import os.path as osp
import random
from collections import defaultdict

import mmcv
import numpy as np
import torch
from detectron2.data.detection_utils import read_image
from mmdet.datasets import DATASETS, CocoPanopticDataset
from mmdet.datasets.coco_panoptic import COCOPanoptic
from mmdet.datasets.pipelines import Compose
from panopticapi.utils import rgb2id

from openpsg.evaluation import sgg_evaluation
from openpsg.models.relation_heads.approaches import Result


@DATASETS.register_module()
class PanopticSceneGraphDataset(CocoPanopticDataset):
    def __init__(
            self,
            ann_file,
            pipeline,
            classes=None,
            data_root=None,
            img_prefix='',
            seg_prefix=None,
            proposal_file=None,
            test_mode=False,
            filter_empty_gt=True,
            file_client_args=dict(backend='disk'),
            # New args
            split: str = 'train',  # {"train", "test"}
            all_bboxes: bool = False,  # load all bboxes (thing, stuff) for SG
    ):
        self.ann_file = ann_file
        self.data_root = data_root
        self.img_prefix = img_prefix
        self.seg_prefix = seg_prefix
        self.proposal_file = proposal_file
        self.test_mode = test_mode
        self.filter_empty_gt = filter_empty_gt
        self.file_client = mmcv.FileClient(**file_client_args)

        # join paths if data_root is specified
        if self.data_root is not None:
            if not osp.isabs(self.ann_file):
                self.ann_file = osp.join(self.data_root, self.ann_file)
            if not (self.img_prefix is None or osp.isabs(self.img_prefix)):
                self.img_prefix = osp.join(self.data_root, self.img_prefix)
            if not (self.seg_prefix is None or osp.isabs(self.seg_prefix)):
                self.seg_prefix = osp.join(self.data_root, self.seg_prefix)
            if not (self.proposal_file is None
                    or osp.isabs(self.proposal_file)):
                self.proposal_file = osp.join(self.data_root,
                                              self.proposal_file)

        self.proposal_file = None
        self.proposals = None

        self.all_bboxes = all_bboxes
        self.split = split

        # Load dataset
        dataset = mmcv.load(ann_file)

        for d in dataset['data']:
            # NOTE: 0-index for object class labels
            # for s in d['segments_info']:
            #     s['category_id'] += 1

            # for a in d['annotations']:
            #     a['category_id'] += 1

            # NOTE: 1-index for predicate class labels
            for r in d['relations']:
                r[2] += 1

        # NOTE: Filter out images with zero relations
        # dataset['data'] = [
        #     d for d in dataset['data'] if len(d['relations']) != 0
        # ]

        # Get split
        assert split in {'train', 'test'}
        if split == 'train':
            self.data = [
                d for d in dataset['data']
                if d['image_id'] not in dataset['test_image_ids']
            ]
            # self.data = self.data[:1000] # for quick debug
        elif split == 'test':
            self.data = [
                d for d in dataset['data']
                if d['image_id'] in dataset['test_image_ids']
            ]
            # self.data = self.data[:1000] # for quick debug
        # Init image infos
        self.data_infos = []
        for d in self.data:
            self.data_infos.append({
                'filename': d['file_name'],
                'height': d['height'],
                'width': d['width'],
                'id': d['image_id'],
            })
        self.img_ids = [d['id'] for d in self.data_infos]

        # Define classes, 0-index
        # NOTE: Class ids should range from 0 to (num_classes - 1)
        self.THING_CLASSES = dataset['thing_classes']
        self.STUFF_CLASSES = dataset['stuff_classes']
        self.CLASSES = self.THING_CLASSES + self.STUFF_CLASSES
        self.PREDICATES = dataset['predicate_classes']

        # NOTE: For evaluation
        self.coco = self._init_cocoapi()
        self.cat_ids = self.coco.get_cat_ids()
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.categories = self.coco.cats

        # processing pipeline
        self.pipeline = Compose(pipeline)

        if not self.test_mode:
            self._set_group_flag()

    def _init_cocoapi(self):
        auxcoco = COCOPanoptic()

        annotations = []

        # Create mmdet coco panoptic data format
        for d in self.data:

            annotation = {
                'file_name': d['pan_seg_file_name'],
                'image_id': d['image_id'],
            }
            segments_info = []

            for a, s in zip(d['annotations'], d['segments_info']):

                segments_info.append({
                    'id':
                    s['id'],
                    'category_id':
                    s['category_id'],
                    'iscrowd':
                    s['iscrowd'],
                    'area':
                    int(s['area']),
                    # Convert from xyxy to xywh
                    'bbox': [
                        a['bbox'][0],
                        a['bbox'][1],
                        a['bbox'][2] - a['bbox'][0],
                        a['bbox'][3] - a['bbox'][1],
                    ],
                })

            annotation['segments_info'] = segments_info

            annotations.append(annotation)

        thing_categories = [{
            'id': i,
            'name': name,
            'isthing': 1
        } for i, name in enumerate(self.THING_CLASSES)]
        stuff_categories = [{
            'id': i + len(self.THING_CLASSES),
            'name': name,
            'isthing': 0
        } for i, name in enumerate(self.STUFF_CLASSES)]

        # Create `dataset` attr for for `createIndex` method
        auxcoco.dataset = {
            'images': self.data_infos,
            'annotations': annotations,
            'categories': thing_categories + stuff_categories,
        }
        auxcoco.createIndex()
        auxcoco.img_ann_map = auxcoco.imgToAnns
        auxcoco.cat_img_map = auxcoco.catToImgs

        return auxcoco

    def get_ann_info(self, idx):
        d = self.data[idx]

        # Process bbox annotations
        gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        if self.all_bboxes:
            # NOTE: Get all the bbox annotations (thing + stuff)
            gt_bboxes = np.array([a['bbox'] for a in d['annotations']],
                                 dtype=np.float32)
            gt_labels = np.array([a['category_id'] for a in d['annotations']],
                                 dtype=np.int64)

        else:
            gt_bboxes = []
            gt_labels = []

            # FIXME: Do we have to filter out `is_crowd`?
            # Do not train on `is_crowd`,
            # i.e just follow the mmdet dataset classes
            # Or treat them as stuff classes?
            # Can try and train on datasets with iscrowd
            # and without and see the difference

            for a, s in zip(d['annotations'], d['segments_info']):
                # NOTE: Only thing bboxes are loaded
                if s['isthing']:
                    gt_bboxes.append(a['bbox'])
                    gt_labels.append(a['category_id'])

            if gt_bboxes:
                gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
                gt_labels = np.array(gt_labels, dtype=np.int64)
            else:
                gt_bboxes = np.zeros((0, 4), dtype=np.float32)
                gt_labels = np.array([], dtype=np.int64)

        # Process segment annotations
        gt_mask_infos = []
        for s in d['segments_info']:
            gt_mask_infos.append({
                'id': s['id'],
                'category': s['category_id'],
                'is_thing': s['isthing']
            })

        # Process relationship annotations
        gt_rels = d['relations'].copy()

        # Filter out dupes!
        if self.split == 'train':
            all_rel_sets = defaultdict(list)
            for (o0, o1, r) in gt_rels:
                all_rel_sets[(o0, o1)].append(r)
            gt_rels = [(k[0], k[1], np.random.choice(v))
                       for k, v in all_rel_sets.items()]
            gt_rels = np.array(gt_rels, dtype=np.int32)
        else:
            # for test or val set, filter the duplicate triplets,
            # but allow multiple labels for each pair
            all_rel_sets = []
            for (o0, o1, r) in gt_rels:
                if (o0, o1, r) not in all_rel_sets:
                    all_rel_sets.append((o0, o1, r))
            gt_rels = np.array(all_rel_sets, dtype=np.int32)

        # add relation to target
        num_box = len(gt_mask_infos)
        relation_map = np.zeros((num_box, num_box), dtype=np.int64)
        for i in range(gt_rels.shape[0]):
            # If already exists a relation?
            if relation_map[int(gt_rels[i, 0]), int(gt_rels[i, 1])] > 0:
                if random.random() > 0.5:
                    relation_map[int(gt_rels[i, 0]),
                                 int(gt_rels[i, 1])] = int(gt_rels[i, 2])
            else:
                relation_map[int(gt_rels[i, 0]),
                             int(gt_rels[i, 1])] = int(gt_rels[i, 2])

        ann = dict(
            bboxes=gt_bboxes,
            labels=gt_labels,
            rels=gt_rels,
            rel_maps=relation_map,
            bboxes_ignore=gt_bboxes_ignore,
            masks=gt_mask_infos,
            seg_map=d['pan_seg_file_name'],
        )

        return ann

    def pre_pipeline(self, results):
        """Prepare results dict for pipeline."""
        super().pre_pipeline(results)

        results['rel_fields'] = []

    def prepare_test_img(self, idx):
        # For SGG, since the forward process may need gt_bboxes/gt_labels,
        # we should also load annotation as if in the training mode.
        return self.prepare_train_img(idx)

    def evaluate(
        self,
        results,
        metric='predcls',
        logger=None,
        jsonfile_prefix=None,
        classwise=True,
        multiple_preds=False,
        iou_thrs=0.5,
        nogc_thres_num=None,
        detection_method='bbox',
        **kwargs,
    ):
        """Overwritten evaluate API:

        For each metric in metrics, it checks whether to invoke ps or sg
        evaluation. if the metric is not 'sg', the evaluate method of super
        class is invoked to perform Panoptic Segmentation evaluation. else,
        perform scene graph evaluation.
        """
        metrics = metric if isinstance(metric, list) else [metric]

        # Available metrics
        allowed_sg_metrics = ['predcls', 'sgcls', 'sgdet']
        allowed_od_metrics = ['PQ']

        sg_metrics, od_metrics = [], []
        for m in metrics:
            if m in allowed_od_metrics:
                od_metrics.append(m)
            elif m in allowed_sg_metrics:
                sg_metrics.append(m)
            else:
                raise ValueError('Unknown metric {}.'.format(m))

        if len(od_metrics) > 0:
            # invoke object detection evaluation.
            # Temporarily for bbox
            if not isinstance(results[0], Result):
                # it may be the results from the son classes
                od_results = results
            else:
                od_results = [{'pan_results': r.pan_results} for r in results]
            return super().evaluate(
                od_results,
                metric,
                logger,
                jsonfile_prefix,
                classwise=classwise,
                **kwargs,
            )

        if len(sg_metrics) > 0:
            """Invoke scene graph evaluation.

            prepare the groundtruth and predictions. Transform the predictions
            of key-wise to image-wise. Both the value in gt_results and
            det_results are numpy array.
            """
            if not hasattr(self, 'test_gt_results'):
                print('\nLoading testing groundtruth...\n')
                prog_bar = mmcv.ProgressBar(len(self))
                gt_results = []
                for i in range(len(self)):
                    ann = self.get_ann_info(i)

                    # NOTE: Change to object class labels 1-index here
                    ann['labels'] += 1

                    # load gt pan_seg masks
                    segment_info = ann['masks']
                    gt_img = read_image(self.img_prefix + '/' + ann['seg_map'],
                                        format='RGB')
                    gt_img = gt_img.copy()  # (H, W, 3)

                    seg_map = rgb2id(gt_img)

                    # get separate masks
                    gt_masks = []
                    labels_coco = []
                    for _, s in enumerate(segment_info):
                        label = self.CLASSES[s['category']]
                        labels_coco.append(label)
                        gt_masks.append(seg_map == s['id'])
                    # load gt pan seg masks done

                    gt_results.append(
                        Result(
                            bboxes=ann['bboxes'],
                            labels=ann['labels'],
                            rels=ann['rels'],
                            relmaps=ann['rel_maps'],
                            rel_pair_idxes=ann['rels'][:, :2],
                            rel_labels=ann['rels'][:, -1],
                            masks=gt_masks,
                        ))
                    prog_bar.update()

                print('\n')
                self.test_gt_results = gt_results

            return sgg_evaluation(
                sg_metrics,
                groundtruths=self.test_gt_results,
                predictions=results,
                iou_thrs=iou_thrs,
                logger=logger,
                ind_to_predicates=['__background__'] + self.PREDICATES,
                multiple_preds=multiple_preds,
                # predicate_freq=self.predicate_freq,
                nogc_thres_num=nogc_thres_num,
                detection_method=detection_method,
            )

    def get_statistics(self):
        freq_matrix = self.get_freq_matrix()
        eps = 1e-3
        freq_matrix += eps
        pred_dist = np.log(freq_matrix / freq_matrix.sum(2)[:, :, None] + eps)

        result = {
            'freq_matrix': torch.from_numpy(freq_matrix),
            'pred_dist': torch.from_numpy(pred_dist).float(),
        }
        if result['pred_dist'].isnan().any():
            print('check pred_dist: nan')
        return result

    def get_freq_matrix(self):
        num_obj_classes = len(self.CLASSES)
        num_rel_classes = len(self.PREDICATES)

        freq_matrix = np.zeros(
            (num_obj_classes, num_obj_classes, num_rel_classes + 1),
            dtype=np.float)
        progbar = mmcv.ProgressBar(len(self.data))

        for d in self.data:
            segments = d['segments_info']
            relations = d['relations']

            for rel in relations:
                object_index = segments[rel[0]]['category_id']
                subject_index = segments[rel[1]]['category_id']
                relation_index = rel[2]

                freq_matrix[object_index, subject_index, relation_index] += 1

            progbar.update()

        return freq_matrix

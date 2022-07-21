import os.path as osp
import random
from collections import defaultdict

import mmcv
import numpy as np
from mmdet.datasets import DATASETS, CocoDataset
from mmdet.datasets.api_wrappers import COCO
from mmdet.datasets.pipelines import Compose

from openpsg.evaluation import sgg_evaluation
from openpsg.models.relation_heads.approaches import Result


@DATASETS.register_module()
class SceneGraphDataset(CocoDataset):
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
        dataset['data'] = [
            d for d in dataset['data'] if len(d['relations']) != 0
        ]

        # Get split
        assert split in {'train', 'test'}
        if split == 'train':
            self.data = [
                d for d in dataset['data']
                if d['image_id'] not in dataset['test_image_ids']
            ]
        elif split == 'test':
            self.data = [
                d for d in dataset['data']
                if d['image_id'] in dataset['test_image_ids']
            ]

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
        self.CLASSES = dataset['thing_classes'] + dataset['stuff_classes']
        self.PREDICATES = dataset['predicate_classes']

        # NOTE: For od evaluation
        self.cat_ids = list(range(0, len(self.CLASSES)))
        self.coco = self._init_cocoapi()

        # processing pipeline
        self.pipeline = Compose(pipeline)

        if not self.test_mode:
            self._set_group_flag()

    def _init_cocoapi(self):
        auxcoco = COCO()

        anns = []

        # Create COCO data format
        for d in self.data:
            for a in d['annotations']:

                anns.append({
                    'area':
                    float((a['bbox'][2] - a['bbox'][0] + 1) *
                          (a['bbox'][3] - a['bbox'][1] + 1)),
                    # Convert from xyxy to xywh
                    'bbox': [
                        a['bbox'][0],
                        a['bbox'][1],
                        a['bbox'][2] - a['bbox'][0],
                        a['bbox'][3] - a['bbox'][1],
                    ],
                    'category_id':
                    a['category_id'],
                    'id':
                    len(anns),
                    'image_id':
                    d['image_id'],
                    'iscrowd':
                    0,
                })

        auxcoco.dataset = {
            'images':
            self.data_infos,
            'categories': [{
                'id': i,
                'name': name
            } for i, name in enumerate(self.CLASSES)],
            'annotations':
            anns,
        }
        auxcoco.createIndex()

        return auxcoco

    def get_ann_info(self, idx):
        d = self.data[idx]

        gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        # Process bbox annotations
        gt_bboxes = np.array([a['bbox'] for a in d['annotations']],
                             dtype=np.float32)
        gt_labels = np.array([a['category_id'] for a in d['annotations']],
                             dtype=np.int64)

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
            # for test or val set, filter the duplicate triplets, but allow multiple labels for each pair
            all_rel_sets = []
            for (o0, o1, r) in gt_rels:
                if (o0, o1, r) not in all_rel_sets:
                    all_rel_sets.append((o0, o1, r))
            gt_rels = np.array(all_rel_sets, dtype=np.int32)

        # add relation to target
        num_box = len(gt_bboxes)
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
            masks=None,
            seg_map=None,
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
        classwise=False,
        multiple_preds=False,
        iou_thrs=0.5,
        nogc_thres_num=None,
        **kwargs,
    ):
        """
        **kwargs: contain the paramteters specifically for OD, e.g., proposal_nums.
        Overwritten evaluate API:
            For each metric in metrics, it checks whether to invoke od or sg evaluation.
            if the metric is not 'sg', the evaluate method of super class is invoked
            to perform Object Detection evaluation.
            else, perform scene graph evaluation.
        """
        metrics = metric if isinstance(metric, list) else [metric]

        # Available metrics
        allowed_sg_metrics = ['predcls', 'sgcls', 'sgdet']
        allowed_od_metrics = ['bbox', 'segm', 'proposal', 'proposal_fast']

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
                # it may be the reuslts from the son classes
                od_results = results
            else:
                od_results = [(r.formatted_bboxes, r.formatted_masks)
                              for r in results]
            return super().evaluate(
                od_results,
                metric,
                logger,
                jsonfile_prefix,
                classwise=classwise,
                iou_thrs=None,
                **kwargs,
            )

        if len(sg_metrics) > 0:
            """Invoke scenen graph evaluation.

            prepare the groundtruth and predictions. Transform the predictions
            of key-wise to image-wise. Both the value in gt_results and
            det_results are numpy array.
            """
            if not hasattr(self, 'test_gt_results'):
                print('\nLooading testing groundtruth...\n')
                prog_bar = mmcv.ProgressBar(len(self))
                gt_results = []
                for i in range(len(self)):
                    ann = self.get_ann_info(i)

                    # NOTE: Change to object class labels 1-index here
                    ann['labels'] += 1

                    gt_results.append(
                        Result(
                            bboxes=ann['bboxes'],
                            labels=ann['labels'],
                            rels=ann['rels'],
                            relmaps=ann['rel_maps'],
                            rel_pair_idxes=ann['rels'][:, :2],
                            rel_labels=ann['rels'][:, -1],
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
            )

import copy
import itertools

import mmcv
import numpy as np
import torch
import torch.nn.functional as F
from mmcv.runner import BaseModule
from mmdet.core import bbox2roi
from mmdet.models import HEADS, builder
from mmdet.models.losses import accuracy

from .approaches import (FrequencyBias, PostProcessor, RelationSampler,
                         get_weak_key_rel_labels)


@HEADS.register_module()
class RelationHead(BaseModule):
    """The basic class of all the relation head."""
    def __init__(
        self,
        object_classes,
        predicate_classes,
        head_config,
        bbox_roi_extractor=None,
        relation_roi_extractor=None,
        relation_sampler=None,
        relation_ranker=None,
        dataset_config=None,
        use_bias=False,
        use_statistics=False,
        num_classes=151,
        num_predicates=51,
        loss_object=dict(type='CrossEntropyLoss',
                         use_sigmoid=False,
                         loss_weight=1.0),
        loss_relation=None,
        init_cfg=None,
    ):
        """The public parameters that shared by various relation heads are
        initialized here."""
        super(RelationHead, self).__init__(init_cfg)

        self.use_bias = use_bias
        self.num_classes = num_classes
        self.num_predicates = num_predicates

        # upgrade some submodule attribute to this head
        self.head_config = head_config
        self.use_gt_box = self.head_config.use_gt_box
        self.use_gt_label = self.head_config.use_gt_label
        self.with_visual_bbox = (bbox_roi_extractor is not None
                                 and bbox_roi_extractor.with_visual_bbox) or (
                                     relation_roi_extractor is not None and
                                     relation_roi_extractor.with_visual_bbox)
        self.with_visual_mask = (bbox_roi_extractor is not None
                                 and bbox_roi_extractor.with_visual_mask) or (
                                     relation_roi_extractor is not None and
                                     relation_roi_extractor.with_visual_mask)
        self.with_visual_point = (bbox_roi_extractor is not None and
                                  bbox_roi_extractor.with_visual_point) or (
                                      relation_roi_extractor is not None and
                                      relation_roi_extractor.with_visual_point)

        self.dataset_config = dataset_config

        if self.use_gt_box:
            if self.use_gt_label:
                self.mode = 'predcls'
            else:
                self.mode = 'sgcls'
        else:
            self.mode = 'sgdet'
        if bbox_roi_extractor is not None:
            self.bbox_roi_extractor = builder.build_roi_extractor(
                bbox_roi_extractor)
        if relation_roi_extractor is not None:
            self.relation_roi_extractor = builder.build_roi_extractor(
                relation_roi_extractor)
        if relation_sampler is not None:
            relation_sampler.update(dict(use_gt_box=self.use_gt_box))
            self.relation_sampler = RelationSampler(**relation_sampler)

        self.post_processor = PostProcessor()

        # relation ranker: a standard component
        if relation_ranker is not None:
            ranker = relation_ranker.pop('type')
            # self.supervised_form = relation_ranker.pop('supervised_form')
            self.comb_factor = relation_ranker.pop('comb_factor', 0.5)
            self.area_form = relation_ranker.pop('area_form', 'rect')
            loss_ranking_relation = relation_ranker.pop('loss')
            self.loss_ranking_relation = builder.build_loss(
                loss_ranking_relation)
            if loss_ranking_relation.type != 'CrossEntropyLoss':
                num_out = 1
            else:
                num_out = 2
            relation_ranker.update(dict(num_out=num_out))
            self.relation_ranker = eval(ranker)(**relation_ranker)

        if loss_object is not None:
            self.loss_object = builder.build_loss(loss_object)

        if loss_relation is not None:
            self.loss_relation = builder.build_loss(loss_relation)

        if use_statistics:
            cache_dir = dataset_config['cache']
            self.statistics = torch.load(cache_dir,
                                         map_location=torch.device('cpu'))
            print('\n Statistics loaded!')

        self.obj_classes, self.rel_classes = (
            object_classes,
            predicate_classes,
        )
        self.obj_classes.insert(0, '__background__')
        self.rel_classes.insert(0, '__background__')

        assert self.num_classes == len(self.obj_classes)
        assert self.num_predicates == len(self.rel_classes)

        if self.use_bias:
            assert self.with_statistics
            # convey statistics into FrequencyBias to avoid loading again
            self.freq_bias = FrequencyBias(self.head_config, self.statistics)

    @property
    def with_bbox_roi_extractor(self):
        return (hasattr(self, 'bbox_roi_extractor')
                and self.bbox_roi_extractor is not None)

    @property
    def with_relation_roi_extractor(self):
        return (hasattr(self, 'relation_roi_extractor')
                and self.relation_roi_extractor is not None)

    @property
    def with_statistics(self):
        return hasattr(self, 'statistics') and self.statistics is not None

    @property
    def with_bias(self):
        return hasattr(self, 'freq_bias') and self.freq_bias is not None

    @property
    def with_loss_object(self):
        return hasattr(self, 'loss_object') and self.loss_object is not None

    @property
    def with_loss_relation(self):
        return hasattr(self,
                       'loss_relation') and self.loss_relation is not None

    @property
    def with_relation_ranker(self):
        return hasattr(self,
                       'relation_ranker') and self.relation_ranker is not None

    def init_weights(self):
        if self.with_bbox_roi_extractor:
            self.bbox_roi_extractor.init_weights()
        if self.with_relation_roi_extractor:
            self.relation_roi_extractor.init_weights()
        self.context_layer.init_weights()

    def frontend_features(self, img, img_meta, det_result, gt_result):
        bboxes, masks, points = (
            det_result.bboxes,
            det_result.masks,
            copy.deepcopy(det_result.points),
        )

        # train/val or: for finetuning on the dataset without
        # relationship annotations
        if gt_result is not None and gt_result.rels is not None:
            if self.mode in ['predcls', 'sgcls']:
                sample_function = self.relation_sampler.gtbox_relsample
            else:
                sample_function = self.relation_sampler.detect_relsample

            sample_res = sample_function(det_result, gt_result)
            if len(sample_res) == 4:
                rel_labels, rel_pair_idxes, rel_matrix, \
                    key_rel_labels = sample_res
            else:
                rel_labels, rel_pair_idxes, rel_matrix = sample_res
                key_rel_labels = None
        else:
            rel_labels, rel_matrix, key_rel_labels = None, None, None
            rel_pair_idxes = self.relation_sampler.prepare_test_pairs(
                det_result)

        det_result.rel_pair_idxes = rel_pair_idxes
        det_result.relmaps = rel_matrix
        det_result.target_rel_labels = rel_labels
        det_result.target_key_rel_labels = key_rel_labels

        rois = bbox2roi(bboxes)
        # merge image-wise masks or points
        if masks is not None:
            masks = list(itertools.chain(*masks))
        if points is not None:
            aug_points = []
            for pts_list in points:
                for pts in pts_list:
                    pts = pts.view(-1, 2)  # (:, [x, y])
                    pts += torch.from_numpy(
                        np.random.normal(0, 0.02, size=pts.shape)).to(pts)
                    # pts -= torch.mean(pts, dim=0, keepdim=True)
                    pts /= torch.max(torch.sqrt(torch.sum(pts**2, dim=1)))
                    aug_points.append(pts)
            points = aug_points

        # extract the unary roi features and union roi features.
        roi_feats = self.bbox_roi_extractor(img,
                                            img_meta,
                                            rois,
                                            masks=masks,
                                            points=points)
        union_feats = self.relation_roi_extractor(img,
                                                  img_meta,
                                                  rois,
                                                  rel_pair_idx=rel_pair_idxes,
                                                  masks=masks,
                                                  points=points)

        return roi_feats + union_feats + (det_result, )
        # return roi_feats, union_feats, (det_result,)

    def forward(self, **kwargs):
        raise NotImplementedError

    def relation_ranking_forward(self, input, det_result, gt_result, num_rels,
                                 is_testing):
        # predict the ranking

        # tensor
        ranking_scores = self.relation_ranker(
            input.detach(), det_result, self.relation_roi_extractor.union_rois)

        # (1) weak supervision, KLDiv:
        if self.loss_ranking_relation.__class__.__name__ == 'KLDivLoss':
            if not is_testing:  # include training and validation
                # list form
                det_result.target_key_rel_labels = get_weak_key_rel_labels(
                    det_result, gt_result, self.comb_factor, self.area_form)
                ranking_scores = ranking_scores.view(-1)
                ranking_scores = ranking_scores.split(num_rels, 0)
            else:
                ranking_scores = ranking_scores.view(-1)
                ranking_scores = torch.sigmoid(ranking_scores).split(num_rels,
                                                                     dim=0)

        # (2) CEloss: the predicted one is the binary classification, 2 columns
        if self.loss_ranking_relation.__class__.__name__ == 'CrossEntropyLoss':
            if not is_testing:
                det_result.target_key_rel_labels = torch.cat(
                    det_result.target_key_rel_labels, dim=-1)
            else:
                ranking_scores = (F.softmax(ranking_scores,
                                            dim=-1)[:, 1].view(-1).split(
                                                num_rels, 0))
        # Margin loss, DR loss
        elif self.loss_ranking_relation.__class__.__name__ == 'SigmoidDRLoss':
            if not is_testing:
                ranking_scores = ranking_scores.view(-1)
                ranking_scores = ranking_scores.split(num_rels, 0)
            else:
                ranking_scores = ranking_scores.view(-1)
                ranking_scores = torch.sigmoid(ranking_scores).split(num_rels,
                                                                     dim=0)

        det_result.ranking_scores = ranking_scores
        return det_result

    def loss(self, det_result):
        (
            obj_scores,
            rel_scores,
            target_labels,
            target_rel_labels,
            add_for_losses,
            head_spec_losses,
        ) = (
            det_result.refine_scores,
            det_result.rel_scores,
            det_result.target_labels,
            det_result.target_rel_labels,
            det_result.add_losses,
            det_result.head_spec_losses,
        )

        losses = dict()
        if self.with_loss_object and obj_scores is not None:
            # fix: the val process
            if isinstance(target_labels, (tuple, list)):
                target_labels = torch.cat(target_labels, dim=-1)
            if isinstance(obj_scores, (tuple, list)):
                obj_scores = torch.cat(obj_scores, dim=0)

            losses['loss_object'] = self.loss_object(obj_scores, target_labels)
            losses['acc_object'] = accuracy(obj_scores, target_labels)

        if self.with_loss_relation and rel_scores is not None:
            if isinstance(target_rel_labels, (tuple, list)):
                target_rel_labels = torch.cat(target_rel_labels, dim=-1)
            if isinstance(rel_scores, (tuple, list)):
                rel_scores = torch.cat(rel_scores, dim=0)
            losses['loss_relation'] = self.loss_relation(
                rel_scores, target_rel_labels)
            losses['acc_relation'] = accuracy(rel_scores, target_rel_labels)

        if self.with_relation_ranker:
            target_key_rel_labels = det_result.target_key_rel_labels
            ranking_scores = det_result.ranking_scores

            avg_factor = (torch.nonzero(
                target_key_rel_labels != -1).view(-1).size(0) if isinstance(
                    target_key_rel_labels, torch.Tensor) else None)
            losses['loss_ranking_relation'] = self.loss_ranking_relation(
                ranking_scores, target_key_rel_labels, avg_factor=avg_factor)
            # if self.supervised_form == 'weak':
            #     # use the KLdiv loss: the label is the soft distribution
            #     bs = 0
            #     losses['loss_ranking_relation'] = 0
            #     for ranking_score, target_key_rel_label in zip(ranking_scores, target_key_rel_labels):
            #         bs += ranking_score.size(0)
            #         losses['loss_ranking_relation'] += torch.nn.KLDivLoss(reduction='none')(F.log_softmax(ranking_score, dim=-1),
            #                                                                     target_key_rel_label).sum(-1)
            #     losses['loss_ranking_relation'] = losses['loss_ranking_relation'] / bs
            # else:
            #     #TODO: firstly try the CE loss function, or you may try the margin loss
            #     #TODO: Check the margin loss
            #     #loss_func = builder.build_loss(self.loss_ranking_relation)
            #     losses['loss_ranking_relation'] = self.loss_ranking_relation(ranking_scores, target_key_rel_labels)

        if add_for_losses is not None:
            for loss_key, loss_item in add_for_losses.items():
                if isinstance(loss_item, list):  # loss_vctree_binary
                    loss_ = [
                        F.binary_cross_entropy_with_logits(l[0], l[1])
                        for l in loss_item
                    ]
                    loss_ = sum(loss_) / len(loss_)
                    losses[loss_key] = loss_
                elif isinstance(loss_item, tuple):
                    if isinstance(loss_item[1], (list, tuple)):
                        target = torch.cat(loss_item[1], -1)
                    else:
                        target = loss_item[1]
                    losses[loss_key] = F.cross_entropy(loss_item[0], target)
                else:
                    raise NotImplementedError

        if head_spec_losses is not None:
            # this losses have been calculated in the specific relation head
            losses.update(head_spec_losses)

        return losses

    def get_result(self, det_result, scale_factor, rescale, key_first=False):
        """for test forward.

        :param det_result:
        :return:
        """
        result = self.post_processor(det_result, key_first=key_first)

        for k, v in result.__dict__.items():
            if (k != 'add_losses' and k != 'head_spec_losses' and v is not None
                    and len(v) > 0):
                _v = v[0]  # remove the outer list
                if isinstance(_v, torch.Tensor):
                    result.__setattr__(k, _v.cpu().numpy())
                elif isinstance(_v, list):  # for mask
                    result.__setattr__(k, [__v.cpu().numpy() for __v in _v])
                else:
                    result.__setattr__(k, _v)  # e.g., img_shape, is a tuple

        if rescale:
            if result.bboxes is not None:
                result.bboxes[:, :4] = result.bboxes[:, :4] / scale_factor
            if result.refine_bboxes is not None:
                result.refine_bboxes[:, :
                                     4] = result.refine_bboxes[:, :
                                                               4] / scale_factor

            if result.masks is not None:
                resize_masks = []
                for bbox, mask in zip(result.refine_bboxes, result.masks):
                    _bbox = bbox.astype(np.int32)
                    w = max(_bbox[2] - _bbox[0] + 1, 1)
                    h = max(_bbox[3] - _bbox[1] + 1, 1)
                    resize_masks.append(
                        mmcv.imresize(mask.astype(np.uint8), (w, h)))
                result.masks = resize_masks

            if result.points is not None:
                resize_points = []
                for points in result.points:
                    resize_points.append(points / scale_factor)
                result.points = resize_points

        # if needed, adjust the form for object detection evaluation
        result.formatted_bboxes, result.formatted_masks = [], []

        if result.refine_bboxes is None:
            result.formatted_bboxes = [
                np.zeros((0, 5), dtype=np.float32)
                for i in range(self.num_classes - 1)
            ]
        else:
            result.formatted_bboxes = [
                result.refine_bboxes[result.refine_labels == i + 1, :]
                for i in range(self.num_classes - 1)
            ]

        if result.masks is None:
            result.formatted_masks = [[] for i in range(self.num_classes - 1)]
        else:
            result.formatted_masks = [[] for i in range(self.num_classes - 1)]
            for i in range(len(result.masks)):
                result.formatted_masks[result.refine_labels[i] - 1].append(
                    result.masks[i])

        # to save the space, drop the saliency maps, if it exists
        if result.saliency_maps is not None:
            result.saliency_maps = None

        return result

    def process_ignore_objects(self, input, ignore_classes):
        """An API used in inference stage for processing the data when some
        object classes should be ignored."""
        ignored_input = input.clone()
        ignored_input[:, ignore_classes] = 0.0
        return ignored_input

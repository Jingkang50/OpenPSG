# ---------------------------------------------------------------
# gps_head.py
# Set-up time: 2021/3/31 17:13
# Copyright (c) 2020 ICT
# Licensed under The MIT License [see LICENSE for details]
# Written by Kenneth-Wong (Wenbin-Wang) @ VIPL.ICT
# Contact: wenbin.wang@vipl.ict.ac.cn [OR] nkwangwenbin@gmail.com
# ---------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models import HEADS

from .approaches import DirectionAwareMessagePassing
from .relation_head import RelationHead


@HEADS.register_module()
class GPSHead(RelationHead):
    def __init__(self, **kwargs):
        super(GPSHead, self).__init__(**kwargs)

        # 1. Initialize the interaction pattern templates

        self.context_layer = DirectionAwareMessagePassing(
            self.head_config, self.obj_classes)

        if self.use_bias:
            self.wp = nn.Linear(self.head_config.roi_dim, self.num_predicates)
        self.w_proj1 = nn.Linear(self.head_config.roi_dim,
                                 self.head_config.roi_dim)
        self.w_proj2 = nn.Linear(self.head_config.roi_dim,
                                 self.head_config.roi_dim)
        self.w_proj3 = nn.Linear(self.head_config.roi_dim,
                                 self.head_config.roi_dim)
        self.out_rel = nn.Linear(self.head_config.roi_dim,
                                 self.num_predicates,
                                 bias=True)

    def init_weights(self):
        self.bbox_roi_extractor.init_weights()
        self.relation_roi_extractor.init_weights()

    def relation_infer(self,
                       pair_reps,
                       union_reps,
                       proj1,
                       proj2,
                       proj3,
                       out_rel,
                       wp=None,
                       log_freq=None):
        dim = pair_reps.shape[-1]
        t1, t2, t3 = proj1(pair_reps[:, :dim // 2]), \
            proj2(pair_reps[:, dim // 2:]), proj3(union_reps)
        t4 = (F.relu(t1 + t2) - (t1 - t2) * (t1 - t2))
        rel_scores = out_rel(F.relu(t4 + t3) - (t4 - t3) * (t4 - t3))
        if wp is not None and log_freq is not None:
            tensor_d = F.sigmoid(wp(union_reps))
            rel_scores += tensor_d * log_freq
        return rel_scores

    def forward(self,
                img,
                img_meta,
                det_result,
                gt_result=None,
                is_testing=False,
                ignore_classes=None):
        """Obtain the relation prediction results based on detection results.

        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.
            img_meta (list[dict]): list of image info dict where each dict has:
                'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.
            det_result: (Result): Result containing bbox, label, mask,
                point, rels, etc. According to different mode, all the
                contents have been set correctly. Feel free to use it.
            gt_result : (Result): The ground truth information.
            is_testing:
        Returns:
            det_result with the following newly added keys:
                refine_scores (list[Tensor]): logits of object
                rel_scores (list[Tensor]): logits of relation
                rel_pair_idxes (list[Tensor]): (num_rel, 2) index of
                    subject and object
                relmaps (list[Tensor]): (num_obj, num_obj):
                target_rel_labels (list[Tensor]): the target relation label.
        """
        roi_feats, union_feats, det_result = self.frontend_features(
            img, img_meta, det_result, gt_result)
        if roi_feats.shape[0] == 0:
            return det_result

        num_rels = [r.shape[0] for r in det_result.rel_pair_idxes]
        num_objs = [len(b) for b in det_result.bboxes]
        assert len(num_rels) == len(num_objs)

        # 1. Message Passing with visual texture features
        refine_obj_scores, obj_preds, roi_context_feats = self.context_layer(
            roi_feats, union_feats, det_result)
        obj_preds = obj_preds.split(num_objs, 0)
        split_roi_context_feats = roi_context_feats.split(num_objs)
        pair_reps = []
        pair_preds = []
        for pair_idx, obj_rep, obj_pred in zip(det_result.rel_pair_idxes,
                                               split_roi_context_feats,
                                               obj_preds):
            pair_preds.append(
                torch.stack(
                    (obj_pred[pair_idx[:, 0]], obj_pred[pair_idx[:, 1]]),
                    dim=1))
            pair_reps.append(
                torch.cat((obj_rep[pair_idx[:, 0]], obj_rep[pair_idx[:, 1]]),
                          dim=-1))
        pair_reps = torch.cat(pair_reps, dim=0)
        pair_preds = torch.cat(pair_preds, dim=0)

        # 3. build different relation head
        log_freq = None
        if self.use_bias:
            log_freq = F.log_softmax(
                self.freq_bias.index_with_labels(
                    pair_preds.long() -
                    1))  # USE 0-index when getting frequency vec!
            if log_freq.isnan().any():  # TODO:why?
                log_freq = None

        rel_scores = self.relation_infer(pair_reps, union_feats, self.w_proj1,
                                         self.w_proj2, self.w_proj3,
                                         self.out_rel,
                                         self.wp if self.use_bias else None,
                                         log_freq)

        # make some changes: list to tensor or tensor to tuple
        if not is_testing:
            det_result.target_labels = torch.cat(det_result.target_labels,
                                                 dim=-1)
            det_result.target_rel_labels = torch.cat(
                det_result.target_rel_labels,
                dim=-1) if det_result.target_rel_labels is not None else None

        else:
            refine_obj_scores = refine_obj_scores.split(num_objs, dim=0)
            rel_scores = rel_scores.split(num_rels, dim=0)

        det_result.refine_scores = refine_obj_scores
        det_result.rel_scores = rel_scores

        # ranking prediction:
        if self.with_relation_ranker:
            det_result = self.relation_ranking_forward(pair_reps, det_result,
                                                       gt_result, num_rels,
                                                       is_testing)
        return det_result

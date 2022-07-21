# ---------------------------------------------------------------
# imp_head.py
# Set-up time: 2020/5/21 下午11:22
# Copyright (c) 2020 ICT
# Licensed under The MIT License [see LICENSE for details]
# Written by Kenneth-Wong (Wenbin-Wang) @ VIPL.ICT
# Contact: wenbin.wang@vipl.ict.ac.cn [OR] nkwangwenbin@gmail.com
# ---------------------------------------------------------------

import torch
from mmdet.models import HEADS

from .approaches import IMPContext
from .relation_head import RelationHead


@HEADS.register_module()
class IMPHead(RelationHead):
    def __init__(self, **kwargs):
        super(IMPHead, self).__init__(**kwargs)

        self.context_layer = IMPContext(self.head_config, self.obj_classes,
                                        self.rel_classes)

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
            det_result: (Result): Result containing bbox, label, mask, point, rels,
                etc. According to different mode, all the contents have been
                set correctly. Feel free to  use it.
            gt_result : (Result): The ground truth information.
            is_testing:
        Returns:
            det_result with the following newly added keys:
                refine_scores (list[Tensor]): logits of object
                rel_scores (list[Tensor]): logits of relation
                rel_pair_idxes (list[Tensor]): (num_rel, 2) index of subject and object
                relmaps (list[Tensor]): (num_obj, num_obj):
                target_rel_labels (list[Tensor]): the target relation label.
        """
        roi_feats, union_feats, det_result = self.frontend_features(
            img, img_meta, det_result, gt_result)
        if roi_feats.shape[0] == 0:
            return det_result

        refine_obj_scores, rel_scores = self.context_layer(
            roi_feats, union_feats, det_result)

        num_rels = [r.shape[0] for r in det_result.rel_pair_idxes]
        num_objs = [len(b) for b in det_result.bboxes]
        assert len(num_rels) == len(num_objs)

        if self.use_bias:
            obj_preds = refine_obj_scores.max(-1)[1]
            obj_preds = obj_preds.split(num_objs, dim=0)

            pair_preds = []
            for pair_idx, obj_pred in zip(det_result.rel_pair_idxes,
                                          obj_preds):
                pair_preds.append(
                    torch.stack(
                        (obj_pred[pair_idx[:, 0]], obj_pred[pair_idx[:, 1]]),
                        dim=1))
            pair_pred = torch.cat(pair_preds, dim=0)

            rel_scores = rel_scores + self.freq_bias.index_with_labels(
                pair_pred.long())

        # make some changes: list to tensor or tensor to tuple
        if self.training:
            det_result.target_labels = torch.cat(det_result.target_labels,
                                                 dim=-1)
            det_result.target_rel_labels = torch.cat(
                det_result.target_rel_labels, dim=-1)
        else:
            refine_obj_scores = refine_obj_scores.split(num_objs, dim=0)
            rel_scores = rel_scores.split(num_rels, dim=0)

        det_result.refine_scores = refine_obj_scores
        det_result.rel_scores = rel_scores
        return det_result

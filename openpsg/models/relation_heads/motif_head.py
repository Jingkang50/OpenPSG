# ---------------------------------------------------------------
# motif_head.py
# Set-up time: 2020/4/27 下午8:08
# Copyright (c) 2020 ICT
# Licensed under The MIT License [see LICENSE for details]
# Written by Kenneth-Wong (Wenbin-Wang) @ VIPL.ICT
# Contact: wenbin.wang@vipl.ict.ac.cn [OR] nkwangwenbin@gmail.com
# ---------------------------------------------------------------

import torch
import torch.nn as nn
from mmcv.cnn import normal_init, xavier_init
from mmdet.models import HEADS

from .approaches import LSTMContext
from .relation_head import RelationHead


@HEADS.register_module()
class MotifHead(RelationHead):
    def __init__(self, **kwargs):
        super(MotifHead, self).__init__(**kwargs)

        self.context_layer = LSTMContext(self.head_config, self.obj_classes,
                                         self.rel_classes)

        # post decoding
        self.use_vision = self.head_config.use_vision
        self.hidden_dim = self.head_config.hidden_dim
        self.context_pooling_dim = self.head_config.context_pooling_dim
        self.post_emb = nn.Linear(self.hidden_dim, self.hidden_dim * 2)
        self.post_cat = nn.Linear(self.hidden_dim * 2,
                                  self.context_pooling_dim)
        self.rel_compress = nn.Linear(self.context_pooling_dim,
                                      self.num_predicates,
                                      bias=True)

        if self.context_pooling_dim != self.head_config.roi_dim:
            self.union_single_not_match = True
            self.up_dim = nn.Linear(self.head_config.roi_dim,
                                    self.context_pooling_dim)
        else:
            self.union_single_not_match = False

    def init_weights(self):
        self.bbox_roi_extractor.init_weights()
        self.relation_roi_extractor.init_weights()
        self.context_layer.init_weights()

        normal_init(self.post_emb,
                    mean=0,
                    std=10.0 * (1.0 / self.hidden_dim)**0.5)
        xavier_init(self.post_cat)
        xavier_init(self.rel_compress)

        if self.union_single_not_match:
            xavier_init(self.up_dim)

    def forward(
        self,
        img,
        img_meta,
        det_result,
        gt_result=None,
        is_testing=False,
        ignore_classes=None,
    ):
        """
        Obtain the relation prediction results based on detection results.
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

        # (N_b, N_c + 1), (N_b),
        refine_obj_scores, obj_preds, edge_ctx, _ = self.context_layer(
            roi_feats, det_result)

        if is_testing and ignore_classes is not None:
            refine_obj_scores = self.process_ignore_objects(
                refine_obj_scores, ignore_classes)
            obj_preds = refine_obj_scores[:, 1:].max(1)[1] + 1

        # post decode
        edge_rep = self.post_emb(edge_ctx)
        edge_rep = edge_rep.view(edge_rep.size(0), 2, self.hidden_dim)
        head_rep = edge_rep[:, 0].contiguous().view(-1, self.hidden_dim)
        tail_rep = edge_rep[:, 1].contiguous().view(-1, self.hidden_dim)

        num_rels = [r.shape[0] for r in det_result.rel_pair_idxes]
        num_objs = [len(b) for b in det_result.bboxes]
        assert len(num_rels) == len(num_objs)

        head_reps = head_rep.split(num_objs, dim=0)
        tail_reps = tail_rep.split(num_objs, dim=0)
        obj_preds = obj_preds.split(num_objs, dim=0)

        prod_reps = []
        pair_preds = []
        for pair_idx, head_rep, tail_rep, obj_pred in zip(
                det_result.rel_pair_idxes, head_reps, tail_reps, obj_preds):
            prod_reps.append(
                torch.cat((head_rep[pair_idx[:, 0]], tail_rep[pair_idx[:, 1]]),
                          dim=-1))
            pair_preds.append(
                torch.stack(
                    (obj_pred[pair_idx[:, 0]], obj_pred[pair_idx[:, 1]]),
                    dim=1))
        prod_rep = torch.cat(prod_reps, dim=0)
        pair_pred = torch.cat(pair_preds, dim=0)

        prod_rep = self.post_cat(prod_rep)

        if self.use_vision:
            if self.union_single_not_match:
                prod_rep = prod_rep * self.up_dim(union_feats)
            else:
                prod_rep = prod_rep * union_feats

        rel_scores = self.rel_compress(prod_rep)

        if self.use_bias:
            rel_scores = rel_scores + self.freq_bias.index_with_labels(
                pair_pred.long())

        # make some changes: list to tensor or tensor to tuple
        if self.training:
            det_result.target_labels = torch.cat(det_result.target_labels,
                                                 dim=-1)
            det_result.target_rel_labels = (torch.cat(
                det_result.target_rel_labels,
                dim=-1) if det_result.target_rel_labels is not None else None)
        else:
            refine_obj_scores = refine_obj_scores.split(num_objs, dim=0)
            rel_scores = rel_scores.split(num_rels, dim=0)

        # we use obj_preds instead of pred from obj_dists
        # because in decoder_rnn, preds has been through a nms stage

        det_result.refine_scores = refine_obj_scores
        det_result.rel_scores = rel_scores

        # ranking prediction:
        if self.with_relation_ranker:
            det_result = self.relation_ranking_forward(prod_rep, det_result,
                                                       gt_result, num_rels,
                                                       is_testing)

        return det_result

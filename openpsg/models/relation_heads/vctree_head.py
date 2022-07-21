# ---------------------------------------------------------------
# vctree_head.py
# Set-up time: 2020/6/4 上午9:35
# Copyright (c) 2020 ICT
# Licensed under The MIT License [see LICENSE for details]
# Written by Kenneth-Wong (Wenbin-Wang) @ VIPL.ICT
# Contact: wenbin.wang@vipl.ict.ac.cn [OR] nkwangwenbin@gmail.com
# ---------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import normal_init, xavier_init
from mmdet.models import HEADS

from .approaches import VCTreeLSTMContext
from .relation_head import RelationHead


@HEADS.register_module()
class VCTreeHead(RelationHead):
    def __init__(self, **kwargs):
        super(VCTreeHead, self).__init__(**kwargs)
        self.context_layer = VCTreeLSTMContext(self.head_config,
                                               self.obj_classes,
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

    def forward(self,
                img,
                img_meta,
                det_result,
                gt_result=None,
                is_testing=False,
                ignore_classes=None):
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

        refine_obj_scores, obj_preds, edge_ctx, binary_preds = self.context_layer(
            roi_feats, det_result)

        # post decode
        edge_rep = F.relu(self.post_emb(edge_ctx))
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
            det_result.target_rel_labels = torch.cat(
                det_result.target_rel_labels, dim=-1)
        else:
            refine_obj_scores = refine_obj_scores.split(num_objs, dim=0)
            rel_scores = rel_scores.split(num_rels, dim=0)

        det_result.refine_scores = refine_obj_scores
        det_result.rel_scores = rel_scores

        # add additional auxiliary loss
        add_for_losses = {}
        if not is_testing:
            binary_loss_items = []
            for bi_gt, bi_pred in zip(det_result.relmaps, binary_preds):
                bi_gt = (bi_gt > 0).float()
                binary_loss_items.append((bi_pred, bi_gt))
            add_for_losses['loss_vctree_binary'] = binary_loss_items
        det_result.add_losses = add_for_losses

        # ranking prediction:
        if self.with_relation_ranker:
            det_result = self.relation_ranking_forward(prod_rep, det_result,
                                                       gt_result, num_rels,
                                                       is_testing)

        return det_result

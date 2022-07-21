# ---------------------------------------------------------------
# relation_ranker.py
# Set-up time: 2021/5/11 16:21
# Copyright (c) 2020 ICT
# Licensed under The MIT License [see LICENSE for details]
# Written by Kenneth-Wong (Wenbin-Wang) @ VIPL.ICT
# Contact: wenbin.wang@vipl.ict.ac.cn [OR] nkwangwenbin@gmail.com
# ---------------------------------------------------------------
# from .transformer import Encoder, EncoderLayer, MultiHeadedAttention, PositionwiseFeedForward
import copy

import numpy as np
import torch
from mmcv.cnn import xavier_init
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import PackedSequence

from .motif_util import center_x, sort_by_score
from .relation_util import Result

# class TransformerRanker(nn.Module):
#     def __init__(self, num_head=8, input_dim=1024, hidden_dim=512, inner_dim=1024, dropout_rate=0.1, nl_layer=6, num_out=1):
#         super(TransformerRanker, self).__init__()
#         self.num_head = num_head
#         self.input_dim = input_dim
#         self.hidden_dim = hidden_dim
#         self.inner_dim = inner_dim
#         self.dropout_rate = dropout_rate
#         self.nl_layer = nl_layer
#         self.num_out = num_out
#         c = copy.deepcopy
#         attn = MultiHeadedAttention(self.num_head, self.hidden_dim, self.dropout_rate)
#         ff = PositionwiseFeedForward(self.hidden_dim, self.inner_dim, self.dropout_rate)
#         self.ranking_context = Encoder(EncoderLayer(self.hidden_dim, c(attn), c(ff), self.dropout_rate), self.nl_layer)
#         self.proj = nn.Linear(self.input_dim, self.hidden_dim)
#         self.rank_proj = nn.Linear(self.hidden_dim, self.num_out)

#     def forward(self, union_feats, det_result=None, union_rois=None):
#         rel_pair_idxes = det_result.rel_pair_idxes
#         num_rels = [len(r) for r in rel_pair_idxes]
#         return self.rank_proj(self.ranking_context(self.proj(union_feats), num_rels))


class LSTMRanker(nn.Module):
    def __init__(self,
                 input_dim=1024,
                 hidden_dim=512,
                 dropout_rate=0.2,
                 nl_layer=1,
                 bidirectional=True,
                 num_out=1):
        super(LSTMRanker, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        self.nl_layer = nl_layer
        self.bidirectional = bidirectional
        self.num_out = num_out
        self.ranking_ctx_rnn = torch.nn.LSTM(input_size=self.input_dim,
                                             hidden_size=self.hidden_dim,
                                             num_layers=self.nl_layer,
                                             dropout=self.dropout_rate,
                                             bidirectional=self.bidirectional)

        self.rank_proj = nn.Linear(self.hidden_dim, self.num_out)

    def sort_rois(self, result):
        """
        :param batch_idx: tensor with what index we're on
        :param confidence: tensor with confidences between [0,1)
        :param boxes: tensor with (x1, y1, x2, y2)
        :return: Permutation, inverse permutation, and the lengths transposed (same as _sort_by_score)
        """
        c_x = center_x(result)
        scores = c_x / (c_x.max() + 1)
        return sort_by_score(result, scores)

    def forward(self, union_feats, det_result, union_rois):
        """Forward pass through the object and edge context.

        :param obj_priors:
        :param obj_fmaps:
        :param im_inds:
        :param obj_labels:
        :param boxes:
        :return:
        """
        rel_pair_idxes = det_result.rel_pair_idxes
        num_rels = [len(r) for r in rel_pair_idxes]
        result = Result(bboxes=union_rois.split(num_rels, 0))
        perm, inv_perm, ls_transposed = self.sort_rois(result)

        rel_inpunt_rep = union_feats[perm].contiguous()
        rel_input_packed = PackedSequence(rel_inpunt_rep, ls_transposed)

        rel_rank_rep = self.ranking_ctx_rnn(rel_input_packed)[0][0]
        if self.bidirectional:
            rel_rank_rep = torch.mean(
                torch.stack((rel_rank_rep[:, :self.hidden_dim],
                             rel_rank_rep[:, self.hidden_dim:])), 0)
        rel_rank_rep = rel_rank_rep[inv_perm].contiguous()

        ranking_scores = self.rank_proj(rel_rank_rep)

        return ranking_scores


class LinearRanker(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=512, nl_layer=1, num_out=1):
        super(LinearRanker, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.nl_layer = nl_layer
        self.num_out = num_out
        ranking_net = []
        for i in range(self.nl_layer):
            dim = self.input_dim if i == 0 else self.hidden_dim
            ranking_net += [
                nn.Linear(dim, self.hidden_dim),
                nn.ReLU(inplace=True)
            ]
        ranking_net.append(nn.Linear(self.hidden_dim, self.num_out))
        self.ranking_net = nn.Sequential(*ranking_net)

    def forward(self, union_feats, det_result=None, union_rois=None):
        """Forward pass through the object and edge context.

        :param obj_priors:
        :param obj_fmaps:
        :param im_inds:
        :param obj_labels:
        :param boxes:
        :return:
        """
        ranking_scores = self.ranking_net(union_feats)
        return ranking_scores


def get_size_maps(size, boxes_int, form='rect'):
    h, w = size
    #boxes_int = bbox.long()
    boxes_w, boxes_h = boxes_int[:,
                                 2] - boxes_int[:,
                                                0] + 1, boxes_int[:,
                                                                  3] - boxes_int[:,
                                                                                 1] + 1
    #boxes_w, boxes_h = bbox[:, 2] - bbox[:, 0] + 1, bbox[:, 3] - boxes_int[:, 1] + 1
    areas = boxes_w * boxes_h
    areas_ratios = areas.float() / (h * w)
    ##TODO: maybe there exists better area maps
    # sigma1 = boxes_w / 6
    # sigma2 = boxes_h / 6
    # mus = torch.cat(((boxes_int[:, 0] + boxes_w // 2)[:, None], (boxes_int[:, 1] + boxes_h // 2)[:, None]), dim=-1)
    # x = torch.arange(0, w).long().to(bbox.device)
    # y = torch.arange(0, h).long().to(bbox.device)
    # xx, yy = torch.meshgrid(x, y)
    # # evaluate kernels at grid points
    # xys = torch.cat((xx.view(-1, 1), yy.view(-1, 1)), dim=-1)
    # for sid, (box_int, areas_ratio, mu, sig1, sig2) in enumerate(zip(boxes_int, areas_ratios, mus, sigma1, sigma2)):
    #     xxyy = xys.clone()
    #     xxyy -= mu
    #     x_term = xxyy[:, 0] ** 2 / sig1 ** 2
    #     y_term = xxyy[:, 1] ** 2 / sig2 ** 2
    #     exp_value = - (x_term + y_term) / 2
    #     area_map = torch.exp(exp_value)
    #     area_map = area_map.view((h, w))
    #     area_map = area_map / area_map.max() * areas_ratio
    return areas_ratios


def get_weak_key_rel_labels(det_result,
                            gt_result,
                            comb_factor=0.5,
                            area_form='rect'):
    gt_bboxes = gt_result.bboxes
    det_bboxes = det_result.bboxes
    saliency_maps = det_result.saliency_maps
    key_rel_labels = []
    rel_pair_idxes = det_result.rel_pair_idxes
    for rel_pair_idx, gt_bbox, det_bbox, saliency_map in zip(
            rel_pair_idxes, gt_bboxes, det_bboxes, saliency_maps):
        assert det_bbox.shape[0] == gt_bbox.shape[0]
        det_bbox_int = det_bbox.clone()
        det_bbox_int = det_bbox_int.long()
        h, w = saliency_map.shape[1:]
        det_bbox_int[:, 0::2] = torch.clamp(det_bbox_int[:, 0::2],
                                            min=0,
                                            max=w - 1)
        det_bbox_int[:, 1::2] = torch.clamp(det_bbox_int[:, 1::2],
                                            min=0,
                                            max=h - 1)
        object_saliency = torch.cat([
            torch.mean(saliency_map[0, box[1]:box[3] + 1,
                                    box[0]:box[2] + 1])[None]
            for box in det_bbox_int
        ], 0).float().to(det_bbox_int.device)
        object_area = get_size_maps(saliency_map.shape[1:], det_bbox_int,
                                    area_form)
        object_importance = object_saliency * comb_factor + (
            1.0 - comb_factor) * object_area
        pair_importance = object_importance[
            rel_pair_idx[:, 0]] + object_importance[rel_pair_idx[:, 1]]
        pair_importance = F.softmax(pair_importance)
        key_rel_labels.append(pair_importance)
    return key_rel_labels

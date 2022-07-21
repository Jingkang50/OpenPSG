# ---------------------------------------------------------------
# imp.py
# Set-up time: 2020/5/21 下午11:26
# Copyright (c) 2020 ICT
# Licensed under The MIT License [see LICENSE for details]
# Written by Kenneth-Wong (Wenbin-Wang) @ VIPL.ICT
# Contact: wenbin.wang@vipl.ict.ac.cn [OR] nkwangwenbin@gmail.com
# ---------------------------------------------------------------
import torch
from mmcv.cnn import kaiming_init
from torch import nn
from torch.nn import functional as F

from .motif_util import to_onehot


class IMPContext(nn.Module):
    def __init__(self, config, obj_classes, rel_classes):
        super(IMPContext, self).__init__()
        self.cfg = config
        in_channels = self.cfg.roi_dim
        self.num_object_classes = len(obj_classes)
        self.num_predicates = len(rel_classes)
        self.hidden_dim = self.cfg.hidden_dim
        self.num_iter = self.cfg.num_iter
        # mode
        if self.cfg.use_gt_box:
            if self.cfg.use_gt_label:
                self.mode = 'predcls'
            else:
                self.mode = 'sgcls'
        else:
            self.mode = 'sgdet'

        self.rel_fc = nn.Linear(self.hidden_dim, self.num_predicates)
        self.obj_fc = nn.Linear(self.hidden_dim, self.num_object_classes)

        self.obj_unary = nn.Linear(in_channels, self.hidden_dim)
        self.edge_unary = nn.Linear(in_channels, self.hidden_dim)

        self.edge_gru = nn.GRUCell(input_size=self.hidden_dim,
                                   hidden_size=self.hidden_dim)
        self.node_gru = nn.GRUCell(input_size=self.hidden_dim,
                                   hidden_size=self.hidden_dim)

        self.sub_vert_w_fc = nn.Sequential(nn.Linear(self.hidden_dim * 2, 1),
                                           nn.Sigmoid())
        self.obj_vert_w_fc = nn.Sequential(nn.Linear(self.hidden_dim * 2, 1),
                                           nn.Sigmoid())
        self.out_edge_w_fc = nn.Sequential(nn.Linear(self.hidden_dim * 2, 1),
                                           nn.Sigmoid())
        self.in_edge_w_fc = nn.Sequential(nn.Linear(self.hidden_dim * 2, 1),
                                          nn.Sigmoid())

    def init_weights(self):
        for module in [
                self.sub_vert_w_fc, self.obj_vert_w_fc, self.out_edge_w_fc,
                self.in_edge_w_fc
        ]:
            for m in module:
                if isinstance(m, nn.Linear):
                    kaiming_init(m, distribution='uniform', a=1)
        for module in [
                self.rel_fc, self.obj_fc, self.obj_unary, self.edge_unary
        ]:
            kaiming_init(module, distribution='uniform', a=1)

    def forward(self, x, union_features, det_result, logger=None):
        num_objs = [len(b) for b in det_result.bboxes]
        rel_pair_idxes = det_result.rel_pair_idxes

        obj_rep = self.obj_unary(x)
        rel_rep = F.relu(self.edge_unary(union_features))

        obj_count = obj_rep.shape[0]
        rel_count = rel_rep.shape[0]

        # generate sub-rel-obj mapping
        sub2rel = torch.zeros(obj_count, rel_count).to(obj_rep)
        obj2rel = torch.zeros(obj_count, rel_count).to(obj_rep)
        obj_offset = 0
        rel_offset = 0
        sub_global_inds = []
        obj_global_inds = []
        for pair_idx, num_obj in zip(rel_pair_idxes, num_objs):
            num_rel = pair_idx.shape[0]
            sub_idx = pair_idx[:, 0].contiguous().long().view(-1) + obj_offset
            obj_idx = pair_idx[:, 1].contiguous().long().view(-1) + obj_offset
            rel_idx = torch.arange(num_rel).to(
                obj_rep.device).long().view(-1) + rel_offset

            sub_global_inds.append(sub_idx)
            obj_global_inds.append(obj_idx)

            sub2rel[sub_idx, rel_idx] = 1.0
            obj2rel[obj_idx, rel_idx] = 1.0

            obj_offset += num_obj
            rel_offset += num_rel

        sub_global_inds = torch.cat(sub_global_inds, dim=0)
        obj_global_inds = torch.cat(obj_global_inds, dim=0)

        # iterative message passing
        hx_obj = torch.zeros(obj_count, self.hidden_dim,
                             requires_grad=False).to(obj_rep)
        hx_rel = torch.zeros(rel_count, self.hidden_dim,
                             requires_grad=False).to(obj_rep)

        vert_factor = [self.node_gru(obj_rep, hx_obj)]
        edge_factor = [self.edge_gru(rel_rep, hx_rel)]

        for i in range(self.num_iter):
            # compute edge context
            sub_vert = vert_factor[i][sub_global_inds]
            obj_vert = vert_factor[i][obj_global_inds]
            weighted_sub = self.sub_vert_w_fc(
                torch.cat((sub_vert, edge_factor[i]), 1)) * sub_vert
            weighted_obj = self.obj_vert_w_fc(
                torch.cat((obj_vert, edge_factor[i]), 1)) * obj_vert

            edge_factor.append(
                self.edge_gru(weighted_sub + weighted_obj, edge_factor[i]))

            # Compute vertex context
            pre_out = self.out_edge_w_fc(
                torch.cat((sub_vert, edge_factor[i]), 1)) * edge_factor[i]
            pre_in = self.in_edge_w_fc(torch.cat(
                (obj_vert, edge_factor[i]), 1)) * edge_factor[i]
            vert_ctx = sub2rel @ pre_out + obj2rel @ pre_in
            vert_factor.append(self.node_gru(vert_ctx, vert_factor[i]))

        if self.mode == 'predcls':
            obj_labels = torch.cat(det_result.labels, dim=0)
            obj_dists = to_onehot(obj_labels, self.num_object_classes)
        else:
            obj_dists = self.obj_fc(vert_factor[-1])

        rel_dists = self.rel_fc(edge_factor[-1])

        return obj_dists, rel_dists

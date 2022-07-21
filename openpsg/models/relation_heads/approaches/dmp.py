# ---------------------------------------------------------------
# dmp.py
# Set-up time: 2020/10/7 22:23
# Copyright (c) 2020 ICT
# Licensed under The MIT License [see LICENSE for details]
# Written by Kenneth-Wong (Wenbin-Wang) @ VIPL.ICT
# Contact: wenbin.wang@vipl.ict.ac.cn [OR] nkwangwenbin@gmail.com
# ---------------------------------------------------------------
import torch
from torch import nn
from torch.nn import functional as F

from .motif_util import encode_box_info, obj_edge_vectors, to_onehot


def matmul(tensor3d, mat):
    out = []
    for i in range(tensor3d.size(-1)):
        out.append(torch.mm(tensor3d[:, :, i], mat))
    return torch.cat(out, -1)


class DirectionAwareMessagePassing(nn.Module):
    """Adapted from the [CVPR 2020] GPS-Net: Graph Property Scensing Network
    for Scene Graph Generation]"""
    def __init__(self, config, obj_classes):
        super(DirectionAwareMessagePassing, self).__init__()
        self.cfg = config
        self.obj_classes = obj_classes
        self.num_obj_classes = len(obj_classes)
        in_channels = self.cfg.roi_dim
        self.use_gt_box = self.cfg.use_gt_box
        self.use_gt_label = self.cfg.use_gt_label
        # mode
        if self.cfg.use_gt_box:
            if self.cfg.use_gt_label:
                self.mode = 'predcls'
            else:
                self.mode = 'sgcls'
        else:
            self.mode = 'sgdet'

        # word embedding
        self.embed_dim = self.cfg.embed_dim
        self.obj_embed = nn.Embedding(self.num_obj_classes, self.embed_dim)
        obj_embed_vecs = obj_edge_vectors(self.obj_classes,
                                          wv_dir=self.cfg.glove_dir,
                                          wv_dim=self.embed_dim)
        with torch.no_grad():
            self.obj_embed.weight.copy_(obj_embed_vecs, non_blocking=True)

        # position embedding
        self.pos_embed = nn.Sequential(*[
            nn.Linear(9, 32),
            nn.BatchNorm1d(32, momentum=0.001),
            nn.Linear(32, 128),
            nn.ReLU(inplace=True),
        ])

        self.obj_dim = in_channels
        self.obj_input_dim = self.obj_dim + self.embed_dim + 128
        # 1024 + 200 + 128

        # set the direction-aware attention mapping
        self.ws = nn.Linear(self.obj_dim, self.obj_dim)
        self.wo = nn.Linear(self.obj_dim, self.obj_dim)
        self.wu = nn.Linear(self.obj_dim, self.obj_dim)
        self.w = nn.Linear(self.obj_dim, 1)

        # now begin to set the DMP
        self.project_input = nn.Sequential(*[
            nn.Linear(self.obj_input_dim, self.obj_dim),
            nn.ReLU(inplace=True)
        ])
        self.trans = nn.Sequential(*[
            nn.Linear(self.obj_dim, self.obj_dim // 4),
            nn.LayerNorm(self.obj_dim // 4),
            nn.ReLU(inplace=True),
            nn.Linear(self.obj_dim // 4, self.obj_dim)
        ])
        self.W_t3 = nn.Sequential(*[
            nn.Linear(self.obj_dim, self.obj_dim // 2),
            nn.ReLU(inplace=True)
        ])

        # object classifier
        self.out_obj = nn.Linear(self.obj_dim, self.num_obj_classes)

    def get_attention(self, obj_feat, union_feat, rel_pair_idx):
        num_obj = obj_feat.shape[0]
        atten_coeff = self.w(
            self.ws(obj_feat[rel_pair_idx[:, 0]]) *
            self.wo(obj_feat[rel_pair_idx[:, 1]]) * self.wu(union_feat))
        atten_tensor = torch.zeros(num_obj, num_obj, 1).to(atten_coeff)
        atten_tensor[rel_pair_idx[:, 0], rel_pair_idx[:, 1]] += atten_coeff
        atten_tensor = F.sigmoid(atten_tensor)
        atten_tensor = atten_tensor * (
            1 - torch.eye(num_obj).unsqueeze(-1).to(atten_tensor))

        atten_tensor_sum = torch.sum(atten_tensor, dim=1, keepdim=True)

        # handle 1 object case, avoid divideByZero
        if atten_tensor.shape[0] == 1:
            atten_tensor_sum = torch.ones(
                atten_tensor_sum.size()).to(atten_tensor_sum)
        # handle 1 object case done

        return atten_tensor / atten_tensor_sum

    def forward(self, obj_feats, union_feats, det_result):
        if self.training or self.use_gt_box:
            # predcls or sgcls or training, just put obj_labels here
            obj_labels = torch.cat(det_result.labels)
        else:
            obj_labels = None

        if self.use_gt_label:  # predcls
            obj_embed = self.obj_embed(obj_labels.long())
        else:
            obj_dists = torch.cat(det_result.dists, dim=0).detach()
            obj_embed = obj_dists @ self.obj_embed.weight
        pos_embed = self.pos_embed(encode_box_info(det_result))  # N x 128

        obj_rep = torch.cat((obj_feats, obj_embed, pos_embed),
                            -1)  # N x (1024 + 200 + 128)
        obj_rep = self.project_input(obj_rep)  # N x 1024

        rel_pair_idxes = det_result.rel_pair_idxes
        num_rels = [r.shape[0] for r in det_result.rel_pair_idxes]
        num_objs = [len(b) for b in det_result.bboxes]
        neighbour_feats = []
        split_obj_rep = obj_rep.split(num_objs)
        split_union_rep = union_feats.split(num_rels)
        for obj_feat, union_feat, rel_pair_idx in zip(split_obj_rep,
                                                      split_union_rep,
                                                      rel_pair_idxes):
            atten_tensor = self.get_attention(obj_feat, union_feat,
                                              rel_pair_idx)  # N x N x 1
            atten_tensor_t = torch.transpose(atten_tensor, 1, 0)
            atten_tensor = torch.cat((atten_tensor, atten_tensor_t),
                                     dim=-1)  # N x N x 2
            context_feats = matmul(atten_tensor, self.W_t3(obj_feat))
            neighbour_feats.append(self.trans(context_feats))

        obj_context_rep = F.relu(obj_rep + torch.cat(neighbour_feats, 0),
                                 inplace=True)

        if self.mode != 'predcls':
            obj_scores = self.out_obj(obj_context_rep)
            obj_dists = F.softmax(obj_scores, dim=1)
            obj_preds = obj_dists[:, 1:].max(1)[1] + 1
        else:
            assert obj_labels is not None
            obj_preds = obj_labels
            obj_scores = to_onehot(obj_preds, self.num_obj_classes)

        return obj_scores, obj_preds, obj_context_rep

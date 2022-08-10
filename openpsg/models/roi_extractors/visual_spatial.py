# ---------------------------------------------------------------
# visual_spatial.py
# Set-up time: 2020/4/28 下午8:46
# Copyright (c) 2020 ICT
# Licensed under The MIT License [see LICENSE for details]
# Written by Kenneth-Wong (Wenbin-Wang) @ VIPL.ICT
# Contact: wenbin.wang@vipl.ict.ac.cn [OR] nkwangwenbin@gmail.com
# ---------------------------------------------------------------

from __future__ import division

import numpy as np
import torch
import torch.nn as nn
from mmcv import ops
from mmcv.cnn import ConvModule, kaiming_init, normal_init
from mmcv.runner import BaseModule, force_fp32
from mmdet.models import ROI_EXTRACTORS
from torch.nn.modules.utils import _pair

from openpsg.models.relation_heads.approaches import PointNetFeat
from openpsg.utils.utils import enumerate_by_image


@ROI_EXTRACTORS.register_module()
class VisualSpatialExtractor(BaseModule):
    """Extract RoI features from a single level feature map.

    If there are multiple input feature levels, each RoI is mapped to a level
    according to its scale.

    Args:
        roi_layer (dict): Specify RoI layer type and arguments.
        out_channels (int): Output channels of RoI layers.
        featmap_strides (int): Strides of input feature maps.
        finest_scale (int): Scale threshold of mapping to level 0.
    """
    def __init__(
            self,
            bbox_roi_layer,
            in_channels,
            featmap_strides,
            roi_out_channels=256,
            fc_out_channels=1024,
            finest_scale=56,
            mask_roi_layer=None,
            with_avg_pool=False,
            with_visual_bbox=True,
            with_visual_mask=False,
            with_visual_point=False,
            with_spatial=False,
            separate_spatial=False,
            gather_visual='sum',
            conv_cfg=None,
            norm_cfg=dict(type='BN', requires_grad=True),
            init_cfg=None,
    ):
        super(VisualSpatialExtractor, self).__init__(init_cfg)
        self.roi_feat_size = _pair(bbox_roi_layer.get('output_size', 7))
        self.roi_feat_area = self.roi_feat_size[0] * self.roi_feat_size[1]
        self.in_channels = in_channels
        self.roi_out_channels = roi_out_channels
        self.fc_out_channels = fc_out_channels
        self.featmap_strides = featmap_strides
        self.finest_scale = finest_scale
        self.fp16_enabled = False
        self.with_avg_pool = with_avg_pool
        self.with_visual_bbox = with_visual_bbox
        self.with_visual_mask = with_visual_mask
        self.with_visual_point = with_visual_point
        self.with_spatial = with_spatial
        self.separate_spatial = separate_spatial
        self.gather_visual = gather_visual
        # NOTE: do not include the visual_point_head
        self.num_visual_head = int(self.with_visual_bbox) + int(
            self.with_visual_mask)
        if self.num_visual_head == 0:
            raise ValueError('There must be at least one visual head. ')

        in_channels = self.in_channels
        if self.with_avg_pool:
            self.avg_pool = nn.AvgPool2d(self.roi_feat_size)
        else:
            in_channels *= self.roi_feat_area

        # set some caches
        self._union_rois = None
        self._pair_rois = None

        # build visual head: extract visual features.
        if self.with_visual_bbox:
            assert bbox_roi_layer is not None
            self.bbox_roi_layers = self.build_roi_layers(
                bbox_roi_layer, featmap_strides)
            self.visual_bbox_head = nn.Sequential(*[
                nn.Linear(in_channels, self.fc_out_channels),
                nn.ReLU(inplace=True),
                nn.Linear(self.fc_out_channels, self.fc_out_channels),
                nn.ReLU(inplace=True),
            ])

        if self.with_visual_mask:
            assert mask_roi_layer is not None
            self.mask_roi_layers = self.build_roi_layers(
                mask_roi_layer, featmap_strides)
            self.visual_mask_head = nn.Sequential(*[
                nn.Linear(in_channels, self.fc_out_channels),
                nn.ReLU(inplace=True),
                nn.Linear(self.fc_out_channels, self.fc_out_channels),
                nn.ReLU(inplace=True),
            ])

        if self.with_visual_point:
            # TODO: build the point feats extraction head.
            self.pointFeatExtractor = PointNetFeat()

        if self.num_visual_head > 1:
            gather_in_channels = (self.fc_out_channels *
                                  2 if self.gather_visual == 'cat' else
                                  self.fc_out_channels)
            self.gather_visual_head = nn.Sequential(*[
                nn.Linear(gather_in_channels, self.fc_out_channels),
                nn.ReLU(inplace=True),
            ])

        # build spatial_head
        if self.with_spatial:
            self.spatial_size = self.roi_feat_size[0] * 4 - 1
            self.spatial_conv = nn.Sequential(*[
                ConvModule(
                    2,
                    self.in_channels // 2,
                    kernel_size=7,
                    stride=2,
                    padding=3,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    order=('conv', 'act', 'norm'),
                ),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                ConvModule(
                    self.in_channels // 2,
                    self.roi_out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    order=('conv', 'act', 'norm'),
                ),
            ])
            if self.separate_spatial:
                self.spatial_head = nn.Sequential(*[
                    nn.Linear(in_channels, self.fc_out_channels),
                    nn.ReLU(inplace=True),
                    nn.Linear(self.fc_out_channels, self.fc_out_channels),
                    nn.ReLU(inplace=True),
                ])

    @property
    def num_inputs(self):
        """int: Input feature map levels."""
        return len(self.featmap_strides)

    @property
    def union_rois(self):
        return self._union_rois

    @property
    def pair_rois(self):
        return self._pair_rois

    def init_weights(self):
        if self.with_visual_bbox:
            for m in self.visual_bbox_head:
                if isinstance(m, nn.Linear):
                    kaiming_init(m, distribution='uniform', a=1)
        if self.with_visual_mask:
            for m in self.visual_mask_head:
                if isinstance(m, nn.Linear):
                    kaiming_init(m, distribution='uniform', a=1)
        if self.with_visual_point:
            pass
            # for the pointNet head, just leave it there, do not
        if self.num_visual_head > 1:
            for m in self.gather_visual_head:
                if isinstance(m, nn.Linear):
                    kaiming_init(m, distribution='uniform', a=1)

        if self.with_spatial:
            for m in self.spatial_conv:
                if isinstance(m, ConvModule):
                    normal_init(m.conv, std=0.01)
            if self.separate_spatial:
                for m in self.spatial_head:
                    if isinstance(m, nn.Linear):
                        kaiming_init(m, distribution='uniform', a=1)

    def build_roi_layers(self, layer_cfg, featmap_strides):
        cfg = layer_cfg.copy()
        layer_type = cfg.pop('type')
        assert hasattr(ops, layer_type)
        layer_cls = getattr(ops, layer_type)
        roi_layers = nn.ModuleList(
            [layer_cls(spatial_scale=1 / s, **cfg) for s in featmap_strides])
        return roi_layers

    def map_roi_levels(self, rois, num_levels):
        """Map rois to corresponding feature levels by scales.

        - scale < finest_scale * 2: level 0
        - finest_scale * 2 <= scale < finest_scale * 4: level 1
        - finest_scale * 4 <= scale < finest_scale * 8: level 2
        - scale >= finest_scale * 8: level 3

        Args:
            rois (Tensor): Input RoIs, shape (k, 5).
            num_levels (int): Total level number.

        Returns:
            Tensor: Level index (0-based) of each RoI, shape (k, )
        """
        scale = torch.sqrt(
            (rois[:, 3] - rois[:, 1] + 1) * (rois[:, 4] - rois[:, 2] + 1))
        target_lvls = torch.floor(torch.log2(scale / self.finest_scale + 1e-6))
        target_lvls = target_lvls.clamp(min=0, max=num_levels - 1).long()
        return target_lvls

    def roi_rescale(self, rois, scale_factor):
        cx = (rois[:, 1] + rois[:, 3]) * 0.5
        cy = (rois[:, 2] + rois[:, 4]) * 0.5
        w = rois[:, 3] - rois[:, 1] + 1
        h = rois[:, 4] - rois[:, 2] + 1
        new_w = w * scale_factor
        new_h = h * scale_factor
        x1 = cx - new_w * 0.5 + 0.5
        x2 = cx + new_w * 0.5 - 0.5
        y1 = cy - new_h * 0.5 + 0.5
        y2 = cy + new_h * 0.5 - 0.5
        new_rois = torch.stack((rois[:, 0], x1, y1, x2, y2), dim=-1)
        return new_rois

    def roi_forward(self,
                    roi_layers,
                    feats,
                    rois,
                    masks=None,
                    roi_scale_factor=None):
        if len(feats) == 1:
            if roi_layers[0].__class__.__name__ == 'ShapeAwareRoIAlign':
                assert masks is not None
                roi_feats = roi_layers[0](feats[0], rois, masks)
            else:
                roi_feats = roi_layers[0](feats[0], rois)
        else:
            out_size = roi_layers[0].output_size
            num_levels = self.num_inputs
            target_lvls = self.map_roi_levels(rois, num_levels)
            roi_feats = feats[0].new_zeros(rois.size(0), self.roi_out_channels,
                                           *out_size)
            if roi_scale_factor is not None:
                assert masks is None  # not applicated for shape-aware roi align
                rois = self.roi_rescale(rois, roi_scale_factor)

            for i in range(num_levels):
                inds = target_lvls == i
                if inds.any():
                    rois_ = rois[inds, :]
                    if roi_layers[
                            i].__class__.__name__ == 'ShapeAwareRoIAlign':
                        masks_ = [
                            masks[idx] for idx in torch.nonzero(inds).view(-1)
                        ]
                        roi_feats_t = roi_layers[i](feats[i], rois_, masks_)
                    else:
                        roi_feats_t = roi_layers[i](feats[i], rois_)
                    roi_feats[inds] = roi_feats_t
        return roi_feats

    def single_roi_forward(self,
                           feats,
                           rois,
                           masks=None,
                           points=None,
                           roi_scale_factor=None):
        roi_feats_bbox, roi_feats_mask, roi_feats_point = None, None, None
        # 1. Use the visual and spatial head to extract roi features.
        if self.with_visual_bbox:
            roi_feats_bbox = self.roi_forward(self.bbox_roi_layers, feats,
                                              rois, masks, roi_scale_factor)
        if self.with_visual_mask:
            roi_feats_mask = self.roi_forward(self.mask_roi_layers, feats,
                                              rois, masks, roi_scale_factor)
        if self.with_visual_point:
            # input: (N_entity, Ndim(2), N_point)
            # output: (N_entity, feat_dim(1024))
            roi_feats_point, trans_matrix, _ = self.pointFeatExtractor(
                torch.stack(points).transpose(2, 1))

        roi_feats_result = []
        # gather the visual features, do not include the features from points
        for roi_feats, head in (
            (roi_feats_bbox, getattr(self, 'visual_bbox_head', None)),
            (roi_feats_mask, getattr(self, 'visual_mask_head', None)),
        ):
            if head is not None:
                roi_feats_result.append(
                    head(roi_feats.view(roi_feats.size(0), -1)))
        if self.num_visual_head > 1:
            if self.gather_visual == 'cat':
                roi_feats_result = torch.cat(roi_feats_result, dim=-1)
            elif self.gather_visual == 'sum':
                roi_feats_result = torch.stack(roi_feats_result).sum(0)
            elif self.gather_visual == 'prod':
                roi_feats_result = torch.stack(roi_feats_result).prod(0)
            else:
                raise NotImplementedError(
                    'The gathering operation {} is not implemented yet.'.
                    format(self.gather_visual))
            roi_feats = self.gather_visual_head(roi_feats_result)
        else:
            roi_feats = roi_feats_result[0]
        if self.with_visual_point:
            return (roi_feats, roi_feats_point, trans_matrix)
        else:
            return (roi_feats, )

    def union_roi_forward(
        self,
        feats,
        img_metas,
        rois,
        rel_pair_idx,
        masks=None,
        points=None,
        roi_scale_factor=None,
    ):
        assert self.with_spatial
        num_images = feats[0].size(0)
        assert num_images == len(rel_pair_idx)
        rel_pair_index = []
        im_inds = rois[:, 0]
        acc_obj = 0
        for i, s, e in enumerate_by_image(im_inds):
            num_obj_i = e - s
            rel_pair_idx_i = rel_pair_idx[i].clone()
            rel_pair_idx_i[:, 0] += acc_obj
            rel_pair_idx_i[:, 1] += acc_obj
            acc_obj += num_obj_i
            rel_pair_index.append(rel_pair_idx_i)
        rel_pair_index = torch.cat(rel_pair_index, 0)

        # prepare the union rois
        head_rois = rois[rel_pair_index[:, 0], :]
        tail_rois = rois[rel_pair_index[:, 1], :]

        head_rois_int = head_rois.cpu().numpy().astype(np.int32)
        tail_rois_int = tail_rois.cpu().numpy().astype(np.int32)
        union_rois = torch.stack(
            [
                head_rois[:, 0],
                torch.min(head_rois[:, 1], tail_rois[:, 1]),
                torch.min(head_rois[:, 2], tail_rois[:, 2]),
                torch.max(head_rois[:, 3], tail_rois[:, 3]),
                torch.max(head_rois[:, 4], tail_rois[:, 4]),
            ],
            -1,
        )

        self._union_rois = union_rois[:, 1:]
        self._pair_rois = torch.cat((head_rois[:, 1:], tail_rois[:, 1:]),
                                    dim=-1)

        # OPTIONAL: prepare the union masks
        union_masks = None
        if masks is not None and self.with_visual_mask:
            union_rois_int = union_rois.cpu().numpy().astype(np.int32)
            union_heights = union_rois_int[:, 4] - union_rois_int[:, 2] + 1
            union_widths = union_rois_int[:, 3] - union_rois_int[:, 1] + 1
            union_masks = []
            for i, pair_idx in enumerate(rel_pair_index.cpu().numpy()):
                head_mask, tail_mask = masks[pair_idx[0]], masks[pair_idx[1]]
                union_mask = torch.zeros(union_heights[i],
                                         union_widths[i]).to(head_mask)
                base_x, base_y = union_rois_int[i, 1], union_rois_int[i, 2]
                union_mask[(head_rois_int[i, 2] -
                            base_y):(head_rois_int[i, 4] - base_y + 1),
                           (head_rois_int[i, 1] -
                            base_x):(head_rois_int[i, 3] - base_x +
                                     1), ] = head_mask
                union_mask[(tail_rois_int[i, 2] -
                            base_y):(tail_rois_int[i, 4] - base_y + 1),
                           (tail_rois_int[i, 1] -
                            base_x):(tail_rois_int[i, 3] - base_x +
                                     1), ] = tail_mask
                union_masks.append(union_mask)

        # OPTIONAL: prepare the union points
        union_points = None
        if points is not None and self.with_visual_point:
            union_points = []
            for i, pair_idx in enumerate(rel_pair_index.cpu().numpy()):
                head_points, tail_points = points[pair_idx[0]], points[
                    pair_idx[1]]
                pts = torch.cat((head_points, tail_points), dim=0)
                union_points.append(pts)

        roi_feats_bbox, roi_feats_mask, roi_feats_point, rect_feats = (
            None,
            None,
            None,
            None,
        )

        # 1. Use the visual and spatial head to extract roi features.
        if self.with_visual_bbox:
            roi_feats_bbox = self.roi_forward(self.bbox_roi_layers, feats,
                                              union_rois, union_masks,
                                              roi_scale_factor)
        if self.with_visual_mask:
            roi_feats_mask = self.roi_forward(self.mask_roi_layers, feats,
                                              union_rois, union_masks,
                                              roi_scale_factor)
        if self.with_visual_point:
            roi_feats_point, trans_matrix, _ = self.pointFeatExtractor(
                torch.stack(union_points, dim=0).transpose(2, 1))

        # rect_feats: use range to construct rectangle, sized (rect_size, rect_size)
        num_rel = len(rel_pair_index)
        dummy_x_range = (torch.arange(self.spatial_size).to(
            rel_pair_index.device).view(1, 1,
                                        -1).expand(num_rel, self.spatial_size,
                                                   self.spatial_size))
        dummy_y_range = (torch.arange(self.spatial_size).to(
            rel_pair_index.device).view(1, -1,
                                        1).expand(num_rel, self.spatial_size,
                                                  self.spatial_size))
        size_list = [
            np.array(img_meta['img_shape'][:2]).reshape(1, -1)
            for img_meta in img_metas
        ]
        img_input_sizes = np.empty((0, 2), dtype=np.float32)
        for img_id in range(len(rel_pair_idx)):
            num_rel = len(rel_pair_idx[img_id])
            img_input_sizes = np.vstack(
                (img_input_sizes, np.tile(size_list[img_id], (num_rel, 1))))
        img_input_sizes = torch.from_numpy(img_input_sizes).to(rois)

        # resize bbox to the scale rect_size
        head_proposals = head_rois.clone()
        head_proposals[:, 1::2] *= self.spatial_size / img_input_sizes[:, 1:2]
        head_proposals[:, 2::2] *= self.spatial_size / img_input_sizes[:, 0:1]
        tail_proposals = tail_rois.clone()
        tail_proposals[:, 1::2] *= self.spatial_size / img_input_sizes[:, 1:2]
        tail_proposals[:, 2::2] *= self.spatial_size / img_input_sizes[:, 0:1]

        head_rect = ((dummy_x_range >= head_proposals[:, 1].floor().view(
            -1, 1, 1).long())
                     & (dummy_x_range <= head_proposals[:, 3].ceil().view(
                         -1, 1, 1).long())
                     & (dummy_y_range >= head_proposals[:, 2].floor().view(
                         -1, 1, 1).long())
                     & (dummy_y_range <= head_proposals[:, 4].ceil().view(
                         -1, 1, 1).long())).float()
        tail_rect = ((dummy_x_range >= tail_proposals[:, 1].floor().view(
            -1, 1, 1).long())
                     & (dummy_x_range <= tail_proposals[:, 2].ceil().view(
                         -1, 1, 1).long())
                     & (dummy_y_range >= tail_proposals[:, 3].floor().view(
                         -1, 1, 1).long())
                     & (dummy_y_range <= tail_proposals[:, 4].ceil().view(
                         -1, 1, 1).long())).float()

        rect_input = torch.stack((head_rect, tail_rect),
                                 dim=1)  # (num_rel, 2, rect_size, rect_size)

        rect_feats = self.spatial_conv(rect_input)

        # gather the different visual features and spatial features
        if self.separate_spatial:  # generally, it is False
            roi_feats_result = []
            for roi_feats, head in (
                (roi_feats_bbox, getattr(self, 'visual_bbox_head', None)),
                (roi_feats_mask, getattr(self, 'visual_mask_head', None)),
            ):
                if head is not None:
                    roi_feats_result.append(
                        head(roi_feats.view(roi_feats.size(0), -1)))
            if self.num_visual_head > 1:
                if self.gather_visual == 'cat':
                    roi_feats_result = torch.cat(roi_feats_result, dim=-1)
                elif self.gather_visual == 'sum':
                    roi_feats_result = torch.stack(roi_feats_result).sum(0)
                elif self.gather_visual == 'prod':
                    roi_feats_result = torch.stack(roi_feats_result).prod(0)
                else:
                    raise NotImplementedError(
                        'The gathering operation {} is not implemented yet.'.
                        format(self.gather_visual))
                roi_feats = self.gather_visual_head(roi_feats_result)
            else:
                roi_feats = roi_feats_result[0]
            roi_feats_spatial = self.spatial_head(rect_feats)
            if self.with_visual_point:
                return (roi_feats, roi_feats_spatial, roi_feats_point,
                        trans_matrix)
            else:
                return (roi_feats, roi_feats_spatial)
        else:
            roi_feats_result = []
            for roi_feats, head in (
                (roi_feats_bbox, getattr(self, 'visual_bbox_head', None)),
                (roi_feats_mask, getattr(self, 'visual_mask_head', None)),
            ):
                if head is not None:
                    roi_feats_result.append(
                        head((roi_feats + rect_feats).view(
                            roi_feats.size(0), -1)))
            if self.num_visual_head > 1:
                if self.gather_visual == 'cat':
                    roi_feats_result = torch.cat(roi_feats_result, dim=-1)
                elif self.gather_visual == 'sum':
                    roi_feats_result = torch.stack(roi_feats_result).sum(0)
                elif self.gather_visual == 'prod':
                    roi_feats_result = torch.stack(roi_feats_result).prod(0)
                else:
                    raise NotImplementedError(
                        'The gathering operation {} is not implemented yet.'.
                        format(self.gather_visual))
                roi_feats = self.gather_visual_head(roi_feats_result)
            else:
                roi_feats = roi_feats_result[0]
            if self.with_visual_point:
                return (roi_feats, roi_feats_point, trans_matrix)
            else:
                return (roi_feats, )

    @force_fp32(apply_to=('feats', ), out_fp16=True)
    def forward(
        self,
        feats,
        img_metas,
        rois,
        rel_pair_idx=None,
        masks=None,
        points=None,
        roi_scale_factor=None,
    ):
        if rois.shape[0] == 0:
            return torch.from_numpy(np.empty(
                (0, self.fc_out_channels))).to(feats[0])
        if self.with_spatial:
            assert rel_pair_idx is not None
            return self.union_roi_forward(feats, img_metas, rois, rel_pair_idx,
                                          masks, points, roi_scale_factor)
        else:
            return self.single_roi_forward(feats, rois, masks, points,
                                           roi_scale_factor)

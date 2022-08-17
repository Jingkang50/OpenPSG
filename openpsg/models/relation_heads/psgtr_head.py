# Copyright (c) OpenMMLab. All rights reserved.
import time
from collections import defaultdict
from inspect import signature

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from mmcv.cnn import Conv2d, Linear, build_activation_layer
from mmcv.cnn.bricks.transformer import FFN, build_positional_encoding
from mmcv.ops import batched_nms
from mmcv.runner import force_fp32
from mmdet.core import (bbox_cxcywh_to_xyxy, bbox_xyxy_to_cxcywh,
                        build_assigner, build_sampler, multi_apply,
                        reduce_mean)
from mmdet.datasets.coco_panoptic import INSTANCE_OFFSET
from mmdet.models.builder import HEADS, build_loss
from mmdet.models.dense_heads import AnchorFreeHead
from mmdet.models.utils import build_transformer
#####imports for tools
from packaging import version

if version.parse(torchvision.__version__) < version.parse('0.7'):
    from torchvision.ops import _new_empty_tensor
    from torchvision.ops.misc import _output_size


@HEADS.register_module()
class PSGTrHead(AnchorFreeHead):

    _version = 2

    def __init__(
            self,
            num_classes,
            in_channels,
            num_relations,
            object_classes,
            predicate_classes,
            use_mask=True,
            num_query=100,
            num_reg_fcs=2,
            transformer=None,
            n_heads=8,
            swin_backbone=None,
            sync_cls_avg_factor=False,
            bg_cls_weight=0.02,
            positional_encoding=dict(type='SinePositionalEncoding',
                                     num_feats=128,
                                     normalize=True),
            sub_loss_cls=dict(type='CrossEntropyLoss',
                              use_sigmoid=False,
                              loss_weight=1.0,
                              class_weight=1.0),
            sub_loss_bbox=dict(type='L1Loss', loss_weight=5.0),
            sub_loss_iou=dict(type='GIoULoss', loss_weight=2.0),
            sub_focal_loss=dict(type='BCEFocalLoss', loss_weight=1.0),
            sub_dice_loss=dict(type='DiceLoss', loss_weight=1.0),
            obj_loss_cls=dict(type='CrossEntropyLoss',
                              use_sigmoid=False,
                              loss_weight=1.0,
                              class_weight=1.0),
            obj_loss_bbox=dict(type='L1Loss', loss_weight=5.0),
            obj_loss_iou=dict(type='GIoULoss', loss_weight=2.0),
            obj_focal_loss=dict(type='BCEFocalLoss', loss_weight=1.0),
            obj_dice_loss=dict(type='DiceLoss', loss_weight=1.0),
            rel_loss_cls=dict(type='CrossEntropyLoss',
                              use_sigmoid=False,
                              loss_weight=2.0,
                              class_weight=1.0),
            train_cfg=dict(assigner=dict(
                type='HTriMatcher',
                s_cls_cost=dict(type='ClassificationCost', weight=1.),
                s_reg_cost=dict(type='BBoxL1Cost', weight=5.0),
                s_iou_cost=dict(type='IoUCost', iou_mode='giou', weight=2.0),
                o_cls_cost=dict(type='ClassificationCost', weight=1.),
                o_reg_cost=dict(type='BBoxL1Cost', weight=5.0),
                o_iou_cost=dict(type='IoUCost', iou_mode='giou', weight=2.0),
                r_cls_cost=dict(type='ClassificationCost', weight=2.))),
            test_cfg=dict(max_per_img=100),
            init_cfg=None,
            **kwargs):

        super(AnchorFreeHead, self).__init__(init_cfg)
        self.sync_cls_avg_factor = sync_cls_avg_factor
        # NOTE following the official DETR rep0, bg_cls_weight means
        # relative classification weight of the no-object class.
        assert isinstance(bg_cls_weight, float), 'Expected ' \
            'bg_cls_weight to have type float. Found ' \
            f'{type(bg_cls_weight)}.'
        self.bg_cls_weight = bg_cls_weight

        assert isinstance(use_mask, bool), 'Expected ' \
            'use_mask to have type bool. Found ' \
            f'{type(use_mask)}.'
        self.use_mask = use_mask

        s_class_weight = sub_loss_cls.get('class_weight', None)
        assert isinstance(s_class_weight, float), 'Expected ' \
            'class_weight to have type float. Found ' \
            f'{type(s_class_weight)}.'

        s_class_weight = torch.ones(num_classes + 1) * s_class_weight
        #NOTE set background class as the last indice
        s_class_weight[-1] = bg_cls_weight
        sub_loss_cls.update({'class_weight': s_class_weight})

        o_class_weight = obj_loss_cls.get('class_weight', None)
        assert isinstance(o_class_weight, float), 'Expected ' \
            'class_weight to have type float. Found ' \
            f'{type(o_class_weight)}.'

        o_class_weight = torch.ones(num_classes + 1) * o_class_weight
        #NOTE set background class as the last indice
        o_class_weight[-1] = bg_cls_weight
        obj_loss_cls.update({'class_weight': o_class_weight})

        r_class_weight = rel_loss_cls.get('class_weight', None)
        assert isinstance(r_class_weight, float), 'Expected ' \
            'class_weight to have type float. Found ' \
            f'{type(r_class_weight)}.'

        r_class_weight = torch.ones(num_relations + 1) * r_class_weight
        #NOTE set background class as the first indice for relations as they are 1-based
        r_class_weight[0] = bg_cls_weight
        rel_loss_cls.update({'class_weight': r_class_weight})
        if 'bg_cls_weight' in rel_loss_cls:
            rel_loss_cls.pop('bg_cls_weight')

        if train_cfg:
            assert 'assigner' in train_cfg, 'assigner should be provided '\
                'when train_cfg is set.'
            assigner = train_cfg['assigner']
            assert sub_loss_cls['loss_weight'] == assigner['s_cls_cost']['weight'], \
                'The classification weight for loss and matcher should be' \
                'exactly the same.'
            assert obj_loss_cls['loss_weight'] == assigner['o_cls_cost']['weight'], \
                'The classification weight for loss and matcher should be' \
                'exactly the same.'
            assert rel_loss_cls['loss_weight'] == assigner['r_cls_cost']['weight'], \
                'The classification weight for loss and matcher should be' \
                'exactly the same.'
            assert sub_loss_bbox['loss_weight'] == assigner['s_reg_cost'][
                'weight'], 'The regression L1 weight for loss and matcher ' \
                'should be exactly the same.'
            assert obj_loss_bbox['loss_weight'] == assigner['o_reg_cost'][
                'weight'], 'The regression L1 weight for loss and matcher ' \
                'should be exactly the same.'
            assert sub_loss_iou['loss_weight'] == assigner['s_iou_cost']['weight'], \
                'The regression iou weight for loss and matcher should be' \
                'exactly the same.'
            assert obj_loss_iou['loss_weight'] == assigner['o_iou_cost']['weight'], \
                'The regression iou weight for loss and matcher should be' \
                'exactly the same.'
            self.assigner = build_assigner(assigner)
            # following DETR sampling=False, so use PseudoSampler
            sampler_cfg = dict(type='PseudoSampler')
            self.sampler = build_sampler(sampler_cfg, context=self)
        self.num_query = num_query
        self.num_classes = num_classes
        self.num_relations = num_relations
        self.object_classes = object_classes
        self.predicate_classes = predicate_classes
        self.in_channels = in_channels
        self.num_reg_fcs = num_reg_fcs
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.fp16_enabled = False
        self.swin = swin_backbone

        self.obj_loss_cls = build_loss(obj_loss_cls)
        self.obj_loss_bbox = build_loss(obj_loss_bbox)
        self.obj_loss_iou = build_loss(obj_loss_iou)

        self.sub_loss_cls = build_loss(sub_loss_cls)
        self.sub_loss_bbox = build_loss(sub_loss_bbox)
        self.sub_loss_iou = build_loss(sub_loss_iou)
        if self.use_mask:
            # self.obj_focal_loss = build_loss(obj_focal_loss)
            self.obj_dice_loss = build_loss(obj_dice_loss)
            # self.sub_focal_loss = build_loss(sub_focal_loss)
            self.sub_dice_loss = build_loss(sub_dice_loss)

        self.rel_loss_cls = build_loss(rel_loss_cls)

        if self.obj_loss_cls.use_sigmoid:
            self.obj_cls_out_channels = num_classes
        else:
            self.obj_cls_out_channels = num_classes + 1

        if self.sub_loss_cls.use_sigmoid:
            self.sub_cls_out_channels = num_classes
        else:
            self.sub_cls_out_channels = num_classes + 1

        if rel_loss_cls['use_sigmoid']:
            self.rel_cls_out_channels = num_relations
        else:
            self.rel_cls_out_channels = num_relations + 1

        self.act_cfg = transformer.get('act_cfg',
                                       dict(type='ReLU', inplace=True))
        self.activate = build_activation_layer(self.act_cfg)
        self.positional_encoding = build_positional_encoding(
            positional_encoding)
        self.transformer = build_transformer(transformer)
        self.n_heads = n_heads
        self.embed_dims = self.transformer.embed_dims
        assert 'num_feats' in positional_encoding
        num_feats = positional_encoding['num_feats']
        assert num_feats * 2 == self.embed_dims, 'embed_dims should' \
            f' be exactly 2 times of num_feats. Found {self.embed_dims}' \
            f' and {num_feats}.'
        self._init_layers()

    def _init_layers(self):
        """Initialize layers of the transformer head."""
        self.input_proj = Conv2d(self.in_channels,
                                 self.embed_dims,
                                 kernel_size=1)
        self.query_embed = nn.Embedding(self.num_query, self.embed_dims)

        self.obj_cls_embed = Linear(self.embed_dims, self.obj_cls_out_channels)
        self.obj_box_embed = MLP(self.embed_dims, self.embed_dims, 4, 3)
        self.sub_cls_embed = Linear(self.embed_dims, self.sub_cls_out_channels)
        self.sub_box_embed = MLP(self.embed_dims, self.embed_dims, 4, 3)
        self.rel_cls_embed = Linear(self.embed_dims, self.rel_cls_out_channels)

        if self.use_mask:
            self.sub_bbox_attention = MHAttentionMap(self.embed_dims,
                                                     self.embed_dims,
                                                     self.n_heads,
                                                     dropout=0.0)
            self.obj_bbox_attention = MHAttentionMap(self.embed_dims,
                                                     self.embed_dims,
                                                     self.n_heads,
                                                     dropout=0.0)
            if not self.swin:
                self.sub_mask_head = MaskHeadSmallConv(
                    self.embed_dims + self.n_heads, [1024, 512, 256],
                    self.embed_dims)
                self.obj_mask_head = MaskHeadSmallConv(
                    self.embed_dims + self.n_heads, [1024, 512, 256],
                    self.embed_dims)
            elif self.swin:
                self.sub_mask_head = MaskHeadSmallConv(
                    self.embed_dims + self.n_heads, self.swin, self.embed_dims)
                self.obj_mask_head = MaskHeadSmallConv(
                    self.embed_dims + self.n_heads, self.swin, self.embed_dims)

    def init_weights(self):
        """Initialize weights of the transformer head."""
        # The initialization for transformer is important
        self.transformer.init_weights()

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        """load checkpoints."""
        version = local_metadata.get('version', None)
        if (version is None or version < 2):
            convert_dict = {
                '.self_attn.': '.attentions.0.',
                '.ffn.': '.ffns.0.',
                '.multihead_attn.': '.attentions.1.',
                '.decoder.norm.': '.decoder.post_norm.',
                '.query_embedding.': '.query_embed.'
            }
            state_dict_keys = list(state_dict.keys())
            for k in state_dict_keys:
                for ori_key, convert_key in convert_dict.items():
                    if ori_key in k:
                        convert_key = k.replace(ori_key, convert_key)
                        state_dict[convert_key] = state_dict[k]
                        del state_dict[k]
        super(AnchorFreeHead,
              self)._load_from_state_dict(state_dict, prefix, local_metadata,
                                          strict, missing_keys,
                                          unexpected_keys, error_msgs)

    def forward(self, feats, img_metas):
        # construct binary masks which used for the transformer.
        # NOTE following the official DETR repo, non-zero values representing
        # ignored positions, while zero values means valid positions.
        last_features = feats[
            -1]  ####get feature outputs of intermediate layers
        batch_size = last_features.size(0)
        input_img_h, input_img_w = img_metas[0]['batch_input_shape']
        masks = last_features.new_ones((batch_size, input_img_h, input_img_w))
        for img_id in range(batch_size):
            img_h, img_w, _ = img_metas[img_id]['img_shape']
            masks[img_id, :img_h, :img_w] = 0

        last_features = self.input_proj(last_features)
        # interpolate masks to have the same spatial shape with feats
        masks = F.interpolate(masks.unsqueeze(1),
                              size=last_features.shape[-2:]).to(
                                  torch.bool).squeeze(1)
        # position encoding
        pos_embed = self.positional_encoding(masks)  # [bs, embed_dim, h, w]
        # outs_dec: [nb_dec, bs, num_query, embed_dim]
        outs_dec, memory = self.transformer(last_features, masks,
                                            self.query_embed.weight, pos_embed)

        sub_outputs_class = self.sub_cls_embed(outs_dec)
        sub_outputs_coord = self.sub_box_embed(outs_dec).sigmoid()
        obj_outputs_class = self.obj_cls_embed(outs_dec)
        obj_outputs_coord = self.obj_box_embed(outs_dec).sigmoid()

        all_cls_scores = dict(sub=sub_outputs_class, obj=obj_outputs_class)
        rel_outputs_class = self.rel_cls_embed(outs_dec)
        all_cls_scores['rel'] = rel_outputs_class
        if self.use_mask:
            ###########for segmentation#################
            sub_bbox_mask = self.sub_bbox_attention(outs_dec[-1],
                                                    memory,
                                                    mask=masks)
            obj_bbox_mask = self.obj_bbox_attention(outs_dec[-1],
                                                    memory,
                                                    mask=masks)
            sub_seg_masks = self.sub_mask_head(last_features, sub_bbox_mask,
                                               [feats[2], feats[1], feats[0]])
            outputs_sub_seg_masks = sub_seg_masks.view(batch_size,
                                                       self.num_query,
                                                       sub_seg_masks.shape[-2],
                                                       sub_seg_masks.shape[-1])
            obj_seg_masks = self.obj_mask_head(last_features, obj_bbox_mask,
                                               [feats[2], feats[1], feats[0]])
            outputs_obj_seg_masks = obj_seg_masks.view(batch_size,
                                                       self.num_query,
                                                       obj_seg_masks.shape[-2],
                                                       obj_seg_masks.shape[-1])

            all_bbox_preds = dict(sub=sub_outputs_coord,
                                  obj=obj_outputs_coord,
                                  sub_seg=outputs_sub_seg_masks,
                                  obj_seg=outputs_obj_seg_masks)
        else:
            all_bbox_preds = dict(sub=sub_outputs_coord, obj=obj_outputs_coord)
        return all_cls_scores, all_bbox_preds

    @force_fp32(apply_to=('all_cls_scores_list', 'all_bbox_preds_list'))
    def loss(self,
             all_cls_scores_list,
             all_bbox_preds_list,
             gt_rels_list,
             gt_bboxes_list,
             gt_labels_list,
             gt_masks_list,
             img_metas,
             gt_bboxes_ignore=None):
        # NOTE defaultly only the outputs from the last feature scale is used.
        all_cls_scores = all_cls_scores_list
        all_bbox_preds = all_bbox_preds_list
        assert gt_bboxes_ignore is None, \
            'Only supports for gt_bboxes_ignore setting to None.'
        all_s_cls_scores = all_cls_scores['sub']
        all_o_cls_scores = all_cls_scores['obj']

        all_s_bbox_preds = all_bbox_preds['sub']
        all_o_bbox_preds = all_bbox_preds['obj']

        num_dec_layers = len(all_s_cls_scores)

        if self.use_mask:
            all_s_mask_preds = all_bbox_preds['sub_seg']
            all_o_mask_preds = all_bbox_preds['obj_seg']
            all_s_mask_preds = [
                all_s_mask_preds for _ in range(num_dec_layers)
            ]
            all_o_mask_preds = [
                all_o_mask_preds for _ in range(num_dec_layers)
            ]

        all_gt_bboxes_list = [gt_bboxes_list for _ in range(num_dec_layers)]
        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
        all_gt_rels_list = [gt_rels_list for _ in range(num_dec_layers)]
        all_gt_bboxes_ignore_list = [
            gt_bboxes_ignore for _ in range(num_dec_layers)
        ]
        all_gt_masks_list = [gt_masks_list for _ in range(num_dec_layers)]
        img_metas_list = [img_metas for _ in range(num_dec_layers)]

        all_r_cls_scores = [None for _ in range(num_dec_layers)]
        all_r_cls_scores = all_cls_scores['rel']

        if self.use_mask:
            # s_losses_cls, o_losses_cls, r_losses_cls, s_losses_bbox, o_losses_bbox, s_losses_iou, o_losses_iou, s_focal_losses, s_dice_losses, o_focal_losses, o_dice_losses = multi_apply(
            #     self.loss_single, all_s_cls_scores, all_o_cls_scores, all_r_cls_scores, all_s_bbox_preds, all_o_bbox_preds,
            #     all_s_mask_preds, all_o_mask_preds,
            #     all_gt_rels_list,all_gt_bboxes_list, all_gt_labels_list,
            #     all_gt_masks_list, img_metas_list,
            #     all_gt_bboxes_ignore_list)
            s_losses_cls, o_losses_cls, r_losses_cls, s_losses_bbox, o_losses_bbox, s_losses_iou, o_losses_iou, s_dice_losses, o_dice_losses = multi_apply(
                self.loss_single, all_s_cls_scores, all_o_cls_scores,
                all_r_cls_scores, all_s_bbox_preds, all_o_bbox_preds,
                all_s_mask_preds, all_o_mask_preds, all_gt_rels_list,
                all_gt_bboxes_list, all_gt_labels_list, all_gt_masks_list,
                img_metas_list, all_gt_bboxes_ignore_list)
        else:
            all_s_mask_preds = [None for _ in range(num_dec_layers)]
            all_o_mask_preds = [None for _ in range(num_dec_layers)]
            s_losses_cls, o_losses_cls, r_losses_cls, s_losses_bbox, o_losses_bbox, s_losses_iou, o_losses_iou, s_dice_losses, o_dice_losses = multi_apply(
                self.loss_single, all_s_cls_scores, all_o_cls_scores,
                all_r_cls_scores, all_s_bbox_preds, all_o_bbox_preds,
                all_s_mask_preds, all_o_mask_preds, all_gt_rels_list,
                all_gt_bboxes_list, all_gt_labels_list, all_gt_masks_list,
                img_metas_list, all_gt_bboxes_ignore_list)

        loss_dict = dict()
        # loss from the last decoder layer
        loss_dict['s_loss_cls'] = s_losses_cls[-1]
        loss_dict['o_loss_cls'] = o_losses_cls[-1]
        loss_dict['r_loss_cls'] = r_losses_cls[-1]
        loss_dict['s_loss_bbox'] = s_losses_bbox[-1]
        loss_dict['o_loss_bbox'] = o_losses_bbox[-1]
        loss_dict['s_loss_iou'] = s_losses_iou[-1]
        loss_dict['o_loss_iou'] = o_losses_iou[-1]
        if self.use_mask:
            # loss_dict['s_focal_losses'] = s_focal_losses[-1]
            # loss_dict['o_focal_losses'] = o_focal_losses[-1]
            loss_dict['s_dice_losses'] = s_dice_losses[-1]
            loss_dict['o_dice_losses'] = o_dice_losses[-1]

        # loss from other decoder layers
        num_dec_layer = 0
        for s_loss_cls_i, o_loss_cls_i, r_loss_cls_i, \
            s_loss_bbox_i, o_loss_bbox_i, \
            s_loss_iou_i, o_loss_iou_i in zip(s_losses_cls[:-1], o_losses_cls[:-1], r_losses_cls[:-1],
                                          s_losses_bbox[:-1], o_losses_bbox[:-1],
                                          s_losses_iou[:-1], o_losses_iou[:-1]):
            loss_dict[f'd{num_dec_layer}.s_loss_cls'] = s_loss_cls_i
            loss_dict[f'd{num_dec_layer}.o_loss_cls'] = o_loss_cls_i
            loss_dict[f'd{num_dec_layer}.r_loss_cls'] = r_loss_cls_i
            loss_dict[f'd{num_dec_layer}.s_loss_bbox'] = s_loss_bbox_i
            loss_dict[f'd{num_dec_layer}.o_loss_bbox'] = o_loss_bbox_i
            loss_dict[f'd{num_dec_layer}.s_loss_iou'] = s_loss_iou_i
            loss_dict[f'd{num_dec_layer}.o_loss_iou'] = o_loss_iou_i
            num_dec_layer += 1
        return loss_dict

    def loss_single(self,
                    s_cls_scores,
                    o_cls_scores,
                    r_cls_scores,
                    s_bbox_preds,
                    o_bbox_preds,
                    s_mask_preds,
                    o_mask_preds,
                    gt_rels_list,
                    gt_bboxes_list,
                    gt_labels_list,
                    gt_masks_list,
                    img_metas,
                    gt_bboxes_ignore_list=None):
        num_imgs = s_cls_scores.size(0)

        s_cls_scores_list = [s_cls_scores[i] for i in range(num_imgs)]
        o_cls_scores_list = [o_cls_scores[i] for i in range(num_imgs)]
        r_cls_scores_list = [r_cls_scores[i] for i in range(num_imgs)]
        s_bbox_preds_list = [s_bbox_preds[i] for i in range(num_imgs)]
        o_bbox_preds_list = [o_bbox_preds[i] for i in range(num_imgs)]

        if self.use_mask:
            s_mask_preds_list = [s_mask_preds[i] for i in range(num_imgs)]
            o_mask_preds_list = [o_mask_preds[i] for i in range(num_imgs)]
        else:
            s_mask_preds_list = [None for i in range(num_imgs)]
            o_mask_preds_list = [None for i in range(num_imgs)]

        cls_reg_targets = self.get_targets(
            s_cls_scores_list, o_cls_scores_list, r_cls_scores_list,
            s_bbox_preds_list, o_bbox_preds_list, s_mask_preds_list,
            o_mask_preds_list, gt_rels_list, gt_bboxes_list, gt_labels_list,
            gt_masks_list, img_metas, gt_bboxes_ignore_list)

        (s_labels_list, o_labels_list, r_labels_list, s_label_weights_list,
         o_label_weights_list, r_label_weights_list, s_bbox_targets_list,
         o_bbox_targets_list, s_bbox_weights_list, o_bbox_weights_list,
         s_mask_targets_list, o_mask_targets_list, num_total_pos,
         num_total_neg, s_mask_preds_list, o_mask_preds_list) = cls_reg_targets
        s_labels = torch.cat(s_labels_list, 0)
        o_labels = torch.cat(o_labels_list, 0)
        r_labels = torch.cat(r_labels_list, 0)

        s_label_weights = torch.cat(s_label_weights_list, 0)
        o_label_weights = torch.cat(o_label_weights_list, 0)
        r_label_weights = torch.cat(r_label_weights_list, 0)

        s_bbox_targets = torch.cat(s_bbox_targets_list, 0)
        o_bbox_targets = torch.cat(o_bbox_targets_list, 0)

        s_bbox_weights = torch.cat(s_bbox_weights_list, 0)
        o_bbox_weights = torch.cat(o_bbox_weights_list, 0)

        if self.use_mask:
            s_mask_targets = torch.cat(s_mask_targets_list,
                                       0).float().flatten(1)
            o_mask_targets = torch.cat(o_mask_targets_list,
                                       0).float().flatten(1)

            s_mask_preds = torch.cat(s_mask_preds_list, 0).flatten(1)
            o_mask_preds = torch.cat(o_mask_preds_list, 0).flatten(1)
            num_matches = o_mask_preds.shape[0]

            # mask loss
            # s_focal_loss = self.sub_focal_loss(s_mask_preds,s_mask_targets,num_matches)
            s_dice_loss = self.sub_dice_loss(
                s_mask_preds, s_mask_targets,
                num_matches)

            # o_focal_loss = self.obj_focal_loss(o_mask_preds,o_mask_targets,num_matches)
            o_dice_loss = self.obj_dice_loss(
                o_mask_preds, o_mask_targets,
                num_matches) 
        else:
            s_dice_loss = None
            o_dice_loss = None

        # classification loss
        s_cls_scores = s_cls_scores.reshape(-1, self.sub_cls_out_channels)
        o_cls_scores = o_cls_scores.reshape(-1, self.obj_cls_out_channels)
        r_cls_scores = r_cls_scores.reshape(-1, self.rel_cls_out_channels)

        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + \
            num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                s_cls_scores.new_tensor([cls_avg_factor]))
        cls_avg_factor = max(cls_avg_factor, 1)

        ###NOTE change cls_avg_factor for objects as we do not calculate object classification loss for unmatched ones

        s_loss_cls = self.sub_loss_cls(s_cls_scores,
                                       s_labels,
                                       s_label_weights,
                                       avg_factor=num_total_pos * 1.0)

        o_loss_cls = self.obj_loss_cls(o_cls_scores,
                                       o_labels,
                                       o_label_weights,
                                       avg_factor=num_total_pos * 1.0)

        r_loss_cls = self.rel_loss_cls(r_cls_scores,
                                       r_labels,
                                       r_label_weights,
                                       avg_factor=cls_avg_factor)

        # Compute the average number of gt boxes across all gpus, for
        # normalization purposes
        num_total_pos = o_loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # construct factors used for rescale bboxes
        factors = []
        for img_meta, bbox_pred in zip(img_metas, s_bbox_preds):
            img_h, img_w, _ = img_meta['img_shape']
            factor = bbox_pred.new_tensor([img_w, img_h, img_w,
                                           img_h]).unsqueeze(0).repeat(
                                               bbox_pred.size(0), 1)
            factors.append(factor)
        factors = torch.cat(factors, 0)

        # DETR regress the relative position of boxes (cxcywh) in the image,
        # thus the learning target is normalized by the image size. So here
        # we need to re-scale them for calculating IoU loss
        s_bbox_preds = s_bbox_preds.reshape(-1, 4)
        s_bboxes = bbox_cxcywh_to_xyxy(s_bbox_preds) * factors
        s_bboxes_gt = bbox_cxcywh_to_xyxy(s_bbox_targets) * factors

        o_bbox_preds = o_bbox_preds.reshape(-1, 4)
        o_bboxes = bbox_cxcywh_to_xyxy(o_bbox_preds) * factors
        o_bboxes_gt = bbox_cxcywh_to_xyxy(o_bbox_targets) * factors

        # regression IoU loss, defaultly GIoU loss
        s_loss_iou = self.sub_loss_iou(s_bboxes,
                                       s_bboxes_gt,
                                       s_bbox_weights,
                                       avg_factor=num_total_pos)
        o_loss_iou = self.obj_loss_iou(o_bboxes,
                                       o_bboxes_gt,
                                       o_bbox_weights,
                                       avg_factor=num_total_pos)

        # regression L1 loss
        s_loss_bbox = self.sub_loss_bbox(s_bbox_preds,
                                         s_bbox_targets,
                                         s_bbox_weights,
                                         avg_factor=num_total_pos)
        o_loss_bbox = self.obj_loss_bbox(o_bbox_preds,
                                         o_bbox_targets,
                                         o_bbox_weights,
                                         avg_factor=num_total_pos)
        # return s_loss_cls, o_loss_cls, r_loss_cls, s_loss_bbox, o_loss_bbox, s_loss_iou, o_loss_iou, s_focal_loss, s_dice_loss, o_focal_loss, o_dice_loss
        return s_loss_cls, o_loss_cls, r_loss_cls, s_loss_bbox, o_loss_bbox, s_loss_iou, o_loss_iou, s_dice_loss, o_dice_loss

    def get_targets(self,
                    s_cls_scores_list,
                    o_cls_scores_list,
                    r_cls_scores_list,
                    s_bbox_preds_list,
                    o_bbox_preds_list,
                    s_mask_preds_list,
                    o_mask_preds_list,
                    gt_rels_list,
                    gt_bboxes_list,
                    gt_labels_list,
                    gt_masks_list,
                    img_metas,
                    gt_bboxes_ignore_list=None):

        assert gt_bboxes_ignore_list is None, \
            'Only supports for gt_bboxes_ignore setting to None.'
        num_imgs = len(s_cls_scores_list)
        gt_bboxes_ignore_list = [
            gt_bboxes_ignore_list for _ in range(num_imgs)
        ]

        (s_labels_list, o_labels_list, r_labels_list, s_label_weights_list,
         o_label_weights_list, r_label_weights_list, s_bbox_targets_list,
         o_bbox_targets_list, s_bbox_weights_list, o_bbox_weights_list,
         s_mask_targets_list, o_mask_targets_list, pos_inds_list,
         neg_inds_list, s_mask_preds_list, o_mask_preds_list) = multi_apply(
             self._get_target_single, s_cls_scores_list, o_cls_scores_list,
             r_cls_scores_list, s_bbox_preds_list, o_bbox_preds_list,
             s_mask_preds_list, o_mask_preds_list, gt_rels_list,
             gt_bboxes_list, gt_labels_list, gt_masks_list, img_metas,
             gt_bboxes_ignore_list)
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (s_labels_list, o_labels_list, r_labels_list,
                s_label_weights_list, o_label_weights_list,
                r_label_weights_list, s_bbox_targets_list, o_bbox_targets_list,
                s_bbox_weights_list, o_bbox_weights_list, s_mask_targets_list,
                o_mask_targets_list, num_total_pos, num_total_neg,
                s_mask_preds_list, o_mask_preds_list)

    def _get_target_single(self,
                           s_cls_score,
                           o_cls_score,
                           r_cls_score,
                           s_bbox_pred,
                           o_bbox_pred,
                           s_mask_preds,
                           o_mask_preds,
                           gt_rels,
                           gt_bboxes,
                           gt_labels,
                           gt_masks,
                           img_meta,
                           gt_bboxes_ignore=None):
        """"Compute regression and classification targets for one image.

        Outputs from a single decoder layer of a single feature level are used.

        Args:
            s_cls_score (Tensor): Subject box score logits from a single decoder layer
                for one image. Shape [num_query, cls_out_channels].
            o_cls_score (Tensor): Object box score logits from a single decoder layer
                for one image. Shape [num_query, cls_out_channels].
            r_cls_score (Tensor): Relation score logits from a single decoder layer
                for one image. Shape [num_query, cls_out_channels].
            s_bbox_pred (Tensor): Sigmoid outputs of Subject bboxes from a single decoder layer
                for one image, with normalized coordinate (cx, cy, w, h) and
                shape [num_query, 4].
            o_bbox_pred (Tensor): Sigmoid outputs of object bboxes from a single decoder layer
                for one image, with normalized coordinate (cx, cy, w, h) and
                shape [num_query, 4].
            s_mask_preds (Tensor): Logits before sigmoid subject masks from a single decoder layer
                for one image, with shape [num_query, H, W].
            o_mask_preds (Tensor): Logits before sigmoid object masks from a single decoder layer
                for one image, with shape [num_query, H, W].
            gt_rels (Tensor): Ground truth relation triplets for one image with
                shape (num_gts, 3) in [gt_sub_id, gt_obj_id, gt_rel_class] format.
            gt_bboxes (Tensor): Ground truth bboxes for one image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (Tensor): Ground truth class indices for one image
                with shape (num_gts, ).
            img_meta (dict): Meta information for one image.
            gt_bboxes_ignore (Tensor, optional): Bounding boxes
                which can be ignored. Default None.

        Returns:
            tuple[Tensor]: a tuple containing the following for one image.

                - s/o/r_labels (Tensor): Labels of each image.
                - s/o/r_label_weights (Tensor]): Label weights of each image.
                - s/o_bbox_targets (Tensor): BBox targets of each image.
                - s/o_bbox_weights (Tensor): BBox weights of each image.
                - s/o_mask_targets (Tensor): Mask targets of each image.
                - pos_inds (Tensor): Sampled positive indices for each image.
                - neg_inds (Tensor): Sampled negative indices for each image.
                - s/o_mask_preds (Tensor): Matched mask preds of each image.
        """

        num_bboxes = s_bbox_pred.size(0)
        gt_sub_bboxes = []
        gt_obj_bboxes = []
        gt_sub_labels = []
        gt_obj_labels = []
        gt_rel_labels = []
        if self.use_mask:
            gt_sub_masks = []
            gt_obj_masks = []

        assert len(gt_masks) == len(gt_bboxes)

        for rel_id in range(gt_rels.size(0)):
            gt_sub_bboxes.append(gt_bboxes[int(gt_rels[rel_id, 0])])
            gt_obj_bboxes.append(gt_bboxes[int(gt_rels[rel_id, 1])])
            gt_sub_labels.append(gt_labels[int(gt_rels[rel_id, 0])])
            gt_obj_labels.append(gt_labels[int(gt_rels[rel_id, 1])])
            gt_rel_labels.append(gt_rels[rel_id, 2])
            if self.use_mask:
                gt_sub_masks.append(gt_masks[int(gt_rels[rel_id,
                                                         0])].unsqueeze(0))
                gt_obj_masks.append(gt_masks[int(gt_rels[rel_id,
                                                         1])].unsqueeze(0))

        gt_sub_bboxes = torch.vstack(gt_sub_bboxes).type_as(gt_bboxes).reshape(
            -1, 4)
        gt_obj_bboxes = torch.vstack(gt_obj_bboxes).type_as(gt_bboxes).reshape(
            -1, 4)
        gt_sub_labels = torch.vstack(gt_sub_labels).type_as(gt_labels).reshape(
            -1)
        gt_obj_labels = torch.vstack(gt_obj_labels).type_as(gt_labels).reshape(
            -1)
        gt_rel_labels = torch.vstack(gt_rel_labels).type_as(gt_labels).reshape(
            -1)

        # assigner and sampler, only return subject&object assign result
        s_assign_result, o_assign_result = self.assigner.assign(
            s_bbox_pred, o_bbox_pred, s_cls_score, o_cls_score, r_cls_score,
            gt_sub_bboxes, gt_obj_bboxes, gt_sub_labels, gt_obj_labels,
            gt_rel_labels, img_meta, gt_bboxes_ignore)

        s_sampling_result = self.sampler.sample(s_assign_result, s_bbox_pred,
                                                gt_sub_bboxes)
        o_sampling_result = self.sampler.sample(o_assign_result, o_bbox_pred,
                                                gt_obj_bboxes)
        pos_inds = o_sampling_result.pos_inds
        neg_inds = o_sampling_result.neg_inds  #### no-rel class indices in prediction

        # label targets
        s_labels = gt_sub_bboxes.new_full(
            (num_bboxes, ), self.num_classes,
            dtype=torch.long)  ### 0-based, class [num_classes]  as background
        s_labels[pos_inds] = gt_sub_labels[
            s_sampling_result.pos_assigned_gt_inds]
        s_label_weights = gt_sub_bboxes.new_zeros(num_bboxes)
        s_label_weights[pos_inds] = 1.0

        o_labels = gt_obj_bboxes.new_full(
            (num_bboxes, ), self.num_classes,
            dtype=torch.long)  ### 0-based, class [num_classes] as background
        o_labels[pos_inds] = gt_obj_labels[
            o_sampling_result.pos_assigned_gt_inds]
        o_label_weights = gt_obj_bboxes.new_zeros(num_bboxes)
        o_label_weights[pos_inds] = 1.0

        r_labels = gt_obj_bboxes.new_full(
            (num_bboxes, ), 0,
            dtype=torch.long)  ### 1-based, class 0 as background
        r_labels[pos_inds] = gt_rel_labels[
            o_sampling_result.pos_assigned_gt_inds]
        r_label_weights = gt_obj_bboxes.new_ones(num_bboxes)

        if self.use_mask:

            gt_sub_masks = torch.cat(gt_sub_masks, axis=0).type_as(gt_masks[0])
            gt_obj_masks = torch.cat(gt_obj_masks, axis=0).type_as(gt_masks[0])

            assert gt_sub_masks.size() == gt_obj_masks.size()
            # mask targets for subjects and objects
            s_mask_targets = gt_sub_masks[
                s_sampling_result.pos_assigned_gt_inds,
                ...]  
            s_mask_preds = s_mask_preds[pos_inds]
            

            o_mask_targets = gt_obj_masks[
                o_sampling_result.pos_assigned_gt_inds, ...]
            o_mask_preds = o_mask_preds[pos_inds]
            
            s_mask_preds = interpolate(s_mask_preds[:, None],
                                       size=gt_sub_masks.shape[-2:],
                                       mode='bilinear',
                                       align_corners=False).squeeze(1)

            o_mask_preds = interpolate(o_mask_preds[:, None],
                                       size=gt_obj_masks.shape[-2:],
                                       mode='bilinear',
                                       align_corners=False).squeeze(1)
        else:
            s_mask_targets = None
            s_mask_preds = None
            o_mask_targets = None
            o_mask_preds = None

        # bbox targets for subjects and objects
        s_bbox_targets = torch.zeros_like(s_bbox_pred)
        s_bbox_weights = torch.zeros_like(s_bbox_pred)
        s_bbox_weights[pos_inds] = 1.0

        o_bbox_targets = torch.zeros_like(o_bbox_pred)
        o_bbox_weights = torch.zeros_like(o_bbox_pred)
        o_bbox_weights[pos_inds] = 1.0
        img_h, img_w, _ = img_meta['img_shape']

        # DETR regress the relative position of boxes (cxcywh) in the image.
        # Thus the learning target should be normalized by the image size, also
        # the box format should be converted from defaultly x1y1x2y2 to cxcywh.
        factor = o_bbox_pred.new_tensor([img_w, img_h, img_w,
                                         img_h]).unsqueeze(0)

        pos_gt_s_bboxes_normalized = s_sampling_result.pos_gt_bboxes / factor
        pos_gt_s_bboxes_targets = bbox_xyxy_to_cxcywh(
            pos_gt_s_bboxes_normalized)
        s_bbox_targets[pos_inds] = pos_gt_s_bboxes_targets

        pos_gt_o_bboxes_normalized = o_sampling_result.pos_gt_bboxes / factor
        pos_gt_o_bboxes_targets = bbox_xyxy_to_cxcywh(
            pos_gt_o_bboxes_normalized)
        o_bbox_targets[pos_inds] = pos_gt_o_bboxes_targets

        return (s_labels, o_labels, r_labels, s_label_weights, o_label_weights,
                r_label_weights, s_bbox_targets, o_bbox_targets,
                s_bbox_weights, o_bbox_weights, s_mask_targets, o_mask_targets,
                pos_inds, neg_inds, s_mask_preds, o_mask_preds
                )  ###return the interpolated predicted masks

    # over-write because img_metas are needed as inputs for bbox_head.
    def forward_train(self,
                      x,
                      img_metas,
                      gt_rels,
                      gt_bboxes,
                      gt_labels=None,
                      gt_masks=None,
                      gt_bboxes_ignore=None,
                      proposal_cfg=None,
                      **kwargs):
        """Forward function for training mode.

        Args:
            x (list[Tensor]): Features from backbone.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_rels (Tensor): Ground truth relation triplets for one image with
                shape (num_gts, 3) in [gt_sub_id, gt_obj_id, gt_rel_class] format.
            gt_bboxes (Tensor): Ground truth bboxes of the image,
                shape (num_gts, 4).
            gt_labels (Tensor): Ground truth labels of each box,
                shape (num_gts,).
            gt_bboxes_ignore (Tensor): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 4).
            proposal_cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert proposal_cfg is None, '"proposal_cfg" must be None'
        outs = self(x, img_metas)
        if gt_labels is None:
            loss_inputs = outs + (gt_rels, gt_bboxes, gt_masks, img_metas)
        else:
            loss_inputs = outs + (gt_rels, gt_bboxes, gt_labels, gt_masks,
                                  img_metas)
        losses = self.loss(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        return losses

    @force_fp32(apply_to=('all_cls_scores_list', 'all_bbox_preds_list'))
    def get_bboxes(self, cls_scores, bbox_preds, img_metas, rescale=False):

        # NOTE defaultly only using outputs from the last feature level,
        # and only the outputs from the last decoder layer is used.

        result_list = []
        for img_id in range(len(img_metas)):
            s_cls_score = cls_scores['sub'][-1, img_id, ...]
            o_cls_score = cls_scores['obj'][-1, img_id, ...]
            r_cls_score = cls_scores['rel'][-1, img_id, ...]
            s_bbox_pred = bbox_preds['sub'][-1, img_id, ...]
            o_bbox_pred = bbox_preds['obj'][-1, img_id, ...]
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            if self.use_mask:
                s_mask_pred = bbox_preds['sub_seg'][img_id, ...]
                o_mask_pred = bbox_preds['obj_seg'][img_id, ...]
            else:
                s_mask_pred = None
                o_mask_pred = None
            triplets = self._get_bboxes_single(s_cls_score, o_cls_score,
                                               r_cls_score, s_bbox_pred,
                                               o_bbox_pred, s_mask_pred,
                                               o_mask_pred, img_shape,
                                               scale_factor, rescale)
            result_list.append(triplets)

        return result_list

    def _get_bboxes_single(self,
                           s_cls_score,
                           o_cls_score,
                           r_cls_score,
                           s_bbox_pred,
                           o_bbox_pred,
                           s_mask_pred,
                           o_mask_pred,
                           img_shape,
                           scale_factor,
                           rescale=False):

        assert len(s_cls_score) == len(o_cls_score)
        assert len(s_cls_score) == len(s_bbox_pred)
        assert len(s_cls_score) == len(o_bbox_pred)

        mask_size = (round(img_shape[0] / scale_factor[1]),
                     round(img_shape[1] / scale_factor[0]))
        max_per_img = self.test_cfg.get('max_per_img', self.num_query)

        assert self.sub_loss_cls.use_sigmoid == False
        assert self.obj_loss_cls.use_sigmoid == False
        assert self.rel_loss_cls.use_sigmoid == False
        assert len(s_cls_score) == len(r_cls_score)

        # 0-based label input for objects and self.num_classes as default background cls
        s_logits = F.softmax(s_cls_score, dim=-1)[..., :-1]
        o_logits = F.softmax(o_cls_score, dim=-1)[..., :-1]

        s_scores, s_labels = s_logits.max(-1)
        o_scores, o_labels = o_logits.max(-1)

        r_lgs = F.softmax(r_cls_score, dim=-1)
        r_logits = r_lgs[..., 1:]
        r_scores, r_indexes = r_logits.reshape(-1).topk(max_per_img)
        r_labels = r_indexes % self.num_relations + 1
        triplet_index = r_indexes // self.num_relations

        s_scores = s_scores[triplet_index]
        s_labels = s_labels[triplet_index] + 1
        s_bbox_pred = s_bbox_pred[triplet_index]

        o_scores = o_scores[triplet_index]
        o_labels = o_labels[triplet_index] + 1
        o_bbox_pred = o_bbox_pred[triplet_index]

        r_dists = r_lgs.reshape(
            -1, self.num_relations +
            1)[triplet_index]  #### NOTE: to match the evaluation in vg

        if self.use_mask:
            s_mask_pred = s_mask_pred[triplet_index]
            o_mask_pred = o_mask_pred[triplet_index]
            s_mask_pred = F.interpolate(s_mask_pred.unsqueeze(1),
                                        size=mask_size).squeeze(1)
            o_mask_pred = F.interpolate(o_mask_pred.unsqueeze(1),
                                        size=mask_size).squeeze(1)

            s_mask_pred_logits = s_mask_pred
            o_mask_pred_logits = o_mask_pred
            
            s_mask_pred = torch.sigmoid(s_mask_pred) > 0.85
            o_mask_pred = torch.sigmoid(o_mask_pred) > 0.85
            ### triplets deduplicate####
            relation_classes = defaultdict(lambda: [])
            for k, (s_l,o_l,r_l) in enumerate(zip(s_labels,o_labels,r_labels)):
                relation_classes[(s_l.item(),o_l.item(),r_l.item())].append(k)
            s_binary_masks = s_mask_pred.to(torch.float).flatten(1)
            o_binary_masks = o_mask_pred.to(torch.float).flatten(1)

            def dedup_triplets(triplets_ids, s_binary_masks, o_binary_masks, keep_tri):
                while len(triplets_ids) > 1:
                    base_s_mask = s_binary_masks[triplets_ids[0]].unsqueeze(0)
                    base_o_mask = o_binary_masks[triplets_ids[0]].unsqueeze(0)
                    other_s_masks = s_binary_masks[triplets_ids[1:]]
                    other_o_masks = o_binary_masks[triplets_ids[1:]]
                    # calculate ious
                    s_ious = base_s_mask.mm(other_s_masks.transpose(0,1))/((base_s_mask+other_s_masks)>0).sum(-1)
                    o_ious = base_o_mask.mm(other_o_masks.transpose(0,1))/((base_o_mask+other_o_masks)>0).sum(-1)
                    ids_left = []
                    for s_iou, o_iou, other_id in zip(s_ious[0],o_ious[0],triplets_ids[1:]):
                        if (s_iou>0.5) & (o_iou>0.5):
                            keep_tri[other_id] = False
                        else:
                            ids_left.append(other_id)
                    triplets_ids = ids_left
                return keep_tri
            
            keep_tri = torch.ones_like(r_labels,dtype=torch.bool)
            for triplets_ids in relation_classes.values():
                if len(triplets_ids)>1:
                    keep_tri = dedup_triplets(triplets_ids, s_binary_masks, o_binary_masks, keep_tri)

            s_labels = s_labels[keep_tri] 
            o_labels = o_labels[keep_tri]
            s_mask_pred = s_mask_pred[keep_tri]
            o_mask_pred = o_mask_pred[keep_tri]

            complete_labels = torch.cat((s_labels, o_labels), 0)
            output_masks = torch.cat((s_mask_pred, o_mask_pred), 0)
            r_scores = r_scores[keep_tri]
            r_labels = r_labels[keep_tri]
            r_dists = r_dists[keep_tri]
            rel_pairs = torch.arange(keep_tri.sum()*2,
                            dtype=torch.int).reshape(2, -1).T
            complete_r_labels = r_labels
            complete_r_dists = r_dists
            
            s_binary_masks = s_binary_masks[keep_tri]
            o_binary_masks = o_binary_masks[keep_tri]

            s_mask_pred_logits = s_mask_pred_logits[keep_tri]
            o_mask_pred_logits = o_mask_pred_logits[keep_tri]

            ###end triplets deduplicate####
            
            #### for panoptic postprocessing ####
            keep = (s_labels != (s_logits.shape[-1] - 1)) & (
                    o_labels != (s_logits.shape[-1] - 1)) & (
                    s_scores[keep_tri]>0.5) & (o_scores[keep_tri] > 0.5) & (r_scores > 0.3) ## the threshold is set to 0.85
            r_scores = r_scores[keep]
            r_labels = r_labels[keep]
            r_dists = r_dists[keep]

            labels = torch.cat((s_labels[keep], o_labels[keep]), 0) - 1
            masks = torch.cat((s_mask_pred[keep], o_mask_pred[keep]), 0)
            binary_masks = masks.to(torch.float).flatten(1)
            s_mask_pred_logits = s_mask_pred_logits[keep]
            o_mask_pred_logits = o_mask_pred_logits[keep]
            mask_logits = torch.cat((s_mask_pred_logits, o_mask_pred_logits), 0)

            h, w = masks.shape[-2:]

            if labels.numel() == 0:
                pan_img = torch.ones(mask_size).cpu().to(torch.long)
                pan_masks = pan_img.unsqueeze(0).cpu().to(torch.long)
                pan_rel_pairs = torch.arange(len(labels), dtype=torch.int).to(masks.device).reshape(2, -1).T
                rels = torch.tensor([0,0,0]).view(-1,3)
                pan_labels = torch.tensor([0])
            else:
                stuff_equiv_classes = defaultdict(lambda: [])
                thing_classes = defaultdict(lambda: [])
                thing_dedup = defaultdict(lambda: [])
                for k, label in enumerate(labels):
                    if label.item() >= 80:
                        stuff_equiv_classes[label.item()].append(k)
                    else:
                        thing_classes[label.item()].append(k)

                pan_rel_pairs = torch.arange(len(labels), dtype=torch.int).to(masks.device)

                def dedup_things(pred_ids, binary_masks):
                    while len(pred_ids) > 1:
                        base_mask = binary_masks[pred_ids[0]].unsqueeze(0)
                        other_masks = binary_masks[pred_ids[1:]]
                        # calculate ious
                        ious = base_mask.mm(other_masks.transpose(0,1))/((base_mask+other_masks)>0).sum(-1)
                        ids_left = []
                        thing_dedup[pred_ids[0]].append(pred_ids[0])
                        for iou, other_id in zip(ious[0],pred_ids[1:]):
                            if iou>0.5:
                                thing_dedup[pred_ids[0]].append(other_id)
                            else:
                                ids_left.append(other_id)
                        pred_ids = ids_left
                    if len(pred_ids) == 1:
                        thing_dedup[pred_ids[0]].append(pred_ids[0])

                # create dict that groups duplicate masks
                for thing_pred_ids in thing_classes.values():
                    if len(thing_pred_ids) > 1:
                      dedup_things(thing_pred_ids, binary_masks)
                    else:
                        thing_dedup[thing_pred_ids[0]].append(thing_pred_ids[0])

                def get_ids_area(masks, pan_rel_pairs, r_labels, r_dists, dedup=False):
                    # This helper function creates the final panoptic segmentation image
                    # It also returns the area of the masks that appears on the image
                    masks = masks.flatten(1)
                    m_id = masks.transpose(0, 1).softmax(-1)

                    if m_id.shape[-1] == 0:
                        # We didn't detect any mask :(
                        m_id = torch.zeros((h, w),
                                           dtype=torch.long,
                                           device=m_id.device)
                    else:
                        m_id = m_id.argmax(-1).view(h, w)

                    if dedup:
                        # Merge the masks corresponding to the same stuff class
                        for equiv in stuff_equiv_classes.values():
                            if len(equiv) > 1:
                                for eq_id in equiv:
                                    m_id.masked_fill_(m_id.eq(eq_id), equiv[0])
                                    pan_rel_pairs[eq_id] = equiv[0]
                        # Merge the masks corresponding to the same thing instance
                        for equiv in thing_dedup.values():
                            if len(equiv) > 1:
                                for eq_id in equiv:
                                    m_id.masked_fill_(m_id.eq(eq_id), equiv[0])
                                    pan_rel_pairs[eq_id] = equiv[0]
                    
                    m_ids_remain,_ = m_id.unique().sort()

                    pan_rel_pairs = pan_rel_pairs.reshape(2, -1).T
                    no_obj_filter = torch.zeros(pan_rel_pairs.shape[0],dtype=torch.bool)
                    for triplet_id in range(pan_rel_pairs.shape[0]):
                        if pan_rel_pairs[triplet_id,0] in m_ids_remain and pan_rel_pairs[triplet_id,1] in m_ids_remain:
                            no_obj_filter[triplet_id]=True
                    pan_rel_pairs = pan_rel_pairs[no_obj_filter]
                    r_labels, r_dists = r_labels[no_obj_filter], r_dists[no_obj_filter]
                    pan_labels = [] 
                    pan_masks = []
                    for i, m_id_remain in enumerate(m_ids_remain):
                        pan_masks.append(m_id.eq(m_id_remain).unsqueeze(0))
                        pan_labels.append(labels[m_id_remain].unsqueeze(0))
                        m_id.masked_fill_(m_id.eq(m_id_remain), i)
                        pan_rel_pairs.masked_fill_(pan_rel_pairs.eq(m_id_remain), i)
                    pan_masks = torch.cat(pan_masks, 0)
                    pan_labels = torch.cat(pan_labels, 0)
                    seg_img = m_id * INSTANCE_OFFSET + pan_labels[m_id]
                    seg_img = seg_img.view(h, w).cpu().to(torch.long)
                    m_id = m_id.view(h, w).cpu()
                    area = []
                    for i in range(len(masks)):
                        area.append(m_id.eq(i).sum().item())
                    return area, seg_img, pan_rel_pairs, pan_masks, r_labels, r_dists, pan_labels

                area, pan_img, pan_rel_pairs, pan_masks, r_labels, r_dists, pan_labels = get_ids_area(mask_logits, pan_rel_pairs, r_labels, r_dists, dedup=True)
                if r_labels.numel() == 0:
                    rels = torch.tensor([0,0,0]).view(-1,3)
                else:
                    rels = torch.cat((pan_rel_pairs,r_labels.unsqueeze(-1)),-1)
                # if labels.numel() > 0:
                #     # We know filter empty masks as long as we find some
                #     while True:
                #         filtered_small = torch.as_tensor(
                #             [area[i] <= 4 for i, c in enumerate(labels)],
                #             dtype=torch.bool,
                #             device=keep.device)
                #         if filtered_small.any().item():
                #             scores = scores[~filtered_small]
                #             labels = labels[~filtered_small]
                #             masks = masks[~filtered_small]
                #             area, pan_img = get_ids_area(masks, scores)
                #         else:
                #             break

        s_det_bboxes = bbox_cxcywh_to_xyxy(s_bbox_pred)
        s_det_bboxes[:, 0::2] = s_det_bboxes[:, 0::2] * img_shape[1]
        s_det_bboxes[:, 1::2] = s_det_bboxes[:, 1::2] * img_shape[0]
        s_det_bboxes[:, 0::2].clamp_(min=0, max=img_shape[1])
        s_det_bboxes[:, 1::2].clamp_(min=0, max=img_shape[0])
        if rescale:
            s_det_bboxes /= s_det_bboxes.new_tensor(scale_factor)
        s_det_bboxes = torch.cat((s_det_bboxes, s_scores.unsqueeze(1)), -1)

        o_det_bboxes = bbox_cxcywh_to_xyxy(o_bbox_pred)
        o_det_bboxes[:, 0::2] = o_det_bboxes[:, 0::2] * img_shape[1]
        o_det_bboxes[:, 1::2] = o_det_bboxes[:, 1::2] * img_shape[0]
        o_det_bboxes[:, 0::2].clamp_(min=0, max=img_shape[1])
        o_det_bboxes[:, 1::2].clamp_(min=0, max=img_shape[0])
        if rescale:
            o_det_bboxes /= o_det_bboxes.new_tensor(scale_factor)
        o_det_bboxes = torch.cat((o_det_bboxes, o_scores.unsqueeze(1)), -1)

        det_bboxes = torch.cat((s_det_bboxes[keep_tri], o_det_bboxes[keep_tri]), 0)

        if self.use_mask:
            return det_bboxes, complete_labels, rel_pairs, output_masks, pan_rel_pairs, \
                pan_img, complete_r_labels, complete_r_dists, r_labels, r_dists, pan_masks, rels, pan_labels
        else:
            return det_bboxes, labels, rel_pairs, r_labels, r_dists

    def simple_test_bboxes(self, feats, img_metas, rescale=False):
        
        # forward of this head requires img_metas
        # start = time.time()
        outs = self.forward(feats, img_metas)
        # forward_time =time.time()
        # print('------forward-----')
        # print(forward_time - start)
        results_list = self.get_bboxes(*outs, img_metas, rescale=rescale)
        # print('-----get_bboxes-----')
        # print(time.time() - forward_time)
        return results_list


class MLP(nn.Module):
    """Very simple multi-layer perceptron (also called FFN) Copied from
    hoitr."""
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def _expand(tensor, length: int):
    return tensor.unsqueeze(1).repeat(1, int(length), 1, 1, 1).flatten(0, 1)


class MaskHeadSmallConv(nn.Module):
    """Simple convolutional head, using group norm.

    Upsampling is done using a FPN approach
    """
    def __init__(self, dim, fpn_dims, context_dim):
        super().__init__()

        inter_dims = [
            dim, context_dim // 2, context_dim // 4, context_dim // 8,
            context_dim // 16, context_dim // 64
        ]
        self.lay1 = torch.nn.Conv2d(dim, dim, 3, padding=1)
        self.gn1 = torch.nn.GroupNorm(8, dim)
        self.lay2 = torch.nn.Conv2d(dim, inter_dims[1], 3, padding=1)
        self.gn2 = torch.nn.GroupNorm(8, inter_dims[1])
        self.lay3 = torch.nn.Conv2d(inter_dims[1], inter_dims[2], 3, padding=1)
        self.gn3 = torch.nn.GroupNorm(8, inter_dims[2])
        self.lay4 = torch.nn.Conv2d(inter_dims[2], inter_dims[3], 3, padding=1)
        self.gn4 = torch.nn.GroupNorm(8, inter_dims[3])
        self.lay5 = torch.nn.Conv2d(inter_dims[3], inter_dims[4], 3, padding=1)
        self.gn5 = torch.nn.GroupNorm(8, inter_dims[4])
        self.out_lay = torch.nn.Conv2d(inter_dims[4], 1, 3, padding=1)

        self.dim = dim

        self.adapter1 = torch.nn.Conv2d(fpn_dims[0], inter_dims[1], 1)
        self.adapter2 = torch.nn.Conv2d(fpn_dims[1], inter_dims[2], 1)
        self.adapter3 = torch.nn.Conv2d(fpn_dims[2], inter_dims[3], 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, bbox_mask, fpns):
        x = torch.cat(
            [_expand(x, bbox_mask.shape[1]),
             bbox_mask.flatten(0, 1)], 1)

        x = self.lay1(x)
        x = self.gn1(x)
        x = F.relu(x)
        x = self.lay2(x)
        x = self.gn2(x)
        x = F.relu(x)

        cur_fpn = self.adapter1(fpns[0])
        if cur_fpn.size(0) != x.size(0):
            cur_fpn = _expand(cur_fpn, x.size(0) // cur_fpn.size(0))
        x = cur_fpn + F.interpolate(x, size=cur_fpn.shape[-2:], mode='nearest')
        x = self.lay3(x)
        x = self.gn3(x)
        x = F.relu(x)

        cur_fpn = self.adapter2(fpns[1])
        if cur_fpn.size(0) != x.size(0):
            cur_fpn = _expand(cur_fpn, x.size(0) // cur_fpn.size(0))
        x = cur_fpn + F.interpolate(x, size=cur_fpn.shape[-2:], mode='nearest')
        x = self.lay4(x)
        x = self.gn4(x)
        x = F.relu(x)

        cur_fpn = self.adapter3(fpns[2])
        if cur_fpn.size(0) != x.size(0):
            cur_fpn = _expand(cur_fpn, x.size(0) // cur_fpn.size(0))
        x = cur_fpn + F.interpolate(x, size=cur_fpn.shape[-2:], mode='nearest')
        x = self.lay5(x)
        x = self.gn5(x)
        x = F.relu(x)

        x = self.out_lay(x)
        return x


class MHAttentionMap(nn.Module):
    """This is a 2D attention module, which only returns the attention softmax
    (no multiplication by value)"""
    def __init__(self,
                 query_dim,
                 hidden_dim,
                 num_heads,
                 dropout=0.0,
                 bias=True):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout)

        self.q_linear = nn.Linear(query_dim, hidden_dim, bias=bias)
        self.k_linear = nn.Linear(query_dim, hidden_dim, bias=bias)

        nn.init.zeros_(self.k_linear.bias)
        nn.init.zeros_(self.q_linear.bias)
        nn.init.xavier_uniform_(self.k_linear.weight)
        nn.init.xavier_uniform_(self.q_linear.weight)
        self.normalize_fact = float(hidden_dim / self.num_heads)**-0.5

    def forward(self, q, k, mask=None):
        q = self.q_linear(q)
        k = F.conv2d(k,
                     self.k_linear.weight.unsqueeze(-1).unsqueeze(-1),
                     self.k_linear.bias)
        qh = q.view(q.shape[0], q.shape[1], self.num_heads,
                    self.hidden_dim // self.num_heads)
        kh = k.view(k.shape[0], self.num_heads,
                    self.hidden_dim // self.num_heads, k.shape[-2],
                    k.shape[-1])
        weights = torch.einsum('bqnc,bnchw->bqnhw', qh * self.normalize_fact,
                               kh)

        if mask is not None:
            weights.masked_fill_(mask.unsqueeze(1).unsqueeze(1), float('-inf'))
        weights = F.softmax(weights.flatten(2), dim=-1).view(weights.size())
        weights = self.dropout(weights)
        return weights


def interpolate(input,
                size=None,
                scale_factor=None,
                mode='nearest',
                align_corners=None):
    """Equivalent to nn.functional.interpolate, but with support for empty
    batch sizes.

    This will eventually be supported natively by PyTorch, and this class can
    go away.
    """
    if version.parse(torchvision.__version__) < version.parse('0.7'):
        if input.numel() > 0:
            return torch.nn.functional.interpolate(input, size, scale_factor,
                                                   mode, align_corners)

        output_shape = _output_size(2, input, size, scale_factor)
        output_shape = list(input.shape[:-2]) + list(output_shape)
        return _new_empty_tensor(input, output_shape)
    else:
        return torchvision.ops.misc.interpolate(input, size, scale_factor,
                                                mode, align_corners)

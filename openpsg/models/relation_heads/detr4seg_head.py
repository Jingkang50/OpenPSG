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

coco_id = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20,
           21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
           41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58,
           59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79,
           80, 81, 82, 84, 85, 86, 87, 88, 89, 90, 92, 93, 95, 100, 107, 109,
           112, 118, 119, 122, 125, 128, 130, 133, 138, 141, 144, 145, 147,
           148, 149, 151, 154, 155, 156, 159, 161, 166, 168, 171, 175, 176,
           177, 178, 180, 181, 184, 185, 186, 187, 188, 189, 190, 191, 192,
           193, 194, 195, 196, 197, 198, 199, 200)


####default: 0-index with last index as the background class
@HEADS.register_module()
class detr4segHead(AnchorFreeHead):

    _version = 2

    def __init__(
            self,
            num_classes,
            in_channels,
            object_classes,
            num_query=100,
            num_reg_fcs=2,
            transformer=None,
            n_heads=8,
            swin_backbone=None,
            sync_cls_avg_factor=False,
            bg_cls_weight=0.1,
            positional_encoding=dict(type='SinePositionalEncoding',
                                     num_feats=128,
                                     normalize=True),
            loss_cls=dict(type='CrossEntropyLoss',
                          use_sigmoid=False,
                          loss_weight=1.0,
                          class_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=5.0),
            loss_iou=dict(type='GIoULoss', loss_weight=2.0),
            focal_loss=dict(type='BCEFocalLoss', loss_weight=1.0),
            dice_loss=dict(type='DiceLoss', loss_weight=1.0),
            train_cfg=dict(assigner=dict(
                type='HungarianAssigner',
                cls_cost=dict(type='ClassificationCost', weight=1.),
                reg_cost=dict(type='BBoxL1Cost', weight=5.0),
                iou_cost=dict(type='IoUCost', iou_mode='giou', weight=2.0))),
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

        class_weight = loss_cls.get('class_weight', None)
        assert isinstance(class_weight, float), 'Expected ' \
            'class_weight to have type float. Found ' \
            f'{type(class_weight)}.'

        class_weight = torch.ones(num_classes + 1) * class_weight
        # set background class as the last indice
        class_weight[num_classes] = bg_cls_weight
        loss_cls.update({'class_weight': class_weight})

        if train_cfg:
            assert 'assigner' in train_cfg, 'assigner should be provided '\
                'when train_cfg is set.'
            assigner = train_cfg['assigner']
            assert loss_cls['loss_weight'] == assigner['cls_cost']['weight'], \
                'The classification weight for loss and matcher should be' \
                'exactly the same.'
            assert loss_bbox['loss_weight'] == assigner['reg_cost'][
                'weight'], 'The regression L1 weight for loss and matcher ' \
                'should be exactly the same.'
            assert loss_iou['loss_weight'] == assigner['iou_cost']['weight'], \
                'The regression iou weight for loss and matcher should be' \
                'exactly the same.'
            self.assigner = build_assigner(assigner)
            # DETR sampling=False, so use PseudoSampler
            sampler_cfg = dict(type='PseudoSampler')
            self.sampler = build_sampler(sampler_cfg, context=self)
        self.num_query = num_query
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.num_reg_fcs = num_reg_fcs
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.fp16_enabled = False
        self.swin = swin_backbone
        self.CLASSES = object_classes

        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)
        self.loss_iou = build_loss(loss_iou)
        self.focal_loss = build_loss(focal_loss)
        self.dice_loss = build_loss(dice_loss)

        if self.loss_cls.use_sigmoid:
            self.cls_out_channels = num_classes
        else:
            self.cls_out_channels = num_classes + 1

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

        self.class_embed = Linear(self.embed_dims, self.cls_out_channels)
        self.bbox_embed = MLP(self.embed_dims, self.embed_dims, 4, 3)

        self.bbox_attention = MHAttentionMap(self.embed_dims,
                                             self.embed_dims,
                                             self.n_heads,
                                             dropout=0.0)
        if not self.swin:
            self.mask_head = MaskHeadSmallConv(self.embed_dims + self.n_heads,
                                               [1024, 512, 256],
                                               self.embed_dims)
        elif self.swin:
            self.mask_head = MaskHeadSmallConv(self.embed_dims + self.n_heads,
                                               self.swin, self.embed_dims)

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
        """Forward function.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.
            img_metas (list[dict]): List of image information.

        Returns:
            all_cls_scores [dict[Tensor]]: Outputs from the classification heads(human,object,action),
                shape [nb_dec, bs, num_query, cls_out_channels]. Note
                cls_out_channels should includes background.
            all_bbox_preds [dict[Tensor]]: Sigmoid outputs from the regression
                heads(human,object) with normalized coordinate format (cx, cy, w, h).
                Shape [nb_dec, bs, num_query, 4].
        """
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

        outputs_class = self.class_embed(outs_dec)
        outputs_coord = self.bbox_embed(outs_dec).sigmoid()

        all_cls_scores = outputs_class

        ###########for segmentation#################
        bbox_mask = self.bbox_attention(outs_dec[-1], memory, mask=masks)
        seg_masks = self.mask_head(last_features, bbox_mask,
                                   [feats[2], feats[1], feats[0]])
        seg_masks = seg_masks.view(batch_size, self.num_query,
                                   seg_masks.shape[-2], seg_masks.shape[-1])

        all_bbox_preds = dict(bbox=outputs_coord, masks=seg_masks)
        return all_cls_scores, all_bbox_preds

    @force_fp32(apply_to=('all_cls_scores_list', 'all_bbox_preds_list'))
    def loss(self,
             all_cls_scores_list,
             all_bbox_preds_list,
             gt_bboxes_list,
             gt_labels_list,
             gt_masks_list,
             img_metas,
             gt_bboxes_ignore=None):
        """"Loss function.

        Only outputs from the last feature level are used for computing
        losses by default.

        Args:
            all_cls_scores_list (list[dict[Tensor]]): Classification outputs
                for each feature level. Each is a 4D-tensor with shape
                [nb_dec, bs, num_query, cls_out_channels].
            all_bbox_preds_list (list[dict[Tensor]]): Sigmoid regression
                outputs for each feature level. Each is a 4D-tensor with
                normalized coordinate format (cx, cy, w, h) and shape
                [nb_dec, bs, num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            img_metas (list[dict]): List of image meta information.
            gt_bboxes_ignore (list[Tensor], optional): Bounding boxes
                which can be ignored for each image. Default None.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        # NOTE defaultly only the outputs from the last feature scale is used.
        all_cls_scores = all_cls_scores_list
        all_bbox_preds = all_bbox_preds_list
        assert gt_bboxes_ignore is None, \
            'Only supports for gt_bboxes_ignore setting to None.'
        all_mask_preds = all_bbox_preds['masks']

        all_bbox_preds = all_bbox_preds['bbox']

        num_dec_layers = len(all_cls_scores)

        all_mask_preds = [all_mask_preds for _ in range(num_dec_layers)]

        all_gt_bboxes_list = [gt_bboxes_list for _ in range(num_dec_layers)]
        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
        all_gt_bboxes_ignore_list = [
            gt_bboxes_ignore for _ in range(num_dec_layers)
        ]
        all_gt_masks_list = [gt_masks_list for _ in range(num_dec_layers)]
        img_metas_list = [img_metas for _ in range(num_dec_layers)]

        losses_cls, losses_bbox, losses_iou, dice_losses, focal_losses = multi_apply(
            self.loss_single, all_cls_scores, all_bbox_preds, all_mask_preds,
            all_gt_bboxes_list, all_gt_labels_list, all_gt_masks_list,
            img_metas_list, all_gt_bboxes_ignore_list)

        loss_dict = dict()
        # loss from the last decoder layer
        loss_dict['loss_cls'] = losses_cls[-1]
        loss_dict['loss_bbox'] = losses_bbox[-1]
        loss_dict['loss_iou'] = losses_iou[-1]

        loss_dict['focal_losses'] = focal_losses[-1]
        loss_dict['dice_losses'] = dice_losses[-1]

        # loss from other decoder layers
        num_dec_layer = 0
        for loss_cls_i, loss_bbox_i, loss_iou_i in zip(losses_cls[:-1],
                                                       losses_bbox[:-1],
                                                       losses_iou[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_bbox'] = loss_bbox_i
            loss_dict[f'd{num_dec_layer}.loss_iou'] = loss_iou_i
            num_dec_layer += 1
        return loss_dict

    def loss_single(self,
                    cls_scores,
                    bbox_preds,
                    mask_preds,
                    gt_bboxes_list,
                    gt_labels_list,
                    gt_masks_list,
                    img_metas,
                    gt_bboxes_ignore_list=None):
        """"Loss function for outputs from a single decoder layer of a single
        feature level.

        Args:
            cls_scores dict[Tensor]: Score logits from a single decoder layer
                for all images. Shape [bs, num_query, cls_out_channels].
            bbox_preds dict[Tensor]: Sigmoid outputs from a single decoder layer
                for all images, with normalized coordinate (cx, cy, w, h) and
                shape [bs, num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            img_metas (list[dict]): List of image meta information.
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.

        Returns:
            dict[str, Tensor]: A dictionary of loss components for outputs from
                a single decoder layer.
        """
        num_imgs = cls_scores.size(0)

        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]
        mask_preds_list = [mask_preds[i] for i in range(num_imgs)]

        cls_reg_targets = self.get_targets(cls_scores_list, bbox_preds_list,
                                           mask_preds_list, gt_bboxes_list,
                                           gt_labels_list, gt_masks_list,
                                           img_metas, gt_bboxes_ignore_list)

        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         mask_targets_list, num_total_pos, num_total_neg,
         mask_preds_list) = cls_reg_targets

        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)

        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)

        mask_targets = torch.cat(mask_targets_list, 0).float().flatten(1)

        mask_preds = torch.cat(mask_preds_list, 0).flatten(1)
        num_matches = mask_preds.shape[0]

        # mask loss
        focal_loss = self.focal_loss(mask_preds, mask_targets, num_matches)
        dice_loss = self.dice_loss(
            mask_preds, mask_targets,
            num_matches)  #,s_mask_weights,avg_factor=num_total_pos)

        # classification loss
        cls_scores = cls_scores.reshape(-1, self.cls_out_channels)

        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + \
            num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor]))
        cls_avg_factor = max(cls_avg_factor, 1)

        loss_cls = self.loss_cls(cls_scores,
                                 labels,
                                 label_weights,
                                 avg_factor=cls_avg_factor)

        # Compute the average number of gt boxes across all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # construct factors used for rescale bboxes
        factors = []
        for img_meta, bbox_pred in zip(img_metas, bbox_preds):
            img_h, img_w, _ = img_meta['img_shape']
            factor = bbox_pred.new_tensor([img_w, img_h, img_w,
                                           img_h]).unsqueeze(0).repeat(
                                               bbox_pred.size(0), 1)
            factors.append(factor)
        factors = torch.cat(factors, 0)

        # DETR regress the relative position of boxes (cxcywh) in the image,
        # thus the learning target is normalized by the image size. So here
        # we need to re-scale them for calculating IoU loss
        bbox_preds = bbox_preds.reshape(-1, 4)
        bboxes = bbox_cxcywh_to_xyxy(bbox_preds) * factors
        bboxes_gt = bbox_cxcywh_to_xyxy(bbox_targets) * factors

        # regression IoU loss, defaultly GIoU loss
        loss_iou = self.loss_iou(bboxes,
                                 bboxes_gt,
                                 bbox_weights,
                                 avg_factor=num_total_pos)

        # regression L1 loss
        loss_bbox = self.loss_bbox(bbox_preds,
                                   bbox_targets,
                                   bbox_weights,
                                   avg_factor=num_total_pos)

        return loss_cls, loss_bbox, loss_iou, dice_loss, focal_loss

    def get_targets(self,
                    cls_scores_list,
                    bbox_preds_list,
                    mask_preds_list,
                    gt_bboxes_list,
                    gt_labels_list,
                    gt_masks_list,
                    img_metas,
                    gt_bboxes_ignore_list=None):

        assert gt_bboxes_ignore_list is None, \
            'Only supports for gt_bboxes_ignore setting to None.'
        num_imgs = len(cls_scores_list)
        gt_bboxes_ignore_list = [
            gt_bboxes_ignore_list for _ in range(num_imgs)
        ]

        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         mask_targets_list, pos_inds_list, neg_inds_list,
         mask_preds_list) = multi_apply(self._get_target_single,
                                        cls_scores_list, bbox_preds_list,
                                        mask_preds_list, gt_bboxes_list,
                                        gt_labels_list, gt_masks_list,
                                        img_metas, gt_bboxes_ignore_list)
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (labels_list, label_weights_list, bbox_targets_list,
                bbox_weights_list, mask_targets_list, num_total_pos,
                num_total_neg, mask_preds_list)

    def _get_target_single(self,
                           cls_score,
                           bbox_pred,
                           mask_preds,
                           gt_bboxes,
                           gt_labels,
                           gt_masks,
                           img_meta,
                           gt_bboxes_ignore=None):

        num_bboxes = bbox_pred.size(0)
        assert len(gt_masks) == len(gt_bboxes)

        # print('o_pred mask shape after interpolating')
        # print(o_mask_preds.shape)

        # assigner and sampler, only return human&object assign result
        assign_result = self.assigner.assign(bbox_pred, cls_score, gt_bboxes,
                                             gt_labels, img_meta,
                                             gt_bboxes_ignore)
        sampling_result = self.sampler.sample(assign_result, bbox_pred,
                                              gt_bboxes)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds  #### no-rel class indices in prediction

        # label targets
        labels = gt_bboxes.new_full((num_bboxes, ),
                                    self.num_classes,
                                    dtype=torch.long)  ### 0-based
        labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        label_weights = gt_bboxes.new_ones(num_bboxes)

        # mask targets for subjects and objects
        mask_targets = gt_masks[sampling_result.pos_assigned_gt_inds,
                                ...]  ###FIXME some transform might be needed
        mask_preds = mask_preds[pos_inds]
        mask_preds = interpolate(mask_preds[:, None],
                                 size=gt_masks.shape[-2:],
                                 mode='bilinear',
                                 align_corners=False).squeeze(1)

        # bbox targets for subjects and objects
        bbox_targets = torch.zeros_like(bbox_pred)
        bbox_weights = torch.zeros_like(bbox_pred)
        bbox_weights[pos_inds] = 1.0

        img_h, img_w, _ = img_meta['img_shape']

        # DETR regress the relative position of boxes (cxcywh) in the image.
        # Thus the learning target should be normalized by the image size, also
        # the box format should be converted from defaultly x1y1x2y2 to cxcywh.
        factor = bbox_pred.new_tensor([img_w, img_h, img_w,
                                       img_h]).unsqueeze(0)

        pos_gt_bboxes_normalized = sampling_result.pos_gt_bboxes / factor
        pos_gt_bboxes_targets = bbox_xyxy_to_cxcywh(pos_gt_bboxes_normalized)
        bbox_targets[pos_inds] = pos_gt_bboxes_targets

        # print('---single--')
        # print(s_mask_targets.shape)

        return (labels, label_weights, bbox_targets, bbox_weights,
                mask_targets, pos_inds, neg_inds, mask_preds
                )  ###return the interpolated predicted masks

    # over-write because img_metas are needed as inputs for bbox_head.
    def forward_train(self,
                      x,
                      img_metas,
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
            loss_inputs = outs + (gt_bboxes, gt_masks, img_metas)
        else:
            loss_inputs = outs + (gt_bboxes, gt_labels, gt_masks, img_metas)
        losses = self.loss(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        return losses

    @force_fp32(apply_to=('all_cls_scores_list', 'all_bbox_preds_list'))
    def get_bboxes(self, cls_scores, bbox_preds, img_metas, rescale=False):
        """NOTE:Transform network outputs for a batch into psg predictions, but
        still use the name of get_bboxes for now.

        Args:
            all_cls_scores_list (list[Tensor]): Classification outputs
                for each feature level. Each is a 4D-tensor with shape
                [nb_dec, bs, num_query, cls_out_channels].
            all_bbox_preds_list (list[Tensor]): Sigmoid regression
                outputs for each feature level. Each is a 4D-tensor with
                normalized coordinate format (cx, cy, w, h) and shape
                [nb_dec, bs, num_query, 4].
            img_metas (list[dict]): Meta information of each image.
            rescale (bool, optional): If True, return boxes in original
                image space. Default False.

        Returns:
            list[list[Tensor, Tensor]]: Each item in result_list is 2-tuple. \
                The first item is an (n, 5) tensor, where the first 4 columns \
                are bounding box positions (tl_x, tl_y, br_x, br_y) and the \
                5-th column is a score between 0 and 1. The second item is a \
                (n,) tensor where each item is the predicted class label of \
                the corresponding box.
        """
        # NOTE defaultly only using outputs from the last feature level,
        # and only the outputs from the last decoder layer is used.

        result_list = []
        for img_id in range(len(img_metas)):
            cls_score = cls_scores[-1, img_id, ...]
            bbox_pred = bbox_preds['bbox'][-1, img_id, ...]
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            mask_pred = bbox_preds['masks'][img_id, ...]
            triplets = self._get_bboxes_single(cls_score, bbox_pred, mask_pred,
                                               img_shape, scale_factor,
                                               rescale)
            result_list.append(triplets)

        return result_list

    def _get_bboxes_single(self,
                           cls_score,
                           bbox_pred,
                           mask_pred,
                           img_shape,
                           scale_factor,
                           rescale=False):
        """Transform outputs from the last decoder layer into bbox predictions
        for each image.

        Args:
            h_cls_score/o_cls_score/i_cls_score (dict[Tensor]): Box score logits from the last decoder layer
                for each image. Each tensor shape [num_query, h/o/i_cls_out_channels].
            h_bbox_pred/o_bbox_pred (dict[Tensor]): Sigmoid outputs from the last decoder layer
                for each image, each tensor with coordinate format (cx, cy, w, h) and
                shape [num_query, 4].
            img_shape (tuple[int]): Shape of input image, (height, width, 3).
            scale_factor (ndarray, optional): Scale factor of the image arange
                as (w_scale, h_scale, w_scale, h_scale).
            rescale (bool, optional): If True, return boxes in original image
                space. Default False.

        Returns:
            tuple[Tensor]: Results of detected bboxes and labels.

                - det_bboxes: Predicted bboxes with shape [num_query, 5], \
                    where the first 4 columns are bounding box positions \
                    (tl_x, tl_y, br_x, br_y) and the 5-th column are scores \
                    between 0 and 1.
                - det_labels: Predicted labels of the corresponding box with \
                    shape [num_query].
        """

        mask_size = (round(img_shape[0] / scale_factor[1]),
                     round(img_shape[1] / scale_factor[0]))
        max_per_img = self.test_cfg.get('max_per_img', self.num_query)

        # 1-based label input and 0 as default background cls

        logits = F.softmax(cls_score, dim=-1)
        scores, labels = logits.max(-1)
        scores, bbox_index = scores.topk(max_per_img)
        bbox_pred = bbox_pred[bbox_index]
        labels = labels[bbox_index]
        mask_pred = mask_pred[bbox_index]

        keep = (labels != logits.shape[-1] - 1) & (
            scores > 0.85)  ## the threshold is set to 0.85
        bbox_pred = bbox_pred[keep]
        det_labels = labels[keep]
        det_masks = mask_pred[keep]
        scores = scores[keep]
        det_masks = F.interpolate(det_masks.unsqueeze(1),
                                  size=mask_size,
                                  mode='bilinear').squeeze(1)

        h, w = det_masks.shape[-2:]

        assert len(det_labels) == len(bbox_pred)
        det_bboxes = bbox_cxcywh_to_xyxy(bbox_pred)
        det_bboxes[:, 0::2] = det_bboxes[:, 0::2] * img_shape[1]
        det_bboxes[:, 1::2] = det_bboxes[:, 1::2] * img_shape[0]
        det_bboxes[:, 0::2].clamp_(min=0, max=img_shape[1])
        det_bboxes[:, 1::2].clamp_(min=0, max=img_shape[0])
        if rescale:
            det_bboxes /= det_bboxes.new_tensor(scale_factor)
        det_bboxes = torch.cat((det_bboxes, scores.unsqueeze(1)), -1)
        bbox_labels = det_labels
        if det_labels.numel() == 0:
            pan_img = torch.ones(mask_size).cpu().to(torch.long)
            return det_bboxes, bbox_labels, pan_img

        # It may be that we have several predicted masks for the same stuff class.
        # In the following, we track the list of masks ids for each stuff class (they are merged later on)
        det_masks = det_masks.flatten(1)
        stuff_equiv_classes = defaultdict(lambda: [])
        for k, label in enumerate(det_labels):
            if label.item() >= 80:
                stuff_equiv_classes[label.item()].append(k)

        def get_ids_area(masks, scores, dedup=False):
            # This helper function creates the final panoptic segmentation image
            # It also returns the area of the masks that appears on the image

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

            seg_img = m_id * INSTANCE_OFFSET + det_labels[m_id]
            seg_img = seg_img.view(h, w).cpu().to(torch.long)
            m_id = m_id.view(h, w).cpu()
            area = []
            for i in range(len(scores)):
                area.append(m_id.eq(i).sum().item())
            return area, seg_img

        area, pan_img = get_ids_area(det_masks, scores, dedup=True)
        if det_labels.numel() > 0:
            # We know filter empty masks as long as we find some
            while True:
                filtered_small = torch.as_tensor(
                    [area[i] <= 4 for i, c in enumerate(det_labels)],
                    dtype=torch.bool,
                    device=keep.device)
                if filtered_small.any().item():
                    scores = scores[~filtered_small]
                    det_labels = det_labels[~filtered_small]
                    det_masks = det_masks[~filtered_small]
                    area, pan_img = get_ids_area(det_masks, scores)
                else:
                    break

        return det_bboxes, bbox_labels, pan_img

    def simple_test_bboxes(self, feats, img_metas, rescale=False):
        """Test det bboxes without test-time augmentation.

        Args:
            feats (tuple[torch.Tensor]): Multi-level features from the
                upstream network, each is a 4D-tensor.
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[tuple[Tensor]]: Each item in result_list is 6-tuple:
                bbox, labels, bbox, labels, scores, labels
                The first item is ``bboxes`` with shape (n, 5),
                where 5 represent (tl_x, tl_y, br_x, br_y, score).
                The shape of the second tensor in the tuple is ``labels``
                with shape (n,)
        """
        # forward of this head requires img_metas
        outs = self.forward(feats, img_metas)
        results_list = self.get_bboxes(*outs, img_metas, rescale=rescale)
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

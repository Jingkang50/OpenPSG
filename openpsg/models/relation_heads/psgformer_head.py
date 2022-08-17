# Copyright (c) OpenMMLab. All rights reserved.
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from mmcv.cnn import Conv2d, Linear, build_activation_layer
from mmcv.cnn.bricks.transformer import build_positional_encoding
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
class PSGFormerHead(AnchorFreeHead):

    _version = 2

    def __init__(self,
                 num_classes,
                 in_channels,
                 num_relations,
                 object_classes,
                 predicate_classes,
                 num_obj_query=100,
                 num_rel_query=100,
                 num_reg_fcs=2,
                 use_mask=True,
                 temp=0.1,
                 transformer=None,
                 n_heads=8,
                 sync_cls_avg_factor=False,
                 bg_cls_weight=0.02,
                 positional_encoding=dict(type='SinePositionalEncoding',
                                          num_feats=128,
                                          normalize=True),
                 rel_loss_cls=dict(type='CrossEntropyLoss',
                                   use_sigmoid=False,
                                   loss_weight=2.0,
                                   class_weight=1.0),
                 sub_id_loss=dict(type='CrossEntropyLoss',
                                  use_sigmoid=False,
                                  loss_weight=2.0,
                                  class_weight=1.0),
                 obj_id_loss=dict(type='CrossEntropyLoss',
                                  use_sigmoid=False,
                                  loss_weight=2.0,
                                  class_weight=1.0),
                 loss_cls=dict(type='CrossEntropyLoss',
                               use_sigmoid=False,
                               loss_weight=1.0,
                               class_weight=1.0),
                 loss_bbox=dict(type='L1Loss', loss_weight=5.0),
                 loss_iou=dict(type='GIoULoss', loss_weight=2.0),
                 focal_loss=dict(type='BCEFocalLoss', loss_weight=1.0),
                 dice_loss=dict(type='DiceLoss', loss_weight=1.0),
                 train_cfg=dict(id_assigner=dict(
                     type='IdMatcher',
                     sub_id_cost=dict(type='ClassificationCost', weight=1.),
                     obj_id_cost=dict(type='ClassificationCost', weight=1.),
                     r_cls_cost=dict(type='ClassificationCost', weight=1.)),
                                bbox_assigner=dict(
                                    type='HungarianAssigner',
                                    cls_cost=dict(type='ClassificationCost',
                                                  weight=1.),
                                    reg_cost=dict(type='BBoxL1Cost',
                                                  weight=5.0),
                                    iou_cost=dict(type='IoUCost',
                                                  iou_mode='giou',
                                                  weight=2.0))),
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
            assert 'id_assigner' in train_cfg, 'id_assigner should be provided '\
                'when train_cfg is set.'
            assert 'bbox_assigner' in train_cfg, 'bbox_assigner should be provided '\
                'when train_cfg is set.'
            id_assigner = train_cfg['id_assigner']
            bbox_assigner = train_cfg['bbox_assigner']
            assert loss_cls['loss_weight'] == bbox_assigner['cls_cost']['weight'], \
                'The classification weight for loss and matcher should be' \
                'exactly the same.'
            assert loss_bbox['loss_weight'] == bbox_assigner['reg_cost'][
                'weight'], 'The regression L1 weight for loss and matcher ' \
                'should be exactly the same.'
            assert loss_iou['loss_weight'] == bbox_assigner['iou_cost']['weight'], \
                'The regression iou weight for loss and matcher should be' \
                'exactly the same.'
            self.id_assigner = build_assigner(id_assigner)
            self.bbox_assigner = build_assigner(bbox_assigner)
            # DETR sampling=False, so use PseudoSampler
            sampler_cfg = dict(type='PseudoSampler')
            self.sampler = build_sampler(sampler_cfg, context=self)
        assert num_obj_query == num_rel_query
        self.num_obj_query = num_obj_query
        self.num_rel_query = num_rel_query
        self.use_mask = use_mask
        self.temp = temp
        self.num_classes = num_classes
        self.num_relations = num_relations
        self.object_classes = object_classes
        self.predicate_classes = predicate_classes
        self.in_channels = in_channels
        self.num_reg_fcs = num_reg_fcs
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.fp16_enabled = False

        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)
        self.loss_iou = build_loss(loss_iou)
        self.focal_loss = build_loss(focal_loss)
        self.dice_loss = build_loss(dice_loss)
        self.rel_loss_cls = build_loss(rel_loss_cls)

        ### id losses
        self.sub_id_loss = build_loss(sub_id_loss)
        self.obj_id_loss = build_loss(obj_id_loss)

        if self.loss_cls.use_sigmoid:
            self.cls_out_channels = num_classes
        else:
            self.cls_out_channels = num_classes + 1

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
        self.obj_query_embed = nn.Embedding(self.num_obj_query,
                                            self.embed_dims)
        self.rel_query_embed = nn.Embedding(self.num_rel_query,
                                            self.embed_dims)

        self.class_embed = Linear(self.embed_dims, self.cls_out_channels)
        self.box_embed = MLP(self.embed_dims, self.embed_dims, 4, 3)

        self.sub_query_update = nn.Sequential(
            Linear(self.embed_dims, self.embed_dims), nn.ReLU(inplace=True),
            Linear(self.embed_dims, self.embed_dims))

        self.obj_query_update = nn.Sequential(
            Linear(self.embed_dims, self.embed_dims), nn.ReLU(inplace=True),
            Linear(self.embed_dims, self.embed_dims))

        self.sop_query_update = nn.Sequential(
            Linear(2 * self.embed_dims, self.embed_dims),
            nn.ReLU(inplace=True), Linear(self.embed_dims, self.embed_dims))

        self.rel_query_update = nn.Identity()

        self.rel_cls_embed = Linear(self.embed_dims, self.rel_cls_out_channels)

        self.bbox_attention = MHAttentionMap(self.embed_dims,
                                             self.embed_dims,
                                             self.n_heads,
                                             dropout=0.0)
        self.mask_head = MaskHeadSmallConv(self.embed_dims + self.n_heads,
                                           [1024, 512, 256], self.embed_dims)

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
                '.decoder1.norm.': '.decoder1.post_norm.',
                '.decoder2.norm.': '.decoder2.post_norm.',
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

    def forward(self, feats, img_metas, train_mode=False):

        # construct binary masks which used for the transformer.
        # NOTE following the official DETR repo, non-zero values representing
        # ignored positions, while zero values means valid positions.
        last_features = feats[-1]
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
        outs_obj_dec, outs_rel_dec, memory \
            = self.transformer(last_features, masks,
                               self.obj_query_embed.weight,
                               self.rel_query_embed.weight,
                               pos_embed)

        outputs_class = self.class_embed(outs_obj_dec)
        outputs_coord = self.box_embed(outs_obj_dec).sigmoid()
        bbox_mask = self.bbox_attention(outs_obj_dec[-1], memory, mask=masks)
        seg_masks = self.mask_head(last_features, bbox_mask,
                                   [feats[2], feats[1], feats[0]])
        seg_masks = seg_masks.view(batch_size, self.num_obj_query,
                                   seg_masks.shape[-2], seg_masks.shape[-1])

        ### interaction
        updated_sub_embed = self.sub_query_update(outs_obj_dec)
        updated_obj_embed = self.obj_query_update(outs_obj_dec)
        sub_q_normalized = F.normalize(updated_sub_embed[-1],
                                       p=2,
                                       dim=-1,
                                       eps=1e-12)
        obj_q_normalized = F.normalize(updated_obj_embed[-1],
                                       p=2,
                                       dim=-1,
                                       eps=1e-12)

        updated_rel_embed = self.rel_query_update(outs_rel_dec)
        rel_q_normalized = F.normalize(updated_rel_embed[-1],
                                       p=2,
                                       dim=-1,
                                       eps=1e-12)

        #### relation-oriented search
        subject_scores = torch.matmul(
            rel_q_normalized, sub_q_normalized.transpose(1, 2)) / self.temp
        object_scores = torch.matmul(
            rel_q_normalized, obj_q_normalized.transpose(1, 2)) / self.temp
        _, subject_ids = subject_scores.max(-1)
        _, object_ids = object_scores.max(-1)

        # prediction
        sub_outputs_class = torch.empty_like(outputs_class)
        sub_outputs_coord = torch.empty_like(outputs_coord)
        obj_outputs_class = torch.empty_like(outputs_class)
        obj_outputs_coord = torch.empty_like(outputs_coord)
        outputs_sub_seg_masks = torch.empty_like(seg_masks)
        outputs_obj_seg_masks = torch.empty_like(seg_masks)
        triplet_sub_ids = []
        triplet_obj_ids = []
        for i in range(len(subject_ids)):
            triplet_sub_id = subject_ids[i]
            triplet_obj_id = object_ids[i]
            sub_outputs_class[:, i] = outputs_class[:, i, triplet_sub_id, :]
            sub_outputs_coord[:, i] = outputs_coord[:, i, triplet_sub_id, :]
            obj_outputs_class[:, i] = outputs_class[:, i, triplet_obj_id, :]
            obj_outputs_coord[:, i] = outputs_coord[:, i, triplet_obj_id, :]
            outputs_sub_seg_masks[i] = seg_masks[i, triplet_sub_id, :, :]
            outputs_obj_seg_masks[i] = seg_masks[i, triplet_obj_id, :, :]
            triplet_sub_ids.append(triplet_sub_id)
            triplet_obj_ids.append(triplet_obj_id)

        all_cls_scores = dict(cls=outputs_class,
                              sub=sub_outputs_class,
                              obj=obj_outputs_class)

        rel_outputs_class = self.rel_cls_embed(outs_rel_dec)
        all_cls_scores['rel'] = rel_outputs_class
        all_cls_scores['sub_ids'] = triplet_sub_ids
        all_cls_scores['obj_ids'] = triplet_obj_ids
        all_cls_scores['subject_scores'] = subject_scores
        all_cls_scores['object_scores'] = object_scores

        all_bbox_preds = dict(bbox=outputs_coord,
                              sub=sub_outputs_coord,
                              obj=obj_outputs_coord,
                              mask=seg_masks,
                              sub_seg=outputs_sub_seg_masks,
                              obj_seg=outputs_obj_seg_masks)
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
        ### object detection and panoptic segmentation
        all_od_cls_scores = all_cls_scores['cls']
        all_od_bbox_preds = all_bbox_preds['bbox']
        all_mask_preds = all_bbox_preds['mask']

        num_dec_layers = len(all_od_cls_scores)

        all_mask_preds = [all_mask_preds for _ in range(num_dec_layers)]

        all_s_bbox_preds = all_bbox_preds['sub']
        all_o_bbox_preds = all_bbox_preds['obj']

        all_gt_bboxes_list = [gt_bboxes_list for _ in range(num_dec_layers)]
        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
        all_gt_rels_list = [gt_rels_list for _ in range(num_dec_layers)]
        all_gt_bboxes_ignore_list = [
            gt_bboxes_ignore for _ in range(num_dec_layers)
        ]
        all_gt_masks_list = [gt_masks_list for _ in range(num_dec_layers)]
        img_metas_list = [img_metas for _ in range(num_dec_layers)]

        all_r_cls_scores = all_cls_scores['rel']

        subject_scores = all_cls_scores['subject_scores']
        object_scores = all_cls_scores['object_scores']
        subject_scores = [subject_scores for _ in range(num_dec_layers)]
        object_scores = [object_scores for _ in range(num_dec_layers)]

        losses_cls, losses_bbox, losses_iou, dice_losses, focal_losses, \
        r_losses_cls, loss_subject_match, loss_object_match= multi_apply(
            self.loss_single, subject_scores, object_scores,
            all_od_cls_scores, all_od_bbox_preds, all_mask_preds,
            all_r_cls_scores, all_s_bbox_preds, all_o_bbox_preds,
            all_gt_rels_list, all_gt_bboxes_list, all_gt_labels_list,
            all_gt_masks_list, img_metas_list, all_gt_bboxes_ignore_list)

        loss_dict = dict()
        ## loss of relation-oriented matching
        loss_dict['loss_subject_match'] = loss_subject_match[-1]
        loss_dict['loss_object_match'] = loss_object_match[-1]

        ## loss of object detection and segmentation
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

        ## loss of scene graph
        # loss from the last decoder layer
        loss_dict['r_loss_cls'] = r_losses_cls[-1]

        # loss from other decoder layers
        num_dec_layer = 0
        for r_loss_cls_i in r_losses_cls[:-1]:
            loss_dict[f'd{num_dec_layer}.r_loss_cls'] = r_loss_cls_i
            num_dec_layer += 1
        return loss_dict

    def loss_single(self,
                    subject_scores,
                    object_scores,
                    od_cls_scores,
                    od_bbox_preds,
                    mask_preds,
                    r_cls_scores,
                    s_bbox_preds,
                    o_bbox_preds,
                    gt_rels_list,
                    gt_bboxes_list,
                    gt_labels_list,
                    gt_masks_list,
                    img_metas,
                    gt_bboxes_ignore_list=None):

        ## before get targets
        num_imgs = r_cls_scores.size(0)
        # obj det&seg
        cls_scores_list = [od_cls_scores[i] for i in range(num_imgs)]
        bbox_preds_list = [od_bbox_preds[i] for i in range(num_imgs)]
        mask_preds_list = [mask_preds[i] for i in range(num_imgs)]

        # scene graph
        r_cls_scores_list = [r_cls_scores[i] for i in range(num_imgs)]
        s_bbox_preds_list = [s_bbox_preds[i] for i in range(num_imgs)]
        o_bbox_preds_list = [o_bbox_preds[i] for i in range(num_imgs)]

        # matche scores
        subject_scores_list = [subject_scores[i] for i in range(num_imgs)]
        object_scores_list = [object_scores[i] for i in range(num_imgs)]

        cls_reg_targets = self.get_targets(
            subject_scores_list, object_scores_list, cls_scores_list,
            bbox_preds_list, mask_preds_list, r_cls_scores_list,
            s_bbox_preds_list, o_bbox_preds_list, gt_rels_list, gt_bboxes_list,
            gt_labels_list, gt_masks_list, img_metas, gt_bboxes_ignore_list)

        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         mask_targets_list, num_total_od_pos, num_total_od_neg,
         mask_preds_list, r_labels_list, r_label_weights_list, num_total_pos,
         num_total_neg, filtered_subject_scores, filtered_object_scores,
         gt_subject_id_list, gt_object_id_list) = cls_reg_targets

        # obj det&seg
        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)

        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)

        mask_targets = torch.cat(mask_targets_list, 0).float().flatten(1)

        mask_preds = torch.cat(mask_preds_list, 0).flatten(1)
        num_od_matches = mask_preds.shape[0]

        # id loss
        filtered_subject_scores = torch.cat(
            filtered_subject_scores,
            0).reshape(len(filtered_subject_scores[0]), -1)
        filtered_object_scores = torch.cat(filtered_object_scores, 0).reshape(
            len(filtered_object_scores[0]), -1)
        gt_subject_id = torch.cat(gt_subject_id_list, 0)
        gt_subject_id = F.one_hot(
            gt_subject_id, num_classes=filtered_subject_scores.shape[-1])
        gt_object_id = torch.cat(gt_object_id_list, 0)
        gt_object_id = F.one_hot(gt_object_id,
                                 num_classes=filtered_object_scores.shape[-1])
        loss_subject_match = self.sub_id_loss(filtered_subject_scores,
                                              gt_subject_id)
        loss_object_match = self.obj_id_loss(filtered_object_scores,
                                             gt_object_id)

        # mask loss
        focal_loss = self.focal_loss(mask_preds, mask_targets, num_od_matches)
        dice_loss = self.dice_loss(mask_preds, mask_targets, num_od_matches)

        # classification loss
        od_cls_scores = od_cls_scores.reshape(-1, self.cls_out_channels)

        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_od_pos * 1.0 + \
            num_total_od_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                od_cls_scores.new_tensor([cls_avg_factor]))
        cls_avg_factor = max(cls_avg_factor, 1)

        loss_cls = self.loss_cls(od_cls_scores,
                                 labels,
                                 label_weights,
                                 avg_factor=cls_avg_factor)

        # Compute the average number of gt boxes across all gpus, for
        # normalization purposes
        num_total_od_pos = loss_cls.new_tensor([num_total_od_pos])
        num_total_od_pos = torch.clamp(reduce_mean(num_total_od_pos),
                                       min=1).item()

        # construct factors used for rescale bboxes
        factors = []
        for img_meta, bbox_pred in zip(img_metas, od_bbox_preds):
            img_h, img_w, _ = img_meta['img_shape']
            factor = bbox_pred.new_tensor([img_w, img_h, img_w,
                                           img_h]).unsqueeze(0).repeat(
                                               bbox_pred.size(0), 1)
            factors.append(factor)
        factors = torch.cat(factors, 0)

        # DETR regress the relative position of boxes (cxcywh) in the image,
        # thus the learning target is normalized by the image size. So here
        # we need to re-scale them for calculating IoU loss
        od_bbox_preds = od_bbox_preds.reshape(-1, 4)
        bboxes = bbox_cxcywh_to_xyxy(od_bbox_preds) * factors
        bboxes_gt = bbox_cxcywh_to_xyxy(bbox_targets) * factors

        # regression IoU loss, defaultly GIoU loss
        loss_iou = self.loss_iou(bboxes,
                                 bboxes_gt,
                                 bbox_weights,
                                 avg_factor=num_total_od_pos)

        # regression L1 loss
        loss_bbox = self.loss_bbox(od_bbox_preds,
                                   bbox_targets,
                                   bbox_weights,
                                   avg_factor=num_total_od_pos)

        # scene graph
        r_labels = torch.cat(r_labels_list, 0)
        r_label_weights = torch.cat(r_label_weights_list, 0)

        # classification loss
        r_cls_scores = r_cls_scores.reshape(-1, self.rel_cls_out_channels)

        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + \
            num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                r_cls_scores.new_tensor([cls_avg_factor]))
        cls_avg_factor = max(cls_avg_factor, 1)

        r_loss_cls = self.rel_loss_cls(r_cls_scores,
                                       r_labels,
                                       r_label_weights,
                                       avg_factor=cls_avg_factor)

        return loss_cls, loss_bbox, loss_iou, dice_loss, focal_loss, r_loss_cls, loss_subject_match, loss_object_match

    def get_targets(self,
                    subject_scores_list,
                    object_scores_list,
                    cls_scores_list,
                    bbox_preds_list,
                    mask_preds_list,
                    r_cls_scores_list,
                    s_bbox_preds_list,
                    o_bbox_preds_list,
                    gt_rels_list,
                    gt_bboxes_list,
                    gt_labels_list,
                    gt_masks_list,
                    img_metas,
                    gt_bboxes_ignore_list=None):

        assert gt_bboxes_ignore_list is None, \
            'Only supports for gt_bboxes_ignore setting to None.'
        num_imgs = len(r_cls_scores_list)
        gt_bboxes_ignore_list = [
            gt_bboxes_ignore_list for _ in range(num_imgs)
        ]

        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         mask_targets_list, od_pos_inds_list, od_neg_inds_list,
         mask_preds_list, r_labels_list, r_label_weights_list, pos_inds_list,
         neg_inds_list, filtered_subject_scores, filtered_object_scores,
         gt_subject_id_list, gt_object_id_list) = multi_apply(
             self._get_target_single, subject_scores_list, object_scores_list,
             cls_scores_list, bbox_preds_list, mask_preds_list,
             r_cls_scores_list, s_bbox_preds_list, o_bbox_preds_list,
             gt_rels_list, gt_bboxes_list, gt_labels_list, gt_masks_list,
             img_metas, gt_bboxes_ignore_list)

        num_total_od_pos = sum((inds.numel() for inds in od_pos_inds_list))
        num_total_od_neg = sum((inds.numel() for inds in od_neg_inds_list))

        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (labels_list, label_weights_list, bbox_targets_list,
                bbox_weights_list, mask_targets_list, num_total_od_pos,
                num_total_od_neg, mask_preds_list, r_labels_list,
                r_label_weights_list, num_total_pos, num_total_neg,
                filtered_subject_scores, filtered_object_scores,
                gt_subject_id_list, gt_object_id_list)

    def _get_target_single(self,
                           subject_scores,
                           object_scores,
                           cls_score,
                           bbox_pred,
                           mask_preds,
                           r_cls_score,
                           s_bbox_pred,
                           o_bbox_pred,
                           gt_rels,
                           gt_bboxes,
                           gt_labels,
                           gt_masks,
                           img_meta,
                           gt_bboxes_ignore=None):

        assert len(gt_masks) == len(gt_bboxes)

        ###### obj det&seg
        num_bboxes = bbox_pred.size(0)
        assert len(gt_masks) == len(gt_bboxes)

        # assigner and sampler, only return human&object assign result
        od_assign_result = self.bbox_assigner.assign(bbox_pred, cls_score,
                                                     gt_bboxes, gt_labels,
                                                     img_meta,
                                                     gt_bboxes_ignore)
        sampling_result = self.sampler.sample(od_assign_result, bbox_pred,
                                              gt_bboxes)
        od_pos_inds = sampling_result.pos_inds
        od_neg_inds = sampling_result.neg_inds  #### no-rel class indices in prediction

        # label targets
        labels = gt_bboxes.new_full((num_bboxes, ),
                                    self.num_classes,
                                    dtype=torch.long)  ### 0-based
        labels[od_pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        label_weights = gt_bboxes.new_ones(num_bboxes)

        # mask targets for subjects and objects
        mask_targets = gt_masks[sampling_result.pos_assigned_gt_inds,
                                ...]  ###FIXME some transform might be needed
        mask_preds = mask_preds[od_pos_inds]
        mask_preds = interpolate(mask_preds[:, None],
                                 size=gt_masks.shape[-2:],
                                 mode='bilinear',
                                 align_corners=False).squeeze(1)

        # bbox targets for subjects and objects
        bbox_targets = torch.zeros_like(bbox_pred)
        bbox_weights = torch.zeros_like(bbox_pred)
        bbox_weights[od_pos_inds] = 1.0

        img_h, img_w, _ = img_meta['img_shape']

        # DETR regress the relative position of boxes (cxcywh) in the image.
        # Thus the learning target should be normalized by the image size, also
        # the box format should be converted from defaultly x1y1x2y2 to cxcywh.
        factor = bbox_pred.new_tensor([img_w, img_h, img_w,
                                       img_h]).unsqueeze(0)

        pos_gt_bboxes_normalized = sampling_result.pos_gt_bboxes / factor
        pos_gt_bboxes_targets = bbox_xyxy_to_cxcywh(pos_gt_bboxes_normalized)
        bbox_targets[od_pos_inds] = pos_gt_bboxes_targets

        gt_label_assigned_query = torch.ones_like(gt_labels)
        gt_label_assigned_query[
            sampling_result.pos_assigned_gt_inds] = od_pos_inds

        ###### scene graph
        num_rels = s_bbox_pred.size(0)
        # separate human boxes and object boxes from gt_bboxes and generate labels
        gt_sub_bboxes = []
        gt_obj_bboxes = []
        gt_sub_labels = []
        gt_obj_labels = []
        gt_rel_labels = []
        gt_sub_ids = []
        gt_obj_ids = []

        for rel_id in range(gt_rels.size(0)):
            gt_sub_bboxes.append(gt_bboxes[int(gt_rels[rel_id, 0])])
            gt_obj_bboxes.append(gt_bboxes[int(gt_rels[rel_id, 1])])
            gt_sub_labels.append(gt_labels[int(gt_rels[rel_id, 0])])
            gt_obj_labels.append(gt_labels[int(gt_rels[rel_id, 1])])
            gt_rel_labels.append(gt_rels[rel_id, 2])
            gt_sub_ids.append(gt_label_assigned_query[int(gt_rels[rel_id, 0])])
            gt_obj_ids.append(gt_label_assigned_query[int(gt_rels[rel_id, 1])])

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
        gt_sub_ids = torch.vstack(gt_sub_ids).type_as(gt_labels).reshape(-1)
        gt_obj_ids = torch.vstack(gt_obj_ids).type_as(gt_labels).reshape(-1)

        ########################################
        #### overwrite relation labels above####
        ########################################
        # assigner and sampler for relation-oriented id match
        s_assign_result, o_assign_result = self.id_assigner.assign(
            subject_scores, object_scores, r_cls_score, gt_sub_ids, gt_obj_ids,
            gt_rel_labels, img_meta, gt_bboxes_ignore)

        s_sampling_result = self.sampler.sample(s_assign_result, s_bbox_pred,
                                                gt_sub_bboxes)
        o_sampling_result = self.sampler.sample(o_assign_result, o_bbox_pred,
                                                gt_obj_bboxes)
        pos_inds = o_sampling_result.pos_inds
        neg_inds = o_sampling_result.neg_inds  #### no-rel class indices in prediction

        #match id targets
        gt_subject_ids = gt_sub_bboxes.new_full((num_rels, ),
                                                -1,
                                                dtype=torch.long)
        gt_subject_ids[pos_inds] = gt_sub_ids[
            s_sampling_result.pos_assigned_gt_inds]

        gt_object_ids = gt_obj_bboxes.new_full((num_rels, ),
                                               -1,
                                               dtype=torch.long)

        gt_object_ids[pos_inds] = gt_obj_ids[
            o_sampling_result.pos_assigned_gt_inds]

        # filtering unmatched subject/object id predictions
        gt_subject_ids = gt_subject_ids[pos_inds]
        gt_subject_ids_res = torch.zeros_like(gt_subject_ids)
        for idx, gt_subject_id in enumerate(gt_subject_ids):
            gt_subject_ids_res[idx] = ((od_pos_inds == gt_subject_id).nonzero(
                as_tuple=True)[0])
        gt_subject_ids = gt_subject_ids_res

        gt_object_ids = gt_object_ids[pos_inds]
        gt_object_ids_res = torch.zeros_like(gt_object_ids)
        for idx, gt_object_id in enumerate(gt_object_ids):
            gt_object_ids_res[idx] = ((od_pos_inds == gt_object_id).nonzero(
                as_tuple=True)[0])
        gt_object_ids = gt_object_ids_res

        filtered_subject_scores = subject_scores[pos_inds]
        filtered_subject_scores = filtered_subject_scores[:, od_pos_inds]
        filtered_object_scores = object_scores[pos_inds]
        filtered_object_scores = filtered_object_scores[:, od_pos_inds]

        r_labels = gt_obj_bboxes.new_full((num_rels, ), 0,
                                          dtype=torch.long)  ### 1-based

        r_labels[pos_inds] = gt_rel_labels[
            o_sampling_result.pos_assigned_gt_inds]
        r_label_weights = gt_obj_bboxes.new_ones(num_rels)

        return (labels, label_weights, bbox_targets, bbox_weights,
                mask_targets, od_pos_inds, od_neg_inds, mask_preds, r_labels,
                r_label_weights, pos_inds, neg_inds, filtered_subject_scores,
                filtered_object_scores, gt_subject_ids, gt_object_ids
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
            # od_cls_score = cls_scores['cls'][-1, img_id, ...]
            # bbox_pred = bbox_preds['bbox'][-1, img_id, ...]
            # mask_pred = bbox_preds['mask'][img_id, ...]
            all_cls_score = cls_scores['cls'][-1, img_id, ...]
            all_masks = bbox_preds['mask'][img_id, ...]

            s_cls_score = cls_scores['sub'][-1, img_id, ...]
            o_cls_score = cls_scores['obj'][-1, img_id, ...]
            r_cls_score = cls_scores['rel'][-1, img_id, ...]
            s_bbox_pred = bbox_preds['sub'][-1, img_id, ...]
            o_bbox_pred = bbox_preds['obj'][-1, img_id, ...]
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            s_mask_pred = bbox_preds['sub_seg'][img_id, ...]
            o_mask_pred = bbox_preds['obj_seg'][img_id, ...]
            triplet_sub_ids = cls_scores['sub_ids'][img_id]
            triplet_obj_ids = cls_scores['obj_ids'][img_id]
            triplets = self._get_bboxes_single(all_masks, all_cls_score,
                                               s_cls_score, o_cls_score,
                                               r_cls_score, s_bbox_pred,
                                               o_bbox_pred, s_mask_pred,
                                               o_mask_pred, img_shape,
                                               triplet_sub_ids,
                                               triplet_obj_ids,
                                               scale_factor, rescale)
            result_list.append(triplets)

        return result_list

    def _get_bboxes_single(self,
                           all_masks,
                           all_cls_score,
                           s_cls_score,
                           o_cls_score,
                           r_cls_score,
                           s_bbox_pred,
                           o_bbox_pred,
                           s_mask_pred,
                           o_mask_pred,
                           img_shape,
                           triplet_sub_ids,
                           triplet_obj_ids,
                           scale_factor,
                           rescale=False):

        assert len(s_cls_score) == len(o_cls_score)
        assert len(s_cls_score) == len(s_bbox_pred)
        assert len(s_cls_score) == len(o_bbox_pred)

        mask_size = (round(img_shape[0] / scale_factor[1]),
                     round(img_shape[1] / scale_factor[0]))
        max_per_img = self.test_cfg.get('max_per_img', self.num_obj_query)

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

        labels = torch.cat((s_labels, o_labels), 0)
        complete_labels = labels
        complete_r_labels = r_labels
        complete_r_dists = r_dists

        if self.use_mask:
            s_mask_pred = s_mask_pred[triplet_index]
            o_mask_pred = o_mask_pred[triplet_index]
            s_mask_pred = F.interpolate(s_mask_pred.unsqueeze(1),
                                        size=mask_size).squeeze(1)
            s_mask_pred = torch.sigmoid(s_mask_pred) > 0.85
            o_mask_pred = F.interpolate(o_mask_pred.unsqueeze(1),
                                        size=mask_size).squeeze(1)
            o_mask_pred = torch.sigmoid(o_mask_pred) > 0.85
            output_masks = torch.cat((s_mask_pred, o_mask_pred), 0)

            all_logits = F.softmax(all_cls_score, dim=-1)[..., :-1]

            all_scores, all_labels = all_logits.max(-1)
            all_masks = F.interpolate(all_masks.unsqueeze(1),
                                      size=mask_size).squeeze(1)
            #### for panoptic postprocessing ####
            triplet_sub_ids = triplet_sub_ids[triplet_index].view(-1,1)
            triplet_obj_ids = triplet_obj_ids[triplet_index].view(-1,1)
            pan_rel_pairs = torch.cat((triplet_sub_ids,triplet_obj_ids), -1).to(torch.int).to(all_masks.device)
            tri_obj_unique = pan_rel_pairs.unique()
            keep = all_labels != (s_logits.shape[-1] - 1)
            tmp = torch.zeros_like(keep, dtype=torch.bool)
            for id in tri_obj_unique:
                tmp[id] = True
            keep = keep & tmp

            all_labels = all_labels[keep]
            all_masks = all_masks[keep]
            all_scores = all_scores[keep]
            h, w = all_masks.shape[-2:]

            no_obj_filter = torch.zeros(pan_rel_pairs.shape[0],dtype=torch.bool)
            for triplet_id in range(pan_rel_pairs.shape[0]):
                if keep[pan_rel_pairs[triplet_id,0]] and keep[pan_rel_pairs[triplet_id,1]]:
                    no_obj_filter[triplet_id]=True
            pan_rel_pairs = pan_rel_pairs[no_obj_filter]
            if keep.sum() != len(keep):
                for new_id, past_id in enumerate(keep.nonzero().view(-1)):
                    pan_rel_pairs.masked_fill_(pan_rel_pairs.eq(past_id), new_id)
            r_labels, r_dists = r_labels[no_obj_filter], r_dists[no_obj_filter]

            if all_labels.numel() == 0:
                pan_img = torch.ones(mask_size).cpu().to(torch.long)
                pan_masks = pan_img.unsqueeze(0).cpu().to(torch.long)
                pan_rel_pairs = torch.arange(len(labels), dtype=torch.int).to(masks.device).reshape(2, -1).T
                rels = torch.tensor([0,0,0]).view(-1,3)
                pan_labels = torch.tensor([0])
            else:
                all_masks = all_masks.flatten(1)
                stuff_equiv_classes = defaultdict(lambda: [])
                thing_classes = defaultdict(lambda: [])
                thing_dedup = defaultdict(lambda: [])
                for k, label in enumerate(all_labels):
                    if label.item() >= 80:
                        stuff_equiv_classes[label.item()].append(k)
                    else:
                        thing_classes[label.item()].append(k)


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
                
                all_binary_masks = (torch.sigmoid(all_masks) > 0.85).to(torch.float)
                # create dict that groups duplicate masks
                for thing_pred_ids in thing_classes.values():
                    if len(thing_pred_ids) > 1:
                      dedup_things(thing_pred_ids, all_binary_masks)
                    else:
                        thing_dedup[thing_pred_ids[0]].append(thing_pred_ids[0])

                def get_ids_area(all_masks, pan_rel_pairs, r_labels, r_dists, dedup=False):
                    # This helper function creates the final panoptic segmentation image
                    # It also returns the area of the masks that appears on the image

                    m_id = all_masks.transpose(0, 1).softmax(-1)

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
                                    pan_rel_pairs.masked_fill_(pan_rel_pairs.eq(eq_id), equiv[0])
                        # Merge the masks corresponding to the same thing instance
                        for equiv in thing_dedup.values():
                            if len(equiv) > 1:
                                for eq_id in equiv:
                                    m_id.masked_fill_(m_id.eq(eq_id), equiv[0])
                                    pan_rel_pairs.masked_fill_(pan_rel_pairs.eq(eq_id), equiv[0])
                    m_ids_remain,_ = m_id.unique().sort()
                    no_obj_filter2 = torch.zeros(pan_rel_pairs.shape[0],dtype=torch.bool)
                    for triplet_id in range(pan_rel_pairs.shape[0]):
                        if pan_rel_pairs[triplet_id,0] in m_ids_remain and pan_rel_pairs[triplet_id,1] in m_ids_remain:
                            no_obj_filter2[triplet_id]=True
                    pan_rel_pairs = pan_rel_pairs[no_obj_filter2]
                    r_labels, r_dists = r_labels[no_obj_filter2], r_dists[no_obj_filter2]

                    pan_labels = [] 
                    pan_masks = []
                    for i, m_id_remain in enumerate(m_ids_remain):
                        pan_masks.append(m_id.eq(m_id_remain).unsqueeze(0))
                        pan_labels.append(all_labels[m_id_remain].unsqueeze(0))
                        m_id.masked_fill_(m_id.eq(m_id_remain), i)
                        pan_rel_pairs.masked_fill_(pan_rel_pairs.eq(m_id_remain), i)
                    pan_masks = torch.cat(pan_masks, 0)
                    pan_labels = torch.cat(pan_labels, 0)

                    seg_img = m_id * INSTANCE_OFFSET + pan_labels[m_id]
                    seg_img = seg_img.view(h, w).cpu().to(torch.long)
                    m_id = m_id.view(h, w).cpu()
                    area = []
                    for i in range(len(all_masks)):
                        area.append(m_id.eq(i).sum().item())
                    return area, seg_img, pan_rel_pairs, pan_masks, r_labels, r_dists, pan_labels

                area, pan_img, pan_rel_pairs, pan_masks, r_labels, r_dists, pan_labels = \
                    get_ids_area(all_masks, pan_rel_pairs, r_labels, r_dists, dedup=True)

                if r_labels.numel() == 0:
                    rels = torch.tensor([0,0,0]).view(-1,3)
                else:
                    rels = torch.cat((pan_rel_pairs,r_labels.unsqueeze(-1)),-1)
                # if all_labels.numel() > 0:
                #     # We know filter empty masks as long as we find some
                #     while True:
                #         filtered_small = torch.as_tensor(
                #             [area[i] <= 4 for i, c in enumerate(all_labels)],
                #             dtype=torch.bool,
                #             device=keep.device)
                #         if filtered_small.any().item():
                #             all_scores = all_scores[~filtered_small]
                #             all_labels = all_labels[~filtered_small]
                #             all_masks = all_masks[~filtered_small]
                #             area, pan_img = get_ids_area(all_masks, all_scores)
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

        det_bboxes = torch.cat((s_det_bboxes, o_det_bboxes), 0)
        rel_pairs = torch.arange(len(det_bboxes),
                                 dtype=torch.int).reshape(2, -1).T

        if self.use_mask:
            return det_bboxes, complete_labels, rel_pairs, output_masks, pan_rel_pairs, \
                pan_img, complete_r_labels, complete_r_dists, r_labels, r_dists, pan_masks, rels, pan_labels
        else:
            return det_bboxes, labels, rel_pairs, r_scores, r_labels, r_dists

    def simple_test_bboxes(self, feats, img_metas, rescale=False):

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

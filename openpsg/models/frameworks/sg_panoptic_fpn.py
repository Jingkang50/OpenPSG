import mmcv
import numpy as np
import torch
import torch.nn.functional as F
from detectron2.utils.visualizer import VisImage, Visualizer
from mmdet.core import BitmapMasks, bbox2roi, build_assigner, multiclass_nms
from mmdet.datasets.coco_panoptic import INSTANCE_OFFSET
from mmdet.models import DETECTORS, PanopticFPN
from mmdet.models.builder import build_head

from openpsg.models.relation_heads.approaches import Result
from openpsg.utils.utils import adjust_text_color, draw_text, get_colormap


@DETECTORS.register_module()
class SceneGraphPanopticFPN(PanopticFPN):
    def __init__(
        self,
        backbone,
        neck=None,
        rpn_head=None,
        roi_head=None,
        train_cfg=None,
        test_cfg=None,
        pretrained=None,
        init_cfg=None,
        # for panoptic segmentation
        semantic_head=None,
        panoptic_fusion_head=None,
        # for scene graph
        relation_head=None,
    ):
        super(SceneGraphPanopticFPN, self).__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg,
            semantic_head=semantic_head,
            panoptic_fusion_head=panoptic_fusion_head,
        )

        # Init relation head
        if relation_head is not None:
            self.relation_head = build_head(relation_head)

        # Cache the detection results to speed up the sgdet training.
        self.rpn_results = dict()
        self.det_results = dict()

    @property
    def with_relation(self):
        return hasattr(self,
                       'relation_head') and self.relation_head is not None

    def simple_test_sg_bboxes(self,
                              x,
                              img_metas,
                              proposals=None,
                              rescale=False):
        """Test without Augmentation; convert panoptic segments to bounding
        boxes."""

        if proposals is None:
            proposal_list = self.rpn_head.simple_test_rpn(x, img_metas)
        else:
            proposal_list = proposals

        bboxes, scores = self.roi_head.simple_test_bboxes(x,
                                                          img_metas,
                                                          proposal_list,
                                                          None,
                                                          rescale=rescale)

        pan_cfg = self.test_cfg.panoptic
        # class-wise predictions
        det_bboxes = []
        det_labels = []
        for bbox, score in zip(bboxes, scores):
            det_bbox, det_label = multiclass_nms(bbox, score,
                                                 pan_cfg.score_thr,
                                                 pan_cfg.nms,
                                                 pan_cfg.max_per_img)
            det_bboxes.append(det_bbox)
            det_labels.append(det_label)

        mask_results = self.simple_test_mask(x,
                                             img_metas,
                                             det_bboxes,
                                             det_labels,
                                             rescale=rescale)
        masks = mask_results['masks']  # List[(1, H, W)]

        # (B, N_stuff_classes, H, W)
        # Resizes all to same size; uses img_metas[0] only
        # seg_preds_alt = self.semantic_head.simple_test(x, img_metas, rescale)

        # seg_preds = self.semantic_head.forward(x)['seg_preds']

        seg_preds = [
            self.semantic_head.simple_test(
                [f[i][None, ...] for f in x],
                [img_metas[i]],
                rescale,
            )[0] for i in range(len(img_metas))
        ]

        results = []
        for i in range(len(det_bboxes)):
            # Shape mismatch
            # i: 1
            # img: [16, 3, 800, 1216]
            # masks[i]: [1, 800, 1120]
            # seg_preds[i]: [54, 800, 1216], problem here??
            pan_results = self.panoptic_fusion_head.simple_test(
                det_bboxes[i], det_labels[i], masks[i], seg_preds[i])
            pan_results = pan_results.int().detach().cpu().numpy()  # (H, W)

            # Convert panoptic segments to bboxes
            ids = np.unique(pan_results)[::-1]
            legal_indices = ids != self.num_classes  # for VOID label
            ids = ids[legal_indices]  # exclude VOID label

            # Extract class labels, (N), 1-index?
            labels = np.array([id % INSTANCE_OFFSET for id in ids],
                              dtype=np.int64) + 1
            # labels = np.array([id % INSTANCE_OFFSET for id in ids],
            # dtype=np.int64)
            # Binary masks for each object, (N, H, W)
            segms = pan_results[None] == ids[:, None, None]

            # Convert to bboxes
            height, width = segms.shape[1:]
            masks_to_bboxes = BitmapMasks(segms, height, width).get_bboxes()

            # Convert to torch tensor
            # (N_b, 4)
            masks_to_bboxes = torch.tensor(masks_to_bboxes).to(det_bboxes[0])
            labels = torch.tensor(labels).to(det_labels[0])

            result = dict(pan_results=pan_results,
                          masks=segms,
                          bboxes=masks_to_bboxes,
                          labels=labels)
            results.append(result)

        return results

    def forward_train(
        self,
        img,
        img_metas,
        # all_gt_bboxes,
        # all_gt_labels,
        gt_bboxes,
        gt_labels,
        gt_bboxes_ignore=None,
        gt_masks=None,
        proposals=None,
        gt_rels=None,
        gt_keyrels=None,
        gt_relmaps=None,
        gt_scenes=None,
        rescale=False,
        **kwargs,
    ):
        # img: (B, C, H, W)

        x = self.extract_feat(img)

        ################################################################
        #        Specifically for Relation Prediction / SGG.           #
        #        The detector part must perform as if at test mode.    #
        ################################################################
        if self.with_relation:
            # # assert gt_rels is not None and gt_relmaps is not None
            # if self.relation_head.with_visual_mask and (not self.with_mask):
            #     raise ValueError("The basic detector did not provide masks.")

            # NOTE: Change gt to 1-index here:
            gt_labels = [label + 1 for label in gt_labels]
            """
            NOTE: (for VG) When the gt masks is None, but the head needs mask,
            we use the gt_box and gt_label (if needed) to generate the fake
            mask.
            """
            (
                bboxes,
                labels,
                target_labels,
                dists,  # Can this be `None`?
                pan_masks,
                pan_results,
                points,
            ) = self.detector_simple_test(
                x,
                img_metas,
                # all_gt_bboxes,
                # all_gt_labels,
                gt_bboxes,
                gt_labels,
                gt_masks,
                proposals,
                use_gt_box=self.relation_head.use_gt_box,
                use_gt_label=self.relation_head.use_gt_label,
                rescale=rescale,
            )

            # Filter out empty predictions
            idxes_to_filter = [i for i, b in enumerate(bboxes) if len(b) == 0]

            param_need_filter = [
                bboxes, labels, dists, target_labels, gt_bboxes, gt_labels,
                gt_rels, img_metas, gt_scenes, gt_keyrels, points, pan_results,
                gt_masks, gt_relmaps, pan_masks
            ]
            for idx, param in enumerate(param_need_filter):
                if param_need_filter[idx]:
                    param_need_filter[idx] = [
                        x for i, x in enumerate(param)
                        if i not in idxes_to_filter
                    ]

            (bboxes, labels, dists, target_labels, gt_bboxes, gt_labels,
             gt_rels, img_metas, gt_scenes, gt_keyrels, points, pan_results,
             gt_masks, gt_relmaps, pan_masks) = param_need_filter
            # Filter done

            if idxes_to_filter and len(gt_bboxes) == 16:
                print('sg_panoptic_fpn: not filtered!')

            filtered_x = []
            for idx in range(len(x)):
                filtered_x.append(
                    torch.stack([
                        e for i, e in enumerate(x[idx])
                        if i not in idxes_to_filter
                    ]))
            x = filtered_x

            gt_result = Result(
                # bboxes=all_gt_bboxes,
                # labels=all_gt_labels,
                bboxes=gt_bboxes,
                labels=gt_labels,
                rels=gt_rels,
                relmaps=gt_relmaps,
                masks=gt_masks,
                rel_pair_idxes=[rel[:, :2].clone() for rel in gt_rels]
                if gt_rels is not None else None,
                rel_labels=[rel[:, -1].clone() for rel in gt_rels]
                if gt_rels is not None else None,
                key_rels=gt_keyrels if gt_keyrels is not None else None,
                img_shape=[meta['img_shape'] for meta in img_metas],
                scenes=gt_scenes,
            )

            det_result = Result(
                bboxes=bboxes,
                labels=labels,
                dists=dists,
                masks=pan_masks,
                pan_results=pan_results,
                points=points,
                target_labels=target_labels,
                target_scenes=gt_scenes,
                img_shape=[meta['img_shape'] for meta in img_metas],
            )

            det_result = self.relation_head(x, img_metas, det_result,
                                            gt_result)

            # Loss performed here
            return self.relation_head.loss(det_result)

    def forward_test(self, imgs, img_metas, **kwargs):
        """
        Args:
            imgs (List[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            img_metas (List[List[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch.
        """
        for var, name in [(imgs, 'imgs'), (img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError(f'{name} must be a list, but got {type(var)}')

        num_augs = len(imgs)
        if num_augs != len(img_metas):
            raise ValueError(f'num of augmentations ({len(imgs)}) '
                             f'!= num of image meta ({len(img_metas)})')

        # NOTE the batched image size information may be useful, e.g.
        # in DETR, this is needed for the construction of masks, which is
        # then used for the transformer_head.
        for img, img_meta in zip(imgs, img_metas):
            batch_size = len(img_meta)
            for img_id in range(batch_size):
                img_meta[img_id]['batch_input_shape'] = tuple(img.size()[-2:])

        key_first = kwargs.pop('key_first', False)

        # if relation_mode:
        assert num_augs == 1
        return self.relation_simple_test(imgs[0],
                                         img_metas[0],
                                         key_first=key_first,
                                         **kwargs)

        # if num_augs == 1:
        #     # proposals (List[List[Tensor]]): the outer list indicates
        #     # test-time augs (multiscale, flip, etc.) and the inner list
        #     # indicates images in a batch.
        #     # The Tensor should have a shape Px4, where P is the number of
        #     # proposals.
        #     if "proposals" in kwargs:
        #         kwargs["proposals"] = kwargs["proposals"][0]
        #     return self.simple_test(imgs[0], img_metas[0], **kwargs)
        # else:
        #     assert imgs[0].size(0) == 1, (
        #         "aug test does not support "
        #         "inference with batch size "
        #         f"{imgs[0].size(0)}"
        #     )
        #     # TODO: support test augmentation for predefined proposals
        #     assert "proposals" not in kwargs
        #     return self.aug_test(imgs, img_metas, **kwargs)

    def detector_simple_test(
        self,
        x,
        img_meta,
        gt_bboxes,
        gt_labels,
        gt_masks,
        proposals=None,
        use_gt_box=False,
        use_gt_label=False,
        rescale=False,
        is_testing=False,
    ):
        """Test without augmentation. Used in SGG.

        Return:
            det_bboxes: (list[Tensor]): The boxes may have 5 columns (sgdet)
                or 4 columns (predcls/sgcls).
            det_labels: (list[Tensor]): 1D tensor, det_labels (sgdet) or
                gt_labels (predcls/sgcls).
            det_dists: (list[Tensor]): 2D tensor, N x Nc, the bg column is 0.
                detected dists (sgdet/sgcls), or None (predcls).
            masks: (list[list[Tensor]]): Mask is associated with box. Thus,
                in predcls/sgcls mode, it will firstly return the gt_masks.
                But some datasets do not contain gt_masks. We try to use the
                gt box to obtain the masks.
        """
        assert self.with_bbox, 'Bbox head must be implemented.'

        pan_seg_masks = None
        if use_gt_box and use_gt_label:  # predcls
            # if is_testing:
            #     det_results = self.simple_test_sg_bboxes(x, img_meta,
            # rescale=True)
            #     pan_results = [r['pan_results'] for r in det_results]

            target_labels = gt_labels
            pan_seg_masks = gt_masks
            return gt_bboxes, gt_labels, target_labels, None, gt_masks, None, None

        # NOTE: Sgcls should not be performed
        elif use_gt_box and not use_gt_label:  # sgcls
            """The self implementation return 1-based det_labels."""
            target_labels = gt_labels
            _, det_labels, det_dists = self.detector_simple_test_det_bbox(
                x, img_meta, proposals=gt_bboxes, rescale=rescale)
            return gt_bboxes, det_labels, target_labels, det_dists, gt_masks, None, None

        elif not use_gt_box and not use_gt_label:  # sgdet
            """It returns 1-based det_labels."""
            # get target labels for the det bboxes: make use of the
            # bbox head assigner
            if not is_testing:  # excluding the testing phase
                det_results = self.simple_test_sg_bboxes(x,
                                                         img_meta,
                                                         rescale=rescale)
                det_bboxes = [r['bboxes'] for r in det_results]
                det_labels = [r['labels'] for r in det_results]  # 1-index
                pan_results = None

                target_labels = []
                # MaxIOUAssigner
                bbox_assigner = build_assigner(self.train_cfg.rcnn.assigner)
                for i in range(len(img_meta)):
                    assign_result = bbox_assigner.assign(
                        # gt_labels: 1-index
                        det_bboxes[i],
                        gt_bboxes[i],
                        gt_labels=gt_labels[i] - 1,
                    )
                    target_labels.append(assign_result.labels + 1)
                    # assign_result.labels[assign_result.labels == -1] = 0
                    # target_labels.append(assign_result.labels)

            else:
                det_results = self.simple_test_sg_bboxes(x,
                                                         img_meta,
                                                         rescale=rescale)
                det_bboxes = [r['bboxes'] for r in det_results]
                det_labels = [r['labels'] for r in det_results]  # 1-index
                pan_seg_masks = [r['masks'] for r in det_results]

                # to reshape pan_seg_masks
                mask_size = (img_meta[0]['ori_shape'][0],
                             img_meta[0]['ori_shape'][1])
                pan_seg_masks = F.interpolate(
                    torch.Tensor(pan_seg_masks[0]).unsqueeze(1),
                    size=mask_size).squeeze(1).bool()
                pan_seg_masks = [pan_seg_masks.numpy()]

                # TODO: why number of bboxes/masks will differ between 2 tests?

                det_results_for_pan = self.simple_test_sg_bboxes(x,
                                                                 img_meta,
                                                                 rescale=True)
                pan_results = [r['pan_results'] for r in det_results_for_pan]
                # pan_seg_masks = [r['masks'] for r in det_results_for_pan]

                target_labels = None

            # det_dists: Tuple[(N_b, N_c + 1)]
            # Temp set as one-hot labels
            det_dists = [
                F.one_hot(det_label,
                          num_classes=self.num_classes + 1).to(det_bboxes[0])
                for det_label in det_labels
            ]
            # det_dists = [
            #     F.one_hot(det_label - 1, num_classes=self.num_classes)
            # .to(det_bboxes[0])
            #     for det_label in det_labels
            # ]

            det_bboxes = [
                torch.cat([b, b.new_ones(len(b), 1)], dim=-1)
                for b in det_bboxes
            ]

            # det_bboxes: List[B x (N_b, 5)]
            # det_labels: List[B x (N_b)]
            # target_labels: List[B x (N_b)]
            # det_dists: List[B x (N_b, N_c + 1)]
            return det_bboxes, det_labels, target_labels, \
                det_dists, pan_seg_masks, pan_results, None

    def detector_simple_test_det_bbox(self,
                                      x,
                                      img_meta,
                                      proposals=None,
                                      rescale=False):
        """Run the detector in test mode, given gt_bboxes, return the labels, dists
        Return:
            det_labels: 1 based.
        """
        num_levels = len(x)

        if proposals is None:
            proposal_list = self.rpn_head.simple_test_rpn(x, img_meta)
        else:
            proposal_list = proposals
        """Support multi-image per batch"""
        det_bboxes, det_labels, score_dists = [], [], []
        for img_id in range(len(img_meta)):
            x_i = tuple([x[i][img_id][None] for i in range(num_levels)])
            # img_meta_i = [img_meta[img_id]]
            proposal_list_i = [proposal_list[img_id]]
            det_labels_i, score_dists_i = self.simple_test_given_bboxes(
                x_i, proposal_list_i)
            det_bboxes.append(proposal_list[img_id])
            det_labels.append(det_labels_i)
            score_dists.append(score_dists_i)

        return det_bboxes, det_labels, score_dists

    def detector_simple_test_det_bbox_mask(self, x, img_meta, rescale=False):
        """Run the detector in test mode, return the detected boxes, labels,
        dists, and masks."""
        """RPN phase"""
        # num_levels = len(x)
        # proposal_list =
        # self.rpn_head.simple_test_rpn(x, img_meta, self.test_cfg.rpn)
        proposal_list = self.rpn_head.simple_test_rpn(x, img_meta)
        # List[Tensor(1000, 5)]
        """Support multi-image per batch"""
        # det_bboxes, det_labels, score_dists = [], [], []
        # # img_meta: List[metadata]
        # for img_id in range(len(img_meta)):
        #     x_i = tuple([x[i][img_id][None] for i in range(num_levels)])
        #     img_meta_i = [img_meta[img_id]]
        #     proposal_list_i = [proposal_list[img_id]]
        #     (
        #         det_bboxes_i,
        #         det_labels_i,
        #         score_dists_i,
        #     ) = self.roi_head.simple_test_bboxes(
        #         x_i,
        #         img_meta_i,
        #         proposal_list_i,
        #         self.test_cfg.rcnn,
        #         rescale=rescale,
        #         # return_dist=True,
        #     )

        #     det_bboxes.append(det_bboxes_i)
        #     det_labels.append(det_labels_i + 1)
        #     score_dists.append(score_dists_i)

        det_bboxes, det_labels = self.roi_head.simple_test_bboxes(
            x,
            img_meta,
            proposal_list,
            self.test_cfg.rcnn,
            rescale=rescale,
            # return_dist=True,
        )
        det_labels, score_dists = zip(*det_labels)

        # det_bboxes: (N_b, 5)
        # det_labels: (N_b)
        # score_dists: (N_b, N_c + 1)
        return det_bboxes, det_labels, score_dists, None, None

    def simple_test_given_bboxes(self, x, proposals):
        """For SGG~SGCLS mode: Given gt boxes, extract its predicted scores and
        score dists.

        Without any post-process.
        """
        rois = bbox2roi(proposals)
        roi_feats = self.roi_head.bbox_roi_extractor(
            x[:len(self.roi_head.bbox_roi_extractor.featmap_strides)], rois)
        if self.with_shared_head:
            roi_feats = self.roi_head.shared_head(roi_feats)
        cls_score, _ = self.roi_head.bbox_head(roi_feats)
        cls_score[:, 1:] = F.softmax(cls_score[:, 1:], dim=1)
        _, labels = torch.max(cls_score[:, 1:], dim=1)
        labels += 1
        cls_score[:, 0] = 0
        return labels, cls_score

    def relation_simple_test(
        self,
        img,
        img_meta,
        # all_gt_bboxes=None,
        # all_gt_labels=None,
        gt_bboxes=None,
        gt_labels=None,
        gt_rels=None,
        gt_masks=None,
        gt_scenes=None,
        rescale=False,
        ignore_classes=None,
        key_first=False,
    ):
        """
        :param img:
        :param img_meta:
        :param gt_bboxes: Usually, under the forward (train/val/test),
        it should not be None. But when for demo (inference), it should
        be None. The same for gt_labels.
        :param gt_labels:
        :param gt_rels: You should make sure that the gt_rels should not
        be passed into the forward process in any mode. It is only used to
        visualize the results.
        :param gt_masks:
        :param rescale:
        :param ignore_classes: For practice, you may want to ignore some
        object classes
        :return:
        """
        # Extract the outer list: Since the aug test is
        # temporarily not supported.
        # if all_gt_bboxes is not None:
        #     all_gt_bboxes = all_gt_bboxes[0]
        # if all_gt_labels is not None:
        #     all_gt_labels = all_gt_labels[0]
        if gt_bboxes is not None:
            gt_bboxes = gt_bboxes[0]
        if gt_labels is not None:
            gt_labels = gt_labels[0]
        if gt_masks is not None:
            gt_masks = gt_masks[0]

        x = self.extract_feat(img)
        """
        NOTE: (for VG) When the gt masks is None, but the head needs mask,
        we use the gt_box and gt_label (if needed) to generate the fake mask.
        """

        # NOTE: Change to 1-index here:
        gt_labels = [label + 1 for label in gt_labels]

        # Rescale should be forbidden here since the bboxes and masks will
        # be used in relation module.
        bboxes, labels, target_labels, dists, pan_masks, pan_results, points \
            = self.detector_simple_test(
            x,
            img_meta,
            gt_bboxes,
            gt_labels,
            gt_masks,
            use_gt_box=self.relation_head.use_gt_box,
            use_gt_label=self.relation_head.use_gt_label,
            rescale=False,
            is_testing=True,
        )

        # saliency_maps = (
        #     self.saliency_detector_test(img, img_meta) if \
        #     self.with_saliency else None
        # )

        det_result = Result(
            bboxes=bboxes,
            labels=labels,
            dists=dists,
            masks=pan_masks,
            pan_results=pan_results,
            points=points,
            target_labels=target_labels,
            # saliency_maps=saliency_maps,
            img_shape=[meta['img_shape'] for meta in img_meta],
        )

        # If empty prediction
        if len(bboxes[0]) == 0:
            return det_result

        det_result = self.relation_head(x,
                                        img_meta,
                                        det_result,
                                        is_testing=True,
                                        ignore_classes=ignore_classes)
        """
        Transform the data type, and rescale the bboxes and masks if needed
        (for visual, do not rescale, for evaluation, rescale).
        """
        scale_factor = img_meta[0]['scale_factor']
        det_result = self.relation_head.get_result(det_result,
                                                   scale_factor,
                                                   rescale=rescale,
                                                   key_first=key_first)

        if pan_masks is not None:
            det_result.masks = np.array(pan_masks[0])

        return det_result

    def show_result(
        self,
        img,
        result,
        score_thr=0.3,
        bbox_color=(72, 101, 241),
        text_color=(72, 101, 241),
        mask_color=None,
        thickness=2,
        font_size=13,
        win_name='',
        show=False,
        wait_time=0,
        out_file=None,
    ):
        """Draw `result` over `img`.

        Args:
            img (str or Tensor): The image to be displayed.
            result (Tensor or tuple): The results to draw over `img`
                bbox_result or (bbox_result, segm_result).
            score_thr (float, optional): Minimum score of bboxes to be shown.
                Default: 0.3.
            bbox_color (str or tuple(int) or :obj:`Color`):Color of bbox lines.
               The tuple of color should be in BGR order. Default: 'green'
            text_color (str or tuple(int) or :obj:`Color`):Color of texts.
               The tuple of color should be in BGR order. Default: 'green'
            mask_color (None or str or tuple(int) or :obj:`Color`):
               Color of masks. The tuple of color should be in BGR order.
               Default: None
            thickness (int): Thickness of lines. Default: 2
            font_size (int): Font size of texts. Default: 13
            win_name (str): The window name. Default: ''
            wait_time (float): Value of waitKey param.
                Default: 0.
            show (bool): Whether to show the image.
                Default: False.
            out_file (str or None): The filename to write the image.
                Default: None.

        Returns:
            img (Tensor): Only if not `show` or `out_file`
        """
        self.CLASSES = [
            'airplane',
            'apple',
            'backpack',
            'banana',
            'baseball bat',
            'baseball glove',
            'bear',
            'bed',
            'bench',
            'bicycle',
            'bird',
            'boat',
            'book',
            'bottle',
            'bowl',
            'broccoli',
            'bus',
            'cake',
            'car',
            'carrot',
            'cat',
            'cell phone',
            'chair',
            'clock',
            'couch',
            'cow',
            'cup',
            'dining table',
            'dog',
            'donut',
            'elephant',
            'fire hydrant',
            'fork',
            'frisbee',
            'giraffe',
            'hair drier',
            'handbag',
            'horse',
            'hot dog',
            'keyboard',
            'kite',
            'knife',
            'laptop',
            'microwave',
            'motorcycle',
            'mouse',
            'orange',
            'oven',
            'parking meter',
            'person',
            'pizza',
            'potted plant',
            'refrigerator',
            'remote',
            'sandwich',
            'scissors',
            'sheep',
            'sink',
            'skateboard',
            'skis',
            'snowboard',
            'spoon',
            'sports ball',
            'stop sign',
            'suitcase',
            'surfboard',
            'teddy bear',
            'tennis racket',
            'tie',
            'toaster',
            'toilet',
            'toothbrush',
            'traffic light',
            'train',
            'truck',
            'tv',
            'umbrella',
            'vase',
            'wine glass',
            'zebra',
            'banner',
            'blanket',
            'bridge',
            'building',
            'cabinet',
            'cardboard',
            'ceiling',
            'counter',
            'curtain',
            'dirt',
            'door',
            'fence',
            'floor',
            'floor-wood',
            'flower',
            'food',
            'fruit',
            'grass',
            'gravel',
            'house',
            'light',
            'mirror',
            'mountain',
            'net',
            'paper',
            'pavement',
            'pillow',
            'platform',
            'playingfield',
            'railroad',
            'river',
            'road',
            'rock',
            'roof',
            'rug',
            'sand',
            'sea',
            'shelf',
            'sky',
            'snow',
            'stairs',
            'table',
            'tent',
            'towel',
            'tree',
            'wall-brick',
            'wall',
            'wall-stone',
            'wall-tile',
            'wall-wood',
            'water',
            'window-blind',
            'window',
        ]

        # Load image
        img = mmcv.imread(img)
        img = img.copy()  # (H, W, 3)
        img_h, img_w = img.shape[:-1]

        if True:
            # Draw masks
            pan_results = result.pan_results

            ids = np.unique(pan_results)[::-1]
            legal_indices = ids != self.num_classes  # for VOID label
            ids = ids[legal_indices]

            # Get predicted labels
            labels = np.array([id % INSTANCE_OFFSET for id in ids],
                              dtype=np.int64)
            labels = [self.CLASSES[label] for label in labels]

            # (N_m, H, W)
            segms = pan_results[None] == ids[:, None, None]
            # Resize predicted masks
            segms = [
                mmcv.image.imresize(m.astype(float), (img_w, img_h))
                for m in segms
            ]

            # Choose colors for each instance in coco
            colormap_coco = get_colormap(len(segms))
            colormap_coco = (np.array(colormap_coco) / 255).tolist()

            viz = Visualizer(img)
            viz.overlay_instances(
                labels=labels,
                masks=segms,
                assigned_colors=colormap_coco,
            )
            viz_img = viz.get_output().get_image()

        else:
            # Draw bboxes
            bboxes = result.refine_bboxes[:, :4]

            # Choose colors for each instance in coco
            colormap_coco = get_colormap(len(bboxes))
            colormap_coco = (np.array(colormap_coco) / 255).tolist()

            # 1-index
            labels = [self.CLASSES[label - 1] for label in result.labels]

            viz = Visualizer(img)
            viz.overlay_instances(
                labels=labels,
                boxes=bboxes,
                assigned_colors=colormap_coco,
            )
            viz_img = viz.get_output().get_image()

        # Draw relations

        # Filter out relations
        n_rel_topk = 20

        # Exclude background class
        rel_dists = result.rel_dists[:, 1:]
        # rel_dists = result.rel_dists

        rel_scores = rel_dists.max(1)
        # rel_scores = result.triplet_scores

        # Extract relations with top scores
        rel_topk_idx = np.argpartition(rel_scores, -n_rel_topk)[-n_rel_topk:]
        rel_labels_topk = rel_dists[rel_topk_idx].argmax(1)
        rel_pair_idxes_topk = result.rel_pair_idxes[rel_topk_idx]
        relations = np.concatenate(
            [rel_pair_idxes_topk, rel_labels_topk[..., None]], axis=1)

        n_rels = len(relations)

        top_padding = 20
        bottom_padding = 20
        left_padding = 20

        text_size = 10
        text_padding = 5
        text_height = text_size + 2 * text_padding

        row_padding = 10

        height = (top_padding + bottom_padding + n_rels *
                  (text_height + row_padding) - row_padding)
        width = viz_img.shape[1]

        curr_x = left_padding
        curr_y = top_padding

        # Adjust colormaps
        colormap_coco = [adjust_text_color(c, viz) for c in colormap_coco]

        viz_graph = VisImage(np.full((height, width, 3), 255))

        for i, r in enumerate(relations):
            s_idx, o_idx, rel_id = r

            s_label = labels[s_idx]
            o_label = labels[o_idx]
            # Becomes 0-index
            rel_label = self.PREDICATES[rel_id]

            # Draw subject text
            text_width = draw_text(
                viz_img=viz_graph,
                text=s_label,
                x=curr_x,
                y=curr_y,
                color=colormap_coco[s_idx],
                size=text_size,
                padding=text_padding,
                # font=font,
            )
            curr_x += text_width

            # Draw relation text
            text_width = draw_text(
                viz_img=viz_graph,
                text=rel_label,
                x=curr_x,
                y=curr_y,
                size=text_size,
                padding=text_padding,
                box_color='gainsboro',
                # font=font,
            )
            curr_x += text_width

            # Draw object text
            text_width = draw_text(
                viz_img=viz_graph,
                text=o_label,
                x=curr_x,
                y=curr_y,
                color=colormap_coco[o_idx],
                size=text_size,
                padding=text_padding,
                # font=font,
            )
            curr_x += text_width

            curr_x = left_padding
            curr_y += text_height + row_padding

        viz_graph = viz_graph.get_image()

        viz_final = np.vstack([viz_img, viz_graph])

        if out_file is not None:
            mmcv.imwrite(viz_final, out_file)

        if not (show or out_file):
            return viz_final

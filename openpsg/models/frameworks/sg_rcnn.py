import mmcv
import numpy as np
import torch
import torch.nn.functional as F
from detectron2.utils.visualizer import VisImage, Visualizer
from mmdet.core import bbox2roi, build_assigner
from mmdet.models import DETECTORS, TwoStageDetector
from mmdet.models.builder import build_head

from openpsg.models.relation_heads.approaches import Result
from openpsg.utils.utils import adjust_text_color, draw_text, get_colormap


@DETECTORS.register_module()
class SceneGraphRCNN(TwoStageDetector):
    def __init__(
        self,
        backbone,
        rpn_head,
        roi_head,
        train_cfg,
        test_cfg,
        neck=None,
        pretrained=None,
        init_cfg=None,
        relation_head=None,
    ):
        super(SceneGraphRCNN, self).__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg,
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

    def forward_train(
        self,
        img,
        img_metas,
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
        x = self.extract_feat(img)

        ################################################################
        #        Specifically for Relation Prediction / SGG.           #
        #        The detector part must perform as if at test mode.    #
        ################################################################
        if self.with_relation:
            # # assert gt_rels is not None and gt_relmaps is not None
            # if self.relation_head.with_visual_mask and (not self.with_mask):
            #     raise ValueError("The basic detector did not provide masks.")

            # Change to 1-index here:
            gt_labels = [l + 1 for l in gt_labels]
            """
            NOTE: (for VG) When the gt masks is None, but the head needs mask,
            we use the gt_box and gt_label (if needed) to generate the fake mask.
            """
            (
                bboxes,
                labels,
                target_labels,
                dists,
                masks,
                points,
            ) = self.detector_simple_test(
                x,
                img_metas,
                gt_bboxes,
                gt_labels,
                gt_masks,
                proposals,
                use_gt_box=self.relation_head.use_gt_box,
                use_gt_label=self.relation_head.use_gt_label,
                rescale=rescale,
            )

            # saliency_maps = (
            #     self.saliency_detector_test(img, img_meta)
            #     if self.with_saliency
            #     else None
            # )

            gt_result = Result(
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
                masks=masks,
                points=points,
                target_labels=target_labels,
                target_scenes=gt_scenes,
                img_shape=[meta['img_shape'] for meta in img_metas],
            )

            det_result = self.relation_head(x, img_metas, det_result,
                                            gt_result)

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
            det_bboxes: (list[Tensor]): The boxes may have 5 columns (sgdet) or 4 columns (predcls/sgcls).
            det_labels: (list[Tensor]): 1D tensor, det_labels (sgdet) or gt_labels (predcls/sgcls).
            det_dists: (list[Tensor]): 2D tensor, N x Nc, the bg column is 0. detected dists (sgdet/sgcls), or
                None (predcls).
            masks: (list[list[Tensor]]): Mask is associated with box. Thus, in predcls/sgcls mode, it will
                firstly return the gt_masks. But some datasets do not contain gt_masks. We try to use the gt box
                to obtain the masks.
        """
        assert self.with_bbox, 'Bbox head must be implemented.'

        if use_gt_box and use_gt_label:  # predcls
            target_labels = gt_labels
            return gt_bboxes, gt_labels, target_labels, None, None, None

        elif use_gt_box and not use_gt_label:  # sgcls
            """The self implementation return 1-based det_labels."""
            target_labels = gt_labels
            _, det_labels, det_dists = self.detector_simple_test_det_bbox(
                x, img_meta, proposals=gt_bboxes, rescale=rescale)
            return gt_bboxes, det_labels, target_labels, det_dists, None, None

        elif not use_gt_box and not use_gt_label:
            """It returns 1-based det_labels."""
            (
                det_bboxes,
                det_labels,
                det_dists,
                _,
                _,
            ) = self.detector_simple_test_det_bbox_mask(x,
                                                        img_meta,
                                                        rescale=rescale)
            # get target labels for the det bboxes: make use of the bbox head assigner
            if not is_testing:  # excluding the testing phase
                target_labels = []
                bbox_assigner = build_assigner(self.train_cfg.rcnn.assigner)
                for i in range(len(img_meta)):
                    assign_result = bbox_assigner.assign(
                        det_bboxes[i],
                        gt_bboxes[i],
                        gt_labels=gt_labels[i] - 1)
                    target_labels.append(assign_result.labels + 1)
            else:
                target_labels = None

            # det_bboxes: List[B x (N_b, 5)], last dim is probability
            # det_labels: List[B x (N_b)]
            # target_labels: List[B x (N_b)]
            # det_dists: List[B x (N_b, N_c + 1)], doesn't sum to 1
            return det_bboxes, det_labels, target_labels, det_dists, None, None

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
            img_meta_i = [img_meta[img_id]]
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
        num_levels = len(x)
        # proposal_list = self.rpn_head.simple_test_rpn(x, img_meta, self.test_cfg.rpn)
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
        # det_labels = [l + 1 for l in det_labels]

        # det_bboxes: (N_b, 5)
        # det_labels: (N_b)
        # score_dists: (N_b, N_c + 1)
        return det_bboxes, det_labels, score_dists, None, None

        # if not self.with_mask:
        #     return det_bboxes, det_labels, score_dists, None, None
        # else:
        #     if self.bbox_head.__class__.__name__ == "ExtrDetWeightSharedFCBBoxHead":
        #         det_weight = self.bbox_head.det_weight_hook()
        #     else:
        #         det_weight = None
        #     segm_masks = []
        #     points = []
        #     for img_id in range(len(img_meta)):
        #         x_i = tuple([x[i][img_id][None] for i in range(num_levels)])
        #         img_meta_i = [img_meta[img_id]]
        #         test_result = self.simple_test_mask(
        #             x_i,
        #             img_meta_i,
        #             det_bboxes[img_id],
        #             det_labels[img_id] - 1,
        #             rescale=rescale,
        #             det_weight=det_weight,
        #             with_point=self.with_point,
        #         )
        #         if isinstance(test_result, tuple):
        #             segm_masks_i, points_i = test_result
        #             points.append(points_i)
        #         else:
        #             segm_masks_i = test_result
        #         segm_masks.append(segm_masks_i)
        #     return det_bboxes, det_labels, score_dists, segm_masks, points

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
        :param gt_bboxes: Usually, under the forward (train/val/test), it should not be None. But
        when for demo (inference), it should be None. The same for gt_labels.
        :param gt_labels:
        :param gt_rels: You should make sure that the gt_rels should not be passed into the forward
        process in any mode. It is only used to visualize the results.
        :param gt_masks:
        :param rescale:
        :param ignore_classes: For practice, you may want to ignore some object classes
        :return:
        """
        # Extract the outer list: Since the aug test is temporarily not supported.
        if gt_bboxes is not None:
            gt_bboxes = gt_bboxes[0]
        if gt_labels is not None:
            gt_labels = gt_labels[0]
        if gt_masks is not None:
            gt_masks = gt_masks[0]

        x = self.extract_feat(img)

        # if self.relation_head.with_visual_mask and (not self.with_mask):
        #     raise ValueError("The basic detector did not provide masks.")
        """
        NOTE: (for VG) When the gt masks is None, but the head needs mask,
        we use the gt_box and gt_label (if needed) to generate the fake mask.
        """

        # Change to 1-index here:
        gt_labels = [l + 1 for l in gt_labels]

        # Rescale should be forbidden here since the bboxes and masks will be used in relation module.
        bboxes, labels, target_labels, dists, masks, points = self.detector_simple_test(
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
        #     self.saliency_detector_test(img, img_meta) if self.with_saliency else None
        # )

        det_result = Result(
            bboxes=bboxes,
            labels=labels,
            dists=dists,
            masks=masks,
            points=points,
            target_labels=target_labels,
            # saliency_maps=saliency_maps,
            img_shape=[meta['img_shape'] for meta in img_meta],
        )

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
        return self.relation_head.get_result(det_result,
                                             scale_factor,
                                             rescale=rescale,
                                             key_first=key_first)

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
        img = mmcv.imread(img)
        img = img.copy()  # (H, W, 3)

        # TODO: Use threshold to filter out
        # Draw bboxes
        bboxes = result.refine_bboxes[:, :4]

        # Choose colors for each instance in coco
        colormap_coco = get_colormap(len(bboxes))
        colormap_coco = (np.array(colormap_coco) / 255).tolist()

        # 1-index
        labels = [self.CLASSES[l - 1] for l in result.labels]

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

        # To exclude background class?
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

            # # Filter for specific relations
            # if rel_ids_to_keep:
            #     if rel_id not in rel_ids_to_keep:
            #         continue
            # elif rel_ids_to_filter:
            #     if rel_id in rel_ids_to_filter:
            #         continue

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

from typing import Tuple
import os.path as osp
import PIL
import mmcv
import mmcv.ops as ops
import numpy as np
import torch
from detectron2.utils.colormap import colormap
from detectron2.utils.visualizer import VisImage, Visualizer
from mmdet.datasets.coco_panoptic import INSTANCE_OFFSET
import matplotlib.pyplot as plt

# from mmcv.ops.nms import batched_nms


def enumerate_by_image(im_inds):
    im_inds_np = im_inds.cpu().numpy()
    initial_ind = int(im_inds_np[0])
    s = 0
    for i, val in enumerate(im_inds_np):
        if val != initial_ind:
            yield initial_ind, s, i
            initial_ind = int(val)
            s = i
    yield initial_ind, s, len(im_inds_np)


def get_colormap(num_colors: int):
    return (np.resize(colormap(), (num_colors, 3))).tolist()


def adjust_text_color(color: Tuple[float, float, float],
                      viz: Visualizer) -> Tuple[float, float, float]:
    color = viz._change_color_brightness(color, brightness_factor=0.7)
    color = np.maximum(color, 0.2)
    color[np.argmax(color)] = max(0.8, np.max(color))
    return color


def draw_text(
    viz_img: VisImage = None,
    text: str = None,
    x: float = None,
    y: float = None,
    color: Tuple[float, float, float] = [0, 0, 0],
    size: float = 10,
    padding: float = 5,
    box_color: str = 'black',
    font: str = None,
) -> float:
    text_obj = viz_img.ax.text(
        x,
        y,
        text,
        size=size,
        # family="sans-serif",
        bbox={
            'facecolor': box_color,
            'alpha': 0.8,
            'pad': padding,
            'edgecolor': 'none',
        },
        verticalalignment='top',
        horizontalalignment='left',
        color=color,
        zorder=10,
        rotation=0,
    )
    viz_img.get_image()
    text_dims = text_obj.get_bbox_patch().get_extents()

    return text_dims.width


def multiclass_nms_alt(
    multi_bboxes,
    multi_scores,
    score_thr,
    nms_cfg,
    max_num=-1,
    score_factors=None,
    return_dist=False,
):
    """NMS for multi-class bboxes.

    Args:
        multi_bboxes (Tensor): shape (n, #class*4) or (n, 4)
        multi_scores (Tensor): shape (n, #class), where the last column
            contains scores of the background class, but this will be ignored.
        score_thr (float): bbox threshold, bboxes with scores lower than it
            will not be considered.
        nms_cfg (float): NMS IoU threshold
        max_num (int): if there are more than max_num bboxes after NMS,
            only top max_num will be kept.
        score_factors (Tensor): The factors multiplied to scores before
            applying NMS
        return_dist (bool): whether to return score dist.

    Returns:
        tuple: (bboxes, labels), tensors of shape (k, 5) and (k). Labels
            are 0-based.
    """
    num_classes = multi_scores.size(1) - 1
    # exclude background category
    if multi_bboxes.shape[1] > 4:
        # (N_b, N_c, 4)
        bboxes = multi_bboxes.view(multi_scores.size(0), -1, 4)
    else:
        bboxes = multi_bboxes[:, None].expand(-1, num_classes, 4)
    scores = multi_scores[:, :-1]

    # filter out boxes with low scores
    valid_mask = scores > score_thr  # (N_b, N_c)
    valid_box_idxes = torch.nonzero(valid_mask)[:, 0].view(-1)
    bboxes = bboxes[valid_mask]
    if score_factors is not None:
        scores = scores * score_factors[:, None]
    score_dists = scores[valid_box_idxes, :]
    # add bg column for later use.
    score_dists = torch.cat(
        (torch.zeros(score_dists.size(0), 1).to(score_dists), score_dists),
        dim=-1)
    scores = scores[valid_mask]
    labels = valid_mask.nonzero()[:, 1]

    if bboxes.numel() == 0:
        bboxes = multi_bboxes.new_zeros((0, 5))
        labels = multi_bboxes.new_zeros((0, ), dtype=torch.long)
        if return_dist:
            return bboxes, (labels, multi_bboxes.new_zeros(
                (0, num_classes + 1)))
        else:
            return bboxes, labels

    # Modified from https://github.com/pytorch/vision/blob
    # /505cd6957711af790211896d32b40291bea1bc21/torchvision/ops/boxes.py#L39.
    # strategy: in order to perform NMS independently per class.
    # we add an offset to all the boxes. The offset is dependent
    # only on the class idx, and is large enough so that boxes
    # from different classes do not overlap
    max_coordinate = bboxes.max()
    offsets = labels.to(bboxes) * (max_coordinate + 1)
    bboxes_for_nms = bboxes + offsets[:, None]
    nms_cfg_ = nms_cfg.copy()
    nms_type = nms_cfg_.pop('type', 'nms')
    nms_op = getattr(ops, nms_type)
    dets, keep = nms_op(bboxes_for_nms, scores, **nms_cfg_)
    bboxes = bboxes[keep]
    scores = dets[:, -1]  # soft_nms will modify scores
    labels = labels[keep]
    score_dists = score_dists[keep]

    if keep.size(0) > max_num:
        _, inds = scores.sort(descending=True)
        inds = inds[:max_num]
        bboxes = bboxes[inds]
        scores = scores[inds]
        labels = labels[inds]
        score_dists = score_dists[inds]

    if return_dist:
        # score_dists has bg_column
        return torch.cat([bboxes, scores[:, None]],
                         1), (labels.view(-1), score_dists)
    else:
        return torch.cat([bboxes, scores[:, None]], 1), labels.view(-1)

CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
    'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
    'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
    'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
    'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
    'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant',
    'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
    'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush', 'banner', 'blanket', 'bridge', 'cardboard',
    'counter', 'curtain', 'door-stuff', 'floor-wood', 'flower', 'fruit',
    'gravel', 'house', 'light', 'mirror-stuff', 'net', 'pillow', 'platform',
    'playingfield', 'railroad', 'river', 'road', 'roof', 'sand', 'sea',
    'shelf', 'snow', 'stairs', 'tent', 'towel', 'wall-brick', 'wall-stone',
    'wall-tile', 'wall-wood', 'water-other', 'window-blind', 'window-other',
    'tree-merged', 'fence-merged', 'ceiling-merged', 'sky-other-merged',
    'cabinet-merged', 'table-merged', 'floor-other-merged', 'pavement-merged',
    'mountain-merged', 'grass-merged', 'dirt-merged', 'paper-merged',
    'food-other-merged', 'building-other-merged', 'rock-merged',
    'wall-other-merged', 'rug-merged', 'background'
]

PREDICATES = [
    'over',
    'in front of',
    'beside',
    'on',
    'in',
    'attached to',
    'hanging from',
    'on back of',
    'falling off',
    'going down',
    'painted on',
    'walking on',
    'running on',
    'crossing',
    'standing on',
    'lying on',
    'sitting on',
    'flying over',
    'jumping over',
    'jumping from',
    'wearing',
    'holding',
    'carrying',
    'looking at',
    'guiding',
    'kissing',
    'eating',
    'drinking',
    'feeding',
    'biting',
    'catching',
    'picking',
    'playing with',
    'chasing',
    'climbing',
    'cleaning',
    'playing',
    'touching',
    'pushing',
    'pulling',
    'opening',
    'cooking',
    'talking to',
    'throwing',
    'slicing',
    'driving',
    'riding',
    'parked on',
    'driving on',
    'about to hit',
    'kicking',
    'swinging',
    'entering',
    'exiting',
    'enclosing',
    'leaning on',
]


def show_result(img,
                result,
                is_one_stage,
                num_rel=20,
                show=False,
                out_dir=None,
                out_file=None):
    # Load image
    img = mmcv.imread(img)
    img = img.copy()  # (H, W, 3)
    img_h, img_w = img.shape[:-1]
    
    # Decrease contrast
    img = PIL.Image.fromarray(img)
    converter = PIL.ImageEnhance.Color(img)
    img = converter.enhance(0.01)
    if out_file is not None:
        mmcv.imwrite(np.asarray(img), 'bw'+out_file)

    # Draw masks
    pan_results = result.pan_results

    ids = np.unique(pan_results)[::-1]
    num_classes = 133
    legal_indices = (ids != num_classes)  # for VOID label
    ids = ids[legal_indices]

    # Get predicted labels
    labels = np.array([id % INSTANCE_OFFSET for id in ids], dtype=np.int64)
    labels = [CLASSES[l] for l in labels]

    #For psgtr
    rel_obj_labels = result.labels
    rel_obj_labels = [CLASSES[l - 1] for l in rel_obj_labels]

    # (N_m, H, W)
    segms = pan_results[None] == ids[:, None, None]
    # Resize predicted masks
    segms = [
        mmcv.image.imresize(m.astype(float), (img_w, img_h)) for m in segms
    ]
    # One stage segmentation
    masks = result.masks

    # Choose colors for each instance in coco
    colormap_coco = get_colormap(len(masks)) if is_one_stage else get_colormap(len(segms))
    colormap_coco = (np.array(colormap_coco) / 255).tolist()

    # Viualize masks
    viz = Visualizer(img)
    viz.overlay_instances(
        labels=rel_obj_labels if is_one_stage else labels,
        masks=masks if is_one_stage else segms,
        assigned_colors=colormap_coco,
    )
    viz_img = viz.get_output().get_image()
    if out_file is not None:
        mmcv.imwrite(viz_img, out_file)

    # Draw relations

    # Filter out relations
    n_rel_topk = num_rel
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
    width = img_w
    curr_x = left_padding
    curr_y = top_padding
    
    # # Adjust colormaps
    # colormap_coco = [adjust_text_color(c, viz) for c in colormap_coco]
    viz_graph = VisImage(np.full((height, width, 3), 255))
    
    for i, r in enumerate(relations):
        s_idx, o_idx, rel_id = r
        s_label = rel_obj_labels[s_idx]
        o_label = rel_obj_labels[o_idx]
        rel_label = PREDICATES[rel_id]
        viz = Visualizer(img)
        viz.overlay_instances(
            labels=[s_label, o_label],
            masks=[masks[s_idx], masks[o_idx]],
            assigned_colors=[colormap_coco[s_idx], colormap_coco[o_idx]],
        )
        viz_masked_img = viz.get_output().get_image()

        viz_graph = VisImage(np.full((40, width, 3), 255))
        curr_x = 2
        curr_y = 2
        text_size = 25
        text_padding = 20
        font = 36
        text_width = draw_text(
            viz_img=viz_graph,
            text=s_label,
            x=curr_x,
            y=curr_y,
            color=colormap_coco[s_idx],
            size=text_size,
            padding=text_padding,
            font=font,
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
            font=font,
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
            font=font,
        )
        output_viz_graph = np.vstack([viz_masked_img, viz_graph.get_image()])
        if out_file is not None:
            mmcv.imwrite(output_viz_graph, osp.join(out_dir, '{}.jpg'.format(i)))

    # if out_file is not None:
    #     mmcv.imwrite(output_viz_graph, out_file)

    if not (show or out_file):
        return viz_final


# def multiclass_nms_alt(
#     multi_bboxes,
#     multi_scores,
#     score_thr,
#     nms_cfg,
#     max_num=-1,
#     score_factors=None,
#     return_inds=False,
#     return_dist=False,
# ):
#     """NMS for multi-class bboxes.

#     Args:
#         multi_bboxes (Tensor): shape (n, #class*4) or (n, 4)
#         multi_scores (Tensor): shape (n, #class), where the last column
#             contains scores of the background class, but this will be ignored.
#         score_thr (float): bbox threshold, bboxes with scores lower than it
#             will not be considered.
#         nms_thr (float): NMS IoU threshold
#         max_num (int, optional): if there are more than max_num bboxes after
#             NMS, only top max_num will be kept. Default to -1.
#         score_factors (Tensor, optional): The factors multiplied to scores
#             before applying NMS. Default to None.
#         return_inds (bool, optional): Whether return the indices of kept
#             bboxes. Default to False.

#     Returns:
#         tuple: (dets, labels, indices (optional)), tensors of shape (k, 5),
#             (k), and (k). Dets are boxes with scores. Labels are 0-based.
#     """
#     num_classes = multi_scores.size(1) - 1
#     # exclude background category
#     if multi_bboxes.shape[1] > 4:
#         bboxes = multi_bboxes.view(multi_scores.size(0), -1, 4)
#     else:
#         bboxes = multi_bboxes[:, None].expand(multi_scores.size(0), num_classes, 4)

#     scores = multi_scores[:, :-1]
#     valid_box_idxes = torch.nonzero(scores > score_thr)[:, 0].view(-1)

#     labels = torch.arange(num_classes, dtype=torch.long, device=scores.device)
#     labels = labels.view(1, -1).expand_as(scores)

#     bboxes = bboxes.reshape(-1, 4)
#     scores = scores.reshape(-1)  # flattened
#     labels = labels.reshape(-1)

#     if not torch.onnx.is_in_onnx_export():
#         # NonZero not supported  in TensorRT
#         # remove low scoring boxes
#         valid_mask = scores > score_thr
#     # multiply score_factor after threshold to preserve more bboxes, improve
#     # mAP by 1% for YOLOv3
#     if score_factors is not None:
#         # expand the shape to match original shape of score
#         score_factors = score_factors.view(-1, 1).expand(
#             multi_scores.size(0), num_classes
#         )
#         score_factors = score_factors.reshape(-1)
#         scores = scores * score_factors

#     score_dists = scores.reshape(-1, num_classes)
#     scores = score_dists[valid_box_idxes, :]
#     # add bg column for later use.
#     score_dists = torch.cat(
#         (torch.zeros(score_dists.size(0), 1).to(score_dists), score_dists), dim=-1
#     )

#     if not torch.onnx.is_in_onnx_export():
#         # NonZero not supported  in TensorRT
#         inds = valid_mask.nonzero(as_tuple=False).squeeze(1)
#         bboxes, scores, labels = bboxes[inds], scores[inds], labels[inds]
#     else:
#         # TensorRT NMS plugin has invalid output filled with -1
#         # add dummy data to make detection output correct.
#         bboxes = torch.cat([bboxes, bboxes.new_zeros(1, 4)], dim=0)
#         scores = torch.cat([scores, scores.new_zeros(1)], dim=0)
#         labels = torch.cat([labels, labels.new_zeros(1)], dim=0)

#     if bboxes.numel() == 0:
#         if torch.onnx.is_in_onnx_export():
#             raise RuntimeError(
#                 "[ONNX Error] Can not record NMS "
#                 "as it has not been executed this time"
#             )
#         dets = torch.cat([bboxes, scores[:, None]], -1)
#         if return_inds:
#             return dets, labels, inds
#         elif return_dist:
#             return dets, (labels, multi_bboxes.new_zeros((0, num_classes + 1)))
#         else:
#             return dets, labels

#     dets, keep = batched_nms(bboxes, scores, labels, nms_cfg)
#     # NOTE: `keep` is for each class independently right?

#     if max_num > 0:
#         dets = dets[:max_num]
#         keep = keep[:max_num]

#     if return_inds:
#         return dets, labels[keep], inds[keep]
#     else:
#         return dets, labels[keep]

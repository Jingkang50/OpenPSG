from typing import Tuple

import mmcv.ops as ops
import numpy as np
import torch
from detectron2.utils.colormap import colormap
from detectron2.utils.visualizer import VisImage, Visualizer

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

# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os.path as osp

import mmcv
import numpy as np
from detectron2.utils.visualizer import VisImage, Visualizer
from mmcv import Config, DictAction
from mmdet.datasets import build_dataset, replace_ImageToTensor
from mmdet.datasets.coco_panoptic import INSTANCE_OFFSET
from sentry_sdk import flush

from openpsg.utils import adjust_text_color, draw_text, get_colormap

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
                out_file=None):
    # Load image
    img = mmcv.imread(img)
    img = img.copy()  # (H, W, 3)
    img_h, img_w = img.shape[:-1]

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
    width = viz_img.shape[1]
    curr_x = left_padding
    curr_y = top_padding
    # Adjust colormaps
    colormap_coco = [adjust_text_color(c, viz) for c in colormap_coco]
    viz_graph = VisImage(np.full((height, width, 3), 255))
    for i, r in enumerate(relations):
        s_idx, o_idx, rel_id = r
        if is_one_stage:
            s_label = rel_obj_labels[s_idx]
            o_label = rel_obj_labels[o_idx]
        else:
            s_label = labels[s_idx]
            o_label = labels[o_idx]
        # Becomes 0-index
        rel_label = PREDICATES[rel_id]
        # Draw subject text
        if is_one_stage:
            text_width = draw_text(
                viz_img=viz_graph,
                text=s_label,
                x=curr_x,
                y=curr_y,
                # color=colormap_coco[s_idx],
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
                # color=colormap_coco[o_idx],
                size=text_size,
                padding=text_padding,
                # font=font,
            )
        else:
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


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet eval image prediction result for each')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('prediction_path',
                        help='prediction path where test pkl result')
    parser.add_argument('show_dir',
                        help='directory where painted images will be saved')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument('--img_idx',
                        default=[25, 73],
                        nargs='+',
                        type=int,
                        help='which image to show')
    parser.add_argument('--wait-time',
                        type=float,
                        default=0,
                        help='the interval of show (s), 0 is block')
    parser.add_argument('--topk',
                        default=20,
                        type=int,
                        help='saved Number of the highest topk '
                        'and lowest topk after index sorting')
    parser.add_argument('--show-score-thr',
                        type=float,
                        default=0,
                        help='score threshold (default: 0.)')
    parser.add_argument('--one_stage', default=False, action='store_true')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    mmcv.check_file_exist(args.prediction_path)

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # in case the test dataset is concatenated
    samples_per_gpu = 1
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
        if samples_per_gpu > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.test.pipeline = replace_ImageToTensor(
                cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        samples_per_gpu = max(
            [ds_cfg.pop('samples_per_gpu', 1) for ds_cfg in cfg.data.test])
        if samples_per_gpu > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

    # build the dataloader
    dataset = build_dataset(cfg.data.test)
    outputs = mmcv.load(args.prediction_path)

    for idx in args.img_idx:
        print(idx, flush=True)
        img = dataset[idx]['img_metas'][0].data['filename']
        result = outputs[idx]
        out_filepath = osp.join(args.show_dir, f'{idx}.png')
        show_result(img,
                    result,
                    args.one_stage,
                    args.topk,
                    out_file=out_filepath)


if __name__ == '__main__':
    main()

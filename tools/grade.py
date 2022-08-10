# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import json
import os
import random

import numpy as np
import PIL
from mmcv import Config
from mmdet.datasets.coco_panoptic import INSTANCE_OFFSET
from panopticapi.utils import rgb2id
from PIL import Image
from tqdm import tqdm

from openpsg.datasets import build_dataset
from openpsg.models.relation_heads.approaches import Result


def save_results(results):
    all_img_dicts = []
    if not os.path.isdir('submission/panseg/'):
        os.makedirs('submission/panseg/')
    for idx, result in enumerate(results):
        if not isinstance(result, Result):
            continue

        labels = result.labels
        rels = result.rels
        masks = result.masks

        segments_info = []
        img = np.full(masks.shape[1:3], 0)
        for label, mask in zip(labels, masks):
            r, g, b = random.choices(range(0, 255), k=3)
            coloring_mask = 1 * np.vstack([[mask]] * 3)
            for j, color in enumerate([r, g, b]):
                coloring_mask[j, :, :] = coloring_mask[j, :, :] * color
            img = img + coloring_mask

            segment = dict(category_id=int(label), id=rgb2id((r, g, b)))
            segments_info.append(segment)

        image_path = 'submission/panseg/%d.png' % idx
        # image_array = np.uint8(img).transpose((2,1,0))
        image_array = np.uint8(img).transpose((1, 2, 0))
        PIL.Image.fromarray(image_array).save(image_path)

        single_result_dict = dict(
            relations=rels.astype(np.int32).tolist(),
            segments_info=segments_info,
            pan_seg_file_name='%d.png' % idx,
        )

        all_img_dicts.append(single_result_dict)
    if not os.path.isdir('submission'):
        os.mkdir('submission')
    with open('submission/relation.json', 'w') as outfile:
        json.dump(all_img_dicts, outfile, default=str)


def load_results(loadpath):
    with open(os.path.join(loadpath, 'relation.json')) as infile:
        all_img_dicts = json.load(infile)

    results = []
    for single_result_dict in tqdm(all_img_dicts,
                                   desc='Loading results from json...'):
        pan_seg_filename = single_result_dict['pan_seg_file_name']
        pan_seg_filename = os.path.join(loadpath, 'panseg', pan_seg_filename)
        pan_seg_img = np.array(Image.open(pan_seg_filename))
        pan_seg_img = pan_seg_img.copy()  # (H, W, 3)
        seg_map = rgb2id(pan_seg_img)

        segments_info = single_result_dict['segments_info']
        num_obj = len(segments_info)

        # get separate masks
        labels = []
        masks = []
        for _, s in enumerate(segments_info):
            label = int(s['category_id'])
            labels.append(label)  # TODO:1-index for gt?
            masks.append(seg_map == s['id'])

        count = dict()
        pan_result = seg_map.copy()
        for _, s in enumerate(segments_info):
            label = int(s['category_id'])
            if label not in count.keys():
                count[label] = 0
            pan_result[seg_map == int(
                s['id']
            )] = label - 1 + count[label] * INSTANCE_OFFSET  # change index?
            count[label] += 1

        rel_array = np.asarray(single_result_dict['relations'])
        if len(rel_array) > 20:
            rel_array = rel_array[:20]
        rel_dists = np.zeros((len(rel_array), 57))
        for idx_rel, rel in enumerate(rel_array):
            rel_dists[idx_rel, rel[2]] += 1  # TODO:1-index for gt?

        result = Result(
            rels=rel_array,
            rel_pair_idxes=rel_array[:, :2],
            masks=masks,
            labels=np.asarray(labels),
            rel_dists=rel_dists,
            refine_bboxes=np.ones((num_obj, 5)),
            pan_results=pan_result,
        )
        results.append(result)

    return results


def parse_args():
    parser = argparse.ArgumentParser(description='MMDet eval a model')
    parser.add_argument('input_path', help='input file path')
    parser.add_argument('output_path', help='output file path')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    cfg = Config.fromfile('configs/_base_/datasets/psg_val.py')

    dataset = build_dataset(cfg.data.test)
    outputs = load_results(args.input_path)
    metric1 = dataset.evaluate(outputs, **cfg.evaluation1)
    metric2 = dataset.evaluate(outputs, **cfg.evaluation2)

    output_filename = os.path.join(args.output_path, 'scores.txt')

    with open(output_filename, 'w+') as f3:
        f3.write('Recall R 20: {}\n'.format(metric1['sgdet_recall_R_20']))
        f3.write('MeanRecall R 20: {}\n'.format(
            metric1['sgdet_mean_recall_mR_20']))
        f3.write('PQ: {}\n'.format(metric2['PQ']))


if __name__ == '__main__':
    main()

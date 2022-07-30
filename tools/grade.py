# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import random
import json

from mmcv import Config
from openpsg.datasets import build_dataset
from PIL import Image
import numpy as np
import PIL
from openpsg.models.relation_heads.approaches import Result
from panopticapi.utils import rgb2id

def save_results(results):
    all_img_dicts = []
    for idx, result in enumerate(results):
        if not isinstance(result, Result):
            continue

        labels = result.labels
        rels = result.rels
        masks = result.masks

        segments_info = []
        img = np.full(masks.shape[1:3], 0)
        for label, mask in zip(labels, masks):
            r,g,b = random.choices(range(0,255), k=3)
            coloring_mask = 1 * np.vstack([[mask]]*3)
            for j, color in enumerate([r,g,b]):
                coloring_mask[j,:,:] = coloring_mask[j,:,:] * color
            img = img + coloring_mask

            segment = dict(
                category_id=int(label),
                id=rgb2id((r,g,b))
            )
            segments_info.append(segment)

        image_path = 'submission/images/%d.png'%idx
        # image_array = np.uint8(img).transpose((2,1,0))
        image_array = np.uint8(img).transpose((1,2,0))
        PIL.Image.fromarray(image_array).save(image_path)

        single_result_dict = dict(
            relations=rels.astype(np.int32).tolist(),
            segments_info=segments_info,
            pan_seg_file_name=image_path,
        )

        all_img_dicts.append(single_result_dict)

    with open('submission/submission_result.json', 'w') as outfile:
        json.dump(all_img_dicts, outfile, default=str)


def load_results(filename):
    with open(filename) as infile:
        all_img_dicts = json.load(infile)

    results=[]
    for single_result_dict in all_img_dicts:
        pan_seg_filename = single_result_dict['pan_seg_file_name']
        # pan_seg_img = np.array(Image.open(pan_seg_filename)).transpose((1, 0, 2))
        pan_seg_img = np.array(Image.open(pan_seg_filename))
        pan_seg_img = pan_seg_img.copy()  # (H, W, 3)
        seg_map = rgb2id(pan_seg_img)

        segments_info = single_result_dict['segments_info']
        num_obj = len(segments_info)

        # get separate masks
        labels = []
        masks = []
        for _, s in enumerate(segments_info):
            label = s['category_id']
            labels.append(label) #TODO:1-index for gt?
            masks.append(seg_map == s['id'])

        rel_array = np.asarray(single_result_dict['relations'])
        if len(rel_array) > 20:
            rel_array = rel_array[:20]
        rel_dists = np.zeros((len(rel_array), 57))
        for idx_rel, rel in enumerate(rel_array):
            rel_dists[idx_rel, rel[2]] += 1 # TODO:1-index for gt?

        result = Result(
            rels=rel_array,
            rel_pair_idxes=rel_array[:, :2],
            masks=masks,
            labels=np.asarray(labels),
            rel_dists=rel_dists,
            refine_bboxes=np.ones((num_obj, 5)),
        )
        results.append(result)
    
    return results


evaluation1 = dict(metric=['sgdet'],
                  relation_mode=True,
                  classwise=True,
                  iou_thrs=0.5,
                  detection_method='pan_seg')

evaluation2 = dict(metric=['PQ'],
                  relation_mode=True,
                  classwise=True,
                  iou_thrs=0.5,
                  detection_method='pan_seg')


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet eval a model')
    parser.add_argument('config', help='config file path') # configs/_base_/datasets/psg.py
    parser.add_argument('input_path', help='input file path')
    parser.add_argument('output_path', help='output file path')
    
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)

    dataset = build_dataset(cfg.data.test)
    outputs = load_results(args.input_path)
    metric1 = dataset.evaluate(outputs, **evaluation1)
    metric2 = dataset.evaluate(outputs, **evaluation2)
    
    output_filename = os.path.join(args.output_dir, 'scores.txt')
    
    with open(output_filename, 'w') as f3:
        f3.write('Recall R 20: {}\n'.format(metric1['sgdet_recall_R_20']))
        f3.write('MeanRecall R 20: {}\n'.format(metric1['sgdet_mean_recall_mR_20']))
        f3.write('PQ: {}\n'.format(metric2['PQ']))
        f3.write('Final Score: {}\n'.format(metric1['sgdet_recall_R_20'] * 0.3 + metric1['sgdet_mean_recall_mR_20'] * 0.6 + 0.1 * metric2['PQ']))


if __name__ == '__main__':
    main()

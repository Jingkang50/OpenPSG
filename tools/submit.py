# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os

from mmcv import Config
from openpsg.datasets import build_dataset
from submit_result import load_results

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

    eval_kwargs = evaluation1
    metric = dataset.evaluate(outputs, **eval_kwargs)
    output_filename = os.path.join(args.output_dir, 'scores.txt')
    with open(output_filename, 'w') as f3:
        f3.write('Recall R 20: {}\n'.format(metric['sgdet_recall_R_20']))
        f3.write('MeanRecall R 20: {}\n'.format(metric['sgdet_mean_recall_mR_20']))

    eval_kwargs = evaluation2
    metric = dataset.evaluate(outputs, **eval_kwargs)
    output_filename = os.path.join(args.output_dir, 'scores.txt')
    with open(output_filename, 'w') as f3:
        f3.write('PQ: {}\n'.format(metric['PQ']))
        f3.write('Final Score: {}\n'.format(metric['sgdet_recall_R_20'] * 0.3 + metric['sgdet_mean_recall_mR_20'] * 0.6 + 0.1 * metric['PQ']))


if __name__ == '__main__':
    main()

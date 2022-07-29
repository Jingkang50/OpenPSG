# Copyright (c) OpenMMLab. All rights reserved.
import argparse

from mmcv import Config
from openpsg.datasets import build_dataset
from submit_result import load_results

evaluation = dict(metric=['sgdet'],
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

    eval_kwargs = evaluation
    metric = dataset.evaluate(outputs, **eval_kwargs)
    print(metric)

    # TODO:print result to output_path(args.output_path)...

    # metric_dict = dict(config=args.config, metric=metric)
    # if args.work_dir is not None:
    #     mmcv.dump(metric_dict, json_file)


if __name__ == '__main__':
    main()

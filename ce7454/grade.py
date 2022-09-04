import argparse
import os

import numpy as np


def compute_recall(gt_list, pred_list):
    score_list = np.zeros([56, 2], dtype=int)
    for gt, pred in zip(gt_list, pred_list):
        for gt_id in gt:
            # pos 0 for counting all existing relations
            score_list[gt_id][0] += 1
            if gt_id in pred:
                # pos 1 for counting relations that is recalled
                score_list[gt_id][1] += 1
    score_list = score_list[6:]
    # to avoid nan, but test set does not have empty predict
    # score_list[:,0][score_list[:,0] == 0] = 1
    meanrecall = np.mean(score_list[:, 1] / score_list[:, 0])

    return meanrecall


def parse_args():
    parser = argparse.ArgumentParser(description='MMDet eval a model')
    parser.add_argument('input_path', help='input file path')
    parser.add_argument('output_path', help='output file path')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    submit_dir = os.path.join(args.input_path, 'res')
    groundtruth_dir = os.path.join(args.input_path, 'ref')

    gt_list = []
    with open(os.path.join(groundtruth_dir, 'psg_cls_gt.txt'), 'r') as reader:
        for line in reader.readlines():
            gt_list.append(
                [int(label) for label in line.strip('/n').split(' ')])

    pred_list = []
    with open(os.path.join(submit_dir, 'result.txt'), 'r') as reader:
        for line in reader.readlines():
            pred_list.append(
                [int(label) for label in line.strip('/n').split(' ')])

    assert np.array(pred_list).shape == (
        500, 3), 'make sure the submitted file is 500 x 3'
    result = compute_recall(gt_list, pred_list)
    output_filename = os.path.join(args.output_path, 'scores.txt')

    with open(output_filename, 'w') as f3:
        f3.write('score: {}\n'.format(result))


if __name__ == '__main__':
    main()

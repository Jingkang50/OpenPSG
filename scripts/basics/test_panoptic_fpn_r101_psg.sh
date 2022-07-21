#!/bin/bash
# sh scripts/basics/test_panoptic_fpn_r101_psg.sh

GPU=1
CPU=1
node=73
PORT=29500
jobname=openpsg

PYTHONPATH='.':$PYTHONPATH \
srun -p dsta --mpi=pmi2 --gres=gpu:${GPU} -n1 \
--cpus-per-task=${CPU} --ntasks-per-node=${GPU} \
--kill-on-bad-exit=1 --job-name=${jobname} -w SG-IDC1-10-51-2-${node} \
python tools/test.py \
  configs/_base_/models/panoptic_fpn_r101_fpn_psg.py \
  work_dirs/checkpoints/panoptic_fpn_r101_fpn_1x_coco_20210820_193950-ab9157a2.pth \
  --out work_dirs/panoptic_fpn_r101_fpn/result.pkl \
  --eval PQ

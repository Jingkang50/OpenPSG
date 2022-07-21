#!/bin/bash
# sh scripts/gpsnet/train_panoptic_fpn_r50_predcls.sh

GPU=1
CPU=1
node=63
PORT=29500
jobname=openpsg

PYTHONPATH='.':$PYTHONPATH \
srun -p dsta --mpi=pmi2 --gres=gpu:${GPU} -n1 \
--cpus-per-task=${CPU} --ntasks-per-node=${GPU} \
--kill-on-bad-exit=1 --job-name=${jobname} -w SG-IDC1-10-51-2-${node} \
python tools/train.py \
  configs/gpsnet/panoptic_fpn_r50_fpn_1x_predcls_psg.py

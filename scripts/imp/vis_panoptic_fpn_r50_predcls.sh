#!/bin/bash
# sh scripts/imp/vis_panoptic_fpn_r50_predcls.sh

GPU=1
CPU=1
node=73
PORT=29500
jobname=openpsg

PYTHONPATH='.':$PYTHONPATH \
srun -p dsta --mpi=pmi2 --gres=gpu:${GPU} -n1 --ntasks-per-node=${GPU} \
--kill-on-bad-exit=1 --job-name=${jobname} -w SG-IDC1-10-51-2-${node} \
python tools/vis_results.py \
  configs/imp/panoptic_fpn_r50_fpn_1x_sgdet_psg.py \
  work_dirs/imp_panoptic_fpn_r50_fpn_1x_sgdet_psg/epoch_12.pkl \
  work_dirs/imp_panoptic_fpn_r50_fpn_1x_sgdet_psg/analyze_viz \
  --topk 20 \
  --show-score-thr 0.3

#!/bin/bash
# sh scripts/vctree/vis.sh

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
  configs/vctree/panoptic_fpn_r50_fpn_1x_sgdet_psg.py \
  work_dirs/vctree_panoptic_fpn_r50_fpn_1x_sgdet_psg/epoch_12.pth \
  --out work_dirs/vctree_panoptic_fpn_r50_fpn_1x_sgdet_psg/epoch_12.pkl

PYTHONPATH='.':$PYTHONPATH \
srun -p dsta --mpi=pmi2 --gres=gpu:${GPU} -n1 --ntasks-per-node=${GPU} \
--kill-on-bad-exit=1 --job-name=${jobname} -w SG-IDC1-10-51-2-${node} \
python tools/vis_results.py \
  configs/vctree/panoptic_fpn_r50_fpn_1x_sgdet_psg.py \
  work_dirs/vctree_panoptic_fpn_r50_fpn_1x_sgdet_psg/epoch_12.pkl \
  work_dirs/vctree_panoptic_fpn_r50_fpn_1x_sgdet_psg/analyze_viz

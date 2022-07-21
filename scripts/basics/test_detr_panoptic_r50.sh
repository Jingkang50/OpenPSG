#!/bin/bash
# sh scripts/basics/test_detr_panoptic_r50.sh

GPU=1
CPU=1
node=75
PORT=29500
jobname=openpsg

PYTHONPATH='.':$PYTHONPATH \
srun -p dsta --mpi=pmi2 --gres=gpu:${GPU} -n1 \
--cpus-per-task=${CPU} --ntasks-per-node=${GPU} \
--kill-on-bad-exit=1 --job-name=${jobname} -w SG-IDC1-10-51-2-${node} \
python -m pdb -c continue tools/test.py \
  configs/_base_/models/detr4seg_r50_psg.py \
  ./work_dirs/checkpoints/detr_pan_r50.pth \
  --out work_dirs/detr4seg_r50_psg/result.pkl \
  --show-dir work_dirs/detr4seg_r50_psg/ \
  --eval PQ

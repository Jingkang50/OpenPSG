#!/bin/bash
# sh scripts/basics/test_panoptic_fpn_r50_psg.sh

GPU=1
CPU=1
node=63
PORT=29500
jobname=openpsg

PYTHONPATH='.':$PYTHONPATH \
srun -p dsta --mpi=pmi2 --gres=gpu:${GPU} -n1 \
--cpus-per-task=${CPU} --ntasks-per-node=${GPU} \
--kill-on-bad-exit=1 --job-name=${jobname} -w SG-IDC1-10-51-2-${node} \
python -m pdb -c continue tools/test.py \
  configs/detseg/detr4seg_r101_psg.py \
  detr_pan_r101.pth \
  --out work_dirs/detr4seg_r101_psg/result.pkl \
  --eval PQ

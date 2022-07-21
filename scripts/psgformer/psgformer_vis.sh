#!/bin/bash
# sh scripts/psgformer/psgformer_vis.sh

GPU=1
CPU=4
node=79
PORT=29500
jobname=openpsg

PYTHONPATH='.':$PYTHONPATH \
srun -p dsta --mpi=pmi2 --gres=gpu:${GPU} -n1 --ntasks-per-node=${GPU} \
--kill-on-bad-exit=1 --job-name=${jobname} -w SG-IDC1-10-51-2-${node} \
python tools/vis_results_one_stage.py \
  configs/psgformer/psgformer_r50_psg.py \
  work_dirs/psgformer_r50/epoch_60_extra.pkl \
  work_dirs/psgformer_r50/epoch_60_extra \
  --topk 20 \
  --show-score-thr 0.3 \
  --one_stage

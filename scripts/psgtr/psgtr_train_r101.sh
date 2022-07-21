#!/bin/bash
# sh scripts/psgformer/psgtr_train.sh
# check gpu usage:
# srun -p dsta -w SG-IDC1-10-51-2-73 nvidia-smi
# squeue -w SG-IDC1-10-51-2-73

GPU=1
CPU=2
node=73
PORT=29500
jobname=openpsg

PYTHONPATH='.':$PYTHONPATH \
srun -p dsta --mpi=pmi2 --gres=gpu:${GPU} -n1 \
--cpus-per-task=${CPU} --ntasks-per-node=${GPU} \
--kill-on-bad-exit=1 --job-name=${jobname} -w SG-IDC1-10-51-2-${node} \
mim train mmdet \
  configs/psgtr/psgtr_r101_psg.py \
  --gpus ${GPU} \
  --launcher pytorch

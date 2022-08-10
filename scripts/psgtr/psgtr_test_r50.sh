#!/bin/bash
# sh scripts/psgtr/psgtr_test_r50.sh

GPU=1
CPU=4
node=76
PORT=29500
jobname=openpsg

PYTHONPATH='.':$PYTHONPATH \
srun -p dsta --mpi=pmi2 --gres=gpu:${GPU} -n1 \
--cpus-per-task=${CPU} --ntasks-per-node=${GPU} \
--kill-on-bad-exit=1 --job-name=${jobname} -w SG-IDC1-10-51-2-${node} \
python tools/test.py \
     configs/psgtr/psgtr_r50_psg.py \
     work_dirs/psgtr_r50_e60/epoch_60.pth \
     --eval sgdet

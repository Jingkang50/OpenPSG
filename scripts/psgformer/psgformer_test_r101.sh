#!/bin/bash
# sh scripts/psgformer/psgformer_test_r101.sh

GPU=1
CPU=4
node=73
PORT=29500
jobname=openpsg

PYTHONPATH='.':$PYTHONPATH \
srun -p dsta --mpi=pmi2 --gres=gpu:${GPU} -n1 \
--cpus-per-task=${CPU} --ntasks-per-node=${GPU} \
--kill-on-bad-exit=1 --job-name=${jobname} -w SG-IDC1-10-51-2-${node} \
python tools/test.py \
     configs/psgformer/psgformer_r101_psg.py \
     work_dirs/psgformer_r101/epoch_60.pth \
     --eval PQ

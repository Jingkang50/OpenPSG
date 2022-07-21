#!/bin/bash
# sh scripts/psgformer/psgformer_train.sh
# check gpu usage:
# srun -p dsta -w SG-IDC1-10-51-2-73 nvidia-smi
# squeue -w SG-IDC1-10-51-2-73

GPU=4
CPU=7
node=66
PORT=29500
jobname=openpsg

PYTHONPATH='.':$PYTHONPATH \
srun -p dsta --mpi=pmi2 --gres=gpu:${GPU} -n1 \
--cpus-per-task=${CPU} --ntasks-per-node=${GPU} \
--kill-on-bad-exit=1 --job-name=${jobname} -w SG-IDC1-10-51-2-${node} \
python -m torch.distributed.launch \
--nproc_per_node=${GPU} \
--master_port=$PORT \
  tools/train.py \
  configs/psgformer/psgformer_r50_psg.py \
  --gpus ${GPU} \
  --launcher pytorch

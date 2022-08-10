#!/bin/bash
# sh scripts/grade.sh

PYTHONPATH='.':$PYTHONPATH \
srun -p dsta python tools/grade.py ./submission ./submission

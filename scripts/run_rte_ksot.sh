#!/bin/bash
set -e

SEED=${SEED:-88}
DEVICE=${DEVICE:-0}

CUDA_VISIBLE_DEVICES=${DEVICE} python train.py \
  --cfg cfg/final/exps/mamba-130m/glue_rte/ksot.yaml \
  --seed ${SEED} \
  --overwrite

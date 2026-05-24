#!/bin/bash
set -e

SEED=88
DEVICE=${DEVICE:-0}

TASKS=(
  cola
  mnli
  mrpc
  qnli
  qqp
  rte
  sst2
)

for TASK in "${TASKS[@]}"; do
  echo "Running GLUE ${TASK} K-SOT with seed ${SEED}"
  CUDA_VISIBLE_DEVICES=${DEVICE} python train.py \
    --cfg cfg/final/exps/mamba-130m/glue_${TASK}/ksot.yaml \
    --seed ${SEED} \
    --overwrite
done

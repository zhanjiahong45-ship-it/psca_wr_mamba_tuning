#!/bin/bash
set -e

SEED=${SEED:-88}
DEVICE=${DEVICE:-0}

echo "Using Random Seed: ${SEED}"

echo "Starting CF-SOT MRPC..."
CUDA_VISIBLE_DEVICES=${DEVICE} python train.py \
  --cfg cfg/final/exps/mamba-130m/glue_mrpc/cf_sot.yaml \
  --seed ${SEED} \
  --overwrite \
  --output_dir outputs/cf_sot/mamba-130m/glue_mrpc

echo "Starting CF-SOT RTE..."
CUDA_VISIBLE_DEVICES=${DEVICE} python train.py \
  --cfg cfg/final/exps/mamba-130m/glue_rte/cf_sot.yaml \
  --seed ${SEED} \
  --overwrite \
  --output_dir outputs/cf_sot/mamba-130m/glue_rte

echo "Starting CF-SOT CoLA..."
CUDA_VISIBLE_DEVICES=${DEVICE} python train.py \
  --cfg cfg/final/exps/mamba-130m/glue_cola/cf_sot.yaml \
  --seed ${SEED} \
  --overwrite \
  --output_dir outputs/cf_sot/mamba-130m/glue_cola

echo "All CF-SOT GLUE experiments completed with seed ${SEED}."

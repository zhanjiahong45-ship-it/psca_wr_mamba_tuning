#!/bin/bash
set -e

SEED=${SEED:-88}
DEVICE=${DEVICE:-0}

echo "Using Random Seed: ${SEED}"

echo "Starting SNOFT-E MRPC..."
CUDA_VISIBLE_DEVICES=${DEVICE} python train.py \
  --cfg cfg/final/exps/mamba-130m/glue_mrpc/snoft_e.yaml \
  --seed ${SEED} \
  --overwrite \
  --output_dir outputs/snoft_e/mamba-130m/glue_mrpc

echo "Starting SNOFT-E RTE..."
CUDA_VISIBLE_DEVICES=${DEVICE} python train.py \
  --cfg cfg/final/exps/mamba-130m/glue_rte/snoft_e.yaml \
  --seed ${SEED} \
  --overwrite \
  --output_dir outputs/snoft_e/mamba-130m/glue_rte

echo "Starting SNOFT-E CoLA..."
CUDA_VISIBLE_DEVICES=${DEVICE} python train.py \
  --cfg cfg/final/exps/mamba-130m/glue_cola/snoft_e.yaml \
  --seed ${SEED} \
  --overwrite \
  --output_dir outputs/snoft_e/mamba-130m/glue_cola

echo "SNOFT-E GLUE experiments completed with seed ${SEED}."

#!/bin/bash
set -e

SEED=${SEED:-88}
DEVICE=${DEVICE:-0}

echo "Using Random Seed: ${SEED}"

echo "Starting TF-SOT MRPC..."
CUDA_VISIBLE_DEVICES=${DEVICE} python train.py \
  --cfg cfg/final/exps/mamba-130m/glue_mrpc/tf_sot.yaml \
  --seed ${SEED} \
  --overwrite \
  --output_dir outputs/tf_sot/mamba-130m/glue_mrpc

echo "Starting TF-SOT RTE..."
CUDA_VISIBLE_DEVICES=${DEVICE} python train.py \
  --cfg cfg/final/exps/mamba-130m/glue_rte/tf_sot.yaml \
  --seed ${SEED} \
  --overwrite \
  --output_dir outputs/tf_sot/mamba-130m/glue_rte

echo "Starting TF-SOT CoLA..."
CUDA_VISIBLE_DEVICES=${DEVICE} python train.py \
  --cfg cfg/final/exps/mamba-130m/glue_cola/tf_sot.yaml \
  --seed ${SEED} \
  --overwrite \
  --output_dir outputs/tf_sot/mamba-130m/glue_cola

echo "Starting TF-SOT SST2..."
CUDA_VISIBLE_DEVICES=${DEVICE} python train.py \
  --cfg cfg/final/exps/mamba-130m/glue_sst2/tf_sot.yaml \
  --seed ${SEED} \
  --overwrite \
  --output_dir outputs/tf_sot/mamba-130m/glue_sst2

echo "Starting TF-SOT QNLI..."
CUDA_VISIBLE_DEVICES=${DEVICE} python train.py \
  --cfg cfg/final/exps/mamba-130m/glue_qnli/tf_sot.yaml \
  --seed ${SEED} \
  --overwrite \
  --output_dir outputs/tf_sot/mamba-130m/glue_qnli

echo "Starting TF-SOT QQP..."
CUDA_VISIBLE_DEVICES=${DEVICE} python train.py \
  --cfg cfg/final/exps/mamba-130m/glue_qqp/tf_sot.yaml \
  --seed ${SEED} \
  --overwrite \
  --output_dir outputs/tf_sot/mamba-130m/glue_qqp

echo "Starting TF-SOT MNLI..."
CUDA_VISIBLE_DEVICES=${DEVICE} python train.py \
  --cfg cfg/final/exps/mamba-130m/glue_mnli/tf_sot.yaml \
  --seed ${SEED} \
  --overwrite \
  --output_dir outputs/tf_sot/mamba-130m/glue_mnli

echo "All TF-SOT GLUE experiments completed with seed ${SEED}."

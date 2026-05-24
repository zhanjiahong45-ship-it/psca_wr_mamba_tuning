#!/usr/bin/env bash
set -euo pipefail

LRS=(
  0.001
  0.003
)

GPU="${CUDA_VISIBLE_DEVICES:-0}"

for lr in "${LRS[@]}"; do
  echo "===== Running PSCA-WR QNLI seed42 batch_size=16 lr=${lr} ====="
  CUDA_VISIBLE_DEVICES="${GPU}" python train.py \
    --cfg cfg/final/exps/mamba-130m/glue_qnli/psca_wr.yaml \
    --output_dir "outputs/psca_wr/mamba-130m/glue_qnli/seed42_bs16_lr${lr}" \
    --overwrite \
    --seed 42 \
    --batch_size 16 \
    --learning_rate "${lr}" \
    --use_psca_wr True \
    --psca_rank 8 \
    --psca_alpha 1.0 \
    --psca_adapt_b True \
    --psca_adapt_c True \
    --psca_init_zero True \
    --psca_use_projector_shift True \
    --psca_projector_residual True \
    --psca_projector_scale 0.01 \
    --bridge_enabled False
done

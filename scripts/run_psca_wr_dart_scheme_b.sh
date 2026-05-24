#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="${OUT_DIR:-outputs/psca_wr/mamba-130m/dart/seed88}"
CFG="${CFG:-cfg/final/exps/mamba-130m/dart/psca_wr.yaml}"
GPU="${CUDA_VISIBLE_DEVICES:-0}"
CLEAN_OUTPUT="${CLEAN_OUTPUT:-1}"

export CUDA_VISIBLE_DEVICES="${GPU}"
export TOKENIZERS_PARALLELISM=false
export WANDB_DISABLED=true
export WANDB_MODE=disabled

if [[ "${CLEAN_OUTPUT}" == "1" ]]; then
  rm -rf "${OUT_DIR}"
fi

python train.py \
  --cfg "${CFG}" \
  --output_dir "${OUT_DIR}" \
  --overwrite \
  --seed 88 \
  --learning_rate 0.003 \
  --batch_size 16 \
  --skip_eval \
  --use_psca_wr True \
  --psca_rank 8 \
  --psca_alpha 1.0 \
  --psca_adapt_b True \
  --psca_adapt_c True \
  --psca_init_zero True \
  --psca_use_projector_shift True \
  --psca_projector_residual True \
  --psca_projector_scale 0.01 \
  --psca_fallback_lite False \
  --bridge_enabled False

python scripts/reval_nlg_checkpoints.py \
  --cfg "${CFG}" \
  --output_dir "${OUT_DIR}" \
  --eval_batch_size 1 \
  --num_data_workers 0

#!/usr/bin/env bash
set -euo pipefail

TASKS=(
  glue_mrpc
  glue_rte
  glue_cola
  glue_sst2
  glue_qnli
  glue_qqp
  glue_mnli
)

COMMON_ARGS=(
  --overwrite
  --seed 42
  --learning_rate 0.002
  --use_psca_wr True
  --psca_rank 8
  --psca_alpha 1.0
  --psca_adapt_b True
  --psca_adapt_c True
  --psca_init_zero True
  --psca_use_projector_shift True
  --psca_projector_residual True
  --psca_projector_scale 0.01
  --bridge_enabled False
)

for task in "${TASKS[@]}"; do
  echo "===== Running PSCA-WR on ${task} ====="
  python train.py \
    --cfg "cfg/final/exps/mamba-130m/${task}/psca_wr.yaml" \
    --output_dir "outputs/psca_wr/mamba-130m/${task}/seed42" \
    "${COMMON_ARGS[@]}"
done

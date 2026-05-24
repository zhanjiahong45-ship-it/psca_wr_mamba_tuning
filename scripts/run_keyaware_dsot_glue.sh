#!/usr/bin/env bash
set -e

METHOD=${1:-keyaware_dsot}
SEED=${2:-88}
BETA=${3:-0.1}
BACKEND=${4:-cuda}

TASKS=(mrpc rte cola sst2 qnli qqp mnli)

for TASK in "${TASKS[@]}"; do
  echo "Running ${TASK}"
  python scripts/train_keyaware_dsot_glue.py \
    --method ${METHOD} \
    --model_name_or_path state-spaces/mamba-130m \
    --task_name ${TASK} \
    --output_dir outputs/keyaware_dsot_glue/${TASK}/seed_${SEED}_beta${BETA} \
    --align_with_sot_config true \
    --target_layer 12 \
    --delta 2 \
    --key_select_mode percentile_band \
    --key_percentile_low 0.7 \
    --key_percentile_high 0.9 \
    --beta_key_context ${BETA} \
    --lambda_key_recall 0.0 \
    --max_length 128 \
    --always_train_probe true \
    --seed ${SEED} \
    --backend ${BACKEND} \
    --device cuda
done

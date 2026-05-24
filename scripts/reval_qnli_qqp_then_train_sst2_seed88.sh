#!/usr/bin/env bash
set -euo pipefail

mkdir -p \
  outputs/sdft/mamba-130m/glue_qnli/seed88 \
  outputs/sdft/mamba-130m/glue_qqp/seed88 \
  logs/sdft

#echo "[1/3] Re-evaluating QNLI SDFT seed88 checkpoints"
#CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true WANDB_MODE=disabled python scripts/reval_mnli_sdft_checkpoints.py \
#  --output_dir outputs/sdft/mamba-130m/glue_qnli/seed88 \
#  --cfg cfg/final/exps/mamba-130m/glue_qnli/sdft.yaml \
#  --save_file outputs/sdft/mamba-130m/glue_qnli/seed88/qnli_reval_results.json \
#  --num_data_workers 8 \
#  2>&1 | tee outputs/sdft/mamba-130m/glue_qnli/seed88/qnli_reval.log

echo "[2/3] Re-evaluating QQP SDFT seed88 checkpoints"
CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true WANDB_MODE=disabled python scripts/reval_mnli_sdft_checkpoints.py \
  --output_dir outputs/sdft/mamba-130m/glue_qqp/seed88 \
  --cfg cfg/final/exps/mamba-130m/glue_qqp/sdft.yaml \
  --checkpoint checkpoint-90962 \
  --checkpoint checkpoint-181924 \
  --checkpoint checkpoint-272886 \
  --save_file outputs/sdft/mamba-130m/glue_qqp/seed88/qqp_reval_results.json \
  --num_data_workers 8 \
  2>&1 | tee outputs/sdft/mamba-130m/glue_qqp/seed88/qqp_reval.log

echo "[3/3] Training SST2 SDFT seed88"
CUDA_VISIBLE_DEVICES=0 WANDB_MODE=offline python train.py \
  --cfg cfg/final/exps/mamba-130m/glue_sst2/sdft.yaml \
  --overwrite \
  2>&1 | tee logs/sdft/sst2_seed88.log

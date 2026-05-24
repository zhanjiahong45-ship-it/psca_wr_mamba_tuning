#!/bin/bash
set -e

export CUDA_VISIBLE_DEVICES=0
SEED=88

echo "Starting MRPC AdaMix-SOT..."
python train.py --cfg cfg/final/exps/mamba-130m/glue_mrpc/adamix_sot.yaml --seed ${SEED} --overwrite

echo "Starting RTE AdaMix-SOT..."
python train.py --cfg cfg/final/exps/mamba-130m/glue_rte/adamix_sot.yaml --seed ${SEED} --overwrite

echo "Starting CoLA AdaMix-SOT..."
python train.py --cfg cfg/final/exps/mamba-130m/glue_cola/adamix_sot.yaml --seed ${SEED} --overwrite

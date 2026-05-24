#!/bin/bash
SEED=88

echo "Using Random Seed: $SEED"

echo "Starting MRPC..."
python run_all.py train.py --device 0 --cfg cfg/final/exps/mamba-130m/glue_mrpc/state_tuning.yaml -- --seed $SEED --overwrite

echo "Starting RTE..."
python run_all.py train.py --device 0 --cfg cfg/final/exps/mamba-130m/glue_rte/state_tuning.yaml -- --seed $SEED --overwrite

echo "Starting CoLA..."
python run_all.py train.py --device 0 --cfg cfg/final/exps/mamba-130m/glue_cola/state_tuning.yaml -- --seed $SEED --overwrite

echo "Starting SST2..."
python run_all.py train.py --device 0 --cfg cfg/final/exps/mamba-130m/glue_sst2/state_tuning.yaml -- --seed $SEED --overwrite

echo "Starting QNLI..."
python run_all.py train.py --device 0 --cfg cfg/final/exps/mamba-130m/glue_qnli/state_tuning.yaml -- --seed $SEED --overwrite

echo "Starting QQP..."
python run_all.py train.py --device 0 --cfg cfg/final/exps/mamba-130m/glue_qqp/state_tuning.yaml -- --seed $SEED --overwrite

echo "Starting MNLI..."
python run_all.py train.py --device 0 --cfg cfg/final/exps/mamba-130m/glue_mnli/state_tuning.yaml -- --seed $SEED --overwrite

echo "All SOT experiments completed with seed $SEED!"

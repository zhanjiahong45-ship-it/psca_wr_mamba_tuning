#!/bin/bash
SEED=88

echo "Using Random Seed: $SEED"

echo "Starting MRPC..."
python run_all.py train.py --device 0 --cfg cfg/final/exps/mamba-130m/glue_mrpc/dsot.yaml -- --seed $SEED --overwrite

echo "Starting RTE..."
python run_all.py train.py --device 0 --cfg cfg/final/exps/mamba-130m/glue_rte/dsot.yaml -- --seed $SEED --overwrite

echo "Starting COLA..."
python run_all.py train.py --device 0 --cfg cfg/final/exps/mamba-130m/glue_cola/dsot.yaml -- --seed $SEED --overwrite

echo "Starting SST2..."
python run_all.py train.py --device 0 --cfg cfg/final/exps/mamba-130m/glue_sst2/dsot.yaml -- --seed $SEED --overwrite

echo "Starting QNLI..."
python run_all.py train.py --device 0 --cfg cfg/final/exps/mamba-130m/glue_qnli/dsot.yaml -- --seed $SEED --overwrite

echo "Starting QQP..."
python run_all.py train.py --device 0 --cfg cfg/final/exps/mamba-130m/glue_qqp/dsot.yaml -- --seed $SEED --overwrite

echo "Starting MNLI..."
python run_all.py train.py --device 0 --cfg cfg/final/exps/mamba-130m/glue_mnli/dsot.yaml -- --seed $SEED --overwrite

echo "All DSOT experiments completed with seed $SEED!"

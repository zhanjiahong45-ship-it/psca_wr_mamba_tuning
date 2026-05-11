#!/bin/bash

echo "Starting RTE..."
python run_all.py train.py --device 0 --cfg cfg/final/exps/mamba-130m/glue_rte/dsot.yaml

echo "Starting COLA..."
python run_all.py train.py --device 0 --cfg cfg/final/exps/mamba-130m/glue_cola/dsot.yaml

echo "Starting SST2..."
python run_all.py train.py --device 0 --cfg cfg/final/exps/mamba-130m/glue_sst2/dsot.yaml

echo "Starting QNLI..."
python run_all.py train.py --device 0 --cfg cfg/final/exps/mamba-130m/glue_qnli/dsot.yaml

echo "Starting QQP..."
python run_all.py train.py --device 0 --cfg cfg/final/exps/mamba-130m/glue_qqp/dsot.yaml

echo "Starting MNLI..."
python run_all.py train.py --device 0 --cfg cfg/final/exps/mamba-130m/glue_mnli/dsot.yaml

echo "All DSOT experiments completed!"
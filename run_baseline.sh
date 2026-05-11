cat << 'EOF' > run_baseline.sh
#!/bin/bash
# 依次运行 GLUE 的 7 个子任务 (使用原作者的 state_tuning.yaml 配置)

#echo "Starting MRPC..."
#python run_all.py train.py --device 0 --cfg cfg/final/exps/mamba-130m/glue_mrpc/state_tuning.yaml

echo "Starting RTE..."
python run_all.py train.py --device 0 --cfg cfg/final/exps/mamba-130m/glue_rte/state_tuning.yaml

echo "Starting CoLA..."
python run_all.py train.py --device 0 --cfg cfg/final/exps/mamba-130m/glue_cola/state_tuning.yaml

echo "Starting SST-2..."
python run_all.py train.py --device 0 --cfg cfg/final/exps/mamba-130m/glue_sst2/state_tuning.yaml


echo "Starting QNLI..."
python run_all.py train.py --device 0 --cfg cfg/final/exps/mamba-130m/glue_qnli/state_tuning.yaml

echo "Starting QQP..."
python run_all.py train.py --device 0 --cfg cfg/final/exps/mamba-130m/glue_qqp/state_tuning.yaml

echo "Starting MNLI..."
python run_all.py train.py --device 0 --cfg cfg/final/exps/mamba-130m/glue_mnli/state_tuning.yaml

echo "All GLUE tasks finished!"
EOF
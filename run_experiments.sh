#!/bin/bash
# 初始化 Conda 环境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate mamba-ssm

# 定义日志文件，带上当前时间戳 (注意：等号两边不能有空格)
LOG_FILE="exp_logs_$(date +%Y%m%d_%H%M).log"
echo "Logging all output to $LOG_FILE..."

# 使用大括号包裹整个执行块
{
    echo "===== Experiments Started at $(date) ====="

    # 定义要运行的配置文件后缀列表 (注意：等号两边不能有空格)
    CONFIGS=(
        "glue_qnli/state_tuning.yaml"
        "glue_qqp/state_tuning.yaml"
        "glue_mnli/state_tuning.yaml"
        "glue_qnli/dsot.yaml"
        "glue_qqp/dsot.yaml"
        "glue_mnli/dsot.yaml"
    )

    # 循环执行每个任务
    for cfg in "${CONFIGS[@]}"; do
        full_path="cfg/final/exps/mamba-130m/$cfg"
        echo "--------------------------------------------------------"
        echo ">>> [$(date +'%H:%M:%S')] Starting: $cfg"

        # 执行训练脚本
        python run_all.py train.py --device 0 --cfg "$full_path"

        # 检查上一条命令的退出状态码
        if [ $? -eq 0 ]; then
            echo ">>> [SUCCESS] Finished $cfg at $(date +'%H:%M:%S')"
        else
            echo ">>> [ERROR] Failed on $cfg at $(date +'%H:%M:%S')! Moving to next..."
        fi
    done

    echo "--------------------------------------------------------"
    echo "===== All experiments completed at $(date) ====="
} 2>&1 | tee "$LOG_FILE"
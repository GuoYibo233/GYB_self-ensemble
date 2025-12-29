#!/bin/bash

# 切换到脚本所在目录的上级目录（GYB_self-ensemble）
cd "$(dirname "$0")/.." || exit 1

# ======================= 全局变量配置 =======================
# 可编辑的全局变量，用于指定 GPU、数据集、模型等
GPU_IDS="0 1 2 3"  # 要使用的GPU ID列表，用空格分隔，如 "0 2 4 6" 或 "0 1 2 3"
FEWSHOTS_LIST="0 5"  # Few-shot数量列表，支持多个值用空格分隔，如 "0 5 10"
DATASETS="commonsense mmlu logiqa"  # 数据集，支持多个用空格分隔，如 "commonsense mmlu logiqa"
MODELS="llama3.2_3b qwen2.5_3b qwen3_4b pythia_2.8b"  # 模型列表，用空格分隔

# Ensemble参数配置
NUM_PARAPHRASES=3
LOGITS_ENSEMBLE_METHOD="avg"
ENSEMBLE_METHOD="layer_output_avg"
ENSEMBLE_ALPHA=0.2
TOKEN_MODE="all"
MULTILAYER_FLAG="--multilayer"  # 如果不需要multilayer，设置为空字符串 ""
# ============================================================

# 设置日志
SCRIPT_NAME=$(basename "$0" .sh)
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="$HOME/self-ensemble/logs"
LOG_FILE="${LOG_DIR}/${SCRIPT_NAME}_${TIMESTAMP}.log"

# 创建日志目录（如果不存在）
mkdir -p "$LOG_DIR"

# 重定向所有输出到日志文件，同时显示在终端
exec > >(tee -a "$LOG_FILE") 2>&1

echo "==============================================="
echo "Script started at: $(date)"
echo "Log file: $LOG_FILE"
echo "==============================================="
echo ""
echo "Running ensemble ppl evaluation with the following configuration:"
echo "GPU_IDS: $GPU_IDS"
echo "FEWSHOTS_LIST: $FEWSHOTS_LIST"
echo "DATASETS: $DATASETS"
echo "MODELS: $MODELS"
echo "Working directory: $(pwd)"

# 函数：根据模型名称获取对应的层数
get_layer_for_model() {
    local model=$1
    case $model in
        llama3.2_1b|llama3.2_1b_it)
            echo 12
            ;;
        llama3.2_3b|llama3.2_3b_it)
            echo 21
            ;;
        llama3.1_8b|llama3.1_8b_it)
            echo 24
            ;;
        qwen2.5_3b|qwen2.5_3b_it)
            echo 27
            ;;
        qwen2.5_7b|qwen2.5_7b_it)
            echo 21
            ;;
        qwen2.5_14b|qwen2.5_14b_it)
            echo 36
            ;;
        qwen3_4b)
            echo 27
            ;;
        qwen3_30b)
            echo 36
            ;;
        qwen3_235b)
            echo 71
            ;;
        pythia_2.8b)
            echo 24
            ;;
        *)
            echo "Unknown MODEL: $model" >&2
            exit 1
            ;;
    esac
}

# 收集所有任务
TASKS=()
for MODEL in $MODELS ; do
    for DATASET in $DATASETS; do
        for FEWSHOTS in $FEWSHOTS_LIST; do
            LAYER=$(get_layer_for_model $MODEL)
            
            # 添加 Baseline 任务
            TASKS+=("baseline|$MODEL|$DATASET|$FEWSHOTS|$LAYER")
            
            # 添加 Ensemble 任务
            TASKS+=("ensemble|$MODEL|$DATASET|$FEWSHOTS|$LAYER")
        done
    done
done

echo "Total tasks: ${#TASKS[@]}"

# 定义执行任务的函数
run_task_group() {
    local device=$1
    shift
    local tasks=("$@")
    
    for task in "${tasks[@]}"; do
        IFS='|' read -r TASK_TYPE MODEL DATASET FEWSHOTS LAYER <<< "$task"
        
        if [ "$TASK_TYPE" == "baseline" ]; then
            echo "GPU $device: Running Baseline for $MODEL on $DATASET with $FEWSHOTS few-shots"
            CUDA_VISIBLE_DEVICES=$device python3 parallel_ensemble_perplexity.py \
                --model $MODEL \
                --dataset $DATASET \
                --num_paraphrases $NUM_PARAPHRASES \
                --num_fewshots $FEWSHOTS \
                --is_baseline
        elif [ "$TASK_TYPE" == "ensemble" ]; then
            echo "GPU $device: Running Ensemble for $MODEL on $DATASET with $FEWSHOTS few-shots (Layer $LAYER)"
            CUDA_VISIBLE_DEVICES=$device python3 parallel_ensemble_perplexity.py \
                --model $MODEL \
                --dataset $DATASET \
                --num_paraphrases $NUM_PARAPHRASES \
                --num_fewshots $FEWSHOTS \
                --logits_ensemble_method $LOGITS_ENSEMBLE_METHOD \
                --ensemble_method $ENSEMBLE_METHOD \
                --ensemble_layer $LAYER \
                --ensemble_alpha $ENSEMBLE_ALPHA \
                --token_mode $TOKEN_MODE \
                $MULTILAYER_FLAG
        fi
    done
}

# 将任务分配到指定的GPU
GPU_ARRAY=($GPU_IDS)
NUM_GPUS=${#GPU_ARRAY[@]}
TASKS_PER_GPU=$((${#TASKS[@]} / NUM_GPUS))
REMAINDER=$((${#TASKS[@]} % NUM_GPUS))

echo "Number of GPUs: $NUM_GPUS"
echo "Tasks per GPU: $TASKS_PER_GPU (with $REMAINDER extra tasks distributed)"

# 启动并行进程，每个GPU一个
for idx in $(seq 0 $((NUM_GPUS - 1))); do
    gpu_id=${GPU_ARRAY[$idx]}
    start_idx=$((idx * TASKS_PER_GPU + (idx < REMAINDER ? idx : REMAINDER)))
    if [ $idx -lt $REMAINDER ]; then
        count=$((TASKS_PER_GPU + 1))
    else
        count=$TASKS_PER_GPU
    fi
    
    # 提取该GPU的任务
    gpu_tasks=("${TASKS[@]:$start_idx:$count}")
    
    if [ ${#gpu_tasks[@]} -gt 0 ]; then
        echo "Assigning ${#gpu_tasks[@]} tasks to GPU $gpu_id"
        run_task_group $gpu_id "${gpu_tasks[@]}" &
    fi
done

# 等待所有后台进程完成
wait
echo ""
echo "==============================================="
echo "All tasks completed!"
echo "Script ended at: $(date)"
echo "Log file: $LOG_FILE"
echo "==============================================="

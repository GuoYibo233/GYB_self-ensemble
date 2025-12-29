#!/bin/bash

# 切换到脚本所在目录的上级目录（GYB_self-ensemble）
cd "$(dirname "$0")/.." || exit 1

NUM_FEWSHOTS=${1:-5}
DATASETS=${2:-"hotpot"} # TODO: Support other datasets, including commonsense, mmlu, logiqa, hotpotqa
echo "Running baseline methods with $NUM_FEWSHOTS few-shots."
echo "Working directory: $(pwd)"

MODELS="llama3.2_3b qwen2.5_3b qwen3_4b pythia_2.8b qwen3_30b" 
#MODELS="qwen3_30b"

# 收集所有任务
TASKS=()
for MODEL in $MODELS ; do
    for DATASET in $DATASETS; do
        for FEWSHOTS in 0 5; do
            if [ "$MODEL" == "llama3.2_1b" ] || [ "$MODEL" == "llama3.2_1b_it" ]; then
        LAYER=12
    elif [ "$MODEL" == "llama3.2_3b" ] || [ "$MODEL" == "llama3.2_3b_it" ]; then
        LAYER=21
    elif [ "$MODEL" == "llama3.1_8b" ] || [ "$MODEL" == "llama3.1_8b_it" ]; then
        LAYER=24
    elif [ "$MODEL" == "qwen2.5_3b" ] || [ "$MODEL" == "qwen2.5_3b_it" ]; then
        LAYER=27
    elif [ "$MODEL" == "qwen2.5_7b" ] || [ "$MODEL" == "qwen2.5_7b_it" ]; then
        LAYER=21
    elif [ "$MODEL" == "qwen2.5_14b" ] || [ "$MODEL" == "qwen2.5_14b_it" ]; then
        LAYER=36
    elif [ "$MODEL" == "qwen3_4b" ]; then
        LAYER=27
    elif [ "$MODEL" == "qwen3_30b" ]; then
        LAYER=36
    elif [ "$MODEL" == "qwen3_235b" ]; then
        LAYER=71
    elif [ "$MODEL" == "pythia_2.8b" ]; then
        LAYER=24
    else
        echo "Unknown MODEL: $MODEL"
        exit 1
    fi
            TASKS+=("$MODEL|$DATASET|$FEWSHOTS|$LAYER")
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
        IFS='|' read -r MODEL DATASET FEWSHOTS LAYER <<< "$task"
        echo "GPU $device: Running $MODEL on $DATASET with $FEWSHOTS few-shots (Layer $LAYER)"
        CUDA_VISIBLE_DEVICES=$device python3 parallel_ensemble.py \
            --ensemble_method layer_output_avg \
            --model $MODEL \
            --dataset $DATASET \
            --num_paraphrases 5 \
            --num_fewshots $FEWSHOTS \
            --num_samples 1 \
            --token_mode last \
            --ensemble_layer $LAYER
    done
}

# 将任务分配到10个GPU
NUM_GPUS=10
TASKS_PER_GPU=$((${#TASKS[@]} / NUM_GPUS))
REMAINDER=$((${#TASKS[@]} % NUM_GPUS))

echo "Tasks per GPU: $TASKS_PER_GPU (with $REMAINDER extra tasks distributed)"

# 启动10个并行进程，每个GPU一个
for gpu_id in {0..9}; do
    start_idx=$((gpu_id * TASKS_PER_GPU + (gpu_id < REMAINDER ? gpu_id : REMAINDER)))
    if [ $gpu_id -lt $REMAINDER ]; then
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
echo "All tasks completed!"

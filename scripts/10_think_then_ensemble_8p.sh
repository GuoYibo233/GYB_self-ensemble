

DEVICE=$1

MODEL=${2:-"qwen3_14b"} # qwen3_4b, qwen3_1.7b, qwen3_8b, qwen3_14b, qwen3_32b
DO_SELFPARAPHRASE=${3:-"false"} # "true" or "false"


BASELINE_DATA_ROOT=/home/xzhao/workspace/GYB_self-ensemble/datasets/hotpot-reasoning-debug/$MODEL
PARAPHRASE_DATA_ROOT=/home/xzhao/workspace/GYB_self-ensemble/datasets/hotpot-debug/$MODEL
if [ "$DO_SELFPARAPHRASE" = "true" ]; then
    PARAPHRASE_FLAG="--additional_paraphrases_file $PARAPHRASE_DATA_ROOT/paraphrases_dataset/paraphrases.num_para4.max_rounds5.temp1.5.topp0.95.jsonl"
    ENSEMBLE_FILE1=$BASELINE_DATA_ROOT/baseline_per_prompt.0shots.0paras.temp1.0.topp0.95.maxnew2048.repeat8.selfparas.feather
    ENSEMBLE_FILE2=$BASELINE_DATA_ROOT/baseline_per_prompt.0shots.3paras.temp1.0.topp0.95.maxnew2048.repeat2.selfparas.feather
else
    PARAPHRASE_FLAG=""
    ENSEMBLE_FILE1=$BASELINE_DATA_ROOT/baseline_per_prompt.0shots.0paras.temp1.0.topp0.95.maxnew2048.repeat8.feather
    ENSEMBLE_FILE2=$BASELINE_DATA_ROOT/baseline_per_prompt.0shots.3paras.temp1.0.topp0.95.maxnew2048.repeat2.feather
fi

CUDA_VISIBLE_DEVICES=$DEVICE python3 generate_baseline.py \
    --model $MODEL \
    --method per_prompt \
    --dataset hotpot \
    --num_fewshots 0 \
    --temperature 1 \
    --top_p 0.95 \
    --repeat 8 \
    --reasoning \
    --debug \
    --max_new_tokens 2048 \
    --num_paraphrases 0 \
    $PARAPHRASE_FLAG


CUDA_VISIBLE_DEVICES=$DEVICE python3 generate_baseline.py \
    --model $MODEL \
    --method per_prompt \
    --dataset hotpot \
    --num_fewshots 0 \
    --temperature 1 \
    --top_p 0.95 \
    --repeat 2 \
    --reasoning \
    --debug \
    --max_new_tokens 2048 \
    --num_paraphrases 3 \
    $PARAPHRASE_FLAG

CUDA_VISIBLE_DEVICES=$DEVICE python3 parallel_ensemble.py \
    --logits_ensemble_method avg \
    --model $MODEL \
    --dataset hotpot \
    --reasoning \
    --logits_ensemble_method avg \
    --debug \
    --baseline_file $ENSEMBLE_FILE1

CUDA_VISIBLE_DEVICES=$DEVICE python3 parallel_ensemble.py \
    --logits_ensemble_method avg \
    --model $MODEL \
    --dataset hotpot \
    --reasoning \
    --logits_ensemble_method avg \
    --debug \
    --baseline_file $ENSEMBLE_FILE2
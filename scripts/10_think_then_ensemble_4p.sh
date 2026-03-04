

DEVICE=$1
MODEL=${2:-"qwen3_4b"} # qwen3_4b, qwen3_1.7b, qwen3_8b, qwen3_14b, qwen3_32b
DO_SELFPARAPHRASE=${3:-"false"} # "true" or "false"
VERSION=${4:-""}
VERSION_SUFFIX=""
VERSION_FLAG=""

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
elif [ "$MODEL" == "qwen3_1.7b" ]; then
    LAYER=21
elif [ "$MODEL" == "qwen3_4b" ]; then
    LAYER=27
elif [ "$MODEL" == "qwen3_8b" ]; then
    LAYER=27
elif [ "$MODEL" == "qwen3_14b" ]; then
    LAYER=30
elif [ "$MODEL" == "qwen3_30b" ]; then
    LAYER=36
elif [ "$MODEL" == "qwen3_32b" ]; then
    LAYER=48
elif [ "$MODEL" == "qwen3_235b" ]; then
    LAYER=71
elif [ "$MODEL" == "pythia_2.8b" ]; then
    LAYER=24
else
    echo "Unknown MODEL: $MODEL"
    exit 1
fi

if [ "$VERSION" != "" ]; then
    VERSION_SUFFIX=".$VERSION"
    VERSION_FLAG="--version $VERSION"
fi

BASELINE_DATA_ROOT=/home/xzhao/workspace/GYB_self-ensemble/datasets/hotpot-reasoning-debug/$MODEL
PARAPHRASE_DATA_ROOT=/home/xzhao/workspace/GYB_self-ensemble/datasets/hotpot-debug/$MODEL
if [ "$DO_SELFPARAPHRASE" = "true" ]; then
    PARAPHRASE_FLAG="--additional_paraphrases_file $PARAPHRASE_DATA_ROOT/paraphrases_dataset/paraphrases.num_para4.max_rounds5.temp1.5.topp0.95.jsonl"
    ENSEMBLE_FILE1=$BASELINE_DATA_ROOT/baseline_per_prompt.0shots.0paras.temp1.0.topp0.95.maxnew2048.repeat4.selfparas$VERSION_SUFFIX.feather
    ENSEMBLE_FILE2=$BASELINE_DATA_ROOT/baseline_per_prompt.0shots.1paras.temp1.0.topp0.95.maxnew2048.repeat2.selfparas$VERSION_SUFFIX.feather
else
    PARAPHRASE_FLAG=""
    ENSEMBLE_FILE1=$BASELINE_DATA_ROOT/baseline_per_prompt.0shots.0paras.temp1.0.topp0.95.maxnew2048.repeat4$VERSION_SUFFIX.feather
    ENSEMBLE_FILE2=$BASELINE_DATA_ROOT/baseline_per_prompt.0shots.1paras.temp1.0.topp0.95.maxnew2048.repeat2$VERSION_SUFFIX.feather
fi

CUDA_VISIBLE_DEVICES=$DEVICE python3 generate_baseline.py \
    --model $MODEL \
    --method per_prompt \
    --dataset hotpot \
    --num_fewshots 0 \
    --temperature 1 \
    --top_p 0.95 \
    --repeat 4 \
    --reasoning \
    --debug \
    --max_new_tokens 2048 \
    --num_paraphrases 0 \
    $VERSION_FLAG \
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
    --num_paraphrases 1 \
    $VERSION_FLAG \
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


CUDA_VISIBLE_DEVICES=$DEVICE python3 parallel_ensemble.py \
    --logits_ensemble_method avg \
    --model $MODEL \
    --dataset hotpot \
    --reasoning \
    --debug \
    --baseline_file $ENSEMBLE_FILE1 \
    --ensemble_method layer_output_avg \
    --ensemble_layer $LAYER \
    --ensemble_alpha 1 \
    --token_mode last \
    --multilayer

CUDA_VISIBLE_DEVICES=$DEVICE python3 parallel_ensemble.py \
    --logits_ensemble_method avg \
    --model $MODEL \
    --dataset hotpot \
    --reasoning \
    --debug \
    --baseline_file $ENSEMBLE_FILE2 \
    --ensemble_method layer_output_avg \
    --ensemble_layer $LAYER \
    --ensemble_alpha 1 \
    --token_mode last \
    --multilayer

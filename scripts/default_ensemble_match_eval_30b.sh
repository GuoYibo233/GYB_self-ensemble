DEVICE=$1
MODEL=${2:-"gpt_20b"} # qwen3_30b, gpt_20b
NUM_FEWSHOTS=${3:-0} # 0, 5
REVERSED=${4:-false} # false, true
# MODELS="gpt_20b qwen3_30b"

TASKS="myriadlama hotpot commonsense mmlu logiqa"

if [ "$REVERSED" = true ] ; then
        TASKS="logiqa mmlu commonsense hotpot myriadlama"
fi

if [ "$MODEL" == "qwen3_30b" ]; then
        LAYER=36
elif [ "$MODEL" == "gpt_20b" ]; then
        LAYER=18
fi

for DATASET in $TASKS; do
        if [ "$DATASET" == "myriadlama" ]; then
                NUM_SAMPLES=5
        else
                NUM_SAMPLES=1
        fi

        CUDA_VISIBLE_DEVICES=$DEVICE python3 generate_baseline.py \
        --model $MODEL \
        --method per_prompt \
        --dataset $DATASET \
        --num_fewshots $NUM_FEWSHOTS

        CUDA_VISIBLE_DEVICES=$DEVICE python3 parallel_ensemble.py \
        --logits_ensemble_method avg \
        --model $MODEL \
        --dataset $DATASET \
        --num_paraphrases 5 \
        --num_fewshots $NUM_FEWSHOTS \
        --num_samples $NUM_SAMPLES \
        --ensemble_method layer_output_avg \
        --ensemble_layer $LAYER \
        --ensemble_alpha 1 \
        --token_mode last \
        --multilayer    
done

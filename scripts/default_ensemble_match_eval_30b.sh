DEVICE=$1
NUM_FEWSHOTS=${2:-0} # 0, 5

MODEL="qwen3_30b"
NUM_SAMPLES=1


for DATASET in "myriadlama" "hotpot" "commonsense" "mmlu" "logiqa"; do
        if [ "$DATASET" == "myriadlama" ]; then
                NUM_SAMPLES=5
        else
                NUM_SAMPLES=1
        fi

        LAYER=36

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

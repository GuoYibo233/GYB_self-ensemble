DEVICE=$1
NUM_FEWSHOTS=${2:-5}

MODEL="qwen3_30b"

for NUM_FEWSHOTS in 0 5; do
        for DATASET in "commonsense" "mmlu" "logiqa"; do
                LAYER=36

                # Baseline
                CUDA_VISIBLE_DEVICES=$DEVICE python3 parallel_ensemble_perplexity.py \
                        --model $MODEL \
                        --dataset $DATASET \
                        --num_paraphrases 5 \
                        --num_fewshots $NUM_FEWSHOTS \
                        --is_baseline

                # Ensemble
                CUDA_VISIBLE_DEVICES=$DEVICE python3 parallel_ensemble_perplexity.py \
                        --model $MODEL \
                        --dataset $DATASET \
                        --num_paraphrases 5 \
                        --num_fewshots $NUM_FEWSHOTS \
                        --logits_ensemble_method avg \
                        --ensemble_method layer_output_avg \
                        --ensemble_layer $LAYER \
                        --ensemble_alpha 1 \
                        --token_mode last \
                        --multilayer
        done
done
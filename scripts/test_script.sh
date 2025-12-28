DEVICE=$1
ENSEMBLE_METHOD=${2:-"series"}
NUM_FEWSHOTS=${3:-5}
DATASET=${4:-"myriadlama"}

if [[ $ENSEMBLE_METHOD == "base" ]]; then
    # CUDA_VISIBLE_DEVICES=$DEVICE python3 generate_baseline.py \
    #         --debug \
    #         --rewrite \
    #         --model llama3.2_3b \
    #         --method per_prompt \
    #         --dataset $DATASET \
    #         --num_fewshots $NUM_FEWSHOTS
        
    CUDA_VISIBLE_DEVICES=$DEVICE python3 parallel_ensemble_perplexity.py \
            --debug \
            --rewrite \
            --model llama3.2_3b \
            --dataset $DATASET \
            --num_paraphrases 3 \
            --num_fewshots $NUM_FEWSHOTS \
            --is_baseline

elif [[ $ENSEMBLE_METHOD == "series" ]]; then
    CUDA_VISIBLE_DEVICES=$DEVICE python3 series_ensemble.py \
            --debug \
            --model llama3.2_3b \
            --single_para_qapair \
            --num_paraphrases 5 \
            --num_samples 5 \
            --num_fewshots $NUM_FEWSHOTS \
            --modify_attn \
            --modify_rope \
            --rewrite \
            --scale_factor
elif [[ $ENSEMBLE_METHOD == "parallel" ]]; then
    CUDA_VISIBLE_DEVICES=$DEVICE python3 parallel_ensemble_perplexity.py \
            --debug \
            --rewrite \
            --model llama3.2_3b \
            --dataset $DATASET \
            --num_paraphrases 3 \
            --num_fewshots $NUM_FEWSHOTS \
            --logits_ensemble_method avg \
            --ensemble_method layer_output_avg \
            --ensemble_layer 16 \
            --ensemble_alpha 0.2 \
            --token_mode all \
            --multilayer
fi
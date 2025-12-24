DEVICE=$1
ENSEMBLE_METHOD=${2:-"series"}

if [[ $ENSEMBLE_METHOD == "series" ]]; then
    CUDA_VISIBLE_DEVICES=$DEVICE python3 series_ensemble.py \
            --debug \
            --model llama3.2_1b \
            --single_para_qapair \
            --num_paraphrases 5 \
            --num_fewshots 5 \
            --modify_attn \
            --modify_rope \
            --rewrite \
            --scale_factor 0.2
elif [[ $ENSEMBLE_METHOD == "parallel" ]]; then
    CUDA_VISIBLE_DEVICES=$DEVICE python3 parallel_ensemble.py \
            --debug \
            --rewrite \
            --model $MODEL \
            --dataset myriadlama \
            --num_paraphrases 5 \
            --num_fewshots 5 \
            --num_samples 5 \
            --logits_ensemble_method avg \
            --ensemble_method ffn_activation_avg \
            --ensemble_layer 16 \
            --ensemble_alpha 0.2 \
            --token_mode all \
            --multilayer
fi
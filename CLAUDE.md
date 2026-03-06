# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a research codebase for self-ensemble methods applied to LLM question answering, targeting ACL 2026. The core idea: given multiple paraphrases of the same question, ensemble LLM predictions at the logit or hidden-state level to improve accuracy.

## Environment Setup

```bash
# Install PyTorch (nightly for FlexAttention support in series_ensemble.py)
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121

# Install other dependencies
pip install -r requirements.txt

# Download spacy model (required for lemmatization-based evaluation)
python -m spacy download en_core_web_lg
```

HuggingFace cache and model paths are configured in [constants.py](constants.py) based on the `$USER` environment variable. Models are stored on a shared network drive (`/net/tokyo100-10g/data/str01_01/`).

## Running Experiments

All scripts take `DEVICE` (GPU index) as the first argument:

```bash
# Full pipeline: baseline + parallel ensemble + series ensemble
bash scripts/default_ensemble_match_eval.sh <DEVICE> <DATASET> <MODEL_TYPE> <NUM_PARAS> <NUM_FEWSHOTS>
# DATASET options: myriadlama, hotpot, commonsense, mmlu, logiqa
# MODEL_TYPE options: base, it, llama, qwen3, others, or a specific model name

# Run a single script directly
CUDA_VISIBLE_DEVICES=0 python3 generate_baseline.py \
    --model qwen3_8b --dataset myriadlama --method per_prompt \
    --num_fewshots 5 --debug

CUDA_VISIBLE_DEVICES=0 python3 parallel_ensemble.py \
    --model qwen3_8b --dataset myriadlama \
    --logits_ensemble_method avg --num_paraphrases 5 --num_samples 1 \
    --baseline_file <path_to_baseline.feather>

CUDA_VISIBLE_DEVICES=0 python3 series_ensemble.py \
    --model qwen3_8b --dataset myriadlama \
    --num_paraphrases 5 --num_samples 5 \
    --modify_attn --modify_rope --scale_factor \
    --baseline_file <path_to_baseline.feather>
```

## Architecture

### Pipeline Flow

1. **Paraphrase generation** ([paraphrase.py](paraphrase.py)): Generates paraphrases of questions for each dataset item.

2. **Baseline generation** ([generate_baseline.py](generate_baseline.py)): Runs each paraphrase independently through the LLM (`per_prompt` method), producing a `.feather` file with predictions, prompts, and optionally `<think>` reasoning traces.

3. **Ensemble inference** - two approaches:
   - **Parallel ensemble** ([parallel_ensemble.py](parallel_ensemble.py)): Feeds all paraphrase prompts as a batch, then merges logits via `avg`, `max`, `weighted_avg`, or `weighted_max`. Optionally hooks into transformer internals (`layer_output_avg`, `ffn_activation_avg`, `ffn_activation_max`) at a specified layer to average hidden states across the batch before continuing the forward pass.
   - **Series ensemble** ([series_ensemble.py](series_ensemble.py)): Concatenates all paraphrases into a single sequence using FlexAttention with custom block masks so each paraphrase segment attends only causally within itself during encoding, but the generated answer can attend to all segments.

4. **Evaluation**: Results are stored as `.feather` files. Accuracy is computed via lemmatization-based matching (`notebooks/_utils.py` + `utils.py:partial_match_scores`).

### Key Files

- [constants.py](constants.py): Model name -> path mapping (`MODEL_PATHs`), HF cache configuration.
- [dataset.py](dataset.py): Dataset classes for myriadlama, webqa, commonsense, mmlu, logiqa, hotpot. `get_dataset_instance()` is the main entry point. Each dataset manages paraphrase loading, few-shot example construction, and prompt formatting. Results stored under `DATASET_ROOT/{ds_name}/{model_name}/`.
- [utils.py](utils.py): Model loading (`load_model_tokenizer`, `load_tokenizer`), generation helpers (`single_generation`, `thinking_generation`), lemmatization workers (`init_spacy`, `lemmaize_chunk`, `append_lemmas`), accuracy metrics (`partial_match_scores`).
- [parallel_ensemble.py](parallel_ensemble.py): Core logit-level and hidden-state-level parallel ensemble logic. Hook-based intervention via `register_forward_hook` / `register_forward_pre_hook`.
- [series_ensemble.py](series_ensemble.py): FlexAttention-based series ensemble. Requires PyTorch nightly. Modifies attention masks and optionally RoPE embeddings.
- [notebooks/_utils.py](notebooks/_utils.py): Evaluation utilities (`calculate_baseline_accuracy`, `calculate_parallel_ensemble_accuracy`, `calculate_oracle_accuracy`, `get_layers`).

### Output File Naming Convention

Files are saved as `.feather` format. Names encode the configuration:
- Baseline: `{ds_name}paraphrase.perprompt.{n}fshots.{model}.feather`
- Parallel ensemble: `parallel.avg.avglayer.layer{N}.alpha{N}.token-last.multilayer.{S}samples.{P}paras.feather`
- Series ensemble: `modifyattn.modifyrope.scalescore.{S}samples.{P}paras.feather`

### Model Layer Defaults

Each model has a tuned target layer for hidden-state ensemble (from `notebooks/_utils.py:get_layers`):
- llama3.2_1b: 12, llama3.2_3b: 21, llama3.1_8b: 24
- qwen3_1.7b: 21, qwen3_4b/8b: 27, qwen3_14b: 30, qwen3_30b: 36, qwen3_32b: 48

### Thinking Mode

Pass `--thinking` to use chain-of-thought reasoning traces. Baseline generation stores `<think>...</think>` blocks; parallel ensemble then conditions on these traces before the final answer token.

### Paraphrase Sampling

`sample_paraphrases_per_item()` in [parallel_ensemble.py](parallel_ensemble.py) generates all permutations of paraphrase indices and randomly samples `num_samples` combinations using `uuid` as the random seed, ensuring reproducibility. Pass `--num_paraphrases -1` to use all available paraphrases without permutation sampling.

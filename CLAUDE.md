# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research codebase for self-ensemble methods applied to LLM question answering, targeting ACL 2026. The core idea: given multiple paraphrases of the same question, ensemble LLM predictions at the logit or hidden-state level to improve accuracy.

Two ensemble strategies:
- **Parallel ensemble**: batch all paraphrase prompts, merge logits or hidden states before generating the answer
- **Series ensemble**: concatenate all paraphrases into one sequence with FlexAttention custom masks so each paraphrase segment attends only within itself during encoding, but the generated answer attends to all segments

## Environment Setup

```bash
# Install PyTorch nightly (required for FlexAttention in series_ensemble.py)
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121

# Install other dependencies
pip install -r requirements.txt

# Download spacy model (required for lemmatization-based evaluation)
python -m spacy download en_core_web_lg
```

HuggingFace cache and model paths are configured in [constants.py](constants.py) based on `$USER`. Models live on a shared network drive (`/net/tokyo100-10g/data/str01_01/`). Dataset root differs by user: `y-guo` uses `/home/y-guo/self-ensemble/`, `xzhao` uses `/home/xzhao/workspace/GYB_self-ensemble/datasets`.

## Repository Layout

```
GYB_self-ensemble/
├── constants.py             # Model name → path mapping, HF cache config
├── utils.py                 # Model loading, generation helpers, lemmatization, metrics
├── dataset.py               # Dataset classes and get_dataset_instance() factory
├── paraphrase.py            # Few-shot paraphrase generation from a base model
├── self_paraphrase.py       # Self-paraphrase generation using the target model itself
├── generate_baseline.py     # Step 1: per-prompt baseline inference → .feather
├── parallel_ensemble.py     # Step 2a: logit/hidden-state parallel ensemble
├── series_ensemble.py       # Step 2b: FlexAttention series ensemble
├── confidence.py            # Token-level confidence scoring utilities
├── notebooks/
│   ├── _utils.py            # Evaluation helpers (accuracy, layer config, filenames)
│   ├── test_eval.py
│   ├── test_attention_mask.py
│   └── test_report_acc.py
├── scripts/                 # 35 shell scripts for experiments
├── paraphrase_instructions.json
├── paraphrase.ipynb
├── requirements.txt
└── environment.yml
```

## Running Experiments

All scripts take `DEVICE` (GPU index) as the first argument.

### Full pipeline

```bash
bash scripts/default_ensemble_match_eval.sh <DEVICE> <DATASET> <MODEL_TYPE> <NUM_PARAS> <NUM_FEWSHOTS>
# DATASET: myriadlama | hotpot | commonsense | mmlu | logiqa
# MODEL_TYPE: base | it | llama | qwen3 | others | <specific-model-name>
```

### Individual steps

```bash
# Step 1 – Baseline generation
CUDA_VISIBLE_DEVICES=0 python3 generate_baseline.py \
    --model qwen3_8b --dataset myriadlama --method per_prompt \
    --num_fewshots 5 --debug

# Step 2a – Parallel ensemble (logit averaging)
CUDA_VISIBLE_DEVICES=0 python3 parallel_ensemble.py \
    --model qwen3_8b --dataset myriadlama \
    --logits_ensemble_method avg --num_paraphrases 5 --num_samples 1 \
    --baseline_file <path_to_baseline.feather>

# Step 2a – Parallel ensemble (hidden-state averaging at a specific layer)
CUDA_VISIBLE_DEVICES=0 python3 parallel_ensemble.py \
    --model qwen3_8b --dataset myriadlama \
    --ensemble_method layer_output_avg --layer 27 \
    --num_paraphrases 5 --num_samples 1 \
    --baseline_file <path_to_baseline.feather>

# Step 2b – Series ensemble (full modification set)
CUDA_VISIBLE_DEVICES=0 python3 series_ensemble.py \
    --model qwen3_8b --dataset myriadlama \
    --num_paraphrases 5 --num_samples 5 \
    --modify_attn --modify_rope --scale_factor \
    --baseline_file <path_to_baseline.feather>

# Self-paraphrase generation (generates paraphrases using the target model)
bash scripts/paraphrase.sh <DEVICE> <DATASET> <MODEL>
```

### Thinking mode (Qwen3 only)

Pass `--thinking` to any script. Baseline stores `<think>…</think>` blocks (token ID 151668); parallel ensemble conditions on those traces before the final answer token.

```bash
bash scripts/10_think_then_ensemble_4p.sh <DEVICE> <DATASET> <MODEL>
```

## Architecture

### Pipeline Flow

```
Paraphrase generation (paraphrase.py / self_paraphrase.py)
          ↓
Step 1: Baseline generation (generate_baseline.py)
    per_prompt: each paraphrase → independent greedy generation
    output: baseline.feather (uuid, answers, paraphrase, prompt, generation, [thinking])
          ↓
Step 2: Ensemble inference
    ┌─────────────────────────────────┬──────────────────────────────────┐
    │ Parallel (parallel_ensemble.py) │ Series (series_ensemble.py)      │
    │ • avg / max / weighted logits   │ • FlexAttention block masks      │
    │ • layer_output_avg at layer N   │ • RoPE reset per segment         │
    │ • ffn_activation_avg/max        │ • Log-space score scaling        │
    └─────────────────────────────────┴──────────────────────────────────┘
          ↓
Step 3: Evaluation (notebooks/_utils.py, utils.py)
    • Lemma-based partial matching (spaCy en_core_web_lg)
    • Label probability for multi-choice (CommonsenseQA, MMLU, LogiQA)
    • Oracle accuracy: best paraphrase accuracy upper-bound
```

### Key Files

| File | Purpose | Key functions |
|---|---|---|
| `constants.py` | Model name → HF path mapping, cache paths | `MODEL_PATHs` dict |
| `utils.py` | Model loading, generation, lemmatization, metrics | `load_model_tokenizer`, `single_generation`, `thinking_generation`, `partial_match_scores`, `init_spacy`, `append_lemmas` |
| `dataset.py` | Dataset classes, prompt formatting, few-shot | `get_dataset_instance()`, `construct_prompts()`, `construct_prompts_for_thinking()` |
| `paraphrase.py` | Paraphrase generation with few-shot examples | `generate_paraphrases()` |
| `self_paraphrase.py` | Self-paraphrase using the target model | Multi-round generation with temperature sampling |
| `generate_baseline.py` | Per-prompt baseline inference | `generate_baseline_per_prompt()` |
| `parallel_ensemble.py` | Logit/hidden-state parallel ensemble | `ensemble_generation()`, `sample_paraphrases_per_item()`, `craft_prompts_from_baseline_file()` |
| `series_ensemble.py` | FlexAttention series ensemble | `FlexAttentionWrapper`, `create_myriadlama_mask_mod()`, `create_myriadlama_score_mod()` |
| `confidence.py` | Token-level confidence scoring | Confidence utilities |
| `notebooks/_utils.py` | Evaluation, accuracy, filename construction | `get_layers()`, `calculate_baseline_accuracy()`, `calculate_parallel_ensemble_accuracy()`, `calculate_series_ensemble_accuracy()`, `calculate_oracle_accuracy()` |

### Dataset Classes (`dataset.py`)

Abstract base: `ParaPharaseDataset` (note: intentional typo in codebase).

| Class | Dataset | Paraphrases |
|---|---|---|
| `WebQAParaphraseDataset` | stanfordnlp/web_questions | 5 variants per question |
| `MyriadLamaParaphraseDataset` | iszhaoxin/MyriadLAMA | 5 manual + 5 auto per UUID |
| `MyriadLama100ParaphraseDataset` | iszhaoxin/MyriadLAMA | 100 paraphrases |
| `CommonsenseParaphraseDataset` | CommonsenseQA | Multi-choice [A–E] |
| `MMLUParaphraseDataset` | MMLU | Multi-choice [A–D] |
| `LogiQAParaphraseDataset` | LogiQA | Multi-choice [A–D] |
| `HotpotParaphraseDataset` | HotpotQA | 1 manual + 10 auto |

Entry point: `get_dataset_instance(dataset_name, model_name, ...)`.

Results are stored under `DATASET_ROOT/{ds_name}/{model_name}/`.

### Supported Models (`constants.py`)

- **LLaMA**: `llama3.2_1b`, `llama3.2_3b`, `llama3.1_8b`, `llama3.1_70b`
- **Qwen2.5**: `qwen2.5_7b`, `qwen2.5_7b_it`, `qwen2.5_72b`
- **Qwen3**: `qwen3_0.6b`, `qwen3_1.7b`, `qwen3_4b`, `qwen3_8b`, `qwen3_14b`, `qwen3_30b`, `qwen3_32b`, `qwen3_235b`
- **DeepSeek**: `deepseek_r1_1.5b`, `deepseek_r1_7b`, `deepseek_r1_8b`, `deepseek_r1_14b`, `deepseek_r1_32b`
- **Others**: `bloom_1b`, `bloom_3b`, `pythia_1b`, `pythia_2.8b`, `phi3.5_mini`, `gpt-oss-20b`

### Model Layer Defaults for Hidden-State Ensemble

Tuned target layers per model (from `notebooks/_utils.py:get_layers()`):

| Model | Layer |
|---|---|
| llama3.2_1b | 12 |
| llama3.2_3b | 21 |
| llama3.1_8b | 24 |
| qwen3_1.7b | 21 |
| qwen3_4b / qwen3_8b | 27 |
| qwen3_14b | 30 |
| qwen3_30b | 36 |
| qwen3_32b | 48 |

### Parallel Ensemble Methods

Controlled by `--logits_ensemble_method` (for logit-level) or `--ensemble_method` (for hidden-state-level):

| Method | Description |
|---|---|
| `avg` | Average logits across paraphrases |
| `max` | Take element-wise max of logits |
| `weighted_avg` | Softmax-weighted average by confidence |
| `weighted_max` | Confidence-weighted max |
| `layer_output_avg` | Average hidden states at layer N via forward hook |
| `ffn_activation_avg` | Average FFN activations at layer N |
| `ffn_activation_max` | Max-pool FFN activations at layer N |

Hook-based intervention uses `register_forward_hook` / `register_forward_pre_hook` on the target layer.

### Series Ensemble (FlexAttention)

Requires PyTorch nightly ≥ 2.5 with CUDA 12.1.

- `FlexAttentionWrapper`: patches model attention layers with custom score functions
- `create_myriadlama_mask_mod()`: block mask isolating paraphrase segments during encoding; generation attends to all segments
- `create_myriadlama_score_mod()`: applies log-space scaling to paraphrase attention scores
- `tokenize_with_segment()`: tracks per-token segment IDs for block mask construction
- `--modify_attn`: enables custom FlexAttention mask
- `--modify_rope`: resets RoPE position embeddings to 0 at the start of each paraphrase segment
- `--scale_factor`: enables attention score modification

### Paraphrase Sampling

`sample_paraphrases_per_item()` in `parallel_ensemble.py`:
- Generates all permutations of paraphrase indices
- Randomly samples `num_samples` combinations using `uuid` as the random seed (reproducible)
- Pass `--num_paraphrases -1` to use all available paraphrases without permutation sampling

### Output File Naming Convention

All outputs are `.feather` format (fast columnar I/O via PyArrow).

| Stage | Filename pattern |
|---|---|
| Baseline | `{ds_name}paraphrase.perprompt.{N}fshots.{model}.feather` |
| Parallel (logits) | `parallel.{method}.{S}samples.{P}paras.feather` |
| Parallel (hidden-state) | `parallel.avg.avglayer.layer{N}.alpha{N}.token-last.multilayer.{S}samples.{P}paras.feather` |
| Series | `modifyattn.modifyrope.scalescore.{S}samples.{P}paras.feather` |

#### DataFrame columns

**Baseline feather**: `uuid`, `answers`, `paraphrase`, `prompt`, `generation`, `thinking` (optional), `is_orig`, `instruction`, `few_shot_context`, `generation_lemmas`, `answer_lemmas`

**Ensemble feather**: `uuid`, `prediction`, `generation`, `paraphrases`, `is_orig`, `answers`, `labels`, `label_probs`, `generation_lemmas`, `answer_lemmas`

### Evaluation

- **Open-ended QA**: lemma-based partial matching via spaCy `en_core_web_lg` (`partial_match_scores` in `utils.py`)
- **Multi-choice**: label probability extraction (`get_label_prob()`) then argmax; also supports direct token matching
- **Oracle accuracy**: best-of-N paraphrase accuracy upper-bound (`calculate_oracle_accuracy()`)
- Lemmatization is parallelized with 8 multiprocessing workers (`append_lemmas()`)

## Scripts Reference

| Script | Purpose |
|---|---|
| `default_ensemble_match_eval.sh` | Full pipeline: baseline + parallel + series |
| `paraphrase.sh` | Self-paraphrase generation via `self_paraphrase.py` |
| `parallel_myriadlama_default.sh` | Parallel ensemble experiments on MyriadLAMA |
| `series_myriadlama_default.sh` | Series ensemble experiments on MyriadLAMA |
| `00_nlp2026_ensemble_match_eval.sh` | ACL 2026 submission pipeline |
| `10_think_then_ensemble_4p.sh` | Thinking-based ensemble (4 paraphrases) |
| `10_think_then_ensemble_8p.sh` | Thinking-based ensemble (8 paraphrases) |
| `parallel_myriadlama_diff_layers.sh` | Layer sensitivity experiments |
| `parallel_myriadlama_hype_search.sh` | Hyperparameter search |
| `series_myriadlama_0shot.sh` | Zero-shot series ensemble |
| `series_multichoice_default.sh` | Multi-choice series ensemble |
| `consistency.sh` | Consistency evaluation across paraphrases |
| `lemmaize.sh` | Standalone lemmatization for existing feathers |
| `default_ensemble_match_eval_30b.sh` | 30B model variant of full pipeline |

## Development Conventions

### Code style
- Python files use standard library + HuggingFace Transformers + PyTorch patterns
- `argparse` for CLI arguments in all main scripts
- Batch size, debug mode, and num_workers are runtime arguments, not hardcoded
- `--debug` flag limits dataset to 200 samples for fast iteration

### Path configuration
- All paths derived from `$USER` in `constants.py` and `utils.py`; do not hardcode paths
- Adding a new user: update the `if os.environ.get("USER") == ...` blocks in both files

### Adding a new model
1. Add entry to `MODEL_PATHs` dict in `constants.py`
2. Add default target layer to `get_layers()` in `notebooks/_utils.py`
3. If thinking mode needed, verify token ID (Qwen3 uses 151668 for `<think>`)

### Adding a new dataset
1. Subclass `ParaPharaseDataset` (or `MultiChoiceParaphraseDataset`) in `dataset.py`
2. Implement: `load_dataset()`, `get_dataloader()`, `collate_fn()`, `get_few_shot_examples()`
3. Register in `get_dataset_instance()` factory
4. Add paraphrase loading logic and prompt templates

### Feather file conventions
- Always write with `df.to_feather(path)` and read with `pd.read_feather(path)`
- Lemma columns (`generation_lemmas`, `answer_lemmas`) added as post-processing step via `append_lemmas()`; re-run `lemmaize.sh` if missing

### Reproducibility
- Paraphrase sampling uses `uuid` string as random seed → deterministic across runs
- `--num_samples` controls how many permutation samples to draw per dataset item
- Series ensemble generation uses greedy decoding (`do_sample=False`)

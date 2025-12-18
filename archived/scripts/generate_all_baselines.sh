#!/bin/bash
# 
# Generate baselines for all existing model directories
# 
# This script automatically detects which models have been run
# and generates baseline results for each of them.
#
# Usage:
#   bash scripts/generate_all_baselines.sh [--rewrite] [--dry-run]
#
# Options:
#   --rewrite   Regenerate baselines even if they already exist
#   --dry-run   Show what would be done without actually running
#

set -e  # Exit on error

# Parse arguments
REWRITE_FLAG=""
DRY_RUN=false

for arg in "$@"; do
    case $arg in
        --rewrite)
            REWRITE_FLAG="--rewrite"
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        *)
            # Unknown option
            ;;
    esac
done

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Base directory for datasets
DATASET_ROOT="/net/tokyo100-10g/data/str01_01/xzhao/datasets/self-ensemble"

# Script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "========================================================================"
echo "Generate Baselines for All Existing Models"
echo "========================================================================"
echo "Dataset root: $DATASET_ROOT"
echo "Project root: $PROJECT_ROOT"
echo "Rewrite: ${REWRITE_FLAG:-false}"
echo "Dry run: $DRY_RUN"
echo ""

# Track statistics
TOTAL_MODELS=0
SKIPPED_MODELS=0
GENERATED_MODELS=0
FAILED_MODELS=0

# Function to generate baseline for a dataset/model pair
generate_baseline() {
    local dataset=$1
    local model=$2
    local model_dir=$3
    
    echo ""
    echo "--------------------------------------------------------------------"
    echo -e "${BLUE}Dataset: $dataset | Model: $model${NC}"
    echo "--------------------------------------------------------------------"
    
    # Check if paraphrases_dataset exists
    if [ "$dataset" = "webqa" ]; then
        if [ ! -d "$model_dir/paraphrases_dataset" ]; then
            echo -e "${YELLOW}⚠ No paraphrases_dataset found, skipping...${NC}"
            ((SKIPPED_MODELS++))
            return
        fi
    fi
    
    # Check if baselines already exist
    BASELINE_ORIGIN="$model_dir/baseline_origin.feather"
    BASELINE_PER_PROMPT="$model_dir/baseline_per_prompt.feather"
    
    if [ -f "$BASELINE_ORIGIN" ] && [ -f "$BASELINE_PER_PROMPT" ] && [ -z "$REWRITE_FLAG" ]; then
        echo -e "${GREEN}✓ Baselines already exist${NC}"
        echo "  - $BASELINE_ORIGIN"
        echo "  - $BASELINE_PER_PROMPT"
        echo "  Use --rewrite to regenerate"
        ((SKIPPED_MODELS++))
        return
    fi
    
    # Build command
    CMD="python3 $PROJECT_ROOT/src/generate_baseline.py --method all --dataset $dataset --model $model $REWRITE_FLAG"
    
    if [ "$DRY_RUN" = true ]; then
        echo -e "${YELLOW}[DRY RUN] Would execute:${NC}"
        echo "  $CMD"
        ((GENERATED_MODELS++))
    else
        echo "Executing: $CMD"
        echo ""
        
        # Run the command
        if $CMD; then
            echo ""
            echo -e "${GREEN}✅ Successfully generated baselines for $dataset/$model${NC}"
            ((GENERATED_MODELS++))
        else
            echo ""
            echo -e "${RED}❌ Failed to generate baselines for $dataset/$model${NC}"
            ((FAILED_MODELS++))
        fi
    fi
}

# Process all models
echo ""
echo "========================================================================"
echo "Processing All Models"
echo "========================================================================"

# ============================================================================
# Model List Configuration
# ============================================================================
# 在这里配置要运行 baseline 的模型列表
# 格式: "dataset:model"
# 
# 说明:
# - dataset 可以是 webqa 或 myriadlama
# - 同一个模型可以在不同的 dataset 上运行
# - 可以在运行实验之前就生成 baseline
# - 只需要确保模型目录存在且有 paraphrases_dataset (对于 webqa)
#
# 示例:
#   "webqa:llama3.2_1b"           # 在 WebQA 数据集上运行 llama3.2_1b
#   "myriadlama:llama3.2_1b"      # 在 MyriadLAMA 数据集上运行同一个模型
# ============================================================================

MODELS=(
    # WebQA models
    "webqa:llama3.2_1b"
    "webqa:llama3.2_1b_it"
    "webqa:llama3.2_3b"
    "webqa:llama3.2_3b_it"
    "webqa:qwen2.5_3b_it"
    "webqa:qwen2.5_7b_it"
    "webqa:qwen3_1.7b"
    "webqa:qwen3_4b"
    "webqa:qwen3_8b"
    
    # MyriadLAMA models
    "myriadlama:qwen2.5_7b_it"
    
    # 可以在这里添加更多模型，例如:
    # "myriadlama:llama3.2_3b_it"
    # "webqa:qwen3_14b"
)

# ============================================================================

for entry in "${MODELS[@]}"; do
    # Skip empty lines and comments
    if [[ -z "$entry" || "$entry" =~ ^[[:space:]]*# ]]; then
        continue
    fi
    
    # Split dataset:model
    dataset="${entry%%:*}"
    model="${entry##*:}"
    model_dir="$DATASET_ROOT/$dataset/$model"
    
    if [ -d "$model_dir" ]; then
        ((TOTAL_MODELS++))
        generate_baseline "$dataset" "$model" "$model_dir"
    else
        echo -e "${YELLOW}⚠ Model directory not found: $model_dir${NC}"
        echo "   You may need to create it first or check the path"
    fi
done

# Print summary
echo ""
echo "========================================================================"
echo "Summary"
echo "========================================================================"
echo "Total models found: $TOTAL_MODELS"
echo -e "${GREEN}Generated: $GENERATED_MODELS${NC}"
echo -e "${YELLOW}Skipped: $SKIPPED_MODELS${NC}"
if [ $FAILED_MODELS -gt 0 ]; then
    echo -e "${RED}Failed: $FAILED_MODELS${NC}"
fi
echo ""

if [ "$DRY_RUN" = true ]; then
    echo -e "${YELLOW}This was a dry run. Run without --dry-run to actually generate baselines.${NC}"
fi

if [ $FAILED_MODELS -gt 0 ]; then
    echo -e "${RED}Some models failed. Check the logs above for details.${NC}"
    exit 1
fi

echo "Done!"
echo "========================================================================"

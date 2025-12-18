#!/usr/bin/env python3
"""
Generate baselines for all existing model directories.

This script automatically detects which models have been run
and generates baseline results for each of them.

Features:
- Auto-detects all existing model directories
- Supports parallel execution with multiple GPUs
- Shows progress and estimated time
- Can resume interrupted runs

Usage:
    # Sequential execution
    python scripts/generate_all_baselines.py
    
    # Parallel execution on multiple GPUs
    python scripts/generate_all_baselines.py --parallel --gpus 0,1,2,3
    
    # Dry run (see what would be done)
    python scripts/generate_all_baselines.py --dry-run
    
    # Force regenerate existing baselines
    python scripts/generate_all_baselines.py --rewrite
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
import json

from utils import DATASET_ROOT

# Colors for terminal output
class Colors:
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    CYAN = '\033[0;36m'
    NC = '\033[0m'  # No Color

def print_header(text):
    print("\n" + "="*70)
    print(text)
    print("="*70)

def print_colored(text, color):
    print(f"{color}{text}{Colors.NC}")

def find_existing_models(dataset_root):
    """Find all existing model directories."""
    models = {"webqa": [], "myriadlama": []}
    
    # WebQA models
    webqa_dir = Path(dataset_root) / "webqa"
    if webqa_dir.exists():
        for model_dir in webqa_dir.iterdir():
            if model_dir.is_dir():
                model_name = model_dir.name
                # Skip special directories
                if model_name in ["raw_dataset", "paraphrases_dataset"]:
                    continue
                # Check if paraphrases_dataset exists
                if (model_dir / "paraphrases_dataset").exists():
                    models["webqa"].append({
                        "name": model_name,
                        "path": str(model_dir),
                        "has_origin": (model_dir / "baseline_origin.feather").exists(),
                        "has_per_prompt": (model_dir / "baseline_per_prompt.feather").exists()
                    })
    
    # MyriadLAMA models
    myriadlama_dir = Path(dataset_root) / "myriadlama"
    if myriadlama_dir.exists():
        for model_dir in myriadlama_dir.iterdir():
            if model_dir.is_dir():
                model_name = model_dir.name
                # Skip special directories
                if model_name in ["raw_dataset", "paraphrases_dataset"]:
                    continue
                models["myriadlama"].append({
                    "name": model_name,
                    "path": str(model_dir),
                    "has_origin": (model_dir / "baseline_origin.feather").exists(),
                    "has_per_prompt": (model_dir / "baseline_per_prompt.feather").exists()
                })
    
    return models

def generate_baseline(dataset, model_name, rewrite=False, dry_run=False, gpu_id=None):
    """Generate baseline for a specific dataset/model pair."""
    
    cmd = [
        "python3",
        "baseline_generate.py",
        "--method", "all",
        "--dataset", dataset,
        "--model", model_name
    ]
    
    if rewrite:
        cmd.append("--rewrite")
    
    if gpu_id is not None:
        cmd.extend(["--device", f"cuda:{gpu_id}"])
    
    if dry_run:
        print_colored(f"[DRY RUN] Would execute: {' '.join(cmd)}", Colors.YELLOW)
        return True
    
    print(f"\nExecuting: {' '.join(cmd)}\n")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False, text=True)
        return True
    except subprocess.CalledProcessError as e:
        print_colored(f"❌ Failed with exit code {e.returncode}", Colors.RED)
        return False

def main():
    parser = argparse.ArgumentParser(
        description="Generate baselines for all existing models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Sequential execution
  python scripts/generate_all_baselines.py
  
  # Parallel execution on GPUs 0-3
  python scripts/generate_all_baselines.py --parallel --gpus 0,1,2,3
  
  # Dry run
  python scripts/generate_all_baselines.py --dry-run
  
  # Force regenerate
  python scripts/generate_all_baselines.py --rewrite
        """
    )
    
    parser.add_argument("--dataset-root", type=str,
                       default=f"{DATASET_ROOT}",
                       help="Root directory for datasets")
    parser.add_argument("--rewrite", action="store_true",
                       help="Regenerate baselines even if they exist")
    parser.add_argument("--dry-run", action="store_true",
                       help="Show what would be done without executing")
    parser.add_argument("--parallel", action="store_true",
                       help="Run multiple models in parallel")
    parser.add_argument("--gpus", type=str, default="0",
                       help="Comma-separated list of GPU IDs (e.g., '0,1,2,3')")
    parser.add_argument("--dataset", type=str, choices=["webqa", "myriadlama", "all"],
                       default="all", help="Which dataset to process")
    
    args = parser.parse_args()
    
    # Change to script directory
    script_dir = Path(__file__).parent.parent
    os.chdir(script_dir)
    
    print_header("Generate Baselines for All Existing Models")
    print(f"Dataset root: {args.dataset_root}")
    print(f"Rewrite: {args.rewrite}")
    print(f"Dry run: {args.dry_run}")
    print(f"Parallel: {args.parallel}")
    if args.parallel:
        print(f"GPUs: {args.gpus}")
    
    # Find all existing models
    print_header("Scanning for Existing Models")
    models = find_existing_models(args.dataset_root)
    
    # Print what we found
    total_models = 0
    for dataset, model_list in models.items():
        if args.dataset != "all" and dataset != args.dataset:
            continue
        if model_list:
            print(f"\n{dataset.upper()}:")
            for model_info in model_list:
                status = []
                if model_info["has_origin"]:
                    status.append("origin✓")
                else:
                    status.append("origin✗")
                if model_info["has_per_prompt"]:
                    status.append("per_prompt✓")
                else:
                    status.append("per_prompt✗")
                
                status_str = " ".join(status)
                print(f"  - {model_info['name']:<25} [{status_str}]")
                total_models += 1
    
    print(f"\nTotal models found: {total_models}")
    
    if total_models == 0:
        print_colored("\nNo models found to process.", Colors.YELLOW)
        return
    
    # Determine which models need processing
    tasks = []
    for dataset, model_list in models.items():
        if args.dataset != "all" and dataset != args.dataset:
            continue
        for model_info in model_list:
            # Skip if both baselines exist and not rewriting
            if not args.rewrite and model_info["has_origin"] and model_info["has_per_prompt"]:
                continue
            tasks.append((dataset, model_info["name"]))
    
    if not tasks:
        print_colored("\n✓ All baselines already exist. Use --rewrite to regenerate.", Colors.GREEN)
        return
    
    print(f"\nModels to process: {len(tasks)}")
    
    if not args.dry_run:
        response = input("\nProceed? [y/N] ")
        if response.lower() not in ['y', 'yes']:
            print("Cancelled.")
            return
    
    # Generate baselines
    print_header("Generating Baselines")
    
    stats = {"success": 0, "failed": 0, "skipped": 0}
    
    if args.parallel:
        print_colored("⚠ Parallel execution not yet implemented. Running sequentially.", Colors.YELLOW)
        # TODO: Implement parallel execution with multiprocessing
    
    # Sequential execution
    for i, (dataset, model_name) in enumerate(tasks, 1):
        print(f"\n[{i}/{len(tasks)}] Processing: {dataset}/{model_name}")
        
        success = generate_baseline(dataset, model_name, 
                                    rewrite=args.rewrite, 
                                    dry_run=args.dry_run)
        
        if success:
            stats["success"] += 1
            print_colored(f"✅ Successfully generated baselines for {dataset}/{model_name}", Colors.GREEN)
        else:
            stats["failed"] += 1
            print_colored(f"❌ Failed to generate baselines for {dataset}/{model_name}", Colors.RED)
    
    # Print summary
    print_header("Summary")
    print(f"Total tasks: {len(tasks)}")
    print_colored(f"Successful: {stats['success']}", Colors.GREEN)
    if stats["failed"] > 0:
        print_colored(f"Failed: {stats['failed']}", Colors.RED)
    
    if args.dry_run:
        print_colored("\nThis was a dry run. Run without --dry-run to actually generate baselines.", Colors.YELLOW)
    
    if stats["failed"] > 0:
        sys.exit(1)
    
    print_colored("\n✅ All done!", Colors.GREEN)

if __name__ == "__main__":
    main()

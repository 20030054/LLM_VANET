#!/usr/bin/env bash
# Build the training set from the data-generation runs, fine-tune the LLM, train the Random Forest.
#   sbatch --gres=gpu:1 hpc/train.sbatch
#SBATCH --job-name=kvrsu-train
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
source hpc/env.sh
mkdir -p data models
"$PYTHON" engine/scripts/build_dataset.py --logs "$RESULTS"/greedy-datagen_*/windows.jsonl --out data/train.jsonl
"$PYTHON" engine/scripts/train_rf.py --data data/train.jsonl --out "$RF_MODEL"
"$PYTHON" engine/scripts/finetune_lora.py --data data/train.jsonl --model "$BASE_MODEL" --out "$MODEL"
"$PYTHON" engine/scripts/check_equivalence.py --model "$MODEL"

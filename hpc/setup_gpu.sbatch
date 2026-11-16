#!/usr/bin/env bash
# Step E0 + E1 on one GPU: verify the engine on the real model, calibrate the cost model, run the
# engine micro-benchmark.   sbatch --gres=gpu:1 hpc/setup_gpu.sbatch [MODEL_DIR]
#SBATCH --job-name=kvrsu-setup
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
source hpc/env.sh
M="${1:-$BASE_MODEL}"
mkdir -p "$RESULTS"
nvidia-smi
"$PYTHON" engine/scripts/check_equivalence.py --model "$M"
"$PYTHON" engine/scripts/profile_engine.py --model "$M" --profile-out "$PROFILE" --bench-out "$RESULTS/e1_engine.csv"
"$PYTHON" engine/scripts/profile_engine.py --model "$M" --profile-in "$PROFILE" --gpu-kv-budget-mb 32 \
    --bench-out "$RESULTS/e1_engine_b32.csv" --requests 4 --churn 0.1

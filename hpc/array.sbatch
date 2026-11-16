#!/usr/bin/env bash
# One simulation per array task.  Usage (from the repository root):
#
#   GPU list:  sbatch --array=1-$(wc -l < hpc/runs/e2e3_main_gpu.txt)%8 --gres=gpu:1 hpc/array.sbatch hpc/runs/e2e3_main_gpu.txt
#   CPU list:  DEVICE=cpu sbatch --array=1-$(wc -l < hpc/runs/e2e3_main_cpu.txt)%32 hpc/array.sbatch hpc/runs/e2e3_main_cpu.txt
#
# Use ONE GPU PER TASK and do not share it: inference time is measured and enters the results.
# Add your site's --partition / --account on the sbatch command line or below.
#SBATCH --job-name=kvrsu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%A_%a.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
source hpc/env.sh
LIST=${1:?run list}
IFS='|' read -r POLICY N SPEED SEED TAG SARGS SIMARGS < <(sed -n "${SLURM_ARRAY_TASK_ID}p" "$LIST")
NAME="${POLICY}${TAG:+-$TAG}_n${N}_v${SPEED}_s${SEED}"
if [ -f "$RESULTS/$NAME/meta.json" ]; then echo "already done: $NAME"; exit 0; fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
SERVER_ARGS="$SARGS" SIM_ARGS="$SIMARGS" hpc/run_one.sh "$POLICY" "$N" "$SPEED" "$SEED" "$TAG"

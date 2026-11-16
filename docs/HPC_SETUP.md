# HPC setup

Assumptions: Linux x86-64 cluster with SLURM, NVIDIA GPUs, internet on the login node, no root.
For PBS/LSF only the three `hpc/*.sbatch` headers and the array variable need changing.

## Requirements

| Item | Version | Note |
|---|---|---|
| C++ compiler | GCC ≥ 9 or Clang, C++17 | to build OMNeT++, Veins and the model |
| bison, flex, make, unzip, curl | any recent | OMNeT++ build |
| Python | 3.10 – 3.12 | 3.13+ works only if PyTorch has wheels for it |
| CUDA GPU | ≥ 16 GB memory | one per simulation; 24 GB+ for fine-tuning |
| OMNeT++ | 6.1 | built without Qtenv/OSG |
| Veins | 5.3.1 | |
| SUMO | 1.25.0 | installed by pip (`eclipse-sumo`), no root |
| INET | — | **not needed**; Veins' own 802.11p stack is used |
| Disk | ~15 GB | tools 4 GB, model 3 GB, results 5–8 GB |

## Steps

1. **Copy the repository** to the cluster (without `.venv`, `sim/src/out`, `results`).
2. **Edit `hpc/env.sh`**: uncomment and adapt the `module load` line. Check with `gcc --version`,
   `python3 --version`, `nvidia-smi` (on a GPU node).
3. **Install the toolchain** on the login node:
   ```bash
   TORCH_CUDA=cu121 bash hpc/install_toolchain.sh      # cu121 / cu124 ...: match the cluster's CUDA
   ```
   It builds everything into `tools/` and ends by running the unit tests (expect `23 passed`).
4. **Check the simulator** without GPU:
   ```bash
   source hpc/env.sh
   MODEL=tiny-random DEVICE=cpu SIM_TIME=60 hpc/run_one.sh kvrsu 40 15 1
   ```
   Expect `done: .../kvrsu_n40_v15_s1` and a non-empty `run-tasks.csv` there.
5. **Download the base model** on the login node. Llama 3.2 is gated: accept the licence on its
   Hugging Face page first, then
   ```bash
   source hpc/env.sh
   HF_HUB_OFFLINE=0 tools/venv/bin/huggingface-cli login
   HF_HUB_OFFLINE=0 tools/venv/bin/huggingface-cli download meta-llama/Llama-3.2-1B --local-dir models/llama32-1b
   ```
6. **Verify and calibrate on a GPU node**:
   ```bash
   mkdir -p logs && sbatch --gres=gpu:1 hpc/setup_gpu.sbatch
   ```
   In `logs/kvrsu-setup-*.out` check: both equivalence checks print `PASS`; the calibration line
   reports a mean relative error (below about 20 % is fine; the engine corrects the rest online).
   Outputs: `results/profile.json`, `results/e1_engine.csv`.

Then follow `docs/EXPERIMENTS.md`.

## Rules for valid timing

- One GPU per simulation, not shared with other jobs (`--gres=gpu:1`, exclusive process mode if offered).
- Use the same GPU type for all runs that are compared; record it (it is printed in each job log).
- Recalibrate (`setup_gpu.sbatch`) when the GPU type or the model changes.

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `'fstream' file not found` when building | Clang does not find libstdc++: `KVRSU_CXX="clang++ --gcc-install-dir=<gcc dir>" sim/build.sh`, or configure OMNeT++ with `PREFER_CLANG=no` |
| `decision server died` | read `results/<run>/server.log` (wrong model path, no GPU, out of memory) |
| `simulation failed` | read `results/<run>/sim.log`; `sumo: command not found` means `SUMO_HOME`/`PATH` not set — `source hpc/env.sh` |
| label is not a single token | the tokenizer splits an answer letter; change `CAND_LABELS` in `engine/kvrsu/prompt.py` |
| equivalence check `CHECK` in float16 | near-ties from rounding; rerun with `--dtype float32` to confirm |
| CUDA errors mentioning streams | add `--no-streams` to `SERVER_ARGS` (copies become synchronous) and report it |

## What has not been tested

Everything in this repository was tested on a workstation without GPU. Not yet exercised:
- `hpc/install_toolchain.sh` (written from the standard OMNeT++/Veins build procedure; the configure
  options may need adjusting to the cluster),
- GPU code paths: half precision, pinned-memory transfers, CUDA streams,
- a real Llama checkpoint (the forward pass was verified against Hugging Face on a small random
  Llama; `check_equivalence.py` repeats the check on the real one),
- SLURM submission, and `eclipse-sumo` from pip (locally SUMO 1.25.0 came from the system package).
Step 6 is designed to surface problems in all GPU items before any experiment runs.

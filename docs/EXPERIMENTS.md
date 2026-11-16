# Experiment protocol

All commands are run from the repository root after `docs/HPC_SETUP.md` is complete.
`python hpc/make_runlist.py` writes the run lists to `hpc/runs/`.

## Fixed setting

| Item | Value |
|---|---|
| Area / roads | 1 km², 5×5 grid, 250 m blocks, 2 lanes, traffic lights |
| Vehicles | constant number (rerouters), so vehicles = vehicles/km² |
| Radio | IEEE 802.11p, 6 Mbps, 20 dBm, path-loss exponent 2.2, Nakagami m = 1.5, unicast with ACK |
| RSU | centre of the area, edge CPU 2·10¹⁰ cycles/s, decision window 100 ms |
| Tasks | Poisson 0.2 /s per vehicle; 1–8·10⁸ cycles; 2–8 kB input; deadline 0.5–1.5 s |
| Vehicle CPU | 0.5–2·10⁹ cycles/s; state report every 0.5 s |
| Run | 150 s simulated, first 30 s warm-up, 5 seeds |
| Model | Llama-3.2-1B, fine-tuned (step 2), half precision |

Same seed ⇒ same vehicles and same tasks for every policy (separate random stream for the workload).
Change workload values only in `sim/simulations/omnetpp.ini` and report them.

## Methods compared

Published work (three journal papers) plus one established technique:

| Label in figures | Policy | Source | Fidelity |
|---|---|---|---|
| Per-request LLM | `sequential` | Trabelsi et al., *Sci. Reports* 2026 | same decision model, served as described there |
| TightLLM-style batching | `batched` | Hu et al., *IEEE Trans. Computers* 2025 | **re-implementation of the policy, not their code**; with single-token decisions its adaptivity reduces to memory-bounded batching |
| Random Forest | `rf` | Ali et al., *IEEE OJ-COMS* 2026 | forest size as published; credibility module not reproduced |
| Prefix cache | `prefix_cache` | prompt caching as in SGLang / vLLM (conference systems) | technique re-implemented in this engine |

`weighted` (Eq. 21 of the Sci. Reports paper) and `greedy` are references, not counted as related work.
State these fidelity notes in the paper.

## Steps

**1. Engine benchmark (E1)** — produced by `hpc/setup_gpu.sbatch` in the setup: service time of each
policy vs table size, requests per window and churn, without the network.

**2. Training data and models**
```bash
DEVICE=cpu sbatch --array=1-$(wc -l < hpc/runs/d_datagen_cpu.txt)%32 hpc/array.sbatch hpc/runs/d_datagen_cpu.txt
sbatch --gres=gpu:1 hpc/train.sbatch          # after the array has finished
```
Check in `logs/kvrsu-train-*.out`: sample count and label mix, RF held-out accuracy, LLM held-out
accuracy, equivalence `PASS`. **Gate:** if the LLM accuracy is not clearly above the RF's, stop and
report. Data-generation seeds (101, 102) are never used for evaluation.

**3. Main sweeps (E2 density, E3 speed)**
```bash
sbatch --array=1-$(wc -l < hpc/runs/e2e3_main_gpu.txt)%8 --gres=gpu:1 hpc/array.sbatch hpc/runs/e2e3_main_gpu.txt
DEVICE=cpu sbatch --array=1-$(wc -l < hpc/runs/e2e3_main_cpu.txt)%32 hpc/array.sbatch hpc/runs/e2e3_main_cpu.txt
```
E2: 50–300 vehicles/km² at 15 m/s. E3: 5–30 m/s at 150 vehicles/km². 7 methods × 5 seeds.

**4. Ablation (E4), memory budget (E5), slower hardware (E6)**
```bash
for L in e4_ablation_gpu e5_budget_gpu e6_scale_gpu; do
  sbatch --array=1-$(wc -l < hpc/runs/$L.txt)%8 --gres=gpu:1 hpc/array.sbatch hpc/runs/$L.txt; done
```
E4 removes one component at a time (ordering, admission, adaptive spill) under a 32 MB KV budget.
E5 varies that budget (8–128 MB) at 300 vehicles. E6 multiplies measured inference time by 2, 4, 8
to emulate edge-class GPUs. E5 and E6 are **emulations** on the HPC GPU and must be described as such.

**5. Collect and plot**
```bash
source hpc/env.sh
$PYTHON analysis/collect.py --results results --out results/summary.csv
$PYTHON analysis/plots.py --summary results/summary.csv --e1 results/e1_engine.csv --out figures
```
`collect.py` prints a table of finished runs per variant and lists skipped (incomplete) runs.
Re-submitting an array skips runs that already have `meta.json`.

## Metrics

Defined in the header of `analysis/collect.py`. Main ones: task completion ratio, decision response
time (mean, 95th percentile), task latency, throughput (decisions/s, completed tasks/s), engine
service time, reused share of the table cache, share of requests decided by the policy.
Figures show the mean over seeds with a 95 % confidence band; every figure has a `.csv` with the numbers.

## Sanity checks before trusting results

- `tasks` per (vehicles, speed, seed) is identical across policies in `summary.csv`.
- `share_nocoverage` is similar across policies (it depends on the radio, not on the policy).
- `sequential`, `batched`, `prefix_cache`, `kvrsu` use the same model, so differences in completion
  ratio come from latency and admission only.
- The measured mean speed is far below the speed limit at high density (traffic lights, queues):
  report the limit as the control variable and say so.

## What to send back for the paper

`results/summary.csv`, `results/e1_engine*.csv`, `results/profile.json`, `figures/`, the three
`logs/kvrsu-setup-*`, `logs/kvrsu-train-*` logs, and the GPU model. Raw run directories are not needed.

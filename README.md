# kvrsu — mobility-aware KV-cache serving of LLM offloading decisions at an RSU

Code for the experiments of the paper in preparation. An RSU runs a fine-tuned LLM that decides, for
every task request of a vehicle, whether to run the task locally, on a nearby vehicle (V2V) or on the
RSU edge server. All requests of a decision window share the RSU's resource table as prompt prefix.
The engine computes that prefix once per window, reuses its key/value (KV) cache across windows up to
the first changed table slot, orders slots so that stable vehicles come first, and — when the cache
does not fit the GPU — chooses per window how much to load from host memory and how much to recompute.
The decisions are identical to per-request serving; only the work differs.

## Layout

| Path | Content |
|---|---|
| `engine/kvrsu/` | Python package: layer-wise Llama forward pass, serving policies, cost model, prompt/table manager, decision rules, TCP decision server |
| `engine/scripts/` | equivalence check, calibration + engine benchmark, dataset builder, LoRA fine-tuning, Random-Forest training |
| `engine/tests/` | unit tests (CPU, tiny random model, no download) |
| `sim/src/kvrsu/` | OMNeT++/Veins modules: vehicle app, RSU app, messages, network |
| `sim/simulations/` | `omnetpp.ini`, radio configuration |
| `scenario/` | SUMO scenario generator (grid, constant density) |
| `hpc/` | installer, site configuration, SLURM job files, run lists |
| `analysis/` | result collection and figures |
| `docs/` | `HPC_SETUP.md`, `EXPERIMENTS.md`, `DESIGN.md` |

## How the pieces talk

```
SUMO  <--TraCI-->  OMNeT++ / Veins  (vehicles, 802.11p, RSU app)
                         |  one text message per decision window (TCP, localhost)
                         v
                 decision server (Python, GPU)  ->  decisions + measured inference time
```

The RSU app delivers each decision after the inference time the server measured, so GPU latency
enters the network simulation. One simulation = one server process = one GPU.

## Quick start (laptop, no GPU, no model download)

```bash
python3 -m venv .venv && .venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest engine/tests -q                       # 23 tests

source <omnetpp>/setenv
VEINS_ROOT=<veins-5.3.1> ./sim/build.sh
VEINS_ROOT=<veins-5.3.1> PYTHON=$PWD/.venv/bin/python MODEL=tiny-random DEVICE=cpu SIM_TIME=60 \
    ./hpc/run_one.sh kvrsu 40 15 1                               # policy, vehicles, speed, seed
.venv/bin/python analysis/collect.py --results results --out results/summary.csv
```

`tiny-random` is a small random model for testing the pipeline. Its decisions are meaningless and
its timings say nothing about a real model.

## Policies (`--policy`)

| Name | What it is | Stands for |
|---|---|---|
| `kvrsu` | shared prefix + cross-window reuse + stability ordering + adaptive host spill + deadline-aware admission | proposed |
| `kvrsu_static` | same, but always loads the whole valid cache | ablation |
| `prefix_cache` | shared prefix + reuse, GPU only, arrival order | prompt caching |
| `batched` | memory-bounded batching, nothing shared | TightLLM-style serving (IEEE TC 2025) |
| `sequential` | one forward pass per request | per-request LLM (Sci. Reports 2026) |
| `rf` | Random Forest on hand-made features | TinyML offloading (IEEE OJ-COMS 2026), simplified |
| `weighted`, `greedy` | rule-based references | — |

## Status of testing

Tested on a workstation (OMNeT++ 6.1, Veins 5.3.1, SUMO 1.25.0, PyTorch CPU): unit tests, model
equivalence, all policies end to end through the simulator, training scripts, analysis.
**Not yet run:** anything on a GPU, a real Llama model, SLURM, the installer script. See
`docs/HPC_SETUP.md` section "What has not been tested".

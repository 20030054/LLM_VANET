#!/usr/bin/env python3
"""Write the run lists of all experiments (docs/EXPERIMENTS.md).

    python hpc/make_runlist.py --out hpc/runs [--seeds 5]

One file per experiment and resource type; one run per line:
    POLICY|VEHICLES|SPEED|SEED|TAG|SERVER_ARGS|SIM_ARGS      (fields may be empty)
Submit a file with:  sbatch --array=1-$(wc -l < FILE) hpc/array.sbatch FILE
"""
import argparse
import os

DENSITIES = [50, 100, 150, 200, 250, 300]
SPEEDS = [5, 10, 15, 20, 25, 30]
REF_N, REF_V = 150, 15
LLM = ["sequential", "batched", "prefix_cache", "kvrsu"]
CPU = ["rf", "weighted", "greedy"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="hpc/runs")
    ap.add_argument("--seeds", type=int, default=5)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    seeds = range(1, a.seeds + 1)
    lists = {}

    def add(name, policy, n, v, seed, tag="", server="", sim=""):
        lists.setdefault(name, []).append("|".join(map(str, (policy, n, v, seed, tag, server, sim))))

    # D: data generation for training (seeds 101.. are never used for evaluation)
    for n in DENSITIES:
        for v in (5, 15, 25):
            for seed in (101, 102):
                add("d_datagen_cpu", "greedy", n, v, seed, "datagen")

    # E2 density sweep and E3 speed sweep
    points = [(n, REF_V) for n in DENSITIES] + [(REF_N, v) for v in SPEEDS if v != REF_V]
    for n, v in points:
        for seed in seeds:
            for p in LLM:
                add("e2e3_main_gpu", p, n, v, seed)
            for p in CPU:
                add("e2e3_main_cpu", p, n, v, seed)

    # E4 ablation of the proposed method (GPU budget fixed so that spilling matters)
    budget = "--gpu-kv-budget-mb 32"
    for n in (100, 200, 300):
        for seed in seeds:
            add("e4_ablation_gpu", "kvrsu", n, REF_V, seed, "full-b32", budget)
            add("e4_ablation_gpu", "kvrsu", n, REF_V, seed, "noorder-b32", budget + " --ordering arrival")
            add("e4_ablation_gpu", "kvrsu", n, REF_V, seed, "noadmit-b32", budget + " --admission off")
            add("e4_ablation_gpu", "kvrsu_static", n, REF_V, seed, "b32", budget)
            add("e4_ablation_gpu", "prefix_cache", n, REF_V, seed, "b32", budget)
            add("e4_ablation_gpu", "batched", n, REF_V, seed, "b32", budget)

    # E5 GPU memory budget for the KV cache
    for mb in (8, 16, 32, 64, 128):
        for seed in seeds:
            for p in ("prefix_cache", "kvrsu_static", "kvrsu"):
                add("e5_budget_gpu", p, 300, REF_V, seed, f"b{mb}", f"--gpu-kv-budget-mb {mb}")

    # E6 slower edge hardware, emulated by scaling the measured inference time
    for scale in (2, 4, 8):
        for seed in seeds:
            for p in LLM:
                add("e6_scale_gpu", p, 200, REF_V, seed, f"x{scale}", "", f"--*.rsu[0].appl.computeScale={scale}")

    for name, rows in lists.items():
        path = os.path.join(a.out, name + ".txt")
        with open(path, "w") as f:
            f.write("\n".join(rows) + "\n")
        print(f"{len(rows):4d} runs  {path}")


if __name__ == "__main__":
    main()

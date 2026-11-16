#!/usr/bin/env python3
"""Calibrate the cost model and run the engine micro-benchmark (experiment E1).

    python engine/scripts/profile_engine.py --model meta-llama/Llama-3.2-1B \
        --profile-out results/profile.json --bench-out results/e1_engine.csv

Step 1 (calibration) measures on this GPU
    * the per-layer compute constants a1, a2 and the fixed cost t0, by timing
      recomputation of N table tokens with a context of c tokens, and
    * the host->GPU and GPU->host copy bandwidth,
and writes them to ``--profile-out``.  Pass that file to the server with
``--profile``.  Recalibrate for every GPU type and model.

Step 2 (benchmark) replays a synthetic RSU workload without the network
simulator and records the service time of every policy:
    vehicles x requests-per-window x churn  ->  one CSV row per policy.
``churn`` is the share of table slots that change between two windows; only
the ``mobile`` share of the vehicles ever changes (the rest are parked or
queued), which is what the stability ordering exploits.
"""
import argparse
import csv
import random
import statistics
import time

import _common  # noqa: F401
import numpy as np
import torch

from kvrsu.costmodel import CostModel, CostParams
from kvrsu.model import load_model
from kvrsu.prompt import Request, TableManager, Vehicle, suffix_text
from kvrsu.serving import PrefixEngine, make_engine

POLICIES = ["sequential", "batched", "prefix_cache", "kvrsu_static", "kvrsu"]


def calibrate(model, encode_len, reps=3):
    """Fit t = L * (a1 * N + a2 * N * c) + t0 from timed recomputations."""
    dev = model.device
    params = CostParams(n_layers=model.n_layers, kappa=float(model.kv_bytes_per_token_layer))
    eng = PrefixEngine(model, CostModel(params), spill="none")
    rng = random.Random(0)
    lo, hi = min(1000, model.vocab // 4), min(20000, model.vocab - 1)   # ordinary tokens of any vocabulary
    rows = []
    for c in (512, 1024, 2048, 4096, 8192):
        base = [rng.randrange(lo, hi) for _ in range(c)]
        for n in (64, 256, 1024, c):
            if n > c:
                continue
            ts = []
            for _ in range(reps + 1):
                eng.reset()
                eng.serve(base, [])                                   # fill the cache
                changed = base[: c - n] + [rng.randrange(lo, hi) for _ in range(n)]
                _, st = eng.serve(changed, [])                        # recompute the last n tokens
                ts.append(st["t_srv"])
            rows.append((n, c, min(ts[1:])))                          # first repetition is warm-up
    L = model.n_layers
    A = np.array([[L * n, L * n * c, 1.0] for n, c, _ in rows])
    y = np.array([t for _, _, t in rows])
    (a1, a2, t0), *_ = np.linalg.lstsq(A, y, rcond=None)
    params.a1, params.a2, params.t0 = float(max(a1, 1e-9)), float(max(a2, 0.0)), float(max(t0, 0.0))

    if torch.device(dev).type == "cuda":
        host = torch.empty(256 * 2**20, dtype=torch.uint8, pin_memory=True)
        gpu = host.to(dev)
        torch.cuda.synchronize()
        t = time.perf_counter(); gpu = host.to(dev, non_blocking=True); torch.cuda.synchronize()
        params.b_cg = host.numel() / (time.perf_counter() - t)
        t = time.perf_counter(); host.copy_(gpu, non_blocking=True); torch.cuda.synchronize()
        params.b_gc = host.numel() / (time.perf_counter() - t)
    pred = A @ np.array([params.a1, params.a2, params.t0])
    err = float(np.mean(np.abs(pred - y) / y))
    print(f"calibration: a1={params.a1:.3e} a2={params.a2:.3e} t0={params.t0:.3e} "
          f"b_cg={params.b_cg / 1e9:.2f} GB/s b_gc={params.b_gc / 1e9:.2f} GB/s  mean rel. error {err:.1%}")
    return params


def workload(n_veh, n_req, churn, mobile, windows, seed):
    """Synthetic windows: vehicle states and requests."""
    rng = random.Random(seed)
    vehs = [Vehicle(i, str(i), rng.uniform(0, 1000), rng.uniform(0, 1000), rng.uniform(0, 20),
                    rng.uniform(0, 6.28), rng.choice([1e9, 2e9]), 0.0) for i in range(n_veh)]
    movers = list(range(int(mobile * n_veh)))
    for _ in range(windows):
        for i in rng.sample(movers, min(len(movers), int(round(churn * n_veh)))):
            vehs[i].x = (vehs[i].x + rng.choice([20, 30, 40])) % 1000
        reqs = []
        for k in range(n_req):
            a = rng.randrange(n_veh)
            cands = rng.sample([j for j in range(n_veh) if j != a], min(8, n_veh - 1))
            reqs.append(Request(k, a, rng.uniform(1e8, 8e8), rng.uniform(2e3, 8e3), 1.0, 0.0, cands))
        yield vehs, reqs


def bench(model, encode, bos, params, a, writer):
    for n_veh in a.vehicles:
        for n_req in a.requests:
            for churn in a.churn:
                for policy in a.policies:
                    cost = CostModel(CostParams(**vars(params)))
                    eng = make_engine(policy, model, cost, a.gpu_kv_budget_mb, not a.no_streams)
                    ordering = "stability" if policy.startswith("kvrsu") else "arrival"
                    tm = TableManager(encode, bos, ordering, resort_every=10)
                    ts, us, ms, ps, ss = [], [], [], [], []
                    for w, (vehs, reqs) in enumerate(workload(n_veh, n_req, churn, a.mobile, a.windows + a.warmup, 0)):
                        t = time.perf_counter()
                        table, handle = tm.update(vehs)
                        suffixes = [encode(suffix_text(r, handle)) for r in reqs]
                        _, st = eng.serve(table, suffixes)
                        if model.device.type == "cuda":
                            torch.cuda.synchronize()
                        if w >= a.warmup:
                            ts.append(time.perf_counter() - t); us.append(st["u"]); ms.append(st["m"])
                            ps.append(st["p"]); ss.append(st["s"])
                    mean_t = statistics.mean(ts)
                    row = dict(policy=policy, ordering=ordering, budget_mb=a.gpu_kv_budget_mb, vehicles=n_veh,
                               requests=n_req, churn=churn, mobile=a.mobile, p=statistics.mean(ps),
                               u=statistics.mean(us), m=statistics.mean(ms), s=statistics.mean(ss),
                               t_mean=mean_t, t_p95=float(np.percentile(ts, 95)),
                               decisions_per_s=n_req / mean_t,
                               prompt_tokens_per_s=n_req * (statistics.mean(ps) + statistics.mean(ss)) / mean_t)
                    writer.writerow(row)
                    print({k: (round(v, 5) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
                    del eng
                    if model.device.type == "cuda":
                        torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="meta-llama/Llama-3.2-1B")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--load-4bit", action="store_true")
    ap.add_argument("--profile-out", default="profile.json")
    ap.add_argument("--profile-in", help="skip calibration and use this profile")
    ap.add_argument("--bench-out", help="CSV for the benchmark; omit to calibrate only")
    ap.add_argument("--policies", nargs="+", default=POLICIES)
    ap.add_argument("--vehicles", nargs="+", type=int, default=[50, 100, 150, 200, 250, 300])
    ap.add_argument("--requests", nargs="+", type=int, default=[1, 4, 16])
    ap.add_argument("--churn", nargs="+", type=float, default=[0.0, 0.1, 0.3])
    ap.add_argument("--mobile", type=float, default=0.6, help="share of vehicles that ever change")
    ap.add_argument("--windows", type=int, default=40)
    ap.add_argument("--warmup", type=int, default=12)
    ap.add_argument("--gpu-kv-budget-mb", type=float)
    ap.add_argument("--no-streams", action="store_true")
    a = ap.parse_args()

    model, encode, bos = load_model(a.model, a.device, a.dtype, a.load_4bit)
    if a.profile_in:
        params = CostParams.load(a.profile_in)
    else:
        params = calibrate(model, encode)
        params.save(a.profile_out)
        print("profile written to", a.profile_out)
    if a.bench_out:
        with open(a.bench_out, "w", newline="") as f:
            fields = ["policy", "ordering", "budget_mb", "vehicles", "requests", "churn", "mobile", "p", "u", "m",
                      "s", "t_mean", "t_p95", "decisions_per_s", "prompt_tokens_per_s"]
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            bench(model, encode, bos, params, a, writer)
        print("benchmark written to", a.bench_out)


if __name__ == "__main__":
    main()

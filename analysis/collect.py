#!/usr/bin/env python3
"""Collect all runs under a results directory into one table.

    python analysis/collect.py --results results --out results/summary.csv

One row per run (directory with meta.json, run-tasks.csv, run-windows.csv).

Metric definitions (measurement interval = sim_time - warm-up)
    tcr              tasks finished before their deadline / all finished-or-missed tasks
    tcr_covered      same, without tasks created outside RSU coverage
    resp_mean/p95    decision response time: task creation -> decision received (s)
    lat_mean/p95     task latency: creation -> result back, completed tasks only (s)
    thr_decisions    decisions delivered per second (policy + fallback)
    thr_tasks        tasks completed in time per second
    share_*          share of tasks decided by the policy / fallback / timeout / no coverage
    svc_mean/p95     engine service time per window (s)
    reuse            mean reused share of the table KV cache (u / p)
    util             engine busy time / measurement interval
    table_tok_per_s  table tokens served per second of engine time (n_policy * p / t_service)
"""
import argparse
import glob
import json
import os

import pandas as pd

WARMUP = 30.0


def one_run(d):
    meta = json.load(open(os.path.join(d, "meta.json")))
    t = pd.read_csv(os.path.join(d, "run-tasks.csv"))
    w = pd.read_csv(os.path.join(d, "run-windows.csv"))
    dur = meta["sim_time"] - WARMUP
    t = t[t.status != "censored"]
    dec = t[t.t_decision >= 0]
    ok = t[t.status == "ok"]
    cov = t[t["mode"] != "nocoverage"]
    row = dict(meta)
    row.update(
        density=meta["vehicles"] / meta["area_km2"], tasks=len(t),
        tcr=(t.status == "ok").mean() if len(t) else float("nan"),
        tcr_covered=(cov.status == "ok").mean() if len(cov) else float("nan"),
        resp_mean=dec.t_decision.mean(), resp_p95=dec.t_decision.quantile(0.95),
        lat_mean=ok.t_done.mean(), lat_p95=ok.t_done.quantile(0.95),
        thr_decisions=len(dec) / dur, thr_tasks=len(ok) / dur,
        svc_mean=w.t_service.mean(), svc_p95=w.t_service.quantile(0.95),
        reuse=(w.u / w.p.where(w.p > 0)).mean(), util=w.t_service.sum() / dur,
        table_tok_per_s=(w.n_policy * w.p).sum() / max(w.t_service.sum(), 1e-12),
        n_veh_table=w.n_veh.mean(), req_per_window=w.n_req.mean(),
    )
    for m in ("policy", "fallback", "timeout", "nocoverage"):
        row["share_" + m] = (t["mode"] == m).mean() if len(t) else float("nan")
    for tg in ("local", "edge", "v2v"):
        row["target_" + tg] = (t.target == tg).mean() if len(t) else float("nan")
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="results/summary.csv")
    a = ap.parse_args()
    rows, bad = [], []
    for meta in sorted(glob.glob(os.path.join(a.results, "**", "meta.json"), recursive=True)):
        try:
            rows.append(one_run(os.path.dirname(meta)))
        except Exception as e:  # incomplete run: report it, do not hide it
            bad.append((os.path.dirname(meta), repr(e)))
    df = pd.DataFrame(rows)
    df.to_csv(a.out, index=False)
    print(f"{len(df)} runs -> {a.out}")
    for d, e in bad:
        print("SKIPPED", d, e)
    if len(df):
        df["variant"] = df.policy + df.tag.map(lambda s: "-" + s if s else "")
        print(df.groupby(["variant", "vehicles", "speed"]).size().unstack(fill_value=0).to_string())


if __name__ == "__main__":
    main()

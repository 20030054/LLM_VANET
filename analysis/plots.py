#!/usr/bin/env python3
"""Paper figures from results/summary.csv (and the E1 benchmark CSV).

    python analysis/plots.py --summary results/summary.csv --e1 results/e1_engine.csv --out figures

Figures (PDF + PNG), one metric per panel, one line per method, mean over
seeds with a 95 % confidence band:
    density_<metric>   metric vs vehicle density at the reference speed
    speed_<metric>     metric vs speed limit at the reference density
    ablation           bars for the ablation variants (rows with a tag)
    e1_service_time    engine service time vs table size (no network)
A table with the same numbers is written next to every figure (.csv).
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Fixed colour, marker and label per method: a method keeps its colour in every figure.
METHODS = {
    "kvrsu":        ("#2a78d6", "o", "Proposed"),
    "prefix_cache": ("#eb6834", "s", "Prefix cache"),
    "batched":      ("#1baf7a", "^", "TightLLM-style batching"),
    "sequential":   ("#eda100", "D", "Per-request LLM"),
    "rf":           ("#e87ba4", "v", "Random Forest"),
    "weighted":     ("#008300", "P", "Weighted heuristic"),
    "greedy":       ("#6250d6", "X", "Greedy estimate"),
    "kvrsu_static": ("#e34948", "*", "Proposed, static spill"),
}
METRICS = {
    "tcr": ("Task completion ratio", 1.0), "resp_mean": ("Decision response time (ms)", 1e3),
    "resp_p95": ("95th pct. response time (ms)", 1e3), "lat_mean": ("Task latency (ms)", 1e3),
    "thr_decisions": ("Decisions per second", 1.0), "thr_tasks": ("Completed tasks per second", 1.0),
    "svc_mean": ("Engine service time (ms)", 1e3), "reuse": ("Reused share of table cache", 1.0),
    "util": ("Engine utilisation", 1.0), "share_policy": ("Share decided by the policy", 1.0),
}
INK, MUTED, GRID = "#1a1a19", "#6b6a63", "#e4e3dd"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "lines.linewidth": 2.0, "lines.markersize": 6, "legend.frameon": False,
                     "figure.facecolor": "white", "savefig.bbox": "tight", "pdf.fonttype": 42})


def ci95(x):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    return 1.96 * x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0


def save(fig, out, name, table=None):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"{name}.{ext}"), dpi=200)
    plt.close(fig)
    if table is not None:
        table.to_csv(os.path.join(out, f"{name}.csv"))


def sweep(df, xcol, xlabel, prefix, out):
    for metric, (ylabel, scale) in METRICS.items():
        fig, ax = plt.subplots(figsize=(3.5, 2.6))
        g = df.groupby(["policy", xcol])[metric]
        mean, err = g.mean().unstack(0) * scale, g.apply(ci95).unstack(0) * scale
        for pol, (color, marker, label) in METHODS.items():
            if pol not in mean or mean[pol].isna().all():
                continue
            ax.plot(mean.index, mean[pol], color=color, marker=marker, label=label,
                    markeredgecolor="white", markeredgewidth=0.8)
            ax.fill_between(mean.index, mean[pol] - err[pol], mean[pol] + err[pol], color=color, alpha=0.15, lw=0)
        ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_xticks(list(mean.index))
        if metric in ("resp_mean", "resp_p95", "svc_mean") and (mean.max().max() / max(mean.min().min(), 1e-9)) > 30:
            ax.set_yscale("log")
        ax.legend(fontsize=7, loc="best")
        save(fig, out, f"{prefix}_{metric}", pd.concat([mean.round(4), err.round(4).add_suffix("_ci95")], axis=1))


def ablation(df, out):
    df = df.copy()
    df["variant"] = df.policy + df.tag.map(lambda s: "-" + s if isinstance(s, str) and s else "")
    for metric in ("tcr", "resp_mean", "svc_mean", "reuse"):
        ylabel, scale = METRICS[metric]
        g = df.groupby(["variant", "vehicles"])[metric]
        mean, err = g.mean().unstack(1) * scale, g.apply(ci95).unstack(1) * scale
        fig, ax = plt.subplots(figsize=(max(3.5, 0.9 * len(mean)), 2.6))
        width = 0.8 / max(1, mean.shape[1])
        blues = ["#9cc3ee", "#5a9be0", "#2a78d6", "#1b56a0"]      # one hue, light -> dark = density
        for i, n in enumerate(mean.columns):
            ax.bar(np.arange(len(mean)) + i * width, mean[n], width * 0.92, yerr=err[n], color=blues[i % 4],
                   label=f"{n} vehicles", error_kw=dict(ecolor=MUTED, lw=0.8))
        ax.set_xticks(np.arange(len(mean)) + width * (mean.shape[1] - 1) / 2)
        ax.set_xticklabels(mean.index, rotation=25, ha="right", fontsize=7)
        ax.set_ylabel(ylabel); ax.grid(axis="x", visible=False); ax.legend(fontsize=7)
        save(fig, out, f"ablation_{metric}", pd.concat([mean.round(4), err.round(4).add_suffix("_ci95")], axis=1))


def e1(path, out):
    b = pd.read_csv(path)
    for (req, churn), part in b.groupby(["requests", "churn"]):
        fig, ax = plt.subplots(figsize=(3.5, 2.6))
        piv = part.pivot_table(index="vehicles", columns="policy", values="t_mean") * 1e3
        for pol, (color, marker, label) in METHODS.items():
            if pol in piv:
                ax.plot(piv.index, piv[pol], color=color, marker=marker, label=label,
                        markeredgecolor="white", markeredgewidth=0.8)
        ax.set_yscale("log"); ax.set_xlabel("Vehicles in the table"); ax.set_ylabel("Service time per window (ms)")
        ax.set_xticks(list(piv.index)); ax.legend(fontsize=7)
        save(fig, out, f"e1_service_time_r{req}_c{churn}", piv.round(3))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--summary", default="results/summary.csv")
    ap.add_argument("--e1")
    ap.add_argument("--out", default="figures")
    ap.add_argument("--ref-speed", type=float, default=15)
    ap.add_argument("--ref-vehicles", type=int, default=150)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if os.path.exists(a.summary):
        df = pd.read_csv(a.summary)
        df["tag"] = df.tag.fillna("")
        main_runs = df[df.tag == ""]
        sweep(main_runs[main_runs.speed == a.ref_speed], "density", "Vehicle density (vehicles/km$^2$)", "density", a.out)
        sweep(main_runs[main_runs.vehicles == a.ref_vehicles], "speed", "Speed limit (m/s)", "speed", a.out)
        if (df.tag != "").any():
            ablation(df[(df.tag != "") | (df.policy == "kvrsu")], a.out)
    if a.e1:
        e1(a.e1, a.out)
    print("figures written to", a.out)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build supervised training data from decision-server logs.

    python engine/scripts/build_dataset.py --logs results/datagen/*/windows.jsonl \
        --out data/train.jsonl

Input: ``windows.jsonl`` files written by the decision server (``--log``)
during data-generation runs (any policy; ``greedy`` is used in the protocol).

Label: for every logged request the *hindsight-best* option.  The logs hold
the position of every vehicle in every later window, so for each option we
can check with the real future trajectories whether the radio link would
have lasted until the result came back, and when the task would have
finished.  The label is the fastest option whose link survives and that
meets the deadline (if none meets it: the fastest whose link survives).
The model therefore learns to predict, from the present state only, what
the kinematic rule (constant-velocity extrapolation) can only guess.

Output: one JSON object per request with
    entries  table lines (shuffled: the model must not depend on slot order)
    suffix   request text
    label    correct answer letter
    options  per-option feature vector and 0/1 label (for the RF baseline)
"""
import argparse
import bisect
import json
import math
import random

import _common  # noqa: F401

from kvrsu import decide
from kvrsu.decide import Env, TARGET_EDGE, TARGET_LOCAL
from kvrsu.prompt import HEADER, Request, TableManager, Vehicle, suffix_text


class Trajectories:
    """Positions of all vehicles over time, as seen in the RSU table."""

    def __init__(self, records):
        self.t, self.xy = {}, {}
        for rec in records:
            for v in rec["veh"]:
                self.t.setdefault(v[0], []).append(rec["t"])
                self.xy.setdefault(v[0], []).append((v[2], v[3], v[4], v[5]))

    def at(self, addr, t, max_gap=2.0):
        """Position at time t, or None if the vehicle is unknown around t."""
        ts = self.t.get(addr)
        if not ts:
            return None
        i = bisect.bisect_right(ts, t)
        if i == 0:
            return None
        x0, y0, sp, hd = self.xy[addr][i - 1]
        if i < len(ts) and ts[i] - ts[i - 1] <= max_gap:
            x1, y1, _, _ = self.xy[addr][i]
            f = (t - ts[i - 1]) / (ts[i] - ts[i - 1])
            return x0 + f * (x1 - x0), y0 + f * (y1 - y0)
        dt = t - ts[i - 1]
        if dt > max_gap:                      # left the RSU's view: treat the link as lost
            return None
        return x0 + sp * math.cos(hd) * dt, y0 + sp * math.sin(hd) * dt


def link_alive(traj, a, b_pos_fn, t0, t1, rng, step=0.1):
    t = t0
    while True:
        pa, pb = traj.at(a, t), b_pos_fn(t)
        if pa is None or pb is None or math.hypot(pa[0] - pb[0], pa[1] - pb[1]) > rng:
            return False
        if t >= t1:
            return True
        t = min(t + step, t1)


def hindsight_label(req, opts, traj, env, now):
    rem = req.deadline - req.age
    scored = []
    for o in opts:
        if o.target == TARGET_LOCAL:
            alive = True
        elif o.target == TARGET_EDGE:
            alive = link_alive(traj, req.addr, lambda t: (env.rsu_x, env.rsu_y), now, now + o.t_done, env.v2i_range)
        else:
            alive = link_alive(traj, req.addr, lambda t, a=o.target: traj.at(a, t), now, now + o.t_done, env.v2v_range)
        scored.append((not alive, o.t_done > rem, o.t_done, o.label))
    return min(scored)[3]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--logs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-window", type=int, default=2, help="requests sampled per window")
    ap.add_argument("--max-samples", type=int, default=60000)
    ap.add_argument("--v2v-range", type=float, default=250.0)
    ap.add_argument("--v2i-range", type=float, default=600.0)
    ap.add_argument("--rate-bps", type=float, default=3.0e6)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    rng = random.Random(a.seed)
    samples = []
    for path in a.logs:
        with open(path) as f:
            records = [json.loads(line) for line in f]
        traj = Trajectories(records)
        tm = TableManager(lambda s: [s], None, "random", seed=rng.randrange(10**9))  # "tokens" are whole lines
        env = Env(v2v_range=a.v2v_range, v2i_range=a.v2i_range, rate_bps=a.rate_bps)
        for rec in records:
            vehicles = [Vehicle(*v) for v in rec["veh"]]
            vehs = {v.addr: v for v in vehicles}
            env.rsu_x, env.rsu_y, env.rsu_cpu_hz, env.rsu_backlog = rec["rsu"]
            pieces, handle = tm.update(vehicles)
            reqs = [r for r in rec["req"] if r[1] in vehs]
            for r in rng.sample(reqs, min(a.per_window, len(reqs))):
                req = Request(r[0], r[1], r[2], r[3], r[4], r[5], r[6])
                opts = decide.build_options(req, vehs, env)
                label = hindsight_label(req, opts, traj, env, rec["t"])
                samples.append(dict(entries=pieces[1:], suffix=suffix_text(req, handle), label=label,
                                    options=[dict(label=o.label, feat=o.feat, y=int(o.label == label)) for o in opts]))
    rng.shuffle(samples)
    samples = samples[: a.max_samples]
    with open(a.out, "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")
    counts = {}
    for s in samples:
        k = "vehicle" if s["label"] in "ABCDEFGH" else ("local" if s["label"] == "L" else "edge")
        counts[k] = counts.get(k, 0) + 1
    assert all(s["entries"] and HEADER for s in samples)
    print(f"{len(samples)} samples written to {a.out}; label mix: {counts}")


if __name__ == "__main__":
    main()

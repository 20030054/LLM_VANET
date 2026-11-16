"""Offloading options, kinematic estimates and the non-LLM decision rules.

For each request the RSU considers up to ``max_cands`` nearby vehicles
(labels A..H), local execution (L) and the RSU edge server (R).  This module
computes, from the RSU's current view only (no future knowledge):

* an estimate of the completion time of every option,
* a kinematic estimate of how long the radio link will last,
* the decision slack used by deadline-aware admission,
* the heuristic decisions used as fallback and as baselines,
* the feature vectors of the Random-Forest baseline.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

from .prompt import CAND_LABELS, EDGE, LOCAL, Request, Vehicle

TARGET_LOCAL, TARGET_EDGE = -1, -2
FEATURES = ["is_local", "is_edge", "cpu_ghz", "backlog", "dist", "rel_speed", "lifetime",
            "t_tx", "t_exec", "t_done", "deadline_rem", "gcycles", "kbytes", "done_ratio"]


@dataclass
class Env:
    rsu_x: float = 0.0
    rsu_y: float = 0.0
    rsu_cpu_hz: float = 2.0e10
    rsu_backlog: float = 0.0
    v2v_range: float = 250.0      # candidate radius and assumed V2V range (m)
    v2i_range: float = 600.0      # assumed vehicle-RSU range (m)
    rate_bps: float = 3.0e6       # effective application-level data rate
    t_fixed: float = 0.005        # fixed per-transfer overhead (s)
    max_cands: int = len(CAND_LABELS)


@dataclass
class Option:
    label: str
    target: int        # vehicle addr, TARGET_LOCAL or TARGET_EDGE
    t_done: float      # estimated time from now until the result is back
    lifetime: float    # estimated remaining link lifetime (inf for local)
    feasible: bool     # link is expected to outlive the task
    feat: List[float]


def _vel(v: Vehicle) -> Tuple[float, float]:
    return v.speed * math.cos(v.heading), v.speed * math.sin(v.heading)


def link_lifetime(ax, ay, avx, avy, bx, by, bvx, bvy, rng: float, cap: float = 60.0) -> float:
    """Time until two nodes moving at constant velocity are more than ``rng`` apart."""
    dx, dy, dvx, dvy = bx - ax, by - ay, bvx - avx, bvy - avy
    if dx * dx + dy * dy > rng * rng:
        return 0.0
    a = dvx * dvx + dvy * dvy
    if a < 1e-9:
        return cap
    b = 2 * (dx * dvx + dy * dvy)
    c = dx * dx + dy * dy - rng * rng
    disc = b * b - 4 * a * c
    return min(cap, (-b + math.sqrt(max(disc, 0.0))) / (2 * a))


def pick_candidates(req: Request, vehs: Dict[int, Vehicle], env: Env) -> List[int]:
    me = vehs[req.addr]
    near = []
    for a, v in vehs.items():
        if a == req.addr:
            continue
        d = math.hypot(v.x - me.x, v.y - me.y)
        if d <= env.v2v_range:
            near.append((d, a))
    near.sort()
    return [a for _, a in near[: env.max_cands]]


def build_options(req: Request, vehs: Dict[int, Vehicle], env: Env) -> List[Option]:
    """All options of a request, candidates first (label order), then L and R."""
    me = vehs[req.addr]
    mvx, mvy = _vel(me)
    rem = max(req.deadline - req.age, 1e-3)
    t_tx = req.size_bytes * 8 / env.rate_bps + env.t_fixed
    out: List[Option] = []

    def add(label, target, cpu, backlog, dist, rel_speed, life, remote):
        t_exec = req.cycles / cpu
        t_done = backlog + t_exec + ((t_tx + env.t_fixed) if remote else 0.0)
        feat = [float(target == TARGET_LOCAL), float(target == TARGET_EDGE), cpu / 1e9, backlog, dist,
                rel_speed, life, t_tx if remote else 0.0, t_exec, t_done, rem, req.cycles / 1e9,
                req.size_bytes / 1e3, t_done / rem]
        out.append(Option(label, target, t_done, life, life >= t_done, feat))

    for i, a in enumerate(req.candidates):
        v = vehs[a]
        vx, vy = _vel(v)
        life = link_lifetime(me.x, me.y, mvx, mvy, v.x, v.y, vx, vy, env.v2v_range)
        add(CAND_LABELS[i], a, v.cpu_hz, v.backlog, math.hypot(v.x - me.x, v.y - me.y),
            math.hypot(vx - mvx, vy - mvy), life, True)
    add(LOCAL, TARGET_LOCAL, me.cpu_hz, me.backlog, 0.0, 0.0, 60.0, False)
    life_r = link_lifetime(me.x, me.y, mvx, mvy, env.rsu_x, env.rsu_y, 0.0, 0.0, env.v2i_range)
    add(EDGE, TARGET_EDGE, env.rsu_cpu_hz, env.rsu_backlog, math.hypot(env.rsu_x - me.x, env.rsu_y - me.y),
        me.speed, life_r, True)
    return out


def best_estimate(opts: Sequence[Option]) -> Option:
    feas = [o for o in opts if o.feasible]
    return min(feas or opts, key=lambda o: o.t_done)


def decision_slack(req: Request, opts: Sequence[Option]) -> float:
    """Time the request can still wait for a decision and meet its deadline."""
    return (req.deadline - req.age) - best_estimate(opts).t_done


def greedy(req: Request, opts: Sequence[Option]) -> Option:
    """Minimum estimated completion time among link-feasible options."""
    return best_estimate(opts)


def weighted(req: Request, opts: Sequence[Option], env: Env,
             w=(0.4, 0.3, 0.3), threshold: float = 0.35) -> Option:
    """Weighted-suitability rule after Eq. (21) of the Sci. Reports 2026 paper.

    S_v = w1 * resource ratio - w2 * normalised distance + w3 * link reliability.
    The bandwidth ratio of the original is omitted: all links share one
    802.11p channel, so it is the same for every candidate.
    """
    rem = max(req.deadline - req.age, 1e-3)
    best, best_s = None, -1e9
    for o in opts:
        if o.target < 0:
            continue
        cpu_hz, backlog, dist = o.feat[2] * 1e9, o.feat[3], o.feat[4]
        ratio = min(max(cpu_hz * (rem - backlog), 0.0) / req.cycles, 3.0) / 3.0
        s = w[0] * ratio - w[1] * dist / env.v2v_range + w[2] * math.exp(-(dist / env.v2v_range) ** 2)
        if s > best_s:
            best, best_s = o, s
    by = {o.label: o for o in opts}
    if best is not None and best_s >= threshold and best.feasible and best.t_done <= rem:
        return best
    if by[EDGE].feasible and by[EDGE].t_done <= rem:
        return by[EDGE]
    return by[LOCAL]

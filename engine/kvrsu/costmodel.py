"""Cost model and per-window optimisation (problem P_w in docs/DESIGN.md).

Per-layer time of a window with a shared table of ``p`` tokens, of which
``m`` are taken from cache (``ghat`` of them already on the GPU) and
``p - m`` are recomputed, serving ``n`` requests of ``s`` suffix tokens:

    T_cmp  = t(p - m, p) + t(n * s, p + s)
    T_load = kappa * (m - ghat) / B_cg
    T_off  = kappa * (p - m)   / B_gc
    T_srv  = L * max(T_cmp, T_load, T_off) + t0          (pipelined)

with ``t(N, c) = a1 * N + a2 * N * c``.  The constants come from
``scripts/profile_engine.py``; an online correction factor absorbs what the
linear model misses (TightLLM shows GPU utilisation varies with length).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import List, Sequence, Tuple


@dataclass
class CostParams:
    n_layers: int = 16
    kappa: float = 2048.0     # KV bytes per token per layer
    a1: float = 2.0e-7        # s per token per layer
    a2: float = 2.0e-11       # s per (token x context token) per layer
    b_cg: float = 6.0e9       # host -> GPU bytes/s
    b_gc: float = 6.0e9       # GPU -> host bytes/s
    t0: float = 1.0e-3        # fixed cost per window (embedding, head, Python)
    pipelined: bool = True    # False: transfers are not overlapped (sum, not max)

    @classmethod
    def load(cls, path: str) -> "CostParams":
        with open(path) as f:
            d = json.load(f)
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)


class CostModel:
    def __init__(self, params: CostParams, ewma: float = 0.2):
        self.p = params
        self.ewma = ewma
        self.correction = 1.0  # measured / predicted, smoothed

    # ------------------------------------------------------------ prediction
    def t_layer(self, n_tokens: float, context: float) -> float:
        return self.p.a1 * n_tokens + self.p.a2 * n_tokens * context

    def predict(self, p: int, m: int, ghat: int, n: int, s: int) -> float:
        """Predicted service time of one window in seconds."""
        c = self.p
        t_cmp = self.t_layer(p - m, p) + self.t_layer(n * s, p + s)
        t_load = c.kappa * max(m - ghat, 0) / c.b_cg
        t_off = c.kappa * (p - m) / c.b_gc
        per_layer = max(t_cmp, t_load, t_off) if c.pipelined else t_cmp + t_load + t_off
        return self.correction * (c.n_layers * per_layer + c.t0)

    def observe(self, predicted: float, measured: float) -> None:
        """Update the online correction with one measured window."""
        if predicted > 0 and measured > 0:
            raw = predicted / self.correction
            self.correction = (1 - self.ewma) * self.correction + self.ewma * (measured / raw)

    # ---------------------------------------------------------- optimisation
    def choose_m(self, p: int, u: int, ghat: int, n: int, s: int, step: int = 16) -> Tuple[int, float]:
        """Best number of cached prefix tokens to use, ``ghat <= m <= u``.

        ``T_cmp`` and ``T_off`` fall and ``T_load`` rises with ``m``, so the
        max is unimodal; a coarse scan is exact enough and costs microseconds.
        """
        best_m, best_t = ghat, self.predict(p, ghat, ghat, n, s)
        cands = list(range(ghat, u + 1, max(step, 1)))
        if not cands or cands[-1] != u:
            cands.append(u)
        for m in cands:
            t = self.predict(p, m, ghat, n, s)
            if t < best_t - 1e-12:
                best_m, best_t = m, t
        return best_m, best_t

    def admit(
        self, slacks: Sequence[float], p: int, u: int, ghat: int, s: int,
        max_service: float = float("inf"), adaptive_m: bool = True,
    ) -> Tuple[List[int], int, float]:
        """Deadline-aware admission.

        ``slacks[i]`` is the time request ``i`` can still wait for a decision.
        The service time depends on the batch only through its size, so the
        best batch of size ``n`` is the ``n`` requests with the largest slack.
        Returns ``(indices of admitted requests, m, predicted service time)``;
        requests that are not admitted fall back to the heuristic.
        """
        order = sorted(range(len(slacks)), key=lambda i: -slacks[i])
        for n in range(len(order), 0, -1):
            if adaptive_m:
                m, t = self.choose_m(p, u, ghat, n, s)
            else:
                m, t = u, self.predict(p, u, ghat, n, s)
            if t <= slacks[order[n - 1]] and t <= max_service:
                return order[:n], m, t
        m, t = (self.choose_m(p, u, ghat, 0, s) if adaptive_m else (u, self.predict(p, u, ghat, 0, s)))
        return [], m, t

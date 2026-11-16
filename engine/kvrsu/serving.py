"""Serving engines: how one decision window is turned into LLM forward passes.

Every engine has the same interface::

    logits, stats = engine.serve(table_ids, suffix_ids)

``table_ids`` is the tokenised RSU resource table shared by all requests of
the window; ``suffix_ids`` holds one short token list per request.  The
prompt of request ``r`` is ``table_ids + suffix_ids[r]`` and ``logits[r]``
are the next-token logits after it.  All engines compute the same function,
so their decisions are identical up to floating-point noise; they differ
only in how much work they do.

Engines
-------
``PlainEngine("sequential")``  one forward pass per request (per-request LLM
                               serving, as in the Sci. Reports 2026 system).
``PlainEngine("batched")``     requests batched up to the GPU memory budget,
                               nothing shared (throughput-oriented serving in
                               the style of TightLLM / FlexGen).
``PrefixEngine``               the table is processed once per window and
                               shared by the batch; its KV cache is reused
                               across windows up to the longest common prefix.
                               ``spill`` selects what happens when the cache
                               does not fit the GPU budget:
                                 "none"     keep what fits, recompute the rest
                                 "static"   keep the rest on the host and
                                            always load it
                                 "adaptive" choose per window how much to load
                                            and how much to recompute
"""
from __future__ import annotations

import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .costmodel import CostModel
from .model import LayerwiseLlama


def lcp(a: Sequence[int], b: Sequence[int]) -> int:
    """Length of the longest common prefix of two token sequences."""
    n = min(len(a), len(b))
    if n == 0:
        return 0
    diff = np.nonzero(np.asarray(a[:n]) != np.asarray(b[:n]))[0]
    return int(diff[0]) if len(diff) else n


def _sync(device) -> None:
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize()


def _pad(suffix_ids: List[List[int]], device) -> Tuple[torch.Tensor, torch.Tensor]:
    s = max(len(x) for x in suffix_ids)
    ids = torch.zeros(len(suffix_ids), s, dtype=torch.long)
    for i, x in enumerate(suffix_ids):
        ids[i, : len(x)] = torch.tensor(x, dtype=torch.long)
    last = torch.tensor([len(x) - 1 for x in suffix_ids], dtype=torch.long)
    return ids.to(device), last.to(device)


# --------------------------------------------------------------------- plain
class PlainEngine:
    """Baselines without any reuse of the shared table."""

    def __init__(self, model: LayerwiseLlama, mode: str = "sequential",
                 gpu_kv_budget_mb: Optional[float] = None, max_batch_tokens: int = 65536):
        assert mode in ("sequential", "batched")
        self.model, self.mode = model, mode
        self.budget = None if gpu_kv_budget_mb is None else gpu_kv_budget_mb * 2**20
        self.max_batch_tokens = max_batch_tokens

    def reset(self) -> None:
        pass

    def _batch_size(self, prompt_len: int) -> int:
        if self.mode == "sequential":
            return 1
        if self.budget is not None:  # transient K/V of one layer must fit the budget
            return max(1, int(self.budget // (self.model.kv_bytes_per_token_layer * prompt_len)))
        return max(1, self.max_batch_tokens // prompt_len)

    @torch.no_grad()
    def serve(self, table_ids: List[int], suffix_ids: List[List[int]]):
        m = self.model
        _sync(m.device)
        t0 = time.perf_counter()
        p, n = len(table_ids), len(suffix_ids)
        suf, last = _pad(suffix_ids, m.device)
        table = torch.tensor(table_ids, dtype=torch.long, device=m.device)
        bs = self._batch_size(p + suf.shape[1])
        out = []
        for i in range(0, n, bs):
            chunk = suf[i:i + bs]
            ids = torch.cat((table.expand(chunk.shape[0], -1), chunk), dim=1)
            out.append(m.forward_full(ids, last[i:i + bs] + p))
        logits = torch.cat(out) if out else torch.empty(0, m.vocab)
        _sync(m.device)
        stats = dict(p=p, u=0, ghat=0, m=0, n_rec=p * n, n_load=0, n=n, s=int(suf.shape[1]),
                     t_srv=time.perf_counter() - t0, t_pred=0.0, batch=bs)
        return logits, stats


# -------------------------------------------------------------------- prefix
class _Mover:
    """Moves KV blocks between host and GPU, overlapped with computation.

    On CUDA the copies run on their own streams so that they overlap with the
    compute kernels of the default stream (the pipelining assumed by the cost
    model).  On CPU, or with ``use_streams=False``, copies are synchronous.
    """

    def __init__(self, device, use_streams: bool):
        self.cuda = torch.device(device).type == "cuda"
        self.streams = self.cuda and use_streams
        if self.streams:
            self.load_stream = torch.cuda.Stream(device)
            self.off_stream = torch.cuda.Stream(device)
        self.device = device
        self._keep: list = []

    def load(self, host: torch.Tensor, a: int, b: int) -> torch.Tensor:
        """Start loading host[a:b]; returns a (1, n_kv, b-a, hd) GPU view."""
        if self.streams:
            with torch.cuda.stream(self.load_stream):
                t = host[a:b].to(self.device, non_blocking=True)
        else:
            t = host[a:b].to(self.device)
        return t.permute(1, 0, 2).unsqueeze(0)

    def wait_load(self) -> None:
        if self.streams:
            self.load_stream.synchronize()

    def offload(self, host: torch.Tensor, a: int, kv: torch.Tensor) -> None:
        """Start copying new KV (1, n_kv, T, hd) into host[a:a+T]."""
        src = kv[0].permute(1, 0, 2).contiguous()
        if self.streams:
            with torch.cuda.stream(self.off_stream):
                self.off_stream.wait_stream(torch.cuda.current_stream(self.device))
                host[a:a + src.shape[0]].copy_(src, non_blocking=True)
            self._keep.append(src)  # keep alive until the copy has finished
        else:
            host[a:a + src.shape[0]].copy_(src)

    def finish(self) -> None:
        if self.streams:
            self.off_stream.synchronize()
        self._keep.clear()


class PrefixEngine:
    """Shared-prefix serving with cross-window reuse and host spill."""

    def __init__(self, model: LayerwiseLlama, cost: CostModel, spill: str = "adaptive",
                 gpu_kv_budget_mb: Optional[float] = None, suffix_chunk: int = 32,
                 max_prefix_tokens: int = 16384, use_streams: bool = True):
        assert spill in ("none", "static", "adaptive")
        self.model, self.cost, self.spill = model, cost, spill
        self.suffix_chunk = suffix_chunk
        self.host_cap = max_prefix_tokens
        per_token = model.n_layers * model.kv_bytes_per_token_layer
        self.g_cap = 10**9 if gpu_kv_budget_mb is None else int(gpu_kv_budget_mb * 2**20 // per_token)
        self.mover = _Mover(model.device, use_streams)
        self.reset()

    def reset(self) -> None:
        L = self.model.n_layers
        self.tokens: List[int] = []          # sequence whose KV is cached
        self.g_res = 0                       # leading tokens resident on the GPU
        self.gpu_k: List[Optional[torch.Tensor]] = [None] * L
        self.gpu_v: List[Optional[torch.Tensor]] = [None] * L
        self.host_k: List[Optional[torch.Tensor]] = [None] * L
        self.host_v: List[Optional[torch.Tensor]] = [None] * L

    # ------------------------------------------------------------- host store
    def _ensure_host(self, p: int) -> None:
        if self.host_k[0] is not None and self.host_k[0].shape[0] >= p:
            return
        m = self.model
        cap = max(self.host_cap, 2 * p)
        pin = self.mover.cuda
        for l in range(m.n_layers):
            for store in (self.host_k, self.host_v):
                new = torch.empty(cap, m.n_kv, m.head_dim, dtype=m.dtype, pin_memory=pin)
                if store[l] is not None:
                    new[: store[l].shape[0]] = store[l]
                store[l] = new

    # ------------------------------------------------------------------ plan
    def plan(self, table_ids: List[int], n: int, s: int) -> Dict[str, int]:
        """Reuse figures for the next window without running it."""
        p = len(table_ids)
        u = lcp(self.tokens, table_ids)
        ghat = min(self.g_res, u)
        return dict(p=p, u=u, ghat=ghat)

    # ----------------------------------------------------------------- serve
    @torch.no_grad()
    def serve(self, table_ids: List[int], suffix_ids: List[List[int]], m_override: Optional[int] = None):
        model = self.model
        dev = model.device
        _sync(dev)
        t0 = time.perf_counter()

        p, n = len(table_ids), len(suffix_ids)
        suf, last = _pad(suffix_ids, dev) if n else (torch.zeros(0, 1, dtype=torch.long, device=dev), None)
        s = int(suf.shape[1]) if n else 0
        u = lcp(self.tokens, table_ids)
        ghat = min(self.g_res, u)

        if self.spill == "none":
            m, t_pred = ghat, self.cost.predict(p, ghat, ghat, n, s)
        elif self.spill == "static":
            m, t_pred = u, self.cost.predict(p, u, ghat, n, s)
        else:
            m, t_pred = self.cost.choose_m(p, u, ghat, n, s)
        if m_override is not None:
            m = max(ghat, min(u, m_override))
            t_pred = self.cost.predict(p, m, ghat, n, s)
        n_load, n_rec = m - ghat, p - m
        use_host = self.spill != "none"
        if use_host:
            self._ensure_host(p)

        g_new = min(self.g_cap, p)
        table = torch.tensor(table_ids, dtype=torch.long, device=dev)
        h_tail = model.embed_tokens(table[m:].unsqueeze(0)) if n_rec else None
        pos_tail = torch.arange(m, p, device=dev)
        h_suf = model.embed_tokens(suf) if n else None
        pos_suf = torch.arange(p, p + s, device=dev)

        pending = None
        if n_load:
            pending = (self.mover.load(self.host_k[0], ghat, m), self.mover.load(self.host_v[0], ghat, m))

        for l in range(model.n_layers):
            parts_k, parts_v = [], []
            if ghat:
                parts_k.append(self.gpu_k[l][:, :, :ghat])
                parts_v.append(self.gpu_v[l][:, :, :ghat])
            if n_load:
                self.mover.wait_load()
                parts_k.append(pending[0])
                parts_v.append(pending[1])
                if l + 1 < model.n_layers:  # prefetch the next layer while this one computes
                    pending = (self.mover.load(self.host_k[l + 1], ghat, m),
                               self.mover.load(self.host_v[l + 1], ghat, m))
            if n_rec:
                past_k = torch.cat(parts_k, dim=2) if parts_k else None
                past_v = torch.cat(parts_v, dim=2) if parts_v else None
                h_tail, k_t, v_t = model.layer_step(l, h_tail, pos_tail, past_k, past_v)
                if use_host:
                    self.mover.offload(self.host_k[l], m, k_t)
                    self.mover.offload(self.host_v[l], m, v_t)
                parts_k.append(k_t)
                parts_v.append(v_t)
            k_full = torch.cat(parts_k, dim=2) if parts_k else None
            v_full = torch.cat(parts_v, dim=2) if parts_v else None

            if n:
                outs = []
                for i in range(0, n, self.suffix_chunk):
                    o, _, _ = model.layer_step(l, h_suf[i:i + self.suffix_chunk], pos_suf, k_full, v_full)
                    outs.append(o)
                h_suf = torch.cat(outs) if len(outs) > 1 else outs[0]

            # keep the first g_new tokens of this layer resident on the GPU
            if g_new == 0:
                self.gpu_k[l] = self.gpu_v[l] = None
            elif ghat < g_new or self.gpu_k[l].shape[2] != g_new:
                self.gpu_k[l] = k_full[:, :, :g_new].clone()
                self.gpu_v[l] = v_full[:, :, :g_new].clone()
            del k_full, v_full, parts_k, parts_v

        self.mover.finish()
        self.g_res = g_new
        self.tokens = list(table_ids) if use_host else list(table_ids[:g_new])

        if n:
            logits = model.head(h_suf[torch.arange(n, device=dev), last])
        else:
            logits = torch.empty(0, model.vocab, device=dev)
        _sync(dev)
        t_srv = time.perf_counter() - t0
        self.cost.observe(t_pred, t_srv)
        stats = dict(p=p, u=u, ghat=ghat, m=m, n_rec=n_rec, n_load=n_load, n=n, s=s,
                     t_srv=t_srv, t_pred=t_pred, batch=n)
        return logits, stats


def make_engine(policy: str, model: LayerwiseLlama, cost: CostModel,
                gpu_kv_budget_mb: Optional[float] = None, use_streams: bool = True):
    """Engine for a policy name used on the command line.

    sequential    per-request serving                (Sci. Reports 2026)
    batched       memory-bounded batching, no reuse  (TightLLM-style)
    prefix_cache  shared prefix + reuse, GPU only    (prompt caching)
    kvrsu_static  + host spill, always load          (ablation)
    kvrsu         + adaptive load/recompute          (proposed)
    """
    if policy in ("sequential", "batched"):
        return PlainEngine(model, policy, gpu_kv_budget_mb)
    spill = {"prefix_cache": "none", "kvrsu_static": "static", "kvrsu": "adaptive"}[policy]
    return PrefixEngine(model, cost, spill, gpu_kv_budget_mb, use_streams=use_streams)

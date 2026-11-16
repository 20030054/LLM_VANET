#!/usr/bin/env python3
"""Verify, for a real model, that the engine computes what Hugging Face computes.

    python engine/scripts/check_equivalence.py --model meta-llama/Llama-3.2-1B

Run this once after installing the environment and again after changing the
model or the transformers version.  It checks

  1. the layer-wise forward pass against ``model(...)`` of Hugging Face, and
  2. every serving policy against per-request serving on synthetic windows
     (same decision for every request).

Exit code 0 means both checks passed.
"""
import argparse
import random
import sys

import _common  # noqa: F401
import torch

from kvrsu.costmodel import CostModel, CostParams
from kvrsu.model import load_model
from kvrsu.prompt import LabelSet
from kvrsu.serving import PlainEngine, make_engine


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.2-1B")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--load-4bit", action="store_true")
    a = ap.parse_args()

    model, encode, bos = load_model(a.model, a.device, a.dtype, a.load_4bit)
    labels = torch.tensor(sorted(LabelSet(encode).ids.values()), device=model.device)
    text = "RSU table: id x y speed heading cpu queue\n" + "".join(
        f"{i:03d} x{37 * i % 100:03d} y{53 * i % 100:03d} v{i % 20:02d} h{i % 8} c02 q00\n" for i in range(60))
    ids = torch.tensor([[bos] + encode(text)], device=model.device)
    last = torch.tensor([ids.shape[1] - 1], device=model.device)

    ours = model.forward_full(ids, last).float()
    with torch.no_grad():
        ref = model.hf(ids).logits[:, -1].float()
    err = (ours - ref).abs().max().item()
    tol = 1e-3 if model.dtype == torch.float32 else 0.15   # half precision accumulates rounding noise
    ok1 = err < tol and ours.argmax().item() == ref.argmax().item()
    print(f"[1] layer-wise vs Hugging Face: max |diff| = {err:.4g}  (tolerance {tol})  {'PASS' if ok1 else 'FAIL'}")

    rng = random.Random(0)
    lines = [encode(f"{i:03d} x{rng.randrange(100):03d} y{rng.randrange(100):03d} v10 h1 c02 q00\n") for i in range(80)]
    ref_engine = PlainEngine(model, "sequential")
    ok2 = True
    for policy in ("batched", "prefix_cache", "kvrsu_static", "kvrsu"):
        eng = make_engine(policy, model, CostModel(CostParams(n_layers=model.n_layers,
                          kappa=model.kv_bytes_per_token_layer)), gpu_kv_budget_mb=4.0)
        agree = total = 0
        worst = 0.0
        for _ in range(6):
            for i in rng.sample(range(80), 12):
                lines[i] = encode(f"{i:03d} x{rng.randrange(100):03d} y{rng.randrange(100):03d} v10 h1 c02 q00\n")
            table = [bos] + [t for ln in lines for t in ln]
            suffixes = [encode(f"task c{rng.randrange(900):04d} b004 d09 from {rng.randrange(80):03d} cand A=001 B=002 ; answer:")
                        for _ in range(5)]
            x, _ = ref_engine.serve(table, suffixes)
            y, _ = eng.serve(table, suffixes)
            agree += (x[:, labels].argmax(-1) == y[:, labels].argmax(-1)).sum().item()
            total += len(suffixes)
            worst = max(worst, (x.float() - y.float()).abs().max().item())
        good = agree == total
        ok2 &= good
        print(f"[2] {policy:13s} same decision in {agree}/{total} requests, max |logit diff| = {worst:.4g}  "
              f"{'PASS' if good else 'CHECK'}")
    if not ok2:
        print("    A few differing decisions in half precision are near-ties caused by rounding; "
              "rerun with --dtype float32 to confirm exact agreement.")
    sys.exit(0 if (ok1 and ok2) else 1)


if __name__ == "__main__":
    main()

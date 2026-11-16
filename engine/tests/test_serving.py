"""All serving engines must produce the same logits on the same windows."""
import random
import pytest
import torch
from kvrsu.costmodel import CostModel, CostParams
from kvrsu.model import load_model
from kvrsu.serving import PlainEngine, PrefixEngine, lcp, make_engine


def windows(n_windows=6, slots=12, slot_len=9, seed=0):
    """A table whose slots change at random, plus a few request suffixes."""
    rng = random.Random(seed)
    table = [[rng.randrange(256) for _ in range(slot_len)] for _ in range(slots)]
    for w in range(n_windows):
        for i in range(len(table)):
            if rng.random() < 0.25:
                table[i] = [rng.randrange(256) for _ in range(slot_len)]
        if w == 3:
            table = table[:-2]                     # vehicles leave
        if w == 4:
            table.append([rng.randrange(256) for _ in range(slot_len)])
        flat = [t for slot in table for t in slot]
        suffixes = [[rng.randrange(256) for _ in range(rng.randrange(4, 9))] for _ in range(rng.randrange(1, 5))]
        yield flat, suffixes


@pytest.mark.parametrize("policy,budget", [
    ("batched", None), ("batched", 0.01), ("prefix_cache", None), ("prefix_cache", 0.02),
    ("kvrsu_static", 0.02), ("kvrsu", 0.02), ("kvrsu", None), ("kvrsu", 0.0),
])
def test_engines_agree_with_sequential(policy, budget):
    model, _, _ = load_model("tiny-random", "cpu")
    ref = PlainEngine(model, "sequential")
    eng = make_engine(policy, model, CostModel(CostParams(n_layers=model.n_layers)), budget)
    for table, suffixes in windows():
        a, _ = ref.serve(table, suffixes)
        b, st = eng.serve(table, suffixes)
        assert torch.allclose(a, b, atol=2e-5), (policy, (a - b).abs().max())
        assert (a.argmax(-1) == b.argmax(-1)).all()
        assert st["p"] == len(table) and st["n"] == len(suffixes)


def test_reuse_is_longest_common_prefix():
    model, _, _ = load_model("tiny-random", "cpu")
    eng = PrefixEngine(model, CostModel(CostParams(n_layers=model.n_layers)), spill="static")
    prev = None
    for table, suffixes in windows(seed=3):
        _, st = eng.serve(table, suffixes)
        if prev is not None:
            assert st["u"] == lcp(prev, table)
            assert st["m"] == st["u"] and st["n_rec"] == len(table) - st["u"]
        prev = table


def test_forced_split_between_load_and_recompute():
    """Any split m between loading and recomputing must give the same logits."""
    model, _, _ = load_model("tiny-random", "cpu")
    table, suffixes = next(windows(seed=5))
    ref, _ = PlainEngine(model, "sequential").serve(table, suffixes)
    for m in (0, 7, 40, len(table)):
        eng = PrefixEngine(model, CostModel(CostParams(n_layers=model.n_layers)), spill="adaptive",
                           gpu_kv_budget_mb=0.005)
        eng.serve(table, suffixes)                       # fill the cache
        out, st = eng.serve(table, suffixes, m_override=m)
        assert st["m"] == max(st["ghat"], m)
        assert torch.allclose(ref, out, atol=2e-5)

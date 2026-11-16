import itertools, random
from kvrsu.costmodel import CostModel, CostParams
from kvrsu.prompt import LabelSet, Quant, TableManager, Vehicle, entry_text, expected_reuse

enc = lambda s: list(s.encode())


def test_stable_first_order_maximises_expected_reuse():
    q = [0.9, 0.05, 0.5, 0.2, 0.7]
    best = max(itertools.permutations(q), key=expected_reuse)
    assert list(best) == sorted(q)


def test_unchanged_vehicle_keeps_its_line_and_handle():
    tm = TableManager(enc)
    v = [Vehicle(7, "a", 100, 200, 0.0, 0.0), Vehicle(9, "b", 50, 60, 12.0, 1.0)]
    ids1, h1 = tm.update(v)
    v[1].x += 40                                         # only vehicle 9 moves
    ids2, h2 = tm.update(v)
    assert h1 == h2
    n_head = len(tm.head_ids)
    slot = len(enc(entry_text(0, v[0], Quant())))
    assert ids1[: n_head + slot] == ids2[: n_head + slot] and ids1 != ids2


def test_stability_ordering_moves_stable_vehicles_forward():
    rng = random.Random(0)
    tm = TableManager(enc, ordering="stability", resort_every=5)
    vs = [Vehicle(i, str(i), 10 * i, 0, 10.0 if i < 5 else 0.0) for i in range(10)]
    for _ in range(40):
        for v in vs[:5]:
            v.x += rng.choice([20, 30])                  # first five keep moving
        tm.update(vs)
    assert set(tm.order[:5]) == {5, 6, 7, 8, 9} and tm.resorts >= 1


def test_labels_are_single_tokens():
    assert len(set(LabelSet(enc).ids.values())) == 10


def test_cost_model_trades_load_against_recompute():
    slow_bus = CostModel(CostParams(b_cg=1e6, b_gc=1e12))
    fast_bus = CostModel(CostParams(b_cg=1e12, b_gc=1e12))
    m_slow, _ = slow_bus.choose_m(4000, 4000, 0, 4, 60)
    m_fast, _ = fast_bus.choose_m(4000, 4000, 0, 4, 60)
    assert m_fast == 4000 and 0 <= m_slow < 4000


def test_admission_keeps_largest_slack_and_respects_deadlines():
    cm = CostModel(CostParams())
    t1 = cm.predict(2000, 2000, 2000, 1, 60)
    t3 = cm.predict(2000, 2000, 2000, 3, 60)
    idx, _, t = cm.admit([t3 * 2, t1 * 0.5, t3 * 3, t3 * 1.5], 2000, 2000, 2000, 60)
    assert sorted(idx) == [0, 2, 3] and t <= t3 * 1.5
    assert cm.admit([0.0, -1.0], 2000, 2000, 2000, 60)[0] == []

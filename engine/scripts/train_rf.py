#!/usr/bin/env python3
"""Train the Random-Forest baseline on the same data as the LLM.

    python engine/scripts/train_rf.py --data data/train.jsonl --out models/rf.joblib

One row per (request, option) with the features of ``kvrsu.decide.FEATURES``
and the label "this option is the hindsight-best one".  At run time the
server scores every option of a request and picks the most probable one.
The forest (100 trees, depth 10) follows the configuration of the authors'
TinyML paper (IEEE OJ-COMS 2026); its credibility module is not reproduced.
"""
import argparse
import json

import _common  # noqa: F401
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--trees", type=int, default=100)
    ap.add_argument("--depth", type=int, default=10)
    ap.add_argument("--holdout", type=float, default=0.1)
    a = ap.parse_args()

    with open(a.data) as f:
        samples = [json.loads(line) for line in f]
    cut = int(len(samples) * (1 - a.holdout))
    rows = lambda part: ([o["feat"] for s in part for o in s["options"]], [o["y"] for s in part for o in s["options"]])
    X, y = rows(samples[:cut])
    rf = RandomForestClassifier(n_estimators=a.trees, max_depth=a.depth, n_jobs=-1, random_state=0,
                                class_weight="balanced").fit(np.array(X), np.array(y))
    hits = 0
    for s in samples[cut:]:
        proba = rf.predict_proba(np.array([o["feat"] for o in s["options"]]))[:, 1]
        hits += s["options"][int(proba.argmax())]["y"]
    joblib.dump(rf, a.out)
    print(f"trained on {cut} requests; held-out top-1 accuracy {hits / max(1, len(samples) - cut):.3f}; saved {a.out}")


if __name__ == "__main__":
    main()

"""
Fish-level K-fold assignment for the cross-validation check (MODE = "extra" in run_all.py).

  python make_folds.py --cache data_cache --k 5

Every group (one MatsyaDx fish, or one near-duplicate group of BD-Freshwater-Fish photos) goes to exactly
one fold. Groups are dealt round-robin within each stratum (MatsyaDx: health label; BD-Freshwater-Fish:
species), so each fold gets about 1/K of every class, including 2 of the 10 healthy fish.
For fold k: test = fold k, validation = fold k+1, training = the other K-2 folds (60/20/20 for K = 5).
Writes <cache>/folds.json.
"""
import argparse, csv, json, os
from collections import defaultdict

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="data_cache")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    rows = list(csv.DictReader(open(os.path.join(a.cache, "manifest.csv"), encoding="utf-8")))
    vocab = json.load(open(os.path.join(a.cache, "vocab.json"), encoding="utf-8"))
    stratum_of = {}
    for r in rows:
        if r["group"] not in stratum_of:
            stratum_of[r["group"]] = ("matsya", int(r["health"])) if r["source"] == "matsya" else ("bdff", int(r["species"]))
    by_stratum = defaultdict(list)
    for g, s in sorted(stratum_of.items()):
        by_stratum[s].append(g)

    rng = np.random.default_rng(a.seed)
    fold_of, start = {}, 0
    for s in sorted(by_stratum):
        gs = by_stratum[s]
        rng.shuffle(gs)
        for i, g in enumerate(gs):
            fold_of[g] = (start + i) % a.k
        start = (start + len(gs)) % a.k            # rotate so small strata do not all start in fold 0

    json.dump({"k": a.k, "seed": a.seed, "fold_of_group": fold_of},
              open(os.path.join(a.cache, "folds.json"), "w", encoding="utf-8"))
    healthy = vocab["health"].index("Healthy")
    for q in range(a.k):
        imgs = sum(1 for r in rows if fold_of[r["group"]] == q)
        fish = {h: len({g for g, s in stratum_of.items() if s == ("matsya", h) and fold_of[g] == q})
                for h in range(len(vocab["health"]))}
        print(f"fold {q}: {imgs} images | MatsyaDx fish per health class "
              + ", ".join(f"{vocab['health'][h]} {n}" for h, n in fish.items())
              + ("" if fish[healthy] else "  <-- no healthy fish"))


if __name__ == "__main__":
    main()

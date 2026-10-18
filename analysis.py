"""Aggregate results -> tables with proper statistics.

  python analysis.py --results results

Unit of analysis for tests = CLIENT (subject): accuracies are first averaged over
seeds/folds per client, then compared across clients (paired Wilcoxon + Holm +
bootstrap CI). Never treat seeds as independent samples.
"""
import argparse
import glob
import itertools
import json
import os

import numpy as np
import pandas as pd

from metrics import chance_cosine, cosine, pearson
from stats_utils import bootstrap_ci, holm, paired_test

COMPARISONS = [("fedavg", "solo"), ("fedper", "solo"), ("fedbn", "solo"),
               ("fedavg_ft", "solo"), ("fedper", "fedavg"), ("fedbn", "fedavg"),
               ("fedavg_ft", "fedavg"), ("central", "fedavg")]


def load_runs(res_dir):
    runs = []
    for d in sorted(glob.glob(os.path.join(res_dir, "seed*_fold*"))):
        mp, ap = os.path.join(d, "metrics.json"), os.path.join(d, "arrays.npz")
        if not (os.path.exists(mp) and os.path.exists(ap)):
            continue
        base = os.path.basename(d)
        seed, fold = int(base.split("_")[0][4:]), int(base.split("_")[1][4:])
        runs.append(dict(seed=seed, fold=fold, m=json.load(open(mp)),
                         arr=dict(np.load(ap))))
    return runs


def accuracy_tables(runs, out):
    rows = [dict(seed=r["seed"], fold=r["fold"], method=m, client=c, acc=a)
            for r in runs for m, d in r["m"]["acc"].items() for c, a in d.items()]
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, "accuracy_long.csv"), index=False)
    per_client = df.groupby(["client", "method"]).acc.mean().unstack("method")
    per_client.to_csv(os.path.join(out, "accuracy_per_client.csv"))

    summ = pd.DataFrame({"mean_%": per_client.mean() * 100,
                         "std_across_clients_%": per_client.std() * 100,
                         "n_clients": per_client.count()})
    summ.to_csv(os.path.join(out, "accuracy_summary.csv"))

    tests = []
    for a, b in COMPARISONS:
        if a in per_client and b in per_client:
            x, y = per_client[a].values, per_client[b].values
            ci = bootstrap_ci(x - y)
            tests.append(dict(A=a, B=b, mean_diff_pp=100 * (x - y).mean(),
                              ci_low_pp=100 * ci[0], ci_high_pp=100 * ci[1],
                              p_raw=paired_test(x, y), n_clients=len(x)))
    t = pd.DataFrame(tests)
    if len(t):
        t["p_holm"] = holm(t.p_raw.values)
    t.to_csv(os.path.join(out, "accuracy_tests.csv"), index=False)
    return summ, t


def importance_similarity(runs, out):
    methods = sorted({m for r in runs for m in r["m"]["acc"]})
    clients = sorted(runs[0]["m"]["acc"][methods[0]])
    rows = []
    for r in runs:
        for c in clients:
            for a, b in itertools.combinations(methods, 2):
                ia, ib = r["arr"][f"imp|{a}|{c}"], r["arr"][f"imp|{b}|{c}"]
                rows.append(dict(client=c, A=a, B=b, cosine=cosine(ia, ib), pearson=pearson(ia, ib)))
    df = pd.DataFrame(rows)
    per_client = df.groupby(["A", "B", "client"])[["cosine", "pearson"]].mean().reset_index()
    tab = per_client.groupby(["A", "B"])[["cosine", "pearson"]].agg(["mean", "std"])
    tab.to_csv(os.path.join(out, "backbone_similarity.csv"))

    # null A: same method, same client, same fold, different seeds
    null_rows = []
    by = {}
    for r in runs:
        by.setdefault(r["fold"], []).append(r)
    for fold, rs in by.items():
        for r1, r2 in itertools.combinations(rs, 2):
            for m in methods:
                for c in clients:
                    ia, ib = r1["arr"][f"imp|{m}|{c}"], r2["arr"][f"imp|{m}|{c}"]
                    null_rows.append(dict(method=m, cosine=cosine(ia, ib), pearson=pearson(ia, ib)))
    n_ch = len(runs[0]["m"]["channels"])
    chance = chance_cosine(n_ch)
    null = pd.DataFrame(null_rows)
    if len(null):
        null.groupby("method")[["cosine", "pearson"]].agg(["mean", "std"]).to_csv(
            os.path.join(out, "backbone_null_across_seeds.csv"))
    return tab, chance


def head_similarity(runs, out):
    methods = sorted({m for r in runs for m in r["m"]["acc"]})
    clients = sorted(runs[0]["m"]["acc"][methods[0]])
    pairs = list(itertools.combinations(clients, 2))
    rng = np.random.RandomState(0)
    if len(pairs) > 200:
        pairs = [pairs[i] for i in rng.choice(len(pairs), 200, replace=False)]
    rows = []
    for r in runs:
        for m in methods:
            sims = [cosine(r["arr"][f"head|{m}|{a}"], r["arr"][f"head|{m}|{b}"]) for a, b in pairs]
            rows.append(dict(seed=r["seed"], fold=r["fold"], method=m, mean_pair_cosine=np.mean(sims)))
    df = pd.DataFrame(rows)
    tab = df.groupby("method").mean_pair_cosine.agg(["mean", "std"])
    tab.to_csv(os.path.join(out, "head_similarity.csv"))
    return tab


def cka_table(runs, out):
    rows = [dict(pair=p, client=c, cka=v) for r in runs for p, d in r["m"]["cka"].items() for c, v in d.items()]
    df = pd.DataFrame(rows)
    per_client = df.groupby(["pair", "client"]).cka.mean().reset_index()
    tab = per_client.groupby("pair").cka.agg(["mean", "std"])
    tab.to_csv(os.path.join(out, "cka_between_methods.csv"))
    return tab


def main():
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 50)
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    res = ap.parse_args().results
    out = os.path.join(res, "analysis")
    os.makedirs(out, exist_ok=True)
    runs = load_runs(res)
    assert runs, f"no finished runs in {res}"
    print(f"{len(runs)} runs loaded\n")

    summ, tests = accuracy_tables(runs, out)
    print("ACCURACY (client-level)\n", summ.round(2), "\n")
    print("PAIRED TESTS (pp = percentage points)\n", tests.round(4), "\n")
    sim, chance = importance_similarity(runs, out)
    print(f"BACKBONE SPATIAL-FILTER SIMILARITY\n(chance-level cosine of random min-max vectors = {chance:.3f})\n",
          sim.round(3), "\n")
    print("CROSS-CLIENT HEAD COSINE\n", head_similarity(runs, out).round(3), "\n")
    print("CKA BETWEEN METHODS\n", cka_table(runs, out).round(3))
    print(f"\nCSV files written to {out}")


if __name__ == "__main__":
    main()

"""Results separately for BCI IV 2a clients and Lee2019 clients (no retraining needed).

  python per_dataset.py --results results_main
Reads  results_main/analysis/accuracy_per_client.csv  (written by analysis.py).
"""
import argparse
import os

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

from stats_utils import bootstrap_ci, holm, paired_test

COMPARISONS = [("fedavg", "solo"), ("fedper", "solo"), ("fedbn", "solo"), ("fedavg_ft", "solo"),
               ("fedper", "fedavg"), ("fedbn", "fedavg"), ("fedavg_ft", "fedavg"), ("central", "fedavg")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results_main")
    res = ap.parse_args().results
    pc = pd.read_csv(os.path.join(res, "analysis", "accuracy_per_client.csv"), index_col=0) * 100
    out = os.path.join(res, "analysis")
    groups = {"BCI IV 2a": [c for c in pc.index if str(c).startswith("2a")],
              "Lee2019": [c for c in pc.index if str(c).startswith("lee")]}
    summary_rows, test_rows = [], []
    for g, idx in groups.items():
        sub = pc.loc[idx]
        for m in sub.columns:
            if m.endswith("_rep"):
                continue
            summary_rows.append(dict(dataset=g, method=m, n_clients=len(idx),
                                     mean=sub[m].mean(), sd=sub[m].std()))
        rows = []
        for a, b in COMPARISONS:
            if a in sub and b in sub:
                x, y = sub[a].values, sub[b].values
                lo, hi = bootstrap_ci(x - y)
                rows.append(dict(dataset=g, A=a, B=b, n_clients=len(idx), mean_diff_pp=(x - y).mean(),
                                 ci_low=lo, ci_high=hi, p_raw=paired_test(x, y)))
        p_adj = holm([r["p_raw"] for r in rows])
        for r, p in zip(rows, p_adj):
            r["p_holm"] = p
        test_rows += rows
    S = pd.DataFrame(summary_rows)
    T = pd.DataFrame(test_rows)
    S.to_csv(os.path.join(out, "per_dataset_summary.csv"), index=False)
    T.to_csv(os.path.join(out, "per_dataset_tests.csv"), index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 50)
    print("MEAN ACCURACY PER DATASET (%)\n", S.pivot(index="method", columns="dataset", values="mean").round(2))
    print("\nSD ACROSS CLIENTS\n", S.pivot(index="method", columns="dataset", values="sd").round(2))
    print("\nPAIRED TESTS PER DATASET (Holm within dataset, 8 comparisons)\n", T.round(4).to_string(index=False))

    # does the federation penalty differ between datasets?
    d = {g: (pc.loc[idx, "fedavg"] - pc.loc[idx, "solo"]).values for g, idx in groups.items()}
    u = mannwhitneyu(d["BCI IV 2a"], d["Lee2019"])
    print(f"\nFedAvg minus Solo, 2a mean {d['BCI IV 2a'].mean():.2f} pp vs Lee2019 mean {d['Lee2019'].mean():.2f} pp; "
          f"Mann-Whitney p = {u.pvalue:.3f}")


if __name__ == "__main__":
    main()

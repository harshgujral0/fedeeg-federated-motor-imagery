"""Figures and tables for the paper, built from the CSVs that analysis.py and
deep_analysis.py already wrote (no retraining needed).

  python make_figures.py --main results_main --evallocal results_main_evallocal \
                         --crypto crypto_benchmark.csv --out figures
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LABELS = {"solo": "Solo", "fedavg": "FedAvg", "fedper": "FedPer", "fedbn": "FedBN",
          "fedavg_ft": "FedAvg+FT", "central": "Central (pooled)",
          "fedper_local": "FedPer (local eval)", "fedbn_local": "FedBN (local eval)"}
ORDER = ["solo", "fedavg", "fedper", "fedbn", "fedper_local", "fedbn_local",
         "fedavg_ft", "central"]


def boot_ci(x, n=10000, seed=0):
    rng = np.random.RandomState(seed)
    x = np.asarray(x, float)
    m = np.array([rng.choice(x, len(x)).mean() for _ in range(n)])
    return np.quantile(m, 0.025), np.quantile(m, 0.975)


def save(fig, out, name):
    fig.tight_layout()
    fig.savefig(os.path.join(out, name + ".png"), dpi=200)
    fig.savefig(os.path.join(out, name + ".pdf"))
    plt.close(fig)
    print("wrote", name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--main", default="results_main")
    ap.add_argument("--evallocal", default="results_main_evallocal")
    ap.add_argument("--crypto", default="crypto_benchmark.csv")
    ap.add_argument("--out", default="figures")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    an = os.path.join(a.main, "analysis")

    # ---------------- accuracy table (per client) ----------------------------
    pc = pd.read_csv(os.path.join(an, "accuracy_per_client.csv"), index_col=0) * 100
    ev_path = os.path.join(a.evallocal, "analysis", "accuracy_per_client.csv")
    if os.path.exists(ev_path):
        ev = pd.read_csv(ev_path, index_col=0) * 100
        for m in ("fedper", "fedbn"):
            if m in ev:
                pc[m + "_local"] = ev[m].reindex(pc.index)
    cols = [m for m in ORDER if m in pc]

    # Fig 1: mean accuracy with 95% bootstrap CI over clients
    fig, ax = plt.subplots(figsize=(8, 4))
    means = [pc[m].mean() for m in cols]
    cis = [boot_ci(pc[m].dropna().values) for m in cols]
    err = np.array([[mu - lo, hi - mu] for mu, (lo, hi) in zip(means, cis)]).T
    ax.bar(range(len(cols)), means, yerr=err, capsize=4, color="#4c78a8")
    ax.axhline(50, ls=":", color="gray")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([LABELS[m] for m in cols], rotation=30, ha="right")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_ylim(40, 90)
    ax.set_title(f"Accuracy across {len(pc)} clients (mean, 95% bootstrap CI)")
    save(fig, a.out, "fig1_accuracy")
    pc[cols].describe().T.round(2).to_csv(os.path.join(a.out, "table_accuracy.csv"))

    # Fig 2: per-client scatter vs Solo
    fig, axes = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)
    is2a = np.array([str(c).startswith("2a") for c in pc.index])
    for ax, m in zip(axes, ["fedavg", "fedavg_ft"]):
        if m not in pc:
            continue
        ax.scatter(pc["solo"][is2a], pc[m][is2a], label="BCI IV 2a", c="#e45756")
        ax.scatter(pc["solo"][~is2a], pc[m][~is2a], label="Lee2019", c="#4c78a8")
        lo, hi = 40, 100
        ax.plot([lo, hi], [lo, hi], "k--", lw=1)
        ax.set_xlabel("Solo accuracy (%)")
        ax.set_ylabel(f"{LABELS[m]} accuracy (%)")
        ax.set_title(f"{LABELS[m]} vs Solo (per client)")
    axes[0].legend()
    save(fig, a.out, "fig2_scatter_vs_solo")

    # ---------------- interpretability figures -------------------------------
    def bar_from_csv(path, value_col, title, ylabel, name, exclude=("fedavg_rep", "solo_rep")):
        if not os.path.exists(path):
            print("missing", path)
            return
        df = pd.read_csv(path, header=[0, 1], index_col=0) if value_col is None else None
        return df

    p = os.path.join(an, "deep_decision_agreement.csv")
    if os.path.exists(p):
        d = pd.read_csv(p, index_col=0)
        d = d.drop([i for i in d.index if i.endswith("_rep")], errors="ignore")
        idx = [m for m in ["solo", "fedper", "fedavg_ft", "fedbn", "fedavg", "central"] if m in d.index]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(range(len(idx)), d.loc[idx, "mean"], yerr=d.loc[idx, "std"], capsize=4, color="#72b7b2")
        ax.axhline(0.5, ls=":", color="gray")
        ax.set_xticks(range(len(idx)))
        ax.set_xticklabels([LABELS[m] for m in idx], rotation=30, ha="right")
        ax.set_ylabel("Pairwise decision agreement between clients")
        ax.set_ylim(0.4, 1.05)
        ax.set_title("Do different clients' models decide alike?")
        save(fig, a.out, "fig3_decision_agreement")

    p = os.path.join(an, "deep_init_control.csv")
    if os.path.exists(p):
        d = pd.read_csv(p, header=[0, 1], index_col=0)
        d = d.drop([i for i in d.index if i.endswith("_rep")], errors="ignore")
        idx = [m for m in ["solo", "fedavg", "fedper", "fedbn", "fedavg_ft", "central"] if m in d.index]
        mu = d[("pearson_vs_init", "mean")].loc[idx]
        sd = d[("pearson_vs_init", "std")].loc[idx]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(range(len(idx)), mu, yerr=sd, capsize=4, color="#f58518")
        ax.set_xticks(range(len(idx)))
        ax.set_xticklabels([LABELS[m] for m in idx], rotation=30, ha="right")
        ax.set_ylabel("Correlation of channel importance with\nthe model's OWN random initialisation")
        ax.set_ylim(-0.1, 1.1)
        ax.set_title("Weight-based importance mostly reflects initialisation")
        save(fig, a.out, "fig4_init_control")

    # ---------------- encryption tables ---------------------------------------
    if os.path.exists(a.crypto):
        c = pd.read_csv(a.crypto)
        est = c[c.kind == "estimated_time"]
        est5 = est[est.n_clients == 5][["key_bits", "scope", "n_params", "client_encrypt_s",
                                        "server_add_s", "decrypt_s", "upload_MB_per_client"]]
        est5.round(3).to_csv(os.path.join(a.out, "table_crypto_cost_5clients.csv"), index=False)
        err = c[c.kind == "measured_error"][["fp_bits", "max_abs_err"]]
        err.to_csv(os.path.join(a.out, "table_crypto_error.csv"), index=False)
        print("\nCrypto cost (5 clients):\n", est5.round(3).to_string(index=False))
        print("\nFixed-point error:\n", err.to_string(index=False))

    # ---------------- accuracy tests (copied for the paper) -------------------
    t = os.path.join(an, "accuracy_tests.csv")
    if os.path.exists(t):
        pd.read_csv(t).round(4).to_csv(os.path.join(a.out, "table_accuracy_tests.csv"), index=False)
        print("wrote table_accuracy_tests.csv")


if __name__ == "__main__":
    main()

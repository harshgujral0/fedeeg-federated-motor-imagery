"""Controlled representation analysis (needs runs made with the updated runner,
which saves the trained models).

  python deep_analysis.py --results results_main

Fixes the weak measures of the first analysis:
 1. init control  : similarity of each trained model's channel importance to ITS OWN
                    initialisation (is the "importance" mostly init noise?).
 2. spatial patterns (Haufe transform) instead of raw filter weights, and a
    motor-cortex ROI score (C3/C4/Cz neighbourhood).
 3. pattern reproducibility: same method, different init (_rep runs).
 4. cross-client pattern consistency per method.
 5. decision agreement between clients' models on a pooled probe set
    (init-free replacement for 'head cosine').
"""
import argparse
import glob
import itertools
import json
import os

import numpy as np
import pandas as pd
import torch

from config import Config
from data import build_clients
from federated import new_model
from metrics import cosine, pearson, spatial_importance
from model import EEGNet
from run_experiments import prepare_fold
from stats_utils import bootstrap_ci, holm, paired_test

MOTOR = ["C3", "C1", "Cz", "C2", "C4", "FC3", "FC4", "CP3", "CP4", "C5", "C6"]


def load_cfg(mjson):
    cfg = Config()
    for k, v in mjson["config"].items():
        if hasattr(cfg, k):
            setattr(cfg, k, tuple(v) if isinstance(v, list) else v)
    cfg.device = "cpu"
    return cfg


def load_model(arr, method, client, n_chans):
    prefix = f"state|{method}|{client}|"
    sd = {k[len(prefix):]: torch.from_numpy(arr[k]) for k in arr if k.startswith(prefix)}
    m = EEGNet(n_chans)
    m.load_state_dict(sd)
    return m.eval()


def init_importance(method, i, seed, n_chans):
    base = method[:-4] if method.endswith("_rep") else method
    off = 777 if method.endswith("_rep") else 0
    s = seed * 1000 + i + off if base == "solo" else seed + off
    return spatial_importance(new_model(n_chans, s))


@torch.no_grad()
def haufe_pattern(model, X):
    """Mean |activation pattern| per channel (Haufe et al. 2014) for the spatial
    filters, computed on the temporal-conv output of this client's own test data."""
    model.eval()
    z = model.bn1(model.temporal(X.unsqueeze(1)))             # (B, F1, C, T)
    F1, C = z.shape[1], z.shape[2]
    W = model.spatial.weight[:, 0, :, 0]                      # (F1*D, C)
    D = W.shape[0] // F1
    pats = []
    for g in range(F1):
        zg = z[:, g].permute(1, 0, 2).reshape(C, -1)
        zg = zg - zg.mean(1, keepdim=True)
        cov = zg @ zg.T / zg.shape[1]
        for d in range(D):
            w = W[g * D + d]
            a = cov @ w
            a = a / (w @ a + 1e-12)
            pats.append(a.abs())
    return torch.stack(pats).mean(0).numpy()


@torch.no_grad()
def predict(model, X):
    model.eval()
    return torch.cat([model(X[i:i + 256]).argmax(1) for i in range(0, X.shape[0], 256)]).numpy()


def roi_ratio(p, idx):
    return float(p[idx].mean() / (p.mean() + 1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    res = ap.parse_args().results
    out = os.path.join(res, "analysis")
    os.makedirs(out, exist_ok=True)

    dirs = sorted(glob.glob(os.path.join(res, "seed*_fold*")))
    first = json.load(open(os.path.join(dirs[0], "metrics.json")))
    cfg = load_cfg(first)
    clients, channels = build_clients(cfg)
    roi_idx = [i for i, c in enumerate(channels) if c in MOTOR]
    print(f"motor ROI channels: {[channels[i] for i in roi_idx]}")
    n_ch = len(channels)

    R_init, R_roi, R_agree, R_cons, R_rep = [], [], [], [], []
    for d in dirs:
        mp, ap_ = os.path.join(d, "metrics.json"), os.path.join(d, "arrays.npz")
        if not (os.path.exists(mp) and os.path.exists(ap_)):
            continue
        base = os.path.basename(d)
        seed, fold = int(base.split("_")[0][4:]), int(base.split("_")[1][4:])
        arr = dict(np.load(ap_))
        if not any(k.startswith("state|") for k in arr):
            print(f"[skip] {d}: no saved models (made with the old runner)")
            continue
        meta = json.load(open(mp))
        methods = list(meta["acc"].keys())
        names = list(meta["acc"][methods[0]].keys())
        data = prepare_fold(clients, cfg, seed, fold)
        by_name = {c["name"]: c for c in data}
        probe = torch.cat([by_name[n]["Xte"] for n in names])

        pats = {}
        for m in methods:
            models = {n: load_model(arr, m, n, n_ch) for n in names}
            pats[m] = {}
            preds = {}
            for i, n in enumerate(names):
                w_imp = spatial_importance(models[n])
                i_imp = init_importance(m, i, seed, n_ch)
                R_init.append(dict(seed=seed, fold=fold, method=m, client=n,
                                   pearson_vs_init=pearson(w_imp, i_imp),
                                   cosine_vs_init=cosine(w_imp, i_imp)))
                p = haufe_pattern(models[n], by_name[n]["Xte"])
                pats[m][n] = p
                R_roi.append(dict(seed=seed, fold=fold, method=m, client=n,
                                  roi_pattern=roi_ratio(p, roi_idx),
                                  roi_weight=roi_ratio(w_imp, roi_idx)))
                preds[n] = predict(models[n], probe)
            ag = [float((preds[a] == preds[b]).mean()) for a, b in itertools.combinations(names, 2)]
            R_agree.append(dict(seed=seed, fold=fold, method=m, agreement=float(np.mean(ag))))
            cs = [pearson(pats[m][a], pats[m][b]) for a, b in itertools.combinations(names, 2)]
            R_cons.append(dict(seed=seed, fold=fold, method=m, cross_client_pattern_pearson=float(np.mean(cs))))
        for m in methods:
            if m + "_rep" in methods:
                for n in names:
                    R_rep.append(dict(seed=seed, fold=fold, method=m, client=n,
                                      pattern_pearson_vs_reinit=pearson(pats[m][n], pats[m + "_rep"][n])))
        print(f"[done] {d}")

    assert R_init, "no runs with saved models found; rerun with the updated run_experiments.py"
    tabs = {}
    df = pd.DataFrame(R_init)
    tabs["init_control"] = df.groupby("method")[["pearson_vs_init", "cosine_vs_init"]].agg(["mean", "std"])
    roi = pd.DataFrame(R_roi)
    per_client = roi.groupby(["client", "method"])[["roi_pattern", "roi_weight"]].mean().reset_index()
    tabs["roi_score"] = per_client.groupby("method")[["roi_pattern", "roi_weight"]].agg(["mean", "std"])
    tabs["decision_agreement"] = pd.DataFrame(R_agree).groupby("method").agreement.agg(["mean", "std"])
    tabs["pattern_consistency_across_clients"] = pd.DataFrame(R_cons).groupby("method").cross_client_pattern_pearson.agg(["mean", "std"])
    if R_rep:
        rep = pd.DataFrame(R_rep).groupby(["client", "method"]).pattern_pearson_vs_reinit.mean().reset_index()
        tabs["pattern_reproducibility_vs_reinit"] = rep.groupby("method").pattern_pearson_vs_reinit.agg(["mean", "std"])

    # paired tests on motor-ROI pattern score (client = unit)
    piv = per_client.pivot(index="client", columns="method", values="roi_pattern")
    tests = []
    for a in [m for m in piv.columns if not m.endswith("_rep") and m != "solo"]:
        if "solo" in piv:
            diff = piv[a].values - piv["solo"].values
            lo, hi = bootstrap_ci(diff)
            tests.append(dict(A=a, B="solo", mean_diff=diff.mean(), ci_low=lo, ci_high=hi,
                              p_raw=paired_test(piv[a].values, piv["solo"].values)))
    t = pd.DataFrame(tests)
    if len(t):
        t["p_holm"] = holm(t.p_raw.values)
        tabs["roi_pattern_tests_vs_solo"] = t
    for name, tab in tabs.items():
        tab.to_csv(os.path.join(out, f"deep_{name}.csv"))
        print(f"\n== {name} ==\n{tab.round(3)}")


if __name__ == "__main__":
    main()

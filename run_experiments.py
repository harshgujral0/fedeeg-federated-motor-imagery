"""Run the expanded FedEEG-Interpret experiments.

  python run_experiments.py --synthetic --seeds 42 --run_folds 0      # smoke test
  python run_experiments.py --n_lee 5 --seeds 42 --run_folds 0        # small real run
  python run_experiments.py                                           # full run
  python run_experiments.py --mode loso --max_holdouts 5              # unseen subjects

Resumable: finished (seed, fold) runs are skipped.
"""
import argparse
import itertools
import json
import os
import time
from dataclasses import fields

import numpy as np
import torch

from config import Config
from data import build_clients, fold_split, standardize
from federated import (extract_features, finetune, predict_acc, train_federated,
                       train_method)
from metrics import linear_cka, spatial_importance


def parse_args():
    cfg = Config()
    p = argparse.ArgumentParser()
    for f in fields(Config):
        v = getattr(cfg, f.name)
        if isinstance(v, bool):
            p.add_argument(f"--{f.name}", action="store_true", default=v)
        elif isinstance(v, tuple):
            typ = type(v[0]) if v else int
            if f.name in ("methods", "datasets"):
                typ = str
            p.add_argument(f"--{f.name}", nargs="*", type=typ, default=list(v))
        else:
            p.add_argument(f"--{f.name}", type=type(v), default=v)
    a = vars(p.parse_args())
    for k, v in a.items():
        if isinstance(getattr(cfg, k), tuple):
            v = tuple(v)
        setattr(cfg, k, v)
    return cfg


def prepare_fold(clients, cfg, seed, fold):
    dev = cfg.resolve_device()
    data = []
    for c in clients:
        tr, te = fold_split(c.y, cfg.n_folds, seed, fold)
        Xtr, Xte = standardize(c.X[tr], c.X[te])
        data.append(dict(name=c.name,
                         Xtr=torch.tensor(Xtr, device=dev), ytr=torch.tensor(c.y[tr], device=dev),
                         Xte=torch.tensor(Xte, device=dev), yte=torch.tensor(c.y[te], device=dev)))
    return data


def run_within(cfg):
    clients, channels = build_clients(cfg)
    methods = list(cfg.methods)
    if cfg.with_null:
        methods += ["solo_rep", "fedavg_rep"]
    folds = list(cfg.run_folds) if cfg.run_folds else list(range(cfg.n_folds))
    os.makedirs(cfg.out_dir, exist_ok=True)

    for seed, fold in itertools.product(cfg.seeds, folds):
        run_dir = os.path.join(cfg.out_dir, f"seed{seed}_fold{fold}")
        if os.path.exists(os.path.join(run_dir, "metrics.json")):
            print(f"[skip] {run_dir}")
            continue
        os.makedirs(run_dir, exist_ok=True)
        t_run = time.time()
        data = prepare_fold(clients, cfg, seed, fold)
        names = [c["name"] for c in data]

        trained, info, cache = {}, {}, {}
        for m in methods:
            t0 = time.time()
            if m == "fedavg_ft":
                if "fedavg" not in cache:
                    cache["fedavg"], _ = train_method("fedavg", data, cfg, seed)
                trained[m] = finetune(cache["fedavg"], data, cfg, seed)
            else:
                trained[m], info[m] = train_method(m, data, cfg, seed)
                if m == "fedavg":
                    cache["fedavg"] = trained[m]
            print(f"  seed {seed} fold {fold} {m:15s} {time.time() - t0:6.1f}s")

        # ---- per-method outputs -------------------------------------------
        acc, arrays = {}, {}
        for m, models in trained.items():
            acc[m] = {}
            for c, mod in zip(data, models):
                acc[m][c["name"]] = predict_acc(mod, c["Xte"], c["yte"])
                arrays[f"imp|{m}|{c['name']}"] = spatial_importance(mod)
                arrays[f"head|{m}|{c['name']}"] = mod.classifier.weight.detach().cpu().numpy().ravel()
                for k, v in mod.state_dict().items():      # full model, for deep_analysis.py
                    arrays[f"state|{m}|{c['name']}|{k}"] = v.detach().cpu().numpy()

        # ---- CKA between methods, on a pooled probe set -----------------------
        probe = torch.cat([c["Xte"][:cfg.probe_size] for c in data])
        feats = {m: [extract_features(mod, probe) for mod in models]
                 for m, models in trained.items()}
        cka = {}
        for a, b in itertools.combinations(list(trained), 2):
            cka[f"{a}|{b}"] = {n: linear_cka(feats[a][i], feats[b][i])
                               for i, n in enumerate(names)}

        crypto = {m: v.get("crypto_log") for m, v in info.items() if v.get("crypto_log")}
        json.dump(dict(acc=acc, cka=cka, crypto=crypto, channels=channels,
                       probe_n=int(probe.shape[0]), config=cfg.__dict__),
                  open(os.path.join(run_dir, "metrics.json"), "w"), default=str)
        np.savez_compressed(os.path.join(run_dir, "arrays.npz"), **arrays)
        print(f"[done] {run_dir} in {time.time() - t_run:.0f}s")


def run_loso(cfg):
    """Leave-one-subject-out: federation trained on all OTHER clients; evaluate the
    global model on the held-out client. Target data are z-scored with their own
    (label-free) statistics."""
    clients, _ = build_clients(cfg)
    dev = cfg.resolve_device()
    holdouts = clients[:cfg.max_holdouts] if cfg.max_holdouts else clients
    os.makedirs(os.path.join(cfg.out_dir, "loso"), exist_ok=True)
    for seed in cfg.seeds:
        path = os.path.join(cfg.out_dir, "loso", f"seed{seed}.json")
        res = json.load(open(path)) if os.path.exists(path) else {}
        for h in holdouts:
            if h.name in res:
                continue
            train = []
            for c in clients:
                if c.name == h.name:
                    continue
                Xs, _ = standardize(c.X, c.X)
                train.append(dict(name=c.name, Xtr=torch.tensor(Xs, device=dev),
                                  ytr=torch.tensor(c.y, device=dev)))
            Xh, _ = standardize(h.X, h.X)
            Xh, yh = torch.tensor(Xh, device=dev), torch.tensor(h.y, device=dev)
            res[h.name] = {}
            for m in ("fedavg", "central"):
                models, _ = train_method(m, train, cfg, seed)
                res[h.name][m] = predict_acc(models[0], Xh, yh)
            print(f"[loso] seed {seed} {h.name}: {res[h.name]}")
            json.dump(res, open(path, "w"))


if __name__ == "__main__":
    cfg = parse_args()
    print(cfg)
    (run_loso if cfg.mode == "loso" else run_within)(cfg)

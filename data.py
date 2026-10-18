"""Data loading: MOABB -> per-subject clients on a COMMON channel set.

Each subject is one federated client. Both datasets are preprocessed like the
paper (4-38 Hz, 128 Hz, 0-4 s after the cue, microvolts). Instead of
zero-padding to 22 channels we use the channels the datasets share.
"""
import hashlib
import json
import os

import numpy as np
from sklearn.model_selection import StratifiedKFold

TARGET_T = 512  # 4 s at 128 Hz


class Client:
    def __init__(self, name, X, y):
        self.name, self.X, self.y = name, X.astype(np.float32), y.astype(np.int64)


# --------------------------------------------------------------------------
# MOABB access
# --------------------------------------------------------------------------
def _moabb_epochs(key, subject):
    from moabb.paradigms import LeftRightImagery
    if key == "2a":
        from moabb.datasets import BNCI2014_001
        ds = BNCI2014_001()
    elif key == "lee":
        from moabb.datasets import Lee2019_MI
        ds = Lee2019_MI()
    else:
        raise ValueError(f"unknown dataset key {key}")
    paradigm = LeftRightImagery(fmin=4.0, fmax=38.0, tmin=0.0, tmax=4.0,
                                resample=128.0)
    epochs, labels, _ = paradigm.get_data(ds, subjects=[subject],
                                          return_epochs=True)
    return epochs, np.asarray(labels)


def dataset_channels(key, cache_dir):
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"channels_{key}.json")
    if os.path.exists(path):
        return json.load(open(path))
    epochs, _ = _moabb_epochs(key, 1)
    names = list(epochs.ch_names)
    json.dump(names, open(path, "w"))
    return names


def common_channels(keys, cache_dir):
    lists = [dataset_channels(k, cache_dir) for k in keys]
    common = set.intersection(*[{n.upper() for n in l} for l in lists])
    return [n for n in lists[0] if n.upper() in common]


def load_subject(key, subject, channels, cache_dir):
    tag = hashlib.md5(",".join(channels).encode()).hexdigest()[:8]
    path = os.path.join(cache_dir, f"{key}_s{subject}_{tag}.npz")
    if os.path.exists(path):
        d = np.load(path)
        return d["X"], d["y"]
    epochs, labels = _moabb_epochs(key, subject)
    names = list(epochs.ch_names)
    upper = {n.upper(): i for i, n in enumerate(names)}
    idx = [upper[c.upper()] for c in channels]
    X = epochs.get_data()[:, idx, :] * 1e6            # volts -> microvolts
    if X.shape[2] > TARGET_T:
        X = X[:, :, :TARGET_T]
    elif X.shape[2] < TARGET_T:
        pad = TARGET_T - X.shape[2]
        print(f"[warn] {key} s{subject}: {X.shape[2]} samples < {TARGET_T}, zero-padding")
        X = np.pad(X, ((0, 0), (0, 0), (0, pad)))
    y = np.array([0 if str(l) == "left_hand" else 1 for l in labels])
    assert set(np.unique(y)) <= {0, 1}
    np.savez_compressed(path, X=X.astype(np.float32), y=y)
    return X.astype(np.float32), y


# --------------------------------------------------------------------------
# Synthetic data for smoke tests (no downloads)
# --------------------------------------------------------------------------
def make_synthetic(n_clients, n_chans=20, n_trials=160, seed=0):
    rng = np.random.RandomState(seed)
    clients = []
    for i in range(n_clients):
        y = np.tile([0, 1], n_trials // 2)
        rng.shuffle(y)
        X = rng.randn(n_trials, n_chans, TARGET_T) * rng.uniform(0.5, 2.0)
        X[y == 0, 3, :] *= 1.8      # class-dependent variance on two channels
        X[y == 1, 7, :] *= 1.8
        clients.append(Client(f"synth_s{i + 1}", X, y))
    return clients, [f"CH{i}" for i in range(n_chans)]


def build_clients(cfg):
    if cfg.synthetic:
        return make_synthetic(cfg.n_synth_clients)
    channels = common_channels(cfg.datasets, cfg.cache_dir)
    print(f"[data] {len(channels)} common channels: {channels}")
    clients = []
    for key in cfg.datasets:
        n = cfg.n_2a if key == "2a" else cfg.n_lee
        for s in range(1, n + 1):
            X, y = load_subject(key, s, channels, cfg.cache_dir)
            clients.append(Client(f"{key}_s{s}", X, y))
            print(f"[data] {key} subject {s}: {X.shape}")
    return clients, channels


# --------------------------------------------------------------------------
# Splitting / normalisation
# --------------------------------------------------------------------------
def fold_split(y, n_folds, seed, fold):
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for i, (tr, te) in enumerate(skf.split(np.zeros(len(y)), y)):
        if i == fold:
            return tr, te
    raise ValueError("fold out of range")


def standardize(Xtr, Xte):
    """Per-channel z-score using TRAIN statistics only."""
    mu = Xtr.mean(axis=(0, 2), keepdims=True)
    sd = Xtr.std(axis=(0, 2), keepdims=True) + 1e-8
    return (Xtr - mu) / sd, (Xte - mu) / sd

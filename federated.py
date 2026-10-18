"""Training procedures. All methods get the same total epoch budget."""
import copy
import time

import numpy as np
import torch
import torch.nn.functional as F

from model import EEGNet, is_bn, is_head


# --------------------------------------------------------------------------
# basic train / eval
# --------------------------------------------------------------------------
def local_train(model, X, y, epochs, lr, bs, gen):
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = X.shape[0]
    for _ in range(epochs):
        perm = torch.randperm(n, generator=gen).to(X.device)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            if len(idx) < 2:        # BatchNorm needs >1 sample
                continue
            opt.zero_grad()
            loss = F.cross_entropy(model(X[idx]), y[idx])
            loss.backward()
            opt.step()


@torch.no_grad()
def predict_acc(model, X, y):
    model.eval()
    correct = 0
    for i in range(0, X.shape[0], 256):
        correct += (model(X[i:i + 256]).argmax(1) == y[i:i + 256]).sum().item()
    return correct / X.shape[0]


@torch.no_grad()
def extract_features(model, X):
    model.eval()
    return torch.cat([model.features(X[i:i + 256]) for i in range(0, X.shape[0], 256)]
                     ).cpu().numpy()


def new_model(n_chans, seed):
    torch.manual_seed(seed)
    return EEGNet(n_chans)


def _gen(*parts):
    """Reproducible generator (python's hash() is salted per process, so use md5)."""
    import hashlib
    g = torch.Generator()
    g.manual_seed(int(hashlib.md5(repr(parts).encode()).hexdigest(), 16) % (2 ** 31))
    return g


# --------------------------------------------------------------------------
# aggregation
# --------------------------------------------------------------------------
def snapshot(model):
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def aggregate_plain(states, weights, key_filter):
    w = torch.tensor(weights, dtype=torch.float32)
    w = w / w.sum()
    out = {}
    for k in states[0]:
        if not key_filter(k) or not states[0][k].dtype.is_floating_point:
            continue
        out[k] = sum(wi * s[k] for wi, s in zip(w, states))
    return out


def aggregate_crypto(states, weights, key_filter, aggregator):
    keys = [k for k in states[0] if key_filter(k) and states[0][k].dtype.is_floating_point]
    shapes = {k: states[0][k].shape for k in keys}
    vecs = [torch.cat([s[k].flatten() for k in keys]).cpu().numpy() for s in states]
    agg = aggregator.aggregate(vecs, weights)
    out, pos = {}, 0
    for k in keys:
        n = int(np.prod(shapes[k]))
        out[k] = torch.from_numpy(agg[pos:pos + n].copy()).reshape(shapes[k])
        pos += n
    return out


# --------------------------------------------------------------------------
# methods
# --------------------------------------------------------------------------
def train_solo(data, cfg, seed, offset=0):
    dev = cfg.resolve_device()
    models = []
    for i, c in enumerate(data):
        m = new_model(c["Xtr"].shape[1], seed * 1000 + i + offset).to(dev)
        local_train(m, c["Xtr"], c["ytr"], cfg.rounds * cfg.local_epochs,
                    cfg.lr, cfg.batch_size, _gen("solo", seed, i, offset))
        models.append(m)
    return models, {}


def train_central(data, cfg, seed, offset=0):
    dev = cfg.resolve_device()
    X = torch.cat([c["Xtr"] for c in data])
    y = torch.cat([c["ytr"] for c in data])
    m = new_model(X.shape[1], seed + offset).to(dev)
    local_train(m, X, y, cfg.rounds * cfg.local_epochs, cfg.lr,
                cfg.batch_size, _gen("central", seed, offset))
    return [m for _ in data], {}


def train_federated(kind, data, cfg, seed, offset=0):
    """kind: fedavg | fedper | fedbn | encrypted_head | encrypted_full"""
    dev = cfg.resolve_device()
    init = new_model(data[0]["Xtr"].shape[1], seed + offset).to(dev)
    models = [copy.deepcopy(init) for _ in data]
    weights = [c["Xtr"].shape[0] for c in data]

    crypto, aggregator = None, None
    if kind == "fedavg":
        shared = lambda k: True
    elif kind == "fedper":
        shared = lambda k: not is_head(k)
    elif kind == "fedbn":
        shared = lambda k: not is_bn(k)
    elif kind == "encrypted_head":
        from crypto import PaillierAggregator
        aggregator = PaillierAggregator(cfg.paillier_bits, cfg.fp_bits)
        shared = lambda k: not is_head(k)       # plaintext part
        crypto = lambda k: is_head(k)           # encrypted part
    elif kind == "encrypted_full":
        from crypto import PaillierAggregator
        aggregator = PaillierAggregator(cfg.paillier_bits, cfg.fp_bits)
        shared = lambda k: False
        crypto = lambda k: True
    else:
        raise ValueError(kind)

    for r in range(cfg.rounds):
        for i, c in enumerate(data):
            local_train(models[i], c["Xtr"], c["ytr"], cfg.local_epochs, cfg.lr,
                        cfg.batch_size, _gen("fed", kind, seed, offset, r, i))
        if cfg.eval_local and kind in ("fedper", "fedbn") and r == cfg.rounds - 1:
            break   # keep the just-trained local models (backbone not overwritten) for evaluation
        states = [snapshot(m) for m in models]
        agg = aggregate_plain(states, weights, shared)
        if crypto is not None:
            agg.update(aggregate_crypto(states, weights, crypto, aggregator))
        for m in models:
            m.load_state_dict({k: v.to(dev) for k, v in agg.items()}, strict=False)

    info = {}
    if aggregator is not None:
        info["crypto_log"] = aggregator.log
    return models, info


def finetune(models, data, cfg, seed):
    out = []
    for i, (m, c) in enumerate(zip(models, data)):
        m2 = copy.deepcopy(m)
        local_train(m2, c["Xtr"], c["ytr"], cfg.ft_epochs, cfg.lr,
                    cfg.batch_size, _gen("ft", seed, i))
        out.append(m2)
    return out


def train_method(method, data, cfg, seed):
    base = method[:-4] if method.endswith("_rep") else method
    offset = 777 if method.endswith("_rep") else 0
    if base == "solo":
        return train_solo(data, cfg, seed, offset)
    if base == "central":
        return train_central(data, cfg, seed, offset)
    if base in ("fedavg", "fedper", "fedbn", "encrypted_head", "encrypted_full"):
        return train_federated(base, data, cfg, seed, offset)
    raise ValueError(f"unknown method {method}")

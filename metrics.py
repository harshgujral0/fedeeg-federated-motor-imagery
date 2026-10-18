"""Representation metrics."""
import numpy as np


def spatial_importance(model):
    """Mean |depthwise weight| per channel, min-max scaled (as in the paper)."""
    w = model.spatial.weight.detach().abs().mean(dim=(0, 1, 3)).cpu().numpy()
    return (w - w.min()) / (w.max() - w.min() + 1e-12)


def cosine(a, b):
    a, b = np.asarray(a, np.float64).ravel(), np.asarray(b, np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def pearson(a, b):
    """Mean-centred cosine. Unlike plain cosine of non-negative importance
    vectors it is NOT inflated by the shared positive offset."""
    a, b = np.asarray(a, np.float64).ravel(), np.asarray(b, np.float64).ravel()
    return cosine(a - a.mean(), b - b.mean())


def linear_cka(X, Y):
    X = np.asarray(X, np.float64)
    Y = np.asarray(Y, np.float64)
    X = X - X.mean(0, keepdims=True)
    Y = Y - Y.mean(0, keepdims=True)
    num = np.linalg.norm(X.T @ Y, "fro") ** 2
    den = np.linalg.norm(X.T @ X, "fro") * np.linalg.norm(Y.T @ Y, "fro")
    return float(num / (den + 1e-12))


def chance_cosine(n_chans, n_draws=20000, seed=0):
    """Expected cosine between two RANDOM min-max-scaled importance vectors."""
    rng = np.random.RandomState(seed)
    vals = []
    for _ in range(n_draws // 100):
        a = np.abs(rng.randn(100, n_chans))
        b = np.abs(rng.randn(100, n_chans))
        a = (a - a.min(1, keepdims=True)) / (a.max(1, keepdims=True) - a.min(1, keepdims=True))
        b = (b - b.min(1, keepdims=True)) / (b.max(1, keepdims=True) - b.min(1, keepdims=True))
        vals.append((a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)))
    return float(np.mean(vals))

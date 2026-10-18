import numpy as np
from scipy.stats import wilcoxon


def holm(pvals):
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def bootstrap_ci(diffs, n_boot=10000, alpha=0.05, seed=0):
    diffs = np.asarray(diffs, float)
    rng = np.random.RandomState(seed)
    means = np.array([rng.choice(diffs, len(diffs)).mean() for _ in range(n_boot)])
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def paired_test(a, b):
    """Wilcoxon signed-rank on paired client-level scores."""
    d = np.asarray(a, float) - np.asarray(b, float)
    if np.allclose(d, 0):
        return 1.0
    return float(wilcoxon(a, b).pvalue)

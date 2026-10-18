"""Measure what Paillier aggregation actually costs and what fixed-point
quantisation does to the result (instead of claiming 'identical').

  python crypto_benchmark.py          # writes crypto_benchmark.csv
"""
import time

import numpy as np
import pandas as pd

from crypto import PaillierAggregator


def per_element_cost(bits, n=100):
    from phe import paillier
    pub, priv = paillier.generate_paillier_keypair(n_length=bits)
    vals = list(range(1, n + 1))
    t0 = time.perf_counter(); enc = [pub.encrypt(v) for v in vals]; t_enc = (time.perf_counter() - t0) / n
    t0 = time.perf_counter(); s = enc[0]
    for e in enc[1:]:
        s = s + e
    t_add = (time.perf_counter() - t0) / (n - 1)
    t0 = time.perf_counter(); [priv.decrypt(e) for e in enc]; t_dec = (time.perf_counter() - t0) / n
    return t_enc, t_add, t_dec


def main():
    from model import EEGNet
    n_head = sum(p.numel() for p in EEGNet(21).classifier.parameters())
    n_full = sum(p.numel() for p in EEGNet(21).parameters())
    rows = []
    for bits in (1024, 2048):
        te, ta, td = per_element_cost(bits)
        for scope, n_el in (("head", n_head), ("full_model", n_full)):
            for k in (2, 5, 10, 20):
                rows.append(dict(kind="estimated_time", key_bits=bits, scope=scope, n_params=n_el,
                                 n_clients=k,
                                 client_encrypt_s=te * n_el, server_add_s=ta * n_el * (k - 1),
                                 decrypt_s=td * n_el, upload_MB_per_client=n_el * 2 * bits / 8 / 1e6))
    rng = np.random.RandomState(0)
    for fp in (8, 12, 16, 20, 24):
        agg = PaillierAggregator(1024, fp)
        vecs = [rng.randn(n_head).astype(np.float32) * 0.1 for _ in range(5)]
        agg.aggregate(vecs, [100, 120, 90, 150, 110])
        rows.append(dict(kind="measured_error", key_bits=1024, scope="head", n_params=n_head,
                         n_clients=5, fp_bits=fp, max_abs_err=agg.log[-1]["max_abs_err"]))
    pd.DataFrame(rows).to_csv("crypto_benchmark.csv", index=False)
    print(pd.DataFrame(rows).to_string())


if __name__ == "__main__":
    main()

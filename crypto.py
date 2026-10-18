"""Paillier aggregation with explicit fixed-point encoding.

Clients scale their vector by their (public) aggregation weight, quantise to
integers (fp_bits fractional bits), encrypt element-wise. The server only
multiplies ciphertexts (= adds plaintexts). The decrypted sum is the weighted
average. In this simulation one process holds the private key; the server step
uses ciphertexts + public key only.

Because of the quantisation, the result differs from plaintext averaging by at
most ~ n_clients * 2^-(fp_bits+1) per element: this is what we MEASURE instead of
claiming "identical".
"""
import operator
import time
from functools import reduce

import numpy as np


class PaillierAggregator:
    def __init__(self, bits=1024, fp_bits=24):
        try:
            from phe import paillier
        except ImportError as e:  # pragma: no cover
            raise ImportError("pip install phe (and optionally gmpy2)") from e
        self.bits, self.fp_bits = bits, fp_bits
        self.scale = 2 ** fp_bits
        self.pub, self.priv = paillier.generate_paillier_keypair(n_length=bits)
        self.log = []   # one dict per call

    def aggregate(self, vectors, weights):
        w = np.asarray(weights, dtype=np.float64)
        w = w / w.sum()
        plain = sum(wi * v.astype(np.float64) for wi, v in zip(w, vectors))

        t0 = time.perf_counter()
        enc_clients = []
        for v, wi in zip(vectors, w):
            q = [int(round(float(x) * wi * self.scale)) for x in v]
            enc_clients.append([self.pub.encrypt(i) for i in q])
        t_enc = time.perf_counter() - t0

        t0 = time.perf_counter()
        summed = [reduce(operator.add, col) for col in zip(*enc_clients)]
        t_add = time.perf_counter() - t0

        t0 = time.perf_counter()
        dec = np.array([float(self.priv.decrypt(c)) for c in summed]) / self.scale
        t_dec = time.perf_counter() - t0

        n_el = len(vectors[0])
        self.log.append(dict(
            n_clients=len(vectors), n_elements=n_el, key_bits=self.bits,
            fp_bits=self.fp_bits, t_encrypt_s=t_enc, t_server_add_s=t_add,
            t_decrypt_s=t_dec,
            upload_bytes_per_client=n_el * (2 * self.bits // 8),
            max_abs_err=float(np.max(np.abs(dec - plain))),
        ))
        return dec.astype(np.float32)

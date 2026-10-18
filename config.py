"""Central configuration for the expanded FedEEG-Interpret experiments."""
from dataclasses import dataclass


@dataclass
class Config:
    # ---- paths -------------------------------------------------------------
    out_dir: str = "results"
    cache_dir: str = "cache"

    # ---- protocol ----------------------------------------------------------
    mode: str = "within"            # "within" (k-fold per client) | "loso"
    seeds: tuple = (42, 123, 256, 789, 1024)
    n_folds: int = 5
    run_folds: tuple = ()           # empty = run every fold
    max_holdouts: int = 0           # loso only: 0 = hold out every client

    # ---- training (equal budget for every method) --------------------------
    rounds: int = 10
    local_epochs: int = 5           # Solo/Central train rounds*local_epochs epochs
    batch_size: int = 32
    lr: float = 1e-3
    ft_epochs: int = 10             # local fine-tuning epochs for fedavg_ft

    # ---- methods -----------------------------------------------------------
    # solo, central, fedavg, fedper (head local), fedbn (BN local),
    # fedavg_ft (FedAvg then local fine-tune), encrypted_head, encrypted_full
    methods: tuple = ("solo", "central", "fedavg", "fedper", "fedbn", "fedavg_ft")
    with_null: bool = False         # adds solo_rep, fedavg_rep (re-init noise floor)
    eval_local: bool = False        # fedper/fedbn: evaluate the locally trained models right after the
                                    # LAST local training (skip the final aggregation) instead of after it

    # ---- data --------------------------------------------------------------
    datasets: tuple = ("2a", "lee")
    n_2a: int = 9                   # BCI IV 2a has 9 subjects
    n_lee: int = 20                 # Lee2019 has 54; start small (big downloads)
    probe_size: int = 24            # test trials per client used for CKA probe
    synthetic: bool = False         # smoke-test data, no downloads
    n_synth_clients: int = 6

    # ---- crypto ------------------------------------------------------------
    paillier_bits: int = 1024
    fp_bits: int = 24               # fixed-point fractional bits

    device: str = ""                # "" = cuda if available else cpu

    def resolve_device(self) -> str:
        if self.device:
            return self.device
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"

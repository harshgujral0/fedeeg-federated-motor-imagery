# FedEEG-Interpret: expanded experiments

Each subject = one federated client, on the channels BCI IV 2a and Lee2019 share.

## Setup (Windows / ASUS TUF with NVIDIA GPU)
```
python -m venv .venv && .venv\Scripts\activate
# 1) torch with CUDA: copy the command from https://pytorch.org/get-started/locally/
# 2) the rest:
pip install -r requirements.txt
```
Optional: set `MNE_DATA` to a drive with space. BCI IV 2a is small; **Lee2019 is large
(tens of GB for all 54 subjects)**, so start with `--n_lee 5`.

## Run order
```
python run_experiments.py --synthetic --seeds 42 --run_folds 0        # 1. smoke test (no downloads)
python run_experiments.py --n_lee 3 --seeds 42 --run_folds 0          # 2. tiny real run, check shapes/accuracy
python analysis.py --results results                                   # 3. tables
python run_experiments.py --n_lee 20 --with_null                       # 4. main run (resumable)
python run_experiments.py --mode loso --max_holdouts 5 --seeds 42 123  # 5. unseen-subject test
python crypto_benchmark.py                                             # 6. encryption cost + quantisation error
python run_experiments.py --methods encrypted_head fedavg --seeds 42 --run_folds 0 --n_lee 3 \
       --out_dir results_crypto                                         # 7. confirm encrypted == plaintext (compare accuracies, crypto max_abs_err in metrics.json)
```
## Methods
solo, central (pooled upper bound), fedavg, fedper (head local), fedbn (BN local),
fedavg_ft (FedAvg + local fine-tune), encrypted_head, encrypted_full, and `_rep` nulls
(`--with_null`: same method, different init, to get the noise floor).

## Differences from the original paper (state them in the new paper)
- Equal epoch budget for all methods (paper: Solo 30 epochs vs 100 for FL).
- k-fold per client instead of one 80/20 split.
- Shared-channel set instead of zero-padding.
- Aggregation weighted by client train-set size.
- Backbone similarity reported as cosine AND Pearson, plus the chance level.

## Outputs
`results/seed*_fold*/{metrics.json,arrays.npz}` and `results/analysis/*.csv`.
Statistics use CLIENTS as the unit (paired Wilcoxon + Holm + bootstrap CI).

## Status
All code in this repository was run on the authors' machine (Windows, NVIDIA RTX 2050) with Python 3.11.9,
PyTorch 2.14.1 (CUDA 12.6), MOABB 1.7.2, MNE 1.13.2 and phe 1.5.0. See `REPRODUCE.md` for the exact command
behind every table and figure of the paper. GPU arithmetic is not bit-exact, so numbers can differ slightly.

## Update: deep analysis (run order)
```
python run_experiments.py --datasets 2a lee --n_2a 2 --n_lee 2 --seeds 42 --run_folds 0 --methods solo fedavg fedper --with_null --out_dir results_test   # quick test (also saves models)
python deep_analysis.py --results results_test
```
Runs made BEFORE this update do not contain saved models; deep_analysis skips them.
Tuning check (does FedAvg lose because of client drift?):
```
python run_experiments.py --datasets 2a --rounds 30 --local_epochs 2 --methods solo central fedavg fedper fedavg_ft --out_dir results_tune
```

## Update 2: evaluation timing for personalised methods
By default fedper/fedbn are evaluated AFTER the last aggregation (shared layers overwritten by the average,
local head/BN left as trained on the old backbone). `--eval_local` evaluates the local models right after the
final local training instead. Run both variants and report which you use:
```
python run_experiments.py --datasets 2a lee --n_lee 20 --seeds 42 --run_folds 0 1 --methods solo fedavg fedper fedbn --eval_local --out_dir results_evallocal
```

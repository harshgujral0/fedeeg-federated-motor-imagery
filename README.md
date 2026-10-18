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
The numpy/pandas/scipy parts (metrics, stats, analysis) were tested on fake data.
The PyTorch / MOABB / phe parts could NOT be run where this was written: run the
smoke test (step 1) first and send me any error.

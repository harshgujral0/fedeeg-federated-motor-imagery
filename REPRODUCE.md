# Reproducing the tables and figures of the paper

Environment: Python 3.11.9, PyTorch 2.14.1 (CUDA 12.6), MOABB 1.7.2, MNE 1.13.2, phe 1.5.0 (`requirements.txt`).
Run all commands in this folder with the virtual environment active.

| Paper item | Command | Output |
|---|---|---|
| Parameter counts (1954 / 514) | `python check_model.py` | printed |
| Main runs (29 clients, 10 seeds x 5 folds) | `python run_experiments.py --datasets 2a lee --n_lee 20 --seeds 42 123 256 789 1024 2024 7 99 555 31337 --out_dir results_main --with_null` | `results_main/` |
| Accuracy table, tests | `python analysis.py --results results_main` | `results_main/analysis/` |
| Per-dataset table | `python per_dataset.py --results results_main` | `results_main/analysis/per_dataset_*.csv` |
| Representation table, Figs 3-4 data | `python deep_analysis.py --results results_main` | `results_main/analysis/deep_*.csv` |
| Evaluation-protocol table | `python run_experiments.py --datasets 2a lee --n_lee 20 --seeds 42 123 256 789 1024 2024 7 99 555 31337 --methods solo fedavg fedper fedbn --eval_local --out_dir results_main_evallocal` then `python analysis.py --results results_main_evallocal` | `results_main_evallocal/analysis/` |
| Training-length check (2 runs) | `python run_experiments.py --datasets 2a lee --n_lee 20 --seeds 42 --run_folds 0 1 --rounds 100 --local_epochs 1 --methods solo central fedavg fedper fedbn fedavg_ft --out_dir results_tune_r100` then `python analysis.py --results results_tune_r100` | `results_tune_r100/analysis/` |
| Encrypted run (6 clients) | `python run_experiments.py --datasets 2a lee --n_2a 3 --n_lee 3 --seeds 42 --run_folds 0 --methods fedavg encrypted_head --out_dir results_crypto` | `results_crypto/` |
| Encryption cost / error tables | `python crypto_benchmark.py` | `crypto_benchmark.csv` |
| Figures 1-4 and CSV tables | `python make_figures.py --main results_main --evallocal results_main_evallocal --crypto crypto_benchmark.csv --out figures` | `figures/` |

Notes
- Runs are resumable; the first run downloads the datasets (Lee2019 is large).
- GPU arithmetic is not bit-exact, so numbers can differ slightly between machines.
- The full-model rows of the encryption cost table are scaled from per-element timings measured on the 514-parameter output layer.

# CCAR-NRS — code (Supplemental File S2)

Implementation and experiment scripts for *Core-Centric Attribute Reduction for Neighborhood Rough Sets without Discernibility Matrix Construction*.

## Layout

| Path | Content |
|---|---|
| `ccar/` | CCAR-NRS: shared blocked NRS kernel (`nrs.py`), one-pass core identification (`core.py`), conflicting pairs and residual family (`conflicting.py`), hitting-set solvers, CCAR-E / CCAR-H / surrogate pipeline |
| `baselines/` | Core-init-Greedy, QuickReduct-NRS, NIP-NRS (`nicp.py`), GBNRS, WNRS, Wang kNN-NRS |
| `experiments/` | Benchmark runner, ablation, delta sweep, wide-table stress test, Covertype scalability, dataset loaders |
| `scripts/` | Regenerate every benchmark number, table and figure of the paper from Supplemental File S1 |
| `tests/` | Unit tests |

## Environment

The published results were produced with Python 3.9.6, NumPy 2.0.2, SciPy 1.13.1, scikit-learn 1.6.1, pandas 2.3.3 and matplotlib 3.9.4 on an Apple M4 Pro (48 GB RAM, macOS 26.6), running one dataset at a time. `requirements.txt` pins these versions.

```sh
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q tests
```

## Reproducing the results (writes to `../results_v7/`)

Run from this directory:

```sh
python -m experiments.run_benchmark --jobs 1 --results-dir ../results_v7   # main benchmark + ablation (~1 h)
python -m experiments.run_delta_sweep_v5                                   # delta sweep
python -m experiments.run_wide_table_stress                                # |C|-axis stress test
python -m experiments.run_scalability                                      # Covertype subsets (several hours)
```

`run_benchmark.py` times only the reduction call of each method; the gamma_delta validity and minimality checks and classifier training run outside the timer. It writes one `<dataset>_perfold.csv` per dataset (reduct, times, accuracies, validity, minimality for every method and fold) and the aggregate `benchmark_20datasets.csv`.

## Regenerating the paper's numbers and figures from S1

```sh
python scripts/manuscript_numbers.py ../results_v7/benchmark_20datasets.csv   # all benchmark statistics and table rows
python scripts/plot_scaling_axes.py ../results_v7/benchmark_20datasets.csv ../results_v7/benchmark_20datasets.csv scaling_axes.png
python scripts/plot_delta_sweep.py ../results_v7/delta_sweep_v7.csv delta_sensitivity.png
```

## Licence

MIT (see `LICENSE`).

# Supplemental File S1 — raw data

Results of every experiment in the paper (canonical run "v7", 2026-09-26). The code that produced them and the scripts that regenerate every number, table and figure are in Supplemental File S2.

## Main benchmark (18 UCI datasets, 5-fold stratified CV, seed 42, delta = 0.1)

- `<dataset>_perfold.csv` — **raw per-fold log**, one row per method and fold (18 x 8 x 5 = 720 rows). Columns: `dataset, algorithm, fold, n_train, reduct_set` (selected attribute indices, `;`-separated), `n_reduct, time_reduct_sec` (wall-clock time of the reduction call only), `time_eval_sec` (kNN + SVM training/scoring), `knn, svm` (test accuracy), `gamma_C` (gamma_delta(C,D) on the training fold), `valid` (gamma_delta(R) >= gamma_delta(C)), `minimal` (valid and no single attribute removable), `n_core, n_pairs, used_surrogate`.
- `<dataset>_folds.csv` — per-dataset aggregates (fold means and standard deviations) derived from the per-fold log, plus one `_ablation` row (full-table conflicting-pair ablation).
- `benchmark_aggregate.csv` — the 18 `<dataset>_folds.csv` files concatenated (162 rows).

Algorithm keys: `ccar_exact` = CCAR-E, `ccar_h` = CCAR-H, `ccar_surrogate` = CCAR with explicit surrogate call (identical to CCAR-E), `core_init_greedy` = Core-init, `quick_reduct` = QuickReduct-NRS, `nicp` = NIP-NRS, `gbnrs` = GBNRS, `wnrs` = WNRS.

## Other experiments

- `delta_sweep_v7.csv` — CCAR-E on 8 datasets for delta in {0.05, 0.10, 0.15, 0.20, 0.30}.
- `wide_table_stress_v7.csv` — synthetic tables, |U| = 400, |C| = 20..200.
- `scalability_v7.csv` — Covertype stratified subsamples, n = 1000..10000 (CCAR-H, Core-init, QuickReduct; reduct, time, validity).
- `core_stability.json` — CORE_delta on every training fold of the 18 datasets.
- `manuscript_numbers_v7.txt` — output of `scripts/manuscript_numbers.py` on `benchmark_aggregate.csv`: every Friedman/Nemenyi/Wilcoxon-Holm statistic, rank, fit and table row printed in the paper.
- `RUN_INFO.txt` — hardware, software versions, command lines and run times. Console logs are omitted from this GitHub package because they contain local filesystem paths; the underlying results are included above.

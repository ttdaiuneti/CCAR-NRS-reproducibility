"""Run 20-dataset benchmark: 5-fold CV, all methods, CSV + LaTeX table fragments."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from ccar.ccar_exact import ccar_nrs_exact, verify_reduct
from ccar.ccar_heuristic import ccar_nrs_heuristic
from ccar.ccar_surrogate import ccar_nrs_surrogate, verify_minimal
from ccar.nrs import normalize_features
from ccar.core import compute_core
from ccar.result import CCARResult
from baselines.gbnrs import gbnrs_reduction
from baselines.greedy_core_init import core_init_greedy
from baselines.nicp import nicp_reduction
from baselines.quick_reduct_nrs import quick_reduct_nrs
from baselines.wnrs import wnrs_reduction
from experiments.ablation import time_full_matrix_proxy, time_pair_only_path
from experiments.datasets import BENCHMARK_DATASETS, DatasetInfo

RANDOM_STATE = 42
N_SPLITS = 5
DELTA_DEFAULT = 0.1

ALGORITHMS: dict[str, str] = {
    "ccar_exact": "CCAR-E",
    "ccar_h": "CCAR-H",
    "ccar_surrogate": "Surr.",
    "core_init_greedy": "Core-init",
    "quick_reduct": "QuickReduct",
    "nicp": "NICP",
    "gbnrs": "GBNRS",
    "wnrs": "WNRS",
}


def _reduce_ccar_exact(X: np.ndarray, y: np.ndarray, delta: float) -> CCARResult:
    return ccar_nrs_exact(X, y, delta)


def _reduce_ccar_h(X: np.ndarray, y: np.ndarray, delta: float) -> CCARResult:
    return ccar_nrs_heuristic(X, y, delta)


def _reduce_ccar_surr(X: np.ndarray, y: np.ndarray, delta: float) -> CCARResult:
    return ccar_nrs_surrogate(X, y, delta)


def _reduce_core_init(X: np.ndarray, y: np.ndarray, delta: float) -> list[int]:
    return core_init_greedy(X, y, delta)


def _reduce_nicp(X: np.ndarray, y: np.ndarray, delta: float) -> list[int]:
    return nicp_reduction(X, y, delta)


def _reduce_gbnrs(X: np.ndarray, y: np.ndarray, delta: float) -> list[int]:
    return gbnrs_reduction(X, y, delta, purity_threshold=0.9)


def _reduce_quick_reduct(X: np.ndarray, y: np.ndarray, delta: float) -> list[int]:
    return quick_reduct_nrs(X, y, delta)


def _reduce_wnrs(X: np.ndarray, y: np.ndarray, delta: float) -> list[int]:
    return wnrs_reduction(X, y, delta)


REDUCERS: dict[str, Callable[..., Any]] = {
    "ccar_exact": _reduce_ccar_exact,
    "ccar_h": _reduce_ccar_h,
    "ccar_surrogate": _reduce_ccar_surr,
    "core_init_greedy": _reduce_core_init,
    "quick_reduct": _reduce_quick_reduct,
    "nicp": _reduce_nicp,
    "gbnrs": _reduce_gbnrs,
    "wnrs": _reduce_wnrs,
}


def run_reduction(
    algo: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    delta: float,
) -> tuple[list[int], float, dict[str, Any]]:
    # The timer covers ONLY the reducer call. Prior to 2026-09 it also covered
    # the validity check (normalize + compute_core + gamma(R) for baselines,
    # verify_reduct for CCAR); for baselines that evaluation-only work was
    # 74-93% of CCAR-E's own runtime, inflating every runtime comparison in
    # CCAR's favour (review_opus_peerj_round2.md, row 16).
    meta: dict[str, Any] = {}
    if algo in ("ccar_exact", "ccar_h", "ccar_surrogate"):
        t0 = time.perf_counter()
        res: CCARResult = REDUCERS[algo](X_train, y_train, delta)
        runtime = time.perf_counter() - t0
        reduct = res.reduct_indices
        meta = {
            "n_core": len(res.core_indices),
            "n_pairs": res.n_pairs,
            "gamma": res.gamma_c,
            "used_surrogate": res.used_surrogate,
            "valid": verify_reduct(X_train, y_train, reduct, res.gamma_c, delta),
            # Universe size of the FOLD the pair count was measured on. The
            # pair_ratio column below must divide by this, not by the full-data
            # n; prior to 2026-08 it used n*(n-1)/2 over the whole dataset,
            # deflating the per-method ratio by ~1.56x at a 0.8 train fraction.
            "n_universe": int(X_train.shape[0]),
        }
    else:
        t0 = time.perf_counter()
        reduct = REDUCERS[algo](X_train, y_train, delta)
        runtime = time.perf_counter() - t0
        from ccar.nrs import dependency_degree, normalize_features

        # gamma_c MUST be computed on the same normalized representation as
        # gamma_r below. Prior to 2026-08 compute_core() was called on the raw
        # X_train while gamma_r used normalize_features(X_train); with delta=0.1
        # on unscaled features almost no pair falls inside the neighborhood, so
        # gamma_c was spuriously 1.0 and every baseline was judged invalid
        # whenever the true gamma_delta(C) < 1. CCAR was unaffected because
        # ccar_nrs_exact() normalizes before calling compute_core(), so only the
        # baselines carried the penalty -- a systematic bias in their disfavour.
        Xn = normalize_features(X_train)
        core, gamma_c = compute_core(Xn, y_train, delta)
        idx = np.array(reduct, dtype=int) if reduct else np.array([], dtype=int)
        gamma_r = dependency_degree(Xn, y_train, idx, delta) if reduct else 0.0
        meta = {
            "n_core": len(core),
            "n_pairs": None,
            "gamma": gamma_c,
            "used_surrogate": False,
            "valid": bool(reduct) and gamma_r >= gamma_c - 1e-9,
        }
    return reduct, runtime, meta


def fold_test_scores(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    reduct: list[int],
) -> dict[str, float]:
    if not reduct:
        return {"knn": 0.0, "svm": 0.0}
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_train[:, reduct])
    X_te_s = scaler.transform(X_test[:, reduct])
    knn = KNeighborsClassifier(n_neighbors=3).fit(X_tr_s, y_train).score(X_te_s, y_test)
    svm = (
        SVC(kernel="rbf", C=1.0, gamma="scale")
        .fit(X_tr_s, y_train)
        .score(X_te_s, y_test)
    )
    return {"knn": float(knn), "svm": float(svm)}


def _progress_path(results_dir: Path, key: str) -> Path:
    d = results_dir / ".progress"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{key}.json"


def _write_progress(results_dir: Path, key: str, payload: dict[str, Any]) -> None:
    payload = {
        **payload,
        "dataset": key,
        "updated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
    }
    path = _progress_path(results_dir, key)
    path.write_text(json.dumps(payload, indent=2))


def _log(msg: str) -> None:
    print(msg, flush=True)


def benchmark_dataset(
    info: DatasetInfo,
    delta: float,
    *,
    n_splits: int = N_SPLITS,
    results_dir: Path | None = None,
) -> list[dict[str, Any]]:
    results_dir = results_dir or Path(__file__).resolve().parents[2] / "results"
    X, y = info.loader()
    n, p = X.shape
    # gamma_full MUST be computed on the Min--Max normalized representation, the
    # same one every reducer and every gamma_delta check uses. Prior to 2026-08
    # this line called compute_core() on the raw X: with delta=0.1 on unscaled
    # features (Glass oxides ~1.5-13, Segment pixel coordinates in the hundreds)
    # essentially no pair falls inside the neighborhood, so gamma_c was pushed to
    # a spurious 1.0 on 10 of the 18 datasets and the "consistent table" claim in
    # the manuscript was overstated (see BUGFIX_NOTES B7). The same defect was
    # fixed inside run_reduction() as B2; this metadata path was missed.
    from ccar.nrs import normalize_features as _norm_full

    gamma_c = compute_core(_norm_full(X), y, delta)[1]
    rows: list[dict[str, Any]] = []
    fold_records: list[dict[str, Any]] = []

    skf =StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    folds = list(skf.split(X, y))
    algo_keys = list(ALGORITHMS.keys())

    _write_progress(
        results_dir,
        info.key,
        {
            "status": "running",
            "algorithm": algo_keys[0],
            "algorithm_label": ALGORITHMS[algo_keys[0]],
            "fold": 0,
            "n_folds": n_splits,
            "algo_index": 1,
            "n_algorithms": len(algo_keys),
        },
    )
    _log(f"[{info.key}] START ({n}×{p}, γ={gamma_c:.3f})")

    for ai, algo in enumerate(algo_keys, start=1):
        fold_reduct: list[int] = []
        fold_runtime: list[float] = []
        fold_knn: list[float] = []
        fold_svm: list[float] = []
        # Validity and surrogate use are aggregated over ALL folds. Prior to
        # 2026-09 only the last fold's meta was kept, so a reducer that failed
        # gamma_delta on folds 1-4 but passed fold 5 was reported as valid
        # (review_opus_peerj_round2.md, row 17).
        fold_valid: list[bool] = []
        fold_surr: list[bool] = []
        fold_minimal: list[bool] = []
        meta_last: dict[str, Any] = {}

        for fi, (train_idx, test_idx) in enumerate(folds, start=1):
            _write_progress(
                results_dir,
                info.key,
                {
                    "status": "running",
                    "algorithm": algo,
                    "algorithm_label": ALGORITHMS[algo],
                    "fold": fi,
                    "n_folds": n_splits,
                    "algo_index": ai,
                    "n_algorithms": len(algo_keys),
                },
            )
            X_tr, y_tr = X[train_idx], y[train_idx]
            X_te, y_te = X[test_idx], y[test_idx]
            t_fold = time.perf_counter()
            reduct, rt, meta = run_reduction(algo, X_tr, y_tr, delta)
            fold_reduct.append(len(reduct))
            fold_runtime.append(rt)
            t_eval = time.perf_counter()
            scores = fold_test_scores(X_tr, y_tr, X_te, y_te, reduct)
            t_eval = time.perf_counter() - t_eval
            fold_knn.append(scores["knn"])
            fold_svm.append(scores["svm"])
            # Minimality (no single attribute removable while keeping
            # gamma_delta(R) >= gamma_delta(C)); outside the timer.
            minimal = bool(meta.get("valid")) and verify_minimal(
                normalize_features(X_tr), y_tr, list(reduct), meta["gamma"], delta
            )
            fold_valid.append(bool(meta.get("valid")))
            fold_minimal.append(minimal)
            fold_surr.append(bool(meta.get("used_surrogate")))
            fold_records.append(
                {
                    "dataset": info.key,
                    "algorithm": algo,
                    "fold": fi,
                    "n_train": int(len(train_idx)),
                    "reduct_set": ";".join(str(int(a)) for a in sorted(reduct)),
                    "n_reduct": len(reduct),
                    "time_reduct_sec": rt,
                    "time_eval_sec": t_eval,
                    "knn": scores["knn"],
                    "svm": scores["svm"],
                    "gamma_C": meta.get("gamma"),
                    "valid": bool(meta.get("valid")),
                    "minimal": minimal,
                    "n_core": meta.get("n_core"),
                    "n_pairs": meta.get("n_pairs"),
                    "used_surrogate": bool(meta.get("used_surrogate")),
                }
            )
            meta_last = meta
            if meta.get("n_core") is not None:
                meta_last["n_core"] = meta.get("n_core")
            _log(
                f"[{info.key}] {ALGORITHMS[algo]:>10} fold {fi}/{n_splits} "
                f"|R|={len(reduct)} {rt:.1f}s (total fold {time.perf_counter() - t_fold:.1f}s)"
            )

        rows.append(
            {
                "dataset": info.key,
                "dataset_name": info.display_name,
                "n_samples": n,
                "n_features": p,
                "gamma_full": gamma_c,
                "algorithm": algo,
                "algorithm_label": ALGORITHMS[algo],
                "n_reduct_mean": float(np.mean(fold_reduct)),
                "n_reduct_std": float(np.std(fold_reduct)),
                "runtime_mean": float(np.mean(fold_runtime)),
                "runtime_std": float(np.std(fold_runtime)),
                "knn_mean": float(np.mean(fold_knn)),
                "knn_std": float(np.std(fold_knn)),
                "svm_mean": float(np.mean(fold_svm)),
                "svm_std": float(np.std(fold_svm)),
                "n_core": meta_last.get("n_core"),
                "n_pairs": meta_last.get("n_pairs"),
                "pair_ratio": (
                    meta_last["n_pairs"]
                    / (
                        meta_last["n_universe"]
                        * (meta_last["n_universe"] - 1)
                        / 2
                    )
                    if meta_last.get("n_pairs") is not None
                    and meta_last.get("n_universe")
                    else None
                ),
                "used_surrogate": any(fold_surr),
                "n_folds_surrogate": int(sum(fold_surr)),
                "valid_reduct": all(fold_valid),
                "n_folds_valid": int(sum(fold_valid)),
                "n_folds_minimal": int(sum(fold_minimal)),
            }
        )

    _write_progress(
        results_dir,
        info.key,
        {
            "status": "running",
            "algorithm": "_ablation",
            "algorithm_label": "ablation",
            "fold": 1,
            "n_folds": 1,
            "algo_index": len(algo_keys) + 1,
            "n_algorithms": len(algo_keys) + 1,
        },
    )
    write_csv(fold_records, results_dir / f"{info.key}_perfold.csv")
    _log(f"[{info.key}] ablation...")

    # Ablation on full data (once per dataset)
    t_pair, n_core, n_pairs, p_ratio = time_pair_only_path(X, y, delta)
    t_full, n_m = time_full_matrix_proxy(X, y, delta)
    speedup = t_full / t_pair if t_pair > 1e-9 else float("nan")
    rows.append(
        {
            "dataset": info.key,
            "dataset_name": info.display_name,
            "n_samples": n,
            "n_features": p,
            "gamma_full": gamma_c,
            "algorithm": "_ablation",
            "algorithm_label": "ablation",
            "n_reduct_mean": n_core,
            "n_reduct_std": 0.0,
            "runtime_mean": t_pair,
            "runtime_std": 0.0,
            "knn_mean": np.nan,
            "knn_std": 0.0,
            "svm_mean": np.nan,
            "svm_std": 0.0,
            "n_core": n_core,
            "n_pairs": n_pairs,
            "pair_ratio": p_ratio,
            "used_surrogate": False,
            "valid_reduct": True,
            "t_full_matrix": t_full,
            "n_matrix_entries": n_m,
            "speedup_pair_vs_full": speedup,
        }
    )
    _write_progress(
        results_dir,
        info.key,
        {
            "status": "done",
            "algorithm": None,
            "algorithm_label": None,
            "fold": n_splits,
            "n_folds": n_splits,
            "n_algorithms": len(algo_keys),
        },
    )
    return rows


def _dataset_info(key: str) -> DatasetInfo:
    for d in BENCHMARK_DATASETS:
        if d.key == key:
            return d
    raise KeyError(f"Unknown dataset key: {key}")


def _limit_blas_threads() -> None:
    """Avoid oversubscription when many processes each call NumPy/BLAS."""
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    warnings.filterwarnings("ignore", category=RuntimeWarning)


def _run_one_dataset_task(
    key: str,
    delta: float,
    n_splits: int,
    results_dir: str,
) -> tuple[str, float, int]:
    """Worker: benchmark one dataset and write its folds CSV."""
    _limit_blas_threads()
    info = _dataset_info(key)
    out_dir = Path(results_dir)
    t0 = time.perf_counter()
    try:
        rows = benchmark_dataset(
            info, delta, n_splits=n_splits, results_dir=out_dir
        )
        out = out_dir / f"{key}_folds.csv"
        write_csv(rows, out)
        elapsed = time.perf_counter() - t0
        _log(f"[{key}] DONE in {elapsed:.1f}s → {out.name}")
        return key, elapsed, len(rows)
    except Exception as exc:
        _write_progress(
            out_dir,
            key,
            {"status": "failed", "error": str(exc)},
        )
        _log(f"[{key}] FAILED — {exc}")
        raise


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for k in row:
            if k not in seen:
                keys.append(k)
                seen.add(k)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def friedman_ranks(
    summary: list[dict],
    metric: str,
    algo_keys: list[str],
    *,
    higher_better: bool = False,
) -> dict[str, float]:
    """Average rank per algorithm across datasets (1 = best)."""
    datasets = sorted({r["dataset"] for r in summary if r["algorithm"] in algo_keys})
    ranks: dict[str, list[float]] = {a: [] for a in algo_keys}
    for ds in datasets:
        vals = []
        for a in algo_keys:
            row = next(
                x
                for x in summary
                if x["dataset"] == ds and x["algorithm"] == a
            )
            vals.append((a, float(row[metric])))
        vals.sort(key=lambda t: t[1], reverse=higher_better)
        for rank, (a, _) in enumerate(vals, start=1):
            ranks[a].append(float(rank))
    return {a: float(np.mean(rs)) for a, rs in ranks.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--delta", type=float, default=DELTA_DEFAULT)
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=[d.key for d in BENCHMARK_DATASETS],
    )
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument(
        "--jobs",
        type=int,
        default=None,
        help="Parallel dataset workers (default: min(len(datasets), cpu_count); 1 = sequential)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip datasets that already have {key}_folds.csv in results-dir",
    )
    args = parser.parse_args()

    results_dir = args.results_dir or Path(__file__).resolve().parents[2] / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    keys: list[str] = []
    for info in BENCHMARK_DATASETS:
        if info.key not in args.datasets:
            continue
        out = results_dir / f"{info.key}_folds.csv"
        if args.skip_existing and out.is_file():
            print(f"Skip {info.key} (exists: {out.name})")
            continue
        keys.append(info.key)

    if not keys:
        print("No datasets to run.")
        return

    cpu = os.cpu_count() or 4
    jobs = args.jobs if args.jobs is not None else min(len(keys), cpu)
    jobs = max(1, min(jobs, len(keys)))

    t_all = time.perf_counter()
    completed: list[tuple[str, float, int]] = []

    _log(f"Progress files: {results_dir / '.progress'}/")
    _log(f"Status anytime: python -m experiments.benchmark_status")

    if jobs == 1:
        for key in keys:
            info = _dataset_info(key)
            _log(f"Running {info.display_name} ({key})...")
            key, elapsed, nrows = _run_one_dataset_task(
                key, args.delta, N_SPLITS, str(results_dir)
            )
            completed.append((key, elapsed, nrows))
    else:
        _log(f"Parallel benchmark: {len(keys)} datasets, jobs={jobs} (cpu_count={cpu})")
        _log(f"Queued: {', '.join(keys)}")
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            futures = {
                pool.submit(
                    _run_one_dataset_task,
                    key,
                    args.delta,
                    N_SPLITS,
                    str(results_dir),
                ): key
                for key in keys
            }
            for fut in as_completed(futures):
                key = futures[fut]
                try:
                    key, elapsed, nrows = fut.result()
                    completed.append((key, elapsed, nrows))
                except Exception:
                    raise

    # Merge all per-dataset CSVs on disk (includes previously completed sets)
    all_rows: list[dict] = []
    for info in BENCHMARK_DATASETS:
        path = results_dir / f"{info.key}_folds.csv"
        if not path.is_file():
            continue
        with path.open() as f:
            all_rows.extend(list(csv.DictReader(f)))

    out_main = results_dir / "benchmark_20datasets.csv"
    write_csv(all_rows, out_main)
    write_csv(all_rows, results_dir / "benchmark_8datasets.csv")

    algo_keys = list(ALGORITHMS.keys())
    main_rows = [r for r in all_rows if r["algorithm"] in algo_keys]
    if main_rows:
        rank_r = friedman_ranks(main_rows, "n_reduct_mean", algo_keys)
        rank_t = friedman_ranks(main_rows, "runtime_mean", algo_keys)
        rank_acc = friedman_ranks(main_rows, "knn_mean", algo_keys)
        _log(f"Mean ranks |R|: {rank_r}")
        _log(f"Mean ranks runtime: {rank_t}")
        _log(f"Mean ranks kNN acc: {rank_acc}")
    elapsed = time.perf_counter() - t_all
    _log(
        f"Wrote {out_main} ({elapsed:.1f}s wall time, "
        f"{len(completed)} dataset(s) run this session, "
        f"{len({r['dataset'] for r in main_rows})} total in CSV)"
    )


if __name__ == "__main__":
    main()

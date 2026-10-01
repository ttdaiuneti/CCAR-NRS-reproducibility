"""Leave-one-dataset-out Friedman rank sensitivity for CCAR-E narrative."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CSV = ROOT / "results_v5" / "benchmark_canonical_v5.csv"
OUT = ROOT / "results_v5" / "rank_sensitivity_loo.json"

METHODS = [
    "ccar_exact",
    "ccar_h",
    "core_init_greedy",
    "quick_reduct",
    "nicp",
    "gbnrs",
    "wnrs",
]
LABEL = {
    "ccar_exact": "CCAR-E",
    "ccar_h": "CCAR-H",
    "core_init_greedy": "Core-init",
    "quick_reduct": "QuickReduct",
    "nicp": "NICP",
    "gbnrs": "GBNRS",
    "wnrs": "WNRS",
}


def friedman_ranks(matrix: np.ndarray, higher_better: bool) -> np.ndarray:
    """matrix: (n_datasets, k_methods). Return average ranks (1=best)."""
    n, k = matrix.shape
    ranks = np.zeros(k)
    for i in range(n):
        row = matrix[i]
        # argsort: for lower-better, smaller value -> rank 1
        order = np.argsort(-row if higher_better else row)
        # handle ties by averaging
        vals = -row if higher_better else row
        # scipy-free tie-aware rank
        sorter = np.argsort(vals)
        sorted_vals = vals[sorter]
        dense = np.empty(k, dtype=float)
        i0 = 0
        while i0 < k:
            i1 = i0
            while i1 + 1 < k and sorted_vals[i1 + 1] == sorted_vals[i0]:
                i1 += 1
            avg = 0.5 * ((i0 + 1) + (i1 + 1))
            for j in range(i0, i1 + 1):
                dense[sorter[j]] = avg
            i0 = i1 + 1
        ranks += dense
    return ranks / n


def main() -> None:
    df = pd.read_csv(CSV)
    df = df[df.algorithm.isin(METHODS)].copy()
    datasets = sorted(df.dataset.unique())
    metrics = {
        "runtime": ("runtime_mean", False),
        "|R|": ("n_reduct_mean", False),
        "kNN": ("knn_mean", True),
    }

    # Full ranks
    full = {}
    for mname, (col, higher) in metrics.items():
        mat = np.zeros((len(datasets), len(METHODS)))
        for i, d in enumerate(datasets):
            sub = df[df.dataset == d].set_index("algorithm")
            for j, a in enumerate(METHODS):
                mat[i, j] = float(sub.loc[a, col])
        full[mname] = {
            LABEL[a]: float(r) for a, r in zip(METHODS, friedman_ranks(mat, higher))
        }

    # LOO
    loo = {}
    for hold in datasets:
        keep = [d for d in datasets if d != hold]
        entry = {}
        for mname, (col, higher) in metrics.items():
            mat = np.zeros((len(keep), len(METHODS)))
            for i, d in enumerate(keep):
                sub = df[df.dataset == d].set_index("algorithm")
                for j, a in enumerate(METHODS):
                    mat[i, j] = float(sub.loc[a, col])
            ranks = friedman_ranks(mat, higher)
            entry[mname] = {LABEL[a]: float(r) for a, r in zip(METHODS, ranks)}
        loo[hold] = entry

    # Summarise CCAR-E rank stability
    summary = {}
    for mname in metrics:
        ccar_ranks = [loo[d][mname]["CCAR-E"] for d in datasets]
        summary[mname] = {
            "full_rank_CCAR_E": full[mname]["CCAR-E"],
            "loo_min": float(min(ccar_ranks)),
            "loo_max": float(max(ccar_ranks)),
            "loo_mean": float(np.mean(ccar_ranks)),
            "best_method_counts": {},
        }
        # how often CCAR-E is best (rank closest to 1 among methods) under LOO
        wins = 0
        for d in datasets:
            ranks = loo[d][mname]
            if ranks["CCAR-E"] <= min(ranks.values()) + 1e-9:
                wins += 1
        summary[mname]["CCAR_E_best_or_tied_loo"] = wins

    out = {"full_average_ranks": full, "leave_one_out": loo, "summary": summary}
    OUT.write_text(json.dumps(out, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()

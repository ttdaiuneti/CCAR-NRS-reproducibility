"""Friedman + Nemenyi post-hoc tests (Demsar 2006) for multi-dataset benchmarks."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from scipy import stats

# Table 5(a), Demsar (2006): two-tailed Nemenyi, alpha=0.05
NEMENYI_Q_005: dict[int, float] = {
    2: 1.960,
    3: 2.343,
    4: 2.569,
    5: 2.728,
    6: 2.850,
    7: 2.949,
    8: 3.031,
    9: 3.102,
    10: 3.164,
}

ALGO_ORDER = [
    "ccar_exact",
    "ccar_h",
    "core_init_greedy",
    "quick_reduct",
    "gbnrs",
    "wnrs",
]
ALGO_LABEL = {
    "ccar_exact": "CCAR-E",
    "ccar_h": "CCAR-H",
    "core_init_greedy": "Core-init",
    "quick_reduct": "QuickReduct",
    "gbnrs": "GBNRS",
    "wnrs": "WNRS",
}
METRICS: dict[str, dict] = {
    "n_reduct_mean": {
        "label": "$|R|$",
        "direction": "min",
        "cd_key": "reduct",
    },
    "runtime_mean": {
        "label": "runtime",
        "direction": "min",
        "cd_key": "runtime",
    },
    "knn_mean": {
        "label": "kNN accuracy",
        "direction": "max",
        "cd_key": "knn",
    },
}


@dataclass
class MetricTestResult:
    metric: str
    label: str
    direction: str
    n_datasets: int
    n_methods: int
    friedman_chi2: float
    friedman_p: float
    avg_ranks: dict[str, float]
    cd: float
    q_alpha: float
    alpha: float
    cliques: list[list[str]]  # groups not significantly different


def _rank_matrix(
    values: np.ndarray,
    *,
    direction: Literal["min", "max"],
) -> np.ndarray:
    """Per-row ranks (1=best), tie-aware average ranks."""
    n_rows, k = values.shape
    ranks = np.zeros_like(values, dtype=float)
    for i in range(n_rows):
        row = values[i]
        order = np.argsort(row) if direction == "min" else np.argsort(-row)
        sorted_vals = row[order]
        j = 0
        while j < k:
            t = j
            while t + 1 < k and sorted_vals[t + 1] == sorted_vals[j]:
                t += 1
            avg_rank = 0.5 * (j + t) + 1.0
            for p in range(j, t + 1):
                ranks[i, order[p]] = avg_rank
            j = t + 1
    return ranks


def nemenyi_cd(k: int, n_datasets: int, *, alpha: float = 0.05) -> tuple[float, float]:
    if alpha != 0.05:
        raise ValueError("Only alpha=0.05 q-table is bundled")
    if k not in NEMENYI_Q_005:
        raise ValueError(f"k={k} not in Nemenyi table")
    q = NEMENYI_Q_005[k]
    cd = q * math.sqrt(k * (k + 1) / (6.0 * n_datasets))
    return cd, q


def _bron_kerbosch(
    R: set[str],
    P: set[str],
    X: set[str],
    adj: dict[str, set[str]],
    cliques: list[set[str]],
) -> None:
    """Bron–Kerbosch with pivoting for maximal clique enumeration."""
    if not P and not X:
        cliques.append(R)
        return
    # Pivot: choose vertex with max |P ∩ N(u)| to reduce branching
    pivot = max(P | X, key=lambda u: len(P & adj[u]), default=None)
    if pivot is None:
        candidates = P
    else:
        candidates = P - adj[pivot]
    for v in list(candidates):
        nbrs = adj[v]
        _bron_kerbosch(
            R | {v},
            P & nbrs,
            X & nbrs,
            adj,
            cliques,
        )
        P.remove(v)
        X.add(v)


def nemenyi_cliques(avg_ranks: dict[str, float], cd: float, borderline_tol: float = 0.0) -> list[list[str]]:
    """Maximal cliques on the non-significant graph (|rank_i - rank_j| <= CD + tol).

    Args:
        borderline_tol: Additional tolerance for treating pairs as non-significant.
                        Use 0.06 to treat ΔR−CD < 0.06 as "not decisive" (borderline).

    Returns maximal cliques as lists of algorithm keys, each sorted by rank.
    Bars in CD diagrams may overlap (standard Demšar convention).
    """
    algos = list(avg_ranks.keys())
    threshold = cd + borderline_tol + 1e-12
    # Build adjacency: edge exists if NOT significantly different (within threshold)
    adj: dict[str, set[str]] = {a: set() for a in algos}
    for i, a in enumerate(algos):
        for b in algos[i + 1 :]:
            if abs(avg_ranks[a] - avg_ranks[b]) <= threshold:
                adj[a].add(b)
                adj[b].add(a)
    # Find all maximal cliques
    cliques: list[set[str]] = []
    _bron_kerbosch(set(), set(algos), set(), adj, cliques)
    # Sort each clique by rank and then by algorithm order for stability
    def sort_key(clique):
        return tuple(sorted(clique, key=lambda a: (avg_ranks[a], algos.index(a))))
    cliques = [sort_key(c) for c in cliques]
    # Sort cliques: larger first, then by first element rank
    cliques.sort(key=lambda c: (-len(c), avg_ranks[c[0]]))
    return [list(c) for c in cliques]


def build_value_matrix(
    rows: list[dict],
    metric: str,
    datasets: list[str],
    algos: list[str],
) -> np.ndarray:
    mat = np.zeros((len(datasets), len(algos)))
    for i, ds in enumerate(datasets):
        for j, algo in enumerate(algos):
            row = next(
                r for r in rows if r["dataset"] == ds and r["algorithm"] == algo
            )
            mat[i, j] = float(row[metric])
    return mat


def analyze_metric(
    rows: list[dict],
    metric: str,
    *,
    datasets: list[str] | None = None,
    algos: list[str] | None = None,
    alpha: float = 0.05,
) -> MetricTestResult:
    meta = METRICS[metric]
    algos = algos or list(ALGO_ORDER)
    datasets = datasets or sorted({r["dataset"] for r in rows})
    mat = build_value_matrix(rows, metric, datasets, algos)
    k = len(algos)
    n = len(datasets)

    rank_mat = _rank_matrix(mat, direction=meta["direction"])
    avg = {algos[j]: float(rank_mat[:, j].mean()) for j in range(k)}

    samples = [mat[:, j] for j in range(k)]
    chi2, p = stats.friedmanchisquare(*samples)
    cd, q = nemenyi_cd(k, n, alpha=alpha)
    cliques = nemenyi_cliques(avg, cd)

    return MetricTestResult(
        metric=metric,
        label=meta["label"],
        direction=meta["direction"],
        n_datasets=n,
        n_methods=k,
        friedman_chi2=float(chi2),
        friedman_p=float(p),
        avg_ranks=avg,
        cd=cd,
        q_alpha=q,
        alpha=alpha,
        cliques=cliques,
    )


def analyze_all(rows: list[dict]) -> dict[str, MetricTestResult]:
    main = [r for r in rows if r["algorithm"] in ALGO_ORDER]
    return {m: analyze_metric(main, m) for m in METRICS}


def result_to_dict(res: MetricTestResult) -> dict:
    return {
        "metric": res.metric,
        "label": res.label,
        "friedman_chi2": res.friedman_chi2,
        "friedman_p": res.friedman_p,
        "cd": res.cd,
        "q_alpha": res.q_alpha,
        "avg_ranks": {ALGO_LABEL[k]: v for k, v in res.avg_ranks.items()},
        "cliques": [[ALGO_LABEL[a] for a in g] for g in res.cliques],
    }


def significant_pairs(res: MetricTestResult) -> list[tuple[str, str, float]]:
    """Pairs with |rank_i - rank_j| >= CD (significantly different)."""
    pairs = []
    algos = list(res.avg_ranks.keys())
    for i, a in enumerate(algos):
        for b in algos[i + 1 :]:
            diff = abs(res.avg_ranks[a] - res.avg_ranks[b])
            if diff >= res.cd - 1e-12:
                pairs.append((ALGO_LABEL[a], ALGO_LABEL[b], diff))
    return sorted(pairs, key=lambda t: -t[2])


def write_latex_summary(results: dict[str, MetricTestResult], path: Path) -> None:
    n_ds = next(iter(results.values())).n_datasets
    lines = [
        "% Auto-generated Friedman/Nemenyi summary",
        "\\begin{table}[t]",
        "  \\centering",
        f"  \\caption{{Friedman test ($\\alpha=0.05$) and Nemenyi critical difference (CD) over {n_ds} datasets.}}",
        "  \\label{tab:friedman-nemenyi}",
        "  \\small",
        "  \\begin{tabular}{lrrrr}",
        "    \\toprule",
        "    Metric & $\\chi^2_F$ & $p$-value & CD & Significant? \\\\",
        "    \\midrule",
    ]
    for key in METRICS:
        r = results[key]
        sig = "yes" if r.friedman_p < 0.05 else "no"
        lines.append(
            f"    {r.label} & {r.friedman_chi2:.3f} & {r.friedman_p:.4f} & "
            f"{r.cd:.3f} & {sig} \\\\"
        )
    lines.extend(
        [
            "    \\bottomrule",
            "  \\end{tabular}",
            "\\end{table}",
            "",
            "% Average ranks (lower is better for $|R|$ and runtime; higher for accuracy)",
            "\\begin{table*}[t]",
            "  \\centering",
            f"  \\caption{{Average ranks (Friedman) over {n_ds} datasets.}}",
            "  \\label{tab:avg-ranks}",
            "  \\small",
            "  \\begin{tabular}{l" + "c" * len(ALGO_ORDER) + "}",
            "    \\toprule",
            "    Metric & " + " & ".join(ALGO_LABEL[a] for a in ALGO_ORDER) + " \\\\",
            "    \\midrule",
        ]
    )
    order = list(ALGO_ORDER)
    for key in METRICS:
        r = results[key]
        vals = " & ".join(f"{r.avg_ranks[a]:.2f}" for a in order)
        lines.append(f"    {r.label} & {vals} \\\\")
    lines.extend(["    \\bottomrule", "  \\end{tabular}", "\\end{table*}"])
    path.write_text("\n".join(lines) + "\n")

    pairs_path = path.parent / "exp_nemenyi_pairs.tex"
    prose = ["% Auto-generated Nemenyi pairwise notes"]
    for key in METRICS:
        r = results[key]
        pairs = significant_pairs(r)
        if not pairs:
            prose.append(
                f"% {r.label}: no pairwise difference exceeds CD={r.cd:.2f}"
            )
            continue
        items = ", ".join(f"{a} vs.\\ {b} ($\\Delta R={d:.2f}$)" for a, b, d in pairs[:6])
        prose.append(f"% {r.label}: {items}")
    pairs_path.write_text("\n".join(prose) + "\n")


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    csv_path = root / "results" / "benchmark_merged_with_wnrs.csv"
    if not csv_path.is_file():
        csv_path = root / "results" / "benchmark_20datasets.csv"
    if not csv_path.is_file():
        csv_path = root / "results" / "benchmark_8datasets.csv"
    import csv

    rows = list(csv.DictReader(csv_path.open()))
    results = analyze_all(rows)

    out_json = root / "results" / "friedman_nemenyi.json"
    payload = {k: result_to_dict(v) for k, v in results.items()}
    out_json.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {out_json}")

    for gen_dir in (root / "paper" / "generated", root / "eswa-submission" / "generated"):
        gen_dir.mkdir(parents=True, exist_ok=True)
        write_latex_summary(results, gen_dir / "exp_stats.tex")

    from experiments.export_pgfplots_data import export_cd_diagrams, export_runtime_scatter

    for gen_dir in (root / "paper" / "generated", root / "eswa-submission" / "generated"):
        export_cd_diagrams(out_json, gen_dir / "cd_panels.tex")
    export_runtime_scatter(
        csv_path,
        root / "paper" / "figures" / "data" / "runtime_scatter.dat",
    )
    export_runtime_scatter(
        csv_path,
        root / "eswa-submission" / "figures" / "data" / "runtime_scatter.dat",
    )
    print("Wrote pgfplots data (cd_panels.tex, runtime_scatter.dat)")

    for k, r in results.items():
        print(
            f"{r.label}: chi2={r.friedman_chi2:.3f} p={r.friedman_p:.4f} "
            f"CD={r.cd:.3f} ranks={{{', '.join(f'{ALGO_LABEL[a]}:{r.avg_ranks[a]:.2f}' for a in ALGO_ORDER)}}}"
        )


if __name__ == "__main__":
    main()

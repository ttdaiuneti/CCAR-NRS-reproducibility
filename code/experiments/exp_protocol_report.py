"""Generate protocol, reduct-validity, and Wilcoxon LaTeX fragments for §5."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from scipy import stats

from experiments.statistical_tests import ALGO_LABEL, ALGO_ORDER

CANONICAL_CSV = "benchmark_merged_with_wnrs.csv"

ALGO_DISPLAY = {
    "ccar_exact": "CCAR-E",
    "ccar_h": "CCAR-H",
    "ccar_surrogate": "CCAR-Surrogate",
    "core_init_greedy": "Core-init-Greedy",
    "quick_reduct": "QuickReduct-NRS",
    "gbnrs": "GBNRS",
    "wnrs": "WNRS",
}

METRICS = [
    ("n_reduct_mean", "$|R|$", "less"),
    ("runtime_mean", "runtime (s)", "less"),
    ("knn_mean", "$k$NN accuracy", "greater"),
]

WILCOXON_PAIRS = [
    ("ccar_exact", "quick_reduct", "CCAR-E", "QuickReduct"),
    ("ccar_exact", "core_init_greedy", "CCAR-E", "Core-init"),
]


def _gen_dirs(root: Path) -> list[Path]:
    return [
        root / "paper" / "generated",
        root / "eswa-submission" / "generated",
    ]


def load_rows(root: Path) -> list[dict]:
    path = root / "results" / CANONICAL_CSV
    if not path.is_file():
        raise FileNotFoundError(path)
    return [r for r in csv.DictReader(path.open()) if r.get("algorithm") != "_ablation"]


def write_all(gen_dirs: list[Path], name: str, content: str) -> None:
    for d in gen_dirs:
        d.mkdir(parents=True, exist_ok=True)
        (d / name).write_text(content)


def protocol_table() -> str:
    lines = [
        "% Auto-generated benchmark protocol",
        "\\begin{table}[t]",
        "  \\centering",
        "  \\caption{Unified benchmark protocol (all methods, $N{=}18$ datasets).}",
        "  \\label{tab:exp-protocol}",
        "  \\small",
        "  \\begin{tabular}{@{}p{0.28\\linewidth}p{0.62\\linewidth}@{}}",
        "    \\toprule",
        "    Component & Setting \\\\",
        "    \\midrule",
        "    Cross-validation & Five stratified folds; \\texttt{random\\_state}$=42$; no per-dataset tuning of CCAR variants. \\\\",
        "    Neighborhood & $\\delta=0.1$ on Min--Max normalized features (Hu-style NRS). \\\\",
        "    CCAR / Core-init / QuickReduct / WNRS & Min--Max to $[0,1]$ fit on each training fold; $\\gamma_\\delta$ and reduct checks on training data. \\\\",
        "    GBNRS & \\texttt{StandardScaler} on training features; granular-ball purity $0.9$ (implementation in \\texttt{gbnrs\\_baseline.py}). \\\\",
        "    Post-reduction metrics & $|R|$; $k$NN ($k{=}3$) and SVM-RBF ($C{=}1$, \\texttt{gamma=scale}) on the same reduct within each fold. \\\\",
        "    Aggregation & Per-dataset mean $\\pm$ std over folds; Friedman/Nemenyi over dataset means ($k{=}7$). \\\\",
        "    Canonical results & \\texttt{results/benchmark\\_merged\\_with\\_wnrs.csv} (126 rows $=$ $18\\times 7$, regenerated from unified run). \\\\",
        "    \\bottomrule",
        "  \\end{tabular}",
        "\\end{table}",
    ]
    return "\n".join(lines) + "\n"


def validity_table(rows: list[dict]) -> str:
    by_algo: dict[str, list[dict]] = {a: [] for a in ALGO_ORDER}
    for r in rows:
        if r["algorithm"] in by_algo:
            by_algo[r["algorithm"]].append(r)

    lines = [
        "% Auto-generated reduct validity (training-fold gamma check)",
        "\\begin{table}[t]",
        "  \\centering",
        "  \\caption{Training-fold reduct validity ($\\gamma_\\delta(R)\\geq\\gamma_\\delta(\\C)$ on the training split).}",
        "  \\label{tab:reduct-validity}",
        "  \\small",
        "  \\begin{threeparttable}",
        "  \\begin{tabular}{lcc}",
        "    \\toprule",
        "    Method & Valid / 18 & Invalid datasets \\\\",
        "    \\midrule",
    ]
    for algo in ALGO_ORDER:
        subset = by_algo[algo]
        invalid = [
            r["dataset"]
            for r in subset
            if str(r.get("valid_reduct", "True")).lower() in ("false", "0")
        ]
        n_valid = 18 - len(invalid)
        inv_txt = ", ".join(invalid) if invalid else "---"
        lines.append(f"    {ALGO_DISPLAY[algo]} & {n_valid}/18 & {inv_txt} \\\\")
    lines.extend(
        [
            "    \\bottomrule",
            "  \\end{tabular}",
            "  \\begin{tablenotes}[flushleft]",
            "    \\small",
            "    \\item Validity is reported exactly from the unified benchmark logs with the same $\\gamma_\\delta$ check across methods.",
            "    Non-CCAR baselines can violate dependency preservation on some datasets; we keep these rows for transparent comparison.",
            "  \\end{tablenotes}",
            "  \\end{threeparttable}",
            "\\end{table}",
        ]
    )
    return "\n".join(lines) + "\n"


def _paired_values(
    rows: list[dict], algo_a: str, algo_b: str, metric: str
) -> tuple[np.ndarray, np.ndarray]:
    by_ds: dict[str, dict[str, float]] = {}
    for r in rows:
        if r["algorithm"] not in (algo_a, algo_b):
            continue
        if str(r.get("valid_reduct", "True")).lower() in ("false", "0"):
            continue
        try:
            val = float(r[metric])
        except (TypeError, ValueError):
            continue
        by_ds.setdefault(r["dataset"], {})[r["algorithm"]] = val
    common = sorted(d for d, m in by_ds.items() if algo_a in m and algo_b in m)
    a = np.array([by_ds[d][algo_a] for d in common])
    b = np.array([by_ds[d][algo_b] for d in common])
    return a, b


def wilcoxon_table(rows: list[dict]) -> str:
    lines = [
        "% Auto-generated Wilcoxon signed-rank (paired over datasets)",
        "\\begin{table}[t]",
        "  \\centering",
        "  \\caption{Wilcoxon signed-rank tests on per-dataset means (two-sided; $N$ = overlapping valid datasets).}",
        "  \\label{tab:wilcoxon}",
        "  \\small",
        "  \\begin{tabular}{llrrrr}",
        "    \\toprule",
        "    Comparison & Metric & $N$ & $p$-value & $r$ & Significant? \\\\",
        "    \\midrule",
    ]
    for algo_a, algo_b, lab_a, lab_b in WILCOXON_PAIRS:
        for metric, label, alt in METRICS:
            a, b = _paired_values(rows, algo_a, algo_b, metric)
            if len(a) < 3:
                continue
            diff = a - b
            stat = stats.wilcoxon(a, b, alternative="two-sided")
            n = len(a)
            z = stat.zstatistic if hasattr(stat, "zstatistic") else None
            if z is None and stat.statistic is not None:
                z = stats.norm.isf(stat.pvalue / 2) * (1 if diff.mean() > 0 else -1)
            r_eff = abs(z) / np.sqrt(n) if z is not None else float("nan")
            sig = "yes" if stat.pvalue < 0.05 else "no"
            lines.append(
                f"    {lab_a} vs.\\ {lab_b} & {label} & {n} & {stat.pvalue:.4f} & "
                f"{r_eff:.2f} & {sig} \\\\"
            )
    lines.extend(["    \\bottomrule", "  \\end{tabular}", "\\end{table}"])
    return "\n".join(lines) + "\n"


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    rows = load_rows(root)
    gen_dirs = _gen_dirs(root)

    write_all(gen_dirs, "exp_protocol.tex", protocol_table())
    write_all(gen_dirs, "exp_validity.tex", validity_table(rows))
    write_all(gen_dirs, "exp_wilcoxon.tex", wilcoxon_table(rows))

    for d in gen_dirs:
        print(f"Wrote {d}/exp_protocol.tex, exp_validity.tex, exp_wilcoxon.tex")


if __name__ == "__main__":
    main()

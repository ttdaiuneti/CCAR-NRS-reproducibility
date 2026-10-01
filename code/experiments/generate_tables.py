"""Generate LaTeX table fragments from benchmark_8datasets.csv."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Literal

import numpy as np
from sklearn.model_selection import StratifiedKFold

from experiments.datasets import BENCHMARK_DATASETS, MAX_SAMPLES, dataset_shape
from experiments.run_benchmark import (
    N_SPLITS,
    RANDOM_STATE,
    fold_test_scores,
)
from experiments.statistical_tests import analyze_metric

def tie_aware_avg_ranks(rows: list[dict], metric: str) -> dict[str, float]:
    """Friedman average ranks (tie-aware), consistent with statistical_tests / tab:avg-ranks."""
    from experiments.statistical_tests import (
        ALGO_ORDER as STAT_ALGOS,
        _rank_matrix,
        build_value_matrix,
    )

    if metric in ("n_reduct_mean", "runtime_mean", "knn_mean"):
        return analyze_metric(rows, metric).avg_ranks
    datasets = sorted({r["dataset"] for r in rows if r.get("algorithm") != "_ablation"})
    mat = build_value_matrix(rows, metric, datasets, STAT_ALGOS)
    direction = "max" if metric.endswith("_mean") and "reduct" not in metric else "min"
    rank_mat = _rank_matrix(mat, direction=direction)
    return {STAT_ALGOS[j]: float(rank_mat[:, j].mean()) for j in range(len(STAT_ALGOS))}

ALGO_ORDER = [
    "ccar_exact",
    "ccar_h",
    "core_init_greedy",
    "quick_reduct",
    "gbnrs",
    "wnrs",
]
DATASET_ORDER = [d.key for d in BENCHMARK_DATASETS]
DISPLAY = {d.key: d.display_name for d in BENCHMARK_DATASETS}


def datasets_in_benchmark(rows: list[dict]) -> list[str]:
    present = {r["dataset"] for r in rows if r.get("algorithm") != "_ablation"}
    return [d for d in DATASET_ORDER if d in present]
ALGO_HEADER = {
    "ccar_exact": "CCAR-E",
    "ccar_h": "CCAR-H",
    "core_init_greedy": "Core-init",
    "quick_reduct": "QR",
    "gbnrs": "GBNRS",
    "wnrs": "WNRS",
}
BASELINE = "ccar_exact"
ACC_DECIMALS = 2  # Tables 3--4 (main kNN / SVM): 2 decimal places for readability


def load_csv(path: Path) -> list[dict]:
    with path.open() as f:
        return list(csv.DictReader(f))


def _f(row: dict, key: str) -> float:
    v = row.get(key, "")
    if v in ("", "None", None):
        return float("nan")
    return float(v)


def _row(rows: list[dict], ds: str, algo: str) -> dict:
    return next(x for x in rows if x["dataset"] == ds and x["algorithm"] == algo)


def acc_full_scores(ds_key: str, clf: str = "knn") -> tuple[float, float]:
    """5-fold CV accuracy on all features (no reduction), for classifier `clf`.

    NOTE: prior to the 2026-08 revision this function always returned the kNN
    scores regardless of the caller, which silently populated the SVM table's
    Acc_full column with kNN accuracies. Always pass `clf` explicitly.
    """
    info = next(d for d in BENCHMARK_DATASETS if d.key == ds_key)
    X, y = info.loader()
    p = X.shape[1]
    reduct = list(range(p))
    skf = StratifiedKFold(
        n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE
    )
    knn_scores: list[float] = []
    svm_scores: list[float] = []
    for train_idx, test_idx in skf.split(X, y):
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_te, y_te = X[test_idx], y[test_idx]
        s = fold_test_scores(X_tr, y_tr, X_te, y_te, reduct)
        knn_scores.append(s["knn"])
        svm_scores.append(s["svm"])
    scores = knn_scores if clf == "knn" else svm_scores
    return float(np.mean(scores)), float(np.std(scores))


def rank_styles(
    values: list[tuple[str, float]],
    *,
    minimize: bool,
) -> dict[str, str | None]:
    clean = [(k, v) for k, v in values if v == v]
    if not clean:
        return {k: None for k, _ in values}
    ordered = sorted(clean, key=lambda t: t[1])
    if not minimize:
        ordered = list(reversed(ordered))
    best_val = ordered[0][1]
    eps = 1e-9
    best_keys = {k for k, v in ordered if abs(v - best_val) <= eps}
    rest = [(k, v) for k, v in ordered if k not in best_keys]
    second_keys: set[str] = set()
    if rest:
        second_val = rest[0][1]
        second_keys = {k for k, v in rest if abs(v - second_val) <= eps}
    out: dict[str, str | None] = {}
    for k, _ in values:
        if k in best_keys:
            out[k] = "best"
        elif k in second_keys:
            out[k] = "second"
        else:
            out[k] = None
    return out


def wrap_cell(text: str, style: str | None) -> str:
    if style == "best":
        return f"\\textbf{{{text}}}"
    if style == "second":
        return f"\\underline{{{text}}}"
    return text


def fmt_pm(mean: float, std: float, decimals: int = 4) -> str:
    if mean != mean:
        return "---"
    return f"{mean:.{decimals}f}\\,{{\\scriptsize$\\pm$}}\\,{std:.{decimals}f}"


def fmt_r(mean: float, std: float = 0.0) -> str:
    if mean != mean:
        return "---"
    m = int(round(mean))
    if std > 0.05:
        return f"{m}"
    return str(m)


def wtl_vs_baseline(
    rows: list[dict],
    metric: str,
    *,
    minimize: bool,
    baseline: str = BASELINE,
    datasets: list[str] | None = None,
) -> dict[str, str]:
    """Win / tie / loss counts vs baseline over datasets."""
    ds_order = datasets or datasets_in_benchmark(rows)
    counts = {a: [0, 0, 0] for a in ALGO_ORDER if a != baseline}
    for ds in ds_order:
        b_val = _f(_row(rows, ds, baseline), metric)
        if b_val != b_val:
            continue
        for a in counts:
            v = _f(_row(rows, ds, a), metric)
            if v != v:
                continue
            if abs(v - b_val) <= 1e-9:
                counts[a][1] += 1
            elif (v < b_val) if minimize else (v > b_val):
                counts[a][0] += 1
            else:
                counts[a][2] += 1
    return {a: f"{w}/{t}/{l}" for a, (w, t, l) in counts.items()}


def table_combined_tabular(
    rows: list[dict],
    clf: Literal["knn", "svm"],
    datasets: list[str] | None = None,
) -> str:
    """Full tabular: header, dataset rows, summary (Average, rank, W/T/L)."""
    ds_order = datasets or datasets_in_benchmark(rows)
    key_m, key_s = f"{clf}_mean", f"{clf}_std"
    cols = "l r " + " | ".join(["c"] * len(ALGO_ORDER))
    lines = [
        f"\\begin{{tabular}}{{{cols}}}",
        "  \\toprule",
        "  Dataset & Acc\\textsubscript{full}"
        + " & "
        + " & ".join(
            f"\\multicolumn{{1}}{{c}}{{{ALGO_HEADER[a]}}}" for a in ALGO_ORDER
        )
        + " \\\\",
    ]
    lines.append("  \\midrule")
    acc_full: dict[str, tuple[float, float]] = {
        ds: acc_full_scores(ds, clf) for ds in ds_order
    }

    for ds in ds_order:
        af_m, af_s = acc_full[ds]
        parts = [DISPLAY[ds], fmt_pm(af_m, af_s, decimals=ACC_DECIMALS)]

        a_vals = [(a, _f(_row(rows, ds, a), key_m)) for a in ALGO_ORDER]
        a_styles = rank_styles(a_vals, minimize=False)

        for a in ALGO_ORDER:
            r = _row(rows, ds, a)
            m, s = _f(r, key_m), _f(r, key_s)
            parts.append(
                wrap_cell(fmt_pm(m, s, decimals=ACC_DECIMALS), a_styles[a])
            )

        lines.append("  " + " & ".join(parts) + " \\\\")

    # --- summary rows ---
    avg_acc = {a: np.mean([_f(_row(rows, ds, a), key_m) for ds in ds_order]) for a in ALGO_ORDER}
    af_avg = np.mean([acc_full[ds][0] for ds in ds_order])

    rank_a = tie_aware_avg_ranks(rows, key_m)
    wtl_a = wtl_vs_baseline(rows, key_m, minimize=False, datasets=ds_order)

    avg_line = [
        f"  \\multicolumn{{1}}{{l}}{{Average$^\\dagger$}}",
        f"{af_avg:.{ACC_DECIMALS}f}",
    ]
    for a in ALGO_ORDER:
        avg_line.append(f"{avg_acc[a]:.{ACC_DECIMALS}f}")
    lines.append(" & ".join(avg_line) + " \\\\")

    rank_line = [f"  \\multicolumn{{1}}{{l}}{{Avg.\\ rank$^{{\\ddagger}}$}}", ""]
    for a in ALGO_ORDER:
        rank_line.append(f"\\multicolumn{{1}}{{c}}{{{rank_a[a]:.2f}}}")
    lines.append(" & ".join(rank_line) + " \\\\")

    wtl_line = [
        f"  \\multicolumn{{1}}{{l}}{{W/T/L vs.\\ {ALGO_HEADER[BASELINE]}}}",
        "",
    ]
    for a in ALGO_ORDER:
        if a == BASELINE:
            wtl_line.append("\\multicolumn{1}{c}{---}")
        else:
            wtl_line.append(f"\\multicolumn{{1}}{{c}}{{{wtl_a[a]}}}")
    lines.append(" & ".join(wtl_line) + " \\\\")
    lines.extend(["  \\bottomrule", "\\end{tabular}"])
    return "\n".join(lines)


def write_main_table(
    rows: list[dict],
    clf: Literal["knn", "svm"],
    path: Path,
    *,
    table_num_label: str,
    datasets: list[str] | None = None,
) -> None:
    ds_order = datasets or datasets_in_benchmark(rows)
    n_ds = len(ds_order)
    clf_name = "$k$NN ($k{=}3$)" if clf == "knn" else "SVM-RBF"
    tabular = table_combined_tabular(rows, clf, datasets=ds_order)
    tex = "\n".join(
        [
            "% Auto-generated — main results table (Information Sciences style)",
            "\\begin{table*}[t]",
            "  \\centering",
            f"  \\caption{{Comparison on {n_ds} UCI benchmarks ({clf_name}, stratified 5-fold CV, $\\delta=0.1$). "
            "Acc\\textsubscript{full}: test accuracy using all features. "
            "Bold: best per dataset; underline: second-best. "
            "QR: QuickReduct-NRS. "
            f"W/T/L: win/tie/loss vs.\\ {ALGO_HEADER[BASELINE]} over datasets.}}",
            f"  \\label{{{table_num_label}}}",
            "  \\footnotesize",
            "  \\setlength{\\tabcolsep}{2.5pt}",
            "  \\renewcommand{\\arraystretch}{1.02}",
            "  \\begin{threeparttable}",
            "  \\adjustbox{max width=\\dimexpr2\\textwidth+\\columnsep-4pt\\relax,center}{%",
            tabular,
            "  }%",
            "  \\begin{tablenotes}[flushleft]",
            "    \\footnotesize",
            f"    \\item[$^\\dagger$] Mean over {n_ds} datasets.",
            "    \\item[$^{\\ddagger}$] Tie-aware average Friedman rank (consistent with Table~\\ref{tab:avg-ranks}; higher is better).",
            "  \\end{tablenotes}",
            "  \\end{threeparttable}",
            "\\end{table*}",
        ]
    )
    path.write_text(tex + "\n")


def write_reduct_size_table(
    rows: list[dict],
    path: Path,
    *,
    table_num_label: str,
    datasets: list[str] | None = None,
) -> None:
    ds_order = datasets or datasets_in_benchmark(rows)
    n_ds = len(ds_order)
    tabular = table_reduct_size(rows, datasets=ds_order)
    tex = "\n".join(
        [
            "% Auto-generated — reduct size table",
            "\\begin{table*}[t]",
            "  \\centering",
            f"  \\caption{{Reduct size comparison on {n_ds} UCI benchmarks (stratified 5-fold CV, $\\delta=0.1$). "
            "$|\\C|$: number of condition attributes. "
            "Bold: best per dataset; underline: second-best. "
            "QR: QuickReduct-NRS. "
            f"W/T/L: win/tie/loss vs.\\ {ALGO_HEADER[BASELINE]} over datasets.}}",
            f"  \\label{{{table_num_label}}}",
            "  \\footnotesize",
            "  \\setlength{\\tabcolsep}{2.5pt}",
            "  \\renewcommand{\\arraystretch}{1.02}",
            "  \\begin{threeparttable}",
            "  \\adjustbox{max width=\\dimexpr2\\textwidth+\\columnsep-4pt\\relax,center}{%",
            tabular,
            "  }%",
            "  \\begin{tablenotes}[flushleft]",
            "    \\footnotesize",
            f"    \\item[$^\\dagger$] Mean over {n_ds} datasets.",
            "    \\item[$^{\\ddagger}$] Tie-aware average Friedman rank (consistent with Table~\\ref{tab:avg-ranks}; lower is better).",
            "  \\end{tablenotes}",
            "  \\end{threeparttable}",
            "\\end{table*}",
        ]
    )
    path.write_text(tex + "\n")


def table_reduct_size(rows: list[dict], datasets: list[str] | None = None) -> str:
    """Reduct size table with grouped columns (|R|, mean over folds)."""
    ds_order = datasets or datasets_in_benchmark(rows)
    cols = "l r " + " | ".join(["c"] * len(ALGO_ORDER))
    lines = [
        f"\\begin{{tabular}}{{{cols}}}",
        "  \\toprule",
        "  Dataset & \\multicolumn{1}{c|}{$|\\C|$} & "
        + " & ".join(f"\\multicolumn{{1}}{{c}}{{{ALGO_HEADER[a]}}}" for a in ALGO_ORDER)
        + " \\\\",
        "  \\midrule",
    ]
    for ds in ds_order:
        _, _, p = dataset_shape(ds)
        vals = [(a, _f(_row(rows, ds, a), "n_reduct_mean")) for a in ALGO_ORDER]
        styles = rank_styles(vals, minimize=True)
        parts = [DISPLAY[ds], str(p)]
        for a in ALGO_ORDER:
            r = _row(rows, ds, a)
            nr = _f(r, "n_reduct_mean")
            nr_std = _f(r, "n_reduct_std")
            parts.append(wrap_cell(fmt_r(nr, nr_std), styles[a]))
        lines.append("  " + " & ".join(parts) + " \\\\")
    
    # Summary rows
    avg_r = {a: np.mean([_f(_row(rows, ds, a), "n_reduct_mean") for ds in ds_order]) for a in ALGO_ORDER}
    rank_r = tie_aware_avg_ranks(rows, "n_reduct_mean")
    wtl_r = wtl_vs_baseline(rows, "n_reduct_mean", minimize=True, datasets=ds_order)
    
    avg_line = [f"  \\multicolumn{{1}}{{l}}{{Average$^\\dagger$}}", ""]
    for a in ALGO_ORDER:
        avg_line.append(f"{avg_r[a]:.1f}")
    lines.append(" & ".join(avg_line) + " \\\\")
    
    rank_line = [f"  \\multicolumn{{1}}{{l}}{{Avg.\\ rank$^{{\\ddagger}}$}}", ""]
    for a in ALGO_ORDER:
        rank_line.append(f"\\multicolumn{{1}}{{c}}{{{rank_r[a]:.2f}}}")
    lines.append(" & ".join(rank_line) + " \\\\")
    
    wtl_line = [
        f"  \\multicolumn{{1}}{{l}}{{W/T/L vs.\\ {ALGO_HEADER[BASELINE]}}}",
        "",
    ]
    for a in ALGO_ORDER:
        if a == BASELINE:
            wtl_line.append("\\multicolumn{1}{c}{---}")
        else:
            wtl_line.append(f"\\multicolumn{{1}}{{c}}{{{wtl_r[a]}}}")
    lines.append(" & ".join(wtl_line) + " \\\\")
    
    lines.extend(["  \\bottomrule", "\\end{tabular}"])
    return "\n".join(lines)


def table_runtime_compact(rows: list[dict], datasets: list[str] | None = None) -> str:
    """Runtime table with grouped columns (seconds, mean over folds)."""
    ds_order = datasets or datasets_in_benchmark(rows)
    cols = "l r " + " | ".join(["c"] * len(ALGO_ORDER))
    lines = [
        f"\\begin{{tabular}}{{{cols}}}",
        "  \\toprule",
        "  Dataset & \\multicolumn{1}{c|}{$|\\U|$} & "
        + " & ".join(f"\\multicolumn{{1}}{{c}}{{{ALGO_HEADER[a]}}}" for a in ALGO_ORDER)
        + " \\\\",
        "  \\midrule",
    ]
    for ds in ds_order:
        n, _, _ = dataset_shape(ds)
        vals = [(a, _f(_row(rows, ds, a), "runtime_mean")) for a in ALGO_ORDER]
        styles = rank_styles(vals, minimize=True)
        parts = [DISPLAY[ds], str(n)]
        for a in ALGO_ORDER:
            v = _f(_row(rows, ds, a), "runtime_mean")
            parts.append(wrap_cell(f"{v:.3f}", styles[a]))
        lines.append("  " + " & ".join(parts) + " \\\\")
    rank_line = ["  \\multicolumn{1}{l}{Avg.\\ rank}", ""] + [
        f"\\multicolumn{{1}}{{c}}{{{tie_aware_avg_ranks(rows, 'runtime_mean')[a]:.2f}}}"
        for a in ALGO_ORDER
    ]
    lines.append(" & ".join(rank_line) + " \\\\")
    lines.extend(["  \\bottomrule", "\\end{tabular}"])
    return "\n".join(lines)


def _dataset_group(n: int) -> str:
    if n <= 400:
        return "small"
    if n <= 1000:
        return "medium"
    return "large"


GROUP_LABEL = {
    "small": "Small-scale",
    "medium": "Medium-scale",
    "large": "Large-scale",
}


def table_datasets(delta: float = 0.1) -> str:
    from ccar.core import compute_core

    grouped: dict[str, list[tuple[str, int, int, int, float]]] = {
        "small": [],
        "medium": [],
        "large": [],
    }
    for d in BENCHMARK_DATASETS:
        X, y = d.loader()
        n, p, k = dataset_shape(d.key)
        _, gamma = compute_core(X, y, delta)
        cap = MAX_SAMPLES.get(d.key)
        cap_mark = f"$^{{\\ddagger}}$" if cap and n >= cap else ""
        grouped[_dataset_group(n)].append(
            (f"{d.display_name}{cap_mark}", n, p, k, gamma)
        )

    body_parts: list[str] = []
    for gkey in ("small", "medium", "large"):
        rows = grouped[gkey]
        if not rows:
            continue
        label = GROUP_LABEL[gkey]
        if len(rows) == 1:
            name, n, p, k, gamma = rows[0]
            body_parts.append(
                f"    \\multicolumn{{1}}{{l}}{{{label}}} & {name} & {n} & {p} & {k} & {gamma:.3f} \\\\"
            )
        else:
            name0, n0, p0, k0, g0 = rows[0]
            body_parts.append(
                f"    \\multirow{{{len(rows)}}}{{*}}{{{label}}} & {name0} & {n0} & {p0} & {k0} & {g0:.3f} \\\\"
            )
            for name, n, p, k, gamma in rows[1:]:
                body_parts.append(
                    f"    & {name} & {n} & {p} & {k} & {gamma:.3f} \\\\"
                )
        body_parts.append("    \\midrule")

    if body_parts and body_parts[-1].strip() == "\\midrule":
        body_parts.pop()
    body = "\n".join(body_parts)
    cap_note = ", ".join(f"{k} ($n\\leq${v})" for k, v in sorted(MAX_SAMPLES.items()))
    return "\n".join(
        [
            "% Auto-generated dataset summary (KNOSYS-style groups)",
            "\\begin{table}[t]",
            "  \\centering",
            "  \\caption{UCI benchmark datasets ($\\delta=0.1$, full-data $\\gamma_\\delta(\\C,\\D)$). "
            f"Stratified subsample caps$^{{\\ddagger}}$: {cap_note}.}}",
            "  \\label{tab:datasets}",
            "  \\small",
            "  \\setlength{\\tabcolsep}{5pt}",
            "  \\begin{tabular}{llrrrr}",
            "    \\toprule",
            "    Group & Dataset & $|\\U|$ & $|\\C|$ & $|\\D|$ & $\\gamma_\\delta(\\C,\\D)$ \\\\",
            "    \\midrule",
            body,
            "    \\bottomrule",
            "  \\end{tabular}",
            "\\end{table}",
        ]
    )


def table_ablation(rows: list[dict], datasets: list[str] | None = None) -> str:
    ds_order = datasets or datasets_in_benchmark(rows)
    body = []
    for ds in ds_order:
        r = next(x for x in rows if x["dataset"] == ds and x["algorithm"] == "_ablation")
        pr = _f(r, "pair_ratio")
        nc = int(_f(r, "n_core"))
        sp = _f(r, "speedup_pair_vs_full")
        sp_txt = f"{sp:.1f}$\\times$" if sp == sp and sp > 0.05 else "---"
        rho_txt = f"{pr:.2e}" if pr < 0.01 else f"{pr:.4f}"
        body.append(f"    {DISPLAY[ds]} & {nc} & {rho_txt} & {sp_txt} \\\\")
    return "\n".join(
        [
            "% Auto-generated ablation table (non-floating; stays with §5.5 text)",
            "\\begin{center}",
            "  \\small",
            "  \\setlength{\\tabcolsep}{4pt}",
            "  \\renewcommand{\\arraystretch}{0.95}",
            "  \\captionof{table}{Conflicting-pair ablation ($\\delta=0.1$, full data). "
            "$\\rho=|P_\\delta|/\\binom{|\\U|}{2}$; Speedup $=|M|_{\\mathrm{proxy}}/|P_\\delta|$ "
            "(proxy cross-class scan vs.\\ pair-only enumeration).}",
            "  \\label{tab:ablation-pairs}",
            "  \\begin{tabular}{@{}lrrr@{}}",
            "    \\toprule",
            "    Dataset & $|\\CORE|$ & $\\rho$ & Speedup \\\\",
            "    \\midrule",
            "\n".join(body),
            "    \\bottomrule",
            "  \\end{tabular}",
            "\\end{center}",
        ]
    )


def ranks_paragraph(rows: list[dict]) -> str:
    short = ALGO_HEADER
    parts = []
    for label, metric, hib in (
        ("$|R|$", "n_reduct_mean", False),
        ("runtime", "runtime_mean", False),
        ("$k$NN accuracy", "knn_mean", True),
    ):
        ranks = tie_aware_avg_ranks(rows, metric)
        ordered = sorted(ranks.items(), key=lambda t: t[1])
        txt = ", ".join(f"{short[a]}={r:.2f}" for a, r in ordered)
        parts.append(f"{label}: {txt}")
    return " ".join(parts) + "."


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    results = root / "results"
    out_dirs = [root / "paper" / "generated", root / "eswa-submission" / "generated"]
    for d in out_dirs:
        d.mkdir(parents=True, exist_ok=True)
    bench_csv = results / "benchmark_merged_with_wnrs.csv"
    if not bench_csv.is_file():
        bench_csv = results / "benchmark_18_full.csv"
    if not bench_csv.is_file():
        bench_csv = results / "benchmark_8datasets.csv"
    rows = load_csv(bench_csv)
    main_rows = [r for r in rows if r["algorithm"] != "_ablation"]
    
    # Load ablation rows from original file (not in merged)
    old_csv = results / "benchmark_20datasets.csv"
    if old_csv.is_file():
        old_rows = load_csv(old_csv)
        ab_rows = [r for r in old_rows if r["algorithm"] == "_ablation" and r["dataset"] != "spambase"]
    else:
        ab_rows = []
    datasets = datasets_in_benchmark(main_rows)
    print(f"Benchmark CSV: {bench_csv.name} ({len(datasets)} datasets)")

    runtime_block = "\n".join(
        [
            "% Auto-generated runtime table",
            "\\begin{table*}[t]",
            "  \\centering",
            "  \\caption{Reduction runtime (seconds, mean over folds, $\\delta=0.1$). "
            "Bold: fastest per dataset; underline: second-fastest.}",
            "  \\label{tab:runtime}",
            "  \\small",
            "  \\setlength{\\tabcolsep}{5pt}",
            table_runtime_compact(main_rows, datasets=datasets),
            "\\end{table*}",
        ]
    )

    files = {
        "exp_table_datasets.tex": table_datasets(),
        "exp_table_ablation.tex": table_ablation(ab_rows, datasets=datasets),
        "exp_table_runtime.tex": runtime_block,
        "exp_ranks.txt": ranks_paragraph(main_rows),
    }

    for paper_gen in out_dirs:
        write_main_table(
            main_rows,
            "knn",
            paper_gen / "exp_table_knn.tex",
            table_num_label="tab:main-knn",
            datasets=datasets,
        )
        write_main_table(
            main_rows,
            "svm",
            paper_gen / "exp_table_svm.tex",
            table_num_label="tab:main-svm",
            datasets=datasets,
        )
        write_reduct_size_table(
            main_rows,
            paper_gen / "exp_table_reduct.tex",
            table_num_label="tab:reduct-size",
            datasets=datasets,
        )
        for name, body in files.items():
            (paper_gen / name).write_text(body.rstrip() + "\n")
            print(f"Wrote {paper_gen / name}")


if __name__ == "__main__":
    main()

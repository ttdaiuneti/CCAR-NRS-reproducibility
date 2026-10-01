"""Per-fold CORE_delta stability for ESWA domain / interpretability subsection."""

from __future__ import annotations

import json
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import StratifiedKFold

from ccar.core import compute_core
from ccar.nrs import normalize_features
from experiments.datasets import BENCHMARK_DATASETS

DELTA = 0.1
N_SPLITS = 5
RANDOM_STATE = 42
STABLE_FOLD_FRAC = 0.8  # attribute in >= 4/5 folds

# UCI-style labels for domain narrative (German: Statlog credit coding scheme).
GERMAN_ATTR = [
    "checking status",
    "duration",
    "credit history",
    "purpose",
    "credit amount",
    "savings",
    "employment",
    "installment rate",
    "personal status",
    "other debtors",
    "residence since",
    "property",
    "age",
    "other plans",
    "housing",
    "existing credits",
    "job",
    "people liable",
    "telephone",
    "foreign worker",
]

HEART_ATTR = [
    "age",
    "sex",
    "chest pain type",
    "resting BP",
    "cholesterol",
    "fasting BS",
    "resting ECG",
    "max heart rate",
    "exercise angina",
    "ST depression",
    "slope",
    "vessels",
    "thalassemia",
]


def fold_core_sets(X: np.ndarray, y: np.ndarray, delta: float) -> list[set[int]]:
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    cores: list[set[int]] = []
    for train_idx, _ in skf.split(X, y):
        Xn = normalize_features(X[train_idx])
        core_idx, _ = compute_core(Xn, y[train_idx], delta)
        cores.append(set(core_idx))
    return cores


def mean_pairwise_jaccard(sets: list[set[int]]) -> float:
    if len(sets) < 2:
        return 1.0
    vals = []
    for a, b in combinations(sets, 2):
        u = a | b
        vals.append(len(a & b) / len(u) if u else 1.0)
    return float(np.mean(vals))


def stable_core(sets: list[set[int]]) -> set[int]:
    need = int(np.ceil(STABLE_FOLD_FRAC * len(sets)))
    universe = set().union(*sets) if sets else set()
    return {i for i in universe if sum(1 for s in sets if i in s) >= need}


def attr_names_for(key: str, n_attr: int) -> list[str]:
    if key == "wdbc":
        return list(load_breast_cancer().feature_names)
    if key == "credit":
        return GERMAN_ATTR[:n_attr]
    if key == "heart":
        return HEART_ATTR[:n_attr]
    return [f"a{i + 1}" for i in range(n_attr)]


def summarize_dataset(key: str) -> dict:
    info = next(d for d in BENCHMARK_DATASETS if d.key == key)
    X, y = info.loader()
    p = X.shape[1]
    cores = fold_core_sets(X, y, DELTA)
    sizes = [len(c) for c in cores]
    stab = stable_core(cores)
    names = attr_names_for(key, p)
    return {
        "dataset": key,
        "display_name": info.display_name,
        "n_features": p,
        "core_sizes": sizes,
        "core_size_mean": float(np.mean(sizes)),
        "core_size_std": float(np.std(sizes)),
        "jaccard_mean": mean_pairwise_jaccard(cores),
        "stable_core_indices": sorted(stab),
        "stable_core_names": [names[i] for i in sorted(stab)],
        "fold_cores": [sorted(c) for c in cores],
    }


def write_latex_table(summaries: list[dict], path: Path) -> None:
    # Three domain-focused rows + summary row for all 18
    domain_keys = ["wdbc", "credit", "heart"]
    domain_rows = [s for s in summaries if s["dataset"] in domain_keys]
    all_j = [s["jaccard_mean"] for s in summaries]
    nonempty = [s for s in summaries if s["core_size_mean"] > 0]

    lines = [
        "% Auto-generated — CORE_delta stability across 5-fold training splits",
        "\\begin{table*}[t]",
        "  \\centering",
        "  \\caption{Stability of $\\CORE_\\delta$ across stratified folds ($\\delta=0.1$, Min--Max on training split). "
        "Jaccard: mean pairwise similarity of fold-wise cores; "
        "Stable core: attributes in $\\geq 4/5$ folds.}",
        "  \\label{tab:core-stability}",
        "  \\small",
        "  \\setlength{\\tabcolsep}{4pt}",
        "  \\begin{tabular}{@{}lrrrr@{}}",
        "    \\toprule",
        "    Domain / dataset & $|\\C|$ & $\\overline{|\\CORE|}$ & Jaccard & $|\\mathrm{Stable}|$ \\\\",
        "    \\midrule",
    ]
    domain_labels = {
        "wdbc": "Medical (WDBC)",
        "credit": "Credit (German)",
        "heart": "Medical (Heart)",
    }
    for s in domain_rows:
        lines.append(
            f"    {domain_labels[s['dataset']]} & {s['n_features']} & "
            f"{s['core_size_mean']:.1f} $\\pm$ {s['core_size_std']:.1f} & "
            f"{s['jaccard_mean']:.2f} & {len(s['stable_core_indices'])} \\\\"
        )
    lines.append("    \\midrule")
    lines.append(
        f"    All 18 benchmarks (mean) & --- & "
        f"{np.mean([s['core_size_mean'] for s in summaries]):.1f} & "
        f"{np.mean(all_j):.2f} & --- \\\\"
    )
    if nonempty:
        lines.append(
            f"    Datasets with $\\CORE_\\delta\\neq\\emptyset$ ({len(nonempty)}/18) & --- & "
            f"{np.mean([s['core_size_mean'] for s in nonempty]):.1f} & "
            f"{np.mean([s['jaccard_mean'] for s in nonempty]):.2f} & --- \\\\"
        )
    lines.extend(
        [
            "    \\bottomrule",
            "  \\end{tabular}",
            "\\end{table*}",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def _tex_label(name: str, max_len: int = 22) -> str:
    s = name.replace("_", " ")
    if len(s) > max_len:
        s = s[: max_len - 1] + "."
    return s.replace("&", "\\&")


def write_wdbc_stability_figure(s: dict, tex_path: Path, data_path: Path) -> None:
    """Bar chart: how often each attribute appears in fold-wise CORE_delta (WDBC)."""
    if s["dataset"] != "wdbc":
        return
    names_all = attr_names_for("wdbc", s["n_features"])
    cnt: Counter[str] = Counter()
    for fold in s["fold_cores"]:
        for i in fold:
            cnt[names_all[i]] += 1
    top = cnt.most_common(10)
    if not top:
        return

    data_path.parent.mkdir(parents=True, exist_ok=True)
    with data_path.open("w") as f:
        f.write("# idx count stable(1 if >=4 folds) label\n")
        for idx, (name, count) in enumerate(top):
            stable = 1 if count >= 4 else 0
            f.write(f"{idx} {count} {stable} {name.replace(' ', '_')}\n")

    labels = ",".join(_tex_label(n) for n, _ in top)

    def _coord(name: str, count: int) -> str:
        return f"{{{_tex_label(name)}}},{count}"

    stable_coords = "\n      ".join(
        f"({_coord(n, c)})" for n, c in top if c >= 4
    )
    other_coords = "\n      ".join(
        f"({_coord(n, c)})" for n, c in top if c < 4
    )

    tex = f"""% Auto-generated — WDBC fold-wise CORE_delta frequency (see core_stability_study.py)
\\begin{{figure*}}[t]
  \\centering
  \\ifpfgplotsloaded
  \\begin{{tikzpicture}}
    \\begin{{axis}}[
      width=0.88\\textwidth,
      height=5.8cm,
      ybar,
      bar width=0.62cm,
      ymin=0,
      ymax=5.5,
      ylabel={{Fold count (out of 5)}},
      symbolic x coords={{{labels}}},
      xtick=data,
      x tick label style={{rotate=40, anchor=east, font=\\scriptsize}},
      ytick={{0,1,2,3,4,5}},
      enlarge x limits=0.12,
      legend style={{font=\\footnotesize, at={{(0.5,0.97)}}, anchor=north, draw=none}},
      grid=major,
      grid style={{dashed,gray!25}},
    ]
      \\addplot[fill=orange!70!white, draw=orange!80!black] coordinates {{
      {other_coords}
      }};
      \\addplot[fill=blue!55!white, draw=blue!70!black] coordinates {{
      {stable_coords}
      }};
      \\legend{{Occasional ($<4$ folds), Stable core ($\\geq 4$ folds)}}
    \\end{{axis}}
  \\end{{tikzpicture}}
  \\else
  \\fbox{{\\parbox{{0.9\\linewidth}}{{\\centering Run \\texttt{{python -m experiments.core\\_stability\\_study}}.}}}}
  \\fi
  \\caption{{WDBC: frequency of each attribute in $\\CORE_\\delta$ across five stratified training folds ($\\delta=0.1$). Blue bars form the audit-ready stable core; orange bars are fold-specific.}}
  \\label{{fig:wdbc-core-stability}}
\\end{{figure*}}
"""
    tex_path.write_text(tex)


def write_wdbc_detail(s: dict, path: Path) -> str:
    """Return LaTeX fragment listing stable WDBC attributes."""
    if s["dataset"] != "wdbc":
        return ""
    names = s["stable_core_names"]
    if not names:
        # Report most frequent across folds
        from collections import Counter

        cnt: Counter[str] = Counter()
        names_all = attr_names_for("wdbc", s["n_features"])
        for fold in s["fold_cores"]:
            for i in fold:
                cnt[names_all[i]] += 1
        top = [n for n, c in cnt.most_common(5) if c >= 3]
        names = top
    short = ", ".join(
        n.replace("_", "\\_") for n in names[:8]
    )
    if len(names) > 8:
        short += ", \\ldots"
    return short


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    out_dirs = [
        root / "results",
        root / "eswa-submission" / "generated",
        root / "paper" / "generated",
    ]
    summaries = [summarize_dataset(d.key) for d in BENCHMARK_DATASETS]
    wdbc = next(s for s in summaries if s["dataset"] == "wdbc")
    credit = next(s for s in summaries if s["dataset"] == "credit")
    heart = next(s for s in summaries if s["dataset"] == "heart")

    payload = {
        "delta": DELTA,
        "n_splits": N_SPLITS,
        "summaries": summaries,
        "wdbc_stable_names": wdbc["stable_core_names"],
        "credit_stable_names": credit["stable_core_names"],
        "heart_stable_names": heart["stable_core_names"],
    }
    for d in out_dirs:
        d.mkdir(parents=True, exist_ok=True)
    (root / "results" / "core_stability.json").write_text(
        json.dumps(payload, indent=2) + "\n"
    )

    tex_path = root / "eswa-submission" / "generated" / "exp_core_stability.tex"
    write_latex_table(summaries, tex_path)
    (root / "paper" / "generated" / "exp_core_stability.tex").write_text(
        tex_path.read_text()
    )

    fig_dir = root / "eswa-submission" / "figures"
    write_wdbc_stability_figure(
        wdbc,
        fig_dir / "wdbc_core_stability.tex",
        fig_dir / "data" / "wdbc_core_fold_frequency.dat",
    )
    paper_fig = root / "paper" / "figures"
    if paper_fig.parent.is_dir():
        paper_fig.mkdir(parents=True, exist_ok=True)
        (paper_fig / "wdbc_core_stability.tex").write_text(
            (fig_dir / "wdbc_core_stability.tex").read_text()
        )

    wdbc_tex = write_wdbc_detail(wdbc, tex_path)
    print("WDBC stable / frequent attributes:", wdbc["stable_core_names"] or wdbc_tex)
    print("Credit stable:", credit["stable_core_names"])
    print("Heart stable:", heart["stable_core_names"])
    print(f"Wrote {tex_path}")
    for s in summaries:
        print(
            f"  {s['display_name']:18} |CORE|={s['core_size_mean']:.1f} "
            f"J={s['jaccard_mean']:.2f} stable={len(s['stable_core_indices'])}"
        )


if __name__ == "__main__":
    main()

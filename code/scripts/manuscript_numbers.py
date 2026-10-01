"""Compute every benchmark-derived number printed in the manuscript from one CSV.

Usage: python scripts/manuscript_numbers.py <benchmark.csv> [--perfold-dir DIR]

Prints Friedman/Nemenyi statistics, average ranks, Holm-corrected Wilcoxon
tests, the NICP head-to-head, consistency strata, leave-one-dataset-out rank
bands, per-method cost-exponent fits and the LaTeX rows of the validity,
reduct-size and runtime tables. Nothing is typed by hand: the manuscript
numbers are copied from this output.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

METHODS = ["ccar_exact", "ccar_h", "core_init_greedy", "quick_reduct", "nicp", "gbnrs", "wnrs"]
LABEL = {
    "ccar_exact": "CCAR-E", "ccar_h": "CCAR-H", "core_init_greedy": "Core-init",
    "quick_reduct": "QuickReduct", "nicp": "NICP", "gbnrs": "GBNRS", "wnrs": "WNRS",
}
# Row order of the manuscript tables.
DS_ORDER = [
    "wine", "wdbc", "sonar", "ionosphere", "heart", "credit", "diabetes", "iris", "glass",
    "vehicle", "vowel", "australian", "ecoli", "yeast", "dermatology", "segment", "chess", "vote",
]
DS_NAME = {
    "wine": "Wine", "wdbc": "WDBC", "sonar": "Sonar", "ionosphere": "Ionosphere", "heart": "Heart",
    "credit": "German Credit", "diabetes": "Diabetes", "iris": "Iris", "glass": "Glass",
    "vehicle": "Vehicle", "vowel": "Vowel", "australian": "Australian", "ecoli": "Ecoli",
    "yeast": "Yeast", "dermatology": "Dermatology", "segment": "Segment",
    "chess": "Chess (KR-KP)", "vote": "Congressional",
}
Q_ALPHA_05 = {7: 2.949, 5: 2.728, 6: 2.850}
HIGHER_BETTER = {"n_reduct_mean": False, "runtime_mean": False, "knn_mean": True, "svm_mean": True}
METRICS = ["n_reduct_mean", "runtime_mean", "knn_mean", "svm_mean"]


def pivot(df: pd.DataFrame, metric: str, methods=METHODS, datasets=None) -> pd.DataFrame:
    p = df.pivot(index="dataset", columns="algorithm", values=metric)[methods]
    return p.loc[datasets] if datasets is not None else p.loc[[d for d in DS_ORDER if d in p.index]]


def avg_ranks(p: pd.DataFrame, higher_better: bool) -> pd.Series:
    vals = -p.values if higher_better else p.values
    r = np.apply_along_axis(stats.rankdata, 1, vals)
    return pd.Series(r.mean(axis=0), index=p.columns)


def friedman(p: pd.DataFrame, higher_better: bool) -> tuple[float, float, float, pd.Series]:
    n, k = p.shape
    R = avg_ranks(p, higher_better)
    # Tie-corrected statistic (scipy), as reported in the manuscript.
    vals = -p.values if higher_better else p.values
    chi2, pval = stats.friedmanchisquare(*vals.T)
    cd = Q_ALPHA_05[k] * np.sqrt(k * (k + 1) / (6 * n))
    return chi2, pval, cd, R


def holm(pvals: list[float]) -> list[float]:
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    running = 0.0
    for i, idx in enumerate(order):
        running = max(running, (m - i) * pvals[idx])
        adj[idx] = min(1.0, running)
    return adj.tolist()


def wilcoxon(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Two-sided Wilcoxon signed-rank, normal approximation; r = |z|/sqrt(#non-zero diffs)."""
    res = stats.wilcoxon(x, y, method="approx")
    z = abs(stats.norm.isf(res.pvalue / 2))
    return float(res.pvalue), float(z / np.sqrt(int(np.sum(x != y))))


def wtl(a: np.ndarray, b: np.ndarray, decimals: int, higher_better: bool) -> str:
    """W/T/L of method b's CCAR-E comparison, counted from CCAR-E's viewpoint (a)."""
    if decimals is not None:
        a, b = np.round(a, decimals), np.round(b, decimals)
    better = (a > b) if higher_better else (a < b)
    worse = (a < b) if higher_better else (a > b)
    return f"{int(better.sum())}/{int((a == b).sum())}/{int(worse.sum())}"


def cost_fit(df: pd.DataFrame, method: str, n_col: str) -> dict:
    sub = df[df.algorithm == method].set_index("dataset").loc[DS_ORDER]
    X = np.column_stack([np.ones(len(sub)), np.log(sub[n_col]), np.log(sub.n_features)])
    y = np.log(sub.runtime_mean.values)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = len(y) - X.shape[1]
    s2 = resid @ resid / dof
    se = np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X)))
    r2 = 1 - resid @ resid / np.sum((y - y.mean()) ** 2)
    return {"a": beta[0], "b": beta[1], "c": beta[2], "se_b": se[1], "se_c": se[2], "r2": r2}


def mark_best(values: dict[str, float], decimals: int, lower_better: bool, eligible: set[str]) -> dict[str, str]:
    """Bold best / underline second-best among eligible cells, comparing rounded values."""
    fmt = f"{{:.{decimals}f}}"
    rounded = {m: round(v, decimals) for m, v in values.items()}
    distinct = sorted({rounded[m] for m in eligible}, reverse=not lower_better)
    out = {}
    for m, v in values.items():
        s = fmt.format(v)
        if m in eligible and distinct and rounded[m] == distinct[0]:
            s = f"\\textbf{{{s}}}"
        elif m in eligible and len(distinct) > 1 and rounded[m] == distinct[1]:
            s = f"\\underline{{{s}}}"
        out[m] = s
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", type=Path)
    ap.add_argument("--n-col", default="n_samples", help="object count used in the cost fit")
    args = ap.parse_args()

    raw = pd.read_csv(args.csv)
    df = raw[raw.algorithm.isin(METHODS)].copy()
    assert len(df) == 18 * 7, f"expected 126 comparison cells, got {len(df)}"
    assert not df[METRICS + ["valid_reduct"]].isna().any().any(), "missing cells"
    df["valid_reduct"] = df.valid_reduct.astype(str).str.lower() == "true"

    print("=== Friedman / Nemenyi (N=18, k=7) ===")
    ranks = {}
    for m in METRICS:
        chi2, pval, cd, R = friedman(pivot(df, m), HIGHER_BETTER[m])
        ranks[m] = R
        print(f"{m:15s} chi2={chi2:.3f} p={pval:.4g} CD={cd:.3f}  " +
              " ".join(f"{LABEL[a]}={R[a]:.2f}" for a in METHODS))
    rt = ranks["runtime_mean"]
    print("runtime rank gaps from CCAR-E: " + " ".join(f"{LABEL[a]}={rt[a]-rt['ccar_exact']:.2f}" for a in METHODS[1:]))

    print("\n=== CCAR-E vs six baselines, Wilcoxon, Holm over 6 (per metric) ===")
    for m in METRICS:
        p = pivot(df, m)
        res = [wilcoxon(p.ccar_exact.values, p[b].values) for b in METHODS[1:]]
        adj = holm([r[0] for r in res])
        print(f"{m:15s} " + " ".join(f"{LABEL[b]}: p={r[0]:.4f} pH={h:.3f} r={r[1]:.2f}"
                                      for b, r, h in zip(METHODS[1:], res, adj)))
    print("mean runtime (s): " + " ".join(f"{LABEL[a]}={pivot(df,'runtime_mean')[a].mean():.3f}" for a in METHODS))

    print("\n=== CCAR-E vs NICP, Holm over 4 metrics ===")
    dec = {"n_reduct_mean": 1, "runtime_mean": None, "knn_mean": 2, "svm_mean": 2}
    res = []
    for m in METRICS:
        p = pivot(df, m)
        res.append((m, *wilcoxon(p.ccar_exact.values, p.nicp.values),
                    p.ccar_exact.mean(), p.nicp.mean(),
                    wtl(p.ccar_exact.values, p.nicp.values, dec[m], HIGHER_BETTER[m])))
    adj = holm([r[1] for r in res])
    for (m, pv, r, me, mn, w), h in zip(res, adj):
        print(f"{m:15s} E={me:.3f} NICP={mn:.3f} p={pv:.4f} pH={h:.4f} r={r:.2f} W/T/L={w}")
    prt = pivot(df, "runtime_mean")
    ratio = prt.nicp / prt.ccar_exact
    print(f"median NICP/E speedup={ratio.median():.4f}; E faster on {(ratio>1).sum()}/18; "
          f"losses: {list(ratio[ratio<1].index)}")

    print("\n=== Validity (all folds) ===")
    for a in METHODS:
        sub = df[df.algorithm == a].set_index("dataset").loc[DS_ORDER]
        bad = [DS_NAME[d] for d in DS_ORDER if not sub.loc[d, "valid_reduct"]]
        extra = ""
        if "n_folds_valid" in sub:
            extra = "  folds valid: " + " ".join(f"{d}={int(sub.loc[d,'n_folds_valid'])}" for d in DS_ORDER if not sub.loc[d, "valid_reduct"])
        print(f"    {LABEL[a]} & {18-len(bad)}/18 & {', '.join(bad) if bad else '---'} \\\\{extra}")
    valid_all = [a for a in METHODS if df[df.algorithm == a].valid_reduct.all()]
    print("18/18-valid methods:", [LABEL[a] for a in valid_all])

    print("\n=== Validity-conditioned Friedman on |R| (18/18-valid methods only) ===")
    k = len(valid_all)
    if k in Q_ALPHA_05:
        chi2, pval, cd, R = friedman(pivot(df, "n_reduct_mean", methods=valid_all), False)
        print(f"k={k} chi2={chi2:.3f} p={pval:.4g} CD={cd:.3f} " + " ".join(f"{LABEL[a]}={R[a]:.2f}" for a in valid_all))
    # Alternative: invalid cells ranked last (ties among invalid averaged).
    p = pivot(df, "n_reduct_mean")
    v = df.pivot(index="dataset", columns="algorithm", values="valid_reduct")[METHODS].loc[p.index]
    pen = p.where(v, np.inf)
    chi2, pval, cd, R = friedman(pen, False)
    print(f"invalid-last k=7 chi2={chi2:.3f} p={pval:.4g} CD={cd:.3f} " + " ".join(f"{LABEL[a]}={R[a]:.2f}" for a in METHODS))

    print("\n=== Consistency strata ===")
    g = raw[raw.algorithm == "ccar_exact"].set_index("dataset").gamma_full
    strata = {"consistent": [d for d in DS_ORDER if g[d] >= 1 - 1e-12],
              "inconsistent": [d for d in DS_ORDER if g[d] < 1 - 1e-12]}
    for name, ds in strata.items():
        pr = pivot(df, "n_reduct_mean", datasets=ds)
        chi2, pval, _, R = friedman(pr, False)
        n, kk = pr.shape
        cd = Q_ALPHA_05[kk] * np.sqrt(kk * (kk + 1) / (6 * n))
        chi2t, pvalt, _, Rt = friedman(pivot(df, "runtime_mean", datasets=ds), False)
        print(f"{name} ({len(ds)}): |R| chi2={chi2:.3f} p={pval:.4g} CD={cd:.3f}; runtime chi2={chi2t:.3f} p={pvalt:.3g} E-rank={Rt['ccar_exact']:.2f}")
        for a in METHODS:
            vv = df[(df.algorithm == a) & df.dataset.isin(ds)].valid_reduct.sum()
            print(f"    {LABEL[a]} & {pr[a].mean():.2f} & {R[a]:.2f} & {vv}/{len(ds)}")

    print("\n=== Leave-one-dataset-out rank bands (CCAR-E) ===")
    for m in ["runtime_mean", "n_reduct_mean"]:
        full = pivot(df, m)
        rs, best = [], []
        for d in full.index:
            R = avg_ranks(full.drop(index=d), HIGHER_BETTER[m])
            rs.append(R["ccar_exact"])
            best.append(R["ccar_exact"] <= R.min() + 1e-12)
        print(f"{m}: [{min(rs):.2f},{max(rs):.2f}] best-or-tied in {sum(best)}/18")

    print("\n=== Cost-exponent fits log t = a + b log|U| + c log|C| ===")
    fits = {a: cost_fit(df, a, args.n_col) for a in METHODS}
    for a in METHODS:
        f = fits[a]
        print(f"{LABEL[a]:12s} b={f['b']:.2f}±{f['se_b']:.2f} c={f['c']:.2f}±{f['se_c']:.2f} R2={f['r2']:.2f}")
    e = fits["ccar_exact"]
    print("ratio t_base/t_E at |U|=1000:")
    for a in ["ccar_h", "core_init_greedy", "quick_reduct", "nicp", "wnrs"]:
        f = fits[a]
        r = lambda n, pp: np.exp(f["a"] - e["a"] + (f["b"] - e["b"]) * np.log(n) + (f["c"] - e["c"]) * np.log(pp))
        cross = np.exp(-(f["a"] - e["a"] + (f["b"] - e["b"]) * np.log(1000)) / (f["c"] - e["c"]))
        print(f"  {LABEL[a]:12s} " + " ".join(f"{r(1000,pp):.2f}" for pp in [10, 30, 60, 100, 300, 1000]) +
              f"  crossover |C|≈{cross:.0f}  |C|=40 over |U|=1e3,1e4,1e5: " +
              " ".join(f"{r(n,40):.1f}" for n in [1e3, 1e4, 1e5]))

    print("\n=== Table rows: reduct size (bold among valid cells) ===")
    pr = pivot(df, "n_reduct_mean")
    v = df.pivot(index="dataset", columns="algorithm", values="valid_reduct")
    for d in DS_ORDER:
        elig = {a for a in METHODS if v.loc[d, a]}
        cells = mark_best(pr.loc[d].to_dict(), 1, True, elig)
        p_ = int(raw[raw.dataset == d].n_features.iloc[0])
        print(f"  {DS_NAME[d]} & {p_} & " + " & ".join(cells[a] for a in METHODS) + r" \\")
    avg = mark_best(pr.mean().to_dict(), 1, True, set(valid_all))
    print(r"  \emph{Average} & --- & " + " & ".join(avg[a] for a in METHODS) + r" \\")
    print(r"  \emph{W/T/L} & --- & --- & " + " & ".join(wtl(pr.ccar_exact.values, pr[a].values, 1, False) for a in METHODS[1:]) + r" \\")
    print("mean |R|: " + " ".join(f"{LABEL[a]}={pr[a].mean():.2f}" for a in METHODS))

    for metric, label in [("knn_mean", "kNN"), ("svm_mean", "SVM")]:
        print(f"\n=== Table rows: {label} accuracy (mean \\pm std; Acc_full column not included) ===")
        pm = pivot(df, metric)
        ps = pivot(df, metric.replace("_mean", "_std"))
        for d in DS_ORDER:
            vals = pm.loc[d].to_dict()
            marked = mark_best(vals, 2, False, set(METHODS))
            cells = []
            for a in METHODS:
                cell = f"{vals[a]:.2f}\\,$\\pm$\\,{ps.loc[d, a]:.2f}"
                if marked[a].startswith("\\textbf"):
                    cell = f"\\textbf{{{cell}}}"
                elif marked[a].startswith("\\underline"):
                    cell = f"\\underline{{{cell}}}"
                cells.append(cell)
            print(f"  {DS_NAME[d]} & " + " & ".join(cells) + r" \\")
        avg = mark_best(pm.mean().to_dict(), 2, False, set(METHODS))
        print(r"  \emph{Average} & " + " & ".join(avg[a] for a in METHODS) + r" \\")
        print(r"  \emph{W/T/L} & --- & " + " & ".join(wtl(pm.ccar_exact.values, pm[a].values, None, True) for a in METHODS[1:]) + r" \\")

    print("\n=== Table rows: runtime ===")
    for d in DS_ORDER:
        cells = mark_best(prt.loc[d].to_dict(), 3, True, set(METHODS))
        n_ = int(raw[raw.dataset == d].n_samples.iloc[0])
        print(f"  {DS_NAME[d]} & {n_} & " + " & ".join(cells[a] for a in METHODS) + r" \\")
    avg = mark_best(prt.mean().to_dict(), 3, True, set(METHODS))
    print(r"  \emph{Average} & --- & " + " & ".join(avg[a] for a in METHODS) + r" \\")
    print(r"  \emph{W/T/L} & --- & --- & " + " & ".join(wtl(prt.ccar_exact.values, prt[a].values, None, False) for a in METHODS[1:]) + r" \\")
    eh = prt.ccar_exact <= prt.ccar_h
    print(f"CCAR-E <= CCAR-H on {eh.sum()}/18; CCAR-H faster on {[DS_NAME[d] for d in prt.index[~eh]]}; "
          f"margins ms: {[(DS_NAME[d], round(1000*(prt.ccar_exact[d]-prt.ccar_h[d]),1)) for d in prt.index[~eh]]}")

    print("\n=== runtime_scatter.dat ===")
    print("dataset n ccar-e ccar-h core-init quickreduct nicp gbnrs wnrs label")
    for d in DS_ORDER:
        n_ = int(raw[raw.dataset == d].n_samples.iloc[0])
        print(f"{d} {n_} " + " ".join(f"{prt.loc[d, a]:.3f}" for a in METHODS) + " " + DS_NAME[d].replace(" ", "_").replace("(KR-KP)", "").rstrip("_"))


if __name__ == "__main__":
    main()

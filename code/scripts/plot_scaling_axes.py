"""Figure scaling_axes.png: (a) measured ablation speedup vs |C|; (b) fitted cost ratios.

Usage: python scripts/plot_scaling_axes.py <ablation.csv> <benchmark.csv> <out.png>

(a) reads the `_ablation` rows (core-first pair path vs direct proxy scan);
(b) reads the per-method runtime_mean of the comparison block and fits
log t = a + b log|U| + c log|C| per method (same fit as manuscript_numbers.py).
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from manuscript_numbers import DS_NAME, cost_fit  # noqa: E402

matplotlib.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"],
    "mathtext.fontset": "cm", "axes.unicode_minus": True, "font.size": 9,
})


def style_axes(ax):
    ax.tick_params(direction="in", top=True, right=True, which="both")
    for s in ax.spines.values():
        s.set_visible(True)


def main() -> None:
    abl_csv, bench_csv, out = map(Path, sys.argv[1:4])
    abl = pd.read_csv(abl_csv)
    abl = abl[abl.algorithm == "_ablation"].set_index("dataset")
    bench = pd.read_csv(bench_csv)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.0))

    # (a) measured ablation speedup
    p, sp, core = abl.n_features.values, abl.speedup_pair_vs_full.values, abl.n_core.values
    ne = core > 0
    ax1.axhline(1, ls="--", color="0.5", lw=0.8)
    ax1.scatter(p[ne], sp[ne], s=22, color="#1f5fd6", label=f"core non-empty ({ne.sum()})", zorder=3)
    ax1.scatter(p[~ne], sp[~ne], s=22, facecolors="none", edgecolors="#c8414b", lw=1.1,
                label=f"core empty ({(~ne).sum()})", zorder=3)
    for d in ["iris", "sonar", "diabetes"]:
        ax1.annotate(DS_NAME[d].split(" ")[0], (abl.loc[d, "n_features"], abl.loc[d, "speedup_pair_vs_full"]),
                     xytext=(0, 7), textcoords="offset points", ha="center", fontsize=7, color="0.35")
    rho, pv = stats.spearmanr(p, sp)
    ptxt = f"{pv:.2f}" if pv >= 0.01 else "{}\\times10^{{{}}}".format(*f"{pv:.1e}".split("e"))
    ax1.text(0.04, 0.05, rf"Spearman $\rho={rho:.2f}$, $p={ptxt}$", transform=ax1.transAxes, fontsize=8)
    ax1.set_xscale("log", base=2); ax1.set_yscale("log"); ax1.set_ylim(0.5, 7)
    ax1.set_xticks([4, 8, 16, 32, 64]); ax1.set_xticklabels(["4", "8", "16", "32", "64"])
    ax1.set_yticks([0.6, 1, 2, 3, 5]); ax1.set_yticklabels(["0.6", "1", "2", "3", "5"])
    ax1.set_xlabel(r"number of condition attributes $|C|$")
    ax1.set_ylabel("speedup of core-first\npair enumeration")
    ax1.legend(fontsize=7, frameon=False, loc="upper left")
    ax1.set_title("(a) measured ablation speedup", fontsize=9)
    style_axes(ax1)

    # (b) extrapolated cost ratios at |U| = 1000
    methods = [m for m in ["ccar_exact", "ccar_h", "core_init_greedy", "quick_reduct", "nicp", "wnrs"]]
    df = bench[bench.algorithm.isin(methods)]
    fits = {m: cost_fit(df, m, "n_samples") for m in methods}
    e = fits["ccar_exact"]
    cs = np.logspace(np.log10(8), 3, 200)
    colors = {"ccar_h": "#c8414b", "core_init_greedy": "0.5", "quick_reduct": "#b07a1a",
              "nicp": "#1f5fd6", "wnrs": "#3f8a4f"}
    labels = {"ccar_h": "CCAR-H", "core_init_greedy": "Core-init", "quick_reduct": "QuickReduct",
              "nicp": "NIP-NRS", "wnrs": "WNRS"}
    ax2.axvspan(8, 60, color="0.92", lw=0)
    ax2.text(9, 0.18, "measured\nrange", fontsize=7, color="0.4", va="bottom")
    ax2.axhline(1, ls="--", color="0.5", lw=0.8)
    ends = []
    for m in labels:
        f = fits[m]
        base = f["a"] - e["a"] + (f["b"] - e["b"]) * np.log(1000)
        r = np.exp(base + (f["c"] - e["c"]) * np.log(cs))
        ax2.plot(cs, r, color=colors[m], lw=1.4, ls="--" if m == "core_init_greedy" else "-")
        ends.append([np.log(r[-1]), labels[m], colors[m]])
        if f["c"] != e["c"]:
            cross = np.exp(-base / (f["c"] - e["c"]))
            if 8 <= cross <= 1000:
                ax2.plot(cross, 1, "o", ms=3.5, color=colors[m], mec="white", mew=0.6)
    ends.sort()
    for k in range(1, len(ends)):
        ends[k][0] = max(ends[k][0], ends[k - 1][0] + 0.28)
    for ly, lab, col in ends:
        ax2.text(1050, np.exp(ly), lab, color=col, fontsize=7, va="center")
    ax2.set_xscale("log"); ax2.set_yscale("log")
    ax2.set_xlim(8, 1000); ax2.set_ylim(0.15, 40)
    ax2.set_xticks([10, 30, 100, 300, 1000]); ax2.set_xticklabels(["10", "30", "100", "300", "1000"])
    ax2.set_yticks([0.3, 1, 3, 10, 30]); ax2.set_yticklabels(["0.3", "1", "3", "10", "30"])
    ax2.set_xlabel(r"number of condition attributes $|C|$")
    ax2.set_ylabel(r"cost ratio $t_{\mathrm{baseline}}/t_{\mathrm{CCAR\text{-}E}}$")
    ax2.set_title(r"(b) fitted ratio at $|U|=1000$ ($>1$: CCAR-E faster)", fontsize=9)
    style_axes(ax2)

    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()

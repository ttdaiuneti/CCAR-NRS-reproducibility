"""Figure delta_sensitivity.png from the delta-sweep CSV (run_delta_sweep_v5.py).

Usage: python scripts/plot_delta_sweep.py <delta_sweep.csv> <out.png>

Three panels: mean |R|, gamma_delta(C,D) on the full data, and kNN accuracy
versus delta. Solid blue: tables consistent at delta = 0.1; dashed grey: the
others. The dotted line marks the operating point delta = 0.1.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

matplotlib.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"],
    "mathtext.fontset": "cm", "axes.unicode_minus": True, "font.size": 9,
})


def style_axes(ax):
    ax.tick_params(direction="in", top=True, right=True, which="both")
    for s in ax.spines.values():
        s.set_visible(True)


def main() -> None:
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    df = pd.read_csv(src)
    at01 = df[(df.delta - 0.1).abs() < 1e-9].set_index("dataset").gamma_full
    consistent = set(at01[at01 >= 1 - 1e-12].index)
    names = df.groupby("dataset").dataset_name.first()
    panels = [("n_reduct_mean", r"mean $|R|$", r"(a) reduct size"),
              ("gamma_full", r"$\gamma_\delta(C,D)$ on full data", r"(b) consistency"),
              ("knn_mean", r"$k$NN accuracy", r"(c) $k$NN accuracy")]
    fig, axes = plt.subplots(1, 3, figsize=(7.8, 2.7))
    for ax, (col, ylab, title) in zip(axes, panels):
        ax.axvline(0.1, ls=":", color="#c8414b", lw=1)
        # Offset overlapping end labels.
        ends = []
        for d, g in df.groupby("dataset"):
            g = g.sort_values("delta")
            c = d in consistent
            ax.plot(g.delta, g[col], marker="o", ms=3, lw=1.3,
                    color="#1f5fd6" if c else "0.55", ls="-" if c else "--")
            ends.append([g[col].iloc[-1], names[d].split(" ")[0], "#1f5fd6" if c else "0.45"])
        ends.sort()
        span = ax.get_ylim()[1] - ax.get_ylim()[0]
        for k in range(1, len(ends)):
            ends[k][0] = max(ends[k][0], ends[k - 1][0] + 0.055 * span)
        for yv, lab, colr in ends:
            ax.text(0.31, yv, lab, fontsize=6.5, color=colr, va="center")
        ax.set_xticks([0.05, 0.10, 0.15, 0.20, 0.30])
        ax.set_xticklabels([".05", ".10", ".15", ".20", ".30"], fontsize=7.5)
        ax.set_xlim(0.03, 0.37)
        ax.set_xlabel(r"neighborhood radius $\delta$")
        ax.set_ylabel(ylab)
        ax.set_title(title, fontsize=9)
        style_axes(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=360, bbox_inches="tight")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()

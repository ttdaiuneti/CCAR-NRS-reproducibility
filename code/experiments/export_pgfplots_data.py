"""Export .dat files and TikZ snippets for pgfplots figures in paper/figures/."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from experiments.statistical_tests import (
    ALGO_LABEL,
    ALGO_ORDER,
    METRICS,
    analyze_all,
    nemenyi_cliques,
)

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "paper" / "figures" / "data"
GEN_DIR = ROOT / "paper" / "generated"

DISPLAY_DS = {
    "wine": "Wine",
    "wdbc": "WDBC",
    "sonar": "Sonar",
    "ionosphere": "Ion.",
    "heart": "Heart",
    "credit": "Credit",
    "diabetes": "Diab.",
    "iris": "Iris",
}


def export_runtime_scatter(csv_path: Path, out: Path) -> None:
    rows = list(csv.DictReader(csv_path.open()))
    # Header with all algorithm columns
    header = "dataset n " + " ".join(ALGO_LABEL[a].lower().replace(".", "") for a in ALGO_ORDER) + " label"
    lines = [header]
    order = list(DISPLAY_DS.keys())
    for ds in order:
        n = int(next(r["n_samples"] for r in rows if r["dataset"] == ds))
        values = []
        for algo in ALGO_ORDER:
            try:
                rt = float(
                    next(
                        r["runtime_mean"]
                        for r in rows
                        if r["dataset"] == ds and r["algorithm"] == algo
                    )
                )
            except StopIteration:
                rt = 0.0
            values.append(f"{rt:.6f}")
        lab = DISPLAY_DS[ds].replace(" ", "_")
        lines.append(f"{ds} {n} " + " ".join(values) + f" {lab}")
    out.write_text("\n".join(lines) + "\n")


def _tex_label(name: str) -> str:
    return name.replace(".", "\\.")


def export_cd_panel(
    avg_ranks: dict[str, float],
    cliques: list[list[str]],
    cd: float,
    *,
    title: str,
    lower_better: bool,
    at: str,
) -> str:
    """One pgfplots axis (side-by-side in a single tikzpicture)."""
    ranks = dict(avg_ranks)
    labels = list(ranks.keys())
    ordered = sorted(labels, key=lambda lb: ranks[lb])
    y_coords = {lb: i for i, lb in enumerate(ordered)}
    k = len(ordered)
    xmin = max(0.8, min(ranks.values()) - 0.35)
    xmax = min(float(k) + 0.2, max(ranks.values()) + 0.35)

    lines = [
        f"% CD panel: {title}",
        "\\begin{axis}[",
        f"  at={{{at}}}, anchor=south west,",
        "  width=5.4cm, height=5.0cm,",
        f"  title={{{title}}},",
        f"  xmin={xmin:.3f}, xmax={xmax:.3f},",
        f"  ymin=-0.6, ymax={k - 0.4},",
        "  ytick={%s}," % ",".join(str(i) for i in range(k)),
        "  yticklabels={%s}," % ",".join(ordered),
        "  yticklabel style={font=\\footnotesize},",
        "  xlabel style={font=\\footnotesize},",
        "  title style={font=\\small},",
        "  tick label style={font=\\scriptsize},",
        (
            "  xlabel={Avg.\\ rank (lower better)},"
            if lower_better
            else "  xlabel={Avg.\\ rank (1=best acc.)},"
        ),
        "  xmajorgrids=true,",
        "  grid style={dashed,gray!30},",
        "  axis line style={gray!60},",
        "  clip=false,",
        "]",
    ]
    for lb in ordered:
        y = y_coords[lb]
        r = ranks[lb]
        lines.append(
            f"  \\addplot[only marks, mark=*, mark size=2.2pt, "
            f"draw=blue!70!black, fill=blue!40] coordinates {{({r:.4f},{y})}};"
        )
    bar_y = k - 0.2
    for group in cliques:
        if len(group) < 2:
            continue
        g_labels = sorted(group, key=lambda lb: ranks[lb])
        lo = ranks[g_labels[0]]
        hi = ranks[g_labels[-1]]
        lines.append(
            f"  \\draw[line width=1.6pt] (axis cs:{lo:.4f},{bar_y}) -- "
            f"(axis cs:{hi:.4f},{bar_y});"
        )
        bar_y += 0.32
    mid = sum(ranks.values()) / len(ranks)
    lines.append(
        f"  \\draw[|<->|,gray] (axis cs:{mid:.4f},{k + 0.05}) -- "
        f"(axis cs:{mid + cd:.4f},{k + 0.05}) node[midway,above,font=\\scriptsize] "
        f"{{CD={cd:.2f}}};"
    )
    lines.append("\\end{axis}")
    return "\n".join(lines)


def export_cd_diagrams(json_path: Path, out_path: Path, borderline_tol: float = 0.06) -> None:
    """Generate CD panels with cliques that treat borderline pairs (ΔR−CD < tol) as connected.

    This ensures the visual bars match prose that calls borderline differences "not decisive".
    """
    data = json.loads(json_path.read_text())
    lines = ["% Auto-generated CD panels for pgfplots (Nemenyi)"]
    x_positions = ["(0cm,0cm)", "(5.7cm,0cm)", "(11.4cm,0cm)"]
    for idx, (key, meta) in enumerate(METRICS.items()):
        block = data[key]
        ranks = {lb: block["avg_ranks"][lb] for lb in block["avg_ranks"]}
        # Regenerate cliques with borderline tolerance for consistent visualization
        cd = block["cd"]
        cliques_borderline = nemenyi_cliques(ranks, cd, borderline_tol=borderline_tol)
        lines.append(
            export_cd_panel(
                ranks,
                cliques_borderline,
                cd,
                title=meta["label"],
                lower_better=meta["direction"] == "min",
                at=x_positions[idx],
            )
        )
    out_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    GEN_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = ROOT / "results" / "benchmark_merged_with_wnrs.csv"
    if not csv_path.is_file():
        csv_path = ROOT / "results" / "benchmark_20datasets.csv"
    if not csv_path.is_file():
        csv_path = ROOT / "results" / "benchmark_8datasets.csv"
    json_path = ROOT / "results" / "friedman_nemenyi.json"

    export_runtime_scatter(csv_path, DATA_DIR / "runtime_scatter.dat")
    print(f"Wrote {DATA_DIR / 'runtime_scatter.dat'}")

    if not json_path.is_file():
        rows = list(csv.DictReader(csv_path.open()))
        results = analyze_all(rows)
        payload = {
            k: {
                "label": v.label,
                "cd": v.cd,
                "avg_ranks": {ALGO_LABEL[a]: v.avg_ranks[a] for a in ALGO_ORDER},
            }
            for k, v in results.items()
        }
        json_path.write_text(json.dumps(payload, indent=2))

    export_cd_diagrams(json_path, GEN_DIR / "cd_panels.tex", borderline_tol=0.0)
    print(f"Wrote {GEN_DIR / 'cd_panels.tex'} (strict CD, tol=0)")


if __name__ == "__main__":
    main()

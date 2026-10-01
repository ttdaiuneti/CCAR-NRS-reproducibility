"""Export pgfplots data for runtime scatter (replaces matplotlib PDF)."""

from __future__ import annotations

from experiments.export_pgfplots_data import export_runtime_scatter, main as export_main
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    csv_path = ROOT / "results" / "benchmark_8datasets.csv"
    out = ROOT / "paper" / "figures" / "data" / "runtime_scatter.dat"
    if not csv_path.is_file():
        export_main()
        return
    export_runtime_scatter(csv_path, out)
    print(f"Wrote {out} (use paper/figures/runtime_scatter.tex)")


if __name__ == "__main__":
    main()

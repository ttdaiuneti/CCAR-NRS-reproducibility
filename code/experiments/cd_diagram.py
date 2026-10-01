"""Export pgfplots/TikZ CD diagram data (replaces matplotlib PDF output)."""

from __future__ import annotations

from experiments.export_pgfplots_data import export_cd_diagrams, main as export_main
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    json_path = ROOT / "results" / "friedman_nemenyi.json"
    if not json_path.is_file():
        export_main()
        return
    export_cd_diagrams(json_path, ROOT / "paper" / "generated" / "cd_panels.tex")
    print(f"Wrote {ROOT / 'paper' / 'generated' / 'cd_panels.tex'}")


if __name__ == "__main__":
    main()

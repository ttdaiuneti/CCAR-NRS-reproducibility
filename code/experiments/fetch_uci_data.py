"""Download UCI CSV files into data/uci/ (offline experiment runs)."""

from __future__ import annotations

import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "uci"
BASE = "https://raw.githubusercontent.com/jbrownlee/Datasets/master"


def _curl(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["curl", "-fsSL", "-o", str(dest), url], check=True)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name in ("sonar.csv", "ionosphere.csv", "german.csv", "pima-indians-diabetes.csv"):
        _curl(f"{BASE}/{name}", OUT / name)
    heart_zip = OUT / "heart.zip"
    _curl("https://archive.ics.uci.edu/static/public/45/heart+disease.zip", heart_zip)
    with zipfile.ZipFile(heart_zip) as zf:
        for member in ("processed.cleveland.data", "cleveland.data"):
            if member in zf.namelist():
                zf.extract(member, OUT)
                (OUT / member).rename(OUT / "heart.csv")
                break
    heart_zip.unlink(missing_ok=True)
    print(f"Core UCI files in {OUT}")
    print("Also run: python -m experiments.fetch_extended_uci")


if __name__ == "__main__":
    main()

"""Merge per-dataset *_folds.csv into benchmark_20datasets.csv."""

from __future__ import annotations

import csv
from pathlib import Path

from experiments.datasets import BENCHMARK_DATASETS


def main() -> None:
    results = Path(__file__).resolve().parents[2] / "results"
    keys = [d.key for d in BENCHMARK_DATASETS]
    all_rows: list[dict] = []
    missing: list[str] = []
    for key in keys:
        path = results / f"{key}_folds.csv"
        if not path.is_file():
            missing.append(key)
            continue
        with path.open() as f:
            all_rows.extend(list(csv.DictReader(f)))
    if not all_rows:
        raise SystemExit("No *_folds.csv found in results/")
    out = results / "benchmark_20datasets.csv"
    keys_out: list[str] = []
    seen: set[str] = set()
    for row in all_rows:
        for k in row:
            if k not in seen:
                keys_out.append(k)
                seen.add(k)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys_out, extrasaction="ignore")
        w.writeheader()
        w.writerows(all_rows)
    legacy = results / "benchmark_8datasets.csv"
    legacy.write_text(out.read_text())
    print(f"Merged {len(keys) - len(missing)}/{len(keys)} datasets -> {out}")
    if missing:
        print("Missing:", ", ".join(missing))


if __name__ == "__main__":
    main()

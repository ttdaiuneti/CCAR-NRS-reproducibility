"""Download extended benchmarks into data/uci/ (fast parallel UCI curl, no OpenML).

OpenML via sklearn.fetch_openml is slow or times out on many networks.
This script pulls CSV directly from archive.ics.uci.edu (~1–2 min for 12 sets).
"""

from __future__ import annotations

import csv
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "uci"

# UCI "static/public/{id}/data.csv" (new archive API)
UCI_STATIC: dict[str, int] = {
    "glass": 42,
    "vehicle": 52,
    "australian": 143,
    "ecoli": 40,
    "yeast": 110,
    "dermatology": 33,
    "segment": 147,
    # NOTE: these are UCI *static* archive ids, NOT OpenML data_ids. Static id 3
    # is Annealing, which is how data/uci/chess.csv came to hold the anneal data
    # (BUGFIX_NOTES B8). KR-vs-KP must be fetched from OpenML data_id=3 instead.
    "chess": None,  # fetch via OpenML data_id=3, not the static archive
    "spambase": 94,
    "vote": 105,
}

# Legacy .data paths (not in static CSV or wrong id mapping)
UCI_LEGACY: dict[str, tuple[str, str]] = {
    "vowel": (
        "https://archive.ics.uci.edu/ml/machine-learning-databases/"
        "undocumented/connectionist-bench/vowel/vowel-context.data",
        "vowel",
    ),
    "splice": (
        "https://archive.ics.uci.edu/ml/machine-learning-databases/"
        "molecular-biology/splice-junction-gene-sequences/splice.data",
        "splice",
    ),
}


def _curl(url: str, dest: Path, timeout: int = 120) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["curl", "-fsSL", "--retry", "3", "--connect-timeout", "15", "--max-time", str(timeout), "-o", str(dest), url],
        check=True,
    )


def _write_csv(path: Path, rows: list[list]) -> None:
    with path.open("w", newline="") as f:
        csv.writer(f).writerows(rows)


def _parse_vowel(raw_path: Path, dest: Path) -> None:
    rows: list[list] = []
    with raw_path.open() as f:
        for line in f:
            parts = line.split()
            if len(parts) < 11:
                continue
            *feat, label = parts
            rows.append(feat + [label])
    _write_csv(dest, rows)
    raw_path.unlink(missing_ok=True)


def _parse_splice(raw_path: Path, dest: Path) -> None:
    rows: list[list] = []
    with raw_path.open() as f:
        for line in f:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 3:
                continue
            cls, _name, seq = parts[0], parts[1], parts[2]
            if len(seq) != 60:
                continue
            rows.append(list(seq) + [cls])
    _write_csv(dest, rows)
    raw_path.unlink(missing_ok=True)


def fetch_one(key: str) -> str:
    dest = OUT / f"{key}.csv"
    if dest.is_file() and dest.stat().st_size > 200:
        return f"skip {key}"

    if key in UCI_STATIC:
        tmp = OUT / f".{key}.download.csv"
        url = f"https://archive.ics.uci.edu/static/public/{UCI_STATIC[key]}/data.csv"
        _curl(url, tmp)
        tmp.replace(dest)
        return f"ok {key} (UCI static {UCI_STATIC[key]})"

    if key in UCI_LEGACY:
        url, kind = UCI_LEGACY[key]
        tmp = OUT / f".{key}.raw"
        _curl(url, tmp)
        if kind == "vowel":
            _parse_vowel(tmp, dest)
        elif kind == "splice":
            _parse_splice(tmp, dest)
        else:
            tmp.replace(dest)
        return f"ok {key} (legacy {kind})"

    raise KeyError(key)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    keys = list(UCI_STATIC.keys()) + list(UCI_LEGACY.keys())
    print(f"Fetching {len(keys)} datasets (parallel curl, no OpenML)...")
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = {pool.submit(fetch_one, k): k for k in keys}
        for fut in as_completed(futures):
            key = futures[fut]
            try:
                print(f"  {fut.result()}")
            except Exception as exc:
                print(f"  FAIL {key}: {exc}")
    print(f"Done. Files in {OUT}")


if __name__ == "__main__":
    main()

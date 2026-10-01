"""Print benchmark progress: dataset × algorithm grid + running workers."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from experiments.datasets import BENCHMARK_DATASETS
from experiments.run_benchmark import ALGORITHMS, N_SPLITS


def _load_progress(progress_dir: Path, key: str) -> dict | None:
    path = progress_dir / f"{key}.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def _fmt_cell(
    results_dir: Path,
    progress_dir: Path,
    key: str,
    algo: str,
) -> str:
    prog = _load_progress(progress_dir, key)
    csv_path = results_dir / f"{key}_folds.csv"
    if csv_path.is_file():
        if prog and prog.get("status") == "running":
            return "re-run"
        return "done"

    if not prog:
        return "—"

    status = prog.get("status")
    if status == "done":
        return "done"
    if status == "failed":
        return "FAIL"

    cur_algo = prog.get("algorithm", "")
    if cur_algo != algo:
        algo_order = list(ALGORITHMS.keys())
        if status == "running" and cur_algo in algo_order and algo in algo_order:
            if algo_order.index(algo) < algo_order.index(cur_algo):
                return "done?"
            if algo_order.index(algo) > algo_order.index(cur_algo):
                return "wait"
        return "—"

    fold = prog.get("fold", 0)
    n_folds = prog.get("n_folds", N_SPLITS)
    return f"{fold}/{n_folds}"


def print_status(results_dir: Path) -> None:
    progress_dir = results_dir / ".progress"
    keys = [d.key for d in BENCHMARK_DATASETS]
    algos = list(ALGORITHMS.items())

    name_w = max(len(d.key) for d in BENCHMARK_DATASETS)
    col_w = 9
    header = f"{'dataset':<{name_w}}  " + " ".join(
        f"{label:^{col_w}}" for _, label in algos
    ) + "  status"
    print(header)
    print("-" * len(header))

    n_done = 0
    n_running = 0
    for info in BENCHMARK_DATASETS:
        key = info.key
        prog = _load_progress(progress_dir, key)
        csv_ok = (results_dir / f"{key}_folds.csv").is_file()

        cells = [_fmt_cell(results_dir, progress_dir, key, algo_key) for algo_key, _ in algos]

        if csv_ok and (not prog or prog.get("status") != "running"):
            status = "OK"
            n_done += 1
        elif prog and prog.get("status") == "running":
            cur = prog.get("algorithm_label", prog.get("algorithm", "?"))
            fold = prog.get("fold", "?")
            n_folds = prog.get("n_folds", N_SPLITS)
            status = f"RUN {cur} f{fold}/{n_folds}"
            n_running += 1
        elif prog and prog.get("status") == "failed":
            status = f"FAIL: {prog.get('error', '?')}"
        else:
            status = "pending"

        row = f"{key:<{name_w}}  " + " ".join(
            f"{c:^{col_w}}" for c in cells
        ) + f"  {status}"
        print(row)

    print()
    print(f"Summary: {n_done}/{len(keys)} datasets complete, {n_running} running")

    running_detail = []
    for info in BENCHMARK_DATASETS:
        prog = _load_progress(progress_dir, info.key)
        if prog and prog.get("status") == "running":
            ts = prog.get("updated_at", "")
            running_detail.append(
                f"  • {info.key}: {prog.get('algorithm_label')} "
                f"fold {prog.get('fold')}/{prog.get('n_folds')} "
                f"(updated {ts})"
            )
    if running_detail:
        print("\nCurrently running:")
        print("\n".join(running_detail))

    if not progress_dir.is_dir() or not any(progress_dir.glob("*.json")):
        print(
            "\nNote: no .progress/ yet — this run started before progress tracking."
        )
        print("Rule of thumb: OK = *_folds.csv exists; no csv = still running/pending.")


def main() -> None:
    results_dir = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else Path(__file__).resolve().parents[2] / "results"
    )
    print_status(results_dir)


if __name__ == "__main__":
    main()

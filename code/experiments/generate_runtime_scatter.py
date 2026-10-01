"""Generate runtime scatter data for all 6 methods."""

import csv
from pathlib import Path

def main() -> None:
    results_dir = Path(__file__).resolve().parents[2] / "results"
    bench_csv = results_dir / "benchmark_merged.csv"
    
    with bench_csv.open() as f:
        rows = list(csv.DictReader(f))
    
    # Filter main algorithms only
    main_algos = ["ccar_exact", "ccar_h", "ccar_surrogate", "core_init_greedy", "quick_reduct", "gbnrs"]
    rows = [r for r in rows if r["algorithm"] in main_algos]
    
    # Group by dataset
    datasets = {}
    for r in rows:
        ds = r["dataset"]
        if ds not in datasets:
            datasets[ds] = {}
        datasets[ds][r["algorithm"]] = float(r["runtime_mean"])
    
    # Write scatter data (use CCAR-E as reference)
    out_dir = Path(__file__).resolve().parents[2] / "eswa-submission" / "figures" / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    out_file = out_dir / "runtime_scatter.dat"
    with out_file.open("w") as f:
        f.write("dataset n ccar h surr core quick gbnrs\n")
        for ds in sorted(datasets.keys()):
            data = datasets[ds]
            n = int([r["n_samples"] for r in rows if r["dataset"] == ds][0])
            ccar = data.get("ccar_exact", 0)
            h = data.get("ccar_h", 0)
            surr = data.get("ccar_surrogate", 0)
            core = data.get("core_init_greedy", 0)
            quick = data.get("quick_reduct", 0)
            gbnrs = data.get("gbnrs", 0)
            f.write(f"{ds} {n} {ccar:.3f} {h:.3f} {surr:.3f} {core:.3f} {quick:.3f} {gbnrs:.3f}\n")
    
    print(f"Wrote {out_file}")

if __name__ == "__main__":
    main()

"""Merge new baseline results with existing CCAR-NRS results."""

import csv
from pathlib import Path

def load_csv(csv_path: Path) -> list[dict]:
    with csv_path.open() as f:
        return list(csv.DictReader(f))

def main() -> None:
    results_dir = Path(__file__).resolve().parents[2] / "results"
    
    # Load existing results (CCAR methods + Core-init)
    old_file = results_dir / "benchmark_20datasets.csv"
    old_rows = load_csv(old_file)
    
    # Filter: keep only CCAR-E, CCAR-H, Surr., Core-init, and exclude spambase
    keep_algos = ["ccar_exact", "ccar_h", "ccar_surrogate", "core_init_greedy"]
    filtered_old = [r for r in old_rows if r["algorithm"] in keep_algos and r["dataset"] != "spambase"]
    
    # Load new results (QuickReduct, GBNRS)
    new_file = results_dir / "benchmark_new_baselines_20260526_073243.csv"
    new_rows = load_csv(new_file)
    
    # Normalize column names (n_attrs -> n_features)
    for r in new_rows:
        if "n_attrs" in r:
            r["n_features"] = r["n_attrs"]
            del r["n_attrs"]
        # Add missing columns for compatibility
        r["gamma_full"] = ""
        r["n_core"] = ""
        r["n_pairs"] = ""
        r["pair_ratio"] = ""
        r["used_surrogate"] = ""
        r["valid_reduct"] = ""
        r["t_full_matrix"] = ""
        r["n_matrix_entries"] = ""
        r["speedup_pair_vs_full"] = ""
    
    # Merge
    merged = filtered_old + new_rows
    
    # Write merged file
    out_file = results_dir / "benchmark_merged.csv"
    if merged:
        fieldnames = list(merged[0].keys())
        with out_file.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(merged)
    
    print(f"Merged {len(filtered_old)} old rows + {len(new_rows)} new rows = {len(merged)} total")
    print(f"Wrote {out_file}")
    
    # Print summary
    print("\nAlgorithm counts:")
    from collections import Counter
    algo_counts = Counter(r["algorithm"] for r in merged)
    for algo, count in sorted(algo_counts.items()):
        print(f"  {algo}: {count}")

if __name__ == "__main__":
    main()

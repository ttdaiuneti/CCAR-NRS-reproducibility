"""Analyze delta sensitivity results."""

import csv
from pathlib import Path
import numpy as np

def load_results(csv_path: Path) -> list[dict]:
    """Load CSV results."""
    with csv_path.open() as f:
        return list(csv.DictReader(f))

def print_delta_summary(results: list[dict]) -> None:
    """Print delta sensitivity summary."""
    print("\n" + "="*100)
    print("Delta Sensitivity Analysis (CCAR-E)")
    print("="*100)
    print(f"{'Delta':<10} {'|R|':<10} {'kNN':<10} {'SVM':<10} {'Runtime':<10}")
    print("-"*100)
    
    deltas = sorted(set(float(r['delta']) for r in results))
    for delta in deltas:
        delta_rows = [r for r in results if float(r['delta']) == delta and r['algorithm'] == 'ccar_exact']
        if delta_rows:
            avg_r = np.mean([float(r['n_reduct_mean']) for r in delta_rows])
            avg_knn = np.mean([float(r['knn_mean']) for r in delta_rows])
            avg_svm = np.mean([float(r['svm_mean']) for r in delta_rows])
            avg_rt = np.mean([float(r['runtime_mean']) for r in delta_rows])
            print(f"{delta:<10.2f} {avg_r:<10.2f} {avg_knn:<10.3f} {avg_svm:<10.3f} {avg_rt:<10.3f}")
    
    print("-"*100)
    
    # Per dataset breakdown
    print("\nPer dataset breakdown (CCAR-E):")
    print(f"{'Dataset':<12} {'Delta':<10} {'|R|':<10} {'kNN':<10}")
    print("-"*100)
    
    datasets = sorted(set(r['dataset'] for r in results))
    for ds in datasets:
        ds_rows = [r for r in results if r['dataset'] == ds and r['algorithm'] == 'ccar_exact']
        ds_rows_sorted = sorted(ds_rows, key=lambda x: float(x['delta']))
        for r in ds_rows_sorted:
            print(f"{r['dataset_name']:<12} {float(r['delta']):<10.2f} {float(r['n_reduct_mean']):<10.2f} {float(r['knn_mean']):<10.3f}")
    
    # Compare with Core-init
    print("\n" + "="*100)
    print("Comparison: CCAR-E vs Core-init (average across datasets)")
    print("="*100)
    print(f"{'Delta':<10} {'CCAR-E |R|':<12} {'Core-init |R|':<12} {'CCAR-E kNN':<12} {'Core-init kNN':<12}")
    print("-"*100)
    
    for delta in deltas:
        ccar_rows = [r for r in results if float(r['delta']) == delta and r['algorithm'] == 'ccar_exact']
        core_rows = [r for r in results if float(r['delta']) == delta and r['algorithm'] == 'core_init_greedy']
        if ccar_rows and core_rows:
            ccar_r = np.mean([float(r['n_reduct_mean']) for r in ccar_rows])
            core_r = np.mean([float(r['n_reduct_mean']) for r in core_rows])
            ccar_knn = np.mean([float(r['knn_mean']) for r in ccar_rows])
            core_knn = np.mean([float(r['knn_mean']) for r in core_rows])
            print(f"{delta:<10.2f} {ccar_r:<12.2f} {core_r:<12.2f} {ccar_knn:<12.3f} {core_knn:<12.3f}")

def main() -> None:
    results_dir = Path(__file__).resolve().parents[2] / "results_delta"
    csv_path = results_dir / "delta_sensitivity.csv"
    
    results = load_results(csv_path)
    print_delta_summary(results)

if __name__ == "__main__":
    main()

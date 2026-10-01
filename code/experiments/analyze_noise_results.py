"""Analyze noise robustness results and generate comparison table."""

import csv
from pathlib import Path

def load_results(csv_path: Path) -> list[dict]:
    """Load CSV results."""
    with csv_path.open() as f:
        return list(csv.DictReader(f))

def compare_noise_impact(no_noise_path: Path, noisy_path: Path) -> dict:
    """Compare results between no noise and noisy conditions."""
    no_noise = load_results(no_noise_path)
    noisy = load_results(noisy_path)
    
    comparison = {}
    for nn_row in no_noise:
        key = (nn_row['dataset'], nn_row['algorithm'])
        for n_row in noisy:
            if n_row['dataset'] == nn_row['dataset'] and n_row['algorithm'] == nn_row['algorithm']:
                comparison[key] = {
                    'dataset': nn_row['dataset'],
                    'dataset_name': nn_row['dataset_name'],
                    'algorithm': nn_row['algorithm'],
                    'algorithm_label': nn_row['algorithm_label'],
                    'no_noise_r': float(nn_row['n_reduct_mean']),
                    'no_noise_knn': float(nn_row['knn_mean']),
                    'no_noise_svm': float(nn_row['svm_mean']),
                    'noisy_r': float(n_row['n_reduct_mean']),
                    'noisy_knn': float(n_row['knn_mean']),
                    'noisy_svm': float(n_row['svm_mean']),
                    'r_change': float(n_row['n_reduct_mean']) - float(nn_row['n_reduct_mean']),
                    'knn_change': float(n_row['knn_mean']) - float(nn_row['knn_mean']),
                    'svm_change': float(n_row['svm_mean']) - float(nn_row['svm_mean']),
                }
                break
    return comparison

def print_summary(comparison: dict) -> None:
    """Print summary comparison."""
    print("\n" + "="*80)
    print("Noise Robustness Analysis (10% label noise)")
    print("="*80)
    print(f"{'Dataset':<12} {'Method':<10} {'|R| (no)':<10} {'|R| (noisy)':<10} {'Δ|R|':<10} {'kNN (no)':<10} {'kNN (noisy)':<10} {'ΔkNN':<10}")
    print("-"*80)
    
    datasets = sorted(set(v['dataset'] for v in comparison.values()))
    for ds in datasets:
        for algo in ['ccar_exact', 'ccar_h', 'ccar_surrogate', 'core_init_greedy']:
            key = (ds, algo)
            if key in comparison:
                v = comparison[key]
                print(f"{v['dataset_name']:<12} {v['algorithm_label']:<10} {v['no_noise_r']:<10.2f} {v['noisy_r']:<10.2f} {v['r_change']:<10.2f} {v['no_noise_knn']:<10.3f} {v['noisy_knn']:<10.3f} {v['knn_change']:<10.3f}")
    
    print("-"*80)
    
    # Average changes per method
    print("\nAverage changes per method:")
    methods = ['ccar_exact', 'ccar_h', 'ccar_surrogate', 'core_init_greedy']
    for algo in methods:
        values = [v for v in comparison.values() if v['algorithm'] == algo]
        if values:
            avg_r_change = sum(v['r_change'] for v in values) / len(values)
            avg_knn_change = sum(v['knn_change'] for v in values) / len(values)
            label = values[0]['algorithm_label']
            print(f"{label:<10} Δ|R|: {avg_r_change:+.2f}, ΔkNN: {avg_knn_change:+.3f}")

def main() -> None:
    results_dir = Path(__file__).resolve().parents[2] / "results_noise"
    no_noise_path = results_dir / "noise_label_0%.csv"
    
    # Label noise analysis
    print("\n" + "="*80)
    print("LABEL NOISE ANALYSIS (10%)")
    print("="*80)
    noisy_path = results_dir / "noise_label_10%.csv"
    comparison = compare_noise_impact(no_noise_path, noisy_path)
    print_summary(comparison)
    
    # Feature noise analysis
    print("\n" + "="*80)
    print("FEATURE NOISE ANALYSIS (10%)")
    print("="*80)
    noisy_path = results_dir / "noise_feature_10%.csv"
    comparison = compare_noise_impact(no_noise_path, noisy_path)
    print_summary(comparison)

if __name__ == "__main__":
    main()

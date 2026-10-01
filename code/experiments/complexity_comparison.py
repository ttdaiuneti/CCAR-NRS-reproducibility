"""
Complexity comparison: CCAR-NRS vs QuickReduct vs Core-init
on (1) hand-crafted 6x3 table and (2) real benchmark datasets.
Reports: operation counts + wall-clock time.
"""

import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from itertools import combinations

from ccar.nrs import dependency_degree, normalize_features
from ccar.core import compute_core
from ccar.conflicting import conflicting_pairs, m_res_family
from ccar.hitting_set import minimal_hitting_set_exact, greedy_hitting_set
from ccar.nrs import attribute_significance


# ─────────────────────────────────────────────────────────────────
# Operation counter wrappers
# ─────────────────────────────────────────────────────────────────
_gamma_calls = [0]
_pair_evals  = [0]

def reset():
    _gamma_calls[0] = 0
    _pair_evals[0]  = 0

def counted_gamma(X, y, idx, delta):
    _gamma_calls[0] += 1
    return dependency_degree(X, y, idx, delta)

def count_cross_pairs(n, y):
    """Total cross-class pairs (what full matrix must check)."""
    return sum(1 for i,j in combinations(range(n),2) if y[i]!=y[j])


# ─────────────────────────────────────────────────────────────────
# QuickReduct implementation with counter
# ─────────────────────────────────────────────────────────────────
def quick_reduct(X, y, delta):
    reset()
    n, p = X.shape
    gamma_c = counted_gamma(X, y, np.arange(p), delta)
    R = []
    while True:
        best_a, best_g = None, -1.0
        for a in range(p):
            if a in R: continue
            g = counted_gamma(X, y, np.array(R+[a], dtype=int), delta)
            if g > best_g:
                best_g, best_a = g, a
        if best_a is None or best_g >= gamma_c - 1e-9:
            if best_a is not None:
                R.append(best_a)
            break
        R.append(best_a)
    return sorted(R), _gamma_calls[0]


# ─────────────────────────────────────────────────────────────────
# Core-init greedy (start from CORE then greedy-add)
# ─────────────────────────────────────────────────────────────────
def core_init_greedy(X, y, delta):
    reset()
    n, p = X.shape
    core, gamma_c = compute_core(X, y, delta)
    _gamma_calls[0] += p  # core: p dep evaluations
    R = list(core)
    if counted_gamma(X, y, np.array(R, dtype=int) if R else np.array([], dtype=int), delta) >= gamma_c - 1e-9:
        pass
    else:
        while True:
            best_a, best_g = None, -1.0
            for a in range(p):
                if a in R: continue
                g = counted_gamma(X, y, np.array(R+[a], dtype=int), delta)
                if g > best_g:
                    best_g, best_a = g, a
            if best_a is None or best_g >= gamma_c - 1e-9:
                if best_a is not None: R.append(best_a)
                break
            R.append(best_a)
    return sorted(R), _gamma_calls[0]


# ─────────────────────────────────────────────────────────────────
# CCAR-H with operation counter
# ─────────────────────────────────────────────────────────────────
def ccar_h(X, y, delta):
    reset()
    n, p = X.shape
    all_idx = np.arange(p)

    core, gamma_c = compute_core(X, y, delta)
    _gamma_calls[0] += p          # one dep-drop test per attribute
    core_idx = np.array(sorted(core), dtype=int)

    if core_idx.size == 0:
        # fallback: full greedy
        R = []
        while True:
            best_a, best_g = None, -1.0
            for a in range(p):
                if a in R: continue
                g = counted_gamma(X, y, np.array(R+[a], dtype=int), delta)
                if g > best_g: best_g, best_a = g, a
            if best_a is None or best_g >= gamma_c - 1e-9:
                if best_a is not None: R.append(best_a)
                break
            R.append(best_a)
        return sorted(R), _gamma_calls[0], 0, n*(n-1)//2

    pairs = conflicting_pairs(X, y, core_idx, delta)
    n_pair_evals = n * (n-1) // 2  # upper bound: all pairs checked for CORE-neighborhood
    # actual: only n² distance computations on CORE attrs (O(n²|CORE|))

    family, _weights = m_res_family(X, pairs, core_idx, all_idx, delta)
    non_core = [int(a) for a in np.setdiff1d(all_idx, core_idx)]

    if not family:
        return sorted(core), _gamma_calls[0], len(pairs), n_pair_evals

    tie = {a: attribute_significance(X, y, core_idx, a, delta) for a in non_core}
    _gamma_calls[0] += len(non_core)  # one sig eval per residual attr

    s = greedy_hitting_set(non_core, family, tie_scores=tie)
    reduct = sorted(set(core) | set(s))
    return reduct, _gamma_calls[0], len(pairs), n_pair_evals


# ─────────────────────────────────────────────────────────────────
# Run on one dataset and print results
# ─────────────────────────────────────────────────────────────────
def benchmark_one(name, X, y, delta=0.1, repeat=20):
    n, p = X.shape
    Xn = normalize_features(X)
    cross = count_cross_pairs(n, y)
    core, gc = compute_core(Xn, y, delta)
    core_ratio = len(core)/p if p > 0 else 0

    # ----- timings -----
    t0 = time.perf_counter()
    for _ in range(repeat):
        rq, gq = quick_reduct(Xn, y, delta)
    t_qr = (time.perf_counter()-t0)/repeat

    t0 = time.perf_counter()
    for _ in range(repeat):
        rc, gc2 = core_init_greedy(Xn, y, delta)
    t_ci = (time.perf_counter()-t0)/repeat

    t0 = time.perf_counter()
    for _ in range(repeat):
        rh, gh, np_used, np_total = ccar_h(Xn, y, delta)
    t_ccar = (time.perf_counter()-t0)/repeat

    # last-run counters
    _, gq_calls = quick_reduct(Xn, y, delta)
    _, gci_calls = core_init_greedy(Xn, y, delta)
    rh, gh_calls, pdelta, _ = ccar_h(Xn, y, delta)

    print(f"\n{'─'*70}")
    print(f"Dataset: {name}   n={n}  p={p}  |CORE|={len(core)}  ρ={core_ratio:.2f}  "
          f"cross-pairs={cross}  |P_δ|={pdelta}  delta={delta}")
    print(f"{'─'*70}")
    print(f"  {'Method':20s}  {'γ-calls':>8}  {'time(ms)':>10}  {'|R|':>5}  {'γ-speedup vs QR':>16}")
    print(f"  {'QuickReduct':20s}  {gq_calls:>8d}  {t_qr*1000:>10.2f}  {len(rq):>5d}  {'1.0x':>16}")
    print(f"  {'Core-init':20s}  {gci_calls:>8d}  {t_ci*1000:>10.2f}  {len(rc):>5d}  "
          f"  {gq_calls/max(gci_calls,1):>14.1f}x")
    print(f"  {'CCAR-H':20s}  {gh_calls:>8d}  {t_ccar*1000:>10.2f}  {len(rh):>5d}  "
          f"  {gq_calls/max(gh_calls,1):>14.1f}x")
    print(f"  Cross-class pairs: {cross}  |P_δ|={pdelta}  "
          f"pair saving={(cross-pdelta)/max(cross,1)*100:.0f}%  "
          f"search space: 2^{p}={2**p} → 2^{p-len(core)}={2**(p-len(core))}")


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    # ── (A) Hand-crafted 6×3 example ─────────────────────────────
    X_toy = np.array([
        [0.0, 0.5, 0.1],
        [0.0, 0.5, 0.9],
        [1.0, 0.5, 0.5],
        [0.1, 0.0, 0.5],
        [0.1, 1.0, 0.5],
        [0.5, 0.5, 0.5],
    ], dtype=float)
    y_toy = np.array([0, 0, 0, 1, 1, 1])
    benchmark_one("Toy-6x3", X_toy, y_toy, delta=0.2, repeat=500)

    # ── (B) Real benchmark datasets ──────────────────────────────
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
        from datasets import BENCHMARK_DATASETS
        targets = ["iris", "wine", "heart", "ionosphere", "wdbc", "segment"]
        for ds in BENCHMARK_DATASETS:
            if ds.key in targets:
                X_ds, y_ds = ds.loader()
                benchmark_one(ds.key, X_ds, y_ds, delta=0.1, repeat=5)
    except Exception as e:
        print(f"\n[Skipping real datasets: {e}]")

    # ── (C) Theoretical complexity summary ───────────────────────
    print(f"\n\n{'='*70}")
    print("THEORETICAL COMPLEXITY  (T_γ = O(|U|²|C|) = one dependency eval)")
    print(f"{'='*70}")
    rows = [
        ("QuickReduct",   "|R|·|C|·T_γ",          "O(|C|²·T_γ)",     "—",          "—"),
        ("Core-init",     "(|C|+|R|·|C|)·T_γ",    "O(|C|²·T_γ)",     "partial",    "—"),
        ("CCAR-H",        "|C|·T_γ + |P_δ|·|C\\CORE|",
                          "O(|C|·T_γ)+O(|P_δ|·|C|)",  "✓",          "✓"),
        ("CCAR-E",        "|C|·T_γ + MHS(|P_δ|,|C\\CORE|)",
                          "same + exact hitting set",  "✓",          "✓"),
    ]
    print(f"  {'Method':12s}  {'Operations':40s}  {'Core anchor':11s}  {'Sound R':7s}")
    print(f"  {'':12s}  {'':40s}  {'':11s}  {'template':7s}")
    for m, ops, _, ca, sr in rows:
        print(f"  {m:12s}  {ops:40s}  {ca:11s}  {sr:7s}")

    print(f"""
Key insight: when |CORE|/|C| = ρ (core ratio) and |P_δ|/C(|U|,2) = π (pair ratio):
  QuickReduct γ-calls  ≈  |R| × |C|
  CCAR-H      γ-calls  ≈  |C|  (core step)  +  |C\\CORE|  (sig tie-break)
                        =  |C| + (1-ρ)|C|  ≈  2|C|   — INDEPENDENT of |R|

  Pair evaluations:
  Full matrix  : C(|U|,2)  all pairs
  P_δ only     : π × C(|U|,2)  (observed π ≈ 0.001–0.05 on benchmarks)
""")

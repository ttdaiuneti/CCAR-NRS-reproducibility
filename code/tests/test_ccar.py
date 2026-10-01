import numpy as np
from sklearn.datasets import load_wine
from sklearn.preprocessing import LabelEncoder

from ccar.ccar_exact import ccar_nrs_exact, verify_reduct
from ccar.ccar_surrogate import ccar_nrs_surrogate
from ccar.conflicting import conflicting_pairs, m_res_family
from ccar.core import compute_core
from ccar.hitting_set import minimal_hitting_set_exact
from ccar.nrs import dependency_degree, normalize_features


def test_ccar_exact_wine():
    data = load_wine()
    y = LabelEncoder().fit_transform(data.target)
    res = ccar_nrs_exact(data.data, y, delta=0.15)
    assert len(res.reduct_indices) >= len(res.core_indices)
    if res.reduct_indices:
        assert verify_reduct(data.data, y, res.reduct_indices, res.gamma_c, 0.15)


def test_surrogate_returns_reduct():
    X = np.random.default_rng(1).random((30, 5))
    y = np.array([i % 3 for i in range(30)])
    res = ccar_nrs_surrogate(X, y, delta=0.2, tau=2)
    assert len(res.reduct_indices) >= 1
    assert res.used_surrogate or len(res.core_indices) > 0


def test_theorem2_decomposition_toy_dataset():
    """Verify Theorem 2: R = CORE ∪ S_min where S_min is minimal hitting set of M_RES.
    
    Uses a toy Pawlak-style dataset (discrete-like with small delta) to test
    the exact decomposition theorem without approximation effects.
    """
    # Toy dataset: 3 attributes, 6 objects, 2 classes
    # Designed to have non-empty core and conflicting pairs
    X = np.array([
        [0.0, 0.0, 0.0],  # class 0
        [0.1, 0.0, 0.0],  # class 0
        [0.0, 1.0, 0.0],  # class 1
        [0.1, 1.0, 0.0],  # class 1
        [0.0, 0.0, 1.0],  # class 0
        [0.1, 0.0, 1.0],  # class 1
    ])
    y = np.array([0, 0, 1, 1, 0, 1])
    delta = 0.05  # Small delta to make it Pawlak-like
    
    X_norm = normalize_features(X)
    n_attr = X_norm.shape[1]
    all_idx = np.arange(n_attr)
    
    # Step 1: Compute CORE
    core, gamma_c = compute_core(X_norm, y, delta)
    core_idx = np.array(core, dtype=int)
    
    # Step 2: Get conflicting pairs
    pairs = conflicting_pairs(X_norm, y, core_idx, delta)
    
    # Step 3: Build M_RES family
    family, weights = m_res_family(X_norm, pairs, core_idx, all_idx, delta)
    non_core = [int(a) for a in np.setdiff1d(all_idx, core_idx)]
    
    # Step 4: Find minimal hitting set
    s_min = minimal_hitting_set_exact(non_core, family, weights)
    
    # Step 5: Construct reduct per Theorem 2
    reduct = sorted(set(core) | set(s_min))
    
    # Verify: reduct achieves gamma_c
    reduct_idx = np.array(reduct, dtype=int)
    gamma_reduct = dependency_degree(X_norm, y, reduct_idx, delta)
    
    assert abs(gamma_reduct - gamma_c) < 1e-9, \
        f"Theorem 2 failed: gamma(R)={gamma_reduct} != gamma_C={gamma_c}"
    
    # Verify: minimality (removing any attribute breaks gamma)
    for a in reduct:
        reduct_without_a = [attr for attr in reduct if attr != a]
        if reduct_without_a:  # Skip if reduct would be empty
            idx_without = np.array(reduct_without_a, dtype=int)
            gamma_without = dependency_degree(X_norm, y, idx_without, delta)
            assert gamma_without < gamma_c - 1e-9, \
                f"Reduct not minimal: removing {a} still gives gamma={gamma_without}"
    
    # Verify: CCAR-NRS-Exact produces same result
    res = ccar_nrs_exact(X, y, delta=delta, normalize=False)
    assert set(res.reduct_indices) == set(reduct), \
        f"CCAR result {res.reduct_indices} != theoretical reduct {reduct}"

import numpy as np

from ccar.nrs import dependency_degree, normalize_features, positive_region


def test_gamma_monotone_with_more_attributes():
    rng = np.random.default_rng(0)
    X = normalize_features(rng.random((20, 4)))
    y = np.array([0] * 10 + [1] * 10)
    g2 = dependency_degree(X, y, np.array([0, 1]), 0.2)
    g4 = dependency_degree(X, y, np.arange(4), 0.2)
    assert g4 >= g2 - 1e-9


def test_positive_region_subset():
    X = normalize_features(np.array([[0.0, 0.0], [0.05, 0.0], [0.9, 0.9], [0.95, 0.9]]))
    y = np.array([0, 0, 1, 1])
    pos = positive_region(X, y, np.arange(2), 0.1)
    assert pos[0] and pos[2]

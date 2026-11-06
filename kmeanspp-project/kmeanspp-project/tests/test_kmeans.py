"""Sanity checks for the from-scratch implementation.

scikit-learn is used here ONLY as an independent reference to verify our own
code (it is never used to produce results in the experiments).
Run with:  python -m tests.test_kmeans   (from the repository root)
"""
import numpy as np
from sklearn.metrics import adjusted_rand_score

from src.kmeans import (adjusted_rand_index, init_kmeanspp, inertia, kmeans,
                        squared_distances)


def test_distances():
    rng = np.random.default_rng(0)
    X, C = rng.normal(size=(50, 7)), rng.normal(size=(4, 7))
    brute = ((X[:, None, :] - C[None, :, :]) ** 2).sum(-1)
    assert np.allclose(squared_distances(X, C), brute)


def test_ari_matches_reference():
    rng = np.random.default_rng(1)
    for _ in range(20):
        a, b = rng.integers(0, 5, 300), rng.integers(0, 7, 300)
        assert np.isclose(adjusted_rand_index(a, b), adjusted_rand_score(a, b))
    a = rng.integers(0, 5, 300)
    assert np.isclose(adjusted_rand_index(a, (a + 2) % 5), 1.0)


def test_objective_never_increases():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(500, 3))
    for init in ("random", "kmeans++"):
        res = kmeans(X, 8, init=init, seed=3)
        h = np.array(res["history"])
        assert np.all(np.diff(h) <= 1e-8 * h[0]), init
        assert np.isclose(res["inertia"], inertia(X, res["centres"]))


def test_seeding_picks_distinct_points():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(200, 2))
    C = init_kmeanspp(X, 20, rng)
    assert len(np.unique(C, axis=0)) == 20


def test_finds_obvious_clusters():
    rng = np.random.default_rng(5)
    centres = np.array([[0, 0], [100, 0], [0, 100]])
    X = np.vstack([c + rng.normal(size=(100, 2)) for c in centres])
    y = np.repeat(np.arange(3), 100)
    res = kmeans(X, 3, init="kmeans++", seed=0)
    assert adjusted_rand_index(res["labels"], y) == 1.0


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok ", name)

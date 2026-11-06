"""
Datasets used in the experiments.

Synthetic data is generated with numpy from an explicit, seeded process.
The real-world dataset is the UCI "Optical Recognition of Handwritten Digits"
data, which ships inside scikit-learn (sklearn.datasets.load_digits), so the
whole project runs offline and is fully reproducible. scikit-learn is used
ONLY to load this file - never for clustering.
"""

import numpy as np


def norm_clusters(n_centres=25, dim=15, n_per_cluster=200, side=500.0,
                  sigma=1.0, seed=0):
    """'Norm-25'-style data from Arthur & Vassilvitskii (2007), Section 6.

    The paper draws 25 "true" centres uniformly at random in a hypercube of
    side 500 in 15 dimensions and adds isotropic Gaussian points with unit
    variance around each. The clusters are therefore very well separated and
    the optimal partition is known, which makes it easy to see when a bad
    initialisation gets stuck (e.g. two centres inside one true cluster).

    Returns X (n, dim), y (true cluster id, used only for evaluation),
    and the true centres.
    """
    rng = np.random.default_rng(seed)
    centres = rng.uniform(0.0, side, size=(n_centres, dim))
    X = np.vstack([c + sigma * rng.standard_normal((n_per_cluster, dim))
                   for c in centres])
    y = np.repeat(np.arange(n_centres), n_per_cluster)
    perm = rng.permutation(len(X))
    return X[perm], y[perm], centres


def blobs_2d(n_centres=9, n_per_cluster=150, sigma=0.6, seed=0):
    """Nine clusters on a 3x3 grid in the plane - used only for pictures,
    so that a bad local minimum can be seen with the naked eye."""
    rng = np.random.default_rng(seed)
    grid = np.array([(i, j) for i in range(3) for j in range(3)], float) * 5.0
    centres = grid[:n_centres]
    X = np.vstack([c + sigma * rng.standard_normal((n_per_cluster, 2))
                   for c in centres])
    y = np.repeat(np.arange(n_centres), n_per_cluster)
    perm = rng.permutation(len(X))
    return X[perm], y[perm]


def overlapping_clusters(sigma, n_centres=10, dim=10, n_per_cluster=200,
                         side=10.0, seed=0):
    """Extension: same generator as norm_clusters but on a small cube, so the
    cluster spread sigma controls how much clusters overlap."""
    return norm_clusters(n_centres=n_centres, dim=dim,
                         n_per_cluster=n_per_cluster, side=side,
                         sigma=sigma, seed=seed)


def digits():
    """UCI optical digits: 1797 images of 8x8 pixels (64 features, values
    0..16), 10 classes. A small, MNIST-like, high-dimensional multiclass set.
    All features share the same scale (pixel intensity), so no
    standardisation is applied: rescaling would change the geometry that
    k-means relies on without any benefit."""
    from sklearn.datasets import load_digits  # data loading only
    d = load_digits()
    return d.data.astype(np.float64), d.target.astype(np.int64)

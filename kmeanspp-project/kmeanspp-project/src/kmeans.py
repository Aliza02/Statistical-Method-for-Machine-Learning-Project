"""
k-means (Lloyd's algorithm) with two seeding strategies, implemented from scratch.

Only numpy is used, and only for vectorised arithmetic (distances, means,
random sampling). No clustering routine from any library is called.

Seeding strategies
------------------
* "random"   : the classical initialisation. k centres are chosen uniformly at
               random (without replacement) among the data points.
* "kmeans++" : D^2 seeding from Arthur & Vassilvitskii (SODA 2007).
               The first centre is a uniformly random data point; every next
               centre is a data point x chosen with probability
               D(x)^2 / sum_y D(y)^2, where D(x) is the distance from x to the
               nearest centre already chosen.

After seeding, both variants run exactly the same Lloyd iterations, so any
difference in the result is caused by the initialisation alone.
"""

import numpy as np


# --------------------------------------------------------------------------- #
# Basic building blocks
# --------------------------------------------------------------------------- #
def squared_distances(X, C):
    """Matrix of squared Euclidean distances, shape (n_points, n_centres).

    Uses ||x - c||^2 = ||x||^2 - 2 x.c + ||c||^2, which avoids building an
    (n, k, d) tensor. Tiny negative values caused by floating point are
    clipped to 0.
    """
    X_sq = np.einsum("ij,ij->i", X, X)[:, None]
    C_sq = np.einsum("ij,ij->i", C, C)[None, :]
    D = X_sq - 2.0 * X @ C.T + C_sq
    np.maximum(D, 0.0, out=D)
    return D


def inertia(X, C, labels=None):
    """k-means objective: sum of squared distances of each point to the
    centre of its cluster (to its closest centre if labels is None)."""
    D = squared_distances(X, C)
    if labels is None:
        return float(D.min(axis=1).sum())
    return float(D[np.arange(X.shape[0]), labels].sum())


# --------------------------------------------------------------------------- #
# Seeding
# --------------------------------------------------------------------------- #
def init_random(X, k, rng):
    """Pick k distinct data points uniformly at random."""
    idx = rng.choice(X.shape[0], size=k, replace=False)
    return X[idx].copy()


def init_kmeanspp(X, k, rng):
    """D^2 seeding (k-means++), Algorithm from Arthur & Vassilvitskii (2007).

    1. c_1 <- uniformly random data point
    2. for i = 2..k:
         D(x)^2 = squared distance of x to its nearest chosen centre
         c_i <- x with probability D(x)^2 / sum_y D(y)^2
    """
    n = X.shape[0]
    centres = np.empty((k, X.shape[1]), dtype=X.dtype)

    first = rng.integers(n)
    centres[0] = X[first]
    # closest squared distance of every point to the current set of centres
    closest_sq = squared_distances(X, centres[:1]).ravel()

    for i in range(1, k):
        total = closest_sq.sum()
        if total <= 0.0:
            # every point coincides with a chosen centre (degenerate data):
            # fall back to a uniform choice
            nxt = rng.integers(n)
        else:
            probs = closest_sq / total
            nxt = rng.choice(n, p=probs)
        centres[i] = X[nxt]
        # update nearest-centre distances with the new centre only
        new_sq = squared_distances(X, centres[i:i + 1]).ravel()
        np.minimum(closest_sq, new_sq, out=closest_sq)

    return centres


INITIALISERS = {"random": init_random, "kmeans++": init_kmeanspp}


# --------------------------------------------------------------------------- #
# Lloyd's algorithm
# --------------------------------------------------------------------------- #
def lloyd(X, centres, max_iter=300, tol=0.0, rng=None):
    """Run Lloyd iterations from the given centres until the assignment stops
    changing (or the relative decrease of the objective falls below tol).

    Empty clusters: if a cluster loses all its points, its centre is moved to
    the data point that is currently farthest from its own centre. This is a
    standard fix that keeps exactly k clusters and never increases the
    objective; it is applied identically to both seeding methods.

    Returns (centres, labels, n_iter, history) where history[t] is the
    objective after iteration t (history[0] = objective of the seeding).
    """
    k = centres.shape[0]
    C = centres.copy()
    D = squared_distances(X, C)
    labels = D.argmin(axis=1)
    history = [float(D[np.arange(len(X)), labels].sum())]

    n_iter = 0
    for n_iter in range(1, max_iter + 1):
        # ---- update step: each centre becomes the mean of its points ----
        counts = np.bincount(labels, minlength=k)
        sums = np.zeros_like(C)
        np.add.at(sums, labels, X)
        nonempty = counts > 0
        C[nonempty] = sums[nonempty] / counts[nonempty, None]

        if not nonempty.all():
            point_cost = D[np.arange(len(X)), labels]
            for j in np.flatnonzero(~nonempty):
                far = int(point_cost.argmax())
                C[j] = X[far]
                point_cost[far] = 0.0

        # ---- assignment step: each point goes to its closest centre ----
        D = squared_distances(X, C)
        new_labels = D.argmin(axis=1)
        obj = float(D[np.arange(len(X)), new_labels].sum())
        history.append(obj)

        converged = np.array_equal(new_labels, labels)
        labels = new_labels
        if converged:
            break
        if tol > 0 and history[-2] - obj <= tol * history[-2]:
            break

    return C, labels, n_iter, history


def kmeans(X, k, init="kmeans++", seed=None, max_iter=300):
    """Full k-means run: seeding + Lloyd. Returns a result dictionary."""
    rng = np.random.default_rng(seed)
    centres0 = INITIALISERS[init](X, k, rng)
    seed_inertia = inertia(X, centres0)
    C, labels, n_iter, history = lloyd(X, centres0, max_iter=max_iter, rng=rng)
    return {
        "centres": C,
        "labels": labels,
        "seed_inertia": seed_inertia,
        "inertia": history[-1],
        "n_iter": n_iter,
        "history": history,
    }


# --------------------------------------------------------------------------- #
# Evaluation helper: Adjusted Rand Index (from scratch)
# --------------------------------------------------------------------------- #
def _comb2(x):
    x = np.asarray(x, dtype=np.float64)
    return x * (x - 1.0) / 2.0


def adjusted_rand_index(a, b):
    """Adjusted Rand Index between two labelings (Hubert & Arabie, 1985).

    1.0 = identical partitions (up to renaming the clusters),
    ~0  = agreement no better than chance.
    """
    a = np.asarray(a)
    b = np.asarray(b)
    _, a_idx = np.unique(a, return_inverse=True)
    _, b_idx = np.unique(b, return_inverse=True)
    table = np.zeros((a_idx.max() + 1, b_idx.max() + 1), dtype=np.int64)
    np.add.at(table, (a_idx, b_idx), 1)

    sum_cells = _comb2(table).sum()
    sum_rows = _comb2(table.sum(axis=1)).sum()
    sum_cols = _comb2(table.sum(axis=0)).sum()
    total = _comb2(len(a))

    expected = sum_rows * sum_cols / total
    max_index = 0.5 * (sum_rows + sum_cols)
    if max_index == expected:
        return 1.0
    return float((sum_cells - expected) / (max_index - expected))

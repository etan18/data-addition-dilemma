import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, List


# =========================
# Mondrian Tree Node
# =========================

@dataclass
class _Node:
    lo: np.ndarray
    hi: np.ndarray
    split_dim: int = -1
    split_val: float = 0.0
    left: Optional["_Node"] = None
    right: Optional["_Node"] = None
    # counts[h] = [n0, n1] for hospital h
    counts: Optional[Dict[int, List[int]]] = None


def _leaf_prob(counts, h, prior=0.5, strength=2.0):
    """Hospital-aware smoothed probability with fallback."""
    if h in counts:
        n0, n1 = counts[h]
    else:
        n0 = sum(v[0] for v in counts.values())
        n1 = sum(v[1] for v in counts.values())
    return (n1 + strength * prior) / (n0 + n1 + strength)


def _sample_mondrian(
    X, y, g, lo, hi, rng, lifetime, time, max_depth, depth
):
    node = _Node(lo=lo.copy(), hi=hi.copy(), counts={})

    # accumulate leaf stats
    for yi, gi in zip(y, g):
        if gi not in node.counts:
            node.counts[gi] = [0, 0]
        node.counts[gi][yi] += 1

    if depth >= max_depth or len(y) < 2:
        return node

    widths = hi - lo
    rate = widths.sum()
    if rate <= 0:
        return node

    E = rng.exponential(1.0 / rate)
    if time + E >= lifetime:
        return node

    d = rng.choice(len(widths), p=widths / rate)
    split = rng.uniform(lo[d], hi[d])

    mask = X[:, d] <= split
    if not mask.any() or mask.all():
        return node

    node.split_dim = d
    node.split_val = split

    node.left = _sample_mondrian(
        X[mask], y[mask], g[mask],
        np.minimum.reduce(X[mask], axis=0),
        np.maximum.reduce(X[mask], axis=0),
        rng, lifetime, time + E, max_depth, depth + 1
    )
    node.right = _sample_mondrian(
        X[~mask], y[~mask], g[~mask],
        np.minimum.reduce(X[~mask], axis=0),
        np.maximum.reduce(X[~mask], axis=0),
        rng, lifetime, time + E, max_depth, depth + 1
    )
    return node


def _predict_tree(node, x, h, prior, strength):
    while node.left is not None:
        if x[node.split_dim] <= node.split_val:
            node = node.left
        else:
            node = node.right
    return _leaf_prob(node.counts, h, prior, strength)


# =========================
# Mondrian Forest
# =========================

class K29:
    """
    Mondrian Forest version of K29.
    - Last column is categorical hospital ID
    - Trees split only on continuous features
    - Leaves store hospital-conditional statistics
    """

    def __init__(
        self,
        n_trees=50,
        lifetime=5.0,
        max_depth=30,
        random_state=None,
        prior=0.5,
        prior_strength=2.0,
    ):
        self.n_trees = n_trees
        self.lifetime = lifetime
        self.max_depth = max_depth
        self.random_state = random_state
        self.prior = prior
        self.prior_strength = prior_strength

        self.trees = []
        self.fitted_ = False

    def _split_features(self, X):
        X = np.asarray(X)
        if X.ndim == 1:
            return X[:-1], int(X[-1])
        return X[:, :-1], X[:, -1].astype(int)

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y).astype(int)
        z, g = self._split_features(X)

        lo = z.min(axis=0)
        hi = z.max(axis=0)

        rng = np.random.RandomState(self.random_state)
        self.trees = []

        for _ in range(self.n_trees):
            tree = _sample_mondrian(
                z, y, g, lo, hi,
                rng, self.lifetime, 0.0,
                self.max_depth, 0
            )
            self.trees.append(tree)

        self.fitted_ = True
        return self

    def predict(self, X):
        if not self.fitted_:
            raise ValueError("Model not fitted")

        X = np.asarray(X)
        single = X.ndim == 1
        if single:
            X = X[None, :]

        z, g = self._split_features(X)
        out = np.zeros(len(z))

        for i in range(len(z)):
            s = 0.0
            for t in self.trees:
                s += _predict_tree(
                    t, z[i], g[i],
                    self.prior, self.prior_strength
                )
            out[i] = s / len(self.trees)

        return float(out[0]) if single else out

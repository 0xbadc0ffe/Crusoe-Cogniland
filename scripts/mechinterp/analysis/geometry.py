"""Dimensionality reduction + direction geometry (the entanglement maths).

Everything here is label-agnostic numpy/sklearn so it is reused unchanged for BT
(no belief/skill) — the *caller* decides which direction sets exist. A "direction"
is a unit vector in an activation space; the scientific question is whether the
belief directions and the skill directions span the same subspace (entangled) or
are geometrically separate.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import PCA


# --------------------------------------------------------------------- DR
@dataclass
class Projection:
    coords: np.ndarray          # (n, k)
    name: str                   # 'pca' | 'umap' | 'tsne'
    explained: np.ndarray | None = None   # PCA only
    model: object | None = None            # fitted estimator (PCA: supports .transform)


def pca_project(X: np.ndarray, n_components: int = 10, seed: int = 0) -> Projection:
    p = PCA(n_components=min(n_components, X.shape[1]), random_state=seed)
    coords = p.fit_transform(X)
    return Projection(coords=coords, name="pca",
                      explained=p.explained_variance_ratio_, model=p)


def umap_project(X, n_neighbors=30, min_dist=0.1, seed=0) -> Projection:
    import umap
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist,
                        n_components=2, random_state=seed)
    return Projection(coords=reducer.fit_transform(X), name="umap")


def tsne_project(X, seed=0) -> Projection:
    from sklearn.manifold import TSNE
    perp = min(30, max(5, X.shape[0] // 100))
    coords = TSNE(n_components=2, perplexity=perp, random_state=seed,
                  init="pca").fit_transform(X)
    return Projection(coords=coords, name="tsne")


# --------------------------------------------------------------- directions
def unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def diff_of_means(X: np.ndarray, labels: np.ndarray, pos, neg) -> np.ndarray:
    """mean(X[labels in pos]) - mean(X[labels in neg]); returned NON-normalised
    so callers can inspect the raw gap norm before unit()-ing."""
    pos = {pos} if isinstance(pos, str) else set(pos)
    neg = {neg} if isinstance(neg, str) else set(neg)
    mp = X[np.isin(labels, list(pos))].mean(0)
    mn = X[np.isin(labels, list(neg))].mean(0)
    return mp - mn


def class_centroids(X: np.ndarray, labels: np.ndarray, order) -> dict:
    return {c: X[labels == c].mean(0) for c in order if (labels == c).any()}


# --------------------------------------------------------------- alignment
def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(unit(a), unit(b)))


def cosine_matrix(dirs_a: dict, dirs_b: dict) -> tuple[np.ndarray, list, list]:
    """|rows|x|cols| cosine-similarity matrix between two named direction sets."""
    ra, rb = list(dirs_a), list(dirs_b)
    M = np.array([[cosine(dirs_a[i], dirs_b[j]) for j in rb] for i in ra])
    return M, ra, rb


def projection_fraction(vec: np.ndarray, basis: list[np.ndarray]) -> float:
    """Fraction of ||vec|| that lies inside span(basis): ||proj|| / ||vec||.
    1.0 = fully inside the subspace (maximally entangled), 0.0 = orthogonal."""
    if not basis:
        return 0.0
    B = np.linalg.svd(np.stack([unit(b) for b in basis]).T, full_matrices=False)[0]
    proj = B @ (B.T @ vec)
    nv = np.linalg.norm(vec)
    return float(np.linalg.norm(proj) / nv) if nv > 0 else 0.0


def principal_angles(basis_a: list[np.ndarray], basis_b: list[np.ndarray]) -> np.ndarray:
    """Principal angles (degrees) between span(A) and span(B). Small angles =>
    overlapping subspaces => belief and skill are entangled."""
    if not basis_a or not basis_b:
        return np.array([])
    Qa = np.linalg.qr(np.stack([unit(b) for b in basis_a]).T)[0]
    Qb = np.linalg.qr(np.stack([unit(b) for b in basis_b]).T)[0]
    s = np.linalg.svd(Qa.T @ Qb, compute_uv=False)
    return np.degrees(np.arccos(np.clip(s, -1, 1)))

#!/usr/bin/env python
"""Environment-agnostic neural-geometry toolkit (arrays in, arrays out).

Implements the techniques of Goodfire's "Manifold Steering" (arXiv:2605.05115)
and "The Shape of Beliefs" (arXiv:2602.02315) in a form reusable across
environments (MemoryEnv, bridge_tunnel/cogniland, ...): nothing here imports an
environment — every function operates on plain numpy arrays:

  X      (N, D)  hidden states / activations
  P      (N, K)  probability vectors (belief or behavior distributions)
  labels (N,)    integer class labels
  param  (N,)    a continuous/ordinal task parameter (e.g. corridor column)

Contents
  inpca            intensive PCA of probability vectors (Bhattacharyya /
                   Minkowski double-centering; signed eigenvalues)
  hellinger_map    p -> sqrt(p) (linearizes the simplex; Goodfire's behavior map)
  bhattacharyya    distance matrix / pairwise
  fit_fiber        polyline manifold through param-ordered class centroids
  tree_geodesics   geodesic distances on a trunk+fibers tree manifold
  lfp_fit          linear field probe family over param bins
  lfp_gram         probe-field Gram (cos-sim) + transfer-accuracy matrices
  smooth_field     spline-smooth a vector field over bins (per-coordinate)
  additive_r2      additivity test  mu(c,x) ~ a_c + b_x
  find_fixed_points  gradient-descent fixed points of a step map + stability
  path_energy      cumulative Bhattacharyya energy of a distribution path
"""
from __future__ import annotations

import numpy as np


# ── probability-space embeddings ─────────────────────────────────────────────
def bhattacharyya(P, Q=None, eps=1e-12):
    """Pairwise Bhattacharyya distance  -log sum_k sqrt(p_k q_k).
    P (N,K), Q (M,K) -> (N,M); Q=None -> (N,N)."""
    Q = P if Q is None else Q
    B = np.sqrt(np.clip(P, 0, None)) @ np.sqrt(np.clip(Q, 0, None)).T
    return -np.log(np.clip(B, eps, None))


def hellinger_map(P):
    """p -> sqrt(p): the simplex becomes (a patch of) a Euclidean sphere."""
    return np.sqrt(np.clip(P, 0, None))


def inpca(P, n_components=3, eps=1e-12):
    """Intensive PCA of probability vectors (Sethna et al., PNAS 2019).

    Double-centers the Bhattacharyya divergence matrix and eigendecomposes;
    coordinates are eigenvectors scaled by sqrt(|lambda|) with SIGNED
    eigenvalues (a Minkowski embedding — negative-lambda directions are
    "time-like"). Returns (coords (N,n_components), eigvals, sign mask)."""
    D = bhattacharyya(P, eps=eps)
    n = D.shape[0]
    J = np.eye(n) - 1.0 / n
    W = -0.5 * J @ D @ J
    W = 0.5 * (W + W.T)
    lam, V = np.linalg.eigh(W)
    order = np.argsort(-np.abs(lam))
    lam, V = lam[order], V[:, order]
    k = n_components
    coords = V[:, :k] * np.sqrt(np.abs(lam[:k]))[None, :]
    return coords, lam, np.sign(lam[:k])


# ── manifold fitting (trunk + fibers tree) ───────────────────────────────────
def fit_fiber(X, param, bins):
    """Per-bin centroids along `param` -> a polyline fiber.

    Returns dict(bins=used_bins, mu=(B,D) centroids, arc=(B,) cumulative
    arc-length). Bins with no samples are dropped."""
    mus, used = [], []
    for b in bins:
        m = param == b
        if m.sum() >= 3:
            mus.append(X[m].mean(0))
            used.append(b)
    mu = np.stack(mus)
    seg = np.linalg.norm(np.diff(mu, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(seg)])
    return dict(bins=np.asarray(used), mu=mu, arc=arc)


def tree_geodesics(trunk, fibers, split_bin):
    """Geodesic distance matrix between all (fiber, bin) nodes of a tree
    manifold: a shared `trunk` fiber that splits into per-class `fibers` at
    `split_bin`. Distances within a fiber follow its arc-length; across fibers
    they pass through the split point.

    trunk/fibers: dicts from fit_fiber (fiber bins must start >= split_bin).
    Returns (nodes [(class_id, bin)], Dgeo (M,M))."""
    nodes, arcs = [], []
    for cid, f in fibers.items():
        for i, b in enumerate(f["bins"]):
            nodes.append((cid, int(b)))
            arcs.append((cid, f["arc"][i]))
    M = len(nodes)
    Dg = np.zeros((M, M))
    for i in range(M):
        ci, ai = arcs[i]
        for j in range(i + 1, M):
            cj, aj = arcs[j]
            d = abs(ai - aj) if ci == cj else ai + aj
            Dg[i, j] = Dg[j, i] = d
    return nodes, Dg


# ── linear field probes ──────────────────────────────────────────────────────
def lfp_fit(X, y, param, bins, C=1.0, min_per_bin=40):
    """A family of multinomial probes, one per param bin (the "field").

    Returns dict(bins, W (B, K, D) class-row weight tensor, acc (B,) in-bin CV
    accuracy, classes)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    classes = np.unique(y)
    Ws, accs, used = [], [], []
    for b in bins:
        m = param == b
        if m.sum() < min_per_bin or len(np.unique(y[m])) < len(classes):
            continue
        clf = LogisticRegression(max_iter=2000, C=C).fit(X[m], y[m])
        cv = cross_val_score(LogisticRegression(max_iter=2000, C=C),
                             X[m], y[m], cv=3).mean()
        W = clf.coef_
        if W.shape[0] == 1:                       # binary -> symmetrize rows
            W = np.vstack([-W, W]) / 2
        Ws.append(W)
        accs.append(cv)
        used.append(b)
    return dict(bins=np.asarray(used), W=np.stack(Ws), acc=np.asarray(accs),
                classes=classes)


def lfp_gram(field):
    """Cosine-similarity Gram matrix K(b, b') of the probe field (rows
    flattened across classes) — smooth banded structure = rotating code,
    near-constant = stationary code."""
    W = field["W"].reshape(len(field["bins"]), -1)
    Wn = W / (np.linalg.norm(W, axis=1, keepdims=True) + 1e-12)
    return Wn @ Wn.T


def lfp_transfer(X, y, param, field):
    """Transfer accuracy T[i, j]: probe trained at bin i applied at bin j."""
    B = len(field["bins"])
    T = np.full((B, B), np.nan)
    cls = field["classes"]
    for j, bj in enumerate(field["bins"]):
        m = param == bj
        Xj, yj = X[m], y[m]
        for i in range(B):
            z = Xj @ field["W"][i].T
            T[i, j] = float((cls[z.argmax(1)] == yj).mean())
    return T


def smooth_field(bins, U, s=None):
    """Spline-smooth a per-bin vector field U (B, D) coordinate-wise."""
    from scipy.interpolate import UnivariateSpline
    B, D = U.shape
    out = np.zeros_like(U)
    x = np.asarray(bins, float)
    for d in range(D):
        try:
            sp = UnivariateSpline(x, U[:, d], k=min(3, B - 1), s=s)
            out[:, d] = sp(x)
        except Exception:
            out[:, d] = U[:, d]
    return out


# ── structure tests ──────────────────────────────────────────────────────────
def additive_r2(M):
    """Additivity of a centroid grid M (C, B, D):  M ~ mean + a_c + b_x.
    Returns (R^2 of the additive model, interaction fraction = 1 - R^2)."""
    mean = M.mean((0, 1), keepdims=True)
    a = M.mean(1, keepdims=True) - mean
    b = M.mean(0, keepdims=True) - mean
    resid = M - (mean + a + b)
    tot = ((M - mean) ** 2).sum()
    return float(1 - (resid ** 2).sum() / tot), float((resid ** 2).sum() / tot)


def path_energy(P_path, P_target):
    """Cumulative Bhattacharyya energy of a distribution path to a target
    distribution (Goodfire's naturalness energy, discretized)."""
    d = bhattacharyya(P_path, P_target[None, :])[:, 0]
    return float(d.mean()), d


# ── dynamics: fixed points of a step map ─────────────────────────────────────
def find_fixed_points(step_fn, inits, iters=1500, lr=5e-2, tol=1e-4,
                      merge_tol=1.0):
    """Gradient-descent minimization of ||step(h) - h||^2 from many inits.

    step_fn: callable (N, D) -> (N, D) (jax or numpy; must be jax-traceable
    for the stability analysis). Returns list of dicts(h, speed, eig_max,
    stable, n_merged)."""
    import jax
    import jax.numpy as jnp
    import optax

    H = jnp.asarray(inits, jnp.float32)

    def loss(h):
        d = step_fn(h) - h
        return (d * d).sum(-1)

    opt = optax.adam(lr)
    state = opt.init(H)

    @jax.jit
    def sweep(H, state):
        g = jax.grad(lambda hh: loss(hh).sum())(H)
        upd, state = opt.update(g, state)
        return optax.apply_updates(H, upd), state

    for _ in range(iters):
        H, state = sweep(H, state)
    sp = np.sqrt(np.asarray(loss(H)))
    Hn = np.asarray(H)
    good = Hn[sp < tol]
    sp = sp[sp < tol]
    # merge duplicates
    out = []
    used = np.zeros(len(good), bool)
    single = lambda h: step_fn(h[None])[0]          # noqa: E731
    for i in np.argsort(sp):
        if used[i]:
            continue
        close = np.linalg.norm(good - good[i], axis=1) < merge_tol
        used |= close
        Jm = np.asarray(jax.jacobian(single)(jnp.asarray(good[i])))
        ev = np.abs(np.linalg.eigvals(Jm)).max()
        out.append(dict(h=good[i], speed=float(sp[i]), eig_max=float(ev),
                        stable=bool(ev < 1.0), n_merged=int(close.sum())))
    return out

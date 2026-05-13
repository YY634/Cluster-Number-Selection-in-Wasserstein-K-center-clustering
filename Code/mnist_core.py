#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mnist_core.py
=============

Core algorithms for the MNIST Wasserstein clustering experiments.

Supports both K-means and K-median clustering on fixed pixel support, selected
via the ``method`` argument:

    method="kmeans"   - cluster center = uniform-weight Wasserstein barycenter
                        (Frechet mean)
    method="kmedian"  - cluster center = Wasserstein Frechet median, computed
                        by iteratively reweighted least squares (IRLS)

Public entry points used by the multi-seed runner:
    mnist_to_fixed_support_weights(...)   # data loading + binning
    validate_over_Ks_fixed(...)           # full K-sweep with sil / RWSD / DBI
                                          #   (method="kmeans" or "kmedian")

Backward-compatible aliases:
    validate_over_Ks_kmeans_fixed(...)         -> method="kmeans"
    wasserstein_k_means_fixed_support(...)     -> method="kmeans"
"""

import numpy as np
from joblib import Parallel, delayed
from sklearn.metrics import silhouette_score
import ot
from torchvision import datasets

EPS = 1e-12


# ============================================================
# Pixel support and MNIST loading
# ============================================================

def build_pixel_support_28x28(normalize=True, dtype=np.float64):
    H, W = 28, 28
    yy, xx = np.meshgrid(np.arange(H, dtype=dtype), np.arange(W, dtype=dtype), indexing='ij')
    coords = np.column_stack((yy.ravel(), xx.ravel()))
    if normalize:
        coords[:, 0] /= (H - 1)
        coords[:, 1] /= (W - 1)
    return coords.astype(dtype, copy=False)  # (784, 2)


def load_mnist_torchvision(split="train", root="./data"):
    is_train = (split == "train")
    ds = datasets.MNIST(root=root, train=is_train, download=True, transform=None)
    X = ds.data.numpy().astype(np.uint8)   # (N, 28, 28)
    y = ds.targets.numpy().astype(np.int64)
    return X, y


def normalize_rows_stochastic(W, min_w=1e-12):
    W = np.asarray(W, dtype=np.float64)
    W = np.maximum(W, min_w)
    s = W.sum(axis=1, keepdims=True)
    s[s <= 0] = 1.0
    return W / s


def mnist_to_fixed_support_weights(sample_plan, seed=42, intensity="linear", gamma=1.0,
                                   split="train", root="./data", dtype=np.float64):
    rng = np.random.default_rng(seed)
    X_img, y_all = load_mnist_torchvision(split=split, root=root)

    idx_sel_list = []
    for d, num in sample_plan.items():
        idx_d = np.where(y_all == d)[0]
        if len(idx_d) < num:
            raise ValueError(f"Digit {d} has only {len(idx_d)} samples, fewer than requested {num}.")
        pick = rng.choice(idx_d, size=num, replace=False)
        idx_sel_list.append(pick)

    idx_sel = np.hstack(idx_sel_list)
    X_sel = X_img[idx_sel].astype(np.float64)  # (n, 28, 28)
    n = X_sel.shape[0]
    m = 28 * 28

    if intensity == "linear":
        W = X_sel.reshape(n, m)
    elif intensity == "gamma":
        W = np.power(X_sel / 255.0, gamma).reshape(n, m)
    elif intensity == "binarize":
        W = (X_sel > 0).reshape(n, m).astype(np.float64)
    else:
        raise ValueError("intensity must be one of {'linear', 'gamma', 'binarize'}")

    W = normalize_rows_stochastic(W).astype(dtype, copy=False)
    X_support = build_pixel_support_28x28(normalize=True, dtype=dtype)
    y = y_all[idx_sel]
    return X_support, W, y


# ============================================================
# Wasserstein / Sinkhorn primitives
# ============================================================

def sqdist(X, Y=None):
    Y = X if Y is None else Y
    XX = (X**2).sum(1, keepdims=True)
    YY = (Y**2).sum(1, keepdims=True)
    return (XX + YY.T - 2.0 * (X @ Y.T)).clip(min=0.0)


def sinkhorn_reg_ot_value_fixed_support(a, b, C, eps, n_iter=400, tol=1e-9):
    m = C.shape[0]
    a = np.asarray(a, dtype=np.float64).clip(1e-300)
    b = np.asarray(b, dtype=np.float64).clip(1e-300)
    K = np.exp(-C / max(eps, 1e-12))
    K = np.maximum(K, 1e-300)
    u = np.full(m, 1.0/m, dtype=np.float64)
    v = np.full(m, 1.0/m, dtype=np.float64)
    for _ in range(n_iter):
        Kv = K @ v; Kv = np.maximum(Kv, 1e-300)
        u_new = a / Kv
        Ktu = K.T @ u_new; Ktu = np.maximum(Ktu, 1e-300)
        v_new = b / Ktu
        if tol and tol > 0:
            row = u_new * (K @ v_new)
            col = v_new * (K.T @ u_new)
            err = max(np.abs(row - a).sum(), np.abs(col - b).sum())
            u, v = u_new, v_new
            if err < tol:
                break
        else:
            u, v = u_new, v_new
    T = (u[:, None] * K) * v[None, :]
    T = np.maximum(T, 1e-300)
    reg_term = (T * (np.log(T) - 1.0)).sum()
    val = float((T * C).sum() + eps * reg_term)
    return val


def sinkhorn_divergence(a, b, C, eps, **kwargs):
    return (sinkhorn_reg_ot_value_fixed_support(a, b, C, eps, **kwargs)
            - 0.5 * (sinkhorn_reg_ot_value_fixed_support(a, a, C, eps, **kwargs)
                     + sinkhorn_reg_ot_value_fixed_support(b, b, C, eps, **kwargs)))


def lp_w2_distance(a, b, C):
    return np.sqrt(ot.emd2(a, b, C))


# ============================================================
# Cluster-center updates
# ============================================================

def fixed_support_mean_barycenter(W_sub, C, backend="sinkhorn", eps=5e-3,
                                  n_iter=1000, tol=1e-9):
    """
    K-means center update: uniform-weight Wasserstein barycenter (Frechet mean).

        argmin_a  (1/n) * sum_i W_2^2(a, W_sub[i])
    """
    n = W_sub.shape[0]
    if n == 0:
        return None

    if backend == "sinkhorn":
        weights = np.full(n, 1.0 / n, dtype=np.float64)
        a = ot.bregman.barycenter(
            W_sub.T, C, eps,
            weights=weights, numItermax=n_iter, stopThr=tol,
        )
        a = np.maximum(a, 1e-12)
        a /= a.sum()
        return a
    elif backend == "lp":
        # Uniform-weight average on the weight simplex (mirrors the LP path
        # of the median IRLS, which also uses a weighted average rather than
        # a true LP barycenter).
        a = W_sub.mean(axis=0).astype(np.float64, copy=False)
        a = np.maximum(a, 1e-12)
        a /= a.sum()
        return a
    else:
        raise ValueError("backend must be one of {'sinkhorn', 'lp'}")


def fixed_support_median_barycenter(W_sub, C, backend="sinkhorn", eps=5e-3,
                                    n_iter=1000, tol=1e-9,
                                    irls_max_iter=20, irls_tol=1e-7,
                                    inner_sinkhorn_iter=400):
    """
    K-median center update: Wasserstein Frechet median, computed via IRLS.

        argmin_a  sum_i W_2(a, W_sub[i])

    Iteration:  w_i^{(t)} = 1 / W_2(a^{(t)}, W_sub[i])  (truncated below)
                a^{(t+1)} = weighted Wasserstein barycenter with weights w^{(t)}.
    """
    n = W_sub.shape[0]
    if n == 0:
        return None

    # Initialize with the uniform-weight barycenter.
    a = fixed_support_mean_barycenter(
        W_sub, C, backend=backend, eps=eps, n_iter=n_iter, tol=tol
    )
    if a is None:
        return None

    for _ in range(irls_max_iter):
        # Distances from current center to each cluster member.
        dists = np.empty(n, dtype=np.float64)
        for i in range(n):
            if backend == "sinkhorn":
                d = max(sinkhorn_divergence(a, W_sub[i], C, eps,
                                            n_iter=inner_sinkhorn_iter), 0.0)
                dists[i] = np.sqrt(d)
            else:
                dists[i] = lp_w2_distance(a, W_sub[i], C)

        # IRLS weights, truncated below to avoid blow-up at zero.
        w = 1.0 / np.maximum(dists, 1e-12)
        s = w.sum()
        if s <= 0:
            break
        w = w / s

        # Weighted Wasserstein barycenter.
        if backend == "sinkhorn":
            a_new = ot.bregman.barycenter(
                W_sub.T, C, eps,
                weights=w, numItermax=n_iter, stopThr=tol,
            )
        else:
            a_new = (w[:, None] * W_sub).sum(axis=0)
        a_new = np.maximum(a_new, 1e-12)
        a_new /= a_new.sum()

        diff = float(np.abs(a_new - a).sum())
        a = a_new
        if diff < irls_tol:
            break

    return a


def _center_update(W_sub, C, method, backend, eps,
                   bary_max_iter, bary_tol, inner_sinkhorn_iter,
                   irls_max_iter=20, irls_tol=1e-7):
    """Dispatch table for cluster-center updates."""
    if method == "kmeans":
        return fixed_support_mean_barycenter(
            W_sub=W_sub, C=C, backend=backend, eps=eps,
            n_iter=bary_max_iter, tol=bary_tol,
        )
    elif method == "kmedian":
        return fixed_support_median_barycenter(
            W_sub=W_sub, C=C, backend=backend, eps=eps,
            n_iter=bary_max_iter, tol=bary_tol,
            irls_max_iter=irls_max_iter, irls_tol=irls_tol,
            inner_sinkhorn_iter=inner_sinkhorn_iter,
        )
    else:
        raise ValueError("method must be one of {'kmeans', 'kmedian'}")


# ============================================================
# PAM helpers and warm starts
# ============================================================

def pam_build(D, K):
    n = D.shape[0]
    costs = D.sum(axis=1)
    medoids = [int(np.argmin(costs))]
    while len(medoids) < K:
        current = np.min(D[:, medoids], axis=1)
        best_gain, best_idx = np.inf, None
        for i in range(n):
            if i in medoids:
                continue
            cand = np.minimum(current, D[:, i])
            gain = cand.sum()
            if gain < best_gain:
                best_gain, best_idx = gain, i
        medoids.append(int(best_idx))
    return np.array(medoids, dtype=int)


def pam_swap(D, medoids, max_iter=20):
    n = D.shape[0]
    medoids = medoids.copy()
    for _ in range(max_iter):
        improved = False
        non = np.setdiff1d(np.arange(n), medoids, assume_unique=False)
        base = np.min(D[:, medoids], axis=1).sum()
        for mi, m in enumerate(medoids):
            for h in non:
                trial = medoids.copy()
                trial[mi] = h
                newc = np.min(D[:, trial], axis=1).sum()
                if newc + 1e-12 < base:
                    medoids = trial
                    base = newc
                    improved = True
        if not improved:
            break
    return medoids


def init_assign_by_mean_kmeanspp(W, K, rng):
    n, m = W.shape
    idx = np.arange(m, dtype=np.float64)
    means = W @ idx  # (n,)
    centers = []
    first = int(rng.integers(0, n))
    centers.append(first)
    for _ in range(1, K):
        d2 = np.minimum.reduce([(means - means[c])**2 for c in centers])
        s = d2.sum()
        p = d2 / (s if s > 0 else 1.0)
        nxt = int(rng.choice(n, p=p))
        centers.append(nxt)
    labels = np.argmin(np.stack([(means - means[c])**2 for c in centers], axis=1), axis=1)
    return labels


# ============================================================
# Wasserstein K-clustering on fixed support (K-means / K-median)
# ============================================================

def wasserstein_k_clustering_fixed_support(
    X_support, W, K,
    method="kmeans",
    backend="sinkhorn", eps=5e-3,
    pam_warm_start=True, use_sa=False, T0=0.5, alpha=0.9,
    max_outer_rounds=12, bary_max_iter=1000, bary_tol=1e-9,
    inner_sinkhorn_iter=400, assign_n_jobs=-1, seed=42,
    irls_max_iter=20, irls_tol=1e-7,
    D_precomputed=None,
):
    """
    Lloyd-style Wasserstein K-clustering on a shared fixed support.

    Parameters
    ----------
    method : {"kmeans", "kmedian"}
        Center-update rule. "kmeans" uses the uniform-weight Wasserstein
        barycenter (Frechet mean); "kmedian" uses the Wasserstein Frechet
        median computed via IRLS.

    Notes
    -----
    The assignment step is identical for both methods (argmin over W_2),
    only the per-cluster center update differs.
    """
    if method not in ("kmeans", "kmedian"):
        raise ValueError("method must be one of {'kmeans', 'kmedian'}")

    rng = np.random.default_rng(seed)
    n, m = W.shape
    assert X_support.shape[0] == m
    C = sqdist(X_support, X_support)

    # ---- Initialization ----
    if D_precomputed is not None:
        medoids = pam_build(D_precomputed, K)
        medoids = pam_swap(D_precomputed, medoids, max_iter=20)
        labels = np.argmin(D_precomputed[:, medoids], axis=1)
    elif pam_warm_start:
        def _pair(i, j):
            if backend == "sinkhorn":
                s = max(sinkhorn_divergence(W[i], W[j], C, eps,
                                            n_iter=inner_sinkhorn_iter), 0.0)
                s = np.sqrt(s)
            else:
                s = lp_w2_distance(W[i], W[j], C)
            return i, j, s

        D = np.zeros((n, n), dtype=np.float64)
        pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
        results = Parallel(n_jobs=assign_n_jobs, backend="loky")(
            delayed(_pair)(i, j) for (i, j) in pairs
        )
        for i, j, v in results:
            D[i, j] = D[j, i] = v
        medoids = pam_build(D, K)
        medoids = pam_swap(D, medoids, max_iter=20)
        labels = np.argmin(D[:, medoids], axis=1)
    else:
        labels = init_assign_by_mean_kmeanspp(W, K, rng)

    # ---- Lloyd-style outer loop ----
    centers = [None for _ in range(K)]
    T = T0
    for r in range(max_outer_rounds):
        # Center update (mean barycenter for K-means, IRLS median for K-median).
        for k in range(K):
            idx = np.where(labels == k)[0]
            if len(idx) == 0:
                centers[k] = None
            else:
                centers[k] = _center_update(
                    W_sub=W[idx], C=C, method=method,
                    backend=backend, eps=eps,
                    bary_max_iter=bary_max_iter, bary_tol=bary_tol,
                    inner_sinkhorn_iter=inner_sinkhorn_iter,
                    irls_max_iter=irls_max_iter, irls_tol=irls_tol,
                )

        # Assignment step: argmin_k W_2(center_k, w_i).
        # Identical for K-means and K-median (both monotone in W_2).
        def _row_cost(i):
            row = np.empty(K, dtype=np.float64)
            wi = W[i]
            for k in range(K):
                if centers[k] is None:
                    row[k] = np.inf
                else:
                    if backend == "sinkhorn":
                        val = max(sinkhorn_divergence(centers[k], wi, C, eps,
                                                      n_iter=inner_sinkhorn_iter), 0.0)
                        row[k] = np.sqrt(val)
                    else:
                        row[k] = lp_w2_distance(centers[k], wi, C)
            return i, row

        results = Parallel(n_jobs=assign_n_jobs, backend="loky")(
            delayed(_row_cost)(i) for i in range(n)
        )
        D_assign = np.zeros((n, K), dtype=np.float64)
        for i, row in results:
            D_assign[i] = row

        if use_sa:
            new_labels = np.zeros(n, dtype=int)
            for i in range(n):
                d = D_assign[i]
                p = np.exp(-d / max(T, 1e-12))
                s = p.sum()
                new_labels[i] = (np.argmin(d) if s <= 0
                                 else np.random.default_rng(seed + r + i).choice(K, p=p/s))
            T *= alpha
        else:
            new_labels = np.argmin(
                D_assign + 1e-12 * np.random.default_rng(seed + r).standard_normal(D_assign.shape),
                axis=1
            )

        changes = int(np.sum(new_labels != labels))
        labels = new_labels
        if changes == 0:
            break

    return labels, centers, C


# Backward-compatible alias.
def wasserstein_k_means_fixed_support(*args, **kwargs):
    """Deprecated alias. Calls wasserstein_k_clustering_fixed_support(method='kmeans')."""
    kwargs.setdefault("method", "kmeans")
    return wasserstein_k_clustering_fixed_support(*args, **kwargs)


# ============================================================
# Wasserstein Spatial Depth (within / between)
# ============================================================

def compute_WSD_X(X, labels, n_jobs=-1):
    if X.ndim != 4 or X.shape[-1] != 2:
        raise ValueError("X must be (M, n, d, 2)")
    M, n, d, _ = X.shape
    coords = X[..., 0].astype(np.float64, copy=False)
    weights = X[:, :, 0, 1].astype(np.float64, copy=False)
    s = weights.sum(axis=1, keepdims=True)
    s[s <= 0] = 1.0
    weights = weights / s

    def _wsd_i(i):
        wi = np.ascontiguousarray(weights[i], dtype=np.float64)
        Xi = np.ascontiguousarray(coords[i], dtype=np.float64)
        members = np.where(labels == labels[i])[0]
        members = members[members != i]
        if members.size == 0:
            return 1.0
        m_acc = np.zeros((n, d), dtype=np.float64)
        for j in members:
            wj = np.ascontiguousarray(weights[j], dtype=np.float64)
            Xj = np.ascontiguousarray(coords[j], dtype=np.float64)
            C = np.ascontiguousarray(ot.dist(Xi, Xj, metric="euclidean")**2, dtype=np.float64)
            Pi = ot.emd(wi, wj, C)
            T_vals = (Pi @ Xj) / (wi[:, None] + EPS)
            W2 = np.sqrt((Pi * C).sum()) + EPS
            m_acc += (Xi - T_vals) / W2
        m_vec = m_acc / members.size
        norm_sq = np.sum(m_vec**2, axis=1)
        return 1.0 - np.sqrt(np.sum(wi * norm_sq))

    depths = Parallel(n_jobs=n_jobs)(delayed(_wsd_i)(i) for i in range(M))
    return np.array(depths)


def compute_WSD_Y(Y, X, n_jobs=1):
    if Y.ndim != 4 or X.ndim != 4 or Y.shape[-1] != 2 or X.shape[-1] != 2:
        raise ValueError("Y/X must be (M, n, d, 2)")
    if (Y.shape[1], Y.shape[2]) != (X.shape[1], X.shape[2]):
        raise ValueError("Y and X must share the same (n, d) for EMD pairing.")
    My, n, d, _ = Y.shape
    Mx = X.shape[0]
    coordsY = Y[..., 0].astype(np.float64, copy=False)
    coordsX = X[..., 0].astype(np.float64, copy=False)
    weightsY = Y[:, :, 0, 1].astype(np.float64, copy=False)
    weightsX = X[:, :, 0, 1].astype(np.float64, copy=False)
    sY = weightsY.sum(axis=1, keepdims=True)
    sY[sY <= 0] = 1.0
    sX = weightsX.sum(axis=1, keepdims=True)
    sX[sX <= 0] = 1.0
    weightsY = weightsY / sY
    weightsX = weightsX / sX

    def _wsd_y(m):
        wy = np.ascontiguousarray(weightsY[m], dtype=np.float64)
        Yp = np.ascontiguousarray(coordsY[m], dtype=np.float64)
        if Mx == 0:
            return 1.0
        m_acc = np.zeros((n, d), dtype=np.float64)
        for k in range(Mx):
            wk = np.ascontiguousarray(weightsX[k], dtype=np.float64)
            Xp = np.ascontiguousarray(coordsX[k], dtype=np.float64)
            C = np.ascontiguousarray(ot.dist(Yp, Xp, metric="euclidean")**2, dtype=np.float64)
            Pi = ot.emd(wy, wk, C)
            T_vals = (Pi @ Xp) / (wy[:, None] + EPS)
            W2 = np.sqrt((Pi * C).sum()) + EPS
            m_acc += (Yp - T_vals) / W2
        m_vec = m_acc / Mx
        norm_sq = np.sum(m_vec**2, axis=1)
        return 1.0 - np.sqrt(np.sum(wy * norm_sq))

    depths = Parallel(n_jobs=n_jobs)(delayed(_wsd_y)(m) for m in range(My))
    return np.array(depths)


def to_Mnd2_from_fixed_support(X_support, W):
    X_support = np.asarray(X_support, dtype=float)
    W = np.asarray(W, dtype=float)
    n, m = W.shape
    d = X_support.shape[1]
    Xw = np.zeros((n, m, d, 2), dtype=np.float64)
    Xw[:, :, :, 0] = X_support[None, :, :]
    Xw[:, :, 0, 1] = W
    return Xw


def _pairwise_D_backend(X_support, W, backend="sinkhorn", eps=5e-3,
                        n_jobs=-1, inner_sinkhorn_iter=400):
    n = W.shape[0]
    C = sqdist(X_support, X_support)
    D = np.zeros((n, n), dtype=float)
    pairs = [(i, j) for i in range(n) for j in range(i+1, n)]

    def _pair(i, j):
        if backend == "sinkhorn":
            sdiv = max(sinkhorn_divergence(W[i], W[j], C, eps, n_iter=inner_sinkhorn_iter), 0.0)
            return i, j, np.sqrt(sdiv)
        else:
            return i, j, lp_w2_distance(W[i], W[j], C)

    results = Parallel(n_jobs=n_jobs, backend="loky")(delayed(_pair)(i, j) for (i, j) in pairs)
    for i, j, v in results:
        D[i, j] = D[j, i] = v
    np.fill_diagonal(D, 0.0)
    return D, C


# ========= Davies-Bouldin Index on Wasserstein space =========
def compute_DBI(W, labels, centers, C,
                backend="sinkhorn", eps=5e-3,
                inner_sinkhorn_iter=400, n_jobs=-1):
    """
    Davies-Bouldin Index (DBI) using Wasserstein / Sinkhorn distance.

    S_i: mean distance from points in cluster i to its center.
    M_ij: distance between cluster centers i and j.
    DBI = (1 / K') * sum_i max_{j != i} (S_i + S_j) / M_ij,
    where K' is the number of non-empty clusters.
    """
    K = len(centers)
    clusters = [np.where(labels == k)[0] for k in range(K)]
    non_empty = [k for k in range(K) if clusters[k].size > 0 and centers[k] is not None]

    if len(non_empty) <= 1:
        return np.nan

    S = np.zeros(K, dtype=float)

    def _dist_center_point(center_vec, wi):
        if backend == "sinkhorn":
            val = max(sinkhorn_divergence(center_vec, wi, C, eps,
                                          n_iter=inner_sinkhorn_iter), 0.0)
            return np.sqrt(val)
        else:
            return lp_w2_distance(center_vec, wi, C)

    for k in non_empty:
        idx = clusters[k]
        ck = np.ascontiguousarray(centers[k], dtype=np.float64)
        if idx.size == 0:
            S[k] = 0.0
            continue
        dists = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(_dist_center_point)(ck, W[i]) for i in idx
        )
        S[k] = float(np.mean(dists))

    M = np.zeros((K, K), dtype=float)

    def _dist_center_center(ci, cj):
        if backend == "sinkhorn":
            val = max(sinkhorn_divergence(ci, cj, C, eps,
                                          n_iter=inner_sinkhorn_iter), 0.0)
            return np.sqrt(val)
        else:
            return lp_w2_distance(ci, cj, C)

    for i in non_empty:
        ci = np.ascontiguousarray(centers[i], dtype=np.float64)
        for j in non_empty:
            if j <= i:
                continue
            cj = np.ascontiguousarray(centers[j], dtype=np.float64)
            dij = _dist_center_center(ci, cj)
            M[i, j] = M[j, i] = dij

    R = np.full(K, np.nan, dtype=float)
    for i in non_empty:
        best = -np.inf
        for j in non_empty:
            if j == i:
                continue
            if M[i, j] <= 0:
                continue
            val = (S[i] + S[j]) / M[i, j]
            if val > best:
                best = val
        if best > -np.inf:
            R[i] = best

    dbi = float(np.nanmean(R[non_empty]))
    return dbi


# ============================================================
# Validation over K (K-means or K-median)
# ============================================================

def validate_over_Ks_fixed(
    X_support, W,
    Ks=tuple(range(2, 10)),
    method="kmeans",
    backend="sinkhorn", eps=5e-3,
    pam_warm_start=True, use_sa=False, T0=0.5, alpha=0.9,
    max_outer_rounds=12, bary_max_iter=1000, bary_tol=1e-9,
    inner_sinkhorn_iter=400, assign_n_jobs=-1, seed=42,
    irls_max_iter=20, irls_tol=1e-7,
    n_jobs_D=-1,
):
    """
    Sweep K, run Wasserstein clustering, record silhouette / RWSD / DBI.

    Parameters
    ----------
    method : {"kmeans", "kmedian"}
        Selects the center-update rule used inside the inner clustering loop.
    """
    if method not in ("kmeans", "kmedian"):
        raise ValueError("method must be one of {'kmeans', 'kmedian'}")

    D, C = _pairwise_D_backend(
        X_support, W, backend=backend, eps=eps,
        n_jobs=n_jobs_D, inner_sinkhorn_iter=inner_sinkhorn_iter,
    )

    Xw = to_Mnd2_from_fixed_support(X_support, W)
    n = W.shape[0]

    sil_vals, red_vals, dbi_vals, perK = {}, {}, {}, {}

    for K in Ks:
        labels, centers, _ = wasserstein_k_clustering_fixed_support(
            X_support, W, K,
            method=method,
            backend=backend, eps=eps,
            pam_warm_start=pam_warm_start, use_sa=use_sa, T0=T0, alpha=alpha,
            max_outer_rounds=max_outer_rounds,
            bary_max_iter=bary_max_iter, bary_tol=bary_tol,
            inner_sinkhorn_iter=inner_sinkhorn_iter,
            assign_n_jobs=assign_n_jobs, seed=seed + K,
            irls_max_iter=irls_max_iter, irls_tol=irls_tol,
            D_precomputed=D,
        )

        # Silhouette based on pairwise W_2 / Sinkhorn distance.
        sil = float(silhouette_score(D, labels, metric='precomputed'))

        # ReD = Dw - Db (Wasserstein spatial depth, within - between).
        Dw = compute_WSD_X(Xw, labels, n_jobs=assign_n_jobs)
        Db = np.full(n, np.nan, dtype=float)
        for c in range(K):
            idx_c = np.where(labels == c)[0]
            if idx_c.size == 0:
                continue
            ref_sets = []
            for t in range(K):
                if t == c:
                    continue
                idx_t = np.where(labels == t)[0]
                if idx_t.size > 0:
                    ref_sets.append(Xw[idx_t])
            for i in idx_c:
                if len(ref_sets) == 0:
                    Db[i] = np.nan
                else:
                    yi = Xw[i:i+1]
                    best = -np.inf
                    for Xref in ref_sets:
                        dy = compute_WSD_Y(yi, Xref, n_jobs=1)
                        if dy[0] > best:
                            best = float(dy[0])
                    Db[i] = best
        ReD = Dw - Db
        mean_ReD = float(np.nanmean(ReD))

        dbi = compute_DBI(
            W=W, labels=labels, centers=centers, C=C,
            backend=backend, eps=eps,
            inner_sinkhorn_iter=inner_sinkhorn_iter,
            n_jobs=assign_n_jobs,
        )

        sil_vals[K] = sil
        red_vals[K] = mean_ReD
        dbi_vals[K] = dbi
        perK[K] = dict(
            labels=labels,
            silhouette=sil,
            Dw=Dw,
            Db=Db,
            ReD=ReD,
            mean_ReD=mean_ReD,
            dbi=dbi,
            centers=centers,
        )
        print(f"[{method}] K={K}: silhouette={sil:.6f}, mean_ReD={mean_ReD:.6f}, DBI={dbi:.6f}")

    K_star_sil = max(sil_vals, key=lambda k: sil_vals[k])
    K_star_red = max(red_vals, key=lambda k: red_vals[k])
    K_star_dbi = min(dbi_vals, key=lambda k: dbi_vals[k] if np.isfinite(dbi_vals[k]) else np.inf)

    return dict(
        method=method,
        D=D, C=C, perK=perK,
        sil_vals=sil_vals,
        red_vals=red_vals,
        dbi_vals=dbi_vals,
        K_star_sil=K_star_sil,
        K_star_red=K_star_red,
        K_star_dbi=K_star_dbi,
    )


# Backward-compatible alias.
def validate_over_Ks_kmeans_fixed(*args, **kwargs):
    """Deprecated alias. Calls validate_over_Ks_fixed(method='kmeans')."""
    kwargs.setdefault("method", "kmeans")
    return validate_over_Ks_fixed(*args, **kwargs)

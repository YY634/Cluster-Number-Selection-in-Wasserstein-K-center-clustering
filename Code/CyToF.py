#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from sklearn.metrics import silhouette_score
from sklearn.manifold import MDS
import ot

EPS = 1e-12

# ============================================================
# Free-support Wasserstein K-median clustering & validation
# ============================================================

# ===================== WSD (depth) =====================

def compute_WSD_X(X, labels, n_jobs=-1):
    """
    Wasserstein spatial depth of each distribution in X with respect
    to its own cluster.

    Parameters
    ----------
    X : ndarray, shape (M, n, d, 2)
        For each distribution: coordinates and weights on a per-cloud support.
    labels : ndarray, shape (M,)
        Cluster labels for each distribution.
    n_jobs : int
        Number of parallel jobs.

    Returns
    -------
    depths : ndarray, shape (M,)
        Depth values in [0, 1], larger is deeper.
    """
    if X.ndim != 4 or X.shape[-1] != 2:
        raise ValueError("X must be (M, n, d, 2)")
    M, n, d, _ = X.shape
    coords  = X[..., 0].astype(np.float64, copy=False)
    weights = X[:, :, 0, 1].astype(np.float64, copy=False)
    s = weights.sum(axis=1, keepdims=True); s[s <= 0] = 1.0
    weights = weights / s

    def _wsd_i(i):
        wi = np.ascontiguousarray(weights[i], dtype=np.float64)
        Xi = np.ascontiguousarray(coords[i],  dtype=np.float64)
        members = np.where(labels == labels[i])[0]
        members = members[members != i]
        if members.size == 0:
            # If the point is the only member of its cluster, give full depth
            return 1.0
        m_acc = np.zeros((n, d), dtype=np.float64)
        for j in members:
            wj = np.ascontiguousarray(weights[j], dtype=np.float64)
            Xj = np.ascontiguousarray(coords[j],  dtype=np.float64)
            C  = np.ascontiguousarray(ot.dist(Xi, Xj, metric="euclidean")**2, dtype=np.float64)
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
    """
    Wasserstein spatial depth of Y with respect to a reference set X.

    Parameters
    ----------
    Y : ndarray, shape (My, n, d, 2)
        Distributions whose depth is to be evaluated.
    X : ndarray, shape (Mx, n, d, 2)
        Reference distributions.
    n_jobs : int
        Number of parallel jobs.

    Returns
    -------
    depths : ndarray, shape (My,)
        Depth of each distribution in Y with respect to X.
    """
    if Y.ndim != 4 or X.ndim != 4 or Y.shape[-1] != 2 or X.shape[-1] != 2:
        raise ValueError("Y/X must be (M, n, d, 2)")
    if (Y.shape[1], Y.shape[2]) != (X.shape[1], X.shape[2]):
        raise ValueError("Y and X must share the same (n, d) for EMD pairing.")
    My, n, d, _ = Y.shape
    Mx = X.shape[0]
    coordsY  = Y[..., 0].astype(np.float64, copy=False)
    coordsX  = X[..., 0].astype(np.float64, copy=False)
    weightsY = Y[:, :, 0, 1].astype(np.float64, copy=False)
    weightsX = X[:, :, 0, 1].astype(np.float64, copy=False)
    sY = weightsY.sum(axis=1, keepdims=True); sY[sY <= 0] = 1.0
    sX = weightsX.sum(axis=1, keepdims=True); sX[sX <= 0] = 1.0
    weightsY = weightsY / sY
    weightsX = weightsX / sX

    def _wsd_y(m):
        wy = np.ascontiguousarray(weightsY[m], dtype=np.float64)
        Yp = np.ascontiguousarray(coordsY[m],  dtype=np.float64)
        if Mx == 0:
            return 1.0
        m_acc = np.zeros((n, d), dtype=np.float64)
        for k in range(Mx):
            wk = np.ascontiguousarray(weightsX[k], dtype=np.float64)
            Xp = np.ascontiguousarray(coordsX[k],  dtype=np.float64)
            C  = np.ascontiguousarray(ot.dist(Yp, Xp, metric="euclidean")**2, dtype=np.float64)
            Pi = ot.emd(wy, wk, C)
            T_vals = (Pi @ Xp) / (wy[:, None] + EPS)
            W2 = np.sqrt((Pi * C).sum()) + EPS
            m_acc += (Yp - T_vals) / W2
        m_vec = m_acc / Mx
        norm_sq = np.sum(m_vec**2, axis=1)
        return 1.0 - np.sqrt(np.sum(wy * norm_sq))

    depths = Parallel(n_jobs=n_jobs)(delayed(_wsd_y)(m) for m in range(My))
    return np.array(depths)


def list_to_mnd2_equal(clouds):
    """
    Pack a list of point clouds into the (M, n, d, 2) representation
    used by the WSD routines, assuming equal weights within each cloud.
    """
    clouds = [np.asarray(c, float) for c in clouds]
    m, d = clouds[0].shape
    Xw = np.zeros((len(clouds), m, d, 2), dtype=np.float64)
    for i, c in enumerate(clouds):
        if c.shape != (m, d):
            raise ValueError("All clouds must share the same (m, d).")
        Xw[i, :, :, 0] = c
        Xw[i, :, 0, 1] = 1.0 / m
    return Xw

# ===================== Sinkhorn distances =====================

def _sinkhorn_divergence_log(X, Y, reg=0.12):
    """
    Sinkhorn divergence between two empirical distributions with uniform weights,
    computed in log-domain to improve numerical stability.
    """
    a = np.full(len(X), 1.0/len(X))
    b = np.full(len(Y), 1.0/len(Y))
    Mxy = ot.dist(X, Y, metric='sqeuclidean')
    Mxx = ot.dist(X, X, metric='sqeuclidean')
    Myy = ot.dist(Y, Y, metric='sqeuclidean')
    ot_xy = ot.sinkhorn2(a, b, Mxy, reg, method='sinkhorn_log', stopThr=1e-6)
    ot_xx = ot.sinkhorn2(a, a, Mxx, reg, method='sinkhorn_log', stopThr=1e-6)
    ot_yy = ot.sinkhorn2(b, b, Myy, reg, method='sinkhorn_log', stopThr=1e-6)
    return max(0.0, ot_xy - 0.5*(ot_xx + ot_yy))


def _sinkhorn_divergence_log_weighted(X, a, Y, b, reg=0.12):
    """
    Sinkhorn divergence between two weighted empirical distributions.
    """
    a = np.asarray(a, float); b = np.asarray(b, float)
    a = a/np.sum(a); b = b/np.sum(b)
    Mxy = ot.dist(X, Y, metric='sqeuclidean')
    Mxx = ot.dist(X, X, metric='sqeuclidean')
    Myy = ot.dist(Y, Y, metric='sqeuclidean')
    ot_xy = ot.sinkhorn2(a, b, Mxy, reg, method='sinkhorn_log', stopThr=1e-6)
    ot_xx = ot.sinkhorn2(a, a, Mxx, reg, method='sinkhorn_log', stopThr=1e-6)
    ot_yy = ot.sinkhorn2(b, b, Myy, reg, method='sinkhorn_log', stopThr=1e-6)
    return max(0.0, ot_xy - 0.5*(ot_xx + ot_yy))


def compute_pairwise_D(distributions, reg=0.12, n_jobs=-1, normalize_global_std=False, verbose=True):
    """
    Compute pairwise Sinkhorn distances between empirical clouds.

    Parameters
    ----------
    distributions : list of ndarray
        Each element is (m_i, d) array of points.
    reg : float
        Entropic regularization.
    n_jobs : int
        Parallel jobs.
    normalize_global_std : bool
        If True, globally scale all points by overall std before distance.
    verbose : bool
        If True, print scaling information.

    Returns
    -------
    D : ndarray, shape (N, N)
        Pairwise distance matrix (square root of Sinkhorn divergence).
    """
    dists = distributions
    if normalize_global_std:
        allpts = np.vstack(distributions)
        scale = float(np.std(allpts)); scale = 1.0 if scale <= 0 else scale
        dists = [x/scale for x in distributions]
        if verbose:
            print(f"[D] Global std used for scaling: {scale:.6f}")
    N = len(dists)
    D = np.zeros((N, N), dtype=float)
    pairs = [(i, j) for i in range(N) for j in range(i+1, N)]

    def _pair(i, j):
        d_ij = np.sqrt(_sinkhorn_divergence_log(dists[i], dists[j], reg=reg))
        return i, j, d_ij

    results = Parallel(n_jobs=n_jobs, backend="loky")(delayed(_pair)(i, j) for (i, j) in pairs)
    for i, j, v in results:
        D[i, j] = D[j, i] = v
    np.fill_diagonal(D, 0.0)
    return D

# ===================== Numerical utilities & free-support barycenter / IRLS =====================

def _logsumexp(A, axis=None, keepdims=False):
    m = np.max(A, axis=axis, keepdims=True)
    out = m + np.log(np.sum(np.exp(A - m), axis=axis, keepdims=True))
    return out if keepdims else out.squeeze(axis)


def _sqdist(X, Y):
    return ((X[:, None, :] - Y[None, :, :])**2).sum(-1)


def _median_scale2(clouds, max_samp=4000, rng=None):
    """
    Estimate a typical squared distance scale from a random subset of points,
    used to normalize cost matrices.
    """
    Xs = clouds.reshape(-1, clouds.shape[-1])
    rng = np.random.default_rng(0) if rng is None else rng
    idx = rng.choice(Xs.shape[0], size=min(max_samp, Xs.shape[0]), replace=False)
    D2 = ((Xs[idx, None, :] - Xs[None, idx, :])**2).sum(-1)
    med = float(np.median(D2[D2 > 0]))
    return max(med, 1e-12)


def sinkhorn_log_domain(C, a, b, eps, niter=600, f0=None, g0=None, tol=1e-9):
    """
    Log-domain Sinkhorn iterations with optional warm-start (f0, g0).
    """
    k, m = C.shape
    eps = float(max(eps, 1e-12))
    Klog = -C / eps
    loga = np.log(np.clip(a, 1e-300, None))
    logb = np.log(np.clip(b, 1e-300, None))
    f = np.zeros(k) if f0 is None else f0.copy()
    g = np.zeros(m) if g0 is None else g0.copy()
    for _ in range(niter):
        f_prev, g_prev = f, g
        f = loga - _logsumexp(Klog + g[None, :], axis=1)
        g = logb - _logsumexp(Klog.T + f[None, :], axis=1)
        if max(np.max(np.abs(f - f_prev)), np.max(np.abs(g - g_prev))) < tol:
            break
    return f, g, Klog


def row_mass_and_moment_softmax(Klog, g, Y, a_row):
    """
    Compute per-row transported mass times destination positions in softmax form.
    """
    k, m = Klog.shape
    mmt = np.empty((k, Y.shape[1]))
    for p in range(k):
        z   = Klog[p, :] + g
        lse = _logsumexp(z)
        q   = np.exp(z - lse)
        mmt[p, :] = a_row[p] * (q @ Y)
    return mmt


def free_support_barycenter(clouds, k, eps_abs, scale2,
                            outer_it=40, inner_it=600,
                            eta=0.3, a_floor=2e-2, a_temp=1.8, a_unif_mix=0.10, a_mix=0.6,
                            seed=0):
    """
    Compute a semi-free-support entropic Wasserstein barycenter for a set of clouds.

    Parameters
    ----------
    clouds : ndarray, shape (n, m, d)
        Input point clouds.
    k : int
        Number of support points in the barycenter.
    eps_abs : float
        Regularization parameter in the Sinkhorn objective (absolute scale).
    scale2 : float
        Squared distance scale used to normalize cost.
    outer_it, inner_it : int
        Numbers of outer (barycenter) and inner (Sinkhorn) iterations.
    eta : float
        Relaxation parameter for updating support locations.
    a_floor, a_temp, a_unif_mix, a_mix : float
        Parameters controlling the sparse/soft support weight updates.
    seed : int
        Random seed.

    Returns
    -------
    X : ndarray, shape (k, d)
        Learned support locations.
    a : ndarray, shape (k,)
        Support weights.
    fg : list of tuples
        Potentials (f,g) for each input cloud.
    Klogs : list of ndarray
        Log-kernel matrices for each cloud.
    """
    rng = np.random.default_rng(seed)
    clouds = np.asarray(clouds, dtype=np.float64)
    n, m, d = clouds.shape
    w = np.ones(m) / m
    X_all = clouds.reshape(-1, d)

    def kmeanspp(X, kk):
        N = X.shape[0]; C = [X[rng.integers(N)]]
        d2 = ((X - C[0])**2).sum(1)
        for _ in range(1, kk):
            p = d2 / (d2.sum() + 1e-18)
            C.append(X[rng.choice(N, p=p)])
            d2 = np.minimum(d2, ((X - C[-1])**2).sum(1))
        return np.array(C)

    def fps_from_seed(X, C, kk):
        centers = C.copy() if C.size else X[rng.integers(X.shape[0])][None,:]
        Dmin = np.min(((X[:,None,:]-centers[None,:,:])**2).sum(2), axis=1)
        while centers.shape[0] < kk:
            idx = np.argmax(Dmin)
            centers = np.vstack([centers, X[idx]])
            Dmin = np.minimum(Dmin, ((X-centers[-1])**2).sum(1))
        return centers

    k1 = max(1, int(0.5*k))
    X  = fps_from_seed(X_all, kmeanspp(X_all, k1), k) if k>k1 else kmeanspp(X_all, k)
    a  = np.ones(k)/k

    fg, Klogs = [], []
    for i in range(n):
        C = _sqdist(X, clouds[i]) / scale2
        f, g, Klog = sinkhorn_log_domain(C, a, w, eps_abs, niter=inner_it)
        fg.append((f,g)); Klogs.append(Klog)

    for _ in range(outer_it):
        X_prev, a_prev = X.copy(), a.copy()

        # Update support weights a via averaged dual potentials.
        alpha_bar = np.zeros_like(a)
        for f,_ in fg:
            alpha_bar += -eps_abs * f
        alpha_bar /= len(fg)
        logits = -alpha_bar / max(a_temp,1.0); logits -= logits.max()
        a_new = np.exp(logits); a_new /= a_new.sum()
        a_new = (1-a_unif_mix)*a_new + a_unif_mix*(np.ones_like(a_new)/k)
        a_new = np.maximum(a_new, a_floor/k); a_new /= a_new.sum()
        a = (1 - a_mix) * a + a_mix * a_new

        # Update support locations X by averaging mapped positions.
        num = np.zeros_like(X); den = a.reshape(-1,1).copy()
        for (f,g),Klog,Y in zip(fg, Klogs, clouds):
            m_i = row_mass_and_moment_softmax(Klog, g, Y, a)
            num += m_i / len(clouds)
        Z = num / np.maximum(den, 1e-12)
        X = (1-eta)*X + eta*Z

        # Recompute Sinkhorn potentials with new (X,a).
        new_fg, new_Klogs = [], []
        for i in range(n):
            C = _sqdist(X, clouds[i]) / scale2
            f0,g0 = fg[i]
            f,g,Klog = sinkhorn_log_domain(C, a, w, eps_abs, niter=inner_it, f0=f0, g0=g0)
            new_fg.append((f,g)); new_Klogs.append(Klog)
        fg, Klogs = new_fg, new_Klogs

        dz = np.linalg.norm(X - X_prev) / (np.linalg.norm(X_prev)+1e-16)
        da = np.linalg.norm(a - a_prev, 1)
        if dz < 2e-6 and da < 2e-6:
            break

    return X, a, fg, Klogs


def irls_once(X, a, clouds, fg, Klogs, scale2, eps_abs, delta_coef=0.10):
    """
    One IRLS update step for free-support Wasserstein median.
    """
    n = clouds.shape[0]; k, d = X.shape
    delta = delta_coef * np.sqrt(scale2)
    num = np.zeros_like(X); den = np.zeros((k,1))
    for i in range(n):
        Y_i   = clouds[i]; Klog = Klogs[i]; g_i = fg[i][1]
        for p in range(k):
            z   = Klog[p, :] + g_i
            lse = _logsumexp(z)
            q   = np.exp(z - lse)
            r2  = ((X[p] - Y_i)**2).sum(1)
            w   = 1.0 / np.sqrt(r2 + delta**2)
            qw  = q * w
            num[p,:] += (qw @ Y_i) * a[p] / n
            den[p,0] += (qw.sum()) * a[p] / n
    return num / np.maximum(den, 1e-12)


def recompute_potentials(X, a, clouds, scale2, eps_abs, fg_warm=None, inner_it=500):
    """
    Recompute Sinkhorn dual potentials and log-kernels for updated support (X,a).
    """
    n = clouds.shape[0]
    if fg_warm is None:
        fg_warm = [(None,None)]*n
    new_fg, new_Klogs = [], []
    w = np.ones(clouds.shape[1]) / clouds.shape[1]
    for i in range(n):
        C = _sqdist(X, clouds[i]) / scale2
        f0,g0 = fg_warm[i]
        f,g,Klog = sinkhorn_log_domain(C, a, w, eps_abs, niter=inner_it, f0=f0, g0=g0)
        new_fg.append((f,g)); new_Klogs.append(Klog)
    return new_fg, new_Klogs


def irls_to_convergence(X0, a, clouds, fg, Klogs, scale2, eps_abs,
                        max_iter=30, tol=2e-4):
    """
    Run IRLS updates until convergence (or until max_iter).
    """
    X = X0.copy()
    for _ in range(1, max_iter+1):
        X_new = irls_once(X, a, clouds, fg, Klogs, scale2, eps_abs, delta_coef=0.10)
        disp = np.linalg.norm(X_new - X, axis=1)
        X = X_new
        fg, Klogs = recompute_potentials(X, a, clouds, scale2, eps_abs, fg_warm=fg)
        if disp.max() < tol:
            break
    return X, (fg, Klogs)


def ot_epsilon_value(X, a, Y, eps_abs, scale2, inner_it=400):
    """
    Compute entropically regularized OT value between a free-support measure (X,a)
    and an empirical cloud Y with uniform weights.
    """
    m = Y.shape[0]
    b = np.ones(m)/m
    C = _sqdist(X, Y) / scale2
    f,g,Klog = sinkhorn_log_domain(C, a, b, eps_abs, niter=inner_it)
    cost = 0.0; ent = 0.0
    for p in range(X.shape[0]):
        z   = Klog[p,:] + g
        lse = _logsumexp(z)
        q   = np.exp(z - lse)
        cost += a[p] * np.dot(q, C[p,:])
        ent  += a[p] * (np.log(max(a[p],1e-300)) + np.sum(q * np.log(np.clip(q,1e-300,None))))
    return cost + eps_abs*ent - eps_abs


def _init_assign_by_mean_kmeanspp(clouds, k_clusters, rng):
    """
    Simple k-means++-style initialization using cloud means as features.
    """
    n, m, d = clouds.shape
    means = clouds.mean(axis=1)
    idx0 = int(rng.integers(n)); centers = [means[idx0]]
    d2 = np.sum((means - centers[0])**2, axis=1)
    while len(centers) < k_clusters:
        p = d2 / (d2.sum() + 1e-18)
        j = int(rng.choice(n, p=p)); centers.append(means[j])
        d2 = np.minimum(d2, np.sum((means - centers[-1])**2, axis=1))
    C = np.stack(centers, axis=0)
    DM = np.sum((means[:,None,:]-C[None,:,:])**2, axis=2)
    labels = np.argmin(DM, axis=1)
    # Ensure each cluster has at least one point.
    for k in range(k_clusters):
        if np.sum(labels==k)==0:
            best = DM.min(axis=1)
            second = np.partition(DM, 1, axis=1)[:,1]
            i_star = int(np.argmax(second - best))
            labels[i_star] = k
    return labels

# ===================== PAM warm start =====================

def _pam_total_cost(D, medoids):
    dmin = np.min(D[:, medoids], axis=1)
    return float(dmin.sum())


def _pam_build_greedy(D, K, rng):
    """
    Greedy BUILD phase for PAM using a random initial medoid.
    """
    N = D.shape[0]
    medoids = [int(rng.integers(N))]
    current_cost = _pam_total_cost(D, medoids)
    while len(medoids) < K:
        best_c = None
        best_cost = np.inf
        in_set = set(medoids)
        for c in range(N):
            if c in in_set:
                continue
            cand = medoids + [c]
            cost = _pam_total_cost(D, cand)
            if cost < best_cost:
                best_cost = cost
                best_c = c
        medoids.append(best_c)
        current_cost = best_cost
    return np.array(medoids, dtype=int)


def _assign_by_medoids(D, medoids):
    DM = D[:, medoids]
    labs = np.argmin(DM, axis=1)
    return labs


def pam_kmedoids_warm_start_labels(D, K, rng):
    """
    Obtain initial labels via K-medoids (PAM) warm-start.
    """
    medoids = _pam_build_greedy(D, K, rng)
    labels = _assign_by_medoids(D, medoids)
    return labels

# ===================== Free-support K-median clustering (returns centers) =====================

def wasserstein_median_IRLS_via_barycenter(clouds, k_support, eps_abs, scale2,
                                           outer_it=35, inner_it=600, irls_iter=20, seed=0):
    """
    Compute a free-support Wasserstein median by:
      1) free-support entropic barycenter,
      2) IRLS refinement on the support.
    """
    Xc, ac, fg, Klogs = free_support_barycenter(
        clouds, k_support, eps_abs, scale2,
        outer_it=outer_it, inner_it=inner_it, seed=seed
    )
    Xc_final, (fg2, Klogs2) = irls_to_convergence(
        Xc, ac, clouds, fg, Klogs, scale2, eps_abs,
        max_iter=irls_iter, tol=2e-4
    )
    return Xc_final, ac


def wasserstein_k_median_clustering(
    clouds, K, k_support,
    eps_abs=1e-3, outer_it=35, inner_it=600, irls_iter=20,
    seed=42, n_jobs=-1, max_outer_rounds=8,
    pam_warm_start=False, D_for_pam=None
):
    """
    Free-support Wasserstein K-median clustering of empirical point clouds.

    Parameters
    ----------
    clouds : ndarray, shape (n, m, d)
        Input distributions.
    K : int
        Number of clusters.
    k_support : int
        Support size of each free-support median.
    eps_abs, outer_it, inner_it, irls_iter : see above.
    pam_warm_start : bool
        If True, use PAM warm-start with distance matrix D_for_pam.
    D_for_pam : ndarray or None
        Precomputed pairwise distance matrix (for PAM warm-start).

    Returns
    -------
    labels : ndarray, shape (n,)
        Final cluster labels.
    centers : list of dict
        Each dict contains 'support', 'weights', 'size', 'members' for a cluster.
    """
    rng = np.random.default_rng(seed)
    n, m, d = clouds.shape
    scale2 = _median_scale2(clouds, rng=rng)

    if pam_warm_start:
        if D_for_pam is None:
            raise ValueError("pam_warm_start=True requires a precomputed distance matrix D_for_pam.")
        labels = pam_kmedoids_warm_start_labels(D_for_pam, K, rng)
    else:
        labels = _init_assign_by_mean_kmeanspp(clouds, K, rng)

    for r in range(max_outer_rounds):
        clusters = [np.where(labels==k)[0] for k in range(K)]

        def _fit_one(k):
            idx = clusters[k]
            if len(idx) == 0:
                return None
            sub = clouds[idx]
            Xk, ak = wasserstein_median_IRLS_via_barycenter(
                sub, k_support, eps_abs, scale2,
                outer_it=outer_it, inner_it=inner_it, irls_iter=irls_iter, seed=seed+k
            )
            return dict(support=Xk, weights=ak, size=len(idx), members=idx)

        centers = Parallel(n_jobs=n_jobs, backend="loky")(delayed(_fit_one)(k) for k in range(K))

        def _cost_row(i):
            row = np.empty(K)
            for k in range(K):
                if centers[k] is None:
                    row[k] = np.inf
                else:
                    Xk, ak = centers[k]["support"], centers[k]["weights"]
                    row[k] = ot_epsilon_value(Xk, ak, clouds[i], eps_abs, scale2, inner_it=inner_it//2)
            return i, row

        results = Parallel(n_jobs=n_jobs, backend="loky")(delayed(_cost_row)(i) for i in range(n))
        D_assign = np.zeros((n, K))
        for i, row in results:
            D_assign[i] = row
        new_labels = np.argmin(D_assign + 1e-12*np.random.standard_normal(D_assign.shape), axis=1)
        changes = int(np.sum(new_labels != labels))
        print(f"[WS-kmedian round {r+1}] reassignment changes: {changes}")
        labels = new_labels
        if changes == 0:
            break
    return labels, centers

# ===================== DBI (median-center version based on free-support medians) =====================

def dbi_centroid_like_from_centers(distributions, labels, centers, reg=0.12):
    """
    Compute Davies–Bouldin Index using free-support K-median prototypes.

    For cluster i:
      S_i  = mean Wasserstein distance (sqrt Sinkhorn divergence) from members
             to the cluster prototype (free-support median).
      M_ij = Wasserstein distance between cluster prototypes i and j.

    DBI = mean_i max_{j != i} (S_i + S_j) / M_ij.
    """
    labels = np.asarray(labels)
    cats = np.unique(labels)
    K = len(cats)
    if K <= 1:
        return np.nan

    # Prepare cluster prototypes (Xk, ak).
    proto = {}
    for k_idx, c in enumerate(cats):
        info = centers[k_idx] if centers[k_idx] is not None else None
        if info is None:
            return np.nan
        proto[c] = (info["support"], info["weights"])

    # Within-cluster scatter S_i.
    S = {}
    for k_idx, c in enumerate(cats):
        Xc, ac = proto[c]
        members = np.where(labels == c)[0]
        if len(members) <= 1:
            S[c] = 0.0
            continue
        dlist = []
        for i in members:
            Xi = distributions[i]
            mi = np.full(len(Xi), 1.0/len(Xi))
            div = _sinkhorn_divergence_log_weighted(Xi, mi, Xc, ac, reg=reg)
            dlist.append(np.sqrt(div))
        S[c] = float(np.mean(dlist))

    # Between-center distances M_ij.
    M = np.zeros((K, K), dtype=float)
    for a_idx, ca in enumerate(cats):
        Xa, aa = proto[ca]
        for b_idx, cb in enumerate(cats):
            if a_idx == b_idx:
                M[a_idx, b_idx] = 0.0
            else:
                Xb, ab = proto[cb]
                div = _sinkhorn_divergence_log_weighted(Xa, aa, Xb, ab, reg=reg)
                M[a_idx, b_idx] = np.sqrt(div)

    # DBI.
    Rmax = []
    for i_idx, ci in enumerate(cats):
        Si = S[ci]
        worst = -np.inf
        for j_idx, cj in enumerate(cats):
            if i_idx == j_idx:
                continue
            Sj = S[cj]
            Mij = M[i_idx, j_idx]
            val = np.inf if Mij <= 0 else (Si + Sj) / Mij
            if val > worst:
                worst = val
        Rmax.append(worst)
    return float(np.mean(Rmax))

# ===================== Evaluation (with optional DBI from free-support medians) =====================

def evaluate_with_labels(
    distributions, labels,
    reg=0.12, normalize_global_std=False, n_jobs=-1, verbose=True,
    D_precomputed=None, centers_for_dbi=None
):
    """
    Evaluate a clustering of free-support distributions.

    Returns
    -------
    D : ndarray
        Pairwise distance matrix (Sinkhorn-based).
    silhouette : float
        Silhouette score based on D.
    red_mean : float
        Mean ReD (WSD-based relative depth).
    dbi : float
        DBI based on free-support median prototypes (if centers_for_dbi is given),
        otherwise NaN.
    """
    if D_precomputed is None:
        D = compute_pairwise_D(distributions, reg=reg, n_jobs=n_jobs,
                               normalize_global_std=normalize_global_std, verbose=verbose)
    else:
        D = D_precomputed

    sil = float(silhouette_score(D, labels, metric='precomputed'))

    Xw = list_to_mnd2_equal(distributions)
    Dw = compute_WSD_X(Xw, labels, n_jobs=n_jobs).astype(float)
    M = len(distributions); Db = np.full(M, -np.inf, dtype=float)
    cats = np.unique(labels)
    for c in cats:
        idx_c = np.where(labels == c)[0]
        for t in cats:
            if t == c:
                continue
            idx_t = np.where(labels == t)[0]
            if idx_t.size == 0:
                continue
            dy = compute_WSD_Y(Xw[idx_c], Xw[idx_t], n_jobs=n_jobs).astype(float)
            Db[idx_c] = np.maximum(Db[idx_c], dy)
    ReD_i = Dw - Db
    red_mean = float(np.nanmean(ReD_i))

    if centers_for_dbi is not None:
        dbi = dbi_centroid_like_from_centers(distributions, labels, centers_for_dbi, reg=reg)
    else:
        dbi = np.nan

    return D, sil, red_mean, dbi

# ===================== Select K (free-support K-median, DBI from medians) =====================

def validate_over_Ks_kmedian_free(distributions, Ks=(2,3,4,5),
                                  clustering_kwargs=None,
                                  reg=0.12, normalize_global_std=False,
                                  n_jobs=-1, rng_seed=42,
                                  plot=True,
                                  use_pam_warm_start=False):
    """
    Validate K for free-support Wasserstein K-median clustering.

    For each K in Ks:
      - run clustering,
      - compute silhouette, ReD (mean WSD-based relative depth), and DBI
        based on free-support medians as prototypes.

    Returns
    -------
    out : dict
        Contains D, per-K metrics, and selected K according to each criterion.
    """
    if clustering_kwargs is None:
        clustering_kwargs = dict(k_support=32, eps_abs=1e-3,
                                 outer_it=35, inner_it=600, irls_iter=20,
                                 seed=rng_seed, n_jobs=n_jobs, max_outer_rounds=8)

    clouds = np.stack(distributions, axis=0)   # (N,m,d)
    sil_vals, red_vals, dbi_vals, labels_perK, centers_perK = {}, {}, {}, {}, {}

    # Precompute distance matrix once (for silhouette/MDS/PAM).
    D_all = compute_pairwise_D(distributions, reg=reg, n_jobs=n_jobs,
                               normalize_global_std=normalize_global_std, verbose=True)

    for K in Ks:
        print(f"\n=== Free-support WS K-median with K={K} ===")
        labs, centers = wasserstein_k_median_clustering(
            clouds, K, pam_warm_start=use_pam_warm_start, D_for_pam=D_all,
            **clustering_kwargs
        )
        labels_perK[K]  = labs
        centers_perK[K] = centers

        # Unified evaluation (DBI via free-support medians).
        _, sil, red, dbi = evaluate_with_labels(
            distributions, labs, reg=reg, normalize_global_std=normalize_global_std,
            n_jobs=n_jobs, verbose=False, D_precomputed=D_all, centers_for_dbi=centers
        )
        sil_vals[K] = sil
        red_vals[K] = red
        dbi_vals[K] = dbi
        print(f"[K={K}] Silhouette={sil:.4f} | ReD(mean)={red:.4f} | DBI={dbi:.4f}")

    Ks_ok = [k for k in Ks if np.isfinite(sil_vals[k])]
    K_star_sil = max(Ks_ok, key=lambda k: sil_vals[k]) if Ks_ok else None
    K_star_red = max(Ks,    key=lambda k: red_vals[k]) if len(Ks) > 0 else None
    K_star_dbi = min(Ks,    key=lambda k: dbi_vals[k]) if len(Ks) > 0 else None

    print("\n=== Summary over K (free-support WS K-median) ===")
    for K in Ks:
        print(f"K={K}: Silhouette={sil_vals[K]:.4f} | ReD(mean)={red_vals[K]:.4f} | DBI={dbi_vals[K]:.4f}")
    print(f"Best K by Silhouette: {K_star_sil}")
    print(f"Best K by ReD:        {K_star_red}")
    print(f"Best K by DBI:        {K_star_dbi}")

    if plot:
        Ks_sorted = sorted(Ks)

        plt.figure(figsize=(5.2,3.8))
        plt.plot(Ks_sorted, [sil_vals[k] for k in Ks_sorted], marker='o')
        if K_star_sil is not None:
            plt.axvline(K_star_sil, ls='--', alpha=.6)
        plt.title('Silhouette vs K'); plt.xlabel('K'); plt.ylabel('Silhouette')
        plt.tight_layout(); plt.show()

        plt.figure(figsize=(5.2,3.8))
        plt.plot(Ks_sorted, [red_vals[k] for k in Ks_sorted], marker='o')
        if K_star_red is not None:
            plt.axvline(K_star_red, ls='--', alpha=.6)
        plt.title('ReD (WSD) vs K'); plt.xlabel('K'); plt.ylabel('ReD mean')
        plt.tight_layout(); plt.show()

        plt.figure(figsize=(5.2,3.8))
        plt.plot(Ks_sorted, [dbi_vals[k] for k in Ks_sorted], marker='o')
        if K_star_dbi is not None:
            plt.axvline(K_star_dbi, ls='--', alpha=.6)
        plt.title('DBI vs K'); plt.xlabel('K'); plt.ylabel('DBI (lower is better)')
        plt.tight_layout(); plt.show()

        X2 = MDS(dissimilarity='precomputed', random_state=0).fit_transform(D_all)
        for title, Ksel in [('Best by Silhouette', K_star_sil),
                            ('Best by ReD',        K_star_red),
                            ('Best by DBI',        K_star_dbi)]:
            if Ksel is None:
                continue
            labs = labels_perK[Ksel]
            cmap = plt.get_cmap('tab10') if len(np.unique(labs))<=10 else plt.get_cmap('tab20')
            plt.figure(figsize=(5.2,5.2))
            plt.scatter(X2[:,0], X2[:,1], c=labs, cmap=cmap, s=36, edgecolors='k')
            plt.title(f"MDS on D — {title} (K={Ksel})")
            plt.tight_layout(); plt.show()

    return dict(
        D=D_all,
        sil_vals=sil_vals, red_vals=red_vals, dbi_vals=dbi_vals,
        K_star_sil=K_star_sil, K_star_red=K_star_red, K_star_dbi=K_star_dbi,
        labels_perK=labels_perK, centers_perK=centers_perK
    )

# ===================== Visualization helpers =====================

def plot_true_clouds(distributions, y_true, centers=None, title="Empirical clouds"):
    """
    Quick scatter-plot of 2D empirical clouds, optionally with cluster centers.
    """
    colors = {0: "#1f77b4", 1: "#2ca02c", 2: "#d62728", -1: "#7f7f7f"}
    plt.figure(figsize=(6,6))
    for i, X in enumerate(distributions):
        plt.scatter(X[:,0], X[:,1], s=6, c=colors.get(int(y_true[i]), "#7f7f7f"), alpha=0.25)
    if centers is not None:
        for j, ct in enumerate(centers):
            plt.scatter([ct[0]], [ct[1]], c=colors.get(j, "#000000"),
                        s=120, marker='*', edgecolors='k', linewidths=1.0)
    plt.title(title); plt.axis('equal'); plt.tight_layout(); plt.show()

# ============================================================
# Main script: load empirical CSV, build distributions, validate K
# ============================================================

ROOT = Path(".")
EMP_CSV = ROOT / "P1_empirical_1000_norm.csv"

KS = (2, 3, 4, 5, 6)
CLUSTERING_KW = dict(
    k_support=1, eps_abs=1e-3,
    outer_it=35, inner_it=600, irls_iter=20,
    seed=42, n_jobs=-1, max_outer_rounds=8
)
REG = 0.10
NORMALIZE_GLOBAL_STD = False
N_JOBS = -1
RNG_SEED_VALID = 42
PLOT = True

DO_EXTRA_ZSCORE = False

if not EMP_CSV.exists():
    sys.exit(f"Missing {EMP_CSV.name}.")

df = pd.read_csv(EMP_CSV)

df = df.drop(columns=["panel", "source_file"], errors="ignore")

need_cols = {"sample_id", "label"}
if not need_cols.issubset(df.columns):
    sys.exit(f"{EMP_CSV.name} lacks required columns {need_cols}.")

exclude = {"sample_id", "label"}
marker_cols = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
if not marker_cols:
    sys.exit("No numeric marker columns detected.")

df[marker_cols] = df[marker_cols].apply(pd.to_numeric, errors="coerce")
before = len(df)
df = df.dropna(subset=marker_cols, how="any").reset_index(drop=True)
dropped = before - len(df)
if dropped > 0:
    print(f"[WARN] Dropped {dropped} rows with NaN in marker columns.")

if DO_EXTRA_ZSCORE:
    X = df[marker_cols].to_numpy(dtype=float)
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd == 0.0] = 1.0
    df.loc[:, marker_cols] = (X - mu) / sd
    print("[INFO] Applied an extra global Z-score on top of your normalized CSV.")

# -------------------- Group rows by sample_id -> distributions --------------------
distributions, dist_ids, dist_labels = [], [], []
sizes = []

for sid, g in df.groupby("sample_id", sort=True):
    labs = g["label"].unique()
    if len(labs) != 1:
        warnings.warn(f"[WARN] sample_id={sid} has multiple labels {labs}; using the first.")
    X = g[marker_cols].to_numpy(dtype=float, copy=False)
    if X.shape[0] == 0:
        continue
    distributions.append(X)
    dist_ids.append(str(sid))
    dist_labels.append(str(labs[0]))
    sizes.append(X.shape[0])

if len(distributions) < 2:
    sys.exit("Need at least 2 distributions (distinct sample_id) to run validation.")

print(f"[INFO] Loaded {len(distributions)} empirical distributions from {EMP_CSV.name}.")
print(f"[INFO] Marker dimension d = {len(marker_cols)}")
print(f"[INFO] Cells per distribution (min/median/max): "
      f"{min(sizes)}/{int(np.median(sizes))}/{max(sizes)}")
labs, cnts = np.unique(dist_labels, return_counts=True)
print("[INFO] Label counts per distribution:", dict(zip(labs, cnts)))

# -------------------- Run validation --------------------
out = validate_over_Ks_kmedian_free(
    distributions,
    Ks=KS,
    clustering_kwargs=CLUSTERING_KW,
    reg=REG,
    normalize_global_std=NORMALIZE_GLOBAL_STD,
    n_jobs=N_JOBS,
    rng_seed=RNG_SEED_VALID,
    plot=PLOT,
    use_pam_warm_start=False
)

# -------------------- Report --------------------
print("\n=== Validation over K (from P1_empirical_1000_norm.csv grouped by sample_id) ===")
print("Best K by Silhouette:", out.get("K_star_sil"))
print("Best K by ReD:",        out.get("K_star_red"))
print("Best K by DBI:",        out.get("K_star_dbi"))
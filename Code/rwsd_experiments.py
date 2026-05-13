"""
rwsd_experiments.py
===================

Five data-generating processes used in the RWSD simulations, plus a
registry that the multi-seed runner consumes. The generators are taken
verbatim from the experiment cells of
``rwsd_simulations_data_generators_in_experiments.ipynb``; the only thing
removed is the inline visualization / validation driver code, which now
lives in ``run_multi_seed.py``.

Each entry in ``EXPERIMENTS`` carries:
    - ``generator``       : a callable returning ``(clouds, labels, ...)``
    - ``default_args``    : kwargs passed to the generator (excluding ``seed``)
    - ``true_K``          : ground-truth number of clusters (excluding noise)
    - ``method``          : ``"kmeans"`` or ``"kmedian"``
    - ``Ks``              : list of K values to sweep
    - ``has_noise``       : whether label ``-1`` is used for noise distributions
"""

import numpy as np


# ============================================================
# Experiment 1: imbalanced dispersion (Gaussian far-field)
# Matches paper Section 5.1: each empirical distribution is N(c, I).
# ============================================================

def generate_two_families_gaussian_far(n_per_family=50, m_points=100, cov=None, seed=0):
    """
    Two distribution families with strongly imbalanced dispersion.
      Cluster 1: centers uniformly in disk of radius 100 around (0, 0).
      Cluster 2: centers uniformly in disk of radius 0.1 around (120, 0).
    Each empirical distribution is a 2D Gaussian cloud N(mu, cov).
    """
    rng = np.random.default_rng(seed)
    if cov is None:
        cov = np.eye(2)

    theta_A = rng.uniform(0.0, 2 * np.pi, size=n_per_family)
    r_A = 100.0 * np.sqrt(rng.random(n_per_family))
    mus_A = np.stack([r_A * np.cos(theta_A), r_A * np.sin(theta_A)], axis=1)

    theta_B = rng.uniform(0.0, 2 * np.pi, size=n_per_family)
    r_B = 0.1 * np.sqrt(rng.random(n_per_family))
    mus_B = np.stack([120.0 + r_B * np.cos(theta_B), r_B * np.sin(theta_B)], axis=1)

    clouds = []
    for mu in mus_A:
        clouds.append(rng.multivariate_normal(mean=mu, cov=cov, size=m_points).astype(float))
    for mu in mus_B:
        clouds.append(rng.multivariate_normal(mean=mu, cov=cov, size=m_points).astype(float))

    labels = np.r_[np.zeros(n_per_family, dtype=int), np.ones(n_per_family, dtype=int)]
    return clouds, labels, mus_A, mus_B


# ============================================================
# Experiment 2: unbalanced separations (three reflected Gaussians)
# ============================================================

def generate_three_families_uniform_reflect(
    n1=200, n2=200, n3=200, m_points=50, sigma=0.12, seed=42,
):
    rng = np.random.default_rng(seed)

    x1 = rng.uniform(-1.8, -1.7, size=n1)
    y1 = rng.uniform(0.0, 0.5, size=n1)
    mus1 = np.column_stack([x1, y1])

    if n2 == n1:
        x2 = -2.0 - x1
    else:
        idx = rng.integers(0, len(x1), size=n2)
        x2 = -2.0 - x1[idx]
    y2 = rng.uniform(0.0, 0.5, size=n2)
    mus2 = np.column_stack([x2, y2])

    x3 = rng.uniform(20.0, 21.0, size=n3)
    y3 = rng.uniform(0.0, 0.5, size=n3)
    mus3 = np.column_stack([x3, y3])

    cov = (sigma ** 2) * np.eye(2)
    clouds = []
    for mu in np.vstack([mus1, mus2, mus3]):
        X = rng.multivariate_normal(mean=mu, cov=cov, size=m_points)
        clouds.append(X.astype(float))

    labels = np.r_[
        np.zeros(n1, dtype=int),
        np.ones(n2, dtype=int),
        2 * np.ones(n3, dtype=int),
    ]
    return clouds, labels, mus1, mus2, mus3


# ============================================================
# Experiment 3: heterogeneous scales (three concentric Gaussians)
# ============================================================

def generate_three_gaussian_families_tri_sigma(nA=100, nB=100, nC=100, m_points=200, seed=0):
    """
    Three 2D Gaussian families with common mean (0, 0) but different scales.
      A: sigma ~ Triangular(0.0, 0.2, 0.4)
      B: sigma ~ Triangular(1.0, 1.2, 1.4)
      C: sigma ~ Triangular(5.0, 5.2, 5.4)
    """
    rng = np.random.default_rng(seed)
    mu0 = np.zeros(2)
    I2 = np.eye(2)

    sigA = rng.triangular(left=0.0, mode=0.2, right=0.4, size=nA)
    sigB = rng.triangular(left=1.0, mode=1.2, right=1.4, size=nB)
    sigC = rng.triangular(left=5.0, mode=5.2, right=5.4, size=nC)

    clouds, sigmas = [], []
    for s in sigA:
        X = rng.multivariate_normal(mean=mu0, cov=(s ** 2) * I2, size=m_points)
        clouds.append(X.astype(float))
        sigmas.append(s)
    for s in sigB:
        X = rng.multivariate_normal(mean=mu0, cov=(s ** 2) * I2, size=m_points)
        clouds.append(X.astype(float))
        sigmas.append(s)
    for s in sigC:
        X = rng.multivariate_normal(mean=mu0, cov=(s ** 2) * I2, size=m_points)
        clouds.append(X.astype(float))
        sigmas.append(s)

    labels_true = np.r_[np.zeros(nA, int), np.ones(nB, int), 2 * np.ones(nC, int)]
    sigmas = np.array(sigmas, dtype=float)
    return clouds, labels_true, dict(sigA=sigA, sigB=sigB, sigC=sigC), sigmas


# ============================================================
# Experiment 4: outlier contamination (3 Gaussians + box noise)
# ============================================================

def triangle_centers_for_noise(rho=1.5):
    return (
        np.array([-rho, 0.0]),
        np.array([rho, 0.0]),
        np.array([0.0, np.sqrt(3) * rho]),
    )


def build_distributions_three_gaussian_families_with_box_noise(
    n_each=60, m_points=200, sigma_inner=0.08, sigma_mean=0.4, rho=0.8,
    noise_per_family=40, box_pad=0.8, seed=42,
):
    """Three Gaussian families plus uniformly placed noise clouds in a bounding box."""
    rng = np.random.default_rng(seed)
    c1, c2, c3 = triangle_centers_for_noise(rho=rho)

    mu1 = rng.normal(loc=c1, scale=sigma_mean, size=(n_each, 2))
    mu2 = rng.normal(loc=c2, scale=sigma_mean, size=(n_each, 2))
    mu3 = rng.normal(loc=c3, scale=sigma_mean, size=(n_each, 2))

    def _make_clouds(means):
        return [rng.normal(loc=mu, scale=sigma_inner, size=(m_points, 2)) for mu in means]

    fam1 = _make_clouds(mu1)
    fam2 = _make_clouds(mu2)
    fam3 = _make_clouds(mu3)

    all_means = np.vstack([mu1, mu2, mu3])
    min_xy = all_means.min(axis=0) - box_pad
    max_xy = all_means.max(axis=0) + box_pad

    mu_noise = rng.uniform(low=min_xy, high=max_xy, size=(3 * noise_per_family, 2))
    noise = _make_clouds(mu_noise)

    distributions = [*fam1, *fam2, *fam3, *noise]
    y_true = np.array(
        [0] * len(fam1) + [1] * len(fam2) + [2] * len(fam3) + [-1] * len(noise),
        dtype=int,
    )
    centers = (c1, c2, c3)
    return distributions, y_true, centers, (min_xy, max_xy)


# ============================================================
# Experiment 5: heavy-tailed dispersion (Pareto-radial centers)
# ============================================================

def triangle_centers_for_pareto(rho=0.60):
    return (
        np.array([-rho, 0.0]),
        np.array([rho, 0.0]),
        np.array([0.0, np.sqrt(3) * rho]),
    )


def sample_centers_pareto(rng, n_centers, center, r_min, alpha, r_cap=None):
    U = rng.uniform(0.0, 1.0, size=n_centers)
    r = r_min * (U ** (-1.0 / float(alpha)))
    if r_cap is not None:
        r = np.minimum(r, float(r_cap))
    theta = rng.uniform(0.0, 2 * np.pi, size=n_centers)
    return np.stack(
        [center[0] + r * np.cos(theta), center[1] + r * np.sin(theta)],
        axis=1,
    )


def make_empirical_gaussians(rng, means, sigma, m_points):
    return [rng.normal(loc=mu, scale=sigma, size=(m_points, 2)) for mu in means]


def build_distributions_three_families_pareto_means_with_noise(
    n_each=45, m_points=160, sigma=0.10, rho=0.55,
    r_min=0.30, alpha=2.2, r_cap=2.2, seed=0,
):
    rng = np.random.default_rng(seed)
    c1, c2, c3 = triangle_centers_for_pareto(rho=rho)

    mu1 = sample_centers_pareto(rng, n_each, c1, r_min=r_min, alpha=alpha, r_cap=r_cap)
    mu2 = sample_centers_pareto(rng, n_each, c2, r_min=r_min, alpha=alpha, r_cap=r_cap)
    mu3 = sample_centers_pareto(rng, n_each, c3, r_min=r_min, alpha=alpha, r_cap=r_cap)

    fam1 = make_empirical_gaussians(rng, mu1, sigma, m_points)
    fam2 = make_empirical_gaussians(rng, mu2, sigma, m_points)
    fam3 = make_empirical_gaussians(rng, mu3, sigma, m_points)

    distributions = [*fam1, *fam2, *fam3]
    y_true = np.array([0] * len(fam1) + [1] * len(fam2) + [2] * len(fam3), dtype=int)
    centers = (c1, c2, c3)
    return distributions, y_true, centers


# ============================================================
# Registry
# ============================================================

def _exp1_call(seed):
    clouds, labels, _, _ = generate_two_families_gaussian_far(
        n_per_family=200, m_points=50, cov=np.eye(2), seed=seed,
    )
    return clouds, labels


def _exp2_call(seed):
    clouds, labels, _, _, _ = generate_three_families_uniform_reflect(
        n1=200, n2=200, n3=200, m_points=50, sigma=0.12, seed=seed,
    )
    return clouds, labels


def _exp3_call(seed):
    clouds, labels, _, _ = generate_three_gaussian_families_tri_sigma(
        nA=50, nB=150, nC=200, m_points=50, seed=seed,
    )
    return clouds, labels


def _exp4_call(seed):
    clouds, labels, _, _ = build_distributions_three_gaussian_families_with_box_noise(
        n_each=200, m_points=100, sigma_inner=0.08, sigma_mean=0.6, rho=0.8,
        noise_per_family=40, box_pad=0.8, seed=seed,
    )
    return clouds, labels


def _exp5_call(seed):
    clouds, labels, _ = build_distributions_three_families_pareto_means_with_noise(
        n_each=200, m_points=50, sigma=0.2, rho=1.0,
        r_min=0.35, alpha=0.45, r_cap=5.0, seed=seed,
    )
    return clouds, labels


EXPERIMENTS = {
    "exp1_imbalanced_dispersion": {
        "call": _exp1_call,
        "true_K": 2,
        "method": "kmeans",
        "Ks": list(range(2, 11)),
        "has_noise": False,
        "description": "Two Laplace families: dispersed disk vs tight disk.",
    },
    "exp2_unbalanced_separations": {
        "call": _exp2_call,
        "true_K": 3,
        "method": "kmeans",
        "Ks": list(range(2, 7)),
        "has_noise": False,
        "description": "Three Gaussian families with two close, one far.",
    },
    "exp3_heterogeneous_scales": {
        "call": _exp3_call,
        "true_K": 3,
        "method": "kmeans",
        "Ks": list(range(2, 11)),
        "has_noise": False,
        "description": "Three concentric Gaussians differing only in scale.",
    },
    "exp4_outlier_contamination": {
        "call": _exp4_call,
        "true_K": 3,
        "method": "kmedian",
        "Ks": list(range(2, 11)),
        "has_noise": True,
        "description": "Triangle Gaussians plus uniform-box noise clouds.",
    },
    "exp5_heavy_tailed_dispersion": {
        "call": _exp5_call,
        "true_K": 3,
        "method": "kmedian",
        "Ks": list(range(2, 11)),
        "has_noise": False,
        "description": "Triangle Gaussians with Pareto-radial center sampling.",
    },
}

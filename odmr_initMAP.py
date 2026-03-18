"""odmr_map2.py

MAP fitting utilities for ODMR spectra with two Lorentzian dips (plus baseline),
using a Poisson noise model and a pragmatic multi-start strategy designed for
difficult cases (two dips on same side, very close dips, shoulder-like second dip).

Main entry point (API-compatible with your previous module):
    fit_MAP_two_lorentz(nu, y, T=1.0, prior_means=None, prior_sigmas=None,
                        theta_init=None, bounds=None)

Returns:
    theta_map, I_map, result, prior_means, prior_sigmas

Notes
-----
- The optimizer (L-BFGS-B) is local; this module reduces failure modes by:
  (1) fitting a 1-dip model first,
  (2) using the 1-dip residual to generate many 2-dip initializations,
  (3) selecting the best MAP (lowest negative log-posterior) among starts.
- Priors are automatically derived from the data if not provided.
- You can choose residual type for seed generation:
    residual_mode='raw'     uses r = y - I1
    residual_mode='pearson' uses r = (y - lambda1)/sqrt(lambda1)   (Poisson-normalized)
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.signal import savgol_filter


# ------------------------ Model ------------------------
def lorentzian(nu: np.ndarray, nu0: float, gamma: float) -> np.ndarray:
    """Normalized Lorentzian: gamma^2 / ((nu-nu0)^2 + gamma^2)."""
    return gamma * gamma / ((nu - nu0) ** 2 + gamma * gamma)


def intensity_two_lorentz(nu: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """I(nu) = B - A1*L(nu;nu1,g1) - A2*L(nu;nu2,g2)."""
    B, A1, A2, nu1, nu2, g1, g2 = theta
    return B - A1 * lorentzian(nu, nu1, g1) - A2 * lorentzian(nu, nu2, g2)


#def lambda_two_lorentz(nu: np.ndarray, theta: np.ndarray, T: float = 1.0) -> np.ndarray:
#    """Poisson mean lambda_i = T * I(nu_i; theta)."""
#    return T * intensity_two_lorentz(nu, theta)


def intensity_one_lorentz(nu: np.ndarray, th: np.ndarray) -> np.ndarray:
    """1-dip model: I1(nu) = B - A*L(nu; nu0, g)."""
    B, A, nu0, g = th
    return B - A * lorentzian(nu, nu0, g)


#def lambda_one_lorentz(nu: np.ndarray, th: np.ndarray, T: float = 1.0) -> np.ndarray:
#    return T * intensity_one_lorentz(nu, th)

def lambda_two_lorentz(nu: np.ndarray, theta: np.ndarray, T: float = 1.0, scale: float = 1.0) -> np.ndarray:
    """Poisson mean lambda_i = scale * T * I(nu_i; theta)."""
    return float(scale) * T * intensity_two_lorentz(nu, theta)


def lambda_one_lorentz(nu: np.ndarray, th: np.ndarray, T: float = 1.0, scale: float = 1.0) -> np.ndarray:
    return float(scale) * T * intensity_one_lorentz(nu, th)



# ------------------------ Poisson log-likelihood ------------------------
def loglike_poisson(lam: np.ndarray, y: np.ndarray) -> float:
    """Sum_i [ y_i log lam_i - lam_i ], ignoring -log(y_i!) constants."""
    if np.any(lam <= 0):
        return -np.inf
    return float(np.sum(y * np.log(lam) - lam))


# ------------------------ Automatic priors ------------------------
def _smooth_for_stats(y: np.ndarray, frac: float = 0.03, polyorder: int = 3) -> np.ndarray:
    """Light Savitzky–Golay smoothing for robust extrema/quantiles."""
    n = int(len(y))
    win = max(7, int(frac * n) // 2 * 2 + 1)  # odd, >=7
    win = min(win, n if n % 2 == 1 else n - 1)  # ensure <= n and odd
    if win < 5:
        return y
    return savgol_filter(y, window_length=win, polyorder=min(polyorder, win - 2), mode="interp")


def make_priors_from_data(
    nu: np.ndarray,
    y: np.ndarray,
    delta_mu_bins: float = 8.0,
    delta_sigma_bins: float = 20.0,
    smooth_frac: float = 0.03,
) -> tuple[np.ndarray, np.ndarray, tuple[float, float]]:
    """Build reasonable prior_means and prior_sigmas from data (no manual peak guesses).

    Returns:
        prior_means: (7,) for (B,A1,A2,nu1,nu2,g1,g2)
        prior_sigmas: (7,)
        (delta_mu, delta_sigma): parameters for a soft Gaussian prior on Δ=nu2-nu1
    """
    nu = np.asarray(nu, float)
    y = np.asarray(y, float)
    if nu.ndim != 1 or y.ndim != 1 or len(nu) != len(y):
        raise ValueError("nu and y must be 1D arrays with the same length")

    dnu = float(nu[1] - nu[0]) if len(nu) > 1 else 1.0
    span = float(nu.max() - nu.min()) + 1e-12

    ys = _smooth_for_stats(y, frac=smooth_frac)

    B0 = float(np.percentile(y, 90))
    i0 = int(np.argmin(ys))
    nu0 = float(nu[i0])

    eps = 1e-6 * max(B0, 1.0)            # plancher relatif (quasi zéro)
    depth = float(max(B0 - ys[i0], eps)) #1 avant mais c'était faux
    # Split depth into two initial amplitudes (big + small) without a hard ratio:
    A1_0 = max(0.7 * depth, eps)
    A2_0 = max(0.7 * depth, eps) #0.3 avant

    g0 = float(max(5.0 * dnu, span / 30.0))

    delta_mu = float(delta_mu_bins * dnu)
    nu1_0 = nu0 - 0.5 * delta_mu
    nu2_0 = nu0 + 0.5 * delta_mu

    # Ensure positivity of intensity at dip centers (rough safeguard)
    B0 = max(B0, A1_0 + A2_0 + 1e-3)

    prior_means = np.array([B0, A1_0, A2_0, nu1_0, nu2_0, g0, g0], dtype=float)

    # Wide-but-not-flat: still guides against noise dips.
    prior_sigmas = np.array(
        [
            0.3 * B0,               # B
            0.8 * max(A1_0, 1.0),   # A1
            0.8 * max(A2_0, 1.0),   # A2
            0.2 * span,             # nu1
            0.2 * span,             # nu2
            1.0 * g0,               # g1
            1.0 * g0,               # g2
        ],
        dtype=float,
    )

    delta_sigma = float(delta_sigma_bins * dnu)
    return prior_means, prior_sigmas, (delta_mu, delta_sigma)


# ------------------------ 2-dip log-prior / posterior ------------------------
def logprior_two_dip(
    theta: np.ndarray,
    prior_means: np.ndarray,
    prior_sigmas: np.ndarray,
    delta_mu: float,
    delta_sigma: float,
) -> float:
    """Truncated factorized Gaussian prior + soft prior on Δ=nu2-nu1."""
    B, A1, A2, nu1, nu2, g1, g2 = theta

    # Hard constraints (truncate)
    if (B <= 0) or (A1 <= 0) or (A2 <= 0) or (g1 <= 0) or (g2 <= 0):
        return -np.inf
    if nu1 >= nu2:
        return -np.inf
    if B <= (A1 + A2):
        return -np.inf

    diffs = (theta - prior_means) / prior_sigmas
    lp = -0.5 * float(np.sum(diffs * diffs))

    delta = nu2 - nu1
    lp += -0.5 * float(((delta - delta_mu) / delta_sigma) ** 2)
    return lp


def logposterior_two_dip(
    theta: np.ndarray,
    nu: np.ndarray,
    y: np.ndarray,
    T: float,
    scale: float,
    prior_means: np.ndarray,
    prior_sigmas: np.ndarray,
    delta_mu: float,
    delta_sigma: float,
) -> float:
    lp = logprior_two_dip(theta, prior_means, prior_sigmas, delta_mu, delta_sigma)
    if not np.isfinite(lp):
        return -np.inf
    ll = loglike_poisson(lambda_two_lorentz(nu, theta, T=T, scale=scale), float(scale) * y)
    if not np.isfinite(ll):
        return -np.inf
    return lp + ll


def neg_logposterior_two_dip(
    theta: np.ndarray,
    nu: np.ndarray,
    y: np.ndarray,
    T: float,
    scale: float,
    prior_means: np.ndarray,
    prior_sigmas: np.ndarray,
    delta_mu: float,
    delta_sigma: float,
) -> float:
    v = logposterior_two_dip(theta, nu, y, T, float(scale), prior_means, prior_sigmas, delta_mu, delta_sigma)
    return 1e100 if not np.isfinite(v) else -v


# ------------------------ 1-dip fit (for seeding) ------------------------

def neg_loglike_one_dip(th: np.ndarray, nu: np.ndarray, y: np.ndarray, T: float, scale: float) -> float:
    B, A, nu0, g = th
    if (B <= 0) or (A <= 0) or (g <= 0):
        return 1e100
    lam = lambda_one_lorentz(nu, th, T=T, scale=scale)
    if np.any(lam <= 0):
        return 1e100
    y_counts = float(scale) * y
    return -loglike_poisson(lam, y_counts)



def fit_one_dip(
    nu: np.ndarray,
    y: np.ndarray,
    T: float = 1.0,
    scale: float = 1.0,
    smooth_frac: float = 0.03,
) -> tuple[np.ndarray, "OptimizeResult"]:
    """Fast Poisson MLE for 1-dip model; used only to generate good seeds."""
    nu = np.asarray(nu, float)
    y = np.asarray(y, float)

    dnu = float(nu[1] - nu[0]) if len(nu) > 1 else 1.0
    span = float(nu.max() - nu.min()) + 1e-12

    ys = _smooth_for_stats(y, frac=smooth_frac)

    B0 = float(np.percentile(y, 90))
    i0 = int(np.argmin(ys))
    nu0 = float(nu[i0])

    eps = 1e-6 * max(B0, 1.0)
    depth = float(max(B0 - ys[i0], eps))
    A0 = depth
    g0 = float(max(5.0 * dnu, span / 30.0))

    bnds = [
        (1e-6, None),                # B
        (1e-6, None),                # A
        (float(nu.min()), float(nu.max())),  # nu0
        (2.0 * dnu, span),            # g
    ]

    res = minimize(
        fun=neg_loglike_one_dip,
        x0=np.array([B0, A0, nu0, g0], dtype=float),
        args=(nu, y, T, float(scale)),
        method="L-BFGS-B",
        bounds=bnds,
    )
    return res.x, res



def residual_from_one_dip(
    nu: np.ndarray,
    y: np.ndarray,
    th1: np.ndarray,
    T: float = 1.0,
    scale: float = 1.0,
    mode: str = "raw",
) -> np.ndarray:
    """
    mode='raw':      r = y - (T * I1)                (mêmes unités que y)
    mode='pearson':  r = (y_counts - lambda_counts)/sqrt(lambda_counts)
                     avec y_counts = scale*y et lambda_counts = scale*T*I1
    """
    I1 = intensity_one_lorentz(nu, th1)
    if mode == "raw":
        return y - (T * I1)
    if mode == "pearson":
        lam = float(scale) * T * I1
        lam = np.maximum(lam, 1e-12)
        y_counts = float(scale) * y
        return (y_counts - lam) / np.sqrt(lam)
    raise ValueError("mode must be 'raw' or 'pearson'")



# ------------------------ Seed generation & multi-start MAP ------------------------
def make_seeds_from_residual(
    nu: np.ndarray,
    y: np.ndarray,
    T: float = 1.0,
    scale: float = 1.0,
    residual_mode: str = "raw",
    n_seps: int = 8,
    n_g2: int = 4,
    smooth_frac: float = 0.03,
) -> tuple[list[np.ndarray], np.ndarray, "OptimizeResult"]:
    """Generate multiple 2-dip initial points from a 1-dip fit + residual analysis.

    Returns:
        seeds: list of theta0 (B,A1,A2,nu1,nu2,g1,g2)
        th1:   fitted 1-dip parameters (B,A,nu0,g)
        res1:  OptimizeResult for the 1-dip fit
    """
    nu = np.asarray(nu, float)
    y = np.asarray(y, float)

    dnu = float(nu[1] - nu[0]) if len(nu) > 1 else 1.0
    span = float(nu.max() - nu.min()) + 1e-12

    th1, res1 = fit_one_dip(nu, y, T=T, smooth_frac=smooth_frac, scale=scale)
    B1, A, nu0, g1 = map(float, th1)

    resid = residual_from_one_dip(nu, y, th1, T=T, mode=residual_mode, scale=scale)

    # Window around main dip
    w = 10 * g1 #4.0 * g1
    mask = (nu >= nu0 - w) & (nu <= nu0 + w)
    if int(np.sum(mask)) < 5:
        mask = slice(None)

    # Main candidate: minimum of residual in the window
    idx = int(np.argmin(resid[mask]))
    nu_candidates = [float(np.asarray(nu)[mask][idx])]

    # Symmetric offsets (ultra-close to moderately close)
    seps = np.linspace(2.0 * dnu, 1.2 * g1, max(1, int(n_seps)))
    for s in seps:
        nu_candidates.extend([nu0 - float(s), nu0 + float(s)])

    # Clamp to band
    nu_min, nu_max = float(nu.min()), float(nu.max())
    nu_candidates = [min(max(v, nu_min), nu_max) for v in nu_candidates]

    # Candidate g2 values
    g2_list = [max(2.0 * dnu, float(f) * g1) for f in np.linspace(0.4, 1.2, max(1, int(n_g2)))]

    # Amplitude split for seeding (shoulder small dip)
    epsA = 1e-6 * max(B1, 1.0)
    A1_init = max(0.75 * A, epsA)
    A2_init = max(0.5* A, epsA)  #0.4 avant

    seeds: list[np.ndarray] = []
    for nu2 in nu_candidates:
        for g2 in g2_list:
            #a, b = sorted([nu0, float(nu2)])
            #B_init = max(B1, A1_init + A2_init + 1e-3)
            #seeds.append(np.array([B_init, A1_init, A2_init, a, b, g1, float(g2)], dtype=float))
            # au lieu de: a, b = sorted([nu0, nu2]); seeds.append([..., A1_init, A2_init, a, b, ...])
            if nu2 >= nu0:
                nu1, nu2_ = nu0, float(nu2)
                A1, A2 = A1_init, A2_init      # A1 attaché au dip principal (nu0)
            else:
                nu1, nu2_ = float(nu2), nu0
                A1, A2 = A2_init, A1_init      # swap amplitudes pour que le dip principal reste profond

            B_init = max(B1, A1 + A2 + 1e-3)
            seeds.append(np.array([B_init, A1, A2, nu1, nu2_, g1, float(g2)], dtype=float))


    return seeds, th1, res1


def fit_MAP_two_lorentz(
    nu,
    y,
    T: float = 1.0,
    scale: float = 1.0,
    prior_means=None,
    prior_sigmas=None,
    theta_init=None,
    bounds=None,
    *,
    residual_mode: str = "raw",
    n_seps: int = 10,
    n_g2: int = 4,
    delta_mu_bins: float = 8.0,
    delta_sigma_bins: float = 20.0,
    smooth_frac: float = 0.03,
    gmin_factor: float = 2.0,
):
    """MAP fit for 2 Lorentzian dips + constant baseline, using multi-start seeding.

    Parameters
    ----------
    nu, y : array-like (1D)
        Frequency axis and observed photon counts.
    T : float
        Integration time (or scale). Lambda = T * I.
    prior_means, prior_sigmas : array-like shape (7,), optional
        If provided, used as Gaussian prior centers/spreads for (B,A1,A2,nu1,nu2,g1,g2).
        If not provided, they are derived automatically from data.
    theta_init : array-like shape (7,), optional
        If provided, included as an additional seed (and can be the only seed if you want).
    bounds : list of 7 (low, high) tuples, optional
        L-BFGS-B box constraints. If None, defaults are created from nu-range and gmin_factor.
    residual_mode : {'raw','pearson'}
        Residual type used to generate seeds from the 1-dip diagnostic.
    n_seps, n_g2 : int
        Number of separation offsets and g2 values used to generate seeds.
    delta_mu_bins, delta_sigma_bins : float
        Hyperparameters for the soft prior on separation Δ=nu2-nu1 (expressed in bins).
    smooth_frac : float
        Smoothing fraction used for robust extrema finding in 1-dip fit and priors.
    gmin_factor : float
        Lower bound for gamma as gmin_factor * dnu.

    Returns
    -------
    theta_map : ndarray (7,)
    I_map : ndarray (len(nu),)
    result : scipy.optimize.OptimizeResult
        Best local optimum among starts. Extra diagnostics are attached as fields:
            result.one_dip_theta, result.one_dip_result, result.seeds, result.best_nlp
    prior_means, prior_sigmas : ndarray (7,), ndarray(7,)
    """
    nu = np.asarray(nu, float)
    scale = float(np.asarray(scale).item())
    y = np.asarray(y, float)
    if nu.ndim != 1 or y.ndim != 1 or len(nu) != len(y):
        raise ValueError("nu and y must be 1D arrays with the same length")

    nu_min, nu_max = float(nu.min()), float(nu.max())
    span = (nu_max - nu_min) + 1e-12
    dnu = float(nu[1] - nu[0]) if len(nu) > 1 else 1.0

    # Priors (auto if not supplied)
    delta_mu = None
    delta_sigma = None
    if prior_means is None or prior_sigmas is None:
        pm, ps, (dmu, dsig) = make_priors_from_data(
            nu, y, delta_mu_bins=delta_mu_bins, delta_sigma_bins=delta_sigma_bins, smooth_frac=smooth_frac
        )
        if prior_means is None:
            prior_means = pm
        if prior_sigmas is None:
            prior_sigmas = ps
        delta_mu, delta_sigma = dmu, dsig
    else:
        prior_means = np.asarray(prior_means, float)
        prior_sigmas = np.asarray(prior_sigmas, float)
        # Still need Δ prior hyperparameters; derive from bin defaults
        delta_mu = float(delta_mu_bins * dnu)
        delta_sigma = float(delta_sigma_bins * dnu)

    if prior_means.shape != (7,) or prior_sigmas.shape != (7,):
        raise ValueError("prior_means and prior_sigmas must have shape (7,)")

    # Bounds
    if bounds is None:
        gmin = max(1e-12, float(gmin_factor) * dnu)
        gmax = span
        bounds = [
            (1e-6, None),       # B
            (1e-6, None),       # A1
            (1e-6, None),       # A2
            (nu_min, nu_max),   # nu1
            (nu_min, nu_max),   # nu2
            (gmin, gmax),       # g1
            (gmin, gmax),       # g2
        ]

    # Seeds from 1-dip residual
    seeds, th1, res1 = make_seeds_from_residual(
        nu, y, T=T, residual_mode=residual_mode, n_seps=n_seps, n_g2=n_g2, smooth_frac=smooth_frac, scale=scale,
    )

    # If user provided theta_init, include it too (and keep it first for reproducibility)
    if theta_init is not None:
        theta_init = np.asarray(theta_init, float)
        if theta_init.shape != (7,):
            raise ValueError("theta_init must have shape (7,)")
        seeds = [theta_init] + seeds

    # Multi-start optimization
    best = None
    best_val = np.inf
    for th0 in seeds:
        r = minimize(
            fun=neg_logposterior_two_dip,
            x0=th0,
            args=(nu, y, T, float(scale), prior_means, prior_sigmas, delta_mu, delta_sigma),
            method="L-BFGS-B",
            bounds=bounds,
        )
        if np.isfinite(r.fun) and (r.fun < best_val):
            best_val = float(r.fun)
            best = r

    if best is None:
        # Should not happen; fallback to a single run at prior_means
        best = minimize(
            fun=neg_logposterior_two_dip,
            x0=np.asarray(prior_means, float),
            args=(nu, y, T, float(scale), prior_means, prior_sigmas, delta_mu, delta_sigma),
            method="L-BFGS-B",
            bounds=bounds,
        )
        best_val = float(best.fun) if np.isfinite(best.fun) else np.inf

    theta_map = np.asarray(best.x, float)
    I_map = intensity_two_lorentz(nu, theta_map)

    # Attach diagnostics without changing the return signature
    try:
        best.one_dip_theta = th1
        best.one_dip_result = res1
        best.seeds = seeds
        best.best_nlp = best_val
        best.delta_mu = delta_mu
        best.delta_sigma = delta_sigma
        best.residual_mode = residual_mode
    except Exception:
        pass

    return theta_map, I_map, best, np.asarray(prior_means, float), np.asarray(prior_sigmas, float)


# Backwards-friendly alias (if you prefer the original name)
fit_MAP_two_lorentz1 = fit_MAP_two_lorentz

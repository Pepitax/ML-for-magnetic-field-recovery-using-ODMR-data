import pandas as pd
import numpy as np
import warnings
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

from pathlib import Path
import os

from scipy.signal import find_peaks

from functools import partial

from odmr_initMAP import fit_MAP_two_lorentz
from helpers import build_x_axis, maybe_normalize, maybe_smooth

from matplotlib.backends.backend_pdf import PdfPages

import json

from helpers import (
    load_df1, load_df2,
    concatener_data_df1, concatener_data_df2,
    plot_on_ax,
    compute_B_components,
    maybe_normalize, maybe_smooth,
    reconstruction_B, plot_B_difference,
    _add_roi_box,   # (oui, underscore, mais pratique ici)
    plot_contrast_argmin,
    plot_B1_B2_DeltaB_projectionNV
)


#************************************************
#***** FITS & BASE FUNCTIONS ********************
#************************************************

######### fonctions de base pour les fits ##############

def _lorentz(x, x0, gamma):
    return 1.0 / (1.0 + ((x - x0) / gamma) ** 2)

def _gauss(x, x0, sigma):
    return np.exp(-0.5 * ((x - x0) / sigma) ** 2)

def _model_two_lorentz(x, a1, x1, g1, a2, x2, g2, b0, b1):
    # baseline linéaire (b0 + b1*x) - deux dips Lorentziens, en pratique pas toujours optimal (risque de fits incohérent à cause de grandes pentes lié à b1)
    return (b0 + b1 * x) - a1 * _lorentz(x, x1, g1) - a2 * _lorentz(x, x2, g2)

def _model_two_lorentz_b0(x, a1, x1, g1, a2, x2, g2, b0):
    # model avec b1=0, globalement les fits sont un peu mieux
    return b0 - a1 * _lorentz(x, x1, g1) - a2 * _lorentz(x, x2, g2)

def _model_tree_lorentz(x, a1, x1, g1, a2, x2, g2, a3, x3, g3, b0, b1):
    return (b0 + b1 * x) - a1 * _lorentz(x, x1, g1) - a2 * _lorentz(x, x2, g2) - a3 * _lorentz(x, x3, g3)

def _model_two_gauss(x, a1, x1, s1, a2, x2, s2, b0, b1):
    return (b0 + b1 * x) - a1 * _gauss(x, x1, s1) - a2 * _gauss(x, x2, s2)

def _model_two_gauss_b0(x, a1, x1, s1, a2, x2, s2, b0):
    return b0 - a1 * _gauss(x, x1, s1) - a2 * _gauss(x, x2, s2)


def _model_one_lorentz(x, a, x0, g, b0, b1):
    return (b0 + b1*x) - a * _lorentz(x, x0, g)

def _model_one_gauss(x, a, x0, s, b0, b1):
    return (b0 + b1*x) - a * _gauss(x, x0, s)

def _model_one_lorentz_b0(x, a, x0, g, b0):
    return b0 - a * _lorentz(x, x0, g)

def _model_one_gauss_b0(x, a, x0, s, b0):
    return b0 - a * _gauss(x, x0, s)
    
    

######### initialisations of fits  #################

def _init_guesses(x, y, amp_guess=None, b1_to_0=False, thr=0.95):
    """Guesses basique : baseline médiane + pente, minimum gauche/droite."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    if b1_to_0:
        b1=0
    else:
        b1 = (y[-1] - y[0]) / (x[-1] - x[0] + 1e-12)
    b0 = np.median(y)
    span_x = x.max() - x.min() + 1e-12
    depth_peak = max(np.median(y) - np.min(y), 1e-12)  #y.max() - y.min() + 1e-12
    span_y = np.ptp(y) + 1e-12

    mid = len(x) // 2

    left_idx  = np.argmin(y[:mid]) if mid > 1 else 0
    right_idx = np.argmin(y[mid:]) + mid if len(x) - mid > 1 else len(x) - 1
    x1, x2 = float(x[left_idx]), float(x[right_idx])

    if amp_guess is not None:
        a1 = a2 = float(amp_guess)
    else:
        a1 = max(b0 - y[left_idx] -0.01,  np.ptp(y) * 0.1)
        a2 = max(b0 - y[right_idx] -0.01, np.ptp(y) * 0.1)

    b0=np.median(y) - b1 * x[0]
    w = span_x / 20.0

    if b1_to_0:
        return (a1, x1, w, a2, x2, w, b0)
    else:
        return(a1, x1, w, a2, x2, w, b0, b1)



def _init_guesses_symmetric(x, y, amp_guess=None, b1_to_0=False, oldversion=True):
    """Guesses initiaux pour deux dips symétriques autour d'une vallée (creux global), avec largeur et écart dépendant du contraste."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)

    span_x = x.max() - x.min() + 1e-12

    # ----- baseline -----
    if b1_to_0:
        b1 = 0.0
    else:
        b1 = (y[-1] - y[0]) / (x[-1] - x[0] + 1e-12)

    y_med = np.median(y)
    b0 = float(y_med) - b1 * float(x[0])

    # ----- contraste -----
    y_min = float(y.min())
    depth = max(y_med - y_min, 1e-12)
    ptp   = np.ptp(y) + 1e-12
    c = depth / 0.12        # avant 2.0,  2.0 * depth / ptp  #grande profondeur a souvent un contraste de 0.2
    if oldversion:
        c=depth / ptp
    c = np.clip(c, 0.0, 1.0)

    # ----- centre de la vallée -----
    thr = y_med - 0.4 * depth  # seuil pour définir la vallée   #0.5
    mask_valley = y < thr

    if mask_valley.sum() >= 3:
        x0 = float(np.mean(x[mask_valley]))
    else:
        # fallback : minimum global
        x0 = float(x[np.argmin(y)])

    xmin, xmax = float(x.min()), float(x.max())
    half_sep_max = max(1e-12, min(x0 - xmin, xmax - x0))

    # ----- demi-écart en fonction du contraste -----
    d_min = span_x / 40.0
    d_max = min(span_x / 5.0, 0.99 * half_sep_max) #/5.0

    if d_max < d_min:
        d_max = d_min

    d = d_min + (1.0 - c) * (d_max - d_min)

    x1 = x0 - d
    x2 = x0 + d

    # ----- amplitude -----
    if amp_guess is not None:
        a1 = a2 = float(amp_guess)
    else:
        # amplitude typique ~ profondeur de la vallée
        a_est = max(depth, ptp * 0.1)
        a1 = a2 = float(a_est) * 0.6  #0.6
        if oldversion:
            a1 = a2 = float(a_est) * 0.5

    # ----- largeur en fonction du contraste -----
    w_min = span_x / 20.0   #50  #500
    w_max = span_x / 20.0    #5   #3
    w = w_min + (1.0 - c) * (w_max - w_min)

    if b1_to_0:
        return (a1, x1, w, a2, x2, w, b0)
    else:
        return(a1, x1, w, a2, x2, w, b0, b1)




def _init_guess_one_peak(x, y):
    """Guesses pour 1 dip + baseline linéaire."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    span = float(x.max() - x.min() + 1e-12)
    dx   = float(np.median(np.diff(x)))

    b0 = float(np.median(y))
    b1 = float((y[-1] - y[0]) / (x[-1] - x[0] + 1e-12))

    k0 = int(np.argmin(y))
    x0 = float(x[k0])
    a0 = float(max(b0 - y[k0], 0.1 * np.ptp(y)))  # amplitude ≥ 0
    w0 = float(max(span / 20.0, 2.0 * dx))        # largeur raisonnable
    return (a0, x0, w0, b0, b1)


##########  fits (avec centrage des fréquences x pour stabilité)  ##########

def fit_two_peaks(x, y, model="lorentz", min_sep_frac=0.02, maxfev=20000, amp_guess=None, init=None, maponly=None, baseline_to_0=False, scale=1.0, oldversion=True):
    """
    Fit deux dips (Lorentzien / Gaussien) + baseline linéaire ou constante.
    Return: dict(success, yfit, x1, x2, popt, note, model).
    """
    x = np.asarray(x, float); y = np.asarray(y, float)

    # --- essaye 2 init et garder la plus petite erreur ---
    # if init == "symmetric_and_map":
    #     r_sym = fit_two_peaks(x, y, model=model, min_sep_frac=min_sep_frac, maxfev=maxfev,
    #                           amp_guess=amp_guess, init="symmetric", maponly=False,
    #                           baseline_to_0=baseline_to_0, scale=scale, oldversion=False)
                              
    #     r_map = fit_two_peaks(x, y, model=model, min_sep_frac=min_sep_frac, maxfev=maxfev,
    #                           amp_guess=amp_guess, init="map", maponly=False,
    #                           baseline_to_0=baseline_to_0, scale=scale, oldversion=oldversion)

    #     if r_sym.get("success") and r_map.get("success"):
    #         err_sym = np.nanmean((y - r_sym["yfit"])**2)
    #         err_map = np.nanmean((y - r_map["yfit"])**2)
    #         return r_sym if err_sym <= err_map else r_map

    #     return r_sym if r_sym.get("success") else r_map
    if init == "symmetric_and_map":
        r_sym1 = fit_two_peaks(x, y, model=model, min_sep_frac=min_sep_frac, maxfev=maxfev,
                              amp_guess=amp_guess, init="symmetric", maponly=False,
                              baseline_to_0=baseline_to_0, scale=scale, oldversion=False)
        r_sym2 = fit_two_peaks(x, y, model=model, min_sep_frac=min_sep_frac, maxfev=maxfev,
                              amp_guess=amp_guess, init="symmetric", maponly=False,
                              baseline_to_0=baseline_to_0, scale=scale, oldversion=True)
        r_map = fit_two_peaks(x, y, model=model, min_sep_frac=min_sep_frac, maxfev=maxfev,
                              amp_guess=amp_guess, init="map", maponly=False,
                              baseline_to_0=baseline_to_0, scale=scale, oldversion=oldversion)

        if r_sym1.get("success") and r_sym2.get("success") and r_map.get("success"):
            err_sym1 = np.nanmean((y - r_sym1["yfit"])**2)
            err_sym2 = np.nanmean((y - r_sym2["yfit"])**2)
            err_map = np.nanmean((y - r_map["yfit"])**2)
            if (err_sym1 <= err_map) and (err_sym1 <= err_sym2):
                return r_sym1
            elif (err_sym2 <= err_map) and (err_sym2 <= err_sym1):
                return r_sym2
            else:
                return r_map
            
        # --- fallback: retourner le meilleur parmi ceux qui ont success ---
        if r_sym1.get("success") or r_sym2.get("success") or r_map.get("success"):
            candidates = []
            if r_sym1.get("success"):
                candidates.append((np.nanmean((y - r_sym1["yfit"])**2), r_sym1))
            if r_sym2.get("success"):
                candidates.append((np.nanmean((y - r_sym2["yfit"])**2), r_sym2))
            if r_map.get("success"):
                candidates.append((np.nanmean((y - r_map["yfit"])**2), r_map))
            return min(candidates, key=lambda t: t[0])[1]

        return r_map


    # ----- CAS MAP ONLY : arrête la fonction et garde les pics estimés par MAP uniquement -----
    if maponly:
        theta_map, I_map, result, prior_means, prior_sigmas = fit_MAP_two_lorentz( #param de base : n_g2=4, n_steps=10
            x, y, T=1.0, scale=scale, n_g2=2, n_seps=3
        )
        if not result.success:
            return {"success": False, "reason": result.message,
                    "yfit": None, "x1": None, "x2": None, "popt": None}
        B, A1, A2, nu1, nu2, g1, g2 = theta_map
        popt_adj = [A1, nu1, g1, A2, nu2, g2, B, 0.0]
        return {
            "success": True,
            "yfit": I_map,
            "x1": float(nu1),
            "x2": float(nu2),
            "popt": popt_adj, # theta_map,   # ici popt = paramètres MAP (B, A1, A2, nu1, nu2, g1, g2)
            "note": "MAP only",
            "model": "map",
        }
    # -----------------------------------------

    # --- centrage pour stabilité numérique pour les fits---
    x0ref  = float(np.mean(x))
    xprime = x - x0ref
    span   = xprime.max() - xprime.min() + 1e-12

    # ========== CHOIX DE L'INIT ==========
    # 1) init "classique"
    if not init:
        p0 = _init_guesses(xprime, y, amp_guess, b1_to_0=baseline_to_0)

    # 2) init symétrique autour du min global
    if init == "symmetric":
        p0 = _init_guesses_symmetric(xprime, y, amp_guess, b1_to_0=baseline_to_0, oldversion=oldversion)

    # 3) init à partir du MAP (2 Lorentziens + fond constant)
    if init == "map":
        theta_map, _, _, _, _ = fit_MAP_two_lorentz(x, y, T=1.0, n_g2=2, n_seps=3, scale=scale)  #MODIFIER HYPERPARAM
        B, A1, A2, nu1, nu2, g1, g2 = theta_map
        if baseline_to_0:
            p0 = [A1, nu1 - x0ref, g1, A2, nu2 - x0ref, g2, B]
        else:
            p0 = [A1, nu1 - x0ref, g1, A2, nu2 - x0ref, g2, B, 0.0]  #obligé d'ajouter 0.0 pour eviter les erreurs lors des fits, qui attendent 8 arguments, on initialise juste b1 à 0.0
    # =====================================

    if baseline_to_0:
        lower = [0, xprime.min(), span/1000, 0, xprime.min(), span/1000, -np.inf]  # b0 seul
        upper = [np.inf, xprime.max(), span,  np.inf, xprime.max(), span,  np.inf]
    else:
        lower = [0, xprime.min(), span/1000, 0, xprime.min(), span/1000, -np.inf, -np.inf]
        upper = [np.inf, xprime.max(), span,  np.inf, xprime.max(), span,  np.inf,  np.inf]


    if model == "lorentz":
        #fun = partial(_model_two_lorentz, baseline_to_zero=baseline_to_0)
        fun = _model_two_lorentz_b0 if baseline_to_0 else _model_two_lorentz
    elif model == "gauss":
        #fun = partial(_model_two_gauss, baseline_to_zero=baseline_to_0)
        fun = _model_two_gauss_b0   if baseline_to_0 else _model_two_gauss
    else:
        raise ValueError("model doit être 'lorentz' ou 'gauss'.")

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            popt, _ = curve_fit(fun, xprime, y, p0=p0, bounds=(lower, upper), maxfev=maxfev)
    except Exception as e:
        return {"success": False, "reason": str(e), "yfit": None, "x1": None, "x2": None, "popt": None}

    # popt = [a1, x1', w1, a2, x2', w2, b0', b1'] dans l’axe centré
    #a1, x1p, w1, a2, x2p, w2, b0p, b1p = popt
    if baseline_to_0:
        a1, x1p, w1, a2, x2p, w2, b0p = popt
        b1p = 0.0
    else:
        a1, x1p, w1, a2, x2p, w2, b0p, b1p = popt


    # reprojection dans l’axe original
    x1 = x1p + x0ref
    x2 = x2p + x0ref
    if x2 < x1:
        a1, x1, w1, a2, x2, w2 = a2, x2, w2, a1, x1, w1

    # baseline dans l’axe original: (b0'+b1'*(x - x0ref)) = (b0_adj + b1'*x)
    #b0_adj = b0p - b1p * x0ref
    #b1_adj = b1p
    # baseline (original vs centré)
    if baseline_to_0:
        # baseline constante : pas de transformation liée au centrage
        b0_adj = b0p
        b1_adj = 0.0
    else:
        # baseline linéaire: (b0'+b1'*(x - x0ref)) = (b0_adj + b1'*x)
        b0_adj = b0p - b1p * x0ref
        b1_adj = b1p

    #yfit = fun(x, a1, x1, w1, a2, x2, w2, b0_adj, b1_adj)
    if baseline_to_0:
        yfit = fun(x, a1, x1, w1, a2, x2, w2, b0_adj)
    else:
        yfit = fun(x, a1, x1, w1, a2, x2, w2, b0_adj, b1_adj)

    note = "pics très proches (fusion)" if abs(x2 - x1) < (min_sep_frac * (x.max()-x.min()+1e-12)) else None
    #popt_adj = [a1, x1, w1, a2, x2, w2, b0_adj, b1_adj]
    if baseline_to_0:
        popt_adj = [a1, x1, w1, a2, x2, w2, b0_adj, 0.0]      # pas de b1, mais on met 0.0 quand même pour pas causer de problème lors de l'appel par plot_fits_on_spectrograms
    else:
        popt_adj = [a1, x1, w1, a2, x2, w2, b0_adj, b1_adj]

    return {"success": True, "yfit": yfit, "x1": float(x1), "x2": float(x2),
            "popt": popt_adj, "note": note, "model": model}



def fit_one_peak(x, y, model="lorentz", maxfev=20000, x0_hint=None, w_hint=None, a_hint=None, baseline_to_0=False):
    """
    Fit 1 dip (Lorentz/Gauss) + baseline linéaire.
    Centre x pendant le fit, puis re-projette les paramètres dans l'axe original.
    Retourne dict: success, yfit, popt=[a, x0, w, b0, b1] (tous dans l'axe original).
    """
    x = np.asarray(x, float); y = np.asarray(y, float)

    x0ref  = float(np.mean(x))
    xprime = x - x0ref

    a0, x0, w0, b0, b1 = _init_guess_one_peak(xprime, y)
    if x0_hint is not None: x0 = float(x0_hint - x0ref)  # transformer l'indice original en centré
    if w_hint  is not None: w0 = float(w_hint)
    if a_hint  is not None: a0 = float(a_hint)

    span  = float(xprime.max() - xprime.min() + 1e-12)
    #lower = [0.0, xprime.min(), span/1000.0, -np.inf, -np.inf]
    #upper = [np.inf, xprime.max(), span,          np.inf,  np.inf]
    if baseline_to_0:
        lower = [0.0, xprime.min(), span/1000.0, -np.inf]
        upper = [np.inf, xprime.max(), span,      np.inf]
        p0_fit = [a0, x0, w0, b0]
    else:
        lower = [0.0, xprime.min(), span/1000.0, -np.inf, -np.inf]
        upper = [np.inf, xprime.max(), span,      np.inf,  np.inf]
        p0_fit = [a0, x0, w0, b0, b1]

    if model == "lorentz":
        #fun = partial(_model_one_lorentz, baseline_to_zero=baseline_to_0)
        fun = _model_one_lorentz_b0 if baseline_to_0 else _model_one_lorentz
    elif model == "gauss":
        #fun = partial(_model_one_gauss, baseline_to_zero=baseline_to_0)
        fun = _model_one_gauss_b0 if baseline_to_0 else _model_one_gauss
    else:
        raise ValueError("model doit être 'lorentz' ou 'gauss'.")

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            #popt, _ = curve_fit(fun, xprime, y, p0=[a0, x0, w0, b0, b1], bounds=(lower, upper), maxfev=maxfev)
            popt, _ = curve_fit(fun, xprime, y, p0=p0_fit, bounds=(lower, upper), maxfev=maxfev)

        #a, x0p, w, b0p, b1p = popt
        if baseline_to_0:
            a, x0p, w, b0p = popt            
            b1p = 0.0
        else:
            a, x0p, w, b0p, b1p = popt

        x0_orig = x0p + x0ref

        #b0_adj  = b0p - b1p * x0ref
        #b1_adj  = b1p
            # baseline (original vs centré)
        if baseline_to_0:
            # baseline constante : pas de transformation liée au centrage
            b0_adj = b0p
            b1_adj = 0.0
        else:
            # baseline linéaire: (b0'+b1'*(x - x0ref)) = (b0_adj + b1'*x)
            b0_adj = b0p - b1p * x0ref
            b1_adj = b1p

        #yfit = fun(x, a, x0_orig, w, b0_adj, b1_adj)
        if baseline_to_0:
            yfit = fun(x, a, x0_orig, w, b0_adj)
        else:
            yfit = fun(x, a, x0_orig, w, b0_adj, b1_adj)

        return {"success": True, "yfit": yfit, "popt": [a, x0_orig, w, b0_adj, b1_adj]}
    except Exception as e:
        return {"success": False, "reason": str(e), "yfit": None, "popt": None}




#***** GROS FIT GRID ********************

def _subset_by_D(
    x, y, D, side="auto", min_frac=0.5, margin=0.0, n_min=20,
    keep_if_global_min=True, prom_sigma=3.0, prom_rel=0.05
):
    """
    Rognage d'un seul côté de D, mais si le minimum global significatif
    est du côté coupé, on ne rogne pas (on garde toutes les données).

    - side="right" -> garde x >= D+margin
    - side="left"  -> garde x <= D-margin
    - side="auto"  -> garde le côté majoritaire si >= min_frac, sinon pas de rognage
    - keep_if_global_min=True : annule le rognage si la dip globale significative
      est du côté opposé au côté choisi.
    """
    x = np.asarray(x); y = np.asarray(y)
    nl = int(np.sum(x <  D))
    nr = int(np.sum(x >  D))
    n  = len(x)

    choose = side
    if side == "auto":
        if nr/n >= min_frac and nr >= n_min:
            choose = "right"
        elif nl/n >= min_frac and nl >= n_min:
            choose = "left"
        else:
            return x, y, np.ones_like(x, dtype=bool), None  # pas de rognage

    # --- Protection: si la dip globale "forte" est du côté coupé, on annule le rognage
    if keep_if_global_min and choose in ("left", "right"):
        # seuil de proéminence (significativité) : max(k*sigma_bruit, alpha*ptp)
        def _noise_level(arr):
            d = np.diff(arr)
            if len(d) < 5:
                return 1.4826 * np.median(np.abs(arr - np.median(arr)))
            return 1.4826 * np.median(np.abs(d - np.median(d))) / np.sqrt(2.0)

        sig = _noise_level(y)
        prom_min = max(prom_sigma * sig, prom_rel * np.ptp(y))

        peaks, props = find_peaks(-y, prominence=prom_min)
        if len(peaks) > 0:
            # prend la dip la plus proéminente (ou la plus basse)
            idx = peaks[np.argmax(props["prominences"])]
            xg  = x[idx]
            # côté de cette dip globale (en tenant compte d'un margin)
            if xg <= (D - margin):
                side_g = "left"
            elif xg >= (D + margin):
                side_g = "right"
            else:
                side_g = None  # trop près de D → on ne coupe pas
            if (side_g is None) or (side_g != choose):
                # la dip principale est justement du côté qu'on allait couper → pas de rognage
                return x, y, np.ones_like(x, dtype=bool), None

    # --- rognage normal
    if choose == "right":
        mask = x >= (D + margin)
    elif choose == "left":
        mask = x <= (D - margin)
    else:
        mask = np.ones_like(x, dtype=bool)

    if mask.sum() < n_min:
        return x, y, np.ones_like(x, dtype=bool), None

    return x[mask], y[mask], mask, choose


def plot_fits_on_spectrograms(df, NORMALIZE, SMOOTH_WIN,
                  center_i, center_j, half=2, share_axes=True,
                  per_panel_x=False,         # NEW: si True -> x indépendants
                  mark_two_min=False, fit_model="lorentz",
                  initialisation=None, 
                  maponly_=None,
                  min_peak_amp=0, min_curve_amp=0.02, maxcurve=2, min_separation_peak=0.02,
                  show_components=True,
                  # présentation
                  freq_mode=None,
                  grid=True,
                  legend="small",              # "small" ou "none"
                  legend_loc="lower right",
                  legend_fontsize=5,
                  panel_size=(2.2, 1.8),       # (largeur, hauteur) par sous-figure en pouces
                  # sortie PDF géant
                  pdf_path=None,               # ex: "odmr_grid.pdf" -> si None, pas de sauvegarde
                  pdf_metadata=None,
                  # NEW
                  show_plots=True,
                  # --- rognage par D ---
                  D=None,                     # ex: 2.87 ou 2.87e9 (mêmes unités que x)
                  crop_side="auto",           # "left" | "right" | "auto"
                  crop_min_frac=0.5,          # fraction mini du côté majoritaire
                  crop_margin=0.0,            # marge exclue autour de D
                  keep_if_global_min=True,    # protège la dip principale
                  prom_sigma=3.0, prom_rel=0.05,  # seuils proéminence
                  show_D=True,                # trace une ligne verticale à D si affichage
                  freq_à_gauche=False,
                  baseline_to_zero=False,
                  old_init_sym=True,
                  n_peaks=2,
                  ):              
    """
    Grille de sweeps autour de (center_i, center_j).
    Retourne un DataFrame listant, pour chaque (i,j), les deux abscisses utilisées pour les axvline.

    Colonnes retournées: i, j, xv1, xv2, mode
      - two_peaks : xv1=x_left, xv2=x_right (2 pics retenus)
      - one_peak  : xv1=xv2=x_peak (1 pic refitté)
      - minima    : xv1, xv2 = positions des minima (fit échoué)
      - none      : NaN, NaN (mark_two_min=False)

    Paramètre:
      - show_plots: si False, ne construit ni n'affiche de figure (fits uniquement).
    """

    records = []  # on accumule ce qu'on va retourner

    n_i = int(df["i"].max() + 1)
    n_j = int(df["j"].max() + 1)

    # Étendue de la grille
    i_range = np.arange(max(0, center_i - half), min(n_i, center_i + half + 1))
    j_range = np.arange(max(0, center_j - half), min(n_j, center_j + half + 1))
    if i_range.size == 0 or j_range.size == 0:
        raise ValueError("ROI hors limites — ajuste center_i/center_j/half.")

    # Figure (seulement si on affiche)
    if show_plots:
        figsize = (panel_size[0] * len(j_range), panel_size[1] * len(i_range))
        sharex = share_axes and not per_panel_x   # si per_panel_x=True -> x NON partagés
        sharey = False #share_axes                       # y reste partagé si share_axes=True

        fig, axes = plt.subplots(len(i_range), len(j_range), figsize=figsize, sharex=sharex, sharey=sharey,)
        #fig, axes = plt.subplots(len(i_range), len(j_range),
        #                         figsize=figsize, sharex=share_axes, sharey=share_axes)
        if not hasattr(axes, "ndim") or axes.ndim == 1:
            axes = np.atleast_2d(axes)
    else:
        axes = None  # pas de plotting

    for ii, i_ in enumerate(i_range):
        for jj, j_ in enumerate(j_range):
            ax = axes[ii, jj] if show_plots else None

            # Données
            sub = df[(df["i"] == i_) & (df["j"] == j_)].sort_values("k")
            if sub.empty:
                if show_plots:
                    ax.set_visible(False)
                continue

            x, label = build_x_axis(sub, freq_mode=freq_mode)
            y = sub["sig"].to_numpy(dtype=float)
            y, med = maybe_normalize(y, NORMALIZE, return_median=True)
            y = maybe_smooth(y, SMOOTH_WIN)
            scale = med if NORMALIZE else 1.0

            # --- Rognage éventuel autour de D (pour le FIT uniquement) ---
            x_used, y_used = x, y
            used_side = None
            if D is not None:
                x_used, y_used, mask_used, used_side = _subset_by_D(
                    x, y, D=float(D), side=crop_side, min_frac=crop_min_frac, margin=crop_margin,
                    keep_if_global_min=keep_if_global_min, prom_sigma=prom_sigma, prom_rel=prom_rel
                )

            # Tracé brut (facultatif)
            # if show_plots:
            #     ax.plot(x, y)
            #     ax.set_title(f"({i_},{j_})", fontsize=8)
            #     if ii == len(i_range) - 1:
            #         ax.set_xlabel(F_LABEL, fontsize=8)
            #     if jj == 0:
            #         ax.set_ylabel("counts (norm)" if NORMALIZE else "counts", fontsize=8)
            #     if grid:
            #         ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.5)
                #if (D is not None) and show_D:
                #    ax.axvline(float(D), linestyle="--", linewidth=0.8, alpha=0.6, color="k")
                        # Tracé brut (facultatif)
            if show_plots:
                ax.plot(x, y)
                ax.set_title(f"({i_},{j_})", fontsize=8)
                #ax.tick_params(axis="y", which="both", left=False, right=False, labelleft=False)


                # Y label seulement sur la première colonne
                if jj == 0:
                    ax.set_ylabel("counts (norm)" if NORMALIZE else "counts", fontsize=8)

                if grid:
                    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.5)

                if per_panel_x:
                    # --- CAS x indépendants : chaque subplot a son propre axe x ---
                    ax.set_xlim(x.min(), x.max())      # ou x_used.min()/max() si tu préfères
                    ax.tick_params(axis="x", labelsize=6, pad=1, labelbottom=True)
                    ax.set_xlabel(label, fontsize=7)
                else:
                    # --- CAS ancien : x partagé, un seul label sur la ligne du bas ---
                    if ii == len(i_range) - 1:
                        ax.set_xlabel(label, fontsize=8)

                if (D is not None) and show_D:
                    ax.axvline(float(D), linestyle="--", linewidth=0.8, alpha=0.6, color="k")



            # --- aucun fit / aucune ligne verticale ---
            #if not mark_two_min:
            #    records.append({"i": int(i_), "j": int(j_), "xv1": np.nan, "xv2": np.nan, "mode": "none"})
            #    continue
            if not mark_two_min:
                records.append({"i": int(i_), "j": int(j_), "xv1": np.nan, "xv2": np.nan,
                                "mode": "none", "subset": used_side})
                continue

            if n_peaks == 1:
                fit1 = fit_one_peak(
                    x_used, y_used,
                    model=fit_model,
                    baseline_to_0=baseline_to_zero
                )

                if fit1.get("success"):
                    a1p, x1p, w1p, b0p, b1p = fit1["popt"]
                    if show_plots:
                        ax.plot(x_used, fit1["yfit"], linestyle="--", linewidth=1, label="fit (1-peak)")
                        ax.axvline(x1p, linestyle=":", linewidth=0.9)

                    records.append({
                        "i": int(i_), "j": int(j_),
                        "xv1": float(x1p), "xv2": float(x1p),
                        "mode": "one_peak", "subset": used_side
                    })
                else:
                    k0 = int(np.argmin(y_used))
                    x0 = float(x_used[k0])
                    if show_plots:
                        ax.axvline(x0, linestyle=":", linewidth=0.9)

                    records.append({
                        "i": int(i_), "j": int(j_),
                        "xv1": float(x0), "xv2": float(x0),
                        "mode": "one_peak", "subset": used_side
                    })

                if show_plots:
                    ax.set_ylim(0.80, 1.09)
                    if mark_two_min and legend != "none":
                        ax.legend(fontsize=legend_fontsize, loc=legend_loc,
                                  frameon=False, handlelength=1.0, handletextpad=0.3, borderpad=0.2)
                continue

            # --- Fit 2-pics + décision ---
            #fit = fit_two_peaks(x, y, model=fit_model, min_sep_frac=min_separation_peak)
            fit = fit_two_peaks(x_used, y_used, model=fit_model, min_sep_frac=min_separation_peak, init=initialisation, maponly=maponly_, baseline_to_0=baseline_to_zero, scale=scale, oldversion=old_init_sym)

            if fit.get("success"):
                a1, x1, w1, a2, x2, w2, b0, b1 = fit["popt"]

                # Pic principal/secondaire par amplitude
                if a1 >= a2:
                    a_big, x_big, w_big = a1, x1, w1
                    a_small, x_small, w_small = a2, x2, w2
                else:
                    a_big, x_big, w_big = a2, x2, w2
                    a_small, x_small, w_small = a1, x1, w1

                curv_big   = a_big   / (w_big + 1e-12)
                curv_small = a_small / (w_small + 1e-12)

                reject_second = (
                    (curv_small < min_curve_amp * curv_big) or
                    (curv_big > maxcurve) or (curv_small > maxcurve) or
                    (a_small < min_peak_amp * a_big)
                )

                if reject_second:
                    # Refit 1 pic avec hints
                    if curv_big > maxcurve:
                        x_hint, w_hint, a_hint = x_small, w_small, a_small
                    else:
                        x_hint, w_hint, a_hint = x_big, w_big, a_big

                    fit1 = fit_one_peak(x_used, y_used, model=fit_model,
                                        x0_hint=x_hint, w_hint=w_hint, a_hint=a_hint,
                                        baseline_to_0=baseline_to_zero)

                    if fit1.get("success"):
                        a1p, x1p, w1p, b0p, b1p = fit1["popt"]
                        if show_plots:
                            #ax.plot(x, fit1["yfit"], linestyle="--", linewidth=1, label="fit (1-peak)")
                            ax.plot(x_used, fit1["yfit"], linestyle="--", linewidth=1, label="fit (1-peak)")
                            ax.axvline(x1p, linestyle=":", linewidth=0.9)
                        records.append({"i": int(i_), "j": int(j_), "xv1": float(x1p), "xv2": float(x1p),
                                        #"mode": "one_peak"})
                                        "mode": "one_peak", "subset": used_side})
                    else:
                        # Fallback : 1 pic basé sur le plus "crédible"
                        if fit_model == "lorentz":
                            if baseline_to_zero:
                                yfit1 = _model_one_lorentz_b0(x_used, a_big, x_big, w_big, b0)
                            else:
                                yfit1 = _model_one_lorentz(x_used, a_big, x_big, w_big, b0, b1)
                            #yfit1 = _model_one_lorentz(x_used, a_big, x_big, w_big, b0, b1, baseline_to_zero=baseline_to_zero)
                        else:
                            if baseline_to_zero:
                                yfit1 = _model_one_gauss_b0(x_used, a_big, x_big, w_big, b0)
                            else:
                                yfit1 = _model_one_gauss(x_used, a_big, x_big, w_big, b0, b1)
                            #yfit1 = _model_one_gauss(x_used, a_big, x_big, w_big, b0, b1, baseline_to_zero=baseline_to_zero)
                        if show_plots:
                            #ax.plot(x, yfit1, linestyle="--", linewidth=1, label="fit (1-peak)")
                            ax.plot(x_used, yfit1, linestyle="--", linewidth=1, label="fit (1-peak)")
                            ax.axvline(x_big, linestyle=":", linewidth=0.9)
                        records.append({"i": int(i_), "j": int(j_), "xv1": float(x_big), "xv2": float(x_big),
                                        #"mode": "one_peak"})
                                        "mode": "one_peak", "subset": used_side})

                    if show_plots:
                        ax.text(0.98, 0.05, "2nd peak rejected",
                                ha="right", va="bottom", fontsize=7, transform=ax.transAxes)

                else:
                    # Deux pics retenus -> fit global + 2 centres
                    if show_plots:
                        #ax.plot(x, fit["yfit"], linestyle="--", linewidth=1, label="fit (2-peaks)")
                        ax.plot(x_used, fit["yfit"], linestyle="--", linewidth=1, label="fit (2-peaks)")

                    x_left, x_right = sorted([x1, x2])
                    if show_plots:
                        ax.axvline(x_left,  linestyle=":", linewidth=0.99)
                        ax.axvline(x_right, linestyle=":", linewidth=0.99)

                        if show_components:
                            if fit_model == "lorentz":
                                if baseline_to_zero:
                                    y1 = _model_one_lorentz_b0(x, a1, x1, w1, b0)
                                    y2 = _model_one_lorentz_b0(x, a2, x2, w2, b0)
                                else:
                                    y1 = _model_one_lorentz(x, a1, x1, w1, b0, b1)
                                    y2 = _model_one_lorentz(x, a2, x2, w2, b0, b1)
                                #y1 = _model_one_lorentz(x, a1, x1, w1, b0, b1, baseline_to_zero=baseline_to_zero)
                                #y2 = _model_one_lorentz(x, a2, x2, w2, b0, b1, baseline_to_zero=baseline_to_zero)
                                lab1 = fr"lorentz $x_1$ = {x1:.3g}"
                                lab2 = fr"lorentz $x_2$ = {x2:.3g}"
                            else:
                                if baseline_to_zero:
                                    y1 = _model_one_gauss_b0(x, a1, x1, w1, b0)
                                    y2 = _model_one_gauss_b0(x, a2, x2, w2, b0)
                                else:
                                    y1 = _model_one_gauss(x, a1, x1, w1, b0, b1)
                                    y2 = _model_one_gauss(x, a2, x2, w2, b0, b1)
                                #y1 = _model_one_gauss(x, a1, x1, w1, b0, b1, baseline_to_zero=baseline_to_zero)
                                #y2 = _model_one_gauss(x, a2, x2, w2, b0, b1, baseline_to_zero=baseline_to_zero,)
                                lab1 = fr"gauss $x_1$ = {x1:.3g}"
                                lab2 = fr"gauss $x_2$ = {x2:.3g}"
                            ax.plot(x, y1, linestyle="-.", linewidth=1, alpha=0.9, label=lab1)
                            ax.plot(x, y2, linestyle="-.", linewidth=1, alpha=0.9, label=lab2)

                        if fit.get("note"):
                            ax.text(0.98, 0.05, f"fusion ? (<{int(100*min_separation_peak)}%)",
                                    ha="right", va="bottom", fontsize=7, transform=ax.transAxes)

                    records.append({"i": int(i_), "j": int(j_), "xv1": float(x_left), "xv2": float(x_right),
                                    #"mode": "two_peaks"})
                                    "mode": "two_peaks", "subset": used_side})

            else:
                # fit échoué : on marque les minima gauche/droite
                mid = len(x) // 2
                kL = np.argmin(y[:mid]) if mid > 1 else 0
                kR = (np.argmin(y[mid:]) + mid) if (len(x) - mid) > 1 else len(x) - 1
                xL, xR = float(x[kL]), float(x[kR])
                if show_plots:
                    ax.axvline(xL, linestyle=":", linewidth=0.9)
                    ax.axvline(xR, linestyle=":", linewidth=0.9)
                records.append({"i": int(i_), "j": int(j_), 
                                #"xv1": xL, "xv2": xR,
                                "xv1": np.nan, "xv2": np.nan,
                                #"mode": "minima"})
                                "subset": used_side})

            if show_plots:
                # Légendes compactes
                ax.set_ylim(0.80, 1.09)                 #LIMITE AXE Y
                if mark_two_min and legend != "none":
                    ax.legend(fontsize=legend_fontsize, loc=legend_loc,
                              frameon=False, handlelength=1.0, handletextpad=0.3, borderpad=0.2)

    if show_plots:
        fig.suptitle(
            f"ODMR sweeps, ROI grid around pixel ({center_i},{center_j}) | fit: {fit_model}", # | "
            #f"thr={int(100*min_peak_amp)}%",
            fontsize=12
        )
        # NEW : resserrer légèrement la grille
        fig.subplots_adjust(hspace=0.15, wspace=0.10)
        try:
            plt.tight_layout(rect=[0, 0, 1, 0.98])
        except Exception:
            plt.tight_layout()

        if pdf_path is not None:
            meta = dict() if pdf_metadata is None else dict(pdf_metadata)
            meta.setdefault("Title", f"ODMR grid ({center_i},{center_j}) half={half}")
            with PdfPages(pdf_path, metadata=meta) as pdf:
                pdf.savefig(fig, bbox_inches="tight")

        plt.show()

    # Retourne les abscisses des lignes verticales (par panneau)
    #return pd.DataFrame.from_records(records, columns=["i", "j", "xv1", "xv2", "mode", "subset"]).sort_values(["i", "j"]).reset_index(drop=True)

    # Retourne les abscisses des lignes verticales (par panneau)
    df_out = pd.DataFrame.from_records(records, columns=["i", "j", "xv1", "xv2", "mode", "subset"])

    # --- Réorganisation/mirroir des fréquences si freq_à_gauche=True ---
    if freq_à_gauche and (D is not None):
        Df = float(D)

        # lignes pour lesquelles on a bien deux abscisses définies
        mask_valid = df_out[["xv1", "xv2"]].notna().all(axis=1)

        # cas où les deux minima sont du côté droit : D <= xv1 <= xv2
        mask_right = mask_valid & (df_out[["xv1", "xv2"]].min(axis=1) >= Df)

        # on effectue le mirroir par rapport à D :
        # D <= xL <= xR  -->  xL' = 2D - xR  (le plus à gauche après mirroir)
        #                    xR' = 2D - xL  (le plus proche de D)
        x1 = df_out.loc[mask_right, "xv1"].to_numpy()
        x2 = df_out.loc[mask_right, "xv2"].to_numpy()

        df_out.loc[mask_right, "xv1"] = 2 * Df - x2   # xL'  (tout à gauche)
        df_out.loc[mask_right, "xv2"] = 2 * Df - x1   # xR'  (plus à droite, mais ≤ D)

    return df_out.sort_values(["i", "j"]).reset_index(drop=True)






#***** PLOT INIT GUESSES ********************
# PLOT INIT GUESSES 

def plot_init_guesses(
    df: pd.DataFrame,
    indice_i: int,
    indice_j: int,
    FREQ_MODE: str,
    NORMALIZE: bool,
    SMOOTH_WIN: int,
    T: float = 1.0,
    figsize=(15, 4),        # un peu plus large pour 3 subplots
    infos=True,
    D=None,
    crop_side="auto", 
    crop_min_frac=0.5, 
    crop_margin=0.0, 
    keep_if_global_min=True, 
    prom_sigma=3.0, 
    prom_rel=0.05,
    b1_to_zero=False,
    oldversion=True
) -> tuple:
    """
    Trace le fit 2-Lorentziens (init classique + init symétrique + MAP) 
    pour un pixel (i,j) donné.
    """
    # ----- extraction du pixel -----
    sub = df[(df["i"] == indice_i) & (df["j"] == indice_j)].copy()
    if sub.empty:
        raise ValueError(f"(i,j)=({indice_i},{indice_j}) demandé inexistant dans le fichier.")
    sub = sub.sort_values("k")

    # ----- axe x et pré-traitement y -----
    x, label = build_x_axis(sub, FREQ_MODE)

    y = sub["sig"].to_numpy(dtype=float)
    y, med = maybe_normalize(y, NORMALIZE, return_median=True)
    y = maybe_smooth(y, SMOOTH_WIN)

    scale = med if NORMALIZE else 1.0

    # --- Rognage éventuel autour de D (pour le FIT uniquement) ---
    x_used, y_used = x, y
    used_side = None
    if D is not None:
        x_used, y_used, mask_used, used_side = _subset_by_D(
            x, y, D=float(D), side=crop_side, min_frac=crop_min_frac, margin=crop_margin,
            keep_if_global_min=keep_if_global_min, prom_sigma=prom_sigma, prom_rel=prom_rel
        )
        # debug éventuel :
        # print(f"D={D}, used_side={used_side}, len(x)={len(x)}, len(x_used)={len(x_used)}")

    # ----- init classique et sym -----
    p0 = _init_guesses(x_used, y_used, b1_to_0=b1_to_zero)
    p0_sym = _init_guesses_symmetric(x_used, y_used, b1_to_0=b1_to_zero, oldversion=oldversion)
    if b1_to_zero:
        y0     = _model_two_lorentz_b0(x_used, *p0)
        y0_sym = _model_two_lorentz_b0(x_used, *p0_sym)
    else:
        y0     = _model_two_lorentz(x_used, *p0)
        y0_sym = _model_two_lorentz(x_used, *p0_sym)
    #y0 = _model_two_lorentz(x_used, *p0)

    # ----- MAP (2 Lorentziens) -----
    theta_map, I_map, result, prior_means, prior_sigmas = fit_MAP_two_lorentz(
        x_used, y_used, T=T, n_g2=2, n_seps=3, scale=scale
    )
    
    if infos:
        names = ["B", "A1", "A2", " nu1", "nu2", "g1", "g2"]

        print("Succès optimisation :", result.success, "|", result.message)
        print("\nParamètres (MAP vs prior):")
        for n, m, pm, ps in zip(names, theta_map, prior_means, prior_sigmas):
            print(f"{n:>4}: MAP={m: .6g}, prior_mu={pm: .6g}, prior_sigma={ps: .6g}")

    # ----- figure avec 3 sous-plots -----
    fig, axs = plt.subplots(1, 3, figsize=figsize, sharey=True)

    # subplot init classique
    axs[0].plot(x, y, "-", alpha=0.25)
    axs[0].scatter(x, y, s=8, alpha=0.7, label="données")
    axs[0].plot(x_used, y0, label="init classic (2 Lorentz)")
    axs[0].set_xlabel(label)
    axs[0].set_ylabel("counts (median normalized)" if NORMALIZE else "counts")
    axs[0].set_title(f"Init classic  (i={indice_i}, j={indice_j})")
    axs[0].grid(True)
    axs[0].legend()

    # subplot init symétrique
    axs[1].plot(x, y, "-",alpha=0.25)
    axs[1].scatter(x, y, s=8, alpha=0.7, label="données")
    axs[1].plot(x_used, y0_sym, label="init symmetric (2 Lorentz)")
    axs[1].set_xlabel(label)
    axs[1].set_title(f"Init symmetric  (i={indice_i}, j={indice_j})")
    axs[1].grid(True)
    axs[1].legend()

    # subplot MAP
    axs[2].plot(x, y, "-",alpha=0.25)
    axs[2].scatter(x, y, s=8, alpha=0.7, label="données")
    axs[2].plot(x_used, I_map, "-", linewidth=2, label="fit MAP")
    axs[2].set_xlabel(label)
    axs[2].set_title(f"MAP  (i={indice_i}, j={indice_j})")
    axs[2].grid(True)
    axs[2].legend()

    plt.tight_layout()
    plt.show()


    # return theta_map, I_map, result, prior_means, prior_sigmas, p0, p0_sym

    #return theta_map, I_map, result, prior_means, prior_sigmas, p0












# helper 
def _contrast_map(df, NORMALIZE, SMOOTH_WIN, roi, use_processed=True):
        # --- Ranges issus du ROI ---
        slice_i, slice_j = roi
        i0, i1 = slice_i.start, slice_i.stop - 1
        j0, j1 = slice_j.start, slice_j.stop - 1
        i_range = np.arange(i0, i1 + 1)
        j_range = np.arange(j0, j1 + 1)

        # --- Sous-DF ROI ---
        mask_roi = (df["i"].between(i0, i1)) & (df["j"].between(j0, j1))
        df_roi = df.loc[mask_roi, ["i","j","k","sig"]].copy().sort_values(["i","j","k"])

        # --- fonction pour calculer le contrast ---      
        def contrast(sig_series):
            y = sig_series.to_numpy(dtype=float)
            if use_processed:
                y = maybe_normalize(y, NORMALIZE)
                y = maybe_smooth(y, SMOOTH_WIN)
            return float(np.nanmax(y) - np.nanmin(y))

        contrast_df = (df_roi.groupby(["i","j"])["sig"].agg(contrast)
                .unstack("j")
                .reindex(index=i_range, columns=j_range)
        )
        
        # xk = df_roi.groupby("k")["frequency"].median().to_numpy()
        # def agg_min_x(s_counts):
        #     y = s_counts.to_numpy(dtype=float)
        #     if use_processed:
        #         y = maybe_normalize(y, NORMALIZE)
        #         y = maybe_smooth(y, SMOOTH_WIN)
        #     k0 = int(np.nanargmin(y))
        #     return xk[k0]

        # min_x = (df_roi.groupby(["i", "j"])["sig"].agg(agg_min_x)
        #         .unstack("j").reindex(index=i_range, columns=j_range))
        return contrast_df #, min_x


def overlay_percentile_band_grid(g2d, ax, pct=25.0, alpha=0.3, *, side="top", extent=None, color=(1.0, 0.0, 0.0), mirror_x=False):
    arr = np.asarray(g2d, float)
    vals = arr[np.isfinite(arr)]
    if vals.size == 0:
        return

    thr  = np.percentile(vals, 100 - pct) if side == "top" else np.percentile(vals, pct)
    mask = (arr >= thr) if side == "top" else (arr <= thr)

    if mirror_x:                     
        mask = mask[:, ::-1] 

    overlay = np.zeros(mask.shape + (4,), float)
    overlay[..., 0] = color[0]
    overlay[..., 1] = color[1]
    overlay[..., 2] = color[2]
    overlay[..., 3] = mask.astype(float) * alpha

    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    ax.imshow(overlay, origin="lower", interpolation="nearest", extent=extent, aspect="auto")
    ax.set_xlim(xlim); ax.set_ylim(ylim)

def _robust_vmin_vmax(A, pct=2.5):
    v = np.asarray(A, float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return None, None
    return np.percentile(v, [pct, 100 - pct])


def format_length_um(length_um):
    # length_um est en micromètres
    if length_um < 1.0:
        return length_um * 1000.0, "nm"   # 1 µm = 1000 nm
    else:
        return length_um, "µm"
    
# big function 
def plot_big_function(
    date: str,
    data_directory: Path,
    *,
    # AFM preprocess
    preprocess_default="parabolic X",
    preprocess_by_date=None,

    # ROI / fits
    center_i=35, center_j=35,
    roi_box=None,
    half_fit=100,
    init="symmetric",
    fit_model="lorentz",
    FREQ_MODE="frequency",
    D=2.87e9,
    NORMALIZE=True,
    SMOOTH_WIN=7,

    # limites colorbar:
    clim_mT=None,          # (vmin, vmax) communs en mT pour les 3 cartes
    clim1_mT=None,         # (vmin, vmax) pour |B_par,1|
    clim2_mT=None,         # (vmin, vmax) pour |B_par,2|
    clim3_mT=None,         # (vmin, vmax) pour ΔB_par

    # overlay rouge via contrast calculé
    overlay_top_pct=25.0,
    overlay_alpha=0.30,

    # Reconstruction/diff figs
    dark_bg=True,
    step_quiver=2,
    s=1,

    # cas “bande”
    band_date="2025-11-16-23-41-29",
    band_source="2025-11-17-18-36-10",

    # Output sauvegarde plots
    png1=None,
    png2=None,
    png3=None,
    png4=None,
    png5=None,
    png6=None,
    png7=None,

    scalebar_frac=0.25,
    scalebar_xy=(0.05, 0.05),
    quiver_plots=True,
):
    preprocess_by_date = preprocess_by_date or {}
    preprocess = preprocess_by_date.get(date, preprocess_default)

    base_folder = data_directory / f"{date}-odmr_hardware"
    scalar_file = base_folder / "scalarData.txt"
    meta_path   = base_folder / "imageMeta.json"
    odmr_path   = base_folder / "seq0" / "eval" / "odmr.txt"

    # ---------- load scalar (AFM) ----------
    df2 = load_df2(scalar_file)

    # ---------- load ODMR + meta ----------
    with open(meta_path, "r") as f:
        meta = json.load(f)
    z_meas_physical = meta["scanHeightControl"]["scanDistance"]
    B_vector_magnet = meta["vectorMagnetSettings"]["field_strength"]
    Size            = meta["rect"]["size"][0]
    Size_um         = Size * 1e6
    microwavepower  = meta["microwavePower"]["Main Source"]
    theta_deg       = meta["vectorMagnetSettings"]["theta"] #125.5
    phi_deg         = meta["vectorMagnetSettings"]["phi"] #90.0 
    
    # axe NV de la pointe de scan
    theta = np.deg2rad(theta_deg)
    phi   = np.deg2rad(phi_deg)
    nv_axis = np.array([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)])

    df = load_df1(odmr_path, DOWNCAST_INT=True)

    # ---------- ajout bande si besoin ----------
    if date == band_date:
        band_folder2 = data_directory / f"{band_source}-odmr_hardware"

        df2_band = load_df2(band_folder2 / "scalarData.txt")
        df2 = concatener_data_df2(df2, df2_band)

        band_odmr = band_folder2 / "seq0" / "eval" / "odmr.txt"
        df_band = load_df1(band_odmr, DOWNCAST_INT=True)
        df = concatener_data_df1(df, df_band, infos=False)

    # dims ODMR (après rotation faite dans load_df1)
    n_i = int(df["i"].max() + 1)
    n_j = int(df["j"].max() + 1)

    # ---------- FIG 1 -----------
    plot_contrast_argmin(df, NORMALIZE, SMOOTH_WIN,
            center_i, center_j, half=half_fit, use_processed=True, grid=False, panel_size=(5.2,4.5), #5.5
            roi_box=roi_box,
            png_path=png1,
            suptitle=f"ROI maps (distance={z_meas_physical*10**9:.0f} nm, B_ext={B_vector_magnet:.1f} mT, scan length={Size*1e6:.0f} $\mu$m, MW power={microwavepower:.0f} dBm), theta={theta_deg:.0f}, phi={phi_deg:.0f}",
            Size_um=Size_um,
            )

    # ---------- fits ROI ----------
    f = plot_fits_on_spectrograms(
        df, NORMALIZE, SMOOTH_WIN,
        center_i, center_j, half=half_fit, share_axes=True,
        per_panel_x=True,
        mark_two_min=True,
        fit_model=fit_model,
        initialisation=init,
        maponly_=False,
        freq_mode=FREQ_MODE,
        min_curve_amp=0.1, min_peak_amp=0.2, maxcurve=1, min_separation_peak=0.02,
        show_components=True,
        legend="none",
        D=D,
        show_plots=False,
        keep_if_global_min=False,
        freq_à_gauche=True,
        baseline_to_zero=True,
    )

    i0, i1 = int(f["i"].min()), int(f["i"].max())
    j0, j1 = int(f["j"].min()), int(f["j"].max())
    roi = (slice(i0, i1 + 1), slice(j0, j1 + 1))
    extent_roi = [j0, j1 + 1, i0, i1 + 1]

    # ---------- B maps ----------
    _, B1_projectionNV, B2_projectionNV, DeltaB_projectionNV = compute_B_components(f, n_i, n_j, clip_negative=False)

    A1 = np.ma.masked_invalid(B1_projectionNV * 1e3)
    A2 = np.ma.masked_invalid(B2_projectionNV * 1e3)
    A3 = np.ma.masked_invalid(DeltaB_projectionNV * 1e3)
    def _vv(local):
        if local is not None:
            return local
        return clim_mT if clim_mT is not None else (None, None)
    v1min, v1max = _vv(clim1_mT)
    v2min, v2max = _vv(clim2_mT)
    v3min, v3max = _vv(clim3_mT)

    # --- robust clim pour B1/B2 (si pas fourni explicitement) ---
    if clim_mT is None and clim1_mT is None and clim2_mT is None:
        v1min, v1max = _robust_vmin_vmax(A1, pct=2.5)
        v2min, v2max = v1min, v1max   # même échelle pour comparer


    # ---------- contrast calculé (depuis df1) et agmin ----------
    contrast_df = _contrast_map(df, NORMALIZE, SMOOTH_WIN, roi, use_processed=True) #, argmin_df


    # ---------- FIG 2x3 ----------
    xmin, xmax, ymin, ymax = extent_roi

    fig, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)
    fig.suptitle(
        f"{date}   (distance={z_meas_physical*1e9:.0f} nm, B_ext={B_vector_magnet:.1f} mT, "
        f"scan length={Size*1e6:.0f} $\\mu$m, MW power={microwavepower:.0f} dBm)",
        fontsize=12
    )

    # Row 1: AFM phase / AFM height / contrast
    ax = axes[0, 0]
    im = plot_on_ax(df2, "afm:phase", ax, robust_pct=5.5)
    ax.set_title("AFM phase")
    if im is not None:
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="phase [deg]")

    overlay_percentile_band_grid(
        contrast_df.to_numpy(), ax,
        pct=overlay_top_pct, alpha=overlay_alpha,
        side="top", extent=extent_roi, mirror_x=True
    )
    # IMPORTANT: remettre l'aspect APRES l'overlay (sinon overlay peut le réécrire)
    ax.set_aspect("equal", adjustable="box")


    ax = axes[0, 1]
    im = plot_on_ax(df2, "afm:height", ax, robust_pct=5.5, preprocess=preprocess, cmap="copper")
    ax.set_title(f"AFM height ({preprocess})")
    if im is not None:
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="height [m]")

    overlay_percentile_band_grid(
        contrast_df.to_numpy(), ax,
        pct=overlay_top_pct, alpha=overlay_alpha,
        side="top", extent=extent_roi, mirror_x=True
    )
    ax.set_aspect("equal", adjustable="box")


    ax = axes[0, 2]
    im0 = ax.imshow(
        contrast_df.to_numpy(),
        origin="lower", interpolation="nearest",
        extent=extent_roi, cmap="viridis"
    )
    ax.set_title("Contraste (max - min)")
    ax.set_xlabel("j"); ax.set_ylabel("i")
    ax.set_xlim(xmin, xmax); ax.set_ylim(ymax, ymin)
    fig.colorbar(im0, ax=ax, label="intensity of contrast ($\\propto$ counts)")
    if roi_box is not None:
        _add_roi_box(ax, roi_box)
    ax.set_aspect("equal", adjustable="box")


    # Row 2: B1_nv / B2_nv / Delta
    ax = axes[1, 0]
    im1 = ax.imshow(
        A1, origin="lower", interpolation="nearest",
        extent=extent_roi, cmap="plasma", vmin=v1min, vmax=v1max
    )
    ax.set_title(r"$B_{\parallel,1}$ (mT)  = $(\nu_{\rm high} - D)/\gamma_{\rm NV}$")
    ax.set_xlabel("j"); ax.set_ylabel("i")
    fig.colorbar(im1, ax=ax, label="mT")
    if roi_box is not None:
        _add_roi_box(ax, roi_box)

    overlay_percentile_band_grid(
        contrast_df.to_numpy(), ax,
        pct=overlay_top_pct, alpha=overlay_alpha,
        side="top", extent=extent_roi
    )
    # IMPORTANT: remettre limites + aspect APRES l'overlay
    ax.set_xlim(xmin, xmax); ax.set_ylim(ymax, ymin)
    ax.set_aspect("equal", adjustable="box")


    ax = axes[1, 1]
    im2 = ax.imshow(
        A2, origin="lower", interpolation="nearest",
        extent=extent_roi, cmap="plasma", vmin=v2min, vmax=v2max
    )
    ax.set_title(r"$B_{\parallel,2}$ (mT)  = $(f_{\rm low} - D)/\gamma_{\rm NV}$")
    ax.set_xlabel("j"); ax.set_ylabel("i")
    im2.set_norm(im1.norm)
    fig.colorbar(im2, ax=ax, label="mT")
    if roi_box is not None:
        _add_roi_box(ax, roi_box)
    ax.set_xlim(xmin, xmax); ax.set_ylim(ymax, ymin)
    ax.set_aspect("equal", adjustable="box")


    ax = axes[1, 2]
    im3 = ax.imshow(
        A3, origin="lower", interpolation="nearest",
        extent=extent_roi, cmap="turbo", vmin=v3min, vmax=v3max
    )
    ax.set_title(r"$\Delta B_{\parallel}$ (mT) $= B_{\parallel,1}-B_{\parallel,2} = \Delta f / \gamma_{\rm NV}$")
    ax.set_xlabel("j"); ax.set_ylabel("i")
    fig.colorbar(im3, ax=ax, label="mT")
    if roi_box is not None:
        _add_roi_box(ax, roi_box)
    ax.set_xlim(xmin, xmax); ax.set_ylim(ymax, ymin)
    ax.set_aspect("equal", adjustable="box")


    # ---- BARRE D'ÉCHELLE (noir + pointillé blanc) ----
    if (Size_um is not None):
        bar_um  = scalebar_frac * Size_um
        x0, y0  = scalebar_xy
        x1, y1  = x0 + scalebar_frac, y0
        
        for ax in axes.ravel():
            ann1 = ax.annotate(
                "", xy=(x1, y1), xytext=(x0, y0),
                xycoords="axes fraction",
                arrowprops=dict(arrowstyle="-", lw=4, color="k"),
                annotation_clip=False
            )
            ann1.set_zorder(10000)

            ann2 = ax.annotate(
                "", xy=(x1, y1), xytext=(x0, y0),
                xycoords="axes fraction",
                arrowprops=dict(arrowstyle="-", lw=2, color="w", linestyle=(0, (4, 4))),
                annotation_clip=False
            )
            ann2.set_zorder(10001)

            val, unit = format_length_um(bar_um)
            txt = ax.text(
                x0 + scalebar_frac/2, y0 + 0.03, f"{val:.0f} {unit}",
                transform=ax.transAxes, color="w",
                ha="center", va="bottom", fontsize=9,
                bbox=dict(facecolor="black", alpha=0.5, edgecolor="none", pad=1.5)
            )
            txt.set_zorder(10002)
            txt.set_clip_on(False)

    if png2 is not None:
        fig.savefig(png2, dpi=300, bbox_inches="tight")
    plt.show()


    # ---------- FIG 1x4 (B1_parr / contrast / AFM phase / AFM height) ----------
    fig, axes = plt.subplots(1, 4, figsize=(22, 5.5), constrained_layout=True)
    fig.suptitle(
        f"{date} Contrast on top in translucent color  (distance={z_meas_physical*10**9:.0f} nm, B_ext={B_vector_magnet:.1f} mT, scan length={Size*1e6:.0f} $\mu$m, MW power={microwavepower:.0f} dBm)",
        fontsize=12
    )

    xmin, xmax, ymin, ymax = extent_roi  # (x0, x1, y0, y1)

    # 1) B1_parr
    ax = axes[0]
    im1 = ax.imshow(
        A1, origin="lower", interpolation="nearest",
        extent=extent_roi, cmap="plasma", vmin=v1min, vmax=v1max
    )
    ax.set_title(r"$B_{\parallel,1}$ (mT) $= (\nu_{\rm high}-D)/\gamma_{\rm NV}$")
    ax.set_xlabel("j"); ax.set_ylabel("i")
    ax.set_xlim(xmin, xmax); ax.set_ylim(ymax, ymin)          # y inversé, sans déformation
    ax.set_box_aspect(A1.shape[0] / A1.shape[1])              # pixels carrés (même si image rectangulaire)
    fig.colorbar(im1, ax=ax, label="mT", fraction=0.046, pad=0.04)
    if roi_box is not None:
        _add_roi_box(ax, roi_box)
    overlay_percentile_band_grid(
        contrast_df.to_numpy(), ax,
        pct=overlay_top_pct, alpha=overlay_alpha,
        side="top", extent=extent_roi
    )

    # 2) Contraste
    ax = axes[1]
    C = contrast_df.to_numpy()
    p_lo, p_hi = 2, 98
    vmin, vmax = np.nanpercentile(C, [p_lo, p_hi])
    imC = ax.imshow(
        C, origin="lower", interpolation="nearest",
        extent=extent_roi, cmap="viridis",
        vmin=vmin, vmax=vmax
    )
    ax.set_title(f"Contraste (max - min)") #  (clip {p_lo}-{p_hi}%)")
    ax.set_xlabel("j"); ax.set_ylabel("i")
    ax.set_xlim(xmin, xmax); ax.set_ylim(ymax, ymin)
    ax.set_box_aspect(C.shape[0] / C.shape[1])
    fig.colorbar(imC, ax=ax, label="intensity of contrast ($\propto$ counts)", fraction=0.046, pad=0.04)
    if roi_box is not None:
        _add_roi_box(ax, roi_box)


    # 3) AFM phase
    ax = axes[2]
    imP = plot_on_ax(df2, "afm:phase", ax, robust_pct=5.5)
    ax.set_title("AFM phase")
    ax.set_xlabel("j"); ax.set_ylabel("i")
    ax.set_xlim(xmin, xmax); ax.set_ylim(ymax, ymin)
    ax.invert_xaxis()
    if imP is not None:
        imP.set_extent(extent_roi)          # force la même géométrie/repère
        imP.set_interpolation("nearest")
        arrP = imP.get_array()
        ax.set_box_aspect(arrP.shape[0] / arrP.shape[1])
        fig.colorbar(imP, ax=ax, fraction=0.046, pad=0.04, label="phase [deg]")
    overlay_percentile_band_grid(
        C, ax,
        pct=overlay_top_pct, alpha=overlay_alpha,
        side="top", extent=extent_roi, mirror_x=True
    )

    # 4) AFM height
    ax = axes[3]
    imH = plot_on_ax(df2, "afm:height", ax, robust_pct=5.5, preprocess=preprocess, cmap="hot")
    ax.set_title(f"AFM height ({preprocess})")
    ax.set_xlabel("j"); ax.set_ylabel("i")
    ax.set_xlim(xmin, xmax); ax.set_ylim(ymax, ymin)
    ax.invert_xaxis()
    if imH is not None:
        imH.set_extent(extent_roi)
        imH.set_interpolation("nearest")
        arrH = imH.get_array()
        ax.set_box_aspect(arrH.shape[0] / arrH.shape[1])
        fig.colorbar(imH, ax=ax, fraction=0.046, pad=0.04, label="height [m]")
    overlay_percentile_band_grid(
        C, ax,
        pct=overlay_top_pct, alpha=overlay_alpha,
        side="top", extent=extent_roi, mirror_x=True,
        color=(0.0, 0.0, 1.0)
    )

    # ---- BARRE D'ÉCHELLE (noir + pointillé blanc) ----
    if (Size_um is not None):
        bar_um  = scalebar_frac * Size_um
        x0, y0  = scalebar_xy
        x1, y1  = x0 + scalebar_frac, y0
        
        for ax in axes.ravel():
            ann1 = ax.annotate(
                "", xy=(x1, y1), xytext=(x0, y0),
                xycoords="axes fraction",
                arrowprops=dict(arrowstyle="-", lw=4, color="k"),
                annotation_clip=False
            )
            ann1.set_zorder(10000)

            ann2 = ax.annotate(
                "", xy=(x1, y1), xytext=(x0, y0),
                xycoords="axes fraction",
                arrowprops=dict(arrowstyle="-", lw=2, color="w", linestyle=(0, (4, 4))),
                annotation_clip=False
            )
            ann2.set_zorder(10001)

            val, unit = format_length_um(bar_um)
            txt = ax.text(
                x0 + scalebar_frac/2, y0 + 0.03, f"{val:.0f} {unit}",
                transform=ax.transAxes, color="w",
                ha="center", va="bottom", fontsize=9,
                bbox=dict(facecolor="black", alpha=0.5, edgecolor="none", pad=1.5)
            )
            txt.set_zorder(10002)
            txt.set_clip_on(False)


    if png3 is not None:
        fig.savefig(png3, dpi=300, bbox_inches="tight")

    plt.show()




    # ---------- FIG 1x4 (B1_parr / contrast / AFM phase / AFM height) ----------
    fig, axes = plt.subplots(1, 4, figsize=(22, 5.5), constrained_layout=True)
    fig.suptitle(
        f"{date}   (distance={z_meas_physical*10**9:.0f} nm, B_ext={B_vector_magnet:.1f} mT, scan length={Size*1e6:.0f} $\mu$m, MW power={microwavepower:.0f} dBm)",
        fontsize=12
    )

    xmin, xmax, ymin, ymax = extent_roi  # (x0, x1, y0, y1)

    # 1) B1_parr
    ax = axes[0]
    im1 = ax.imshow(
        A1, origin="lower", interpolation="nearest",
        extent=extent_roi, cmap="plasma", vmin=v1min, vmax=v1max
    )
    ax.set_title(r"$B_{\parallel,1}$ (mT) $= (\nu_{\rm high}-D)/\gamma_{\rm NV}$")
    ax.set_xlabel("j"); ax.set_ylabel("i")
    ax.set_xlim(xmin, xmax); ax.set_ylim(ymax, ymin)          # y inversé, sans déformation
    ax.set_box_aspect(A1.shape[0] / A1.shape[1])              # pixels carrés (même si image rectangulaire)
    fig.colorbar(im1, ax=ax, label="mT", fraction=0.046, pad=0.04)
    if roi_box is not None:
        _add_roi_box(ax, roi_box)


    # 2) Contraste
    ax = axes[1]
    C = contrast_df.to_numpy()
    p_lo, p_hi = 2, 98
    vmin, vmax = np.nanpercentile(C, [p_lo, p_hi])
    imC = ax.imshow(
        C, origin="lower", interpolation="nearest",
        extent=extent_roi, cmap="viridis",
        vmin=vmin, vmax=vmax
    )
    ax.set_title(f"Contraste (max - min)") #  (clip {p_lo}-{p_hi}%)")
    ax.set_xlabel("j"); ax.set_ylabel("i")
    ax.set_xlim(xmin, xmax); ax.set_ylim(ymax, ymin)
    ax.set_box_aspect(C.shape[0] / C.shape[1])
    fig.colorbar(imC, ax=ax, label="intensity of contrast ($\propto$ counts)", fraction=0.046, pad=0.04)
    if roi_box is not None:
        _add_roi_box(ax, roi_box)


    # 3) AFM phase
    ax = axes[2]
    imP = plot_on_ax(df2, "afm:phase", ax, robust_pct=5.5)
    ax.set_title("AFM phase")
    ax.set_xlabel("j"); ax.set_ylabel("i")
    ax.set_xlim(xmin, xmax); ax.set_ylim(ymax, ymin)
    ax.invert_xaxis()
    if imP is not None:
        imP.set_extent(extent_roi)          # force la même géométrie/repère
        imP.set_interpolation("nearest")
        arrP = imP.get_array()
        ax.set_box_aspect(arrP.shape[0] / arrP.shape[1])
        fig.colorbar(imP, ax=ax, fraction=0.046, pad=0.04, label="phase [deg]")


    # 4) AFM height
    ax = axes[3]
    imH = plot_on_ax(df2, "afm:height", ax, robust_pct=5.5, preprocess=preprocess, cmap="hot")
    ax.set_title(f"AFM height ({preprocess})")
    ax.set_xlabel("j"); ax.set_ylabel("i")
    ax.set_xlim(xmin, xmax); ax.set_ylim(ymax, ymin)
    ax.invert_xaxis()
    if imH is not None:
        imH.set_extent(extent_roi)
        imH.set_interpolation("nearest")
        arrH = imH.get_array()
        ax.set_box_aspect(arrH.shape[0] / arrH.shape[1])
        fig.colorbar(imH, ax=ax, fraction=0.046, pad=0.04, label="height [m]")

    # ---- BARRE D'ÉCHELLE (noir + pointillé blanc) ----
    if (Size_um is not None):
        bar_um  = scalebar_frac * Size_um
        x0, y0  = scalebar_xy
        x1, y1  = x0 + scalebar_frac, y0
        
        for ax in axes.ravel():
            ann1 = ax.annotate(
                "", xy=(x1, y1), xytext=(x0, y0),
                xycoords="axes fraction",
                arrowprops=dict(arrowstyle="-", lw=4, color="k"),
                annotation_clip=False
            )
            ann1.set_zorder(10000)

            ann2 = ax.annotate(
                "", xy=(x1, y1), xytext=(x0, y0),
                xycoords="axes fraction",
                arrowprops=dict(arrowstyle="-", lw=2, color="w", linestyle=(0, (4, 4))),
                annotation_clip=False
            )
            ann2.set_zorder(10001)

            val, unit = format_length_um(bar_um)
            txt = ax.text(
                x0 + scalebar_frac/2, y0 + 0.03, f"{val:.0f} {unit}",
                transform=ax.transAxes, color="w",
                ha="center", va="bottom", fontsize=9,
                bbox=dict(facecolor="black", alpha=0.5, edgecolor="none", pad=1.5)
            )
            txt.set_zorder(10002)
            txt.set_clip_on(False)

    if png4 is not None:
        fig.savefig(png4, dpi=300, bbox_inches="tight")

    plt.show()


    plot_B1_B2_DeltaB_projectionNV(
        B1_projectionNV[roi],
        B2_projectionNV[roi],
        DeltaB_projectionNV[roi],
        extent=extent_roi,
        clim3_mT=clim3_mT,
        png_path=png5,
        suptitle=f"Projection B on NV axis from the ODMR spectra, {date}  (distance={z_meas_physical*10**9:.0f} nm, B_ext={B_vector_magnet:.1f} mT, scan length={Size*1e6:.0f} $\mu$m, MW power={microwavepower:.0f} dBm)",
        scalebar=True,
        Size_um=Size_um,
        # roi_box=roi_box
    )

    if quiver_plots:
        # --- SECONDE FIGURE ---
        (Bx1, By1, Bz1), (Bx2, By2, Bz2), (BxD, ByD, BzD) = reconstruction_B(
            B1_T=(B1_projectionNV[roi]) + B_vector_magnet*1e-3, #met en T
            B2_T=(B2_projectionNV[roi]) + B_vector_magnet*1e-3,
            nv_axis=np.asarray(nv_axis, float),
            dx=1,
            d_meas=z_meas_physical,
            extent=extent_roi,
            step=step_quiver,
            s=s,
            with_norm_2D=True,
            color_mode="Bz",
            dark_bg=dark_bg,
            cmap_name="jet",
            diff_mode="Bpar",
            clim3_mT=clim3_mT,
            png_path=png6,
            Size_um=Size_um,
            scalebar=True,
        )

        plot_B_difference(
            Bx1, By1, Bz1,
            Bx2, By2, Bz2,
            extent=extent_roi,
            png_path=png7,
        )

    return fig, dict(
        roi=(i0, i1, j0, j1),
        extent_roi=extent_roi,
        z_meas=z_meas_physical,
        B1_projectionNV=B1_projectionNV,
        B2_projectionNV=B2_projectionNV,
        DeltaB_projectionNV=DeltaB_projectionNV,
    )
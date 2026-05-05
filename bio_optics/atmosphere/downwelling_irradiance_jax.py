"""
JAX implementation of downwelling spectral irradiance (Gege 2012 / WASI).

Two-layer architecture:
  - Layer 1: precompute() — numpy/scipy, runs once outside JIT.
    Loads spectral data files (E0_sun, a_ozone, a_oxygen, a_wv) resampled to sensor
    wavelengths.  Optionally pre-computes Ed_d / Ed_sr / Ed_sa as static arrays when
    atmospheric parameters are fixed per scene.
  - Layer 2: _compute_Ed_d / _compute_Ed_sr / _compute_Ed_sa — pure JAX arithmetic,
    JIT-compilable.  These are called from _forward_core of coupled models.

Dual-mode design
----------------
Mode A — fixed atmosphere (zero per-call overhead):
    Pass theta_sun and all atmospheric scalars to precompute().
    The returned dict contains 'Ed_d', 'Ed_sr', 'Ed_sa' as static jnp arrays.
    Coupled model _forward_core() reads them directly.

    pre = downwelling_irradiance_jax.precompute(
              wavelengths, theta_sun=np.radians(30), P=1013.25, ...)
    # pre['Ed_d'], pre['Ed_sr'], pre['Ed_sa'] are ready to use

Mode B — atmosphere retrieval (theta_sun / aerosol params as fit parameters):
    Call precompute() without theta_sun (and without other scene scalars).
    The returned dict contains only the spectral data ('E0', 'a_oz', 'a_ox', 'a_wv').
    Coupled model _forward_core() calls _compute_Ed_d(p, pre) etc. on every forward
    call, computing Ed from p['theta_sun'], p['alpha'], p['beta'], etc.

    pre = downwelling_irradiance_jax.precompute(wavelengths)
    # pre has no 'Ed_d'; make_forward_vec must include 'theta_sun' etc. in param_names

Reference:
    Gege, P. (2012): Analytic model for the direct and diffuse components of downwelling
    spectral irradiance in water. [10.1364/AO.51.001407]
    Gege, P. (2021): The Water Colour Simulator WASI. User manual for WASI version 6.
"""

import numpy as np
import jax
import jax.numpy as jnp

from ..helper import resampling
from . import transmittance_jax as _T

jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------
# Layer 1 — Precompute (numpy, runs once outside JIT)
# ---------------------------------------------------------------------------

def precompute(wavelengths,
               theta_sun=None,
               P=1013.25,
               AM=5.0,
               RH=80.0,
               H_oz=0.381,
               WV=2.5,
               alpha=1.317,
               beta=0.2602,
               lambda_a=550.0):
    """
    Load and resample spectral data files; optionally pre-compute Ed arrays.

    Must NOT be called inside a jax.jit context.

    Args:
        wavelengths: wavelengths [nm], numpy array of shape (n_wavelengths,)
        theta_sun: sun zenith angle [radians].  If provided (Mode A), Ed_d / Ed_sr /
            Ed_sa are computed once and stored in the returned dict as static arrays.
            If None (Mode B), only the raw spectral files are loaded.
        P: atmospheric pressure [mbar], default: 1013.25  (Mode A only)
        AM: air mass type [1: open ocean .. 10: continental], default: 5  (Mode A only)
        RH: relative humidity [%], default: 80  (Mode A only)
        H_oz: ozone scale height [cm], default: 0.381  (Mode A only)
        WV: precipitable water [cm], default: 2.5  (Mode A only)
        alpha: Ångström exponent (typically 0.2–2), default: 1.317  (Mode A only)
        beta: turbidity coefficient (typically 0.16–0.50), default: 0.2602  (Mode A only)
        lambda_a: aerosol reference wavelength [nm], default: 550  (Mode A only)

    Returns:
        pre: dict of JAX arrays with keys:
            'wavelengths' — wavelengths [nm], shape (n_wavelengths,)
            'E0'          — extraterrestrial solar irradiance [mW m-2 nm-1],
                            shape (n_wavelengths,)
            'a_oz'        — ozone absorption coefficient [cm-1], shape (n_wavelengths,)
            'a_ox'        — oxygen absorption coefficient [cm-1], shape (n_wavelengths,)
            'a_wv'        — water vapour absorption coefficient [cm-1], shape (n_wavelengths,)
            --- Mode A only (theta_sun is not None) ---
            'Ed_d'        — direct downwelling irradiance [W m-2 nm-1], shape (n_wavelengths,)
            'Ed_sr'       — Rayleigh-scattered irradiance [W m-2 nm-1], shape (n_wavelengths,)
            'Ed_sa'       — aerosol-scattered irradiance [W m-2 nm-1], shape (n_wavelengths,)
    """
    wl = np.asarray(wavelengths, dtype=np.float64)

    pre = {
        'wavelengths': jnp.array(wl),
        'E0':   jnp.array(resampling.resample_E0(wl)),          # [mW m-2 nm-1]
        'a_oz': jnp.array(resampling.resample_a_oz(wl)),        # [cm-1]
        'a_ox': jnp.array(resampling.resample_a_ox(wl)),        # [cm-1]
        'a_wv': jnp.array(resampling.resample_a_wv(wl)),        # [cm-1]
    }

    if theta_sun is not None:
        p_atm = dict(theta_sun=float(theta_sun), P=float(P),
                     AM=float(AM), RH=float(RH), H_oz=float(H_oz),
                     WV=float(WV), alpha=float(alpha), beta=float(beta),
                     lambda_a=float(lambda_a))
        pre['Ed_d']  = _compute_Ed_d(p_atm, pre)
        pre['Ed_sr'] = _compute_Ed_sr(p_atm, pre)
        pre['Ed_sa'] = _compute_Ed_sa(p_atm, pre)

    return pre


# ---------------------------------------------------------------------------
# Layer 2 — JAX component functions (JIT-compilable)
# ---------------------------------------------------------------------------

def _compute_Ed_d(p, pre):
    """
    Direct component of downwelling irradiance [W m-2 nm-1].

    Args:
        p: dict with keys theta_sun, P, AM, RH, H_oz, WV, alpha, beta, lambda_a
        pre: dict from precompute() containing E0, a_oz, a_ox, a_wv, wavelengths

    Returns:
        Ed_d: shape (n_wavelengths,)
    """
    ts   = p['theta_sun']
    wl   = pre['wavelengths']
    # E0 is in mW m-2 nm-1; result in W m-2 nm-1
    return (pre['E0'] * 1e-3 * jnp.cos(ts)
            * _T.T_r(wl, ts, p['P'])
            * _T.T_aa(wl, ts, p['AM'], p['RH'], p['lambda_a'], p['alpha'], p['beta'])
            * _T.T_as(wl, ts, p['AM'], p['RH'], p['lambda_a'], p['alpha'], p['beta'])
            * _T.T_oz(wl, ts, p['H_oz'], pre['a_oz'])
            * _T.T_ox(wl, ts, p['P'],    pre['a_ox'])
            * _T.T_wv(wl, ts, p['WV'],   pre['a_wv']))


def _compute_Ed_sr(p, pre):
    """
    Rayleigh-scattered component of diffuse downwelling irradiance [W m-2 nm-1].

    Args:
        p: dict with keys theta_sun, P, AM, RH, H_oz, WV, alpha, beta, lambda_a
        pre: dict from precompute() containing E0, a_oz, a_ox, a_wv, wavelengths

    Returns:
        Ed_sr: shape (n_wavelengths,)
    """
    ts   = p['theta_sun']
    wl   = pre['wavelengths']
    return (0.5 * pre['E0'] * 1e-3 * jnp.cos(ts)
            * (1.0 - _T.T_r(wl, ts, p['P']) ** 0.95)
            * _T.T_aa(wl, ts, p['AM'], p['RH'], p['lambda_a'], p['alpha'], p['beta'])
            * _T.T_oz(wl, ts, p['H_oz'], pre['a_oz'])
            * _T.T_ox(wl, ts, p['P'],    pre['a_ox'])
            * _T.T_wv(wl, ts, p['WV'],   pre['a_wv']))


def _compute_Ed_sa(p, pre):
    """
    Aerosol-scattered component of diffuse downwelling irradiance [W m-2 nm-1].

    Args:
        p: dict with keys theta_sun, P, AM, RH, H_oz, WV, alpha, beta, lambda_a
        pre: dict from precompute() containing E0, a_oz, a_ox, a_wv, wavelengths

    Returns:
        Ed_sa: shape (n_wavelengths,)
    """
    ts   = p['theta_sun']
    wl   = pre['wavelengths']
    return (pre['E0'] * 1e-3 * jnp.cos(ts)
            * _T.T_r(wl, ts, p['P']) ** 1.5
            * _T.T_aa(wl, ts, p['AM'], p['RH'], p['lambda_a'], p['alpha'], p['beta'])
            * _T.T_oz(wl, ts, p['H_oz'], pre['a_oz'])
            * _T.T_ox(wl, ts, p['P'],    pre['a_ox'])
            * _T.T_wv(wl, ts, p['WV'],   pre['a_wv'])
            * (1.0 - _T.T_as(wl, ts, p['AM'], p['RH'], p['lambda_a'], p['alpha'], p['beta']))
            * _T.F_a(ts, p['alpha']))


def get_Ed_d(p, pre):
    """
    Return Ed_d: uses pre-computed array if available (Mode A), else computes on-the-fly.

    Args:
        p: parameter dict (needs atmosphere keys only in Mode B)
        pre: dict from precompute()

    Returns:
        Ed_d: shape (n_wavelengths,)
    """
    if 'Ed_d' in pre:
        return pre['Ed_d']
    return _compute_Ed_d(p, pre)


def get_Ed_sr(p, pre):
    """
    Return Ed_sr: uses pre-computed array if available (Mode A), else computes on-the-fly.

    Args:
        p: parameter dict (needs atmosphere keys only in Mode B)
        pre: dict from precompute()

    Returns:
        Ed_sr: shape (n_wavelengths,)
    """
    if 'Ed_sr' in pre:
        return pre['Ed_sr']
    return _compute_Ed_sr(p, pre)


def get_Ed_sa(p, pre):
    """
    Return Ed_sa: uses pre-computed array if available (Mode A), else computes on-the-fly.

    Args:
        p: parameter dict (needs atmosphere keys only in Mode B)
        pre: dict from precompute()

    Returns:
        Ed_sa: shape (n_wavelengths,)
    """
    if 'Ed_sa' in pre:
        return pre['Ed_sa']
    return _compute_Ed_sa(p, pre)

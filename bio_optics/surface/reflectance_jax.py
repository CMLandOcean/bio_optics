"""
JAX implementation of the surface reflectance (sky glint) model (Gege 2021 / WASI).

Two-layer architecture:
  - Layer 1: precompute() — numpy/scipy, runs once outside JIT.
    Delegates to downwelling_irradiance_jax.precompute(); supports the same dual-mode
    atmosphere design (Mode A: theta_sun supplied → Ed arrays pre-baked; Mode B:
    theta_sun=None → Ed computed on-the-fly from p, enabling atmosphere retrieval).
  - Layer 2: _forward_core() — pure JAX arithmetic, JIT-compilable, vmap-able.

Dual-mode atmosphere (inherited from downwelling_irradiance_jax):
    Mode A — fixed atmosphere (zero per-call overhead):
        pre = reflectance_jax.precompute(wavelengths, theta_sun=np.radians(30), ...)
        # pre['Ed_d'], pre['Ed_sr'], pre['Ed_sa'] are static jnp arrays

    Mode B — atmosphere retrieval (theta_sun / aerosol params as fit parameters):
        pre = reflectance_jax.precompute(wavelengths)   # no theta_sun
        f = reflectance_jax.make_forward_vec(
                ['theta_sun', 'alpha', 'beta', 'fd_d', 'g_dd', 'g_dsr', 'g_dsa',
                 'rho_L', 'd_r'], pre)

Surface reflectance parameters (always required in p):
    fd_d   — fractional contribution of direct irradiance [dimensionless]
    fd_s   — fractional contribution of diffuse irradiance [dimensionless]
    g_dd   — intensity of direct solar component [sr-1]
    g_dsr  — intensity of Rayleigh-scattered component [sr-1]
    g_dsa  — intensity of aerosol-scattered component [sr-1]
    theta_view — sensor viewing angle [radians]
    n1     — refractive index of air, default ~1
    n2     — refractive index of water, default ~1.33
    d_r    — additive offset [sr-1]

Atmosphere parameters (required in Mode B, i.e. when theta_sun not in precompute):
    theta_sun, P, AM, RH, H_oz, WV, alpha, beta, lambda_a

Reference:
    Gege, P. (2021): The Water Colour Simulator WASI. User manual for WASI version 6.
"""

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from ..atmosphere import downwelling_irradiance_jax as _di
from ..atmosphere import sky_radiance_jax as _sr
from . import air_water_jax


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
    Load atmospheric spectral data; optionally pre-compute Ed arrays.

    Delegates entirely to downwelling_irradiance_jax.precompute().  If theta_sun
    is supplied (Mode A), Ed_d / Ed_sr / Ed_sa are pre-baked into the returned
    dict — zero per-call overhead in _forward_core.  If theta_sun is None (Mode B),
    only the raw spectral files are loaded; Ed is computed on-the-fly from p at each
    forward call, enabling atmosphere / geometry retrieval.

    Must NOT be called inside a jax.jit context.

    Args:
        wavelengths: wavelengths [nm], numpy array of shape (n_wavelengths,)
        theta_sun: sun zenith angle [radians].  If provided (Mode A), Ed arrays are
            pre-computed.  If None (Mode B), atmosphere params must be in p at call time.
        P: atmospheric pressure [mbar], default: 1013.25  (Mode A only)
        AM: air mass type [1: open ocean .. 10: continental], default: 5  (Mode A only)
        RH: relative humidity [%], default: 80  (Mode A only)
        H_oz: ozone scale height [cm], default: 0.381  (Mode A only)
        WV: precipitable water [cm], default: 2.5  (Mode A only)
        alpha: Ångström exponent, default: 1.317  (Mode A only)
        beta: turbidity coefficient, default: 0.2602  (Mode A only)
        lambda_a: aerosol reference wavelength [nm], default: 550  (Mode A only)

    Returns:
        pre: dict of JAX arrays — same as downwelling_irradiance_jax.precompute():
            'wavelengths', 'E0', 'a_oz', 'a_ox', 'a_wv' always present
            'Ed_d', 'Ed_sr', 'Ed_sa' present only in Mode A (theta_sun is not None)
    """
    return _di.precompute(
        wavelengths,
        theta_sun=theta_sun,
        P=P, AM=AM, RH=RH,
        H_oz=H_oz, WV=WV,
        alpha=alpha, beta=beta,
        lambda_a=lambda_a,
    )


# ---------------------------------------------------------------------------
# Layer 2 — Core forward model (jnp, JIT-compilable)
# ---------------------------------------------------------------------------

def _forward_core(p, pre):
    """
    Core forward simulation — pure JAX arithmetic, JIT-compilable.

    Computes the surface reflectance (sky glint) contribution to above-water Rrs.

    Args:
        p: dict of scalar parameters.  Required keys:
            fd_d, fd_s, g_dd, g_dsr, g_dsa — sky radiance weighting [sr-1]
            theta_view — sensor viewing zenith angle [radians]
            n1, n2 — refractive indices of air and water
            d_r — residual / additive offset [sr-1]
            In Mode B (no Ed_d/sr/sa in pre): also theta_sun, P, AM, RH, H_oz, WV,
            alpha, beta, lambda_a.
        pre: dict of JAX arrays from precompute()

    Returns:
        Rrs_surface: surface reflectance contribution [sr-1], shape (n_wavelengths,)
    """
    Ed_d  = _di.get_Ed_d(p, pre)
    Ed_sr = _di.get_Ed_sr(p, pre)
    Ed_sa = _di.get_Ed_sa(p, pre)

    L_s = _sr.L_s(p["fd_d"], p["g_dd"],  Ed_d,
                  p["fd_s"], p["g_dsr"], Ed_sr,
                            p["g_dsa"], Ed_sa)

    Ed = p["fd_d"] * Ed_d + p["fd_s"] * (Ed_sr + Ed_sa)

    rho_L = air_water_jax.fresnel(p["theta_view"], n1=p["n1"], n2=p["n2"])

    return rho_L * L_s / Ed + p["d_r"]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def forward(params, precomputed):
    """
    Forward simulation of the surface reflectance (sky glint) contribution.

    Args:
        params: lmfit Parameters object or plain dict mapping parameter names to values
        precomputed: dict of JAX arrays returned by precompute()

    Returns:
        Rrs_surface: surface reflectance contribution [sr-1], shape (n_wavelengths,)
    """
    p = {k: float(v) for k, v in params.items()}
    return _forward_core(p, precomputed)


def make_forward_vec(param_names, precomputed):
    """
    Return a function f(params_vec) -> Rrs_surface suitable for jax.jit / jax.jacobian.

    Args:
        param_names: ordered list of parameter name strings
        precomputed: dict of JAX arrays returned by precompute()

    Returns:
        f: callable f(params_vec) -> Rrs_surface, params_vec shape (len(param_names),)
    """
    def f(params_vec):
        p = dict(zip(param_names, params_vec))
        return _forward_core(p, precomputed)
    return f

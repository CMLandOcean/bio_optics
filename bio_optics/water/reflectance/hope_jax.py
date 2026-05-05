"""
JAX implementation of the HOPE shallow water reflectance model (Lee et al. 1998 / 1999).

Two-layer architecture:
  - Layer 1: precompute() — numpy/scipy, runs once outside JIT.
    Loads a_w, bb_w (Morel), empirical A0/A1 (for a_Phi), and bottom reflectance R_b_i.
  - Layer 2: _forward_core() — pure JAX arithmetic, JIT-compilable, vmap-able.

Usage::

    import numpy as np
    import jax
    from bio_optics.water.reflectance import hope_jax

    pre  = hope_jax.precompute(wavelengths)
    Rrs  = hope_jax.forward(params, pre)

    names = list(params.keys())
    f_vec = hope_jax.make_forward_vec(names, pre)
    Rrs   = jax.jit(f_vec)(params_vec)
    J     = jax.jacobian(f_vec)(params_vec)   # shape (n_wl, n_params)

Parameters in p (required by _forward_core):
    C_Mie      — non-algal particle concentration [mg L-1] (represents X when bb_Mie_spec=1)
    C_Y        — CDOM absorption at lambda_0 [m-1]
    a_phy_440  — phytoplankton absorption at 440 nm [m-1]
    zB         — water depth [m]
    f_0..f_5   — fractional bottom cover per type [dimensionless]
    B_0..B_5   — bidirectional reflectance factor per type [sr-1]
    lambda_0   — CDOM reference wavelength [nm]
    lambda_S   — particle backscattering reference wavelength [nm]
    S          — CDOM spectral slope [nm-1]
    bb_Mie_spec — specific backscattering coefficient [m2 mg-1]; set to 1 for Lee formulation
    n          — Mie scattering exponent (negative, related to Y = -n)
    n1         — refractive index of air
    n2         — refractive index of water
    theta_sun  — sun zenith angle [radians]
    g_0        — deep-water rrs empirical constant [sr-1]
    g_1        — deep-water rrs empirical constant [sr-1]
    offset     — additive spectral offset [sr-1]

References:
    [1] Lee et al. (1998): Hyperspectral remote sensing for shallow waters: 1. A semianalytical
        model [10.1364/AO.37.006329]
    [2] Lee et al. (1999): Hyperspectral remote sensing for shallow waters: 2. Deriving bottom
        depths and water properties by optimization [10.1364/ao.38.003831]
    [3] Albert & Mobley (2003): An analytical model for subsurface irradiance and remote sensing
        reflectance in deep and shallow case-2 waters. [10.1364/OE.11.002873]
"""

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from ...helper import resampling
from ..backscattering import morel


# ---------------------------------------------------------------------------
# Layer 1 — Precompute (numpy, runs once outside JIT)
# ---------------------------------------------------------------------------

def precompute(wavelengths, fresh=False):
    """
    Load and resample all static spectral lookup tables.

    Must NOT be called inside a jax.jit context.

    Args:
        wavelengths: wavelengths [nm], numpy array of shape (n_wavelengths,)
        fresh: True for fresh water (controls bb_w via Morel formula), default: False

    Returns:
        pre: dict of JAX arrays with keys:
            'wavelengths' — wavelengths [nm], shape (n_wavelengths,)
            'a_w'         — pure water absorption [m-1], shape (n_wavelengths,)
            'bb_w'        — pure water backscattering [m-1], shape (n_wavelengths,)
            'A0'          — empirical phytoplankton absorption factor A0 [m-1], shape (n_wavelengths,)
            'A1'          — empirical phytoplankton absorption factor A1 [dimensionless], shape (n_wavelengths,)
            'R_b_i'       — bottom reflectance spectra [sr-1], shape (n_wavelengths, 6)
    """
    wl = np.asarray(wavelengths, dtype=np.float64)
    A0, A1 = resampling.resample_A(wl)
    return {
        "wavelengths": jnp.array(wl),
        "a_w":   jnp.array(resampling.resample_a_w(wl)),
        "bb_w":  jnp.array(morel(wavelengths=wl, fresh=fresh)),
        "A0":    jnp.array(A0),
        "A1":    jnp.array(A1),
        "R_b_i": jnp.array(resampling.resample_R_b_i(wl)),  # (n_wl, 6)
    }


# ---------------------------------------------------------------------------
# Layer 2 — Core forward model (jnp, JIT-compilable)
# ---------------------------------------------------------------------------

def _forward_core(p, pre):
    """
    Core forward simulation — pure JAX arithmetic, JIT-compilable.

    Implements Lee et al. (1998, 1999) shallow water model, converting subsurface
    rrs_sh to above-water Rrs via the Lee et al. (1998) air-water interface correction.

    Args:
        p: dict of scalar parameters (see module docstring for full list)
        pre: dict of JAX arrays from precompute()

    Returns:
        Rrs: above-water remote sensing reflectance [sr-1], shape (n_wavelengths,)
    """
    wl = pre["wavelengths"]

    # --- Backscattering ---
    bb_Mie = p["C_Mie"] * p["bb_Mie_spec"] * (wl / p["lambda_S"]) ** p["n"]
    bb = pre["bb_w"] + bb_Mie

    # --- Absorption ---
    a_Y   = p["C_Y"] * jnp.exp(-p["S"] * (wl - p["lambda_0"]))
    a_Phi = (pre["A0"] + pre["A1"] * jnp.log(p["a_phy_440"])) * p["a_phy_440"]
    a = pre["a_w"] + a_Y + a_Phi

    # --- IOP ratios ---
    kappa = a + bb
    u = bb / kappa

    # --- Path elongation factors (Lee et al. 1999 Eq. 5) ---
    D_u_C = 1.03 * (1.0 + 2.4 * u) ** 0.5   # f1=1.03, f2=2.4
    D_u_B = 1.04 * (1.0 + 5.4 * u) ** 0.5   # f1=1.04, f2=5.4

    # --- In-water refraction angle ---
    theta_w = jnp.arcsin(p["n1"] / p["n2"] * jnp.sin(p["theta_sun"]))
    inv_cos_w = 1.0 / jnp.cos(theta_w)

    # --- Deep-water subsurface rrs ---
    rrs_deep = (p["g_0"] + p["g_1"] * u) * u

    # --- Bottom reflectance weighted sum ---
    R_b = (p["f_0"] * p["B_0"] * pre["R_b_i"][:, 0] +
           p["f_1"] * p["B_1"] * pre["R_b_i"][:, 1] +
           p["f_2"] * p["B_2"] * pre["R_b_i"][:, 2] +
           p["f_3"] * p["B_3"] * pre["R_b_i"][:, 3] +
           p["f_4"] * p["B_4"] * pre["R_b_i"][:, 4] +
           p["f_5"] * p["B_5"] * pre["R_b_i"][:, 5])

    # --- Shallow water subsurface rrs (Lee et al. 1999 Eq. 9) ---
    depth_C = kappa * p["zB"] * (inv_cos_w + D_u_C)
    depth_B = kappa * p["zB"] * (inv_cos_w + D_u_B)
    rrs_sh = rrs_deep * (1.0 - jnp.exp(-depth_C)) + R_b * jnp.exp(-depth_B)

    # --- Air-water interface correction (Lee et al. 1998) ---
    # below2above: Rrs = zeta * rrs / (1 - Gamma * rrs)
    rrs_total = rrs_sh + p["offset"]
    return (0.52 * rrs_total) / (1.0 - 1.6 * rrs_total)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def forward(params, precomputed):
    """
    Forward simulation of shallow water remote sensing reflectance (Lee et al. 1998 / 1999).

    Args:
        params: lmfit Parameters object or plain dict mapping parameter names to values
        precomputed: dict of JAX arrays returned by precompute()

    Returns:
        Rrs: above-water remote sensing reflectance [sr-1], shape (n_wavelengths,)
    """
    p = {k: float(v) for k, v in params.items()}
    return _forward_core(p, precomputed)


def make_forward_vec(param_names, precomputed):
    """
    Return a function f(params_vec) -> Rrs suitable for jax.jit / jax.jacobian.

    Args:
        param_names: ordered list of parameter name strings
        precomputed: dict of JAX arrays returned by precompute()

    Returns:
        f: callable f(params_vec) -> Rrs, params_vec shape (len(param_names),)
    """
    def f(params_vec):
        p = dict(zip(param_names, params_vec))
        return _forward_core(p, precomputed)
    return f

"""
JAX implementation of the coupled SBOP model: shallow water Rrs + surface reflectance.

Two-layer architecture:
  - Layer 1: precompute() — numpy/scipy, runs once outside JIT.
    Merges sbop_jax.precompute() (water tables) with downwelling_irradiance_jax.precompute()
    (atmosphere tables).  Supports dual-mode atmosphere: Mode A (theta_sun supplied →
    Ed arrays pre-baked, zero overhead) or Mode B (theta_sun=None → Ed on-the-fly).
  - Layer 2: _forward_core() — pure JAX arithmetic, JIT-compilable, vmap-able.

Usage::

    import numpy as np
    import jax
    from bio_optics.coupled_models import sbop_3C_jax

    pre  = sbop_3C_jax.precompute(wavelengths, theta_sun=np.radians(35))
    Rrs  = sbop_3C_jax.forward(params, pre)

    names  = list(params.keys())
    f_vec  = sbop_3C_jax.make_forward_vec(names, pre)
    Rrs    = jax.jit(f_vec)(params_vec)
    J      = jax.jacobian(f_vec)(params_vec)   # shape (n_wl, n_params)

References:
    [1] Li et al. (2017): Remote sensing estimation of colored dissolved organic matter (CDOM)
        in optically shallow waters [10.1016/j.isprsjprs.2017.03.015]
    [2] Gege, P. (2021): The Water Colour Simulator WASI. User manual for WASI version 6.
"""

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from ..water.reflectance import sbop_jax
from ..atmosphere import downwelling_irradiance_jax as _di
from ..surface import air_water_jax


# ---------------------------------------------------------------------------
# Layer 1 — Precompute (numpy, runs once outside JIT)
# ---------------------------------------------------------------------------

def precompute(wavelengths,
               fresh=False,
               # Atmosphere — supply theta_sun for Mode A (pre-baked Ed arrays);
               # pass theta_sun=None for Mode B (Ed computed on-the-fly in _forward_core).
               theta_sun=np.radians(30),
               P=1013.25,
               AM=5,
               RH=80,
               H_oz=0.381,
               WV=2.5,
               alpha=1.317,
               beta=0.2602,
               # Optional pre-computed irradiance arrays (override downwelling_irradiance_jax)
               Ed_d_res=None,
               Ed_sa_res=None,
               Ed_sr_res=None,
               # Optional pre-computed sky-to-Ed ratio spectrum
               Ls_Ed=None):
    """
    Load all static spectral lookup tables; optionally pre-compute Ed arrays.

    Merges sbop_jax.precompute() (a_w, bb_w, R_b_i) with
    downwelling_irradiance_jax.precompute() (E0, a_oz, a_ox, a_wv; plus Ed arrays
    in Mode A).  Must NOT be called inside a jax.jit context.

    Args:
        wavelengths: wavelengths [nm], numpy array of shape (n_wavelengths,)
        fresh: True for fresh water (controls bb_w), default: False
        theta_sun: sun zenith angle [radians], default: np.radians(30).  Pass None for
            Mode B (atmosphere / geometry retrieval via lmfit).
        P: atmospheric pressure [mbar], default: 1013.25
        AM: air mass type [1..10], default: 5
        RH: relative humidity [%], default: 80
        H_oz: ozone scale height [cm], default: 0.381
        WV: precipitable water [cm], default: 2.5
        alpha: Angström exponent, default: 1.317
        beta: turbidity coefficient, default: 0.2602
        Ed_d_res: optional pre-computed direct downwelling irradiance [W m-2 nm-1]
        Ed_sa_res: optional pre-computed aerosol-scattered irradiance [W m-2 nm-1]
        Ed_sr_res: optional pre-computed Rayleigh-scattered irradiance [W m-2 nm-1]
        Ls_Ed: optional sky radiance / downwelling irradiance ratio [sr-1], shape (n_wavelengths,)

    Returns:
        pre: merged dict of JAX arrays from sbop_jax.precompute() and
             downwelling_irradiance_jax.precompute(), plus 'Ls_Ed'
    """
    pre = sbop_jax.precompute(wavelengths, fresh=fresh)

    atm_pre = _di.precompute(wavelengths,
                             theta_sun=theta_sun, P=P, AM=AM, RH=RH,
                             H_oz=H_oz, WV=WV, alpha=alpha, beta=beta)
    pre.update(atm_pre)

    if Ed_d_res is not None:
        pre["Ed_d"]  = jnp.array(np.asarray(Ed_d_res))
    if Ed_sr_res is not None:
        pre["Ed_sr"] = jnp.array(np.asarray(Ed_sr_res))
    if Ed_sa_res is not None:
        pre["Ed_sa"] = jnp.array(np.asarray(Ed_sa_res))

    pre["Ls_Ed"] = jnp.array(np.asarray(Ls_Ed) if Ls_Ed is not None
                              else np.zeros(len(wavelengths)))

    return pre


# ---------------------------------------------------------------------------
# Layer 2 — Core forward model (jnp, JIT-compilable)
# ---------------------------------------------------------------------------

def _forward_core(p, pre):
    """
    Core forward simulation — pure JAX arithmetic, JIT-compilable.

    Computes shallow water Rrs (sbop_jax._forward_core) and adds sky glint surface
    reflectance following Gege (2021).

    Args:
        p: dict of scalar parameters.  All keys from sbop_jax._forward_core plus:
            Surface / sky: fd_d, fd_s, g_dd, g_dsr, g_dsa, theta_view, n1, n2, d_r
            (In Mode B also: theta_sun, P, AM, RH, H_oz, WV, alpha, beta, lambda_a)
        pre: dict of JAX arrays from precompute()

    Returns:
        Rrs: above-water remote sensing reflectance [sr-1], shape (n_wavelengths,)
    """
    Rrs_water = sbop_jax._forward_core(p, pre)

    # --- irradiance components (Mode A: from pre; Mode B: computed from p) ---
    Ed_d  = _di.get_Ed_d(p, pre)
    Ed_sr = _di.get_Ed_sr(p, pre)
    Ed_sa = _di.get_Ed_sa(p, pre)

    # --- Sky radiance (Gege 2021) ---
    L_s = (p["fd_d"] * p["g_dd"]  * Ed_d
           + p["fd_s"] * (p["g_dsr"] * Ed_sr + p["g_dsa"] * Ed_sa))

    # --- Total downwelling irradiance ---
    Ed = p["fd_d"] * Ed_d + p["fd_s"] * (Ed_sr + Ed_sa)

    # --- Fresnel reflectance for viewing direction ---
    rho_L = air_water_jax.fresnel(p["theta_view"], n1=p["n1"], n2=p["n2"])

    # --- Surface contribution ---
    Rrs_surf = rho_L * L_s / Ed + p["d_r"] + rho_L * pre["Ls_Ed"]

    return Rrs_water + Rrs_surf


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def forward(params, precomputed):
    """
    Forward simulation: shallow water Rrs (Li 2017) + surface reflectance (Gege 2021).

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
    Return a function f(params_vec, aux=None) -> Rrs suitable for jax.jit / jax.jacobian.

    Args:
        param_names: ordered list of parameter name strings
        precomputed: dict of JAX arrays returned by precompute()

    Returns:
        f: callable f(params_vec, aux=None) -> Rrs.
           When aux is a dict it is merged with precomputed (aux takes precedence).
    """
    def f(params_vec, aux=None):
        p   = dict(zip(param_names, params_vec))
        pre = precomputed if aux is None else {**precomputed, **aux}
        return _forward_core(p, pre)
    return f

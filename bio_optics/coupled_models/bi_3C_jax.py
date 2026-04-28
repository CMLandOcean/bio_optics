"""
JAX implementation of the coupled HEREON model: water-leaving Rrs + surface reflectance.

This is a two-component model (water + surface). For the three-component version
that additionally includes fluorescence, see ``bio_optics.coupled_models.bi_fluo_3C_jax``.

Two-layer architecture:
  - Layer 1: precompute() — numpy/scipy, runs once outside JIT, converts to JAX arrays
  - Layer 2: _forward_core() — pure JAX arithmetic, JIT-compilable, vmap-able

Usage::

    import numpy as np
    import jax
    from bio_optics.coupled_models import bi_3C_jax

    pre  = bi_3C_jax.precompute(wavelengths, theta_sun=np.radians(35))
    Rrs  = bi_3C_jax.forward(params, pre)

    # JIT-compiled vectorised forward
    names  = list(params.keys())
    f_vec  = bi_3C_jax.make_forward_vec(names, pre)
    Rrs    = jax.jit(f_vec)(params_vec)
    J      = jax.jacobian(f_vec)(params_vec)   # shape (n_wl, n_params)

References:
    [1] Bi et al. (2023): Bio-geo-optical modelling of natural waters [10.3389/fmars.2023.1196352]
    [2] Gege, P. (2021): The Water Colour Simulator WASI. User manual for WASI version 6.
"""

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from ..atmosphere import downwelling_irradiance
from ..surface import air_water_jax
from ..water.reflectance import bi_jax


# ---------------------------------------------------------------------------
# Layer 1 — Precompute (numpy, runs once outside JIT)
# ---------------------------------------------------------------------------

def precompute(wavelengths,
               fresh=False,
               phy_source='a_phy_EnSAD',
               b_phy_source='b_phy_EnSAD',
               # Atmospheric scalars used to compute Ed components
               theta_sun=np.radians(30),
               P=1013.25,
               AM=5,
               RH=80,
               H_oz=0.381,
               WV=2.5,
               alpha=1.317,
               beta=0.2602,
               # Optional precomputed atmospheric absorption tables
               E0_res=[],
               a_oz_res=[],
               a_ox_res=[],
               a_wv_res=[],
               # Optional precomputed irradiance components (skips recomputation)
               Ed_d_res=[],
               Ed_sa_res=[],
               Ed_sr_res=[],
               # Optional precomputed sky-to-Ed ratio spectrum
               Ls_Ed=[]):
    """
    Resample all static spectral lookup tables once and return as a dict of JAX arrays.

    Calls ``bi_jax.precompute()`` for water optical tables and adds atmospheric
    irradiance components for the surface reflectance model.
    Must NOT be called inside a jax.jit context.

    Args:
        wavelengths: wavelengths [nm], numpy array of shape (n_wavelengths,)
        fresh: True for fresh water, False for oceanic water (controls bb_w), default: False
        phy_source: phytoplankton absorption library keyword or file path, default: 'a_phy_EnSAD'
        b_phy_source: phytoplankton scattering library keyword or file path, default: 'b_phy_EnSAD'
        theta_sun: sun zenith angle [radians], default: np.radians(30)
        P: atmospheric pressure [mbar], default: 1013.25
        AM: air mass type [1..10], default: 5
        RH: relative humidity [%], default: 80
        H_oz: ozone scale height [cm], default: 0.381
        WV: precipitable water [cm], default: 2.5
        alpha: Angström exponent, default: 1.317
        beta: turbidity coefficient, default: 0.2602
        E0_res: optional precomputed extraterrestrial solar irradiance
        a_oz_res: optional precomputed ozone absorption coefficient
        a_ox_res: optional precomputed oxygen absorption coefficient
        a_wv_res: optional precomputed water vapour absorption coefficient
        Ed_d_res: optional precomputed direct downwelling irradiance
        Ed_sa_res: optional precomputed aerosol-scattered downwelling irradiance
        Ed_sr_res: optional precomputed Rayleigh-scattered downwelling irradiance
        Ls_Ed: optional ratio of sky radiance to downwelling irradiance [sr-1], shape (n_wavelengths,)

    Returns:
        precomputed: dict of JAX arrays. All keys from ``bi_jax.precompute()`` plus:
            "Ed_d"   — direct downwelling irradiance [W m-2 nm-1], shape (n_wavelengths,)
            "Ed_sr"  — Rayleigh-scattered downwelling irradiance [W m-2 nm-1], shape (n_wavelengths,)
            "Ed_sa"  — aerosol-scattered downwelling irradiance [W m-2 nm-1], shape (n_wavelengths,)
            "Ls_Ed"  — sky radiance / downwelling irradiance ratio [sr-1], shape (n_wavelengths,)
    """
    pre = bi_jax.precompute(wavelengths, fresh=fresh,
                             phy_source=phy_source, b_phy_source=b_phy_source)

    atm_kwargs = dict(
        theta_sun=theta_sun, P=P, AM=AM, RH=RH,
        H_oz=H_oz, WV=WV, alpha=alpha, beta=beta,
        E0_res=E0_res, a_oz_res=a_oz_res, a_ox_res=a_ox_res, a_wv_res=a_wv_res,
    )
    Ed_d  = (np.array(Ed_d_res)  if len(Ed_d_res)  > 0
             else downwelling_irradiance.Ed_d( wavelengths, **atm_kwargs))
    Ed_sr = (np.array(Ed_sr_res) if len(Ed_sr_res) > 0
             else downwelling_irradiance.Ed_sr(wavelengths, **atm_kwargs))
    Ed_sa = (np.array(Ed_sa_res) if len(Ed_sa_res) > 0
             else downwelling_irradiance.Ed_sa(wavelengths, **atm_kwargs))
    Ls_Ed_arr = (np.array(Ls_Ed) if len(Ls_Ed) > 0 else np.zeros(len(wavelengths)))

    return {
        **pre,
        "Ed_d":  jnp.array(Ed_d),
        "Ed_sr": jnp.array(Ed_sr),
        "Ed_sa": jnp.array(Ed_sa),
        "Ls_Ed": jnp.array(Ls_Ed_arr),
    }


# ---------------------------------------------------------------------------
# Layer 2 — Core forward model (jnp, JIT-compilable)
# ---------------------------------------------------------------------------

def _forward_core(p, pre):
    """
    Core forward simulation — pure JAX arithmetic, JIT-compilable.

    Combines water-leaving Rrs (via ``bi_jax._forward_core``) with sky/sun glint
    surface reflectance following Gege (2021) [2].

    Args:
        p: dict of scalar parameters (floats or JAX 0-d arrays). All keys required by
           ``bi_jax._forward_core`` plus:
               Surface / viewing geometry: theta_view, n1, n2, fd_d, fd_s, g_dd, g_dsr, g_dsa, d_r
        pre: dict of precomputed JAX arrays from precompute()

    Returns:
        Rrs: above-water remote sensing reflectance [sr-1], shape (n_wavelengths,)
    """
    Rrs_water = bi_jax._forward_core(p, pre)

    # --- Surface reflectance (sky glint; Gege 2021) ---
    # Sky radiance: L_s = fd_d * g_dd * Ed_d + fd_s * (g_dsr * Ed_sr + g_dsa * Ed_sa)
    L_s = (p["fd_d"] * p["g_dd"]  * pre["Ed_d"]
           + p["fd_s"] * (p["g_dsr"] * pre["Ed_sr"] + p["g_dsa"] * pre["Ed_sa"]))

    # Total downwelling irradiance: Ed = fd_d * Ed_d + fd_s * (Ed_sr + Ed_sa)
    Ed = p["fd_d"] * pre["Ed_d"] + p["fd_s"] * (pre["Ed_sr"] + pre["Ed_sa"])

    # Fresnel reflectance for the viewing direction
    rho_L = air_water_jax.fresnel(p["theta_view"], n1=p["n1"], n2=p["n2"])

    # Surface contribution: sky glint + white-cap/foam term + residual sky radiance
    Rrs_surf = rho_L * L_s / Ed + p["d_r"] + rho_L * pre["Ls_Ed"]

    return Rrs_water + Rrs_surf


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def forward(params, precomputed):
    """
    Forward simulation: water-leaving Rrs (Bi et al. 2023) [1] + surface reflectance (Gege 2021) [2].

    [1] Bi et al. (2023): Bio-geo-optical modelling of natural waters [10.3389/fmars.2023.1196352]
    [2] Gege, P. (2021): The Water Colour Simulator WASI. User manual for WASI version 6.

    Args:
        params: lmfit Parameters object or plain dict mapping parameter names to scalar values
        precomputed: dict of JAX arrays returned by precompute()

    Returns:
        Rrs: above-water remote sensing reflectance [sr-1], shape (n_wavelengths,)
    """
    p = {k: float(v) for k, v in params.items()}
    return _forward_core(p, precomputed)


def make_forward_vec(param_names, precomputed):
    """
    Return a function f(params_vec, aux=None) -> Rrs suitable for jax.jit, jax.jacobian, and jax.vmap.

    The returned function takes a 1-D JAX array of parameter values (in the order given by
    param_names) and returns the simulated above-water reflectance spectrum.

    The optional ``aux`` argument is a dict that overrides entries in ``precomputed`` on a
    per-call basis, enabling per-pixel variation of any precomputed spectral quantity.

    Example usage::

        pre    = precompute(wavelengths, theta_sun=np.radians(35))
        names  = ["C_0", "C_Y", "C_ism", "offset", "fd_d", ...]
        f_vec  = make_forward_vec(names, pre)

        Rrs = jax.jit(f_vec)(params_vec)
        J   = jax.jacobian(f_vec)(params_vec)             # (n_wl, n_params)
        Rrs_batch = jax.vmap(f_vec)(params_matrix)        # (n_pixels, n_wl)

    Args:
        param_names: ordered list of parameter name strings matching the columns of params_vec
        precomputed: dict of JAX arrays returned by precompute()

    Returns:
        f: callable f(params_vec, aux=None) -> Rrs where params_vec has shape (len(param_names),).
           When aux is a dict it is merged with precomputed (aux takes precedence).
    """
    def f(params_vec, aux=None):
        p   = dict(zip(param_names, params_vec))
        pre = precomputed if aux is None else {**precomputed, **aux}
        return _forward_core(p, pre)
    return f

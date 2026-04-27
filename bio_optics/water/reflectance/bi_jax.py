"""
JAX implementation of the HEREON bi water optical model (Bi et al. 2023).

Water-leaving Rrs only — no fluorescence, no surface reflectance.
For the coupled model with surface reflectance see ``bio_optics.coupled_models.bi_3C_jax``.
For the coupled model with fluorescence and surface see ``bio_optics.coupled_models.bi_fluo_3C_jax``.

Two-layer architecture:
  - Layer 1: precompute() — numpy/scipy, runs once outside JIT, converts to JAX arrays
  - Layer 2: _forward_core() — pure JAX arithmetic, JIT-compilable, vmap-able

Usage::

    import numpy as np
    import jax
    from bio_optics.water.reflectance import bi_jax

    pre  = bi_jax.precompute(wavelengths)
    Rrs  = bi_jax.forward(params, pre)

    # JIT-compiled vectorised forward
    names  = list(params.keys())
    f_vec  = bi_jax.make_forward_vec(names, pre)
    Rrs    = jax.jit(f_vec)(params_vec)
    J      = jax.jacobian(f_vec)(params_vec)   # shape (n_wl, n_params)

References:
    [1] Bi et al. (2023): Bio-geo-optical modelling of natural waters [10.3389/fmars.2023.1196352]
    [2] Lee et al. (2011): Deriving inherent optical properties from water color [10.1364/AO.38.003628]
"""

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from ..backscattering import morel
from ...helper import resampling

MAX_PHY_CLASSES = 8


# ---------------------------------------------------------------------------
# Layer 1 — Precompute (numpy, runs once outside JIT)
# ---------------------------------------------------------------------------

def precompute(wavelengths, fresh=False,
               phy_source='a_phy_EnSAD', b_phy_source='b_phy_EnSAD'):
    """
    Resample all static spectral lookup tables once and return as a dict of JAX arrays.

    Must NOT be called inside a jax.jit context.

    Args:
        wavelengths: wavelengths [nm], numpy array of shape (n_wavelengths,)
        fresh: True for fresh water, False for oceanic water (controls bb_w), default: False
        phy_source: phytoplankton absorption library. Keyword from the internal registry
                    ('a_phy', 'a_phy_EnSAD') or path to a custom CSV file. Default: 'a_phy_EnSAD'
        b_phy_source: phytoplankton scattering library. Keyword or file path. Default: 'b_phy_EnSAD'

    Returns:
        precomputed: dict of JAX arrays with keys:
            "wavelengths"  — wavelengths [nm], shape (n_wavelengths,)
            "a_w"          — pure water absorption [m-1], shape (n_wavelengths,)
            "da_w_div_dT"  — temperature gradient of pure water absorption [m-1 K-1], shape (n_wavelengths,)
            "a_i_spec"     — specific phytoplankton absorption [m2 mg-1], shape (n_wavelengths, MAX_PHY_CLASSES)
            "b_i_spec"     — specific phytoplankton scattering [m2 mg-1], shape (n_wavelengths, MAX_PHY_CLASSES)
            "bb_w"         — pure water backscattering [m-1], shape (n_wavelengths,)
            "n_classes"    — int: actual number of spectral classes in the absorption library
    """
    a_i_spec_raw = resampling.resample_phy_spec(wavelengths, source=phy_source)
    n_classes = a_i_spec_raw.shape[1]
    a_i_spec = resampling.resample_phy_spec(wavelengths, source=phy_source, n_pad=MAX_PHY_CLASSES)
    b_i_spec = resampling.resample_phy_spec(wavelengths, source=b_phy_source, n_pad=MAX_PHY_CLASSES)

    return {
        "wavelengths":  jnp.array(wavelengths),
        "a_w":          jnp.array(resampling.resample_a_w(wavelengths)),
        "da_w_div_dT":  jnp.array(resampling.resample_da_w_div_dT(wavelengths)),
        "a_i_spec":     jnp.array(a_i_spec),
        "b_i_spec":     jnp.array(b_i_spec),
        "bb_w":         jnp.array(morel(wavelengths=wavelengths, fresh=fresh)),
        "n_classes":    n_classes,
    }


# ---------------------------------------------------------------------------
# Layer 2 — Core forward model (jnp, JIT-compilable)
# ---------------------------------------------------------------------------

def _forward_core(p, pre):
    """
    Core forward simulation — pure JAX arithmetic, JIT-compilable.

    Implements the HEREON water optical model (Bi et al. 2023) [1] computing
    water-leaving Rrs using the Lee et al. (2011) [2] deep-water formula.

    Args:
        p: dict of scalar parameters (floats or JAX 0-d arrays). Required keys:
            Phytoplankton concentrations [ug/L]:  C_0 .. C_7
            CDOM:                                 C_Y, S_cdom, lambda_0_cdom, K
            ISM:                                  C_ism
            Minerogenic detritus absorption:      A_md, S_md, C_md, lambda_0_md
            Biogenic detritus absorption:         A_bd, S_bd, C_bd, lambda_0_bd
            Detritus attenuation:                 gamma_d, x0, x1, x2, lambda_0_c_d
            Temperature:                          T_W, T_W_0
            Phy packaging correction:             A_phy, E0, E1, lambda_0_phy
            Phy backscattering ratios:            b_ratio_C_0 .. b_ratio_C_7
            Detritus backscattering ratios:       b_ratio_md, b_ratio_bd
            Lee (2011) coefficients:              Gw0, Gw1, Gp0, Gp1
            Spectral offset:                      offset
        pre: dict of precomputed JAX arrays from precompute()

    Returns:
        Rrs_water: water-leaving remote sensing reflectance [sr-1], shape (n_wavelengths,)
    """
    wl   = pre["wavelengths"]
    bb_w = pre["bb_w"]

    # --- Phytoplankton concentrations ---
    C_i = jnp.array([p["C_0"], p["C_1"], p["C_2"], p["C_3"],
                     p["C_4"], p["C_5"], p["C_6"], p["C_7"]])
    C_phy = jnp.sum(C_i)

    # --- Absorption ---

    # Pure water (temperature-corrected)
    a_water = pre["a_w"] + (p["T_W"] - p["T_W_0"]) * pre["da_w_div_dT"]

    # Phytoplankton: raw dot product then packaging correction
    a_phy_raw = jnp.dot(pre["a_i_spec"], C_i)
    a_phy_at_ref = jnp.interp(p["lambda_0_phy"], wl, a_phy_raw)
    E_phy = jnp.where(C_phy <= 1.0, p["E0"], p["E1"])
    # When C_phy = 0 the raw spectrum is zero; guard against 0/0
    correction = jnp.where(
        C_phy > 0.0,
        (p["A_phy"] * C_phy ** E_phy) / jnp.maximum(a_phy_at_ref, 1e-30),
        1.0,
    )
    a_phy = a_phy_raw * correction

    # CDOM
    a_Y = p["C_Y"] * jnp.exp(-p["S_cdom"] * (wl - p["lambda_0_cdom"])) + p["K"]

    # Minerogenic detritus absorption (Eq. 7a in Bi et al. 2023)
    a_md_spec = p["A_md"] * jnp.exp(-p["S_md"] * (wl - p["lambda_0_md"])) + p["C_md"]
    a_md = p["C_ism"] * a_md_spec

    # Biogenic detritus absorption (Eq. 7b in Bi et al. 2023)
    a_bd_spec = p["A_bd"] * jnp.exp(-p["S_bd"] * (wl - p["lambda_0_bd"])) + p["C_bd"]
    a_bd = C_phy * a_bd_spec

    # Total absorption
    a = a_water + a_phy + a_Y + a_md + a_bd

    # --- Detritus attenuation and scattering ---
    # Single scattering albedo of detritus at reference wavelength (Eq. 11 in Bi et al. 2023)
    omega_d = p["x0"] - p["x1"] ** p["x2"]
    one_minus_omega_d = jnp.maximum(1.0 - omega_d, 1e-10)

    # Minerogenic detrital attenuation (Eq. 9 in Bi et al. 2023)
    a_md_at_ref = p["C_ism"] * (p["A_md"] * jnp.exp(-p["S_md"] * (p["lambda_0_c_d"] - p["lambda_0_md"])) + p["C_md"])
    c_md_at_ref = a_md_at_ref / one_minus_omega_d
    c_md = c_md_at_ref * (p["lambda_0_c_d"] / wl) ** p["gamma_d"]

    # Biogenic detrital attenuation (Eq. 9 in Bi et al. 2023)
    a_bd_at_ref = C_phy * (p["A_bd"] * jnp.exp(-p["S_bd"] * (p["lambda_0_c_d"] - p["lambda_0_bd"])) + p["C_bd"])
    c_bd_at_ref = a_bd_at_ref / one_minus_omega_d
    c_bd = c_bd_at_ref * (p["lambda_0_c_d"] / wl) ** p["gamma_d"]

    # Detrital scattering: b = c - a
    b_md = c_md - a_md
    b_bd = c_bd - a_bd

    # --- Backscattering ---

    # Phytoplankton (per-type backscattering ratios)
    bb_ratio_C_i = jnp.array([p["b_ratio_C_0"], p["b_ratio_C_1"], p["b_ratio_C_2"], p["b_ratio_C_3"],
                               p["b_ratio_C_4"], p["b_ratio_C_5"], p["b_ratio_C_6"], p["b_ratio_C_7"]])
    bb_phy = jnp.dot(pre["b_i_spec"], C_i * bb_ratio_C_i)

    # Detrital backscattering
    bb_md = p["b_ratio_md"] * b_md
    bb_bd = p["b_ratio_bd"] * b_bd

    # Total particulate and total backscattering
    bb_p = bb_phy + bb_md + bb_bd
    bb   = bb_w + bb_p

    # --- Water-leaving Rrs: Lee et al. (2011) deep-water formula ---
    # Includes air–water transmission via the Lee (1998) parameterisation
    k = a + bb
    Rrs_water = (
        (p["Gw0"] + p["Gw1"] * bb_w / k) * bb_w / k
        + (p["Gp0"] + p["Gp1"] * bb_p / k) * bb_p / k
    )

    return Rrs_water + p["offset"]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def forward(params, precomputed):
    """
    Forward simulation of water-leaving Rrs after Bi et al. (2023) [1].

    [1] Bi et al. (2023): Bio-geo-optical modelling of natural waters [10.3389/fmars.2023.1196352]

    Args:
        params: lmfit Parameters object or plain dict mapping parameter names to scalar values
        precomputed: dict of JAX arrays returned by precompute()

    Returns:
        Rrs_water: water-leaving remote sensing reflectance [sr-1], shape (n_wavelengths,)
    """
    p = {k: float(v) for k, v in params.items()}
    return _forward_core(p, precomputed)


def make_forward_vec(param_names, precomputed):
    """
    Return a function f(params_vec, aux=None) -> Rrs suitable for jax.jit, jax.jacobian, and jax.vmap.

    The returned function takes a 1-D JAX array of parameter values (in the order given by
    param_names) and returns the simulated water-leaving reflectance spectrum.

    The optional ``aux`` argument is a dict that overrides entries in ``precomputed`` on a
    per-call basis, enabling per-pixel variation of any precomputed spectral quantity.

    Example usage::

        pre    = precompute(wavelengths)
        names  = ["C_0", "C_Y", "C_ism", "offset", ...]
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

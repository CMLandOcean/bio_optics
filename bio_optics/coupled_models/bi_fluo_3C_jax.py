"""
JAX implementation of the full coupled HEREON model: water-leaving Rrs + fluorescence + surface reflectance.

Three-component model building on ``bi_3C_jax``:
  1. Water-leaving Rrs (via ``bi_jax._forward_core``)
  2. Phytoplankton fluorescence: Chl-a, phycocyanin, phycoerythrin (via ``fluorescence_jax``)
  3. Surface reflectance: sky/sun glint (via ``bi_3C_jax._forward_core``)

For the two-component version without fluorescence see ``bio_optics.coupled_models.bi_3C_jax``.

Two-layer architecture:
  - Layer 1: precompute() — numpy/scipy, runs once outside JIT, converts to JAX arrays
  - Layer 2: _forward_core() — pure JAX arithmetic, JIT-compilable, vmap-able

Usage::

    import numpy as np
    import jax
    from bio_optics.coupled_models import bi_fluo_3C_jax

    pre  = bi_fluo_3C_jax.precompute(wavelengths, theta_sun=np.radians(35))
    Rrs  = bi_fluo_3C_jax.forward(params, pre)

    names  = list(params.keys())
    f_vec  = bi_fluo_3C_jax.make_forward_vec(names, pre)
    Rrs    = jax.jit(f_vec)(params_vec)
    J      = jax.jacobian(f_vec)(params_vec)

References:
    [1] Bi et al. (2023): Bio-geo-optical modelling of natural waters [10.3389/fmars.2023.1196352]
    [2] Gege, P. (2021): The Water Colour Simulator WASI. User manual for WASI version 6.
"""

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from ..water import fluorescence as fluorescence_np
from ..water import fluorescence_jax
from . import bi_3C_jax


# ---------------------------------------------------------------------------
# Layer 1 — Precompute (numpy, runs once outside JIT)
# ---------------------------------------------------------------------------

def precompute(wavelengths,
               fresh=False,
               phy_source='a_phy_EnSAD',
               b_phy_source='b_phy_EnSAD',
               # Chlorophyll-a fluorescence shape
               fwhm_chl=25,
               lambda_C1=685,
               double_chl=False,
               W=0.75,
               fwhm_chl2=50,
               lambda_C2=730,
               # Phycocyanin fluorescence shape
               fwhm_phycocyanin=20,
               lambda_C_phycocyanin=644,
               # Phycoerythrin fluorescence shape
               fwhm_phycoerythrin=20,
               lambda_C_phycoerythrin=573,
               # Atmospheric scalars (passed through to bi_3C_jax.precompute)
               theta_sun=np.radians(30),
               P=1013.25,
               AM=5,
               RH=80,
               H_oz=0.381,
               WV=2.5,
               alpha=1.317,
               beta=0.2602,
               Ed_d_res=None,
               Ed_sa_res=None,
               Ed_sr_res=None,
               Ls_Ed=None):
    """
    Resample all static spectral lookup tables once and return as a dict of JAX arrays.

    Calls ``bi_3C_jax.precompute()`` for water and atmospheric tables, then adds
    the fluorescence Gaussian emission spectra.
    Must NOT be called inside a jax.jit context.

    Args:
        wavelengths: wavelengths [nm], numpy array of shape (n_wavelengths,)
        fresh: True for fresh water, False for oceanic water (controls bb_w), default: False
        phy_source: phytoplankton absorption library keyword or file path, default: 'a_phy_EnSAD'
        b_phy_source: phytoplankton scattering library keyword or file path, default: 'b_phy_EnSAD'
        fwhm_chl: FWHM of Chl-a fluorescence Gaussian [nm], default: 25
        lambda_C1: peak wavelength of Chl-a fluorescence [nm], default: 685
        double_chl: if True, use double-Gaussian Chl-a model, default: False
        W: weight of first Gaussian in double-peak Chl-a model, default: 0.75
        fwhm_chl2: FWHM of second Chl-a Gaussian [nm], default: 50
        lambda_C2: peak wavelength of second Chl-a Gaussian [nm], default: 730
        fwhm_phycocyanin: FWHM of phycocyanin fluorescence [nm], default: 20
        lambda_C_phycocyanin: peak wavelength of phycocyanin fluorescence [nm], default: 644
        fwhm_phycoerythrin: FWHM of phycoerythrin fluorescence [nm], default: 20
        lambda_C_phycoerythrin: peak wavelength of phycoerythrin fluorescence [nm], default: 573
        theta_sun .. Ls_Ed: atmospheric parameters, forwarded to ``bi_3C_jax.precompute()``.
            Pass theta_sun=None for Mode B (atmosphere / geometry retrieval via lmfit).

    Returns:
        precomputed: dict of JAX arrays. All keys from ``bi_3C_jax.precompute()`` plus:
            "h_C"               — Chl-a fluorescence emission spectrum [nm-1], shape (n_wavelengths,)
            "h_C_phycocyanin"   — phycocyanin emission spectrum [nm-1], shape (n_wavelengths,)
            "h_C_phycoerythrin" — phycoerythrin emission spectrum [nm-1], shape (n_wavelengths,)
    """
    pre = bi_3C_jax.precompute(
        wavelengths, fresh=fresh,
        phy_source=phy_source, b_phy_source=b_phy_source,
        theta_sun=theta_sun, P=P, AM=AM, RH=RH,
        H_oz=H_oz, WV=WV, alpha=alpha, beta=beta,
        Ed_d_res=Ed_d_res, Ed_sa_res=Ed_sa_res, Ed_sr_res=Ed_sr_res, Ls_Ed=Ls_Ed,
    )

    if double_chl:
        h_C_arr = fluorescence_np.h_C_double(
            W=W, wavelengths=wavelengths,
            fwhm1=fwhm_chl, fwhm2=fwhm_chl2,
            lambda_C1=lambda_C1, lambda_C2=lambda_C2,
        )
    else:
        h_C_arr = fluorescence_np.h_C(wavelengths=wavelengths, fwhm=fwhm_chl, lambda_C=lambda_C1)

    h_C_phycocyanin_arr = fluorescence_np.h_C(
        wavelengths=wavelengths, fwhm=fwhm_phycocyanin, lambda_C=lambda_C_phycocyanin
    )
    h_C_phycoerythrin_arr = fluorescence_np.h_C(
        wavelengths=wavelengths, fwhm=fwhm_phycoerythrin, lambda_C=lambda_C_phycoerythrin
    )

    return {
        **pre,
        "h_C":               jnp.array(h_C_arr),
        "h_C_phycocyanin":   jnp.array(h_C_phycocyanin_arr),
        "h_C_phycoerythrin": jnp.array(h_C_phycoerythrin_arr),
    }


# ---------------------------------------------------------------------------
# Layer 2 — Core forward model (jnp, JIT-compilable)
# ---------------------------------------------------------------------------

def _forward_core(p, pre):
    """
    Core forward simulation — pure JAX arithmetic, JIT-compilable.

    Computes water-leaving Rrs + fluorescence + surface reflectance.
    Fluorescence is additive and conditionally applied via jnp.where:
      - Chl-a fluorescence when total C_phy > 0.1 mg/m³ (empirical threshold from Bi et al.)
      - Phycocyanin fluorescence when C_3 > 0.1 mg/m³ (cyanobacteria blue)
      - Phycoerythrin fluorescence when C_4 > 0.1 mg/m³ (cyanobacteria red)
    Note: jnp.where always evaluates both branches; the threshold zeros the result,
    it does not skip computation.

    Args:
        p: dict of scalar parameters (floats or JAX 0-d arrays). All keys required by
           ``bi_3C_jax._forward_core`` plus:
               Fluorescence amplitudes: L_fl_lambda0, L_fl_phycocyanin, L_fl_phycoerythrin
        pre: dict of precomputed JAX arrays from precompute()

    Returns:
        Rrs: above-water remote sensing reflectance [sr-1], shape (n_wavelengths,)
    """
    # Water + surface (from bi_3C_jax)
    Rrs = bi_3C_jax._forward_core(p, pre)

    # Total phytoplankton concentration for fluorescence threshold
    C_phy = jnp.sum(jnp.array([p["C_0"], p["C_1"], p["C_2"], p["C_3"],
                                p["C_4"], p["C_5"], p["C_6"], p["C_7"]]))

    # Fluorescence (additive)
    Rrs = Rrs + jnp.where(
        C_phy > 0.1, fluorescence_jax.Rrs_fl(p["L_fl_lambda0"], pre["h_C"]), 0.0
    )
    Rrs = Rrs + jnp.where(
        p["C_3"] > 0.1, fluorescence_jax.Rrs_fl_phycocyanin(p["L_fl_phycocyanin"], pre["h_C_phycocyanin"]), 0.0
    )
    Rrs = Rrs + jnp.where(
        p["C_4"] > 0.1, fluorescence_jax.Rrs_fl_phycoerythrin(p["L_fl_phycoerythrin"], pre["h_C_phycoerythrin"]), 0.0
    )

    return Rrs


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def forward(params, precomputed):
    """
    Forward simulation: water-leaving Rrs + fluorescence + surface reflectance.

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

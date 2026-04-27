"""
JAX implementation of the coupled HEREON model (Bi et al. 2023): water-leaving Rrs + surface reflectance.

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
"""

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from ..atmosphere import downwelling_irradiance
from ..surface import air_water_jax
from ..water import fluorescence as fluorescence_np
from ..water.backscattering import morel
from ..helper import resampling


# ---------------------------------------------------------------------------
# Layer 1 — Precompute (numpy, runs once outside JIT)
# ---------------------------------------------------------------------------

def precompute(wavelengths,
               fresh=False,
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

    Must NOT be called inside a jax.jit context.

    Args:
        wavelengths: wavelengths [nm], numpy array of shape (n_wavelengths,)
        fresh: True for fresh water, False for oceanic water (controls bb_w), default: False
        fwhm_chl: FWHM of chlorophyll-a fluorescence Gaussian [nm], default: 25
        lambda_C1: peak wavelength of Chl-a fluorescence [nm], default: 685
        double_chl: if True, use double-Gaussian Chl-a fluorescence model, default: False
        W: weight of first Chl-a Gaussian in double-peak model, default: 0.75
        fwhm_chl2: FWHM of second Chl-a Gaussian [nm], default: 50
        lambda_C2: peak wavelength of second Chl-a Gaussian [nm], default: 730
        fwhm_phycocyanin: FWHM of phycocyanin fluorescence [nm], default: 20
        lambda_C_phycocyanin: peak wavelength of phycocyanin fluorescence [nm], default: 644
        fwhm_phycoerythrin: FWHM of phycoerythrin fluorescence [nm], default: 20
        lambda_C_phycoerythrin: peak wavelength of phycoerythrin fluorescence [nm], default: 573
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
        Ed_d_res: optional precomputed direct downwelling irradiance; skips computation if provided
        Ed_sa_res: optional precomputed aerosol-scattered downwelling irradiance; skips computation if provided
        Ed_sr_res: optional precomputed Rayleigh-scattered downwelling irradiance; skips computation if provided
        Ls_Ed: optional ratio of sky radiance to downwelling irradiance [sr-1], shape (n_wavelengths,)

    Returns:
        precomputed: dict of JAX arrays with keys:
            "wavelengths"          — wavelengths [nm], shape (n_wavelengths,)
            "a_w"                  — pure water absorption [m-1], shape (n_wavelengths,)
            "da_w_div_dT"          — temperature gradient of pure water absorption [m-1 K-1], shape (n_wavelengths,)
            "a_i_spec_EnSAD"       — specific phytoplankton absorption (8 types) [m2 mg-1], shape (n_wavelengths, 8)
            "b_i_spec_EnSAD"       — specific phytoplankton scattering (8 types) [m2 mg-1], shape (n_wavelengths, 8)
            "bb_w"                 — pure water backscattering [m-1], shape (n_wavelengths,)
            "h_C"                  — Chl-a fluorescence emission spectrum [nm-1], shape (n_wavelengths,)
            "h_C_phycocyanin"      — phycocyanin fluorescence emission spectrum [nm-1], shape (n_wavelengths,)
            "h_C_phycoerythrin"    — phycoerythrin fluorescence emission spectrum [nm-1], shape (n_wavelengths,)
            "Ed_d"                 — direct downwelling irradiance [W m-2 nm-1], shape (n_wavelengths,)
            "Ed_sr"                — Rayleigh-scattered downwelling irradiance [W m-2 nm-1], shape (n_wavelengths,)
            "Ed_sa"                — aerosol-scattered downwelling irradiance [W m-2 nm-1], shape (n_wavelengths,)
            "Ls_Ed"                — sky radiance / downwelling irradiance ratio [sr-1], shape (n_wavelengths,)
    """
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

    Ls_Ed_arr = (np.array(Ls_Ed) if len(Ls_Ed) > 0
                 else np.zeros(len(wavelengths)))

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
        "wavelengths":        jnp.array(wavelengths),
        "a_w":                jnp.array(resampling.resample_a_w(wavelengths)),
        "da_w_div_dT":        jnp.array(resampling.resample_da_w_div_dT(wavelengths)),
        "a_i_spec_EnSAD":     jnp.array(resampling.resample_a_i_spec_EnSAD(wavelengths)),
        "b_i_spec_EnSAD":     jnp.array(resampling.resample_b_i_spec_EnSAD(wavelengths)),
        "bb_w":               jnp.array(morel(wavelengths=wavelengths, fresh=fresh)),
        "h_C":                jnp.array(h_C_arr),
        "h_C_phycocyanin":    jnp.array(h_C_phycocyanin_arr),
        "h_C_phycoerythrin":  jnp.array(h_C_phycoerythrin_arr),
        "Ed_d":               jnp.array(Ed_d),
        "Ed_sr":              jnp.array(Ed_sr),
        "Ed_sa":              jnp.array(Ed_sa),
        "Ls_Ed":              jnp.array(Ls_Ed_arr),
    }


# ---------------------------------------------------------------------------
# Layer 2 — Core forward model (jnp, JIT-compilable)
# ---------------------------------------------------------------------------

def _forward_core(p, pre):
    """
    Core forward simulation — pure JAX arithmetic, JIT-compilable.

    Implements the HEREON bio-geo-optical model (Bi et al. 2023) [1] for water-leaving
    Rrs combined with sky/sun glint surface reflectance (Gege 2021) [2].

    [1] Bi et al. (2023): Bio-geo-optical modelling of natural waters [10.3389/fmars.2023.1196352]
    [2] Gege, P. (2021): The Water Colour Simulator WASI. User manual for WASI version 6.

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
            Fluorescence amplitudes:              L_fl_lambda0, L_fl_phycocyanin, L_fl_phycoerythrin
            Spectral offset:                      offset
            Surface / viewing geometry:           theta_view, n1, n2, fd_d, fd_s, g_dd, g_dsr, g_dsa, d_r
        pre: dict of precomputed JAX arrays from precompute()

    Returns:
        Rrs: above-water remote sensing reflectance [sr-1], shape (n_wavelengths,)
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

    # Phytoplankton (EnSAD 8-type spectra), raw dot product then packaging correction
    a_phy_raw = jnp.dot(pre["a_i_spec_EnSAD"], C_i)
    a_phy_at_ref = jnp.interp(p["lambda_0_phy"], wl, a_phy_raw)
    E_phy = jnp.where(C_phy <= 1.0, p["E0"], p["E1"])
    # When C_phy = 0 the raw spectrum is zero; skip correction to avoid 0/0
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

    # Phytoplankton (EnSAD 8-type spectra, per-type backscattering ratios)
    bb_ratio_C_i = jnp.array([p["b_ratio_C_0"], p["b_ratio_C_1"], p["b_ratio_C_2"], p["b_ratio_C_3"],
                               p["b_ratio_C_4"], p["b_ratio_C_5"], p["b_ratio_C_6"], p["b_ratio_C_7"]])
    bb_phy = jnp.dot(pre["b_i_spec_EnSAD"], C_i * bb_ratio_C_i)

    # Detrital backscattering
    bb_md = p["b_ratio_md"] * b_md
    bb_bd = p["b_ratio_bd"] * b_bd

    # Total particulate and total backscattering
    bb_p = bb_phy + bb_md + bb_bd
    bb   = bb_w + bb_p

    # --- Water-leaving Rrs: Lee et al. (2011) deep-water formula ---
    # Rrs_deep already includes the air–water transmission (Lee 1998 parameterisation)
    k = a + bb
    Rrs_water = (
        (p["Gw0"] + p["Gw1"] * bb_w / k) * bb_w / k
        + (p["Gp0"] + p["Gp1"] * bb_p / k) * bb_p / k
    )

    # --- Fluorescence ---
    Rrs_water = Rrs_water + jnp.where(
        C_phy > 0.1, p["L_fl_lambda0"] * pre["h_C"], 0.0
    )
    Rrs_water = Rrs_water + jnp.where(
        p["C_3"] > 0.1, p["L_fl_phycocyanin"] * pre["h_C_phycocyanin"], 0.0
    )
    Rrs_water = Rrs_water + jnp.where(
        p["C_4"] > 0.1, p["L_fl_phycoerythrin"] * pre["h_C_phycoerythrin"], 0.0
    )

    # Spectral offset and water-leaving total
    Rrs_water = Rrs_water + p["offset"]

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

        # JIT-compiled single-pixel forward
        Rrs = jax.jit(f_vec)(params_vec)

        # Jacobian w.r.t. all fitted parameters
        J = jax.jacobian(f_vec)(params_vec)   # shape (n_wavelengths, n_params)

        # Vectorized over many pixels
        Rrs_batch = jax.vmap(f_vec)(params_matrix)  # shape (n_pixels, n_wavelengths)

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

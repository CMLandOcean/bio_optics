import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from ..water import absorption_jax, backscattering_jax, attenuation_jax, bottom_reflectance_jax
from ..water.backscattering import morel
from ..surface import air_water_jax
from ..helper import resampling


# ---------------------------------------------------------------------------
# Helper functions (mirrors of albert_mobley.py local functions)
# ---------------------------------------------------------------------------

def f_rs(omega_b, cos_t_sun_p, cos_t_view_p):
    """
    Irradiance-to-reflectance conversion factor after Albert & Mobley (2003) [1].

    [1] Albert & Mobley (2003): An analytical model for subsurface irradiance and remote sensing reflectance
        in deep and shallow case-2 waters. [10.1364/OE.11.002873]

    Args:
        omega_b: single scattering albedo bb / (a + bb) [dimensionless]
        cos_t_sun_p: cosine of the refracted sun zenith angle in water
        cos_t_view_p: cosine of the refracted viewing angle in water

    Returns:
        f_rs: irradiance-to-reflectance conversion factor [sr-1]
    """
    return (0.0512
            * (1 + omega_b * (4.6659 + omega_b * (-7.8387 + omega_b * 5.4571)))
            * (1 + 0.1098 / cos_t_sun_p)
            * (1 + 0.4021 / cos_t_view_p))


def rrs_deep(frs, omega_b):
    """
    Subsurface radiance reflectance of optically deep water after Albert & Mobley (2003) [1].

    [1] Albert & Mobley (2003): An analytical model for subsurface irradiance and remote sensing reflectance
        in deep and shallow case-2 waters. [10.1364/OE.11.002873]

    Args:
        frs: irradiance-to-reflectance conversion factor [sr-1]
        omega_b: single scattering albedo [dimensionless]

    Returns:
        rrs_deep: subsurface radiance reflectance of deep water [sr-1]
    """
    return frs * omega_b


def rrs_shallow(rrsd, Kd, ku_w, zB, Rrs_b, ku_b, A_rs1=1.1576, A_rs2=1.0389):
    """
    Subsurface radiance reflectance of optically shallow water after Albert & Mobley (2003) [1].

    [1] Albert & Mobley (2003): An analytical model for subsurface irradiance and remote sensing reflectance
        in deep and shallow case-2 waters. [10.1364/OE.11.002873]

    Args:
        rrsd: subsurface radiance reflectance of optically deep water [sr-1]
        Kd: downwelling diffuse attenuation coefficient [m-1]
        ku_w: upwelling attenuation coefficient for the water column [m-1]
        zB: water depth [m]
        Rrs_b: bottom reflectance [sr-1]
        ku_b: upwelling attenuation coefficient for the bottom [m-1]
        A_rs1: empirical constant, default: 1.1576
        A_rs2: empirical constant, default: 1.0389

    Returns:
        rrs_shallow: subsurface radiance reflectance of shallow water [sr-1]
    """
    return (rrsd * (1 - A_rs1 * jnp.exp(-(Kd + ku_w) * zB))
            + A_rs2 * Rrs_b * jnp.exp(-(Kd + ku_b) * zB))


# ---------------------------------------------------------------------------
# Layer 1 — Precompute (numpy, runs once outside JIT)
# ---------------------------------------------------------------------------

def precompute(wavelengths, fresh=False, b_X_norm_factor=1.0):
    """
    Resample all static spectral lookup tables once and return as a dict of JAX arrays.

    This function uses numpy/pandas resampling internally and must NOT be called inside
    a jax.jit context. Call it once before inversion and pass the result to forward().

    Args:
        wavelengths: wavelengths to resample to [nm], numpy array of shape (n_wavelengths,)
        fresh: True for fresh water, False for oceanic water (controls pure water backscattering), default: False
        b_X_norm_factor: spectrally flat normalization factor for type I particle scattering [dimensionless], default: 1.0

    Returns:
        precomputed: dict of JAX arrays with keys:
            "wavelengths"   — wavelengths [nm], shape (n_wavelengths,)
            "a_w"           — pure water absorption [m-1], shape (n_wavelengths,)
            "da_w_div_dT"   — temperature gradient of pure water absorption [m-1 degrees_C-1], shape (n_wavelengths,)
            "a_i_spec"      — specific phytoplankton absorption [m2 mg-1], shape (n_wavelengths, 6)
            "b_phy_norm"    — normalized phytoplankton backscattering [dimensionless], shape (n_wavelengths,)
            "bb_w"          — pure water backscattering [m-1], shape (n_wavelengths,)
            "b_X_norm"      — normalized type I particle scattering [dimensionless], shape (n_wavelengths,)
            "R_b_i"         — bottom albedo spectra [dimensionless], shape (n_wavelengths, 6)
    """
    return {
        "wavelengths":  jnp.array(wavelengths),
        "a_w":          jnp.array(resampling.resample_a_w(wavelengths)),
        "da_w_div_dT":  jnp.array(resampling.resample_da_w_div_dT(wavelengths)),
        "a_i_spec":     jnp.array(resampling.resample_a_i_spec(wavelengths)),
        "b_phy_norm":   jnp.array(resampling.resample_b_phy_norm(wavelengths)),
        "bb_w":         jnp.array(morel(wavelengths=wavelengths, fresh=fresh)),
        "b_X_norm":     jnp.ones(len(wavelengths)) * b_X_norm_factor,
        "R_b_i":        jnp.array(resampling.resample_R_b_i(wavelengths)),
    }


# ---------------------------------------------------------------------------
# Layer 2 — Core forward model (jnp, JIT-compilable)
# ---------------------------------------------------------------------------

def _forward_core(p, pre):
    """
    Core forward simulation — pure JAX arithmetic, JIT-compilable.

    Args:
        p: dict of scalar parameters (floats or JAX 0-d arrays)
        pre: dict of precomputed JAX arrays from precompute()

    Returns:
        Rrs: above-water remote sensing reflectance [sr-1], shape (n_wavelengths,)
    """
    wl = pre["wavelengths"]

    # Refraction angles
    ctsp = jnp.cos(air_water_jax.snell(p["theta_sun"],  n1=p["n1"], n2=p["n2"]))
    ctvp = jnp.cos(air_water_jax.snell(p["theta_view"], n1=p["n1"], n2=p["n2"]))

    # Total absorption
    a_sim = absorption_jax.a(
        C_0=p["C_0"], C_1=p["C_1"], C_2=p["C_2"],
        C_3=p["C_3"], C_4=p["C_4"], C_5=p["C_5"],
        C_Y=p["C_Y"], C_X=p["C_X"], C_Mie=p["C_Mie"],
        S=p["S"], S_NAP=p["S_NAP"],
        lambda_0=p["lambda_0"], K=p["K"],
        T_W=p["T_W"], T_W_0=p["T_W_0"],
        a_NAP_spec_lambda_0=p["a_NAP_spec_lambda_0"],
        wavelengths=wl,
        a_w=pre["a_w"],
        da_w_div_dT=pre["da_w_div_dT"],
        a_i_spec=pre["a_i_spec"],
    )

    # Total backscattering
    C_phy = p["C_0"] + p["C_1"] + p["C_2"] + p["C_3"] + p["C_4"] + p["C_5"]
    bb_sim = backscattering_jax.bb(
        C_X=p["C_X"], C_Mie=p["C_Mie"], C_phy=C_phy,
        bb_phy_spec=p["bb_phy_spec"],
        bb_Mie_spec=p["bb_Mie_spec"],
        bb_X_spec=p["bb_X_spec"],
        lambda_S=p["lambda_S"], n=p["n"],
        wavelengths=wl,
        bb_w=pre["bb_w"],
        b_phy_norm=pre["b_phy_norm"],
        b_X_norm=pre["b_X_norm"],
    )

    # Bottom reflectance
    f_i = jnp.array([p["f_0"], p["f_1"], p["f_2"], p["f_3"], p["f_4"], p["f_5"]])
    B_i = jnp.array([p["B_0"], p["B_1"], p["B_2"], p["B_3"], p["B_4"], p["B_5"]])
    Rrs_b = bottom_reflectance_jax.Rrs_b(f_i, B_i, pre["R_b_i"])

    # Attenuation
    ob   = attenuation_jax.omega_b(a_sim, bb_sim)
    frs  = f_rs(ob, ctsp, ctvp)
    rrsd = rrs_deep(frs, ob)
    Kd   = attenuation_jax.Kd(a_sim, bb_sim, ctsp, p["kappa_0"])
    kuW  = attenuation_jax.ku_w(a_sim, bb_sim, ob, ctsp, ctvp)
    kuB  = attenuation_jax.ku_b(a_sim, bb_sim, ob, ctsp, ctvp)

    return air_water_jax.below2above(rrs_shallow(rrsd, Kd, kuW, p["zB"], Rrs_b, kuB))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def forward(params, precomputed):
    """
    Forward simulation of water-leaving remote sensing reflectance after Albert & Mobley (2003) [1].

    [1] Albert & Mobley (2003): An analytical model for subsurface irradiance and remote sensing reflectance
        in deep and shallow case-2 waters. [10.1364/OE.11.002873]

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
    Return a function f(params_vec) -> Rrs suitable for jax.jit, jax.jacobian, and jax.vmap.

    The returned function takes a 1-D JAX array of parameter values (in the order given by
    param_names) and returns the simulated above-water reflectance spectrum.

    Example usage::

        pre = precompute(wavelengths)
        names = ["C_0", "C_Y", "C_X", "C_Mie", "zB", "f_0", ...]
        f_vec = make_forward_vec(names, pre)

        # JIT-compiled single-pixel forward
        Rrs = jax.jit(f_vec)(params_vec)

        # Jacobian w.r.t. all fitted parameters
        J = jax.jacobian(f_vec)(params_vec)   # shape (n_wavelengths, n_params)

        # Vectorized over many pixels
        Rrs_batch = jax.vmap(f_vec)(params_matrix)   # params_matrix shape (n_pixels, n_params)

    Args:
        param_names: ordered list of parameter name strings matching the columns of params_vec
        precomputed: dict of JAX arrays returned by precompute()

    Returns:
        f: callable f(params_vec) -> Rrs where params_vec has shape (len(param_names),)
    """
    def f(params_vec):
        p = dict(zip(param_names, params_vec))
        return _forward_core(p, precomputed)
    return f

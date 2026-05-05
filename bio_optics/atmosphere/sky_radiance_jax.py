"""
JAX implementation of sky radiance model (Gege 2021 / WASI).

Component module — no precompute / make_forward_vec.  Pure JAX arithmetic;
inputs are precomputed irradiance spectral arrays from
``downwelling_irradiance_jax.precompute()``.

Reference:
    Gege, P. (2021): The Water Colour Simulator WASI. User manual for WASI version 6.
"""

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


def L_s(fd_d, g_dd,  Ed_d,
        fd_s, g_dsr, Ed_sr,
        g_dsa, Ed_sa):
    """
    Sky radiance as a weighted sum of three downwelling irradiance components.

    The parameters g_dd, g_dsr, g_dsa are the intensities [sr-1] of the direct,
    Rayleigh-scattered and aerosol-scattered components respectively (Gege 2021).

    Args:
        fd_d: fractional contribution of direct irradiance [dimensionless]
        g_dd: intensity of direct solar irradiance [sr-1]
        Ed_d: direct downwelling irradiance [W m-2 nm-1], shape (n_wavelengths,)
        fd_s: fractional contribution of diffuse irradiance [dimensionless]
        g_dsr: intensity of Rayleigh-scattered irradiance [sr-1]
        Ed_sr: Rayleigh-scattered downwelling irradiance [W m-2 nm-1], shape (n_wavelengths,)
        g_dsa: intensity of aerosol-scattered irradiance [sr-1]
        Ed_sa: aerosol-scattered downwelling irradiance [W m-2 nm-1], shape (n_wavelengths,)

    Returns:
        L_s: sky radiance [W m-2 nm-1 sr-1], shape (n_wavelengths,)
    """
    return fd_d * (g_dd * Ed_d) + fd_s * (g_dsr * Ed_sr + g_dsa * Ed_sa)

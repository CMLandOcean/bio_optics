import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


def Rrs_b(f_i, B_i, R_b_i):
    """
    Radiance reflectance of benthic substrate [sr-1] as a mixture of up to 6 bottom types [1].

    [1] Gege, P. (2021): The Water Colour Simulator WASI. User manual for WASI version 6.

    Args:
        f_i: fractional cover of each bottom type [dimensionless], shape (6,)
        B_i: proportion of radiation reflected towards sensor from each bottom type [sr-1], shape (6,)
        R_b_i: precomputed bottom albedo spectra [dimensionless], shape (n_wavelengths, 6)

    Returns:
        Rrs_b: radiance reflectance of benthic substrate [sr-1], shape (n_wavelengths,)
    """
    return jnp.dot(R_b_i, f_i * B_i)

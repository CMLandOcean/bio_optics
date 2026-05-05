"""
JAX implementation of Rayleigh scattering spectrum.

Component module — no precompute / make_forward_vec.  Pure JAX arithmetic.
"""

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


def b_rayleigh(wavelengths, lambda_r=400.0, b_r_spec=1.0, n_r=-4.0):
    """
    Rayleigh scattering spectrum as a power law.

    Args:
        wavelengths: wavelengths [nm], shape (n_wavelengths,)
        lambda_r: reference wavelength [nm], default: 400
        b_r_spec: specific scattering intensity at lambda_r [dimensionless], default: 1
        n_r: wavelength exponent, default: -4

    Returns:
        b_rayleigh: Rayleigh scattering spectrum [dimensionless], shape (n_wavelengths,)
    """
    return b_r_spec * (wavelengths / lambda_r) ** n_r

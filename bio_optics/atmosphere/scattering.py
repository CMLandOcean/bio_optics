import numpy as np


def b_rayleigh(wavelengths=np.arange(400, 800),
               lambda_r=400,
               b_r_spec=1,
               n_r=-4):
    """
    Rayleigh scattering.

    Args:
        wavelengths: wavelengths to compute Rayleigh scattering for [nm]. Defaults to np.arange(400, 800).
        lambda_r: reference wavelength for normalization [nm]. Defaults to 400.
        b_r_spec: specific intensity at reference wavelength. Defaults to 1.
        n_r: exponent for Rayleigh scattering. Defaults to -4.

    Returns:
        Rayleigh scattering spectrum.
    """
    return b_r_spec * ((wavelengths / lambda_r) ** n_r)

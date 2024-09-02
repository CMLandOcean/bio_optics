import numpy as np


def b_rayleigh(wavelengths = np.arange(400,800),
               lambda_r = 400,
               b_r_spec = 1,
               n_r = -4):
    """
    Rayleigh scattering.

    Args:
        wavelengths (_type_, optional): _description_. Defaults to np.arange(400,800).
        lambda_r (int, optional): Reference wavelength for normalization [nm]. Defaults to 400.
        b_r_spec (int, optional): Specific intensity at reference wavelength. Defaults to 1.
        n_r (int, optional): Exponent for Rayleigh scattering. Defaults to -4.

    Returns:
        b_rayleigh: Rayleigh scattering spectrum.
    """
    b_rayleigh = b_r_spec * ((wavelengths/lambda_r)**n_r)

    return b_rayleigh


def R_rs_adjacency(C_adj=0.1,
                   wavelengths = np.arange(400,800),
                   lambda_r = 400,
                   b_r_spec = 1,
                   n_r = -4,
                   R_bg=[],
                   b_ray=[]):
    """_summary_

    Args:
        C_adj (float, optional): Scalar for adjacency signal. Defaults to 0.1.
        wavelengths (np.array, optional): wavelengths to compute adjacency effect. Defaults to np.arange(400,800).
        lambda_r (int, optional): Reference wavelength for normalization [nm]. Defaults to 400.
        b_r_spec (int, optional): Specific intensity at reference wavelength. Defaults to 1.
        n (int, optional): Exponent for Rayleigh scattering. Defaults to -4.
        R_bg (np.array, optional): Background reflectance spectrum. Defaults to None.
        b_ray (np.array, optional): Rayleigh scattering spectrum.

    Returns:
        R_rs_adjacency: Estimate of adjacency radiance reflectance spectrum.
    """
    # if no R_bg is provided set to zero, i.e. no adjacency effect
    if len(R_bg)==0:
        R_bg = np.zeros(len(wavelengths))

    if len(b_ray)==0:
        b_ray = b_rayleigh(wavelengths=wavelengths, lambda_r=lambda_r, b_r_spec=b_r_spec, n_r=n_r)
    
    R_rs_adjacency = C_adj * b_ray * R_bg
    
    return R_rs_adjacency
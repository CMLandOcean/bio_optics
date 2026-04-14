import numpy as np
from ..helper.utils import find_closest
from ..surface.air_water import above2below


def stumpf(R_w,
           wavelengths,
           lambda1 = 466,
           lambda2 = 536,
           m1 = 1,
           m0 = 0,
           n = 1,
           normalized = True):
    """
    Relative water depth following Stumpf et al. (2003) [1]. 
    
    [1] Stumpf et al. (2003): Determination of water depth with high-resolution satellite imagery over variable bottom types [10.4319/lo.2003.48.1_part_2.0547]
    
    Args:
        R_w: water-leaving reflectance [-] spectrum
        wavelengths: corresponding wavelengths [nm]
        lambda1: blue band wavelength [nm] (numerator), default: 466
        lambda2: green band wavelength [nm] (denominator), default: 536
        m1: scaling constant to map log-ratio to depth, default: 1
        m0: offset constant for zero depth, default: 0
        n: fixed value ensuring a positive logarithm and linear depth response, default: 1
        normalized: if True, normalize output to 0..1 range, default: True

    Returns:
        zB: relative bathymetry [dimensionless if normalized, else in depth units]
    """

    band1 = R_w[find_closest(wavelengths, lambda1)[1]]
    band2 = R_w[find_closest(wavelengths, lambda2)[1]]
    
    # Eq. 9
    zB = m1 * (np.log(n * band1) / np.log(n * band2)) - m0
    
    if normalized:
        zB *= 1/zB.max()
    
    return zB


def li(Rrs, wavelengths, lambda1=466, lambda2=536, lambda3=652, chl_a=None, n=1000, normalized=False):
    """
    Adaptive bathymetry estimation for shallow coastal chl-a dominated waters (Case-I waters) following Li et al. (2019) [1].
     
    [1] Li et al. (2019): Adaptive bathymetry estimation for shallow coastal waters using Planet Dove satellites [10.1016/j.rse.2019.111302]
            
    Args:
        Rrs: remote sensing reflectance [sr-1] spectrum
        wavelengths: corresponding wavelengths [nm]
        lambda1: blue band wavelength [nm] for attenuation index and depth estimation, default: 466
        lambda2: green band wavelength [nm] for attenuation index and depth estimation, default: 536
        lambda3: red band wavelength [nm] for light attenuation index, default: 652
        chl_a: chl-a concentration [mg m-3]; if None, estimated from Rrs, default: None
        n: multiplicative factor in Stumpf log-ratio (ensures positive log), default: 1000
        normalized: if True, normalize output to 0..1 range, default: False

    Returns:
        zB: water depth estimate [m]
    """
    band1 = Rrs[find_closest(wavelengths,lambda1)[1]]
    band2 = Rrs[find_closest(wavelengths,lambda2)[1]]
    band3 = Rrs[find_closest(wavelengths,lambda3)[1]]

    omega = band2 - 0.46 * band3 - 0.54 * band1

    if chl_a == None:
        chl_a = 10**(-0.4909 + 191.659 * omega)

    # Note that m0 and m1 are switched in [1] compared to stumpf()
    m0 = 50.156 * np.exp(0.957 * chl_a)
    m1 = 52.083 * np.exp(0.957 * chl_a)

    # Note that depth is computed with subsurface r_rs in [1] instead of R_w as in stumpf()
    zB = stumpf(above2below(Rrs), wavelengths=wavelengths, m1=m1, m0=m0, n=n, normalized=normalized)

    return zB
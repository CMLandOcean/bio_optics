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
    Relative water depth [1] 
    
    [1] Stumpf et al. (2003): Determination of water depth with high-resolution satellite imagery over variable bottom types [10.4319/lo.2003.48.1_part_2.0547]
    
    :param R_w: water-leaving reflectance [-] spectrum
    :param wavelengths: corresponding wavelengths [nm]
    :param lambda1: wavelength of first band used for depth estimation (usually blue; default: 466 nm for GAO data based in comparison with ENVI Relative Water Depth)
    :param lambda2: index of the second band used for depth estimation (usually green; default: 536 nm for GAO data based in comparison with ENVI Relative Water Depth)
    :param m1: tunable constant to scale the ratio to depth (default: 1)
    :param m0: tunable constant to set offset for a depth of 0 m (default: 0)
    :param n: fixed value chosen to assure both that the logarithm will be positive under any condition and that the ratio will produce a linear response with depth (default: 1)
    :param normalied: boolean to decide if output should be normalized to the range 0..1 (default: True).
    :return: relative bathymetry
    """

    band1 = R_w[find_closest(wavelengths, lambda1)[1]]
    band2 = R_w[find_closest(wavelengths, lambda2)[1]]
    
    # Eq. 9
    Z = m1 * (np.log(n * band1) / np.log(n * band2)) - m0
    
    if normalized:
        Z *= 1/Z.max()
    
    return Z


def li(R_rs, wavelengths, lambda1=466, lambda2=536, lambda3=652, chl_a=None, n=1000, normalized=False):
    """
    Adaptive bathymetry estimation for shallow coastal chl-a dominated waters (Case-I waters) using Planet Dove satellites
     
    [1] Li et al. (2019): Adaptive bathymetry estimation for shallow coastal waters using Planet Dove satellites [10.1016/j.rse.2019.111302]
            
    :param R_rs: remote-sensing reflectance [sr-1] spectrum
    :param wavelengths: corresponding wavelengths [nm]
    :param lambda1: wavelength [nm] of blue band used for light attenuation index (omega) and depth estimation (default: 466 for GAO data)
    :param lambda2: wavelength [nm] of green band used for light attenuation index (omega) and depth estimation (default: 536 for GAO data)
    :param lambda3: wavelength [nm] of red band used for light attenuation index (omega) (default: 652 for GAO data) 
    :param chl_a: concentration of chl-a [mg m-3] (default: None, computed from r_rs)
    :param n: factor in stumpf() (default: 1000, according to [1]])
    :param normalied: boolean to decide if output should be normalized to the range 0..1 (default: False).
    :return: bathymetry
    """
    band1 = R_rs[find_closest(wavelengths,lambda1)[1]]
    band2 = R_rs[find_closest(wavelengths,lambda2)[1]]
    band3 = R_rs[find_closest(wavelengths,lambda3)[1]]

    omega = band2 - 0.46 * band3 - 0.54 * band1

    if chl_a == None:
        chl_a = 10**(-0.4909 + 191.659 * omega)

    # Note that m0 and m1 are switched in [1] compared to stumpf()
    m0 = 50.156 * np.exp(0.957 * chl_a)
    m1 = 52.083 * np.exp(0.957 * chl_a)

    # Note that depth is computed with subsurface r_rs in [1] instead of R_w as in stumpf()
    return stumpf(above2below(R_rs), wavelengths=wavelengths, m1=m1, m0=m0, n=n, normalized=normalized)

    
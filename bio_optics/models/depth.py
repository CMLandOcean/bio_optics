import numpy as np
from .. helper.utils import find_closest


def stumpf(R_w,
           wavelengths,
           lambda1: int = 466,
           lambda2: int = 536,
           m1: float = 1,
           m0: float = 0,
           n: float = 1,
           normalized: bool = True):
    """
    Relative water depth [1] 
    
    [1] Stumpf et al. (2003): Determination of water depth with high-resolution satellite imagery over variable bottom types [10.4319/lo.2003.48.1_part_2.0547].
    
    :param R_w: water-leaving reflectance [-] spectrum, must include 'wavelength' attribute
    :param lambda1: wavelength of first band used for depth estimation (usually blue; default: 466 nm for GAO data based in comparison with ENVI Relative Water Depth)
    :param lambda2: index of the second band used for depth estimation (usually green; default: 536 nm for GAO data based in comparison with ENVI Relative Water Depth)
    :param m1: tunable constant to scale the ratio to depth (default: 1)
    :param m0: tunable constant to set offset for a depth of 0 m (default: 0)
    :param n: fixed value chosen to assure both that the logarithm will be positive under any condition and that the ratio will produce a linear response with depth (default: 1)
    :param normalied: boolean to decide if output should be normalized to the range 0..1 (default: True).
    :return: xarray.DataArray of relative bathymetry
    """

    band1 = R_w[find_closest(wavelengths, lambda1)]
    band2 = R_w[find_closest(wavelengths, lambda2)]
    
    # Eq. 9
    Z = m1 * (np.log(n * band1) / np.log(n * band2)) - m0
    
    if normalized:
        Z *= 1/Z.max()
    
    return Z
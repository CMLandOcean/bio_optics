import numpy as np
from .. helper.utils import find_closest


def petus(R_rs, wavelengths, a=26110, b=604.5, c=0.24, lambda0=860):
    """
    Empirical turbidity model for MODIS [1] 

    [1] Petus et al. (2010): Estimating turbidity and total suspended matter in the Adour River plume (South Bay of Biscay) using MODIS 250-m imagery [10.1016/j.csr.2009.12.007]

    Args:
        R_rs: remote sensing reflectance [sr-1] spectrum
        wavelengths: corresponding wavelengths [nm]
        a: Defaults to 26110.
        b: Defaults to 604.5.
        c: Defaults to 0.24.
        lambda0: wavelength for turbidity estimation. Defaults to 860.
    Returns: turbidity [NTU]
    """
    x = R_rs[find_closest(wavelengths, 860)[1]]
    return a*x**2 + b*x + c


def chen(R_rs, wavelengths, a=1203.9, b=1.087, lambda0=645):
    """
    Empirical turbidity algorithm for MODIS, valid for 0.9 < turbidity < 8.0 [1].

    [1] Chen et al. (2007): Monitoring turbidity in Tampa Bay using MODIS/Aqua 250-m imagery [10.1016/j.rse.2006.12.019]

    Args:
        R_rs (_type_): remote sensing reflectance [sr-1] spectrum
        wavelengths (_type_): correspondong wavelengths [nm]
        a (float, optional): multiplier. Defaults to 1203.9.
        b (float, optional): exponent. Defaults to 1.087.
        lambda0 (int, optional): wavelength for turbidity estimation. Defaults to 645.
    Returns: turbidity [NTU]
    """
    return a * R_rs[find_closest(wavelengths, lambda0)[1]]**b


def hico(R_rs, wavelengths, a=2e6, b=2.7848, lambda0=646):
    """
    Empirical turbidity algorithm for the HICO mission [1] based on the method by Chen et al. [2]

    [1] Keith et al. (2014): Remote sensing of selected water-quality indicators with the hyperspectral imager for the coastal ocean (HICO) sensor [10.1080/01431161.2014.894663]
    [2] Chen et al. (2007): Monitoring turbidity in Tampa Bay using MODIS/Aqua 250-m imagery [10.1016/j.rse.2006.12.019]

    Args:
        R_rs (_type_): remote sensing reflectance [sr-1] spectrum
        wavelengths (_type_): correspondong wavelengths [nm]
        a (float, optional): multiplier. Defaults to 1203.9.
        b (float, optional): exponent. Defaults to 1.087.
        lambda0 (int, optional): wavelength for turbidity estimation. Defaults to 646.
    Returns: turbidity [NTU]
    """
    return chen(R_rs, wavelengths, a, b, lambda0)
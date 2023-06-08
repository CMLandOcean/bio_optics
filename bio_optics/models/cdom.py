import numpy as np
from .. helper.utils import find_closest


def hico(R_rs, wavelengths, lambda1=670, lambda2=490):
    """
    Empirical CDOM model for the HICO mission [1]
    
    [1] Keith et al. (2014): Remote sensing of selected water-quality indicators with the hyperspectral imager for the coastal ocean (HICO) sensor [10.1080/01431161.2014.894663]
    
    :param R_rs: R_rs spectrum [sr-1] with first axis = bands
    :param wavelengths: corresponding wavelengths [nm]
    :param lambda_1: wavelength of first band [nm], default: 490 
    :param lambda_2: wavelength of second band [nm], default: 670
    :return: CDOM absorption at 412 nm [m-1]
    """
    band1 = R_rs[find_closest(wavelengths, lambda1)[1]]
    band2 = R_rs[find_closest(wavelengths, lambda2)[1]]

    return 0.8426 * (band1/band2) - 0.032


def loisel(R_rs, wavelengths, lambda1=412, lambda2=555, sza=0):
    """
    Semi-empirial CDOM model based on K_d [1]
    
    [1] Loisel et al. (2014): Assessment of the colored dissolved organic matter in coastal waters from ocean color remote sensing [10.1364/oe.22.013109]

    :param R_rs: R_rs spectrum [sr-1] with first axis = bands
    :param wavelengths: corresponding wavelengths [nm]
    :param lambda1: first wavelength for ratio [nm]
    :param lambda2: second wavelength for ration [nm]
    :param sza: sun zenith angle [deg]
    :return: CDOM absorption at 412 nm [m-1]
    """
    band1 = R_rs[find_closest(wavelengths, lambda1)[1]]
    band2 = R_rs[find_closest(wavelengths, lambda2)[1]]
    
    if sza==0:
        A, B, C, D = -0.0634808, 0.254858, -1.22384, -0.89454
    elif sza==30:
        A, B, C, D = -0.12484, 0.160857, -1.2292, -0.886471
    elif sza==60:
        A, B, C, D = -0.535652, -0.224119, -1.18114, -0.840784
        
    # Eq. 8, and inline Eq. after Eq. 7
    Y = 10**(A * np.log10(band1/band2)**3 + B * np.log10(band1/band2)**2 + C * np.log10(band1/band2) + D) 
    # Eq. 6 & Eq. 7
    X = Y - 10**(-0.009 * np.log10(Y)**2 + 1.147 * np.log10(Y) - 0.26)
    # Eq. 6
    a_CDOM412 = 10**(0.1548 * np.log10(X)**2 + 1.1939 * np.log10(X) + 0.0689)
    
    return a_CDOM412


def mannino(R_rs, wavelengths, lambda0=443, lambda1=490, lambda2=551):
    """
    Empirical model to estimate CDOM absorption [m-1] at a reference wavelengths (lambda0) at 443 nm, 412 nm or 355 nm.
    Coefficients are for MODIS-Aqua.

    "The form of the algorithm is the nonlinear one-phase exponential decay regression model. The non-linear function was
    solved for a_CDOM yielding the following equation. The Rrs(490 nm/551 nm) band ratio algorithms were applied to
    derive a_CDOM from MODIS-Aqua, and no adjustments were made to field-derived R_rs at 490 nm to match the 488 nm MODIS-Aqua band" [1].

    [1] Mannino et al. (2008): Algorithm development and validation for satellite-derived distributions of DOC and CDOM in the U.S. Middle Atlantic Bight [10.1029/2007JC004493]

    Args:
        R_rs: remote sensing reflectance [sr-1] spectrum
        wavelengths: corresponding wavelengths [nm]
        lambda0: reference wavelength [nm] to compute a_cdom for. Defaults to 443. Alternatives are 355 and 412.
        lambda1: TBD
        lambda2: TBD      
    Returns: 
        CDOM absorption [m-1] at a reference wavelength (lambda0) 
    """
    if lambda0==443:
        a, b, c = 0.4363, 2.221, 13.126

    elif lambda0==412:
        a, b, c = 0.4553, 2.345, 8.045

    elif lambda0==355:
        a, b, c = 0.4934, 2.731, 3.512

    band1 = R_rs[find_closest(wavelengths, lambda1)[1]]
    band2 = R_rs[find_closest(wavelengths, lambda2)[1]]

    return np.log((band1/band2 - a) / b) / (-c)


def ficek(R_rs, wavelengths, a=3.65, b=-1.93, lambda1=570, lambda2=655):
    """
    Empirical model to estimate CDOM absorption [m-1] at 440 nm [1]

    [1] Ficek et al. (2011): Remote sensing reflectance of Pomeranian lakes and the Baltic [10.1016/j.csr.2009.12.007]

    Args:
        R_rs: remote sensing reflectance [sr-1] spectrum
        wavelengths: corresponding wavelengths [nm]
        lambda1: TBD
        lambda2: TBD      
    Returns:
        CDOM absorption [m-1] at 440 nm
    """
    band1 = R_rs[find_closest(wavelengths, lambda1)[1]]
    band2 = R_rs[find_closest(wavelengths, lambda2)[1]]

    return a * (band1/band2)**b





     


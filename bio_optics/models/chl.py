import numpy as np
from .. helper.utils import find_closest
from .. helper.indices import ndi


def pigment_concentration(band1, band2, band3):
    """
    General formulation of a three band reflectance model to estimate pigment concentration (Eq. 3 in [1]). 
    Originally developed for terrestrial vegetation [2,3] but also applicable to turbid waters [4].

    [1] Gitelson et al. (2008): A simple semi-analytical model for remote estimation of chlorophyll-a in turbid waters: Validation [10.1016/j.rse.2008.04.015]
    [2] tba
    [3] tba
    [4] tba

    Args:
        band1: band one has to be restricted within the range of 660 to 690 nm
        band2: band two should be in the range from 710 to 730 nm
        band3: band three should be from a range where reflectance is minimally affected by a_chl, a_tripton, and a_CDOM
    Returns:
        pigment concentration
    """
    return (band1**(-1) - band2**(-1)) * band3


def gitelson(R, wavelengths, a=117.42, b=23.09, lambda1=660, lambda2=715, lambda3=750):
    """
    Semi-analytical model that relates chlorophyll-a pigment concentration to reflectance R in three spectral bands [1].

    [1] Gitelson et al. (2008): A simple semi-analytical model for remote estimation of chlorophyll-a in turbid waters: Validation [10.1016/j.rse.2008.04.015]

    Args:
        R (_type_): irradiance reflectance [-] spectrum
        wavelengths (_type_): corresponding wavelengths [nm]
        lambda1 (int, optional): _description_. Defaults to 675.
        lambda2 (int, optional): _description_. Defaults to 720.
        lambda3 (int, optional): _description_. Defaults to 748.
    Returns:
        chlorophyll-a pigment concentration [ug L-1]
    """
    band1 = R[find_closest(wavelengths, lambda1)[1]]
    band2 = R[find_closest(wavelengths, lambda2)[1]]
    band3 = R[find_closest(wavelengths, lambda3)[1]]

    return a * pigment_concentration(band1, band2, band3) + b


def hico(R_rs, wavelengths, a=17.477, b=6.152, lambda1 = 686, lambda2 = 703, lambda3 = 735):
    """
    Semi-analytical model chl model for the HICO mission [1] that relates chlorophyll-a pigment concentration to reflectance R_rs in three spectral bands [2].

    [1] Keith et al. (2014): Remote sensing of selected water-quality indicators with the hyperspectral imager for the coastal ocean (HICO) sensor [10.1080/01431161.2014.894663]
    [2] Gitelson et al. (2008): A simple semi-analytical model for remote estimation of chlorophyll-a in turbid waters: Validation [10.1016/j.rse.2008.04.015]

    Args:
        R_rs (_type_): remote sensing reflectance [sr-1] spectrum
        wavelengths (_type_): corresponding wavelengths [nm]
        lambda1 (int, optional): _description_. Defaults to 686.
        lambda2 (int, optional): _description_. Defaults to 703.
        lambda3 (int, optional): _description_. Defaults to 735.
    Returns:
        chlorophyll-a pigment concentration [ug L-1]
    """
    band1 = R_rs[find_closest(wavelengths, lambda1)[1]]
    band2 = R_rs[find_closest(wavelengths, lambda2)[1]]
    band3 = R_rs[find_closest(wavelengths, lambda3)[1]]

    return a * pigment_concentration(band1, band2, band3) + b


def flh(R_rs, wavelengths, lambda1=665, lambda2=681, lambda3=705, k=1.005):
    """
    Fluorescence Line Height (FLH) also known as Maximum Chlorophyll Index (MCI) (FLI/MCI) [1,2]
    Estimates magnitude of sun induces chlorophyll fluorescence at 681 nm above a baseline between 665 and 705 nm.

    [1] Gower et al. (2010): Interpretation of the 685nm peak in water-leaving radiance spectra in terms of fluorescence, absorption and scattering, and its observation by MERIS [doi.org/10.1080/014311699212470].
    [2] Mishra et al. (2017): Bio-optical Modeling and Remote Sensing of Inland Waters, p. 211.

    Args:
        R_rs (_type_): remote sensing reflectance [sr-1] spectrum
        wavelengths (_type_): corresponding wavelengths [nm]
        lambda1 (int, optional): _description_. Defaults to 680.
        lambda2 (int, optional): _description_. Defaults to 708.
        lambda3 (int, optional): _description_. Defaults to 753.
        k (float, optional): _description_. Defaults to 1.005.
    """
    L1 = R_rs[find_closest(wavelengths, lambda1)[1]]
    L2 = R_rs[find_closest(wavelengths, lambda2)[1]]
    L3 = R_rs[find_closest(wavelengths, lambda3)[1]]

    k = ((R_rs[find_closest(wavelengths, lambda3)[1]]-R_rs[find_closest(wavelengths, lambda2)[1]]) / (R_rs[find_closest(wavelengths, lambda3)[1]]-R_rs[find_closest(wavelengths, lambda1)[1]]))

    return L2 - (L3 + (L1 - L3) * k)


def ndci(R_rs, wavelengths, lambda1=665, lambda2=708, a0=14.039, a1=86.115, a2=194.325):
    """
    Normalized Difference Chlorophyll Index [1]
    Coefficients are from Table 2, 2nd box [1]
    
    [1] Mishra & Mishra (2012): Normalized difference chlorophyll index: A novel model for remote estimation of chlorophyll-a concentration in turbid productive waters [10.1016/j.rse.2011.10.016]

    Args:
        R_rs (_type_): remote sensing reflectance [sr-1] spectrum
        wavelengths (_type_): corresponding wavelengths [nm]
        lambda1 (int, optional): _description_. Defaults to 665.
        lambda2 (int, optional): _description_. Defaults to 708.
    """
    band1 = R_rs[find_closest(wavelengths,lambda1)[1]]
    band2 = R_rs[find_closest(wavelengths,lambda2)[1]]

    return a0 + a1 *  ndi(band2, band1) + a2 * ndi(band2, band1)**2


def li(R_rs, wavelengths, blue=466.79, green=536.90, red=652.07, a=-0.4909, b=191.659):
    """
    Simple chl-a retrieval after Li et al. (2019) [1].
    Part of adaptive bathymetry estimation for shallow coastal chl-a dominated waters (Case-I waters).

    [1] Li et al. (2019): Adaptive bathymetry estimation for shallow coastal waters using Planet Dove satellites [10.1016/j.rse.2019.111302]

    Args:
        R_rs (_type_): remote sensing reflectance [sr-1] spectrum
        wavelengths (_type_): corresponding wavelengths [nm]
        blue (float, optional): Wavelength of blue band [nm]. Defaults to 466.79.
        green (float, optional): Wavelength of green band [nm]. Defaults to 536.90.
        red (float, optional): Wavelength of red band [nm]. Defaults to 652.07.
    """
    omega = R_rs[find_closest(wavelengths, green)[1]] - 0.46 * R_rs[find_closest(wavelengths, red)[1]] - 0.54 * R_rs[find_closest(wavelengths, blue)[1]]
    
    return 10**(a + b * omega)
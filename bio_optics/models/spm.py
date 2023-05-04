import numpy as np
from .. helper.utils import find_closest


def petus(R_rs, wavelengths, a=12450, b=666.1, c=0.48, lambda0=860):
    """
    Empirical TSS model for MODIS [1] 

    [1] Petus et al. (2010): Estimating turbidity and total suspended matter in the Adour River plume (South Bay of Biscay) using MODIS 250-m imagery [10.1016/j.csr.2009.12.007]

    Args:
        R_rs: remote sensing reflectance [sr-1] spectrum
        wavelengths: corresponding wavelengths [nm]
        a: Defaults to 12450.
        b: Defaults to 666.1.
        c: Defaults to 0.48.
        lambda0: wavelength for turbidity estimation. Defaults to 860.
    Returns: spm [mg L-1]
    """
    x = R_rs[find_closest(wavelengths, 860)[1]]
    return a*x**2 + b*x + c


def miller(R, wavelengths, a=-1.91, b=1140.25, lambda0=645):
    """
    Empirical linear TSM model for MODIS band 1 (620 – 670 nm) [1].
    I think there is an error in the formular in Fig. 2: It should be + instead of *.

    [1] Miller & McKee (2004): Using MODIS Terra 250 m imagery to map concentrations of total suspended matter in coastal waters [10.1016/j.rse.2004.07.012]

    Args:
        R: irradiance reflectance [-] spectrum
        wavelengths: corresponding wavelengths [nm]
        a (float, optional): Defaults to -1.91.
        b (float, optional): Defaults to 1140.25.
        lambda0: wavelength for TSM estimation. Defaults to 645 as the center wavelength of MODIS band 1.

    Returns:
       tsm concentration [mg L-1]
    """
    band1 = R[find_closest(wavelengths, lambda0)[1]]
    return a + b * band1


def novoa(R_w, wavelengths, a=130.1 , b=531.5, c=37150, d=1751, lambda1=561, lambda2=665, lambda3=865):
    """
    A switching method that automatically selects the most sensitive SPM vs. R_w relationship, to avoid saturation effects when computing the SPM concentration [1].
    Note that this function only applies the proposed method for the Gironde data set (see Table 3 in [1]).

    [1] Novoa et al. (2017): Atmospheric corrections and multi-conditional algorithm for multi-sensor remote sensing of suspended particulate matter in low-to-high turbidity levels coastal waters [10.3390/rs9010061]

    Args:
        R_w: water-leaving reflectance [-] spectrum
        wavelengths: corresponding wavelengths [nm]
        a (float, optional): Defaults to 130.1.
        b (float, optional): Defaults to 531.5.
        c (int, optional): Defaults to 37150.
        d (int, optional): Defaults to 1751.
        lambda1: wavelength. Defaults at 561 nm.
        lambda2: wavelength. Defaults at 665 nm.
        lambda3: wavelength. Defaults at 865 nm.

    Returns:
        spm: concentration of SPM [mg L-1]
    """
    band1 = R_w[find_closest(wavelengths, lambda1)[1]]
    band2 = R_w[find_closest(wavelengths, lambda2)[1]]
    band3 = R_w[find_closest(wavelengths, lambda3)[1]]

    # Models described in Table 2
    linear_green = a * band1
    linear_red = b * band2
    poly_nir = c * band3**2 + d * band3

    # Tresholds described in Table 3, Columns 1 & 2
    S_GL95_minus = 0.007
    S_GL95_plus = 0.016
    S_GH95_minus = 0.08
    S_GH95_plus = 0.12   

    # Eq. 4
    alpha1 = np.log(S_GL95_plus / band2) / np.log(S_GL95_plus / S_GL95_minus)
    beta1 = np.log(band2 / S_GL95_minus) / np.log(S_GL95_plus / S_GL95_minus)
    # Eq. 6
    alpha2 = np.log(S_GH95_plus / band2) / np.log(S_GH95_plus / S_GH95_minus)
    beta2 = np.log(band2 / S_GH95_minus) / np.log(S_GH95_plus / S_GH95_minus)

    # Descision tree described in Table 3, Columns 1 & 2
    type0 = band2 < S_GL95_minus
    type1 = (S_GL95_minus <= band2) & (band2 <= S_GL95_plus)
    type2 = (S_GL95_plus < band2) & (band2 < S_GH95_minus)
    type3 = (S_GH95_minus <= band2) & (band2 <= S_GH95_plus)
    type4 = S_GH95_plus < band2

    # Create empty array to fill
    spm = np.empty(type1.shape)
    spm[:] = np.nan

    # Select appropriate model based on decision tree (Table 3, Column 3)
    spm = np.where(type0, linear_green,
                   np.where(type1, (alpha1 * linear_green + beta1 * linear_red),
                            np.where(type2, linear_red,
                                     np.where(type3, (alpha2 * linear_red + beta2 * poly_nir),
                                              np.where(type4, poly_nir, np.nan)))))

    return spm


def gaa(R_rs, wavelengths, lambda1=486, lambda2=551, lambda3=671, lambda4=745, lambda5=862, a1=20.43, a2=2.15, c0=0.04, c1=1.17, c2=0.4, c3=14.86):
    """
    An empirical globally applicable algorithm to to seamlessly retrieve the concentration of suspended particulate matter (SPM)
    from remote sensing reflectance across ocean to turbid river mouths without any hard-switching [1]
    
    [1] Yu et al. (2019): An empirical algorithm to seamlessly retrieve the concentration of suspended particulate matter from water color across ocean to turbid river mouths [doi.org/10.1016/j.rse.2019.111491].

    Args:
        R_rs: remote sensing reflectance [sr-1] spectrum
        wavelengths: corresponding wavelengths [nm]
    Returns:
        spm [mg L-1]
    """
    band1 = R_rs[find_closest(wavelengths, lambda1)[1]]
    band2 = R_rs[find_closest(wavelengths, lambda2)[1]]
    band3 = R_rs[find_closest(wavelengths, lambda3)[1]]
    band4 = R_rs[find_closest(wavelengths, lambda4)[1]]
    band5 = R_rs[find_closest(wavelengths, lambda5)[1]]

    prod1 = c1 * (band3 / np.sum([band3, band4, band5], axis=0)) * (band3/band2)
    prod2 = c2 * (band4 / np.sum([band3, band4, band5], axis=0)) * (band4/band2)
    prod3 = c3 * (band5 / np.sum([band3, band4, band5], axis=0)) * (band5/band2)

    GI_SPM = c0 * (band2 / band1) + np.sum([prod1, prod2, prod3], axis=0)
    C_SPM = a1 * GI_SPM**a2

    return C_SPM


def dsa(R_rs, wavelengths, lambda1=671, lambda2=551, a=1.25, b=1.11):
    """
    Empirical two band SPM retrieval algorithm [1]

    [1] D'Sa et al. (2007): Suspended particulate matter dynamics in coastal waters from ocean color: Application to the northern Gulf of Mexico [10.4319/lo.2007.52.6.2418 ]

    Args:
        R_rs: remote sensing reflectance [sr-1] spectrum
        wavelengths: corresponding wavelengths [nm]
        lambda1: Defaults to 671.
        lambda2: Defaults to 551.
    """
    X = R_rs[find_closest(wavelengths, lambda1)[1]] / R_rs[find_closest(wavelengths, lambda2)[1]]
    return 10**(a + b * np.log(X))

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


def ci(R_rs, wavelengths, lambda1=443.0, lambda2=555.0, lambda3=670.0, x=0.5, y=1.0):
    """
    Color Index (CI) as described in Hu et al. (2012) [1] Eq. 3.

    [1] Hu et al. (2012): Chlorophyll aalgorithms for oligotrophic oceans: A novel approach based on three-band reflectance difference [10.1029/2011JC007395]

    Args:
        R_rs: remote sensing reflectance [sr-1] spectrum
        wavelengths: corresponding wavelengths [nm]
        lambda1 (float, optional): Wavelength of blue band [nm]. Defaults to 443.0
        lambda2 (float, optional): Wavelength of green band [nm]. Defaults to 555.0
        lambda3 (float, optional): Wavelength of red band [nm]. Defaults to 670.0

    Returns:
        chl concentration [mg m-3]
    """
    ci = R_rs[find_closest(wavelengths, lambda2)[1]] - x * R_rs[find_closest(wavelengths, lambda1)[1]] + y * R_rs[find_closest(wavelengths, lambda3)[1]]

    return ci


def cia(R_rs, wavelengths, lambda1=443.0, lambda2=555.0, lambda3=670.0, x=0.5, y=1.0, a=-0.4909, b=191.659):
    """
    CI-based Algorithm (CIA) to retrieve Chlorophyll a concentration in oligotrophic oceans [1]

    !!! Only valid for CI <= 0.0005 sr-1 !!!

    [1] Hu et al. (2012): Chlorophyll aalgorithms for oligotrophic oceans: A novel approach based on three-band reflectance difference [10.1029/2011JC007395]

    Args:
        R_rs: remote sensing reflectance [sr-1] spectrum
        wavelengths: corresponding wavelengths [nm]
        lambda1 (float, optional): Wavelength of blue band [nm]. Defaults to 443.0
        lambda2 (float, optional): Wavelength of green band [nm]. Defaults to 555.0
        lambda3 (float, optional): Wavelength of red band [nm]. Defaults to 670.0
        x (float, optional): Factor for blue band. Defaults to 0.5.
        y (float, optional): Factor for red band. Defaults to 1.0.
        a (float, optional): Empirical component. Defaults to -0.4909.
        b (float, optional): Empirical component. Defaults to 191.659.
    
    Returns:
        chl concentration [mg m-3]
    """
    cia = 10**(a + b * ci(R_rs=R_rs, wavelengths=wavelengths, lambda1=lambda1, lambda2=lambda2, lambda3=lambda3, x=x, y=y))

    return cia 


def li(R_rs, wavelengths, lambda3=466.79, lambda2=536.90, lambda1=652.07, a=-0.4909, b=191.659):
    """
    Chl-a retrieval for Planet Dove data as described in Li et al. (2019) [1] after Hu et al. (2012) [2]
    Part of adaptive bathymetry estimation for shallow coastal chl-a dominated waters (Case-I waters).

    !!! Note that compared to Hu et al., the red and blue band are interchanged !!!    
    !!! Only valid for opticall deep water !!!

    [1] Li et al. (2019): Adaptive bathymetry estimation for shallow coastal waters using Planet Dove satellites [10.1016/j.rse.2019.111302]
    [2] Hu et al. (2012): Chlorophyll aalgorithms for oligotrophic oceans: A novel approach based on three-band reflectance difference [10.1029/2011JC007395]
    
    Args:
        R_rs (_type_): remote sensing reflectance [sr-1] spectrum
        wavelengths (_type_): corresponding wavelengths [nm]
        lambda1 (float, optional): Wavelength of red band [nm]. Defaults to 652.07.
        lambda2 (float, optional): Wavelength of green band [nm]. Defaults to 536.90.
        lambda3 (float, optional): Wavelength of blue band [nm]. Defaults to 466.79.

    Returns:
        chl concentration [mg m-3]
    """
    chl = cia(R_rs=R_rs, wavelengths=wavelengths, lambda1=lambda1, lambda2=lambda2, lambda3=lambda3, x=x, y=y, a=a, b=b)
  
    return chl


def guc2(R_rs, wavelengths, lambda1=663, lambda2=623, a=113.112, b=58.408, c=8.669, d=0.0384):
    """
    Goa University Case II semianalytical algorithm to retrieve chlorophyll-a in optically complex waters [1].
    Fit to data from the Arabian Sea.
    
    [1] Menon & Adhikari (2018): Remote Sensing of Chlorophyll-A in Case II Waters: A Novel Approach With Improved Accuracy Over Widely Implemented Turbid Water Indices [10.1029/2018JC014052]

    Args:
        R_rs (_type_): remote sensing reflectance [sr-1] spectrum
        wavelengths (_type_): corresponding wavelengths [nm]
        lambda1 (int, optional): Wavelength of first band [nm]. Defaults to 663.
        lambda2 (int, optional): Wavelength of second band [nm]. Defaults to 623.
        a (float, optional): Defaults to 113.112.
        b (float, optional): Defaults to 58.408.
        c (float, optional): Defaults to 8.669.
        d (float, optional): Defaults to 0.0384.

    Returns:
        chlorophyll-a pigment concentration [ug L-1]
    """
    band1 = R_rs[find_closest(wavelengths,lambda1)[1]]
    band2 = R_rs[find_closest(wavelengths,lambda2)[1]]    

    x = (band1**(-1) - band2**(-1)) * band2
    return a*x**3 - b*x**2 + c*x - d 


def two_band(R_rs, wavelengths, lambda1=665.0, lambda2=708.0, a=61.324, b=-37.94):
    """
    Two-band ratio algorithm after Eq. 2 (Model A) in the Neil et al. (2019) compilation [1,2].

    The "two-band ratio algorithm of Dall'Olmo et al. (2003), Moses et al. (2009) and Gitelson et al. (2011), originally proposed by Gitelson and Kondratyev (1991) and later adapted to MERIS bands. 
    This is an empirical formula based on a linear relationship between in-situ Chla and the ratio of MERIS satellite remote sensing reflectance, measured at NIR, Rrs(708), and red, Rrs(665)" [1,2].

    [1] Neil et al. (2018): A global approach for chlorophyll-a retrieval across optically complex inland waters based on optical water types [10.1016/j.rse.2019.04.027]
    [2] Neil et al. (2020): Corrigendum to “A global approach for chlorophyll-a retrieval across optically complex inland waters based on optical water types” [Remote Sens. Environ., 229: 159-178] [10.1016/j.rse.2020.111837]

    Args:
        R_rs (_type_): _description_
        wavelengths (_type_): _description_
        lambda1 (optional): _description_. Defaults to 665.
        lambda2 (optional): _description_. Defaults to 708.
        a (float, optional): Defaults to 61.324.
        b (float, optional): Defaults to -37.94.

    Returns:
        _type_: _description_
    """
    band1 = R_rs[find_closest(wavelengths,lambda1)[1]]
    band2 = R_rs[find_closest(wavelengths,lambda2)[1]]    
    
    return a * (band2/band1) + b


def three_band(R_rs, wavelengths, lambda1=665, lambda2=708, lambda3=753, a=232.329, b=23.174):
    """
    Three-band ratio algorithm after Eq. 3 (Model B) in the Neil et al. (2019) compilation [1,2].

    The "three-band algorithm developed by Moses et al. (2009) and adapted by Gitelson et al. (2011)" [1,2].
    "In theory, the combination of three bands alters the model sensitivity to the presence of optically active constituents by removing the effects of SPM and CDOM 
    (Rrs(665) and Rrs(708) are comparably influenced by SPM and CDOM and Rrs(753) is mainly driven by backscattering) " [1,2].

    [1] Neil et al. (2018): A global approach for chlorophyll-a retrieval across optically complex inland waters based on optical water types [10.1016/j.rse.2019.04.027]
    [2] Neil et al. (2020): Corrigendum to “A global approach for chlorophyll-a retrieval across optically complex inland waters based on optical water types” [Remote Sens. Environ., 229: 159-178] [10.1016/j.rse.2020.111837]

    Args:
        R_rs (_type_): _description_
        wavelengths (_type_): _description_
        lambda1 (int, optional): _description_. Defaults to 665.
        lambda2 (int, optional): _description_. Defaults to 708.
        lambda3 (int, optional): _description_. Defaults to 753.
        a (float, optional): _description_. Defaults to 232.329.
        b (float, optional): _description_. Defaults to 23.174.

    Returns:
        _type_: _description_
    """
    band1 = R_rs[find_closest(wavelengths,lambda1)[1]]
    band2 = R_rs[find_closest(wavelengths,lambda2)[1]] 
    band3 = R_rs[find_closest(wavelengths,lambda3)[1]]    
    
    return a * pigment_concentration(band1, band2, band3) + b


def gurlin_two_band(R_rs, wavelengths, lambda1=665, lambda2=708, a=25.28, b=14.85, c=-15.18):
    """
    Two-band empirically derived ratio algorithm of Gurlin et al. (2011) after Eq. 4 (Model C) in the Neil et al. (2019) compilation [1,2].

    [1] Neil et al. (2018): A global approach for chlorophyll-a retrieval across optically complex inland waters based on optical water types [10.1016/j.rse.2019.04.027]
    [2] Neil et al. (2020): Corrigendum to “A global approach for chlorophyll-a retrieval across optically complex inland waters based on optical water types” [Remote Sens. Environ., 229: 159-178] [10.1016/j.rse.2020.111837]

    Args:
        R_rs (_type_): _description_
        wavelengths (_type_): _description_
        lambda1 (int, optional): _description_. Defaults to 665.
        lambda2 (int, optional): _description_. Defaults to 708.
        a (float, optional): _description_. Defaults to 25.28.
        b (float, optional): _description_. Defaults to 14.85.
        c (float, optional): _description_. Defaults to -15.18.

    Returns:
        _type_: _description_
    """       
    band1 = R_rs[find_closest(wavelengths,lambda1)[1]]
    band2 = R_rs[find_closest(wavelengths,lambda2)[1]] 

    return a * (band2/band1)**2 + b * (band2/band1) + c


def gurlin_three_band(R_rs, wavelengths, lambda1=665, lambda2=708, lambda3=753, a=315.50, b=215.95, c=25.66):
    """
    Three-band ratio algorithm of Gurlin et al. (2011) after Eq. 5 (Model D) in the Neil et al. (2019) compilation [1,2].

    "Calibrated using field measurements of Rrs and Chla taken from Fremont lakes Nebraska" [1,2].

    [1] Neil et al. (2018): A global approach for chlorophyll-a retrieval across optically complex inland waters based on optical water types [10.1016/j.rse.2019.04.027]
    [2] Neil et al. (2020): Corrigendum to “A global approach for chlorophyll-a retrieval across optically complex inland waters based on optical water types” [Remote Sens. Environ., 229: 159-178] [10.1016/j.rse.2020.111837]


    Args:
        R_rs (_type_): _description_
        wavelengths (_type_): _description_
        lambda1 (int, optional): _description_. Defaults to 665.
        lambda2 (int, optional): _description_. Defaults to 708.
        lambda3 (int, optional): _description_. Defaults to 753.
        a (float, optional): _description_. Defaults to 315.50.
        b (float, optional): _description_. Defaults to 215.95.
        c (float, optional): _description_. Defaults to 25.66.

    Returns:
        _type_: _description_
    """
            
    band1 = R_rs[find_closest(wavelengths,lambda1)[1]]
    band2 = R_rs[find_closest(wavelengths,lambda2)[1]]    
    band3 = R_rs[find_closest(wavelengths,lambda3)[1]]   

    Chla = a * (band3/(band1-band2))**2 + b * (band3/(band1-band2)) + c
    return Chla


def analytical_two_band(R_rs, wavelengths, lambda1=665.0, lambda2=708.0, a=35.745, b=19.295, c=1.124):
    """
    Advanced two-band semi-analytical algorithm proposed by Gilerson et al. (2010) after Eq. 7 (Model E) in the Neil et al. (2019) compilation [1,2].

    "While this is governed by the ratio of NIR to red reflectance, model coefficients are determined analytically from individual absorption components contributing to 
    the total IOPs of the water body. It is assumed that the water term dominates (at red-NIR wavelengths) where Chla concentration is > 5 mg m-3" [1,2].
    
    [1] Neil et al. (2018): A global approach for chlorophyll-a retrieval across optically complex inland waters based on optical water types [10.1016/j.rse.2019.04.027]
    [2] Neil et al. (2020): Corrigendum to “A global approach for chlorophyll-a retrieval across optically complex inland waters based on optical water types” [Remote Sens. Environ., 229: 159-178] [10.1016/j.rse.2020.111837]

    Args:
        R_rs (_type_): _description_
        wavelengths (_type_): _description_
        lambda1 (float, optional): _description_. Defaults to 665.0.
        lambda2 (float, optional): _description_. Defaults to 708.0.
        a (float, optional): _description_. Defaults to 35.75.
        b (float, optional): _description_. Defaults to 19.30.
        c (float, optional): _description_. Defaults to 1.124.

    Returns:
        _type_: _description_
    """
    band1 = R_rs[find_closest(wavelengths,lambda1)[1]]
    band2 = R_rs[find_closest(wavelengths,lambda2)[1]]    

    return (a * (band2/band1) -b)**c            
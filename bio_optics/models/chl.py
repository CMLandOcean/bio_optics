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


def gitelson(R, wavelengths, a=117.42, b=23.09, lambda1=665, lambda2=715, lambda3=750):
    """
    Semi-analytical model that relates chlorophyll-a pigment concentration to reflectance R in three spectral bands [1].

    [1] Gitelson et al. (2008): A simple semi-analytical model for remote estimation of chlorophyll-a in turbid waters: Validation [10.1016/j.rse.2008.04.015]

    Args:
        R (_type_): irradiance reflectance [-] spectrum
        wavelengths (_type_): corresponding wavelengths [nm]
        lambda1: sensitive to chl a but also other factors. Defaults to 665.
        lambda2: close to lambda 1 but minimally sensitive to chl a abs. Defaults to 715.
        lambda3: wavelength where abs approx by a_water, to account for variability in backscattering. Defaults to 748.
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
        lambda1: _description_. Defaults to 686.
        lambda2: _description_. Defaults to 703.
        lambda3: _description_. Defaults to 735.
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
    Estimates magnitude of sun induced chlorophyll fluorescence at 681 nm above a baseline between 665 and 705 nm.

    [1] Gower et al. (2010): Interpretation of the 685nm peak in water-leaving radiance spectra in terms of fluorescence, absorption and scattering, and its observation by MERIS [doi.org/10.1080/014311699212470].
    [2] Mishra et al. (2017): Bio-optical Modeling and Remote Sensing of Inland Waters, p. 211., Eq. 7.39

    Args:
        R_rs (_type_): remote sensing reflectance [sr-1] spectrum
        wavelengths (_type_): corresponding wavelengths [nm]
        lambda1: _description_. Defaults to 680.
        lambda2: _description_. Defaults to 708.
        lambda3: _description_. Defaults to 753.
        k (float, optional): _description_. Defaults to 1.005.
    """
    L1 = R_rs[find_closest(wavelengths, lambda1)[1]]
    L2 = R_rs[find_closest(wavelengths, lambda2)[1]]
    L3 = R_rs[find_closest(wavelengths, lambda3)[1]]

    k = (find_closest(wavelengths, lambda3)[0]-find_closest(wavelengths, lambda2)[0]) / (find_closest(wavelengths, lambda3)[0]-find_closest(wavelengths, lambda1)[0])

    return L2 - k * L1 - (1-k) * L3


def cyanobacterial_index(R_rs, wavelengths, lambda1=665, lambda2=681, lambda3=709):
    """
    Cyanobacterial index (CI) as described in Kudela et al. (2015) [1] (Table 3) after Wynne et al. (2008) [2].
    The spectral shape equation (SS) in Table 3 is mathematically identical to the fluorescence line hight (FLH) but weirdly gets multiplied by (-1).

    [1] Kudela et al. (2015): Application of hyperspectral remote sensing to cyanobacterial blooms in inland waters [10.1016/j.rse.2015.01.025]
    [2] Wynne et al. (2008): Relating spectral shape to cyanobacterial blooms in the Laurentian Great Lakes [10.1080/01431160802007640]

    Args:
        R_rs (_type_): remote sensing reflectance [sr-1] spectrum
        wavelengths (_type_): corresponding wavelengths [nm]
        lambda1: _description_. Defaults to 665.
        lambda2: _description_. Defaults to 681.
        lambda3: _description_. Defaults to 709.
    Returns: 
        Cyanobacterial index
    """
    return (-1) * flh(R_rs=R_rs, wavelengths=wavelengths, lambda1=lambda1, lambda2=lambda2, lambda3=lambda3)   


def slh(R_rs, wavelengths, lambda1=654, lambda2=714, lambda3=754):
    """
    Scattering line height (SLH) for detection of cyanobacteria as described in Kudela et al. (2015) [1] (Table 3).

    [1] Kudela et al. (2015): Application of hyperspectral remote sensing to cyanobacterial blooms in inland waters [10.1016/j.rse.2015.01.025]

    Args:
        R_rs (_type_): remote sensing reflectance [sr-1] spectrum
        wavelengths (_type_): corresponding wavelengths [nm]
        lambda1: _description_. Defaults to 654.
        lambda2: _description_. Defaults to 714.
        lambda3: _description_. Defaults to 754.
    Returns: 
        Scattering line height
    """
    Rrs1 = R_rs[find_closest(wavelengths, lambda1)[1]]
    Rrs2 = R_rs[find_closest(wavelengths, lambda2)[1]]
    Rrs3 = R_rs[find_closest(wavelengths, lambda3)[1]]

    lambda1 = find_closest(wavelengths, lambda1)[0]
    lambda2 = find_closest(wavelengths, lambda2)[0]
    lambda3 = find_closest(wavelengths, lambda3)[0]

    return  Rrs2 - (Rrs1 + ((Rrs3 - Rrs1)/(lambda3-lambda1)) * (lambda2-lambda1))


def ndci(R_rs, wavelengths, lambda1=665, lambda2=708, a0=14.039, a1=86.115, a2=194.325):
    """
    Normalized Difference Chlorophyll Index [1].
    Coefficients are from Table 2, 2nd box [1].
    
    [1] Mishra & Mishra (2012): Normalized difference chlorophyll index: A novel model for remote estimation of chlorophyll-a concentration in turbid productive waters [10.1016/j.rse.2011.10.016]

    Args:
        R_rs (_type_): remote sensing reflectance [sr-1] spectrum
        wavelengths (_type_): corresponding wavelengths [nm]
        lambda1: _description_. Defaults to 665.
        lambda2: _description_. Defaults to 708.
    """
    band1 = R_rs[find_closest(wavelengths,lambda1)[1]]
    band2 = R_rs[find_closest(wavelengths,lambda2)[1]]

    return a0 + a1 *  ndi(band2, band1) + a2 * ndi(band2, band1)**2


def color_index(R_rs, wavelengths, lambda1=443.0, lambda2=555.0, lambda3=670.0, x=0.5, y=1.0, simple=False):
    """
    Color Index (CI) as described in Hu et al. (2012) [1] Eq. 3. Relative height of Rrs(555) from a background baseline formed linearly between Rrs(443) and Rrs(670)

    [1] Hu et al. (2012): Chlorophyll algorithms for oligotrophic oceans: A novel approach based on three-band reflectance difference [10.1029/2011JC007395]
    
    Args:
        R_rs: remote sensing reflectance [sr-1] spectrum
        wavelengths: corresponding wavelengths [nm]
        lambda1 (float, optional): Wavelength of blue band [nm]. Defaults to 443.0
        lambda2 (float, optional): Wavelength of green band [nm]. Defaults to 555.0
        lambda3 (float, optional): Wavelength of red band [nm]. Defaults to 670.0
    
    Returns:
        color index [sr-1]
    """
    
    if simple:
        ci = R_rs[find_closest(wavelengths, lambda2)[1]] - x * (R_rs[find_closest(wavelengths, lambda1)[1]] + y * R_rs[find_closest(wavelengths, lambda3)[1]])
    else:
        ci = R_rs[find_closest(wavelengths, lambda2)[1]] - (R_rs[find_closest(wavelengths, lambda1)[1]] + (lambda2-lambda1)/(lambda3-lambda1) * (R_rs[find_closest(wavelengths, lambda3)[1]] - R_rs[find_closest(wavelengths, lambda1)[1]]))


    return ci


def cia(R_rs, wavelengths, lambda1=443.0, lambda2=555.0, lambda3=670.0, a=-0.8204, b=49.3352, simple=False):
    """
    Color index-based Algorithm (CIA) to retrieve Chlorophyll a concentration in oligotrophic oceans [1]

    !!! Only valid for CI <= - 0.0005 sr-1 !!! At higher chl concentrations (approx 0.4 mg/m3), use Ocx [1]

    [1] Hu et al. (2012): Chlorophyll a algorithms for oligotrophic oceans: A novel approach based on three-band reflectance difference [10.1029/2011JC007395]
    [2] Hu et al. (2019): Improving Satellite Global Chlorophyll a Data Products Through Algorithm Refinement and Data Recovery [10.1029/2019JC014941]

    Args:
        R_rs: remote sensing reflectance [sr-1] spectrum
        wavelengths: corresponding wavelengths [nm]
        lambda1 (float, optional): Wavelength of blue band [nm]. Defaults to 443.0
        lambda2 (float, optional): Wavelength of green band [nm]. Defaults to 555.0
        lambda3 (float, optional): Wavelength of red band [nm]. Defaults to 670.0
        a (float, optional): Empirical component. Defaults to -0.8204 based on fluorometric Jan 2023 CM data (Use -0.4287 for [2] or -0.4909 for [1]).
        b (float, optional): Empirical component. Defaults to 49.3352 based on fluorometric Jan 2023 CM data (Use 230.47 for [2] or 191.659 for [1]).
    
    Returns:
        chl concentration [mg m-3]
    """
    cia = 10**(a + b * color_index(R_rs=R_rs, wavelengths=wavelengths, lambda1=lambda1, lambda2=lambda2, lambda3=lambda3, simple=simple))

    return cia 


def li(R_rs, wavelengths, lambda3=466.79, lambda2=536.90, lambda1=652.07, x=0.46, y=0.54, a=-0.4909, b=191.659):
    """
    Chl-a retrieval for Planet Dove data as described in Li et al. (2019) [1] after Hu et al. (2012) [2]
    Part of adaptive bathymetry estimation for shallow coastal chl-a dominated waters (Case-I waters).

    !!! Note that compared to Hu et al., the red and blue band are interchanged !!!    
    !!! Only valid for optically deep water !!!

    [1] Li et al. (2019): Adaptive bathymetry estimation for shallow coastal waters using Planet Dove satellites [10.1016/j.rse.2019.111302]
    [2] Hu et al. (2012): Chlorophyll aalgorithms for oligotrophic oceans: A novel approach based on three-band reflectance difference [10.1029/2011JC007395]
    
    Args:
        R_rs (_type_): remote sensing reflectance [sr-1] spectrum
        wavelengths (_type_): corresponding wavelengths [nm]
        lambda1 (float, optional): Wavelength of red band [nm]. Defaults to 652.07.
        lambda2 (float, optional): Wavelength of green band [nm]. Defaults to 536.90.
        lambda3 (float, optional): Wavelength of blue band [nm]. Defaults to 466.79.
        x (float, optional): Factor for red band. Defaults to 0.46.
        y (float, optional): Factor for blue band. Defaults to 0.54.
        a (float, optional): Empirical component. Defaults to -0.4909.
        b (float, optional): Empirical component. Defaults to 191.659.

    Returns:
        chl concentration [mg m-3]
    """
    chl = cia(R_rs=R_rs, wavelengths=wavelengths, lambda1=lambda1, lambda2=lambda2, lambda3=lambda3, x=x, y=y, a=a, b=b, simple=True)
  
    return chl


def guc2(R_rs, wavelengths, lambda1=663, lambda2=623, a=113.112, b=58.408, c=8.669, d=0.0384):
    """
    Goa University Case II semianalytical algorithm to retrieve chlorophyll-a in optically complex waters [1].
    Fit to data from the Arabian Sea.
    
    [1] Menon & Adhikari (2018): Remote Sensing of Chlorophyll-A in Case II Waters: A Novel Approach With Improved Accuracy Over Widely Implemented Turbid Water Indices [10.1029/2018JC014052]

    Args:
        R_rs (_type_): remote sensing reflectance [sr-1] spectrum
        wavelengths (_type_): corresponding wavelengths [nm]
        lambda1: Wavelength of first band [nm]. Defaults to 663.
        lambda2: Wavelength of second band [nm]. Defaults to 623.
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
        lambda1: _description_. Defaults to 665.
        lambda2: _description_. Defaults to 708.
        lambda3: _description_. Defaults to 753.
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
        lambda1: _description_. Defaults to 665.
        lambda2: _description_. Defaults to 708.
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
        lambda1: _description_. Defaults to 665.
        lambda2: _description_. Defaults to 708.
        lambda3: _description_. Defaults to 753.
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


def oc4me(R_rs, wavelengths, lambda1=443, lambda2=489, lambda3=510, lambda4=560, a0=0.450, a1=-3.259, a2=3.523, a3=-3.359, a4=0.950):
    """
    Ocean Color for Meris (OC4Me) algorithm as described in Mishra et al. (2017) [1] Eqs. 6.2f.

    [1] Mishra et al. (2017): Bio-optical Modeling and Remote Sensing of Inland Waters.

    Args:
        R_rs (_type_): remote sensing reflectance [sr-1] spectrum
        wavelengths (_type_): corresponding wavelengths [nm]
        lambda1: _description_. Defaults to 443.
        lambda2: _description_. Defaults to 489.
        lambda3: _description_. Defaults to 510.
        lambda4: _description_. Defaults to 560.
        a0: Defaults to 0.450.
        a1: Defaults to -3.259.
        a2: Defaults to 3.523.
        a3: Defaults to -3.359.
        a4: Defaults to 0.950.
    Returns: 
        Scattering line height
    """

    R_rs1 = np.max(R_rs[[find_closest(wavelengths, lambda1)[1], find_closest(wavelengths, lambda2)[1], find_closest(wavelengths, lambda3)[1]]], axis=0)
    R_rs2 = R_rs[find_closest(wavelengths, lambda4)[1]]

    x = np.log10(R_rs1/R_rs2)

    Chl_a = 10**(a0 + a1*x + a2*x**2 + a3*x**3 + a4*x**4)

    return Chl_a


def potes_cya(R, wavelengths, lambda1=490, lambda2=560, lambda3=620, a=115530.31, b=2.38):
    """
    Empirical algorithm for Cyanobacteria concentration [10**3 cells mL-1] as reported in Petus et al. (2018) [1].
    Originally developed for MERIS but proven to work for Sentinel-2 as well.

    [1] Potes et al. (2018): Use of Sentinel 2-MSI for water quality monitoring at Alqueva reservoir, Portugal [10.5194/piahs-380-73-2018]

    Args:
        R: Water reflectance [-] spectrum
        wavelengths: correspondong wavelengths [nm]

    Returns:
        Cyanobacteria concentration [10**3 cells mL-1]
    """
    return a * ((R[find_closest(wavelengths, lambda2)[1]] * R[find_closest(wavelengths, lambda3)[1]]) / R[find_closest(wavelengths, lambda1)[1]])**b 


def potes_chl(R, wavelengths, lambda1=442.5, lambda2=560, a=4.23, b=3.94):
    """
    Empirical algorithm for Chl a concentration [mg m-3] as reported in Petus et al. (2018) [1].
    Originally developed for MERIS but proven to work for Sentinel-2 as well.

    [1] Potes et al. (2018): Use of Sentinel 2-MSI for water quality monitoring at Alqueva reservoir, Portugal [10.5194/piahs-380-73-2018]

    Args:
        R: Water reflectance [-] spectrum
        wavelengths: correspondong wavelengths [nm]

    Returns:
        Cyanobacteria concentration [10**3 cells mL-1]
    """
    return a * (R[find_closest(wavelengths, lambda2)[1]] / R[find_closest(wavelengths, lambda1)[1]])**b 


def ocx(R_rs, wavelengths, algorithm_name='OC6_ENMAP'):
    """
    Empirical ocean color algorithms for Chlorophyll retrieval as described in O'Reilly & Werdell (2019) [1]

    [1] O'Reilly & Werdell (2019): Chlorophyll algorithms for ocean color sensors - OC4, OC5 & OC6 [10.1016/j.rse.2019.04.021]

    Args:
        R_rs: remote sensing reflectance [sr-1] spectrum
        wavelengths: corresponding wavelengths [nm]
        algorithm_name: algorithm name as described in Tab. 6 [1]. Defaults to 'OC6_ENMAP'.

    Returns:
        Chl: Chlorophyll concentration in mg m-3
    """
    algorithm_names = np.array(
        ['OC6_SEAWIFS', 'OC6_MODIS', 'OC6_MERIS', 'OC6_COCTS', 'OC6_SGLI',
        'OC6_SABIA_MAR', 'OC6_PACE_OCI', 'OC6_OSMI', 'OC6_OLCI', 'OC6_OCTS',
        'OC6_OCM', 'OC6_MOS', 'OC6_MERSI', 'OC6_HICO', 'OC6_HAWKEYE', 
        'OC6_GOCI', 'OC6_GLI', 'OC6_ENMAP', 'OC5_SEAWIFS',  'OC5_OLCI', 
        'OC5_MODIS', 'OC5_MERIS', 'OC5_GOCI', 'OC5_SABIA_MAR', 'OC5_PACE_OCI', 
        'OC5_OSMI', 'OC5_GLI', 'OC5_ENMAP', 'OC5_COCTS', 'OC5_HAWKEYE', 
        'OC5_HICO', 'OC5_MERSI', 'OC5_MOS', 'OC5_OCM', 'OC5_OCTS', 
        'OC4_SEAWIFS', 'OC4_COCTS', 'OC4_VIIRS', 'OC4_SGLI', 'OC4_SABIA_MAR', 
        'OC4_OCM', 'OC4_OCI', 'OC4_MOS',  'OC4_MERSI', 'OC4_HICO', 
        'OC4_HAWKEYE', 'OC4_GOCI', 'OC4_GLI', 'OC4_ENMAP', 'OC4_PACE_OCI', 
        'OC4_MERIS', 'OC4_OLCI', 'OC4_OCTS', 'OC4_OSMI', 'OC4_MODIS', 
        'OC3_POLDER', 'OC3_VIIRS', 'OC3_CZCS', 'OC3_SGLI', 'OC3_POLDER_2', 
        'OC3_MODIS', 'OC3_OLI', 'OC2_POLDER', 'OC2_POLDER_2', 'OC2_MISR'])

    A = np.array([
        [0.92160, -3.17884, 2.39690, -1.30318, 0.20160], 
        [1.22914, -4.99423, 5.64706, -3.53426, 0.69266], 
        [0.95087, -3.05489, 2.18141, -1.11783, 0.15132], 
        [1.11801, -3.48138, 2.74672, -1.38603, 0.19322], 
        [1.28506, -4.20996, 3.83254, -2.03507, 0.32442], 
        [0.90755, -3.17549, 2.43524, -1.34385, 0.21096],
        [0.94297, -3.18493, 2.33682, -1.23923, 0.18697], 
        [0.92160, -3.17884, 2.39690, -1.30318, 0.20160], 
        [0.95039, -3.05404, 2.17992, -1.12097, 0.15262], 
        [1.05968, -3.24992, 2.41784, -1.19442, 0.15412], 
        [0.89280, -3.17118, 2.47461, -1.38801, 0.22203], 
        [0.95411, -3.45810, 2.95256, -1.35470, 0.07931], 
        [1.05578, -3.52403, 3.02209, -1.63058, 0.24777], 
        [0.96178, -3.43787, 2.80047, -1.59267, 0.26869], 
        [0.92160, -3.17884, 2.39690, -1.30318, 0.20160], 
        [1.60887, -1.68050, -0.31117, 0.56459, -0.15294], 
        [1.10656, -3.48994, 2.79927, -1.43087, 0.20257], 
        [0.96229, -3.38589, 2.66366, -1.50367, 0.24946], 
        [0.33899, -3.11338, 3.35701, -2.01792, -0.03811],
        [0.43213, -3.13001, 3.05479, -1.45176, -0.24947],
        [0.42919, -4.88411, 9.57678, -9.24289, 2.51916], 
        [0.43282, -3.12934, 3.04872, -1.43479, -0.25474], 
        [1.60197, -1.80486, -0.37900, 0.72207, -0.20484], 
        [0.33899, -3.11338, 3.35701, -2.01792, -0.03811], 
        [0.33899, -3.11338, 3.35701, -2.01792, -0.03811], 
        [0.33899, -3.11338, 3.35701, -2.01792, -0.03811], 
        [0.57617, -3.72075, 4.39869, -2.57369, 0.10102], 
        [0.33638, -3.34851, 4.17646, -3.10417, 0.32935], 
        [0.57617, -3.72075, 4.39869, -2.57369, 0.10102], 
        [0.33899, -3.11338, 3.35701, -2.01792, -0.03811], 
        [0.34355, -3.40385, 4.34820, -3.26853, 0.41553], 
        [0.57617, -3.72075, 4.39869, -2.57369, 0.10102], 
        [0.66874, -3.67737, 3.84550, -1.77616, -0.13769], 
        [0.33899, -3.11338, 3.35701, -2.01792, -0.03811], 
        [0.55123, -3.44308, 3.61405, -1.78572, -0.15201], 
        [0.32814, -3.20725, 3.22969, -1.36769, -0.81739], 
        [0.57049, -3.79984, 4.25538, -1.87362, -0.62622], 
        [0.26101, -2.53974, 1.63454, -0.21157, -0.66549], 
        [0.43171, -2.46496, 1.25461, 0.36690, -0.80127], 
        [0.32814, -3.20725, 3.22969, -1.36769, -0.81739], 
        [0.32814, -3.20725, 3.22969, -1.36769, -0.81739], 
        [0.32814, -3.20725, 3.22969, -1.36769, -0.81739], 
        [0.66316, -3.75896, 3.67693, -1.03117, -0.84256], 
        [0.57049, -3.79984, 4.25538, -1.87362, -0.62622], 
        [0.33527, -3.48692, 4.20858, -2.64340, -0.35546], 
        [0.32814, -3.20725, 3.22969, -1.36769, -0.81739], 
        [0.28043, -2.49033, 1.53980, -0.09926, -0.68403], 
        [0.57049, -3.79984, 4.25538, -1.87362, -0.62622], 
        [0.33518, -3.42262, 3.96328, -2.20298, -0.61986], 
        [0.32814, -3.20725, 3.22969, -1.36769, -0.81739], 
        [0.42487, -3.20974, 2.89721, -0.75258, -0.98259], 
        [0.42540, -3.21679, 2.86907, -0.62628, -1.09333], 
        [0.54655, -3.51799, 3.39128, -0.91567, -0.97112], 
        [0.32814, -3.20725, 3.22969, -1.36769, -0.81739], 
        [0.27015, -2.47936, 1.53752, -0.13967, -0.66166], 
        [0.41712, -2.56402, 1.22219, 1.02751, -1.56804], 
        [0.23548, -2.63001, 1.65498, 0.16117, -1.37247], 
        [0.31841, -4.56386, 8.63979, -8.41411, 1.91532], 
        [0.41712, -2.56402, 1.22219, 1.02751, -1.56804], 
        [0.41712, -2.56402, 1.22219, 1.02751, -1.56804], 
        [0.26294, -2.64669, 1.28364, 1.08209, -1.76828], 
        [0.30963, -2.40052, 1.28932, 0.52802, -1.33825], 
        [0.19868, -1.78301, 0.84573, 0.19455, -0.95628], 
        [0.19868, -1.78301, 0.84573, 0.19455, -0.95628], 
        [0.10922, -1.82977, 0.95797, 0.00543, -1.13850]
        ])

    bands = [[[412,443,490,510],[555,670]],
            [[412,442,488,531],[554,667]],
            [[412,442,490,510],[560,665]], 
            [[412,443,490,520],[565,670]], 
            [[412,443,490,530],[565,674]], 
            [[412,443,490,510],[555,665]], 
            [[412,443,490,510],[555,678]], 
            [[412,443,490,510],[555,670]], 
            [[413,443,490,510],[560,665]], 
            [[412,443,490,516],[565,667]], 
            [[412,443,490,510],[555,660]], 
            [[408,443,485,520],[570,615]], 
            [[412,443,490,520],[565,650]], 
            [[416,444,490,513],[553,668]], 
            [[412,443,490,510],[555,670]], 
            [[412,443,490,555],[660,680]], 
            [[412,443,490,520],[565,666]], 
            [[424,445,489,513],[554,672]], 
            [[412,443,490,510],[555]],
            [[413,443,490,510],[560]],
            [[412,442,488,531],[554]],
            [[412,442,490,510],[560]],
            [[412,443,490,555],[660]],
            [[412,443,490,510],[555]],
            [[412,443,490,510],[555]],
            [[412,443,490,510],[555]],
            [[412,443,490,520],[565]],
            [[424,445,489,513],[554]],
            [[412,443,490,520],[565]],
            [[412,443,490,510],[555]],
            [[416,444,490,513],[553]],
            [[412,443,490,520],[565]],
            [[408,443,485,520],[570]],
            [[412,443,490,510],[555]],
            [[412,443,490,516],[565]],
            [[443,490,510],[555]],
            [[443,490,520],[565]],
            [[410,443,486],[551]],
            [[412,443,490],[565]],
            [[443,490,510],[555]],
            [[443,490,510],[555]],
            [[443,490,510],[555]],
            [[443,485,520],[570]],
            [[443,490,520],[565]],
            [[444,490,513],[553]],
            [[443,490,510],[555]],
            [[412,443,490],[555]],
            [[443,490,520],[565]],
            [[445,490,513],[554]],
            [[443,490,510],[555]],
            [[442,490,510],[560]],
            [[443,490,510],[560]],
            [[443,490,516],[565]],
            [[443,490,510],[555]],
            [[412,442,488],[554]],
            [[443,490],[565]],
            [[443,486],[551]],
            [[443,520],[550]],
            [[443,490],[565]],
            [[443,490],[565]],
            [[442,488],[554]],
            [[443,482],[561]],
            [[443],[565]],
            [[443],[565]],
            [[446],[557]]]

    selected_model_idx = np.where(algorithm_names==algorithm_name)[0][0]

    wavelengths1, wavelengths2 = bands[selected_model_idx]

    a = A[selected_model_idx]

    selected_wavelengths1, selected_band_idx1 = np.array([find_closest(wavelengths, wl) for wl in wavelengths1]).T 
    selected_wavelengths2, selected_band_idx2 = np.array([find_closest(wavelengths, wl) for wl in wavelengths2]).T

    X = np.max(R_rs.T[selected_band_idx1], axis=0) / np.mean(R_rs.T[selected_band_idx2], axis=0)

    log10_Chl = a[0] + a[1]*X + a[2]*X**2 + a[3]*X**3 + a[4]*X**4

    Chl = 10**(log10_Chl)

    return Chl
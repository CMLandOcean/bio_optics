# -*- coding: utf-8 -*-
#  Copyright 2023 
#  Center for Global Discovery and Conservation Science, Arizona State University
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
#
# Translated to Python by:
#  Marcel König, mkoenig3 AT asu.edu 
#  Phillip Noel, pnoel1 AT asu.edu
#
# bio_optics
#  This code base builds on the extensive work of many researchers. For example, models were developed by Albert & Mobley [1] and Gege [2]; 
#  and the methodology was mainly developed by Gege [3,4,5] and Albert & Gege [6]. Please give proper attribution when using this code for publication.
#  A former version of this code base was developed in the course of the CarbonMapper Land and Ocean Program [7].
#
#  When using this code, please use the following citation:
#
#  König, M., Noel, P., Hondula. K.L., Jamalinia, E., Dai, J., Vaughn, N.R., Asner, G.P. (2023): bio_optics python package (Version x) [Software]. Available from https://github.com/CMLandOcean/bio_optics
#
# [1] Albert & Mobley (2003): An analytical model for subsurface irradiance and remote sensing reflectance in deep and shallow case-2 waters. [10.1364/OE.11.002873]
# [2] Gege (2012): Analytic model for the direct and diffuse components of downwelling spectral irradiance in water. [10.1364/AO.51.001407]
# [3] Gege (2004): The water color simulator WASI: an integrating software tool for analysis and simulation of optical in situ spectra. [10.1016/j.cageo.2004.03.005]
# [4] Gege (2014): WASI-2D: A software tool for regionally optimized analysis of imaging spectrometer data from deep and shallow waters. [10.1016/j.cageo.2013.07.022]
# [5] Gege (2021): The Water Colour Simulator WASI. User manual for WASI version 6. 
# [6] Gege & Albert (2006): A Tool for Inverse Modeling of Spectral Measurements in Deep and Shallow Waters. [10.1007/1-4020-3968-9_4]
# [7] König et al. (2023): WaterQuality python package (Version 1.2.0) [Software]. Available from https://github.com/CMLandOcean/WaterQuality. [10.5281/zenodo.7967294]


import numpy as np


def h_C(wavelengths=np.arange(400,800), fwhm=25, lambda_C=685):
    """
    Gaussian emission function of chloropyhll fluorescence [nm-1] according to Eq. 4 in [1]

    [1] Mobley (2024): https://www.oceanopticsbook.info/view/scattering/level-2/chlorophyll-fluorescence

    Args:
        wavelengths (_type_, optional): Wavelengths [nm]. Defaults to np.arange(400,800).
        fwhm (int, optional): full width at half maximum. Defaults to 25 [nm].
        lambda_C (int, optional): Wavelength of maximum emission [nm]. Defaults to 685.

    Returns:
        h_C: Gaussian emission function of chloropyhll fluorescence [nm-1]
    """
    h_C = np.sqrt((4 * np.log(2)) / np.pi) * 1/fwhm * np.exp(-4 * np.log(2) * ((wavelengths - lambda_C) / 25)**2)
    return h_C


def h_C_double(W=0.75, wavelengths=np.arange(400,800), fwhm1=25, fwhm2=50, lambda_C1=685, lambda_C2=730):
    """
    Emission function of chlorophyll fluorescence as weighted sum of two Gaussians accorting to Eq. 6 in [1]

    [1] Mobley (2024): https://www.oceanopticsbook.info/view/scattering/level-2/chlorophyll-fluorescence

    Args:
        W (float, optional): Weight of the first Gaussian. Defaults to 0.75, which results in the second peak height to be 0.2 * the first one.
        wavelengths (_type_, optional):  Wavelengths [nm]. Defaults to np.arange(400,800).
        fwhm1 (int, optional): full width at half maximum of peak 1 [nm]. Defaults to 25.
        fwhm2 (int, optional): full width at half maximum of peak 2 [nm]. Defaults to 50.
        lambda_C1 (int, optional): Wavelength of maximum emission of peak 1 [nm]. Defaults to 685.
        lambda_C2 (int, optional): Wavelength of maximum emission of peak 2 [nm]. Defaults to 730.

    Returns:
        h_C_double: Gaussian emission function of chloropyhll fluorescence [nm-1]
    """
    h_C_double = W * h_C(wavelengths=wavelengths, fwhm=fwhm1, lambda_C=lambda_C1) + (1-W) * h_C(wavelengths=wavelengths, fwhm=fwhm2, lambda_C=lambda_C2)
    return h_C_double


def Rrs_fl(wavelengths=np.arange(400,800), 
           L_fl_lambda0=0.001, 
           W=0.75, 
           fwhm1=25, 
           fwhm2=50, 
           lambda_C1=685, 
           lambda_C2=730, 
           double=False, 
           h_C_res=[]):
    """
    Fluorescence reflectance accounting for Chl a pigment fluorescence as presented in Eq. 8 in 
    Groetsch et al. (2020) [1] following Eq. 7.36 in Gilerson & Huot (2017) [2, 3].
    Updated following Mobley [4], L_fl_lambda0 is probably not really radiance but is rather treated as a scaler for the peak height.

    [1] Groetsch et al. (2020): Exploring the limits for sky and sun glint correction of hyperspectral above-surface reflectance observations. [10.1364/AO.385853]
    [2] Gilerson & Huot (2017): Sun-induced chlorophyll-a fluorescence, pp. 189-231 in [3].
    [3] Mishra et al. (2017): Bio-optical Modeling and Remote Sensing of Inland Waters.
    [4] Mobley (2024): https://www.oceanopticsbook.info/view/scattering/level-2/chlorophyll-fluorescence

    Args:
        wavelengths: wavelengths [nm]. Defaults to np.arange(400,800).
        L_fl_lambda0: Fluorescence radiance at lambda0 [W m-2 nm-1 sr-1]. Defaults to 0.001.
        W (float, optional): Weight of the first Gaussian. Defaults to 0.75, which results in the second peak height to be 0.2 * the first one.
        wavelengths (_type_, optional):  Wavelengths [nm]. Defaults to np.arange(400,800).
        fwhm1 (int, optional): full width at half maximum of peak 1 [nm]. Defaults to 25.
        fwhm2 (int, optional): full width at half maximum of peak 2 [nm]. Defaults to 50.
        lambda_C1 (int, optional): Wavelength of maximum emission of peak 1 [nm]. Defaults to 685.
        lambda_C2 (int, optional): Wavelength of maximum emission of peak 2 [nm]. Defaults to 730.

    Returns:
        Fluorescence radiance reflectance [sr-1]
    """
    if len(h_C_res)==0:
        if double:
            Rrs_fl = L_fl_lambda0 * h_C_double(W=W, wavelengths=wavelengths, fwhm1=fwhm1, fwhm2=fwhm2, lambda_C1=lambda_C1, lambda_C2=lambda_C2)
        else:
            Rrs_fl = L_fl_lambda0 * h_C(wavelengths=wavelengths, fwhm=fwhm1, lambda_C=lambda_C1)
    else:
        Rrs_fl = L_fl_lambda0 * h_C_res

    return Rrs_fl


def Rrs_fl_phycocyanin(L_fl_phycocyanin=0.001, wavelengths=np.arange(400,800), fwhm=20, lambda_C=644, h_C_phycocyanin_res=[]):
    """
    Fluorescence of phycocyanin (cyano red)

    Args:
        wavelengths: wavelengths [nm]. Defaults to np.arange(400,800).
        fwhm (int, optional): full width at half maximum of peak 1 [nm]. Defaults to 20.
        lambda_C (int, optional): Wavelength of maximum emission of peak [nm]. Defaults to 644.

    Returns:
       Fluorescence radiance reflectance due to phycocyanin [sr-1]
    """
    if len(h_C_phycocyanin_res)==0:
        Rrs_fl_phycocyanin = L_fl_phycocyanin * h_C(wavelengths=wavelengths, fwhm=fwhm, lambda_C=lambda_C)
    else:
        Rrs_fl_phycocyanin = L_fl_phycocyanin * h_C_phycocyanin_res

    return Rrs_fl_phycocyanin


def Rrs_fl_phycoerythrin(L_fl_phycoerythrin=0.001, wavelengths=np.arange(400,800), fwhm=20, lambda_C=573, h_C_phycoerythrin_res=[]):
    """
    Fluorescence of phycoerythrin (cyano blue)

    Args:
        wavelengths: wavelengths [nm]. Defaults to np.arange(400,800).
        fwhm (int, optional): full width at half maximum of peak 1 [nm]. Defaults to 20.
        lambda_C (int, optional): Wavelength of maximum emission of peak [nm]. Defaults to 573.

    Returns:
       Fluorescence radiance reflectance due to phycoerythrin [sr-1]
    """
    if len(h_C_phycoerythrin_res)==0:
        Rrs_fl_phycoerythrin = L_fl_phycoerythrin * h_C(wavelengths=wavelengths, fwhm=fwhm, lambda_C=lambda_C)
    else:
        Rrs_fl_phycoerythrin = L_fl_phycoerythrin * h_C_phycoerythrin_res
        
    return Rrs_fl_phycoerythrin
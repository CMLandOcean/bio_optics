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


import numpy as np
from . utils import find_closest


def ndi(band1, band2):
    """
    Normalized Difference Index (NDI)

    :param band1: first band
    :param band2: second band
    :return: NDI
    """
    return (band1 - band2) / (band1 + band2)


def ndwi(spectrum, wavelengths, green=559, nir=864):
    """
    Normalized Difference Water Index (NDWI) for water body delineation defined as (GREEN-NIR)/(GREEN+NIR) [1]
    
    [1] McFeeters (1996) [doi.org/10.1080/01431169608948714]

    Args:
        spectrum: spectrum or image (bands on first axis)
        wavelengths: wavelengths of spectrum [nm]
        green: green wavelength [nm]
        nir: nir wavelength [nm]
    Return: NDWI
    """
    band1 = spectrum[find_closest(wavelengths, green)[1]]
    band2 = spectrum[find_closest(wavelengths, nir)[1]]
    return ndi(band1, band2)


def mndwi(spectrum, wavelengths, green=559, swir=1240):
    """
    Modified Normalized Difference Water Index (MNDWI) for water body delineation defined as (GREEN-SWIR)/(GREEN+SWIR) [1] 
    
    [1] Xu (2006) [doi.org/10.1080/01431160600589179].

    Args:
        spectrum: spectrum or image (bands on first axis)
        wavelengths: wavelengths of spectrum [nm]
        green: green wavelength [nm]
        swir: swir wavelength [nm]
    Return: MNDWI
    """
    band1 = spectrum[find_closest(wavelengths, green)[1]]
    band2 = spectrum[find_closest(wavelengths, swir)[1]]
    return ndi(band1, band2)


def hdwi(spectrum, wavelengths, red=650, nir1=700, nir2=850):
    """
    Hyperspectral Difference Water Index (HDWI) [1]
    
    [1] Xie et al. (2014) [doi.org/10.1117/1.JRS.8.]

    Args:
        spectrum: spectrum or image (bands on first axis)
        wavelengths: wavelengths of spectrum [nm]
        red: green wavelength [nm]
        nir1: nir1 wavelength [nm]
        nir2: nir2 wavelength [nm]
    Return: HDWI
    """
    band1 = spectrum[find_closest(wavelengths, red)[1]]
    band2 = spectrum[find_closest(wavelengths, nir1)[1]]
    band3 = spectrum[find_closest(wavelengths, nir2)[1]]

    return (np.sum(spectrum[band1:band2], axis=0) - np.sum(spectrum[band2:band3], axis=0)) / (np.sum(spectrum[band1:band2], axis=0) + np.sum(spectrum[band2:band3], axis=0))


def ndti(spectrum, wavelengths, green=545, red=650):
    """
    Normalized Difference Turbidity Index (NDTI) defined as (RED-GREEN)/(RED+GREEN) to estimate different degrees of turbidity [1].
    Can be extended by the NDVI and NDPI and a decision tree presented in [1] to classify pond turbidity.
    
    [1] Lacaux et al. (2007) [doi.org/10.1016/j.rse.2006.07.012].

    Args:
        spectrum: spectrum or image (bands on first axis)
        wavelengths: wavelengths of spectrum [nm]
        green: green wavelength [nm]
        red: red wavelength [nm]
    Return: NDTI
    """
    band1 = spectrum[find_closest(wavelengths, red)[1]]
    band2 = spectrum[find_closest(wavelengths, green)[1]]
    return ndi(band1, band2)


def ndpi(spectrum, wavelengths, green=545, mir=1665):
    """
    Normalized Difference Pond Index (NDPI) defined as (MIR-GREEN)/(MIR+GREEN) to map ponds [1].
    Can be extended by the NDVI and NDTI and a decision tree presented in [1] to classify pond turbidity.
    
    [1] Lacaux et al. (2007) [doi.org/10.1016/j.rse.2006.07.012].

    Args:
        spectrum: spectrum or image (bands on first axis)
        wavelengths: wavelengths of spectrum [nm]
        green: green wavelength [nm]
        mir: mir wavelength [nm]
    Return: NDPI
    """
    band1 = spectrum[find_closest(wavelengths, mir)[1]]
    band2 = spectrum[find_closest(wavelengths, green)[1]]
    return ndi(band1, band2)


def awei(R, wavelengths, band1=485, band2=560, band3=830, band4=1650, band5=2215, shade=False):
    """
    Automated Water Extraction Index (AWEI) [1] 
    to maximize separability of water and nonwater pixels through band differencing, addition and applying different coefficients based on Landsat 5 TM bands.
    Particular emphasis was given to the enhancement of the separability of water and dark surfaces such as shadow and built-up structures that are often difficult 
    to distinguish due to similarities in reflectance patterns [1].

    [1] Feyisa et al. (2014) [doi.org/10.1016/j.rse.2013.08.029]
    
    Args:
        R (_type_): Spectrum in units of Reflectance [-]
        wavelengths (_type_): _description_
        band1 (int, optional): _description_. Defaults to 485.
        band2 (int, optional): _description_. Defaults to 560.
        band3 (int, optional): _description_. Defaults to 830.
        band4 (int, optional): _description_. Defaults to 1650.
        band5 (int, optional): _description_. Defaults to 2215.
    """
    R_b1 = R[find_closest(wavelengths, band1)[1]]
    R_b2 = R[find_closest(wavelengths, band2)[1]]
    R_b4 = R[find_closest(wavelengths, band3)[1]]
    R_b5 = R[find_closest(wavelengths, band4)[1]]
    R_b7 = R[find_closest(wavelengths, band5)[1]]

    if shade==False:
        return 4 * (R_b2 - R_b5) - (0.25 * R_b4 + 2.75 * R_b7)
    else:
        return R_b1 + 2.5 * R_b2 - 1.5 * (R_b4 + R_b5) - 0.25 * R_b7

    
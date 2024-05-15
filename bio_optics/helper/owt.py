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

import os
import numpy as np
import pandas as pd
from . resampling import resample_srf
from . utils import find_closest
from . indices import ndi


# get absolute path to data folder
data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')


def avw(R_rs, wavelengths, sensor='hyperspectral'):
    """
    Apparent visible wavelength (AVW) [1]
    "The Apparent Visible Wavelength (AVW) is calculated as the weighted harmonic mean of all R_rs wavelengths, 
    weighted as a function of the relative intensity of Rrs at each wavelength" [1].
    Polynomial coefficients for multispectral sensors are from [2].

    [1] Vandermeulen (2022) [https://oceancolor.gsfc.nasa.gov/atbd/avw/].
    [2] Vandermeulen (no year) [https://oceancolor.gsfc.nasa.gov/resources/atbd/avw/#:~:text=The%20AVW%20is%20an%20optical,2020)]
    
    :param wavelengths: All available wavelengths between 400–700 nm
    :param R_rs: R_rs at respective wavelengths
    :param sensor: sensor name if not hyperspectral. Defaults to 'hyperspectral'. Options are 'MODIS-Aqua', 'MODIS-Terra', 'OLCI-S3A', OLCI-S3B', 'MERIS', 'SeaWiFS', 'HawkEye', 'OCTS', 'GOCI', 'VIIRS-SNPP', VIIRS-JPSS1', 'CZCS', 'MSI-S2A', 'MSI-S2B', 'OLI', 'SuperDove'.
    :return:
    """ 
    avw = np.sum(R_rs[(wavelengths>=400) & (wavelengths<=700)], axis=0) / np.sum(R_rs[(wavelengths>=400) & (wavelengths<=700)] / wavelengths[(wavelengths>=400) & (wavelengths<=700)], axis=0)
    
    if sensor=='hyperspectral':
        return avw
        
    else:
        avw_coefficients = pd.read_csv(os.path.join(data_dir, 'avw_coefficients.txt'), skiprows=5)
        c0, c1, c2, c3, c4, c5 = avw_coefficients[avw_coefficients['Sensor']==sensor].values[0][1:]
        avw = c0*(avw**5) + c1*(avw**4) + c2*(avw**3) + c3*(avw**2) + c4*(avw) + c5
        return avw
    
    
def qwip(avw):
    """
    Quality Water Index Polynomial (QWIP) [1] 

    [1] Dierssen et al. (2022) [doi.org/10.3389/frsen.2022.869611].
    [2] Vandermeulen (2022) [https://oceancolor.gsfc.nasa.gov/atbd/avw/].
    
    :param avw: Apparent visible wavelength (AVW) [2].
    :return: QWIP
    """
    p = np.array([-8.399885e-9, 1.715532e-5, -1.301670e-2, 4.357838e0, -5.449532e2])    
    qwip = p[0]*avw**4 + p[1]*avw**3 + p[2]*avw**2 + p[3]*avw + p[4]
    return qwip
    
    
def qwip_score(R_rs, wavelengths, sensor='hyperspectral'):
    """
    Quality Water Index Polynomial score (QWIP score) [1].

    [1] Dierssen et al. (2022) [doi.org/10.3389/frsen.2022.869611].
    
    :param wavelengths: All available wavelengths between 400 nm and 700 nm 
    :param R_rs: R_rs at respective wavelengths
    :param sensor: sensor name if not hyperspectral. Defaults to 'hyperspectral'. Options are 'MODIS-Aqua', 'MODIS-Terra', 'OLCI-S3A', OLCI-S3B', 'MERIS', 'SeaWiFS', 'HawkEye', 'OCTS', 'GOCI', 'VIIRS-SNPP', VIIRS-JPSS1', 'CZCS', 'MSI-S2A', 'MSI-S2B', 'OLI', 'SuperDove'.
    :return: QWIP score
    """
    qwip_score = ndi(R_rs[find_closest(wavelengths, 665)[1]], R_rs[find_closest(wavelengths, 492)[1]]) - qwip(avw(R_rs=R_rs, wavelengths=wavelengths, sensor=sensor))
    return qwip_score


def balasubramanian(R_rs, wavelengths):
    """
    Optical water type classification [1].

    [1] Balasubramanian et al. (2020) [doi:10.1016/j.rse.2020.111768]

    Args:
        spectrum: spectrum or image (bands on first axis)
        wavelengths: wavelengths of spectrum [nm]
    Return:
        One of three water types (1: Blue-green water, 2: Green water, 3: Brown water)
    """
    Rrs_665 = R_rs[find_closest(wavelengths, 665)[1]]
    Rrs_492 = R_rs[find_closest(wavelengths, 492)[1]]
    Rrs_560 = R_rs[find_closest(wavelengths, 560)[1]]
    Rrs_740 = R_rs[find_closest(wavelengths, 740)[1]]

    return np.where(((Rrs_665 < Rrs_560) & (Rrs_665 > Rrs_492)), 2,
                np.where(((Rrs_665 > Rrs_560) & (Rrs_740 > 0.01)), 3,
                    np.where((Rrs_560 < Rrs_492), 1, 
                        np.where(np.isnan(Rrs_665), np.nan, 2))))


def jiang(R_rs, wavelengths):
    """
    Optical water type classification [1]. 
    
    [1] Jiang et al. (2021) [10.1016/j.rse.2021.112386].

    Args:
        R_rs: spectrum as np.array in units of R_rs [1/sr]
        wavelengths: wavelengths of spectrum [nm]
    Return:
        One of four water types as int (1: clear waters, 2: moderately turbid waters, 3: highly turbid waters, 4: extremely turbid waters)
    """

    # Find the closest wavelengths to the wavelengths suggested in Jiang et al. (2021)
    Rrs_490 = R_rs[find_closest(wavelengths, 490)[1]]
    Rrs_560 = R_rs[find_closest(wavelengths, 560)[1]]
    Rrs_620 = R_rs[find_closest(wavelengths, 620)[1]]
    Rrs_754 = R_rs[find_closest(wavelengths, 754)[1]]
    
    return np.where((Rrs_490 > Rrs_560), 1,
                    np.where((Rrs_490 > Rrs_620), 2,
                             np.where(((Rrs_754 > Rrs_490) & (Rrs_754 > 0.01)), 4, 
                                      np.where(np.isnan(Rrs_490), np.nan, 3))))



def forel_ule(R_rs, wavelengths, kind='slinear'):
    """
    Discrete Forel-Ule scale [1,2,3,4]

    [1] Wernand & Woerd (2010): Spectral analysis of the Forel-Ule ocean colour comparator scale [10.2971/jeos.2010.10014s]
    [2] Novoa et al. (2013): The Forel-Ule scale revisited spectrally: preparation protocol, transmission measurements and chromaticity [10.2971/jeos.2013.13057]
    [3] Wernand et al. (2013): MERIS-based ocean colour classification with the discrete Forel-Ule scale [10.5194/os-9-477-2013]
    [4] Wang et al. (2021): A dataset of remote-sensed Forel-Ule Index for global inland waters during 2000–2018. [10.1038/s41597-021-00807-z]

    :param R_rs: array of remote sensing reflectance, if more than 1D, first axis needs to be bands
    :param wavelengths: corresponding wavelengths [nm]
    :param kind: specifies kind of interpolation, parameter for interp1d, default: 'slinear'

    """
    fu_scale = pd.read_csv(os.path.join(data_dir, 'fu_scale.txt'), skiprows=7)
    cie = pd.read_csv(os.path.join(data_dir, 'cie.txt'), skiprows=5, sep='\t')

    xyz =resample_srf(srf_wavelengths=cie.iloc[:,0].values, 
                      srf_factors = cie.iloc[:,1:].values, 
                      input_wavelengths = wavelengths, 
                      input_spectrum = R_rs,
                      kind=kind) 

    x = xyz[0] / xyz.sum(axis=0)
    y = xyz[1] / xyz.sum(axis=0)

    alpha = np.degrees(np.arctan2(y-(1/3),x-(1/3)))
    alpha = np.where(alpha<0, 180-alpha, alpha)

    alpha_lut = fu_scale["alpha_Novoa"].values[:21]
    alpha_lut = np.where(np.isnan(alpha_lut), 0, alpha_lut)

    if len(R_rs.shape)==1:
        idx = np.where(alpha_lut-alpha < 0, -1e10, alpha_lut-alpha).argmin(axis=0)
    else:
        # if R_rs has more than 1 dimension
        alpha_lut = np.broadcast_to(alpha_lut, R_rs.T.shape[:-1] + (21,)).T
        idx = np.where(alpha_lut-alpha < 0, -1e10, alpha_lut-alpha).argmin(axis=0)

    fu_class = idx + 1
    dominant_wavelength = fu_scale["dominant_wl"].values[idx]
    
    return fu_class, dominant_wavelength
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
from . indices import ndi


def avw(R_rs, wavelengths):
    """
    Apparent visible wavelength (AVW) [1]
    "The Apparent Visible Wavelength (AVW) is calculated as the weighted harmonic mean of all R_rs wavelengths, 
    weighted as a function of the relative intensity of Rrs at each wavelength" [1].

    [1] Vandermeulen (2022) [https://oceancolor.gsfc.nasa.gov/atbd/avw/].
    
    :param wavelengths: All available wavelengths between 400–700 nm
    :param R_rs: R_rs at respective wavelengths
    :return:
    """ 
    avw = np.sum(R_rs[(wavelengths>=400) & (wavelengths<=700)], axis=0) / np.sum(R_rs[(wavelengths>=400) & (wavelengths<=700)] / wavelengths[(wavelengths>=400) & (wavelengths<=700)], axis=0)
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
    
    
def qwip_score(R_rs, wavelengths):
    """
    Quality Water Index Polynomial score (QWIP score) [1].

    [1] Dierssen et al. (2022) [doi.org/10.3389/frsen.2022.869611].
    
    :param wavelengths: All available wavelengths between 400–700 nm 
    :param R_rs: R_rs at respective wavelengths
    :return: QWIP score
    """
    qwip_score = ndi(R_rs[find_closest(wavelengths, 665)[1]], R_rs[find_closest(wavelengths, 492)[1]]) - qwip(avw(R_rs, wavelengths))
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
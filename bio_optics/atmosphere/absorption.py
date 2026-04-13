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
#  Marcel König, mkoenig3 AT asu.edu / marcel.koenig AT brockmann-consult.de
#
# bio_optics
#  This code base builds on the extensive work of many researchers. For example, models were developed by Albert & Mobley [1] and Gege [2]; 
#  and the methodology was mainly developed by Gege [3,4,5] and Albert & Gege [6]. Please give proper attribution when using this code for publication.
#  A former version of this code base was developed in the course of the CarbonMapper Land and Ocean Program [7]
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
import pandas as pd
from .. helper import resampling


def a_oz(wavelengths=np.arange(400,800), a_oz_res=[]):
    """
    Spectral absorption coefficient of ozone resampled to sensor's spectral sampling rate.
    
    :param wavelengths: wavelengths to resample a_oz to, default: np.arange(400,800)
    :param a_oz_res: optional, preresampling a_oz before inversion saves a lot of time.
    :return: spectral absorption coefficient of ozone
    """
    if len(a_oz_res)==0:
        a_oz = resampling.resample_a_oz(wavelengths=wavelengths)
    else:
        a_oz = a_oz_res
  
    return a_oz


def a_wv(wavelengths=np.arange(400,800), a_wv_res=[]):
    """
    Spectral absorption coefficient of water vapor resampled to sensor's spectral sampling rate.
    
    :param wavelengths: wavelengths to resample a_wv to, default: np.arange(400,800)
    :param a_wv_res: optional, preresampling a_wv before inversion saves a lot of time.
    :return: spectral absorption coefficient of water vapor
    """
    if len(a_wv_res)==0:
        a_wv = resampling.resample_a_wv(wavelengths=wavelengths)
    else:
        a_wv = a_wv_res
  
    return a_wv


def a_ox(wavelengths=np.arange(400,800), a_ox_res=[]):
    """
    Spectral absorption coefficient of oxygen resampled to sensor's spectral sampling rate.
    
    :param wavelengths: wavelengths to resample a_ox to, default: np.arange(400,800)
    :param a_ox_res: optional, preresampling a_ox before inversion saves a lot of time.
    :return: spectral absorption coefficient of oxygen
    """
    if len(a_ox_res)==0:
        a_ox = resampling.resample_a_ox(wavelengths=wavelengths)
    else:
        a_ox = a_ox_res
        
    return a_ox
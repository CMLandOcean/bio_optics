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
#
# WaterQuality
#  Code is provided to Planet, PBC as part of the CarbonMapper Land and Ocean Program.
#  It builds on the extensive work of many researchers. For example, models were developed  
#  by Albert & Mobley [1] and Gege [2]; the methodology was mainly developed 
#  by Gege [3,4,5] and Albert & Gege [6].
#
#  Please give proper attribution when using this code for publication:
#
#  König, M., Hondula. K.L., Jamalinia, E., Dai, J., Vaughn, N.R., Asner, G.P. (2023): WaterQuality python package (Version x) [Software]. Available from https://github.com/CMLandOcean/WaterQuality
#
# [1] Albert & Mobley (2003): An analytical model for subsurface irradiance and remote sensing reflectance in deep and shallow case-2 waters. [10.1364/OE.11.002873]
# [2] Gege (2012): Analytic model for the direct and diffuse components of downwelling spectral irradiance in water. [10.1364/AO.51.001407]
# [3] Gege (2004): The water color simulator WASI: an integrating software tool for analysis and simulation of optical in situ spectra. [10.1016/j.cageo.2004.03.005]
# [4] Gege (2014): WASI-2D: A software tool for regionally optimized analysis of imaging spectrometer data from deep and shallow waters. [10.1016/j.cageo.2013.07.022]
# [5] Gege (2021): The Water Colour Simulator WASI. User manual for WASI version 6. 
# [6] Gege & Albert (2006): A Tool for Inverse Modeling of Spectral Measurements in Deep and Shallow Waters. [10.1007/1-4020-3968-9_4]


import numpy as np
from .. helper import resampling


def R_rs_b(f_0 = 0,
           f_1 = 0,
           f_2 = 0,
           f_3 = 0,
           f_4 = 0,
           f_5 = 0,
           B_0 = 1/np.pi, 
           B_1 = 1/np.pi, 
           B_2 = 1/np.pi, 
           B_3 = 1/np.pi, 
           B_4 = 1/np.pi, 
           B_5 = 1/np.pi, 
           wavelengths=np.arange(400,800),
           R_i_b_res = []):
    """
    Radiance reflectance of benthic substrate [sr-1] as a mixture of up to 6 bottom types [1].
    
    [1] Gege, P. (2021): The Water Colour Simulator WASI. User manual for WASI version 6.
    
    :param f_0: fractional cover of bottom type 0
    :param f_1: fractional cover of bottom type 1
    :param f_2: fractional cover of bottom type 2
    :param f_3: fractional cover of bottom type 3
    :param f_4: fractional cover of bottom type 4
    :param f_5: fractional cover of bottom type 5
    :param B_0: proportion of radiation reflected towards the sensor from bottom type 0
    :param B_1: proportion of radiation reflected towards the sensor from bottom type 1
    :param B_2: proportion of radiation reflected towards the sensor from bottom type 2
    :param B_3: proportion of radiation reflected towards the sensor from bottom type 3
    :param B_4: proportion of radiation reflected towards the sensor from bottom type 4
    :param B_5: proportion of radiation reflected towards the sensor from bottom type 5
    :param wavelengths: wavelengths to resample R_i_b (albedo of bottom types 0..5) to
    :param R_i_b_res: optional, preresampling R_i_b before inversion saves a lot of time.
    :return: radiance reflectance of benthic substrate [sr-1]

    # Math: R_{rs}^b(\lambda) = \sum_{n=0}^{N-1}f_n * B_n * a_n(\lambda)
    """
    f_i = np.array([f_0,f_1,f_2,f_3,f_4,f_5])
    B_i = np.array([B_0,B_1,B_2,B_3,B_4,B_5])
    
    if len(R_i_b_res)==0:
        R_i_b = resampling.resample_R_i_b(wavelengths=wavelengths)
    else:
        R_i_b = R_i_b_res
    
    R_rs_b = 0
    for i in np.arange(R_i_b.shape[1]): R_rs_b += f_i[i] * B_i[i] * R_i_b.T[i]

    return R_rs_b

def dR_rs_b_div_df_i(i, 
           B_0 = 1/np.pi, 
           B_1 = 1/np.pi, 
           B_2 = 1/np.pi, 
           B_3 = 1/np.pi, 
           B_4 = 1/np.pi, 
           B_5 = 1/np.pi, 
           wavelengths=np.arange(400,800),
           R_i_b_res = []):
    """
    # Math: \frac{\partial}{\partial f_i} R_{rs}^b(\lambda) = \frac{\partial}{\partial f_i} \sum_{n=0}^{N-1}f_n * B_n * a_n(\lambda) = B_i * a_i(\lambda)
    """
    B_i = np.array([B_0,B_1,B_2,B_3,B_4,B_5])
    
    if len(R_i_b_res)==0:
        R_i_b = resampling.resample_R_i_b(wavelengths=wavelengths)
    else: 
        R_i_b = R_i_b_res
    
    dR_rs_b_div_df_i = B_i[i] * R_i_b.T[i]
    
    return dR_rs_b_div_df_i

def dR_rs_b_div_dB_i(i, 
           f_0 = 0,
           f_1 = 0,
           f_2 = 0,
           f_3 = 0,
           f_4 = 0,
           f_5 = 0,
           B_0 = 1/np.pi, 
           B_1 = 1/np.pi, 
           B_2 = 1/np.pi, 
           B_3 = 1/np.pi, 
           B_4 = 1/np.pi, 
           B_5 = 1/np.pi, 
           wavelengths=np.arange(400,800),
           R_i_b_res = []):
    """
    # Math: \frac{\partial}{\partial B_i} R_{rs}^b(\lambda) = \frac{\partial}{\partial B_i} \sum_{n=0}^{N-1}f_n * B_n * a_n(\lambda) = f_i * a_i(\lambda)
    """
    f_i = np.array([f_0,f_1,f_2,f_3,f_4,f_5])
    B_i = np.array([B_0,B_1,B_2,B_3,B_4,B_5])
    
    if len(R_i_b_res)==0:
        R_i_b = resampling.resample_R_i_b(wavelengths=wavelengths)
    else: 
        R_i_b = R_i_b_res
    
    dR_rs_b_div_dB_i = f_i[i] * R_i_b.T[i]
    
    return dR_rs_b_div_dB_i

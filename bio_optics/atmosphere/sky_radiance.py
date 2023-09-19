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


def L_s(g_dd, E_dd,
        g_dsr, E_dsr,
        g_dsa, E_dsa):
    """
    Sky radiance in W/m2 nm sr [1]
    
    "A parameterization similar to E_d is implemented for the sky radiance, L_s. The radiance downwelling from a part of the sky is treated 
    as a weighted sum of three wavelength dependent functions, E_dd, E_dsr and E_dsa.In contrast to E_d, the two diffuse components are treated 
    separately since Rayleigh scattering has a much stronger angle dependency than aerosol scattering. The parameters g_dd, g_dsr and g_dsa are 
    the intensities (in units of sr−1) of E_dd, E_dsr and E_dsa, respectively." [1]
    
    [1] Gege, P. (2021): The Water Colour Simulator WASI. User manual for WASI version 6.
    
    :return: sky radiance in W/m2 nm sr
    
    """
    L_s = g_dd  * E_dd + \
          g_dsr * E_dsr + \
          g_dsa * E_dsa
    
    return L_s

def d_LS_div_dg_dd(E_dd):
    return E_dd

def d_LS_div_dg_dsr(E_dsr):
    return E_dsr

def d_LS_div_dg_dsa(E_dsa):
    return E_dsa

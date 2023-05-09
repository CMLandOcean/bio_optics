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


def below2above(r_rs, zeta=0.52, Gamma=1.6):
    """
    Convert subsurface radiance reflectance to remote sensing reflectance after Lee et al. (1998) [1]
    as described in Giardino et al. (2019) [2].

    [1] Lee et al. (1998): Hyperspectral remote sensing for shallow waters: 1. A semianalytical model [10.1364/AO.37.006329]
    [2] Giardino et al. (2019): Imaging Spectrometry of Inland and Coastal Waters: State of the Art, Achievements and Perspectives [10.1007/s10712-018-9476-0]
    
    :param r_rs: subsurface radiance reflectance
    :return: remote sensing reflectance
    """
    return (zeta * r_rs) / (1 - Gamma * r_rs)
    

def above2below(R_rs, zeta=0.52, Gamma=1.6):
    """
    Convert remote sensing reflectance to subsurface radiance reflectance after Lee et al. (1998) [1]
    as described in Giardino et al. (2019) [2].

    [1] Lee et al. (1998): Hyperspectral remote sensing for shallow waters: 1. A semianalytical model [10.1364/AO.37.006329]
    [2] Giardino et al. (2019): Imaging Spectrometry of Inland and Coastal Waters: State of the Art, Achievements and Perspectives [10.1007/s10712-018-9476-0]
    
    :param R_rs: remote sensing reflectance
    :return: subsurface radiance reflectance
    """
    return R_rs / (zeta + Gamma * R_rs)
    
    
def snell(theta_inc, n1=1, n2=1.33):
    """
    Compute the refraction angle using Snell's Law.

    :param theta_inc: Incident angle [radians]
    :param n1: Refrective index of origin medium, default: 1 for air
    :param n2: Refrective index of destination medium, default: 1.33 for water
    :returns theta: refraction angle in radians.
    """
    return np.arcsin(n1 / n2 * np.sin(theta_inc))


def fresnel(theta_inc, n1=1, n2=1.33):
    """
    Fresnel reflectance for unpolarized incoming light for a horizontally flat air-water interface.

    :param theta_inc: Incident angle [radians]
    :param n1: Refrective index of origin medium, default: 1 for air
    :param n2: Refrective index of destination medium, default: 1.33 for water
    :return: Fresnel reflectance for unpolarized incoming light
    """
    theta_inc_w = snell(theta_inc, n1, n2)

    return (((np.sin(theta_inc - theta_inc_w)**2) / (np.sin(theta_inc + theta_inc_w)**2)) + ((np.tan(theta_inc - theta_inc_w)**2) / (np.tan(theta_inc + theta_inc_w)**2))) / 2
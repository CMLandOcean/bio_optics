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
from . import downwelling_irradiance


def L_s(wavelengths=np.arange(400,800), 
        theta_sun=np.radians(30), 
        P=1013.25, 
        AM=5, 
        RH=80, 
        H_oz=0.38, 
        WV=2.5, 
        alpha=1.317,
        beta=0.2606,
        g_dd=0.02,
        g_dsr=1/np.pi,
        g_dsa=1/np.pi,
        E_0_res=[],
        a_oz_res=[],
        a_ox_res=[],
        a_wv_res=[],
        E_dd_res=[],
        E_dsa_res=[],
        E_dsr_res=[]):
    """
    Sky radiance in W/m2 nm sr [1]
    
    "A parameterization similar to E_d is implemented for the sky radiance, L_s. The radiance downwelling from a part of the sky is treated 
    as a weighted sum of three wavelength dependent functions, E_dd, E_dsr and E_dsa.In contrast to E_d, the two diffuse components are treated 
    separately since Rayleigh scattering has a much stronger angle dependency than aerosol scattering. The parameters g_dd, g_dsr and g_dsa are 
    the intensities (in units of sr−1) of E_dd, E_dsr and E_dsa, respectively." [1]
    
    [1] Gege, P. (2021): The Water Colour Simulator WASI. User manual for WASI version 6.
    
    :param wavelengths: wavelengths to compute L_s for [nm], default: np.arange(400,800)
    :param theta_sun: sun zenith angle [radians], default: np.radians(30)
    :param P: atmospheric pressure [mbar], default: 1013.25
    :param AM: air mass type (1: open ocean .. 10: continental), default: 5
    :param RH: relative humidity [%], default: 80
    :param H_oz: ozone scale height [cm], default: 0.38
    :param WV: water vapor in units of precipitable water [cm], default: 2.5
    :param alpha: Angström exponent of aerosol optical thickness
    :param beta: turbidity coefficient of aerosol optical thickness
    :param g_dd: intensity of direct component of E_d [sr-1], default: 0.02
    :param g_dsr: intensity of Rayleigh scattering part of diffuse component of E_d [sr-1], default: 1/np.pi()
    :param g_dsa: intensity of aerosol scattering part of diffuse component of E_d [sr-1], default: 1/np.pi()
    :param E_0_res: optional, preresampling E_0 before inversion saves a lot of time.
    :param a_oz_res: optional, preresampling a_oz before inversion saves a lot of time.
    :param a_ox_res: optional, preresampling a_ox before inversion saves a lot of time.
    :param a_wv_res: optional, preresampling a_wv before inversion saves a lot of time.
    :param E_dd_res: optional, preresampling E_dd before inversion saves a lot of time.
    :param E_dsa_res: optional, preresampling E_dsa before inversion saves a lot of time.
    :param E_dsr_res: optional, preresampling E_dsr before inversion saves a lot of time.
    :return: sky radiance in W/m2 nm sr
    
    """
    L_s = g_dd * downwelling_irradiance.E_dd(wavelengths=wavelengths, theta_sun=theta_sun, P=P, AM=AM, RH=RH, H_oz=H_oz, WV=WV, alpha=alpha, beta=beta, E_0_res=E_0_res, a_oz_res=a_oz_res, a_ox_res=a_ox_res, a_wv_res=a_wv_res, E_dd_res=E_dd_res) + \
          g_dsr * downwelling_irradiance.E_dsr(wavelengths=wavelengths, theta_sun=theta_sun, P=P, AM=AM, RH=RH, H_oz=H_oz, WV=WV, E_0_res=E_0_res, a_oz_res=a_oz_res, a_ox_res=a_ox_res, a_wv_res=a_wv_res, E_dsr_res=E_dsr_res) + \
          g_dsa * downwelling_irradiance.E_dsa(wavelengths=wavelengths, theta_sun=theta_sun, P=P, AM=AM, RH=RH, H_oz=H_oz, WV=WV, alpha=alpha, E_0_res=E_0_res, a_oz_res=a_oz_res, a_ox_res=a_ox_res, a_wv_res=a_wv_res, E_dsa_res=E_dsa_res)
    
    return L_s
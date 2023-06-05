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
import pandas as pd
import math
from . import absorption
from .. helper import resampling


def M(theta_sun=np.radians(30), a=0.50572, b=6.07995, c=1.6364):
    """
    Atmospheric path length [1].
    
    "The numerical values used by Gregg and Carder [2] (a = 0.15, b = 3.885°, c = 1.253) were replaced 
    by updated values a = 0.50572, b = 6.07995°, c = 1.6364 from Kasten and Young (1989) [3]" [1]
    
    [1] Gege, P. (2021): The Water Colour Simulator WASI. User manual for WASI version 6.
    [2] Gregg and Carder (1990): A simple spectral solar irradiance model for cloudless maritime atmospheres.
    [3] Kasten and Young (1989): Revised optical air mass tables and approximation formula.
    
    :param theta_sun: sun zenith angle [radians], default: np.radians(30)
    :param a: numerical value from [2], default: 0.50572
    :param b: numerical value from [2], default: 6.07995°
    :param c: numerical value from [2], default: 1.6364
    :return: atmospheric path length
    """
    M = 1 / (np.cos(theta_sun) + a*(90 + b - np.degrees(theta_sun))**(-c))
    return M
    
    
def M_cor(theta_sun=np.radians(30), P=1013.25):
    """
    Atmospheric path length corrected for nonstandard atmospheric pressure [1].
    
    [1] Gege, P. (2021): The Water Colour Simulator WASI. User manual for WASI version 6.
    
    :param theta_sun: sun zenith angle [radians], default: np.radians(30)
    :param P: atmospheric pressure [mbar], default: 1013.25
    :return: atmospheric path length corrected for nonstandard atmospheric pressure
    """
    M_cor = M(theta_sun=theta_sun) * P / 1013.25
    return M_cor
    
    
def M_oz(theta_sun=np.radians(30)):
    """
    Atmospheric path length for ozone [1].
    
    [1] Gege, P. (2021): The Water Colour Simulator WASI. User manual for WASI version 6.
    
    :param theta_sun: sun zenith angle [radians], default: np.radians(30)
    :return: atmospheric path length for ozone
    """
    M_oz = 1.0035 / (math.cos(theta_sun)**2 + 0.007)**0.5
    return M_oz
    
    
def compute_beta(V=12, H_a=1):
    """
    Turbidity coefficient of aerosol optical thickness computed from V and H_a [1].
    
    [1] Gege, P. (2021): The Water Colour Simulator WASI. User manual for WASI version 6.
    
    :param V: horizontal visibility [km] (typically ranges from 8 to 24)
    :param H_a: aerosol scale height [km] (typically 1)
    :return: turbidity coefficient of aerosol optical thickness
    """
    beta = 3.91 * H_a / V
    return beta
    
    
def tau_a(wavelengths=np.arange(400,800), lambda_a=550, alpha=1.317, beta=0.2606, V=None, H_a=None):
    """
    Aerosol optical thickness [1]
    The turbidity coefficient beta can optionally be computed from V and H_a.
    
    [1] Gege, P. (2021): The Water Colour Simulator WASI. User manual for WASI version 6.
    
    :param wavelengths: wavelengths to compute tau_a for, default: np.arange(400,800)
    :param lambda_a: reference wavelength, default: 550
    :param alpha: Angström exponent determining wavelength dependency (typically ranges from 0.2 to 2 [1]), default: 1.317
    :param beta: turbidity coefficient as a measure of concentration (typically ranges from 0.16 to 0.50 [1]), default: 0.2606
    :param V: horizontal visibility [km] (typically ranges from 8 to 24), default: None
    :param H_a: aerosol scale height [km] (typically 1), default: None
    :return: aerosol optical thickness
    """
    if ((V is not None) & (H_a is not None)):
        beta = compute_beta(V=V, H_a=H_a)       
        
    tau_a = beta * (wavelengths / lambda_a)**(-alpha)
    
    return tau_a
    
    
def omega_a(AM=5, RH=80):
    """
    Aerosol single scattering albedo [1]
    
    [1] Gege, P. (2021): The Water Colour Simulator WASI. User manual for WASI version 6.
    
    :param AM: air mass type [1: open ocean aerosols .. 10: continental aerosols], default: 5
    :param RH: relative humidity [%] (typical values range from 46 to 91 %), default: 80
    :return: aerosol single scattering albedo
    """
    omega_a = (-0.0032 * AM + 0.972) * np.exp(3.06 * 10**(-4) * RH)
    return omega_a
    
    
def F_a(theta_sun=np.radians(30), alpha=1.317):
    """
    Aerosol forward scattering probability [1]
    
    [1] Gege, P. (2021): The Water Colour Simulator WASI. User manual for WASI version 6.
    
    :param theta_sun: sun zenith angle [radians], default: np.radians(30)
    :param alpha: Angström exponent determining wavelength dependency (typically ranges from 0.2 to 2 [1]), default: 1.317
    :return: aerosol forward scattering probability
    """
    B3 = np.log(1 - (-0.1417 * alpha + 0.82))
    B1 = B3 * (1.459 + B3*(0.1595 + 0.4129*B3))
    B2 = B3 * (0.0783 + B3*(-0.3824 - 0.5874*B3))
    
    F_a = 1 - 0.5 * np.exp((B1 + B2 * np.cos(theta_sun)) * np.cos(theta_sun))
    
    return F_a


def T_r(wavelengths=np.arange(400,800), theta_sun=np.radians(30), P=1013.25):
    """
    Rayleigh scattering [1]
    
    [1] Gege, P. (2021): The Water Colour Simulator WASI. User manual for WASI version 6.
    
    :param wavelengths: wavelengths to compute T_r for, default: np.arange(400,800)
    :param theta_sun: sun zenith angle [radians], default: np.radians(30)
    :param P: atmospheric pressure [mbar], default: 1013.25
    :return: rayleigh scattering
    """
    T_r = np.exp(-M_cor(theta_sun=theta_sun, P=P) / (115.6406 * (wavelengths/1000)**4 - 1.335 * (wavelengths/1000)**2))
    return T_r
    
    
def T_aa(wavelengths=np.arange(400,800), theta_sun=np.radians(30), AM=5, RH=80):
    """
    Aerosol absorption [1]
    
    [1] Gege, P. (2021): The Water Colour Simulator WASI. User manual for WASI version 6.
    
    :param wavelengths: wavelengths to compute T_aa for, default: np.arange(400,800)
    :param theta_sun: sun zenith angle [radians], default: np.radians(30)
    :param AM: air mass type [1: open ocean aerosols .. 10: continental aerosols], default: 5
    :param RH: relative humidity [%] (typical values range from 46 to 91 %), default: 80
    :return: aerosol absorption
    """
    T_aa = np.exp(-(1-omega_a(AM=AM, RH=RH)) * tau_a(wavelengths) * M(theta_sun=theta_sun))
    return T_aa
    
    
def T_as(wavelengths=np.arange(400,800), theta_sun=np.radians(30), AM=5, RH=80, lambda_a=550, alpha=1.317, beta=0.2606):
    """
    Aerosol scattering [1]
    
    [1] Gege, P. (2021): The Water Colour Simulator WASI. User manual for WASI version 6.
    
    :param wavelengths: wavelengths to compute T_aa for, default: np.arange(400,800)
    :param theta_sun: sun zenith angle [radians], default: np.radians(30)
    :param AM: air mass type [1: open ocean aerosols .. 10: continental aerosols], default: 5
    :param RH: relative humidity [%] (typical values range from 46 to 91 %), default: 80
    :param lambda_a: reference wavelength, default: 550
    :param alpha: Angström exponent determining wavelength dependency (typically ranges from 0.2 to 2 [1]), default: 1.317
    :param beta: turbidity coefficient as a measure of concentration (typically ranges from 0.16 to 0.50 [1]), default: 0.2606
    :return: aerosol absorption
    """
    T_as = np.exp(-omega_a(AM=AM, RH=RH) * tau_a(wavelengths, lambda_a=lambda_a, alpha=alpha, beta=beta) * M(theta_sun=theta_sun))
    return T_as
    
    
def T_oz(wavelengths=np.arange(400,800), theta_sun=np.radians(30), H_oz=0.381, a_oz_res=[]):
    """
    Ozone absorption [1]
    
    [1] Gege, P. (2021): The Water Colour Simulator WASI. User manual for WASI version 6.
    
    :param wavelengths: wavelengths to compute T_aa for, default: np.arange(400,800)
    :param theta_sun: sun zenith angle [radians], default: np.radians(30)
    :param H_oz: ozone scale height [cm], default: 0.381
    :param a_oz_res: optional, precomputing a_oz saves a lot of time.
    """
    T_oz = np.exp(-absorption.a_oz(wavelengths, a_oz_res=a_oz_res) * H_oz * M_oz(theta_sun=theta_sun))
    return T_oz
    
    
def T_ox(wavelengths=np.arange(400,800), theta_sun=np.radians(30), P=1013.25, a_ox_res=[]):
    """
    Oxygen absorption [1]
    
    [1] Gege, P. (2021): The Water Colour Simulator WASI. User manual for WASI version 6.
    
    :param wavelengths: wavelengths to compute T_aa for, default: np.arange(400,800)
    :param theta_sun: sun zenith angle [radians], default: np.radians(30)
    :param P: atmospheric pressure [mbar], default: 1013.25
    :param a_ox_res: optional, precomputing a_ox saves a lot of time.
    """
    T_ox = np.exp((-1.41*absorption.a_ox(wavelengths, a_ox_res=a_ox_res) * M_cor(theta_sun=theta_sun, P=P)) / (1 + 118.3 * absorption.a_ox(wavelengths, a_ox_res=a_ox_res) * M_cor(theta_sun=theta_sun, P=P))**0.45)
    return T_ox
    
    
def T_wv(wavelengths=np.arange(400,800), theta_sun=np.radians(30), WV=2.5, a_wv_res=[]):
    """
    Water vapor absorption [1]
    
    [1] Gege, P. (2021): The Water Colour Simulator WASI. User manual for WASI version 6.
    
    :param wavelengths: wavelengths to compute T_aa for, default: np.arange(400,800)
    :param theta_sun: sun zenith angle [radians], default: np.radians(30)
    :param WV: precipitable water [cm], default: 2.5
    """
    T_wv = np.exp((-0.2385 * absorption.a_wv(wavelengths, a_wv_res=a_wv_res) * WV * M(theta_sun=theta_sun)) / (1 + 20.07 * absorption.a_wv(wavelengths, a_wv_res=a_wv_res) * WV * M(theta_sun=theta_sun))**0.45)
    return T_wv
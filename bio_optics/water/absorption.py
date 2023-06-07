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
from .. helper import resampling 
from . import temperature_gradient


def a_w(wavelengths = np.arange(400,800), a_w_res=[]):
    """
    Spectral absorption coefficient of pure water [1/m] at a reference temperature of 20 degree C. 
    The spectrum is from WASI6 [1] and a compilation of different sources.
    
    [1] Gege, P. (2021): The Water Colour Simulator WASI. User manual for WASI version 6.
    
    :wavelengths: wavelengths to resample a_w to [nm], default: np.arange(400,800)
    :param a_w_res: optional, preresampling a_w before inversion saves a lot of time.
    :return: spectral absorption coefficient of pure water [m-1]
    """
    if len(a_w_res)==0:
        a_w = resampling.resample_a_w(wavelengths=wavelengths)
    else:
        a_w = a_w_res
        
    return a_w


def a_w_T(wavelengths = np.arange(400,800), T_W_0=20, T_W=20, a_w_res=[], da_W_div_dT_res=[]):
    """
    Spectral absorption coefficient of pure water corrected for actual temperature in degrees C after [1].
    
    [1] Gege, P. (2021): The Water Colour Simulator WASI. User manual for WASI version 6.
    
    :wavelengths: wavelengths to compute a_w_T for [nm], default: np.arange(400,800)
    :param T_W_0: Reference temperature [degrees C], default: 20
    :param T_W: Actual water temperature [degrees C], default: 20
    :param a_w_res: optional, preresampling a_w before inversion saves a lot of time.
    :param da_W_div_dT_res: optional, preresampling da_W_div_dT before inversion saves a lot of time.
    :return: spectral absorption coefficient of pure water corrected for actual temperature
    """
    a_w_T = a_w(wavelengths=wavelengths, a_w_res=a_w_res) + (T_W - T_W_0) * temperature_gradient.da_W_div_dT(wavelengths=wavelengths, da_W_div_dT_res=da_W_div_dT_res)
    return a_w_T


def a_ph(C_0 = 0,
         C_1 = 0,
         C_2 = 0,
         C_3 = 0,
         C_4 = 0,
         C_5 = 0,
         wavelengths = np.arange(400,800),
         a_i_spec_res = []):
    """
    Spectral absorption coefficient of phytoplankton for a mixture of up to 6 phytoplankton classes (C_0..C_5).
    
    :param C_0: concentration of phytoplankton type 0 [ug/L], default: 0
    :param C_1: concentration of phytoplankton type 1 [ug/L], default: 0
    :param C_2: concentration of phytoplankton type 2 [ug/L], default: 0
    :param C_3: concentration of phytoplankton type 3 [ug/L], default: 0
    :param C_4: concentration of phytoplankton type 4 [ug/L], default: 0
    :param C_5: concentration of phytoplankton type 5 [ug/L], default: 0
    :wavelengths: wavelengths to compute a_ph for [nm], default: np.arange(400,800)
    :param a_i_spec_res: optional, preresampling a_i_spec (absorption of phytoplankton types C_0..C_5) before inversion saves a lot of time.
    :return: spectral absorption coefficient of phytoplankton
    """
    C_i = np.array([C_0,C_1,C_2,C_3,C_4,C_5])
    
    if len(a_i_spec_res)==0:
        a_i_spec = resampling.resample_a_i_spec(wavelengths=wavelengths)
    else:
        a_i_spec = a_i_spec_res
    
    a_ph = np.sum([C_i[i] * a_i_spec.T[i] for i in np.arange(a_i_spec.shape[1])], axis=0)
    
    return a_ph


def a_Y_norm(wavelengths = np.arange(400,800),
             S = 0.014,
             lambda_0 = 440):
    """
    Exponential approximation of normalized spectral absorption of CDOM.
    
    :wavelengths: wavelengths to compute a_Y for [nm], default: np.arange(400,800)
    :param S: spectral slope of CDOM absorption spectrum [m-1], default: 0.014
    :param lambda_0: wavelength used for normalization [nm], default: 440
    :return: normalized spectral absorption of CDOM
    """
    return np.exp(-S * (wavelengths - lambda_0))


def a_Y(C_Y = 0, 
        wavelengths = np.arange(400,800),
        S = 0.014, 
        lambda_0 = 440,
        K = 0,
        a_Y_N_res=[]):
    """
    Exponential approximation of spectral absorption of CDOM or yellow substances.

    [1] Mobley (2022): The Oceanic Optics Book [doi.org/10.25607/OBP-1710]
    [2] Grunert et al. (2018): Characterizing CDOM Spectral Variability Across Diverse Regions and Spectral Ranges [doi.org/10.1002/2017GB005756]).
   
    :param C_Y: CDOM absorption coefficient at lambda_0 [m-1]
    :wavelengths: wavelengths to compute a_Y for [nm], default: np.arange(400,800)
    :param S: spectral slope of CDOM absorption spectrum [m-1], default: 0.014
    :param lambda_0: wavelength used for normalization [nm], default: 440
    :param K: Constant added to the exponential function [m-1], default: 0 
              "What this constant represents is not clear. In some cases it is supposed to account for scattering by the dissolved component, 
              however there is no reason to believe such scattering would be spectrally ﬂat (see Bricaud et al. 1981 for an in-depth discussion)" [1].
              "K is a constant addressing background noise and potential instrument bias" [2]     
    :param a_Y_N_res: optional, precomputing a_Y_norm before inversion saves a lot of time.   
    :return: spectral absorption coefficient of CDOM or yellow substances
    """
    if len(a_Y_N_res)==0:
        a_Y_N = a_Y_norm(wavelengths=wavelengths, S=S, lambda_0=lambda_0)
    else:
        a_Y_N = a_Y_N_res
    
    a_Y = C_Y * a_Y_N + K
    
    return a_Y
    

def a_NAP_norm(wavelengths = np.arange(400,800),
               S_NAP = 0.011,
               lambda_0 = 440):
    """
    Normalized absorption spectrum of non-algal particles (NAP).
    Can be approximated reasonably well in many cases with an exponential function.
    Normalized at the same wavelength (lambda_0) as CDOM.
    
    :param wavelengths: wavelengths to compute a_NAP_norm for [nm], default: np.arange(400,800)
    :param S_NAP: spectral slope of NAP absorption spectrum [m-1], default: 0.011
    :param lambda_0: reference wavelength for normalization of NAP absorption spectrum (identical for CDOM) [nm], default: 440 nm
    :return: normalized spectral absorption of NAP
    """
    return np.exp(-S_NAP * (wavelengths - lambda_0))


def a_NAP(C_X = 0,
          C_Mie = 0,
          wavelengths = np.arange(400,800), 
          lambda_0 = 440,
          a_NAP_spec_lambda_0 = 0.041,
          S_NAP = 0.011,
          a_NAP_N_res=[]):
    """
    Spectral absorption of non-algal particles (NAP), also known as detritus, tripton or bleached particles.
    Normalized at the same wavelength (lambda_0) as CDOM.
    
    :param C_X: concentration of non-algal particles type I [mg/L], default: 0
    :param C_Mie: concentration of non-algal particles type II [mg/L], default: 0
    :wavelengths: wavelengths to compute a_NAP for [nm], default: np.arange(400,800)
    :param lambda_0: reference wavelength for normalization of NAP absorption spectrum (identical for CDOM) [nm], default: 440 nm
    :param a_NAP_spec_lambda_0: specific absorption coefficient of NAP at referece wavelength lambda_0 [m2 g-1], default: 0.041
    :param S_NAP: spectral slope of NAP absorption spectrum, default [m-1]: 0.011
    :param a_NAP_norm_res: optional, preresampling a_NAP_norm before inversion saves a lot of time.
    :return: spectral absorption coefficient of non-algal particles (NAP)
    """
    C_NAP = C_X + C_Mie
    
    if len(a_NAP_N_res)==0:
        a_NAP_N = a_NAP_norm(wavelengths=wavelengths, S_NAP=S_NAP, lambda_0=lambda_0)
    else:
        a_NAP_N = a_NAP_N_res
    
    #a_NAP_norm = np.exp(-S_NAP * (wavelengths - lambda_0))
    a_NAP = C_NAP * a_NAP_spec_lambda_0 * a_NAP_N
    
    return a_NAP


def a(C_0 = 0,
      C_1 = 0,
      C_2 = 0,
      C_3 = 0,
      C_4 = 0,
      C_5 = 0,
      C_Y = 0, 
      C_X = 0, 
      C_Mie = 0,
      wavelengths = np.arange(400,800),
      S = 0.014,
      lambda_0 = 440,
      K=0,
      a_NAP_spec_lambda_0 = 0.041,
      S_NAP = 0.011,
      T_W=20,
      T_W_0=20,
      a_w_res=[],
      da_W_div_dT_res=[],
      a_i_spec_res=[],
      a_Y_N_res=[],
      a_NAP_N_res=[]
      ):
    """
    Spectral absorption coefficient of a natural water body.
    
    :param C_0: concentration of phytoplankton type 0 [ug/L], default: 0
    :param C_1: concentration of phytoplankton type 1 [ug/L], default: 0
    :param C_2: concentration of phytoplankton type 2 [ug/L], default: 0
    :param C_3: concentration of phytoplankton type 3 [ug/L], default: 0
    :param C_4: concentration of phytoplankton type 4 [ug/L], default: 0
    :param C_5: concentration of phytoplankton type 5 [ug/L], default: 0
    :param C_Y: CDOM absorption coefficient at lambda_0 [m-1]
    :param C_X: concentration of non-algal particles type I [mg/L], default: 0
    :param C_Mie: concentration of non-algal particles type II [mg/L], default: 0
    :wavelengths: wavelengths to compute a for [nm], default: np.arange(400,800)
    :param S: spectral slope of CDOM absorption spectrum [nm-1], default: 0.014
    :param lambda_0: wavelength used for normalization of CDOM and NAP functions [nm], default: 440
    :param K: constant added to the CDOM exponential function [m-1], default: 0
    :param a_NAP_spec_lambda_0: specific absorption coefficient of NAP at referece wavelength lambda_0 [m2 g-1], default: 0.041
    :param S_NAP: spectral slope of NAP absorption spectrum, default [nm-1]: 0.011
    :param T_W: actual water temperature [degrees C], default: 20
    :param T_W_0: reference temperature [degrees C], default: 20
    :param a_w_res: optional, absorption of pure water resampled to sensor's band settings. Will be computed within function if not provided.
    :param da_W_div_dT_res: optional, temperature gradient of pure water absorption resampled  to sensor's band settings. Will be computed within function if not provided.
    :param a_i_spec_res: optional, specific absorption coefficients of phytoplankton types resampled to sensor's band settings. Will be computed within function if not provided.
    :param a_Y_N_res: optional, normalized absorption coefficients of CDOM resampled to sensor's band settings. Will be computed within function if not provided.
    :param a_NAP_N_res: optional, normalized absorption coefficients of NAP resampled to sensor's band settings. Will be computed within function if not provided.
    :return: spectral absorption coefficient of a natural water body
    """
    a_wc = a_ph(wavelengths=wavelengths, C_0=C_0, C_1=C_1, C_2=C_2, C_3=C_3, C_4=C_4, C_5=C_5, a_i_spec_res=a_i_spec_res) + \
           a_Y(C_Y=C_Y, wavelengths=wavelengths, S=S, lambda_0=lambda_0, K=K, a_Y_N_res=a_Y_N_res) + \
           a_NAP(C_X=C_X, C_Mie=C_Mie, wavelengths=wavelengths, a_NAP_spec_lambda_0=a_NAP_spec_lambda_0, S_NAP=S_NAP, lambda_0=lambda_0, a_NAP_N_res=a_NAP_N_res)
    a = a_w(wavelengths=wavelengths, a_w_res=a_w_res) + (T_W - T_W_0) * temperature_gradient.da_W_div_dT(wavelengths=wavelengths, da_W_div_dT_res=da_W_div_dT_res) + a_wc
    
    return a


def a_Phi(a_phy_440 = 0.01,
          wavelengths=np.arange(400,800),
          A_res=[]):
    """
    Phytoplankton pigment absorption coefficient based on the empirical parameters A0 and A1 (Phi0 and Phi1 in some publications) first described in Lee (1994) [1]
    according to the Eq. 12 and the values for A0 and A1 provided in Tab. 2 in Lee et al. (1998) [2].    
    In most of Lee's work the phytoplankton absorption coefficient at 440 nm (here: a_phy_440) is called P (e.g., [2]).
        
    [1] Lee (1994): Visible-Infrared Remote-Sensing Model and Applications for Ocean Waters. Dissertation.
    [2] Lee et al. (1998): Hyperspectral remote sensing for shallow waters: 1 A semianalytical model [10.1364/ao.37.006329]
    
    :param a_phy_440: Phytoplankton absorption coefficient at 440 nm, usually called P in Lee's work; default: 0.01.
    :param wavelengths: wavelength range to compute a_phy for; default: 400 nm - 800 nm.
    :param A_res: empirical factors A0 and A1 resampled to the band setting of the used sensor; saves a lot of time during inversion.
    :return: phytoplankton pigment absorption coefficient for the provided wavelengths and a given chlorophyll a concentration.
    """    
    if len(A_res)==0:
        A0, A1 = resampling.resample_A(wavelengths=wavelengths)
    else:
        A0, A1 = A_res[0], A_res[1]
        
    a_phy = (A0 + A1 * np.log(a_phy_440)) * a_phy_440
    
    return a_phy

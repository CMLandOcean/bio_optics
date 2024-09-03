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
from .. helper import resampling, utils


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
    a_w_T = a_w(wavelengths=wavelengths, a_w_res=a_w_res) + (T_W - T_W_0) * da_w_div_dT(wavelengths=wavelengths, da_w_div_dT_res=da_W_div_dT_res)
    return a_w_T

def da_w_div_dT(wavelengths = np.arange(400,800), da_w_div_dT_res=[]):
    """
    Temperature gradient of pure water absorption resampled to sensor's spectral sampling rate.
    The spectrum is from Roettgers et al. [1].
    
    [1] Roettgers et al. (2013): Pure water spectral absorption, scattering, and real part of refractive index model.
    
    :param wavelengths: wavelengths to resample da_W_div_dT to
    :param da_W_div_dT_res: optional, preresampling da_W_div_dT before inversion saves a lot of time.
    :return: temperature gradient of pure water absorption
    """
    if len(da_w_div_dT_res) == 0:
        da_w_div_dT = resampling.resample_da_w_div_dT(wavelengths=wavelengths)
    else:
        da_w_div_dT = da_w_div_dT_res
    
    return da_w_div_dT


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
    
    a_ph = 0
    
    for i in range(a_i_spec.shape[1]): a_ph += C_i[i] * a_i_spec[:, i]
    
    return a_ph

def da_ph_div_dCi(i,
              wavelengths = np.arange(400,800),
              a_i_spec_res = []):
    """
    Partial derivative of a_phy with respect to jth phytoplankton concentration

    # Math: \frac{\partial}{\partial C_j} a_{phy}(\lambda) = \frac{\partial}{\partial C_j} \sum_{i=0}^5 C_i * a_i^* (\lambda) = a_j^*(\lambda)
    """
    if len(a_i_spec_res)==0:
        a_i_spec = resampling.resample_a_i_spec(wavelengths=wavelengths)
    else:
        a_i_spec = a_i_spec_res
    
    da_ph_div_dCi = a_i_spec.T[i]
    
    return da_ph_div_dCi


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

    # Math: a_{CDOM}(\lambda) = C_Y * e^{-S (\lambda - \lambda_0)} + K
    """
    if len(a_Y_N_res)==0:
        a_Y_N = a_Y_norm(wavelengths=wavelengths, S=S, lambda_0=lambda_0)
    else:
        a_Y_N = a_Y_N_res
    
    a_Y = C_Y * a_Y_N + K
    
    return a_Y
    
def da_Y_div_dC_Y(wavelengths = np.arange(400,800),
        S = 0.014, 
        lambda_0 = 440,
        a_Y_N_res = []):
    """
    # Math: \frac{\partial}{\partial C_y}\left[C_Y * e^{-S (\lambda - \lambda_0)} + K \right] = e^{-S (\lambda - \lambda_0)}
    """
    if len(a_Y_N_res) == 0:
        da_Y_div_dC_Y = np.exp(-S * (wavelengths - lambda_0))
    else:
        da_Y_div_dC_Y = a_Y_N_res
    
    return da_Y_div_dC_Y

def da_Y_div_dS(C_Y = 0, 
        wavelengths = np.arange(400,800),
        S = 0.014, 
        lambda_0 = 440,
        a_Y_N_res = []):
    """
    # Math: \frac{\partial}{\partial S}\left[C_Y * e^{-S (\lambda - \lambda_0)}\right] = C_Y \frac{\partial}{\partial S} e^{-S (\lambda - \lambda_0)}
    # Math: = C_Y (\lambda_0 - \lambda) e^{-S (\lambda - \lambda_0)}
    """
    if len(a_Y_N_res) == 0:
        a_Y_N = np.exp(-S * (wavelengths - lambda_0))
    else:
        a_Y_N = a_Y_N_res

    da_Y_div_dS = C_Y * (lambda_0 - wavelengths) * a_Y_N
    
    return da_Y_div_dS


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

    # Math: a_{NAP} = C_{NAP} * a_{NAP}^*(\lambda_0) * e^{ -S_{NAP} (\lambda - \lambda_0) }
    # Math: = (C_X + C_{Mie}) * a_{NAP}^*(\lambda_0) * e^{ -S_{NAP} (\lambda - \lambda_0) }
    """
    C_NAP = C_X + C_Mie
    
    if len(a_NAP_N_res)==0:
        a_NAP_N = a_NAP_norm(wavelengths=wavelengths, S_NAP=S_NAP, lambda_0=lambda_0)
    else:
        a_NAP_N = a_NAP_N_res
    
    a_NAP = C_NAP * a_NAP_spec_lambda_0 * a_NAP_N
    
    return a_NAP

def da_NAP_div_dC_X(
          wavelengths = np.arange(400,800), 
          lambda_0 = 440,
          a_NAP_spec_lambda_0 = 0.041,
          S_NAP = 0.011,
          a_NAP_N_res=[]):
    """
    # Math: \frac{\partial}{\partial C_{X}}a_{NAP} = \frac{\partial}{\partial C_{X}}\left[ (C_X + C_{Mie}) * a_{NAP}^*(\lambda_0) * e^{ -S_{NAP} (\lambda - \lambda_0) } \right]
    # Math: = a_{NAP}^*(\lambda_0) * e^{-S(\lambda - \lambda_0)}
    """
    if len(a_NAP_N_res) == 0:
        a_NAP_N = np.exp(-S_NAP * (wavelengths - lambda_0))
    else:
        a_NAP_N = a_NAP_N_res

    da_NAP_div_dC_X = a_NAP_spec_lambda_0 * a_NAP_N
    
    return da_NAP_div_dC_X

def da_NAP_div_dC_Mie(
          wavelengths = np.arange(400,800), 
          lambda_0 = 440,
          a_NAP_spec_lambda_0 = 0.041,
          S_NAP = 0.011,
          a_NAP_N_res=[]):
    """
    # Math: \frac{\partial}{\partial C_{Mie}}a_{NAP} = \frac{\partial}{\partial C_{Mie}}\left[ (C_X + C_{Mie}) * a_{NAP}^*(\lambda_0) * e^{ -S_{NAP} (\lambda - \lambda_0) } \right]
    # Math: = a_{NAP}^*(\lambda_0) * e^{-S(\lambda - \lambda_0)}
    """    
    if len(a_NAP_N_res) == 0:
        a_NAP_N = np.exp(-S_NAP * (wavelengths - lambda_0))
    else:
        a_NAP_N = a_NAP_N_res

    da_NAP_div_dC_Mie = a_NAP_spec_lambda_0 * a_NAP_N
    
    return da_NAP_div_dC_Mie

def da_NAP_div_dS_NAP(C_X = 0,
                      C_Mie = 0,
                      wavelengths = np.arange(400,800), 
                      lambda_0 = 440,
                      a_NAP_spec_lambda_0 = 0.041,
                      S_NAP = 0.011,
                      a_NAP_N_res=[]):
    """
    # Math: \frac{\partial}{\partial S_{NAP}}a_{NAP} = C_{NAP} * a_{NAP}^*(\lambda_0) * \frac{\partial}{\partial S_{NAP}}e^{-S(\lambda - \lambda_0)}
    # Math: = C_{NAP} * a_{NAP}^*(\lambda_0) * (\lambda_0 - \lambda) * e^{-S(\lambda - \lambda_0)}
    """
    C_NAP = C_X + C_Mie
    
    if len(a_NAP_N_res) == 0:
        a_NAP_n = np.exp(-S_NAP * (wavelengths - lambda_0))
    else:
        a_NAP_N = a_NAP_N_res

    da_NAP_div_dS_NAP = C_NAP * a_NAP_spec_lambda_0 * -(wavelengths - lambda_0) * a_NAP_n
    
    return da_NAP_div_dS_NAP


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
      da_w_div_dT_res=[],
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
    
    a = a_w(wavelengths=wavelengths, a_w_res=a_w_res) + (T_W - T_W_0) * da_w_div_dT(wavelengths=wavelengths, da_w_div_dT_res=da_w_div_dT_res) + a_wc
    
    return a

def da_div_dC_i(i,
      wavelengths = np.arange(400,800),
      a_i_spec_res=[],
      ):
    """
    # Math: a(\lambda) = \left[ a_w(\lambda) + (T - T_0)\frac{da_w(\lambda)}{dT} \right] + a_{wc}(\lambda)
    # Math: a_{wc} = a_{CDOM} + a_{phy} + a_{NAP}
    # Math: \frac{\partial}{\partial C_0} a(\lambda) = \frac{\partial}{\partial C_0} a_{phy}
    """
    return da_ph_div_dCi(i, wavelengths=wavelengths, a_i_spec_res=a_i_spec_res)

def da_div_dC_Y(wavelengths = np.arange(400,800),
      S = 0.014,
      lambda_0 = 440,
      a_Y_N_res=[]):
    
    return da_Y_div_dC_Y(wavelengths=wavelengths, S=S, lambda_0=lambda_0, a_Y_N_res=a_Y_N_res)

def da_div_dS(C_Y = 0, 
      wavelengths = np.arange(400,800),
      S = 0.014,
      lambda_0 = 440,
      a_Y_N_res=[]):
    return da_Y_div_dS(C_Y=C_Y, wavelengths=wavelengths, S=S, lambda_0=lambda_0, a_Y_N_res=a_Y_N_res)

def da_div_dC_X(
      wavelengths = np.arange(400,800),
      lambda_0 = 440,
      a_NAP_spec_lambda_0 = 0.041,
      S_NAP = 0.011,
      a_NAP_N_res=[]):
    return da_NAP_div_dC_X(wavelengths=wavelengths, lambda_0=lambda_0, a_NAP_spec_lambda_0=a_NAP_spec_lambda_0, S_NAP=S_NAP, a_NAP_N_res=a_NAP_N_res)

def da_div_dC_Mie(
      wavelengths = np.arange(400,800),
      lambda_0 = 440,
      a_NAP_spec_lambda_0 = 0.041,
      S_NAP = 0.011,
      a_NAP_N_res=[]
      ):
    return da_NAP_div_dC_Mie(wavelengths=wavelengths, lambda_0=lambda_0, a_NAP_spec_lambda_0=a_NAP_spec_lambda_0, S_NAP=S_NAP, a_NAP_N_res=a_NAP_N_res)

def da_div_dS_NAP(C_i,
      C_X = 0, 
      C_Mie = 0,
      wavelengths = np.arange(400,800),
      lambda_0 = 440,
      a_NAP_spec_lambda_0 = 0.041,
      S_NAP = 0.011,
      a_NAP_N_res=[]
      ):
    return da_NAP_div_dS_NAP(C_X=C_X, C_Mie=C_Mie, wavelengths=wavelengths, lambda_0=lambda_0, a_NAP_spec_lambda_0=a_NAP_spec_lambda_0, S_NAP=S_NAP, a_NAP_N_res=a_NAP_N_res)


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

def a_Y_pow(C_Y = 0, 
        wavelengths = np.arange(400,800),
        S = 6.92, 
        lambda_0 = 412,
        K = 0):
    """
    Spectral absorption of CDOM or yellow substances according to Twardowski et al. 2004 [doi.org/10.1016/j.marchem.2004.02.008].
    "Another model that has been found to work even better than the exponential model is a power-law model" (Mobley, OceanOpticsBook 2022).
        
    :param C_Y: CDOM absorption coefficient [1/m]
    :param wavelengths: wavelengths to compute 
    :param S: spectral slope, default: 6.92 
    :param lambda_0: wavelength used for normalization in nm, default: 412 nm
    :return: spectral absorption coefficient of CDOM or yellow substances
    """
    
    a_Y = C_Y * (wavelengths / lambda_0)**(-S) + K
    
    return a_Y

def a_Y_gauss(wavelengths=np.arange(400,800), C_Y=0, phi1=1, mu1=0, sigma1=10, phi2=1, mu2=0, sigma2=10, K=0):
    """
    Gaussian decomposition CDOM model inspired by Gege [1].
    Instead of the commonly used exponential function, CDOM absorption is described by two Gaussian peaks defined by phi, mu, and sigma.
    Two gaussian peaks are fit per default and following the approach of Gege [3] as described in [4]:
    (1) A first peak at 203 nm with variable mu and sigma, and
    (2) a second peak in the wavelength region around 240 nm.
    All components can be used as fit parameters.
    
    [1] Gege, P. (2000): Gaussian model for yellow substance absorption spectra. Proc. Ocean Optics XV conference, October 16-20, 2000, Monaco.
    [2] Göritz, A. (2018): From laboratory spectroscopy to remote sensing : Methods for the retrieval of water constituents in optically complex waters. Dissertation.
    
    :param phi1: height of the first Gaussian peak, default: 1 
    :param mu1: center position of the first peak, default: 0 (= standard exponential model)
    :param sigma1: width of the first peak, default: 10
    :param phi2: height of the second Gaussian peak, default: 1 
    :param mu2: center position of the second peak, default: 0 (= standard exponential model)
    :param sigma2: width of the second peak, default: 10
    :return: spectral absorption coefficient of CDOM or yellow substances
    """
    return C_Y * \
           phi1 * np.exp(-np.power(wavelengths - mu1, 2.) / (2 * np.power(sigma1, 2.))) + \
           phi2 * np.exp(-np.power(wavelengths - mu2, 2.) / (2 * np.power(sigma2, 2.))) + \
           K

def a_Y_exp_gauss(C_Y=0, wavelengths=np.arange(400,800), S=0.014, lambda_0=440, K=0, phi1=1, mu1=0, sigma1=10, phi2=1, mu2=0, sigma2=10):
    """
    Gaussian decomposition CDOM model inspired by Massicotete & Stiig [1], Grunert et al. [2], and Gege [3].
    Standard exponential CDOM model extended by two (2) Gaussian peaks defined by phi, mu, and sigma.
    Two gaussian peaks are fit per default and following the approach of Gege [3] as described in [4]:
    (1) A first peak at 203 nm with variable mu and sigma, and
    (2) a second peak in the wavelength region around 240 nm.
    All components can be used as fit parameters.
    
    [1] 10.1016/j.marchem.2016.01.008
    [2] 10.1002/2017GB005756
    [3] Gege, P. (2000): Gaussian model for yellow substance absorption spectra. Proc. Ocean Optics XV conference, October 16-20, 2000, Monaco.
    [4] Göritz, A. (2018): From laboratory spectroscopy to remote sensing : Methods for the retrieval of water constituents in optically complex waters. Dissertation.
    
    :param C_Y: CDOM absorption coefficient at lambda_0 [1/m]
    :param wavelengths: wavelengths to compute 
    :param S: spectral slope, default: 0.014 [1/nm]
    :param lambda_0: wavelength used for normalization in nm, default: 440 nm
    :param K: Constant added to the exponential function. "What this constant represents is not clear. In some cases it is supposed to account 
              for scattering by the dissolved component, however there is no reason to believe such scattering would be spectrally ﬂat (see Bricaud et al. 1981
              for an in-depth discussion)" (Mobley [OceanOpticsBook], 2022).
              "K is a constant addressing background noise and potential instrument bias (1/m)" (Grunert et al. 2018 [doi.org/10.1002/2017GB005756]).
              , default: 0
    :param phi1: height of the first Gaussian peak, default: 1 
    :param mu1: center position of the first peak, default: 0 (= standard exponential model)
    :param sigma1: width of the first peak, default: 10
    :param phi2: height of the second Gaussian peak, default: 1 
    :param mu2: center position of the second peak, default: 0 (= standard exponential model)
    :param sigma2: width of the second peak, default: 10
    :return: spectral absorption coefficient of CDOM or yellow substances
    """
    return a_Y(C_Y=C_Y, wavelengths=wavelengths, S=S, lambda_0=lambda_0, K=K) + \
           phi1 * np.exp(-np.power(wavelengths - mu1, 2.) / (2 * np.power(sigma1, 2.))) + \
           phi2 * np.exp(-np.power(wavelengths - mu2, 2.) / (2 * np.power(sigma2, 2.)))


################
#### HEREON ####
################


def a_xd_spec(wavelengths=np.arange(400,800),
              A_xd=0,
              S_xd=0,
              C_xd=0,
              lambda_0=550.):
    """
    Generic exponential function incl. offset. 
    E.g., for specific absorption coefficients (Eq. 20 in [1]).

    [1] Bi et al. (2023): Bio-geo-optical modelling of natural waters [10.3389/fmars.2023.11963529]

    Args:
        wavelengths (_type_, optional): _description_. Defaults to np.arange(400,800).
        A_xd (int, optional): _description_. Defaults to 0.
        S_xd (int, optional): _description_. Defaults to 0.
        C_xd (int, optional): Constant. Defaults to 0.
        lambda_0 (_type_, optional): Reference wavelength [nm]. Defaults to 550..
    """
    a_xd_spec = A_xd * np.exp(-S_xd * (wavelengths - lambda_0)) + C_xd
    return a_xd_spec
    

def a_md_spec(wavelengths=np.arange(400,800), 
              A_md=13.4685e-3, 
              S_md=10.3845e-3, 
              C_md=12.1700e-3,
              lambda_0=550.):
    """    
    Mass-specific absorption coefficient of minerogenic detritus [m2/g] [1]

    [1] Bi et al. (2023): Bio-geo-optical modelling of natural waters [10.3389/fmars.2023.11963529]

    Args:
        wavelengths (_type_, optional): _description_. Defaults to np.arange(400,800).
        A_md (_type_, optional): _description_. Defaults to 13.4685e-3.
        S_md (_type_, optional): _description_. Defaults to 10.3845e-3.
        C_md (_type_, optional): Constant. Defaults to 12.1700e-3.
        lambda_0 (_type_, optional): Reference wavelength [nm]. Defaults to 550..

    Returns:
        a_md_spec: Mass-specific absorption coefficient of minerogenic detritus [m2/g]
    """
    a_md_spec = a_xd_spec(wavelengths=wavelengths, A_xd=A_md, S_xd=S_md, C_xd=C_md, lambda_0=lambda_0) 
    return a_md_spec


def a_bd_spec(wavelengths=np.arange(400,800), 
              A_bd=0.3893e-3, 
              S_bd=15.7621e-3, 
              C_bd= 0.9994e-3, 
              lambda_0=550.):
    """
    Chl-specific absorption coefficient of biogenic detritus [m2/mg] [1]

    [1] Bi et al. (2023): Bio-geo-optical modelling of natural waters [10.3389/fmars.2023.11963529]

    Args:
        wavelengths (_type_, optional): _description_. Defaults to np.arange(400,800).
        A_bd (_type_, optional): _description_. Defaults to 0.3893e-3.
        S_bd (_type_, optional): _description_. Defaults to 15.7621e-3.
        C_bd (_type_, optional): Constant. Defaults to 0.9994e-3.
        lambda_0 (_type_, optional): Reference wavelength [nm]. Defaults to 550..

    Returns:
        a_bd_spec: Chl-specific absorption coefficient of biogenic detritus [m2/mg]
    """
    a_bd_spec = a_xd_spec(wavelengths=wavelengths, A_xd=A_bd, S_xd=S_bd, C_xd=C_bd, lambda_0=lambda_0)
    return a_bd_spec


def a_d(wavelengths=np.arange(400,800), 
        C_ism=1., 
        C_phy=1.,
        A_md=13.4685e-3, 
        A_bd=0.3893e-3, 
        S_md=10.3845e-3, 
        S_bd=15.7621e-3, 
        C_md=12.1700e-3,
        C_bd= 0.9994e-3, 
        lambda_0_md=550., 
        lambda_0_bd=550., 
        a_md_spec_res=[],
        a_bd_spec_res=[]):
    """
    Absorption coefficient of detritus (Eq. 7 in [1]).

    [1] Bi et al. (2023): Bio-geo-optical modelling of natural waters [10.3389/fmars.2023.11963529]

    Args:
        wavelengths (_type_): _description_. Defaults to np.arange(400,800).
        C_ism (_type_, optional): Concentration of inorganic suspended matter. Defaults to 1..
        C_phy (_type_, optional): Concentration of chlorophyll a. Defaults to 1..
        A_md (_type_, optional): _description_. Defaults to 13.4685e-3.
        A_bd (_type_, optional): _description_. Defaults to 0.3893e-3.
        S_md (_type_, optional): _description_. Defaults to 10.3845e-3.
        S_bd (_type_, optional): _description_. Defaults to 15.7621e-3.
        C_md (_type_, optional): Constant. Defaults to 12.1700e-3.
        C_bd (_type_, optional): Constant. Defaults to 0.9994e-3.
        lambda_0_md (_type_, optional): _description_. Defaults to 550..
        lambda_0_bd (_type_, optional): _description_. Defaults to 550..
        a_md_spec_res (list, optional): _description_. Defaults to [].
        a_bd_spec_res (list, optional): _description_. Defaults to [].

    Returns:
        a_d: Absorption coefficient of detritus [m-1].
    """
    a_md_spec_res = a_md_spec(wavelengths, A_md, S_md, C_md, lambda_0=lambda_0_md) if len(a_md_spec_res)==0 else a_md_spec_res
    a_bd_spec_res = a_bd_spec(wavelengths, A_bd, S_bd, C_bd, lambda_0=lambda_0_bd)  if len(a_bd_spec_res)==0 else a_bd_spec_res

    a_d = C_ism * a_md_spec_res + C_phy * a_bd_spec_res

    return a_d


def a_phy(C_0 = 0,
          C_1 = 0,
          C_2 = 0,
          C_3 = 0,
          C_4 = 0,
          C_5 = 0,
          C_6 = 0,
          C_7 = 0,
          wavelengths = np.arange(400,800),
          a_i_spec_res = []):
    """
    Spectral scattering coefficient of phytoplankton for a mixture of up to 6 phytoplankton classes (C_0..C_7).
    
    :param C_0: concentration of phytoplankton type 0 [ug/L], default: 0
    :param C_1: concentration of phytoplankton type 1 [ug/L], default: 0
    :param C_2: concentration of phytoplankton type 2 [ug/L], default: 0
    :param C_3: concentration of phytoplankton type 3 [ug/L], default: 0
    :param C_4: concentration of phytoplankton type 4 [ug/L], default: 0
    :param C_5: concentration of phytoplankton type 5 [ug/L], default: 0
    :param C_6: concentration of phytoplankton type 6 [ug/L], default: 0
    :param C_7: concentration of phytoplankton type 7 [ug/L], default: 0
    :wavelengths: wavelengths to compute a_ph for [nm], default: np.arange(400,800)
    :param b_i_spec_res: optional, preresampling b_i_spec (scattering coefficient of phytoplankton types C_0..C_7) before inversion saves a lot of time.
    :return: spectral scattering coefficient of phytoplankton mixture
    """
    C_i = np.array([C_0,C_1,C_2,C_3,C_4,C_5,C_6,C_7])
   
    if len(a_i_spec_res)==0:
        a_i_spec = resampling.resample_a_i_spec_EnSAD(wavelengths=wavelengths)
    else:
        a_i_spec = a_i_spec_res
    
    a_phy = 0
    # shape-1 because there are 7 classes in the sli
    for i in range(a_i_spec.shape[1]): a_phy += C_i[i] * a_i_spec[:, i]
    
    return a_phy


def correct_a_phy(a_phy_res, 
                  wavelengths=np.arange(400,800), 
                  C_phy=1., 
                  A=0.0237, 
                  E0=1., 
                  E1=0.8987, 
                  lambda_0_phy=676.,
                  interpolate=True):
    """
    Correct a_ph for non-linear concentration-related effects (e.g., packaging) following [1] (Eqs. 14 and 21)

     [1] Bi et al. (2023): Bio-geo-optical modelling of natural waters [10.3389/fmars.2023.11963529]

    Args:
        a_ph (np.array): Spectral absorption coefficient of phytoplankton.
        wavelengths (np.array, optional): Corresponding wavelengths [nm]. Defaults to np.arange(400,800).
        C_phy (float, optional): Concentration of phytoplankton [ug/L]. Defaults to 1..
        A (float, optional): Scale factor. Defaults to 0.0237. Between 0.0112 ~ 0.0501.
        E0 (float, optional): Power exponent E0 represents the effects of pigment packaging and its interaction with phytoplankton cell size for phytoplankton concentrations <= 1.. Defaults to 1..
        E1 (float, optional): Power exponent E1 represents the effects of pigment packaging and its interaction with phytoplankton cell size for phytoplankton concentrations > 1.. Defaults to 0.8987.
        lambda_0 (float, optional): Reference wavelength [nm]. Defaults to 676..
        interpolate (bool, optional): Boolean to decide if a_ph at lambda_0 is to be interpolated. If False, a_ph at the closest wavelength will be chosen. Defaults to True.

    Returns:
        a_ph: Spectral absorption coefficient of phytoplankton corrected for non-linear concentration-related effects.
    """
    if interpolate:
        a_phy_lambda_0 = np.interp(lambda_0_phy, wavelengths, a_phy_res)
    else:
        a_phy_lambda_0 = a_phy_res[utils.find_closest(wavelengths, lambda_0_phy)[1]] 

    E = E0 if C_phy <= 1. else E1
    a_phy_res *= (A * C_phy**E) / a_phy_lambda_0  

    return a_phy_res


def a_total(wavelengths=np.arange(400,800), 
            C_0=0., 
            C_1=0., 
            C_2=0., 
            C_3=0., 
            C_4=0., 
            C_5=0.,
            C_6=0.,
            C_7=0.,
            C_Y=0.,
            C_ism=0.,
            A_md=13.4685e-3, 
            A_bd=0.3893e-3, 
            S_md=10.3845e-3, 
            S_bd=15.7621e-3, 
            S_cdom = 0.014,
            C_md=12.1700e-3,
            C_bd= 0.9994e-3,
            K=0,
            lambda_0_cdom = 440,
            lambda_0_md=550., 
            lambda_0_bd=550.,
            lambda_0_phy=676.,
            A=0.0237, 
            E0=1., 
            E1=0.8987, 
            interpolate=True, 
            T_W=20,
            T_W_0=20,
            a_d_res=[],
            a_md_spec_res=[],
            a_bd_spec_res=[],
            a_i_spec_res=[],
            a_phy_res=[],
            a_Y_N_res=[],
            a_w_res=[],
            da_W_div_dT_res=[]):
    
    C_phy = np.sum([C_0, C_1, C_2, C_3, C_4, C_5, C_6, C_7])

    if len(a_d_res)==0:
        a_d_res = a_d(wavelengths=wavelengths, C_phy=C_phy, C_ism=C_ism, A_md=A_md, A_bd=A_bd, S_md=S_md, S_bd=S_bd, C_md=C_md, C_bd=C_bd, lambda_0_md=lambda_0_md, lambda_0_bd=lambda_0_bd, a_md_spec_res=a_md_spec_res, a_bd_spec_res=a_bd_spec_res)
    
    if len(a_phy_res)==0:
        a_phy_res = a_phy(wavelengths=wavelengths, C_0=C_0, C_1=C_1, C_2=C_2, C_3=C_3, C_4=C_4, C_5=C_5, C_6=C_6, C_7=C_7, a_i_spec_res=a_i_spec_res)

    a_wc = correct_a_phy(a_phy_res=a_phy_res, wavelengths=wavelengths, C_phy=C_phy, A=A, E0=E0, E1=E1, lambda_0_phy=lambda_0_phy, interpolate=interpolate) + \
           a_Y(C_Y=C_Y, wavelengths=wavelengths, S=S_cdom, lambda_0=lambda_0_cdom, K=K, a_Y_N_res=a_Y_N_res) + \
           a_d_res
    
    a = a_w(wavelengths=wavelengths, a_w_res=a_w_res) + (T_W - T_W_0) * da_w_div_dT(wavelengths=wavelengths, da_w_div_dT_res=da_W_div_dT_res) + a_wc

    return a
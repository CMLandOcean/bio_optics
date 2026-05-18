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
from .. helper import resampling, utils


def a_w(wavelengths = np.arange(400,800), a_w_res=None):
    """
    Spectral absorption coefficient of pure water [1/m] at a reference temperature of 20 degree C. 
    The spectrum is from WASI6 [1] and a compilation of different sources.
    
    [1] Gege, P. (2021): The Water Colour Simulator WASI. User manual for WASI version 6.
    
    Args:
        wavelengths: wavelengths [nm], default: np.arange(400, 800)
        a_w_res: optional precomputed pure water absorption; if provided, skips resampling

    Returns:
        a_w: spectral absorption coefficient of pure water [m-1]
    """
    if a_w_res is None:
        a_w = resampling.resample_a_w(wavelengths=wavelengths)
    else:
        a_w = a_w_res
        
    return a_w

def a_w_T(wavelengths = np.arange(400,800), T_W_0=20, T_W=20, a_w_res=None, da_W_div_dT_res=None):
    """
    Spectral absorption coefficient of pure water corrected for actual temperature in degrees C after [1].
    
    [1] Gege, P. (2021): The Water Colour Simulator WASI. User manual for WASI version 6.
    
    Args:
        wavelengths: wavelengths [nm], default: np.arange(400, 800)
        T_W_0: reference temperature [degrees C], default: 20
        T_W: actual water temperature [degrees C], default: 20
        a_w_res: optional precomputed pure water absorption; if provided, skips resampling
        da_W_div_dT_res: optional precomputed temperature gradient; if provided, skips resampling

    Returns:
        a_w_T: spectral absorption coefficient of pure water corrected for actual temperature [m-1]
    """
    a_w_T = a_w(wavelengths=wavelengths, a_w_res=a_w_res) + (T_W - T_W_0) * da_w_div_dT(wavelengths=wavelengths, da_w_div_dT_res=da_W_div_dT_res)
    return a_w_T

def da_w_div_dT(wavelengths = np.arange(400,800), da_w_div_dT_res=None):
    """
    Temperature gradient of pure water absorption resampled to sensor's spectral sampling rate.
    The spectrum is from Roettgers et al. [1].
    
    [1] Roettgers et al. (2013): Pure water spectral absorption, scattering, and real part of refractive index model.
    
    Args:
        wavelengths: wavelengths [nm], default: np.arange(400, 800)
        da_w_div_dT_res: optional precomputed temperature gradient; if provided, skips resampling

    Returns:
        da_w_div_dT: temperature gradient of pure water absorption [m-1 degrees C-1]
    """
    if da_w_div_dT_res is None:
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
    
    Args:
        C_0: concentration of phytoplankton type 0 [ug/L], default: 0
        C_1: concentration of phytoplankton type 1 [ug/L], default: 0
        C_2: concentration of phytoplankton type 2 [ug/L], default: 0
        C_3: concentration of phytoplankton type 3 [ug/L], default: 0
        C_4: concentration of phytoplankton type 4 [ug/L], default: 0
        C_5: concentration of phytoplankton type 5 [ug/L], default: 0
        wavelengths: wavelengths [nm], default: np.arange(400, 800)
        a_i_spec_res: optional precomputed specific absorption spectra of phytoplankton types; if provided, skips resampling

    Returns:
        a_ph: spectral absorption coefficient of phytoplankton [m-1]
    """
    C_i = np.array([C_0,C_1,C_2,C_3,C_4,C_5])
    
    if a_i_spec_res is None:
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
    if a_i_spec_res is None:
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
    
    Args:
        wavelengths: wavelengths [nm], default: np.arange(400, 800)
        S: spectral slope of CDOM absorption spectrum [nm-1], default: 0.014
        lambda_0: reference wavelength for normalization [nm], default: 440

    Returns:
        a_Y_norm: normalized spectral absorption of CDOM [dimensionless]
    """
    return np.exp(-S * (wavelengths - lambda_0))

def a_Y(C_Y = 0, 
        wavelengths = np.arange(400,800),
        S = 0.014, 
        lambda_0 = 440,
        K = 0,
        a_Y_N_res=None):
    """
    Exponential approximation of spectral absorption of CDOM or yellow substances.

    [1] Mobley (2022): The Oceanic Optics Book [doi.org/10.25607/OBP-1710]
    [2] Grunert et al. (2018): Characterizing CDOM Spectral Variability Across Diverse Regions and Spectral Ranges [doi.org/10.1002/2017GB005756]).
   
    Args:
        C_Y: CDOM absorption coefficient at lambda_0 [m-1]
        wavelengths: wavelengths [nm], default: np.arange(400, 800)
        S: spectral slope of CDOM absorption spectrum [nm-1], default: 0.014
        lambda_0: reference wavelength for normalization [nm], default: 440
        K: constant offset [m-1], default: 0; "What this constant represents is not clear. In some cases it is supposed to account for scattering by the dissolved
           component, however there is no reason to believe such scattering would be spectrally flat (see Bricaud et al. 1981 for an in-depth discussion)" [1];
           "K is a constant addressing background noise and potential instrument bias" [2]
        a_Y_N_res: optional precomputed normalized CDOM absorption; if provided, skips computation

    Returns:
        a_Y: spectral absorption coefficient of CDOM or yellow substances [m-1]

    # Math: a_{CDOM}(\lambda) = C_Y * e^{-S (\lambda - \lambda_0)} + K
    """
    if a_Y_N_res is None:
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
    if a_Y_N_res is None:
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
    if a_Y_N_res is None:
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
    
    Args:
        wavelengths: wavelengths [nm], default: np.arange(400, 800)
        S_NAP: spectral slope of NAP absorption spectrum [nm-1], default: 0.011
        lambda_0: reference wavelength for normalization [nm], default: 440

    Returns:
        a_NAP_norm: normalized spectral absorption of NAP [dimensionless]
    """
    return np.exp(-S_NAP * (wavelengths - lambda_0))

def a_NAP(C_X = 0,
          C_Mie = 0,
          wavelengths = np.arange(400,800), 
          lambda_0 = 440,
          a_NAP_spec_lambda_0 = 0.041,
          S_NAP = 0.011,
          a_NAP_N_res=None):
    """
    Spectral absorption of non-algal particles (NAP), also known as detritus, tripton or bleached particles.
    Normalized at the same wavelength (lambda_0) as CDOM.
    
    Args:
        C_X: concentration of non-algal particles type I [mg/L], default: 0
        C_Mie: concentration of non-algal particles type II [mg/L], default: 0
        wavelengths: wavelengths [nm], default: np.arange(400, 800)
        lambda_0: reference wavelength for normalization [nm], default: 440
        a_NAP_spec_lambda_0: specific absorption coefficient of NAP at lambda_0 [m2 g-1], default: 0.041
        S_NAP: spectral slope of NAP absorption spectrum [nm-1], default: 0.011
        a_NAP_N_res: optional precomputed normalized NAP absorption; if provided, skips computation

    Returns:
        a_NAP: spectral absorption coefficient of non-algal particles [m-1]

    # Math: a_{NAP} = C_{NAP} * a_{NAP}^*(\lambda_0) * e^{ -S_{NAP} (\lambda - \lambda_0) }
    # Math: = (C_X + C_{Mie}) * a_{NAP}^*(\lambda_0) * e^{ -S_{NAP} (\lambda - \lambda_0) }
    """
    C_NAP = C_X + C_Mie
    
    if a_NAP_N_res is None:
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
          a_NAP_N_res=None):
    """
    # Math: \frac{\partial}{\partial C_{X}}a_{NAP} = \frac{\partial}{\partial C_{X}}\left[ (C_X + C_{Mie}) * a_{NAP}^*(\lambda_0) * e^{ -S_{NAP} (\lambda - \lambda_0) } \right]
    # Math: = a_{NAP}^*(\lambda_0) * e^{-S(\lambda - \lambda_0)}
    """
    if a_NAP_N_res is None:
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
          a_NAP_N_res=None):
    """
    # Math: \frac{\partial}{\partial C_{Mie}}a_{NAP} = \frac{\partial}{\partial C_{Mie}}\left[ (C_X + C_{Mie}) * a_{NAP}^*(\lambda_0) * e^{ -S_{NAP} (\lambda - \lambda_0) } \right]
    # Math: = a_{NAP}^*(\lambda_0) * e^{-S(\lambda - \lambda_0)}
    """    
    if a_NAP_N_res is None:
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
                      a_NAP_N_res=None):
    """
    # Math: \frac{\partial}{\partial S_{NAP}}a_{NAP} = C_{NAP} * a_{NAP}^*(\lambda_0) * \frac{\partial}{\partial S_{NAP}}e^{-S(\lambda - \lambda_0)}
    # Math: = C_{NAP} * a_{NAP}^*(\lambda_0) * (\lambda_0 - \lambda) * e^{-S(\lambda - \lambda_0)}
    """
    C_NAP = C_X + C_Mie
    
    if a_NAP_N_res is None:
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
      a_w_res=None,
      da_w_div_dT_res=None,
      a_i_spec_res=None,
      a_Y_N_res=None,
      a_NAP_N_res=None
      ):
    """
    Spectral absorption coefficient of a natural water body.
    
    Args:
        C_0: concentration of phytoplankton type 0 [ug/L], default: 0
        C_1: concentration of phytoplankton type 1 [ug/L], default: 0
        C_2: concentration of phytoplankton type 2 [ug/L], default: 0
        C_3: concentration of phytoplankton type 3 [ug/L], default: 0
        C_4: concentration of phytoplankton type 4 [ug/L], default: 0
        C_5: concentration of phytoplankton type 5 [ug/L], default: 0
        C_Y: CDOM absorption coefficient at lambda_0 [m-1], default: 0
        C_X: concentration of non-algal particles type I [mg/L], default: 0
        C_Mie: concentration of non-algal particles type II [mg/L], default: 0
        wavelengths: wavelengths [nm], default: np.arange(400, 800)
        S: spectral slope of CDOM absorption spectrum [nm-1], default: 0.014
        lambda_0: reference wavelength for normalization of CDOM and NAP [nm], default: 440
        K: constant offset of CDOM exponential function [m-1], default: 0
        a_NAP_spec_lambda_0: specific absorption coefficient of NAP at lambda_0 [m2 g-1], default: 0.041
        S_NAP: spectral slope of NAP absorption spectrum [nm-1], default: 0.011
        T_W: actual water temperature [degrees C], default: 20
        T_W_0: reference temperature [degrees C], default: 20
        a_w_res: optional precomputed pure water absorption; if provided, skips resampling
        da_w_div_dT_res: optional precomputed temperature gradient; if provided, skips resampling
        a_i_spec_res: optional precomputed specific absorption spectra of phytoplankton types; if provided, skips resampling
        a_Y_N_res: optional precomputed normalized CDOM absorption; if provided, skips computation
        a_NAP_N_res: optional precomputed normalized NAP absorption; if provided, skips computation

    Returns:
        a: spectral absorption coefficient of a natural water body [m-1]
    """
    a_wc = a_ph(wavelengths=wavelengths, C_0=C_0, C_1=C_1, C_2=C_2, C_3=C_3, C_4=C_4, C_5=C_5, a_i_spec_res=a_i_spec_res) + \
           a_Y(C_Y=C_Y, wavelengths=wavelengths, S=S, lambda_0=lambda_0, K=K, a_Y_N_res=a_Y_N_res) + \
           a_NAP(C_X=C_X, C_Mie=C_Mie, wavelengths=wavelengths, a_NAP_spec_lambda_0=a_NAP_spec_lambda_0, S_NAP=S_NAP, lambda_0=lambda_0, a_NAP_N_res=a_NAP_N_res)
    
    a = a_w(wavelengths=wavelengths, a_w_res=a_w_res) + (T_W - T_W_0) * da_w_div_dT(wavelengths=wavelengths, da_w_div_dT_res=da_w_div_dT_res) + a_wc
    
    return a

def da_div_dC_i(i,
      wavelengths = np.arange(400,800),
      a_i_spec_res=None,
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
      a_Y_N_res=None):
    
    return da_Y_div_dC_Y(wavelengths=wavelengths, S=S, lambda_0=lambda_0, a_Y_N_res=a_Y_N_res)

def da_div_dS(C_Y = 0, 
      wavelengths = np.arange(400,800),
      S = 0.014,
      lambda_0 = 440,
      a_Y_N_res=None):
    return da_Y_div_dS(C_Y=C_Y, wavelengths=wavelengths, S=S, lambda_0=lambda_0, a_Y_N_res=a_Y_N_res)

def da_div_dC_X(
      wavelengths = np.arange(400,800),
      lambda_0 = 440,
      a_NAP_spec_lambda_0 = 0.041,
      S_NAP = 0.011,
      a_NAP_N_res=None):
    return da_NAP_div_dC_X(wavelengths=wavelengths, lambda_0=lambda_0, a_NAP_spec_lambda_0=a_NAP_spec_lambda_0, S_NAP=S_NAP, a_NAP_N_res=a_NAP_N_res)

def da_div_dC_Mie(
      wavelengths = np.arange(400,800),
      lambda_0 = 440,
      a_NAP_spec_lambda_0 = 0.041,
      S_NAP = 0.011,
      a_NAP_N_res=None
      ):
    return da_NAP_div_dC_Mie(wavelengths=wavelengths, lambda_0=lambda_0, a_NAP_spec_lambda_0=a_NAP_spec_lambda_0, S_NAP=S_NAP, a_NAP_N_res=a_NAP_N_res)

def da_div_dS_NAP(C_i,
      C_X = 0, 
      C_Mie = 0,
      wavelengths = np.arange(400,800),
      lambda_0 = 440,
      a_NAP_spec_lambda_0 = 0.041,
      S_NAP = 0.011,
      a_NAP_N_res=None
      ):
    return da_NAP_div_dS_NAP(C_X=C_X, C_Mie=C_Mie, wavelengths=wavelengths, lambda_0=lambda_0, a_NAP_spec_lambda_0=a_NAP_spec_lambda_0, S_NAP=S_NAP, a_NAP_N_res=a_NAP_N_res)


def a_Phi(a_phy_440 = 0.01,
          wavelengths=np.arange(400,800),
          A_res=None):
    """
    Phytoplankton pigment absorption coefficient based on the empirical parameters A0 and A1 (Phi0 and Phi1 in some publications) first described in Lee (1994) [1]
    according to the Eq. 12 and the values for A0 and A1 provided in Tab. 2 in Lee et al. (1998) [2].    
    In most of Lee's work the phytoplankton absorption coefficient at 440 nm (here: a_phy_440) is called P (e.g., [2]).
        
    [1] Lee (1994): Visible-Infrared Remote-Sensing Model and Applications for Ocean Waters. Dissertation.
    [2] Lee et al. (1998): Hyperspectral remote sensing for shallow waters: 1 A semianalytical model [10.1364/ao.37.006329]
    
    Args:
        a_phy_440: phytoplankton absorption coefficient at 440 nm (called P in Lee's work) [m-1], default: 0.01
        wavelengths: wavelengths [nm], default: np.arange(400, 800)
        A_res: optional precomputed empirical factors A0 and A1; if provided, skips resampling

    Returns:
        a_phy: phytoplankton pigment absorption coefficient [m-1]
    """    
    if A_res is None:
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
        
    Args:
        C_Y: CDOM absorption coefficient at lambda_0 [m-1], default: 0
        wavelengths: wavelengths [nm], default: np.arange(400, 800)
        S: spectral slope [dimensionless], default: 6.92
        lambda_0: reference wavelength for normalization [nm], default: 412
        K: constant offset [m-1], default: 0

    Returns:
        a_Y: spectral absorption coefficient of CDOM or yellow substances [m-1]
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
    
    Args:
        wavelengths: wavelengths [nm], default: np.arange(400, 800)
        C_Y: CDOM absorption scaling factor [m-1], default: 0
        phi1: height of the first Gaussian peak [m-1], default: 1
        mu1: center wavelength of the first peak [nm], default: 0
        sigma1: width of the first peak [nm], default: 10
        phi2: height of the second Gaussian peak [m-1], default: 1
        mu2: center wavelength of the second peak [nm], default: 0
        sigma2: width of the second peak [nm], default: 10
        K: constant offset [m-1], default: 0

    Returns:
        a_Y: spectral absorption coefficient of CDOM or yellow substances [m-1]
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
    
    Args:
        C_Y: CDOM absorption coefficient at lambda_0 [m-1], default: 0
        wavelengths: wavelengths [nm], default: np.arange(400, 800)
        S: spectral slope [nm-1], default: 0.014
        lambda_0: reference wavelength for normalization [nm], default: 440
        K: constant offset [m-1], default: 0
        phi1: height of the first Gaussian peak [m-1], default: 1
        mu1: center wavelength of the first peak [nm], default: 0
        sigma1: width of the first peak [nm], default: 10
        phi2: height of the second Gaussian peak [m-1], default: 1
        mu2: center wavelength of the second peak [nm], default: 0
        sigma2: width of the second peak [nm], default: 10

    Returns:
        a_Y: spectral absorption coefficient of CDOM or yellow substances [m-1]
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
        wavelengths: wavelengths [nm], default: np.arange(400, 800)
        A_xd: amplitude coefficient [m2/g], default: 0
        S_xd: spectral slope [nm-1], default: 0
        C_xd: constant offset [m2/g], default: 0
        lambda_0: reference wavelength [nm], default: 550

    Returns:
        a_xd_spec: generic specific absorption coefficient [m2/g]
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
        wavelengths: wavelengths [nm], default: np.arange(400, 800)
        A_md: amplitude coefficient [m2/g], default: 13.4685e-3
        S_md: spectral slope [nm-1], default: 10.3845e-3
        C_md: constant offset [m2/g], default: 12.1700e-3
        lambda_0: reference wavelength [nm], default: 550

    Returns:
        a_md_spec: mass-specific absorption coefficient of minerogenic detritus [m2/g]
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
        wavelengths: wavelengths [nm], default: np.arange(400, 800)
        A_bd: amplitude coefficient [m2/mg], default: 0.3893e-3
        S_bd: spectral slope [nm-1], default: 15.7621e-3
        C_bd: constant offset [m2/mg], default: 0.9994e-3
        lambda_0: reference wavelength [nm], default: 550

    Returns:
        a_bd_spec: Chl-specific absorption coefficient of biogenic detritus [m2/mg]
    """
    a_bd_spec = a_xd_spec(wavelengths=wavelengths, A_xd=A_bd, S_xd=S_bd, C_xd=C_bd, lambda_0=lambda_0)
    return a_bd_spec


def a_md(wavelengths=np.arange(400,800),
         C_ism=1.,
         A_md=13.4685e-3,
         S_md=10.3845e-3,
         C_md=12.1700e-3,
         lambda_0_md=550.,
         a_md_spec_res=None):
    """
    Absorption coefficient of minerogenic detritus [m-1] (Bi et al. 2023, Eq. 5).

    Args:
        wavelengths: wavelengths [nm]
        C_ism: inorganic suspended matter concentration [g m-3], default: 1.0
        A_md: amplitude coefficient [m2 g-1], default: 13.4685e-3
        S_md: spectral slope [nm-1], default: 10.3845e-3
        C_md: constant offset [m2 g-1], default: 12.1700e-3
        lambda_0_md: reference wavelength [nm], default: 550
        a_md_spec_res: optional precomputed mass-specific absorption [m2 g-1]

    Returns:
        a_md: absorption coefficient of minerogenic detritus [m-1]
    """
    if a_md_spec_res is None:
        a_md_spec_res = a_md_spec(wavelengths, A_md, S_md, C_md, lambda_0=lambda_0_md)
    return C_ism * a_md_spec_res


def a_bd(wavelengths=np.arange(400,800),
         C_phy=1.,
         A_bd=0.3893e-3,
         S_bd=15.7621e-3,
         C_bd=0.9994e-3,
         lambda_0_bd=550.,
         a_bd_spec_res=None):
    """
    Absorption coefficient of biogenic detritus [m-1] (Bi et al. 2023, Eq. 6).

    Args:
        wavelengths: wavelengths [nm]
        C_phy: phytoplankton (Chl-a) concentration [mg m-3], default: 1.0
        A_bd: amplitude coefficient [m2 mg-1], default: 0.3893e-3
        S_bd: spectral slope [nm-1], default: 15.7621e-3
        C_bd: constant offset [m2 mg-1], default: 0.9994e-3
        lambda_0_bd: reference wavelength [nm], default: 550
        a_bd_spec_res: optional precomputed Chl-specific absorption [m2 mg-1]

    Returns:
        a_bd: absorption coefficient of biogenic detritus [m-1]
    """
    if a_bd_spec_res is None:
        a_bd_spec_res = a_bd_spec(wavelengths, A_bd, S_bd, C_bd, lambda_0=lambda_0_bd)
    return C_phy * a_bd_spec_res


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
        a_md_spec_res=None,
        a_bd_spec_res=None):
    """
    Absorption coefficient of detritus (Eq. 7 in [1]).

    [1] Bi et al. (2023): Bio-geo-optical modelling of natural waters [10.3389/fmars.2023.11963529]

    Args:
        wavelengths: wavelengths [nm], default: np.arange(400, 800)
        C_ism: concentration of inorganic suspended matter [g/m3], default: 1.0
        C_phy: concentration of chlorophyll a [ug/L], default: 1.0
        A_md: amplitude coefficient for minerogenic detritus [m2/g], default: 13.4685e-3
        A_bd: amplitude coefficient for biogenic detritus [m2/mg], default: 0.3893e-3
        S_md: spectral slope of minerogenic detritus [nm-1], default: 10.3845e-3
        S_bd: spectral slope of biogenic detritus [nm-1], default: 15.7621e-3
        C_md: constant offset for minerogenic detritus [m2/g], default: 12.1700e-3
        C_bd: constant offset for biogenic detritus [m2/mg], default: 0.9994e-3
        lambda_0_md: reference wavelength for minerogenic detritus [nm], default: 550
        lambda_0_bd: reference wavelength for biogenic detritus [nm], default: 550
        a_md_spec_res: optional precomputed minerogenic detritus specific absorption; if provided, skips computation
        a_bd_spec_res: optional precomputed biogenic detritus specific absorption; if provided, skips computation

    Returns:
        a_d: absorption coefficient of detritus [m-1]
    """
    a_md_spec_res = a_md_spec(wavelengths, A_md, S_md, C_md, lambda_0=lambda_0_md) if a_md_spec_res is None else a_md_spec_res
    a_bd_spec_res = a_bd_spec(wavelengths, A_bd, S_bd, C_bd, lambda_0=lambda_0_bd)  if a_bd_spec_res is None else a_bd_spec_res

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
    
    Args:
        C_0: concentration of phytoplankton type 0 [ug/L], default: 0
        C_1: concentration of phytoplankton type 1 [ug/L], default: 0
        C_2: concentration of phytoplankton type 2 [ug/L], default: 0
        C_3: concentration of phytoplankton type 3 [ug/L], default: 0
        C_4: concentration of phytoplankton type 4 [ug/L], default: 0
        C_5: concentration of phytoplankton type 5 [ug/L], default: 0
        C_6: concentration of phytoplankton type 6 [ug/L], default: 0
        C_7: concentration of phytoplankton type 7 [ug/L], default: 0
        wavelengths: wavelengths [nm], default: np.arange(400, 800)
        a_i_spec_res: optional precomputed specific absorption spectra of phytoplankton types (EnSAD); if provided, skips resampling

    Returns:
        a_phy: spectral absorption coefficient of phytoplankton mixture [m-1]
    """
    C_i = np.array([C_0,C_1,C_2,C_3,C_4,C_5,C_6,C_7])

    if a_i_spec_res is None:
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
            a_d_res=None,
            a_md_res=None,
            a_bd_res=None,
            a_md_spec_res=None,
            a_bd_spec_res=None,
            a_i_spec_res=None,
            a_phy_res=None,
            a_Y_N_res=None,
            a_w_res=None,
            da_W_div_dT_res=None):
    """
    Total spectral absorption coefficient of a natural water body following Bi et al. (2023) [1].

    [1] Bi et al. (2023): Bio-geo-optical modelling of natural waters [10.3389/fmars.2023.11963529]

    Args:
        wavelengths: wavelengths [nm], default: np.arange(400, 800)
        C_0: concentration of phytoplankton type 0 [ug/L], default: 0
        C_1: concentration of phytoplankton type 1 [ug/L], default: 0
        C_2: concentration of phytoplankton type 2 [ug/L], default: 0
        C_3: concentration of phytoplankton type 3 [ug/L], default: 0
        C_4: concentration of phytoplankton type 4 [ug/L], default: 0
        C_5: concentration of phytoplankton type 5 [ug/L], default: 0
        C_6: concentration of phytoplankton type 6 [ug/L], default: 0
        C_7: concentration of phytoplankton type 7 [ug/L], default: 0
        C_Y: CDOM absorption coefficient at lambda_0_cdom [m-1], default: 0
        C_ism: concentration of inorganic suspended matter [g/m3], default: 0
        A_md: amplitude coefficient for minerogenic detritus specific absorption [m2/g], default: 13.4685e-3
        A_bd: amplitude coefficient for biogenic detritus specific absorption [m2/mg], default: 0.3893e-3
        S_md: spectral slope of minerogenic detritus [nm-1], default: 10.3845e-3
        S_bd: spectral slope of biogenic detritus [nm-1], default: 15.7621e-3
        S_cdom: spectral slope of CDOM absorption [nm-1], default: 0.014
        C_md: constant offset for minerogenic detritus specific absorption [m2/g], default: 12.1700e-3
        C_bd: constant offset for biogenic detritus specific absorption [m2/mg], default: 0.9994e-3
        K: constant offset of CDOM exponential function [m-1], default: 0
        lambda_0_cdom: reference wavelength for CDOM normalization [nm], default: 440
        lambda_0_md: reference wavelength for minerogenic detritus [nm], default: 550
        lambda_0_bd: reference wavelength for biogenic detritus [nm], default: 550
        lambda_0_phy: reference wavelength for phytoplankton packaging correction [nm], default: 676
        A: scale factor for phytoplankton packaging correction, default: 0.0237
        E0: power exponent for C_phy <= 1, default: 1.0
        E1: power exponent for C_phy > 1, default: 0.8987
        interpolate: if True, interpolate a_ph at lambda_0_phy; if False, use nearest band, default: True
        T_W: actual water temperature [degrees C], default: 20
        T_W_0: reference temperature [degrees C], default: 20
        a_d_res: optional precomputed detritus absorption; if provided, skips computation
        a_md_spec_res: optional precomputed minerogenic detritus specific absorption; if provided, skips computation
        a_bd_spec_res: optional precomputed biogenic detritus specific absorption; if provided, skips computation
        a_i_spec_res: optional precomputed specific absorption spectra of phytoplankton types; if provided, skips resampling
        a_phy_res: optional precomputed phytoplankton absorption; if provided, skips computation
        a_Y_N_res: optional precomputed normalized CDOM absorption; if provided, skips computation
        a_w_res: optional precomputed pure water absorption; if provided, skips resampling
        da_W_div_dT_res: optional precomputed temperature gradient; if provided, skips resampling

    Returns:
        a: total spectral absorption coefficient of a natural water body [m-1]
    """
    C_phy = np.sum([C_0, C_1, C_2, C_3, C_4, C_5, C_6, C_7])

    if a_d_res is None:
        if a_md_res is not None and a_bd_res is not None:
            a_d_res = a_md_res + a_bd_res
        else:
            a_d_res = a_d(wavelengths=wavelengths, C_phy=C_phy, C_ism=C_ism, A_md=A_md, A_bd=A_bd, S_md=S_md, S_bd=S_bd, C_md=C_md, C_bd=C_bd, lambda_0_md=lambda_0_md, lambda_0_bd=lambda_0_bd, a_md_spec_res=a_md_spec_res, a_bd_spec_res=a_bd_spec_res)
    
    if a_phy_res is None:
        a_phy_res = a_phy(wavelengths=wavelengths, C_0=C_0, C_1=C_1, C_2=C_2, C_3=C_3, C_4=C_4, C_5=C_5, C_6=C_6, C_7=C_7, a_i_spec_res=a_i_spec_res)

    a_wc = correct_a_phy(a_phy_res=a_phy_res, wavelengths=wavelengths, C_phy=C_phy, A=A, E0=E0, E1=E1, lambda_0_phy=lambda_0_phy, interpolate=interpolate) + \
           a_Y(C_Y=C_Y, wavelengths=wavelengths, S=S_cdom, lambda_0=lambda_0_cdom, K=K, a_Y_N_res=a_Y_N_res) + \
           a_d_res
    
    a = a_w(wavelengths=wavelengths, a_w_res=a_w_res) + (T_W - T_W_0) * da_w_div_dT(wavelengths=wavelengths, da_w_div_dT_res=da_W_div_dT_res) + a_wc

    return a
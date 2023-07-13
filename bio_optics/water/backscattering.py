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


def morel(wavelengths: np.array = np.arange(400,800), 
          fresh: bool = True, 
          lambda_1 = 500):
    """
    Spectral backscattering coefficient of pure water according to Morel (1974) [1].
    
    [1] Morel, A. (1974): Optical properties of pure water and pure Sea water.
    
    :param wavelengths: wavelengths to compute backscattering coefficient of pure water for, default: np.arange(400,800)
    :param fresh: boolean to decide if backscattering coefficient is to be computed for fresh (True, default) or oceanic water (False) with a salinity of 35-38 per mille. Values are only valid of lambda_0==500 nm.
    :param lambda_1: reference wavelength for backscattering of pure water [nm], default: 500
    :return: spectral backscattering coefficient of pure water
    """
    b1 = 0.00111 if fresh==True else 0.00144
        
    b_bw = b1 * (wavelengths / lambda_1)**-4.32
    
    return b_bw


def b_bw(wavelengths: np.array = np.arange(400,800), 
         fresh: bool = True,
         b_bw_res = []):
    """
    Spectral backscattering coefficient of pure water according to Morel (1974) [1].
    
    [1] Morel, A. (1974): Optical properties of pure water and pure Sea water.
    
    :param wavelengths: wavelengths to compute b_bw for, default: np.arange(400,800)
    :param fresh: boolean to decide if to compute b_bw for fresh or oceanic water, default: True
    :param b_bw_res: optional, precomputing b_bw before inversion saves a lot of time.
    :return: spectral backscattering coefficients of pure water [m-1]
    """
    if len(b_bw_res)==0:
        b_bw = morel(wavelengths=wavelengths, fresh=fresh)
    else:
        b_bw = b_bw_res
    
    return b_bw


def b_bphy(C_phy: float = 0, 
           wavelengths: np.array = np.arange(400,800), 
           b_bphy_spec: float = 0.0010,           
           b_phy_norm_res = []):
    """
    Backscattering of phytoplankton resampled to specified wavelengths.
    The normalized phytoplankton scattering coefficient b_phy_norm is imported from file.
    
    [1] Gege, P. (2021): The Water Colour Simulator WASI. User manual for WASI version 6.
    
    :param C_phy: phytoplankton concentration [ug L-1], default: 0
    :param wavelengths: wavelengths to compute b_bphy for [nm], default: np.arange(400,800)
    :param b_bphy_spec:  specific backscattering coefficient of phytoplankton at 550 nm in [m2 mg-1], default: 0.0010
    :param b_phy_norm_res: optional, preresampling b_phy_norm before inversion saves a lot of time.
    :return:
    """       
    if len(b_phy_norm_res)==0:
        b_phy_norm = resampling.resample_b_phy_norm(wavelengths=wavelengths)
    else:
        b_phy_norm = b_phy_norm_res

    b_bphy = C_phy * b_bphy_spec * b_phy_norm
    
    return b_bphy


def b_bX(C_X: float = 0,
         wavelengths: np.array = np.arange(400,800),
         b_bX_spec: float = 0.0086,
         b_bX_norm_factor: float = 1,
         b_X_norm_res = []):
    """
    Spectral backscattering coefficient of particles of type I defined by a normalized scattering coefficient with arbitrary wavelength dependency [1].
    The default parameter setting is representative for Lake Constance [1, 2].
    
    [1] Gege, P. (2021): The Water Colour Simulator WASI. User manual for WASI version 6.
    [2] Heege, T. (2000): Flugzeuggestützte Fernerkundung von Wasserinhaltsstoffen am Bodensee. PhD thesis. DLR-Forschungsbericht 2000-40, 134 p.
    
    :param C_X: concentration of non-algal particles type I [mg L-1], default: 0
    :param wavelengths: wavelengths to compute b_bX for [nm], default: np.arange(400,800)
    :param b_bX_spec: specific backscattering coefficient of non-algal particles type I [m2 g-1], default: 0.0086 [2]
    :param b_bX_norm_factor: normalized scattering coefficient with arbitrary wavelength dependency, default: 1
    :param b_X_norm_res: optional, precomputing b_bX_norm before inversion saves a lot of time.
    :return: spectral backscattering coefficient of particles of type I
    """
    if len(b_X_norm_res)==0:
        b_X_norm = np.ones(wavelengths.shape) * b_bX_norm_factor
    else:
        b_X_norm = b_X_norm_res
        
    b_bX = C_X * b_bX_spec * b_X_norm
    
    return b_bX


def b_bMie(C_Mie: float = 0,
           wavelengths: np.array = np.arange(400,800),
           b_bMie_spec: float = 0.0042,
           lambda_S: float = 500, 
           n: float = -1,
           b_Mie_norm_res=[]):
    """
    Spectral backscattering coefficient of particles of type II "defined by the normalized scattering coefficient (wavelengths/lambda_S)**n, 
    where the Angström exponent n is related to the particle size distribution" [1]. The default parameter setting is representative for Lake Constance [1, 2].
    
    [1] Gege, P. (2021): The Water Colour Simulator WASI. User manual for WASI version 6.
    [2] Heege, T. (2000): Flugzeuggestützte Fernerkundung von Wasserinhaltsstoffen am Bodensee. PhD thesis. DLR-Forschungsbericht 2000-40, 134 p.
    
    :param C_Mie: concentration of non-algal particles type II [mg L-1], default: 0
    :param wavelengths: wavelengths to compute b_bMie for [nm], default: np.arange(400,800)
    :param b_bMie_spec: specific backscattering coefficient of non-algal particles type II [m2 g-1], default: 0.0042
    :param lambda_S: reference wavelength [nm], default: 500 nm
    :param n: Angström exponent of particle type II backscattering, default: -1
    :param b_bMie_norm_res: optional, if n and lambda_S are not fit params, the last part of the equation can be precomputed to save time.
    :return: spectral backscattering coefficient of particles of type II
    """
    if len(b_Mie_norm_res)==0:
        b_bMie = C_Mie * b_bMie_spec * ((wavelengths/lambda_S)**n)
    else:
        b_bMie = C_Mie * b_bMie_spec * b_Mie_norm_res
    
    return b_bMie


def b_bNAP(C_X: float = 0,
           C_Mie: float = 0,
           wavelengths = np.arange(400,800),
           b_bMie_spec: float = 0.0042,
           lambda_S: float = 500, 
           n: float = -1,
           b_bX_spec: float = 0.0086,
           b_bX_norm_factor: float = 1,
           b_X_norm_res = [],
           b_Mie_norm_res = []):
    """
    Spectral backscattering coefficient of non-algal particles (NAP) as a mixture of two types with spectrally different backscattering coefficients [1].
    
    [1] Gege, P. (2021): The Water Colour Simulator WASI. User manual for WASI version 6.
    [2] Heege, T. (2000): Flugzeuggestützte Fernerkundung von Wasserinhaltsstoffen am Bodensee. PhD thesis. DLR-Forschungsbericht 2000-40, 134 p.
    
    :param C_X: concentration of non-algal particles type I [mg L-1], default: 0
    :param C_Mie: concentration of non-algal particles type II [mg L-1], default: 0
    :param wavelengths: wavelengths to compute b_bNAP for [nm], default: np.arange(400,800)
    :param b_bMie_spec: specific backscattering coefficient of non-algal particles type II [m2 g-1] , default: 0.0042
    :param lambda_S: reference wavelength for scattering particles type II [nm], default: 500 nm
    :param n: Angström exponent of particle type II backscattering, default: -1
    :param b_bX_spec: specific backscattering coefficient of non-algal particles type I [m2 g-1], default: 0.0086 [2]
    :param b_bX_norm_factor: normalized scattering coefficient with arbitrary wavelength dependency, default: 1
    :param b_X_norm_res: optional, precomputing b_bX_norm before inversion saves a lot of time.
    :param b_Mie_norm_res: optional, if n and lambda_S are not fit params, the last part of the equation can be precomputed to save time.
    :return: spectral backscattering coefficient of NAP
    """
    b_bNAP = b_bX(C_X=C_X, wavelengths=wavelengths, b_bX_spec=b_bX_spec, b_bX_norm_factor=b_bX_norm_factor, b_X_norm_res=b_X_norm_res) + \
             b_bMie(C_Mie=C_Mie, wavelengths=wavelengths, b_bMie_spec=b_bMie_spec, lambda_S=lambda_S, n=n, b_Mie_norm_res=b_Mie_norm_res)
    
    return b_bNAP


def b_b(C_X: float = 0,
        C_Mie: float = 0,
        C_phy: float  = 0,
        wavelengths: np.array = np.arange(400,800),
        fresh: bool = True,
        b_bMie_spec: float = 0.0042,
        lambda_S: float = 500, 
        n: float = -1,
        b_bX_spec: float = 0.0086,
        b_bX_norm_factor: float = 1,
        b_bphy_spec: float = 0.0010,
        b_bw_res = [],
        b_phy_norm_res = [],
        b_X_norm_res=[],
        b_Mie_norm_res=[],
        b_b_res=[]
        ):
    """
    Spectral backscattering coefficient of a natural water body as the sum of the backscattering coefficients of pure water, phytoplankton and non-algal particles [1].
    
    [1] Gege, P. (2021): The Water Colour Simulator WASI. User manual for WASI version 6.
    [2] Heege, T. (2000): Flugzeuggestützte Fernerkundung von Wasserinhaltsstoffen am Bodensee. PhD thesis. DLR-Forschungsbericht 2000-40, 134 p.
    
    :param C_X: concentration of non-algal particles type I [mg L-1], default: 0
    :param C_Mie: concentration of non-algal particles type II [mg L-1], default: 0
    :param C_phy: phytoplankton concentration [ug L-1], default: 0
    :param wavelengths: wavelengths to compute b_b for [nm], default: np.arange(400,800)
    :param fresh: boolean to decide if to compute b_bw for fresh or oceanic water, default: True
    :param b_bMie_spec: specific backscattering coefficient of non-algal particles type II [m2 g-1] , default: 0.0042
    :param lambda_S: reference wavelength for scattering particles type II [nm], default: 500 nm
    :param n: Angström exponent of particle type II backscattering, default: -1
    :param b_bX_spec: specific backscattering coefficient of non-algal particles type I [m2 g-1], default: 0.0086 [2]
    :param b_bX_norm_factor: normalized scattering coefficient with arbitrary wavelength dependency, default: 1
    :param b_bphy_spec:  specific backscattering coefficient at 550 nm in [m2 mg-1], default: 0.0010
    :param b_bw_res: optional, precomputing b_bw b_bw saves a lot of time during inversion.
    :param b_phy_norm_res: optional, preresampling b_phy_norm saves a lot of time during inversion.
    :param b_X_norm_res: optional, precomputing b_bX_norm before inversion saves a lot of time.
    :param b_Mie_norm_res: optional, if n and lambda_S are not fit params, the last part of the equation can be precomputed to save time.
    :return:
    """  
    if len(b_b_res)==0:
        b_b = b_bw(wavelengths=wavelengths, fresh=fresh, b_bw_res=b_bw_res) + \
            b_bNAP(C_Mie=C_Mie, C_X=C_X, wavelengths=wavelengths, b_bMie_spec=b_bMie_spec, lambda_S=lambda_S, n=n, b_bX_spec=b_bX_spec, b_bX_norm_factor=b_bX_norm_factor, b_X_norm_res=b_X_norm_res, b_Mie_norm_res=b_Mie_norm_res) + \
            b_bphy(wavelengths=wavelengths, C_phy=C_phy, b_bphy_spec=b_bphy_spec, b_phy_norm_res=b_phy_norm_res)
        
    else:
        b_b = b_b_res
    
    return b_b

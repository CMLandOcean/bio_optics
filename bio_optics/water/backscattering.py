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
from . import absorption, attenuation, scattering


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

    # Math: b_{b,phy} = C_{phy} * b_{b, phy}^* * b_{b, phy}^N(\lambda)
    """       
    if len(b_phy_norm_res)==0:
        b_phy_norm = resampling.resample_b_phy_norm(wavelengths=wavelengths)
    else:
        b_phy_norm = b_phy_norm_res

    b_bphy = C_phy * b_bphy_spec * b_phy_norm
    
    return b_bphy

def db_bphy_div_dC_phy(wavelengths: np.array = np.arange(400,800), 
        b_bphy_spec: float = 0.0010,           
        b_phy_norm_res: np.array = []):
    """
    # Math: \frac{\partial}{\partial C_{phy}}b_{b,phy} = \frac{\partial}{\partial C_{phy}}\left[C_{phy} * b_{b, phy}^* * b_{b, phy}^N(\lambda)\right] = b_{b, phy}^* * b_{b, phy}^N(\lambda)
    """
    if len(b_phy_norm_res)==0:
        b_phy_norm = resampling.resample_b_phy_norm(wavelengths=wavelengths)
    else:
        b_phy_norm = b_phy_norm_res

    db_bphy_div_dC_phy = b_bphy_spec * b_phy_norm
    
    return db_bphy_div_dC_phy


def b_bX_norm(b_bX_norm_factor=1, wavelengths=np.arange(400,800)):
    return np.ones(wavelengths.shape) * b_bX_norm_factor

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

def db_bX_div_dC_X(wavelengths: np.array = np.arange(400,800),
        b_bX_spec: float = 0.0086,
        b_bX_norm_factor: float = 1,
        b_X_norm_res = []):
    """
    # Math: \frac{\partial}{\partial C_X} b_{b,x} = \frac{\partial}{\partial C_X}\left[C_X * b_{b,X}^* * b_{b,X}^N(\lambda)\right] = b_{b,X}^* * b_{b,X}^N(\lambda)
    """
    if len(b_X_norm_res)==0:
        b_X_norm = np.ones(wavelengths.shape) * b_bX_norm_factor
    else:
        b_X_norm = b_X_norm_res
        
    b_bX_div_C_X = b_bX_spec * b_X_norm
    
    return b_bX_div_C_X


def b_bMie_norm(wavelengths=np.arange(400, 800), lambda_S=500, n=-1):
    return (wavelengths/lambda_S)**n

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

    # Math: b_{b,Mie} = C_{Mie} * b_{b,Mie} * (\frac{\lambda}{\lambda_S})^n
    """
    if len(b_Mie_norm_res)==0:
        b_bMie = C_Mie * b_bMie_spec * ((wavelengths/lambda_S)**n)
    else:
        b_bMie = C_Mie * b_bMie_spec * b_Mie_norm_res
    
    return b_bMie

def db_bMie_div_dC_Mie(wavelengths: np.array = np.arange(400,800),
        b_bMie_spec: float = 0.0042,
        lambda_S: float = 500, 
        n: float = -1,
        b_bMie_norm_res=[]):
    """
    # Math: \frac{\partial}{\partial C_{Mie}}b_{b,Mie} = \frac{\partial}{\partial C_{Mie}}\left[C_{Mie} * b_{b,Mie} * (\frac{\lambda}{\lambda_S})^n \right] = b_{b,Mie} * (\frac{\lambda}{\lambda_S})^n
    """
    if len(b_bMie_norm_res) == 0:
        b_bMie_norm = (wavelengths/lambda_S)**n
    else:
        b_bMie_norm = b_bMie_norm_res

    db_bMie_div_dC_Mie = b_bMie_spec * b_bMie_norm

    return db_bMie_div_dC_Mie

def db_Mie_div_dn(C_Mie: float = 0,
        wavelengths: np.array = np.arange(400,800),
        b_bMie_spec: float = 0.0042,
        lambda_S: float = 500, 
        n: float = -1,
        b_bMie_norm_res=[]):
    """
    # Math: \frac{\partial}{\partial n} \left[C_{Mie} * b_{b,Mie} * (\frac{\lambda}{\lambda_S})^n \right] = C_{Mie} * b_{b,Mie} * ln(\frac{\lambda}{\lambda_S}) (\frac{\lambda}{\lambda_S})^n
    """
    if len(b_bMie_norm_res) == 0:
        b_bMie_norm = (wavelengths/lambda_S)**n
    else:
        b_bMie_norm = b_bMie_norm_res

    b_bMie_div_dn = C_Mie * b_bMie_spec * np.log(wavelengths/lambda_S) * b_bMie_norm

    return b_bMie_div_dn


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

def db_bNAP_div_dC_X(wavelengths: np.array = np.arange(400,800),
        b_bX_spec: float = 0.0086,
        b_bX_norm_factor: float = 1,
        b_X_norm_res: np.array = []):
    """
    # Math: b_{b,NAP} = b_{b,X} + b_{b,Mie}
    # Math: \frac{\partial}{\partial C_X}b_{b,NAP} = \frac{\partial}{\partial C_X}b_{b,X}
    """
    return db_bX_div_dC_X(wavelengths=wavelengths, b_bX_spec=b_bX_spec, b_bX_norm_factor=b_bX_norm_factor, b_X_norm_res=b_X_norm_res)

def db_bNAP_div_dC_Mie(wavelengths: np.array = np.arange(400,800),
        b_bMie_spec: float = 0.0042,
        lambda_S: float = 500, 
        n=-1,
        b_bMie_norm_res=[]):
    """
    # Math: b_{b,NAP} = b_{b,X} + b_{b,Mie}
    # Math: \frac{\partial}{\partial C_{Mie}}b_{b,NAP} = \frac{\partial}{\partial C_{Mie}}b_{b,Mie}
    """
    return db_bMie_div_dC_Mie(wavelengths=wavelengths, b_bMie_spec=b_bMie_spec, lambda_S=lambda_S, n=n, b_bMie_norm_res=b_bMie_norm_res)

def db_NAP_div_dn(C_Mie: float = 0,
        wavelengths: np.array = np.arange(400,800),
        b_bMie_spec: float = 0.0042,
        lambda_S: float = 500, 
        n: float = -1,
        b_bMie_norm_res=[]):
    """
    # Math: b_{b,NAP} = b_{b,X} + b_{b,Mie}
    # Math: \frac{\partial}{\partial n}b_{b,NAP} = \frac{\partial}{\partial n}b_{b,Mie}
    """
    return db_Mie_div_dn(C_Mie=C_Mie, wavelengths=wavelengths, b_bMie_spec=b_bMie_spec, lambda_S=lambda_S, n=n, b_bMie_norm_res=b_bMie_norm_res)


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

def db_b_div_dC_X(wavelengths: np.array = np.arange(400,800),
        b_bX_spec: float = 0.0086,
        b_bX_norm_factor: float = 1,
        b_X_norm_res=[]
        ):
    """
    # Math: \frac{\partial}{\partial C_X} b_b(\lambda) = \frac{\partial}{\partial C_X} \left[ b_{b,w}(\lambda) + b_{b, phy}(\lambda) + b_{b, NAP}(\lambda) \right] = \frac{\partial}{\partial C_X}b_{b,NAP}(\lambda)
    """  
    db_b_div_dC_X = db_bNAP_div_dC_X(wavelengths=wavelengths, b_bX_spec=b_bX_spec, b_bX_norm_factor=b_bX_norm_factor, b_X_norm_res=b_X_norm_res)
    
    return db_b_div_dC_X

def db_b_div_dC_Mie(wavelengths: np.array = np.arange(400,800),
        n=-1,
        b_bMie_spec: float = 0.0042,
        lambda_S: float = 500, 
        b_bMie_norm_res=[]):
    """
    # Math: \frac{\partial}{\partial C_{Mie}} b_b(\lambda) = \frac{\partial}{\partial C_{Mie}} \left[ b_{b,w}(\lambda) + b_{b, phy}(\lambda) + b_{b, NAP}(\lambda) \right] = \frac{\partial}{\partial C_{Mie}}b_{b,NAP}(\lambda)
    """  
    db_b_div_dC_Mie = db_bNAP_div_dC_Mie(wavelengths=wavelengths, n=n, b_bMie_spec=b_bMie_spec, lambda_S=lambda_S, b_bMie_norm_res=b_bMie_norm_res)

    return db_b_div_dC_Mie

def db_b_div_dn(C_Mie: float = 0,
    wavelengths: np.array = np.arange(400,800),
    b_bMie_spec: float = 0.0042,
    lambda_S: float = 500, 
    n: float = -1,
    b_bMie_norm_res=[]):
    """
    # Math: \frac{\partial}{\partial n} b_b(\lambda) = \frac{\partial}{\partial n} \left[ b_{b,w}(\lambda) + b_{b, phy}(\lambda) + b_{b, NAP}(\lambda) \right] = \frac{\partial}{\partial n}b_{b,NAP}(\lambda)
    """  
    db_b_div_dn = db_NAP_div_dn(C_Mie=C_Mie, wavelengths=wavelengths, b_bMie_spec=b_bMie_spec, lambda_S=lambda_S, n=n, b_bMie_norm_res=b_bMie_norm_res)
    
    return db_b_div_dn

def db_b_div_dC_phy(wavelengths: np.array = np.arange(400,800),
        b_bphy_spec: float = 0.0010,
        b_phy_norm_res: np.array = [],
        ):
    """
    # Math: \frac{\partial}{\partial C_{phy}} b_b(\lambda) = \frac{\partial}{\partial C_{phy}} \left[ b_{b,w}(\lambda) + b_{b, phy}(\lambda) + b_{b, NAP}(\lambda) \right] = \frac{\partial}{\partial C_{phy}}b_{b,phy}(\lambda)
    """  
    db_b_div_dC_phy = db_bphy_div_dC_phy(wavelengths=wavelengths, b_bphy_spec=b_bphy_spec, b_phy_norm_res=b_phy_norm_res)
    
    return db_b_div_dC_phy


################
#### HEREON ####
################


def b_bphy_hereon(C_0 = 0,
                  C_1 = 0,
                  C_2 = 0,
                  C_3 = 0,
                  C_4 = 0,
                  C_5 = 0,
                  C_6 = 0,
                  C_7 = 0,
                  b_ratio_C_0 = 0.002,
                  b_ratio_C_1 = 0.002,
                  b_ratio_C_2 = 0.002,
                  b_ratio_C_3 = 0.002,
                  b_ratio_C_4 = 0.002, 
                  b_ratio_C_5 = 0.002, 
                  b_ratio_C_6 = 0.002, 
                  b_ratio_C_7 = 0.002, 
                  wavelengths = np.arange(400,800),
                  b_i_spec_res = []):
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

    b_ratio_C_i = np.array([b_ratio_C_0, b_ratio_C_1, b_ratio_C_2, b_ratio_C_3, b_ratio_C_4, b_ratio_C_5, b_ratio_C_6, b_ratio_C_7])
    
    if len(b_i_spec_res)==0:
        b_i_spec = resampling.resample_b_i_spec_EnSAD(wavelengths=wavelengths)
    else:
        b_i_spec = b_i_spec_res
    
    b_bphy = 0
    for i in range(b_i_spec.shape[1]): b_bphy += b_ratio_C_i[i] * C_i[i] * b_i_spec[:, i]
    
    return b_bphy


def b_bd(b_d, b_ratio_d=0.0216):
    """
    Backscattering coefficient of detritus [m-1]

    Args:
        b_d (np.array): Scattering coefficient of detritus [m-1]
        b_ratio_d (float, optional): Backscattering ratio or backscattering probability of detritus. Defaults to 0.0216.

    Returns:
        np.array: Backscattering coefficient of detritus [m-1]
    """
    return b_ratio_d * b_d


def b_b_total(wavelengths = np.arange(400,800),
         C_0 = 0,
         C_1 = 0,
         C_2 = 0,
         C_3 = 0,
         C_4 = 0,
         C_5 = 0,
         C_6 = 0,
         C_7 = 0,
         C_ism = 0,
         b_ratio_C_0 = 0.002,
         b_ratio_C_1 = 0.002,
         b_ratio_C_2 = 0.002,
         b_ratio_C_3 = 0.002,
         b_ratio_C_4 = 0.002, 
         b_ratio_C_5 = 0.002, 
         b_ratio_C_6 = 0.002, 
         b_ratio_C_7 = 0.002, 
         b_ratio_d = 0.0216,
         fresh=False,
         A_md=13.4685e-3, 
         A_bd=0.3893e-3, 
         S_md=10.3845e-3, 
         S_bd=15.7621e-3, 
         C_md=12.1700e-3,
         C_bd= 0.9994e-3, 
         lambda_0_md=550., 
         lambda_0_bd=550., 
         lambda_0_c_d=550., 
         gamma_d=0.3835,
         x0=1,
         x1=10,
         x2=-1.3390,
         c_d_lambda_0_res=None,
         a_d_lambda_0_res=None,
         omega_d_lambda_0_res=None,
         interpolate=True,
         a_d_res=[],
         a_md_spec_res=[],
         a_bd_spec_res=[],
         b_d_res = [],
         b_bd_res = [],
         b_bp_res = [],
         b_bw_res = [],
         b_i_spec_res = [],
         c_d_res = []):
    """
    Total backscattering coefficient of natural water and water constituents following [1]

    [1] Bi et al. (2023): Bio-geo-optical modelling of natural waters [10.3389/fmars.2023.11963529]

    Args:
        wavelengths (_type_, optional): _description_. Defaults to np.arange(400,800).
        C_0 (int, optional): _description_. Defaults to 0.
        C_1 (int, optional): _description_. Defaults to 0.
        C_2 (int, optional): _description_. Defaults to 0.
        C_3 (int, optional): _description_. Defaults to 0.
        C_4 (int, optional): _description_. Defaults to 0.
        C_5 (int, optional): _description_. Defaults to 0.
        C_ism (int, optional): _description_. Defaults to 0.
        b_ratio_C_0 (float, optional): _description_. Defaults to 0.002.
        b_ratio_C_1 (float, optional): _description_. Defaults to 0.002.
        b_ratio_C_2 (float, optional): _description_. Defaults to 0.002.
        b_ratio_C_3 (float, optional): _description_. Defaults to 0.002.
        b_ratio_C_4 (float, optional): _description_. Defaults to 0.002.
        b_ratio_C_5 (float, optional): _description_. Defaults to 0.002.
        b_ratio_d (float, optional): _description_. Defaults to 0.0216.
        fresh (bool, optional): _description_. Defaults to False.
        A_md (_type_, optional): _description_. Defaults to 13.4685e-3.
        A_bd (_type_, optional): _description_. Defaults to 0.3893e-3.
        S_md (_type_, optional): _description_. Defaults to 10.3845e-3.
        S_bd (_type_, optional): _description_. Defaults to 15.7621e-3.
        C_md (_type_, optional): _description_. Defaults to 12.1700e-3.
        C_bd (_type_, optional): _description_. Defaults to 0.9994e-3.
        lambda_0_md (_type_, optional): _description_. Defaults to 550..
        lambda_0_bd (_type_, optional): _description_. Defaults to 550..
        lambda_0_c_d (_type_, optional): _description_. Defaults to 550..
        gamma_d (float, optional): _description_. Defaults to 0.3835.
        x0 (int, optional): _description_. Defaults to 1.
        x1 (int, optional): _description_. Defaults to 10.
        x2 (float, optional): _description_. Defaults to -1.3390.
        c_d_lambda_0_res (_type_, optional): _description_. Defaults to None.
        a_d_lambda_0_res (_type_, optional): _description_. Defaults to None.
        omega_d_lambda_0_res (_type_, optional): _description_. Defaults to None.
        interpolate (bool, optional): _description_. Defaults to True.
        a_d_res (list, optional): _description_. Defaults to [].
        a_md_spec_res (list, optional): _description_. Defaults to [].
        a_bd_spec_res (list, optional): _description_. Defaults to [].
        b_d_res (list, optional): _description_. Defaults to [].
        b_bd_res (list, optional): _description_. Defaults to [].
        b_bp_res (list, optional): _description_. Defaults to [].
        b_bw_res (list, optional): _description_. Defaults to [].
        b_i_spec_res (list, optional): _description_. Defaults to [].
        c_d_res (list, optional): _description_. Defaults to [].

    Returns:
        np.array: Total backscattering coefficient of natural water and water constituents [m-1]
    """
    C_phy = np.sum([C_0, C_1, C_2, C_3, C_4, C_5, C_6, C_7])

    if len(b_bp_res)==0:
      # compute b_bp
      if len(b_bd_res)==0:
        # compute b_bd
        if len(b_d_res)==0:
          # compute b_d
          if len(a_d_res)==0:
            # compute a_d
            a_d_res = absorption.a_d(wavelengths=wavelengths,
                                     C_phy=C_phy, 
                                     C_ism=C_ism, 
                                     A_md=A_md, 
                                     A_bd=A_bd, 
                                     S_md=S_md, 
                                     S_bd=S_bd, 
                                     C_md=C_md, 
                                     C_bd=C_bd, 
                                     lambda_0_md=lambda_0_md, 
                                     lambda_0_bd=lambda_0_bd, 
                                     a_md_spec_res=a_md_spec_res, 
                                     a_bd_spec_res=a_bd_spec_res)
            
          a_d_lambda_0_res = np.interp(lambda_0_c_d, wavelengths, a_d_res) if interpolate else a_d_res[utils.find_closest(wavelengths, lambda_0_c_d)[1]]
          
          if len(c_d_res)==0:
            c_d_res = attenuation.c_d(wavelengths=wavelengths, 
                                      C_ism=C_ism, 
                                      C_phy=C_phy,
                                      A_md=A_md, 
                                      A_bd=A_bd, 
                                      S_md=S_md,
                                      S_bd=S_bd, 
                                      C_md=C_md,
                                      C_bd=C_bd, 
                                      lambda_0_c_d=lambda_0_c_d,
                                      lambda_0_md=lambda_0_md, 
                                      lambda_0_bd=lambda_0_bd, 
                                      gamma_d=gamma_d,
                                      x0=x0,
                                      x1=x1,
                                      x2=x2,
                                      c_d_lambda_0_res=c_d_lambda_0_res,
                                      a_d_lambda_0_res=a_d_lambda_0_res,
                                      omega_d_lambda_0_res=omega_d_lambda_0_res,
                                      a_md_spec_res=a_md_spec_res,
                                      a_bd_spec_res=a_bd_spec_res)
          b_d_res = scattering.b(a_d_res, c_d_res)
        b_bd_res = b_bd(b_d_res, b_ratio_d=b_ratio_d)
      b_bp_res = b_bphy_hereon(C_0=C_0, 
                               C_1=C_1, 
                               C_2=C_2, 
                               C_3=C_3, 
                               C_4=C_4, 
                               C_5=C_5, 
                               C_6=C_6,
                               C_7=C_7,
                               b_ratio_C_0=b_ratio_C_0, 
                               b_ratio_C_1=b_ratio_C_1, 
                               b_ratio_C_2=b_ratio_C_2, 
                               b_ratio_C_3=b_ratio_C_3, 
                               b_ratio_C_4=b_ratio_C_4, 
                               b_ratio_C_5=b_ratio_C_5, 
                               b_ratio_C_6=b_ratio_C_6, 
                               b_ratio_C_7=b_ratio_C_7, 
                               wavelengths=wavelengths, 
                               b_i_spec_res=b_i_spec_res) + b_bd_res
    b_b = b_bp_res + b_bw(wavelengths=wavelengths, fresh=fresh, b_bw_res=b_bw_res)

    return b_b
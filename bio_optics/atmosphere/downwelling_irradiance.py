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
from . import transmittance, ET_solar_irradiance


def Ed_d(wavelengths=np.arange(400,800), 
         theta_sun=np.radians(30), 
         P=1013.25, 
         AM=5, 
         RH=80, 
         H_oz=0.381, 
         WV=2.5, 
         alpha=1.317, 
         beta=0.2602, 
         E0_res=[],
         a_oz_res=[],
         a_ox_res=[],
         a_wv_res=[],
         Ed_d_res=[]):
    """
    Ed_d is the direct component of the downwelling irradiance, representing the sun disk in the sky as light source [1]. 
    
    [1] Gege, P. (2021): The Water Colour Simulator WASI. User manual for WASI version 6.
    
    :param wavelengths: wavelengths to compute tau_a fpr, default: np.arange(400,800)
    :param theta_sun: sun zenith angle [radians], default: np.radians(30)
    :param P: atmospheric pressure [mbar], default: 1013.25
    :param AM: air mass type [1: open ocean aerosols .. 10: continental aerosols], default: 5
    :param RH: relative humidity [%] (typical values range from 46 to 91 %), default: 80
    :param H_oz: ozone scale height [cm], default: 0.381
    :param WV: precipitable water [cm], default: 2.5
    :param alpha: Angström exponent determining wavelength dependency (typically ranges from 0.2 to 2 [1]), default: 1.317
    :param beta: turbidity coefficient as a measure of concentration (typically ranges from 0.16 to 0.50 [1]), default: 0.2606
    :param E0_res: optional, precomputing E_0 saves a lot of time.
    :param a_oz_res: optional, precomputing a_oz saves a lot of time.
    :param a_ox_res: optional, precomputing a_ox saves a lot of time.
    :param a_wv_res: optional, precomputing a_wv saves a lot of time.
    :return: Ed_d
    """
    if len(Ed_d_res)==0:
        Ed_d = ET_solar_irradiance.E0(wavelengths,E0_res=E0_res) * np.cos(theta_sun) * \
               transmittance.T_r(wavelengths, theta_sun=theta_sun, P=P) * \
               transmittance.T_aa(wavelengths, theta_sun=theta_sun, AM=AM, RH=RH, alpha=alpha, beta=beta) * \
               transmittance.T_as(wavelengths, theta_sun=theta_sun, AM=AM, RH=RH, alpha=alpha, beta=beta) * \
               transmittance.T_oz(wavelengths, theta_sun=theta_sun, H_oz=H_oz, a_oz_res=a_oz_res) * \
               transmittance.T_ox(wavelengths, theta_sun=theta_sun, P=P, a_ox_res=a_ox_res) * \
               transmittance.T_wv(wavelengths, theta_sun=theta_sun, WV=WV, a_wv_res=a_wv_res)
    else:
        Ed_d=Ed_d_res
        
    return Ed_d
    
def Ed_sr(wavelengths=np.arange(400,800), 
          theta_sun=np.radians(30), 
          P=1013.25, 
          AM=5, 
          RH=80, 
          H_oz=0.380, 
          WV=2.5, 
          alpha=1.317, 
          beta=0.2602, 
          E0_res=[],
          a_oz_res=[],
          a_ox_res=[],
          a_wv_res=[],
          Ed_sr_res=[]):
    """
    Ed_sr represents Rayleigh scattering as part of the diffuse component of downwelling irradiance [1].
    
    [1] Gege, P. (2021): The Water Colour Simulator WASI. User manual for WASI version 6.
    
    :param wavelengths: wavelengths to compute tau_a fpr, default: np.arange(400,800)
    :param theta_sun: sun zenith angle [radians], default: np.radians(30)
    :param P: atmospheric pressure [mbar], default: 1013.25
    :param AM: air mass type [1: open ocean aerosols .. 10: continental aerosols], default: 5
    :param RH: relative humidity [%] (typical values range from 46 to 91 %), default: 80
    :param H_oz: ozone scale height [cm], default: 0.381
    :param WV: precipitable water [cm], default: 2.5
    :param E0_res: optional, precomputing E_0 saves a lot of time.
    :param a_oz_res: optional, precomputing a_oz saves a lot of time.
    :param a_ox_res: optional, precomputing a_ox saves a lot of time.
    :param a_wv_res: optional, precomputing a_wv saves a lot of time.
    :return: Ed_sr
    """
    if len(Ed_sr_res)==0:
        Ed_sr = 0.5 * ET_solar_irradiance.E0(wavelengths, E0_res=E0_res) * np.cos(theta_sun) * \
                (1 - transmittance.T_r(wavelengths, theta_sun=theta_sun, P=P)**0.95) * \
                transmittance.T_aa(wavelengths, theta_sun=theta_sun, AM=AM, RH=RH, alpha=alpha, beta=beta) * \
                transmittance.T_oz(wavelengths, theta_sun=theta_sun, H_oz=H_oz, a_oz_res=a_oz_res) * \
                transmittance.T_ox(wavelengths, theta_sun=theta_sun, P=P, a_ox_res=a_ox_res) * \
                transmittance.T_wv(wavelengths, theta_sun=theta_sun, WV=WV, a_wv_res=a_wv_res)
    else:
        Ed_sr=Ed_sr_res
        
    return Ed_sr
 
def Ed_sa(wavelengths=np.arange(400,800), 
          theta_sun=np.radians(30), 
          P=1013.25, 
          AM=5, 
          RH=80, 
          H_oz=0.38, 
          WV=2.5, 
          alpha=1.317, 
          beta=0.2606, 
          E0_res=[],
          a_oz_res=[],
          a_ox_res=[],
          a_wv_res=[],
          Ed_sa_res=[]):
    """
    Ed_sa represents aerosol scattering as part of the diffuse component of downwelling irradiance [1].
    
    [1] Gege, P. (2021): The Water Colour Simulator WASI. User manual for WASI version 6.
    
    :param wavelengths: wavelengths to compute tau_a fpr, default: np.arange(400,800)
    :param theta_sun: sun zenith angle [radians], default: np.radians(30)
    :param P: atmospheric pressure [mbar], default: 1013.25
    :param AM: air mass type [1: open ocean aerosols .. 10: continental aerosols], default: 5
    :param RH: relative humidity [%] (typical values range from 46 to 91 %), default: 80
    :param H_oz: ozone scale height [cm], default: 0.381
    :param WV: precipitable water [cm], default: 2.5
    :param alpha: Angstroem exponent determining wavelength dependency (typically ranges from 0.2 to 2 [1]), default: 1.317
    :param beta: turbidity coefficient as a measure of concentration (typically ranges from 0.16 to 0.50 [1]), default: 0.2606
    :param E0_res: optional, precomputing E_0 saves a lot of time.
    :param a_oz_res: optional, precomputing a_oz saves a lot of time.
    :param a_ox_res: optional, precomputing a_ox saves a lot of time.
    :param a_wv_res: optional, precomputing a_wv saves a lot of time.
    :return: Ed_sa
    """
    if len(Ed_sa_res)==0:
        Ed_sa = ET_solar_irradiance.E0(wavelengths, E0_res=E0_res) * np.cos(theta_sun) * \
                transmittance.T_r(wavelengths, theta_sun=theta_sun, P=P)**1.5 * \
                transmittance.T_aa(wavelengths, theta_sun=theta_sun, AM=AM, RH=RH, alpha=alpha, beta=beta) * \
                transmittance.T_oz(wavelengths, theta_sun=theta_sun, H_oz=H_oz, a_oz_res=a_oz_res) * \
                transmittance.T_ox(wavelengths, theta_sun=theta_sun, P=P, a_ox_res=a_ox_res) * \
                transmittance.T_wv(wavelengths, theta_sun=theta_sun, WV=WV, a_wv_res=a_wv_res) * \
                (1 - transmittance.T_as(wavelengths, theta_sun=theta_sun, AM=AM, RH=RH, alpha=alpha, beta=beta)) * \
                transmittance.F_a(theta_sun=theta_sun, alpha=alpha)
    else:
        Ed_sa=Ed_sa_res
    
    return Ed_sa
    
def Ed_s(Ed_sr, Ed_sa):
    return Ed_sr + Ed_sa

def Ed(Ed_d,
        E_ds,
        f_dd=1,
        f_ds=1
        ):
    """
    Downwelling irradiance is split into a direct and a diffuse component [1]: 
    * Ed_d is the direct component of the downwelling irradiance, representing the sun disk in the sky as light source. 
    * Ed_s is the radiation from the sky, i.e. the diffuse downwelling irradiance. It is split into two components Ed_sr and Ed_sa:
        * Ed_sr represents Rayleigh scattering
        * Ed_sa represents aerosol scattering
     
     [1] Gege, P. (2021): The Water Colour Simulator WASI. User manual for WASI version 6.
    """

    return f_dd * Ed_d + f_ds * E_ds

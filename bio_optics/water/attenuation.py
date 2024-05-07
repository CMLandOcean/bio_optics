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
from .. surface import air_water
from . import absorption


def omega_b(a, b_b):
    """
    Single scattering albedo
    
    # Math: \omega_b = \frac{b_b}{a + b_b}
    """
    return b_b / (a + b_b)


def domega_b_div_dp(a, b_b, da_div_dp, db_b_div_dp):
    """
    # Math: \frac{\partial}{\partial p}\left[\omega_b\right] = \frac{\partial}{\partial p}\left[ b_b \times (a+b_b)^{-1} \right]
    # Math: =\frac{\partial b_b}{\partial p} (a + b_b)^{-1} - b_b(a+b_b)^{-2}(\frac{\partial a}{\partial p} + \frac{\partial b_b}{\partial p})
    # Math: = \frac{a \frac{\partial b_b}{\partial p} - b \frac{\partial a}{\partial p}}{(a + b_b)^2}
    """
    return (a * db_b_div_dp - b_b * da_div_dp) / (a + b_b)**2


def K_d(a,
        b_b, 
        cos_t_sun_p=np.pi/6,
        kappa_0=1.0546,
        ):
    """
    Diffuse attenuation for downwelling irradiance as implemented in WASI [1].
    
    [1] Gege, P. (2021): The Water Colour Simulator WASI. User manual for WASI version 6.
    [2] Albert, A., & Mobley, C. (2003): An analytical model for subsurface irradiance and remote sensing reflectance in deep and shallow case-2 waters [doi.org/10.1364/OE.11.002873]
    
    :param a: spectral absorption coefficient of a water body
    :param b_b: spectral backscattering coefficient of a water body
    :param cos_t_sun_p: cosine of sun zenith angle (expected to be adjusted for refraction angle using Snell's law) in air
    :param kappa_0: coefficient depending on scattering phase function, default: 1.0546 [2]
    :return: diffuse attenuation for downwelling irradiance

    # Math: K_d(\lambda) = \kappa_0 \frac{a + b_b}{cos\theta_{sun}'}
    """
    K_d = (kappa_0 / cos_t_sun_p) * (a + b_b)
    
    return K_d


def dK_d_div_dp(da_div_dp,
                db_b_div_dp,
                cos_t_sun_p=np.cos(np.pi/6),
                kappa_0=1.0546):
    """
    # Math: \frac{\partial}{\partial p} \left[ \frac{k_0}{cos \theta_{sun}'} (a + b_b) \right] = \frac{k_0}{cos \theta_{sun}'} (\frac{\partial a}{\partial p} + \frac{\partial b_b}{\partial p})
    """
    return (kappa_0 / cos_t_sun_p) * (da_div_dp + db_b_div_dp)
    

def k_uW(a,
         b_b,
         omega_b,
         cos_t_sun_p,
         cos_t_view_p):
    """
    # Math: k_{uW} = \frac{a + b_b}{cos \theta_v'} \times (1 + \omega_b)^{3.5421} \times (1 - \frac{0.2786}{cos \theta_{sun}'})
    """
    return (a + b_b) / cos_t_view_p * (1 + omega_b)**3.5421 * (1 - 0.2786 / cos_t_sun_p)

def dk_uW_div_dp(a, 
                 b_b,
                 omega_b,
                 da_div_dp, 
                 db_b_div_dp,
                 domega_b_div_dp,
                 cos_t_sun_p, 
                 cos_t_view_p):
    """
    # Math: \frac{\partial}{\partial p}\left[k_{uW}\right] = \frac{\partial}{\partial p}\left[\frac{a + b_b}{cos \theta_v'} \times (1 + \omega_b)^{3.5421} \times (1 - \frac{0.2786}{cos \theta_{sun}'})\right]
    # Math: = \frac{1}{cos \theta_v'}\left[ \frac{\partial}{\partial p}(a + b_b) \times (1 + \omega_b)^{3.5421} + (a + b_b) \times \frac{\partial}{\partial p}((1 + \omega_b)^{3.5421})\right]\times (1 - \frac{0.2786}{cos \theta_{sun}'})
    # Math: = \frac{1}{cos \theta_v'}\left[ (\frac{\partial a}{\partial p} + \frac{\partial b}{\partial p}) \times (1 + \omega_b)^{3.5421} + (a + b) \times (3.5421 \times (1 + \omega_b)^{2.5421} \times \frac{\partial \omega_b}{\partial p}) \right] \times (1 - \frac{0.2786}{cos \theta_{sun}'})
    """

    return 1/cos_t_sun_p * \
          (
            ((da_div_dp + db_b_div_dp) * (1 + omega_b)**(3.5421) + \
             (a + b_b) * (3.5421 * (1 + omega_b)**(2.5421) * domega_b_div_dp))
          ) * \
          (1 - 0.2786/ cos_t_view_p)


def k_uB(a,
         b_b,
         omega_b,
         cos_t_sun_p,
         cos_t_view_p):
    """
    # Math: k_{uB} = \frac{a + b_b}{cos \theta_v'} \times (1 + \omega_b)^{2.2658} \times (1 + \frac{0.0577}{cos \theta_{sun}'})
    """
    return (a + b_b) / cos_t_view_p * (1 + omega_b)**2.2658 * (1 + 0.0577 / cos_t_sun_p)


def dk_uB_div_dp(a, 
                 b_b,
                 omega_b, 
                 da_div_dp, 
                 db_b_div_dp,
                 domega_b_div_dp,
                 cos_t_sun_p, 
                 cos_t_view_p):
    """
    # Math: \frac{\partial}{\partial p}\left[k_{uB}\right] = \frac{\partial}{\partial p}\left[\frac{a + b_b}{cos \theta_v'} \times (1 + \omega_b)^{2.2658} \times (1 + \frac{0.0577}{cos \theta_{sun}'})\right]
    # Math: = \frac{1}{cos \theta_v'}\left[ \frac{\partial}{\partial p}(a + b_b) \times (1 + \omega_b)^{2.2658} + (a + b_b) \times \frac{\partial}{\partial p}((1 + \omega_b)^{2.2658})\right]\times (1 + \frac{0.0577}{cos \theta_{sun}'})
    # Math: = \frac{1}{cos \theta_v'}\left[ (\frac{\partial a}{\partial p} + \frac{\partial b}{\partial p}) \times (1 + \omega_b)^{2.2658} + (a + b) \times (2.2658 \times (1 + \omega_b)^{1.2658} \times \frac{\partial \omega_b}{\partial p}) \right] \times (1 + \frac{0.0577}{cos \theta_{sun}'})
    """
    return 1/cos_t_sun_p * \
          (
            ((da_div_dp + db_b_div_dp) * (1 + omega_b)**(2.2658) + \
             (a + b_b) * (2.2658 * (1 + omega_b)**(1.2658) * domega_b_div_dp))
          ) * \
          (1 + 0.0577 / cos_t_view_p)


################
#### HEREON ####
################


def omega_d_lambda_0(x0=1., 
                     x1=10, 
                     x2=-1.3390):
    """
    Single scattering albedo of detritus at a reference wavelength.

    Args:
        x0 (_type_, optional): Minuend. Defaults to 1..
        x1 (int, optional): Base of the power-law function. Defaults to -10.
        x2 (float, optional): Exponent of the power-law function. Defaults to -1.3390 (± 0.0618).

    Returns:
        omega_d_lambda_0: Single scattering albedo of detritus at a reference wavelength.
    """
    omega_d_lambda_0 = x0 - np.power(x1, x2)

    return omega_d_lambda_0


def estimate_c_d_lambda_0(C_ism=1., 
                          C_phy=1.,
                          A_md=13.4685e-3, 
                          A_bd=0.3893e-3, 
                          S_md=10.3845e-3,
                          S_bd=15.7621e-3, 
                          C_md=12.1700e-3,
                          C_bd= 0.9994e-3, 
                          lambda_0_c_d=550., 
                          lambda_0_md=550., 
                          lambda_0_bd=550., 
                          x0=1.,
                          x1=10,
                          x2=-1.3390,
                          omega_d_lambda_0_res=None,
                          a_d_lambda_0_res = None,
                          a_md_spec_res=[],
                          a_bd_spec_res=[]):
    """
    Helper function to estimate the attenuation coefficient of detritus at a reference wavelength (Eq. 10 in [1]).

    [1] Bi et al. (2023): Bio-geo-optical modelling of natural waters [10.3389/fmars.2023.11963529]

    Args:
        lambda_0 (int, optional): _description_. Defaults to 550.
        C_ism (_type_, optional): _description_. Defaults to 1..
        C_phy (_type_, optional): _description_. Defaults to 1..
        A_md (_type_, optional): _description_. Defaults to 13.4685e-3.
        A_bd (_type_, optional): _description_. Defaults to 0.3893e-3.
        S_md (_type_, optional): _description_. Defaults to 10.3845e-3.
        S_bd (_type_, optional): _description_. Defaults to 15.7621e-3.
        C_md (_type_, optional): _description_. Defaults to 12.1700e-3.
        C_bd (_type_, optional): _description_. Defaults to 0.9994e-3.
        lambda_0_md (_type_, optional): _description_. Defaults to 500..
        lambda_0_bd (_type_, optional): _description_. Defaults to 500..
        x0 (int, optional): _description_. Defaults to -1.
        x1 (int, optional): _description_. Defaults to 10.
        x2 (float, optional): _description_. Defaults to -1.3390.
        omega_d_lambda_0_pre (_type_, optional): _description_. Defaults to None.
        a_md_spec_res (list, optional): _description_. Defaults to [].
        a_bd_spec_res (list, optional): _description_. Defaults to [].

    Returns:
        c_d_lambda_0: estimated attenuation coefficient of detritus at reference wavelength [m-1]
    """
    if omega_d_lambda_0_res is None:
        omega_d_550 = omega_d_lambda_0(x0=x0,x1=x1,x2=x2)
    else:
        omega_d_550 = omega_d_lambda_0_res

    if a_d_lambda_0_res is None:
        a_d_lambda_0 = absorption.a_d(wavelengths=lambda_0_c_d,
                                      C_phy=C_phy, 
                                      C_ism=C_ism,
                                      A_md=A_md,
                                      A_bd=A_bd,
                                      S_md=S_md,
                                      S_bd=S_bd,
                                      C_md=C_md,
                                      C_bd= C_bd,
                                      lambda_0_md=lambda_0_md,
                                      lambda_0_bd=lambda_0_bd,
                                      a_md_spec_res=a_md_spec_res,
                                      a_bd_spec_res=a_bd_spec_res)
    else:
        a_d_lambda_0 = a_d_lambda_0_res
           
    c_d_lambda_0 =  a_d_lambda_0 / (1 - omega_d_550)
    
    return c_d_lambda_0


def c_d(wavelengths=np.arange(400,800), 
        C_ism=1., 
        C_phy=1.,
        A_md=13.4685e-3, 
        A_bd=0.3893e-3, 
        S_md=10.3845e-3,
        S_bd=15.7621e-3, 
        C_md=12.1700e-3,
        C_bd= 0.9994e-3, 
        lambda_0_c_d=550., 
        lambda_0_md=550., 
        lambda_0_bd=550., 
        gamma_d=0.3835,
        x0=1,
        x1=10,
        x2=-1.3390,
        c_d_lambda_0_res=None,
        omega_d_lambda_0_res=None,
        a_d_lambda_0_res = None,
        a_md_spec_res=[],
        a_bd_spec_res=[]):
    """
    Attenuation coefficient of detritus (Eq. 9 in [1]).

    [1] Bi et al. (2023): Bio-geo-optical modelling of natural waters [10.3389/fmars.2023.11963529]

    Args:
        wavelengths (_type_, optional): _description_. Defaults to np.arange(400,800).
        C_ism (_type_, optional): _description_. Defaults to 1..
        C_phy (_type_, optional): _description_. Defaults to 1..
        A_md (_type_, optional): _description_. Defaults to 13.4685e-3.
        A_bd (_type_, optional): _description_. Defaults to 0.3893e-3.
        S_md (_type_, optional): _description_. Defaults to 10.3845e-3.
        S_bd (_type_, optional): _description_. Defaults to 15.7621e-3.
        C_md (_type_, optional): _description_. Defaults to 12.1700e-3.
        C_bd (_type_, optional): _description_. Defaults to 0.9994e-3.
        lambda_0 (_type_, optional): _description_. Defaults to 550..
        lambda_0_md (_type_, optional): _description_. Defaults to 550..
        lambda_0_bd (_type_, optional): _description_. Defaults to 550..
        gamma_d (float, optional): Exponential of power-law function [nm-1]. Defaults to 0.3835 (± 0.1277)
        x0 (int, optional): _description_. Defaults to -1.
        x1 (int, optional): _description_. Defaults to 10.
        x2 (float, optional): _description_. Defaults to –1.3390 (± 0.0618).
        c_d_lambda_0 (_type_, optional): _description_. Defaults to None.
        omega_d_lambda_0_pre (_type_, optional): _description_. Defaults to None.
        a_md_spec_res (list, optional): _description_. Defaults to [].
        a_bd_spec_res (list, optional): _description_. Defaults to [].

    Returns:
        c_d: spectral attenuation coefficient of detritus [m-1]
    """
    if c_d_lambda_0_res is None:
        c_d_lambda_0_res = estimate_c_d_lambda_0(C_ism=C_ism,
                                             C_phy=C_phy,
                                             A_md=A_md,
                                             A_bd=A_bd,
                                             S_md=S_md,
                                             S_bd=S_bd,
                                             C_md=C_md,
                                             C_bd=C_bd,
                                             lambda_0_md=lambda_0_md,
                                             lambda_0_bd=lambda_0_bd,
                                             lambda_0_c_d=lambda_0_c_d,
                                             x0=x0,
                                             x1=x1,
                                             x2=x2,
                                             omega_d_lambda_0_res=omega_d_lambda_0_res,
                                             a_d_lambda_0_res = a_d_lambda_0_res,
                                             a_md_spec_res=[], # needs to be empty so function does not return a vector 
                                             a_bd_spec_res=[]) # needs to be empty so function does not return a vector 

    c_d = c_d_lambda_0_res * (lambda_0_c_d / wavelengths)**gamma_d
    return c_d 


def K_d_Lee(a_t, 
            b_b, 
            theta_sun=30, 
            m1=4.18, 
            m2=0.52, 
            m3=-10.8):
    """
    Diffuse attenuation coefficient for downwelling irradiance [m-1] 
    following Lee et al. (2005) [1] as described in Barnes et al. (2013) [2]

    [1] Lee et al. (2005): Diffuse attenuation coefficient of downwelling irradiance: An evaluation of remote sensing methods [10.1029/2004JC002573]
    [2] Barnes et al. (2013): MODIS-derived spatiotemporal water clarity patterns in optically shallow Florida Keys waters: A new approach to remove bottom contamination [10.1016/j.rse.2013.03.016]

    Args:
        a_t (np.array): total absorption coefficient [m-1]
        b_b (np.array): backscattering coefficient [m-1]
        theta_sun (float): solar zenith angle in air [degrees]
        m1 (float, optional): Constant. Defaults to 4.18.
        m2 (float, optional): Constant. Defaults to 0.52.
        m3 (float, optional): Constant. Defaults to -10.8.
    
    Returns: 
        K_d: Diffuse attenuation coefficient for downwelling irradiance [m-1]
    """
    # Eq. 1 in [2]
    K_d = (1 + 0.005*theta_sun)*a_t + m1*(1 - m2*np.exp(m3*a_t)) * b_b
    
    return K_d


def estimate_c(a, b_bp, b_bw, eta_p=0.015, eta_w=0.5):
    """
    Estimation of the beam attenuation coefficient c [m-1] as decsribed in Eq. 5 in McKinna & Werdell (2018) [1].
    In [1] this function is used to estimate c at a reference wavelength of 547 nm.

    [1] McKinna & Werdell (2018): Approach for identifying optically shallow pixels when processing ocean-color imagery [10.1364/OE.26.00A915]

    Args:
        a (np.array): absorption coefficient [m-1]
        b_bp (np.array): particulate backscattering coefficient [m-1]
        b_bw (np.array): backscattering coefficient of water [m-1]
        eta_p (float, optional): particulate backscatter ratio. Defaults to 0.015; halfway between global average oceanic value of 0.01 and well known Petzold average particle value of 0.0183.
        eta_w (float, optional): backscatter ratio of pure water. Defaults to 0.5.

    Returns:
        c: beam attenuation coefficient [m-1]
    """
    c = a + b_bp / eta_p + b_bw / eta_w

    return c
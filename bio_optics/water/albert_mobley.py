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
from .. water import absorption, backscattering, attenuation, bottom_reflectance
from .. helper import resampling

def f_rs(omega_b,
        cos_t_sun_p = np.cos(np.radians(30)),
        cos_t_view_p = np.cos(np.radians(0))):
    """
    # Math: f_{rs} = 0.0512 \times (1 + 4.6659 \times \omega_b - 7.8387 \times \omega_b^2 + 5.4571 \times \omega_b^3) \times (1 + \frac{0.1098}{cos \theta_{sun}'}) \times (1 + \frac{0.4021}{cos \theta_{sun}'})
    
    Applying Horner's method:
    # Math: f_{rs} = 0.0512 \times (1 + \omega_b \times (4.6659 + \omega_b \times(-7.8387 + \omega_b \times (5.4571))) \times (1 + \frac{0.1098}{cos \theta_{sun}'}) \times (1 + \frac{0.4021}{cos \theta_{sun}'})
    """

    return 0.0512 * (1 + omega_b * (4.6659 + omega_b * (-7.8387 + omega_b * (5.4571)))) * (1 + 0.1098 / cos_t_sun_p) * (1 + 0.4021 / cos_t_view_p)
    
def df_rs_div_dp(omega_b,
                domega_b_div_dp,
                cos_t_sun_p = np.cos(np.radians(30)),
                cos_t_view_p = np.cos(np.radians(0))):
    """
    Generalized partial derivative for f_rs with respect to Fit Param (p)
    The only independent variable of the polynomial f_rs is omega_b
    and the derivative will have the same underlying for for all lower
    level variables.

    # Math: f_{rs} = 0.0512 \times (1 + \omega_b \times (4.6659 + \omega_b \times(-7.8387 + \omega_b \times (5.4571))) \times (1 + \frac{0.1098}{cos \theta_{sun}'}) \times (1 + \frac{0.4021}{cos \theta_{sun}'})
    # Math: \frac{\partial}{\partial p}\left[f_{rs}\right] = 0.0512 \times \frac{\partial \omega_b}{\partial p} \times (4.6659 + \omega_b \times (2 \times  (-7.8387) + \omega_b \times (3 \times 5.4571))
    # Math: \frac{\partial}{\partial p}\left[f_{rs}\right] = 0.0512 \times \frac{\partial \omega_b}{\partial p} \times (4.6659 + \omega_b \times (-15.6774 + \omega_b \times (16.3713))
    """
    return 0.0512 * (1 + 0.1098 / cos_t_sun_p) * (1 + 0.4021 / cos_t_view_p) * domega_b_div_dp * (4.6659 + omega_b * ((-15.6774) + (16.3713) * omega_b))


def r_rs_deep(f_rs, omega_b):
    """
    Subsurface radiance reflectance of optically deep water after Albert & Mobley (2003) [1].
    
    [1] Albert & Mobley (2003): An analytical model for subsurface irradiance and remote sensing reflectance in deep and shallow case-2 waters. [10.1364/OE.11.002873]

    :param u: ratio of backscattering coefficient to the sum of absorption and backscattering coefficients
    :param theta_sun: sun zenith angle in air [radians], is converted to in water using Snell's law, default: np.radians(30)
    :param theta_view: viewing angle in air in units [radians], is converted to in water using Snell's law, np.radians(0)
    :param n1: refrective index of origin medium, default: 1 for air
    :param n2: refrective index of destination medium, default: 1.33 for water
    :return: subsurface radiance reflectance of deep water [sr-1]

    # Math: r_{rs}^{deep-} = f_{rs} * \omega_{b}
    """
    return f_rs * omega_b

def dr_rs_deep_div_dp(f_rs,
                      df_rs_div_dp,
                      omega_b,
                      domega_b_div_dp):
    """
    # Math: \frac{\partial}{\partial p} \left[ f_{rs} * \omega_b \right] = \frac{\partial f_{rs}}{\partial p} * \omega_b + f_{rs} * \frac{\partial \omega_b}{\partial p}
    """
    return df_rs_div_dp * omega_b + f_rs * domega_b_div_dp


def r_rs_shallow(r_rs_deep,
                 K_d,
                 k_uW,
                 zB,
                 R_rs_b,
                 k_uB,
                 A_rs1=1.1576,
                 A_rs2=1.0389):
    """
    Subsurface radiance reflectance of optically shallow water after Albert & Mobley (2003) [1].
    
    [1] Albert & Mobley (2003): An analytical model for subsurface irradiance and remote sensing reflectance in deep and shallow case-2 waters. [10.1364/OE.11.002873]
    [2] Heege, T. (2000): Flugzeuggestützte Fernerkundung von Wasserinhaltsstoffen am Bodensee. PhD thesis. DLR-Forschungsbericht 2000-40, 134 p.
    [3] Albert, A., & Mobley, C. (2003): An analytical model for subsurface irradiance and remote sensing reflectance in deep and shallow case-2 waters [doi.org/10.1364/OE.11.002873]

    # Math: r_{rs}^{sh-} = r_{rs}^{deep-} * \left[ 1 - A_{rs,1} * e^{-(K_d + k_{uW}) * zB} \right] + A_{rs,2} * R_{rs}^b * e^{-(K_d + k_{uB}) * zB}
    """
    return r_rs_deep * \
            (1 - A_rs1 * np.exp(-(K_d + k_uW) * zB)) + \
            A_rs2 * R_rs_b * np.exp(-(K_d + k_uB) * zB)

def drs_rs_shallow_div_dp(r_rs_deep,
                          dr_rs_deep_div_dp,
                          K_d,
                          dK_d_div_dp,
                          k_uW,
                          dk_uW_div_dp,
                          r_rs_b,
                          d_r_rs_b_div_dp,
                          k_uB,
                          dk_uB_div_dp,
                          zB,
                          A_rs1=1.1576,
                          A_rs2=1.0389):
    """
    # Math: \frac{\partial}{\partial p}\left[ r_{rs}^{sh-} \right] = \left [ \frac{\partial r_{rs}^{deep-}}{\partial p} * \left[ 1 - A_{rs,1} * e^{-(K_d + k_{uW}) * zB} \right] + r_{rs}^{deep-} * \left[ 1 - A_{rs,1} * \frac{\partial e^{-(K_d + k_{uW}) * zB}}{\partial p} \right] \right] + \left[ A_{rs,2} * \frac{\partial R_{rs}^b}{\partial p} * e^{-(K_d + k_{uB}) * zB} + A_{rs,2} * R_{rs}^b * \frac{\partial e^{-(K_d + k_{uB}) * zB}}{\partial p} \right]
    # Math: = \left [ \frac{\partial r_{rs}^{deep-}}{\partial p} * \left[ 1 - A_{rs,1} * e^{-(K_d + k_{uW}) * zB} \right] + r_{rs}^{deep-} * \left[ 1 - A_{rs,1} * -(\frac{\partial K_d}{\partial p} + \frac{\partial k_{uW}}{\partial p})e^{-(K_d + k_{uW}) * zB} \right] \right] + \left[ A_{rs,2} * \frac{\partial R_{rs}^b}{\partial p} * e^{-(K_d + k_{uB}) * zB} + A_{rs,2} * R_{rs}^b * -(\frac{\partial K_d}{\partial p} + \frac{\partial k_{uB}}{\partial p}e^{-(K_d + k_{uB}) * zB} \right]
    """
    eW = np.exp(-(K_d + k_uW) * zB)
    deW = -zB * (dK_d_div_dp + dk_uW_div_dp) * eW
    eB = np.exp(-(K_d + k_uB) * zB)
    deB = -zB * (dK_d_div_dp + dk_uB_div_dp) * eB
    
    return dr_rs_deep_div_dp * (1 - A_rs1 * eW) + \
            r_rs_deep * (-A_rs1 * deW) + \
            A_rs2 * \
            (
                d_r_rs_b_div_dp * eB + \
                r_rs_b * deB
            )

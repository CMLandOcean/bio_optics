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
from ..atmosphere import sky_radiance, downwelling_irradiance
from . import air_water


def forward(parameters,
            wavelengths,
            E0_res=[],
            a_oz_res=[],
            a_ox_res=[],
            a_wv_res=[],
            Ed_d_res=[],
            Ed_sa_res=[],
            Ed_sr_res=[],
            Ed_s_res=[],
            Ed_res=[],
            n2_res=[],
            Ls_Ed=[]):
    """
    Forward simulation of the surface reflectance (sky glint) contribution to above-water remote sensing reflectance.

    Args:
        parameters: lmfit Parameters object
        wavelengths: wavelengths [nm]
        E0_res: optional precomputed extraterrestrial solar irradiance
        a_oz_res: optional precomputed ozone absorption
        a_ox_res: optional precomputed oxygen absorption
        a_wv_res: optional precomputed water vapour absorption
        Ed_d_res: optional precomputed direct downwelling irradiance
        Ed_sa_res: optional precomputed aerosol-scattered downwelling irradiance
        Ed_sr_res: optional precomputed Rayleigh-scattered downwelling irradiance
        Ed_s_res: optional precomputed diffuse downwelling irradiance
        Ed_res: optional precomputed total downwelling irradiance
        n2_res: optional precomputed refractive index of water
        Ls_Ed: optional ratio of sky radiance to downwelling irradiance

    Returns:
        Rrs_surface: surface reflectance contribution [sr-1]
    """
    n2 = n2_res if len(n2_res) > 0 else parameters["n2"]

    if "rho_L" in parameters:
        rho_L = parameters["rho_L"].value
    else:
        rho_L = air_water.fresnel(parameters["theta_view"], n1=parameters["n1"], n2=n2)

    Ls_Ed = np.zeros_like(wavelengths) if len(Ls_Ed) == 0 else Ls_Ed

    Ed_d  = Ed_d_res  if len(Ed_d_res)  > 0 else downwelling_irradiance.Ed_d( wavelengths, parameters["theta_sun"], parameters["P"], parameters["AM"], parameters["RH"], parameters["H_oz"], parameters["WV"], parameters["alpha"], parameters["beta"], E0_res, a_oz_res, a_ox_res, a_wv_res)
    Ed_sa = Ed_sa_res if len(Ed_sa_res) > 0 else downwelling_irradiance.Ed_sa(wavelengths, parameters["theta_sun"], parameters["P"], parameters["AM"], parameters["RH"], parameters["H_oz"], parameters["WV"], parameters["alpha"], parameters["beta"], E0_res, a_oz_res, a_ox_res, a_wv_res)
    Ed_sr = Ed_sr_res if len(Ed_sr_res) > 0 else downwelling_irradiance.Ed_sr(wavelengths, parameters["theta_sun"], parameters["P"], parameters["AM"], parameters["RH"], parameters["H_oz"], parameters["WV"], parameters["alpha"], parameters["beta"], E0_res, a_oz_res, a_ox_res, a_wv_res)
    Ed_s  = Ed_s_res  if len(Ed_s_res)  > 0 else downwelling_irradiance.Ed_s(Ed_sr, Ed_sa)
    Ed    = Ed_res    if len(Ed_res)    > 0 else downwelling_irradiance.Ed(Ed_d, Ed_s, parameters["fd_d"], parameters["fd_s"])

    L_s = sky_radiance.L_s(parameters["fd_d"], parameters["g_dd"], Ed_d,
                            parameters["fd_s"], parameters["g_dsr"], Ed_sr,
                            parameters["g_dsa"], Ed_sa)

    Rrs_surface = Rrs_surf(L_s, Ed, rho_L, parameters["d_r"])
    Rrs_surface += air_water.fresnel(parameters["theta_view"], n2=n2) * Ls_Ed
    return Rrs_surface


def L_surf(L_s, rho_L):
    return rho_L * L_s

def Rrs_surf(L_s, Ed, rho_L=0.02, d_r=0):
    return (L_surf(L_s, rho_L) / Ed) + d_r

def dRrs_surf_dp(L_s, dL_s_div_dp, Ed, dEd_div_dp, rho_L):
    return rho_L * ((dL_s_div_dp / Ed) + (L_s * Ed**-2 * dEd_div_dp))


def rsoa(wavelengths=np.arange(400, 800), h0=0.0, h1=0.0, lambda0=550.0):
    """
    Power-law glint model after Lin et al. (2023) [1] as part of the revised spectral optimization approach (RSOA).

    [1] Lin et al. (2023): Revised spectral optimization approach to remove surface-reflected radiance for the estimation of remote-sensing reflectance from the above-water method [10.1364/OE.486981]

    Args:
        wavelengths: wavelengths to compute rho for, default: np.arange(400,800)
        h0 (float, optional): Defaults to 0. Boundaries are h0 < 0.5. [1].
        h1 (float, optional): Defaults to 0. Boundaries are -0.1 < h1 < 0.5. [1].
        lambda0 (float, optional): Reference wavelengths. Defaults to 550.

    Returns:
        rho: sea-surface skylight reflectance for provided wavelengths [sr-1]
    """
    rho = h0 * (wavelengths / lambda0) ** h1
    return rho
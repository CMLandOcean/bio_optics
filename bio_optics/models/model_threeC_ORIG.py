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
# bio_optics
#  A former version of this code was provided to Planet, PBC as part of the CarbonMapper Land and Ocean Program.
#  It builds on the extensive work of many researchers. For example, models were developed  
#  by Albert & Mobley [1] and Gege [2]; the methodology was mainly developed 
#  by Gege [3,4,5] and Albert & Gege [6].
#
#  Please give proper attribution when using this code for publication:
#
#  König, M., Hondula. K.L., Jamalinia, E., Dai, J., Vaughn, N.R., Asner, G.P. (2023): bio_optics python package (Version x) [Software]. Available from https://github.com/CMLandOcean/bio_optics
#
# [1] Albert & Mobley (2003): An analytical model for subsurface irradiance and remote sensing reflectance in deep and shallow case-2 waters. [10.1364/OE.11.002873]
# [2] Gege (2012): Analytic model for the direct and diffuse components of downwelling spectral irradiance in water. [10.1364/AO.51.001407]
# [3] Gege (2004): The water color simulator WASI: an integrating software tool for analysis and simulation of optical in situ spectra. [10.1016/j.cageo.2004.03.005]
# [4] Gege (2014): WASI-2D: A software tool for regionally optimized analysis of imaging spectrometer data from deep and shallow waters. [10.1016/j.cageo.2013.07.022]
# [5] Gege (2021): The Water Colour Simulator WASI. User manual for WASI version 6. 
# [6] Gege & Albert (2006): A Tool for Inverse Modeling of Spectral Measurements in Deep and Shallow Waters. [10.1007/1-4020-3968-9_4]


import numpy as np
from lmfit import minimize, Parameters
from ..water import absorption, backscattering, attenuation, bottom_reflectance
from ..surface import surface, air_water, threeC
from ..helper import resampling, utils


def r_rs_dp(u, 
            theta_sun=np.radians(30), 
            theta_view=np.radians(0), 
            n1=1, 
            n2=1.33):
    """
    Subsurface radiance reflectance of optically deep water after Albert & Mobley (2003) [1].
    
    [1] Albert & Mobley (2003): An analytical model for subsurface irradiance and remote sensing reflectance in deep and shallow case-2 waters. [10.1364/OE.11.002873]

    :param u: ratio of backscattering coefficient to the sum of absorption and backscattering coefficients
    :param theta_sun: sun zenith angle in air [radians], is converted to in water using Snell's law, default: np.radians(30)
    :param theta_view: viewing angle in air in units [radians], is converted to in water using Snell's law, np.radians(0)
    :param n1: refrective index of origin medium, default: 1 for air
    :param n2: refrective index of destination medium, default: 1.33 for water
    :return: subsurface radiance reflectance of deep water [sr-1]
    """    
    f_rs = 0.0512 * (1 + 4.6659 * u - 7.8387 * u**2 + 5.4571 * u**3) * (1 + 0.1098/np.cos(air_water.snell(theta_sun, n1, n2))) * (1 + 0.4021/np.cos(air_water.snell(theta_view, n1, n2)))
    r_rs_dp = f_rs * u
    
    return r_rs_dp


def R_dp(u, 
         theta_sun=np.radians(30), 
         n1=1, 
         n2=1.33):
    """
    Subsurface irradiance reflectance of optically deep water after Albert & Mobley (2003) [1.]

    [1] Albert & Mobley (2003): An analytical model for subsurface irradiance and remote sensing reflectance in deep and shallow case-2 waters. [10.1364/OE.11.002873]
    
    :param u: ratio of backscattering coefficient to the sum of absorption and backscattering coefficients
    :param theta_sun: sun zenith angle in air [radians], is converted to in water using Snell's law, default: np.radians(30)
    :param n1: refrective index of origin medium, default: 1 for air
    :param n2: refrective index of destination medium, default: 1.33 for water
    :return: subsurface irradiance reflectance of deep water [sr-1]
    """
    f = 0.1034 * (1 + 3.3586 * u + -6.5358 * u**2 + 4.6638 * u**3) * (1 + 2.4121 / np.cos(air_water.snell(theta_sun, n1, n2)))
    R_dp = f * u

    return R_dp


def R_rs(C_0 = 0,
         C_1 = 0,
         C_2 = 0,
         C_3 = 0,
         C_4 = 0,
         C_5 = 0,
         C_Y: float = 0,
         C_X: float = 0,
         C_Mie: float = 0,
         b_bphy_spec: float = 0.0010,
         b_bMie_spec: float = 0.0042,
         b_bX_spec: float = 0.0086,
         b_bX_norm_factor: float = 1,
         a_NAP_spec_lambda_0: float = 0.041,
         S: float = 0.014,
         K: float = 0.0,
         S_NAP: float = 0.011,
         n: float = -1,
         lambda_0: float = 440,
         lambda_S: float = 500,
         theta_sun = np.radians(30),
         theta_view = np.radians(0),
         n1 = 1,
         n2 = 1.33,
         fresh: bool = False,
         T_W=20,
         T_W_0=20,
         wavelengths: np.array = np.arange(400,800),
         a_i_spec_res=[],
         a_w_res=[],
         a_Y_N_res = [],
         a_NAP_N_res = [],
         b_phy_norm_res = [],
         b_bw_res = [],
         b_X_norm_res=[],
         b_Mie_norm_res=[],
         da_W_div_dT_res=[]
         ):
    """
    Subsurface radiance reflectance of optically shallow water after Albert & Mobley (2003) [1].
    
    [1] Albert & Mobley (2003): An analytical model for subsurface irradiance and remote sensing reflectance in deep and shallow case-2 waters. [10.1364/OE.11.002873]
    [2] Heege, T. (2000): Flugzeuggestützte Fernerkundung von Wasserinhaltsstoffen am Bodensee. PhD thesis. DLR-Forschungsbericht 2000-40, 134 p.
    [3] Albert, A., & Mobley, C. (2003): An analytical model for subsurface irradiance and remote sensing reflectance in deep and shallow case-2 waters [doi.org/10.1364/OE.11.002873]
    
    :param C_0: concentration of phytoplankton type 0 [ug L-1], default: 0
    :param C_1: concentration of phytoplankton type 1 [ug L-1], default: 0
    :param C_2: concentration of phytoplankton type 2 [ug L-1], default: 0
    :param C_3: concentration of phytoplankton type 3 [ug L-1], default: 0
    :param C_4: concentration of phytoplankton type 4 [ug L-1], default: 0
    :param C_5: concentration of phytoplankton type 5 [ug L-1], default: 0
    :param C_Y: CDOM absorption coefficient at lambda_0 [m-1], default: 0
    :param C_X: concentration of non-algal particles type I [mg L-1], default: 0
    :param C_Mie: concentration of non-algal particles type II [mg L-1], default: 0
    :param b_bphy_spec:  specific backscattering coefficient of phytoplankton at 550 nm in [m2 mg-1], default: 0.0010
    :param b_bMie_spec: specific backscattering coefficient of non-algal particles type II [m2 g-1] , default: 0.0042
    :param b_bX_spec: specific backscattering coefficient of non-algal particles type I [m2 g-1], default: 0.0086 [2]
    :param b_bX_norm_factor: normalized scattering coefficient with arbitrary wavelength dependency, default: 1
    :param a_NAP_spec_lambda_0: specific absorption coefficient of NAP at reference wavelength lambda_0 [m2 g-1], default: 0.041
    :param S: spectral slope of CDOM absorption spectrum [nm-1], default: 0.014
    :param K: constant added to the CDOM exponential function [m-1], default: 0
    :param S_NAP: spectral slope of NAP absorption spectrum, default [nm-1]: 0.011
    :param n: Angström exponent of particle type II backscattering, default: -1
    :param lambda_0: reference wavelength for CDOM and NAP absorption [nm], default: 440 nm
    :param lambda_S: reference wavelength for scatteromg of particles type II [nm] , default: 500 nm
    :param theta_sun: sun zenith angle [radians], default: np.radians(30)
    :param theta_view: viewing angle [radians], default: np.radians(0) (nadir) 
    :param n1: refractive index of origin medium, default: 1 for air
    :param n2: refractive index of destination medium, default: 1.33 for water
    :param fresh: boolean to decide if to compute b_bw for fresh or oceanic water, default: False
    :param T_W: actual water temperature [degrees C], default: 20
    :param T_W_0: reference temperature of pure water absorption [degrees C], default: 20
    :wavelengths: wavelengths to compute r_rs_sh for [nm], default: np.arange(400,800) 
    :param a_i_spec_res: optional, specific absorption coefficients of phytoplankton types resampled to sensor's band settings. Will be computed within function if not provided.
    :param a_w_res: optional, absorption of pure water resampled to sensor's band settings. Will be computed within function if not provided.
    :param a_Y_N_res: optional, normalized absorption coefficients of CDOM resampled to sensor's band settings. Will be computed within function if not provided.
    :param a_NAP_N_res: optional, normalized absorption coefficients of NAP resampled to sensor's band settings. Will be computed within function if not provided.
    :param b_phy_norm_res: optional, preresampling b_phy_norm saves a lot of time during inversion. Will be computed within function if not provided.
    :param b_bw_res: optional, precomputing b_bw b_bw saves a lot of time during inversion. Will be computed within function if not provided.
    :param b_X_norm_res: optional, precomputing b_bX_norm before inversion saves a lot of time. Will be computed within function if not provided.
    :param b_Mie_norm_res: optional, if n and lambda_S are not fit params, the last part of the equation can be precomputed to save time. Will be computed within function if not provided.
    :param da_W_div_dT_res: optional, temperature gradient of pure water absorption resampled  to sensor's band settings. Will be computed within function if not provided.
    :return: Remote sensing reflectance of shallow water [sr-1]
    """
    # Backscattering and absorption coefficients of the water body depending on the concentration of optically active water constituents
    bs = backscattering.b_b(C_X=C_X,
                            C_Mie=C_Mie,
                            C_phy=0, # np.sum([C_0,C_1,C_2,C_3,C_4,C_5]), # PG implementation does not consider b_bphy
                            b_bphy_spec=b_bphy_spec,
                            wavelengths=wavelengths,
                            b_bMie_spec=b_bMie_spec,
                            lambda_S=lambda_S,
                            n=n,
                            b_bX_spec=b_bX_spec,
                            b_bX_norm_factor=b_bX_norm_factor,
                            fresh=fresh,
                            b_phy_norm_res = b_phy_norm_res,
                            b_bw_res = b_bw_res,
                            b_Mie_norm_res = b_Mie_norm_res,
                            b_X_norm_res = b_X_norm_res)

    ab = absorption.a(C_0,
                      C_1,
                      C_2,
                      C_3,
                      C_4,
                      C_5,
                      C_Y,
                      C_X,
                      C_Mie,
                      wavelengths=wavelengths,
                      S=S,
                      K=K,
                      a_NAP_spec_lambda_0=a_NAP_spec_lambda_0,
                      S_NAP=S_NAP,
                      lambda_0=lambda_0,
                      T_W=T_W,
                      T_W_0=T_W_0,
                      a_i_spec_res = a_i_spec_res,
                      a_w_res = a_w_res,
                      a_Y_N_res = a_Y_N_res,
                      a_NAP_N_res = a_NAP_N_res,
                      da_W_div_dT_res=da_W_div_dT_res,
                      )

    u = bs / (ab + bs)

    Rrs = 0.518 * r_rs_dp(u, theta_sun=theta_sun, theta_view=theta_view, n1=n1, n2=n2) / (1 - 0.48 * R_dp(u, theta_sun=theta_sun, n1=n1, n2=n2))

    return Rrs
    

#################
### Inversion
#################


def forward(params,
            wavelengths,
            Ls_Ed,
            a_i_spec_res=[],
            a_w_res=[],
            a_Y_N_res = [],
            a_NAP_N_res = [],
            b_phy_norm_res = [],
            b_bw_res = [],
            b_X_norm_res=[],
            b_Mie_norm_res=[],
            da_W_div_dT_res=[]):
    """
    Forward simulation of a shallow water remote sensing reflectance spectrum based on the provided parameterization.
    
    :param params: lmfit Parameters object containing all Parameter objects that are required to specify the model
    :param wavelengths: wavelengths of R_rs bands [nm]
    :param Ls_Ed: Sky reflectance spectrum [sr-1]
    :param a_i_spec_res: optional, specific absorption coefficients of phytoplankton types resampled to sensor's band settings. Will be computed within function if not provided.
    :param a_w_res: optional, absorption of pure water resampled to sensor's band settings. Will be computed within function if not provided.
    :param a_Y_N_res: optional, normalized absorption coefficients of CDOM resampled to sensor's band settings. Will be computed within function if not provided.
    :param a_NAP_N_res: optional, normalized absorption coefficients of NAP resampled to sensor's band settings. Will be computed within function if not provided.
    :param b_phy_norm_res: optional, preresampling b_phy_norm saves a lot of time. Will be computed within function if not provided.
    :param b_bw_res: optional, precomputing b_bw b_bw saves a lot of time . Will be computed within function if not provided.
    :param b_X_norm_res: optional, precomputing b_bX_norm saves a lot of time. Will be computed within function if not provided.
    :param b_Mie_norm_res: optional, if n and lambda_S are not fit params, the last part of the equation can be precomputed to save time. Will be computed within function if not provided.
    :param da_W_div_dT_res: optional, preresampling da_W_div_dT saves a lot of time. Will be computed within function if not provided.
    :return: R_rs: simulated remote sensing reflectance spectrum [sr-1]
    """
    
    if params['fit_surface']==True:
        
        R_rs_sim = R_rs(C_0 = params['C_0'],
                        C_1 = params['C_1'],
                        C_2 = params['C_2'],
                        C_3 = params['C_3'],
                        C_4 = params['C_4'],
                        C_5 = params['C_5'],
                        C_Y = params['C_Y'],
                        C_X = params['C_X'],
                        C_Mie = params['C_Mie'],
                        b_bphy_spec = params['b_bphy_spec'],
                        b_bMie_spec = params['b_bMie_spec'],
                        b_bX_spec = params['b_bX_spec'],
                        b_bX_norm_factor = params['b_bX_norm_factor'],
                        a_NAP_spec_lambda_0 = params['a_NAP_spec_lambda_0'],
                        S = params['S'],
                        K = params['K'],
                        S_NAP = params['S_NAP'],
                        n = params['n'],
                        lambda_0 = params['lambda_0'],
                        lambda_S = params['lambda_S'],
                        theta_sun = params['theta_sun'],
                        theta_view = params['theta_view'],
                        n1 = params['n1'],
                        n2 = params['n2'],
                        fresh = params["fresh"],
                        T_W = params["T_W"],
                        T_W_0 = params["T_W_0"],
                        wavelengths = wavelengths,
                        a_i_spec_res=a_i_spec_res,
                        a_w_res=a_w_res,
                        a_Y_N_res = a_Y_N_res,
                        a_NAP_N_res = a_NAP_N_res,
                        b_phy_norm_res=b_phy_norm_res,
                        b_bw_res=b_bw_res,
                        b_X_norm_res=b_X_norm_res,
                        b_Mie_norm_res=b_Mie_norm_res,
                        da_W_div_dT_res=da_W_div_dT_res) + \
                            air_water.fresnel(params['theta_view'],
                                              n1 = params['n1'],
                                              n2 = params['n2']) * Ls_Ed + \
                            threeC.delta(wavelengths = wavelengths,
                                        rho_dd = params['rho_dd'], 
                                        rho_ds = params['rho_ds'], 
                                        delta = params['delta'],
                                        theta_sun=params['theta_sun'], 
                                        P=params['P'], 
                                        AM=params['AM'], 
                                        RH=params['RH'], 
                                        alpha=params['alpha'],
                                        beta=params['beta']) + \
                            params['offset']
                            
    elif params['fit_surface']==False:

        R_rs_sim = R_rs(C_0 = params['C_0'],
                        C_1 = params['C_1'],
                        C_2 = params['C_2'],
                        C_3 = params['C_3'],
                        C_4 = params['C_4'],
                        C_5 = params['C_5'],
                        C_Y = params['C_Y'],
                        C_X = params['C_X'],
                        C_Mie = params['C_Mie'],
                        b_bphy_spec = params['b_bphy_spec'],
                        b_bMie_spec = params['b_bMie_spec'],
                        b_bX_spec = params['b_bX_spec'],
                        b_bX_norm_factor = params['b_bX_norm_factor'],
                        a_NAP_spec_lambda_0 = params['a_NAP_spec_lambda_0'],
                        S = params['S'],
                        K = params['K'],
                        S_NAP = params['S_NAP'],
                        n = params['n'],
                        lambda_0 = params['lambda_0'],
                        lambda_S = params['lambda_S'],
                        theta_sun = params['theta_sun'],
                        theta_view = params['theta_view'],
                        n1 = params['n1'],
                        n2 = params['n2'],
                        fresh = params["fresh"],
                        T_W = params["T_W"],
                        T_W_0 = params["T_W_0"],
                        wavelengths = wavelengths,
                        a_i_spec_res=a_i_spec_res,
                        a_w_res=a_w_res,
                        a_Y_N_res = a_Y_N_res,
                        a_NAP_N_res = a_NAP_N_res,
                        b_phy_norm_res=b_phy_norm_res,
                        b_bw_res=b_bw_res,
                        b_X_norm_res=b_X_norm_res,
                        b_Mie_norm_res=b_Mie_norm_res,
                        da_W_div_dT_res=da_W_div_dT_res) + \
                    params['offset']
                            
    return R_rs_sim


def func2opt(params, 
             R_rs,
             wavelengths, 
             Ls_Ed,
             weights = [],
             a_i_spec_res=[],
             a_w_res=[],
             a_Y_N_res = [],
             a_NAP_N_res = [],
             b_phy_norm_res = [],
             b_bw_res = [],
             b_X_norm_res=[],
             b_Mie_norm_res=[],
             da_W_div_dT_res=[]):
    """
    Error function around model to be minimized by changing fit parameters.
    
    :param params: lmfit Parameters object containing all Parameter objects that are required to specify the model
    :param R_rs: Remote sensing reflectance spectrum [sr-1]
    :param wavelengths: wavelengths of R_rs bands [nm]
    :param weights: spectral weighing coefficients
    :param a_i_spec_res: optional, specific absorption coefficients of phytoplankton types resampled to sensor's band settings. Will be computed within function if not provided.
    :param a_w_res: optional, absorption of pure water resampled to sensor's band settings. Will be computed within function if not provided.
    :param a_Y_N_res: optional, normalized absorption coefficients of CDOM resampled to sensor's band settings. Will be computed within function if not provided.
    :param a_NAP_N_res: optional, normalized absorption coefficients of NAP resampled to sensor's band settings. Will be computed within function if not provided.
    :param b_phy_norm_res: optional, preresampling b_phy_norm saves a lot of time during inversion. Will be computed within function if not provided.
    :param b_bw_res: optional, precomputing b_bw b_bw saves a lot of time during inversion. Will be computed within function if not provided.
    :param b_X_norm_res: optional, precomputing b_bX_norm before inversion saves a lot of time. Will be computed within function if not provided.
    :param b_Mie_norm_res: optional, if n and lambda_S are not fit params, the last part of the equation can be precomputed to save time. Will be computed within function if not provided.
    :param R_i_b_res: optional, preresampling R_i_b before inversion saves a lot of time. Will be computed within function if not provided.
    :param da_W_div_dT_res: optional, temperature gradient of pure water absorption resampled  to sensor's band settings. Will be computed within function if not provided.
    :param E_0_res: optional, precomputing E_0 saves a lot of time. Will be computed within function if not provided.
    :param a_oz_res: optional, precomputing a_oz saves a lot of time. Will be computed within function if not provided.
    :param a_ox_res: optional, precomputing a_ox saves a lot of time. Will be computed within function if not provided.
    :param a_wv_res: optional, precomputing a_wv saves a lot of time. Will be computed within function if not provided.
    :param E_dd_res: optional, preresampling E_dd before inversion saves a lot of time. Will be computed within function if not provided.
    :param E_dsa_res: optional, preresampling E_dsa before inversion saves a lot of time. Will be computed within function if not provided.
    :param E_dsr_res: optional, preresampling E_dsr before inversion saves a lot of time. Will be computed within function if not provided.
    :param E_d_res: optional, preresampling E_d before inversion saves a lot of time. Will be computed within function if not provided.
    :return: weighted difference between measured and simulated R_rs
    """
    
    if len(weights)==0:
        weights = np.ones(len(wavelengths))

    R_rs_sim = forward(params=params,
                       wavelengths=wavelengths,
                       Ls_Ed = Ls_Ed,
                       a_i_spec_res=a_i_spec_res,
                       a_w_res=a_w_res,
                       a_Y_N_res = a_Y_N_res,
                       a_NAP_N_res = a_NAP_N_res,
                       b_phy_norm_res=b_phy_norm_res,
                       b_bw_res=b_bw_res,
                       b_X_norm_res=b_X_norm_res,
                       b_Mie_norm_res=b_Mie_norm_res,
                       da_W_div_dT_res=da_W_div_dT_res)
           
    return np.sum((R_rs_sim - R_rs)**2 * weights) # utils.compute_residual(R_rs, R_rs_sim, method=params['error_method'], weights=weights)


def invert(params, 
           R_rs,
           wavelengths,
           Ls_Ed, 
           weights=[],
           a_i_spec_res=[],
           a_w_res=[],
           a_Y_N_res = [],
           a_NAP_N_res = [],
           b_phy_norm_res = [],
           b_bw_res = [],
           b_X_norm_res=[],
           b_Mie_norm_res=[],
           da_W_div_dT_res=[],
           method="least-squares", 
           max_nfev=15000
           ):
    """
    Function to inversely fit a modeled spectrum to a measurement spectrum.
    
    :param params: lmfit Parameters object containing all Parameter objects that are required to specify the model
    :param R_rs: Remote sensing reflectance spectrum [sr-1]
    :param Ls_Ed: Sky reflectance spectrum [sr-1]
    :param wavelengths: wavelengths of R_rs bands [nm]
    :param weights: spectral weighing coefficients
    :param a_i_spec_res: optional, specific absorption coefficients of phytoplankton types resampled to sensor's band settings. Will be computed within function if not provided.
    :param a_w_res: optional, absorption of pure water resampled to sensor's band settings. Will be computed within function if not provided.
    :param a_Y_N_res: optional, normalized absorption coefficients of CDOM resampled to sensor's band settings. Will be computed within function if not provided.
    :param a_NAP_N_res: optional, normalized absorption coefficients of NAP resampled to sensor's band settings. Will be computed within function if not provided.
    :param b_phy_norm_res: optional, preresampling b_phy_norm saves a lot of time during inversion. Will be computed within function if not provided.
    :param b_bw_res: optional, precomputing b_bw b_bw saves a lot of time during inversion. Will be computed within function if not provided.
    :param b_X_norm_res: optional, precomputing b_bX_norm before inversion saves a lot of time. Will be computed within function if not provided.
    :param b_Mie_norm_res: optional, if n and lambda_S are not fit params, the last part of the equation can be precomputed to save time. Will be computed within function if not provided.
    :param da_W_div_dT_res: optional, temperature gradient of pure water absorption resampled  to sensor's band settings. Will be computed within function if not provided.
    :param E_0_res: optional, precomputing E_0 saves a lot of time. Will be computed within function if not provided.
    :param a_oz_res: optional, precomputing a_oz saves a lot of time. Will be computed within function if not provided.
    :param a_ox_res: optional, precomputing a_ox saves a lot of time. Will be computed within function if not provided.
    :param a_wv_res: optional, precomputing a_wv saves a lot of time. Will be computed within function if not provided.
    :param E_dd_res: optional, preresampling E_dd before inversion saves a lot of time. Will be computed within function if not provided.
    :param E_dsa_res: optional, preresampling E_dsa before inversion saves a lot of time. Will be computed within function if not provided.
    :param E_dsr_res: optional, preresampling E_dsr before inversion saves a lot of time. Will be computed within function if not provided.
    :param E_d_res: optional, preresampling E_d before inversion saves a lot of time. Will be computed within function if not provided.
    :param method: name of the fitting method to use by lmfit, default: 'least-squares'
    :param max_nfev: maximum number of function evaluations, default: 400
    :return: object containing the optimized parameters and several goodness-of-fit statistics.
    """    

    if len(weights)==0:
        weights = np.ones(len(R_rs))
    
    if params['fit_surface']==True:
        res = minimize(func2opt, 
                       params, 
                       args=(R_rs, 
                             wavelengths, 
                             Ls_Ed,
                             weights, 
                             a_i_spec_res, 
                             a_w_res, 
                             a_Y_N_res, 
                             a_NAP_N_res, 
                             b_phy_norm_res, 
                             b_bw_res, 
                             b_X_norm_res, 
                             b_Mie_norm_res, 
                             da_W_div_dT_res), 
                       method=method, 
                       max_nfev=max_nfev) 
                       
    elif params['fit_surface']==False:

        params.add('P', vary=False) 
        params.add('AM', vary=False) 
        params.add('RH', vary=False) 
        params.add('H_oz', vary=False)
        params.add('WV', vary=False) 
        params.add('alpha', vary=False) 
        params.add('beta', vary=False) 
        params.add('g_dd', vary=False) 
        params.add('g_dsr', vary=False) 
        params.add('g_dsa', vary=False) 
        params.add('f_dd', vary=False) 
        params.add('f_ds', vary=False) 
        params.add('rho_L', vary=False) 

        Ls_Ed = np.zeros(len(R_rs))

        res = minimize(func2opt, 
                       params, 
                       args=(R_rs, 
                            Ls_Ed,
                             wavelengths, 
                             weights, 
                             a_i_spec_res, 
                             a_w_res, 
                             a_Y_N_res, 
                             a_NAP_N_res, 
                             b_phy_norm_res, 
                             b_bw_res, 
                             b_X_norm_res, 
                             b_Mie_norm_res, 
                             da_W_div_dT_res), 
                       method=method, 
                       options={'gtol': 1e-16, 'eps': 1e-07, 'maxiter': 15000, 'ftol': 1e-16, 'maxls': 20, 'maxcor': 20})
                       # max_nfev=max_nfev) 
    return res





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
from ..water import absorption, backscattering, temperature_gradient, attenuation, bottom_reflectance
from ..surface import surface, air_water, threeC
from ..helper import resampling, utils


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
         rho_dd = 0.1, 
        rho_ds = 0.1, 
        delta = 0.0,
        P=1013.25, 
        AM=1, 
        RH=60, 
        alpha=1.317,
        beta=0.2606,
         wavelengths: np.array = np.arange(400,800),
         a_i_spec_res=[],
         a_w_res=[],
         a_Y_N_res = [],
         a_NAP_N_res = [],
         b_phy_norm_res = [],
         b_bw_res = [],
         b_X_norm_res=[],
         b_Mie_norm_res=[],
         da_W_div_dT_res=[],
         Ls_Ed=[]
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

    ###### PG ######

    wl = wavelengths
        
    wl_a = 550

    theta_sun_ = theta_sun * np.pi / 180.

    z3 = -0.1417 * alpha + 0.82
    z2 = 0.65 if alpha > 1.2 else z3
    z1 = 0.82 if alpha < 0 else z2
    
    theta_sun_mean = z1

    B3 = np.log(1 - theta_sun_mean)
    B2 = B3 * (0.0783 + B3 * (-0.3824 - 0.5874 * B3))
    B1 = B3 * (1.459 + B3 * (0.1595 + 0.4129 * B3))
    Fa = 1 - 0.5 * np.exp((B1 + B2 * np.cos(theta_sun_)) * np.cos(theta_sun_))

    omega_a = (-0.0032 * AM + 0.972) * np.exp(3.06 * 1e-4 * RH)
    tau_a = beta*(wl/wl_a)**(-alpha)

    # fixed a bug in M, thanks Jaime! [brackets added]
    M  = 1 / (np.cos(theta_sun_) + 0.50572 * (90 + 6.07995 - theta_sun)**(-1.6364)) 
    M_ = M * P / 1013.25

    Tr = np.exp(- M_ / (115.6406 * (wl / 1000)**4 - 1.335 * (wl / 1000)**2)) 

    Tas = np.exp(- omega_a * tau_a * M)

    Edd = Tr * Tas    
    Edsr = 0.5 * (1 - Tr**0.95)
    Edsa = Tr**1.5 * (1 - Tas) * Fa
    
    Ed = Edd + Edsr + Edsa
    Edd_Ed = Edd / Ed
    Edsr_Ed = Edsr / Ed
    Edsa_Ed = Edsa / Ed
    Eds_Ed = Edsr_Ed + Edsa_Ed

    # calc_a_ph
    a_ph = C_0 * a_i_spec_res.T[0]

    # calc_a_y
    wl_ref_y = 440
    a_y = C_Y * np.exp(- S * (wl - wl_ref_y))

    # calc_a
    T_w_ref = 20.
    a_w_corr = a_w_res + (T_W - T_w_ref) * da_W_div_dT_res
    
    a = a_w_corr + a_ph + a_y

    # calc_bb_sm
    bbstar_sm = 0.0086
    bbstar_mie = 0.0042
    wl_ref_mie = 500
        
    bb_sm = C_X * bbstar_sm + C_Mie * bbstar_mie * (wl / wl_ref_mie)**n

    # calc_bb
    b1 = 0.00144 if n2==1.34 else 0.00111
        
    wl_ref_water = 500
    S_water = -4.32

    bb_water = b1 * (wl / wl_ref_water)**S_water
    bb = bb_water + bb_sm

    # calc omega_b
    omega_b = bb / (bb + a)

    # calc sun and viewing zenith angles under water
    theta_sun_ = theta_sun * np.pi / 180.
    theta_sun_ss = np.arcsin(np.sin(theta_sun_) / n2)
    theta_view_ = theta_view * np.pi / 180.
    theta_view_ss = np.arcsin(np.sin(theta_view_) / n2)

    p_f = [0.1034, 1, 3.3586, -6.5358, 4.6638, 2.4121]
    p_frs = [0.0512, 1, 4.6659, -7.8387, 5.4571, 0.1098, 0.4021]

    # calc subsurface reflectance            
    f = p_f[0] * (p_f[1] + p_f[2] * omega_b + p_f[3] * omega_b**2 + p_f[4] * omega_b**3) * (1 + p_f[5] / np.cos(theta_sun_ss)) 

    R0minus = f * omega_b

    # calc subsurface remote sensing reflectance       
    frs = p_frs[0] * (p_frs[1] + p_frs[2] * omega_b + p_frs[3] * omega_b**2 + p_frs[4] * omega_b**3) * (1 + p_frs[5] / np.cos(theta_sun_ss)) * (1 + p_frs[6] / np.cos(theta_view_ss))

    Rrs0minus = frs * omega_b

    # calc water surface reflected reflectance        
    Rrs_refl = air_water.fresnel(theta_view,n1,n2) * Ls_Ed + rho_dd * Edd_Ed / np.pi + rho_ds * Eds_Ed / np.pi + delta

    # calc_Rrs0plus (Lee1998, eq22), R=Q*Rrs
    gamma = 0.48
    zeta = 0.518

    Rrs = zeta * Rrs0minus / ( 1 - gamma * R0minus )
    
    Lu_Ed = Rrs + Rrs_refl

    return Rrs, Rrs_refl
    

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
                        da_W_div_dT_res=da_W_div_dT_res,
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
           
    return np.sum((R_rs_sim[0] - R_rs)**2 * weights) # utils.compute_residual(R_rs, R_rs_sim, method=params['error_method'], weights=weights)


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
                       options={'gtol': 1e-16, 'eps': 1e-07, 'maxiter': max_nfev, 'ftol': 1e-16, 'maxls': 20, 'maxcor': 20})
                        
    return res





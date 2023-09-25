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
#  Marcel KÃ¶nig, mkoenig3 AT asu.edu 
#
# bio_optics
#  A former version of this code was provided to Planet, PBC as part of the CarbonMapper Land and Ocean Program.
#  It builds on the extensive work of many researchers. For example, models were developed  
#  by Albert & Mobley [1] and Gege [2]; the methodology was mainly developed 
#  by Gege [3,4,5] and Albert & Gege [6].
#
#  Please give proper attribution when using this code for publication:
#
#  KÃ¶nig, M., Hondula. K.L., Jamalinia, E., Dai, J., Vaughn, N.R., Asner, G.P. (2023): bio_optics python package (Version x) [Software]. Available from https://github.com/CMLandOcean/bio_optics
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
from ..water import albert_mobley as water_alg
from .. atmosphere import sky_radiance, downwelling_irradiance
from ..surface import surface, air_water
from ..helper import resampling, utils


def invert(params, 
           R_rs, 
           wavelengths,
           weights=[],
           a_i_spec_res=[],
           a_w_res=[],
           a_Y_N_res = [],
           a_NAP_N_res = [],
           b_phy_norm_res = [],
           b_bw_res = [],
           b_X_norm_res=[],
           b_Mie_norm_res=[],
           R_i_b_res = [],
           da_W_div_dT_res=[],
           E_0_res=[],
           a_oz_res=[],
           a_ox_res=[],
           a_wv_res=[],
           E_dd_res=[],
           E_dsa_res=[],
           E_dsr_res=[],
           E_d_res=[],
           E_ds_res=[],
           method="least-squares", 
           max_nfev=400
           ):
    """
    Function to inversely fit a modeled spectrum to a measurement spectrum.
    
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
    :param method: name of the fitting method to use by lmfit, default: 'least-squares'
    :param max_nfev: maximum number of function evaluations, default: 400
    :return: object containing the optimized parameters and several goodness-of-fit statistics.
    """    

    if len(weights)==0:
        weights = np.ones(len(R_rs))
    
    if params['fit_surface'].value:
        res = minimize(func2opt, 
                       params, 
                       args=(R_rs, 
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
                             R_i_b_res, 
                             da_W_div_dT_res,
                             E_0_res, 
                             a_oz_res, 
                             a_ox_res, 
                             a_wv_res,
                             E_dd_res,
                             E_dsa_res,
                             E_dsr_res,
                             E_d_res,
                             E_ds_res), 
                       method=method, 
                       max_nfev=max_nfev) 
                       
    elif not params['fit_surface'].value:

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

        res = minimize(func2opt, 
                       params, 
                       args=(R_rs, 
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
                             R_i_b_res, 
                             da_W_div_dT_res), 
                       method=method, 
                       max_nfev=max_nfev) 
    return res


def forward(parameters,
        wavelengths,
        a_w_res=[],
        da_W_div_dT_res=[],
        a_i_spec_res=[],
        a_Y_N_res = [],
        a_NAP_N_res = [],
        b_phy_norm_res = [],
        b_bw_res = [],
        b_X_norm_res=[],
        b_Mie_norm_res=[],
        R_i_b_res = [],
        E_0_res=[],
        a_oz_res=[],
        a_ox_res=[],
        a_wv_res=[],
        E_dd_res=[],
        E_dsa_res=[],
        E_dsr_res=[],
        E_d_res=[],
        E_ds_res=[]):
    """
    Forward simulation of a shallow water remote sensing reflectance spectrum based on the provided parameterization.
    """
    ctsp = np.cos(air_water.snell(parameters["theta_sun"],  n1=parameters["n1"], n2=parameters["n2"]))  #cos of theta_sun_prime. theta_sun_prime = snell(theta_sun, n1, n2)
    ctvp = np.cos(air_water.snell(parameters["theta_view"], n1=parameters["n1"], n2=parameters["n2"]))

    a_sim = absorption.a(C_0=parameters["C_0"], C_1=parameters["C_1"], C_2=parameters["C_2"], C_3=parameters["C_3"], C_4=parameters["C_4"], C_5=parameters["C_5"], 
                        C_Y=parameters["C_Y"], C_X=parameters["C_X"], C_Mie=parameters["C_Mie"], S=parameters["S"], 
                        S_NAP=parameters["S_NAP"], 
                        a_NAP_spec_lambda_0=parameters["a_NAP_spec_lambda_0"],
                        lambda_0=parameters["lambda_0"],
                        K=parameters["K"],
                        wavelengths=wavelengths,
                        T_W=parameters["T_W"],
                        T_W_0=parameters["T_W_0"],
                        a_w_res=a_w_res,
                        da_W_div_dT_res=da_W_div_dT_res, 
                        a_i_spec_res=a_i_spec_res, 
                        a_Y_N_res=a_Y_N_res,
                        a_NAP_N_res=a_NAP_N_res)
    
    b_b_sim = backscattering.b_b(C_X=parameters["C_X"], C_Mie=parameters["C_Mie"], C_phy=np.sum([parameters["C_0"], parameters["C_1"], parameters["C_2"], parameters["C_3"], parameters["C_4"], parameters["C_5"]]), wavelengths=wavelengths, 
                        fresh=parameters["fresh"],
                        b_bphy_spec=parameters["b_bphy_spec"],
                        b_bMie_spec=parameters["b_bMie_spec"],
                        b_bX_spec=parameters["b_bX_spec"],
                        b_bX_norm_factor=parameters["b_bX_norm_factor"],
                        lambda_S=parameters["lambda_S"],
                        n=parameters["n"],
                        b_bw_res=b_bw_res, 
                        b_phy_norm_res=b_phy_norm_res, 
                        b_X_norm_res=b_X_norm_res, 
                        b_Mie_norm_res=b_Mie_norm_res)

    Rrsb = bottom_reflectance.R_rs_b(parameters["f_0"], parameters["f_1"], parameters["f_2"], parameters["f_3"], parameters["f_4"], parameters["f_5"], B_0=parameters["B_0"], B_1=parameters["B_1"], B_2=parameters["B_2"], B_3=parameters["B_3"], B_4=parameters["B_4"], B_5=parameters["B_5"], wavelengths=wavelengths, R_i_b_res=R_i_b_res)

    ob = attenuation.omega_b(a_sim, b_b_sim) #ob is omega_b. Shortened to distinguish between new var and function params.

    frs = water_alg.f_rs(omega_b=ob, cos_t_sun_p=ctsp, cos_t_view_p=ctvp)

    rrsd = water_alg.r_rs_deep(f_rs=frs, omega_b=ob)

    Kd =  attenuation.K_d(a=a_sim, b_b=b_b_sim, cos_t_sun_p=ctsp, kappa_0=parameters["kappa_0"])

    kuW = attenuation.k_uW(a=a_sim, b_b=b_b_sim, omega_b=ob, cos_t_sun_p=ctsp, cos_t_view_p=ctvp)

    kuB = attenuation.k_uB(a=a_sim, b_b=b_b_sim, omega_b=ob, cos_t_sun_p=ctsp, cos_t_view_p=ctvp)

    R_rs_water = air_water.below2above(water_alg.r_rs_shallow(r_rs_deep=rrsd, K_d=Kd, k_uW=kuW, zB=parameters["zB"], R_rs_b=Rrsb, k_uB=kuB)) # zeta & gamma

    if parameters["fit_surface"].value:        
        if len(E_dd_res) == 0:
            E_dd  = downwelling_irradiance.E_dd(wavelengths, parameters["theta_sun"], parameters["P"], parameters["AM"], parameters["RH"], parameters["H_oz"], parameters["WV"], parameters["alpha"], parameters["beta"], E_0_res, a_oz_res, a_ox_res, a_wv_res, E_dd_res)
        else:
            E_dd = E_dd_res

        if len(E_dsa_res) == 0:
            E_dsa = downwelling_irradiance.E_dsa(wavelengths, parameters["theta_sun"], parameters["P"], parameters["AM"], parameters["RH"], parameters["H_oz"], parameters["WV"], parameters["alpha"], parameters["beta"], E_0_res, a_oz_res, a_ox_res, a_wv_res, E_dsa_res)
        else:
            E_dsa = E_dsa_res

        if len(E_dsr_res) == 0:
            E_dsr = downwelling_irradiance.E_dsr(wavelengths, parameters["theta_sun"], parameters["P"], parameters["AM"], parameters["RH"], parameters["H_oz"], parameters["WV"], E_0_res, a_oz_res, a_ox_res, a_wv_res, E_dsr_res)
        else:
            E_dsr = E_dsr_res

        if len(E_ds_res) == 0:
            E_ds = downwelling_irradiance.E_ds(E_dsr, E_dsa)
        else:
            E_ds = E_ds_res

        if len(E_d_res) == 0:
            E_d = downwelling_irradiance.E_d(E_dd, E_ds, parameters["f_dd"], parameters["f_ds"])
        else:
            E_d = E_d_res

        L_s = sky_radiance.L_s(parameters["g_dd"], E_dd, parameters["g_dsr"], E_dsr, parameters["g_dsa"], E_dsa)

        R_rs_surface = surface.R_rs_surf(L_s, E_d, parameters["rho_L"])

        R_rs_sim = R_rs_water + R_rs_surface + parameters["offset"]
        return R_rs_sim
     
    else:
        R_rs_sim = R_rs_water + parameters["offset"]
        return R_rs_sim

def forward_glint(parameters,
        wavelengths,
        E_0_res=[],
        a_oz_res=[],
        a_ox_res=[],
        a_wv_res=[],
        E_dd_res=[],
        E_dsa_res=[],
        E_dsr_res=[],
        E_d_res=[],
        E_ds_res=[]):
    if len(E_dd_res) == 0:
        E_dd  = downwelling_irradiance.E_dd(wavelengths, parameters["theta_sun"], parameters["P"], parameters["AM"], parameters["RH"], parameters["H_oz"], parameters["WV"], parameters["alpha"], parameters["beta"], E_0_res, a_oz_res, a_ox_res, a_wv_res, E_dd_res)
    else:
        E_dd = E_dd_res

    if len(E_dsa_res) == 0:
        E_dsa = downwelling_irradiance.E_dsa(wavelengths, parameters["theta_sun"], parameters["P"], parameters["AM"], parameters["RH"], parameters["H_oz"], parameters["WV"], parameters["alpha"], parameters["beta"], E_0_res, a_oz_res, a_ox_res, a_wv_res, E_dsa_res)
    else:
        E_dsa = E_dsa_res

    if len(E_dsr_res) == 0:
        E_dsr = downwelling_irradiance.E_dsr(wavelengths, parameters["theta_sun"], parameters["P"], parameters["AM"], parameters["RH"], parameters["H_oz"], parameters["WV"], E_0_res, a_oz_res, a_ox_res, a_wv_res, E_dsr_res)
    else:
        E_dsr = E_dsr_res

    if len(E_ds_res) == 0:
        E_ds = downwelling_irradiance.E_ds(E_dsr, E_dsa)
    else:
        E_ds = E_ds_res

    if len(E_d_res) == 0:
        E_d = downwelling_irradiance.E_d(E_dd, E_ds, parameters["f_dd"], parameters["f_ds"])
    else:
        E_d = E_d_res

    L_s = sky_radiance.L_s(parameters["g_dd"], E_dd, parameters["g_dsr"], E_dsr, parameters["g_dsa"], E_dsa)

    R_rs_surface = surface.R_rs_surf(L_s, E_d, parameters["rho_L"])

    return R_rs_surface

def func2opt(params, 
             R_rs,
             wavelengths, 
             weights = [],
             a_i_spec_res=[],
             a_w_res=[],
             a_Y_N_res = [],
             a_NAP_N_res = [],
             b_phy_norm_res = [],
             b_bw_res = [],
             b_X_norm_res=[],
             b_Mie_norm_res=[],
             R_i_b_res = [],
             da_W_div_dT_res=[],
             E_0_res=[],
             a_oz_res=[],
             a_ox_res=[],
             a_wv_res=[],
             E_dd_res=[],
             E_dsa_res=[],
             E_dsr_res=[],
             E_d_res=[],
             E_ds_res=[]):
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

    R_rs_sim = forward(parameters=params,
                       wavelengths=wavelengths,
                       a_i_spec_res=a_i_spec_res,
                       a_w_res=a_w_res,
                       a_Y_N_res = a_Y_N_res,
                       a_NAP_N_res = a_NAP_N_res,
                       b_phy_norm_res=b_phy_norm_res,
                       b_bw_res=b_bw_res,
                       b_X_norm_res=b_X_norm_res,
                       b_Mie_norm_res=b_Mie_norm_res,
                       R_i_b_res=R_i_b_res,
                       da_W_div_dT_res=da_W_div_dT_res,
                       E_0_res=E_0_res,
                       a_oz_res=a_oz_res,
                       a_ox_res=a_ox_res,
                       a_wv_res=a_wv_res,
                       E_dd_res=E_dd_res,
                       E_dsa_res=E_dsa_res,
                       E_dsr_res=E_dsr_res,
                       E_d_res=E_d_res,
                       E_ds_res=E_ds_res)
           
    # return utils.compute_residual(R_rs, R_rs_sim, method=params['error_method'], weights=weights)
    return (R_rs - R_rs_sim) * weights


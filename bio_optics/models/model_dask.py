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
#  Phillip Noel, pnoel1 AT asu.edu
#
# bio_optics
#  This code base builds on the extensive work of many researchers. For example, models were developed by Albert & Mobley [1] and Gege [2]; 
#  and the methodology was mainly developed by Gege [3,4,5] and Albert & Gege [6]. Please give proper attribution when using this code for publication.
#  A former version of this code base was developed in the course of the CarbonMapper Land and Ocean Program [7].
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
from lmfit import minimize, Parameters
from ..water import absorption, backscattering, attenuation, bottom_reflectance
from ..water import albert_mobley as water_alg
from .. atmosphere import sky_radiance, downwelling_irradiance
from ..surface import surface, air_water
from ..helper import resampling, utils


def invert(params, 
           Rrs, 
           wavelengths,
           weights=[],
           a_res=[],
           bb_res=[],
           a_i_spec_res=[],
           a_w_res=[],
           a_Y_N_res = [],
           a_NAP_N_res = [],
           b_phy_norm_res = [],
           bb_w_res = [],
           b_X_norm_res=[],
           b_Mie_norm_res=[],
           R_b_i_res = [],
           da_w_div_dT_res=[],
           E0_res=[],
           a_oz_res=[],
           a_ox_res=[],
           a_wv_res=[],
           Ed_d_res=[],
           Ed_sa_res=[],
           Ed_sr_res=[],
           Ed_res=[],
           Ed_s_res=[],
           n2_res=[],
           Ls_Ed=[],
           method="least-squares", 
           max_nfev=400
           ):
    """
    Function to inversely fit a modeled spectrum to a measurement spectrum.
    We made a slight change to this function so it can be parallelized using Dasks map_block() functionality.
    Dask serializes inputs in the parallelization process, but the lmfit.Parameters() object was not serializable.
    In this versoin we implemented a simple fix and hand over a serializable dictionary generated from the lmfit.Parameters object 
    using params.dumps() and then rebuild it inside the function using Parameters().loads(params) where params is the dict.
    
    :param params: dictionary of parameters required to specify the model, generated from an lmfit.Parameters object using params.dumps()
    :param Rrs: Remote sensing reflectance spectrum [sr-1]
    :param wavelengths: wavelengths of R_rs bands [nm]
    :param weights: spectral weighing coefficients
    :param a_i_spec_res: optional, specific absorption coefficients of phytoplankton types resampled to sensor's band settings. Will be computed within function if not provided.
    :param a_w_res: optional, absorption of pure water resampled to sensor's band settings. Will be computed within function if not provided.
    :param a_Y_N_res: optional, normalized absorption coefficients of CDOM resampled to sensor's band settings. Will be computed within function if not provided.
    :param a_NAP_N_res: optional, normalized absorption coefficients of NAP resampled to sensor's band settings. Will be computed within function if not provided.
    :param b_phy_norm_res: optional, preresampling b_phy_norm saves a lot of time during inversion. Will be computed within function if not provided.
    :param bb_w_res: optional, precomputing b_bw b_bw saves a lot of time during inversion. Will be computed within function if not provided.
    :param b_X_norm_res: optional, precomputing b_bX_norm before inversion saves a lot of time. Will be computed within function if not provided.
    :param b_Mie_norm_res: optional, if n and lambda_S are not fit params, the last part of the equation can be precomputed to save time. Will be computed within function if not provided.
    :param R_b_i_res: optional, preresampling R_b_i before inversion saves a lot of time. Will be computed within function if not provided.
    :param da_w_div_dT_res: optional, temperature gradient of pure water absorption resampled  to sensor's band settings. Will be computed within function if not provided.
    :param E0_res: optional, precomputing E0 saves a lot of time. Will be computed within function if not provided.
    :param a_oz_res: optional, precomputing a_oz saves a lot of time. Will be computed within function if not provided.
    :param a_ox_res: optional, precomputing a_ox saves a lot of time. Will be computed within function if not provided.
    :param a_wv_res: optional, precomputing a_wv saves a lot of time. Will be computed within function if not provided.
    :param Ed_d_res: optional, preresampling E_dd before inversion saves a lot of time. Will be computed within function if not provided.
    :param Ed_sa_res: optional, preresampling E_dsa before inversion saves a lot of time. Will be computed within function if not provided.
    :param Ed_sr_res: optional, preresampling E_dsr before inversion saves a lot of time. Will be computed within function if not provided.
    :param Ed_res: optional, preresampling E_d before inversion saves a lot of time. Will be computed within function if not provided.
    :param method: name of the fitting method to use by lmfit, default: 'least-squares'
    :param max_nfev: maximum number of function evaluations, default: 400
    :return: object containing the optimized parameters and several goodness-of-fit statistics.
    """    

    # reconvert the params dict to an lmfit.Parameters object
    # dask cannot handle serialization of lmfit.Parameters objects but it can serialize dictionaries so this is a simple workaround.
    params = Parameters().loads(params)

    if len(weights)==0:
        weights = np.ones(len(Rrs))
    
    if params['fit_surface'].value:
        res = minimize(func2opt, 
                       params, 
                       args=(Rrs, 
                             wavelengths, 
                             weights,
                             a_res,
                             bb_res,
                             a_i_spec_res, 
                             a_w_res, 
                             a_Y_N_res, 
                             a_NAP_N_res, 
                             b_phy_norm_res, 
                             bb_w_res, 
                             b_X_norm_res, 
                             b_Mie_norm_res, 
                             R_b_i_res, 
                             da_w_div_dT_res,
                             E0_res,
                             a_oz_res,
                             a_ox_res,
                             a_wv_res,
                             Ed_d_res,
                             Ed_sa_res,
                             Ed_sr_res,
                             Ed_res,
                             Ed_s_res,
                             n2_res,
                             Ls_Ed), 
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
                       args=(Rrs, 
                             wavelengths, 
                             weights,
                             a_res,
                             bb_res,
                             a_i_spec_res, 
                             a_w_res, 
                             a_Y_N_res, 
                             a_NAP_N_res, 
                             b_phy_norm_res, 
                             bb_w_res, 
                             b_X_norm_res, 
                             b_Mie_norm_res, 
                             R_b_i_res, 
                             da_w_div_dT_res,
                             E0_res,
                             a_oz_res,
                             a_ox_res,
                             a_wv_res,
                             Ed_d_res,
                             Ed_sa_res,
                             Ed_sr_res,
                             Ed_res,
                             Ed_s_res,
                             n2_res,
                             Ls_Ed), 
                       method=method, 
                       max_nfev=max_nfev) 
    return res


def forward(parameters,
        wavelengths,
        a_res=[],
        bb_res=[],
        a_w_res=[],
        da_w_div_dT_res=[],
        a_i_spec_res=[],
        a_Y_N_res = [],
        a_NAP_N_res = [],
        b_phy_norm_res = [],
        bb_w_res = [],
        b_X_norm_res=[],
        b_Mie_norm_res=[],
        R_b_i_res = [],
        E0_res=[],
        a_oz_res=[],
        a_ox_res=[],
        a_wv_res=[],
        Ed_d_res=[],
        Ed_sa_res=[],
        Ed_sr_res=[],
        Ed_res=[],
        Ed_s_res=[],
        n2_res=[],
        Ls_Ed=[]):
    """
    Forward simulation of a shallow water remote sensing reflectance spectrum based on the provided parameterization.

    Args:
        parameters (_type_): _description_
        wavelengths (_type_): _description_
        a_res (list, optional): _description_. Defaults to [].
        bb_res (list, optional): _description_. Defaults to [].
        a_w_res (list, optional): _description_. Defaults to [].
        da_w_div_dT_res (list, optional): _description_. Defaults to [].
        a_i_spec_res (list, optional): _description_. Defaults to [].
        a_Y_N_res (list, optional): _description_. Defaults to [].
        a_NAP_N_res (list, optional): _description_. Defaults to [].
        b_phy_norm_res (list, optional): _description_. Defaults to [].
        b_bw_res (list, optional): _description_. Defaults to [].
        b_X_norm_res (list, optional): _description_. Defaults to [].
        b_Mie_norm_res (list, optional): _description_. Defaults to [].
        R_b_i_res (list, optional): _description_. Defaults to [].
        E0_res (list, optional): _description_. Defaults to [].
        a_oz_res (list, optional): _description_. Defaults to [].
        a_ox_res (list, optional): _description_. Defaults to [].
        a_wv_res (list, optional): _description_. Defaults to [].
        Ed_d_res (list, optional): _description_. Defaults to [].
        Ed_sa_res (list, optional): _description_. Defaults to [].
        Ed_sr_res (list, optional): _description_. Defaults to [].
        Ed_res (list, optional): _description_. Defaults to [].
        Ed_s_res (list, optional): _description_. Defaults to [].
        n2_res (list, optional): _description_. Defaults to [].
        Ls_Ed (list, optional): _description_. Defaults to [].

    Returns:
        _type_: _description_
    """
    if len(n2_res) == 0:
        n2 = parameters["n2"]
    else:
        n2 = n2_res

    if "rho_L" in parameters:
        rho_L = parameters["rho_L"].value
    else:
        rho_L = air_water.fresnel(parameters["theta_view"], n1=parameters["n1"], n2=n2)

    if len(Ls_Ed) == 0:
        Ls_Ed = np.zeros_like(wavelengths)

    ctsp = np.cos(air_water.snell(parameters["theta_sun"],  n1=parameters["n1"], n2=n2))  #cos of theta_sun_prime. theta_sun_prime = snell(theta_sun, n1, n2)
    ctvp = np.cos(air_water.snell(parameters["theta_view"], n1=parameters["n1"], n2=n2))

    if len(a_res) == 0:
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
                            da_w_div_dT_res=da_w_div_dT_res, 
                            a_i_spec_res=a_i_spec_res, 
                            a_Y_N_res=a_Y_N_res,
                            a_NAP_N_res=a_NAP_N_res)
    else:
        a_sim = a_res
    
    if len(bb_res) == 0:
        bb_sim = backscattering.bb(C_X=parameters["C_X"], C_Mie=parameters["C_Mie"], C_phy=np.sum([parameters["C_0"], parameters["C_1"], parameters["C_2"], parameters["C_3"], parameters["C_4"], parameters["C_5"]]), wavelengths=wavelengths, 
                            fresh=parameters["fresh"],
                            bb_phy_spec=parameters["bb_phy_spec"],
                            bb_Mie_spec=parameters["bb_Mie_spec"],
                            bb_X_spec=parameters["bb_X_spec"],
                            b_X_norm_factor=parameters["b_X_norm_factor"],
                            lambda_S=parameters["lambda_S"],
                            n=parameters["n"],
                            bb_w_res=bb_w_res, 
                            b_phy_norm_res=b_phy_norm_res, 
                            b_X_norm_res=b_X_norm_res, 
                            b_Mie_norm_res=b_Mie_norm_res)
    else:
        bb_sim = bb_res

    Rrsb = bottom_reflectance.Rrs_b(parameters["f_0"], parameters["f_1"], parameters["f_2"], parameters["f_3"], parameters["f_4"], parameters["f_5"], B_0=parameters["B_0"], B_1=parameters["B_1"], B_2=parameters["B_2"], B_3=parameters["B_3"], B_4=parameters["B_4"], B_5=parameters["B_5"], wavelengths=wavelengths, R_b_i_res=R_b_i_res)

    ob = attenuation.omega_b(a_sim, bb_sim) #ob is omega_b. Shortened to distinguish between new var and function params.

    frs = water_alg.f_rs(omega_b=ob, cos_t_sun_p=ctsp, cos_t_view_p=ctvp)

    rrsd = water_alg.rrs_deep(f_rs=frs, omega_b=ob)

    Kd =  attenuation.Kd(a=a_sim, bb=bb_sim, cos_t_sun_p=ctsp, kappa_0=parameters["kappa_0"])

    kuW = attenuation.ku_w(a=a_sim, bb=bb_sim, omega_b=ob, cos_t_sun_p=ctsp, cos_t_view_p=ctvp)

    kuB = attenuation.ku_b(a=a_sim, bb=bb_sim, omega_b=ob, cos_t_sun_p=ctsp, cos_t_view_p=ctvp)

    Rrs_water = air_water.below2above(water_alg.rrs_shallow(rrs_deep=rrsd, Kd=Kd, ku_W=kuW, zB=parameters["zB"], Rrs_b=Rrsb, ku_b=kuB)) # zeta & gamma

    if parameters["fit_surface"].value:
        if len(Ed_d_res) == 0:
            Ed_d  = downwelling_irradiance.Ed_d(wavelengths, parameters["theta_sun"], parameters["P"], parameters["AM"], parameters["RH"], parameters["H_oz"], parameters["WV"], parameters["alpha"], parameters["beta"], E0_res, a_oz_res, a_ox_res, a_wv_res, Ed_d_res)
        else:
            Ed_d = Ed_d_res

        if len(Ed_sa_res) == 0:
            Ed_sa = downwelling_irradiance.Ed_sa(wavelengths, parameters["theta_sun"], parameters["P"], parameters["AM"], parameters["RH"], parameters["H_oz"], parameters["WV"], parameters["alpha"], parameters["beta"], E0_res, a_oz_res, a_ox_res, a_wv_res, Ed_sa_res)
        else:
            Ed_sa = Ed_sa_res

        if len(Ed_sr_res) == 0:
            Ed_sr = downwelling_irradiance.Ed_sr(wavelengths, parameters["theta_sun"], parameters["P"], parameters["AM"], parameters["RH"], parameters["H_oz"], parameters["WV"], parameters["alpha"], parameters["beta"], E0_res, a_oz_res, a_ox_res, a_wv_res, Ed_sr_res)
        else:
            Ed_sr = Ed_sr_res

        if len(Ed_s_res) == 0:
            Ed_s = downwelling_irradiance.Ed_s(Ed_sr, Ed_sa)
        else:
            Ed_s = Ed_s_res

        if len(Ed_res) == 0:
            Ed = downwelling_irradiance.Ed(Ed_d, Ed_s, parameters["f_dd"], parameters["f_ds"])
        else:
            Ed = Ed_res

        L_s = sky_radiance.L_s(parameters["f_dd"], parameters["g_dd"], Ed_d, parameters["f_ds"], parameters["g_dsr"], Ed_sr, parameters["g_dsa"], Ed_sa)

        Rrs_surface = surface.Rrs_surf(L_s, Ed, rho_L, parameters["d_r"])

        Rrs_surface += air_water.fresnel(parameters['theta_view'], n2=n2) * Ls_Ed

        Rrs_sim = Rrs_water + Rrs_surface + parameters["offset"]
        return Rrs_sim
     
    else:
        Rrs_sim = Rrs_water + parameters["offset"]
        return Rrs_sim


def forward_glint(parameters,
        wavelengths,
        E0_res=[],
        a_oz_res=[],
        a_ox_res=[],
        a_wv_res=[],
        Ed_d_res=[],
        Ed_sa_res=[],
        Ed_sr_res=[],
        Ed_res=[],
        Ed_s_res=[],
        n2_res=[],
        Ls_Ed=[]):
    """_summary_

    Args:
        parameters (_type_): _description_
        wavelengths (_type_): _description_
        E0_res (list, optional): _description_. Defaults to [].
        a_oz_res (list, optional): _description_. Defaults to [].
        a_ox_res (list, optional): _description_. Defaults to [].
        a_wv_res (list, optional): _description_. Defaults to [].
        Ed_d_res (list, optional): _description_. Defaults to [].
        Ed_sa_res (list, optional): _description_. Defaults to [].
        Ed_sr_res (list, optional): _description_. Defaults to [].
        Ed_res (list, optional): _description_. Defaults to [].
        Ed_s_res (list, optional): _description_. Defaults to [].
        n2_res (list, optional): _description_. Defaults to [].
        Ls_Ed (list, optional): _description_. Defaults to [].

    Returns:
        _type_: _description_
    """
    
    if len(n2_res) == 0:
        n2 = parameters["n2"]
    else:
        n2 = n2_res
    
    if "rho_L" in parameters:
        rho_L = parameters["rho_L"].value
    else:
        rho_L = air_water.fresnel(parameters["theta_view"], n1=parameters["n1"], n2=n2)

    if len(Ls_Ed) == 0:
        Ls_Ed = np.zeros_like(wavelengths)

    if len(Ed_d_res) == 0:
        Ed_d  = downwelling_irradiance.Ed_d(wavelengths, parameters["theta_sun"], parameters["P"], parameters["AM"], parameters["RH"], parameters["H_oz"], parameters["WV"], parameters["alpha"], parameters["beta"], E0_res, a_oz_res, a_ox_res, a_wv_res, Ed_d_res)
    else:
        Ed_d = Ed_d_res

    if len(Ed_sa_res) == 0:
        Ed_sa = downwelling_irradiance.Ed_sa(wavelengths, parameters["theta_sun"], parameters["P"], parameters["AM"], parameters["RH"], parameters["H_oz"], parameters["WV"], parameters["alpha"], parameters["beta"], E0_res, a_oz_res, a_ox_res, a_wv_res, Ed_sa_res)
    else:
        Ed_sa = Ed_sa_res

    if len(Ed_sr_res) == 0:
        Ed_sr = downwelling_irradiance.Ed_sr(wavelengths, parameters["theta_sun"], parameters["P"], parameters["AM"], parameters["RH"], parameters["H_oz"], parameters["WV"], parameters["alpha"], parameters["beta"], E0_res, a_oz_res, a_ox_res, a_wv_res, Ed_sr_res)
    else:
        Ed_sr = Ed_sr_res

    if len(Ed_s_res) == 0:
        Ed_s = downwelling_irradiance.Ed_s(Ed_sr, Ed_sa)
    else:
        Ed_s = Ed_s_res

    if len(Ed_res) == 0:
        Ed = downwelling_irradiance.Ed(Ed_d, Ed_s, parameters["f_dd"], parameters["f_ds"])
    else:
        Ed = Ed_res

    L_s = sky_radiance.L_s(parameters["f_dd"], parameters["g_dd"], Ed_d, parameters["f_ds"], parameters["g_dsr"], Ed_sr, parameters["g_dsa"], Ed_sa)

    Rrs_surface = surface.Rrs_surf(L_s, Ed, rho_L, parameters["d_r"])

    Rrs_surface += air_water.fresnel(parameters['theta_view'], n2=n2) * Ls_Ed

    return Rrs_surface


def func2opt(parameters, 
             Rrs,
             wavelengths,
             weights=[],
             a_res=[],
             bb_res=[],
             a_i_spec_res=[],
             a_w_res=[],
             a_Y_N_res = [],
             a_NAP_N_res = [],
             b_phy_norm_res = [],
             bb_w_res = [],
             b_X_norm_res=[],
             b_Mie_norm_res=[],
             R_b_i_res = [],
             da_w_div_dT_res=[],
             E0_res=[],
             a_oz_res=[],
             a_ox_res=[],
             a_wv_res=[],
             Ed_d_res=[],
             Ed_sa_res=[],
             Ed_sr_res=[],
             Ed_res=[],
             Ed_s_res=[],
             n2_res=[],
             Ls_Ed=[],
             ):
    """
    Error function around model to be minimized by changing fit parameters.
    
    :param params: lmfit Parameters object containing all Parameter objects that are required to specify the model
    :param Rrs: Remote sensing reflectance spectrum [sr-1]
    :param wavelengths: wavelengths of R_rs bands [nm]
    :param weights: spectral weighing coefficients
    :param a_i_spec_res: optional, specific absorption coefficients of phytoplankton types resampled to sensor's band settings. Will be computed within function if not provided.
    :param a_w_res: optional, absorption of pure water resampled to sensor's band settings. Will be computed within function if not provided.
    :param a_Y_N_res: optional, normalized absorption coefficients of CDOM resampled to sensor's band settings. Will be computed within function if not provided.
    :param a_NAP_N_res: optional, normalized absorption coefficients of NAP resampled to sensor's band settings. Will be computed within function if not provided.
    :param b_phy_norm_res: optional, preresampling b_phy_norm saves a lot of time during inversion. Will be computed within function if not provided.
    :param bb_w_res: optional, precomputing bb_w saves a lot of time during inversion. Will be computed within function if not provided.
    :param b_X_norm_res: optional, precomputing b_bX_norm before inversion saves a lot of time. Will be computed within function if not provided.
    :param b_Mie_norm_res: optional, if n and lambda_S are not fit params, the last part of the equation can be precomputed to save time. Will be computed within function if not provided.
    :param R_b_i_res: optional, preresampling R_b_i before inversion saves a lot of time. Will be computed within function if not provided.
    :param da_w_div_dT_res: optional, temperature gradient of pure water absorption resampled  to sensor's band settings. Will be computed within function if not provided.
    :param E0_res: optional, precomputing E0 saves a lot of time. Will be computed within function if not provided.
    :param a_oz_res: optional, precomputing a_oz saves a lot of time. Will be computed within function if not provided.
    :param a_ox_res: optional, precomputing a_ox saves a lot of time. Will be computed within function if not provided.
    :param a_wv_res: optional, precomputing a_wv saves a lot of time. Will be computed within function if not provided.
    :param Ed_d_res: optional, preresampling E_dd before inversion saves a lot of time. Will be computed within function if not provided.
    :param Ed_sa_res: optional, preresampling E_dsa before inversion saves a lot of time. Will be computed within function if not provided.
    :param Ed_sr_res: optional, preresampling E_dsr before inversion saves a lot of time. Will be computed within function if not provided.
    :param Ed_res: optional, preresampling E_d before inversion saves a lot of time. Will be computed within function if not provided.
    :return: weighted difference between measured and simulated R_rs
    """    
    if len(weights)==0:
        weights = np.ones(len(wavelengths))

    if len(n2_res) == 0:
        n2 = parameters["n2"]
    else:
        n2 = n2_res
    
    if "rho_L" in parameters:
        rho_L = parameters["rho_L"].value
    else:
        rho_L = air_water.fresnel(parameters["theta_view"], n1=parameters["n1"], n2=n2)

    if len(Ls_Ed) == 0:
        Ls_Ed = np.zeros_like(wavelengths)

    Rrs_sim = forward(parameters=parameters,
                       wavelengths=wavelengths,
                       a_res=a_res,
                       bb_res=bb_res,
                       a_i_spec_res=a_i_spec_res,
                       a_w_res=a_w_res,
                       a_Y_N_res = a_Y_N_res,
                       a_NAP_N_res = a_NAP_N_res,
                       b_phy_norm_res=b_phy_norm_res,
                       bb_w_res=bb_w_res,
                       b_X_norm_res=b_X_norm_res,
                       b_Mie_norm_res=b_Mie_norm_res,
                       R_b_i_res=R_b_i_res,
                       da_w_div_dT_res=da_w_div_dT_res,
                       E0_res=E0_res,
                       a_oz_res=a_oz_res,
                       a_ox_res=a_ox_res,
                       a_wv_res=a_wv_res,
                       Ed_d_res=Ed_d_res,
                       Ed_sa_res=Ed_sa_res,
                       Ed_sr_res=Ed_sr_res,
                       Ed_res=Ed_res,
                       Ed_s_res=Ed_s_res,
                       n2_res=n2_res,
                       Ls_Ed=Ls_Ed)
    
    return utils.compute_residual(Rrs, Rrs_sim, method=parameters['error_method'], weights=weights)
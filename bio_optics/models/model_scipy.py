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


from    lmfit import Parameters
import  numpy as np
from    scipy.optimize import least_squares
from .. water import absorption, backscattering, attenuation, bottom_reflectance
from .. water import albert_mobley as water_alg
from .. atmosphere import sky_radiance, downwelling_irradiance
from .. surface import surface, air_water

def residual(Rrs_sim, 
             Rrs,
             weights = [],
             ):
    """
    Error function around model to be minimized by changing fit parameters.
    
    :return: weighted difference between measured and simulated R_rs

    # Math: L_i = R_{rs_{sim}}(\lambda_i) - R_{rs}(\lambda_i)
    """
    
    if len(weights)==0:
        err = (Rrs_sim - Rrs)
    else:
        err = (Rrs_sim - Rrs) * weights
        
    return err


def get_residuals(fun, data, weights=[]):
    """
    This is a function decorator i.e. the return value of this function is
    another function.

    In this case, it returns a function that is called exactly as fun and
    returns the value of fun, minus data, times weights.

    compositefun must accept variable parameters because it must accept the same
    parameters as fun, whatever they may be at runtime.
    """
    def compositefun(*args4inner, **kwargs4inner):
          if len(weights)==0:
            return (fun(*args4inner, **kwargs4inner) - data)
          else:
            return (fun(*args4inner, **kwargs4inner) - data) * weights
    
    return compositefun


def get_fun_shim(fun, wavelengths, fit_param_names, params_obj):
    """
    This is a function decorator i.e. the return value of this function is
    another function.

    In this case, a function which accepts a list of values as the first
    argument is return value. The returned function will be the argument to 
    scipy.minimize.least_squares because scipy is written to minimize functions
    accepting a fit params list as the first argument.

    This function decorator is like a shim: it returns a function that accepts
    a fit parameter list as the first arg, and marshalls that into an
    lmfit.Parameters object which the userfun accepts instead of a parameters
    list.
    """
    def outer_fun(x0, *args4fun, **kwargs4fun):

        for i, val in enumerate(fit_param_names):
            params_obj[val].value = x0[i]

        return fun(params_obj, wavelengths, *args4fun, **kwargs4fun)

    return outer_fun


def get_dfun_shim(dfun, wavelengths, fit_param_names, params_obj):
    """
    This is a function decorator i.e. the return value of this function is
    another function.

    In this case, a function which accepts a list of values as the first
    argument is return value. The returned function will be the argument to 
    scipy.minimize.least_squares because scipy is written to minimize functions
    accepting a fit params list as the first argument.

    This function decorator is like a shim: it returns a function that accepts
    a fit parameter list as the first arg, and marshalls that into an
    lmfit.Parameters object which the userfun accepts instead of a parameters
    list.
    """
    def outer_fun(x0, *args4dfun, **kwargs4dfun):

        for i, val in enumerate(fit_param_names):
            params_obj[val].value = x0[i]

        return dfun(params_obj, wavelengths, *args4dfun, **kwargs4dfun)

    return outer_fun


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
        R_i_b_res = [],
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

    Rrsb = bottom_reflectance.Rrs_b(parameters["f_0"], parameters["f_1"], parameters["f_2"], parameters["f_3"], parameters["f_4"], parameters["f_5"], B_0=parameters["B_0"], B_1=parameters["B_1"], B_2=parameters["B_2"], B_3=parameters["B_3"], B_4=parameters["B_4"], B_5=parameters["B_5"], wavelengths=wavelengths, R_i_b_res=R_i_b_res)

    ob = attenuation.omega_b(a_sim, bb_sim) #ob is omega_b. Shortened to distinguish between new var and function params.

    frs = water_alg.f_rs(omega_b=ob, cos_t_sun_p=ctsp, cos_t_view_p=ctvp)

    rrsd = water_alg.rrs_deep(f_rs=frs, omega_b=ob)

    Kd =  attenuation.Kd(a=a_sim, bb=bb_sim, cos_t_sun_p=ctsp, kappa_0=parameters["kappa_0"])

    kuW = attenuation.ku_w(a=a_sim, bb=bb_sim, omega_b=ob, cos_t_sun_p=ctsp, cos_t_view_p=ctvp)

    kuB = attenuation.ku_b(a=a_sim, bb=bb_sim, omega_b=ob, cos_t_sun_p=ctsp, cos_t_view_p=ctvp)

    R_rs_water = air_water.below2above(water_alg.rrs_shallow(rrs_deep=rrsd, Kd=Kd, ku_W=kuW, zB=parameters["zB"], Rrs_b=Rrsb, ku_b=kuB)) # zeta & gamma

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
            E_ds = downwelling_irradiance.Ed_s(Ed_sr, Ed_sa)
        else:
            E_ds = Ed_s_res

        if len(Ed_res) == 0:
            Ed = downwelling_irradiance.Ed(Ed_d, E_ds, parameters["f_dd"], parameters["f_ds"])
        else:
            Ed = Ed_res

        L_s = sky_radiance.L_s(parameters["f_dd"], parameters["g_dd"], Ed_d, parameters["f_ds"], parameters["g_dsr"], Ed_sr, parameters["g_dsa"], Ed_sa)

        Rrs_surface = surface.Rrs_surf(L_s, Ed, rho_L, parameters["d_r"])

        Rrs_surface += air_water.fresnel(parameters['theta_view'], n2=n2) * Ls_Ed

        Rrs_sim = R_rs_water + Rrs_surface + parameters["offset"]
        return Rrs_sim
     
    else:
        Rrs_sim = R_rs_water + parameters["offset"]
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

def dfun(parameters,
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
        R_i_b_res = [],
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

    fit_params = []
    partials   = {}
    jacobian   = []

    ones = np.ones_like(wavelengths)

    for param in parameters.keys():
        if parameters[param].vary or parameters[param].expr:
            fit_params.append(param)
    
    if len(n2_res) == 0:
        n2 = parameters["n2"]
    else:
        n2 = n2_res

    if "rho_L" in parameters:
        rho_L = parameters["rho_L"].value
    else:
        rho_L = air_water.fresnel(parameters["theta_view"], n1=parameters["n1"], n2=n2)

    ctsp = np.cos(air_water.snell(parameters["theta_sun"], n1=parameters["n1"],  n2=n2))  #cos of theta_sun_prime. theta_sun_prime = snell(theta_sun, n1, n2)
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

    Rrsb = bottom_reflectance.Rrs_b(parameters["f_0"], parameters["f_1"], parameters["f_2"], parameters["f_3"], parameters["f_4"], parameters["f_5"], B_0=parameters["B_0"], B_1=parameters["B_1"], B_2=parameters["B_2"], B_3=parameters["B_3"], B_4=parameters["B_4"], B_5=parameters["B_5"], wavelengths=wavelengths, R_i_b_res=R_i_b_res)

    ob = attenuation.omega_b(a_sim, bb_sim) #ob is omega_b. Shortened to distinguish between new var and function params.

    frs = water_alg.f_rs(omega_b=ob, cos_t_sun_p=ctsp, cos_t_view_p=ctvp)

    rrsd = water_alg.rrs_deep(f_rs=frs, omega_b=ob)

    Kd =  attenuation.Kd(a=a_sim, bb=bb_sim, cos_t_sun_p=ctsp, kappa_0=parameters["kappa_0"])

    kuW = attenuation.ku_w(a=a_sim, bb=bb_sim, omega_b=ob, cos_t_sun_p=ctsp, cos_t_view_p=ctvp)

    kuB = attenuation.ku_b(a=a_sim, bb=bb_sim, omega_b=ob, cos_t_sun_p=ctsp, cos_t_view_p=ctvp)

    R_rs_water = air_water.below2above(water_alg.rrs_shallow(rrs_deep=rrsd, Kd=Kd, ku_W=kuW, zB=parameters["zB"], Rrs_b=Rrsb, ku_b=kuB)) # zeta & gamma

    if len(Ed_d_res) == 0:
        Ed_d  = downwelling_irradiance.Ed_d(wavelengths, parameters["theta_sun"], parameters["P"], parameters["AM"], parameters["RH"], parameters["H_oz"], parameters["WV"], parameters["alpha"], parameters["beta"], E0_res, a_oz_res, a_ox_res, a_wv_res, Ed_d_res)
    else:
        Ed_d = Ed_d_res

    if len(Ed_sa_res) == 0:
        Ed_sa = downwelling_irradiance.Ed_sa(wavelengths, parameters["theta_sun"], parameters["P"], parameters["AM"], parameters["RH"], parameters["H_oz"], parameters["WV"], parameters["alpha"], parameters["beta"], E0_res, a_oz_res, a_ox_res, a_wv_res, Ed_sa_res)
    else:
        Ed_sa = Ed_sa_res

    if len(Ed_sr_res) == 0:
        Ed_sr = downwelling_irradiance.Ed_sr(wavelengths, parameters["theta_sun"], parameters["P"], parameters["AM"], parameters["RH"], parameters["H_oz"], parameters["WV"], E0_res, a_oz_res, a_ox_res, a_wv_res, Ed_sr_res)
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

    dbdcphy = backscattering.dbb_div_dC_phy(wavelengths=wavelengths, bb_phy_spec=parameters["b_bphy_spec"], b_phy_norm_res=b_phy_norm_res)

    def df_div_dC_0():
        dadC0   = absorption.da_div_dC_i(i=0, wavelengths=wavelengths, a_i_spec_res=a_i_spec_res)
        domegadC0 = attenuation.domega_b_div_dp(a=a_sim, bb=bb_sim, da_div_dp=dadC0, dbb_div_dp=dbdcphy)
        dfrsdC0 = water_alg.df_rs_div_dp(omega_b=ob, domega_b_div_dp=domegadC0, cos_t_sun_p=ctsp, cos_t_view_p=ctvp)
        df_div_dC_0 = air_water.dbelow2above_div_dp(rrs=R_rs_water, 
                                                dr_rs_div_dp=water_alg.drrs_shallow_div_dp(r_rs_deep=rrsd,
                                                                                drrs_deep_div_dp=water_alg.drrs_deep_div_dp(frs, dfrsdC0, ob, domegadC0),
                                                                                K_d=Kd,
                                                                                dK_d_div_dp=attenuation.dK_d_div_dp(dadC0, dbdcphy, ctsp, parameters["kappa_0"]),
                                                                                k_uW=kuW,
                                                                                dk_uW_div_dp=attenuation.dk_uW_div_dp(a_sim, bb_sim, ob, dadC0, dbdcphy, domegadC0, ctsp, ctvp),
                                                                                r_rs_b=Rrsb,
                                                                                d_r_rs_b_div_dp=0,
                                                                                k_uB=kuB,
                                                                                dk_uB_div_dp=attenuation.dku_B_div_dp(a_sim, bb_sim, ob, dadC0, dbdcphy, domegadC0, ctsp, ctvp),
                                                                                zB=parameters["zB"]
                                                                            )
                                                )
        return df_div_dC_0

    def df_div_dC_1():
        dadC1   = absorption.da_div_dC_i(1, wavelengths, a_i_spec_res)
        domegadC1 = attenuation.domega_b_div_dp(a_sim, bb_sim, dadC1, dbdcphy)
        dfrsdC1 = water_alg.df_rs_div_dp(ob, domegadC1, cos_t_sun_p=ctsp, cos_t_view_p=ctvp)
        df_div_dC_1 = air_water.dbelow2above_div_dp(R_rs_water, 
                                                water_alg.drrs_shallow_div_dp(r_rs_deep=rrsd,
                                                                                drrs_deep_div_dp=water_alg.drrs_deep_div_dp(frs, dfrsdC1, ob, domegadC1),
                                                                                K_d=Kd,
                                                                                dK_d_div_dp=attenuation.dK_d_div_dp(dadC1, dbdcphy, ctsp, parameters["kappa_0"]),
                                                                                k_uW=kuW,
                                                                                dk_uW_div_dp=attenuation.dk_uW_div_dp(a_sim, bb_sim, ob, dadC1, dbdcphy, domegadC1, ctsp, ctvp),
                                                                                r_rs_b=Rrsb,
                                                                                d_r_rs_b_div_dp=0,
                                                                                k_uB=kuB,
                                                                                dk_uB_div_dp=attenuation.dku_B_div_dp(a_sim, bb_sim, ob, dadC1, dbdcphy, domegadC1, ctsp, ctvp),
                                                                                zB=parameters["zB"]
                                                                            )
                                                )
        return df_div_dC_1

    def df_div_dC_2():
        dadC2   = absorption.da_div_dC_i(2, wavelengths, a_i_spec_res)
        domegadC2 = attenuation.domega_b_div_dp(a_sim, bb_sim, dadC2, dbdcphy)
        dfrsdC2 = water_alg.df_rs_div_dp(ob, domegadC2, cos_t_sun_p=ctsp, cos_t_view_p=ctvp)
        df_div_dC_2 = air_water.dbelow2above_div_dp(R_rs_water, 
                                                water_alg.drrs_shallow_div_dp(r_rs_deep=rrsd,
                                                                                drrs_deep_div_dp=water_alg.drrs_deep_div_dp(frs, dfrsdC2, ob, domegadC2),
                                                                                K_d=Kd,
                                                                                dK_d_div_dp=attenuation.dK_d_div_dp(dadC2, dbdcphy, ctsp, parameters["kappa_0"]),
                                                                                k_uW=kuW,
                                                                                dk_uW_div_dp=attenuation.dk_uW_div_dp(a_sim, bb_sim, ob, dadC2, dbdcphy, domegadC2, ctsp, ctvp),
                                                                                r_rs_b=Rrsb,
                                                                                d_r_rs_b_div_dp=0,
                                                                                k_uB=kuB,
                                                                                dk_uB_div_dp=attenuation.dku_B_div_dp(a_sim, bb_sim, ob, dadC2, dbdcphy, domegadC2, ctsp, ctvp),
                                                                                zB=parameters["zB"]
                                                                            )
                                                )
        return df_div_dC_2

    def df_div_dC_3():
        dadC3   = absorption.da_div_dC_i(3, wavelengths, a_i_spec_res)
        domegadC3 = attenuation.domega_b_div_dp(a_sim, bb_sim, dadC3, dbdcphy)
        dfrsdC3 = water_alg.df_rs_div_dp(ob, domegadC3, cos_t_sun_p=ctsp, cos_t_view_p=ctvp)
        df_div_dC_3 = air_water.dbelow2above_div_dp(R_rs_water, 
                                                water_alg.drrs_shallow_div_dp(r_rs_deep=rrsd,
                                                                                drrs_deep_div_dp=water_alg.drrs_deep_div_dp(frs, dfrsdC3, ob, domegadC3),
                                                                                K_d=Kd,
                                                                                dK_d_div_dp=attenuation.dK_d_div_dp(dadC3, dbdcphy, ctsp, parameters["kappa_0"]),
                                                                                k_uW=kuW,
                                                                                dk_uW_div_dp=attenuation.dk_uW_div_dp(a_sim, bb_sim, ob, dadC3, dbdcphy, domegadC3, ctsp, ctvp),
                                                                                r_rs_b=Rrsb,
                                                                                d_r_rs_b_div_dp=0,
                                                                                k_uB=kuB,
                                                                                dk_uB_div_dp=attenuation.dku_B_div_dp(a_sim, bb_sim, ob, dadC3, dbdcphy, domegadC3, ctsp, ctvp),
                                                                                zB=parameters["zB"]
                                                                            )
                                                )
        return df_div_dC_3

    def df_div_dC_4():
        dadC4   = absorption.da_div_dC_i(4, wavelengths, a_i_spec_res)
        domegadC4 = attenuation.domega_b_div_dp(a_sim, bb_sim, dadC4, dbdcphy)
        dfrsdC4 = water_alg.df_rs_div_dp(ob, domegadC4, cos_t_sun_p=ctsp, cos_t_view_p=ctvp)
        df_div_dC_4 = air_water.dbelow2above_div_dp(R_rs_water, 
                                                water_alg.drrs_shallow_div_dp(r_rs_deep=rrsd,
                                                                                drrs_deep_div_dp=water_alg.drrs_deep_div_dp(frs, dfrsdC4, ob, domegadC4),
                                                                                K_d=Kd,
                                                                                dK_d_div_dp=attenuation.dK_d_div_dp(dadC4, dbdcphy, ctsp, parameters["kappa_0"]),
                                                                                k_uW=kuW,
                                                                                dk_uW_div_dp=attenuation.dk_uW_div_dp(a_sim, bb_sim, ob, dadC4, dbdcphy, domegadC4, ctsp, ctvp),
                                                                                r_rs_b=Rrsb,
                                                                                d_r_rs_b_div_dp=0,
                                                                                k_uB=kuB,
                                                                                dk_uB_div_dp=attenuation.dku_B_div_dp(a_sim, bb_sim, ob, dadC4, dbdcphy, domegadC4, ctsp, ctvp),
                                                                                zB=parameters["zB"]
                                                                            )
                                                )
        return df_div_dC_4

    def df_div_dC_5():
        dadC5   = absorption.da_div_dC_i(5, wavelengths, a_i_spec_res)
        domegadC5 = attenuation.domega_b_div_dp(a_sim, bb_sim, dadC5, dbdcphy)
        dfrsdC5 = water_alg.df_rs_div_dp(ob, domegadC5, cos_t_sun_p=ctsp, cos_t_view_p=ctvp)
        df_div_dC_5 = air_water.dbelow2above_div_dp(R_rs_water, 
                                                water_alg.drrs_shallow_div_dp(r_rs_deep=rrsd,
                                                                                drrs_deep_div_dp=water_alg.drrs_deep_div_dp(frs, dfrsdC5, ob, domegadC5),
                                                                                K_d=Kd,
                                                                                dK_d_div_dp=attenuation.dK_d_div_dp(dadC5, dbdcphy, ctsp, parameters["kappa_0"]),
                                                                                k_uW=kuW,
                                                                                dk_uW_div_dp=attenuation.dk_uW_div_dp(a_sim, bb_sim, ob, dadC5, dbdcphy, domegadC5, ctsp, ctvp),
                                                                                r_rs_b=Rrsb,
                                                                                d_r_rs_b_div_dp=0,
                                                                                k_uB=kuB,
                                                                                dk_uB_div_dp=attenuation.dku_B_div_dp(a_sim, bb_sim, ob, dadC5, dbdcphy, domegadC5, ctsp, ctvp),
                                                                                zB=parameters["zB"]
                                                                            )
                                                )
        return df_div_dC_5
        
    def df_div_dC_Y():
        dadCY = absorption.da_div_dC_Y(wavelengths=wavelengths, S=parameters["S"], lambda_0=parameters["lambda_0"], a_Y_N_res=a_Y_N_res)
        dbdCY = 0
        domegadCY = attenuation.domega_b_div_dp(a_sim, bb_sim, dadCY, dbdCY)
        dfrsdCY = water_alg.df_rs_div_dp(ob, domegadCY)
        df_div_dC_Y = air_water.dbelow2above_div_dp(R_rs_water, 
                                                water_alg.drrs_shallow_div_dp(r_rs_deep=rrsd,
                                                                                drrs_deep_div_dp=water_alg.drrs_deep_div_dp(frs, dfrsdCY, ob, domegadCY),
                                                                                K_d=Kd,
                                                                                dK_d_div_dp=attenuation.dK_d_div_dp(dadCY, dbdCY, ctsp, parameters["kappa_0"]),
                                                                                k_uW=kuW, 
                                                                                dk_uW_div_dp=attenuation.dk_uW_div_dp(a_sim, bb_sim, ob, dadCY, dbdCY, domegadCY, ctsp, ctvp),
                                                                                r_rs_b=Rrsb,
                                                                                d_r_rs_b_div_dp=0,
                                                                                k_uB=kuB,
                                                                                dk_uB_div_dp=attenuation.dku_B_div_dp(a_sim, bb_sim, ob, dadCY, dbdCY, domegadCY, ctsp, ctvp),
                                                                                zB=parameters["zB"]
                                                                                )
                                                )
        return df_div_dC_Y
        
    def df_div_dC_X():
        dadCX = absorption.da_div_dC_X(wavelengths=wavelengths, lambda_0=parameters["lambda_0"], a_NAP_spec_lambda_0=parameters["a_NAP_spec_lambda_0"], S_NAP=parameters["S_NAP"], a_NAP_N_res=a_NAP_N_res)
        dbdCX = backscattering.dbb_div_dC_X(wavelengths=wavelengths, b_X_norm_res=b_X_norm_res)
        domegadCX = attenuation.domega_b_div_dp(a_sim, bb_sim, dadCX, dbdCX)
        dfrsdCX = water_alg.df_rs_div_dp(ob, domegadCX)
        df_div_dC_X = air_water.dbelow2above_div_dp(R_rs_water,
                                                water_alg.drrs_shallow_div_dp(r_rs_deep=rrsd,
                                                                                drrs_deep_div_dp=water_alg.drrs_deep_div_dp(frs, dfrsdCX, ob, domegadCX),
                                                                                K_d=Kd,
                                                                                dK_d_div_dp=attenuation.dK_d_div_dp(dadCX, dbdCX, ctsp, ctvp),
                                                                                k_uW=kuW,
                                                                                dk_uW_div_dp=attenuation.dk_uW_div_dp(a_sim, bb_sim, ob, dadCX, dbdCX, domegadCX, ctsp, ctvp),
                                                                                r_rs_b=Rrsb,
                                                                                d_r_rs_b_div_dp=0,
                                                                                k_uB=kuB,
                                                                                dk_uB_div_dp=attenuation.dku_B_div_dp(a_sim, bb_sim, ob, dadCX, dbdCX, domegadCX, ctsp, ctvp),
                                                                                zB=parameters["zB"]
                                                                                )                                                
                    )
        return df_div_dC_X

    def df_div_dC_Mie():
        dadCMie = absorption.da_div_dC_Mie(wavelengths=wavelengths, lambda_0=parameters["lambda_0"], a_NAP_spec_lambda_0=parameters["a_NAP_spec_lambda_0"], S_NAP=parameters["S_NAP"], a_NAP_N_res=a_NAP_N_res)
        dbdCMie = backscattering.dbb_div_dC_Mie(wavelengths=wavelengths, n=parameters["n"], bb_Mie_spec=parameters["b_bMie_spec"], lambda_S=parameters["lambda_S"], bb_Mie_norm_res=b_Mie_norm_res)
        domegadCMie = attenuation.domega_b_div_dp(a_sim, bb_sim, dadCMie, dbdCMie)
        dfrsdCMie = water_alg.df_rs_div_dp(ob, domegadCMie)
        df_div_dC_Mie = air_water.dbelow2above_div_dp(R_rs_water,
                                                  water_alg.drrs_shallow_div_dp(r_rs_deep=rrsd,
                                                                                  drrs_deep_div_dp=water_alg.drrs_deep_div_dp(frs, dfrsdCMie, ob, domegadCMie),
                                                                                  K_d=Kd,
                                                                                  dK_d_div_dp=attenuation.dK_d_div_dp(dadCMie, dbdCMie, ctsp, ctvp),
                                                                                  k_uW=kuW,
                                                                                  dk_uW_div_dp=attenuation.dk_uW_div_dp(a_sim, bb_sim, ob, dadCMie, dbdCMie, domegadCMie, ctsp, ctvp),
                                                                                  r_rs_b=Rrsb,
                                                                                  d_r_rs_b_div_dp=0,
                                                                                  k_uB=kuB,
                                                                                  dk_uB_div_dp=attenuation.dku_B_div_dp(a_sim, bb_sim, ob, dadCMie, dbdCMie, domegadCMie, ctsp, ctvp),
                                                                                  zB=parameters["zB"]              
                                                                                )
                    )
        return df_div_dC_Mie

    def df_div_dd_r():
        return ones

    def df_div_df_0():
        df_div_df_0 = air_water.dbelow2above_div_dp(R_rs_water,
                                                water_alg.drrs_shallow_div_dp(r_rs_deep=rrsd,
                                                                                drrs_deep_div_dp=0,
                                                                                K_d=Kd,
                                                                                dK_d_div_dp=0,
                                                                                k_uW=kuW,
                                                                                dk_uW_div_dp=0,
                                                                                r_rs_b=Rrsb,
                                                                                d_r_rs_b_div_dp=bottom_reflectance.dRrs_b_div_df_i(0, wavelengths=wavelengths, R_i_b_res=R_i_b_res),
                                                                                k_uB=kuB,
                                                                                dk_uB_div_dp=0,
                                                                                zB=parameters["zB"]
                                                )
                                            )
        return df_div_df_0

    def df_div_df_1():
        df_div_df_1 = air_water.dbelow2above_div_dp(R_rs_water,
                                                water_alg.drrs_shallow_div_dp(r_rs_deep=rrsd,
                                                                                drrs_deep_div_dp=0,
                                                                                K_d=Kd,
                                                                                dK_d_div_dp=0,
                                                                                k_uW=kuW,
                                                                                dk_uW_div_dp=0,
                                                                                r_rs_b=Rrsb,
                                                                                d_r_rs_b_div_dp=bottom_reflectance.dRrs_b_div_df_i(1, wavelengths=wavelengths, R_i_b_res=R_i_b_res),
                                                                                k_uB=kuB,
                                                                                dk_uB_div_dp=0,
                                                                                zB=parameters["zB"]
                                                )
                                            )
        return df_div_df_1

    def df_div_df_2():
        df_div_df_2 = air_water.dbelow2above_div_dp(R_rs_water,
                                                water_alg.drrs_shallow_div_dp(r_rs_deep=rrsd,
                                                                                drrs_deep_div_dp=0,
                                                                                K_d=Kd,
                                                                                dK_d_div_dp=0,
                                                                                k_uW=kuW,
                                                                                dk_uW_div_dp=0,
                                                                                r_rs_b=Rrsb,
                                                                                d_r_rs_b_div_dp=bottom_reflectance.dRrs_b_div_df_i(2, wavelengths=wavelengths, R_i_b_res=R_i_b_res),
                                                                                k_uB=kuB,
                                                                                dk_uB_div_dp=0,
                                                                                zB=parameters["zB"]
                                                )
                                            )
        return df_div_df_2

    def df_div_df_3():
        df_div_df_3 = air_water.dbelow2above_div_dp(R_rs_water,
                                                water_alg.drrs_shallow_div_dp(r_rs_deep=rrsd,
                                                                                drrs_deep_div_dp=0,
                                                                                K_d=Kd,
                                                                                dK_d_div_dp=0,
                                                                                k_uW=kuW,
                                                                                dk_uW_div_dp=0,
                                                                                r_rs_b=Rrsb,
                                                                                d_r_rs_b_div_dp=bottom_reflectance.dRrs_b_div_df_i(3, wavelengths=wavelengths, R_i_b_res=R_i_b_res),
                                                                                k_uB=kuB,
                                                                                dk_uB_div_dp=0,
                                                                                zB=parameters["zB"]
                                                )
                                            )
        return df_div_df_3

    def df_div_df_4():
        df_div_df_4 = air_water.dbelow2above_div_dp(R_rs_water,
                                                water_alg.drrs_shallow_div_dp(r_rs_deep=rrsd,
                                                                                drrs_deep_div_dp=0,
                                                                                K_d=Kd,
                                                                                dK_d_div_dp=0,
                                                                                k_uW=kuW,
                                                                                dk_uW_div_dp=0,
                                                                                r_rs_b=Rrsb,
                                                                                d_r_rs_b_div_dp=bottom_reflectance.dRrs_b_div_df_i(4, wavelengths=wavelengths, R_i_b_res=R_i_b_res),
                                                                                k_uB=kuB,
                                                                                dk_uB_div_dp=0,
                                                                                zB=parameters["zB"]
                                                )
                                            )
        return df_div_df_4

    def df_div_df_5():
        df_div_df_5 = air_water.dbelow2above_div_dp(R_rs_water,
                                                water_alg.drrs_shallow_div_dp(r_rs_deep=rrsd,
                                                                                drrs_deep_div_dp=0,
                                                                                K_d=Kd,
                                                                                dK_d_div_dp=0,
                                                                                k_uW=kuW,
                                                                                dk_uW_div_dp=0,
                                                                                r_rs_b=Rrsb,
                                                                                d_r_rs_b_div_dp=bottom_reflectance.dRrs_b_div_df_i(5, wavelengths=wavelengths, R_i_b_res=R_i_b_res),
                                                                                k_uB=kuB,
                                                                                dk_uB_div_dp=0,
                                                                                zB=parameters["zB"]
                                                )
                                            )
        return df_div_df_5

    def df_div_dzB():
        df_div_d_zB = air_water.dbelow2above_div_dp(R_rs_water,
                                                    water_alg.drrs_shallow_div_zB(r_rs_deep=rrsd,
                                                                                K_d=Kd,
                                                                                k_uW=kuW,
                                                                                r_rs_b=Rrsb,
                                                                                k_uB=kuB,
                                                                                zB=parameters["zB"]
                                                        )
                                                    )
        return df_div_d_zB
    
    if parameters["fit_surface"].value:
        def df_div_dg_dd():
            df_div_dg_dd  =  rho_L * (sky_radiance.d_LS_div_dg_dd(Ed_d) / Ed)
            return df_div_dg_dd
        
        def df_div_dg_dsr():
            df_div_dg_dsr = rho_L * (sky_radiance.d_LS_div_dg_dsr(Ed_sr) / Ed)
            return df_div_dg_dsr

        def df_div_dg_dsa():
            df_div_dg_dsa = rho_L * (sky_radiance.d_LS_div_dg_dsa(Ed_sa) / Ed)
            return df_div_dg_dsa
    else:
        # A memoized instance of this variable will improve the relative speed gains
        # when fit_surface == False.
        zero = np.zeros_like(wavelengths)

        def df_div_dg_dd():
            return zero
        def df_div_dg_dsr():
            return zero
        def df_div_dg_dsa():
            return zero
        
    partials = \
    {
        "C_0":   df_div_dC_0,
        "C_1":   df_div_dC_1,
        "C_2":   df_div_dC_2,
        "C_3":   df_div_dC_3,
        "C_4":   df_div_dC_4,
        "C_5":   df_div_dC_5,
        "C_Y":   df_div_dC_Y,
        "C_X":   df_div_dC_X,
        "C_Mie": df_div_dC_Mie,
        "d_r":   df_div_dd_r,
        "f_0":   df_div_df_0,
        "f_1":   df_div_df_1,
        "f_2":   df_div_df_2,
        "f_3":   df_div_df_3,
        "f_4":   df_div_df_4,
        "f_5":   df_div_df_5,
        "zB":    df_div_dzB,
        "g_dd":  df_div_dg_dd,  # this function scoped to module even though defined in an "if"
        "g_dsa": df_div_dg_dsa, # this function scoped to module even though defined in an "if"
        "g_dsr": df_div_dg_dsr  # this function scoped to module even though defined in an "if"
    }

    for param in fit_params:
        jacobian.append(partials[param]())

    return np.array(jacobian).T

def invert(parameters: Parameters, 
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
           R_i_b_res = [],
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
           max_nfev=400,
           analytical=True
           ):
    
    ret_params      = parameters.copy()
    fit_params      = []
    fit_bounds      = [[],[]]
    fit_param_names = []

    for param in ret_params.keys():
        if ret_params[param].vary:
            fit_params.append(ret_params[param].value)
            fit_param_names.append(ret_params[param].name)
            fit_bounds[0].append(ret_params[param].min)
            fit_bounds[1].append(ret_params[param].max)


    userfun  = get_fun_shim(get_residuals(forward, Rrs, weights), wavelengths, fit_param_names, ret_params)
    userdfun = get_dfun_shim(dfun, wavelengths, fit_param_names, ret_params)

    if analytical:
        analytical_fit = least_squares(userfun, fit_params, bounds=fit_bounds, jac=userdfun,
                                    kwargs={
                                        "a_res": a_res,
                                        "bb_res": bb_res,
                                        "a_i_spec_res": a_i_spec_res,
                                        "a_w_res": a_w_res,
                                        "a_Y_N_res": a_Y_N_res,
                                        "a_NAP_N_res": a_NAP_N_res,
                                        "b_phy_norm_res": b_phy_norm_res,
                                        "bb_w_res": bb_w_res,
                                        "b_X_norm_res": b_X_norm_res,
                                        "b_Mie_norm_res": b_Mie_norm_res,
                                        "R_i_b_res": R_i_b_res,
                                        "da_w_div_dT_res": da_w_div_dT_res,
                                        "E0_res": E0_res,
                                        "a_oz_res": a_oz_res,
                                        "a_ox_res": a_ox_res,
                                        "a_wv_res": a_wv_res,
                                        "Ed_d_res": Ed_d_res,
                                        "Ed_sa_res": Ed_sa_res,
                                        "Ed_sr_res": Ed_sr_res,
                                        "Ed_res": Ed_res,
                                        "Ed_s_res": Ed_s_res,
                                        "n2_res": n2_res,
                                        "Ls_Ed": Ls_Ed,
                                    },
                                    max_nfev=max_nfev)
        return analytical_fit, ret_params
    else:
        numerical_fit = least_squares(userfun, fit_params, bounds=fit_bounds,
                                    kwargs={
                                        "a_res": a_res,
                                        "bb_res": bb_res,
                                        "a_i_spec_res": a_i_spec_res,
                                        "a_w_res": a_w_res,
                                        "a_Y_N_res": a_Y_N_res,
                                        "a_NAP_N_res": a_NAP_N_res,
                                        "b_phy_norm_res": b_phy_norm_res,
                                        "b_bw_res": bb_w_res,
                                        "b_X_norm_res": b_X_norm_res,
                                        "b_Mie_norm_res": b_Mie_norm_res,
                                        "R_i_b_res": R_i_b_res,
                                        "da_w_div_dT_res": da_w_div_dT_res,
                                        "E0_res": E0_res,
                                        "a_oz_res": a_oz_res,
                                        "a_ox_res": a_ox_res,
                                        "a_wv_res": a_wv_res,
                                        "Ed_d_res": Ed_d_res,
                                        "Ed_sa_res": Ed_sa_res,
                                        "Ed_sr_res": Ed_sr_res,
                                        "Ed_res": Ed_res,
                                        "Ed_s_res": Ed_s_res,
                                        "n2_res": n2_res,
                                        "Ls_Ed": Ls_Ed,
                                    },
                                    max_nfev=max_nfev)
        return numerical_fit, ret_params

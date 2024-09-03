import numpy as np
from lmfit import minimize, Parameters
from ..water import absorption, backscattering, attenuation, bottom_reflectance
from ..water import albert_mobley as water_alg
from .. atmosphere import sky_radiance, downwelling_irradiance, adjacency_effect
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
           R_bg=[],
           b_ray=[],
           method="least-squares", 
           max_nfev=400
           ):
    """
    Function to inversely fit a modeled spectrum to a measurement spectrum.
    
    :param params: lmfit Parameters object containing all Parameter objects that are required to specify the model
    :param Rrs: Remote sensing reflectance spectrum [sr-1]
    :param wavelengths: wavelengths of Rrs bands [nm]
    :param weights: spectral weighing coefficients
    :param a_i_spec_res: optional, specific absorption coefficients of phytoplankton types resampled to sensor's band settings. Will be computed within function if not provided.
    :param a_w_res: optional, absorption of pure water resampled to sensor's band settings. Will be computed within function if not provided.
    :param a_Y_N_res: optional, normalized absorption coefficients of CDOM resampled to sensor's band settings. Will be computed within function if not provided.
    :param a_NAP_N_res: optional, normalized absorption coefficients of NAP resampled to sensor's band settings. Will be computed within function if not provided.
    :param b_phy_norm_res: optional, preresampling b_phy_norm saves a lot of time during inversion. Will be computed within function if not provided.
    :param bb_w_res: optional, precomputing bb_w bb_w saves a lot of time during inversion. Will be computed within function if not provided.
    :param b_X_norm_res: optional, precomputing b_bX_norm before inversion saves a lot of time. Will be computed within function if not provided.
    :param b_Mie_norm_res: optional, if n and lambda_S are not fit params, the last part of the equation can be precomputed to save time. Will be computed within function if not provided.
    :param R_i_b_res: optional, preresampling R_i_b before inversion saves a lot of time. Will be computed within function if not provided.
    :param da_W_div_dT_res: optional, temperature gradient of pure water absorption resampled  to sensor's band settings. Will be computed within function if not provided.
    :param E0_res: optional, precomputing E0 saves a lot of time. Will be computed within function if not provided.
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
                             R_i_b_res, 
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
                             Ls_Ed,
                             R_bg,
                             b_ray), 
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
                             R_i_b_res, 
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
                             Ls_Ed,
                             R_bg,
                             b_ray), 
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
        Ls_Ed=[],
        R_bg=[],
        b_ray=[]):
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
        bb_w_res (list, optional): _description_. Defaults to [].
        b_X_norm_res (list, optional): _description_. Defaults to [].
        b_Mie_norm_res (list, optional): _description_. Defaults to [].
        R_i_b_res (list, optional): _description_. Defaults to [].
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
        R_bg (list, optional): _description_. Defaults to [].

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

    Rrsb = bottom_reflectance.Rrs_b(parameters["f_0"], parameters["f_1"], parameters["f_2"], parameters["f_3"], parameters["f_4"], parameters["f_5"], B_0=parameters["B_0"], B_1=parameters["B_1"], B_2=parameters["B_2"], B_3=parameters["B_3"], B_4=parameters["B_4"], B_5=parameters["B_5"], wavelengths=wavelengths, R_i_b_res=R_i_b_res)

    ob = attenuation.omega_b(a_sim, bb_sim) #ob is omega_b. Shortened to distinguish between new var and function params.

    frs = water_alg.f_rs(omega_b=ob, cos_t_sun_p=ctsp, cos_t_view_p=ctvp)

    rrsd = water_alg.Rrs_deep(f_rs=frs, omega_b=ob)

    Kd =  attenuation.Kd(a=a_sim, bb=bb_sim, cos_t_sun_p=ctsp, kappa_0=parameters["kappa_0"])

    kuW = attenuation.ku_w(a=a_sim, bb=bb_sim, omega_b=ob, cos_t_sun_p=ctsp, cos_t_view_p=ctvp)

    kuB = attenuation.ku_b(a=a_sim, bb=bb_sim, omega_b=ob, cos_t_sun_p=ctsp, cos_t_view_p=ctvp)

    # NEW: ADJACENCY EFFECT !!!
    Rrs_adj = adjacency_effect.Rrs_adjacency(C_adj=parameters["C_adj"],
                                               wavelengths = wavelengths,
                                               lambda_r = parameters["lambda_r"],
                                               b_r_spec = parameters["b_r_spec"],
                                               n_r = parameters["n_r"],
                                               R_bg=R_bg,
                                               b_ray=b_ray)

    Rrs_sim = air_water.below2above(water_alg.rrs_shallow(Rrs_deep=rrsd, K_d=Kd, k_uW=kuW, zB=parameters["zB"], Rrs_b=Rrsb, k_uB=kuB)) + Rrs_adj# zeta & gamma



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

        return Rrs_sim + Rrs_surface + parameters["offset"]

    else:
        return Rrs_sim + parameters["offset"]


def forward_glint(parameters,
        wavelengths,
        E0_res=[],
        a_oz_res=[],
        a_ox_res=[],
        a_wv_res=[],
        Ed_d_res=[],
        Ed_sa_res=[],
        Ed_sr_res=[],
        Ed__res=[],
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

    if len(Ed__res) == 0:
        Ed = downwelling_irradiance.Ed(Ed_d, Ed_s, parameters["f_dd"], parameters["f_ds"])
    else:
        Ed = Ed__res

    L_s = sky_radiance.L_s(parameters["f_dd"], parameters["g_dd"], Ed_d, parameters["f_ds"], parameters["g_dsr"], Ed_sr, parameters["g_dsa"], Ed_sa)

    Rrs_surface = surface.Rrs_surf(L_s, Ed, rho_L, parameters["d_r"])

    Rrs_surface += air_water.fresnel(parameters['theta_view'], n2=n2) * Ls_Ed

    return Rrs_surface


def forward_adjacency(parameters,
                      wavelengths,
                      R_bg=[], 
                      b_ray=[]):
    """_summary_

    Args:
        parameters (_type_): _description_
        wavelengths (_type_): _description_
        R_bg (list, optional): _description_. Defaults to [].
        b_ray (list, optional): _description_. Defaults to [].

    Returns:
        _type_: _description_
    """
    Rrs_adjacency = adjacency_effect.Rrs_adjacency(C_adj=parameters["C_adj"], wavelengths=wavelengths, lambda_r=parameters["lambda_r"], b_r_spec=parameters["b_r_spec"], n_r=parameters["n_r"], R_bg=R_bg, b_ray=b_ray)
    
    return Rrs_adjacency


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
             R_bg=[],
             b_ray=[]
             ):
    """
    Error function around model to be minimized by changing fit parameters.
    
    :param params: lmfit Parameters object containing all Parameter objects that are required to specify the model
    :param Rrs: Remote sensing reflectance spectrum [sr-1]
    :param wavelengths: wavelengths of Rrs bands [nm]
    :param weights: spectral weighing coefficients
    :param a_i_spec_res: optional, specific absorption coefficients of phytoplankton types resampled to sensor's band settings. Will be computed within function if not provided.
    :param a_w_res: optional, absorption of pure water resampled to sensor's band settings. Will be computed within function if not provided.
    :param a_Y_N_res: optional, normalized absorption coefficients of CDOM resampled to sensor's band settings. Will be computed within function if not provided.
    :param a_NAP_N_res: optional, normalized absorption coefficients of NAP resampled to sensor's band settings. Will be computed within function if not provided.
    :param b_phy_norm_res: optional, preresampling b_phy_norm saves a lot of time during inversion. Will be computed within function if not provided.
    :param bb_w_res: optional, precomputing bb_w bb_w saves a lot of time during inversion. Will be computed within function if not provided.
    :param b_X_norm_res: optional, precomputing b_bX_norm before inversion saves a lot of time. Will be computed within function if not provided.
    :param b_Mie_norm_res: optional, if n and lambda_S are not fit params, the last part of the equation can be precomputed to save time. Will be computed within function if not provided.
    :param R_i_b_res: optional, preresampling R_i_b before inversion saves a lot of time. Will be computed within function if not provided.
    :param da_w_div_dT_res: optional, temperature gradient of pure water absorption resampled  to sensor's band settings. Will be computed within function if not provided.
    :param E0_res: optional, precomputing E0 saves a lot of time. Will be computed within function if not provided.
    :param a_oz_res: optional, precomputing a_oz saves a lot of time. Will be computed within function if not provided.
    :param a_ox_res: optional, precomputing a_ox saves a lot of time. Will be computed within function if not provided.
    :param a_wv_res: optional, precomputing a_wv saves a lot of time. Will be computed within function if not provided.
    :param Ed_d_res: optional, preresampling E_dd before inversion saves a lot of time. Will be computed within function if not provided.
    :param Ed_sa_res: optional, preresampling E_dsa before inversion saves a lot of time. Will be computed within function if not provided.
    :param Ed_sr_res: optional, preresampling E_dsr before inversion saves a lot of time. Will be computed within function if not provided.
    :param Ed_res: optional, preresampling E_d before inversion saves a lot of time. Will be computed within function if not provided.
    :return: weighted difference between measured and simulated Rrs
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
                       R_i_b_res=R_i_b_res,
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
                       Ls_Ed=Ls_Ed,
                       R_bg=R_bg,
                       b_ray=b_ray)
    
    return utils.compute_residual(Rrs, Rrs_sim, method=parameters['error_method'], weights=weights)
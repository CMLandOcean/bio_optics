import numpy as np
from lmfit import minimize, Parameters
from .. water import fluorescence, absorption, backscattering, attenuation, scattering, lee
from .. helper import resampling, utils 
from .. surface import surface, air_water
from .. atmosphere import sky_radiance, downwelling_irradiance

def forward(parameters,
            wavelengths,
            a_d_lambda_0_res=None,
            c_d_lambda_0_res=None,
            omega_d_lambda_0_res=None,
            a_res=[],
            a_d_res=[],
            a_md_spec_res=[],
            a_bd_spec_res=[],
            a_w_res=[],
            a_i_spec_res=[],
            a_phy_res=[],
            a_Y_N_res=[],
            b_b_res=[],
            b_bd_res=[],
            b_bp_res=[],
            b_bphy_res=[],
            b_bw_res=[],
            b_d_res=[],
            b_i_spec_res=[],
            c_d_res=[],
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
            n2_res=[],
            Ls_Ed=[]):
    """
    Forward function of the HEREON model described in [1]

    [1] Bi et al. (2023): Bio-geo-optical modelling of natural waters [10.3389/fmars.2023.11963529]

    Args:
        parameters (_type_): _description_
        wavelengths (_type_): _description_
        a_d_lambda_0_res (_type_, optional): _description_. Defaults to None.
        c_d_lambda_0_res (_type_, optional): _description_. Defaults to None.
        omega_d_lambda_0_res (_type_, optional): _description_. Defaults to None.
        a_res (list, optional): _description_. Defaults to [].
        a_d_res (list, optional): _description_. Defaults to [].
        a_md_spec_res (list, optional): _description_. Defaults to [].
        a_bd_spec_res (list, optional): _description_. Defaults to [].
        a_w_res (list, optional): _description_. Defaults to [].
        a_i_spec_res (list, optional): _description_. Defaults to [].
        a_phy_res (list, optional): _description_. Defaults to [].
        a_Y_N_res (list, optional): _description_. Defaults to [].
        b_b_res (list, optional): _description_. Defaults to [].
        b_bd_res (list, optional): _description_. Defaults to [].
        b_bp_res (list, optional): _description_. Defaults to [].
        b_bphy_res (list, optional): _description_. Defaults to [].
        b_bw_res (list, optional): _description_. Defaults to [].
        b_d_res (list, optional): _description_. Defaults to [].
        b_i_spec_res (list, optional): _description_. Defaults to [].
        c_d_res (list, optional): _description_. Defaults to [].
        da_W_div_dT_res (list, optional): _description_. Defaults to [].

    Returns:
        _type_: _description_
    """    
    C_phy = np.sum([parameters["C_0"], parameters["C_1"], parameters["C_2"], parameters["C_3"], parameters["C_4"], parameters["C_5"]])

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

    # it makes sense to precompute some coefficients outside of a() and b_b() because they are used in both functions
    if len(b_bw_res)==0:
        b_bw_res = resampling.resample_b_bw(wavelengths=wavelengths)

    if len(a_d_res)==0:
        a_d_res = absorption.a_d(wavelengths=wavelengths, 
                                 C_phy=C_phy, 
                                 C_ism=parameters["C_ism"], 
                                 A_md=parameters["A_md"], 
                                 A_bd=parameters["A_bd"], 
                                 S_md=parameters["S_md"], 
                                 S_bd=parameters["S_bd"], 
                                 C_md=parameters["C_md"], 
                                 C_bd=parameters["C_bd"], 
                                 lambda_0_md=parameters["lambda_0_md"], 
                                 lambda_0_bd=parameters["lambda_0_bd"], 
                                 a_bd_spec_res=a_bd_spec_res, 
                                 a_md_spec_res=a_md_spec_res)
    a_d_lambda_0_res = np.interp(parameters["lambda_0_cd"].value, wavelengths, a_d_res) if parameters["interpolate"].value else a_d_res[utils.find_closest(wavelengths, parameters["lambda_0_cd"])[1]]

    if len(c_d_res)==0:
        c_d_res = attenuation.c_d(wavelengths=wavelengths, 
                                  C_phy=C_phy, 
                                  C_ism=parameters["C_ism"], 
                                  A_md=parameters["A_md"], 
                                  A_bd=parameters["A_bd"], 
                                  S_md=parameters["S_md"], 
                                  S_bd=parameters["S_bd"], 
                                  C_md=parameters["C_md"], 
                                  C_bd=parameters["C_bd"], 
                                  lambda_0_cd=parameters["lambda_0_cd"], 
                                  lambda_0_md=parameters["lambda_0_md"], 
                                  lambda_0_bd=parameters["lambda_0_bd"], 
                                  gamma_d=parameters["gamma_d"], 
                                  x0=parameters["x0"], 
                                  x1=parameters["x1"], 
                                  x2=parameters["x2"], 
                                  c_d_lambda_0_res=c_d_lambda_0_res,
                                  omega_d_lambda_0_res=omega_d_lambda_0_res,
                                  a_d_lambda_0_res = a_d_lambda_0_res,
                                  a_md_spec_res=a_md_spec_res,
                                  a_bd_spec_res=a_bd_spec_res)

    if len(b_d_res)==0:
        b_d_res = scattering.b(a=a_d_res, c=c_d_res)
    
    if len(b_bd_res)==0:
        b_bd_res = backscattering.b_bd(b_d=b_d_res, b_ratio_d=parameters["b_ratio_d"])

    if len(b_bphy_res)==0:
        b_bphy_res = backscattering.b_bphy_hereon(wavelengths=wavelengths, 
                                                  C_0=parameters["C_0"], 
                                                  C_1=parameters["C_1"], 
                                                  C_2=parameters["C_2"], 
                                                  C_3=parameters["C_3"], 
                                                  C_4=parameters["C_4"], 
                                                  C_5=parameters["C_5"], 
                                                  b_i_spec_res=b_i_spec_res)
    if len(b_bp_res)==0:
        b_bp_res = b_bd_res + b_bphy_res

    if len(a_res) == 0:
        # C_phy could be used as an argument so it does not need to be recomputed inside functions
        a_res = absorption.a_total(wavelengths=wavelengths, 
                                   C_0=parameters["C_0"], 
                                   C_1=parameters["C_1"], 
                                   C_2=parameters["C_2"], 
                                   C_3=parameters["C_3"], 
                                   C_4=parameters["C_4"], 
                                   C_5=parameters["C_5"], 
                                   C_ism=parameters["C_ism"], 
                                   C_Y=parameters["C_Y"], 
                                   A_md=parameters["A_md"], 
                                   A_bd=parameters["A_bd"], 
                                   S_md=parameters["S_md"], 
                                   S_bd=parameters["S_bd"], 
                                   S_cdom=parameters["S_cdom"], 
                                   C_md=parameters["C_md"], 
                                   C_bd=parameters["C_bd"], 
                                   K=parameters["K"], 
                                   lambda_0_cdom=parameters["lambda_0_cdom"], 
                                   lambda_0_md=parameters["lambda_0_md"], 
                                   lambda_0_bd=parameters["lambda_0_bd"], 
                                   lambda_0_C_phy=parameters["lambda_0_C_phy"].value, 
                                   A=parameters["A"], 
                                   E0=parameters["E0"], 
                                   E1=parameters["E1"], 
                                   interpolate=parameters["interpolate"], 
                                   T_W=parameters["T_W"], 
                                   T_W_0=parameters["T_W_0"], 
                                   a_d_res=a_d_res,
                                   a_md_spec_res=a_md_spec_res,
                                   a_bd_spec_res=a_bd_spec_res,
                                   a_i_spec_res=a_i_spec_res,
                                   a_phy_res=a_phy_res,
                                   a_Y_N_res=a_Y_N_res,
                                   a_w_res=a_w_res,
                                   da_W_div_dT_res=da_W_div_dT_res)

    if len(b_b_res)==0:
        # C_phy could be used as an argument so it does not need to be recomputed inside functions
        b_b_res = backscattering.b_b_total(wavelengths=wavelengths,
                                           C_0=parameters["C_0"], 
                                           C_1=parameters["C_1"], 
                                           C_2=parameters["C_2"], 
                                           C_3=parameters["C_3"], 
                                           C_4=parameters["C_4"], 
                                           C_5=parameters["C_5"], 
                                           C_ism=parameters["C_ism"], 
                                           b_ratio_C_0=parameters["b_ratio_C_0"], 
                                           b_ratio_C_1=parameters["b_ratio_C_1"], 
                                           b_ratio_C_2=parameters["b_ratio_C_2"], 
                                           b_ratio_C_3=parameters["b_ratio_C_3"], 
                                           b_ratio_C_4=parameters["b_ratio_C_4"], 
                                           b_ratio_C_5=parameters["b_ratio_C_5"], 
                                           b_ratio_d=parameters["b_ratio_d"], 
                                           fresh=parameters["fresh"],
                                           A_md=parameters["A_md"],
                                           A_bd=parameters["A_bd"],
                                           S_md=parameters["S_md"],
                                           S_bd=parameters["S_bd"],
                                           C_md=parameters["C_md"],
                                           C_bd=parameters["C_bd"],
                                           lambda_0_md=parameters["lambda_0_md"], 
                                           lambda_0_bd=parameters["lambda_0_bd"], 
                                           lambda_0_cd=parameters["lambda_0_cd"], 
                                           gamma_d=parameters["gamma_d"], 
                                           x0=parameters["x0"], 
                                           x1=parameters["x1"], 
                                           x2=parameters["x2"], 
                                           c_d_lambda_0_res=c_d_lambda_0_res, 
                                           a_d_lambda_0_res=a_d_lambda_0_res,
                                           omega_d_lambda_0_res=omega_d_lambda_0_res, 
                                           interpolate=parameters["interpolate"],
                                           a_d_res = a_d_res,
                                           a_md_spec_res=a_md_spec_res,
                                           a_bd_spec_res=a_bd_spec_res,
                                           b_d_res=b_d_res,
                                           b_bd_res=b_bd_res,
                                           b_bp_res=b_bp_res,
                                           b_bw_res=b_bw_res,
                                           b_i_spec_res=b_i_spec_res,
                                           c_d_res=c_d_res)
    
    R_rs_water = lee.R_rs_deep(a=a_res, 
                               b_b=b_b_res, 
                               b_bp=b_bp_res, 
                               b_bw=b_bw_res,
                               Gw0=parameters["Gw0"],
                               Gw1=parameters["Gw1"],
                               Gp0=parameters["Gp0"],
                               Gp1=parameters["Gp1"]) + fluorescence.R_rs_fl(wavelengths=wavelengths,
                                                                             L_fl_lambda0=parameters['L_fl_lambda0'])
    
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
                E_dsr = downwelling_irradiance.E_dsr(wavelengths, parameters["theta_sun"], parameters["P"], parameters["AM"], parameters["RH"], parameters["H_oz"], parameters["WV"], parameters["alpha"], parameters["beta"], E_0_res, a_oz_res, a_ox_res, a_wv_res, E_dsr_res)
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

            L_s = sky_radiance.L_s(parameters["f_dd"], parameters["g_dd"], E_dd, parameters["f_ds"], parameters["g_dsr"], E_dsr, parameters["g_dsa"], E_dsa)

            R_rs_surface = surface.R_rs_surf(L_s, E_d, rho_L, parameters["d_r"])

            R_rs_surface += air_water.fresnel(parameters['theta_view'], n2=n2) * Ls_Ed

            R_rs_sim = R_rs_water + R_rs_surface + parameters["offset"]
            return R_rs_sim
        
    else:
        R_rs_sim = R_rs_water + parameters["offset"]
        return R_rs_sim


def func2opt(parameters, 
             R_rs,
             wavelengths,
             weights=[],
             a_res=[],
             a_d_res=[],
             a_md_spec_res=[],
             a_bd_spec_res=[],
             a_w_res=[],
             a_i_spec_res=[],
             a_phy_res=[],
             a_Y_N_res=[],
             b_b_res=[],
             b_bd_res=[],
             b_bp_res=[],
             b_bphy_res=[],
             b_bw_res=[],
             b_d_res=[],
             b_i_spec_res=[],
             c_d_res=[],
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
             n2_res=[],
             Ls_Ed=[],
             omega_d_lambda_0_res=None,
             a_d_lambda_0_res=None,
             c_d_lambda_0_res=None
             ):
    """
    Function to optimize during inversion.

    Args:
        parameters (_type_): _description_
        R_rs (_type_): _description_
        wavelengths (_type_): _description_
        weights (list, optional): _description_. Defaults to [].
        a_res (list, optional): _description_. Defaults to [].
        a_d_res (list, optional): _description_. Defaults to [].
        a_md_spec_res (list, optional): _description_. Defaults to [].
        a_bd_spec_res (list, optional): _description_. Defaults to [].
        a_w_res (list, optional): _description_. Defaults to [].
        a_i_spec_res (list, optional): _description_. Defaults to [].
        a_phy_res (list, optional): _description_. Defaults to [].
        a_Y_N_res (list, optional): _description_. Defaults to [].
        b_b_res (list, optional): _description_. Defaults to [].
        b_bd_res (list, optional): _description_. Defaults to [].
        b_bp_res (list, optional): _description_. Defaults to [].
        b_bphy_res (list, optional): _description_. Defaults to [].
        b_bw_res (list, optional): _description_. Defaults to [].
        b_d_res (list, optional): _description_. Defaults to [].
        b_i_spec_res (list, optional): _description_. Defaults to [].
        c_d_res (list, optional): _description_. Defaults to [].
        da_W_div_dT_res (list, optional): _description_. Defaults to [].
        omega_d_lambda_0_res (_type_, optional): _description_. Defaults to None.
        a_d_lambda_0_res (_type_, optional): _description_. Defaults to None.
        c_d_lambda_0_res (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """

    if len(weights)==0:
        weights = np.ones(len(wavelengths))

    R_rs_sim = forward(parameters=parameters,
                       wavelengths=wavelengths,
                       a_res=a_res,
                       a_d_res=a_d_res,
                       a_md_spec_res=a_md_spec_res,
                       a_bd_spec_res=a_bd_spec_res,
                       a_w_res=a_w_res,
                       a_i_spec_res=a_i_spec_res,
                       a_phy_res=a_phy_res,
                       a_Y_N_res=a_Y_N_res,
                       b_b_res=b_b_res,
                       b_bd_res=b_bd_res,
                       b_bp_res=b_bp_res,
                       b_bphy_res=b_bphy_res,
                       b_bw_res=b_bw_res,
                       b_d_res=b_d_res,
                       b_i_spec_res=b_i_spec_res,
                       c_d_res=c_d_res,
                       da_W_div_dT_res=da_W_div_dT_res,
                       omega_d_lambda_0_res=omega_d_lambda_0_res,
                       a_d_lambda_0_res=a_d_lambda_0_res,
                       c_d_lambda_0_res=c_d_lambda_0_res,
                       E_0_res=E_0_res,
                       a_oz_res=a_oz_res,
                       a_ox_res=a_ox_res,
                       a_wv_res=a_wv_res,
                       E_dd_res=E_dd_res,
                       E_dsa_res=E_dsa_res,
                       E_dsr_res=E_dsr_res,
                       E_d_res=E_d_res,
                       E_ds_res=E_ds_res,
                       n2_res=n2_res,
                       Ls_Ed=Ls_Ed)
    
    return utils.compute_residual(R_rs, R_rs_sim, method=parameters['error_method'], weights=weights)


def invert(params,
           R_rs,
           wavelengths,
           weights,
           a_res=[],
           a_d_res=[],
           a_md_spec_res=[],
           a_bd_spec_res=[],
           a_w_res=[],
           a_i_spec_res=[],
           a_phy_res=[],
           a_Y_N_res=[],
           b_b_res=[],
           b_bd_res=[],
           b_bp_res=[],
           b_bphy_res=[],
           b_bw_res=[],
           b_d_res=[],
           b_i_spec_res=[],
           c_d_res=[],
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
           n2_res=[],
           Ls_Ed=[],
           omega_d_lambda_0_res=None,
           a_d_lambda_0_res=None,
           c_d_lambda_0_res=None,
           method="least-squares", 
           max_nfev=400):
    """
    Inversion of HEREON model described in [1].

    [1] Bi et al. (2023): Bio-geo-optical modelling of natural waters [10.3389/fmars.2023.11963529]

    Args:
        params (_type_): _description_
        R_rs (_type_): _description_
        wavelengths (_type_): _description_
        weights (_type_): _description_
        a_res (list, optional): _description_. Defaults to [].
        a_d_res (list, optional): _description_. Defaults to [].
        a_md_spec_res (list, optional): _description_. Defaults to [].
        a_bd_spec_res (list, optional): _description_. Defaults to [].
        a_w_res (list, optional): _description_. Defaults to [].
        a_i_spec_res (list, optional): _description_. Defaults to [].
        a_phy_res (list, optional): _description_. Defaults to [].
        a_Y_N_res (list, optional): _description_. Defaults to [].
        b_b_res (list, optional): _description_. Defaults to [].
        b_bd_res (list, optional): _description_. Defaults to [].
        b_bp_res (list, optional): _description_. Defaults to [].
        b_bphy_res (list, optional): _description_. Defaults to [].
        b_bw_res (list, optional): _description_. Defaults to [].
        b_d_res (list, optional): _description_. Defaults to [].
        b_i_spec_res (list, optional): _description_. Defaults to [].
        c_d_res (list, optional): _description_. Defaults to [].
        da_W_div_dT_res (list, optional): _description_. Defaults to [].
        omega_d_lambda_0_res (_type_, optional): _description_. Defaults to None.
        a_d_lambda_0_res (_type_, optional): _description_. Defaults to None.
        c_d_lambda_0_res (_type_, optional): _description_. Defaults to None.
        method (str, optional): _description_. Defaults to "least-squares".
        max_nfev (int, optional): _description_. Defaults to 400.

    Returns:
        _type_: _description_
    """

    if len(weights)==0:
        weights = np.ones(len(R_rs))

    if params['fit_surface'].value:
        res = minimize(func2opt, 
                        params, 
                        args=(R_rs, 
                                wavelengths, 
                                weights,
                                a_res,
                                a_d_res,
                                a_md_spec_res,
                                a_bd_spec_res,
                                a_w_res,
                                a_i_spec_res,
                                a_phy_res,
                                a_Y_N_res,
                                b_b_res,
                                b_bd_res,
                                b_bp_res,
                                b_bphy_res,
                                b_bw_res,
                                b_d_res,
                                b_i_spec_res,
                                c_d_res,
                                da_W_div_dT_res,
                                omega_d_lambda_0_res,
                                a_d_lambda_0_res,
                                c_d_lambda_0_res,
                                E_0_res,
                                a_oz_res,
                                a_ox_res,
                                a_wv_res,
                                E_dd_res,
                                E_dsa_res,
                                E_dsr_res,
                                E_d_res,
                                E_ds_res,
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
                        args=(R_rs, 
                                wavelengths, 
                                weights,
                                a_res,
                                a_d_res,
                                a_md_spec_res,
                                a_bd_spec_res,
                                a_w_res,
                                a_i_spec_res,
                                a_phy_res,
                                a_Y_N_res,
                                b_b_res,
                                b_bd_res,
                                b_bp_res,
                                b_bphy_res,
                                b_bw_res,
                                b_d_res,
                                b_i_spec_res,
                                c_d_res,
                                da_W_div_dT_res,
                                omega_d_lambda_0_res,
                                a_d_lambda_0_res,
                                c_d_lambda_0_res,
                                E_0_res,
                                a_oz_res,
                                a_ox_res,
                                a_wv_res,
                                E_dd_res,
                                E_dsa_res,
                                E_dsr_res,
                                E_d_res,
                                E_ds_res,
                                n2_res,
                                Ls_Ed), 
                        method=method, 
                        max_nfev=max_nfev) 

    return res
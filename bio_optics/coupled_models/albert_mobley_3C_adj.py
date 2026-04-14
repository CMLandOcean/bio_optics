import numpy as np
from ..water import absorption, backscattering, attenuation, bottom_reflectance
from ..reflectance import albert_mobley as water_alg
from ..atmosphere.adjacency import reflectance as adjacency_effect
from ..surface import reflectance as surface, air_water


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

    Rrsb = bottom_reflectance.Rrs_b(parameters["f_0"], parameters["f_1"], parameters["f_2"], parameters["f_3"], parameters["f_4"], parameters["f_5"], B_0=parameters["B_0"], B_1=parameters["B_1"], B_2=parameters["B_2"], B_3=parameters["B_3"], B_4=parameters["B_4"], B_5=parameters["B_5"], wavelengths=wavelengths, R_b_i_res=R_b_i_res)

    ob = attenuation.omega_b(a_sim, bb_sim) #ob is omega_b. Shortened to distinguish between new var and function params.

    frs = water_alg.f_rs(omega_b=ob, cos_t_sun_p=ctsp, cos_t_view_p=ctvp)

    rrsd = water_alg.rrs_deep(f_rs=frs, omega_b=ob)

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

    Rrs_water = air_water.below2above(water_alg.rrs_shallow(rrs_deep=rrsd, Kd=Kd, ku_w=kuW, zB=parameters["zB"], Rrs_b=Rrsb, ku_b=kuB)) + Rrs_adj

    Rrs_surf = surface.forward(parameters=parameters, wavelengths=wavelengths,
                               E0_res=E0_res, a_oz_res=a_oz_res, a_ox_res=a_ox_res,
                               a_wv_res=a_wv_res, Ed_d_res=Ed_d_res, Ed_sa_res=Ed_sa_res,
                               Ed_sr_res=Ed_sr_res, Ed_s_res=Ed_s_res, Ed_res=Ed_res,
                               n2_res=n2_res, Ls_Ed=Ls_Ed)

    return Rrs_water + Rrs_surf + parameters["offset"]


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

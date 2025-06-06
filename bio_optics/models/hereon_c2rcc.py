import numpy as np
from lmfit import minimize, Parameters
from .. water import fluorescence, absorption, backscattering, attenuation, scattering, lee
from .. helper import resampling, utils
from .. surface import surface, air_water
from .. atmosphere import sky_radiance, downwelling_irradiance


def forward_c2rcc_IOPs(parameters,
            wavelengths,
            a_d_lambda_0_res=None,
            c_d_lambda_0_res=None,
            omega_d_lambda_0_res=None,
            a_res=[],
            # a_d_res=[],
            a_md_res=[],
            a_bd_res=[],
            a_md_spec_res=[],
            a_bd_spec_res=[],
            a_w_res=[],
            a_i_spec_res=[],
            a_phy_res=[],
            a_Y_N_res=[],
            bb_res=[],
            bb_bd_res=[],
            bb_md_res=[],
            bb_p_res=[],
            bb_phy_res=[],
            b_md_res=[],
            b_bd_res=[],
            b_bw_res=[],
            # b_d_res=[],
            b_i_spec_res=[],
            # c_d_res=[],
            c_md_res=[],
            c_bd_res=[],
            h_C_res=[],
            h_C_phycocyanin_res=[],
            h_C_phycoerythrin_res=[],
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

    HERE: IOPs as defined for c2rcc in-water output (apig, adet, agelb, bpart, bwit at 443nm) are calculated and returned

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

    wavelengths = wavelengths
    C_0 = parameters["C_0"]
    C_1 = parameters["C_1"]
    C_2 = parameters["C_2"]
    C_3 = parameters["C_3"]
    C_4 = parameters["C_4"]
    C_5 = parameters["C_5"]
    C_6 = parameters["C_6"]
    C_7 = parameters["C_7"]
    C_ism = parameters["C_ism"]
    C_Y = parameters["C_Y"]
    A_md = parameters["A_md"]
    A_bd = parameters["A_bd"]
    S_md = parameters["S_md"]
    S_bd = parameters["S_bd"]
    S_cdom = parameters["S_cdom"]
    C_md = parameters["C_md"]
    C_bd = parameters["C_bd"]
    K = parameters["K"]
    lambda_0_cdom = parameters["lambda_0_cdom"]
    lambda_0_md = parameters["lambda_0_md"]
    lambda_0_bd = parameters["lambda_0_bd"]
    lambda_0_phy = parameters["lambda_0_phy"].value
    A = parameters["A"]
    E0 = parameters["E0"]
    E1 = parameters["E1"]
    interpolate = parameters["interpolate"]

    b_ratio_C_0 = parameters["b_ratio_C_0"]
    b_ratio_C_1 = parameters["b_ratio_C_1"]
    b_ratio_C_2 = parameters["b_ratio_C_2"]
    b_ratio_C_3 = parameters["b_ratio_C_3"]
    b_ratio_C_4 = parameters["b_ratio_C_4"]
    b_ratio_C_5 = parameters["b_ratio_C_5"]
    b_ratio_C_6 = parameters["b_ratio_C_6"]
    b_ratio_C_7 = parameters["b_ratio_C_7"]
    b_ratio_md = parameters["b_ratio_md"]
    b_ratio_bd = parameters["b_ratio_bd"]
    lambda_0_c_d = parameters["lambda_0_c_d"]
    fresh = parameters["fresh"]

    ###
    # Absorption components of C2RCC
    ###
    C_phy = np.sum([C_0, C_1, C_2, C_3, C_4, C_5, C_6, C_7])

    if len(a_md_res) == 0:
        a_md_res = absorption.a_md(wavelengths=wavelengths, C_ism=C_ism, A_md=A_md, S_md=S_md, C_md=C_md, lambda_0_md=lambda_0_md,
                        a_md_spec_res=a_md_spec_res)

    if len(a_bd_res) == 0:
        a_bd_res = absorption.a_bd(wavelengths=wavelengths, C_phy=C_phy, A_bd=A_bd, S_bd=S_bd, C_bd=C_bd, lambda_0_bd=lambda_0_bd,
                        a_bd_spec_res=a_bd_spec_res)

    # if len(a_d_res)==0:
    #     a_d_res = a_d(wavelengths=wavelengths, C_phy=C_phy, C_ism=C_ism, A_md=A_md, A_bd=A_bd, S_md=S_md, S_bd=S_bd, C_md=C_md, C_bd=C_bd, lambda_0_md=lambda_0_md, lambda_0_bd=lambda_0_bd, a_md_spec_res=a_md_spec_res, a_bd_spec_res=a_bd_spec_res)

    if len(a_phy_res) == 0:
        a_phy_res = absorption.a_phy(wavelengths=wavelengths, C_0=C_0, C_1=C_1, C_2=C_2, C_3=C_3, C_4=C_4, C_5=C_5, C_6=C_6,
                          C_7=C_7, a_i_spec_res=a_i_spec_res)

    apig = absorption.correct_a_phy(a_phy_res=a_phy_res, wavelengths=wavelengths, C_phy=C_phy, A=A, E0=E0, E1=E1,
                         lambda_0_phy=lambda_0_phy, interpolate=interpolate)
    agelb = absorption.a_Y(C_Y=C_Y, wavelengths=wavelengths, S=S_cdom, lambda_0=lambda_0_cdom, K=K, a_Y_N_res=a_Y_N_res)
    adet = a_md_res + a_bd_res

    ###
    # Backscattering of C2RCC
    ###
    if len(bb_p_res) == 0:
        # compute b_bp, backscattering all particles
        if len(bb_bd_res) == 0:
            # compute b_bd, backscattering biogenic detritus
            if len(bb_md_res) == 0:
                # compute b_md, backscattering mineralogenic detritus

                a_md_lambda_0_res = a_md_res[utils.find_closest(wavelengths, lambda_0_c_d)[1]]

                if len(a_bd_res) == 0:
                    # compute a_bd, absorption biogenic detritus
                    a_bd_res = absorption.a_bd(wavelengths=wavelengths,
                                               C_phy=C_phy,
                                               A_bd=A_bd,
                                               S_bd=S_bd,
                                               C_bd=C_bd,
                                               lambda_0_bd=lambda_0_bd,
                                               a_bd_spec_res=a_bd_spec_res)

                # a_bd_lambda_0_res = np.interp(lambda_0_c_d, wavelengths, a_bd_res) if interpolate else a_bd_res[
                #     utils.find_closest(wavelengths, lambda_0_c_d)[1]]

                if len(c_md_res) == 0:
                    # attenuation mineralogenic detritus
                    c_md_res = attenuation.c_md(wavelengths=wavelengths,
                                                    C_ism=parameters["C_ism"],
                                                    A_md=parameters["A_md"],
                                                    S_md=parameters["S_md"],
                                                    C_md=parameters["C_md"],
                                                    lambda_0_c_d=parameters["lambda_0_c_d"],
                                                    lambda_0_md=parameters["lambda_0_md"],
                                                    gamma_d=parameters["gamma_d"],
                                                    x0=parameters["x0"],
                                                    x1=parameters["x1"],
                                                    x2=parameters["x2"],
                                                    omega_d_lambda_0_res=omega_d_lambda_0_res,
                                                    a_md_lambda_0_res=a_md_lambda_0_res)

                b_md_res = scattering.b(a_md_res, c_md_res)

                if len(c_bd_res) == 0:
                    # attenuation biogenic detritus
                    c_bd_res = attenuation.c_bd(wavelengths=wavelengths,
                                                C_phy=C_phy,
                                                A_bd=parameters["A_bd"],
                                                S_bd=parameters["S_bd"],
                                                C_bd=parameters["C_bd"],
                                                lambda_0_c_d=parameters["lambda_0_c_d"],
                                                lambda_0_bd=parameters["lambda_0_bd"],
                                                gamma_d=parameters["gamma_d"],
                                                x0=parameters["x0"],
                                                x1=parameters["x1"],
                                                x2=parameters["x2"],
                                                omega_d_lambda_0_res=omega_d_lambda_0_res,
                                                a_bd_lambda_0_res=a_d_lambda_0_res)

            b_bd_res = scattering.b(a_bd_res, c_bd_res)
            bb_bp_res = backscattering.b_bd(b_bd_res, b_ratio_d=b_ratio_bd) + backscattering.b_bd(b_md_res, b_ratio_d=b_ratio_md)
        bb_p_res = backscattering.b_bphy_hereon(C_0=C_0,
                                 C_1=C_1,
                                 C_2=C_2,
                                 C_3=C_3,
                                 C_4=C_4,
                                 C_5=C_5,
                                 C_6=C_6,
                                 C_7=C_7,
                                 b_ratio_C_0=b_ratio_C_0,
                                 b_ratio_C_1=b_ratio_C_1,
                                 b_ratio_C_2=b_ratio_C_2,
                                 b_ratio_C_3=b_ratio_C_3,
                                 b_ratio_C_4=b_ratio_C_4,
                                 b_ratio_C_5=b_ratio_C_5,
                                 b_ratio_C_6=b_ratio_C_6,
                                 b_ratio_C_7=b_ratio_C_7,
                                 wavelengths=wavelengths,
                                 b_i_spec_res=b_i_spec_res) + bb_bp_res

    bpart = bb_p_res
    bb_w = backscattering.b_bw(wavelengths=wavelengths, fresh=fresh, b_bw_res=b_bw_res)

    return np.array((apig[0], agelb[0], adet[0], bpart[0]))
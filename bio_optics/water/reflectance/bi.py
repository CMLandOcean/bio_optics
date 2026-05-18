import numpy as np
from .. import fluorescence, absorption, backscattering, attenuation, scattering
from . import lee
from ...helper import resampling, utils
from ...surface import air_water

def forward(parameters,
            wavelengths,
            a_d_lambda_0_res=None,
            c_d_lambda_0_res=None,
            omega_d_lambda_0_res=None,
            a_res=None,
            # a_d_res=None,
            a_md_res=None,
            a_bd_res=None,
            a_md_spec_res=None,
            a_bd_spec_res=None,
            a_w_res=None,
            a_i_spec_res=None,
            a_phy_res=None,
            a_Y_N_res=None,
            bb_res=None,
            bb_bd_res=None,
            bb_md_res=None,
            bb_p_res=None,
            bb_phy_res=None,
            b_md_res=None,
            b_bd_res=None,
            bb_w_res=None,
            # b_d_res=None,
            bb_i_spec_res=None,
            # c_d_res=None,
            c_md_res=None,
            c_bd_res=None,
            h_C_res=None,
            h_C_phycocyanin_res=None,
            h_C_phycoerythrin_res=None,
            da_w_div_dT_res=None,
            E0_res=None,
            a_oz_res=None,
            a_ox_res=None,
            a_wv_res=None,
            Ed_d_res=None,
            Ed_sa_res=None,
            Ed_sr_res=None,
            Ed_res=None,
            Ed_s_res=None,
            n2_res=None,
            Ls_Ed=None):
    """
    Forward function of the HEREON model described in [1]
    a_d and b_d split are into a_md/a_bd and b_md/b_bd, respectively.

    [1] Bi et al. (2023): Bio-geo-optical modelling of natural waters [10.3389/fmars.2023.11963529]

    Args:
        parameters: lmfit Parameters object specifying the model configuration
        wavelengths: wavelengths [nm]
        a_d_lambda_0_res: optional precomputed detrital absorption at the reference wavelength [m-1]
        c_d_lambda_0_res: optional precomputed detrital attenuation at the reference wavelength [m-1]
        omega_d_lambda_0_res: optional precomputed detrital single scattering albedo at the reference wavelength
        a_res: optional precomputed total absorption coefficient [m-1]
        a_md_res: optional precomputed mineral-detrital absorption [m-1]
        a_bd_res: optional precomputed biodetrital absorption [m-1]
        a_md_spec_res: optional precomputed specific mineral-detrital absorption spectra [m2 g-1]
        a_bd_spec_res: optional precomputed specific biodetrital absorption spectra [m2 g-1]
        a_w_res: optional precomputed pure water absorption [m-1]
        a_i_spec_res: optional precomputed specific phytoplankton absorption spectra [m2 mg-1]
        a_phy_res: optional precomputed total phytoplankton absorption [m-1]
        a_Y_N_res: optional precomputed normalised CDOM absorption
        bb_res: optional precomputed total backscattering coefficient [m-1]
        bb_bd_res: optional precomputed biodetrital backscattering [m-1]
        bb_md_res: optional precomputed mineral-detrital backscattering [m-1]
        bb_p_res: optional precomputed particulate backscattering [m-1]
        bb_phy_res: optional precomputed phytoplankton backscattering [m-1]
        b_md_res: optional precomputed mineral-detrital scattering [m-1]
        b_bd_res: optional precomputed biodetrital scattering [m-1]
        bb_w_res: optional precomputed water backscattering [m-1]
        bb_i_spec_res: optional precomputed specific backscattering spectra [m2 g-1]
        c_md_res: optional precomputed mineral-detrital attenuation [m-1]
        c_bd_res: optional precomputed biodetrital attenuation [m-1]
        h_C_res: optional precomputed phytoplankton fluorescence spectrum
        h_C_phycocyanin_res: optional precomputed phycocyanin fluorescence spectrum
        h_C_phycoerythrin_res: optional precomputed phycoerythrin fluorescence spectrum
        da_w_div_dT_res: optional precomputed temperature gradient of pure water absorption [m-1 K-1]
        E0_res: optional precomputed extraterrestrial solar irradiance
        a_oz_res: optional precomputed ozone absorption
        a_ox_res: optional precomputed oxygen absorption
        a_wv_res: optional precomputed water vapour absorption
        Ed_d_res: optional precomputed direct downwelling irradiance
        Ed_sa_res: optional precomputed aerosol-scattered downwelling irradiance
        Ed_sr_res: optional precomputed Rayleigh-scattered downwelling irradiance
        Ed_res: optional precomputed total downwelling irradiance
        Ed_s_res: optional precomputed diffuse downwelling irradiance
        n2_res: optional precomputed refractive index of water
        Ls_Ed: optional precomputed ratio of sky radiance to downwelling irradiance

    Returns:
        Rrs_sim: above-water remote sensing reflectance [sr-1]
    """    
    C_phy = np.sum([parameters["C_0"], parameters["C_1"], parameters["C_2"], parameters["C_3"], parameters["C_4"], parameters["C_5"], parameters["C_6"], parameters["C_7"]])

    if n2_res is None:
        n2 = parameters["n2"]
    else:
        n2 = n2_res

    # it makes sense to precompute some coefficients outside of a() and bb() because they are used in both functions
    if bb_w_res is None:
        bb_w_res = resampling.resample_bb_w(wavelengths=wavelengths)

    # if a_d_res is None:
    #     a_d_res = absorption.a_d(wavelengths=wavelengths,
    #                              C_phy=C_phy,
    #                              C_ism=parameters["C_ism"],
    #                              A_md=parameters["A_md"],
    #                              A_bd=parameters["A_bd"],
    #                              S_md=parameters["S_md"],
    #                              S_bd=parameters["S_bd"],
    #                              C_md=parameters["C_md"],
    #                              C_bd=parameters["C_bd"],
    #                              lambda_0_md=parameters["lambda_0_md"],
    #                              lambda_0_bd=parameters["lambda_0_bd"],
    #                              a_bd_spec_res=a_bd_spec_res,
    #                              a_md_spec_res=a_md_spec_res)
    # a_d_lambda_0_res = np.interp(parameters["lambda_0_c_d"].value, wavelengths, a_d_res) if parameters["interpolate"].value else a_d_res[utils.find_closest(wavelengths, parameters["lambda_0_c_d"])[1]]

    if a_md_res is None:
        a_md_res = absorption.a_md(wavelengths=wavelengths,
                                 C_ism=parameters["C_ism"],
                                 A_md=parameters["A_md"],
                                 S_md=parameters["S_md"],
                                 C_md=parameters["C_md"],
                                 lambda_0_md=parameters["lambda_0_md"],
                                 a_md_spec_res=a_md_spec_res)
    a_md_lambda_0_res = np.interp(parameters["lambda_0_c_d"].value, wavelengths, a_md_res) if parameters["interpolate"].value else a_md_res[utils.find_closest(wavelengths, parameters["lambda_0_c_d"])[1]]

    if a_bd_res is None:
        a_bd_res = absorption.a_bd(wavelengths=wavelengths,
                                 C_phy=C_phy,
                                 A_bd=parameters["A_bd"],
                                 S_bd=parameters["S_bd"],
                                 C_bd=parameters["C_bd"],
                                 lambda_0_bd=parameters["lambda_0_bd"],
                                 a_bd_spec_res=a_bd_spec_res)
    a_bd_lambda_0_res = np.interp(parameters["lambda_0_c_d"].value, wavelengths, a_bd_res) if parameters["interpolate"].value else a_bd_res[utils.find_closest(wavelengths, parameters["lambda_0_c_d"])[1]]

    # if c_d_res is None:
    #     c_d_res = attenuation.c_d(wavelengths=wavelengths,
    #                               C_phy=C_phy,
    #                               C_ism=parameters["C_ism"],
    #                               A_md=parameters["A_md"],
    #                               A_bd=parameters["A_bd"],
    #                               S_md=parameters["S_md"],
    #                               S_bd=parameters["S_bd"],
    #                               C_md=parameters["C_md"],
    #                               C_bd=parameters["C_bd"],
    #                               lambda_0_c_d=parameters["lambda_0_c_d"],
    #                               lambda_0_md=parameters["lambda_0_md"],
    #                               lambda_0_bd=parameters["lambda_0_bd"],
    #                               gamma_d=parameters["gamma_d"],
    #                               x0=parameters["x0"],
    #                               x1=parameters["x1"],
    #                               x2=parameters["x2"],
    #                               c_d_lambda_0_res=c_d_lambda_0_res,
    #                               omega_d_lambda_0_res=omega_d_lambda_0_res,
    #                               a_d_lambda_0_res = a_d_lambda_0_res,
    #                               a_md_spec_res=a_md_spec_res,
    #                               a_bd_spec_res=a_bd_spec_res)

    if c_md_res is None:
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
                                  a_md_lambda_0_res = a_md_lambda_0_res)

    c_md_lambda_0_res = np.interp(parameters["lambda_0_c_d"].value, wavelengths, c_md_res) if parameters[
        "interpolate"].value else c_md_res[utils.find_closest(wavelengths, parameters["lambda_0_c_d"])[1]]

    if c_bd_res is None:
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
                                  a_bd_lambda_0_res = a_d_lambda_0_res)

    c_bd_lambda_0_res = np.interp(parameters["lambda_0_c_d"].value, wavelengths, c_bd_res) if parameters[
        "interpolate"].value else c_bd_res[utils.find_closest(wavelengths, parameters["lambda_0_c_d"])[1]]

    if b_md_res is None:
        b_md_res = scattering.b(a=a_md_res, c=c_md_res)

    if b_bd_res is None:
        b_bd_res = scattering.b(a=a_bd_res, c=c_bd_res)
    
    if bb_bd_res is None:
        bb_bd_res = backscattering.bb_d(b_d=b_bd_res, bb_ratio_d=parameters["b_ratio_bd"])

    if bb_md_res is None:
        bb_md_res = backscattering.bb_d(b_d=b_md_res, bb_ratio_d=parameters["b_ratio_md"])

    if bb_phy_res is None:
        bb_phy_res = backscattering.bb_phy_bi(wavelengths=wavelengths,
                                                  C_0=parameters["C_0"], 
                                                  C_1=parameters["C_1"], 
                                                  C_2=parameters["C_2"], 
                                                  C_3=parameters["C_3"], 
                                                  C_4=parameters["C_4"], 
                                                  C_5=parameters["C_5"], 
                                                  C_6=parameters["C_6"], 
                                                  C_7=parameters["C_7"], 
                                                  bb_i_spec_res=bb_i_spec_res)
    if bb_p_res is None:
        bb_p_res = bb_bd_res + bb_md_res + bb_phy_res

    if a_res is None:
        # C_phy could be used as an argument so it does not need to be recomputed inside functions
        a_res = absorption.a_total(wavelengths=wavelengths, 
                                   C_0=parameters["C_0"], 
                                   C_1=parameters["C_1"], 
                                   C_2=parameters["C_2"], 
                                   C_3=parameters["C_3"], 
                                   C_4=parameters["C_4"], 
                                   C_5=parameters["C_5"], 
                                   C_6=parameters["C_6"], 
                                   C_7=parameters["C_7"], 
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
                                   lambda_0_phy=parameters["lambda_0_phy"].value, 
                                   A=parameters["A_phy"],
                                   E0=parameters["E0"], 
                                   E1=parameters["E1"], 
                                   interpolate=parameters["interpolate"], 
                                   T_W=parameters["T_W"], 
                                   T_W_0=parameters["T_W_0"], 
                                   # a_bd_res=a_bd_res,
                                   # a_md_res=a_md_res,
                                   a_md_spec_res=a_md_spec_res,
                                   a_bd_spec_res=a_bd_spec_res,
                                   a_i_spec_res=a_i_spec_res,
                                   a_phy_res=a_phy_res,
                                   a_Y_N_res=a_Y_N_res,
                                   a_w_res=a_w_res,
                                   da_W_div_dT_res=da_w_div_dT_res)

    if bb_res is None:
        # C_phy could be used as an argument so it does not need to be recomputed inside functions
        bb_res = backscattering.bb_total(wavelengths=wavelengths,
                                           C_0=parameters["C_0"], 
                                           C_1=parameters["C_1"], 
                                           C_2=parameters["C_2"], 
                                           C_3=parameters["C_3"], 
                                           C_4=parameters["C_4"], 
                                           C_5=parameters["C_5"], 
                                           C_6=parameters["C_6"], 
                                           C_7=parameters["C_7"], 
                                           C_ism=parameters["C_ism"], 
                                           bb_ratio_C_0=parameters["b_ratio_C_0"], 
                                           bb_ratio_C_1=parameters["b_ratio_C_1"], 
                                           bb_ratio_C_2=parameters["b_ratio_C_2"], 
                                           bb_ratio_C_3=parameters["b_ratio_C_3"], 
                                           bb_ratio_C_4=parameters["b_ratio_C_4"], 
                                           bb_ratio_C_5=parameters["b_ratio_C_5"], 
                                           bb_ratio_C_6=parameters["b_ratio_C_6"], 
                                           bb_ratio_C_7=parameters["b_ratio_C_7"], 
                                           # b_ratio_md=parameters["b_ratio_md"],
                                           # b_ratio_bd=parameters["b_ratio_bd"],
                                           bb_ratio_d=parameters["b_ratio_d"],
                                           fresh=parameters["fresh"],
                                           A_md=parameters["A_md"],
                                           A_bd=parameters["A_bd"],
                                           S_md=parameters["S_md"],
                                           S_bd=parameters["S_bd"],
                                           C_md=parameters["C_md"],
                                           C_bd=parameters["C_bd"],
                                           lambda_0_md=parameters["lambda_0_md"], 
                                           lambda_0_bd=parameters["lambda_0_bd"], 
                                           lambda_0_c_d=parameters["lambda_0_c_d"], 
                                           gamma_d=parameters["gamma_d"], 
                                           x0=parameters["x0"], 
                                           x1=parameters["x1"], 
                                           x2=parameters["x2"], 
                                           # c_md_lambda_0_res=c_md_lambda_0_res,
                                           # a_md_lambda_0_res=a_md_lambda_0_res,
                                           # c_bd_lambda_0_res=c_bd_lambda_0_res,
                                           # a_bd_lambda_0_res=a_bd_lambda_0_res,
                                           omega_d_lambda_0_res=omega_d_lambda_0_res, 
                                           interpolate=parameters["interpolate"],
                                           # a_md_res=a_md_res,
                                           # a_bd_res=a_bd_res,
                                           a_md_spec_res=a_md_spec_res,
                                           a_bd_spec_res=a_bd_spec_res,
                                           # b_md_res=b_md_res,
                                           bb_d_res=b_bd_res,
                                           # bb_bd_res=bb_bd_res,
                                           # bb_md_res=bb_md_res,
                                           bb_p_res=bb_p_res,
                                           bb_w_res=bb_w_res,
                                           b_i_spec_res=bb_i_spec_res,
                                           # c_md_res=c_md_res,
                                           # c_bd_res=c_bd_res
                                         )
    
    R_rs_water = lee.Rrs_deep(a=a_res, 
                               bb=bb_res,
                               bb_p=bb_p_res,
                               bb_w=bb_w_res,
                               Gw0=parameters["Gw0"],
                               Gw1=parameters["Gw1"],
                               Gp0=parameters["Gp0"],
                               Gp1=parameters["Gp1"])
    
    if parameters["C_0"]+parameters["C_1"]+parameters["C_2"]+parameters["C_3"]+parameters["C_4"]+parameters["C_5"]+parameters["C_6"]+parameters["C_7"] >0.1:
        R_rs_water += fluorescence.Rrs_fl(wavelengths=wavelengths,
                                             L_fl_lambda0=parameters['L_fl_lambda0'],
                                             W=parameters['W'],
                                             fwhm1=parameters['fwhm1'],
                                             fwhm2=parameters['fwhm2'],
                                             lambda_C1=parameters['lambda_C1'],
                                             lambda_C2=parameters['lambda_C2'],
                                             double=parameters['double'],
                                             h_C_res=h_C_res)
    if parameters["C_3"] > 0.1:
        R_rs_water += fluorescence.Rrs_fl_phycocyanin(wavelengths=wavelengths,
                                         L_fl_phycocyanin=parameters['L_fl_phycocyanin'],
                                         fwhm=parameters['fwhm_phycocyanin'],
                                         lambda_C=parameters['lambda_C_phycocyanin'],
                                         h_C_phycocyanin_res=h_C_phycocyanin_res)
    if parameters["C_4"] > 0.1:
        R_rs_water += fluorescence.Rrs_fl_phycoerythrin(wavelengths=wavelengths,
                                           L_fl_phycoerythrin=parameters['L_fl_phycoerythrin'],
                                           fwhm=parameters['fwhm_phycoerythrin'],
                                           lambda_C=parameters['lambda_C_phycoerythrin'],
                                           h_C_phycoerythrin_res=h_C_phycoerythrin_res)
    
    return R_rs_water + parameters["offset"]

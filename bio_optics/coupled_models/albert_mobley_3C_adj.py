import numpy as np
from ..water import absorption, backscattering, attenuation, bottom_reflectance
from ..water.reflectance import albert_mobley as water_alg
from ..atmosphere.adjacency import reflectance as adjacency_effect
from ..surface import reflectance as surface, air_water


def forward(parameters,
        wavelengths,
        a_res=None,
        bb_res=None,
        a_w_res=None,
        da_w_div_dT_res=None,
        a_i_spec_res=None,
        a_Y_N_res = [],
        a_NAP_N_res = [],
        b_phy_norm_res = [],
        bb_w_res = [],
        b_X_norm_res=None,
        b_Mie_norm_res=None,
        R_b_i_res = [],
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
        Ls_Ed=None,
        R_bg=None,
        b_ray=None):
    """
    Forward simulation of shallow water Rrs with adjacency effect and surface reflectance after Albert & Mobley (2003) [1].

    [1] Albert & Mobley (2003): An analytical model for subsurface irradiance and remote sensing reflectance in deep and shallow case-2 waters. [10.1364/OE.11.002873]

    Args:
        parameters: lmfit Parameters object (must include C_adj, lambda_r, b_r_spec, n_r for adjacency)
        wavelengths: wavelengths [nm]
        a_res: optional precomputed total absorption coefficient [m-1]
        bb_res: optional precomputed total backscattering coefficient [m-1]
        a_w_res: optional precomputed pure water absorption [m-1]
        da_w_div_dT_res: optional precomputed temperature gradient of pure water absorption [m-1 K-1]
        a_i_spec_res: optional precomputed specific phytoplankton absorption spectra [m2 mg-1]
        a_Y_N_res: optional precomputed normalised CDOM absorption
        a_NAP_N_res: optional precomputed normalised NAP absorption
        b_phy_norm_res: optional precomputed normalised phytoplankton backscattering
        bb_w_res: optional precomputed water backscattering [m-1]
        b_X_norm_res: optional precomputed normalised mineral backscattering
        b_Mie_norm_res: optional precomputed normalised Mie backscattering
        R_b_i_res: optional precomputed bottom reflectance spectra
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
        R_bg: optional background reflectance spectrum for adjacency effect

    Returns:
        Rrs_sim: above-water remote sensing reflectance [sr-1]
    """
    if n2_res is None:
        n2 = parameters["n2"]
    else:
        n2 = n2_res

    if "rho_L" in parameters:
        rho_L = parameters["rho_L"].value
    else:
        rho_L = air_water.fresnel(parameters["theta_view"], n1=parameters["n1"], n2=n2)

    if Ls_Ed is None:
        Ls_Ed = np.zeros_like(wavelengths)

    ctsp = np.cos(air_water.snell(parameters["theta_sun"],  n1=parameters["n1"], n2=n2))  #cos of theta_sun_prime. theta_sun_prime = snell(theta_sun, n1, n2)
    ctvp = np.cos(air_water.snell(parameters["theta_view"], n1=parameters["n1"], n2=n2))

    if a_res is None:
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
    
    if bb_res is None:
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
                      R_bg=None, 
                      b_ray=None):
    """
    Compute the adjacency reflectance contribution using parameters from the lmfit Parameters object.

    Args:
        parameters: lmfit Parameters object (must include C_adj, lambda_r, b_r_spec, n_r)
        wavelengths: wavelengths [nm]
        R_bg: optional background reflectance spectrum; zeros (no adjacency) if not provided
        b_ray: optional precomputed Rayleigh scattering spectrum

    Returns:
        Rrs_adjacency: adjacency radiance reflectance [sr-1]
    """
    Rrs_adjacency = adjacency_effect.Rrs_adjacency(C_adj=parameters["C_adj"], wavelengths=wavelengths, lambda_r=parameters["lambda_r"], b_r_spec=parameters["b_r_spec"], n_r=parameters["n_r"], R_bg=R_bg, b_ray=b_ray)
    
    return Rrs_adjacency

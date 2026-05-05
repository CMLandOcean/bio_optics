"""
Coupled Albert & Mobley model: water-leaving Rrs + surface reflectance (Rrs_surf).
"""
import numpy as np
from ..water.reflectance import albert_mobley
from ..surface import reflectance as surface_reflectance


def forward(parameters,
            wavelengths,
            a_res=None,
            bb_res=None,
            a_w_res=None,
            da_w_div_dT_res=None,
            a_i_spec_res=None,
            a_Y_N_res=None,
            a_NAP_N_res=None,
            b_phy_norm_res=None,
            bb_w_res=None,
            b_X_norm_res=None,
            b_Mie_norm_res=None,
            R_b_i_res=None,
            E0_res=None,
            a_oz_res=None,
            a_ox_res=None,
            a_wv_res=None,
            Ed_d_res=None,
            Ed_sa_res=None,
            Ed_sr_res=None,
            Ed_s_res=None,
            Ed_res=None,
            n2_res=None,
            Ls_Ed=None):
    """
    Forward simulation: water-leaving Rrs (albert_mobley.forward) + surface reflectance (surface_reflectance.forward).

    Args:
        parameters: lmfit Parameters object
        wavelengths: wavelengths [nm]
        a_res: optional precomputed total absorption coefficient
        bb_res: optional precomputed total backscattering coefficient
        a_w_res: optional precomputed pure water absorption
        da_w_div_dT_res: optional precomputed temperature gradient of pure water absorption
        a_i_spec_res: optional precomputed specific phytoplankton absorption
        a_Y_N_res: optional precomputed normalised CDOM absorption
        a_NAP_N_res: optional precomputed normalised NAP absorption
        b_phy_norm_res: optional precomputed normalised phytoplankton backscattering
        bb_w_res: optional precomputed water backscattering
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
        Ed_s_res: optional precomputed diffuse downwelling irradiance
        Ed_res: optional precomputed total downwelling irradiance
        n2_res: optional precomputed refractive index of water
        Ls_Ed: optional ratio of sky radiance to downwelling irradiance

    Returns:
        Rrs_sim: above-water remote sensing reflectance [sr-1]
    """
    Rrs_water = albert_mobley.forward(
        parameters=parameters, wavelengths=wavelengths,
        a_res=a_res, bb_res=bb_res, a_w_res=a_w_res,
        da_w_div_dT_res=da_w_div_dT_res, a_i_spec_res=a_i_spec_res,
        a_Y_N_res=a_Y_N_res, a_NAP_N_res=a_NAP_N_res,
        b_phy_norm_res=b_phy_norm_res, bb_w_res=bb_w_res,
        b_X_norm_res=b_X_norm_res, b_Mie_norm_res=b_Mie_norm_res,
        R_b_i_res=R_b_i_res, n2_res=n2_res)

    Rrs_surf = surface_reflectance.forward(
        parameters=parameters, wavelengths=wavelengths,
        E0_res=E0_res, a_oz_res=a_oz_res, a_ox_res=a_ox_res, a_wv_res=a_wv_res,
        Ed_d_res=Ed_d_res, Ed_sa_res=Ed_sa_res, Ed_sr_res=Ed_sr_res,
        Ed_s_res=Ed_s_res, Ed_res=Ed_res, n2_res=n2_res, Ls_Ed=Ls_Ed)

    return Rrs_water + Rrs_surf + parameters["offset"]

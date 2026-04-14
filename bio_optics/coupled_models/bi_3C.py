"""
Coupled HEREON model: water-leaving Rrs + surface reflectance (Rrs_surf).
"""
import numpy as np
from ..reflectance import bi
from ..surface import reflectance as srf, air_water
from ..atmosphere import sky_radiance, downwelling_irradiance
from ..helper import resampling


def forward(parameters,
            wavelengths,
            a_res=[],
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
            bb_p_res=[],
            bb_phy_res=[],
            bb_w_res=[],
            bb_i_spec_res=[],
            c_md_res=[],
            c_bd_res=[],
            h_C_res=[],
            h_C_phycocyanin_res=[],
            h_C_phycoerythrin_res=[],
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
            omega_d_lambda_0_res=None,
            a_d_lambda_0_res=None,
            c_d_lambda_0_res=None):
    """
    Forward simulation: water-leaving Rrs (bi.forward) + surface reflectance (surface_reflectance.forward).

    Args:
        parameters: lmfit Parameters object specifying the model configuration
        wavelengths: wavelengths [nm]
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
        bb_p_res: optional precomputed particulate backscattering [m-1]
        bb_phy_res: optional precomputed phytoplankton backscattering [m-1]
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
        omega_d_lambda_0_res: optional precomputed detrital single scattering albedo at the reference wavelength
        a_d_lambda_0_res: optional precomputed detrital absorption at the reference wavelength [m-1]
        c_d_lambda_0_res: optional precomputed detrital attenuation at the reference wavelength [m-1]

    Returns:
        Rrs_sim: above-water remote sensing reflectance [sr-1]
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

    Rrs_water = bi.forward(parameters=parameters,
                                wavelengths=wavelengths,
                                a_res=a_res,
                                a_md_res=a_md_res,
                                a_bd_res=a_bd_res,
                                a_md_spec_res=a_md_spec_res,
                                a_bd_spec_res=a_bd_spec_res,
                                a_w_res=a_w_res,
                                a_i_spec_res=a_i_spec_res,
                                a_phy_res=a_phy_res,
                                a_Y_N_res=a_Y_N_res,
                                bb_res=bb_res,
                                bb_bd_res=bb_bd_res,
                                bb_p_res=bb_p_res,
                                bb_phy_res=bb_phy_res,
                                bb_w_res=bb_w_res,
                                bb_i_spec_res=bb_i_spec_res,
                                c_md_res=c_md_res,
                                c_bd_res=c_bd_res,
                                h_C_res=h_C_res,
                                h_C_phycocyanin_res=h_C_phycocyanin_res,
                                h_C_phycoerythrin_res=h_C_phycoerythrin_res,
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
                                omega_d_lambda_0_res=omega_d_lambda_0_res,
                                a_d_lambda_0_res=a_d_lambda_0_res,
                                c_d_lambda_0_res=c_d_lambda_0_res)

    if len(Ed_d_res) == 0:
        Ed_d = downwelling_irradiance.Ed_d(wavelengths, parameters["theta_sun"], parameters["P"], parameters["AM"], parameters["RH"], parameters["H_oz"], parameters["WV"], parameters["alpha"], parameters["beta"], E0_res, a_oz_res, a_ox_res, a_wv_res, Ed_d_res)
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
        Ed = downwelling_irradiance.Ed(Ed_d, Ed_s, parameters["fd_d"], parameters["fd_s"])
    else:
        Ed = Ed_res

    L_s = sky_radiance.L_s(parameters["fd_d"], parameters["g_dd"], Ed_d, parameters["fd_s"], parameters["g_dsr"], Ed_sr, parameters["g_dsa"], Ed_sa)

    Rrs_surface = srf.Rrs_surf(L_s, Ed, rho_L, parameters["d_r"])
    Rrs_surface += air_water.fresnel(parameters['theta_view'], n2=n2) * Ls_Ed

    if np.any(Rrs_surface < 0):
        Rrs_surface = Rrs_surface + 1

    # offset was already added in bi.forward(); subtract it so we add it only once
    return Rrs_water - parameters["offset"] + Rrs_surface + parameters["offset"]

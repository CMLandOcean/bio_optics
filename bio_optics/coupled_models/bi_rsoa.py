"""
Coupled HEREON model: water-leaving Rrs + RSOA surface reflectance model.
"""
from ..reflectance import bi
from ..surface import reflectance as srf


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
            n2_res=[],
            omega_d_lambda_0_res=None,
            a_d_lambda_0_res=None,
            c_d_lambda_0_res=None):
    """
    Forward simulation: water-leaving Rrs (bi.forward) + RSOA power-law surface model.

    Args:
        parameters: lmfit Parameters object specifying the model configuration (must include h0, h1, lambda0 for RSOA)
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
        n2_res: optional precomputed refractive index of water
        omega_d_lambda_0_res: optional precomputed detrital single scattering albedo at the reference wavelength
        a_d_lambda_0_res: optional precomputed detrital absorption at the reference wavelength [m-1]
        c_d_lambda_0_res: optional precomputed detrital attenuation at the reference wavelength [m-1]

    Returns:
        Rrs_sim: above-water remote sensing reflectance [sr-1]
    """
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
                                n2_res=n2_res,
                                omega_d_lambda_0_res=omega_d_lambda_0_res,
                                a_d_lambda_0_res=a_d_lambda_0_res,
                                c_d_lambda_0_res=c_d_lambda_0_res)

    Rrs_surface = srf.rsoa(wavelengths=wavelengths,
                            h0=parameters['h0'],
                            h1=parameters['h1'],
                            lambda0=parameters['lambda0'])

    return Rrs_water + Rrs_surface

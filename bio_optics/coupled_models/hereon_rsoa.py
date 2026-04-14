"""
Coupled HEREON model: water-leaving Rrs + RSOA surface reflectance model.
"""
from ..reflectance import hereon
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
    Forward simulation: water-leaving Rrs (hereon.forward) + RSOA surface model.

    Returns:
        Rrs_sim: above-water remote sensing reflectance [sr-1]
    """
    Rrs_water = hereon.forward(parameters=parameters,
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

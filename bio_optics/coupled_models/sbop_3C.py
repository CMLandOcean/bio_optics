"""
Coupled SBOP model: water-leaving Rrs + surface reflectance term (3-component).
"""
import numpy as np
from ..water.reflectance import sbop
from ..surface import reflectance as srf
from ..helper import utils


def forward(params,
            wavelengths,
            a_w_res=None,
            bb_w_res=None,
            R_b_i_res=None):
    """
    Forward simulation of above-water Rrs including surface reflectance.

    Rrs_total = sbop.forward() + surface.Rrs_surf(L_s, Ed, rho_L, d_r)

    Returns:
        Rrs_sim: simulated above-water remote sensing reflectance [sr-1]
    """
    Rrs_water = sbop.forward(params=params,
                              wavelengths=wavelengths,
                              a_w_res=a_w_res,
                              bb_w_res=bb_w_res,
                              R_b_i_res=R_b_i_res)

    Rrs_surface = srf.Rrs_surf(L_s=params['L_s'],
                                Ed=params['Ed'],
                                rho_L=params['rho_L'],
                                d_r=params['d_r'])

    return Rrs_water + Rrs_surface

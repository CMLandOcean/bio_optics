import numpy as np
from .. import absorption, backscattering, bottom_reflectance
from . import hope
from ...surface import air_water
from ...helper import resampling, utils


def rrs_sh(C_Mie=0,
           C_Y=0,
           zB=2,
           f_0=0,
           f_1=1,
           f_2=0,
           f_3=0,
           f_4=0,
           f_5=0,
           B_0=1/np.pi,
           B_1=1/np.pi,
           B_2=1/np.pi,
           B_3=1/np.pi,
           B_4=1/np.pi,
           B_5=1/np.pi,
           lambda_0=440,
           lambda_S=555,
           S=0.015,
           bb_Mie_spec=1,
           n=-1,
           fresh=False,
           q=0.75,
           g_0=0.089,
           g_1=0.125,
           wavelengths=np.arange(400, 800),
           a_w_res=None,
           bb_w_res=None,
           R_b_i_res=None):
    """
    Shallow water bio-optical properties (SBOP) model after Li et al. (2017) [1].

    [1] Li et al. (2017): Remote sensing estimation of colored dissolved organic matter (CDOM) in optically shallow waters [10.1016/j.isprsjprs.2017.03.015]
    [2] Lee et al. (1999): Hyperspectral remote sensing for shallow waters: 2 Deriving bottom depths and water properties by optimization [10.1364/ao.38.003831]
    [3] Albert & Mobley (2003): An analytical model for subsurface irradiance and remote sensing reflectance in deep and shallow case-2 waters. [10.1364/OE.11.002873]

    Args:
        C_Mie: concentration of non-algal particles type II [mg L-1], default: 0
        C_Y: CDOM absorption coefficient at lambda_0 [m-1], default: 0
        zB: water depth [m], default: 2
        f_0: fractional cover of bottom type 0, default: 0
        f_1: fractional cover of bottom type 1, default: 1
        f_2: fractional cover of bottom type 2, default: 0
        f_3: fractional cover of bottom type 3, default: 0
        f_4: fractional cover of bottom type 4, default: 0
        f_5: fractional cover of bottom type 5, default: 0
        B_0: bidirectional reflectance factor for bottom type 0, default: 1/pi
        B_1: bidirectional reflectance factor for bottom type 1, default: 1/pi
        B_2: bidirectional reflectance factor for bottom type 2, default: 1/pi
        B_3: bidirectional reflectance factor for bottom type 3, default: 1/pi
        B_4: bidirectional reflectance factor for bottom type 4, default: 1/pi
        B_5: bidirectional reflectance factor for bottom type 5, default: 1/pi
        lambda_0: reference wavelength for CDOM absorption [nm], default: 440
        lambda_S: reference wavelength for particle backscattering [nm], default: 555
        S: spectral slope of CDOM absorption [nm-1], default: 0.015
        bb_Mie_spec: specific backscattering of non-algal particles type II [m2 g-1], default: 1
        n: spectral slope exponent for Mie backscattering, default: -1
        fresh: True for fresh water, False for ocean water, default: False
        q: ratio of absorption to scattering for NAP, default: 0.75
        g_0: empirical constant [sr-1], default: 0.089
        g_1: empirical constant [sr-1], default: 0.125
        wavelengths: wavelengths [nm], default: np.arange(400, 800)
        a_w_res: optional precomputed pure water absorption [m-1]
        bb_w_res: optional precomputed water backscattering [m-1]
        R_b_i_res: optional precomputed bottom reflectance spectra

    Returns:
        rrs_sh: subsurface radiance reflectance [sr-1] of shallow water
    """
    bs = backscattering.bb_w(wavelengths=wavelengths, fresh=fresh, bb_w_res=bb_w_res if bb_w_res is not None else []) + \
         backscattering.bb_Mie(C_Mie=C_Mie, wavelengths=wavelengths, bb_Mie_spec=bb_Mie_spec, lambda_S=lambda_S, n=n)

    ab = absorption.a_w(wavelengths=wavelengths, a_w_res=a_w_res if a_w_res is not None else []) + \
         absorption.a_Y(wavelengths=wavelengths, C_Y=C_Y, S=S, lambda_0=lambda_0) + \
         q * backscattering.bb_Mie(C_Mie=C_Mie, wavelengths=wavelengths, bb_Mie_spec=bb_Mie_spec, lambda_S=lambda_S, n=n)

    kappa = ab + bs
    u = bs / kappa

    rrs_sh = hope.rrs_dp(u, g_0=g_0, g_1=g_1) * (1 - np.exp(-hope.D_u_C(u, f1=1, f2=2.4) * kappa * zB)) + \
             bottom_reflectance.Rrs_b(f_0=f_0, f_1=f_1, f_2=f_2, f_3=f_3, f_4=f_4, f_5=f_5,
                                      B_0=B_0, B_1=B_1, B_2=B_2, B_3=B_3, B_4=B_4, B_5=B_5,
                                      wavelengths=wavelengths, R_b_i_res=R_b_i_res if R_b_i_res is not None else []) * \
             np.exp(-hope.D_u_B(u, f1=1, f2=5.5) * kappa * zB)

    return rrs_sh


def forward(params,
            wavelengths,
            a_w_res=None,
            bb_w_res=None,
            R_b_i_res=None):
    """
    Forward simulation of water-leaving remote sensing reflectance after Li et al. (2017) [1].

    [1] Li et al. (2017): Remote sensing estimation of colored dissolved organic matter (CDOM) in optically shallow waters [10.1016/j.isprsjprs.2017.03.015]

    Args:
        params: lmfit Parameters object specifying the model configuration
        wavelengths: wavelengths [nm]
        a_w_res: optional precomputed pure water absorption [m-1]
        bb_w_res: optional precomputed water backscattering [m-1]
        R_b_i_res: optional precomputed bottom reflectance spectra

    Returns:
        Rrs_sim: simulated above-water remote sensing reflectance [sr-1]
    """
    Rrs_sim = air_water.below2above(
        rrs_sh(wavelengths=wavelengths,
               C_Mie=params['C_Mie'],
               C_Y=params['C_Y'],
               zB=params['zB'],
               f_0=params['f_0'],
               f_1=params['f_1'],
               f_2=params['f_2'],
               f_3=params['f_3'],
               f_4=params['f_4'],
               f_5=params['f_5'],
               B_0=params['B_0'],
               B_1=params['B_1'],
               B_2=params['B_2'],
               B_3=params['B_3'],
               B_4=params['B_4'],
               B_5=params['B_5'],
               lambda_0=params['lambda_0'],
               lambda_S=params['lambda_S'],
               S=params['S'],
               bb_Mie_spec=params['bb_Mie_spec'],
               n=params['n'],
               fresh=params['fresh'],
               q=params['q'],
               g_0=params['g_0'],
               g_1=params['g_1'],
               a_w_res=a_w_res,
               bb_w_res=bb_w_res,
               R_b_i_res=R_b_i_res) + params['offset'])

    return Rrs_sim

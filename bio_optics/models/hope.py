import numpy as np
from lmfit import minimize, Parameters
from .. water import absorption, backscattering, temperature_gradient, attenuation, bottom_reflectance
from .. surface import surface, air_water
from .. helper import resampling


def D_u_C(u, f1=1.03, f2=2.4):
    """
    Path elongation factor for scattered photons from the water body after Eq. 5 in [1].

    [1] Lee et al. (1999): Hyperspectral remote sensing for shallow waters: 2 Deriving bottom depths and water properties by optimization [10.1364/ao.38.003831]

    Args:
        u (_type_): _description_
        f1 (float, optional): Factor 1. Defaults to 1.03.
        f2 (float, optional): Factor 2. Defaults to 2.4.

    Returns:
        D_u_C: Path elongation factor for water body.
    """
    D_u_C = f1 * (1 + f2 * u)**0.5

    return D_u_C


def D_u_B(u, f1=1.04, f2=5.4):
    """
    Path enlongation factor for scattered photons from the bottom after Eq. 5 in [1].

    [1] Lee et al. (1999): Hyperspectral remote sensing for shallow waters: 2 Deriving bottom depths and water properties by optimization [10.1364/ao.38.003831]

    Args:
        u (_type_): _description_
        f1 (float, optional): Factor 1. Defaults to 1.04.
        f2 (float, optional): Factor 2. Defaults to 5.4.

    Returns:
        D_u_B: Path enlongation factor for benthos
    """
    D_u_B = f1 * (1 + f2 * u)**0.5

    return D_u_B


def r_rs_dp(u, g_0=0.084, g_1=0.170):
    """
    Subsurface radiance reflectance [sr-1] for optically deep water after Eq. 4 in [1].

    [1] Lee et al. (1999): Hyperspectral remote sensing for shallow waters: 2 Deriving bottom depths and water properties by optimization [10.1364/ao.38.003831]

    Args:
        u (_type_): b_b / (a + b_b)
        g_0 (float, optional): Defaults to 0.084.
        g_1 (float, optional): Defaults to 0.17.

    Returns:
        r_rs_dp: Subsurface radiance reflectance [sr-1] for optically deep water
    """
    r_rs_dp = (g_0 + g_1 * u) * u

    return r_rs_dp


def r_rs_sh(C_Mie = 1,       # represents X from Eq. 10 [1]
            
            b_bMie_spec = 1, # must be 1 so C_Mie can represent X
            lambda_S = 400,
            n = -1,          # should be estimated using utils.estimate_y()*(-1)

            wavelengths = np.arange(400,800),
            fresh = False,

            


            B = 1,
            lambda_B = 560,
            C_Y = 0.0001,
            lambda_0 = 440,
            a_phy_440 = 0.0001,


            f_0 = 0,
            f_1 = 1,
            f_2 = 0,
            f_3 = 0,
            f_4 = 0,
            f_5 = 0,
            B_0 = 1/np.pi,
            B_1 = 1/np.pi,
            B_2 = 1/np.pi,
            B_3 = 1/np.pi,
            B_4 = 1/np.pi,
            B_5 = 1/np.pi,
            S = 0.015,
            zB = 2,
            n1 = 1,
            n2 = 1.33,
            g_0 = 0.084, 
            g_1 = 0.17,
            theta_inc = np.radians(30),
            a_w_res=[],
            A_res=[],
            b_bw_res=[],
            R_i_b_res=[]):
    """
    """
    # Backscattering and absorption coefficients of the water body depending on the concentration of optically active water constituents
    # b_bp is defined as X * (400/wavelengths)**Y in Lee et al. (1998) [1]. Almost the same formulation is used in Albert and Mobley (2003) [2] for b_bMie,
    # thus b_bp() can be exchanged with b_bMie(lambda_S=400, n=-y, b_bMie_spec=1). C_Mie then represents X.
    
    bs = backscattering.b_bw(wavelengths=wavelengths, fresh=fresh, b_bw_res=b_bw_res) + \
         backscattering.b_bMie(C_Mie=C_Mie, wavelengths=wavelengths, b_bMie_spec=b_bMie_spec, lambda_S=lambda_S, n=n)
    
    ab = absorption.a_w(wavelengths=wavelengths, a_w_res=a_w_res) + \
         absorption.a_Y(wavelengths=wavelengths, C_Y=C_Y, S=S, lambda_0=lambda_0) + \
         absorption.a_Phi(wavelengths=wavelengths, a_phy_440=a_phy_440, A_res=A_res)

    kappa = ab + bs    
    u = bs / kappa
    
    r_rs_sh = (r_rs_dp(u=u, g_0=g_0, g_1=g_1) * (1 - np.exp(-kappa*zB* (1/np.cos(air_water.snell(theta_inc=theta_inc, n1=n1, n2=n2)) + D_u_C(u=u))))) + \
              bottom_reflectance.R_rs_b_norm(B=B,
                                             lambda_B=lambda_B,
                                             f_0=f_0,
                                             f_1=f_1, 
                                             f_2=f_2, 
                                             f_3=f_3, 
                                             f_4=f_4, 
                                             f_5=f_5, 
                                             B_0=B_0, 
                                             B_1=B_1, 
                                             B_2=B_2, 
                                             B_3=B_3, 
                                             B_4=B_4, 
                                             B_5=B_5, 
                                             wavelengths=wavelengths, 
                                             R_i_b_res=R_i_b_res) * np.exp(-k*zB * (1/np.cos(air_water.snell(theta_inc=theta_inc, n1=n1, n2=n2)) + D_u_B(u=u)))
                
    return r_rs_sh
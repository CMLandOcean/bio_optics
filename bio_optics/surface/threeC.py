import numpy as np
from ..atmosphere import sky_radiance, downwelling_irradiance

def R_rs_surf(wavelengths, 
                 Ls_Ed, # from measurements, this is the only measurement input to the min_funct() the model() functions
                 rho_s = 0.0256, 
                 rho_dd = 0.1, 
                 rho_ds = 0.1, 
                 delta = 0.0,
                 theta_sun=np.radians(30), 
                 P=1013.25, 
                 AM=1, 
                 RH=60, 
                 H_oz=0.38, 
                 WV=2.5, 
                 alpha=1.317,
                 beta=0.2606,
                 E_0_res=[],
                 a_oz_res=[],
                 a_ox_res=[],
                 a_wv_res=[],
                 E_dd_res=[],
                 E_dsa_res=[],
                 E_dsr_res=[]):
    """
    R_rs_surf implementation as described in Philipp Groetsch's 3C GitHub repo [1]

    [1] https://gitlab.com/pgroetsch/rrs_model_3C/-/blob/master/rrs_model_3C.py
    """
    if len(E_dd_res)==0:
        Edd = downwelling_irradiance.E_dd(wavelengths, theta_sun=theta_sun, P=P, AM=AM, RH=RH, H_oz=H_oz, WV=WV, alpha=alpha, beta=beta, E_0_res=E_0_res, a_oz_res=a_oz_res, a_ox_res=a_ox_res, a_wv_res=a_wv_res)
    else:
        Edd = E_dd_res

    if len(E_dsr_res)==0:
        Edsr = downwelling_irradiance.E_dsr(wavelengths, theta_sun=theta_sun, P=P, AM=AM, RH=RH, H_oz=H_oz, WV=WV, E_0_res=E_0_res, a_oz_res=a_oz_res, a_ox_res=a_ox_res, a_wv_res=a_wv_res)
    else: 
        Edsr = E_dsr_res

    if len(E_dsa_res)==0:
        Edsa = downwelling_irradiance.E_dsa(wavelengths, theta_sun=theta_sun, P=P, AM=AM, RH=RH, H_oz=H_oz, WV=WV, alpha=alpha, E_0_res=E_0_res, a_oz_res=a_oz_res, a_ox_res=a_ox_res, a_wv_res=a_wv_res)
    else:
        Edsa = E_dsa_res

    # Lines 116-120 in [1]
    Ed = Edd + Edsr + Edsa
    Edd_Ed = Edd / Ed
    Edsr_Ed = Edsr / Ed
    Edsa_Ed = Edsa / Ed
    Eds_Ed = Edsr_Ed + Edsa_Ed

    # Line 180 in [1]
    Rrs_refl = rho_s * Ls_Ed + rho_dd * Edd_Ed / np.pi + rho_ds * Eds_Ed / np.pi + delta

    return Rrs_refl
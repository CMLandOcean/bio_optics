import numpy as np
from ..atmosphere import sky_radiance, downwelling_irradiance, transmittance

def delta(wavelengths, 
                 rho_dd = 0.1, 
                 rho_ds = 0.1, 
                 delta = 0.0,
                 theta_sun=np.radians(30), 
                 P=1013.25, 
                 AM=1, 
                 RH=60, 
                 lambda_a=550,
                 alpha=1.317,
                 beta=0.2606):
    """
    R_rs_surf implementation as described in Philipp Groetsch's 3C GitHub repo [1].

    [1] https://gitlab.com/pgroetsch/rrs_model_3C/-/blob/master/rrs_model_3C.py
    """

    Tr = transmittance.T_r(wavelengths=wavelengths, theta_sun=theta_sun, P=P)
    Tas = transmittance.T_as(wavelengths=wavelengths, theta_sun=theta_sun, AM=AM, RH=RH, lambda_a=lambda_a, alpha=alpha, beta=beta)
    Fa = transmittance.F_a(theta_sun=theta_sun, alpha=alpha)

    # Lines 112-114 in [1]
    Edd = Tr * Tas
    Edsr = 0.5 * (1- Tr**0.95)
    Edsa = Tr**1.5 * (1 - Tas) * Fa

    # Lines 116-120 in [1]
    Ed = Edd + Edsr + Edsa
    Edd_Ed = Edd / Ed
    Edsr_Ed = Edsr / Ed
    Edsa_Ed = Edsa / Ed
    Eds_Ed = Edsr_Ed + Edsa_Ed

    # according to Line 180 in [1]
    delta = rho_dd * Edd_Ed / np.pi + rho_ds * Eds_Ed / np.pi + delta

    return delta
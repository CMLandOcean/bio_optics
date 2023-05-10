import numpy as np
from .. helper.utils import find_closest
from .. helper.resampling import resample_a_w, resample_b_bw
from .. surface.air_water import above2below

def qaa(R_rs, 
        wavelengths, 
        lambdas=np.array([412, 443, 490, 555, 640, 670]),
        g0=0.089,
        g1=0.1245,
        h0=-1.146,
        h1=-1.366,
        h2=-0.469,
        j_1=0.63, 
        j_2=0.88,
        a_w_res=[],
        b_bw_res=[]):
    """
    Quasi-Analytical Algorithm (QAA) to derive the absorption and backscattering coefficients by analytically inverting the spectral remote-sensing reflectance [sr-1] [1,2,3],
    and extended with CDOM estimation [4]. Only valid for optically deep water.
    Note that the equations follow the IOCCG updates [2,3] and are not identical to the ones provided in [4].
    The function can run with any number of bands as long as the crucial wavelengths (lambdas) are included.
 
    [1] Lee et al. (2002): Deriving inherent optical properties from water color: A multiband quasi-analytical algorithm for optically deep waters [10.1364/ao.41.005755]
    [2] Lee et al. (no year): An Update of the Quasi-Analytical Algorithm (QAA_v5). Available online: https://www.ioccg.org/groups/Software_OCA/QAA_v5.pdf
    [3] Lee et al. (2014): Update of the Quasi-Analytical Algorithm (QAA_v6). Available online: https://www.ioccg.org/groups/Software_OCA/QAA_v6_2014209.pdf
    [4] Zhu & Yu (2013): Inversion of chromophoric dissolved organic matter from EO-1 hyperion imagery for turbid estuarine and coastal waters [10.1109/TGRS.2012.2224117]

    Args:
        R_rs: remote sensing reflectance [sr-1]
        wavelengths: corresponding wavelengths [nm]
        lambdas: wavelengths used in QAA. Defaults to np.array([412, 443, 490, 555, 640, 670]).
        g0: Defaults to 0.089 [3].
        g1: Defaults to 0.1245 [3].
        h0: Defaults to -1.146 [2].
        h1: Defaults to -1.366 [2].
        h2: Defaults to -0.469 [2].
        j_1: Defaults to 0.63 [4].
        j_2: Defaults to 0.88 [4].
        a_w_res: absorption coefficient of pure water resampled to the sensors band setting [m-1]. Defaults to [], will be computed if not provided.
        b_bw_res: backscattering coefficient of pure water resampled to the sensors band setting [m-1]. Defaults to [], will be computed if not provided.

    Returns:
        a: bulk spectral absorption coefficient [m-1]
        a_ph: spectral absorption coefficient of phytoplankton [m-1]
        a_dg: spectral absorption coefficient of NAP and CDOM [m-1]
        a_p_440: particulate absorption coefficient [m-1] at 440 nm
        a_g_440: CDOM absorption coefficient [m-1] at 440 nm
        b_b: bulk spectral backscattering coefficient [m-1] 
        b_bp: spectral particulate backscattering coefficient [m-1]
    """
    
    idx = np.array([find_closest(wavelengths, wl)[1] for wl in lambdas]).astype(int)
    
    if len(a_w_res)==0:
        a_w_res = resample_a_w(wavelengths)
    if len(b_bw_res)==0:
        b_bw_res = resample_b_bw(wavelengths)

    # Step 0 [3]
    r_rs = above2below(R_rs)

    # Step 1 [3]
    u = (-g0 + np.sqrt(g0**2 + 4*g1 * r_rs)) / 2*g1

    # Step 2 [3]
    chi = np.log((r_rs[idx[1]] + r_rs[idx[2]]) / (r_rs[idx[3]] + 5*(r_rs[idx[5]]/r_rs[idx[2]]) * r_rs[idx[5]]))
    a_lambda_0_v5 = a_w_res[idx[3]] + 10**(h0 + h1*chi + h2*chi**2)
    a_lambda_0_v6 = a_w_res[idx[5]] + 0.39 * (R_rs[idx[5]] / R_rs[idx[1]] + R_rs[idx[2]])**1.14
    # a_lambda_0 = np.where(R_rs[idx[5]] < 0.0015, a_lambda_0_v5, a_lambda_0_v6)

    # Step 3 [3]
    b_bp_lambda_0_v5 = ((u[idx[3]] * a_lambda_0_v5) / (1-u[idx[3]])) - b_bw_res[idx[3]]
    b_bp_lambda_0_v6 = ((u[idx[5]] * a_lambda_0_v6) / (1-u[idx[3]])) - b_bw_res[idx[5]]
    
    # Step 4 [3]
    eta = 2.0 * (1 - 1.2 * np.exp(-0.9 * (r_rs[idx[1]]/r_rs[idx[3]])))

    # Step 5 [3]
    b_bp_v5 = np.asarray([b_bp_lambda_0_v5 * (wavelengths[idx[3]] / lambda_i)**eta for lambda_i in wavelengths])
    b_bp_v6 = np.asarray([b_bp_lambda_0_v6 * (wavelengths[idx[5]] / lambda_i)**eta for lambda_i in wavelengths])
    b_bp = np.where(R_rs[idx[5]] < 0.0015, b_bp_v5, b_bp_v6)
    
    # Step 6 [3]
    b_b = np.asarray([b_bw_res[i] + b_bp[i] for i in range(len(wavelengths))])
    a = (1-u) * b_b / u

    # Step 7 & 8 [3]
    zeta = 0.74 + (0.2 / (0.8 + r_rs[idx[1]] / r_rs[idx[3]]))
    S = 0.015 + (0.002 / (0.6 + r_rs[idx[1]] / r_rs[idx[3]]))
    xi = np.exp(S*(442.5-415.5))

    # Step 9 & 10 [3]
    a_dg_443 = ((a[idx[0]] - zeta*a[idx[1]]) / (xi - zeta)) - ((a_w_res[idx[0]] - zeta*a_w_res[idx[1]]) / (xi - zeta))
    a_dg = np.array([a_dg_443*np.exp(-S*(lambda_i-wavelengths[idx[1]])) for lambda_i in wavelengths])
    a_ph = a - a_dg - a_w_res[idx[1]]

    # CDOM estimation [4]
    a_p_440 = j_1 * b_bp[idx[1]]**j_2
    a_g_440 = a[idx[1]] - a_w_res[idx[1]] - a_p_440

    return a, a_ph, a_dg, a_p_440, a_g_440, b_b, b_bp
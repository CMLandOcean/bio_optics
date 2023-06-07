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
        a_w_res=[],
        b_bw_res=[]):
    """
    Quasi-Analytical Algorithm (QAA) to derive the absorption and backscattering coefficients by analytically inverting the spectral remote-sensing reflectance [sr-1] [1,2,3]. 
    Only valid for optically deep water.    
    The function can run with any number of bands as long as the crucial wavelengths (lambdas) are included.

    I believe that there is a typo in the QAA_v6 documentation (https://www.ioccg.org/groups/Software_OCA/QAA_v6_2014209.pdf): in Steps 9 & 10, where the first line 
    is a_g(443) while I think should be a_dg(443) as in the AQQ_v5 documentation (https://www.ioccg.org/groups/Software_OCA/QAA_v5.pdf).
 
    [1] Lee et al. (2002): Deriving inherent optical properties from water color: A multiband quasi-analytical algorithm for optically deep waters [10.1364/ao.41.005755]
    [2] Lee et al. (no year): An Update of the Quasi-Analytical Algorithm (QAA_v5). Available online: https://www.ioccg.org/groups/Software_OCA/QAA_v5.pdf
    [3] Lee et al. (2014): Update of the Quasi-Analytical Algorithm (QAA_v6). Available online: https://www.ioccg.org/groups/Software_OCA/QAA_v6_2014209.pdf

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
    s = 0.015 + (0.002 / (0.6 + r_rs[idx[1]] / r_rs[idx[3]]))
    xi = np.exp(s*(442.5-415.5))

    # Step 9 & 10 [3]
    a_dg_443 = ((a[idx[0]] - zeta*a[idx[1]]) / (xi - zeta)) - ((a_w_res[idx[0]] - zeta*a_w_res[idx[1]]) / (xi - zeta))
    a_dg = np.array([a_dg_443*np.exp(-s*(lambda_i-wavelengths[idx[1]])) for lambda_i in wavelengths])
    a_ph = a - a_dg - a_w_res[idx[1]]

    return a, a_ph, a_dg, b_b, b_bp


def qaa_cdom(R_rs, 
             wavelengths, 
             lambdas=np.array([440, 490, 555, 640]),
             k0=6.807,
             k1=1.186,
             h0=-1.169,
             h1=-1.468,
             h2=0.274,
             j1=0.63,
             j2=0.88,
             a_w_res=[],
             b_bw_res=[]):
    """
    Quasi-Analytical Algorithm (QAA) to derive absorption of CDOM and particulate matter at a reference wavelength of 440 nm [1] based on the QAA_v5 algorithm [2,3]. 
    Only valid for optically deep water.

    Zhu & Yu made some empirical adjustments based on the NOMAD data set. These adjustments (e.g., computation of u in Step 1) are incorporated in this function,
    except for the Gamma adjustment in u(). 

    !!! Note that this implementation requires R_rs and wavelengths to contain only the relevant bands !!!

    [1] Zhu & Yu (2013): Inversion of chromophoric dissolved organic matter from EO-1 hyperion imagery for turbid estuarine and coastal waters [10.1109/TGRS.2012.2224117]
    [2] Lee et al. (2002): Deriving inherent optical properties from water color: A multiband quasi-analytical algorithm for optically deep waters [10.1364/ao.41.005755]
    [3] Lee et al. (no year): An Update of the Quasi-Analytical Algorithm (QAA_v5). Available online: https://www.ioccg.org/groups/Software_OCA/QAA_v5.pdf

    Args:
        R_rs: remote sensing reflectance [sr-1] for selected bands (lambdas)
        wavelengths: corresponding wavelengths [nm]
        lambdas: wavelengths used in QAAV5. Defaults to np.array([440, 490, 555, 640]).
        k0: Defaults to 6.807 [1].
        k1: Defaults to 1.186 [1].
        h0: Defaults to -1.169 [1].
        h1: Defaults to -1.468 [1].
        h2: Defaults to 0.274 [1].
        j_1: Defaults to 0.63 [1].
        j_2: Defaults to 0.88 [1].
        a_w_res: absorption coefficient of pure water resampled to the sensors band setting [m-1]. Defaults to [], will be computed if not provided.
        b_bw_res: backscattering coefficient of pure water resampled to the sensors band setting [m-1]. Defaults to [], will be computed if not provided.

    Returns:
        a_p_440: particulate absorption coefficient [m-1] at 440 nm
        a_g_440: CDOM absorption coefficient [m-1] at 440 nm
    """
    
    idx = np.array([find_closest(wavelengths, wl)[1] for wl in lambdas]).astype(int)
    
    if len(a_w_res)==0:
        a_w_res = resample_a_w(wavelengths)
    if len(b_bw_res)==0:
        b_bw_res = resample_b_bw(wavelengths)

    # Step 0 [1]
    r_rs = above2below(R_rs, Gamma=[1.8, 2.0, 2.1, 2.2])

    # Step 1 [1]
    u = 1 - np.exp((-k0 * r_rs**k1)/(0.31 - r_rs))

    # Step 2 [1]
    chi = np.log((r_rs[idx[0]] + r_rs[idx[1]]) / (r_rs[idx[2]] + 2*(r_rs[idx[3]]/r_rs[idx[2]]) * r_rs[idx[3]]))
    a_555 = a_w_res[idx[2]] + 10**(h0 + h1*chi + h2*chi**2)

    # Step 3 [1]
    b_bp_555 = ((u[idx[2]] * a_555) / (1-u[idx[2]])) - b_bw_res[idx[2]]
    
    # Step 4 [1]
    eta = 2.2 * (1 - 1.2 * np.exp(-0.9 * (r_rs[idx[0]]/r_rs[idx[2]])))
    b_bp = np.asarray([b_bp_555 * (wavelengths[idx[2]] / lambda_i)**eta for lambda_i in wavelengths])
    
    # Step 5 [1]
    a_440 = (1-u[idx[0]]) * (b_bw_res[idx[0]] + b_bp[idx[0]]) / u[idx[0]]

    # Step 6 [1]
    a_p_440 = j1 * b_bp[idx[2]]**j2

    # Step 7 [1]
    a_g_440 = a_440 - a_w_res[idx[0]] - a_p_440

    return a_p_440, a_g_440
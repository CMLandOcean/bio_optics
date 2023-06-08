import numpy as np
from lmfit import minimize, Parameters
from .. water import absorption, backscattering, temperature_gradient, attenuation, bottom_reflectance
from .. models import hope
from .. surface import surface, air_water
from .. helper import resampling

def r_rs_sh(C_Mie = 0.02,           # represents X from Eq. 19 [2]
            C_Y = 0,                # represents G from Eq. 17 [2]
            zB = 2,                 # represents H from Eq. 9 [2]
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
            lambda_0 = 440,
            lambda_S = 555,
            S = 0.015,
            b_bMie_spec = 1,        # must be 1 so C_Mie can represent X
            n = -1,                 # should be estimated using utils.estimate_y()*(-1), varies between 0 and -2.5 [1]
            fresh = False,
            q = 0.75,
            g_0 = 0.089, 
            g_1 = 0.125,
            wavelengths = np.arange(400,800),
            a_w_res=[],
            b_bw_res=[],
            R_i_b_res=[]):
    """
    Shallow water bio-optical properties (SBOP) model after Li et al. (2017) [1].
    
    In Eq. 10 [1] b_bp is defined as: 
        b_bp = P * (wavelengths/555)**y
    Almost the same formulation is used in Albert and Mobley (2003) [3] for b_bMie, thus b_bp can be exchanged with b_bMie(lambda_S=555, n=-y, b_bMie_spec=1). 
    When b_bMie_spec = 1, C_Mie represents P. The exponent y needs to be multiplied by -1.

    Instead of representing bottom albedo as a spectrum normalized at 560 nm and scaled with fit parameter B, we use the implementation of [3] to model bottom albedo as a mixture of
    up to 6 bottom types.

    [1] Li et al. (2017): Remote sensing estimation of colored dissolved organic matter (CDOM) in optically shallow waters [10.1016/j.isprsjprs.2017.03.015]
    [2] Lee et al. (1999): Hyperspectral remote sensing for shallow waters: 2 Deriving bottom depths and water properties by optimization [10.1364/ao.38.003831]
    [3] Albert & Mobley (2003): An analytical model for subsurface irradiance and remote sensing reflectance in deep and shallow case-2 waters. [10.1364/OE.11.002873]

    Args:
        C_Mie: concentration of non-algal particles type II [mg L-1] from [3], represents particulate backscattering coefficient (plus viewing angle and sea state) at lambda_S=555 nm (P) from Eq. 10 [1] when b_bMie_spec==1, default: 0
        C_Y: CDOM absorption coefficient at lambda_0 [m-1], default: 0
        a_phy_440: phytoplankton absorption coefficient at 440 nm, default: 0
        zB: water depth [m], default: 2
        f_0: fractional cover of bottom type 0, default: 0
        f_1: fractional cover of bottom type 1, default: 0
        f_2: fractional cover of bottom type 2, default: 0
        f_3: fractional cover of bottom type 3, default: 0
        f_4: fractional cover of bottom type 4, default: 0
        f_5: fractional cover of bottom type 5, default: 0
        B_0: proportion of radiation reflected towards the sensor from bottom type 0, default: 1/np.pi
        B_1: proportion of radiation reflected towards the sensor from bottom type 1, default: 1/np.pi
        B_2: proportion of radiation reflected towards the sensor from bottom type 2, default: 1/np.pi
        B_3: proportion of radiation reflected towards the sensor from bottom type 3, default: 1/np.pi
        B_4: proportion of radiation reflected towards the sensor from bottom type 4, default: 1/np.pi
        B_5: proportion of radiation reflected towards the sensor from bottom type 5, default: 1/np.pi
        lambda_0: reference wavelength for CDOM and NAP absorption [nm], default: 440 nm
        lambda_S: reference wavelength for scatteromg of particles type II [nm] , default: 400 nm
        S: spectral slope of CDOM absorption spectrum [nm-1], default: 0.015
        b_bMie_spec: specific backscattering coefficient of non-algal particles type II [m2 g-1] from [3] but used here to compute b_bp' and must be 1, default: 1
        n: Angström exponent of particle type II backscattering usually called y or Y in Lee's work, should be estimated using utils.estimate_y()*(-1), default: -1
        fresh: boolean to decide if to compute b_bw for fresh or oceanic water, default: False
        q: empirical ratio of a_p and b_bp [1], default: 0.75
        g_0: Empirical value. Defaults to 0.089.
        g_1: Empirical value. Defaults to 0.125.
        wavelengths: wavelengths to compute r_rs_sh for [nm], default: np.arange(400,800) 
        a_w_res: optional, absorption of pure water resampled to sensor's band settings. Will be computed within function if not provided.
        A_res: optional, parameters for the empirical a_Phi(lambda) simulation resampled to sensor's band settings. Will be computed within function if not provided.
        b_bw_res: optional, precomputing b_bw b_bw saves a lot of time during inversion. Will be computed within function if not provided.
        R_i_b_res: optional, preresampling R_i_b before inversion saves a lot of time. Will be computed within function if not provided.

    Returns:
        r_rs_sh: subsurface radiance reflectance [sr-1] of shallow water
    """
    bs = backscattering.b_bw(wavelengths=wavelengths, fresh=fresh, b_bw_res=b_bw_res) + \
         backscattering.b_bMie(C_Mie=C_Mie, wavelengths=wavelengths, b_bMie_spec=b_bMie_spec, lambda_S=lambda_S, n=n)
    
    ab = absorption.a_w(wavelengths=wavelengths, a_w_res=a_w_res) + \
         absorption.a_Y(wavelengths=wavelengths, C_Y=C_Y, S=S, lambda_0=lambda_0) + \
         q * backscattering.b_bMie(C_Mie=C_Mie, wavelengths=wavelengths, b_bMie_spec=b_bMie_spec, lambda_S=lambda_S, n=n)

    kappa = ab + bs    
    u = bs / kappa
    
    r_rs_sh = hope.r_rs_dp(u, g_0=g_0, g_1=g_1) * (1 - np.exp(-hope.D_u_C(u, f1=1, f2=2.4) * kappa * zB)) + \
              bottom_reflectance.R_rs_b(f_0=f_0, f_1=f_1, f_2=f_2, f_3=f_3, f_4=f_4, f_5=f_5, B_0=B_0, B_1=B_1, B_2=B_2, B_3=B_3, B_4=B_4, B_5=B_5, wavelengths=wavelengths, R_i_b_res=R_i_b_res) * \
              np.exp(-hope.D_u_B(u, f1=1, f2=5.5) * kappa * zB)
                    
    return r_rs_sh


def func2opt(params, 
             R_rs,
             wavelengths, 
             weights = [],
             a_w_res=[],
             b_bw_res=[],
             R_i_b_res=[]):
    """_summary_

    Args:
        params (_type_): _description_
        R_rs (_type_): _description_
        wavelengths (_type_): _description_
        weights (list, optional): _description_. Defaults to [].
        a_w_res (list, optional): _description_. Defaults to [].
        b_bw_res (list, optional): _description_. Defaults to [].
        R_i_b_res (list, optional): _description_. Defaults to [].

    Returns:
        _type_: _description_
    """
    r_rs = air_water.above2below(R_rs)
    
    r_rs_sim = r_rs_sh(wavelengths = wavelengths,
                       C_Mie = params['C_Mie'],    
                       C_Y = params['C_Y'], 
                       zB = params['zB'], 
                       f_0 = params['f_0'],
                       f_1 = params['f_1'],
                       f_2 = params['f_2'],
                       f_3 = params['f_3'],
                       f_4 = params['f_4'],
                       f_5 = params['f_5'],
                       B_0 = params['B_0'],
                       B_1 = params['B_1'],
                       B_2 = params['B_2'],
                       B_3 = params['B_3'],
                       B_4 = params['B_4'],
                       B_5 = params['B_5'],
                       lambda_0 = params['lambda_0'],
                       lambda_S = params['lambda_S'],
                       S = params['S'],
                       b_bMie_spec = params['b_bMie_spec'],
                       n = params['n'],
                       fresh = params['fresh'],
                       q = params['q'],
                       g_0 = params['g_0'],
                       g_1 = params['g_1'],
                       a_w_res=a_w_res,
                       b_bw_res=b_bw_res,
                       R_i_b_res=R_i_b_res) + params['offset']
    
    error_method = params['error_method']    
    
    if error_method == 1:
        # least squares
        err = (np.abs(r_rs-r_rs_sim))**2
    elif error_method == 2:
        # absolute differences
        err = np.abs(r_rs-r_rs_sim) * weights
    elif error_method == 3:
        # relative differences
        err = np.abs(1 - r_rs_sim/r_rs)
    elif error_method == 4:
        # the one described in Li et al. (2017)
        err = np.sqrt(np.sum((r_rs - r_rs_sim)**2)) / np.sqrt(np.sum(r_rs))
    elif error_method == 5:
        # absolute percentage difference
        err = np.sqrt(np.sum((r_rs - r_rs_sim)**2)) / np.sum(r_rs)

    return err


def invert(params, 
           R_rs, 
           wavelengths,
           weights = [],
           a_w_res=[],
           b_bw_res=[],
           R_i_b_res=[],
           method="least-squares", 
           max_nfev=400
           ):
    """
    Function to inversely fit a modeled spectrum to a measurement spectrum.
    
    :param params: lmfit Parameters object containing all Parameter objects that are required to specify the model
    :param R_rs: Remote sensing reflectance spectrum [sr-1]
    :param wavelengths: wavelengths of R_rs bands [nm]
    :param weights: spectral weighing coefficients
    :param a_w_res: optional, absorption of pure water resampled to sensor's band settings. Will be computed within function if not provided.
    :param b_bw_res: optional, precomputing b_bw b_bw saves a lot of time during inversion. Will be computed within function if not provided.
    :param R_i_b_res: optional, preresampling R_i_b before inversion saves a lot of time. Will be computed within function if not provided.
    :param method: name of the fitting method to use by lmfit, default: 'least-squares'
    :param max_nfev: maximum number of function evaluations, default: 400
    :return: object containing the optimized parameters and several goodness-of-fit statistics.
    """ 

    res = minimize(func2opt,
                   params, 
                   args=(R_rs,
                         wavelengths, 
                         weights,
                         a_w_res, 
                         b_bw_res, 
                         R_i_b_res), 
                   method=method, 
                   max_nfev=max_nfev) 
    return res


def forward(params,
            wavelengths,
            a_w_res=[],
            b_bw_res = [],
            R_i_b_res = [],):
    """
    Forward simulation of a shallow water remote sensing reflectance spectrum based on the provided parameterization.
    
    :param params: lmfit Parameters object containing all Parameter objects that are required to specify the model
    :param wavelengths: wavelengths of R_rs bands [nm]
    :param a_w_res: optional, absorption of pure water resampled to sensor's band settings. Will be computed within function if not provided.
    :param b_bw_res: optional, precomputing b_bw b_bw saves a lot of time . Will be computed within function if not provided.
    :param R_i_b_res: optional, preresampling R_i_b saves a lot of time. Will be computed within function if not provided.
    :return: R_rs: simulated remote sensing reflectance spectrum [sr-1]
    """
    R_rs_sim = air_water.below2above(
                            r_rs_sh(wavelengths = wavelengths,
                                    C_Mie = params['C_Mie'],
                                    C_Y = params['C_Y'], 
                                    zB = params['zB'], 
                                    f_0 = params['f_0'],
                                    f_1 = params['f_1'],
                                    f_2 = params['f_2'],
                                    f_3 = params['f_3'],
                                    f_4 = params['f_4'],
                                    f_5 = params['f_5'],
                                    B_0 = params['B_0'],
                                    B_1 = params['B_1'],
                                    B_2 = params['B_2'],
                                    B_3 = params['B_3'],
                                    B_4 = params['B_4'],
                                    B_5 = params['B_5'],
                                    lambda_0 = params['lambda_0'],
                                    lambda_S = params['lambda_S'],
                                    S = params['S'],
                                    b_bMie_spec = params['b_bMie_spec'],
                                    n = params['n'],
                                    fresh = params['fresh'],
                                    q = params['q'],
                                    g_0 = params['g_0'],
                                    g_1 = params['g_1'],
                                    a_w_res=a_w_res,
                                    b_bw_res=b_bw_res,
                                    R_i_b_res=R_i_b_res) + params['offset'])
    
    return R_rs_sim
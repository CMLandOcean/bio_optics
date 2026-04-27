import numpy as np
from .. import absorption, backscattering, attenuation, bottom_reflectance
from ...surface import air_water
from ...helper import resampling, utils


def D_u_C(u, f1=1.03, f2=2.4):
    """
    Path elongation factor for scattered photons from the water body after Eq. 5 in [1].

    [1] Lee et al. (1999): Hyperspectral remote sensing for shallow waters: 2 Deriving bottom depths and water properties by optimization [10.1364/ao.38.003831]

    Args:
        u: single scattering albedo bb / (a + bb) [dimensionless]
        f1 (float, optional): empirical constant, default: 1.03
        f2 (float, optional): empirical constant, default: 2.4

    Returns:
        D_u_C: path elongation factor for photons scattered in the water body [dimensionless]
    """
    D_u_C = f1 * (1 + f2 * u)**0.5

    return D_u_C


def D_u_B(u, f1=1.04, f2=5.4):
    """
    Path enlongation factor for scattered photons from the bottom after Eq. 5 in [1].

    [1] Lee et al. (1999): Hyperspectral remote sensing for shallow waters: 2 Deriving bottom depths and water properties by optimization [10.1364/ao.38.003831]

    Args:
        u: single scattering albedo bb / (a + bb) [dimensionless]
        f1 (float, optional): empirical constant, default: 1.04
        f2 (float, optional): empirical constant, default: 5.4

    Returns:
        D_u_B: path elongation factor for photons scattered from the bottom [dimensionless]
    """
    D_u_B = f1 * (1 + f2 * u)**0.5

    return D_u_B


def rrs_dp(u, g_0=0.084, g_1=0.170):
    """
    Subsurface radiance reflectance [sr-1] for optically deep water after Eq. 4 in [1].

    [1] Lee et al. (1999): Hyperspectral remote sensing for shallow waters: 2 Deriving bottom depths and water properties by optimization [10.1364/ao.38.003831]

    Args:
        u: single scattering albedo bb / (a + bb) [dimensionless]
        g_0 (float, optional): empirical constant [sr-1], default: 0.084
        g_1 (float, optional): empirical constant [sr-1], default: 0.17

    Returns:
        rrs_dp: subsurface radiance reflectance of optically deep water [sr-1]
    """
    rrs_dp = (g_0 + g_1 * u) * u

    return rrs_dp


def rrs_sh(C_Mie = 0,              # represents X from Eq. 19 [2]
            C_Y = 0,           # represents G from Eq. 17 [2]
            a_phy_440 = 1e-10,     # represents P from Eq. 16 [2]
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
            lambda_S = 400,
            S = 0.015,
            bb_Mie_spec = 1,        # must be 1 so C_Mie can represent X
            n = -1,                 # should be estimated using utils.estimate_y()*(-1), varies between 0 and -2.5 [1]
            fresh = False,
            n1 = 1,
            n2 = 1.33,
            g_0 = 0.084, 
            g_1 = 0.17,
            theta_sun = np.radians(30),
            wavelengths = np.arange(400,800),
            a_w_res=[],
            A_res=[],
            bb_w_res=[],
            R_b_i_res=[]):
    """
    Subsurface reflectance of shallow water [sr-1] after Lee et al. (1998, 1999) [1,2].
    
    In Eq. 19 [2] b_bp' is defined as: 
        b_bp' = X * (400/wavelengths)**Y
    Where X combines the particle- backscattering coefficient, viewing-angle information, as well as sea state into one variable [2]. 
    Almost the same formulation is used in Albert and Mobley (2003) [3] for b_bMie, thus b_bp' can be exchanged with b_bMie(lambda_S=400, n=-Y, bb_Mie_spec=1). 
    When bb_Mie_spec = 1, C_Mie represents X. The exponent Y needs to be multiplied by -1 because the wavelength ratio is the opposite in [3].

    Instead of representing bottom albedo as a spectrum normalized at 560 nm and scaled with fit parameter B, we use the implementation of [3] to model bottom albedo as a mixture of
    up to 6 bottom types.

    [1] Lee et al. (1998): Hyperspectral remote sensing for shallow waters: 1 A semianalytical model [10.1364/AO.37.006329]
    [2] Lee et al. (1999): Hyperspectral remote sensing for shallow waters: 2 Deriving bottom depths and water properties by optimization [10.1364/ao.38.003831]
    [3] Albert & Mobley (2003): An analytical model for subsurface irradiance and remote sensing reflectance in deep and shallow case-2 waters. [10.1364/OE.11.002873]

    Args:
        C_Mie: concentration of non-algal particles type II [mg L-1] from [3], represents particulate backscattering coefficient (plus viewing angle and sea state) at lambda_S=400 nm (X) from Eq. 19 [2] when bb_Mie_spec==1, default: 0
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
        bb_Mie_spec: specific backscattering coefficient of non-algal particles type II [m2 g-1] from [3] but used here to compute b_bp' and must be 1, default: 1
        n: Angström exponent of particle type II backscattering usually called y or Y in Lee's work, should be estimated using utils.estimate_y()*(-1), default: -1
        fresh: boolean to decide if to compute bb_w for fresh or oceanic water, default: False
        n1: refractive index of origin medium, default: 1 for air
        n2: refractive index of destination medium, default: 1.33 for water
        g_0: Empirical value. Defaults to 0.084.
        g_1: Empirical value. Defaults to 0.17.
        theta_sun: sun zenith angle [radians], default: np.radians(30)
        wavelengths: wavelengths to compute Rrs_sh for [nm], default: np.arange(400,800) 
        a_w_res: optional, absorption of pure water resampled to sensor's band settings. Will be computed within function if not provided.
        A_res: optional, parameters for the empirical a_Phi(lambda) simulation resampled to sensor's band settings. Will be computed within function if not provided.
        bb_w_res: optional, precomputing bb_w bb_w saves a lot of time during inversion. Will be computed within function if not provided.
        R_b_i_res: optional, preresampling R_b_i before inversion saves a lot of time. Will be computed within function if not provided.

    Returns:
        rrs_sh: subsurface radiance reflectance [sr-1] of shallow water
    """
    bs = backscattering.bb_w(wavelengths=wavelengths, fresh=fresh, bb_w_res=bb_w_res) + \
         backscattering.bb_Mie(C_Mie=C_Mie, wavelengths=wavelengths, bb_Mie_spec=bb_Mie_spec, lambda_S=lambda_S, n=n)
    
    ab = absorption.a_w(wavelengths=wavelengths, a_w_res=a_w_res) + \
         absorption.a_Y(wavelengths=wavelengths, C_Y=C_Y, S=S, lambda_0=lambda_0) + \
         absorption.a_Phi(wavelengths=wavelengths, a_phy_440=a_phy_440, A_res=A_res)

    kappa = ab + bs    
    u = bs / kappa
    
    rrs_sh = (rrs_dp(u=u, g_0=g_0, g_1=g_1) * (1 - np.exp(-kappa * zB * (1/np.cos(air_water.snell(theta_inc=theta_sun, n1=n1, n2=n2)) + D_u_C(u=u))))) + \
              bottom_reflectance.Rrs_b(f_0=f_0, f_1=f_1, f_2=f_2, f_3=f_3, f_4=f_4, f_5=f_5, B_0=B_0, B_1=B_1, B_2=B_2, B_3=B_3, B_4=B_4, B_5=B_5, wavelengths=wavelengths, R_b_i_res=R_b_i_res) * \
              np.exp(-kappa * zB * (1/np.cos(air_water.snell(theta_inc=theta_sun, n1=n1, n2=n2)) + D_u_B(u=u)))
                
    return rrs_sh


def forward(params,
            wavelengths,
            a_w_res=[],
            A_res = [],
            bb_w_res = [],
            R_i_b_res = [],):
    """
    Forward simulation of a shallow water remote sensing reflectance spectrum based on the provided parameterization.
    
    Args:
        params: lmfit Parameters object specifying the model configuration
        wavelengths: wavelengths [nm]
        a_w_res: optional precomputed pure water absorption [m-1]
        A_res: optional precomputed empirical phytoplankton absorption parameters
        bb_w_res: optional precomputed water backscattering coefficient [m-1]
        R_i_b_res: optional precomputed bottom reflectance spectra

    Returns:
        Rrs_sim: simulated above-water remote sensing reflectance [sr-1]
    """
    Rrs_sim = air_water.below2above(
                            rrs_sh(wavelengths = wavelengths,
                                    C_Mie = params['C_Mie'],
                                    C_Y = params['C_Y'], 
                                    a_phy_440 = params['a_phy_440'], 
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
                                    bb_Mie_spec = params['bb_Mie_spec'],
                                    n = params['n'],
                                    fresh = params['fresh'],
                                    n1 = params['n1'],
                                    n2 = params['n2'],
                                    g_0 = params['g_0'],
                                    g_1 = params['g_1'],
                                    theta_sun = params['theta_sun'],
                                    a_w_res=a_w_res,
                                    A_res=A_res,
                                    bb_w_res=bb_w_res,
                                    R_b_i_res=R_i_b_res) + params['offset'])
    
    return Rrs_sim



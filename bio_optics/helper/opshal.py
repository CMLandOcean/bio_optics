import numpy as np
from .. models.qaa import qaa_shallow
from .. water.attenuation import estimate_c
from .. helper.resampling import resample_a_w, resample_bb_w
from .. helper.utils import find_closest


def estimate_zeta(c, depth):
    """
    Estimated optical depth following Eq. 6 in McKinna & Werdell (2018) [1]

    [1] McKinna & Werdell (2018): Approach for identifying optically shallow pixels when processing ocean-color imagery [10.1364/OE.26.00A915]

    Args:
        c: beam attenuation coefficien [m-1]. Can be computed using estimate_c in water.attenuation. [1] compute c for 547 nm only.
        depth: geometric water depth [m], Delta_z in [1].

    Returns:
        zeta_E: Estimated optical depth
    """
    zeta_E = c * depth

    return zeta_E


def opshal(Rrs, 
           wavelengths,
           depth,
           lambda_0=547,
           threshold=20,
           lambdas=np.array([412, 443, 490, 555, 640, 670]),
           g0=0.089,
           g1=0.125,
           eta_p=0.015,
           eta_w=0.5,
           a_w_res=[],
           bb_w_res=[]):
    """
    Approach for identifying optically shallow pixels when processing ocean-color imagery following McKinna & Werdell (2018) [1] based on an adapted version of the QAA.

    [1] McKinna & Werdell (2018): Approach for identifying optically shallow pixels when processing ocean-color imagery [10.1364/OE.26.00A915]

    Args:
        Rrs: remote sensing reflectance [sr-1]. For matrices, bands need to be first axis.
        wavelengths: corresponding wavelengths [nm]
        depth: geometrical depth [m]
        lambda_0 (int, optional): reference wavelength [nm]. Defaults to 547.
        threshold (int, optional): threshold for binary mask. Defaults to 20.
        lambdas (optional): wavelengths used in QAA. Defaults to np.array([412, 443, 490, 555, 640, 670]).
        g0: Defaults to 0.089 [1].
        g1: Defaults to 0.125 [1].
        eta_p (float, optional): particulate backscatter ratio. Defaults to 0.015; halfway between global average oceanic value of 0.01 and well known Petzold average particle value of 0.0183.
        eta_w (float, optional): backscatter ratio of pure water. Defaults to 0.5.
        a_w_res: absorption coefficient of pure water resampled to the sensors band setting [m-1]. Defaults to [], will be computed if not provided.
        bb_w_res: backscattering coefficient of pure water resampled to the sensors band setting [m-1]. Defaults to [], will be computed if not provided.

    Returns:
        is_optically_shallow: binary mask where optically shallow is True.
    """
       
    if len(a_w_res)==0:
        a_w_res = resample_a_w(wavelengths)
    if len(bb_w_res)==0:
        bb_w_res = resample_bb_w(wavelengths)
    
    # Step 1: Estimate a_t, b_bp and b_bw using the qaa_shallow
    a, bb, bb_p = qaa_shallow(Rrs=Rrs, wavelengths=wavelengths, lambdas=lambdas, g0=g0, g1=g1, a_w_res=a_w_res, bb_w_res=bb_w_res)

    # Step 2: Estimate c
    if len(Rrs.shape)==1:
        c = estimate_c(a, bb_p, bb_w_res, eta_p=eta_p, eta_w=eta_w)
    elif len(Rrs.shape)==2:
        c = estimate_c(a, bb_p, bb_w_res[:, np.newaxis], eta_p=eta_p, eta_w=eta_w)
    elif len(Rrs.shape)==3:
        c = estimate_c(a, bb_p, bb_w_res[:, np.newaxis, np.newaxis], eta_p=eta_p, eta_w=eta_w)

    # Step 3: Estimate zeta
    zeta_E = estimate_zeta(c, depth)

    # Step 4: Create binary mask (optically shallow = True)
    is_optically_shallow = np.where(zeta_E[find_closest(wavelengths, lambda_0)[1]] < threshold, True, False)

    return is_optically_shallow




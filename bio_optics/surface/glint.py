import numpy as np

from .. helper import resampling, utils
from . import air_water


def gao(R, wavelengths, theta_sun=0.001, lambda_nir=1640, n1=1, n2=[]):
    """
    Sun glint correction considering the spectral variation of the refractive index of water [1].
    Assumes zero reflectance of water in the infrared.

    [1] Gao & Li (2021): Correction of Sunglint Effects in High Spatial Resolution Hyperspectral Imagery Using SWIR or NIR Bands and Taking Account of Spectral Variation of Refractive Index of Water [10.21926/aeer.2103017]

    Args:
        R: array of one ore more spectra in units of Reflectance [-]
        wavelengths: corresponding wavelengths [nm]
        theta_sun: solar zenith angle [radians]. Defaults to 0.001.
        lambda_nir: wavelength [nm] of infrared band where reflectance is assumed to be negligible. Defaults to 1640.
        n1 (int, optional): Refractive index of origin medium, default: 1 for air
        n2 (float, optional): Refractive index of destination medium (water), should be pre-resampled using resample_n() from the resampling module and passed to this function. 
                              If a constant value (e.g., 1.33) is used, glint is considered to be spectrally uniform and this function becomes similar to other glint correction methods (e.g., Hedley), default: [].

    Returns:
        glint reflectance [-]
    """
    if len(n2)==0:
         n2 = resampling.resample_n(wavelengths=wavelengths)

    fresnel_reflectance = air_water.fresnel(theta_inc=theta_sun, n1=n1, n2=n2)

    RTO_B_Ref = R[utils.find_closest(wavelengths, lambda_nir)[1]] / fresnel_reflectance[utils.find_closest(wavelengths, lambda_nir)[1]]

    if len(R.shape)==3:
          sun_glint = np.einsum('i,jk->ijk', fresnel_reflectance, RTO_B_Ref)
    elif len(R.shape)==2:
         sun_glint = np.einsum('i,j->ij', fresnel_reflectance, RTO_B_Ref)
    elif len(R.shape)==1:
         sun_glint = fresnel_reflectance * RTO_B_Ref
         
    return sun_glint
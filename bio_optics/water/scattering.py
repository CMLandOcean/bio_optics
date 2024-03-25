import numpy as np
from .. helper import resampling


def b(a,c):
    """
    Compute scattering coefficients from absorption and attenuation coefficients.

    Args:
        a (np.array): Absorption coefficient.
        c (np.array): Attenuation coefficient.

    Returns:
        b: Scattering coefficient.
    """
    b = c - a
    return b


def b_phy(C_0 = 0,
          C_1 = 0,
          C_2 = 0,
          C_3 = 0,
          C_4 = 0,
          C_5 = 0,
          C_6 = 0,
          C_7 = 0,
          wavelengths = np.arange(400,800),
          b_i_spec_res = []):
    """
    Spectral scattering coefficient of phytoplankton for a mixture of up to 6 phytoplankton classes (C_0..C_5).
    
    :param C_0: concentration of phytoplankton type 0 [ug/L], default: 0
    :param C_1: concentration of phytoplankton type 1 [ug/L], default: 0
    :param C_2: concentration of phytoplankton type 2 [ug/L], default: 0
    :param C_3: concentration of phytoplankton type 3 [ug/L], default: 0
    :param C_4: concentration of phytoplankton type 4 [ug/L], default: 0
    :param C_5: concentration of phytoplankton type 5 [ug/L], default: 0
    :param C_6: concentration of phytoplankton type 6 [ug/L], default: 0
    :param C_7: concentration of phytoplankton type 7 [ug/L], default: 0
    :wavelengths: wavelengths to compute a_ph for [nm], default: np.arange(400,800)
    :param b_i_spec_res: optional, preresampling b_i_spec (scattering coefficient of phytoplankton types C_0..C_7) before inversion saves a lot of time.
    :return: spectral scattering coefficient of phytoplankton mixture
    """
    C_i = np.array([C_0,C_1,C_2,C_3,C_4,C_5,C_6,C_7])
    
    if len(b_i_spec_res)==0:
        b_i_spec = resampling.resample_b_i_spec_EnSAD(wavelengths=wavelengths)
    else:
        b_i_spec = b_i_spec_res
    
    b_phy = 0
    for i in range(b_i_spec.shape[1]): b_phy += C_i[i] * b_i_spec[:, i]
    
    return b_phy
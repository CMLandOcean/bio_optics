import numpy as np


def brightness_normalization(spectrum: np.array):
    """
    Brightness normalization for single spectra and images.

    :param spectrum: np.array (first axis needs to be bands)
    :return: brightness normalized image or spectrum as np.array
    """
    return spectrum / np.linalg.norm(spectrum, axis=0)
    
    
    
def wavelength_normalization(spectrum: np.array, band: int):
    """
    Wavelength normalization for single spectra and images.
    
    :param spectrum: np.array (first axis needs to be bands)
    :param band: band index of wavelength for normalization
    :return: image or spectrum normalized at wavelength as np.array
    """
    return spectrum / spectrum[band]
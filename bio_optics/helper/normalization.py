import numpy as np


def brightness_normalization(spectrum: np.array):
    """
    Brightness normalization for single spectra and images.

    Args:
        spectrum: spectral array with bands on the first axis

    Returns:
        brightness-normalised spectrum or image
    """
    return spectrum / np.linalg.norm(spectrum, axis=0)



def wavelength_normalization(spectrum: np.array, band: int):
    """
    Wavelength normalization for single spectra and images.

    Args:
        spectrum: spectral array with bands on the first axis
        band: band index of the wavelength to normalise by

    Returns:
        spectrum normalised at the specified wavelength band
    """
    return spectrum / spectrum[band]
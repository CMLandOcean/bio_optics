import numpy as np
from . import utils


def rgb_to_hex(rgb):
    """
    Convert an BGR-ordered RGB triplet to a hex color string.

    Args:
        rgb: array-like of length 3 in BGR order (as returned by OpenCV-style indexing)

    Returns:
        hex_color: six-character lowercase hex string (without '#' prefix)
    """
    return '%02x%02x%02x' % (rgb[2], rgb[1], rgb[0])


def spectrum_to_hex(spectrum, wavelengths, r=610, g=550, b=480):
    """
    Convert a spectrum to a hex color code by extracting and normalising red, green, and blue bands.

    Args:
        spectrum: spectral array (bands on first axis)
        wavelengths: wavelengths corresponding to spectrum bands [nm]
        r: wavelength of the red band [nm], default: 610
        g: wavelength of the green band [nm], default: 550
        b: wavelength of the blue band [nm], default: 480

    Returns:
        hex_color: hex color string with '#' prefix
    """
    rgb = spectrum[np.isin(wavelengths, [utils.find_closest(wavelengths, r),utils.find_closest(wavelengths, g),utils.find_closest(wavelengths, b)])]
    return '#' + rgb_to_hex(np.round(rgb / np.max(rgb) * 255).astype(int))
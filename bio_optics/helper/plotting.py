import numpy as np

def rgb_to_hex(rgb):
    """
    Convert an RGB into hex color code.
    """
    return '%02x%02x%02x' % (rgb[2], rgb[1], rgb[0]) 


def spectrum_to_hex(spectrum, wavelengths, r=610, g=550, b=480):
    """
    Convert a spectrum into hex color code based on RGB bands.
    """
    rgb = spectrum[np.isin(wavelengths, [r,g,b])]
    return '#' + rgb_to_hex(np.round(rgb / np.max(rgb) * 255).astype(int))
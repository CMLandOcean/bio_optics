import os
import warnings
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from . import utils


# get absolute path to data folder
data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')


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


# ---------------------------------------------------------------------------
# CIE 1931 true colour
# ---------------------------------------------------------------------------

# D65 reference white (Y = 1), as used by the sRGB standard (IEC 61966-2-1)
_XYZ_D65 = np.array([0.95047, 1.0, 1.08883])

# Bradford cone-response matrix (Lam 1985)
_M_BRADFORD = np.array([[ 0.8951,  0.2664, -0.1614],
                        [-0.7502,  1.7135,  0.0367],
                        [ 0.0389, -0.0685,  1.0296]])

# XYZ (D65) -> linear sRGB (IEC 61966-2-1)
_M_XYZ_TO_SRGB = np.array([[ 3.2404542, -1.5371385, -0.4985314],
                           [-0.9692660,  1.8760108,  0.0415560],
                           [ 0.0556434, -0.2040259,  1.0572252]])


def _load_cie():
    """CIE 1931 2° colour matching functions from data/cie.txt -> (wavelengths [nm], cmf (n, 3))."""
    cie = pd.read_csv(os.path.join(data_dir, 'cie.txt'), skiprows=5, sep='\t')
    return cie.iloc[:, 0].values.astype(float), cie.iloc[:, 1:4].values.astype(float)


def _illuminant_on_grid(illuminant, grid, wavelengths):
    """Return illuminant spectral power on the 1 nm integration grid."""
    if illuminant is None or (isinstance(illuminant, str) and illuminant.upper() == 'E'):
        return np.ones_like(grid)
    if isinstance(illuminant, str) and illuminant.lower() in ('sun', 'e0_sun'):
        E0 = pd.read_csv(os.path.join(data_dir, 'E0_sun.txt'), sep=" ", skiprows=12)
        return np.interp(grid, E0.iloc[:, 0].values, E0.iloc[:, 1].values)
    illuminant = np.asarray(illuminant, dtype=float)
    if illuminant.shape != np.shape(wavelengths):
        raise ValueError("A custom illuminant must be given on the same wavelengths as the spectra.")
    order = np.argsort(wavelengths)
    return np.interp(grid, np.asarray(wavelengths)[order], illuminant[order])


def _srgb_gamma(c):
    """sRGB transfer function (linear -> encoded), input in [0, 1]."""
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * np.power(c, 1 / 2.4) - 0.055)


def spectra_to_xyz(spectra, wavelengths, illuminant=None, min_coverage=0.95):
    """
    CIE 1931 2° tristimulus values XYZ of reflectance-like spectra (e.g. Rrs or R) under an illuminant.

    X = k ∑ S(λ) R(λ) x̄(λ) Δλ, Y and Z analogous, with one common factor k = 1 / ∑ S(λ) ȳ(λ) Δλ
    for all three channels, so that a perfect reflector (R = 1) has Y = 1. The spectra are treated as
    piecewise linear between bands and integrated on a 1 nm grid over the overlap of their wavelength
    range with the CIE table (400–830 nm in data/cie.txt).

    Args:
        spectra: spectrum or image, bands on first axis; shape (n_bands,), (n_bands, n) or (n_bands, x, y)
        wavelengths: wavelengths of the bands [nm]
        illuminant: None or 'E' for equal energy (default), 'sun' for data/E0_sun.txt,
            or an array of illuminant values on `wavelengths`
        min_coverage: warn if the overlap holds less than this fraction of the integral of any
            colour matching function (x̄, ȳ or z̄), default: 0.95

    Returns:
        xyz: tristimulus values, shape spectra.shape[1:] + (3,)
        xyz_white: tristimulus values of the illuminant itself (R = 1) over the same overlap, shape (3,)
    """
    spectra = np.asarray(spectra, dtype=float)
    wavelengths = np.asarray(wavelengths, dtype=float)
    if spectra.shape[0] != wavelengths.shape[0]:
        raise ValueError(f"spectra have {spectra.shape[0]} bands on the first axis but {wavelengths.shape[0]} wavelengths were given.")

    order = np.argsort(wavelengths)
    wavelengths = wavelengths[order]
    spectra = spectra[order]

    cie_wl, cmf = _load_cie()
    lo, hi = max(wavelengths[0], cie_wl[0]), min(wavelengths[-1], cie_wl[-1])
    in_overlap = (cie_wl >= lo) & (cie_wl <= hi)
    if not in_overlap.any():
        raise ValueError(f"Spectra ({wavelengths[0]:g}-{wavelengths[-1]:g} nm) do not overlap the CIE colour matching functions ({cie_wl[0]:g}-{cie_wl[-1]:g} nm).")

    coverage = cmf[in_overlap].sum(axis=0) / cmf.sum(axis=0)
    if np.any(coverage < min_coverage):
        warnings.warn(f"Spectra cover only {lo:g}-{hi:g} nm of the CIE colour matching functions "
                      f"(x_bar, y_bar, z_bar coverage = {coverage[0]:.1%}, {coverage[1]:.1%}, {coverage[2]:.1%}); "
                      f"colours may be inaccurate.", stacklevel=3)

    grid = cie_wl[in_overlap]
    weights = cmf[in_overlap] * _illuminant_on_grid(illuminant, grid, np.asarray(wavelengths))[:, None]  # S·cmf, Δλ = 1 nm
    k = 1.0 / weights[:, 1].sum()

    # Linear-interpolation basis functions of the input bands evaluated on the grid: integrating
    # ∑_i R_i·hat_i(λ)·S·cmf equals interpolating the spectra to 1 nm first, without materialising
    # the (n_grid, n_pixels) array.
    if wavelengths.shape[0] == 1:
        basis = np.ones((grid.shape[0], 1))
    else:
        basis = interp1d(wavelengths, np.eye(wavelengths.shape[0]), axis=0, kind='linear')(grid)
    band_weights = k * basis.T @ weights  # (n_bands, 3)

    flat = spectra.reshape(spectra.shape[0], -1)
    xyz = (band_weights.T @ flat).T.reshape(spectra.shape[1:] + (3,))
    xyz_white = k * weights.sum(axis=0)
    return xyz, xyz_white


def spectra_to_srgb(spectra, wavelengths, exposure=None, illuminant=None, Y_ref=None, min_coverage=0.95):
    """
    True-colour sRGB of reflectance-like spectra (e.g. Rrs or R) via CIE 1931 2° XYZ.

    Steps: XYZ under the illuminant (see spectra_to_xyz) -> Bradford chromatic adaptation from the
    illuminant's white (computed over the same wavelength overlap, so a perfect reflector maps to
    exactly D65 white) to D65 -> one brightness scale for the whole call -> linear sRGB -> clip to
    [0, 1] -> sRGB gamma.

    Brightness is scaled once per call, not per spectrum, so relative brightness differences are
    kept and a darker spectrum gives a darker colour. By default the brightest Y in the batch is
    mapped to `exposure`. Pass `Y_ref` to use a fixed scale instead (Y / Y_ref * exposure), e.g. to
    get consistent colours across several calls or image tiles.

    Args:
        spectra: spectrum or image, bands on first axis; shape (n_bands,), (n_bands, n) or (n_bands, x, y)
        wavelengths: wavelengths of the bands [nm]
        exposure: Y value the brightest spectrum (or Y_ref) is mapped to, default: 1.
            Values > 1 brighten dark scenes (with clipping of the brightest colours).
        illuminant: None or 'E' for equal energy (default), 'sun' for data/E0_sun.txt,
            or an array of illuminant values on `wavelengths`
        Y_ref: optional fixed luminance (as returned in spectra_to_xyz units, i.e. relative to a perfect
            reflector) that is mapped to `exposure`; default: brightest Y in the batch
        min_coverage: warn if the spectral overlap with the CIE functions is below this fraction, default: 0.95

    Returns:
        rgb: sRGB values in [0, 1], shape spectra.shape[1:] + (3,), i.e. (3,), (n, 3) or (x, y, 3);
            NaN where the input spectrum contains NaN
    """
    exposure = 1.0 if exposure is None else float(exposure)
    xyz, xyz_white = spectra_to_xyz(spectra, wavelengths, illuminant=illuminant, min_coverage=min_coverage)

    # Bradford adaptation: illuminant white -> D65
    rho_src = _M_BRADFORD @ (xyz_white / xyz_white[1])
    rho_dst = _M_BRADFORD @ _XYZ_D65
    M_adapt = np.linalg.inv(_M_BRADFORD) @ np.diag(rho_dst / rho_src) @ _M_BRADFORD
    M = _M_XYZ_TO_SRGB @ M_adapt

    if Y_ref is None:
        Y = xyz[..., 1]
        Y_ref = np.nanmax(Y) if np.any(np.isfinite(Y)) else np.nan
        if not np.isfinite(Y_ref) or Y_ref <= 0:
            warnings.warn("No spectrum with positive luminance Y; returning black.", stacklevel=2)
            Y_ref = np.inf
    elif Y_ref <= 0:
        raise ValueError("Y_ref must be positive.")

    rgb_lin = (xyz * (exposure / Y_ref)) @ M.T
    nan_mask = np.isnan(rgb_lin)
    rgb = _srgb_gamma(np.clip(np.nan_to_num(rgb_lin), 0.0, 1.0))
    rgb[nan_mask] = np.nan
    return rgb


def spectra_to_srgb_hex(spectra, wavelengths, exposure=None, illuminant=None, Y_ref=None, min_coverage=0.95):
    """
    Hex colour codes ('#rrggbb') of spectra; wrapper around spectra_to_srgb (same arguments).

    Returns:
        hex_colors: str for a 1D spectrum, otherwise array of str with shape spectra.shape[1:];
            None where the input spectrum contains NaN
    """
    rgb = spectra_to_srgb(spectra, wavelengths, exposure=exposure, illuminant=illuminant, Y_ref=Y_ref, min_coverage=min_coverage)
    flat = rgb.reshape(-1, 3)
    valid = ~np.isnan(flat).any(axis=1)
    ints = np.round(np.nan_to_num(flat) * 255).astype(int)
    hex_colors = np.array(['#%02x%02x%02x' % tuple(c) if v else None for c, v in zip(ints, valid)], dtype=object)
    if rgb.ndim == 1:
        return hex_colors[0]
    return hex_colors.reshape(rgb.shape[:-1])

"""
JAX implementation of phytoplankton fluorescence forward models.

Fluorescence is modelled as a purely additive Rrs term: scalar amplitude × precomputed
Gaussian emission spectrum. All functions are pure JAX and JIT-compilable.

Emission spectra (h_C, h_C_phycocyanin, h_C_phycoerythrin) are typically precomputed
once via the numpy ``fluorescence.h_C()`` function and passed as JAX arrays stored in the
``pre`` dict produced by a coupled model's ``precompute()``. They can also be computed
on the fly with :func:`h_C` if wavelengths are known at trace time.

Three fluorescence components:
  - Chlorophyll-a (Chl-a):   peak ≈ 685 nm, FWHM ≈ 25 nm
  - Phycocyanin (cyanobacteria pigment): peak ≈ 644 nm, FWHM ≈ 20 nm
  - Phycoerythrin (cyanobacteria pigment): peak ≈ 573 nm, FWHM ≈ 20 nm
"""

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


def h_C(wavelengths, fwhm, lambda_C):
    """
    Gaussian emission spectrum [nm-1] — pure JAX, JIT-compilable.

    Args:
        wavelengths: wavelengths [nm], shape (n_wavelengths,)
        fwhm: full width at half maximum [nm]
        lambda_C: peak emission wavelength [nm]

    Returns:
        h_C: normalised Gaussian emission function [nm-1], shape (n_wavelengths,)
    """
    return (jnp.sqrt(4.0 * jnp.log(2.0) / jnp.pi) / fwhm
            * jnp.exp(-4.0 * jnp.log(2.0) * ((wavelengths - lambda_C) / fwhm) ** 2))


def Rrs_fl(L_fl_lambda0, h_C_pre):
    """
    Chlorophyll-a fluorescence contribution to Rrs [sr-1].

    Args:
        L_fl_lambda0: fluorescence amplitude [W m-2 nm-1 sr-1]
        h_C_pre: precomputed Gaussian emission spectrum [nm-1], shape (n_wavelengths,)

    Returns:
        Rrs_fl: Chl-a fluorescence Rrs [sr-1], shape (n_wavelengths,)
    """
    return L_fl_lambda0 * h_C_pre


def Rrs_fl_phycocyanin(L_fl_phycocyanin, h_C_phycocyanin_pre):
    """
    Phycocyanin fluorescence contribution to Rrs [sr-1].

    Args:
        L_fl_phycocyanin: phycocyanin fluorescence amplitude [W m-2 nm-1 sr-1]
        h_C_phycocyanin_pre: precomputed Gaussian emission spectrum [nm-1], shape (n_wavelengths,)

    Returns:
        Rrs_fl_phycocyanin: phycocyanin fluorescence Rrs [sr-1], shape (n_wavelengths,)
    """
    return L_fl_phycocyanin * h_C_phycocyanin_pre


def Rrs_fl_phycoerythrin(L_fl_phycoerythrin, h_C_phycoerythrin_pre):
    """
    Phycoerythrin fluorescence contribution to Rrs [sr-1].

    Args:
        L_fl_phycoerythrin: phycoerythrin fluorescence amplitude [W m-2 nm-1 sr-1]
        h_C_phycoerythrin_pre: precomputed Gaussian emission spectrum [nm-1], shape (n_wavelengths,)

    Returns:
        Rrs_fl_phycoerythrin: phycoerythrin fluorescence Rrs [sr-1], shape (n_wavelengths,)
    """
    return L_fl_phycoerythrin * h_C_phycoerythrin_pre


def forward(p, pre):
    """
    Combined fluorescence Rrs (Chl-a + phycocyanin + phycoerythrin) — JIT-compilable.

    Args:
        p: dict with keys:
               L_fl_lambda0       — Chl-a amplitude [W m-2 nm-1 sr-1]
               L_fl_phycocyanin   — phycocyanin amplitude [W m-2 nm-1 sr-1]
               L_fl_phycoerythrin — phycoerythrin amplitude [W m-2 nm-1 sr-1]
        pre: dict with keys h_C, h_C_phycocyanin, h_C_phycoerythrin
             (precomputed Gaussian emission spectra, shape (n_wavelengths,) each)

    Returns:
        Rrs_fl_total: total fluorescence Rrs [sr-1], shape (n_wavelengths,)
    """
    return (Rrs_fl(p["L_fl_lambda0"], pre["h_C"])
            + Rrs_fl_phycocyanin(p["L_fl_phycocyanin"], pre["h_C_phycocyanin"])
            + Rrs_fl_phycoerythrin(p["L_fl_phycoerythrin"], pre["h_C_phycoerythrin"]))

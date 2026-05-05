"""
JAX implementation of atmospheric transmittance functions (Gege 2021 / WASI).

Component module — no precompute / make_forward_vec.  All functions are pure JAX
and accept precomputed spectral absorption arrays from
``downwelling_irradiance_jax.precompute()`` alongside scalar atmospheric parameters.

Reference:
    Gege, P. (2021): The Water Colour Simulator WASI. User manual for WASI version 6.
"""

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------
# Atmospheric path lengths (scalar → scalar)
# ---------------------------------------------------------------------------

def M(theta_sun, a=0.50572, b=6.07995, c=1.6364):
    """
    Atmospheric path length after Kasten and Young (1989).

    Args:
        theta_sun: sun zenith angle [radians]
        a: empirical constant, default: 0.50572
        b: empirical constant [degrees], default: 6.07995
        c: empirical constant, default: 1.6364

    Returns:
        M: atmospheric path length [dimensionless]
    """
    return 1.0 / (jnp.cos(theta_sun) + a * (90.0 + b - jnp.degrees(theta_sun)) ** (-c))


def M_cor(theta_sun, P=1013.25):
    """
    Atmospheric path length corrected for non-standard atmospheric pressure.

    Args:
        theta_sun: sun zenith angle [radians]
        P: atmospheric pressure [mbar], default: 1013.25

    Returns:
        M_cor: pressure-corrected atmospheric path length [dimensionless]
    """
    return M(theta_sun) * P / 1013.25


def M_oz(theta_sun):
    """
    Atmospheric path length for ozone.

    Args:
        theta_sun: sun zenith angle [radians]

    Returns:
        M_oz: ozone path length [dimensionless]
    """
    return 1.0035 / (jnp.cos(theta_sun) ** 2 + 0.007) ** 0.5


# ---------------------------------------------------------------------------
# Aerosol helper functions (scalar → scalar)
# ---------------------------------------------------------------------------

def omega_a(AM=5.0, RH=80.0):
    """
    Aerosol single scattering albedo.

    Args:
        AM: air mass type [1: open ocean .. 10: continental aerosols], default: 5
        RH: relative humidity [%], default: 80

    Returns:
        omega_a: aerosol single scattering albedo [dimensionless]
    """
    return (-0.0032 * AM + 0.972) * jnp.exp(3.06e-4 * RH)


def F_a(theta_sun, alpha=1.317):
    """
    Aerosol forward scattering probability.

    Args:
        theta_sun: sun zenith angle [radians]
        alpha: Ångström exponent (typically 0.2–2), default: 1.317

    Returns:
        F_a: aerosol forward scattering probability [dimensionless]
    """
    B3 = jnp.log(1.0 - (-0.1417 * alpha + 0.82))
    B1 = B3 * (1.459 + B3 * (0.1595 + 0.4129 * B3))
    B2 = B3 * (0.0783 + B3 * (-0.3824 - 0.5874 * B3))
    return 1.0 - 0.5 * jnp.exp((B1 + B2 * jnp.cos(theta_sun)) * jnp.cos(theta_sun))


def tau_a(wavelengths, lambda_a=550.0, alpha=1.317, beta=0.2606):
    """
    Aerosol optical thickness.

    Args:
        wavelengths: wavelengths [nm], shape (n_wavelengths,)
        lambda_a: reference wavelength [nm], default: 550
        alpha: Ångström exponent (typically 0.2–2), default: 1.317
        beta: turbidity coefficient (typically 0.16–0.50), default: 0.2606

    Returns:
        tau_a: aerosol optical thickness [dimensionless], shape (n_wavelengths,)
    """
    return beta * (wavelengths / lambda_a) ** (-alpha)


# ---------------------------------------------------------------------------
# Transmittance functions (spectral → spectral)
# ---------------------------------------------------------------------------

def T_r(wavelengths, theta_sun, P=1013.25):
    """
    Rayleigh scattering transmittance.
    Wavelengths are converted to micrometres internally.

    Args:
        wavelengths: wavelengths [nm], shape (n_wavelengths,)
        theta_sun: sun zenith angle [radians]
        P: atmospheric pressure [mbar], default: 1013.25

    Returns:
        T_r: Rayleigh scattering transmittance [dimensionless], shape (n_wavelengths,)
    """
    wl_um = wavelengths / 1000.0
    return jnp.exp(-M_cor(theta_sun, P) / (115.6406 * wl_um ** 4 - 1.335 * wl_um ** 2))


def T_aa(wavelengths, theta_sun, AM=5.0, RH=80.0, lambda_a=550.0, alpha=1.317, beta=0.2606):
    """
    Aerosol absorption transmittance.

    Args:
        wavelengths: wavelengths [nm], shape (n_wavelengths,)
        theta_sun: sun zenith angle [radians]
        AM: air mass type [1: open ocean .. 10: continental aerosols], default: 5
        RH: relative humidity [%], default: 80
        lambda_a: reference wavelength [nm], default: 550
        alpha: Ångström exponent, default: 1.317
        beta: turbidity coefficient, default: 0.2606

    Returns:
        T_aa: aerosol absorption transmittance [dimensionless], shape (n_wavelengths,)
    """
    return jnp.exp(-(1.0 - omega_a(AM, RH)) * tau_a(wavelengths, lambda_a, alpha, beta) * M(theta_sun))


def T_as(wavelengths, theta_sun, AM=5.0, RH=80.0, lambda_a=550.0, alpha=1.317, beta=0.2606):
    """
    Aerosol scattering transmittance.

    Args:
        wavelengths: wavelengths [nm], shape (n_wavelengths,)
        theta_sun: sun zenith angle [radians]
        AM: air mass type [1: open ocean .. 10: continental aerosols], default: 5
        RH: relative humidity [%], default: 80
        lambda_a: reference wavelength [nm], default: 550
        alpha: Ångström exponent, default: 1.317
        beta: turbidity coefficient, default: 0.2606

    Returns:
        T_as: aerosol scattering transmittance [dimensionless], shape (n_wavelengths,)
    """
    return jnp.exp(-omega_a(AM, RH) * tau_a(wavelengths, lambda_a, alpha, beta) * M(theta_sun))


def T_oz(wavelengths, theta_sun, H_oz, a_oz):
    """
    Ozone absorption transmittance.

    Args:
        wavelengths: wavelengths [nm], shape (n_wavelengths,)
        theta_sun: sun zenith angle [radians]
        H_oz: ozone scale height [cm]
        a_oz: precomputed ozone absorption coefficient [cm-1], shape (n_wavelengths,)

    Returns:
        T_oz: ozone absorption transmittance [dimensionless], shape (n_wavelengths,)
    """
    return jnp.exp(-a_oz * H_oz * M_oz(theta_sun))


def T_ox(wavelengths, theta_sun, P, a_ox):
    """
    Oxygen absorption transmittance.

    Args:
        wavelengths: wavelengths [nm], shape (n_wavelengths,)
        theta_sun: sun zenith angle [radians]
        P: atmospheric pressure [mbar]
        a_ox: precomputed oxygen absorption coefficient [cm-1], shape (n_wavelengths,)

    Returns:
        T_ox: oxygen absorption transmittance [dimensionless], shape (n_wavelengths,)
    """
    x = 1.41 * a_ox * M_cor(theta_sun, P)
    return jnp.exp(-x / (1.0 + 118.3 * a_ox * M_cor(theta_sun, P)) ** 0.45)


def T_wv(wavelengths, theta_sun, WV, a_wv):
    """
    Water vapour absorption transmittance.

    Args:
        wavelengths: wavelengths [nm], shape (n_wavelengths,)
        theta_sun: sun zenith angle [radians]
        WV: precipitable water [cm]
        a_wv: precomputed water vapour absorption coefficient [cm-1], shape (n_wavelengths,)

    Returns:
        T_wv: water vapour absorption transmittance [dimensionless], shape (n_wavelengths,)
    """
    x = 0.2385 * a_wv * WV * M(theta_sun)
    return jnp.exp(-x / (1.0 + 20.07 * a_wv * WV * M(theta_sun)) ** 0.45)

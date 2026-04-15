import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


def bb_phy(C_phy, bb_phy_spec, b_phy_norm):
    """
    Spectral backscattering coefficient of phytoplankton.

    Args:
        C_phy: total phytoplankton concentration [ug/L]
        bb_phy_spec: specific backscattering coefficient of phytoplankton at 550 nm [m2 mg-1]
        b_phy_norm: precomputed normalized phytoplankton scattering spectrum [dimensionless], shape (n_wavelengths,)

    Returns:
        bb_phy: spectral backscattering coefficient of phytoplankton [m-1], shape (n_wavelengths,)
    """
    return C_phy * bb_phy_spec * b_phy_norm


def bb_X(C_X, bb_X_spec, b_X_norm):
    """
    Spectral backscattering coefficient of non-algal particles type I (spectrally flat).

    Args:
        C_X: concentration of non-algal particles type I [mg/L]
        bb_X_spec: specific backscattering coefficient of type I particles [m2 g-1]
        b_X_norm: precomputed normalized scattering spectrum [dimensionless], shape (n_wavelengths,)

    Returns:
        bb_X: spectral backscattering coefficient of type I particles [m-1], shape (n_wavelengths,)
    """
    return C_X * bb_X_spec * b_X_norm


def bb_Mie(C_Mie, bb_Mie_spec, lambda_S, n, wavelengths):
    """
    Spectral backscattering coefficient of non-algal particles type II (Mie / power-law scattering).

    Args:
        C_Mie: concentration of non-algal particles type II [mg/L]
        bb_Mie_spec: specific backscattering coefficient of type II particles [m2 g-1]
        lambda_S: reference wavelength for Mie scattering [nm]
        n: Angström exponent of type II particle backscattering [dimensionless]
        wavelengths: wavelengths [nm], shape (n_wavelengths,)

    Returns:
        bb_Mie: spectral backscattering coefficient of type II particles [m-1], shape (n_wavelengths,)
    """
    return C_Mie * bb_Mie_spec * (wavelengths / lambda_S) ** n


def bb(C_X, C_Mie, C_phy,
       bb_phy_spec, bb_Mie_spec, bb_X_spec,
       lambda_S, n,
       wavelengths,
       bb_w,
       b_phy_norm,
       b_X_norm):
    """
    Spectral backscattering coefficient of a natural water body.

    All static spectral arrays (bb_w, b_phy_norm, b_X_norm) must be provided as precomputed
    JAX arrays from albert_mobley_jax.precompute(). The Mie shape term (wavelengths/lambda_S)**n
    is computed inside JIT as pure arithmetic.

    Args:
        C_X: concentration of non-algal particles type I [mg/L]
        C_Mie: concentration of non-algal particles type II [mg/L]
        C_phy: total phytoplankton concentration [ug/L]
        bb_phy_spec: specific backscattering coefficient of phytoplankton at 550 nm [m2 mg-1]
        bb_Mie_spec: specific backscattering coefficient of type II particles [m2 g-1]
        bb_X_spec: specific backscattering coefficient of type I particles [m2 g-1]
        lambda_S: reference wavelength for Mie scattering [nm]
        n: Angström exponent of type II particle backscattering [dimensionless]
        wavelengths: wavelengths [nm], shape (n_wavelengths,)
        bb_w: precomputed pure water backscattering [m-1], shape (n_wavelengths,)
        b_phy_norm: precomputed normalized phytoplankton scattering [dimensionless], shape (n_wavelengths,)
        b_X_norm: precomputed normalized type I particle scattering [dimensionless], shape (n_wavelengths,)

    Returns:
        bb: spectral backscattering coefficient of a natural water body [m-1], shape (n_wavelengths,)
    """
    return (bb_w
            + bb_phy(C_phy, bb_phy_spec, b_phy_norm)
            + bb_X(C_X, bb_X_spec, b_X_norm)
            + bb_Mie(C_Mie, bb_Mie_spec, lambda_S, n, wavelengths))

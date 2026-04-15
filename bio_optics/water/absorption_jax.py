import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


def a_ph(C_i, a_i_spec):
    """
    Spectral absorption coefficient of phytoplankton for a mixture of up to 6 phytoplankton classes.

    Args:
        C_i: concentrations of phytoplankton types 0-5 [ug/L], shape (6,)
        a_i_spec: precomputed specific absorption spectra of phytoplankton types [m2 mg-1], shape (n_wavelengths, 6)

    Returns:
        a_ph: spectral absorption coefficient of phytoplankton [m-1], shape (n_wavelengths,)
    """
    return jnp.dot(a_i_spec, C_i)


def a_Y(C_Y, S, lambda_0, wavelengths, K=0.0):
    """
    Exponential approximation of spectral absorption of CDOM or yellow substances.

    Args:
        C_Y: CDOM absorption coefficient at lambda_0 [m-1]
        S: spectral slope of CDOM absorption spectrum [nm-1]
        lambda_0: reference wavelength for normalization [nm]
        wavelengths: wavelengths [nm], shape (n_wavelengths,)
        K: constant offset [m-1], default: 0.0

    Returns:
        a_Y: spectral absorption coefficient of CDOM [m-1], shape (n_wavelengths,)
    """
    return C_Y * jnp.exp(-S * (wavelengths - lambda_0)) + K


def a_NAP(C_X, C_Mie, S_NAP, lambda_0, wavelengths, a_NAP_spec_lambda_0=0.041):
    """
    Spectral absorption of non-algal particles (NAP).

    Args:
        C_X: concentration of non-algal particles type I [mg/L]
        C_Mie: concentration of non-algal particles type II [mg/L]
        S_NAP: spectral slope of NAP absorption spectrum [nm-1]
        lambda_0: reference wavelength for normalization [nm]
        wavelengths: wavelengths [nm], shape (n_wavelengths,)
        a_NAP_spec_lambda_0: specific absorption coefficient of NAP at lambda_0 [m2 g-1], default: 0.041

    Returns:
        a_NAP: spectral absorption coefficient of non-algal particles [m-1], shape (n_wavelengths,)
    """
    return (C_X + C_Mie) * a_NAP_spec_lambda_0 * jnp.exp(-S_NAP * (wavelengths - lambda_0))


def a(C_0, C_1, C_2, C_3, C_4, C_5,
      C_Y, C_X, C_Mie,
      S, S_NAP, lambda_0, K,
      T_W, T_W_0,
      a_NAP_spec_lambda_0,
      wavelengths,
      a_w,
      da_w_div_dT,
      a_i_spec):
    """
    Spectral absorption coefficient of a natural water body.

    All spectral lookup tables (a_w, da_w_div_dT, a_i_spec) must be provided as precomputed
    JAX arrays from albert_mobley_jax.precompute(). Parameters S, S_NAP, lambda_0, K are
    computed inside JIT as pure arithmetic.

    Args:
        C_0: concentration of phytoplankton type 0 [ug/L]
        C_1: concentration of phytoplankton type 1 [ug/L]
        C_2: concentration of phytoplankton type 2 [ug/L]
        C_3: concentration of phytoplankton type 3 [ug/L]
        C_4: concentration of phytoplankton type 4 [ug/L]
        C_5: concentration of phytoplankton type 5 [ug/L]
        C_Y: CDOM absorption coefficient at lambda_0 [m-1]
        C_X: concentration of non-algal particles type I [mg/L]
        C_Mie: concentration of non-algal particles type II [mg/L]
        S: spectral slope of CDOM absorption spectrum [nm-1]
        S_NAP: spectral slope of NAP absorption spectrum [nm-1]
        lambda_0: reference wavelength for CDOM and NAP normalization [nm]
        K: constant offset of CDOM exponential function [m-1]
        T_W: actual water temperature [degrees C]
        T_W_0: reference temperature for pure water absorption [degrees C]
        a_NAP_spec_lambda_0: specific absorption coefficient of NAP at lambda_0 [m2 g-1]
        wavelengths: wavelengths [nm], shape (n_wavelengths,)
        a_w: precomputed pure water absorption [m-1], shape (n_wavelengths,)
        da_w_div_dT: precomputed temperature gradient of pure water absorption [m-1 degrees_C-1], shape (n_wavelengths,)
        a_i_spec: precomputed specific absorption spectra of phytoplankton types [m2 mg-1], shape (n_wavelengths, 6)

    Returns:
        a: spectral absorption coefficient of a natural water body [m-1], shape (n_wavelengths,)
    """
    C_i = jnp.array([C_0, C_1, C_2, C_3, C_4, C_5])

    return (a_w + (T_W - T_W_0) * da_w_div_dT
            + a_ph(C_i, a_i_spec)
            + a_Y(C_Y, S, lambda_0, wavelengths, K)
            + a_NAP(C_X, C_Mie, S_NAP, lambda_0, wavelengths, a_NAP_spec_lambda_0))

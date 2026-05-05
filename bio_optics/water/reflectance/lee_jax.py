"""
JAX implementation of the Lee et al. (2011) deep-water remote sensing reflectance model.

Component module — no precompute / make_forward_vec.  These are pure JAX functions
intended to be called from other JAX forward models (e.g. hope_jax, bi_jax).

Reference:
    Lee et al. (2011): An inherent-optical-property-centered approach to correct the
    angular effects in water-leaving radiance [10.1364/AO.50.003155]
"""

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


def Rrs_deep(a, bb, bb_p, bb_w,
             Gw0=0.05881474,
             Gw1=0.05062697,
             Gp0=0.03997009,
             Gp1=0.1398902):
    """
    Remote sensing reflectance of optically deep water following Lee et al. (2011).
    Transfer through the water surface is already included.

    Args:
        a: total absorption coefficient [m-1], shape (n_wavelengths,)
        bb: total backscattering coefficient [m-1], shape (n_wavelengths,)
        bb_p: total particulate backscattering coefficient [m-1], shape (n_wavelengths,)
        bb_w: backscattering coefficient of pure water [m-1], shape (n_wavelengths,)
        Gw0: empirical coefficient for water term (constant part), default: 0.05881474
        Gw1: empirical coefficient for water term (b_bw/k part), default: 0.05062697
        Gp0: empirical coefficient for particle term (constant part), default: 0.03997009
        Gp1: empirical coefficient for particle term (b_bp/k part), default: 0.1398902

    Returns:
        Rrs: remote sensing reflectance of optically deep water [sr-1], shape (n_wavelengths,)
    """
    k = a + bb
    return (Gw0 + Gw1 * bb_w / k) * bb_w / k + (Gp0 + Gp1 * bb_p / k) * bb_p / k

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


def snell(theta_inc, n1=1.0, n2=1.33):
    """
    Compute the refraction angle using Snell's Law.

    Args:
        theta_inc: incident angle [radians]
        n1: refractive index of origin medium, default: 1.0 for air
        n2: refractive index of destination medium, default: 1.33 for water

    Returns:
        theta_refr: refraction angle [radians]
    """
    return jnp.arcsin(n1 / n2 * jnp.sin(theta_inc))


def below2above(rrs, zeta=0.52, Gamma=1.6):
    """
    Convert subsurface radiance reflectance to remote sensing reflectance after Lee et al. (1998) [1]
    as described in Giardino et al. (2019) [2].

    [1] Lee et al. (1998): Hyperspectral remote sensing for shallow waters: 1. A semianalytical model [10.1364/AO.37.006329]
    [2] Giardino et al. (2019): Imaging Spectrometry of Inland and Coastal Waters: State of the Art, Achievements and Perspectives [10.1007/s10712-018-9476-0]

    Args:
        rrs: subsurface radiance reflectance [sr-1]
        zeta: proportionality factor, default: 0.52
        Gamma: proportionality factor, default: 1.6

    Returns:
        Rrs: above-water remote sensing reflectance [sr-1]
    """
    return (zeta * rrs) / (1 - Gamma * rrs)


def above2below(Rrs, zeta=0.52, Gamma=1.6):
    """
    Convert remote sensing reflectance to subsurface radiance reflectance after Lee et al. (1998) [1]
    as described in Giardino et al. (2019) [2].

    [1] Lee et al. (1998): Hyperspectral remote sensing for shallow waters: 1. A semianalytical model [10.1364/AO.37.006329]
    [2] Giardino et al. (2019): Imaging Spectrometry of Inland and Coastal Waters: State of the Art, Achievements and Perspectives [10.1007/s10712-018-9476-0]

    Args:
        Rrs: above-water remote sensing reflectance [sr-1]
        zeta: proportionality factor, default: 0.52
        Gamma: proportionality factor, default: 1.6

    Returns:
        rrs: subsurface radiance reflectance [sr-1]
    """
    return Rrs / (zeta + Gamma * Rrs)


def fresnel(theta_inc, n1=1.0, n2=1.33, method="cos"):
    """
    Fresnel reflectance for unpolarized incoming light at a flat air-water interface.

    Two equivalent formulations are available via `method`:

    ``"cos"`` (default) — cosine form, numerically stable at normal incidence
    (theta_inc = 0) where the sin/tan form produces 0/0::

        rs = ((n1 cos θ_i − n2 cos θ_t) / (n1 cos θ_i + n2 cos θ_t))²
        rp = ((n2 cos θ_i − n1 cos θ_t) / (n2 cos θ_i + n1 cos θ_t))²

    ``"sin_tan"`` — classical textbook form, identical to the cos form for
    theta_inc > 0 but returns nan at theta_inc = 0::

        rs = (sin(θ_i − θ_t) / sin(θ_i + θ_t))²
        rp = (tan(θ_i − θ_t) / tan(θ_i + θ_t))²

    Args:
        theta_inc: incident angle [radians]
        n1: refractive index of origin medium, default: 1.0 for air
        n2: refractive index of destination medium, default: 1.33 for water
        method: ``"cos"`` (default) or ``"sin_tan"``

    Returns:
        rho_F: Fresnel reflectance for unpolarized light [dimensionless]
    """
    theta_w = snell(theta_inc, n1, n2)
    if method == "cos":
        cos_i = jnp.cos(theta_inc)
        cos_w = jnp.cos(theta_w)
        rs = ((n1 * cos_i - n2 * cos_w) / (n1 * cos_i + n2 * cos_w)) ** 2
        rp = ((n2 * cos_i - n1 * cos_w) / (n2 * cos_i + n1 * cos_w)) ** 2
    elif method == "sin_tan":
        rs = (jnp.sin(theta_inc - theta_w) / jnp.sin(theta_inc + theta_w)) ** 2
        rp = (jnp.tan(theta_inc - theta_w) / jnp.tan(theta_inc + theta_w)) ** 2
    else:
        raise ValueError(f"method must be 'cos' or 'sin_tan', got {method!r}")
    return (rs + rp) / 2

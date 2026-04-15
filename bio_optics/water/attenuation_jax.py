import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


def omega_b(a, bb):
    """
    Single scattering albedo of a water body [dimensionless].

    Args:
        a: total absorption coefficient [m-1]
        bb: total backscattering coefficient [m-1]

    Returns:
        omega_b: single scattering albedo [dimensionless]
    """
    return bb / (a + bb)


def Kd(a, bb, cos_t_sun_p, kappa_0=1.0546):
    """
    Diffuse attenuation coefficient for downwelling irradiance after Albert & Mobley (2003) [1].

    [1] Albert & Mobley (2003): An analytical model for subsurface irradiance and remote sensing reflectance
        in deep and shallow case-2 waters [10.1364/OE.11.002873]

    Args:
        a: total absorption coefficient [m-1]
        bb: total backscattering coefficient [m-1]
        cos_t_sun_p: cosine of the refracted sun zenith angle inside water (after Snell's law)
        kappa_0: coefficient depending on scattering phase function, default: 1.0546

    Returns:
        Kd: diffuse attenuation coefficient for downwelling irradiance [m-1]
    """
    return (kappa_0 / cos_t_sun_p) * (a + bb)


def ku_w(a, bb, ob, cos_t_sun_p, cos_t_view_p):
    """
    Upwelling attenuation coefficient for the water column after Albert & Mobley (2003) [1].

    [1] Albert & Mobley (2003): An analytical model for subsurface irradiance and remote sensing reflectance
        in deep and shallow case-2 waters [10.1364/OE.11.002873]

    Args:
        a: total absorption coefficient [m-1]
        bb: total backscattering coefficient [m-1]
        ob: single scattering albedo bb / (a + bb) [dimensionless]
        cos_t_sun_p: cosine of the refracted sun zenith angle inside water (after Snell's law)
        cos_t_view_p: cosine of the refracted view zenith angle inside water (after Snell's law)

    Returns:
        ku_w: upwelling attenuation coefficient for the water column [m-1]
    """
    return (a + bb) / cos_t_view_p * (1 + ob)**3.5421 * (1 - 0.2786 / cos_t_sun_p)


def ku_b(a, bb, ob, cos_t_sun_p, cos_t_view_p):
    """
    Upwelling attenuation coefficient for the bottom after Albert & Mobley (2003) [1].

    [1] Albert & Mobley (2003): An analytical model for subsurface irradiance and remote sensing reflectance
        in deep and shallow case-2 waters [10.1364/OE.11.002873]

    Args:
        a: total absorption coefficient [m-1]
        bb: total backscattering coefficient [m-1]
        ob: single scattering albedo bb / (a + bb) [dimensionless]
        cos_t_sun_p: cosine of the refracted sun zenith angle inside water (after Snell's law)
        cos_t_view_p: cosine of the refracted view zenith angle inside water (after Snell's law)

    Returns:
        ku_b: upwelling attenuation coefficient for the bottom [m-1]
    """
    return (a + bb) / cos_t_view_p * (1 + ob)**2.2658 * (1 + 0.0577 / cos_t_sun_p)

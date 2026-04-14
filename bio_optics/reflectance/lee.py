def Rrs_deep(a,
             bb,
             bb_p,
             bb_w,
             Gw0 = 0.05881474,
             Gw1 = 0.05062697,
             Gp0 = 0.03997009,
             Gp1 = 0.1398902):
    """
    Remote sensing reflectance of optically deep water following Lee et al. (2011) [1]
    Transfer through the water surface is already included.

    [1] Lee et al. (2011): An inherent-optical-property-centered approach to correct the angular effects in water-leaving radiance [10.1364/AO.50.003155]

    Args:
        a: total absorption coefficient [m-1]
        bb: total backscattering coefficient [m-1]
        bb_p: total particulate backscattering coefficient [m-1]
        bb_w: backscattering coefficient of water [m-1]
        Gw0: empirical coefficient for water term (constant part), default: 0.05881474
        Gw1: empirical coefficient for water term (b_bw/k part), default: 0.05062697
        Gp0: empirical coefficient for particle term (constant part), default: 0.03997009
        Gp1: empirical coefficient for particle term (b_bp/k part), default: 0.1398902

    Returns:
        Rrs: remote sensing reflectance of optically deep water [sr-1]
    """
    k = a + bb
    Rrs = (Gw0 + Gw1 * bb_w / k) * bb_w / k + (Gp0 + Gp1 * bb_p / k) * bb_p / k

    return Rrs
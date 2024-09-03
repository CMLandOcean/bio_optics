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
        a (np.array): Total absorption coefficient.
        b_b (np.array): Total backscattering coefficient.
        b_bp (np.array): Total particulate backscattering coefficient.
        b_bw (np.array): Backscattering coefficient of water.
        Gw0 (float, optional): _description_. Defaults to 0.05881474.
        Gw1 (float, optional): _description_. Defaults to 0.05062697.
        Gp0 (float, optional): _description_. Defaults to 0.03997009.
        Gp1 (float, optional): _description_. Defaults to 0.1398902.
    """
    k = a + bb
    Rrs = (Gw0 + Gw1 * bb_w / k) * bb_w / k + (Gp0 + Gp1 * bb_p / k) * bb_p / k

    return Rrs
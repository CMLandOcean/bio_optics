"""
Lee et al. (2011) deep-water remote sensing reflectance model.

Partitions Rrs into water and particle contributions using empirical G coefficients.
JAX-native version: bio_optics.water.reflectance.lee_jax

Reference:
    Lee et al. (2011): An inherent-optical-property-centered approach to correct the
    angular effects in water-leaving radiance [10.1364/AO.50.003155]
"""


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


import scipy.io
import pandas as pd
import xarray as xr
import numpy as np


def read_G_LUT():
    mat = scipy.io.loadmat("../bio_optics/data/G_LUT.mat")
    G_LUT = mat['G_LUT']

    # solz=np.asarray([0.,15,30,45,60,75,80,88])  # solar zenith angle steps in LUT, column 0
    # senz= np.asarray([0,10,20,30,40,50,60,70,80,87.5]) # sensor zenith angle steps in LUT, column 1
    # phi= np.asarray([0.,15,30,45,60,75,90,105,120,135,150,165,180]) # azimuth difference angle steps in LUT, column 2

    ## setup xarray instance with coordinates (solz, senz, phi)
    G_df = pd.DataFrame(G_LUT, columns=['solz', 'senz', 'phi', 'G0w', 'G1w', 'G0p', 'G1p'])
    # G_df.to_csv("../water/G_LUT.csv", index=False)

    solz = np.unique(G_df.iloc[:, 0])
    senz = np.unique(G_df.iloc[:, 1])
    phi = np.unique(G_df.iloc[:, 2])

    return G_df, solz, senz, phi


def setup_xarray_Gx(G_df, solz, senz, phi, varname='G0w'):
    Gx = np.zeros((len(solz), len(senz), len(phi))) + np.nan
    for i, sz in enumerate(solz):
        for j, oz in enumerate(senz):
            for k, aa in enumerate(phi):
                if oz == 0:
                    id_solz = np.where(G_df.iloc[:, 0] == sz)[0]
                    id_senz = np.where(G_df.iloc[:, 1] == oz)[0]
                    id_aa = np.where(G_df.iloc[:, 2] == 0)[0]
                    idx_temp = np.intersect1d(id_solz, id_senz)
                    idx = np.intersect1d(idx_temp, id_aa)
                else:
                    id_solz = np.where(G_df.iloc[:, 0] == sz)[0]
                    id_senz = np.where(G_df.iloc[:, 1] == oz)[0]
                    id_aa = np.where(G_df.iloc[:, 2] == aa)[0]
                    idx_temp = np.intersect1d(id_solz, id_senz)
                    idx = np.intersect1d(idx_temp, id_aa)
                # print(i, j, k, idx)
                if len(idx) == 1:
                    # print(G_df[varname].iloc[idx].values)
                    Gx[i, j, k] = G_df[varname].iloc[idx].values[0]

    da = xr.DataArray(
        data=Gx,
        dims=["solz", "senz", "phi"],
        coords=dict(
            solz=solz,
            senz=senz,
            phi=phi))

    return da
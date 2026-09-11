def R_rs_deep(a,
              b_b,
              b_bp,
              b_bw,
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
    k = a + b_b
    R_rs = (Gw0 + Gw1 * b_bw / k) * b_bw / k + (Gp0 + Gp1 * b_bp / k) * b_bp / k

    return R_rs

import scipy.io
import pandas as pd
import xarray as xr
import numpy as np

def read_G_LUT():
    mat = scipy.io.loadmat("../water/G_LUT.mat")
    G_LUT = mat['G_LUT']

    # solz=np.asarray([0.,15,30,45,60,75,80,88])  # solar zenith angle steps in LUT, column 0
    # senz= np.asarray([0,10,20,30,40,50,60,70,80,87.5]) # sensor zenith angle steps in LUT, column 1
    # phi= np.asarray([0.,15,30,45,60,75,90,105,120,135,150,165,180]) # azimuth difference angle steps in LUT, column 2
    
    ## setup xarray instance with coordinates (solz, senz, phi)
    G_df = pd.DataFrame(G_LUT, columns=['solz', 'senz', 'phi', 'G0w', 'G1w', 'G0p', 'G1p'])
    # G_df.to_csv("../water/G_LUT.csv", index=False)
    
    solz = np.unique(G_df.iloc[:,0])
    senz = np.unique(G_df.iloc[:,1])
    phi = np.unique(G_df.iloc[:,2])

    return G_df, solz, senz, phi
    

def setup_xarray_Gx(G_df, solz, senz, phi, varname='G0w'):
    Gx = np.zeros((len(solz), len(senz), len(phi))) + np.nan
    for i, sz in enumerate(solz):
        for j, oz in enumerate(senz):
            for k, aa in enumerate(phi):
                if oz == 0:
                    id_solz = np.where( G_df.iloc[:, 0] == sz)[0]
                    id_senz = np.where( G_df.iloc[:, 1] == oz)[0]
                    id_aa = np.where( G_df.iloc[:, 2] == 0)[0]
                    idx_temp= np.intersect1d(id_solz, id_senz)
                    idx = np.intersect1d(idx_temp,id_aa)
                else:
                    id_solz = np.where( G_df.iloc[:, 0] == sz)[0]
                    id_senz = np.where( G_df.iloc[:, 1] == oz)[0]
                    id_aa = np.where( G_df.iloc[:, 2] == aa)[0]
                    idx_temp= np.intersect1d(id_solz, id_senz)
                    idx = np.intersect1d(idx_temp,id_aa)
                # print(i, j, k, idx)
                if len(idx) ==1:
                    Gx[i, j, k] = G_df[varname].iloc[idx[0]] #.values


    da = xr.DataArray(
        data=Gx,
        dims=["solz", "senz", "phi"],
        coords=dict(
            solz=solz,
            senz=senz,
            phi=phi))
    
    return da

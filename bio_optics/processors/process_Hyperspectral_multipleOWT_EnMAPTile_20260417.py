# Automatically full inversion process by Bi&Hieronymi OWTs
# dictionary of ranges and parameters per OWT and area
# read in data

### use with env py38_keras3
# from matplotlib import pyplot as plt
# import pandas as pd
# import numpy as np
# import scipy
# from scipy.interpolate import UnivariateSpline
# import lmfit
import ray
# import os
# import timeit

from bio_optics.water import absorption, attenuation, backscattering, scattering, lee, fluorescence
from bio_optics.atmosphere import downwelling_irradiance
from bio_optics.models import hereon, model, bandratioIOPprediction
from bio_optics.helper import resampling, utils, owt, indices, plotting

from matplotlib import pyplot as plt
import pandas as pd
# import geopandas as gpd
import numpy as np
import scipy
from shapely.geometry import box
import lmfit
# import rioxarray
import timeit
import sys
import os
import xarray as xr
# from xcube.core.store import new_data_store
import dask.array as da
# import rasterio
import pickle
from datetime import datetime

import json
# from rasterio.features import rasterize


def set_default_parameters(AlgaeGroupType='Standardv3'):
    ## by Default:
    # - all Phytoplankton groups are value=0, min=0, max=1000 (C_5=10, C_7=1), vary = False
    # - C_Y value=0, min=0, max=30, vary=True
    # - C_ism value=0, min=0, max=100, vary=True
    # - fluorescence all vary = False
    # - offset vary = False
    params = lmfit.Parameters()                             # v3 ,      SummerBloom
    params.add('C_0', value=0, min=0, max=100, vary=False)  # Diatoms
    params.add('C_1', value=0, min=0, max=100, vary=False)  # green
    params.add('C_2', value=0, min=0, max=100, vary=False)  # cryptophyte
    params.add('C_3', value=0, min=0, max=100, vary=False)  # cyano blue
    params.add('C_4', value=0, min=0, max=100, vary=False)  # cyano red
    params.add('C_5', value=0, min=0, max=10, vary=False)   # coccolithophores , Phaeocystis: other ranges
    params.add('C_6', value=0, min=0, max=100, vary=False)  # dinoflagellates
    params.add('C_7', value=0, min=0, max=1, vary=False)    # case-1  , Noctiluca: other ranges
    params.add('C_Y', value=0, min=0, max=30, vary=True)
    params.add('C_ism', value=0, min=0, max=100, vary=True)
    params.add('L_fl_lambda0', value=0, min=0, max=0.2, vary=False)
    params.add('L_fl_phycocyanin', value=0, min=0, max=0.2, vary=False)
    params.add('L_fl_phycoerythrin', value=0, min=0, max=0.2, vary=False)
    params.add('b_ratio_C_0', value=0.002, vary=False)  # Diatoms 0.0058; HEREONweb: 0.002
    params.add('b_ratio_C_1', value=0.007, vary=False)  # green 0.007, HEREONweb: 0.007
    params.add('b_ratio_C_2', value=0.002, vary=False)  # cryptophyte 0.0042, HEREONweb: 0.002
    params.add('b_ratio_C_3', value=0.003, vary=False)  # cyano blue 0.0082, HEREONweb: 0.003
    params.add('b_ratio_C_4', value=0.003, vary=False)  # cyano red 0.001, HEREONweb: 0.003
    # if AlgaeGroupType == 'NSSummerBloomsv3':
    if AlgaeGroupType == 'Summer':
        params.add('b_ratio_C_5', value=0.0034, vary=False)  # Phaeocystis: change to 0.0034
    elif AlgaeGroupType == 'HEREONgold':
        params.add('b_ratio_C_5', value=0.002, vary=False)
    else:
        params.add('b_ratio_C_5', value=0.0129,
                   vary=False)  # coccolithophores 0.0129, , HEREONweb: 0.007, Phaeocystis: change to 0.0034
    params.add('b_ratio_C_6', value=0.0209, vary=False)  # dinoflagellates 0.0209, HEREONweb: not available
    # if AlgaeGroupType == 'NSSummerBloomsv3':
    if AlgaeGroupType == 'Summer':
        params.add('b_ratio_C_7', value=0.0209, vary=False)  # Noctiluca: change to 0.0209
    else:
        params.add('b_ratio_C_7', value=0.007,
                   vary=False)  # case-1 0.0109, HEREONweb: 0.007, Noctiluca: change to 0.0209
    params.add('b_ratio_md', value=0.0216, min=0.021, max=0.3756, vary=False)  # max=0.0756 #todo: vary turn off
    params.add('b_ratio_bd', value=0.0216, min=0.021, max=0.3756, vary=False)  # max=0.0756 #todo: vary turn off
    params.add('b_ratio_d', value=0.0216, min=0.021, max=0.3756, vary=True)
    params.add('A_md', value=13.4685e-3, vary=False)
    params.add('A_bd', value=0.3893e-3, vary=False)
    params.add('S_md', value=10.3845e-3, vary=False)
    params.add('S_bd', value=15.7621e-3, vary=False)
    params.add('S_cdom', value=0.020, min=0.018, max=0.022, vary=True) # Helsinki: 0.018 -0.022; test Baltic: min =0.01
    params.add('C_md', value=12.1700e-3, vary=False)
    params.add('C_bd', value=0.9994e-3, vary=False)
    params.add('K', value=0, min=0, vary=False)
    params.add('lambda_0_cdom', value=440, vary=False)
    params.add('lambda_0_md', value=550, vary=False)
    params.add('lambda_0_bd', value=550, vary=False)
    params.add('lambda_0_c_d', value=550, vary=False)
    params.add('lambda_0_phy', value=676, vary=False)
    params.add('gamma_d', value=0.3835, vary=False)
    params.add('x0', value=1, vary=False)
    params.add('x1', value=10, vary=False)
    params.add('x2', value=-1.3390, min=-1.3390 - 0.0618, max=-1.3390 + 0.0618, vary=False)
    params.add('A', value=0.0237, vary=False)
    params.add('E0', value=1, vary=False)
    params.add('E1', value=0.8987, vary=False)
    params.add('W', value=0.75, vary=False)
    params.add('fwhm1', value=25, vary=False)
    params.add('fwhm2', value=50, vary=False)
    params.add('fwhm_phycocyanin', value=20, vary=False)
    params.add('fwhm_phycoerythrin', value=20, vary=False)
    params.add('lambda_C1', value=685, vary=False)
    params.add('lambda_C2', value=730, vary=False)
    params.add('lambda_C_phycocyanin', value=644, vary=False)
    params.add('lambda_C_phycoerythrin', value=573, vary=False)
    params.add('double', value=True, vary=False)
    params.add('interpolate', value=True, vary=False)
    params.add("Gw0", value=0.05881474, vary=False)
    params.add("Gw1", value=0.05062697, vary=False)
    params.add("Gp0", value=0.03997009, vary=False)
    params.add("Gp1", value=0.1398902, vary=False)
    params.add('error_method', value=0, vary=False)
    params.add('theta_sun', value=np.radians(30), min=np.radians(0), max=np.radians(90), vary=False)
    params.add('theta_view', value=np.radians(1e-10), min=np.radians(1e-10), max=np.radians(90), vary=False)
    params.add('n1', value=1, vary=False)
    params.add('n2', value=1.33, vary=False)
    params.add('kappa_0', value=1.0546, vary=False)
    params.add('fresh', value=False, vary=False)
    params.add('T_W', value=25, min=0, max=40, vary=False)
    params.add('T_W_0', value=20, vary=False)
    params.add('P', value=1013.25, vary=False)
    params.add('AM', value=1, vary=False)
    params.add('RH', value=60, vary=False)
    params.add('H_oz', value=0.38, vary=False)
    params.add('WV', value=2.5, vary=False)
    params.add('alpha', value=1.317, vary=False)
    params.add('beta', value=0.2606, vary=False)
    params.add('g_dd', value=0.02, min=-1, max=10, vary=False)  # glint correction
    params.add('g_dsr', value=1 / np.pi, min=0, max=10, vary=False)  # glint correction
    params.add('g_dsa', value=1 / np.pi, min=0, max=10, vary=False)  # glint correction
    params.add('d_r', value=0, min=0, max=0.1, vary=False)
    params.add('f_dd', value=1, vary=False)
    params.add('f_ds', value=1, vary=False)
    params.add('offset', value=0, min=-0.1, max=0.1, vary=False)
    params.add('fit_surface', value=False, vary=False)
    return params


def set_parameters_byDict(pDict, params):
    for key in pDict.keys():
        thisD = pDict[key]
        params.add(key, value=thisD['value'], min=thisD['min'], max=thisD['max'], vary=True)
    return params


def setDictValues(value, vmin, vmax):
    dd = {'value': value, 'min': vmin, 'max': vmax}
    return dd


def processor_bioOptics_hyperspectral_EnMAPTile(
        versionAC = '',
        regionName = '',
        AlgaeGroupType = '',
        outpath = None, # r"E:\Documents\projects\EnsAD\data\EnMAP_NN_training\extracts_NorthSea\\",
        r_rs=None,
        wlstr=[],
        wavelengths=None):


    paramDict = processingDict_Oder

    ## weights
    weights = np.ones(len(wavelengths))

    @ray.remote
    def invert_chunk(chunk,
                     params,
                     wavelengths,
                     weights,
                     a_md_spec_res,
                     a_bd_spec_res,
                     a_w_res,
                     a_i_spec_res,
                     b_bw_res,
                     b_i_spec_res,
                     h_C_res,
                     h_C_phycocyanin_res,
                     h_C_phycoerythrin_res,
                     da_W_div_dT_res,
                     E_0_res,
                     a_oz_res,
                     a_ox_res,
                     a_wv_res,
                     E_dd_res,
                     E_dsa_res,
                     E_dsr_res,
                     E_d_res,
                     n2_res,
                     method="least_squares",
                     max_nfev=400):
        results = np.array([None] * chunk.shape[0])

        # make a copy of actual params object to enable in-parallel mutation
        chunk_params = params.copy()

        for i in np.arange(chunk.shape[0]):
            # chunk_params.add('theta_sun', value=np.radians(chunk[0,i,0]), vary=False)

            inv = hereon.invert(chunk_params,
                                R_rs=chunk[i, :],
                                Ls_Ed=[],
                                wavelengths=wavelengths,
                                weights=weights,
                                a_md_spec_res=a_md_spec_res,
                                a_bd_spec_res=a_bd_spec_res,
                                a_w_res=a_w_res,
                                a_i_spec_res=a_i_spec_res,
                                b_bw_res=b_bw_res,
                                b_i_spec_res=b_i_spec_res,
                                h_C_res=h_C_res,
                                h_C_phycocyanin_res=h_C_phycocyanin_res,
                                h_C_phycoerythrin_res=h_C_phycoerythrin_res,
                                da_W_div_dT_res=da_W_div_dT_res,
                                E_0_res=E_0_res,
                                a_oz_res=a_oz_res,
                                a_ox_res=a_ox_res,
                                a_wv_res=a_wv_res,
                                E_dd_res=E_dd_res,
                                E_dsa_res=E_dsa_res,
                                E_dsr_res=E_dsr_res,
                                E_d_res=E_d_res,
                                n2_res=n2_res,
                                method=method,
                                max_nfev=max_nfev)
            results[i] = inv

        return results

    data = r_rs.values

    if data.shape[0] < 300:
        num_chunks = 1
    elif data.shape[0] < 1000:
        num_chunks = 3
    else:
        num_chunks = 9  # Number of chunks to split the DF into

    chunk_size = data.shape[0] // num_chunks  # Size of each chunk
    chunks = [data[i:i + chunk_size, :] for i in range(0, data.shape[0], chunk_size)]  # Split the DF into chunks
    while chunks[-1].shape[0] == 1 and num_chunks>0:
        num_chunks -= 1
        if num_chunks >0:
            # print(data.shape[0],  num_chunks)
            chunk_size = data.shape[0] // num_chunks  # Size of each chunk
            chunks = [data[i:i + chunk_size, :] for i in  range(0, data.shape[0], chunk_size)]  # Split the DF into chunks
        else:
            chunks = [data]

    print(chunks[-1].shape[0])
    print('chunks N', len(chunks))
    ray.shutdown()

    # if data.shape[0] < 300:
    #     num_chunks= 1
    # else:
    #     num_chunks = 9  # Number of chunks to split the array into
    #
    # chunk_size = r_rs.values.shape[0] // num_chunks  # Size of each chunk
    # chunks = [data[i:i + chunk_size, :] for i in range(0, data.shape[0], chunk_size)]  # Split the array into chunks

    ray.shutdown()

    start = timeit.default_timer()
    ###
    # Set processing parameters from dictionary!
    params = set_default_parameters(AlgaeGroupType)
    params = set_parameters_byDict(paramDict, params)

    ##
    if params['offset'].vary:
        params.add('fit_surface', value=True, vary=False)

    # Parallelize the processing of the chunks using ray
    chunk_refs = [ray.put(chunk) for chunk in chunks]  # Put the chunks into the object store
    result_refs = [invert_chunk.remote(chunk_ref,
                                       params=params,
                                       wavelengths=wavelengths,
                                       weights=weights,
                                       a_md_spec_res=a_md_spec_res,
                                       a_bd_spec_res=a_bd_spec_res,
                                       a_w_res=a_w_res,
                                       a_i_spec_res=a_i_spec_res,
                                       b_bw_res=b_bw_res,
                                       b_i_spec_res=b_i_spec_res,
                                       h_C_res=h_C_res,
                                       h_C_phycocyanin_res=h_C_phycocyanin_res,
                                       h_C_phycoerythrin_res=h_C_phycoerythrin_res,
                                       da_W_div_dT_res=da_W_div_dT_res,
                                       E_0_res=E_0_res,
                                       a_oz_res=a_oz_res,
                                       a_ox_res=a_ox_res,
                                       a_wv_res=a_wv_res,
                                       E_dd_res=E_dd_res,
                                       E_dsa_res=E_dsa_res,
                                       E_dsr_res=E_dsr_res,
                                       E_d_res=E_d_res,
                                       n2_res=n2_res,
                                       method="least_squares",
                                       max_nfev=1500) for chunk_ref in chunk_refs]  # Process the chunks in parallel

    results = ray.get(result_refs)

    # Concatenate the results from the processed chunks
    processed_data = np.concatenate(results)

    stop = timeit.default_timer()
    print('Time: ', stop - start)

    results = processed_data
    ###
    # Forward Model - Simulate spectra
    ###
    for i in np.arange(len(results)):
        if i == 0:
            R_rs_sim = hereon.forward(parameters=results[i].params,
                                      wavelengths=wavelengths,
                                      a_md_spec_res=a_md_spec_res,
                                      a_bd_spec_res=a_bd_spec_res,
                                      a_w_res=a_w_res,
                                      a_i_spec_res=a_i_spec_res,
                                      b_bw_res=b_bw_res,
                                      b_i_spec_res=b_i_spec_res,
                                      h_C_res=h_C_res,
                                      h_C_phycocyanin_res=h_C_phycocyanin_res,
                                      h_C_phycoerythrin_res=h_C_phycoerythrin_res,
                                      da_W_div_dT_res=da_W_div_dT_res,
                                      E_0_res=E_0_res,
                                      a_oz_res=a_oz_res,
                                      a_ox_res=a_ox_res,
                                      a_wv_res=a_wv_res,
                                      E_dd_res=E_dd_res,
                                      E_dsa_res=E_dsa_res,
                                      E_dsr_res=E_dsr_res,
                                      E_d_res=E_d_res,
                                      n2_res=n2_res,
                                      Ls_Ed=[])
        else:
            R_rs_sim = np.vstack((R_rs_sim, hereon.forward(parameters=results[i].params,
                                                           wavelengths=wavelengths,
                                                           a_md_spec_res=a_md_spec_res,
                                                           a_bd_spec_res=a_bd_spec_res,
                                                           a_w_res=a_w_res,
                                                           a_i_spec_res=a_i_spec_res,
                                                           b_bw_res=b_bw_res,
                                                           b_i_spec_res=b_i_spec_res,
                                                           h_C_res=h_C_res,
                                                           h_C_phycocyanin_res=h_C_phycocyanin_res,
                                                           h_C_phycoerythrin_res=h_C_phycoerythrin_res,
                                                           da_W_div_dT_res=da_W_div_dT_res,
                                                           E_0_res=E_0_res,
                                                           a_oz_res=a_oz_res,
                                                           a_ox_res=a_ox_res,
                                                           a_wv_res=a_wv_res,
                                                           E_dd_res=E_dd_res,
                                                           E_dsa_res=E_dsa_res,
                                                           E_dsr_res=E_dsr_res,
                                                           E_d_res=E_d_res,
                                                           n2_res=n2_res,
                                                           Ls_Ed=[])
                                  ))

    # print('Rrs_sim.shape', R_rs_sim.shape, len(R_rs_sim.shape))
    if len(R_rs_sim.shape) == 1:
        R_rs_sim = R_rs_sim.reshape((1, R_rs_sim.shape[0]))
    R_rs_sim = pd.DataFrame(R_rs_sim, columns=wavelengths.astype(str))


    ###
    # Write results
    ###
    keysList = results[0].params.valuesdict().keys()
    outArr = np.zeros((r_rs.shape[0], len(keysList)))
    for i in range(len(results)):
        for j, key in enumerate(keysList):
            outArr[i, j] = results[i].params.valuesdict().get(key)

    outDF = pd.DataFrame(outArr, columns=keysList)
    outDF['C_phy'] = outDF[['C_0', 'C_1', 'C_2', 'C_3', 'C_4', 'C_5', 'C_6', 'C_7']].sum(axis=1)

    ### create the output-filename
    phyList = ['C_0', 'C_1', 'C_2', 'C_3', 'C_4', 'C_5', 'C_6', 'C_7']
    Ngroups = 0
    groupStr =''
    for phy in phyList:
        if params[phy].vary:
            Ngroups+=1
            groupStr = groupStr+phy.split('_')[1]
    if params['offset'].vary:
        groupStr = groupStr + 'offset'

    # if not outpath is None:
    #     outDF.to_csv(outpath + "inverted_IOP_bio_optics_HEREONfull_" + regionName + "_V"+str(int(Ngroups)) +
    #              "AH_"+groupStr+"_restrict.txt",
    #              sep='\t', header=True, index=False)
    #     R_rs_sim.to_csv(outpath + "inverted_Rrs_bio_optics_HEREONfull_" + regionName + "_V"+str(int(Ngroups)) +
    #              "AH_"+groupStr+"_restrict.txt",
    #                 sep='\t', header=True, index=False)

    return outDF, R_rs_sim

## 20250307
# first test with new specific absorption and scattering values

## Update 20241219:
# - C_3 Cyano_blue for Synechococcus in North Sea
# - add fluorescence 'L_fl_phycoerythrin' for Cryptophytes C_2

### Standard ##

processingDict_Oder = {
    'C_0': setDictValues(0,0,200),
    'C_1': setDictValues(0,0,200),
    'C_2': setDictValues(0,0,300),
    'C_3': setDictValues(0,0,200), # Synechococcus
    # 'C_4': setDictValues(0,0,100),
    'C_5': setDictValues(0,0,10), # coccolith., or gold algae
    # 'C_6': setDictValues(0,0,300),
    'C_Y': setDictValues(0,0,1),
    'C_ism': setDictValues(0,0,50),
    'L_fl_lambda0': setDictValues(0,0,0.2),
    # 'L_fl_phycoerythrin': setDictValues(0,0,0.2),
    'L_fl_phycocyanin': setDictValues(0,0,0.2),
    # 'offset': setDictValues(0, -0.1, 0.1)
}

processingDict_Elbe = {
    # '1': {'C_7': setDictValues(0,0,1)},
    # '2': {'C_0': setDictValues(0,0,1),
    #       'C_2': setDictValues(0,0,1),
    #       'C_3': setDictValues(0,0,1), # Synechococcus
    #       'C_6': setDictValues(0,0,1),
    #       'C_7': setDictValues(0,0,1),
    #       'C_Y': setDictValues(0,0,0.1),
    #       'C_ism': setDictValues(0,0,1)}, # no fluorescence!
    '3a_g': {'C_0': setDictValues(0,0,10),
            # 'C_1': setDictValues(0,0,10),
          # 'C_2': setDictValues(0,0,10),
           'C_3': setDictValues(0,0,10), # Synechococcus
          # 'C_4': setDictValues(0,0,10),
          #  'C_6': setDictValues(0,0,10),
          'C_Y': setDictValues(0,0,1),
          'C_ism': setDictValues(0,0,10),
           'L_fl_lambda0': setDictValues(0,0,0.2),
           # 'L_fl_phycoerythrin': setDictValues(0,0,0.2),
            'L_fl_phycocyanin': setDictValues(0,0,0.2)
        # 'offset': setDictValues(0, -0.1, 0.1)
           },
    '3a_y': {'C_0': setDictValues(0,0,10),
            # 'C_1': setDictValues(0,0,10),
          # 'C_2': setDictValues(0,0,10),
           'C_3': setDictValues(0,0,10), # Synechococcus
          # 'C_4': setDictValues(0,0,10),
          #  'C_6': setDictValues(0,0,10),
          'C_Y': setDictValues(0,0,3),
          'C_ism': setDictValues(0,0,10),
           'L_fl_lambda0': setDictValues(0,0,0.2),
           # 'L_fl_phycoerythrin': setDictValues(0,0,0.2),
            'L_fl_phycocyanin': setDictValues(0,0,0.2)
        # 'offset': setDictValues(0, -0.1, 0.1)
           },
    '3b': {'C_0': setDictValues(0,0,10),
            # 'C_1': setDictValues(0,0,10),
          # 'C_2': setDictValues(0,0,10),
            'C_3': setDictValues(0,0,10),
          #  'C_5': setDictValues(0,0,10), # coccolith.
          # 'C_6': setDictValues(0,0,10),
          'C_Y': setDictValues(0,0,10),
          'C_ism': setDictValues(0,0,10),
           'L_fl_lambda0': setDictValues(0,0,0.2),
            'L_fl_phycocyanin': setDictValues(0,0,0.2)
           # 'L_fl_phycoerythrin': setDictValues(0,0,0.2)
        # 'offset': setDictValues(0, -0.1, 0.1)
           },
    '4a_g': {'C_0': setDictValues(0,0,30),
            # 'C_1': setDictValues(0,0,30),
          # 'C_2': setDictValues(0,0,30),
           'C_3': setDictValues(0,0,30), # Synechococcus
          # 'C_4': setDictValues(0,0,30),
          # 'C_6': setDictValues(0,0,30),
          'C_Y': setDictValues(0,0,5),
          'C_ism': setDictValues(0,0,20),
           'L_fl_lambda0': setDictValues(0,0,0.2),
           # 'L_fl_phycoerythrin': setDictValues(0,0,0.2),
           'L_fl_phycocyanin': setDictValues(0,0,0.2),
        # 'offset': setDictValues(0, -0.1, 0.1)
             },
    '4a_y': {'C_0': setDictValues(0,0,30),
        'C_1': setDictValues(0,0,30),
          # 'C_2': setDictValues(0,0,30),
           'C_3': setDictValues(0,0,30), # Synechococcus
          # 'C_4': setDictValues(0,0,30),
          # 'C_6': setDictValues(0,0,30),
          'C_Y': setDictValues(0,0,1),
          'C_ism': setDictValues(0,0,20),
           'L_fl_lambda0': setDictValues(0,0,0.2),
           # 'L_fl_phycoerythrin': setDictValues(0,0,0.2),
           'L_fl_phycocyanin': setDictValues(0,0,0.2),
        # 'offset': setDictValues(0, -0.1, 0.1)
             },
    '4b': {'C_0': setDictValues(0,0,200),
        'C_1': setDictValues(0,0,200),
        #   'C_2': setDictValues(0,0,300),
        'C_3': setDictValues(0,0, 200), # Synechococcus
        #   'C_4': setDictValues(0,0,100),
        #     'C_5': setDictValues(0,0,10), # coccolith.
        #   'C_6': setDictValues(0,0,300),
          'C_Y': setDictValues(0,0,1),
          'C_ism': setDictValues(0,0,20), #v0.2b: 0-50, v0.3b: 0-50
           'L_fl_lambda0': setDictValues(0,0,0.2),
           # 'L_fl_phycoerythrin': setDictValues(0,0,0.2),
           'L_fl_phycocyanin': setDictValues(0,0,0.2),
        # 'offset': setDictValues(0, -0.1, 0.1)
           },
    '5a': {'C_0': setDictValues(0.,0,200),
           'C_1': setDictValues(0,0,200),
          # 'C_2': setDictValues(0,0,300),
           'C_3': setDictValues(0,0,200), # Synechococcus
          # 'C_4': setDictValues(0,0,300),
          #  'C_6': setDictValues(0,0,300),
          'C_Y': setDictValues(0,0,1),
          'C_ism': setDictValues(0,0,20), #v0.2b: 0-50, v0.3b: 0-50
           'L_fl_lambda0': setDictValues(0,0,0.2),
            # 'L_fl_phycoerythrin': setDictValues(0,0,0.2),
            'L_fl_phycocyanin': setDictValues(0,0,0.2),
           # 'offset': setDictValues(0, -0.1, 0.1)
           },
    '5b': {'C_0': setDictValues(0,0,200),
            'C_1': setDictValues(0,0,200),
          # 'C_2': setDictValues(0,0,1000),
            'C_3': setDictValues(0,0,200), # Synechococcus
          # 'C_4': setDictValues(0,0,300),
          # 'C_6': setDictValues(0,0,1000),
          'C_Y': setDictValues(0,0, 1),
          'C_ism': setDictValues(0,0, 20), #v0.2b: 0-50, v0.3b: 0-50
           'L_fl_lambda0': setDictValues(0,0,0.2),
            # 'L_fl_phycoerythrin': setDictValues(0,0,0.2),
            'L_fl_phycocyanin': setDictValues(0,0,0.2),
           # 'offset': setDictValues(0, -0.1, 0.1)
           },
    '6': {'C_0': setDictValues(0.,0,200),
          'C_1': setDictValues(0.,0,200),
          # 'C_2': setDictValues(0,0,500),
            'C_3': setDictValues(0,0,200), # Synechococcus
          # 'C_4': setDictValues(0,0,300),
          # 'C_6': setDictValues(0,0,500),
          'C_Y': setDictValues(0,0,1),
          'C_ism': setDictValues(0,0, 20), #v0.2b: 0-50, v0.3b: 0-50
           'L_fl_lambda0': setDictValues(0,0,0.2),
          # 'L_fl_phycoerythrin': setDictValues(0,0,0.2),
          'L_fl_phycocyanin': setDictValues(0,0,0.2),
        # 'offset': setDictValues(0, -0.1, 0.1)
          },
    '7': {'C_0': setDictValues(0.,0,200),
          # 'C_1': setDictValues(0.,0,200),
          # 'C_2': setDictValues(0,0,200),
            'C_3': setDictValues(0,0,200), # Synechococcus
          # 'C_4': setDictValues(0,0,200),
          # 'C_6': setDictValues(0,0,200),
          'C_Y': setDictValues(0,0,1),
          'C_ism': setDictValues(0,0,2),
           'L_fl_lambda0': setDictValues(0,0,0.2),
          # 'L_fl_phycoerythrin': setDictValues(0,0,0.2),
          'L_fl_phycocyanin': setDictValues(0,0,0.2),
        # 'offset': setDictValues(0, -0.1, 0.1)
          }
}

### Begin MAIN ###
regionName='Oder'
sensor = 'EnMAP'
versionAC = 'EnMAPLandv1.5.2'
dataType = 'EnMAPTile' # 'Sim'
AlgaeGroupType = 'HEREONgold' #'Standardv2' # 'Standardv3' # 'NSSummerBloomsv3' Standardv2, 'HEREONgold'

# maxWavelength= 750.
maxWavelength = 900.

path = "D:\Documents\projects\EnsAD\EnMAP\L2A_Land\\v010502\Fluss_Elbe_Oder\\"
outpath = "E:\Documents\projects\EnsAD\inversion\\bio_optics_EnMAPtiles\\"
fnames = os.listdir(path)
print(fnames[0])
img = xr.open_zarr(path + fnames[0])

rrs = img.where(img.quality_classes==2)['band_data'].isel(band=slice(0,80)) / np.pi
wavelengths = rrs.band.values

print(rrs.shape)

## HEREON default (like web-version)
## oder_frankfurt:
anap_spec440 = 0.03617
S_xd= 0.0093
a_md_spec_res = absorption.a_md_spec(wavelengths=wavelengths, A_md=anap_spec440, S_md=S_xd, lambda_0=440.)

a_bd_spec_res = absorption.a_bd_spec(wavelengths=wavelengths)
a_w_res = resampling.resample_a_w(wavelengths=wavelengths)
if AlgaeGroupType == 'Standardv2':
    a_i_spec_res = resampling.resample_a_i_spec_EnSAD(wavelengths=wavelengths) # prototype v2, PACEv2
    b_i_spec_res = resampling.resample_b_i_spec_EnSAD(wavelengths=wavelengths)  # prototype v2, PACEv2
if AlgaeGroupType == 'Standardv3':
    a_i_spec_res = resampling.resample_a_i_spec_EnSAD_Standardv3(wavelengths=wavelengths) # prototype v3, PACEv3
    b_i_spec_res = resampling.resample_b_i_spec_EnSAD_Standardv3(wavelengths=wavelengths)  # prototype v3, PACEv3
if AlgaeGroupType == 'NSSummerBloomsv3':
    a_i_spec_res = resampling.resample_a_i_spec_EnSAD_SummerBloomsv3(wavelengths=wavelengths) # prototype v3, PACEv3
    b_i_spec_res = resampling.resample_b_i_spec_EnSAD_SummerBloomsv3(wavelengths=wavelengths)  # prototype v3, PACEv3
if AlgaeGroupType == 'HEREONgold': #HEREON (- coccolithophore + goldalgae) + dinoflagellate
    a_i_spec_res = resampling.resample_a_i_spec_EnSADandGold(wavelengths=wavelengths)  # AQUATIME Vortsjarv
    b_i_spec_res = resampling.resample_b_i_spec_EnSADandGold(wavelengths=wavelengths)  # AQUATIME Vortsjarv

b_bw_res = backscattering.b_bw(wavelengths=wavelengths, fresh=False)

# print(a_md_spec_res.shape, a_bd_spec_res.shape, a_i_spec_res.shape, b_i_spec_res.shape, len(wavelengths))

da_W_div_dT_res = resampling.resample_da_W_div_dT(wavelengths=wavelengths)
h_C_res = fluorescence.h_C_double(wavelengths=wavelengths, W=0.75)
h_C_phycocyanin_res = fluorescence.h_C(wavelengths=wavelengths, fwhm=20, lambda_C=644)
h_C_phycoerythrin_res =fluorescence.h_C(wavelengths=wavelengths, fwhm=20, lambda_C=573)
omega_d_lambda_0_res = attenuation.omega_d_lambda_0()

E_0_res = resampling.resample_E_0(wavelengths=wavelengths)
a_oz_res = resampling.resample_a_oz(wavelengths=wavelengths)
a_ox_res = resampling.resample_a_ox(wavelengths=wavelengths)
a_wv_res = resampling.resample_a_wv(wavelengths=wavelengths)
n2_res = resampling.resample_n(wavelengths=wavelengths)

E_dd_res = downwelling_irradiance.E_dd(wavelengths=wavelengths)
E_dsa_res = downwelling_irradiance.E_dsa(wavelengths=wavelengths)
E_dsr_res = downwelling_irradiance.E_dsr(wavelengths=wavelengths)
E_d_res = E_dd_res + E_dsa_res + E_dsr_res

## Start INVERSION ##
paramDict = processingDict_Oder
params = set_default_parameters(AlgaeGroupType)
params = set_parameters_byDict(paramDict, params)

## weights
weights = np.ones(len(wavelengths))

def invert_dask(cube1,  # this is remote sensing reflectance (R_rs)
                params,
                wavelengths,
                weights,
                a_md_spec_res,
                a_bd_spec_res,
                a_w_res,
                a_i_spec_res,
                b_bw_res,
                b_i_spec_res,
                h_C_res,
                h_C_phycocyanin_res,
                h_C_phycoerythrin_res,
                da_W_div_dT_res,
                E_0_res,
                a_oz_res,
                a_ox_res,
                a_wv_res,
                E_dd_res,
                E_dsa_res,
                E_dsr_res,
                E_d_res,
                n2_res,
                method="least_squares",
                max_nfev=400,
                fit_metadata_attrs=['chisqr', 'nfev', 'success'],  # from lmfit
                return_residual=False):
    # reshape from 3D to 2D to only have one for-loop
    R_rs = np.reshape(cube1, (cube1.shape[0], -1))
    # zB = np.reshape(cube2, (cube2.shape[0], -1))

    # make a copy of actual params str to make it mutable
    chunk_params = params  # here, params is already a str, so assigning it makes a copy internally
    # automatically construct list of varying_params from params by rebuilding the lmfit.Parameters() object to make sure results has the right size
    # this is ugly and there must be a more elegant solution!
    parameters = lmfit.Parameters().loads(params)
    varying_params = [key for key in parameters.keys() if parameters[key].vary or parameters[key].expr]

    # initialize results array
    results = np.zeros((len(varying_params), R_rs.shape[1]))

    if len(fit_metadata_attrs) > 0:
        # initialize array to store metadata about the fit
        fit_metadata = np.zeros((len(fit_metadata_attrs), R_rs.shape[1]))

    if return_residual:
        # initialize array to store band-wise residual
        residuals = np.zeros(R_rs.shape)

    # invert spectrum-wise
    for i in range(R_rs.shape[1]):

        # get R_rs spectrum
        R_rs_i = R_rs[:, i]
        # then dump to make a string for handover to model.invert() ... this is somewhat confusing!
        chunk_params = parameters.dumps()

        # test if any input contains nans
        if np.isnan(R_rs_i).any():
            results[:, i] = np.nan
            continue

        result = hereon.invert(chunk_params,
                                R_rs=R_rs_i,
                                Ls_Ed=[],
                                wavelengths=wavelengths,
                                weights=weights,
                                a_md_spec_res=a_md_spec_res,
                                a_bd_spec_res=a_bd_spec_res,
                                a_w_res=a_w_res,
                                a_i_spec_res=a_i_spec_res,
                                b_bw_res=b_bw_res,
                                b_i_spec_res=b_i_spec_res,
                                h_C_res=h_C_res,
                                h_C_phycocyanin_res=h_C_phycocyanin_res,
                                h_C_phycoerythrin_res=h_C_phycoerythrin_res,
                                da_W_div_dT_res=da_W_div_dT_res,
                                E_0_res=E_0_res,
                                a_oz_res=a_oz_res,
                                a_ox_res=a_ox_res,
                                a_wv_res=a_wv_res,
                                E_dd_res=E_dd_res,
                                E_dsa_res=E_dsa_res,
                                E_dsr_res=E_dsr_res,
                                E_d_res=E_d_res,
                                n2_res=n2_res,
                                method=method,
                                max_nfev=max_nfev)

        # fill results array
        results[:, i] = np.array([param.value for name, param in result.params.items() if param.vary or param.expr])

        if len(fit_metadata_attrs) > 0:
            # fill fit metadata array
            fit_metadata[:, i] = np.array([getattr(result, attr) for attr in fit_metadata_attrs])

        if return_residual:
            # fill residuals array
            residuals[:, i] = result.residual

    output = results.reshape((len(varying_params), cube1.shape[1], cube1.shape[2]))

    if len(fit_metadata_attrs) > 0:
        output = np.concatenate(
            (output, fit_metadata.reshape((len(fit_metadata_attrs), cube1.shape[1], cube1.shape[2]))), axis=0)

    if return_residual:
        output = np.concatenate((output, residuals.reshape((cube1.shape[0], cube1.shape[1], cube1.shape[2]))), axis=0)

    return output

# prep cubes and make sure they both have the same chunking
cube1 = da.stack(rrs)
cube1 = cube1.rechunk(chunks=(cube1.shape[0],32,32))

graph = da.map_blocks(invert_dask,
                      cube1,
                      meta=np.array((), dtype=np.float32),
                      dtype=np.float32,
                      params=params.dumps(), # dump params to convert to str to make params serializable
                       wavelengths=wavelengths,
                       weights=weights,
                       a_md_spec_res=a_md_spec_res,
                       a_bd_spec_res=a_bd_spec_res,
                       a_w_res=a_w_res,
                       a_i_spec_res=a_i_spec_res,
                       b_bw_res=b_bw_res,
                       b_i_spec_res=b_i_spec_res,
                       h_C_res=h_C_res,
                       h_C_phycocyanin_res=h_C_phycocyanin_res,
                       h_C_phycoerythrin_res=h_C_phycoerythrin_res,
                       da_W_div_dT_res=da_W_div_dT_res,
                       E_0_res=E_0_res,
                       a_oz_res=a_oz_res,
                       a_ox_res=a_ox_res,
                       a_wv_res=a_wv_res,
                       E_dd_res=E_dd_res,
                       E_dsa_res=E_dsa_res,
                       E_dsr_res=E_dsr_res,
                       E_d_res=E_d_res,
                       n2_res=n2_res,
                      method="least_squares",
                      max_nfev=400
                      )

start = timeit.default_timer()
result = graph.compute(scheduler='processes')
stop = timeit.default_timer()
print('Time: ', stop - start)

varying_params = [key for key in params.keys() if params[key].vary or params[key].expr]
fit_metadata_attrs = ['chisqr','nfev','success']

bands2plot = varying_params + fit_metadata_attrs

fit_params = xr.Dataset(
    {band: (["y", "x"], result[i]) for i, band in enumerate(bands2plot)},
    coords={"y": rrs.y, "x": rrs.x}
)

fit_params.to_netcdf(outpath + 'fitBioOptics_'+fnames[0].split('.')[0]+'.nc')


# resultInv, resultSim = processor_bioOptics_hyperspectral_EnMAPTile(
#     versionAC = versionAC,
#     regionName = regionName,
#     AlgaeGroupType = AlgaeGroupType,
#     path = path,
#     outpath=None,
#     r_rs=rrs,
#     wavelengths=wavelengths,
#     wlstr=wlstr)



# resultInvDF = pd.DataFrame(resultInvArr, columns=thisResultInv.columns.values)
# resultInvDF['date'] = dat['date'].values
# resultInvDF.to_csv(outpath + "inverted_IOP_bio_optics_HEREONfull_" + datasetName + "_allOWTs_Inv0.1.txt")
# resultSimDF = pd.DataFrame(resultSimArr, columns=thisResultSim.columns.values)
# resultSimDF['date'] = dat['date'].values
# resultSimDF.to_csv(outpath + "inverted_Sim_bio_optics_HEREONfull_" + datasetName + "_allOWTs_Inv0.1.txt")
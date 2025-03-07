# Automatically full inversion process by Bi&Hieronymi OWTs
# dictionary of ranges and parameters per OWT and area
# read in data

### use with env py38_keras3
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import scipy
from scipy.interpolate import UnivariateSpline
import lmfit
import ray
import os
import timeit

from bio_optics.water import absorption, attenuation, backscattering, scattering, lee, fluorescence
from bio_optics.atmosphere import downwelling_irradiance
from bio_optics.models import hereon, model, bandratioIOPprediction
from bio_optics.helper import resampling, utils, owt, indices, plotting

from bio_optics.processors.read_data import read_EnMAP_extracts, set_wavelengths_bySensor, read_PACE_extracts


def set_default_parameters():
    ## by Default:
    # - all Phytoplankton groups are value=0, min=0, max=1000 (C_5=10, C_7=1), vary = False
    # - C_Y value=0, min=0, max=30, vary=True
    # - C_ism value=0, min=0, max=100, vary=True
    # - fluorescence all vary = False
    # - offset vary = False
    params = lmfit.Parameters()
    params.add('C_0', value=0, min=0, max=100, vary=False)  # brown
    params.add('C_1', value=0, min=0, max=100, vary=False)  # green
    params.add('C_2', value=0, min=0, max=100, vary=False)  # cryptophyte
    params.add('C_3', value=0, min=0, max=100, vary=False)  # cyano blue
    params.add('C_4', value=0, min=0, max=100, vary=False)  # cyano red
    params.add('C_5', value=0, min=0, max=10, vary=False)  # coccolithophores
    params.add('C_6', value=0, min=0, max=100, vary=False)  # dinoflagellates
    params.add('C_7', value=0, min=0, max=1, vary=False)  # case-1
    params.add('C_Y', value=0, min=0, max=30, vary=True)
    params.add('C_ism', value=0, min=0, max=100, vary=True)
    params.add('L_fl_lambda0', value=0, min=0, max=0.2, vary=False)
    params.add('L_fl_phycocyanin', value=0, min=0, max=0.2, vary=False)
    params.add('L_fl_phycoerythrin', value=0, min=0, max=0.2, vary=False)
    params.add('b_ratio_C_0', value=0.002, vary=False)  # brown
    params.add('b_ratio_C_1', value=0.007, vary=False)  # green
    params.add('b_ratio_C_2', value=0.002, vary=False)  # cryptophyte
    params.add('b_ratio_C_3', value=0.001, vary=False)  # cyano blue
    params.add('b_ratio_C_4', value=0.001, vary=False)  # cyano red
    params.add('b_ratio_C_5', value=0.007, vary=False)  # coccolithophores
    params.add('b_ratio_C_6', value=0.007, vary=False)  # dinoflagellates , chose 0.007 because of smaller cell size
    params.add('b_ratio_C_7', value=0.007, vary=False)  # case-1
    params.add('b_ratio_md', value=0.0216, min=0.021, max=0.3756, vary=True)  # max=0.0756
    params.add('b_ratio_bd', value=0.0216, min=0.021, max=0.3756, vary=True)  # max=0.0756
    # params.add('b_ratio_d', value=0.0216, min=0.021, max=0.3756, vary=True)
    params.add('A_md', value=13.4685e-3, vary=False)
    params.add('A_bd', value=0.3893e-3, vary=False)
    params.add('S_md', value=10.3845e-3, vary=False)
    params.add('S_bd', value=15.7621e-3, vary=False)
    params.add('S_cdom', value=0.0185, min=0.005, max=0.032, vary=True)
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


def processor_bioOptics_hyperspectral_byOWT(
        datasetDate = '',
        datasetID = '',
        OWTsingleList = [], # single OWTs only!
        outpath = r"E:\Documents\projects\EnsAD\data\EnMAP_NN_training\extracts_NorthSea\\"):

    if sensor == 'EnMAP':
        r_rs, wlstr, wavelengths, datasetName = read_EnMAP_extracts(seaName=seaName,
                                                 OWTList=OWTsingleList,
                                                 outpath=outpath,
                                                 datasetDate=datasetDate,
                                                 datasetID=datasetID)
    if sensor == 'PACE':
        r_rs, wlstr, wavelengths, datasetName = read_PACE_extracts(seaName=seaName,
                                                 OWTList=OWTsingleList,
                                                 outpath=outpath,
                                                 datasetDate=datasetDate)


    if seaName == 'North Sea':
        paramDict = processingDict_NorthSea[OWTsingleList[0]]
    if seaName == 'Baltic Sea':
        paramDict = processingDict_BalticSea[OWTsingleList[0]]

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

    num_chunks = 6  # Number of chunks to split the array into

    chunk_size = r_rs.values.shape[0] // num_chunks  # Size of each chunk
    chunks = [data[i:i + chunk_size, :] for i in range(0, data.shape[0], chunk_size)]  # Split the array into chunks

    ray.shutdown()

    start = timeit.default_timer()
    ###
    # Set processing parameters from dictionary!
    params = set_default_parameters()
    params = set_parameters_byDict(paramDict, params)

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


    outDF.to_csv(outpath + "inverted_IOP_bio_optics_HEREONfull_" + datasetName + "_V"+str(int(Ngroups)) +
                 "AH_"+groupStr+".txt",
                 sep='\t', header=True, index=False)
    R_rs_sim.to_csv(outpath + "inverted_Rrs_bio_optics_HEREONfull_" + datasetName + "_V"+str(int(Ngroups)) +
                 "AH_"+groupStr+".txt",
                    sep='\t', header=True, index=False)

    return None



## Update 20241219:
# - C_3 Cyano_blue for Synechococcus in North Sea
# - add fluorescence 'L_fl_phycoerythrin' for Cryptophytes C_2

processingDict_NorthSea = {
    '1': {'C_7': setDictValues(0,0,1)},
    '2': {'C_0': setDictValues(0,0,1),
          'C_2': setDictValues(0,0,1),
          'C_3': setDictValues(0,0,1), # Synechococcus
          'C_6': setDictValues(0,0,1),
          'C_Y': setDictValues(0,0,0.1),
          'C_ism': setDictValues(0,0,1)}, # no fluorescence!
    '3a': {'C_0': setDictValues(0.1,0,10),
          'C_2': setDictValues(0,0,10),
           'C_3': setDictValues(0,0,10), # Synechococcus
          'C_6': setDictValues(0,0,10),
          'C_Y': setDictValues(0,0,1),
          'C_ism': setDictValues(0,0,10),
           'L_fl_lambda0': setDictValues(0,0,0.2),
           'L_fl_phycoerythrin': setDictValues(0,0,0.2)
           },
    '3b': {'C_0': setDictValues(0.1,0,10),
          'C_2': setDictValues(0,0,10),
           'C_5': setDictValues(0,0,10), # coccolith.
          'C_6': setDictValues(0,0,10),
          'C_Y': setDictValues(0,0,10),
          'C_ism': setDictValues(0,0,10),
           'L_fl_lambda0': setDictValues(0,0,0.2),
           'L_fl_phycoerythrin': setDictValues(0,0,0.2)},
    '4a': {'C_0': setDictValues(0.1,0,30),
          'C_2': setDictValues(0,0,30),
           'C_3': setDictValues(0,0,30), # Synechococcus
          'C_6': setDictValues(0,0,30),
          'C_Y': setDictValues(0,0,5),
          'C_ism': setDictValues(0,0,20),
           'L_fl_lambda0': setDictValues(0,0,0.2),
           'L_fl_phycoerythrin': setDictValues(0,0,0.2)},
    '4b': {'C_0': setDictValues(0.1,0,300),
          'C_2': setDictValues(0,0,300),
            'C_5': setDictValues(0,0,10), # coccolith.
          'C_6': setDictValues(0,0,300),
          'C_Y': setDictValues(0,0,10),
          'C_ism': setDictValues(0,0,100),
           'L_fl_lambda0': setDictValues(0,0,0.2),
           'L_fl_phycoerythrin': setDictValues(0,0,0.2)},
    '5a': {'C_0': setDictValues(0.1,0,300),
          'C_2': setDictValues(0,0,300),
           'C_3': setDictValues(0,0,300), # Synechococcus
          'C_6': setDictValues(0,0,300),
          'C_Y': setDictValues(0,0,100),
          'C_ism': setDictValues(0,0,100),
           'L_fl_lambda0': setDictValues(0,0,0.2),
            'L_fl_phycoerythrin': setDictValues(0,0,0.2),
           'offset': setDictValues(0, -0.1, 0.1)},
    '5b': {'C_0': setDictValues(0.1,0,1000),
            'C_1': setDictValues(0.1,0,100),
          'C_2': setDictValues(0,0,1000),
          'C_6': setDictValues(0,0,1000),
          'C_Y': setDictValues(0,0,100),
          'C_ism': setDictValues(0,0,100),
           'L_fl_lambda0': setDictValues(0,0,0.2),
            'L_fl_phycoerythrin': setDictValues(0,0,0.2),
           'offset': setDictValues(0, -0.1, 0.1)},
    '6': {'C_0': setDictValues(0.1,0,500),
          'C_1': setDictValues(0.,0,500),
          'C_2': setDictValues(0,0,500),
          'C_6': setDictValues(0,0,500),
          'C_Y': setDictValues(0,0,100),
          'C_ism': setDictValues(0,0,1000),
           'L_fl_lambda0': setDictValues(0,0,0.2),
          'L_fl_phycoerythrin': setDictValues(0,0,0.2)},
    '7': {'C_0': setDictValues(0.1,0,200),
          'C_1': setDictValues(0.,0,200),
          'C_2': setDictValues(0,0,200),
          'C_6': setDictValues(0,0,200),
          'C_Y': setDictValues(0,0,30),
          'C_ism': setDictValues(0,0,100),
           'L_fl_lambda0': setDictValues(0,0,0.2),
          'L_fl_phycoerythrin': setDictValues(0,0,0.2)}
}

processingDict_BalticSea = {
    '1': {'C_7': setDictValues(0,0,1)},
    '2': {'C_0': setDictValues(0,0,1),
          'C_2': setDictValues(0,0,1),
          'C_6': setDictValues(0,0,1),
          'C_Y': setDictValues(0,0,0.1),
          'C_ism': setDictValues(0,0,1)}, # no fluorescence!
    '3a': {'C_0': setDictValues(0.1,0,10),
          'C_2': setDictValues(0,0,10),
           'C_3' : setDictValues(0,0,10),
          'C_4' : setDictValues(0,0,10),
          'C_6': setDictValues(0,0,10),
          'C_Y': setDictValues(0,0,1),
          'C_ism': setDictValues(0,0,10),
           'L_fl_lambda0': setDictValues(0,0,0.2),
           'L_fl_phycocyanin': setDictValues(0,0,0.2),
           'L_fl_phycoerythrin': setDictValues(0,0,0.2)},
    '3b': {'C_0': setDictValues(0.1,0,10),
          'C_2': setDictValues(0,0,10),
           'C_3' : setDictValues(0,0,10),
          'C_4' : setDictValues(0,0,10),
          'C_6': setDictValues(0,0,10),
          'C_Y': setDictValues(0,0,10),
          'C_ism': setDictValues(0,0,10),
           'L_fl_lambda0': setDictValues(0,0,0.2),
           'L_fl_phycocyanin': setDictValues(0,0,0.2),
           'L_fl_phycoerythrin': setDictValues(0,0,0.2)},
    '4a': {'C_0': setDictValues(0.1,0,30),
          'C_2': setDictValues(0,0,30),
           'C_3' : setDictValues(0,0,30),
          'C_4' : setDictValues(0,0,30),
          'C_6': setDictValues(0,0,30),
          'C_Y': setDictValues(0,0,5),
          'C_ism': setDictValues(0,0,20),
           'L_fl_lambda0': setDictValues(0,0,0.2),
           'L_fl_phycocyanin': setDictValues(0,0,0.2),
           'L_fl_phycoerythrin': setDictValues(0,0,0.2)},
    '4b': {'C_0': setDictValues(0.1,0,300),
          'C_2': setDictValues(0,0,300),
           'C_3' : setDictValues(0,0,300),
          'C_4' : setDictValues(0,0,300),
          'C_6': setDictValues(0,0,300),
          'C_Y': setDictValues(0,0,10),
          'C_ism': setDictValues(0,0,100),
           'L_fl_lambda0': setDictValues(0,0,0.2),
           'L_fl_phycocyanin': setDictValues(0,0,0.2),
           'L_fl_phycoerythrin': setDictValues(0,0,0.2)},
    '5a': {'C_0': setDictValues(0.1,0,300),
          'C_2': setDictValues(0,0,300),
           'C_3' : setDictValues(0,0,300),
          'C_4' : setDictValues(0,0,300),
          'C_6': setDictValues(0,0,300),
          'C_Y': setDictValues(0,0,100),
          'C_ism': setDictValues(0,0,100),
           'L_fl_lambda0': setDictValues(0,0,0.2),
           'L_fl_phycocyanin': setDictValues(0,0,0.2),
           'L_fl_phycoerythrin': setDictValues(0,0,0.2),
           'offset': setDictValues(0, -0.1, 0.1)},
    '5b': {'C_0': setDictValues(0.1,0,1000),
            'C_1': setDictValues(0.1,0,100),
          'C_2': setDictValues(0,0,1000),
           'C_3' : setDictValues(0,0,1000),
          'C_4' : setDictValues(0,0,1000),
          'C_6': setDictValues(0,0,1000),
          'C_Y': setDictValues(0,0,100),
          'C_ism': setDictValues(0,0,100),
           'L_fl_lambda0': setDictValues(0,0,0.2),
           'L_fl_phycocyanin': setDictValues(0,0,0.2),
           'L_fl_phycoerythrin': setDictValues(0,0,0.2),
           'offset': setDictValues(0, -0.1, 0.1)},
    '6': {'C_0': setDictValues(0.1,0,500),
          'C_1': setDictValues(0.1,0,500),
          'C_2': setDictValues(0,0,500),
          'C_3' : setDictValues(0,0,500),
          'C_4' : setDictValues(0,0,500),
          'C_6': setDictValues(0,0,500),
          'C_Y': setDictValues(0,0,100),
          'C_ism': setDictValues(0,0,1000),
           'L_fl_lambda0': setDictValues(0,0,0.2),
           'L_fl_phycocyanin': setDictValues(0,0,0.2),
           'L_fl_phycoerythrin': setDictValues(0,0,0.2)},
    '7': {'C_0': setDictValues(0.1,0,100),
          'C_1': setDictValues(0.1,0,100),
          'C_2': setDictValues(0,0,100),
          'C_3' : setDictValues(0,0,100),
          'C_4' : setDictValues(0,0,100),
          'C_6': setDictValues(0,0,100),
          'C_Y': setDictValues(0,0,30),
          'C_ism': setDictValues(0,0,100),
           'L_fl_lambda0': setDictValues(0,0,0.2),
           'L_fl_phycocyanin': setDictValues(0,0,0.2),
           'L_fl_phycoerythrin': setDictValues(0,0,0.2)}
}


### Begin MAIN ###
# sensor = 'EnMAP'
# versionAC = 'v010402'
# dataType = 'PCA'
# seaName = 'North Sea'
# datasetDate = '20241118',
# datasetID = 'membSumFilt2_QWIPfilt',
# wavelengths = set_wavelengths_bySensor(sensor, versionAC)

sensor = 'PACE'
versionAC = 2
dataType = 'Orig'
seaName = 'North Sea'
datasetDate = '20241001'
datasetID = 'Rrs_flags'
wavelengths = set_wavelengths_bySensor(sensor, versionAC)

# global inputs that don't change with fit params
a_md_spec_res = absorption.a_md_spec(wavelengths=wavelengths)
a_bd_spec_res = absorption.a_bd_spec(wavelengths=wavelengths)
a_w_res = resampling.resample_a_w(wavelengths=wavelengths)
a_i_spec_res = resampling.resample_a_i_spec_EnSAD(wavelengths=wavelengths)
b_bw_res = backscattering.b_bw(wavelengths=wavelengths, fresh=False)
b_i_spec_res = resampling.resample_b_i_spec_EnSAD(wavelengths=wavelengths)
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


OWTList = ['2', '3a', '3b', '4a', '4b', '5a', '5b', '6', '7']
# outpath = "E:\Documents\projects\EnsAD\data\EnMAP_NN_training\\automated_NorthSea\\"
outpath = "E:\Documents\projects\EnsAD\data\PACE_NN_training\PACE_fullInversion_NorthSea_update_20241219\\"
for owt in OWTList[6:7]:
    OWTsingleList = [owt]
    processor_bioOptics_hyperspectral_byOWT(
        datasetDate = datasetDate,
        datasetID = datasetID,
        OWTsingleList = OWTsingleList, # single OWTs only!
        outpath=outpath )
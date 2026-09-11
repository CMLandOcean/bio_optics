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

from bio_optics.processors.read_data import read_EnMAP_extracts, set_wavelengths_bySensor

from bio_optics.helper.OWT_BI import OWT
from bio_optics.helper.OpticalVariables import OpticalVariables


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
    params.add('C_Y', value=0, min=0, max=30, vary=False)
    params.add('C_ism', value=0, min=0, max=100, vary=False)
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
        params.add('b_ratio_C_5', value=0.0129, vary=False)  # coccolithophores 0.0129, , HEREONweb: 0.007, Phaeocystis: change to 0.0034
    params.add('b_ratio_C_6', value=0.0209, vary=False)  # dinoflagellates 0.0209, HEREONweb: not available
    # if AlgaeGroupType == 'NSSummerBloomsv3':
    if AlgaeGroupType == 'Summer':
        params.add('b_ratio_C_7', value=0.0209, vary=False)  # Noctiluca: change to 0.0209
    else:
        params.add('b_ratio_C_7', value=0.007, vary=False)  # case-1 0.0109, HEREONweb: 0.007, Noctiluca: change to 0.0209
    params.add('b_ratio_md', value=0.0216, min=0.021, max=0.3756, vary=False)  # max=0.0756 #todo: vary turn off
    params.add('b_ratio_bd', value=0.0216, min=0.021, max=0.3756, vary=False)  # max=0.0756 #todo: vary turn off
    params.add('b_ratio_d', value=0.0216, min=0.021, max=0.3756, vary=False)
    params.add('A_md', value=13.4685e-3, vary=False)
    params.add('A_bd', value=0.3893e-3, vary=False)
    params.add('S_md', value=10.3845e-3, vary=False)
    params.add('S_bd', value=15.7621e-3, vary=False)
    params.add('S_cdom', value=0.020, min=0.018, max=0.022, vary=False) # Helsinki: 0.018 -0.022; test Baltic: min =0.01
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
    params.add('fd_d', value=1, vary=False)
    params.add('fd_s', value=1, vary=False)
    params.add('offset', value=0, min=-0.1, max=0.1, vary=True) # simple offset to spectrum
    params.add('fit_surface', value=False, vary=False) # starts the glint correction!
    params.add('estimate_start_values', value=False, vary=False) # use start values from estimates (linear equations)
    return params


def set_parameters_byDict(pDict, params):
    for key in pDict.keys():
        thisD = pDict[key]
        params.add(key, value=thisD['value'], min=thisD['min'], max=thisD['max'], vary=True)
    return params

def set_fixedParameters_byDict(uvDict, params):
    for key in uvDict.keys():
        params.add(key, value=uvDict[key], vary=False)
    return params

def setDictValues(value, vmin, vmax):
    dd = {'value': value, 'min': vmin, 'max': vmax}
    return dd


def processor_bioOptics_hyperspectral_byOWT(
        regionName = '',
        AlgaeGroupType = '',
        OWTsingleList = [], # single OWTs only!
        outpath = None, # r"E:\Documents\projects\EnsAD\data\EnMAP_NN_training\extracts_NorthSea\\",
        r_rs=None,
        wavelengths=None,
        datasetName='',
        checkZeroSpectrum= False):

    # if sensor == 'EnMAP':
    #     r_rs, wlstr, wavelengths, datasetName = read_EnMAP_extracts(seaName=seaName,
    #                                              OWTList=OWTsingleList,
    #                                              outpath=outpath,
    #                                              datasetDate=datasetDate,
    #                                              datasetID=datasetID)
    # if sensor == 'PACE':
    #     r_rs, wlstr, wavelengths, datasetName = read_PACE_extracts(seaName=seaName,
    #                                              regionName = regionName,
    #                                              OWTList=OWTsingleList,
    #                                              versionAC = versionAC,
    #                                              outpath=path,
    #                                              datasetDate=datasetDate)
    #     if r_rs is None:
    #         return None



    if seaName == 'North Sea' or seaName == 'Baltic Sea' or seaName== 'Lakes':
        print('processing Dict North Sea')
        paramDict = processingDict_NorthSea[OWTsingleList[0]]

    if regionName == 'Kellersee':
        paramDict = processingDict_Kellersee[OWTsingleList[0]]
    elif regionName == 'Grosser-Binnensee' or regionName == 'Sibbersdorfer-See' or regionName == 'Stendorfer-See':
        paramDict = processingDict_GrosserBinnensee[OWTsingleList[0]]
    elif regionName == 'Mueggelsee': #HEATWISE
        print('use Mueggelsee Dict')
        paramDict = processingDict_Mueggelsee[OWTsingleList[0]]
    elif regionName == 'Helsinki_SS' or regionName == 'Helsinki_FM': #HEATWISE
        paramDict = processingDict_Helsinki[OWTsingleList[0]]
    elif regionName == 'oder_hohenwutzen' or regionName == 'oder_frankfurt': #AQUATIME
        paramDict = processingDict_Oder[OWTsingleList[0]]
    elif regionName == 'elbe_seemannshöft' or regionName == 'elbe_bunthaus': #AQUATIME
        paramDict = processingDict_Elbe[OWTsingleList[0]]
    elif 'dalaro' in regionName or regionName == 'pyhajarvi':
        paramDict = processingDict_Helsinki[OWTsingleList[0]]
    elif regionName == 'vortsjarv':
        paramDict = processingDict_Vortsjarv[OWTsingleList[0]]
    elif regionName == 'BelgianCoastRT1' or regionName == 'BelgianCoastCPOWER':
        paramDict = processingDict_BelgianCoast[OWTsingleList[0]]
    elif regionName == 'GLORIA' :
        print('processing Dict GLORIA')
        paramDict = processingDict_GLORIA[OWTsingleList[0]]
    elif regionName == 'Helsinki_EnMAP':
        print('processing Dict Helsinki_EnMAP')
        # paramDict = processingDict_Helsinki_EnMAPsimplified[OWTsingleList[0]] # for Land-AC!
        paramDict = processingDict_Helsinki[OWTsingleList[0]]
    elif regionName == 'California':
        paramDict = processingDict_California_EnMAPsimplified[OWTsingleList[0]]
    # if seaName == 'Baltic Sea':
    #     paramDict = processingDict_BalticSea[OWTsingleList[0]]


    # ## Modify start values  in GLORIADict ##
    # Nphyto = 0
    # for p in paramDict.keys():
    #     if p in ['C_0', 'C_1','C_2','C_3','C_4','C_5','C_6','C_7']:
    #         Nphyto +=1
    #
    # for p in paramDict.keys():
    #     if p in ['C_0', 'C_1','C_2','C_3','C_4','C_5','C_6','C_7']:
    #         paramDict[p]['value'] = paramDict[p]['value']/Nphyto

    


    ## weights
    weights = np.ones(len(wavelengths))


    ## modify weights:
    # def gaus2(x, a=0.5, sigma=1, posMax=0):
    #     return a * np.exp(-(x - posMax) ** 2 / (2 * sigma ** 2))
    #
    # modifyDict = {'+': [[550, 20], [640, 20]],
    #               '-': [[700, 20]]}
    #
    # weights_mod = np.zeros(len(wavelengths))
    # for key in modifyDict.keys():
    #     if key == '+':
    #         for posMax, sigma in modifyDict[key]:
    #             weights_mod += gaus2(wavelengths, sigma=np.sqrt(sigma), posMax=posMax)
    #     elif key == '-':
    #         for posMax, sigma in modifyDict[key]:
    #             weights_mod -= gaus2(wavelengths, sigma=np.sqrt(sigma), posMax=posMax)
    # weights += weights_mod

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

        if params['estimate_start_values'].value:
            pred_chl_tot, pred_acdom, pred_TSS = bandratioIOPprediction.predict_start_values_allSensors(chunk,
                                                                                                        wavelengths,
                                                                                                        outlierRemoval=True)

            phytoList = [a for a in params.keys() if a.startswith('C') and params[a].vary and not 'C_Y' and not 'C_ism']
            Nphyto = len(phytoList)

        # make a copy of actual params object to enable in-parallel mutation
        chunk_params = params.copy()

        for i in np.arange(chunk.shape[0]):
            # chunk_params.add('theta_sun', value=np.radians(chunk[0,i,0]), vary=False)
            if checkZeroSpectrum:
                weights = np.ones(len(wavelengths))
                weights[chunk[i,:]==0] = 0

            try:
                if params['estimate_start_values'].value:
                    for key in phytoList:
                        chunk_params.add(key, value=pred_chl_tot[i]/Nphyto, min=0., max=pred_chl_tot[i], vary=True)
                    chunk_params.add('C_Y', value=pred_acdom[i], min=0., max=pred_acdom[i]*2, vary=True)
                    chunk_params.add('C_ism', value=pred_TSS[i], min=0., max=pred_TSS[i] * 1.5, vary=True)

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
            except:
                continue
            results[i] = inv

        return results

    data = r_rs.values

    if data.shape[0] < 100:
        num_chunks = 1
    elif data.shape[0] < 300:
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
    params = set_parameters_byDict(paramDict, params) #variable parameters
    params = set_fixedParameters_byDict(updateVarDict, params)

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

    if not outpath is None:
        outDF.to_csv(outpath + "inverted_IOP_bio_optics_HEREONfull_" + datasetName + "_V"+str(int(Ngroups)) +
                 "AH_"+groupStr+"_restrict.txt",
                 sep='\t', header=True, index=False)
        R_rs_sim.to_csv(outpath + "inverted_Rrs_bio_optics_HEREONfull_" + datasetName + "_V"+str(int(Ngroups)) +
                 "AH_"+groupStr+"_restrict.txt",
                    sep='\t', header=True, index=False)

    return outDF, R_rs_sim

## 20250307
# first test with new specific absorption and scattering values

## Update 20241219:
# - C_3 Cyano_blue for Synechococcus in North Sea
# - add fluorescence 'L_fl_phycoerythrin' for Cryptophytes C_2

### Standard ##
processingDict_NorthSea = {
    '1': {'C_7': setDictValues(0,0,1)},
    '2': {'C_0': setDictValues(0,0,1),
          'C_2': setDictValues(0,0,1),
          'C_3': setDictValues(0,0,1), # Synechococcus
          'C_6': setDictValues(0,0,1),
          'C_7': setDictValues(0,0,1),
          'C_Y': setDictValues(0,0,0.1),
          'C_ism': setDictValues(0,0,1)}, # no fluorescence!
    '3a': {'C_0': setDictValues(0.1,0,10),
          'C_2': setDictValues(0,0,10),
           'C_3': setDictValues(0,0,10), # Synechococcus
          'C_4': setDictValues(0,0,10),
           'C_6': setDictValues(0,0,10),
          'C_Y': setDictValues(0,0,1),
          'C_ism': setDictValues(0,0,10),
           # 'L_fl_lambda0': setDictValues(0,0,0.2),
           # 'L_fl_phycoerythrin': setDictValues(0,0,0.2),
           #  'L_fl_phycocyanin': setDictValues(0,0,0.2)
           },
    '3a_g': {'C_0': setDictValues(0.1,0,10),
          'C_2': setDictValues(0,0,10),
           'C_3': setDictValues(0,0,10), # Synechococcus
          'C_4': setDictValues(0,0,10),
           'C_6': setDictValues(0,0,10),
          'C_Y': setDictValues(0,0,1),
          'C_ism': setDictValues(0,0,10),
           # 'L_fl_lambda0': setDictValues(0,0,0.2),
           # 'L_fl_phycoerythrin': setDictValues(0,0,0.2),
           #  'L_fl_phycocyanin': setDictValues(0,0,0.2)
           },
    '3a_y': {'C_0': setDictValues(0.1,0,10),
          'C_2': setDictValues(0,0,10),
           'C_3': setDictValues(0,0,10), # Synechococcus
          'C_4': setDictValues(0,0,10),
           'C_6': setDictValues(0,0,10),
          'C_Y': setDictValues(0,0,3),
          'C_ism': setDictValues(0,0,10),
           # 'L_fl_lambda0': setDictValues(0,0,0.2),
           # 'L_fl_phycoerythrin': setDictValues(0,0,0.2),
           #  'L_fl_phycocyanin': setDictValues(0,0,0.2)
           },
    '3b': {'C_0': setDictValues(0.1,0,10),
          'C_2': setDictValues(0,0,10),
           'C_5': setDictValues(0,0,10), # coccolith.
          'C_6': setDictValues(0,0,10),
          'C_Y': setDictValues(0,0,3),
          'C_ism': setDictValues(0,0,10),
           # 'L_fl_lambda0': setDictValues(0,0,0.2),
           # 'L_fl_phycoerythrin': setDictValues(0,0,0.2)
           },
    '4a': {'C_0': setDictValues(0.1,0,30),
        # 'C_1': setDictValues(0,0,30),
          'C_2': setDictValues(0,0,30),
           'C_3': setDictValues(0,0,30), # Synechococcus
          'C_4': setDictValues(0,0,30),
          'C_6': setDictValues(0,0,30),
          'C_Y': setDictValues(0,0,3),
          'C_ism': setDictValues(0,0,20),
           # 'L_fl_lambda0': setDictValues(0,0,0.2),
           # 'L_fl_phycoerythrin': setDictValues(0,0,0.2),
           # 'L_fl_phycocyanin': setDictValues(0,0,0.2)
           },
    '4a_g': {'C_0': setDictValues(0.1,0,30),
            # 'C_1': setDictValues(0,0,30),
          'C_2': setDictValues(0,0,30),
           'C_3': setDictValues(0,0,30), # Synechococcus
          'C_4': setDictValues(0,0,30),
          'C_6': setDictValues(0,0,30),
          'C_Y': setDictValues(0,0,3),
          'C_ism': setDictValues(0,0,20),
           # 'L_fl_lambda0': (0,0,0.2),
           # 'L_fl_phycoerythrin': setDictValues(0,0,0.2),
           # 'L_fl_phycocyanin': setDictValues(0,0,0.2)
             },
    '4a_y': {'C_0': setDictValues(0.1,0,30),
        # 'C_1': setDictValues(0,0,30),
          'C_2': setDictValues(0,0,30),
           'C_3': setDictValues(0,0,30), # Synechococcus
          'C_4': setDictValues(0,0,30),
          'C_6': setDictValues(0,0,30),
          'C_Y': setDictValues(0,0,3),
          'C_ism': setDictValues(0,0,20),
           # 'L_fl_lambda0': setDictValues(0,0,0.2),
           # 'L_fl_phycoerythrin': setDictValues(0,0,0.2),
           # 'L_fl_phycocyanin': setDictValues(0,0,0.2)
             },
    '4b': {'C_0': setDictValues(0.1,0,300),
        # 'C_1': setDictValues(0,0,100),
          'C_2': setDictValues(0,0,300),
        'C_3': setDictValues(0,0,100), # Synechococcus
          'C_4': setDictValues(0,0,100),
            'C_5': setDictValues(0,0,10), # coccolith.
          'C_6': setDictValues(0,0,300),
          'C_Y': setDictValues(0,0,3),
          'C_ism': setDictValues(0,0,100),
           # 'L_fl_lambda0': setDictValues(0,0,0.2),
           # 'L_fl_phycoerythrin': setDictValues(0,0,0.2),
           # 'L_fl_phycocyanin': setDictValues(0,0,0.2)
           },
    '5a': {'C_0': setDictValues(0.,0,300),
           # 'C_1': setDictValues(0,0,300),
          'C_2': setDictValues(0,0,300),
           'C_3': setDictValues(0,0,300), # Synechococcus
          'C_4': setDictValues(0,0,300),
           'C_6': setDictValues(0,0,300),
          'C_Y': setDictValues(0,0,3),
          'C_ism': setDictValues(0,0,100),
           # 'L_fl_lambda0': setDictValues(0,0,0.2),
           #  'L_fl_phycoerythrin': setDictValues(0,0,0.2),
           #  'L_fl_phycocyanin': setDictValues(0,0,0.2)
           # 'offset': setDictValues(0, -0.1, 0.1)
           },
    '5b': {'C_0': setDictValues(0.1,0,1000),
            # 'C_1': setDictValues(0.1,0,100),
          'C_2': setDictValues(0,0,1000),
            'C_3': setDictValues(0,0,300), # Synechococcus
          'C_4': setDictValues(0,0,300),
          'C_6': setDictValues(0,0,1000),
          'C_Y': setDictValues(0,0,3),
          'C_ism': setDictValues(0,0,100),
           # 'L_fl_lambda0': setDictValues(0,0,0.2),
           #  'L_fl_phycoerythrin': setDictValues(0,0,0.2),
           #  'L_fl_phycocyanin': setDictValues(0,0,0.2)
           # 'offset': setDictValues(0, -0.1, 0.1)
           },
    '6': {'C_0': setDictValues(0.1,0,500),
          # 'C_1': setDictValues(0.,0,500),
          'C_2': setDictValues(0,0,500),
            'C_3': setDictValues(0,0,300), # Synechococcus
          'C_4': setDictValues(0,0,300),
          'C_6': setDictValues(0,0,500),
          'C_Y': setDictValues(0,0,3),
          'C_ism': setDictValues(0,0,1000),
           # 'L_fl_lambda0': setDictValues(0,0,0.2),
          # 'L_fl_phycoerythrin': setDictValues(0,0,0.2),
          # 'L_fl_phycocyanin': setDictValues(0,0,0.2)
          },
    '7': {'C_0': setDictValues(0.1,0,200),
          'C_1': setDictValues(0.,0,200),
          'C_2': setDictValues(0,0,200),
            'C_3': setDictValues(0,0,200), # Synechococcus
          'C_4': setDictValues(0,0,200),
          'C_6': setDictValues(0,0,200),
          'C_Y': setDictValues(0,0,3),
          'C_ism': setDictValues(0,0,100),
           # 'L_fl_lambda0': setDictValues(0,0,0.2),
          # 'L_fl_phycoerythrin': setDictValues(0,0,0.2),
          # 'L_fl_phycocyanin': setDictValues(0,0,0.2)
          }
}

processingDict_GLORIA = {
    '4a_y' : {
    'C_0': setDictValues( 12.0 , 0., 925.39 ),
    'C_1': setDictValues( 12.0 , 0., 925.39 ),
    'C_2': setDictValues( 12.0 , 0., 925.39 ),
    'C_3': setDictValues( 12.0 , 0., 925.39 ),
    'C_4': setDictValues( 12.0 , 0., 925.39 ),
    # 'C_5': setDictValues( 12.0 , 0., 925.39 ), #coccolithophores
    'C_6': setDictValues( 12.0 , 0., 925.39 ),
    # 'C_7': setDictValues( 12.0 , 0., 925.39 ), # case-1
    'C_ism': setDictValues( 3.6 , 0., 143.83 ),
    'C_Y': setDictValues( 2.58 , 0., 19.74 )
    },
    '4b' : {
    'C_0': setDictValues( 32.25 , 0., 481.85 ),
    'C_1': setDictValues( 32.25 , 0., 481.85 ),
    'C_2': setDictValues( 32.25 , 0., 481.85 ),
    'C_3': setDictValues( 32.25 , 0., 481.85 ),
    'C_4': setDictValues( 32.25 , 0., 481.85 ),
    'C_5': setDictValues( 32.25 , 0., 481.85 ),
    'C_6': setDictValues( 32.25 , 0., 481.85 ),
    # 'C_7': setDictValues( 32.25 , 0., 481.85 ),
    'C_ism': setDictValues( 19.06 , 0., 1195.75 ),
    'C_Y': setDictValues( 1.36 , 0., 11.21 )
    },
    '5a' : {
    'C_0': setDictValues( 24.26 , 0., 351.72 ),
    'C_1': setDictValues( 24.26 , 0., 351.72 ),
    'C_2': setDictValues( 24.26 , 0., 351.72 ),
    'C_3': setDictValues( 24.26 , 0., 351.72 ),
    'C_4': setDictValues( 24.26 , 0., 351.72 ),
    # 'C_5': setDictValues( 24.26 , 0., 351.72 ),
    'C_6': setDictValues( 24.26 , 0., 351.72 ),
    # 'C_7': setDictValues( 24.26 , 0., 351.72 ),
    'C_ism': setDictValues( 10.75 , 0., 159.06 ),
    'C_Y': setDictValues( 1.31 , 0., 11.27 )
    },
    '5b' : {
    'C_0': setDictValues( 16.03 , 0., 323.53 ),
    'C_1': setDictValues( 16.03 , 0., 323.53 ),
    'C_2': setDictValues( 16.03 , 0., 323.53 ),
    'C_3': setDictValues( 16.03 , 0., 323.53 ),
    'C_4': setDictValues( 16.03 , 0., 323.53 ),
    # 'C_5': setDictValues( 16.03 , 0., 323.53 ),
    'C_6': setDictValues( 16.03 , 0., 323.53 ),
    # 'C_7': setDictValues( 16.03 , 0., 323.53 ),
    'C_ism': setDictValues( 10.84 , 0., 113.56 ),
    'C_Y': setDictValues( 0.62 , 0., 6.29 )
    },
    '6' : {
    'C_0': setDictValues( 41.73 , 0., 1334.63 ),
    'C_1': setDictValues( 41.73 , 0., 1334.63 ),
    'C_2': setDictValues( 41.73 , 0., 1334.63 ),
    'C_3': setDictValues( 41.73 , 0., 1334.63 ),
    'C_4': setDictValues( 41.73 , 0., 1334.63 ),
    # 'C_5': setDictValues( 41.73 , 0., 1334.63 ),
    'C_6': setDictValues( 41.73 , 0., 1334.63 ),
    # 'C_7': setDictValues( 41.73 , 0., 1334.63 ),
    'C_ism': setDictValues( 24.9 , 0., 410.04 ),
    'C_Y': setDictValues( 0.87 , 0., 6.26 )
    },
    '7' : {
    'C_0': setDictValues( 15.37 , 0., 121.99 ),
    'C_1': setDictValues( 15.37 , 0., 121.99 ),
    'C_2': setDictValues( 15.37 , 0., 121.99 ),
    'C_3': setDictValues( 15.37 , 0., 121.99 ),
    'C_4': setDictValues( 15.37 , 0., 121.99 ),
    # 'C_5': setDictValues( 15.37 , 0., 121.99 ),
    'C_6': setDictValues( 15.37 , 0., 121.99 ),
    # 'C_7': setDictValues( 15.37 , 0., 121.99 ),
    'C_ism': setDictValues( 3.55 , 0., 18.26 ),
    'C_Y': setDictValues( 1.17 , 0., 6.86 )
    },
    'nan' : {
    'C_0': setDictValues( 2.02 , 0., 42.09 ),
    'C_1': setDictValues( 2.02 , 0., 42.09 ),
    'C_2': setDictValues( 2.02 , 0., 42.09 ),
    'C_3': setDictValues( 2.02 , 0., 42.09 ),
    'C_4': setDictValues( 2.02 , 0., 42.09 ),
    # 'C_5': setDictValues( 2.02 , 0., 42.09 ),
    'C_6': setDictValues( 2.02 , 0., 42.09 ),
    # 'C_7': setDictValues( 2.02 , 0., 42.09 ),
    'C_ism': setDictValues( 3.02 , 0., 90.08 ),
    'C_Y': setDictValues( 0.15 , 0., 8.08 ),
    'offset': setDictValues(0., -0.02, 0.02)
    }
}


processingDict_BelgianCoast = { # use with NSsummer set!

    '3a_g': {'C_0': setDictValues(0.1,0,10),
          # 'C_2': setDictValues(0,0,10),
          #  'C_3': setDictValues(0,0,10), # Synechococcus
          # 'C_4': setDictValues(0,0,10),
            'C_5': setDictValues(0,0,10), # Phaeocystis
           'C_6': setDictValues(0,0,10),
          'C_Y': setDictValues(0,0,1),
          'C_ism': setDictValues(0,0,10),
           'L_fl_lambda0': setDictValues(0,0,0.2),
           # 'L_fl_phycoerythrin': setDictValues(0,0,0.2),
           #  'L_fl_phycocyanin': setDictValues(0,0,0.2)
           },
    '3a_y': {'C_0': setDictValues(0.1,0,10),
          # 'C_2': setDictValues(0,0,10),
          #  'C_3': setDictValues(0,0,10), # Synechococcus
          # 'C_4': setDictValues(0,0,10),
            'C_5': setDictValues(0,0,10), # Phaeocystis
           'C_6': setDictValues(0,0,10),
          'C_Y': setDictValues(0,0,3),
          'C_ism': setDictValues(0,0,10),
           'L_fl_lambda0': setDictValues(0,0,0.2),
           # 'L_fl_phycoerythrin': setDictValues(0,0,0.2),
           #  'L_fl_phycocyanin': setDictValues(0,0,0.2)
           },
    '3b': {'C_0': setDictValues(0.1,0,10),
          # 'C_2': setDictValues(0,0,10),
           'C_5': setDictValues(0,0,10), # pHaeocystis
          'C_6': setDictValues(0,0,10),
          'C_Y': setDictValues(0,0,10),
          'C_ism': setDictValues(0,0,10),
           'L_fl_lambda0': setDictValues(0,0,0.2),
           # 'L_fl_phycoerythrin': setDictValues(0,0,0.2)
           },
    '4a': {'C_0': setDictValues(0.1,0,30),
        # 'C_1': setDictValues(0,0,30),
        #   'C_2': setDictValues(0,0,30),
        #    'C_3': setDictValues(0,0,30), # Synechococcus
          'C_5': setDictValues(0,0,30), # pHaeocystis
          'C_6': setDictValues(0,0,30),
          'C_Y': setDictValues(0,0,5),
          'C_ism': setDictValues(0,0,20),
           'L_fl_lambda0': setDictValues(0,0,0.2),
           # 'L_fl_phycoerythrin': setDictValues(0,0,0.2),
           # 'L_fl_phycocyanin': setDictValues(0,0,0.2)
           },
    '4a_g': {'C_0': setDictValues(0.1,0,30),
          #   'C_1': setDictValues(0,0,30),
          # 'C_2': setDictValues(0,0,30),
          #  'C_3': setDictValues(0,0,30), # Synechococcus
          'C_5': setDictValues(0,0,30), # pHaeocystis
          'C_6': setDictValues(0,0,30),
          'C_Y': setDictValues(0,0,5),
          'C_ism': setDictValues(0,0,20),
           'L_fl_lambda0': setDictValues(0,0,0.2),
           # 'L_fl_phycoerythrin': setDictValues(0,0,0.2),
           # 'L_fl_phycocyanin': setDictValues(0,0,0.2)
             },
    '4a_y': {'C_0': setDictValues(0.1,0,30),
        # 'C_1': setDictValues(0,0,30),
        #   'C_2': setDictValues(0,0,30),
        #    'C_3': setDictValues(0,0,30), # Synechococcus
          'C_5': setDictValues(0,0,30),# pHaeocystis
          'C_6': setDictValues(0,0,30),
          'C_Y': setDictValues(0,0,10),
          'C_ism': setDictValues(0,0,20),
           'L_fl_lambda0': setDictValues(0,0,0.2),
           # 'L_fl_phycoerythrin': setDictValues(0,0,0.2),
           # 'L_fl_phycocyanin': setDictValues(0,0,0.2)
             },
    '4b': {'C_0': setDictValues(0.1,0,300),
        # 'C_1': setDictValues(0,0,100),
        #   'C_2': setDictValues(0,0,300),
        # 'C_3': setDictValues(0,0,100), # Synechococcus
        #   'C_4': setDictValues(0,0,100),
            'C_5': setDictValues(0,0,300), #  pHaeocystis.
          'C_6': setDictValues(0,0,300),
          'C_Y': setDictValues(0,0,10),
          'C_ism': setDictValues(0,0,100),
           'L_fl_lambda0': setDictValues(0,0,0.2),
           # 'L_fl_phycoerythrin': setDictValues(0,0,0.2),
           # 'L_fl_phycocyanin': setDictValues(0,0,0.2)
           },
    '5a': {'C_0': setDictValues(0.,0,300),
          #  'C_1': setDictValues(0,0,300),
          # 'C_2': setDictValues(0,0,300),
          #  'C_3': setDictValues(0,0,300), # Synechococcus
          'C_5': setDictValues(0,0,300), # pHaeocystis
           'C_6': setDictValues(0,0,300),
          'C_Y': setDictValues(0,0,100),
          'C_ism': setDictValues(0,0,100),
           'L_fl_lambda0': setDictValues(0,0,0.2),
            # 'L_fl_phycoerythrin': setDictValues(0,0,0.2),
            # 'L_fl_phycocyanin': setDictValues(0,0,0.2)
           # 'offset': setDictValues(0, -0.1, 0.1)
           },
    '5b': {'C_0': setDictValues(0.1,0,1000),
          #   'C_1': setDictValues(0.1,0,100),
          # 'C_2': setDictValues(0,0,1000),
          #   'C_3': setDictValues(0,0,300), # Synechococcus
          'C_5' : setDictValues(0,0,300), # pHaeocystis
          'C_6': setDictValues(0,0,1000),
          'C_Y': setDictValues(0,0,100),
          'C_ism': setDictValues(0,0,100),
           'L_fl_lambda0': setDictValues(0,0,0.2),
            # 'L_fl_phycoerythrin': setDictValues(0,0,0.2),
            # 'L_fl_phycocyanin': setDictValues(0,0,0.2)
           # 'offset': setDictValues(0, -0.1, 0.1)
           },
    '6': {'C_0': setDictValues(0.1,0,500),
          # 'C_1': setDictValues(0.,0,500),
          # 'C_2': setDictValues(0,0,500),
          #   'C_3': setDictValues(0,0,300), # Synechococcus
          'C_5': setDictValues(0,0,300), # pHaeocystis
          'C_6': setDictValues(0,0,500),
          'C_Y': setDictValues(0,0,100),
          'C_ism': setDictValues(0,0,1000),
           'L_fl_lambda0': setDictValues(0,0,0.2),
          # 'L_fl_phycoerythrin': setDictValues(0,0,0.2),
          # 'L_fl_phycocyanin': setDictValues(0,0,0.2)
          },
    '7': {'C_0': setDictValues(0.1,0,200),
          # 'C_1': setDictValues(0.,0,200),
          # 'C_2': setDictValues(0,0,200),
          #   'C_3': setDictValues(0,0,200), # Synechococcus
          'C_5': setDictValues(0,0,200), # pHaeocystis
          'C_6': setDictValues(0,0,200),
          'C_Y': setDictValues(0,0,30),
          'C_ism': setDictValues(0,0,100),
           'L_fl_lambda0': setDictValues(0,0,0.2),
          # 'L_fl_phycoerythrin': setDictValues(0,0,0.2),
          # 'L_fl_phycocyanin': setDictValues(0,0,0.2)
          }
}

### Standard ##
# Kellersee: only Diatomeen, Green in all classes
processingDict_Kellersee = {
    # '1': {'C_7': setDictValues(0,0,1)},
    # '2': {'C_0': setDictValues(0,0,1),
    #       'C_2': setDictValues(0,0,1),
    #       'C_3': setDictValues(0,0,1), # Synechococcus
    #       'C_6': setDictValues(0,0,1),
    #       'C_7': setDictValues(0,0,1),
    #       'C_Y': setDictValues(0,0,0.1),
    #       'C_ism': setDictValues(0,0,1)}, # no fluorescence!
    '3a_g': {'C_0': setDictValues(0,0,10),
            'C_1': setDictValues(0,0,10),
          # 'C_2': setDictValues(0,0,10),
          #  'C_3': setDictValues(0,0,10), # Synechococcus
          # 'C_4': setDictValues(0,0,10),
          #  'C_6': setDictValues(0,0,10),
          'C_Y': setDictValues(0,0,1),
          'C_ism': setDictValues(0,0,10),
           'L_fl_lambda0': setDictValues(0,0,0.2),
           # 'L_fl_phycoerythrin': setDictValues(0,0,0.2),
           #  'L_fl_phycocyanin': setDictValues(0,0,0.2)
        'offset': setDictValues(0, -0.1, 0.1)
           },
    '3a_y': {'C_0': setDictValues(0,0,10),
            'C_1': setDictValues(0,0,10),
          # 'C_2': setDictValues(0,0,10),
          #  'C_3': setDictValues(0,0,10), # Synechococcus
          # 'C_4': setDictValues(0,0,10),
          #  'C_6': setDictValues(0,0,10),
          'C_Y': setDictValues(0,0,3),
          'C_ism': setDictValues(0,0,10),
           'L_fl_lambda0': setDictValues(0,0,0.2),
           # 'L_fl_phycoerythrin': setDictValues(0,0,0.2),
           #  'L_fl_phycocyanin': setDictValues(0,0,0.2)
        'offset': setDictValues(0, -0.1, 0.1)
           },
    '3b': {'C_0': setDictValues(0,0,10),
            'C_1': setDictValues(0,0,10),
          # 'C_2': setDictValues(0,0,10),
          #  'C_5': setDictValues(0,0,10), # coccolith.
          # 'C_6': setDictValues(0,0,10),
          'C_Y': setDictValues(0,0,10),
          'C_ism': setDictValues(0,0,10),
           'L_fl_lambda0': setDictValues(0,0,0.2),
           # 'L_fl_phycoerythrin': setDictValues(0,0,0.2)
        'offset': setDictValues(0, -0.1, 0.1)
           },
    '4a_g': {'C_0': setDictValues(0,0,30),
            'C_1': setDictValues(0,0,30),
          # 'C_2': setDictValues(0,0,30),
          #  'C_3': setDictValues(0,0,30), # Synechococcus
          # 'C_4': setDictValues(0,0,30),
          # 'C_6': setDictValues(0,0,30),
          'C_Y': setDictValues(0,0,5),
          'C_ism': setDictValues(0,0,20),
           'L_fl_lambda0': setDictValues(0,0,0.2),
           # 'L_fl_phycoerythrin': setDictValues(0,0,0.2),
           # 'L_fl_phycocyanin': setDictValues(0,0,0.2)
        'offset': setDictValues(0, -0.1, 0.1)
             },
    '4a_y': {'C_0': setDictValues(0.1,0,30),
        'C_1': setDictValues(0,0,30),
          # 'C_2': setDictValues(0,0,30),
          #  'C_3': setDictValues(0,0,30), # Synechococcus
          # 'C_4': setDictValues(0,0,30),
          # 'C_6': setDictValues(0,0,30),
          'C_Y': setDictValues(0,0,10),
          'C_ism': setDictValues(0,0,20),
           'L_fl_lambda0': setDictValues(0,0,0.2),
           # 'L_fl_phycoerythrin': setDictValues(0,0,0.2),
           # 'L_fl_phycocyanin': setDictValues(0,0,0.2)
        'offset': setDictValues(0, -0.1, 0.1)
             },
    '4b': {'C_0': setDictValues(0.1,0,300),
        'C_1': setDictValues(0,0,100),
        #   'C_2': setDictValues(0,0,300),
        # 'C_3': setDictValues(0,0,100), # Synechococcus
        #   'C_4': setDictValues(0,0,100),
        #     'C_5': setDictValues(0,0,10), # coccolith.
        #   'C_6': setDictValues(0,0,300),
          'C_Y': setDictValues(0,0,10),
          'C_ism': setDictValues(0,0,100),
           'L_fl_lambda0': setDictValues(0,0,0.2),
           # 'L_fl_phycoerythrin': setDictValues(0,0,0.2),
           # 'L_fl_phycocyanin': setDictValues(0,0,0.2)
        'offset': setDictValues(0, -0.1, 0.1)
           },
    '5a': {'C_0': setDictValues(0.,0,300),
           'C_1': setDictValues(0,0,300),
          # 'C_2': setDictValues(0,0,300),
          #  'C_3': setDictValues(0,0,300), # Synechococcus
          # 'C_4': setDictValues(0,0,300),
          #  'C_6': setDictValues(0,0,300),
          'C_Y': setDictValues(0,0,100),
          'C_ism': setDictValues(0,0,100),
           'L_fl_lambda0': setDictValues(0,0,0.2),
            # 'L_fl_phycoerythrin': setDictValues(0,0,0.2),
            # 'L_fl_phycocyanin': setDictValues(0,0,0.2)
           'offset': setDictValues(0, -0.1, 0.1)
           },
    '5b': {'C_0': setDictValues(0.1,0,1000),
            'C_1': setDictValues(0.1,0,100),
          # 'C_2': setDictValues(0,0,1000),
          #   'C_3': setDictValues(0,0,300), # Synechococcus
          # 'C_4': setDictValues(0,0,300),
          # 'C_6': setDictValues(0,0,1000),
          'C_Y': setDictValues(0,0,100),
          'C_ism': setDictValues(0,0,100),
           'L_fl_lambda0': setDictValues(0,0,0.2),
            # 'L_fl_phycoerythrin': setDictValues(0,0,0.2),
            # 'L_fl_phycocyanin': setDictValues(0,0,0.2)
           'offset': setDictValues(0, -0.1, 0.1)
           },
    '6': {'C_0': setDictValues(0.1,0,500),
          'C_1': setDictValues(0.,0,500),
          # 'C_2': setDictValues(0,0,500),
          #   'C_3': setDictValues(0,0,300), # Synechococcus
          # 'C_4': setDictValues(0,0,300),
          # 'C_6': setDictValues(0,0,500),
          'C_Y': setDictValues(0,0,100),
          'C_ism': setDictValues(0,0,1000),
           'L_fl_lambda0': setDictValues(0,0,0.2),
          # 'L_fl_phycoerythrin': setDictValues(0,0,0.2),
          # 'L_fl_phycocyanin': setDictValues(0,0,0.2)
        'offset': setDictValues(0, -0.1, 0.1)
          },
    '7': {'C_0': setDictValues(0.1,0,200),
          'C_1': setDictValues(0.,0,200),
          # 'C_2': setDictValues(0,0,200),
          #   'C_3': setDictValues(0,0,200), # Synechococcus
          # 'C_4': setDictValues(0,0,200),
          # 'C_6': setDictValues(0,0,200),
          'C_Y': setDictValues(0,0,30),
          'C_ism': setDictValues(0,0,100),
           'L_fl_lambda0': setDictValues(0,0,0.2),
          # 'L_fl_phycoerythrin': setDictValues(0,0,0.2),
          # 'L_fl_phycocyanin': setDictValues(0,0,0.2)
        'offset': setDictValues(0, -0.1, 0.1)
          }
}

# Kellersee: only Diatomeen, Green in all classes
processingDict_GrosserBinnensee = {
    # '1': {'C_7': setDictValues(0,0,1)},
    # '2': {'C_0': setDictValues(0,0,1),
    #       'C_2': setDictValues(0,0,1),
    #       'C_3': setDictValues(0,0,1), # Synechococcus
    #       'C_6': setDictValues(0,0,1),
    #       'C_7': setDictValues(0,0,1),
    #       'C_Y': setDictValues(0,0,0.1),
    #       'C_ism': setDictValues(0,0,1)}, # no fluorescence!
    '3a_g': {'C_0': setDictValues(0,0,10),
            'C_1': setDictValues(0,0,10),
          # 'C_2': setDictValues(0,0,10),
          #  'C_3': setDictValues(0,0,10), # Synechococcus
          # 'C_4': setDictValues(0,0,10),
          #  'C_6': setDictValues(0,0,10),
          'C_Y': setDictValues(0,0,1),
          'C_ism': setDictValues(0,0,10),
           'L_fl_lambda0': setDictValues(0,0,0.2),
           # 'L_fl_phycoerythrin': setDictValues(0,0,0.2),
           #  'L_fl_phycocyanin': setDictValues(0,0,0.2)
        'offset': setDictValues(0, -0.1, 0.1)
           },
    '3a_y': {'C_0': setDictValues(0,0,10),
            'C_1': setDictValues(0,0,10),
          # 'C_2': setDictValues(0,0,10),
          #  'C_3': setDictValues(0,0,10), # Synechococcus
          # 'C_4': setDictValues(0,0,10),
          #  'C_6': setDictValues(0,0,10),
          'C_Y': setDictValues(0,0,3),
          'C_ism': setDictValues(0,0,10),
           'L_fl_lambda0': setDictValues(0,0,0.2),
           # 'L_fl_phycoerythrin': setDictValues(0,0,0.2),
           #  'L_fl_phycocyanin': setDictValues(0,0,0.2)
        'offset': setDictValues(0, -0.1, 0.1)
           },
    '3b': {'C_0': setDictValues(0,0,10),
            'C_1': setDictValues(0,0,10),
          # 'C_2': setDictValues(0,0,10),
          #  'C_5': setDictValues(0,0,10), # coccolith.
          # 'C_6': setDictValues(0,0,10),
          'C_Y': setDictValues(0,0,10),
          'C_ism': setDictValues(0,0,10),
           'L_fl_lambda0': setDictValues(0,0,0.2),
           # 'L_fl_phycoerythrin': setDictValues(0,0,0.2)
        'offset': setDictValues(0, -0.1, 0.1)
           },
    '4a_g': {'C_0': setDictValues(0,0,30),
            'C_1': setDictValues(0,0,30),
          # 'C_2': setDictValues(0,0,30),
           'C_3': setDictValues(0,0,30), # Synechococcus
          # 'C_4': setDictValues(0,0,30),
          # 'C_6': setDictValues(0,0,30),
          'C_Y': setDictValues(0,0,5),
          'C_ism': setDictValues(0,0,20),
           'L_fl_lambda0': setDictValues(0,0,0.2),
           # 'L_fl_phycoerythrin': setDictValues(0,0,0.2),
           'L_fl_phycocyanin': setDictValues(0,0,0.2),
        'offset': setDictValues(0, -0.1, 0.1)
             },
    '4a_y': {'C_0': setDictValues(0.1,0,30),
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
        'offset': setDictValues(0, -0.1, 0.1)
             },
    '4b': {'C_0': setDictValues(0,0,200),
        'C_1': setDictValues(0,0,200),
        #   'C_2': setDictValues(0,0,300),
        'C_3': setDictValues(0,0, 200), # Synechococcus
        #   'C_4': setDictValues(0,0,100),
        #     'C_5': setDictValues(0,0,10), # coccolith.
        #   'C_6': setDictValues(0,0,300),
          'C_Y': setDictValues(0,0,1),
          'C_ism': setDictValues(0,0,2),
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
          'C_ism': setDictValues(0,0,2),
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
          'C_ism': setDictValues(0,0, 2),
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
          'C_ism': setDictValues(0,0,2),
           'L_fl_lambda0': setDictValues(0,0,0.2),
          # 'L_fl_phycoerythrin': setDictValues(0,0,0.2),
          'L_fl_phycocyanin': setDictValues(0,0,0.2),
        # 'offset': setDictValues(0, -0.1, 0.1)
          },
    '7': {'C_0': setDictValues(0.,0,200),
          'C_1': setDictValues(0.,0,200),
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

processingDict_Mueggelsee = {
    # '1': {'C_7': setDictValues(0,0,1)},
    # '2': {'C_0': setDictValues(0,0,1),
    #       'C_2': setDictValues(0,0,1),
    #       'C_3': setDictValues(0,0,1), # Synechococcus
    #       'C_6': setDictValues(0,0,1),
    #       'C_7': setDictValues(0,0,1),
    #       'C_Y': setDictValues(0,0,0.1),
    #       'C_ism': setDictValues(0,0,1)}, # no fluorescence!
    # '3a_g': {'C_0': setDictValues(0,0,10),
    #       #   'C_1': setDictValues(0,0,10),
    #       'C_2': setDictValues(0,0,10),
    #        'C_3': setDictValues(0,0,10), # Synechococcus
    #       'C_5': setDictValues(0,0,10),
    #       #  'C_6': setDictValues(0,0,10),
    #       'C_Y': setDictValues(0,0,1),
    #       'C_ism': setDictValues(0,0,10),
    #        'L_fl_lambda0': setDictValues(0,0,0.2),
    #        # 'L_fl_phycoerythrin': setDictValues(0,0,0.2),
    #         'L_fl_phycocyanin': setDictValues(0,0,0.2)
    #     # 'offset': setDictValues(0, -0.1, 0.1)
    #        },
    # '3a_y': {'C_0': setDictValues(0,0,10),
    #       #   'C_1': setDictValues(0,0,10),
    #       'C_2': setDictValues(0,0,10),
    #        'C_3': setDictValues(0,0,10), # Synechococcus
    #       'C_5': setDictValues(0,0,10), # Goldalge
    #       #  'C_6': setDictValues(0,0,10),
    #       'C_Y': setDictValues(0,0,3),
    #       'C_ism': setDictValues(0,0,10),
    #        'L_fl_lambda0': setDictValues(0,0,0.2),
    #        # 'L_fl_phycoerythrin': setDictValues(0,0,0.2),
    #         'L_fl_phycocyanin': setDictValues(0,0,0.2)
    #     # 'offset': setDictValues(0, -0.1, 0.1)
    #        },
    # '3b': {'C_0': setDictValues(0,0,10),
    #       #   'C_1': setDictValues(0,0,10),
    #       'C_2': setDictValues(0,0,10),
    #         'C_3': setDictValues(0,0,10),
    #        'C_5': setDictValues(0,0,10), # Goldalge in HereonGold
    #       # 'C_6': setDictValues(0,0,10),
    #       'C_Y': setDictValues(0,0,10),
    #       'C_ism': setDictValues(0,0,10),
    #        'L_fl_lambda0': setDictValues(0,0,0.2),
    #         'L_fl_phycocyanin': setDictValues(0,0,0.2)
    #        # 'L_fl_phycoerythrin': setDictValues(0,0,0.2)
    #     # 'offset': setDictValues(0, -0.1, 0.1)
    #        },
    '4a_g': {'C_0': setDictValues(0,0,30),
            # 'C_1': setDictValues(0,0,30),
          # 'C_2': setDictValues(0,0,30),
          #  'C_3': setDictValues(0,0,30), # Synechococcus
          # 'C_5': setDictValues(0,0,30),
          'C_6': setDictValues(0,0,30),
          'C_Y': setDictValues(0,0,5),
          'C_ism': setDictValues(0,0,20),
           'L_fl_lambda0': setDictValues(0,0,0.2),
           # 'L_fl_phycoerythrin': setDictValues(0,0,0.2),
           'L_fl_phycocyanin': setDictValues(0,0,0.2),
        # 'offset': setDictValues(0, -0.1, 0.1)
             },
    '4a_y': {'C_0': setDictValues(0.1,0,30),
            # 'C_1': setDictValues(0,0,30),
            # 'C_2': setDictValues(0,0,30),
           # 'C_3': setDictValues(0,0,30), # Synechococcus
          # 'C_5': setDictValues(0,0,30),
          'C_6': setDictValues(0,0,30),
          'C_Y': setDictValues(0,0,1),
          'C_ism': setDictValues(0,0,20),
           'L_fl_lambda0': setDictValues(0,0,0.2),
           # 'L_fl_phycoerythrin': setDictValues(0,0,0.2),
           'L_fl_phycocyanin': setDictValues(0,0,0.2),
        'offset': setDictValues(0, -0.1, 0.1)
             },
    '4b': {'C_0': setDictValues(0,0,200),
            # 'C_1': setDictValues(0,0,200),
            # 'C_2': setDictValues(0,0,300),
            'C_3': setDictValues(0,0, 200), # Synechococcus
        #   'C_4': setDictValues(0,0,100),
        #     'C_5': setDictValues(0,0,10), # goldalge
          'C_6': setDictValues(0,0,300),
          'C_Y': setDictValues(0,0,1),
          'C_ism': setDictValues(0,0,20),
           'L_fl_lambda0': setDictValues(0,0,0.2),
           # 'L_fl_phycoerythrin': setDictValues(0,0,0.2),
           'L_fl_phycocyanin': setDictValues(0,0,0.2),
        # 'offset': setDictValues(0, -0.1, 0.1)
           },
    '5a': {'C_0': setDictValues(0.,0,200),
           # 'C_1': setDictValues(0,0,200),
           #  'C_2': setDictValues(0,0,300),
           # 'C_3': setDictValues(0,0,200), # Synechococcus
          # 'C_5': setDictValues(0,0,300),
           'C_6': setDictValues(0,0,300),
          'C_Y': setDictValues(0,0,1),
          'C_ism': setDictValues(0,0,20),
           'L_fl_lambda0': setDictValues(0,0,0.2),
            # 'L_fl_phycoerythrin': setDictValues(0,0,0.2),
            'L_fl_phycocyanin': setDictValues(0,0,0.2),
           # 'offset': setDictValues(0, -0.1, 0.1)
           },
    '5b': {'C_0': setDictValues(0,0,200),
            # 'C_1': setDictValues(0,0,200),
          # 'C_2': setDictValues(0,0,1000),
          #   'C_3': setDictValues(0,0,200), # Synechococcus
          # 'C_5': setDictValues(0,0,300),
          'C_6': setDictValues(0,0,1000),
          'C_Y': setDictValues(0,0, 1),
          'C_ism': setDictValues(0,0, 20),
           'L_fl_lambda0': setDictValues(0,0,0.2),
            # 'L_fl_phycoerythrin': setDictValues(0,0,0.2),
            'L_fl_phycocyanin': setDictValues(0,0,0.2),
           # 'offset': setDictValues(0, -0.1, 0.1)/
           },
    '6': {'C_0': setDictValues(0.,0,200),
          # 'C_1': setDictValues(0.,0,200),
          # 'C_2': setDictValues(0,0,500),
          #   'C_3': setDictValues(0,0,200), # Synechococcus
          # 'C_5': setDictValues(0,0,300),
          'C_6': setDictValues(0,0,500),
          'C_Y': setDictValues(0,0,1),
          'C_ism': setDictValues(0,0,20),
           'L_fl_lambda0': setDictValues(0,0,0.2),
          # 'L_fl_phycoerythrin': setDictValues(0,0,0.2),
          'L_fl_phycocyanin': setDictValues(0,0,0.2),
        # 'offset': setDictValues(0, -0.1, 0.1)
          },
    '7': {'C_0': setDictValues(0.,0,200),
          # 'C_1': setDictValues(0.,0,200),
          # 'C_2': setDictValues(0,0,200),
          #   'C_3': setDictValues(0,0,200), # Synechococcus
          # 'C_5': setDictValues(0,0,200),
          'C_6': setDictValues(0,0,200),
          'C_Y': setDictValues(0,0,1),
          'C_ism': setDictValues(0,0,20),
           'L_fl_lambda0': setDictValues(0,0,0.2),
          # 'L_fl_phycoerythrin': setDictValues(0,0,0.2),
          'L_fl_phycocyanin': setDictValues(0,0,0.2),
        # 'offset': setDictValues(0, -0.1, 0.1)
          }
}

processingDict_Helsinki = {
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
            'L_fl_phycocyanin': setDictValues(0,0,0.2),
        'offset': setDictValues(0, -0.1, 0.1)
           },
    '3a_y': {'C_0': setDictValues(0,0,10),
            # 'C_1': setDictValues(0,0,10),
          # 'C_2': setDictValues(0,0,10),
           'C_3': setDictValues(0,0,10), # Synechococcus
          # 'C_4': setDictValues(0,0,10),
          #   'C_6': setDictValues(0,0,10),
          'C_Y': setDictValues(0,0,3),
          'C_ism': setDictValues(0,0,10),
           'L_fl_lambda0': setDictValues(0,0,0.2),
           # 'L_fl_phycoerythrin': setDictValues(0,0,0.2),
            'L_fl_phycocyanin': setDictValues(0,0,0.2),
        'offset': setDictValues(0, -0.1, 0.1)
           },
    '3b': {'C_0': setDictValues(0,0,10),
            # 'C_1': setDictValues(0,0,10),
          # 'C_2': setDictValues(0,0,10),
            'C_3': setDictValues(0,0,10),
          #  'C_5': setDictValues(0,0,10), # coccolith.
          #  'C_6': setDictValues(0,0,10),
          'C_Y': setDictValues(0,0,10),
          'C_ism': setDictValues(0,0,10),
           'L_fl_lambda0': setDictValues(0,0,0.2),
            'L_fl_phycocyanin': setDictValues(0,0,0.2),
           # 'L_fl_phycoerythrin': setDictValues(0,0,0.2)
        'offset': setDictValues(0, -0.1, 0.1)
           },
    '4a_g': {'C_0': setDictValues(0,0,30),
            # 'C_1': setDictValues(0,0,30),
          # 'C_2': setDictValues(0,0,30),
           'C_3': setDictValues(0,0,30), # Synechococcus
          # 'C_4': setDictValues(0,0,30),
           'C_6': setDictValues(0,0,30),
          'C_Y': setDictValues(0,0,5),
          'C_ism': setDictValues(0,0,20),
           'L_fl_lambda0': setDictValues(0,0,0.2),
           # 'L_fl_phycoerythrin': setDictValues(0,0,0.2),
           'L_fl_phycocyanin': setDictValues(0,0,0.2),
        'offset': setDictValues(0, -0.1, 0.1)
             },
    '4a_y': {'C_0': setDictValues(0,0,30),
        # 'C_1': setDictValues(0,0,30),
          # 'C_2': setDictValues(0,0,30),
           'C_3': setDictValues(0,0,30), # Synechococcus
          # 'C_4': setDictValues(0,0,30),
           'C_6': setDictValues(0,0,30),
          'C_Y': setDictValues(0,0,10),
          'C_ism': setDictValues(0,0,20),
           'L_fl_lambda0': setDictValues(0,0,0.2),
           # 'L_fl_phycoerythrin': setDictValues(0,0,0.2),
           'L_fl_phycocyanin': setDictValues(0,0,0.2),
        'offset': setDictValues(0, -0.1, 0.1)
             },
    '4b': {'C_0': setDictValues(0,0,200),
        # 'C_1': setDictValues(0,0,200),
        #   'C_2': setDictValues(0,0,300),
        'C_3': setDictValues(0,0, 200), # Synechococcus
        #   'C_4': setDictValues(0,0,100),
        #     'C_5': setDictValues(0,0,10), # coccolith.
           'C_6': setDictValues(0,0,300),
          'C_Y': setDictValues(0,0,10),
          'C_ism': setDictValues(0,0,20),
           'L_fl_lambda0': setDictValues(0,0,0.2),
           # 'L_fl_phycoerythrin': setDictValues(0,0,0.2),
           'L_fl_phycocyanin': setDictValues(0,0,0.2),
        'offset': setDictValues(0, -0.1, 0.1)
           },
    '5a': {'C_0': setDictValues(0.,0,200),
           # 'C_1': setDictValues(0,0,200),
          # 'C_2': setDictValues(0,0,300),
           'C_3': setDictValues(0,0,200), # Synechococcus
          # 'C_4': setDictValues(0,0,300),
          'C_6': setDictValues(0,0,300),
          'C_Y': setDictValues(0,0,10),
          'C_ism': setDictValues(0,0,20),
           'L_fl_lambda0': setDictValues(0,0,0.2),
            # 'L_fl_phycoerythrin': setDictValues(0,0,0.2),
            'L_fl_phycocyanin': setDictValues(0,0,0.2),
           'offset': setDictValues(0, -0.1, 0.1)
           },
    '5b': {'C_0': setDictValues(0,0,200),
            # 'C_1': setDictValues(0,0,200),
          # 'C_2': setDictValues(0,0,1000),
            'C_3': setDictValues(0,0,200), # Synechococcus
          # 'C_4': setDictValues(0,0,300),
          'C_6': setDictValues(0,0,1000),
          'C_Y': setDictValues(0,0, 10),
          'C_ism': setDictValues(0,0, 20),
           'L_fl_lambda0': setDictValues(0,0,0.2), #chl-a fluorescence
            # 'L_fl_phycoerythrin': setDictValues(0,0,0.2),
            'L_fl_phycocyanin': setDictValues(0,0,0.2),
           'offset': setDictValues(0, -0.1, 0.1)
           },
    '6': {'C_0': setDictValues(0.,0,200),
          # 'C_1': setDictValues(0.,0,200),
          # 'C_2': setDictValues(0,0,500),
            'C_3': setDictValues(0,0,200), # Synechococcus
          # 'C_4': setDictValues(0,0,300),
          'C_6': setDictValues(0,0,500),
          'C_Y': setDictValues(0,0,10),
          'C_ism': setDictValues(0,0,20),
           'L_fl_lambda0': setDictValues(0,0,0.2),
          # 'L_fl_phycoerythrin': setDictValues(0,0,0.2),
          'L_fl_phycocyanin': setDictValues(0,0,0.2),
        'offset': setDictValues(0, -0.1, 0.1)
          },
    '7': {'C_0': setDictValues(0.,0,200),
          # 'C_1': setDictValues(0.,0,200),
          # 'C_2': setDictValues(0,0,200),
            'C_3': setDictValues(0,0,200), # Synechococcus
          # 'C_4': setDictValues(0,0,200),
          'C_6': setDictValues(0,0,200),
          'C_Y': setDictValues(0,0,10),
          'C_ism': setDictValues(0,0,2),
           'L_fl_lambda0': setDictValues(0,0,0.2),
          # 'L_fl_phycoerythrin': setDictValues(0,0,0.2),
          'L_fl_phycocyanin': setDictValues(0,0,0.2),
        'offset': setDictValues(0, -0.1, 0.1)
          },
    'nan': {'C_0': setDictValues(0.,0,200),
          # 'C_1': setDictValues(0.,0,200),
          # 'C_2': setDictValues(0,0,200),
            'C_3': setDictValues(0,0,200), # Synechococcus
          # 'C_4': setDictValues(0,0,200),
          'C_6': setDictValues(0,0,200),
          'C_Y': setDictValues(0,0,10),
          'C_ism': setDictValues(0,0,2),
           'L_fl_lambda0': setDictValues(0,0,0.2),
          # 'L_fl_phycoerythrin': setDictValues(0,0,0.2),
          'L_fl_phycocyanin': setDictValues(0,0,0.2),
        'offset': setDictValues(0, -0.1, 0.1)
          }
}

processingDict_Helsinki_EnMAPsimplified = {
    '1': {
        'C_0': setDictValues(0.5,0,10),
            # 'C_1': setDictValues(0,0,10),
          # 'C_2': setDictValues(0,0,10),
          #  'C_3': setDictValues(0.5,0,10), # Synechococcus
          # 'C_4': setDictValues(0,0,10),
          #  'C_6': setDictValues(0,0,10),
          #   'C_7': setDictValues(0,0,10),
          # 'C_Y': setDictValues(0,0,1),
          'C_ism': setDictValues(0,0,1),
          #'S_cdom': setDictValues(value=0.020, vmin=0.016, vmax=0.022),
           # 'L_fl_lambda0': setDictValues(0,0,0.2),
           # 'L_fl_phycoerythrin': setDictValues(0,0,0.2),
           #  'L_fl_phycocyanin': setDictValues(0,0,0.2)
        'offset': setDictValues(0, -0.01, 0.01)
        },
    '2': {'C_0': setDictValues(1,0,10),
            # 'C_1': setDictValues(0,0,10),
          # 'C_2': setDictValues(0,0,10),
          #  'C_3': setDictValues(1,0,10), # Synechococcus
          # 'C_4': setDictValues(0,0,10),
          #  'C_6': setDictValues(0,0,10),
          #   'C_7': setDictValues(0,0,10),
          'C_Y': setDictValues(0,0,1),
          'C_ism': setDictValues(0,0, 2),
        #'S_cdom': setDictValues(value=0.020, vmin=0.016, vmax=0.022),
           # 'L_fl_lambda0': setDictValues(0,0,0.2),
           # 'L_fl_phycoerythrin': setDictValues(0,0,0.2),
           #  'L_fl_phycocyanin': setDictValues(0,0,0.2)
        'offset': setDictValues(0, -0.01, 0.01)
           },
    '3': {'C_0': setDictValues(1,0,30),
            # 'C_1': setDictValues(0,0,10),
          # 'C_2': setDictValues(0,0,10),
            'C_3': setDictValues(1,0,30),
          #  'C_5': setDictValues(0,0,10), # coccolith.
          # 'C_6': setDictValues(0,0,10),
          'C_Y': setDictValues(0,0,10),
          'C_ism': setDictValues(1,0,10),
          # 'S_cdom': setDictValues(value=0.020, vmin=0.016, vmax=0.022),
           # 'L_fl_lambda0': setDictValues(0,0,0.2),
           #  'L_fl_phycocyanin': setDictValues(0,0,0.2)
           # 'L_fl_phycoerythrin': setDictValues(0,0,0.2)
        'offset': setDictValues(0, -0.01, 0.01)
           },
    '4': {'C_0': setDictValues(10,0,300),
            # 'C_1': setDictValues(0,0,30),
          # 'C_2': setDictValues(0,0,30),
           'C_3': setDictValues(10,0,300), # Synechococcus
          # 'C_4': setDictValues(0,0,30),
          # 'C_6': setDictValues(0,0,30),
          'C_Y': setDictValues(1,0,5),
          'C_ism': setDictValues(10,0,200),
           # 'S_cdom': setDictValues(value=0.020, vmin=0.016, vmax=0.022),
           # 'L_fl_lambda0': setDictValues(0,0,0.2),
           # 'L_fl_phycoerythrin': setDictValues(0,0,0.2),
           # 'L_fl_phycocyanin': setDictValues(0,0,0.2),
        'offset': setDictValues(0, -0.01, 0.01)
             },
    '5': {'C_0': setDictValues(10,0,300),
        # 'C_1': setDictValues(0,0,30),
          # 'C_2': setDictValues(0,0,30),
           'C_3': setDictValues(10,0,300), # Synechococcus
          # 'C_4': setDictValues(0,0,30),
          # 'C_6': setDictValues(0,0,30),
          'C_Y': setDictValues(1,0,10),
          'C_ism': setDictValues(10,0,200),
          #'S_cdom': setDictValues(value=0.020, vmin=0.016, vmax=0.022),
           # 'L_fl_lambda0': setDictValues(0,0,0.2),
           # 'L_fl_phycoerythrin': setDictValues(0,0,0.2),
           # 'L_fl_phycocyanin': setDictValues(0,0,0.2),
        'offset': setDictValues(0, -0.01, 0.01)
        }
}

processingDict_California_EnMAPsimplified = {
    '1': {
        'C_0': setDictValues(0.5,0,10),
            'C_1': setDictValues(0.5,0,10),
          # 'C_2': setDictValues(0,0,10),
           'C_3': setDictValues(0.5,0,10), # Synechococcus
          # 'C_4': setDictValues(0,0,10),
          #  'C_6': setDictValues(0,0,10),
          #   'C_7': setDictValues(0,0,10),
          'C_Y': setDictValues(0,0,1),
          'C_ism': setDictValues(0,0,1),
          #'S_cdom': setDictValues(value=0.020, vmin=0.016, vmax=0.022),
           # 'L_fl_lambda0': setDictValues(0,0,0.2),
           # 'L_fl_phycoerythrin': setDictValues(0,0,0.2),
           #  'L_fl_phycocyanin': setDictValues(0,0,0.2)
        # 'offset': setDictValues(0, -0.01, 0.01)
        },
    '2': {'C_0': setDictValues(1,0,10),
            'C_1': setDictValues(1,0,10),
          # 'C_2': setDictValues(0,0,10),
           'C_3': setDictValues(1,0,10), # Synechococcus
          # 'C_4': setDictValues(0,0,10),
          #  'C_6': setDictValues(0,0,10),
          #   'C_7': setDictValues(0,0,10),
          'C_Y': setDictValues(0,0,1),
          'C_ism': setDictValues(0,0, 2),
        #'S_cdom': setDictValues(value=0.020, vmin=0.016, vmax=0.022),
           # 'L_fl_lambda0': setDictValues(0,0,0.2),
           # 'L_fl_phycoerythrin': setDictValues(0,0,0.2),
           #  'L_fl_phycocyanin': setDictValues(0,0,0.2)
        # 'offset': setDictValues(0, -0.01, 0.01)
           },
    '3': {'C_0': setDictValues(1,0,30),
            'C_1': setDictValues(1,0,10),
          # 'C_2': setDictValues(0,0,10),
            'C_3': setDictValues(1,0,30),
          #  'C_5': setDictValues(0,0,10), # coccolith.
          # 'C_6': setDictValues(0,0,10),
          'C_Y': setDictValues(0,0,10),
          'C_ism': setDictValues(1,0,10),
          # 'S_cdom': setDictValues(value=0.020, vmin=0.016, vmax=0.022),
           # 'L_fl_lambda0': setDictValues(0,0,0.2),
           #  'L_fl_phycocyanin': setDictValues(0,0,0.2)
           # 'L_fl_phycoerythrin': setDictValues(0,0,0.2)
        # 'offset': setDictValues(0, -0.01, 0.01)
           },
    '4': {'C_0': setDictValues(10,0,300),
            'C_1': setDictValues(10,0,30),
          # 'C_2': setDictValues(0,0,30),
           'C_3': setDictValues(10,0,300), # Synechococcus
          # 'C_4': setDictValues(0,0,30),
          # 'C_6': setDictValues(0,0,30),
          'C_Y': setDictValues(1,0,5),
          'C_ism': setDictValues(10,0,200),
           # 'S_cdom': setDictValues(value=0.020, vmin=0.016, vmax=0.022),
           # 'L_fl_lambda0': setDictValues(0,0,0.2),
           # 'L_fl_phycoerythrin': setDictValues(0,0,0.2),
           # 'L_fl_phycocyanin': setDictValues(0,0,0.2),
        # 'offset': setDictValues(0, -0.01, 0.01)
             },
    '5': {'C_0': setDictValues(10,0,300),
        # 'C_1': setDictValues(10,0,300),
          # 'C_2': setDictValues(0,0,30),
           'C_3': setDictValues(10,0,300), # Synechococcus
          # 'C_4': setDictValues(0,0,30),
          # 'C_6': setDictValues(0,0,30),
          'C_Y': setDictValues(1,0,10),
          'C_ism': setDictValues(10,0,200),
          #'S_cdom': setDictValues(value=0.020, vmin=0.016, vmax=0.022),
           # 'L_fl_lambda0': setDictValues(0,0,0.2),
           # 'L_fl_phycoerythrin': setDictValues(0,0,0.2),
           # 'L_fl_phycocyanin': setDictValues(0,0,0.2),
        # 'offset': setDictValues(0, -0.01, 0.01)
        }
}


processingDict_Oder = {
    '1': {'C_7': setDictValues(0,0,1)},
    '2': {'C_0': setDictValues(0,0,1),
       #   'C_2': setDictValues(0,0,1),
       #   'C_3': setDictValues(0,0,1), # Synechococcus
       #   'C_6': setDictValues(0,0,1),
       #   'C_7': setDictValues(0,0,1),
          'C_Y': setDictValues(0,0,0.1),
          'C_ism': setDictValues(0,0,1)}, # no fluorescence!
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
            'C_1': setDictValues(0,0,30),
          'C_2': setDictValues(0,0,30),
           'C_3': setDictValues(0,0,30), # Synechococcus
          # 'C_4': setDictValues(0,0,30),
          # 'C_6': setDictValues(0,0,30),
          'C_Y': setDictValues(0,0,5),
          'C_ism': setDictValues(0,0,20),
           'L_fl_lambda0': setDictValues(0,0,0.2),
           # 'L_fl_phycoerythrin': setDictValues(0,0,0.2),
           'L_fl_phycocyanin': setDictValues(0,0,0.2),
       #  'offset': setDictValues(0, -0.1, 0.1)
             },
    '4a_y': {'C_0': setDictValues(0,0,30),
        'C_1': setDictValues(0,0,30),
           'C_2': setDictValues(0,0,30),
           'C_3': setDictValues(0,0,30), # Synechococcus
          # 'C_4': setDictValues(0,0,30),
          # 'C_6': setDictValues(0,0,30),
          'C_Y': setDictValues(0,0,1),
          'C_ism': setDictValues(0,0,20),
           'L_fl_lambda0': setDictValues(0,0,0.2),
           # 'L_fl_phycoerythrin': setDictValues(0,0,0.2),
           'L_fl_phycocyanin': setDictValues(0,0,0.2),
       #  'offset': setDictValues(0, -0.1, 0.1)
             },
    '4b': {'C_0': setDictValues(0,0,200),
        'C_1': setDictValues(0,0,200),
           'C_2': setDictValues(0,0,300),
        'C_3': setDictValues(0,0, 200), # Synechococcus
        #   'C_4': setDictValues(0,0,100),
        #     'C_5': setDictValues(0,0,10), # coccolith.
        #   'C_6': setDictValues(0,0,300),
          'C_Y': setDictValues(0,0,1),
          'C_ism': setDictValues(0,0,50),
           'L_fl_lambda0': setDictValues(0,0,0.2),
           # 'L_fl_phycoerythrin': setDictValues(0,0,0.2),
           'L_fl_phycocyanin': setDictValues(0,0,0.2),
        # 'offset': setDictValues(0, -0.1, 0.1)
           },
    '5a': {'C_0': setDictValues(0.,0,200),
           'C_1': setDictValues(0,0,200),
           'C_2': setDictValues(0,0,300),
           'C_3': setDictValues(0,0,200), # Synechococcus
          # 'C_4': setDictValues(0,0,300),
          #  'C_6': setDictValues(0,0,300),
          'C_Y': setDictValues(0,0,1),
          'C_ism': setDictValues(0,0,50),
           'L_fl_lambda0': setDictValues(0,0,0.2),
            # 'L_fl_phycoerythrin': setDictValues(0,0,0.2),
            'L_fl_phycocyanin': setDictValues(0,0,0.2),
          #  'offset': setDictValues(0, -0.1, 0.1)
           },
    '5b': {'C_0': setDictValues(0,0,200),
            'C_1': setDictValues(0,0,200),
           'C_2': setDictValues(0,0,200),
            'C_3': setDictValues(0,0,200), # Synechococcus
          # 'C_4': setDictValues(0,0,300),
          # 'C_6': setDictValues(0,0,1000),
          'C_Y': setDictValues(0,0, 1),
          'C_ism': setDictValues(0,0, 50),
           'L_fl_lambda0': setDictValues(0,0,0.2),
            # 'L_fl_phycoerythrin': setDictValues(0,0,0.2),
            'L_fl_phycocyanin': setDictValues(0,0,0.2),
          #  'offset': setDictValues(0, -0.1, 0.1)
           },
    '6': {'C_0': setDictValues(0.,0,200),
          'C_1': setDictValues(0.,0,200),
           'C_2': setDictValues(0,0,500),
            'C_3': setDictValues(0,0,200), # Synechococcus
          # 'C_4': setDictValues(0,0,300),
          # 'C_6': setDictValues(0,0,500),
          'C_Y': setDictValues(0,0,1),
          'C_ism': setDictValues(0,0,50),
           'L_fl_lambda0': setDictValues(0,0,0.2),
          # 'L_fl_phycoerythrin': setDictValues(0,0,0.2),
          'L_fl_phycocyanin': setDictValues(0,0,0.2),
         #'offset': setDictValues(0, -0.1, 0.1)
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

processingDict_Vortsjarv = {

    '4a_g': {
        #'C_0': setDictValues(0,0,30),
            # 'C_1': setDictValues(0,0,30),
          # 'C_2': setDictValues(0,0,30),
          #  'C_3': setDictValues(0,0,30),
           'C_5': setDictValues(0,0,30), # Goldalgae
           'C_6': setDictValues(0,0,30),
          'C_Y': setDictValues(0,0,5),
          'C_ism': setDictValues(0,0,20),
           'L_fl_lambda0': setDictValues(0,0,0.2),
           # 'L_fl_phycoerythrin': setDictValues(0,0,0.2),
           'L_fl_phycocyanin': setDictValues(0,0,0.2),
       #  'offset': setDictValues(0, -0.1, 0.1)
             },
    '4a_y': {
        # 'C_0': setDictValues(0,0,30),
            # 'C_1': setDictValues(0,0,30),
           # 'C_2': setDictValues(0,0,30),
           # 'C_3': setDictValues(0,0,30),
           'C_5': setDictValues(0,0,30), # Goldalgae
           'C_6': setDictValues(0,0,30),
          'C_Y': setDictValues(0,0,5),
          'C_ism': setDictValues(0,0,20),
           'L_fl_lambda0': setDictValues(0,0,0.2),
           # 'L_fl_phycoerythrin': setDictValues(0,0,0.2),
           'L_fl_phycocyanin': setDictValues(0,0,0.2),
       #  'offset': setDictValues(0, -0.1, 0.1)
             },
    '4b': {
        # 'C_0': setDictValues(0,0,200),
            # 'C_1': setDictValues(0,0,200),
           # 'C_2': setDictValues(0,0,300),
           #  'C_3': setDictValues(0,0, 200),
            'C_5': setDictValues(0,0,200), # Goldalgae
          'C_6': setDictValues(0,0,300),
          'C_Y': setDictValues(0,0,5),
          'C_ism': setDictValues(0,0,50),
           'L_fl_lambda0': setDictValues(0,0,0.2),
           # 'L_fl_phycoerythrin': setDictValues(0,0,0.2),
           'L_fl_phycocyanin': setDictValues(0,0,0.2),
        # 'offset': setDictValues(0, -0.1, 0.1)
           },
    '5a': {
        # 'C_0': setDictValues(0.,0,200),
           # 'C_1': setDictValues(0,0,200),
           # 'C_2': setDictValues(0,0,300),
           # 'C_3': setDictValues(0,0,200),
          'C_5': setDictValues(0,0,300), # Goldalgae
           'C_6': setDictValues(0,0,300),
          'C_Y': setDictValues(0,0,5),
          'C_ism': setDictValues(0,0,50),
           'L_fl_lambda0': setDictValues(0,0,0.2),
            # 'L_fl_phycoerythrin': setDictValues(0,0,0.2),
            'L_fl_phycocyanin': setDictValues(0,0,0.2),
          #  'offset': setDictValues(0, -0.1, 0.1)
           },
    '5b': {
        # 'C_0': setDictValues(0,0,200),
            # 'C_1': setDictValues(0,0,200),
           # 'C_2': setDictValues(0,0,200),
           #  'C_3': setDictValues(0,0,200),
          'C_5': setDictValues(0,0,300), # Goldalgae
          'C_6': setDictValues(0,0,1000),
          'C_Y': setDictValues(0,0, 5),
          'C_ism': setDictValues(0,0, 50),
           'L_fl_lambda0': setDictValues(0,0,0.2),
            # 'L_fl_phycoerythrin': setDictValues(0,0,0.2),
            'L_fl_phycocyanin': setDictValues(0,0,0.2),
          #  'offset': setDictValues(0, -0.1, 0.1)
           },
    '6': {
        # 'C_0': setDictValues(0.,0,200),
          # 'C_1': setDictValues(0.,0,200),
          #  'C_2': setDictValues(0,0,500),
          #   'C_3': setDictValues(0,0,200),
          'C_5': setDictValues(0,0,300), # Goldalgae
          'C_6': setDictValues(0,0,500),
          'C_Y': setDictValues(0,0,5),
          'C_ism': setDictValues(0,0,50),
           'L_fl_lambda0': setDictValues(0,0,0.2),
          # 'L_fl_phycoerythrin': setDictValues(0,0,0.2),
          'L_fl_phycocyanin': setDictValues(0,0,0.2),
         #'offset': setDictValues(0, -0.1, 0.1)
          },
    '7': {
    # 'C_0': setDictValues(0.,0,200),
          # 'C_1': setDictValues(0.,0,200),
          #  'C_2': setDictValues(0,0,500),
          #   'C_3': setDictValues(0,0,200),
          'C_5': setDictValues(0,0,300), # Goldalgae
          'C_6': setDictValues(0,0,500),
          'C_Y': setDictValues(0,0,5),
          'C_ism': setDictValues(0,0,50),
           'L_fl_lambda0': setDictValues(0,0,0.2),
          # 'L_fl_phycoerythrin': setDictValues(0,0,0.2),
          'L_fl_phycocyanin': setDictValues(0,0,0.2),
         #'offset': setDictValues(0, -0.1, 0.1)
          }
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


## Summer ##
# processingDict_NorthSea = {
#     '1': {'C_7': setDictValues(0,0,1)},
#     '2': {'C_0': setDictValues(0,0,1),
#           'C_2': setDictValues(0,0,1),
#           'C_4': setDictValues(0,0,1),
#           'C_5': setDictValues(0,0,1), #Phaeocystis
#            'C_6': setDictValues(0,0,1),
#         'C_7': setDictValues(0,0,1), #NOctiluca
#           'C_Y': setDictValues(0,0,0.1),
#           'C_ism': setDictValues(0,0,1)}, # no fluorescence!
#     '3a_g': {'C_0': setDictValues(0.1,0,10),
#           # 'C_2': setDictValues(0,0,10),
#           #  'C_3': setDictValues(0,0,10), # Synechococcus
#           'C_4': setDictValues(0,0,10),
#           'C_5': setDictValues(0,0,10), #Phaeocystis
#            'C_6': setDictValues(0,0,10),
#            'C_7': setDictValues(0,0,10), #NOctiluca
#           'C_Y': setDictValues(0,0,1),
#           'C_ism': setDictValues(0,0,10),
#            'L_fl_lambda0': setDictValues(0,0,0.2),
#            'L_fl_phycoerythrin': setDictValues(0,0,0.2),
#             # 'L_fl_phycocyanin': setDictValues(0,0,0.2)
#            },
#     '3a_y': {'C_0': setDictValues(0.1,0,10),
#           # 'C_2': setDictValues(0,0,10),
#           #  'C_3': setDictValues(0,0,10), # Synechococcus
#           'C_4': setDictValues(0,0,10),
#           'C_5': setDictValues(0,0,10), #Phaeocystis
#            'C_6': setDictValues(0,0,10),
#            'C_7': setDictValues(0,0,10), #NOctiluca
#           'C_Y': setDictValues(0,0,10),
#           'C_ism': setDictValues(0,0,10),
#            'L_fl_lambda0': setDictValues(0,0,0.2),
#            'L_fl_phycoerythrin': setDictValues(0,0,0.2),
#             # 'L_fl_phycocyanin': setDictValues(0,0,0.2)
#            },
#     '3b': {'C_0': setDictValues(0.1,0,10),
#           'C_2': setDictValues(0,0,10),
#            'C_5': setDictValues(0,0,10), # coccolith.
#           'C_6': setDictValues(0,0,10),
#           'C_Y': setDictValues(0,0,10),
#           'C_ism': setDictValues(0,0,10),
#            'L_fl_lambda0': setDictValues(0,0,0.2),
#            'L_fl_phycoerythrin': setDictValues(0,0,0.2)},
#     '4a_g': {'C_0': setDictValues(0.1,0,30),
#           # 'C_2': setDictValues(0,0,10),
#           #  'C_3': setDictValues(0,0,10), # Synechococcus
#           'C_4': setDictValues(0,0,30),
#           'C_5': setDictValues(0,0,30), #Phaeocystis
#            'C_6': setDictValues(0,0,30),
#            'C_7': setDictValues(0,0,30), #NOctiluca
#           'C_Y': setDictValues(0,0,5),
#           'C_ism': setDictValues(0,0,20),
#            'L_fl_lambda0': setDictValues(0,0,0.2),
#            'L_fl_phycoerythrin': setDictValues(0,0,0.2),
#             # 'L_fl_phycocyanin': setDictValues(0,0,0.2)
#            },
#     '4a_y': {'C_0': setDictValues(0.1,0,30),
#           # 'C_2': setDictValues(0,0,10),
#           #  'C_3': setDictValues(0,0,10), # Synechococcus
#           'C_4': setDictValues(0,0,30),
#           'C_5': setDictValues(0,0,30), #Phaeocystis
#            'C_6': setDictValues(0,0,30),
#            'C_7': setDictValues(0,0,30), #NOctiluca
#           'C_Y': setDictValues(0,0,10),
#           'C_ism': setDictValues(0,0,20),
#            'L_fl_lambda0': setDictValues(0,0,0.2),
#            'L_fl_phycoerythrin': setDictValues(0,0,0.2),
#             # 'L_fl_phycocyanin': setDictValues(0,0,0.2)
#            },
#     '4b': {'C_0': setDictValues(0.1,0,300),
#           # 'C_2': setDictValues(0,0,300),
#         # 'C_3': setDictValues(0,0,100), # Synechococcus
#           'C_4': setDictValues(0,0,100),
#             'C_5': setDictValues(0,0,300), #Phaeocystis
#           'C_6': setDictValues(0,0,300),
#             'C_7': setDictValues(0,0,300), #Noctiluca
#           'C_Y': setDictValues(0,0,10),
#           'C_ism': setDictValues(0,0,100),
#            'L_fl_lambda0': setDictValues(0,0,0.2),
#            'L_fl_phycoerythrin': setDictValues(0,0,0.2),
#            'L_fl_phycocyanin': setDictValues(0,0,0.2)},
#     '5a': {'C_0': setDictValues(0.1,0,300),
#           # 'C_2': setDictValues(0,0,300),
#            # 'C_3': setDictValues(0,0,300), # Synechococcus
#           'C_4': setDictValues(0,0,300),
#            'C_5': setDictValues(0,0,300), #Phaeocystis
#           'C_6': setDictValues(0,0,300),
#             'C_7': setDictValues(0,0,300), #Noctiluca
#           'C_Y': setDictValues(0,0,100),
#           'C_ism': setDictValues(0,0,100),
#            'L_fl_lambda0': setDictValues(0,0,0.2),
#             'L_fl_phycoerythrin': setDictValues(0,0,0.2),
#             'L_fl_phycocyanin': setDictValues(0,0,0.2)
#            # 'offset': setDictValues(0, -0.1, 0.1)
#            },
#     '5b': {'C_0': setDictValues(0.1,0,1000),
#             'C_1': setDictValues(0.1,0,100),
#           # 'C_2': setDictValues(0,0,1000),
#           #   'C_3': setDictValues(0,0,300), # Synechococcus
#           'C_4': setDictValues(0,0,300),
#           'C_5': setDictValues(0,0,1000), #Phaeocystis
#           'C_6': setDictValues(0,0,1000),
#             'C_7': setDictValues(0,0,1000), #Noctiluca
#           'C_Y': setDictValues(0,0,100),
#           'C_ism': setDictValues(0,0,100),
#            'L_fl_lambda0': setDictValues(0,0,0.2),
#             'L_fl_phycoerythrin': setDictValues(0,0,0.2),
#             'L_fl_phycocyanin': setDictValues(0,0,0.2)
#            # 'offset': setDictValues(0, -0.1, 0.1)
#            },
#     '6': {'C_0': setDictValues(0.1,0,500),
#           'C_1': setDictValues(0.,0,500),
#           # 'C_2': setDictValues(0,0,500),
#           #   'C_3': setDictValues(0,0,300), # Synechococcus
#           'C_4': setDictValues(0,0,300),
#           'C_5': setDictValues(0,0,500), #Phaeocystis
#           'C_6': setDictValues(0,0,500),
#             'C_7': setDictValues(0,0,500), #Noctiluca
#           'C_Y': setDictValues(0,0,100),
#           'C_ism': setDictValues(0,0,1000),
#            'L_fl_lambda0': setDictValues(0,0,0.2),
#           'L_fl_phycoerythrin': setDictValues(0,0,0.2),
#           'L_fl_phycocyanin': setDictValues(0,0,0.2)},
#     '7': {'C_0': setDictValues(0.1,0,200),
#           'C_1': setDictValues(0.,0,200),
#           # 'C_2': setDictValues(0,0,200),
#           #   'C_3': setDictValues(0,0,200), # Synechococcus
#           'C_4': setDictValues(0,0,200),
#           'C_5': setDictValues(0,0,200), #Phaeocystis
#           'C_6': setDictValues(0,0,200),
#             'C_7': setDictValues(0,0,200), #Noctiluca
#           'C_Y': setDictValues(0,0,30),
#           'C_ism': setDictValues(0,0,100),
#            'L_fl_lambda0': setDictValues(0,0,0.2),
#           'L_fl_phycoerythrin': setDictValues(0,0,0.2),
#           'L_fl_phycocyanin': setDictValues(0,0,0.2)}
# }

# processingDict_BalticSea = {
#     '1': {'C_7': setDictValues(0,0,1)},
#     '2': {'C_0': setDictValues(0,0,1),
#           'C_2': setDictValues(0,0,1),
#           'C_3': setDictValues(0,0,1),
#           'C_6': setDictValues(0,0,1),
#           'C_7': setDictValues(0,0,1),
#           'C_Y': setDictValues(0,0,0.1),
#           'C_ism': setDictValues(0,0,1)}, # no fluorescence!
#     '3a': {'C_0': setDictValues(0.1,0,10),
#           'C_2': setDictValues(0,0,10),
#            'C_3' : setDictValues(0,0,10),
#           'C_4' : setDictValues(0,0,10),
#           'C_6': setDictValues(0,0,10),
#           'C_Y': setDictValues(0,0,1),
#           'C_ism': setDictValues(0,0,10),
#            'L_fl_lambda0': setDictValues(0,0,0.2),
#            'L_fl_phycocyanin': setDictValues(0,0,0.2),
#            'L_fl_phycoerythrin': setDictValues(0,0,0.2)},
#     '3b': {'C_0': setDictValues(0.1,0,10),
#           'C_2': setDictValues(0,0,10),
#            'C_3' : setDictValues(0,0,10),
#           'C_4' : setDictValues(0,0,10),
#           'C_6': setDictValues(0,0,10),
#           'C_Y': setDictValues(0,0,10),
#           'C_ism': setDictValues(0,0,10),
#            'L_fl_lambda0': setDictValues(0,0,0.2),
#            'L_fl_phycocyanin': setDictValues(0,0,0.2),
#            'L_fl_phycoerythrin': setDictValues(0,0,0.2)},
#     '4a': {'C_0': setDictValues(0.1,0,30),
#           'C_2': setDictValues(0,0,30),
#            'C_3' : setDictValues(0,0,30),
#           'C_4' : setDictValues(0,0,30),
#           'C_6': setDictValues(0,0,30),
#           'C_Y': setDictValues(0,0,5),
#           'C_ism': setDictValues(0,0,20),
#            'L_fl_lambda0': setDictValues(0,0,0.2),
#            'L_fl_phycocyanin': setDictValues(0,0,0.2),
#            'L_fl_phycoerythrin': setDictValues(0,0,0.2)},
#     '4b': {'C_0': setDictValues(0.1,0,300),
#           'C_2': setDictValues(0,0,300),
#            'C_3' : setDictValues(0,0,300),
#           'C_4' : setDictValues(0,0,300),
#           'C_6': setDictValues(0,0,300),
#           'C_Y': setDictValues(0,0,10),
#           'C_ism': setDictValues(0,0,100),
#            'L_fl_lambda0': setDictValues(0,0,0.2),
#            'L_fl_phycocyanin': setDictValues(0,0,0.2),
#            'L_fl_phycoerythrin': setDictValues(0,0,0.2)},
#     '5a': {'C_0': setDictValues(0.1,0,300),
#           'C_2': setDictValues(0,0,300),
#            'C_3' : setDictValues(0,0,300),
#           'C_4' : setDictValues(0,0,300),
#           'C_6': setDictValues(0,0,300),
#           'C_Y': setDictValues(0,0,100),
#           'C_ism': setDictValues(0,0,100),
#            'L_fl_lambda0': setDictValues(0,0,0.2),
#            'L_fl_phycocyanin': setDictValues(0,0,0.2),
#            'L_fl_phycoerythrin': setDictValues(0,0,0.2),
#            'offset': setDictValues(0, -0.1, 0.1)},
#     '5b': {'C_0': setDictValues(0.1,0,1000),
#             'C_1': setDictValues(0.1,0,100),
#           'C_2': setDictValues(0,0,1000),
#            'C_3' : setDictValues(0,0,1000),
#           'C_4' : setDictValues(0,0,1000),
#           'C_6': setDictValues(0,0,1000),
#           'C_Y': setDictValues(0,0,100),
#           'C_ism': setDictValues(0,0,100),
#            'L_fl_lambda0': setDictValues(0,0,0.2),
#            'L_fl_phycocyanin': setDictValues(0,0,0.2),
#            'L_fl_phycoerythrin': setDictValues(0,0,0.2),
#            'offset': setDictValues(0, -0.1, 0.1)},
#     '6': {'C_0': setDictValues(0.1,0,500),
#           'C_1': setDictValues(0.1,0,500),
#           'C_2': setDictValues(0,0,500),
#           'C_3' : setDictValues(0,0,500),
#           'C_4' : setDictValues(0,0,500),
#           'C_6': setDictValues(0,0,500),
#           'C_Y': setDictValues(0,0,100),
#           'C_ism': setDictValues(0,0,1000),
#            'L_fl_lambda0': setDictValues(0,0,0.2),
#            'L_fl_phycocyanin': setDictValues(0,0,0.2),
#            'L_fl_phycoerythrin': setDictValues(0,0,0.2)},
#     '7': {'C_0': setDictValues(0.1,0,100),
#           'C_1': setDictValues(0.1,0,100),
#           'C_2': setDictValues(0,0,100),
#           'C_3' : setDictValues(0,0,100),
#           'C_4' : setDictValues(0,0,100),
#           'C_6': setDictValues(0,0,100),
#           'C_Y': setDictValues(0,0,30),
#           'C_ism': setDictValues(0,0,100),
#            'L_fl_lambda0': setDictValues(0,0,0.2),
#            'L_fl_phycocyanin': setDictValues(0,0,0.2),
#            'L_fl_phycoerythrin': setDictValues(0,0,0.2)}
# }


### Begin MAIN ###
sensor = 'CHIME_Center' #'CHIME_Sim' #'EnMAP' #'CHIME_Sim'
versionAC = 'v1'
dataType = 'SimRrs_InsituForward' #'SimRrs_InsituForward' #'EnMAPSuperpixel' # 'Insitu' #'EnMAPSuperpixel'  # 'Sim'
seaName = 'HEATWISE' #'HEATWISE' #'California' #'GLORIA' # 'North Sea' #'AQUATIME' #'Lakes'
regionName =  'Mueggelsee' # 'dalaro-2' #'dalaro' #'Helsinki_EnMAP'# 'California' # #'GLORIA' #'EnMAP_NorthSea' #'BelgianCoastCPOWER' # 'BelgianCoastRT1' # 'vortsjarv', 'oder_frankfurt' # 'dalaro-2' #'pyhajarvi' # #'elbe_seemannshöft', 'elbe_bunthaus' #'oder_frankfurt'#'oder_hohenwutzen' #'Helsinki' # Mueggelsee, Helsinki, //'Stendorfer-See' #'Sibbersdorfer-See' #'Grosser-Binnensee' # 'Kellersee'
AlgaeGroupType = 'HEREONgold' #'Standardv2' #'Standardv2' # 'Standardv3' # 'NSSummerBloomsv3' Standardv2, 'HEREONgold'
datasetDate = '20251211' # '20250516'
# datasetID = 'Rrs_flags'
# maxWavelength= 750.

metaDict = {
    'elbe_bunthaus' : {'path': "Z:\projects\ongoing\AQUATIME\sharepoint\WP2 Representative Dataset\\RTM_simulations\Simulated Spectra\elbe_bunthaus\\",
                    'datasetName' : 'CHIME_sim_AQUATIME_elbe_bunthaus'},
    'dalaro-2': {'path' : "Z:\projects\ongoing\AQUATIME\sharepoint\WP2 Representative Dataset\\RTM_simulations\Simulated Spectra\dalaro-2\\",
             'datasetName': 'CHIME_sim_AQUATIME_dalaro-2'},
    'pyhajarvi': {'path' : "Z:\projects\ongoing\AQUATIME\sharepoint\WP2 Representative Dataset\\RTM_simulations\Simulated Spectra\pyhajarvi\\",
                  'datasetName' :'CHIME_sim_AQUATIME_pyhajarvi'},
    'elbe_seemannshöft' :{ 'path': "Z:\projects\ongoing\AQUATIME\sharepoint\WP2 Representative Dataset\\RTM_simulations\Simulated Spectra\elbe_seemannshöft\\",
                           'datasetName': 'CHIME_sim_AQUATIME_elbe-seemannshöft' },
    'oder_frankfurt': {'path': "Z:\projects\ongoing\AQUATIME\sharepoint\WP2 Representative Dataset\\RTM_simulations\Simulated Spectra\oder_frankfurt\\",
                       'datasetName': 'CHIME_sim_AQUATIME_oder_frankfurt'},
    'oder_hohenwutzen': {'path': "Z:\projects\ongoing\AQUATIME\sharepoint\WP2 Representative Dataset\\RTM_simulations\Simulated Spectra\oder_hohenwutzen\\",
                         'datasetName': 'CHIME_sim_AQUATIME_oder_hohenwutzen'},
    'vortsjarv': { 'path': "Z:\projects\ongoing\AQUATIME\sharepoint\WP2 Representative Dataset\\RTM_simulations\Simulated Spectra\\vortsjarv\\",
                   'datasetName': 'CHIME_sim_AQUATIME_vortsjarv'},
    'Helsinki_SS': {
        'path': "Z:\projects\ongoing\HEATWISE\sharepoint\WP2_workspace\Water Quality\\representative_dataset\helsinki_siljaserenade\\",
        'datasetName': 'CHIME_sim_HEATWISE_helsinki_ss'},
    'Helsinki_FM': {
        'path': "Z:\projects\ongoing\HEATWISE\sharepoint\WP2_workspace\Water Quality\\representative_dataset\helsinki_finnmaid\\"},
     'Mueggelsee': { #'path' : "Z:\projects\ongoing\HEATWISE\sharepoint\WP2_workspace\Water Quality\\representative_dataset\\berlin_muggelsee\\",
    #                'datasetName': ''
                   'path' : "Z:\projects\ongoing\HEATWISE\sharepoint\WP2_workspace\Water Quality\FullInversion\\bio-optics_forward\\",
                    'datasetName': 'SimRrs_InsituForward_HEREONgold_berlin_muggelsee_IOP_daily_CHIMEbands'
                   },
    'BelgianCoastRT1': {'path': "", 'datasetName' :'BelgianCoastRT1'},
    'BelgianCoastCPOWER': {'path': "", 'datasetName' :'BelgianCoastCPOWER'},
    'EnMAP_NorthSea' : {'path': "D:\Documents\projects\EnsAD\EnMAP\dask_oe_processor_2026\\",
                        'datasetName': '_d_withOWT' },
    'GLORIA' : {'path': "D:\Documents\projects\EnsAD\EnMAP\dask_oe_processor_2026\\",
                'datasetName': 'ValidationGLORIA_withOWTrefined'},
    'Helsinki_EnMAP': {'path': "D:\Documents\projects\EnsAD\EnMAP\dask_oe_processor_2026\Helsinki\\",
                # 'datasetName': 'EnMAPsuperpixel_DT0000158841_20251019T101424Z_002_V010502_20251020T203545Z_withOWT',
                # 'datasetName': 'EnMAPsuperpixel_DT0000158841_20251019T101424Z_002_V010502_20251020T203545Z_noCorr_simpleOWT_selection2', # Land AC! run with glint correction
                # 'datasetName': 'EnMAPsuperpixel_DT0000158841_20251019T101424Z_002_V010502_20251020T203545Z_glintCorr_LandAC' # Land AC after glint correction! ??
                #  'datasetName': 'EnMAPsuperpixel_DT0000158841_20251019T101424Z_002_V010505_20260205T211148Z_noCorr_WaterAC_withOWT'
                    'datasetName': 'EnMAPsuperpixel_ENMAP01-____L2A-DT0000158841_20251019T101424Z_002_V010505_20260206T113145Z-oe_dask_am3c_Rrs_Dogleg_GlintCorrLandAC_withOWT' # from LANdAC with glintCorr done by Marcel.

                       },
    'California': {
        'path': "D:\Documents\projects\EnsAD\EnMAP\dask_oe_processor_2026\California\\",
        # 'datasetName': "EnMAPsuperpixel_DT0000009827_20230310T193303Z_001_V010502_20250626T150515Z_noCorr_LandAC"
        'datasetName': "EnMAPsuperpixel_DT0000014484_20230414T194131Z_001_V010502_20250626T143926Z_noCorr_LandAC"
    }
}

if dataType == 'SimRrs_InsituForward' and seaName=='AQUATIME':
    path = "Z:\projects\ongoing\AQUATIME\sharepoint\WP2 Representative Dataset\RTM_simulations\FullInversion\\bio-optics_forward\\"
    datasetName = ""
else:
    path = metaDict[regionName]['path']
    datasetName = metaDict[regionName]['datasetName']

if seaName == 'HEATWISE':
    outpath = "Z:\projects\ongoing\HEATWISE\sharepoint\WP2_workspace\Water Quality\FullInversion\\"
if seaName == 'AQUATIME':
    outpath = "Z:\projects\ongoing\AQUATIME\sharepoint\WP2 Representative Dataset\RTM_simulations\FullInversion\\"
if seaName == 'North Sea' or seaName=='GLORIA':
    # outpath = "D:\Documents\projects\EnsAD\EnMAP\dask_oe_processor_2026\\FullInversion\\"
    outpath = "Z:\projects\ongoing\AQUATIME\sharepoint\WP3 Product Engineering\InversionPhytoplanktonGroups\\validation\FullInversion_bio-optics-hereon\\"
if seaName == 'California':
    outpath = "D:\Documents\projects\EnsAD\EnMAP\dask_oe_processor_2026\California\FullInversion\\"

maxWavelength = 850. #900.
if dataType == 'Sim':
    wavelengths = set_wavelengths_bySensor(sensor, versionAC, maxWL= maxWavelength)
elif dataType == 'SimRrs_InsituForward':
    fname = os.listdir(path)
    fname = [fn for fn in fname if fn.startswith('SimRrs_InsituForward') and 'CHIMEbands' in fn]
    dat = pd.read_csv(path + fname[0], sep='\t')
    wavelengths=np.asarray([float(a) for a in dat.columns.values])
    ID = np.logical_and(wavelengths >= 400, wavelengths <= maxWavelength)
    wavelengths = wavelengths[ID]

elif dataType == 'Insitu':
    if regionName == 'vortsjarv':
        path = "Z:\projects\ongoing\AQUATIME\sharepoint\WP2 Representative Dataset\Case_Study_3\Vortsjarv\Hyperspectral data\\"
        insituRrs = pd.read_csv(path + "Hypstar_radiometric_23-24.csv", header=0)
        wavelengths = np.asarray([float(a) for a in insituRrs.columns.values[1:]])
        ID = np.logical_and(wavelengths>=400, wavelengths<=maxWavelength)
        wavelengths = wavelengths[ID]
    if regionName == 'BelgianCoastRT1' or regionName == 'BelgianCoastCPOWER':
        path = "Z:\projects\ongoing\AQUATIME\sharepoint\WP2 Representative Dataset\RTM_simulations\concistency\\"
        insituRrs = pd.read_csv(path + "Insitu_Rrs_Belgian_Coast_-_RT1.csv", header=0)
        wavelengths = np.asarray([float(a.split('_')[1]) for a in insituRrs.columns.values[1:] if a.startswith('wl_')])
        ID = np.logical_and(wavelengths >= 400, wavelengths <= maxWavelength)
        wavelengths = wavelengths[ID]
    if regionName == 'GLORIA':
        insituRrs= pd.read_csv("D:\Documents\projects\EnsAD\EnMAP\dask_oe_processor_2026\ValidationGLORIA_withOWTrefined.txt", header=0, sep='\t')
        wavelengths = np.asarray([float(a) for a in insituRrs.columns.values[:-1]])
        ID = np.logical_and(wavelengths >= 400, wavelengths <= maxWavelength)
        wavelengths = wavelengths[ID]
elif dataType == 'EnMAPSuperpixel':
    if regionName == 'EnMAP_NorthSea' or regionName=='Helsinki_EnMAP' or regionName=='California':
        fnames = os.listdir(path)
        fnames = [fn for fn in fnames if datasetName in fn]
        print(dataType, fnames)
        d = pd.read_csv(path + fnames[0], header=0, sep='\t')
        wavelengths = np.asarray([float(a) for a in d.columns.values if '.' in a])
        ID = np.logical_and(wavelengths >= 400, wavelengths <= maxWavelength)
        wavelengths = wavelengths[ID]


# global inputs that don't change with fit params
# if some default values in the paramDict needs changing, they are set here!
updateVarDict = {}

## HEREON default (like web-version)
if regionName == 'Helsinki_EnMAP':
    anap_spec440 = 0.0552
    S_md = 0.011
    lambda_0_md = 440.
    updateVarDict = {
        'A_md': anap_spec440,
        'S_md': S_md,
        'lambda_0_md': lambda_0_md,
        'S_cdom': 0.02100542
    }
    a_md_spec_res = absorption.a_xd_spec(wavelengths, anap_spec440, S_xd=S_md, lambda_0=lambda_0_md)
else:
    a_md_spec_res = absorption.a_md_spec(wavelengths=wavelengths)

## todo: replace
## oder_frankfurt:
# anap_spec440 = 0.03617
# S_xd= 0.0093

## Vortsjarv ##
# different anap_spec440 and S_xd for each spectrum!
# iN = 0
# vortsjarv = pd.read_csv("Z:\projects\ongoing\AQUATIME\sharepoint\WP2 Representative Dataset\RTM_simulations\IOPs\Case Study 3 Vörtsjärv\\vortsjarv_v0.3.csv", header=0)
# anap_spec440 = vortsjarv['anap_spec(440) [m2 g-1]'].iloc[iN] # 'anap_spec(440) [m2 g-1]'
# S_xd = vortsjarv['Snap [1/nm]'].iloc[iN]  # 'Snap [1/nm]'
# a_md_spec_res = absorption.a_xd_spec(wavelengths, anap_spec440, S_xd, lambda_0=440.)


a_bd_spec_res = absorption.a_bd_spec(wavelengths=wavelengths)
a_w_res = resampling.resample_a_w(wavelengths=wavelengths)
if AlgaeGroupType == 'Standardv2':
    a_i_spec_res = resampling.resample_a_i_spec_EnSAD(wavelengths=wavelengths) # prototype v2, PACEv2
# a_i_spec_res = None
if AlgaeGroupType == 'Standardv3':
    a_i_spec_res = resampling.resample_a_i_spec_EnSAD_Standardv3(wavelengths=wavelengths) # prototype v3, PACEv3
if AlgaeGroupType == 'NSSummerBloomsv3':
    a_i_spec_res = resampling.resample_a_i_spec_EnSAD_SummerBloomsv3(wavelengths=wavelengths) # prototype v3, PACEv3
if AlgaeGroupType == 'HEREONgold': #HEREON (- coccolithophore + goldalgae) + dinoflagellate
    a_i_spec_res = resampling.resample_a_i_spec_EnSADandGold(wavelengths=wavelengths)  # AQUATIME Vortsjarv

b_bw_res = backscattering.b_bw(wavelengths=wavelengths, fresh=False)

if AlgaeGroupType == 'Standardv2':
    b_i_spec_res = resampling.resample_b_i_spec_EnSAD(wavelengths=wavelengths) # prototype v2, PACEv2
# b_i_spec_res = None
if AlgaeGroupType == 'Standardv3':
    b_i_spec_res = resampling.resample_b_i_spec_EnSAD_Standardv3(wavelengths=wavelengths) # prototype v3, PACEv3
if AlgaeGroupType == 'NSSummerBloomsv3':
    b_i_spec_res = resampling.resample_b_i_spec_EnSAD_SummerBloomsv3(wavelengths=wavelengths) # prototype v3, PACEv3
if AlgaeGroupType == 'HEREONgold': #HEREON (- coccolithophore + goldalgae) + dinoflagellate
    b_i_spec_res = resampling.resample_b_i_spec_EnSADandGold(wavelengths=wavelengths)  # AQUATIME Vortsjarv

print(a_md_spec_res.shape, a_bd_spec_res.shape, a_i_spec_res.shape, b_i_spec_res.shape, len(wavelengths))

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


### 20250516 ###
OWTList = [ '1', '2', '3a', '3a_g', '3a_y', '3b', '4a', '4a_g', '4a_y', '4b', '5a', '5b', '6', '7']
if regionName == 'GLORIA':
    OWTList.append('nan')

if regionName == 'Helsinki_EnMAP':
    # OWTList = ['1', '2', '3', '4', '5'] # imlplified OWT for Land-AC
    OWTList.append('nan')

if regionName == 'California':
    OWTList = ['1', '2', '3', '4', '5']  # simlplified OWT for Land-AC


## DATA to Invert
if dataType == 'Sim': # MOMO simulations
    fname = os.listdir(path)
    # fname = [fn for fn in fname if 'RrsNoGlint_v0.3' in fn]
    fname = [fn for fn in fname if 'RrsNoGlint_v0.4' in fn]
    # dat = pd.read_csv(path + fname[0], sep='\t')
    print(path, fname[0])
    dat = pd.read_csv(path + fname[0], header=0)

    # wlstr = [str(int(wl)) for wl in wavelengths]
    wlstr = [str(wl) for wl in wavelengths]
    Rrs_ = dat[wlstr].values.reshape(dat[wlstr].shape[0], 1, dat[wlstr].shape[1])
    ov = OpticalVariables(Rrs=Rrs_, band=wavelengths)
    ov.run()
    # Create `owt` class to run optical classification
    this_owt = OWT(ov.AVW, ov.Area, ov.NDI, '../helper/data_OWT_Bi/OWT_centroids_refined3a_4a.nc')
    this_owt.run_classification()
    owt_result_str = np.asarray([a[0] for a in this_owt.type_str])
    dat['OWT'] = owt_result_str
elif dataType == 'SimRrs_InsituForward':
    fname = os.listdir(path)
    if regionName == 'Mueggelsee':
        fname = [fn for fn in fname if
                 fn.startswith('SimRrs_InsituForward') and 'CHIMEbands' in fn and 'muggelsee' in fn]
    else:
        fname = [fn for fn in fname if fn.startswith('SimRrs_InsituForward') and 'CHIMEbands' in fn and regionName in fn]
    dat = pd.read_csv(path + fname[0], sep='\t')
    wlstr = [str(wl) for wl in wavelengths]
    # wlstr = dat.columns.values
    datasetName = fname[0].split('.')[0]
    ## calculate OWTs!
    Rrs_ = dat[wlstr].values.reshape(dat[wlstr].shape[0], 1, dat[wlstr].shape[1])
    ov = OpticalVariables(Rrs=Rrs_, band=wavelengths)
    ov.run()
    # Create `owt` class to run optical classification
    this_owt = OWT(ov.AVW, ov.Area, ov.NDI, '../helper/data_OWT_Bi/OWT_centroids_refined3a_4a.nc')
    this_owt.run_classification()
    owt_result_str = np.asarray([a[0] for a in this_owt.type_str])
    # owt_result = this_owt.type_idx[0]
    # owt_names = [b for b in this_owt.dict_idx_name.values()]
    # owt_names = np.asarray(owt_names[1:])
    # owt_result = this_owt.type_idx
    dat['OWT'] = owt_result_str

elif dataType == 'Insitu':
    if regionName == 'vortjarv':
        ## Vortsjarv insitu spectra ##
        path = "Z:\projects\ongoing\AQUATIME\sharepoint\WP2 Representative Dataset\Case_Study_3\Vortsjarv\Hyperspectral data\\"
        fname = ['Insitu_Vorstjarv_Rrs_Hypstar_radiometric_23-24.csv']
        dat = pd.read_csv(path+ fname[0])
        wlstr = [str(wl) for wl in wavelengths]
    if regionName == 'BelgianCoastRT1':
        path = "Z:\projects\ongoing\AQUATIME\sharepoint\WP2 Representative Dataset\RTM_simulations\concistency\\"
        fname = ["Insitu_Rrs_Belgian_Coast_-_RT1.csv"]
        dat = pd.read_csv(path + fname[0], header=0)
        wlstr = np.asarray([a for a in dat.columns.values if a.startswith('wl_')])
        wl = np.asarray([float(a.split('_')[1]) for a in wlstr])
        ID = np.logical_and(wl >= 400, wl <= maxWavelength)
        wlstr = wlstr[ID]
    if regionName == 'BelgianCoastCPOWER':
        path = "Z:\projects\ongoing\AQUATIME\sharepoint\WP2 Representative Dataset\RTM_simulations\concistency\\"
        fname = ["Insitu_Rrs_Belgian_Coast_-_CPOWER.csv"]
        dat = pd.read_csv(path + fname[0], header=0)
        wlstr = np.asarray([a for a in dat.columns.values if a.startswith('wl_')])
        wl = np.asarray([float(a.split('_')[1]) for a in wlstr])
        ID = np.logical_and(wl >= 400, wl <= maxWavelength)
        wlstr = wlstr[ID]
    if regionName == 'GLORIA':
        path = "D:\Documents\projects\EnsAD\EnMAP\dask_oe_processor_2026\\"
        fname = ["ValidationGLORIA_withOWTrefined.txt"]
        dat = pd.read_csv(path + fname[0], header=0, sep='\t')
        dat['OWT'] = np.asarray([str(a) for a in dat['OWT'].values])
        print(np.unique(dat['OWT']))
        wlstr = np.asarray([a for a in dat.columns.values[:-1]])
        wl = np.asarray([float(a) for a in wlstr])
        ID = np.logical_and(wl >= 400, wl <= maxWavelength)
        wlstr = wlstr[ID]
elif dataType == 'EnMAPSuperpixel':
    if regionName == 'EnMAP_NorthSea' :
        fnames = os.listdir(path)
        fnames = [fn for fn in fnames if datasetName in fn]
        dat = pd.read_csv(path + fnames[0], header=0, sep='\t')
        if 'OWT' in dat.columns.values:
            dat['OWT'] = np.asarray([str(a) for a in dat['OWT'].values])
            print(np.unique(dat['OWT']))
        else:
            dat['OWT'] = 'nan'
        wlstr = np.asarray([a for a in dat.columns.values if '.' in a])
        wl = np.asarray([float(a) for a in wlstr])
        ID = np.logical_and(wavelengths >= 400, wavelengths <= maxWavelength)
        wlstr = wlstr[ID]
    if regionName=='Helsinki_EnMAP' or regionName == 'California':
        fnames = os.listdir(path)
        fnames = [fn for fn in fnames if datasetName in fn]
        dat = pd.read_csv(path + fnames[0], header=0, sep='\t')
        if 'OWT' in dat.columns.values:
            # dat['OWT'] = np.asarray([str(int(a)) for a in dat['OWT'].values]) # simplified OWT for Land-AC
            dat['OWT'] = np.asarray([str(a) for a in dat['OWT'].values]) #
            print(np.unique(dat['OWT']))
        else:
            dat['OWT'] = 'nan'
        wlstr = np.asarray([a for a in dat.columns.values if '.' in a])
        wl = np.asarray([float(a) for a in wlstr])
        ID = np.logical_and(wavelengths >= 400, wavelengths <= maxWavelength)
        wlstr = wlstr[ID]

## Datum, Solar Angles, G-Parameters for conversion to Rrs in Inversion
angleDependencyEnMAP = False
if angleDependencyEnMAP:
    glist = ['G0w', 'G1w', 'G0p', 'G1p']

    G_df, solz, senz, phi = lee.read_G_LUT()
    xG0w = lee.setup_xarray_Gx(G_df, solz, senz, phi, varname='G0w')
    xG1w = lee.setup_xarray_Gx(G_df, solz, senz, phi, varname='G1w')
    xG0p = lee.setup_xarray_Gx(G_df, solz, senz, phi, varname='G0p')
    xG1p = lee.setup_xarray_Gx(G_df, solz, senz, phi, varname='G1p')

    # EnMAP is fully normalised! Single geometry for all pixels!
    G = np.zeros(4)
    sun_zenith, senz, phi = (0., 0.,
                             180.)  # (50., 135.)  # (30., 135.) # (0,0) # for consistency with Pitarch et al 2025, the azimuth difference has to be transformed!
    G[0] = xG0w.interp(solz=sun_zenith, senz=senz, phi=180. - phi, method="linear").values.flatten()[0]
    G[1] = xG1w.interp(solz=sun_zenith, senz=senz, phi=180. - phi, method="linear").values.flatten()[0]
    G[2] = xG0p.interp(solz=sun_zenith, senz=senz, phi=180. - phi, method="linear").values.flatten()[0]
    G[3] = xG1p.interp(solz=sun_zenith, senz=senz, phi=180. - phi, method="linear").values.flatten()[0]

    updateVarDict['Gw0'] = G[0]
    updateVarDict['Gw1'] = G[1]
    updateVarDict['Gp0'] = G[2]
    updateVarDict['Gp1'] = G[3]


print(dat.columns.values)
r_rs_all = dat[wlstr]

# resultInvDF = pd.DataFrame()
# resultSimDF = pd.DataFrame()
resultInvArr = None
resultSimArr = None
# resultArr = None

for owt in OWTList[:]:
    print(owt)
    OWTsingleList = [owt]

    ID = dat.OWT == owt
    if np.sum(ID)>0:
        r_rs_sub = r_rs_all.loc[ID,:]
        print(r_rs_sub.shape)

        thisResultInv, thisResultSim = processor_bioOptics_hyperspectral_byOWT(
            regionName = regionName,
            AlgaeGroupType = AlgaeGroupType,
            OWTsingleList = OWTsingleList, # single OWTs only!
            outpath=None,
            r_rs=r_rs_sub,
            wavelengths=wavelengths,
            datasetName=datasetName,
            checkZeroSpectrum=False)

        if resultInvArr is None:
            resultInvArr = np.zeros((len(ID), thisResultInv.shape[1]))
        resultInvArr[ID,:] = thisResultInv.values
            # resultInvDF = thisResultInv.copy()
        # else:
        #     resultInvDF = pd.concat((resultInvDF, thisResultInv), axis=0)

        if resultSimArr is None:
            resultSimArr = np.zeros((len(ID), thisResultSim.shape[1]))
        resultSimArr[ID,:] = thisResultSim.values
        # else:
        #     resultSimDF = pd.concat((resultSimDF, thisResultSim), axis=0)

resultInvDF = pd.DataFrame(resultInvArr, columns=thisResultInv.columns.values)
if 'date' in dat.columns.values:
    resultInvDF['date'] = dat['date'].values
resultInvDF.to_csv(outpath + "inverted_IOP_bio_optics_HEREONfull_" + datasetName + "_v0.3_allOWTs_P06.txt", header=True, index=False)
resultSimDF = pd.DataFrame(resultSimArr, columns=thisResultSim.columns.values)
if 'date' in dat.columns.values:
    resultSimDF['date'] = dat['date'].values
resultSimDF.to_csv(outpath + "inverted_Sim_bio_optics_HEREONfull_" + datasetName + "_v0.3_allOWTs_P06.txt", header=True, index=False)
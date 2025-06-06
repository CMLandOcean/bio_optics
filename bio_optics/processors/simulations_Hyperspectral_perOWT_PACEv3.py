from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import scipy
from scipy.interpolate import UnivariateSpline
from scipy.interpolate import interp1d
import lmfit
import ray
import os
import timeit

from bio_optics.water import absorption, attenuation, backscattering, scattering, lee
from bio_optics.atmosphere import downwelling_irradiance
from bio_optics.models import hereon, model
from bio_optics.helper import resampling, utils, owt, indices, plotting
from bio_optics.water import fluorescence


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
    params.add('b_ratio_C_0', value=0.0058, vary=False)  # Diatoms
    params.add('b_ratio_C_1', value=0.007, vary=False)  # green
    params.add('b_ratio_C_2', value=0.0042, vary=False)  # cryptophyte
    params.add('b_ratio_C_3', value=0.0082, vary=False)  # cyano blue
    params.add('b_ratio_C_4', value=0.001, vary=False)  # cyano red
    # if AlgaeGroupType == 'NSSummerBloomsv3':
    if AlgaeGroupType == 'Summer':
        params.add('b_ratio_C_5', value=0.0034, vary=False)  # Phaeocystis: change to 0.0034
    else:
        params.add('b_ratio_C_5', value=0.0129, vary=False)  # coccolithophores , Phaeocystis: change to 0.0034
    params.add('b_ratio_C_6', value=0.0209, vary=False)  # dinoflagellates
    # if AlgaeGroupType == 'NSSummerBloomsv3':
    if AlgaeGroupType == 'Summer':
        params.add('b_ratio_C_7', value=0.0209, vary=False)  # Noctiluca: change to 0.0209
    else:
        params.add('b_ratio_C_7', value=0.0109, vary=False)  # case-1 , Noctiluca: change to 0.0209
    params.add('b_ratio_md', value=0.0216, min=0.021, max=0.3756, vary=True)  # max=0.0756
    params.add('b_ratio_bd', value=0.0216, min=0.021, max=0.3756, vary=True)  # max=0.0756
    # params.add('b_ratio_d', value=0.0216, min=0.021, max=0.3756, vary=True)
    params.add('A_md', value=13.4685e-3, vary=False)
    params.add('A_bd', value=0.3893e-3, vary=False)
    params.add('S_md', value=10.3845e-3, vary=False)
    params.add('S_bd', value=15.7621e-3, vary=False)
    params.add('S_cdom', value=0.0185, min=0.005, max=0.032, vary=True) # test Baltic: min =0.01
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


# Define wavelength range and sampling rate
wavelengths=np.arange(400,900, 1)

# Select iop-model setup!

AlgaeGroupType = 'Standard' # 'Standardv2': HEREON, 'Standardv3': 'Standard', NSSummerBloomsv3': 'Summer'


# global inputs that don't change with fit params
a_md_spec_res = absorption.a_md_spec(wavelengths=wavelengths)
a_bd_spec_res = absorption.a_bd_spec(wavelengths=wavelengths)
a_w_res = resampling.resample_a_w(wavelengths=wavelengths)
if AlgaeGroupType == 'HEREON':
    a_i_spec_res = resampling.resample_a_i_spec_EnSAD(wavelengths=wavelengths) # prototype v2, PACEv2
    b_i_spec_res = resampling.resample_b_i_spec_EnSAD(wavelengths=wavelengths) # prototype v2, PACEv2
if AlgaeGroupType == 'Standard':
    a_i_spec_res = resampling.resample_a_i_spec_EnSAD_Standardv3(wavelengths=wavelengths) # prototype v3, PACEv3
    b_i_spec_res = resampling.resample_b_i_spec_EnSAD_Standardv3(wavelengths=wavelengths) # prototype v3, PACEv3
if AlgaeGroupType == 'Summer':
    a_i_spec_res = resampling.resample_a_i_spec_EnSAD_SummerBloomsv3(wavelengths=wavelengths) # prototype v3, PACEv3
    b_i_spec_res = resampling.resample_b_i_spec_EnSAD_SummerBloomsv3(wavelengths=wavelengths) # prototype v3, PACEv3
b_bw_res = backscattering.b_bw(wavelengths=wavelengths, fresh=False)

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

## Parallelised version!

## read parameters from inversion results
# OWTList = ['1', '2',  '3b', '4a', '4b', '5a', '5b', '6', '7'] #'3a'
OWTList = ['1', '2', '3a_g', '3a_y', '3b', '4a_g', '4a_y', '4b', '5a', '5b', '6', '7']

# outpath = "E:\Documents\projects\EnsAD\data\PACE_NN_training\TrainingData_PACE_v3_comb_fullSimulation\\"
# path = "E:\Documents\projects\EnsAD\data\PACE_NN_training\TrainingData_PACE_v3_comb\\"

outpath = "E:\Documents\projects\EnsAD\data\PACE_NN_training\TrainingData_PACE_v3_OWTrefined_comb_fullSimulation\\"
path = "E:\Documents\projects\EnsAD\data\PACE_NN_training\TrainingData_PACE_v3_OWTrefined_comb\\"

fnames_ = os.listdir(path)
iopFnames = [fn for fn in fnames_ if fn.startswith('iop_')]
origFnames = [fn for fn in fnames_ if fn.startswith('orig_')]
invFnames = [fn for fn in fnames_ if fn.startswith('inv_')]

for OWT_ in OWTList[:]:
    print('OWT ', OWT_)
    fnames = [fn for fn in iopFnames if 'OWT' + OWT_ in fn]
    origFnames = [fn for fn in origFnames if 'OWT' + OWT_ in fn]
    invFnames = [fn for fn in invFnames if 'OWT' + OWT_ in fn]

    paramDF = pd.read_csv(path + fnames[0], header=0, sep='\t')
    typeL = np.unique(paramDF.viop)
    for t in typeL:
        ID = np.array(paramDF.viop.values == t)
        print(t, np.sum(ID))

    ID = np.array(paramDF.viop.values == AlgaeGroupType)
    print(np.sum(ID))

    if np.sum(ID)>0:
        paramDF = paramDF.loc[ID, :]

        paramList = []
        for v in paramDF.columns.values:
            if len(np.unique(paramDF[v])) > 1 and v != 'C_phy':
                paramList.append(v)


        @ray.remote
        def simulate_chunk(
                           paramDF_chunk,  # iops and variables from Inversion
                           wavelengths,
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
                           n2_res):

            params = set_default_parameters(AlgaeGroupType)

            for i in np.arange(paramDF_chunk.shape[0]):
                for p in paramList:
                    # Change parameters object accordingly.
                    params.add(p, value=paramDF_chunk[p].values[i])

                if i == 0:
                    R_rs_sim = hereon.forward(parameters=params,
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
                    R_rs_sim = np.vstack((R_rs_sim, hereon.forward(parameters=params,
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

            # print(R_rs_sim.shape)
            return R_rs_sim


        if paramDF.shape[0] < 300:
            num_chunks = 1
        elif paramDF.shape[0] < 1000:
            num_chunks = 3
        else:
            num_chunks = 9  # Number of chunks to split the DF into

        chunk_size = paramDF.shape[0] // num_chunks  # Size of each chunk
        chunks = [paramDF.iloc[i:i + chunk_size, :] for i in range(0, paramDF.shape[0], chunk_size)]  # Split the DF into chunks
        while chunks[-1].shape[0] == 1:
            num_chunks -= 1
            chunk_size = paramDF.shape[0] // num_chunks  # Size of each chunk
            chunks = [paramDF.iloc[i:i + chunk_size, :] for i in range(0, paramDF.shape[0], chunk_size)]  # Split the DF into chunks


        print(chunks[-1].shape[0])
        print('chunks N', len(chunks))
        ray.shutdown()

        start = timeit.default_timer()

        # Parallelize the processing of the chunks using ray
        chunk_refs = [ray.put(chunk) for chunk in chunks]  # Put the chunks into the object store
        result_refs = [simulate_chunk.remote(chunk_ref,
                                           # params=params,
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
                                           n2_res=n2_res) for chunk_ref in chunk_refs]  # Process the chunks in parallel

        results = ray.get(result_refs)

        # Concatenate the results from the processed chunks
        processed_data = np.concatenate(results)

        stop = timeit.default_timer()
        print('Time: ', stop - start)


        R_rs_sim = pd.DataFrame(processed_data, columns=wavelengths.astype(str))
        R_rs_sim.to_csv(outpath + "Rrs_" + AlgaeGroupType + "_simulation_" + fnames[0], header=True, sep='\t', index=False)
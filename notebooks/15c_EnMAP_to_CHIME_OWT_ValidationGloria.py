import os
import time
import numpy as np
import xarray as xr
import lmfit
import matplotlib.pyplot as plt
import pandas as pd
import ray
import jax.numpy as jnp
import cmocean.cm as cm
from skimage.segmentation import mark_boundaries
from xcube.core.store import new_data_store
from suncalc import get_position, get_times
from sunpy.coordinates.sun import earth_distance

from bio_optics.helper import resampling
from bio_optics.water.reflectance import bi_jax, bi_fluo, lee, bi
from bio_optics.inversion import lmfit_engine
from bio_optics.image_processing import superpixel_engine
from bio_optics.water import absorption, attenuation, backscattering
from bio_optics.water.owt import OWT, OpticalVariables, owt_qwip_prepare


def find_closest(arr: np.array, val: int):
  """ 
  Find the closest value to a number in an array.
  
  :param arr:  an array or list of numbers
  :param val:  the number  
  :return:     the closest value and its index in arr
  """
  arr = np.asarray(arr)
  idx = (np.abs(arr - val)).argmin()
  
  return arr[idx], idx

regionName='GLORIA'
sensor = 'EnMAP'
versionAC = 'EnMAPWaterv1.5.2'
dataType = 'Insitu' #'EnMAPTile' # 'Sim'
## CAUTION: only Standardv2 is implemented here!! ##
AlgaeGroupType = 'Standardv2' #'Standardv2' # 'Standardv3' # 'NSSummerBloomsv3' Standardv2, 'HEREONgold'

# maxWavelength= 750.
maxWavelength = 850.
n_iter = 1500
noise= 1.

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
    params.add('A_phy', value=0.0237, vary=False)
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
    params.add('interpolate', value=False, vary=False) # value=True
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

## GLORIA setup for Inversion
processingDict_GLORIA = {
 '4a_y' : {
'C_0': setDictValues( 12.0 , 0., 925.39 ),
'C_1': setDictValues( 12.0 , 0., 925.39 ),
'C_2': setDictValues( 12.0 , 0., 925.39 ),
'C_3': setDictValues( 12.0 , 0., 925.39 ),
'C_4': setDictValues( 12.0 , 0., 925.39 ),
'C_5': setDictValues( 12.0 , 0., 925.39 ),
'C_6': setDictValues( 12.0 , 0., 925.39 ),
'C_7': setDictValues( 12.0 , 0., 925.39 ),
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
'C_7': setDictValues( 32.25 , 0., 481.85 ),
'C_ism': setDictValues( 19.06 , 0., 1195.75 ),
'C_Y': setDictValues( 1.36 , 0., 11.21 )
},
'5a' : {
'C_0': setDictValues( 24.26 , 0., 351.72 ),
'C_1': setDictValues( 24.26 , 0., 351.72 ),
'C_2': setDictValues( 24.26 , 0., 351.72 ),
'C_3': setDictValues( 24.26 , 0., 351.72 ),
'C_4': setDictValues( 24.26 , 0., 351.72 ),
'C_5': setDictValues( 24.26 , 0., 351.72 ),
'C_6': setDictValues( 24.26 , 0., 351.72 ),
'C_7': setDictValues( 24.26 , 0., 351.72 ),
'C_ism': setDictValues( 10.75 , 0., 159.06 ),
'C_Y': setDictValues( 1.31 , 0., 11.27 )
},
'5b' : {
'C_0': setDictValues( 16.03 , 0., 323.53 ),
'C_1': setDictValues( 16.03 , 0., 323.53 ),
'C_2': setDictValues( 16.03 , 0., 323.53 ),
'C_3': setDictValues( 16.03 , 0., 323.53 ),
'C_4': setDictValues( 16.03 , 0., 323.53 ),
'C_5': setDictValues( 16.03 , 0., 323.53 ),
'C_6': setDictValues( 16.03 , 0., 323.53 ),
'C_7': setDictValues( 16.03 , 0., 323.53 ),
'C_ism': setDictValues( 10.84 , 0., 113.56 ),
'C_Y': setDictValues( 0.62 , 0., 6.29 )
},
'6' : {
'C_0': setDictValues( 41.73 , 0., 1334.63 ),
'C_1': setDictValues( 41.73 , 0., 1334.63 ),
'C_2': setDictValues( 41.73 , 0., 1334.63 ),
'C_3': setDictValues( 41.73 , 0., 1334.63 ),
'C_4': setDictValues( 41.73 , 0., 1334.63 ),
'C_5': setDictValues( 41.73 , 0., 1334.63 ),
'C_6': setDictValues( 41.73 , 0., 1334.63 ),
'C_7': setDictValues( 41.73 , 0., 1334.63 ),
'C_ism': setDictValues( 24.9 , 0., 410.04 ),
'C_Y': setDictValues( 0.87 , 0., 6.26 )
},
'7' : {
'C_0': setDictValues( 15.37 , 0., 121.99 ),
'C_1': setDictValues( 15.37 , 0., 121.99 ),
'C_2': setDictValues( 15.37 , 0., 121.99 ),
'C_3': setDictValues( 15.37 , 0., 121.99 ),
'C_4': setDictValues( 15.37 , 0., 121.99 ),
'C_5': setDictValues( 15.37 , 0., 121.99 ),
'C_6': setDictValues( 15.37 , 0., 121.99 ),
'C_7': setDictValues( 15.37 , 0., 121.99 ),
'C_ism': setDictValues( 3.55 , 0., 18.26 ),
'C_Y': setDictValues( 1.17 , 0., 6.86 )
},
'NaN' : {
'C_0': setDictValues( 2.02 , 0., 42.09 ),
'C_1': setDictValues( 2.02 , 0., 42.09 ),
'C_2': setDictValues( 2.02 , 0., 42.09 ),
'C_3': setDictValues( 2.02 , 0., 42.09 ),
'C_4': setDictValues( 2.02 , 0., 42.09 ),
'C_5': setDictValues( 2.02 , 0., 42.09 ),
'C_6': setDictValues( 2.02 , 0., 42.09 ),
'C_7': setDictValues( 2.02 , 0., 42.09 ),
'C_ism': setDictValues( 3.02 , 0., 90.08 ),
'C_Y': setDictValues( 0.15 , 0., 8.08 )
}
}

pathGLORIA = "E:\Documents\projects\EnsAD\insitu_data\GLORIA\GLORIA_Rrs_mean_Convolved_EnMAP2.csv"
# here: ending in EnMAP2.csv -> GLORIA_Rrs.csv, ending in EnMAP.csv -> GLORIA_Rrs_mean.csv
Rrs = pd.read_csv(pathGLORIA, header=0, sep='\t')
header = Rrs.columns.values

s = pd.read_csv("E:\Documents\projects\EnsAD\c2rcc_test\EnMAP_SRF\EnMAP_Spectral_Bands.txt", sep='\t', header=0)
# print(s.iloc[:, 1].values)
wlGLORIA = s.iloc[:, 1].values
ID = wlGLORIA < maxWavelength
wlGLORIA = wlGLORIA[ID]
Rrs_col = ['band_%03i' % (i + 1) for i in range(len(wlGLORIA))]

wavelengths = wlGLORIA
sp_spectra = Rrs[Rrs_col].values
valid = np.ones(Rrs.shape[0]) == 1

AVW, NDI, qwip_score = owt_qwip_prepare.calculate_QWIP_hyperspectral(Rrs=sp_spectra[valid,:], wl_=wavelengths, maxwl=700)
Area = owt_qwip_prepare.calculate_area(Rrs=sp_spectra[valid,:], band=wavelengths)
fn_centroids = "../bio_optics/water/owt/data/OWT_centroids_refined3a_4a_final.nc"
# fn_centroids = "../bio_optics/water/owt/data/OWT_centroids.nc"
owt = OWT.OWT(AVW, Area, NDI, fn_centroids)
owt.run_classification()

colorMapDict = owt.dict_idx_color
classNames = owt.dict_idx_name
owt_result_str = owt.type_str
owt_result = owt.type_idx
owt_names = [b for b in owt.dict_idx_name.values()]

membership = owt.u[0]
membSum = np.sum(membership, axis=1)
highMemb = np.sum(membership > 0.5, axis=1)
medMemb = np.sum(np.logical_and(membership > 0.2, membership < 0.5), axis=1)
lowMemb = np.sum(np.logical_and(membership > 0.01, membership < 0.2), axis=1)

outDF = pd.DataFrame()
outDF['AVW'] = AVW
outDF['Area'] = Area
outDF['NDI'] = NDI
outDF['OWT'] = owt_result[0,:]
outDF['OWTstr'] = owt_result_str[0,:]
outDF['membSum'] = membSum

def create_chunks(data):
    if data.shape[0] < 100:
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
    return chunks

angle_dependency = False

# # --- precompute --------------------------
# pre_base     = bi_jax.precompute(wavelengths)
# pre = {**pre_base}
#
# def forward_func(params, wavelengths):
#     """Adapter: lmfit_engine signature → bi_jax.forward."""
#     return np.array(bi_jax.forward(params, pre))
#     # return np.array(bi_fluo.forward(params, pre)) # TODO: does not work yet!

## --- setup bi forward ---
a_md_spec_res = absorption.a_md_spec(wavelengths=wavelengths)
a_bd_spec_res = absorption.a_bd_spec(wavelengths=wavelengths)
a_w_res = absorption.a_w(wavelengths=wavelengths)
# if AlgaeGroupType == 'Standardv2':
a_i_spec_res = resampling.resample_a_i_spec_EnSAD(wavelengths=wavelengths) # prototype v2, PACEv2
b_bw_res = backscattering.bb_w(wavelengths=wavelengths, fresh=False)

b_i_spec_res = resampling.resample_b_i_spec_EnSAD(wavelengths=wavelengths) # prototype v2, PACEv2

da_w_div_dT_res = absorption.da_w_div_dT(wavelengths=wavelengths)
# h_C_res = fluorescence.h_C_double(wavelengths=wavelengths, W=0.75)
# h_C_phycocyanin_res = fluorescence.h_C(wavelengths=wavelengths, fwhm=20, lambda_C=644)
# h_C_phycoerythrin_res =fluorescence.h_C(wavelengths=wavelengths, fwhm=20, lambda_C=573)
omega_d_lambda_0_res = attenuation.omega_d_lambda_0()

# E_0_res = resampling.resample_E_0(wavelengths=wavelengths)
# a_oz_res = resampling.resample_a_oz(wavelengths=wavelengths)
# a_ox_res = resampling.resample_a_ox(wavelengths=wavelengths)
# a_wv_res = resampling.resample_a_wv(wavelengths=wavelengths)
n2_res = resampling.resample_n(wavelengths=wavelengths)

# E_dd_res = downwelling_irradiance.E_dd(wavelengths=wavelengths)
# E_dsa_res = downwelling_irradiance.E_dsa(wavelengths=wavelengths)
# E_dsr_res = downwelling_irradiance.E_dsr(wavelengths=wavelengths)
# E_d_res = E_dd_res + E_dsa_res + E_dsr_res
def forward_func(params, wavelengths):
    return np.array(bi.forward(params,
                               wavelengths,
                               omega_d_lambda_0_res=omega_d_lambda_0_res,
                               a_md_spec_res=a_md_spec_res,
                               a_bd_spec_res=a_bd_spec_res,
                               a_w_res=a_w_res,
                               a_i_spec_res=a_i_spec_res,
                               da_w_div_dT_res=da_w_div_dT_res,
                               bb_w_res=b_bw_res,
                               bb_i_spec_res=b_i_spec_res,
                               n2_res=n2_res
                               ))


# ## test forward function
# paramDict = processingDict_GLORIA['5a']
# thisParamList = list(processingDict_GLORIA['5a'].keys())
# params = set_default_parameters(AlgaeGroupType)
# params = set_parameters_byDict(paramDict, params)
# rrs_test = forward_func(params, wavelengths)
#
# plt.plot(wavelengths, rrs_test[:], '-')
# plt.show()

## --- per OWT ###
# OWTList = [ '1', '2', '3a', '3a_g', '3a_y', '3b', '4a', '4a_g', '4a_y', '4b', '5a', '5b', '6', '7', 'NaN']
OWTList = [ '7', 'NaN']

## check all occuring varying parameters
parList = []
for owt in OWTList:
    if owt in list(processingDict_GLORIA.keys()):
        for p in processingDict_GLORIA[owt].keys():
            parList.append(p)
parList = np.unique(parList)
print(parList)

sp_results = pd.DataFrame(np.zeros((len(valid), len(parList))) + np.nan, columns = parList)

for owt in OWTList[:]:
    ID = outDF.OWTstr == owt
    print(owt, np.sum(ID))
    if np.sum(ID)>0:
        rrs_sub = sp_spectra[valid,:][ID,:]
        counts_sub = 10. #sp_counts[valid][ID]
        paramDict = processingDict_GLORIA[owt]
        thisParamList = list(processingDict_GLORIA[owt].keys())
        params = set_default_parameters(AlgaeGroupType)
        params = set_parameters_byDict(paramDict, params)

        # if angle_dependency:
        #     params.add("Gw0", value=G0w, vary=False)
        #     params.add("Gw1", value=G1w, vary=False)
        #     params.add("Gp0", value=G0p, vary=False)
        #     params.add("Gp1", value=G1p, vary=False)

        setup = lmfit_engine.build_inversion(
                        params,
                        wavelengths,
                        forward_func,
                        method='least-squares',
                        max_nfev=n_iter,
                    )

        @ray.remote
        def invert_chunk(chunk,
                         # params,
                         # method='least-squares',
                         # max_nfev=n_iter,
                         setup):

            # make a copy of actual params object to enable in-parallel mutation
            # chunk_setup = setup
            # results = np.array([None] * chunk.shape[0])
            n_spectra = chunk.shape[0]
            n_fit = len(setup.fit_names)
            results = np.full((n_spectra, n_fit), np.nan)

            for i, spectrum in enumerate(chunk):
                if not np.isfinite(spectrum).all():
                    continue
                p = setup.params.copy()
                result = lmfit_engine.invert(
                    p, spectrum, setup.wavelengths, setup.forward_func,
                    weights=setup.weights, method=setup.method, max_nfev=setup.max_nfev,
                )
                results[i] = [result.params[n].value for n in setup.fit_names]

            return results

        data = rrs_sub
        chunks = create_chunks(data)

        ray.shutdown()

        # Parallelize the processing of the chunks using ray
        chunk_refs = [ray.put(chunk) for chunk in chunks]  # Put the chunks into the object store
        result_refs = [invert_chunk.remote(chunk_ref,
                                           setup=setup
                                           ) for chunk_ref in chunk_refs]  # Process the chunks in parallel

        results = ray.get(result_refs)

        # Concatenate the results from the processed chunks
        processed_data = np.concatenate(results)
        results = processed_data

        for fn in thisParamList:
            y = np.array(sp_results[fn].values)
            i = np.where(np.asarray(setup.fit_names) == fn)[0][0]
            y[ID] = results[:, i]
            sp_results[fn] = y

        outpath = "D:\Documents\projects\EnsAD\EnMAP\dask_oe_processor_2026\FullInversion\\"
        sp_results.to_csv(outpath + "invertedIOPs_Bi_GLORIA_OWTrefined_GLORIA_withOWTNaNclass_OWT"+owt+"_b.txt", header=True,
                          index=False)

Rrs_sim_Arr = np.zeros((len(valid), len(wavelengths)))

for i in range(sp_results.shape[0]):
    if not np.all(np.isnan(sp_results.iloc[i].values)):
        for j, p in enumerate(sp_results.columns.values):
            # Change parameters object accordingly.
            # if p.startswith('C_'):
            #     print(p, np.round(paramDF[p].values[i] / invIOP['C_phy'].values[i], 2))
            if not np.isnan(sp_results[p].values[i]):
                params.add(p, value=sp_results[p].values[i])
            else:
                params.add(p, value=0.)

        R_rs_sim = forward_func(params, wavelengths)
        Rrs_sim_Arr[i,:] = R_rs_sim

## Write to file
outpath = "D:\Documents\projects\EnsAD\EnMAP\dask_oe_processor_2026\FullInversion\\"
sp_results.to_csv(outpath + "invertedIOPs_Bi_GLORIA_OWTrefined_GLORIA_withOWTNaNclass.txt", header=True, index=False)
for i,w in enumerate(wavelengths):
    outDF[str(w)] = Rrs_sim_Arr[:, i]
outDF.to_csv(outpath + "invertedRrs_Bi_GLORIA_OWTrefined_GLORIA_withOWTNaNclass.txt", header=True, index=False)

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
from suncalc import get_position, get_times
from datetime import datetime

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
    params.add('b_ratio_C_0', value=0.002, vary=False)  # Diatoms 0.0058; HEREONweb: 0.002
    params.add('b_ratio_C_1', value=0.007, vary=False)  # green 0.007, HEREONweb: 0.007
    params.add('b_ratio_C_2', value=0.002, vary=False)  # cryptophyte 0.0042, HEREONweb: 0.002
    params.add('b_ratio_C_3', value=0.003, vary=False)  # cyano blue 0.0082, HEREONweb: 0.003
    params.add('b_ratio_C_4', value=0.003, vary=False)  # cyano red 0.001, HEREONweb: 0.003
    # if AlgaeGroupType == 'NSSummerBloomsv3':
    if AlgaeGroupType == 'Summer':
        params.add('b_ratio_C_5', value=0.0034, vary=False)  # Phaeocystis: change to 0.0034
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
    params.add('b_ratio_md', value=0.0216, min=0.021, max=0.3756, vary=True)  # max=0.0756
    params.add('b_ratio_bd', value=0.0216, min=0.021, max=0.3756, vary=True)  # max=0.0756
    params.add('b_ratio_d', value=0.0216, min=0.021, max=0.3756, vary=True)
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

## Exchange columns with input
## Exchange columns with input and order them like inversion results by OWT!
# OWTList = [ '1', '2', '3a_g', '3a_y', '3b', '4a_g', '4a_y', '4b', '5a', '5b', '6', '7']
regionName = 'Mueggelsee' #'Helsinki_SS' #'uto' #'malaren' #'pyhajarvi' #'dalaro-2' #'elbe_bunthaus' #'elbe_seemannshöft' #'oder_hohenwutzen' #'Helsinki_FM' # 'Mueggelsee', 'vortsjarv' # 'oder_frankfurt' #
print(regionName)
angleDependency = False
theta_view=0.

metaDict = {
    'dalaro-2': {'project': 'AQUATIME',
        'path' : "Z:\projects\ongoing\AQUATIME\sharepoint\WP2 Representative Dataset\\RTM_simulations\Simulated Spectra\dalaro-2\\",
        'datasetName': 'CHIME_sim_AQUATIME_dalaro-2',
        'iop_path': 'Z:\projects\ongoing\AQUATIME\sharepoint\WP2 Representative Dataset\RTM_simulations\IOPs\Case Study 2 - Dalarö\\',
        'datasetNameIOP': 'dalaro_dalaro_v0.3_daily',
        'lat': 59.128,
        'lon': 18.411,
        'specNAP_variable': False,
        'PhytoGroupSet' : 'HEREONgold'
                 },
    'pyhajarvi': {'project': 'AQUATIME',
        'path' : "Z:\projects\ongoing\AQUATIME\sharepoint\WP2 Representative Dataset\\RTM_simulations\Simulated Spectra\pyhajarvi\\",
        'datasetName' :'CHIME_sim_AQUATIME_pyhajarvi',
        'iop_path': 'Z:\projects\ongoing\AQUATIME\sharepoint\WP2 Representative Dataset\RTM_simulations\IOPs\Case Study 3 pyhajarvi\\',
        'datasetNameIOP': 'pyhajarvi_continuous_v0.3',
        'lat': 61.025460552597494,
        'lon': 22.205154164905377,
        'specNAP_variable': False,
        'PhytoGroupSet' : 'HEREON'
                  },
    'elbe_bunthaus' : {'project': 'AQUATIME',
        'path': "Z:\projects\ongoing\AQUATIME\sharepoint\WP2 Representative Dataset\\RTM_simulations\Simulated Spectra\elbe_bunthaus\\",
        'datasetName' : 'CHIME_sim_AQUATIME_elbe_bunthaus',
        'iop_path': 'Z:\projects\ongoing\AQUATIME\sharepoint\WP2 Representative Dataset\\RTM_simulations\IOPs\Case Study 3 elbe_bunthaus\\',
        'datasetNameIOP': 'elbe-river_bunthaus_v0.3',
        'lat': 53.461787,
        'lon': 10.064451,
        'specNAP_variable': False,
        'PhytoGroupSet' : 'HEREON'
                       },
    'elbe_seemannshöft' :{'project': 'AQUATIME',
        'path': "Z:\projects\ongoing\AQUATIME\sharepoint\WP2 Representative Dataset\\RTM_simulations\Simulated Spectra\elbe_seemannshöft\\",
        'datasetName': 'CHIME_sim_AQUATIME_elbe-seemannshöft',
        'iop_path': 'Z:\projects\ongoing\AQUATIME\sharepoint\WP2 Representative Dataset\\RTM_simulations\IOPs\Case Study 3 elbe_seemannshöft\\',
        'datasetNameIOP': 'elbe-river_seemannshöft_v0.3',
        'lat': 53.540068,
        'lon': 9.885823,
        'specNAP_variable': False,
        'PhytoGroupSet' : 'HEREON'
                           },
    'oder_frankfurt': {'project': 'AQUATIME',
        'path': "Z:\projects\ongoing\AQUATIME\sharepoint\WP2 Representative Dataset\\RTM_simulations\Simulated Spectra\oder_frankfurt\\",
        'datasetName': 'CHIME_sim_AQUATIME_oder_frankfurt',
        'iop_path': 'Z:\projects\\ongoing\AQUATIME\\sharepoint\WP2 Representative Dataset\\RTM_simulations\IOPs\Case Study 3 Oder_Frankfurt\\',
        'datasetNameIOP': 'oder-river_frankfurt_v0.3',
        'lat': 52.357381058811875,
        'lon': 14.55155726803013,
        'specNAP_variable': False,
        'PhytoGroupSet' : 'HEREON'
                       },
    'oder_hohenwutzen': { 'project': 'AQUATIME',
        'path': "Z:\projects\ongoing\AQUATIME\sharepoint\WP2 Representative Dataset\\RTM_simulations\Simulated Spectra\oder_hohenwutzen\\",
        'datasetName': 'CHIME_sim_AQUATIME_oder_hohenwutzen',
        'iop_path': 'Z:\projects\\ongoing\AQUATIME\\sharepoint\WP2 Representative Dataset\\RTM_simulations\IOPs\Case Study 3 Oder_Hohenwutzen\\',
        'datasetNameIOP': 'oder-river_hohenwutzen_v0.3',
        'lat': 52.83612915898216,
        'lon': 14.122788786414537,
        'specNAP_variable': False,
        'PhytoGroupSet' : 'HEREON'
                         },
    'vortsjarv': {# no inversion available yet!
        'project': 'AQUATIME',
        'path' : "Z:\projects\ongoing\AQUATIME\sharepoint\WP2 Representative Dataset\\RTM_simulations\Simulated Spectra\pyhajarvi\\",
        'datasetName' :'CHIME_sim_AQUATIME_pyhajarvi',
        'iop_path': 'Z:\projects\ongoing\AQUATIME\sharepoint\WP2 Representative Dataset\\RTM_simulations\IOPs\Case Study 3 Vörtsjärv\\',
        'datasetNameIOP': 'vortsjarv',
        'lat': 58.2111,
        'lon': 26.10556,
        'specNAP_variable': True,
        'PhytoGroupSet' : 'HEREONgold'
    },
    'uto' : { 'project': 'AQUATIME',
        'path' : "Z:\projects\ongoing\AQUATIME\sharepoint\WP2 Representative Dataset\\RTM_simulations\Simulated Spectra\pyhajarvi\\",
        'datasetName' :'CHIME_sim_AQUATIME_pyhajarvi',
        'iop_path': 'Z:\projects\ongoing\AQUATIME\sharepoint\WP2 Representative Dataset\\RTM_simulations\IOPs\Case Study 4 Utö\\',
        'datasetNameIOP': 'uto_continuous_v0.3',
        'lat': 59.7788,
        'lon': 21.3558,
        'specNAP_variable': False,
        'PhytoGroupSet' : 'HEREONgold'},
    'malaren' : { 'project': 'AQUATIME',
        'path' : "Z:\projects\ongoing\AQUATIME\sharepoint\WP2 Representative Dataset\\RTM_simulations\Simulated Spectra\malaren-gorvaln\\",
        'datasetName' :'CHIME_sim_AQUATIME_malaren',
        'iop_path': 'Z:\projects\ongoing\AQUATIME\sharepoint\WP2 Representative Dataset\\RTM_simulations\IOPs\Case Study 3 Mälären\\',
        'datasetNameIOP': 'malaren_gorvaln_daily_v0.3',
        'lat': 59.4321,
        'lon': 17.76425,
        'specNAP_variable': False,
        'PhytoGroupSet' : 'HEREONgold'
    },
    'belgian_coast_RT1': { 'project': 'AQUATIME',
        'lat': 51.2464,
        'lon': 2.9193
       },
    'belgian_coast_CPOWER': { 'project': 'AQUATIME',
        'lat': 51.532,
        'lon': 2.955
    },
    'Helsinki_SS': {'project': 'HEATWISE',
        'path': "Z:\projects\ongoing\HEATWISE\sharepoint\WP2_workspace\Water Quality\\representative_dataset\helsinki_siljaserenade\\",
        'datasetName': '',
        'iop_path': "Z:\projects\ongoing\HEATWISE\sharepoint\WP2_workspace\Water Quality\Helsinki\input_simulations\\",
        'datasetNameIOP': "helsinki_silja_serenade_IOP_v0.2", # 'helsinki_silja_serenade_daily_v0.3',
        'lat': 60.15,
        'lon': 25.00,
        'specNAP_variable': False,
        'PhytoGroupSet' : 'HEREON'
    },
    'Helsinki_FM': { 'project': 'HEATWISE',
        'path': "Z:\projects\ongoing\HEATWISE\sharepoint\WP2_workspace\Water Quality\\representative_dataset\helsinki_finnmaid\\",
        'datasetName': '',
        'iop_path': "Z:\projects\ongoing\HEATWISE\sharepoint\WP2_workspace\Water Quality\Helsinki\input_simulations\\",
        'datasetNameIOP': 'helsinki_finnmaid_daily_v0.3',
        'lat': 60.19,
        'lon': 25.213,
        'specNAP_variable': False,
        'PhytoGroupSet' : 'HEREON'
    },
    'Mueggelsee': { 'project': 'HEATWISE',
        'path' : "Z:\projects\ongoing\HEATWISE\sharepoint\WP2_workspace\Water Quality\\representative_dataset\\berlin_muggelsee\\",
        'iop_path': "Z:\projects\ongoing\HEATWISE\sharepoint\WP2_workspace\Water Quality\Muggelsee - Berlin\input_simulations\\",
        'datasetNameIOP': "berlin_muggelsee_IOP_daily_v0.3.", #"berlin_muggelsee_IOP_daily_v0.3_SZA_Gx_OZA2"
        'lat': 52.4373,
        'lon': 13.6474,
        'specNAP_variable': False,
        'PhytoGroupSet' : 'HEREONgold'
        }
}


# Define wavelength range and sampling rate
# wavelengths=np.arange(400.,900., 1.)
## CHIME central wavelengths
wavelengths = np.array((404.2, 412.6, 421. , 429.4, 437.8, 446.2, 454.6, 463. , 471.4,
       479.8, 488.2, 496.6, 505. , 513.4, 521.8, 530.2, 538.6, 547. ,
       555.4, 563.8, 572.2, 580.6, 589. , 597.4, 605.8, 614.2, 622.6,
       631. , 639.4, 647.8, 656.2, 664.6, 673. , 681.4, 689.8, 698.2,
       706.6, 715. , 723.4, 731.8, 740.2, 748.6, 757. , 765.4, 773.8,
       782.2, 790.6, 799. , 807.4, 815.8, 824.2, 832.6, 841. , 849.4,
       857.8, 866.2, 874.6, 883. , 891.4, 899.8, 908.2, 916.6, 925. ,
       933.4, 941.8, 950.2, 958.6, 967. , 975.4, 983.8, 992.2))

# Select iop-model setup!
## HEREONgold: HEATWISE mueggelsee, AQUATIME vortsjarv
AlgaeGroupType = metaDict[regionName]['PhytoGroupSet'] # 'Standardv2': HEREON, 'Standardv3': 'Standard', NSSummerBloomsv3': 'Summer'
projectName =metaDict[regionName]['project']

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
if AlgaeGroupType == 'HEREONgold': #HEREON (- coccolithophore + goldalgae) + dinoflagellate
    a_i_spec_res = resampling.resample_a_i_spec_EnSADandGold(wavelengths=wavelengths)  # AQUATIME Vortsjarv, HEATWISE Mueggelsee
    b_i_spec_res = resampling.resample_b_i_spec_EnSADandGold(wavelengths=wavelengths)  # AQUATIME Vortsjarv, HEATWISE Mueggelsee
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

## Select paths by projectName ##

if angleDependency:
    if projectName == 'HEATWISE':
        outpath = "Z:\projects\\ongoing\HEATWISE\\sharepoint\WP2_workspace\Water Quality\FullInversion\\bio-optics_forward_withSZA\\"
    elif projectName == 'AQUATIME':
        outpath = "Z:\projects\\ongoing\AQUATIME\sharepoint\WP2 Representative Dataset\RTM_simulations\FullInversion\\bio-optics_forward_withSZA\\"
else:
    if projectName == 'HEATWISE':
        outpath = "Z:\projects\\ongoing\HEATWISE\\sharepoint\WP2_workspace\Water Quality\FullInversion\\bio-optics_forward\\"
    elif projectName == 'AQUATIME':
        outpath ="Z:\projects\\ongoing\AQUATIME\sharepoint\WP2 Representative Dataset\RTM_simulations\FullInversion\\bio-optics_forward\\"

if projectName == 'HEATWISE':
    path = "Z:\projects\\ongoing\HEATWISE\\sharepoint\WP2_workspace\Water Quality\FullInversion\\"
elif projectName == 'AQUATIME':
    path = "Z:\projects\\ongoing\AQUATIME\sharepoint\WP2 Representative Dataset\RTM_simulations\FullInversion\\"

# location = 'mueggelsee' # 'mueggelsee', 'helsinki' #'Helsinki'
versionInv = 'Inv0.1' # helsinki_ss: ['Inv0.2b', 'Inv0.2c', 'Inv0.2d', 'Inv0.3', 'Inv0.4'] , Mueggelsee: Inv0.4


## AQUATIME
# outpath = "Z:\projects\\ongoing\AQUATIME\sharepoint\WP2 Representative Dataset\RTM_simulations\FullInversion\\bio-optics_forward\\"
# path = "Z:\projects\\ongoing\AQUATIME\sharepoint\WP2 Representative Dataset\RTM_simulations\FullInversion\\"
# location = 'pyhajarvi' #'vortsjarv' #'oder_frankfurt'
# versionInv = 'Inv0.1'

## HERE: select an arbitrary inversion result file to set up input for the forward model!
fnames = os.listdir(path)
# iopFnames = [fn for fn in fnames if fn.startswith('inverted_IOP') and fn.endswith('allOWTs.txt') and location in fn]
##inverted_IOP_bio_optics_HEREONfull_CHIME_sim_HEATWISE_helsinki_ss_allOWTs_Inv0.2c
iopFnames = [fn for fn in fnames if fn.startswith('inverted_IOP') and fn.endswith('allOWTs_'+versionInv+'.txt') ] #and location in fn
# print(iopFnames)

# paramDF = pd.read_csv(path + iopFnames[0], header=0, sep='\t')
paramDF = pd.read_csv(path + iopFnames[0], header=0)
if 'date' in paramDF.columns.values:
    paramDF = paramDF.drop('date', axis=1)
# print(paramDF.iloc[0,:])
# print(paramDF.columns.values)
paramList = ['C_0', 'C_1', 'C_2', 'C_3', 'C_4', 'C_5', 'C_6', 'C_7', 'C_Y', 'C_ism', 'L_fl_lambda0', 'L_fl_phycocyanin',
             'L_fl_phycoerythrin', 'b_ratio_C_0', 'b_ratio_C_1', 'b_ratio_C_2', 'b_ratio_C_3', 'b_ratio_C_4',
             'b_ratio_C_5', 'b_ratio_C_6', 'b_ratio_C_7', 'b_ratio_md', 'b_ratio_bd', 'A_md', 'A_bd', 'S_md', 'S_bd',
             'S_cdom', 'C_md', 'C_bd', 'K', 'lambda_0_cdom', 'lambda_0_md', 'lambda_0_bd', 'lambda_0_c_d', 'lambda_0_phy',
             'gamma_d', 'x0', 'x1', 'x2', 'A', 'E0', 'E1', 'W', 'fwhm1', 'fwhm2', 'fwhm_phycocyanin', 'fwhm_phycoerythrin',
             'lambda_C1', 'lambda_C2', 'lambda_C_phycocyanin', 'lambda_C_phycoerythrin', 'double', 'interpolate',
             'Gw0', 'Gw1', 'Gp0', 'Gp1', 'error_method', 'theta_sun', 'theta_view', 'n1', 'n2', 'kappa_0', 'fresh',
             'T_W', 'T_W_0', 'offset', 'fit_surface']

paramDF.drop(columns = ['C_phy'], inplace=True)



# path = metaDict[regionName]['path']
specNAP_variable = metaDict[regionName]['specNAP_variable']  # Vortsjarv: True, Mueggelsee: False

### Read Insitu IOPs
iop_path = metaDict[regionName]['iop_path']
datasetNameIOP = metaDict[regionName]['datasetNameIOP']

fname = os.listdir(iop_path)
iop_insitu_fname = [fn for fn in fname if fn.startswith(datasetNameIOP) and 'v0.3' in fn][0] # Silja Serenade v0.2 # Helsinki: 'IOP' in fn and ... 'v0.2'
iop_insitu = pd.read_csv(iop_path+ iop_insitu_fname, sep=',')

if angleDependency:
    glist = ['G0w', 'G1w', 'G0p', 'G1p']
    IDcol = [a for a in glist if a in iop_insitu.columns.values]
    if len(IDcol) <= len(glist):
        dateCol = [a for a in iop_insitu.columns.values if 'date' in a or 'time' in a]
        if regionName == 'Helsinki_SS' and 'v0.2' in iop_insitu_fname:
            # iop_insitu['datetime'] = pd.to_datetime(iop_insitu['datetime_station']) # v0.1
            dateStr = np.asarray([str(t)+'T12:00:00' for t in iop_insitu[dateCol[0]]])
            iop_insitu['datetime'] = pd.to_datetime(dateStr)
        else:
            iop_insitu['datetime'] = pd.to_datetime(iop_insitu[dateCol[0]])
        print(dateCol)
        print(iop_insitu[dateCol].iloc[0])
        G_df, solz, senz, phi = lee.read_G_LUT()
        xG0w = lee.setup_xarray_Gx(G_df, solz, senz, phi, varname='G0w')
        xG1w = lee.setup_xarray_Gx(G_df, solz, senz, phi, varname='G1w')
        xG0p = lee.setup_xarray_Gx(G_df, solz, senz, phi, varname='G0p')
        xG1p = lee.setup_xarray_Gx(G_df, solz, senz, phi, varname='G1p')
        sun_dict = get_position(iop_insitu.datetime, lng=metaDict[regionName]['lon'], lat=metaDict[regionName]['lat'])
        sun_dict['zenith'] = (np.pi / 2. - sun_dict['altitude']) * 180. / np.pi
        G = np.zeros((iop_insitu.shape[0], 4))
        senz, phi = (0.,180.) # (50., 135.)  # (30., 135.) # (0,0) # for consistency with Pitarch et al 2025, the azimuth difference has to be transformed!
        G[:, 0] = xG0w.interp(solz=sun_dict['zenith'], senz=senz, phi=180.-phi, method="linear").values.flatten()
        G[:, 1] = xG1w.interp(solz=sun_dict['zenith'], senz=senz, phi=180.-phi, method="linear").values.flatten()
        G[:, 2] = xG0p.interp(solz=sun_dict['zenith'], senz=senz, phi=180.-phi, method="linear").values.flatten()
        G[:, 3] = xG1p.interp(solz=sun_dict['zenith'], senz=senz, phi=180.-phi, method="linear").values.flatten()
        iop_insitu['G0w'] = G[:, 0]
        iop_insitu['G1w'] = G[:, 1]
        iop_insitu['G0p'] = G[:, 2]
        iop_insitu['G1p'] = G[:, 3]
        iop_insitu['SZA'] = sun_dict['zenith']
        iop_insitu['OZA'] = senz
        iop_insitu['dAA'] = phi

## Vortsjarv: no inversion yet, shorten paramDF
## Muggelsee: expand the paramDF to fit the daily data!
if iop_insitu.shape[0] < paramDF.shape[0]:
    paramDF = paramDF.iloc[0:iop_insitu.shape[0],:]
if iop_insitu.shape[0] > paramDF.shape[0]:
    N1 = iop_insitu.shape[0] // paramDF.shape[0]
    N = iop_insitu.shape[0] % paramDF.shape[0]
    print(N1, N, iop_insitu.shape[0], paramDF.shape[0] )
    paramDF_ = paramDF.copy()
    for i in range(N1-1):
        paramDF = pd.concat([paramDF, paramDF_])
    if N > 0:
        paramDF = pd.concat([paramDF, paramDF_.iloc[:N,:]])
    print(paramDF.shape)

## for replacement:
IOPDict = {
    'C_0': 'Ckie [µg/l]',
    'C_1': 'Cgr [µg/l]',
    'C_2': 'Ccry [µg/l]',
    'C_3': 'Cbl [µg/l]',
    'C_4': 'Cbl2 [µg/l]',
    'C_5': 'Cgold [µg/l]',
    'C_6': 'Cdino [µg/l]',
    'C_Y': 'ag(440) [1/m]',
    'C_ism': 'NAP [mg/l]',
    'S_md': 'Snap [1/nm]',
    'S_cdom' : 'Sg [1/nm]',
    'L_fl_lambda0' : None,  # remove fluoresence
    'L_fl_phycocyanin': None # remove fluoresence
}
if angleDependency:
    angleDict = {'Gw0' : 'G0w',
    'Gw1' : 'G1w',
    'Gp0' : 'G0p',
    'Gp1' : 'G1p',
    'theta_sun': 'SZA',
    'theta_view': 'OZA'}
    for key in angleDict.keys():
        IOPDict[key] = angleDict[key]

for key in IOPDict.keys():
    if IOPDict[key] is None:
        paramDF[key] = 0.
    else:
        y = iop_insitu[IOPDict[key]].values
        ID = np.isnan(y)
        if np.sum(ID)>0:
            y[ID] =  0.
        # print(key, y.shape)
        paramDF[key] = y

## NAP: single specific absorption
if not specNAP_variable:
    anap_spec440 = np.unique(np.round(iop_insitu['anap_spec(440) [m2 g-1]'].values, 5))
    S_xd = np.unique(np.round(iop_insitu['Snap [1/nm]'].values, 5))
    offset_xd = 0.01231 # Bi & Hieronymi uses offset!
    if len(anap_spec440)==1:
        print('set NAP specific absorption')
        a_md_spec_res = absorption.a_xd_spec(wavelengths, anap_spec440, S_xd, C_xd=offset_xd, lambda_0=440.)
else:
    paramDF['anap_spec(440) [m2 g-1]'] = iop_insitu['anap_spec(440) [m2 g-1]'].values
    paramDF['Snap [1/nm]'] = iop_insitu['Snap [1/nm]'].values

# paramDF.to_csv(iop_path + 'forward_vortsjarv_input.csv', header=True, index=False, sep='\t')

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
                  da_W_div_dT_res):

    params = set_default_parameters(AlgaeGroupType)

    for i in np.arange(paramDF_chunk.shape[0]):
        for p in paramList:
            # Change parameters object accordingly.
            params.add(p, value=paramDF_chunk[p].values[i])

        ## variable aNAP_spec!
        if specNAP_variable:
            anap_spec440 = paramDF_chunk['anap_spec(440) [m2 g-1]'].values[i]
            S_xd = paramDF_chunk['Snap [1/nm]'].values[i]
            offset_xd = 0.01231 # Bi & Hieronymi uses offset!
            a_md_spec_res = absorption.a_xd_spec(wavelengths, anap_spec440, S_xd, C_xd=offset_xd, lambda_0=440.)

        if i == 0:
            # print(params)
            a_res, c_d_res, b_d_res, b_phy_res, omega_d_lambda_0_res, c_d_lambda_0_res = hereon.forward_IOPs(parameters=params,
                                      wavelengths=wavelengths,
                                      a_md_spec_res=a_md_spec_res,
                                      a_bd_spec_res=a_bd_spec_res,
                                      a_w_res=a_w_res,
                                      a_i_spec_res=a_i_spec_res,
                                      b_bw_res=b_bw_res,
                                      b_i_spec_res=b_i_spec_res,
                                      da_W_div_dT_res=da_W_div_dT_res)
        else:
            a_res_, c_d_res_, b_d_res_, b_phy_res_, omega_d_lambda_0_res_, c_d_lambda_0_res_ = hereon.forward_IOPs(parameters=params,
                                                           wavelengths=wavelengths,
                                                           a_md_spec_res=a_md_spec_res,
                                                           a_bd_spec_res=a_bd_spec_res,
                                                           a_w_res=a_w_res,
                                                           a_i_spec_res=a_i_spec_res,
                                                           b_bw_res=b_bw_res,
                                                           b_i_spec_res=b_i_spec_res,
                                                           da_W_div_dT_res=da_W_div_dT_res)
            a_res = np.vstack((a_res, a_res_))
            c_d_res = np.vstack((c_d_res, c_d_res_))
            b_d_res = np.vstack((b_d_res, b_d_res_))
            b_phy_res = np.vstack((b_phy_res, b_phy_res_))
            omega_d_lambda_0_res = np.vstack((omega_d_lambda_0_res, omega_d_lambda_0_res_))
            c_d_lambda_0_res = np.vstack((c_d_lambda_0_res, c_d_lambda_0_res_))

    # print(R_rs_sim.shape)
    return a_res, c_d_res, b_d_res, b_phy_res, omega_d_lambda_0_res, c_d_lambda_0_res


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
                                   da_W_div_dT_res=da_W_div_dT_res) for chunk_ref in chunk_refs]  # Process the chunks in parallel

results = ray.get(result_refs)

# Concatenate the results from the processed chunks
processed_data_Dict = {}
varList = ['a_res', 'c_d_res', 'b_d_res', 'b_phy_res', 'omega_d_lambda_0_res', 'c_d_lambda_0_res']
print(len(results))
for i, v in enumerate(varList):
    for j in range(len(results)):
        print(v, results[j][i].shape)
        if j == 0:
            processed_data = results[j][i]
        else:
            processed_data = np.concatenate( (processed_data, results[j][i]))

    processed_data_Dict[v] = processed_data
# processed_data = np.concatenate(results)
#
stop = timeit.default_timer()
print('Time: ', stop - start)
#

for key in processed_data_Dict.keys():
    if not 'lambda_0' in key:
        outDF = pd.DataFrame(processed_data_Dict[key], columns=wavelengths.astype(str))
    else:
        outDF = pd.DataFrame(processed_data_Dict[key])
    outDF.to_csv(outpath + "SimIOPs_"+ key +"_InsituForward_" + AlgaeGroupType + '_'+regionName+'_CHIMEbands.txt', header=True, sep='\t', index=False)

# R_rs_sim = pd.DataFrame(processed_data, columns=wavelengths.astype(str))
# # R_rs_sim.to_csv(outpath + "SimRrs_InsituForward_HEREONorig_" + AlgaeGroupType + iopFnames[0].split('.')[0]+'_'+versionInv+'.txt', header=True, sep='\t', index=False)
# if angleDependency:
#     outfname = "SimRrs_InsituForward_" + AlgaeGroupType + '_'+ iop_insitu_fname.split('.')[0] +'_'+versionInv+'_AngleDepSZA.txt'
# else:
#     outfname = "SimRrs_InsituForward_" + AlgaeGroupType + '_' + iop_insitu_fname.split('.')[0] + '_' + versionInv + '.txt'
# R_rs_sim.to_csv(outpath + outfname, header=True, sep='\t', index=False)
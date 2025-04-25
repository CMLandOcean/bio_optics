### use with env py38_keras3
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import scipy
from scipy.interpolate import UnivariateSpline
import lmfit
import ray
import timeit

from bio_optics.water import absorption, attenuation, backscattering, scattering, lee, fluorescence
from bio_optics.atmosphere import downwelling_irradiance
from bio_optics.models import hereon, model
from bio_optics.helper import resampling, utils, owt, indices, plotting
from netCDF4 import Dataset

def setup_data_forInversion(datasetName = 'NorthSeaEnMAPSouth20240511', writeSpectralAnalysis=False, outpath=''):

    if datasetName == 'NorthSeaEnMAPSouth20240511':
        d = Dataset(r'Z:\projects\ongoing\EnsAD\workspace\data\Database\\20240511_DB\EnsAD_DB_20240603.nc', 'r')

        nc_vars = [var for var in d.variables]
        print(nc_vars)

        print(np.unique(d.variables['Image'][:]))

        OWT = np.asarray([str(b) for b in d['OWT'][:]])
        nc_vars = [var for var in d.variables]
        columns = nc_vars[4:-14]

        pin = [str(a) + '_' + str(b) + '_' + c.split('_V010')[0].split('_')[-2] for a, b, c in
               zip(OWT, d['index'][:], d['Image'][:])]
        Rrs = np.zeros((len(OWT), len(columns)))
        for i, col in enumerate(columns):
            Rrs[:, i] = d[col][:]

        pin = np.asarray(pin)

        wlStand = np.asarray([float(a) for a in columns])
        Rrs_unc = np.zeros(Rrs.shape) + np.nanmax(Rrs) * 0.05

        ID = np.array(np.sum(np.isnan(Rrs), axis=1) == 0)
        Rrs = Rrs[ID, :]
        Rrs_unc = Rrs_unc[ID, :]
        pin = pin[ID]

        wavelengths = wlStand
        wavelengths = wavelengths[(wavelengths > 400) & (wavelengths < 750)]

        r_rs = pd.DataFrame(Rrs.T[(wlStand > 400) & (wlStand < 750)].T)
        print(datasetName, r_rs.shape)

        ## remove NaN OWT
        ID = np.asarray(OWT != 'NaN')
        # np.sum(ID)
        r_rs = r_rs.loc[ID, :]
        OWT = OWT[ID]
        pin = pin[ID]

        df = pd.DataFrame()
        df['OWT'] = OWT
        df['pin'] = pin

        # filter by QWIP
        AVW = r_rs.apply(owt.avw, wavelengths=wavelengths, axis=1)
        NDI = r_rs.apply(lambda x: indices.ndi(x.iloc[utils.find_closest(wavelengths, 665)[1]],
                                               x.iloc[utils.find_closest(wavelengths, 492)[1]]), axis=1)
        qwip = np.array([owt.qwip(x) for x in AVW])
        # qwip_interp = UnivariateSpline(np.sort(AVW), qwip[np.argsort(AVW)])
        df['AVW'] = AVW.values
        df['NDI'] = NDI.values
        df['qwip'] = qwip

        IDqwip = np.array(np.abs(df.qwip - df.NDI) < 0.2)
        print('QWIP: ', np.sum(IDqwip), 'of', len(IDqwip))
        r_rs = r_rs.loc[IDqwip, :]
        # pin = pin[IDqwip]
        df = df.loc[IDqwip, :]

        print(r_rs.shape)
        if writeSpectralAnalysis:
            ## write additional info: OWT, names, etc.
            df.to_csv(outpath + "datasetINFO_OWT_" + datasetName + "_filteredQWIP.txt", header=True, index=False,
                      sep='\t')

        return r_rs, wavelengths, df


def initialise_params():
    params = lmfit.Parameters()
    params.add('C_0', value=0, min=0, max=1000, vary=False)  # brown
    params.add('C_1', value=0, min=0, max=1000, vary=False)  # green
    params.add('C_2', value=0, min=0, max=1000, vary=False)  # cryptophyte
    params.add('C_3', value=0, min=0, max=1000, vary=False)  # cyano blue
    params.add('C_4', value=0, min=0, max=1000, vary=False)  # cyano red
    params.add('C_5', value=0, min=0, max=1000, vary=False)  # coccolithophores
    params.add('C_6', value=0, min=0, max=1000, vary=False)  # dinoflagellates
    params.add('C_7', value=0, min=0, max=1, vary=False)  # case-1
    params.add('C_Y', value=0.1, min=0, max=2, vary=True)
    params.add('C_ism', value=1, min=0, max=1000, vary=True)
    params.add('L_fl_lambda0', value=0, min=0, max=0.2, vary=True)
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
    params.add('g_dd', value=0.02, min=-1, max=10, vary=True)  # glint correction: direct reflection
    params.add('g_dsr', value=1 / np.pi, min=0, max=10, vary=True)  # glint correction: sky glint Rayleigh
    params.add('g_dsa', value=1 / np.pi, min=0, max=10, vary=True)  # glint correction: sky glint aerosol
    params.add('d_r', value=0, min=0, max=0.1, vary=False)
    params.add('f_dd', value=1, vary=False)
    params.add('f_ds', value=1, vary=False)
    params.add('offset', value=0, min=-0.1, max=0.1, vary=False)
    params.add('fit_surface', value=True, vary=False) # value: True starts the glint correction, vary has to be kept as False!
    return params


def inversion_main_onePhytoComp(datasetName, outpath, CHL_param='C_0', writeSpectralAnalysis=False):
    r_rs, wavelengths, df = setup_data_forInversion(datasetName)

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
    h_C_phycoerythrin_res = fluorescence.h_C(wavelengths=wavelengths, fwhm=20, lambda_C=573)
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

    params = initialise_params()
    chunk_params = params.copy()
    if CHL_param == 'C_7':
        chunk_params.add(CHL_param, value=0, min=0, max=1, vary=True)
    else:
        chunk_params.add(CHL_param, value=0, min=0, max=1000, vary=True)

    data = r_rs.values

    num_chunks = 16  # Number of chunks to split the array into

    chunk_size = r_rs.values.shape[0] // num_chunks  # Size of each chunk
    chunks = [data[i:i + chunk_size, :] for i in range(0, data.shape[0], chunk_size)]  # Split the array into chunks

    ## run inversion
    ray.shutdown()
    start = timeit.default_timer()

    # Parallelize the processing of the chunks using ray
    chunk_refs = [ray.put(chunk) for chunk in chunks]  # Put the chunks into the object store
    result_refs = [invert_chunk.remote(chunk_ref,
                                       params=chunk_params,
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

    ## simulate spectra from results
    # Get glint
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

    ### Write output / Get params
    for i in np.arange(len(results)):
        if i == 0:
            params = pd.DataFrame.from_dict(results[i].params.valuesdict().items()).set_index([0]).rename(
                columns={1: i}).T
        else:
            params = pd.concat([params,
                                pd.DataFrame.from_dict(results[i].params.valuesdict().items()).set_index([0]).rename(
                                    columns={1: i}).T])


    params['ID'] = df.pin.values
    params['C_phy'] = params[['C_0', 'C_1', 'C_2', 'C_3', 'C_4', 'C_5', 'C_6', 'C_7']].sum(axis=1)

    if chunk_params['fit_surface'].value:
        R_rs_surf = pd.DataFrame(R_rs_sim, columns=wavelengths.astype(str))
        R_rs_surf.to_csv(
            outpath + "inversion_RrsGlint_" + datasetName + "_varChl1_" + CHL_param + ".txt",
            header=True,
            sep='\t', index=False)
        params.to_csv(outpath + "inversionIOP_Glint_" + datasetName + "_varChl1_" + CHL_param + ".txt", sep='\t',
                      header=True,
                      index=False)
    else:
        R_rs_sim = pd.DataFrame(R_rs_sim, columns=wavelengths.astype(str))
        R_rs_sim.to_csv(
            outpath + "inversion_Rrs_" + datasetName + "_varChl1_" + CHL_param + ".txt",
            header=True,
            sep='\t', index=False)

        params.to_csv(outpath + "inversionIOP_"+datasetName+"_varChl1_"+CHL_param+".txt", sep='\t', header=True,
                 index=False)


def run_one_comp():
    CHL_paramList = ['C_0', 'C_1', 'C_2', 'C_3', 'C_6', 'C_7']
    outpath = r"E:\Documents\projects\EnsAD\inversion\HZG_HEREON_groups\analysis_EnMAP\singleGroupTests_20240511data\\"
    datasetName = 'NorthSeaEnMAPSouth20240511'

    # a, b, c = setup_data_forInversion(datasetName, writeSpectralAnalysis=True, outpath=outpath)

    for CHL_param in CHL_paramList[:]:
        inversion_main_onePhytoComp(datasetName, outpath, CHL_param)


def analyse_one_comp(datasetName, path, plot_OWT_varDistr=False, plot_fMAE_hist=False, check_spectra=False):
    ## North Sea EnMAP
    # 0: brown, 1: green, 2: cryptoph., 3: cyano red, 4: cyano blue, 5: coccolith., 6: dinoflagellates, 7: case-1
    CHL_paramList = ['C_0', 'C_1', 'C_2', 'C_3', 'C_6', 'C_7']
    CHL_paramList = CHL_paramList[:2]

    # path = r"E:\Documents\projects\EnsAD\\inversion\HZG_HEREON_groups\\analysis_EnMAP\singleGroupTests_20240511data\\"
    colorList = ['lightcoral', 'chocolate', 'goldenrod', 'darkkhaki', 'yellowgreen', 'cadetblue', 'orchid', 'slateblue']
    IOPResultDict = {}
    for i, CHL_param in enumerate(CHL_paramList):
        IOPResultDict[CHL_param] = {}
        # IOPResultDict[CHL_param]['fname'] = "inversionIOP_" + datasetName + "_varChl1_" + CHL_param + "_withPosGlint_md_bd_conFL.txt"
        IOPResultDict[CHL_param]['fname'] = "inversionIOP_" + datasetName + "_varChl1_" + CHL_param + ".txt"

    dDict = {}
    for key in IOPResultDict.keys():
        dDict[key] = pd.read_csv(path + IOPResultDict[key]['fname'], sep='\t', header=0)

    RrsResultDict = {}
    for i, CHL_param in enumerate(CHL_paramList):
        RrsResultDict[CHL_param] = {}
        # RrsResultDict[CHL_param]['fname'] = "inversion_Rrs_" + datasetName + "_varChl1_" + CHL_param + "_withPosGlint_md_bd_conFL.txt"
        RrsResultDict[CHL_param]['fname'] = "inversion_Rrs_" + datasetName + "_varChl1_" + CHL_param + ".txt"
        # RrsResultDict[CHL_param]['glint_fname'] = "inversion_RrsGlint_" + datasetName + "_varChl1_" + CHL_param + "_withPosGlint_md_bd_conFL.txt"
        RrsResultDict[CHL_param]['col'] = colorList[i]

    rrsDict = {}
    for key in RrsResultDict.keys():
        rrsDict[key] = pd.read_csv(path + RrsResultDict[key]['fname'], sep='\t', header=0)

    ## read dataset INFO
    # df = pd.read_csv(path + "datasetINFO_OWT_" + datasetName + "_filteredQWIP.txt", header=0, sep='\t')

    r_rs, wavelengths, df = setup_data_forInversion(datasetName)

    # Error, Residuals of spectral fit
    SpectralErrorDict = {}
    for key in RrsResultDict.keys():
        rMAE = np.zeros(r_rs.shape[0])
        sam = np.zeros(r_rs.shape[0])

        for i in range(len(rMAE)):
            rMAE[i] = np.mean(utils.compute_residual(r_rs.iloc[i].values, rrsDict[key].iloc[i, :].values, method=11))
            sam[i] = np.sum(r_rs.iloc[i].values * rrsDict[key].iloc[i, :].values) / \
                     (np.sqrt(np.sum(r_rs.iloc[i].values * r_rs.iloc[i].values)) * np.sqrt(
                         np.sum(rrsDict[key].iloc[i, :].values * rrsDict[key].iloc[i, :].values)))
            sam[i] = np.arccos(sam[i])

        sam *= 180. / np.pi
        SpectralErrorDict[key] = {}
        SpectralErrorDict[key]['rMAE'] = rMAE
        SpectralErrorDict[key]['SAM'] = sam

        # dsurf = pd.read_csv(path + RrsResultDict[key]['glint_fname'], header=0, sep='\t')
        # glintStrength = np.mean(dsurf / r_rs, axis=1)
        # SpectralErrorDict[key]['glint'] = glintStrength

    if check_spectra:
        OWTList = np.unique(df.OWT.values)
        colorList = ['lightcoral', 'chocolate', 'goldenrod', 'darkkhaki', 'yellowgreen', 'cadetblue', 'orchid',
                     'slateblue']

        for i in range(10):
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6,4))
            ax.plot(wavelengths, r_rs.iloc[i], c='#0f7f9b', label="measured")
            for j, CHL_param in enumerate(CHL_paramList):
                ax.plot(wavelengths, rrsDict[CHL_param].iloc[i], '--', color=colorList[j],
                        label=CHL_param + ' '+ str(np.round(SpectralErrorDict[CHL_param]['rMAE'][i], 3)))
            ax.legend()
            fig.tight_layout()
            plt.show()


        # for i, CHL_param in enumerate(CHL_paramList):
        #     ID1 = np.array(SpectralErrorDict[CHL_param]['rMAE'] < 0.01)
        #     ID2 = np.logical_and(np.array(dDict[CHL_param][CHL_param].values > 0.01),
        #                         dDict[CHL_param]['C_ism'].values > 0.01)
        #     for owt_ in OWTList:
        #         ID = np.logical_and(ID2, df.OWT.values == owt_)
        #         ID = np.logical_and(ID, ID1)
        #         if np.sum(ID) > 0:
        #             ax[i, 0].plot(wavelengths, r_rs[ID].T, , alpha=0.4)
        #             ax[i, 1].hist(np.log10(dDict[CHL_param]['C_ism'].values[ID]), binsNAP, color=col, alpha=0.4)
        #             ax[i, 2].hist(dDict[CHL_param]['C_Y'].values[ID], binsCDOM, color=col, alpha=0.4)
        #         ax[i, 0].set_title(CHL_param)
        #         fig.tight_layout()
        #         plt.show()


    if plot_fMAE_hist:

        ## histogram rMAE and glint strength
        Nx = 2
        bins = np.linspace(0.00, 0.06, 30)
        fig, ax = plt.subplots(nrows=1, ncols=Nx, figsize=(8, 4))
        for key in SpectralErrorDict.keys():
            ax[0].hist(SpectralErrorDict[key]['rMAE'], bins=bins, color=RrsResultDict[key]['col'], alpha=0.4, label=key)
            # ax[1].hist(SpectralErrorDict[key]['glint'],  color=RrsResultDict[key]['col'], alpha=0.4, label=key)

        ax[0].set_title('rMAE')
        ax[1].set_title('glint')
        for i in range(Nx):
            ax[i].legend()

        fig.tight_layout()
        plt.show()

    if plot_OWT_varDistr:
        ## CHL, NAP, CDOM
        OWTList = np.unique(df.OWT.values)
        colorList = ['lightcoral', 'chocolate', 'goldenrod', 'darkkhaki', 'yellowgreen', 'cadetblue', 'orchid', 'slateblue']
        binsChl = np.arange(-2, 3, 0.05)
        binsNAP = np.arange(-2, 2, 0.05)
        binsCDOM = np.arange(0, 2, 0.05)
        fig, ax = plt.subplots(nrows=len(CHL_paramList), ncols=3, figsize=(8, 1.5*len(CHL_paramList)))
        for i, CHL_param in enumerate(CHL_paramList):
            for owt_, col in zip(OWTList, colorList):
                ID = np.logical_and(np.array(dDict[CHL_param][CHL_param].values > 0.01),
                                    dDict[CHL_param]['C_ism'].values > 0.01)
                ID = np.logical_and(ID, df.OWT.values == owt_)
                if np.sum(ID) > 0:
                    ax[i, 0].hist(np.log10(dDict[CHL_param][CHL_param].values[ID]), binsChl, color=col, alpha=0.4)
                    ax[i, 1].hist(np.log10(dDict[CHL_param]['C_ism'].values[ID]), binsNAP, color=col, alpha=0.4)
                    ax[i, 2].hist(dDict[CHL_param]['C_Y'].values[ID], binsCDOM, color=col, alpha=0.4)
            ax[i, 0].set_title(CHL_param)
        fig.tight_layout()
        plt.show()


run_one_comp()
# outpath = r"E:\Documents\projects\EnsAD\inversion\HZG_HEREON_groups\analysis_EnMAP\singleGroupTests_20240511data\\"
# datasetName = 'NorthSeaEnMAPSouth20240511'
# analyse_one_comp(datasetName, outpath, check_spectra=True)

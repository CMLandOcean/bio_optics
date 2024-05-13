from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import scipy
from scipy.interpolate import UnivariateSpline
import lmfit
import ray


from bio_optics.water import absorption, attenuation, backscattering, scattering, lee
from bio_optics.atmosphere import downwelling_irradiance
from bio_optics.models import hereon, model
from bio_optics.helper import resampling, utils, owt, indices, plotting

from bio_optics.water import fluorescence

def run_debug_example(plotThis = True):
    ## Jorge: Spec Balaton
    df = pd.read_csv(r'Z:\projects\ongoing\EnsAD\workspace\data\Database\Others\SpecBal_EnMap.txt', index_col=0)

    r_rs = df.iloc[:,:56]
    wavelengths = df.columns[:56].values.astype(float)
    print(df.columns.values)

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

    weights = np.ones(len(wavelengths))

    params = lmfit.Parameters()
    params.add('C_0', value=0, min=0, max=1000, vary=True) # brown
    params.add('C_1', value=1, min=1e-10, max=1000, vary=True) # green
    params.add('C_2', value=0, min=0, max=1000, vary=True) # cryptophyte
    params.add('C_3', value=0, min=0, max=1000, vary=True) # cyano blue
    params.add('C_4', value=0, min=0, max=1000, vary=False) # cyano red
    params.add('C_5', value=0, min=0, max=1000, vary=False) # coccolithophores
    params.add('C_6', value=0, min=0, max=1000, vary=False) # dinoflagellates
    params.add('C_7', value=0, min=0, max=1000, vary=False) # case-1
    params.add('C_Y', value=0.1, min=0, max=20, vary=True)
    params.add('C_ism', value=1, min=0, max=1000, vary=True)
    params.add('L_fl_lambda0', value=0, min=0, max=0.2, vary=True)
    params.add('L_fl_phycocyanin', value=0, min=0, max=0.2, vary=True)
    params.add('L_fl_phycoerythrin', value=0, min=0, max=0.2, vary=True)
    params.add('b_ratio_C_0', value=0.002, vary=False) # brown
    params.add('b_ratio_C_1', value=0.007, vary=False) # green
    params.add('b_ratio_C_2', value=0.002, vary=False) # cryptophyte
    params.add('b_ratio_C_3', value=0.001, vary=False) # cyano blue
    params.add('b_ratio_C_4', value=0.001, vary=False) # cyano red
    params.add('b_ratio_C_5', value=0.007, vary=False) # coccolithophores
    params.add('b_ratio_C_6', value=0.007, vary=False) # dinoflagellates , chose 0.007 because of smaller cell size
    params.add('b_ratio_C_7', value=0.007, vary=False) # case-1
    params.add('b_ratio_md', value=0.0216, min=0.021, max=0.3756, vary=True) # max=0.0756
    params.add('b_ratio_bd', value=0.0216, min=0.021, max=0.3756, vary=True) # max=0.0756
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
    params.add('x2', value=-1.3390, min=-1.3390-0.0618, max=-1.3390+0.0618, vary=False)
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
    params.add('g_dd', value=0.02, min=-1, max=10, vary=True)
    params.add('g_dsr', value=1/np.pi, min=0, max=10, vary=True)
    params.add('g_dsa', value=1/np.pi, min=0, max=10, vary=True)
    params.add('d_r', value=0, min=0, max=0.1, vary=True)
    params.add('f_dd', value=1, vary=False)
    params.add('f_ds', value=1, vary=False)
    params.add('offset', value=0, min=-0.1, max=0.1, vary=False)
    params.add('fit_surface', value=False, vary=False)

    PhytoConcInsituCol = ['Diatoms', 'Green_algae', 'Cryptophyta', 'Cyano', 'CDOM', 'Turbidity']
    a = np.zeros(df.shape[0])
    for phyto in PhytoConcInsituCol[:4]:
        print(phyto, np.sum(np.isnan(df[phyto].values)))
        a += np.isnan(df[phyto].values)
    ID = np.array(a==0)
    print(np.sum(ID), len(a))

    df = df.loc[ID,:]
    r_rs = r_rs.loc[ID,:]

    # print(r_rs.shape[0])


    PhytoConcInsituCol = ['Diatoms', 'Green_algae', 'Cryptophyta', 'Cyano', 'CDOM', 'Turbidity']
    for phyto in PhytoConcInsituCol:
        print(phyto, np.sum(np.isnan(df[phyto].values)))

    out = None

    for i in range(r_rs.shape[0]):
    # print(df[PhytoConcInsituCol].iloc[i])
        chunk_params = params.copy()
        # set fixed concentration values, from table: 'Diatoms' 'Cryptophyta' 'Planktothrix' 'Green_algae' 'Ratio' 'Chl_a' 'Cyano'
        chunk_params.add('C_0', value=df['Diatoms'].values[i], vary=False) # brown
        chunk_params.add('C_1', value=df['Green_algae'].values[i], vary=False) # green
        chunk_params.add('C_2', value=df['Cryptophyta'].values[i], vary=False) # cryptophyte
        chunk_params.add('C_3', value=df['Cyano'].values[i], vary=False) # cyano blue
        # chunk_params.add('C_Y', value=df['CDOM'].values[i], vary=False) # CDOM
        FreeParamList = [a for a in chunk_params.keys() if chunk_params[a].vary]
        inv = hereon.invert(chunk_params,
                            R_rs=r_rs.iloc[i,:],
                            Ls_Ed = [],
                            wavelengths=wavelengths,
                            weights=weights,
                            a_md_spec_res = a_md_spec_res,
                            a_bd_spec_res = a_bd_spec_res,
                            a_w_res = a_w_res,
                            a_i_spec_res = a_i_spec_res,
                            b_bw_res = b_bw_res,
                            b_i_spec_res = b_i_spec_res,
                            h_C_res = h_C_res,
                            h_C_phycocyanin_res=h_C_phycocyanin_res,
                            h_C_phycoerythrin_res=h_C_phycoerythrin_res,
                            da_W_div_dT_res = da_W_div_dT_res,
                            E_0_res = E_0_res,
                            a_oz_res = a_oz_res,
                            a_ox_res = a_ox_res,
                            a_wv_res = a_wv_res,
                            E_dd_res = E_dd_res,
                            E_dsa_res = E_dsa_res,
                            E_dsr_res = E_dsr_res,
                            E_d_res = E_d_res,
                            n2_res = n2_res,
                            method="least_squares",
                            max_nfev=1500)

        if out is None:
            out = np.zeros((r_rs.shape[0], len(FreeParamList)))
        for j, p in enumerate(FreeParamList):
            # print(p, np.round(inv.params[p].value,4))
            out[i, j] = inv.params[p].value

        if plotThis:
            plt.plot(wavelengths, r_rs.iloc[i], c='#0f7f9b', label="measured")
            plt.plot(wavelengths, hereon.forward(parameters=inv.params,
                                                 wavelengths=wavelengths,
                                                 a_md_spec_res = a_md_spec_res,
                                                 a_bd_spec_res = a_bd_spec_res,
                                                 a_w_res = a_w_res,
                                                 a_i_spec_res = a_i_spec_res,
                                                 b_bw_res = b_bw_res,
                                                 b_i_spec_res = b_i_spec_res,
                                                 h_C_res=h_C_res,
                                                 h_C_phycocyanin_res = h_C_phycocyanin_res,
                                                 h_C_phycoerythrin_res = h_C_phycoerythrin_res,
                                                 da_W_div_dT_res = da_W_div_dT_res,
                                                 E_0_res = E_0_res,
                                                 a_oz_res = a_oz_res,
                                                 a_ox_res = a_ox_res,
                                                 a_wv_res = a_wv_res,
                                                 E_dd_res = E_dd_res,
                                                 E_dsa_res = E_dsa_res,
                                                 E_dsr_res = E_dsr_res,
                                                 E_d_res = E_d_res,
                                                 n2_res = n2_res,
                                                 Ls_Ed=[]), '--', c='red', label='modeled')
            # if results[i].params['fit_surface'].value:
                # plt.plot(wavelengths, R_rs_surf, c='skyblue', label="glint")
                # plt.plot(wavelengths, r_rs.iloc[i] - R_rs_surf, c='darkblue', label="measured - glint")
            plt.xlabel('$\lambda$ [nm]')
            plt.ylabel('$\mathrm{r_{rs}} \/ [\mathrm{sr}^{-1}]$')
            plt.hlines(0,300,1000, color='black', linewidth=0.5) #, linestyle='dotted')
            plt.xlim(np.min(wavelengths)-10,np.max(wavelengths)+10)
            # plt.ticklabel_format(style='scientific')
            plt.legend()
            plt.show()

def debug_negative_glint_contribution(plotThis=True):
    ## Jorge: Spec Balaton
    df = pd.read_csv(r'Z:\projects\ongoing\EnsAD\workspace\data\Database\Others\SpecBal_EnMap.txt', index_col=0)

    r_rs = df.iloc[:, :56]
    wavelengths = df.columns[:56].values.astype(float)
    print(df.columns.values)

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

    params = lmfit.Parameters()
    params.add('C_0', value=0, min=0, max=1000, vary=True)  # brown
    params.add('C_1', value=1, min=1e-10, max=1000, vary=True)  # green
    params.add('C_2', value=0, min=0, max=1000, vary=True)  # cryptophyte
    params.add('C_3', value=0, min=0, max=1000, vary=True)  # cyano blue
    params.add('C_4', value=0, min=0, max=1000, vary=False)  # cyano red
    params.add('C_5', value=0, min=0, max=1000, vary=False)  # coccolithophores
    params.add('C_6', value=0, min=0, max=1000, vary=False)  # dinoflagellates
    params.add('C_7', value=0, min=0, max=1000, vary=False)  # case-1
    params.add('C_Y', value=0.1, min=0, max=20, vary=True)
    params.add('C_ism', value=1, min=0, max=1000, vary=True)
    params.add('L_fl_lambda0', value=0, min=0, max=0.2, vary=True)
    params.add('L_fl_phycocyanin', value=0, min=0, max=0.2, vary=True)
    params.add('L_fl_phycoerythrin', value=0, min=0, max=0.2, vary=True)
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
    params.add('g_dd', value=0.02, min=-1, max=10, vary=True)
    params.add('g_dsr', value=1 / np.pi, min=0, max=10, vary=True)
    params.add('g_dsa', value=1 / np.pi, min=0, max=10, vary=True)
    params.add('d_r', value=0, min=0, max=0.1, vary=True)
    params.add('f_dd', value=1, vary=False)
    params.add('f_ds', value=1, vary=False)
    params.add('offset', value=0, min=-0.1, max=0.1, vary=False)
    params.add('fit_surface', value=True, vary=True)

    PhytoConcInsituCol = ['Diatoms', 'Green_algae', 'Cryptophyta', 'Cyano', 'CDOM', 'Turbidity']
    a = np.zeros(df.shape[0])
    for phyto in PhytoConcInsituCol[:4]:
        print(phyto, np.sum(np.isnan(df[phyto].values)))
        a += np.isnan(df[phyto].values)
    ID = np.array(a == 0)
    print(np.sum(ID), len(a))

    df = df.loc[ID, :]
    r_rs = r_rs.loc[ID, :]

    # print(r_rs.shape[0])

    PhytoConcInsituCol = ['Diatoms', 'Green_algae', 'Cryptophyta', 'Cyano', 'CDOM', 'Turbidity']
    for phyto in PhytoConcInsituCol:
        print(phyto, np.sum(np.isnan(df[phyto].values)))

    out = None
    # NList = [50, 51, 53, 54, 57]
    for i in range(r_rs.shape[0]):
        print(i)
        # print(df[PhytoConcInsituCol].iloc[i])
        chunk_params = params.copy()
        # set fixed concentration values, from table: 'Diatoms' 'Cryptophyta' 'Planktothrix' 'Green_algae' 'Ratio' 'Chl_a' 'Cyano'
        chunk_params.add('C_0', value=df['Diatoms'].values[i], vary=False)  # brown
        chunk_params.add('C_1', value=df['Green_algae'].values[i], vary=False)  # green
        chunk_params.add('C_2', value=df['Cryptophyta'].values[i], vary=False)  # cryptophyte
        chunk_params.add('C_3', value=df['Cyano'].values[i], vary=False)  # cyano blue
        # chunk_params.add('C_Y', value=df['CDOM'].values[i], vary=False) # CDOM
        FreeParamList = [a for a in chunk_params.keys() if chunk_params[a].vary]
        inv = hereon.invert(chunk_params,
                            R_rs=r_rs.iloc[i, :],
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
                            method="least_squares",
                            max_nfev=1500)

        if out is None:
            out = np.zeros((r_rs.shape[0], len(FreeParamList)))
        for j, p in enumerate(FreeParamList):
            # print(p, np.round(inv.params[p].value,4))
            out[i, j] = inv.params[p].value

        R_rs_surf = model.forward_glint(wavelengths=wavelengths,
                                        parameters=inv.params,
                                        E_d_res=E_d_res,
                                        E_dd_res=E_dd_res,
                                        E_dsa_res=E_dsa_res,
                                        E_dsr_res=E_dsr_res,
                                        n2_res=n2_res,
                                        Ls_Ed=[])


        if np.any(R_rs_surf<0):

        # if plotThis:
            plt.plot(wavelengths, r_rs.iloc[i], c='#0f7f9b', label="measured")
            plt.plot(wavelengths, hereon.forward(parameters=inv.params,
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
                                                 Ls_Ed=[]), '--', c='red', label='modeled')

            plt.plot(wavelengths, R_rs_surf, '-', c='lightblue', label='glint')

            # if results[i].params['fit_surface'].value:
            # plt.plot(wavelengths, R_rs_surf, c='skyblue', label="glint")
            # plt.plot(wavelengths, r_rs.iloc[i] - R_rs_surf, c='darkblue', label="measured - glint")
            plt.xlabel('$\lambda$ [nm]')
            plt.ylabel('$\mathrm{r_{rs}} \/ [\mathrm{sr}^{-1}]$')
            plt.hlines(0, 300, 1000, color='black', linewidth=0.5)  # , linestyle='dotted')
            plt.xlim(np.min(wavelengths) - 10, np.max(wavelengths) + 10)
            # plt.ticklabel_format(style='scientific')
            plt.legend()
            plt.show()




##
# run_debug_example()
debug_negative_glint_contribution()
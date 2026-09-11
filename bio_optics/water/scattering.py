import numpy as np
from .. helper import resampling


def b(a,c):
    """
    Compute scattering coefficients from absorption and attenuation coefficients.

    Args:
        a (np.array): Absorption coefficient.
        c (np.array): Attenuation coefficient.

    Returns:
        b: Scattering coefficient.
    """
    b = c - a
    return b


def b_phy(C_0 = 0,
          C_1 = 0,
          C_2 = 0,
          C_3 = 0,
          C_4 = 0,
          C_5 = 0,
          C_6 = 0,
          C_7 = 0,
          wavelengths = np.arange(400,800),
          b_i_spec_res = []):
    """
    Spectral scattering coefficient of phytoplankton for a mixture of up to 6 phytoplankton classes (C_0..C_5).
    
    :param C_0: concentration of phytoplankton type 0 [ug/L], default: 0
    :param C_1: concentration of phytoplankton type 1 [ug/L], default: 0
    :param C_2: concentration of phytoplankton type 2 [ug/L], default: 0
    :param C_3: concentration of phytoplankton type 3 [ug/L], default: 0
    :param C_4: concentration of phytoplankton type 4 [ug/L], default: 0
    :param C_5: concentration of phytoplankton type 5 [ug/L], default: 0
    :param C_6: concentration of phytoplankton type 6 [ug/L], default: 0
    :param C_7: concentration of phytoplankton type 7 [ug/L], default: 0
    :wavelengths: wavelengths to compute a_ph for [nm], default: np.arange(400,800)
    :param b_i_spec_res: optional, preresampling b_i_spec (scattering coefficient of phytoplankton types C_0..C_7) before inversion saves a lot of time.
    :return: spectral scattering coefficient of phytoplankton mixture
    """
    C_i = np.array([C_0,C_1,C_2,C_3,C_4,C_5,C_6,C_7])
    # gamma_i = np.array([0.8943513, 0.8938004, 0.8938008, 0.8938008, 0.89379981, 0.8938001, 0.893800, 0.8937999])
    gamma_i = np.asarray([0.8952, 0.8952, 0.8800, 0.9000, 0.9000,  0.8800, 0.8952, 0.8485])  # albedo_ph676
    
    if len(b_i_spec_res)==0:
        b_i_spec = resampling.resample_b_i_spec_EnSAD(wavelengths=wavelengths)
    else:
        b_i_spec = b_i_spec_res
    
    b_phy = 0
    # for i in range(b_i_spec.shape[1]):
    #     if C_i[i] <=1:
    #         b_phy += C_i[i] * b_i_spec[:, i]
    #     else: # from HEREON web implementation
    #         b_phy += C_i[i]**gamma_i[i] * b_i_spec[:, i]

    ## from bi model repository:
    frac_mat = C_i / np.sum(C_i)
    bphs_sum = 0
    gamma_ = 0
    for i in range(len(frac_mat)):
        bphs_sum += frac_mat[i] * b_i_spec[:, i]
        gamma_ += frac_mat[i] * gamma_i[i]
    sumC = np.sum(C_i)
    if sumC <=1:
        b_phy = sumC * bphs_sum
    else:
        b_phy = sumC**gamma_ * bphs_sum

    return b_phy


# def b_phy_web(C_0 = 0,
#           C_1 = 0,
#           C_2 = 0,
#           C_3 = 0,
#           C_4 = 0,
#           C_5 = 0,
#           C_6 = 0,
#           C_7 = 0,
#           a_phy_spec = [], # corrected a_phy
#           wavelengths = np.arange(400,800),
#               albedo_ph676=[0.8952, 0.8952, 0.8800, 0.9000, 0.9000, 0.9562, 0.8485],
#               gamma_ph676=[np.nan, 0.00, 0.00, 0.00, 0.00, np.nan, np.nan] ):
#
#     frac_C =
#
#     def generate_aphs(Chl,
#                       albedo_ph676=[0.8952, 0.8952, 0.8800, 0.9000, 0.9000, 0.9562, 0.8485],
#                       gamma_ph676=[np.nan, 0.00, 0.00, 0.00, 0.00, np.nan, np.nan],
#                       vary_cph=False,
#                       aphs_fun=None,
#                       phytodive_iop_list=None,
#                       a_frac=[1, 1, 1, 1, 1, 1, 1],
#                       **kwargs):
#
#         # Check albedo_ph676 bounds
#         if any((np.array(albedo_ph676) > 1)) or any((np.array(albedo_ph676) < 0)):
#             raise ValueError("albedo of phytoplankton at 676 nm should between 0 and 1")
#
#         # Assign names from phytodive_iop_list$name_phyto keys if available
#         if phytodive_iop_list is not None and 'name_phyto' in phytodive_iop_list:
#             names_albedo = list(phytodive_iop_list['name_phyto'].keys())
#         else:
#             names_albedo = [f'phyto_{i}' for i in range(len(albedo_ph676))]
#
#         albedo_ph676_dict = dict(zip(names_albedo, albedo_ph676))
#
#         # if vary_cph:
#         #     albedo_ph676_new = []
#         #     for i, val in enumerate(albedo_ph676):
#         #         albedo_ph676_new.append(rnorm_bound(1, mean=val, sd=0.03, lo=None, up=0.99)[0])
#         #     albedo_ph676 = albedo_ph676_new
#         #     albedo_ph676_dict = dict(zip(names_albedo, albedo_ph676))
#
#         # Check gamma_ph676 bounds
#         gamma_arr = np.array([g if not np.isnan(g) else np.nan for g in gamma_ph676])
#         if any(gamma_arr[~np.isnan(gamma_arr)] < -1):
#             raise ValueError("Gamma of phytoplankton attenuation normalized at 676 nm should > -1")
#
#         gamma_ph676_dict = dict(zip(names_albedo, gamma_ph676))
#
#         # if vary_cph:
#         #     gamma_ph676_new = gamma_ph676.copy()
#         #     for i, val in enumerate(gamma_ph676_new):
#         #         if np.isnan(val):
#         #             continue
#         #         else:
#         #             gamma_ph676_new[i] = rnorm_bound(1, mean=val, sd=0.01, lo=0.0, up=0.8)[0]
#         #     gamma_ph676 = gamma_ph676_new
#         #     gamma_ph676_dict = dict(zip(names_albedo, gamma_ph676))
#
#         # Determine aph676 from the input function
#         # aph676 = aphs_fun(Chl, **kwargs)
#         #
#         # if hasattr(aph676, '__class__') and "Hereon_aphs_func" in aph676.__class__.__name__:
#         #     attr_aph676 = getattr(aph676, '__dict__', {})
#         # else:
#         #     attr_aph676 = [aphs_fun]
#         #
#         # aphs_676_new = aph676 / Chl if Chl != 0 else 1
#         #
#         # aphs_mat = phytodive_iop_list['aphs'].iloc[:, 1:].to_numpy()
#         # aphs_676 = phytodive_iop_list['aphs'].loc[phytodive_iop_list['aphs']['wv'] == 676].iloc[:, 1:].values.flatten()
#         # aphs_mat_norm = aphs_mat / vec_to_mat(aphs_676, n=aphs_mat.shape[0])
#         # aphs_mat_new = aphs_mat_norm * aphs_676_new
#
#         cphs_mat = phytodive_iop_list['cphs'].iloc[:, 1:].to_numpy()
#         cphs_676 = phytodive_iop_list['cphs'].loc[phytodive_iop_list['cphs']['wv'] == 676].iloc[:, 1:].values.flatten()
#         cphs_mat_norm = cphs_mat / vec_to_mat(cphs_676, n=cphs_mat.shape[0], by=1)
#
#         PG_TBC = [i for i, val in enumerate(gamma_ph676) if not np.isnan(val)]
#         for i in PG_TBC:
#             cphs_mat_norm[:, i] = (676 / phytodive_iop_list['cphs']['wv'].to_numpy()) ** gamma_ph676[i]
#
#         coef_lin = aphs_676_new / (1 - np.array(albedo_ph676))
#         cphs_mat_new = cphs_mat_norm * vec_to_mat(coef_lin, n=cphs_mat.shape[0], by=1)
#         bphs_mat_new = cphs_mat_new - aphs_mat_new
#
#         aphs_mat_new = aphs_mat_new * vec_to_mat(a_frac, n=aphs_mat_new.shape[0], by=1)
#         cphs_mat_new = aphs_mat_new + bphs_mat_new
#
#         bphs = pd.DataFrame(np.column_stack((phytodive_iop_list['bphs']['wv'], bphs_mat_new)),
#                             columns=phytodive_iop_list['bphs'].columns)
#
#
#         return r
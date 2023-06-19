# -*- coding: utf-8 -*-
#  Copyright 2023 
#  Center for Global Discovery and Conservation Science, Arizona State University
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
#
# Translated to Python by:
#  Marcel König, mkoenig3 AT asu.edu 


import numpy as np
from scipy.signal import savgol_filter
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, r2_score
from . import distance


def find_closest(arr: np.array or list, val: int):  
  """ 
  Find the closest value to a number in an array.
  
  :param arr:  an array or list of numbers
  :param val:  the number  
  :return:     the closest value and its index in arr
  """
  arr = np.asarray(arr)
  idx = (np.abs(arr - val)).argmin()
  
  return arr[idx], idx


def band_mask(wavelengths, mask_regions = [[1300,1500],[1800,2000]]):
    """
    Create a band mask for selected regions (default: water vapor regions between 1300-1500 nm an 1800-2000 nm.
    
    :param wavelengths: np.array of wavelengths
    :param bad_regions: list of lists containing the boundaries of the mask regions, default: [[1300,1500],[1800,2000]]
    """
    good_bands_mask = np.ones(wavelengths.shape, dtype=bool)
    for region in mask_regions:
        good_bands_mask = good_bands_mask * ~((wavelengths >= region[0]) & (wavelengths <= region[1]))
        
    return good_bands_mask


def estimate_y(R_rs, wavelengths, lambda1=444., lambda2=555., a=2.0, b=1.0, c=1.2, d=-0.9):
    """
    Two-band estimation of spectral shape parameter of particulate backscattering using an empirical relationship [1,2,3] (named Y in [2]).
    The ratio of the two bands is named chi in [2] and can be in units of R_rs or (subsurface) r_rs.
    Default for lambda1 and lambda2, and coefficients a, b, c, d are from Li et al. (2017) [4].
    For lambda1 = 440 and lambda2 = 490 Lee et al. propose a=3.44, b=1, c=3.17, d=-2.01 [2].
    
    
    [1] Lee et al. (1996): Estimating primary production at depth from remote sensing [10.1364/AO.35.000463]
    [2] Lee et al. (1999): Hyperspectral remote sensing for shallow waters: 2 Deriving bottom depths and water properties by optimization [10.1364/ao.38.003831]
    [3] Lee et al. (2002): Deriving inherent optical properties from water color: A multiband quasi-analytical algorithm for optically deep waters [10.1364/ao.41.005755]
    [4] Li et al. (2017): Remote sensing estimation of colored dissolved organic matter (CDOM) in optically shallow waters [10.1016/j.isprsjprs.2017.03.015])]

    Args:
        R_rs (_type_): Remote sensing reflectance [sr-1]
        wavelengths (_type_): corresponding wavelengths [nm]
        lambda1 (float, optional): wavelength of first band [nm]. Defaults to 444.
        lambda2 (float, optional): wavelength of second band [nm]. Defaults to 555.
        a (float, optional): Defaults to 2.
        b (float, optional): Defaults to 1.
        c (float, optional): Defaults to 1.2.
        d (float, optional): Defaults to -0.9.

    Returns:
        y: spectral shape paramter for particulate backscattering coefficient
    """
    y = a * (b - c*np.exp(d*(R_rs[find_closest(wavelengths, lambda1)[1]] / R_rs[find_closest(wavelengths, lambda2)[1]])))
    
    return y


def compute_residual(y_true, y_pred, method=2, weights=[]):
    """
    Residual computation for comparison of measured and simulated data.

    Args:
        y_true (_type_): array of true values
        y_pred (_type_): array of predicted or simulated values
        method (int, optional): Defaults to 2.
        weights (list, optional): element-wise weighting factors, 1 for each element if not provided. Defaults to [].
    Returns:
        residual
    """

    if len(weights)==0:
        weights = np.ones(len(y_true))
    
    if method == 1:
        # element-wise least squares
        return (y_pred-y_true)**2 * weights
    elif method == 2:
        # element-wise absolute differences
        return np.abs(y_pred-y_true) * weights
    elif method == 3:
        # element-wise relative differences
        return np.abs(1 - y_pred/y_true) * weights
    elif method == 4:
        # the one described in Li et al. (2017) [10.1016/j.isprsjprs.2017.03.015] but element-wise
        return np.sqrt((y_true - y_pred)**2) / np.sqrt(y_true) * weights
    elif method == 5:
        # absolute percentage difference according to Barnes et al. (2018) [10.1016/j.rse.2017.10.013] but element-wise
        return np.sqrt((y_true - y_pred)**2) / y_true * weights
    elif method == 6:
        # element-wise least squares on spectral derivatives after Petit et al. (2017) [10.1016/j.rse.2017.01.004]
        return (savgol_filter(y_pred, window_length=7, polyorder=3, deriv=1) - savgol_filter(y_true, window_length=7, polyorder=3, deriv=1))**2 
    elif method == 7:
        # summed least squared according to Groetsch et al. (2016) [10.1364/OE.25.00A742]
        return np.sum((y_pred - y_true)**2)
    elif method == 8:
        # summed absolute difference
        return np.sum(np.abs(y_pred-y_true) * weights)
    elif method == 9:
        # summed relative differences
        return np.sum(np.abs(1 - y_pred/y_true) * weights)
    elif method == 10:
        # the one described in Li et al. (2017) [10.1016/j.isprsjprs.2017.03.015]
        return np.sqrt(np.sum((y_true - y_pred)**2)) / np.sqrt(np.sum(y_true)) * weights
    elif method == 11:
        # absolute percentage difference according to Barnes et al. (2018) [10.1016/j.rse.2017.10.013]
        return np.sqrt(np.sum((y_true - y_pred)**2)) / np.sum(y_true) * weights
    elif method == 12:
        # RMSE
        return np.sqrt(mean_squared_error(y_true, y_pred, sample_weight=weights))
    elif method == 13:
        # 1 - absolute Pearson r
        return 1 - np.abs(pearsonr(y_true, y_pred)[0])
    elif method == 14:
        # 1 - R2-score
        return 1 - r2_score(y_true, y_pred)
    elif method == 15:
        # SAM
        return distance.spectral_angle(y_true, y_pred)
    elif method == 16:
        # SID
        return distance.spectral_information_divergence(y_true, y_pred)
    elif method == 17:
        # Chebyshev
        return distance.chebyshev_distance(y_true, y_pred) 
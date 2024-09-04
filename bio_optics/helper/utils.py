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
#  Marcel König, mkoenig3 AT asu.edu / marcel.koenig AT brockmann-consult.de
#
# bio_optics
#  This code base builds on the extensive work of many researchers. For example, models were developed by Albert & Mobley [1] and Gege [2]; 
#  and the methodology was mainly developed by Gege [3,4,5] and Albert & Gege [6]. Please give proper attribution when using this code for publication.
#  A former version of this code base was developed in the course of the CarbonMapper Land and Ocean Program [7]
#
#  When using this code, please use the following citation:
#
#  König, M., Noel, P., Hondula. K.L., Jamalinia, E., Dai, J., Vaughn, N.R., Asner, G.P. (2023): bio_optics python package (Version x) [Software]. Available from https://github.com/CMLandOcean/bio_optics
#
# [1] Albert & Mobley (2003): An analytical model for subsurface irradiance and remote sensing reflectance in deep and shallow case-2 waters. [10.1364/OE.11.002873]
# [2] Gege (2012): Analytic model for the direct and diffuse components of downwelling spectral irradiance in water. [10.1364/AO.51.001407]
# [3] Gege (2004): The water color simulator WASI: an integrating software tool for analysis and simulation of optical in situ spectra. [10.1016/j.cageo.2004.03.005]
# [4] Gege (2014): WASI-2D: A software tool for regionally optimized analysis of imaging spectrometer data from deep and shallow waters. [10.1016/j.cageo.2013.07.022]
# [5] Gege (2021): The Water Colour Simulator WASI. User manual for WASI version 6. 
# [6] Gege & Albert (2006): A Tool for Inverse Modeling of Spectral Measurements in Deep and Shallow Waters. [10.1007/1-4020-3968-9_4]
# [7] König et al. (2023): WaterQuality python package (Version 1.2.0) [Software]. Available from https://github.com/CMLandOcean/WaterQuality. [10.5281/zenodo.7967294]


import numpy as np
from scipy.signal import savgol_filter
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, r2_score
from . import distance
from pysolar.solar import get_altitude
import warnings


def find_closest(arr: np.array, val: int, threshold=10):  
  """ 
  Find the closest value to a number in an array.
  
  :param arr:  an array or list of numbers
  :param val:  the number  
  :param threshold: threshold that needs to be breached to send a WARNING message. Defaults to 10.
  :return:     the closest value and its index in arr
  """
  arr = np.asarray(arr)
  distance = (np.abs(arr - val)).min()
  idx = (np.abs(arr - val)).argmin()
  closest_val = arr[idx]

  if distance > threshold:
      warnings.warn("The distance of " + str(val) + " to the closest value ("+str(closest_val)+") is larger than the threshold of " + str(threshold) + '.')
  
  return closest_val, idx


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


def estimate_y(Rrs, wavelengths, lambda1=444., lambda2=555., a=2.0, b=1.0, c=1.2, d=-0.9):
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
        Rrs (_type_): Remote sensing reflectance [sr-1]
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
    y = a * (b - c*np.exp(d*(Rrs[find_closest(wavelengths, lambda1)[1]] / Rrs[find_closest(wavelengths, lambda2)[1]])))
    
    return y


def estimate_S_dg(Rrs, wavelengths, lambda1=443., lambda2=555., a=0.015, b=0.002, c=0.6, d=-1.0):
    """
    Two-band estimation of spectral shape parameter of CDOM and NAP absorption [1,2] (named S_dg in [2]) when using the exponental approximation.
    Default for lambda1 and lambda2, and coefficients a, b, c, d are from Erickson et al. (2023) [2].
    
    
    [1] Lee et al. (2002): Deriving inherent optical properties from water color: A multiband quasi-analytical algorithm for optically deep waters [10.1364/ao.41.005755]
    [4] Erickson et al. (2023): Bayesian approach to a generalized inherent optical property model [10.1364/oe.486581]

    Args:
        Rrs (_type_): Remote sensing reflectance [sr-1]
        wavelengths (_type_): corresponding wavelengths [nm]
        lambda1 (float, optional): wavelength of first band [nm]. Defaults to 443.
        lambda2 (float, optional): wavelength of second band [nm]. Defaults to 555.
        a (float, optional): Defaults to 0.015.
        b (float, optional): Defaults to 0.002.
        c (float, optional): Defaults to 0.6.
        d (float, optional): Defaults to -1.0.

    Returns:
        S_dg: spectral shape paramter for absorption coefficient of CDOM and NAP when using the exponential approximation
    """
    S_dg = a + b * (c + (Rrs[find_closest(wavelengths, lambda1)[1]] / Rrs[find_closest(wavelengths, lambda2)[1]])**d)
    
    return S_dg



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
    if method == 0:
        return (y_pred-y_true) * weights
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
        return np.sum((y_pred - y_true)**2 * weights)
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
        # 1 - R2-score
        return 1 - r2_score(y_true, y_pred)
    elif method == 14:
        # SAM
        return distance.spectral_angle(y_true, y_pred)
    elif method == 15:
        # SID
        return distance.spectral_information_divergence(y_true, y_pred)
    elif method == 16:
        # Chebyshev
        return distance.chebyshev_distance(y_true, y_pred) 
    

def get_solar_zenith_angle(lat, lon, timestamp_utc):
    """
    Compute solar zenith angle from lat, lon and time (in UTC) using get_altitude() from the pysolar package.
    
    :param lat: latitude in decimal degrees
    :param lon: longitude in decimal degrees
    :param timestamp_utc: UTC time as datetime object 
    :return: solar zenith angle respective for time and location
    """
    return 90 - get_altitude(lat, lon, timestamp_utc)
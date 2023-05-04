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
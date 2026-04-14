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
# bio_optics
#  König, M., Noel, P., Hondula. K.L., Jamalinia, E., Dai, J., Vaughn, N.R., Asner, G.P. (2023):
#  bio_optics python package (Version x) [Software]. Available from https://github.com/CMLandOcean/bio_optics

import numpy as np
from lmfit import Parameters
from scipy.optimize import least_squares


def _get_residuals(fun, data, weights):
    """
    Decorator returning a function that computes (fun(...) - data) * weights.
    Suitable as a residual function for scipy.optimize.least_squares.
    """
    def compositefun(*args, **kwargs):
        if len(weights) == 0:
            return fun(*args, **kwargs) - data
        return (fun(*args, **kwargs) - data) * weights
    return compositefun


def _get_fun_shim(fun, wavelengths, fit_param_names, params_obj):
    """
    Decorator returning a function whose first argument is a flat array of fit-parameter values.
    Marshals x0 back into the lmfit Parameters object before calling fun(params_obj, wavelengths, **kwargs).
    This bridges scipy's parameter-list convention with the Parameters-object convention used by forward models.
    """
    def outer_fun(x0, **kwargs):
        for i, name in enumerate(fit_param_names):
            params_obj[name].value = x0[i]
        return fun(params_obj, wavelengths, **kwargs)
    return outer_fun


def invert(params,
           Rrs,
           wavelengths,
           forward_func,
           fixed_params=None,
           weights=[],
           max_nfev=400,
           **fwd_kwargs):
    """
    Inversely fit a forward model to a measured remote sensing reflectance spectrum using
    scipy.optimize.least_squares with numerical Jacobian estimation.

    Args:
        params: lmfit Parameters object specifying the model configuration
        Rrs: measured remote sensing reflectance spectrum [sr-1]
        wavelengths: wavelengths of Rrs bands [nm]
        forward_func: forward model callable with signature forward_func(parameters, wavelengths, **kwargs)
        fixed_params: optional dict of {param_name: value} to fix before inversion,
                      useful for incorporating measured data (e.g. {'zB': measured_depth})
        weights: optional spectral weighting coefficients
        max_nfev: maximum number of function evaluations, default: 400
        **fwd_kwargs: additional keyword arguments forwarded to forward_func (e.g. precomputed arrays)

    Returns:
        scipy OptimizeResult containing optimised parameters and fit diagnostics
    """
    if fixed_params is not None:
        for name, value in fixed_params.items():
            params[name].value = value
            params[name].vary = False

    ret_params = params.copy()
    fit_params, fit_bounds, fit_param_names = [], [[], []], []

    for param in ret_params.keys():
        if ret_params[param].vary:
            fit_params.append(ret_params[param].value)
            fit_param_names.append(ret_params[param].name)
            fit_bounds[0].append(ret_params[param].min)
            fit_bounds[1].append(ret_params[param].max)

    userfun = _get_fun_shim(
        _get_residuals(forward_func, Rrs, weights),
        wavelengths, fit_param_names, ret_params)

    return least_squares(userfun, fit_params, bounds=fit_bounds,
                         max_nfev=max_nfev, kwargs=fwd_kwargs)

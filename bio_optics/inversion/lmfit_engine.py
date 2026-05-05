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
from lmfit import minimize
from ..helper import utils


def invert(params,
           Rrs,
           wavelengths,
           forward_func,
           fixed_params=None,
           weights=None,
           error_method=2,
           method="least-squares",
           max_nfev=400,
           **fwd_kwargs):
    """
    Inversely fit a forward model to a measured remote sensing reflectance spectrum.

    Args:
        params: lmfit Parameters object specifying the model configuration
        Rrs: measured remote sensing reflectance spectrum [sr-1]
        wavelengths: wavelengths of Rrs bands [nm]
        forward_func: forward model callable with signature forward_func(parameters, wavelengths, **kwargs)
        fixed_params: optional dict of {param_name: value} to fix before inversion,
                      useful for incorporating measured data (e.g. {'zB': measured_depth})
        weights: optional spectral weighting coefficients
        error_method: residual method passed to utils.compute_residual, default: 2 (absolute difference)
        method: lmfit minimisation method, default: 'least-squares'
        max_nfev: maximum number of function evaluations, default: 400
        **fwd_kwargs: additional keyword arguments forwarded to forward_func (e.g. precomputed arrays)

    Returns:
        lmfit MinimizerResult containing optimised parameters and goodness-of-fit statistics
    """
    if fixed_params is not None:
        for name, value in fixed_params.items():
            params[name].value = value
            params[name].vary = False

    if weights is None:
        weights = np.ones(len(Rrs))

    def _func2opt(parameters, Rrs, wavelengths, weights):
        Rrs_sim = forward_func(parameters, wavelengths, **fwd_kwargs)
        return utils.compute_residual(Rrs, Rrs_sim, method=error_method, weights=weights)

    return minimize(_func2opt, params, args=(Rrs, wavelengths, weights),
                    method=method, max_nfev=max_nfev)

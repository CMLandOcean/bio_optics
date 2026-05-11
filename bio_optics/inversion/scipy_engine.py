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

from typing import Callable, List, NamedTuple, Optional

import numpy as np
from lmfit import Parameters
from scipy.optimize import least_squares


class ScipySetup(NamedTuple):
    """
    Packaged inversion configuration for scipy-based engines.
    Mirrors the InversionSetup / LmfitSetup pattern so that scipy_engine
    can be plugged directly into superpixel_engine and dask_oe_engine as
    invert_fn=scipy_engine.invert_image.

    Attributes:
        params:       lmfit Parameters template (copied per spectrum in invert_image)
        wavelengths:  band centres in nm, shape (n_obs,)
        forward_func: forward model callable, signature f(params, wavelengths, **kwargs)
        fit_names:    names of free parameters (p.vary == True), in params iteration order
        weights:      optional per-band weights, shape (n_obs,)
        max_nfev:     maximum function evaluations (default 400)
    """
    params:       object           # lmfit.Parameters — avoid circular import in type hint
    wavelengths:  np.ndarray
    forward_func: Callable
    fit_names:    List[str]
    weights:      Optional[np.ndarray]
    max_nfev:     int


def build_inversion(
    params,
    wavelengths,
    forward_func,
    fixed_params=None,
    weights=None,
    max_nfev=400,
) -> ScipySetup:
    """
    Build a ScipySetup ready for invert_image().

    Parameters
    ----------
    params       : lmfit Parameters object (not mutated; copied internally)
    wavelengths  : band centres [nm]
    forward_func : forward model callable f(params, wavelengths, **fwd_kwargs)
    fixed_params : optional {name: value} dict — fixes params before baking into setup
    weights      : optional per-band spectral weights
    max_nfev     : max function evaluations per spectrum (default 400)

    Returns
    -------
    ScipySetup
    """
    if fixed_params is not None:
        params = params.copy()
        for name, value in fixed_params.items():
            params[name].value = value
            params[name].vary = False
    fit_names = [n for n, p in params.items() if p.vary]
    return ScipySetup(
        params=params,
        wavelengths=np.asarray(wavelengths),
        forward_func=forward_func,
        fit_names=fit_names,
        weights=weights,
        max_nfev=max_nfev,
    )


def _get_residuals(fun, data, weights):
    """
    Decorator returning a function that computes (fun(...) - data) * weights.
    Suitable as a residual function for scipy.optimize.least_squares.
    """
    def compositefun(*args, **kwargs):
        if weights is None:
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
           weights=None,
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


def invert_image(spectra, setup, noise, store_chi2_spectral: bool = False, **kwargs):
    """
    Invert a batch of spectra using scipy.optimize.least_squares.

    Compatible with superpixel_engine and dask_engine as invert_fn:
        invert_fn=scipy_engine.invert_image

    Parameters
    ----------
    spectra             : (n_spectra, n_obs)  mean spectra to invert
    setup               : ScipySetup from build_inversion()
    noise               : accepted for interface compatibility; not used
    store_chi2_spectral : if True include ``chi2_spectral`` in output — mean
                          squared difference between observed and simulated
                          spectrum with no prior term (extra forward call per
                          spectrum).
    **kwargs:
        x_a_image : (n_spectra, n_fit) optional warm-start values in physical
                    space.  All other kwargs are silently ignored.

    Returns
    -------
    dict with keys:
        x_hat           : (n_spectra, n_fit)  optimised parameter values
        chi2            : (n_spectra,)        raw sum of squared residuals
        fit_names       : list[str]
        chi2_spectral   : (n_spectra,)        only when store_chi2_spectral=True
    """
    n_spectra = spectra.shape[0]
    n_fit     = len(setup.fit_names)
    x_hat     = np.full((n_spectra, n_fit), np.nan)
    chi2      = np.full(n_spectra, np.nan)
    chi2_sp   = np.full(n_spectra, np.nan) if store_chi2_spectral else None

    x0 = kwargs.get('x_a_image')   # (n_spectra, n_fit) or None

    for i, spectrum in enumerate(spectra):
        if not np.isfinite(spectrum).all():
            continue
        p = setup.params.copy()
        if x0 is not None and np.isfinite(x0[i]).all():
            for j, name in enumerate(setup.fit_names):
                p[name].value = x0[i, j]
        result = invert(
            p, spectrum, setup.wavelengths, setup.forward_func,
            weights=setup.weights, max_nfev=setup.max_nfev,
        )
        if result.success:
            x_hat[i] = result.x
            chi2[i]  = np.sum(result.fun ** 2)
            if store_chi2_spectral:
                p_fit = setup.params.copy()
                for j, name in enumerate(setup.fit_names):
                    p_fit[name].value = result.x[j]
                y_hat = np.asarray(setup.forward_func(p_fit, setup.wavelengths))
                chi2_sp[i] = np.mean((spectrum - y_hat) ** 2)

    out = {'x_hat': x_hat, 'chi2': chi2, 'fit_names': setup.fit_names}
    if store_chi2_spectral:
        out['chi2_spectral'] = chi2_sp
    return out

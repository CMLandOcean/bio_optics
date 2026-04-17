"""
Optimal Estimation inversion engine for JAX forward models.

Implements the Gauss-Newton OE solver of Rodgers (2000, Ch. 5) with exact
Jacobians via jax.jacobian.  The core function solve() is pure JAX and can be
JIT-compiled or vmapped over a pixel batch.  The convenience wrapper invert()
and the batch-setup function build_inversion() bridge the lmfit-Parameters
convention used by the rest of bio_optics.

References:
    Rodgers, C.D. (2000): Inverse Methods for Atmospheric Sounding: Theory
    and Practice. World Scientific, Singapore.

Prior specification
-------------------
All priors are specified in **physical units** (mg/m³, m, …) via the
``sigma_a`` dict.  For parameters that are retrieved in log-space (see
``log_params`` below), the engine converts automatically using the delta
method:

    σ_log ≈ σ_physical / x_a        (valid when σ / x_a ≲ 1)

This keeps the API consistent: a user always supplies physical standard
deviations regardless of whether a parameter is log-transformed.

Log-transform support
---------------------
For parameters that must be strictly positive (concentrations, depth, …)
you can pass a ``log_params`` list to ``build_inversion()`` or ``invert()``.
Those parameters are retrieved in log-space, which is equivalent to placing a
**lognormal prior** on them in physical space.  The forward function closure
applies ``jnp.exp()`` automatically before calling the underlying forward
model, so solve() and invert_pixels() are unaware of the transform.

After inversion, use ``to_physical()`` and ``posterior_sigma_physical()`` to
convert results back to physical space::

    x_phys  = oe_engine.to_physical(result.x_hat, setup.log_mask)
    sigma_ph = oe_engine.posterior_sigma_physical(result.S_hat, x_phys, setup.log_mask)

Typical two-step usage (with log-transform)::

    from bio_optics.reflectance import albert_mobley_jax
    from bio_optics.coupled_models import albert_mobley_3C_jax
    from bio_optics.inversion import oe_engine

    # --- precompute (once per sensor configuration) -------------------------
    pre_3C = albert_mobley_3C_jax.precompute(wavelengths, theta_sun=np.radians(30))
    pre_w  = albert_mobley_jax.precompute(wavelengths)

    # --- Step 1: joint surface + water fit with loose surface priors --------
    f_vec_3C = albert_mobley_3C_jax.make_forward_vec(all_names_3C, pre_3C)
    result1, names1, lm1 = oe_engine.invert(
        params_3C, Rrs, f_vec_3C, noise, sigma_a_loose,
        log_params=['C_0', 'C_Y', 'C_Mie', 'zB'],
    )
    x_phys1 = oe_engine.to_physical(result1.x_hat, lm1)

    # --- subtract surface contribution -------------------------------------
    p_hat = dict(zip(names1, np.array(x_phys1)))
    Rrs_glint     = (albert_mobley_3C_jax.forward(p_hat, pre_3C)
                     - albert_mobley_jax.forward(p_hat, pre_3C))
    Rrs_corrected = Rrs - Rrs_glint

    # --- Step 2: water-only retrieval with tighter priors ------------------
    f_vec_w = albert_mobley_jax.make_forward_vec(all_names_w, pre_w)
    result2, names2, lm2 = oe_engine.invert(
        params_w, Rrs_corrected, f_vec_w, noise, sigma_a_tight,
        log_params=['C_0', 'C_Y', 'C_Mie', 'zB'],
    )
    x_phys2 = oe_engine.to_physical(result2.x_hat, lm2)

    # --- batch inversion (image) -------------------------------------------
    setup = oe_engine.build_inversion(
        params_w, f_vec_w, sigma_a_tight,
        log_params=['C_0', 'C_Y', 'C_Mie', 'zB'],
    )
    results = jax.jit(oe_engine.invert_pixels)(
        setup.f_fit, Rrs_image, noise_std, setup.x_a, setup.S_a_inv,
    )
    x_hat_phys = oe_engine.to_physical(results.x_hat, setup.log_mask)
"""

import numpy as np
import jax
import jax.numpy as jnp
from typing import NamedTuple, Callable, List, Optional

jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

class OEResult(NamedTuple):
    """
    Optimal Estimation retrieval result.

    All fields are JAX arrays (or Python scalars derived from them), so
    OEResult instances can be stacked by jax.vmap and transferred to NumPy
    with np.array().

    Note on retrieval space vs physical space
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    When log-transformed parameters are used (``log_params`` argument of
    ``build_inversion()`` or ``invert()``), ``x_hat`` and the rows/columns of
    ``S_hat``, ``A``, and ``J`` that correspond to those parameters are in
    **log-space**, not physical space.  Use ``to_physical()`` and
    ``posterior_sigma_physical()`` to convert back.

    Attributes:
        x_hat:  posterior mean in retrieval space, shape (n_fit,).
                For log-transformed parameters this is ln(x); apply
                ``to_physical(x_hat, log_mask)`` to recover physical values.
        S_hat:  posterior covariance in retrieval space, shape (n_fit, n_fit).
                Diagonal elements are variances in retrieval space.  For
                log-params, σ_physical ≈ exp(x_hat) * sqrt(S_hat_diag).
        A:      averaging kernel matrix, shape (n_fit, n_fit).
                A[i, i] close to 1 means the i-th parameter is well-constrained
                by the data; close to 0 means the prior dominates.
        dfs:    degrees of freedom for signal (= tr(A)), scalar.
                Measures how many independent pieces of information are
                extracted from the observed spectrum.
        chi2:   chi-squared per observation band, scalar.
                chi2 ≈ 1 indicates a good fit; chi2 >> 1 suggests model error
                or underestimated noise; chi2 << 1 suggests over-fitting or
                overestimated noise.
        J:      Jacobian of the forward model at x_hat, shape (n_obs, n_fit).
                Rows are spectral bands, columns are free parameters (in
                retrieval space).
        y_hat:  simulated spectrum at x_hat, shape (n_obs,).
                Compare to y_obs to assess spectral fit quality.
    """
    x_hat: jnp.ndarray
    S_hat: jnp.ndarray
    A:     jnp.ndarray
    dfs:   float
    chi2:  float
    J:     jnp.ndarray
    y_hat: jnp.ndarray


# ---------------------------------------------------------------------------
# Public conversion helpers (retrieval space ↔ physical space)
# ---------------------------------------------------------------------------

def to_physical(x_retrieval: jnp.ndarray,
                log_mask: jnp.ndarray) -> jnp.ndarray:
    """
    Convert a retrieval-space state vector to physical space.

    For parameters whose ``log_mask`` entry is 1 (log-transformed), applies
    ``exp()``.  For linear parameters (log_mask = 0) the values are returned
    unchanged.

    This is the inverse of the log-transform applied internally by
    ``build_inversion()`` / ``invert()``.  It is a pure JAX operation and
    can be applied inside jax.jit or jax.vmap.

    Args:
        x_retrieval: state vector in retrieval space, shape (..., n_fit).
                     Typically ``OEResult.x_hat`` or a batch of them.
        log_mask:    binary mask, shape (n_fit,).  Entry is 1.0 for
                     log-transformed parameters, 0.0 for linear parameters.
                     Stored as ``InversionSetup.log_mask`` or returned as the
                     third element of ``invert()``.

    Returns:
        x_physical: state vector in physical space, same shape as x_retrieval.

    Example::

        x_phys = oe_engine.to_physical(result.x_hat, setup.log_mask)
        # element-wise: x_phys[i] = exp(x_hat[i]) if log_mask[i] else x_hat[i]
    """
    return jnp.where(log_mask > 0.5, jnp.exp(x_retrieval), x_retrieval)


def posterior_sigma_physical(S_hat: jnp.ndarray,
                              x_hat_physical: jnp.ndarray,
                              log_mask: jnp.ndarray) -> jnp.ndarray:
    """
    Convert posterior standard deviations from retrieval space to physical space.

    For linear parameters the posterior std is ``sqrt(S_hat[i, i])``.
    For log-transformed parameters, the posterior std in physical space is
    obtained via the delta method:

        σ_physical ≈ x_physical · σ_log

    where ``σ_log = sqrt(S_hat[i, i])`` is the posterior std in log-space and
    ``x_physical = exp(x_hat_log)`` is the posterior mean in physical space.
    This approximation is exact for small σ_log (≲ 0.5) and slightly
    underestimates the true lognormal std for wider posteriors.

    Args:
        S_hat:          posterior covariance in retrieval space, shape
                        (n_fit, n_fit) or (n_pixels, n_fit, n_fit).
                        Typically ``OEResult.S_hat``.
        x_hat_physical: posterior mean in **physical** space, shape (n_fit,)
                        or (n_pixels, n_fit).  Obtain via ``to_physical()``.
        log_mask:       binary mask, shape (n_fit,).  See ``to_physical()``.

    Returns:
        sigma_physical: posterior standard deviations in physical space,
                        shape (n_fit,) or (n_pixels, n_fit).

    Example::

        x_phys  = oe_engine.to_physical(result.x_hat, setup.log_mask)
        sigma   = oe_engine.posterior_sigma_physical(result.S_hat, x_phys, setup.log_mask)
    """
    # Works for both single-pixel (S_hat 2-D) and batch (S_hat 3-D)
    sigma_retrieval = jnp.sqrt(jnp.diagonal(S_hat, axis1=-2, axis2=-1))
    sigma_phys = jnp.where(log_mask > 0.5,
                           x_hat_physical * sigma_retrieval,
                           sigma_retrieval)
    return sigma_phys


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_S_eps_inv(noise, n_obs: int) -> jnp.ndarray:
    """
    Build the inverse noise covariance matrix from various input forms.

    Args:
        noise: measurement uncertainty in one of three forms —
               * scalar float: uniform noise std across all bands →
                 diagonal S_eps with constant variance
               * 1-D array of length n_obs: per-band noise std →
                 diagonal S_eps
               * 2-D array of shape (n_obs, n_obs): full noise covariance
                 matrix, **already inverted** — returned as-is
        n_obs: number of observation bands

    Returns:
        S_eps_inv: inverse noise covariance, shape (n_obs, n_obs)
    """
    noise = jnp.asarray(noise, dtype=jnp.float64)
    if noise.ndim == 0:
        return jnp.eye(n_obs) / noise ** 2
    elif noise.ndim == 1:
        return jnp.diag(1.0 / noise ** 2)
    else:
        return noise  # caller supplied pre-inverted full matrix


def _build_f_vec_fit(f_vec: Callable,
                     all_names: list,
                     fit_indices: list,
                     fixed_vals: jnp.ndarray,
                     log_mask: Optional[jnp.ndarray] = None) -> Callable:
    """
    Return a projected forward function that accepts only free-parameter values.

    The returned function inserts the fixed parameter values at the correct
    positions and delegates to the full f_vec.  If ``log_mask`` is provided,
    log-transformed parameters are exponentiated before insertion, so the
    forward model always receives physical-space values.

    Args:
        f_vec:       full forward model, f(params_all) -> y
        all_names:   ordered list of all parameter names expected by f_vec
        fit_indices: indices into all_names for the free parameters
        fixed_vals:  JAX array of fixed parameter values, shape (n_all,).
                     Contains initial/fixed values for ALL parameters; free
                     parameter slots are overwritten by x_fit.
        log_mask:    optional binary mask, shape (n_fit,).  Entry 1.0 means
                     the corresponding element of x_fit is in log-space and
                     must be exponentiated before passing to f_vec.  None or
                     all-zero mask means no log-transforms are applied.

    Returns:
        f_fit: callable f_fit(x_fit) -> y, where x_fit has shape (n_fit,).
               When log_mask is non-None, x_fit is expected in retrieval space
               (log-space for flagged parameters).
    """
    fit_indices_arr = jnp.array(fit_indices, dtype=jnp.int32)

    if log_mask is None or not jnp.any(log_mask > 0.5):
        # Fast path: no log-transforms, insert directly
        def f_fit(x_fit):
            x_all = fixed_vals.at[fit_indices_arr].set(x_fit)
            return f_vec(x_all)
    else:
        # General path: exp() for log-params, identity for linear params
        def f_fit(x_fit):
            x_phys = jnp.where(log_mask > 0.5, jnp.exp(x_fit), x_fit)
            x_all  = fixed_vals.at[fit_indices_arr].set(x_phys)
            return f_vec(x_all)

    return f_fit


# ---------------------------------------------------------------------------
# Core solver — pure JAX, JIT-compilable, vmappable
# ---------------------------------------------------------------------------

def solve(f_vec: Callable,
          y_obs: jnp.ndarray,
          noise,
          x0: jnp.ndarray,
          x_a: jnp.ndarray,
          S_a_inv: jnp.ndarray,
          n_iter: int = 10,
          lm_damping: float = 0.0) -> OEResult:
    """
    Gauss-Newton Optimal Estimation solver (Rodgers 2000, Ch. 5).

    Minimises the OE cost function::

        J(x) = [y − F(x)]ᵀ Sε⁻¹ [y − F(x)] + (x − xa)ᵀ Sa⁻¹ (x − xa)

    via the iterative update::

        H       = Jᵀ Sε⁻¹ J + Sa⁻¹
        g       = Jᵀ Sε⁻¹ [y − F(xᵢ)] − Sa⁻¹ (xᵢ − xa)
        xᵢ₊₁   = xᵢ + H⁻¹ g

    After convergence the posterior statistics are::

        Ŝ   = H⁻¹
        A   = Ŝ Jᵀ Sε⁻¹ J          (averaging kernel)
        DFS = tr(A)
        χ²  = (y − ŷ)ᵀ Sε⁻¹ (y − ŷ) / n_obs

    This function is pure JAX: it can be wrapped with jax.jit or mapped with
    jax.vmap.  The iteration is a Python for-loop that JAX unrolls at trace
    time, so n_iter must be a static Python integer.

    solve() is agnostic to the retrieval space: if log-transformed parameters
    are used, x0, x_a, S_a_inv, and f_vec should already be expressed in
    log-space (as set up by build_inversion()).  The returned OEResult.x_hat
    is then in retrieval space; use to_physical() to convert.

    Args:
        f_vec:      forward model f(x) -> y; x shape (n_fit,), y shape (n_obs,).
                    Must be differentiable with jax.jacobian.
                    Typically the projected function from build_inversion() /
                    invert(), which handles fixed-param insertion and the
                    log → physical conversion internally.
        y_obs:      observed spectrum, shape (n_obs,)
        noise:      measurement uncertainty — scalar std, 1-D std array (n_obs,),
                    or pre-inverted full covariance matrix (n_obs, n_obs).
                    See _build_S_eps_inv() for details.
        x0:         initial state vector in retrieval space, shape (n_fit,).
                    For log-params this should be ln(initial_physical_value).
        x_a:        prior mean in retrieval space, shape (n_fit,).
                    For log-params this should be ln(prior_physical_mean).
        S_a_inv:    inverse prior covariance in retrieval space,
                    shape (n_fit, n_fit).  For independent priors use
                    jnp.diag(1 / sigma_a_retrieval ** 2).
        n_iter:     number of Gauss-Newton iterations (static Python int),
                    default 10.  Increase if the cost function is non-linear.
        lm_damping: Levenberg-Marquardt diagonal damping added to H before
                    solving, expressed as a fraction of each diagonal element
                    (H_damp = H + lm_damping * diag(H)).  Default 0 (pure
                    Gauss-Newton).  Use 0.1 if the solver diverges or produces
                    NaN pixels; log-transforms often make this unnecessary.

    Returns:
        OEResult namedtuple with fields x_hat, S_hat, A, dfs, chi2, J, y_hat.
        All arrays are in retrieval space; see OEResult docstring.
    """
    n_obs = y_obs.shape[0]
    S_eps_inv = _build_S_eps_inv(noise, n_obs)

    x = x0
    for _ in range(n_iter):
        y_i = f_vec(x)
        J   = jax.jacobian(f_vec)(x)                          # (n_obs, n_fit)

        H = J.T @ S_eps_inv @ J + S_a_inv
        if lm_damping > 0.0:
            H = H + lm_damping * jnp.diag(jnp.diag(H))

        g  = J.T @ S_eps_inv @ (y_obs - y_i) - S_a_inv @ (x - x_a)
        dx = jnp.linalg.solve(H, g)
        x  = x + dx

    # Diagnostics at solution
    y_hat = f_vec(x)
    J     = jax.jacobian(f_vec)(x)
    H     = J.T @ S_eps_inv @ J + S_a_inv
    if lm_damping > 0.0:
        H = H + lm_damping * jnp.diag(jnp.diag(H))

    S_hat = jnp.linalg.inv(H)
    A     = S_hat @ J.T @ S_eps_inv @ J
    dfs   = jnp.trace(A)

    residual = y_obs - y_hat
    chi2 = residual @ S_eps_inv @ residual / n_obs

    return OEResult(x_hat=x, S_hat=S_hat, A=A, dfs=dfs, chi2=chi2,
                    J=J, y_hat=y_hat)


# ---------------------------------------------------------------------------
# Inversion setup — packages projected forward function + prior for invert_pixels
# ---------------------------------------------------------------------------

class InversionSetup(NamedTuple):
    """
    Packaged inversion configuration returned by build_inversion().

    All arrays are in **retrieval space**: for log-transformed parameters
    (log_mask == 1), x_a contains ln(physical_prior_mean) and S_a_inv encodes
    the uncertainty in log-space.  Use ``to_physical()`` and
    ``posterior_sigma_physical()`` to interpret results in physical units.

    Attributes:
        f_fit:     projected forward model f(x_fit) -> y, where x_fit contains
                   only the free parameters (shape n_fit) in retrieval space.
                   Log-transformed parameters are exponentiated inside the
                   closure before the underlying physical forward model is
                   called.  Safe to pass directly to invert_pixels() or to
                   jax.jit / jax.vmap.
        fit_names: list of free parameter names, length n_fit.  Maps columns of
                   x_hat / rows & cols of S_hat to parameter names.
        x_a:       prior mean in retrieval space, shape (n_fit,).
                   For log-params: ln(physical prior mean).
                   For linear params: physical prior mean.
        S_a_inv:   diagonal inverse prior covariance in retrieval space,
                   shape (n_fit, n_fit).
                   For log-params: 1 / σ_log² where σ_log ≈ σ_physical / x_a.
                   For linear params: 1 / σ_physical².
        log_mask:  binary mask, shape (n_fit,).  Entry is 1.0 for
                   log-transformed parameters, 0.0 for linear parameters.
                   Pass to to_physical() and posterior_sigma_physical() when
                   interpreting results.
    """
    f_fit:     Callable
    fit_names: List[str]
    x_a:       jnp.ndarray
    S_a_inv:   jnp.ndarray
    log_mask:  jnp.ndarray


def build_inversion(params,
                    f_vec: Callable,
                    sigma_a: dict,
                    fixed_params: dict = None,
                    log_params: list = None) -> InversionSetup:
    """
    Build a projected forward function and prior arrays ready for invert_pixels().

    This is the counterpart to invert() for the batch workflow: it performs the
    same parameter splitting and forward-function projection that invert() does
    internally, but returns the pieces as an InversionSetup so you can pass
    them directly to invert_pixels().

    Prior specification
    ~~~~~~~~~~~~~~~~~~~
    ``sigma_a`` is always specified in **physical units** (the same units as
    the parameters themselves).  For log-transformed parameters (those listed
    in ``log_params``), the conversion to log-space is done automatically
    using the delta method:

        σ_log ≈ σ_physical / x_a

    This means that the width of the prior in log-space equals the **relative
    uncertainty** in physical space (e.g., σ_physical = 1.0, x_a = 2.0 gives
    σ_log = 0.5, i.e., ±50% relative uncertainty).

    The prior mean in log-space is set to ln(params[name].value), so
    ``params[name].value`` should be a reasonable starting guess in physical
    units for both linear and log-transformed parameters.

    Typical usage::

        setup = oe_engine.build_inversion(
            params_w, f_vec_w, sigma_a_w,
            log_params=['C_0', 'C_Y', 'C_Mie', 'zB'],
        )

        # Batch inversion
        results = jax.jit(oe_engine.invert_pixels)(
            setup.f_fit, Rrs_pixels, noise_std, setup.x_a, setup.S_a_inv,
        )

        # Convert to physical space
        x_hat_phys  = oe_engine.to_physical(results.x_hat, setup.log_mask)
        sigma_phys  = oe_engine.posterior_sigma_physical(
                          results.S_hat, x_hat_phys, setup.log_mask)

    Args:
        params:       lmfit Parameters object (or compatible dict with .value
                      and .vary per entry).  Provides parameter names, initial
                      values (used as both x0 and prior mean), and vary flags.
                      Values should always be in physical units.
        f_vec:        full forward model produced by make_forward_vec(all_names,
                      precomputed).  Must accept a 1-D array of length
                      len(params) with all parameters in physical units.
        sigma_a:      dict {param_name: prior_std} for every free (vary=True)
                      parameter.  Always in **physical units** regardless of
                      log_params.  For a log-transformed parameter with
                      x_a = 1.0, sigma_a = 0.5 yields σ_log = 0.5 (50%
                      relative uncertainty).
        fixed_params: optional dict {param_name: value} — overrides
                      params[name].value and marks those parameters as
                      non-varying before the split.
        log_params:   optional list of parameter names to retrieve in
                      log-space.  Those parameters must be strictly positive
                      (x_a > 0).  The transform is applied transparently
                      inside f_fit; sigma_a is still given in physical units.
                      Default None (all parameters retrieved in physical space,
                      identical to the pre-log-transform behaviour).

    Returns:
        InversionSetup namedtuple with fields:
            f_fit, fit_names, x_a, S_a_inv, log_mask.
        All array fields are in retrieval space; see InversionSetup docstring.

    Raises:
        ValueError: if a log-param has a non-positive prior mean (params value).
    """
    log_set = set(log_params or [])

    # Apply fixed_params overrides
    if fixed_params is not None:
        for name, value in fixed_params.items():
            params[name].value = value
            params[name].vary = False

    all_names = list(params.keys())
    fit_names = [n for n in all_names if params[n].vary]
    fit_idx   = [all_names.index(n) for n in fit_names]

    # Full vector of all parameter values in physical space
    x_all_np   = np.array([float(params[n].value) for n in all_names])
    fixed_vals = jnp.array(x_all_np)

    # Build retrieval-space x_a, x0, sigma vectors and log_mask
    x_a_list      = []
    sigma_ret_list = []
    log_mask_list  = []

    for n in fit_names:
        val = float(params[n].value)
        sig = sigma_a[n]

        if n in log_set:
            if val <= 0:
                raise ValueError(
                    f"log_params parameter '{n}' has non-positive prior mean "
                    f"({val}).  Log-transform requires strictly positive values."
                )
            # Convert to log-space: x_a_log = ln(x_a_phys)
            #                       σ_log   ≈ σ_phys / x_a_phys  (delta method)
            x_a_list.append(np.log(val))
            sigma_ret_list.append(sig / val)
            log_mask_list.append(1.0)
        else:
            x_a_list.append(val)
            sigma_ret_list.append(sig)
            log_mask_list.append(0.0)

    log_mask = jnp.array(log_mask_list)
    x_a      = jnp.array(x_a_list)
    S_a_inv  = jnp.diag(1.0 / jnp.array(sigma_ret_list) ** 2)

    # Projected forward function: accepts retrieval-space x_fit, returns Rrs
    f_fit = _build_f_vec_fit(f_vec, all_names, fit_idx, fixed_vals, log_mask)

    return InversionSetup(
        f_fit=f_fit,
        fit_names=fit_names,
        x_a=x_a,
        S_a_inv=S_a_inv,
        log_mask=log_mask,
    )


# ---------------------------------------------------------------------------
# Convenience wrapper — bridges lmfit Parameters convention
# ---------------------------------------------------------------------------

def invert(params,
           Rrs: np.ndarray,
           f_vec: Callable,
           noise,
           sigma_a: dict,
           fixed_params: dict = None,
           log_params: list = None,
           n_iter: int = 10,
           lm_damping: float = 0.0):
    """
    OE inversion with lmfit-Parameters-style parameter specification.

    Convenience wrapper around solve() that reads initial values, prior means,
    and parameter ordering from a lmfit Parameters object, then calls the core
    solver and returns the result alongside the list of fitted parameter names
    and the log-mask for converting results back to physical space.

    Internally calls build_inversion() to construct the projected forward
    function and prior arrays, then calls solve().  For batch pixel inversion
    use build_inversion() + invert_pixels() directly.

    Prior specification
    ~~~~~~~~~~~~~~~~~~~
    See build_inversion() for details on how sigma_a and log_params interact.
    sigma_a is always in **physical units**; the delta-method conversion to
    log-space is applied automatically for parameters listed in log_params.

    Args:
        params:       lmfit Parameters object (or compatible dict with .value
                      and .vary per entry).  Provides parameter names, initial
                      values (used as both x0 and prior mean xa), and vary
                      flags.  Values should be in physical units.
        Rrs:          observed remote sensing reflectance, shape (n_obs,)
        f_vec:        full forward model produced by make_forward_vec(all_names,
                      precomputed).  Must accept a 1-D array of length
                      len(params) with all parameters in physical units.
        noise:        measurement uncertainty — scalar std, 1-D std array, or
                      pre-inverted full covariance matrix.  See solve().
        sigma_a:      dict {param_name: prior_std} in physical units for every
                      free (vary=True) parameter.
        fixed_params: optional dict {param_name: value} — overrides
                      params[name].value and marks those parameters as
                      non-varying.  Same convention as lmfit_engine.invert().
        log_params:   optional list of free parameter names to retrieve in
                      log-space (lognormal prior).  Parameters must have
                      positive prior means.  Default None (all linear/Gaussian).
        n_iter:       Gauss-Newton iterations, passed to solve(), default 10.
        lm_damping:   LM damping factor, passed to solve(), default 0.

    Returns:
        result:    OEResult namedtuple (see solve() for field descriptions).
                   x_hat and related arrays are in **retrieval space**.
        fit_names: list of free parameter names in the order they appear in
                   result.x_hat, result.S_hat, result.A, result.J.
        log_mask:  JAX array of shape (n_fit,) with 1.0 for log-transformed
                   parameters, 0.0 for linear parameters.  Pass to
                   to_physical() and posterior_sigma_physical() to convert
                   result.x_hat and result.S_hat back to physical space.

    Example::

        result, names, lm = oe_engine.invert(
            params_w, Rrs, f_vec_w, noise_std, sigma_a,
            log_params=['C_0', 'C_Y', 'zB'],
        )
        x_phys = oe_engine.to_physical(result.x_hat, lm)
        sigma  = oe_engine.posterior_sigma_physical(result.S_hat, x_phys, lm)
        print(dict(zip(names, np.array(x_phys))))
    """
    setup = build_inversion(params, f_vec, sigma_a,
                            fixed_params=fixed_params,
                            log_params=log_params)

    y_obs  = jnp.array(np.asarray(Rrs), dtype=jnp.float64)
    result = solve(setup.f_fit, y_obs, noise,
                   x0=setup.x_a,    # start from prior mean in retrieval space
                   x_a=setup.x_a,
                   S_a_inv=setup.S_a_inv,
                   n_iter=n_iter,
                   lm_damping=lm_damping)

    return result, setup.fit_names, setup.log_mask


# ---------------------------------------------------------------------------
# Vectorised image inversion — jax.vmap over pixels
# ---------------------------------------------------------------------------

def invert_pixels(f_vec: Callable,
                  Rrs_pixels: jnp.ndarray,
                  noise,
                  x_a: jnp.ndarray,
                  S_a_inv: jnp.ndarray,
                  x0: jnp.ndarray = None,
                  n_iter: int = 10,
                  lm_damping: float = 0.0) -> OEResult:
    """
    Batch OE inversion over a stack of pixels via jax.vmap.

    Applies solve() independently to every row of Rrs_pixels, sharing the
    same forward model, noise model, prior, and initial state.  The function
    is JIT-compilable: wrap with jax.jit for best performance on the first
    call.

    All array arguments (x_a, S_a_inv, x0) are in **retrieval space**.
    When log-transformed parameters are used, pass ``setup.x_a`` and
    ``setup.S_a_inv`` from build_inversion() directly — they are already in
    the correct retrieval space.  Apply ``to_physical()`` to the returned
    ``x_hat`` to recover physical concentrations / depths.

    Usage::

        # Build projected forward function and retrieval-space prior
        setup = oe_engine.build_inversion(
            params_w, f_vec_w, sigma_a_w,
            log_params=['C_0', 'C_Y', 'C_Mie', 'zB'],
        )

        # Stack pixel spectra: shape (n_pixels, n_obs)
        Rrs_pixels = image.reshape(-1, n_bands)

        # Invert all pixels (JIT-compiled)
        results = jax.jit(oe_engine.invert_pixels)(
            setup.f_fit, Rrs_pixels, noise_std, setup.x_a, setup.S_a_inv,
            n_iter=10, lm_damping=0.0,
        )

        # Convert retrieval-space x_hat to physical space
        x_hat_phys = oe_engine.to_physical(results.x_hat, setup.log_mask)
        # x_hat_phys has shape (n_pixels, n_params)

        # Reshape to image grid
        x_hat_img = np.array(x_hat_phys).reshape(n_rows, n_cols, n_params)

    Args:
        f_vec:      forward model f(x) -> y in retrieval space; x shape
                    (n_params,), y shape (n_obs,).  Typically ``setup.f_fit``
                    from build_inversion(), which handles fixed-param insertion
                    and the log → physical conversion internally.
        Rrs_pixels: observed spectra, shape (n_pixels, n_obs)
        noise:      measurement uncertainty — scalar std, 1-D std array (n_obs,),
                    or pre-inverted full covariance matrix (n_obs, n_obs).
                    Applied identically to every pixel.
        x_a:        prior mean in retrieval space.  Two options:
                    * 1-D array (n_params,): same prior mean for every pixel
                    * 2-D array (n_pixels, n_params): per-pixel prior mean
                    The per-pixel form is useful when you have auxiliary maps
                    (e.g. a bathymetry map for zB, or a chlorophyll climatology
                    for C_0) that constrain individual parameters spatially.
                    For log-params this should be ln(physical_prior_mean).
                    When using build_inversion(), pass setup.x_a directly.
        S_a_inv:    inverse prior covariance in retrieval space.  Two options:
                    * 2-D array (n_params, n_params): same for every pixel
                    * 3-D array (n_pixels, n_params, n_params): per-pixel
                    Use the per-pixel form to apply tighter constraints where
                    your auxiliary map is more accurate (e.g. shallow pixels
                    from a high-confidence bathymetry survey).
                    When using build_inversion(), pass setup.S_a_inv directly.
        x0:         initial state in retrieval space.  Three options:
                    * None (default): start every pixel from its prior mean x_a
                    * 1-D array (n_params,): same starting point for every pixel
                    * 2-D array (n_pixels, n_params): per-pixel starting point
        n_iter:     Gauss-Newton iterations (static Python int), default 10.
        lm_damping: LM damping factor (see solve()), default 0.  With
                    log-transformed parameters this is often unnecessary.

    Returns:
        OEResult where every field has an extra leading pixel dimension:
            x_hat  shape (n_pixels, n_params)  — retrieval space
            S_hat  shape (n_pixels, n_params, n_params)
            A      shape (n_pixels, n_params, n_params)
            dfs    shape (n_pixels,)
            chi2   shape (n_pixels,)
            J      shape (n_pixels, n_obs, n_params)
            y_hat  shape (n_pixels, n_obs)
        Apply to_physical(results.x_hat, setup.log_mask) for physical values.
    """
    n_pixels = Rrs_pixels.shape[0]

    # --- broadcast x_a to (n_pixels, n_params) ------------------------------
    x_a_arr  = jnp.asarray(x_a)
    n_params = x_a_arr.shape[-1]
    x_a_batch = (jnp.broadcast_to(x_a_arr, (n_pixels, n_params))
                 if x_a_arr.ndim == 1 else x_a_arr)

    # --- broadcast S_a_inv to (n_pixels, n_params, n_params) ----------------
    S_a_inv_arr   = jnp.asarray(S_a_inv)
    S_a_inv_batch = (jnp.broadcast_to(S_a_inv_arr, (n_pixels, n_params, n_params))
                     if S_a_inv_arr.ndim == 2 else S_a_inv_arr)

    # --- x0: default to per-pixel prior mean --------------------------------
    if x0 is None:
        x0_batch = x_a_batch
    else:
        x0_arr   = jnp.asarray(x0)
        x0_batch = (jnp.broadcast_to(x0_arr, (n_pixels, n_params))
                    if x0_arr.ndim == 1 else x0_arr)

    _solve_one = lambda y, x, xa, Sa: solve(f_vec, y, noise, x, xa, Sa,
                                            n_iter, lm_damping)
    return jax.vmap(_solve_one, in_axes=(0, 0, 0, 0))(
        Rrs_pixels, x0_batch, x_a_batch, S_a_inv_batch
    )

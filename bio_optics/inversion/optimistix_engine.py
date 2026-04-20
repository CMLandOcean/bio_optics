"""
OE inversion engine using optimistix JAX-native solvers.

Provides drop-in replacements for ``oe_engine.solve()`` and
``oe_engine.invert_pixels()`` that use
``optimistix.LevenbergMarquardt`` (default) or ``optimistix.GaussNewton``
instead of the hand-written Gauss-Newton loop in ``oe_engine``.

Key differences from oe_engine
-------------------------------
The iteration loop is ``jax.lax.while_loop`` — **not unrolled** at compile
time.  This means:

* **Much faster JIT compilation** — the computation graph has O(1) nodes
  regardless of ``max_steps``.  With ``oe_engine.solve()`` at ``n_iter=15``,
  JAX unrolls 15 copies of the Jacobian computation; here it compiles one
  loop body that XLA executes iteratively.
* **Convergence-based stopping** — the solver exits as soon as
  ``‖step‖ ≤ atol + rtol · ‖x‖``, so easy pixels don't pay for hard ones.
* **Adaptive LM damping** — ``LevenbergMarquardt`` uses a proper trust-region
  radius update; the damping in ``oe_engine.solve()`` is static.

Identical API surface
---------------------
``solve_optx()`` returns the same ``OEResult`` as ``oe_engine.solve()``.
``invert_pixels_optx()`` returns the same vmapped ``OEResult`` as
``oe_engine.invert_pixels()``.
Both functions accept ``InversionSetup`` from ``oe_engine.build_inversion()``,
so you can mix modules freely.

Formulation
-----------
The OE cost function is written as a sum of squared residuals::

    r_data_k  = (y_k − F_k(x)) / σ_k          (weighted by band noise)
    r_prior_i = (x_i − x_a,i) / σ_a,i          (prior pull)

    J(x) = ‖r_data‖² + ‖r_prior‖²

Optimistix minimises this via GN or LM, then the posterior covariance,
averaging kernel, and χ² are computed analytically at the solution (same
as ``oe_engine.solve()``).

Limitations
-----------
* Only scalar and 1-D (per-band) noise models are supported.
  Full covariance matrix noise requires Cholesky decomposition of
  S_ε^{-1} which is not yet implemented here.
* ``S_a_inv`` is assumed to be diagonal (which ``build_inversion()`` always
  produces).  Off-diagonal prior covariances are not supported.

References
----------
Kidger, P. (2023): On Neural Differential Equations.  PhD thesis, Oxford.
https://github.com/patrick-kidger/optimistix

Rodgers, C.D. (2000): Inverse Methods for Atmospheric Sounding.
World Scientific, Singapore.
"""

import numpy as np
import jax
import jax.numpy as jnp
import optimistix as optx
from typing import Callable, Optional

from bio_optics.inversion.oe_engine import (
    OEResult,
    InversionSetup,
    _build_S_eps_inv,
    to_physical,
    posterior_sigma_physical,
    bottom_fractions,
)

jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_noise_sqrt_inv(noise, n_obs: int, weights=None) -> jnp.ndarray:
    """
    Return a 1-D array of shape (n_obs,) representing sqrt(S_ε⁻¹) diagonal.

    Used to form the data-fit residual vector for optimistix:
    ``r_data = sqrt_inv * (y_obs − f(x))``.

    Args:
        noise:   scalar float or 1-D array of length n_obs.
                 Full covariance matrices are not supported.
        n_obs:   number of observation bands.
        weights: optional per-band weight array, shape (n_obs,).

    Returns:
        1-D JAX array of shape (n_obs,).
    """
    noise_jax = jnp.asarray(noise, dtype=jnp.float64)
    if noise_jax.ndim == 0:
        s = jnp.ones(n_obs, dtype=jnp.float64) / noise_jax
    elif noise_jax.ndim == 1:
        s = 1.0 / noise_jax
    else:
        raise ValueError(
            "optimistix_engine: full covariance matrix noise is not supported. "
            "Pass a scalar or a 1-D per-band noise array."
        )
    if weights is not None:
        s = s * jnp.asarray(weights, dtype=jnp.float64)
    return s


# ---------------------------------------------------------------------------
# Core solver — pure JAX, lax.while_loop-based
# ---------------------------------------------------------------------------

def solve_optx(f_vec: Callable,
               y_obs: jnp.ndarray,
               noise,
               x0: jnp.ndarray,
               x_a: jnp.ndarray,
               S_a_inv: jnp.ndarray,
               max_steps: int = 100,
               rtol: float = 1e-6,
               atol: float = 1e-6,
               use_lm: bool = False,
               weights=None,
               aux=None) -> OEResult:
    """
    OE solver using optimistix (lax.while_loop-based, convergence-stopping).

    Drop-in replacement for ``oe_engine.solve()``.  Returns the same
    ``OEResult`` namedtuple; posterior covariance, averaging kernel, and χ²
    are computed analytically at the optimistix solution.

    The OE cost function is reformulated as nonlinear least-squares::

        r = concat(
            sqrt(S_ε⁻¹) · (y − F(x)),   # (n_obs,)  data-fit residuals
            sqrt(S_a⁻¹) · (x − x_a),    # (n_fit,)  prior pull
        )
        J(x) = ‖r‖²

    Args:
        f_vec:     forward model f(x) → y; x shape (n_fit,), y shape (n_obs,).
                   Must be differentiable with jax.jacobian.
                   Typically ``setup.f_fit`` from ``oe_engine.build_inversion()``.
        y_obs:     observed spectrum, shape (n_obs,).
        noise:     measurement std — scalar or 1-D array (n_obs,).
                   Full covariance not supported (use oe_engine.solve() instead).
        x0:        initial state in retrieval space, shape (n_fit,).
        x_a:       prior mean in retrieval space, shape (n_fit,).
        S_a_inv:   diagonal inverse prior covariance, shape (n_fit, n_fit).
                   Must be diagonal; ``build_inversion()`` always produces this.
        max_steps: maximum solver iterations, default 100.
        rtol:      relative convergence tolerance, default 1e-6.
        atol:      absolute convergence tolerance, default 1e-6.
        use_lm:    if True (default) use LevenbergMarquardt; else GaussNewton.
        weights:   optional per-band weight array, shape (n_obs,).
        aux:       optional per-pixel auxiliary dict forwarded to f_vec as
                   ``f_vec(x, aux)``.  Default None.

    Returns:
        OEResult with the same fields as oe_engine.solve().  All in retrieval
        space; apply ``to_physical()`` / ``posterior_sigma_physical()`` as usual.
    """
    f = (lambda x: f_vec(x, aux)) if aux is not None else f_vec

    n_obs = y_obs.shape[0]
    n_fit = x0.shape[0]

    # sqrt(S_ε⁻¹) diagonal for the residual vector
    eps_sqrt_inv = _build_noise_sqrt_inv(noise, n_obs, weights)

    # sqrt(S_a⁻¹) diagonal — S_a_inv is diagonal by construction
    S_a_inv_diag  = jnp.diag(S_a_inv)                          # (n_fit,)
    sa_sqrt_inv   = jnp.sqrt(S_a_inv_diag)                     # = 1/sigma_a

    # Residual function for optimistix
    def residual_fn(x, args):
        del args
        r_data  = eps_sqrt_inv * (y_obs - f(x))                # (n_obs,)
        r_prior = sa_sqrt_inv  * (x - x_a)                     # (n_fit,)
        return jnp.concatenate([r_data, r_prior])               # (n_obs + n_fit,)

    solver = (optx.LevenbergMarquardt(rtol=rtol, atol=atol)
              if use_lm else
              optx.GaussNewton(rtol=rtol, atol=atol))

    sol = optx.least_squares(residual_fn, solver, x0,
                              max_steps=max_steps, throw=False)
    x_hat = sol.value

    # --- posterior diagnostics at solution (identical to oe_engine.solve) ---
    y_hat     = f(x_hat)
    J         = jax.jacobian(f)(x_hat)                          # (n_obs, n_fit)
    S_eps_inv = _build_S_eps_inv(noise, n_obs, weights)
    H         = J.T @ S_eps_inv @ J + S_a_inv
    S_hat     = jnp.linalg.inv(H)
    A         = S_hat @ J.T @ S_eps_inv @ J
    dfs       = jnp.trace(A)
    residual  = y_obs - y_hat
    chi2      = residual @ S_eps_inv @ residual / n_obs

    return OEResult(x_hat=x_hat, S_hat=S_hat, A=A, dfs=dfs, chi2=chi2,
                    J=J, y_hat=y_hat)


# ---------------------------------------------------------------------------
# Vectorised image inversion — jax.vmap over pixels
# ---------------------------------------------------------------------------

def invert_pixels_optx(f_vec: Callable,
                        Rrs_pixels: jnp.ndarray,
                        noise,
                        x_a: jnp.ndarray,
                        S_a_inv: jnp.ndarray,
                        x0: jnp.ndarray = None,
                        max_steps: int = 100,
                        rtol: float = 1e-6,
                        atol: float = 1e-6,
                        use_lm: bool = False,
                        weights=None,
                        aux_pixels=None) -> OEResult:
    """
    Batch OE inversion via jax.vmap + optimistix.

    Drop-in replacement for ``oe_engine.invert_pixels()``.  Uses
    ``solve_optx()`` per pixel; the while_loop inside each solve is
    transformed by vmap into a batched while_loop — a single loop that
    exits when all pixels in the batch have converged (or hit max_steps).

    Wrap with ``jax.jit`` for best performance::

        results = jax.jit(optimistix_engine.invert_pixels_optx)(
            setup.f_fit, Rrs_pixels, noise, setup.x_a, setup.S_a_inv,
            max_steps=100,
        )

    Args:
        f_vec:      forward model, typically ``setup.f_fit``.
        Rrs_pixels: observed spectra, shape (n_pixels, n_obs).
        noise:      scalar or 1-D noise std.
        x_a:        prior mean, shape (n_fit,) or (n_pixels, n_fit).
        S_a_inv:    inverse prior covariance, shape (n_fit, n_fit) or
                    (n_pixels, n_fit, n_fit).
        x0:         initial state, shape (n_fit,) or (n_pixels, n_fit).
                    Default None → start from x_a.
        max_steps:  maximum solver iterations per pixel, default 100.
        rtol:       relative convergence tolerance, default 1e-6.
        atol:       absolute convergence tolerance, default 1e-6.
        use_lm:     True (default) = LevenbergMarquardt; False = GaussNewton.
        weights:    optional per-band weights, shape (n_obs,).
        aux_pixels: optional per-pixel auxiliary pytree.  Each pixel slice
                    is forwarded to ``f_vec(x, aux)`` via solve_optx().

    Returns:
        OEResult with leading pixel dimension on every field (same as
        ``oe_engine.invert_pixels()``).
    """
    n_pixels = Rrs_pixels.shape[0]

    x_a_arr    = jnp.asarray(x_a)
    n_params   = x_a_arr.shape[-1]
    x_a_batch  = (jnp.broadcast_to(x_a_arr, (n_pixels, n_params))
                  if x_a_arr.ndim == 1 else x_a_arr)

    S_a_inv_arr   = jnp.asarray(S_a_inv)
    S_a_inv_batch = (jnp.broadcast_to(S_a_inv_arr, (n_pixels, n_params, n_params))
                     if S_a_inv_arr.ndim == 2 else S_a_inv_arr)

    if x0 is None:
        x0_batch = x_a_batch
    else:
        x0_arr   = jnp.asarray(x0)
        x0_batch = (jnp.broadcast_to(x0_arr, (n_pixels, n_params))
                    if x0_arr.ndim == 1 else x0_arr)

    if aux_pixels is not None:
        def _solve_one(y, x, xa, Sa, a):
            return solve_optx(f_vec, y, noise, x, xa, Sa,
                               max_steps=max_steps, rtol=rtol, atol=atol,
                               use_lm=use_lm, weights=weights, aux=a)
        return jax.vmap(_solve_one, in_axes=(0, 0, 0, 0, 0))(
            Rrs_pixels, x0_batch, x_a_batch, S_a_inv_batch, aux_pixels
        )
    else:
        def _solve_one(y, x, xa, Sa):
            return solve_optx(f_vec, y, noise, x, xa, Sa,
                               max_steps=max_steps, rtol=rtol, atol=atol,
                               use_lm=use_lm, weights=weights)
        return jax.vmap(_solve_one, in_axes=(0, 0, 0, 0))(
            Rrs_pixels, x0_batch, x_a_batch, S_a_inv_batch
        )

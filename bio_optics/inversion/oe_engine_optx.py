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
* **n_steps per pixel** — ``OEResult.num_steps`` records how many solver
  iterations were taken; exposed via ``invert_image``'s ``n_steps`` key.

Identical API surface
---------------------
``solve_optx()`` returns the same ``OEResult`` as ``oe_engine.solve()``,
with the addition of ``OEResult.num_steps``.
``invert_pixels_optx()`` returns the same vmapped ``OEResult`` as
``oe_engine.invert_pixels()``.
Both functions accept ``InversionSetup`` from ``oe_engine.build_inversion()``,
so you can mix modules freely.

Naming note
-----------
This module was renamed from ``optimistix_engine`` to ``oe_engine_optx`` to
make clear that it is the **OE** (Optimal Estimation) engine backed by
optimistix, as opposed to ``lsq_engine_optx`` which is the pure **LSQ**
(no prior) engine also backed by optimistix.

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
import lineax as lx
from typing import Callable, Dict, Optional

from bio_optics.inversion.oe_engine import (
    OEResult,
    InversionSetup,
    _build_S_eps_inv,
    _build_noise_sqrt_inv,
    to_physical,
    posterior_sigma_physical,
    bottom_fractions,
)

jax.config.update("jax_enable_x64", True)


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
    ``OEResult.num_steps`` records how many iterations were taken.

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
        use_lm:    if True use LevenbergMarquardt; else GaussNewton (default).
        weights:   optional per-band weight array, shape (n_obs,).
        aux:       optional per-pixel auxiliary dict forwarded to f_vec as
                   ``f_vec(x, aux)``.  Default None.

    Returns:
        OEResult with the same fields as oe_engine.solve(), plus num_steps.
        All in retrieval space; apply ``to_physical()`` /
        ``posterior_sigma_physical()`` as usual.
    """
    f = (lambda x: f_vec(x, aux)) if aux is not None else f_vec

    n_obs = y_obs.shape[0]
    n_fit = x0.shape[0]

    eps_sqrt_inv = _build_noise_sqrt_inv(noise, n_obs, weights)

    S_a_inv_diag  = jnp.diag(S_a_inv)
    sa_sqrt_inv   = jnp.sqrt(S_a_inv_diag)

    def residual_fn(x, args):
        del args
        r_data  = eps_sqrt_inv * (y_obs - f(x))
        r_prior = sa_sqrt_inv  * (x - x_a)
        return jnp.concatenate([r_data, r_prior])

    _lin = lx.AutoLinearSolver(well_posed=False)
    solver = (optx.LevenbergMarquardt(rtol=rtol, atol=atol, linear_solver=_lin)
              if use_lm else
              optx.GaussNewton(rtol=rtol, atol=atol, linear_solver=_lin))

    sol = optx.least_squares(residual_fn, solver, x0,
                              max_steps=max_steps, throw=False)
    x_hat = sol.value

    y_hat     = f(x_hat)
    J         = jax.jacfwd(f)(x_hat)
    S_eps_inv = _build_S_eps_inv(noise, n_obs, weights)
    H         = J.T @ S_eps_inv @ J + S_a_inv
    S_hat     = jnp.linalg.inv(H)
    A         = S_hat @ J.T @ S_eps_inv @ J
    dfs       = jnp.trace(A)
    G         = S_hat @ J.T @ S_eps_inv
    H_info    = 0.5 * (jnp.linalg.slogdet(H)[1] - jnp.linalg.slogdet(S_a_inv)[1])
    residual  = y_obs - y_hat
    chi2      = residual @ S_eps_inv @ residual / n_obs

    return OEResult(x_hat=x_hat, S_hat=S_hat, A=A, dfs=dfs, chi2=chi2,
                    J=J, y_hat=y_hat, H_info=H_info, G=G,
                    num_steps=sol.stats['num_steps'])


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

        results = jax.jit(oe_engine_optx.invert_pixels_optx)(
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
        use_lm:     True = LevenbergMarquardt; False = GaussNewton (default).
        weights:    optional per-band weights, shape (n_obs,).
        aux_pixels: optional per-pixel auxiliary pytree.

    Returns:
        OEResult with leading pixel dimension on every field.
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


# ---------------------------------------------------------------------------
# Module-level JIT caches — compiled once per process, reused across calls.
# ---------------------------------------------------------------------------

_invert_pixels_optx_jit = jax.jit(
    invert_pixels_optx,
    static_argnums=(0,),
    static_argnames=('max_steps', 'rtol', 'atol', 'use_lm'),
)


def _solve_postprocess(f_fit, Rrs, noise, x_a, S_a_inv, log_mask,
                        max_steps, rtol, atol, use_lm, weights, aux_pixels):
    """Fused solve + physical-space conversion — single XLA program.

    Putting to_physical / posterior_sigma_physical / jnp.diagonal inside the
    same jit boundary as the solve avoids separate GPU kernel launches and
    device→host synchronisations between the two phases, which caused a
    significant regression on GPU compared to the old single-lambda JIT.
    """
    res = invert_pixels_optx(f_fit, Rrs, noise, x_a, S_a_inv,
                             max_steps=max_steps, rtol=rtol, atol=atol,
                             use_lm=use_lm, weights=weights, aux_pixels=aux_pixels)
    x_hat_phys = to_physical(res.x_hat, log_mask)
    sigma_phys = posterior_sigma_physical(res.S_hat, x_hat_phys, log_mask)
    A_diag     = jnp.diagonal(res.A, axis1=-2, axis2=-1)
    return x_hat_phys, sigma_phys, A_diag, res.chi2, res.H_info, res.y_hat, res.num_steps


_solve_postprocess_jit = jax.jit(
    _solve_postprocess,
    static_argnums=(0,),
    static_argnames=('max_steps', 'rtol', 'atol', 'use_lm'),
)


# ---------------------------------------------------------------------------
# Tile-level function — one Dask task per tile
# ---------------------------------------------------------------------------

def invert_tile_optx(
    Rrs_tile: np.ndarray,
    f_fit,
    x_a: np.ndarray,
    S_a_inv: np.ndarray,
    log_mask: np.ndarray,
    noise,
    weights: Optional[np.ndarray] = None,
    max_steps: int = 100,
    rtol: float = 1e-6,
    atol: float = 1e-6,
    use_lm: bool = False,
    store_y_hat: bool = False,
    store_gain: bool = False,
    aux_tile=None,
    store_chi2_spectral: bool = False,
):
    """
    Run optimistix OE inversion on a single pixel tile.

    Returns tuple: (x_hat_phys, sigma_phys, A_diag, chi2, n_steps, H_info)
    with y_hat appended when store_y_hat=True, G when store_gain=True,
    and chi2_spectral when store_chi2_spectral=True.
    """
    valid_mask  = np.isfinite(Rrs_tile).all(axis=-1)
    n_tile      = Rrs_tile.shape[0]
    n_fit       = np.asarray(x_a).shape[-1]
    n_obs_tile  = Rrs_tile.shape[-1]
    has_invalid = not valid_mask.all()

    if has_invalid and not valid_mask.any():
        nan_fit  = np.full((n_tile, n_fit), np.nan)
        nan_chi  = np.full(n_tile, np.nan)
        nan_step = np.full(n_tile, -1, dtype=np.int32)
        out = (nan_fit, nan_fit.copy(), nan_fit.copy(), nan_chi, nan_step)
        if store_y_hat:
            out = out + (np.full((n_tile, n_obs_tile), np.nan),)
        return out

    if has_invalid:
        Rrs_padded = Rrs_tile.copy()
        Rrs_padded[~valid_mask] = Rrs_tile[valid_mask][0]
    else:
        Rrs_padded = Rrs_tile

    x_a_np     = np.asarray(x_a)
    S_a_inv_np = np.asarray(S_a_inv)

    Rrs_jax      = jnp.asarray(Rrs_padded, dtype=jnp.float64)
    x_a_jax      = jnp.asarray(x_a_np,     dtype=jnp.float64)
    S_a_inv_jax  = jnp.asarray(S_a_inv_np, dtype=jnp.float64)
    log_mask_jax = jnp.asarray(log_mask,    dtype=jnp.float64)
    weights_jax  = jnp.asarray(weights, dtype=jnp.float64) if weights is not None else None

    if aux_tile is not None:
        if isinstance(aux_tile, dict):
            aux_jax = {k: jnp.asarray(v, dtype=jnp.float64) for k, v in aux_tile.items()}
        else:
            aux_jax = jnp.asarray(aux_tile, dtype=jnp.float64)
    else:
        aux_jax = None

    results = _invert_pixels_optx_jit(
        f_fit, Rrs_jax, noise, x_a_jax, S_a_inv_jax,
        max_steps=max_steps, rtol=rtol, atol=atol,
        use_lm=use_lm, weights=weights_jax,
        aux_pixels=aux_jax,
    )

    x_hat_phys = to_physical(results.x_hat, log_mask_jax)
    sigma_phys = posterior_sigma_physical(results.S_hat, x_hat_phys, log_mask_jax)
    A_diag     = jnp.diagonal(results.A, axis1=-2, axis2=-1)

    x_hat_np    = np.array(x_hat_phys)
    sigma_np    = np.array(sigma_phys)
    A_diag_np   = np.array(A_diag)
    chi2_np     = np.array(results.chi2)
    n_steps_np  = np.array(results.num_steps, dtype=np.int32)
    H_info_np   = np.array(results.H_info)

    if has_invalid:
        x_hat_np[~valid_mask]   = np.nan
        sigma_np[~valid_mask]   = np.nan
        A_diag_np[~valid_mask]  = np.nan
        chi2_np[~valid_mask]    = np.nan
        n_steps_np[~valid_mask] = -1
        H_info_np[~valid_mask]  = np.nan

    out = (x_hat_np, sigma_np, A_diag_np, chi2_np, n_steps_np, H_info_np)
    if store_y_hat:
        y_hat_np = np.array(results.y_hat)
        if has_invalid:
            y_hat_np[~valid_mask] = np.nan
        out = out + (y_hat_np,)
    if store_gain:
        G_np = np.array(results.G)
        if has_invalid:
            G_np[~valid_mask] = np.nan
        out = out + (G_np,)
    if store_chi2_spectral:
        chi2_sp_np = np.mean(np.square(Rrs_padded - np.array(results.y_hat)), axis=-1)
        if has_invalid:
            chi2_sp_np[~valid_mask] = np.nan
        out = out + (chi2_sp_np,)
    return out


# ---------------------------------------------------------------------------
# Image-level inversion — Layer-1 standard interface
# ---------------------------------------------------------------------------

def invert_image_optx(
    spectra,
    setup: InversionSetup,
    noise,
    max_steps: int = 100,
    rtol: float = 1e-6,
    atol: float = 1e-6,
    use_lm: bool = False,
    store_y_hat: bool = False,
    store_gain: bool = False,
    store_chi2_spectral: bool = False,
    x_a_image: Optional[np.ndarray] = None,
    S_a_inv_image: Optional[np.ndarray] = None,
    aux_image=None,
    **kwargs,
) -> Dict[str, object]:
    """Batch OE inversion via vmap + optimistix — Layer-1 invert_fn interface.

    Drop-in replacement for ``oe_engine.invert_image`` using ``solve_optx()``
    (lax.while_loop, convergence-stopping) instead of the fixed-iteration
    Gauss-Newton loop.  Tiling and parallelism are delegated to Layer-2 engines
    (``dask_engine``, ``superpixel_engine``).

    Formerly contained internal Dask tiling; ``tile_size`` / ``scheduler``
    arguments are silently absorbed via ``**kwargs`` for backwards compatibility.

    Args:
        spectra:             (n_spectra, n_obs) or (n_rows, n_cols, n_obs).
        setup:               InversionSetup from ``oe_engine.build_inversion()``.
        noise:               scalar or 1-D noise std.
        max_steps:           solver iteration cap per pixel, default 100.
        rtol:                relative convergence tolerance, default 1e-6.
        atol:                absolute convergence tolerance, default 1e-6.
        use_lm:              True = LevenbergMarquardt; False = GaussNewton (default).
        store_y_hat:         include simulated spectra in output.
        store_gain:          include gain matrix G in output.
        store_chi2_spectral: include ``chi2_spectral`` — mean squared difference
                             between observed and simulated spectrum, no noise
                             weighting or prior term.
        x_a_image:           per-spectra prior mean override.
        S_a_inv_image:       per-spectra S_a_inv override.
        aux_image:           per-spectra auxiliary pytree.
        **kwargs:            silently absorbed (e.g. ``tile_size``, ``scheduler``).

    Returns:
        dict with keys ``x_hat``, ``sigma``, ``A_diag``, ``chi2``, ``n_steps``,
        ``H_info``, ``fit_names``, and optionally ``y_hat``, ``G``,
        ``chi2_spectral``.
    """
    spectra_arr   = np.asarray(spectra)
    spatial_shape = None

    if spectra_arr.ndim == 3:
        n_rows, n_cols, n_obs = spectra_arr.shape
        spatial_shape = (n_rows, n_cols)
        Rrs_flat = spectra_arr.reshape(-1, n_obs)
    else:
        Rrs_flat = spectra_arr
        n_obs    = Rrs_flat.shape[1]

    n_pixels = Rrs_flat.shape[0]
    n_fit    = len(setup.fit_names)
    log_mask = jnp.asarray(setup.log_mask, dtype=jnp.float64)

    valid       = np.isfinite(Rrs_flat).all(axis=-1)
    has_invalid = not valid.all()
    Rrs_padded  = Rrs_flat.copy()
    if has_invalid and valid.any():
        Rrs_padded[~valid] = Rrs_flat[valid][0]

    x_hat_np   = np.full((n_pixels, n_fit), np.nan)
    sigma_np   = np.full((n_pixels, n_fit), np.nan)
    A_diag_np  = np.full((n_pixels, n_fit), np.nan)
    chi2_np    = np.full(n_pixels, np.nan)
    n_steps_np = np.full(n_pixels, -1, dtype=np.int32)
    H_info_np  = np.full(n_pixels, np.nan)
    y_hat_np   = None

    if valid.any():
        x_a_jax    = jnp.asarray(x_a_image    if x_a_image    is not None else setup.x_a,     dtype=jnp.float64)
        Sa_inv_jax = jnp.asarray(S_a_inv_image if S_a_inv_image is not None else setup.S_a_inv, dtype=jnp.float64)
        weights_jax = jnp.asarray(setup.weights, dtype=jnp.float64) if setup.weights is not None else None

        if aux_image is not None:
            n_spatial = len(spatial_shape) if spatial_shape is not None else 0
            def _flat(a):
                a = np.asarray(a)
                return a.reshape(n_pixels, *a.shape[n_spatial:])
            aux_jax = ({k: jnp.asarray(_flat(v), dtype=jnp.float64) for k, v in aux_image.items()}
                       if isinstance(aux_image, dict)
                       else jnp.asarray(_flat(aux_image), dtype=jnp.float64))
        else:
            aux_jax = None

        if store_gain:
            # Full OEResult needed for G — use the solve-only JIT and post-process
            # in Python.  store_gain is a diagnostic flag so the extra kernel
            # launches here are acceptable.
            res = _invert_pixels_optx_jit(
                setup.f_fit,
                jnp.asarray(Rrs_padded, dtype=jnp.float64),
                noise, x_a_jax, Sa_inv_jax,
                max_steps=max_steps, rtol=rtol, atol=atol,
                use_lm=use_lm, weights=weights_jax,
                aux_pixels=aux_jax,
            )
            x_hat_phys = to_physical(res.x_hat, log_mask)
            sigma_phys = posterior_sigma_physical(res.S_hat, x_hat_phys, log_mask)
            A_diag     = jnp.diagonal(res.A, axis1=-2, axis2=-1)
            _chi2    = res.chi2
            _H_info  = res.H_info
            _y_hat   = res.y_hat
            _n_steps = res.num_steps
            G_np = np.full((n_pixels, n_fit, n_obs), np.nan)
            G_np[valid] = np.array(res.G)[valid]
        else:
            # Fused path: solve + post-processing in one XLA program.
            x_hat_phys, sigma_phys, A_diag, _chi2, _H_info, _y_hat, _n_steps = \
                _solve_postprocess_jit(
                    setup.f_fit,
                    jnp.asarray(Rrs_padded, dtype=jnp.float64),
                    noise, x_a_jax, Sa_inv_jax, log_mask,
                    max_steps=max_steps, rtol=rtol, atol=atol,
                    use_lm=use_lm, weights=weights_jax,
                    aux_pixels=aux_jax,
                )

        x_hat_np[valid]   = np.array(x_hat_phys)[valid]
        sigma_np[valid]   = np.array(sigma_phys)[valid]
        A_diag_np[valid]  = np.array(A_diag)[valid]
        chi2_np[valid]    = np.array(_chi2)[valid]
        n_steps_np[valid] = np.array(_n_steps, dtype=np.int32)[valid]
        H_info_np[valid]  = np.array(_H_info)[valid]

        if store_y_hat or store_chi2_spectral:
            y_hat_np = np.full((n_pixels, n_obs), np.nan)
            y_hat_np[valid] = np.array(_y_hat)[valid]

    def _reshape(arr, *extra):
        if spatial_shape is not None:
            return arr.reshape(*spatial_shape, *extra)
        return arr

    out: Dict[str, object] = {
        'x_hat':     _reshape(x_hat_np, n_fit),
        'sigma':     _reshape(sigma_np, n_fit),
        'A_diag':    _reshape(A_diag_np, n_fit),
        'chi2':      _reshape(chi2_np),
        'n_steps':   _reshape(n_steps_np),
        'H_info':    _reshape(H_info_np),
        'fit_names': list(setup.fit_names),
    }
    if store_y_hat and y_hat_np is not None:
        out['y_hat'] = _reshape(y_hat_np, n_obs)
    if store_gain and valid.any():
        out['G'] = _reshape(G_np, n_fit, n_obs)
    if store_chi2_spectral and y_hat_np is not None:
        chi2_sp = np.full(n_pixels, np.nan)
        chi2_sp[valid] = np.mean(np.square(Rrs_flat[valid] - y_hat_np[valid]), axis=-1)
        out['chi2_spectral'] = _reshape(chi2_sp)

    return out


# Alias for standard Layer-1 invert_fn interface
invert_image = invert_image_optx

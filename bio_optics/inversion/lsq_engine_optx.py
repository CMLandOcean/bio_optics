"""
Pure weighted least-squares image inversion using JAX + optimistix.

Minimises only the spectral residual — no OE prior term:

    J(x) = || σ_ε⁻¹ (y_obs − F(x)) ||²

Equivalent to lmfit.minimize but vectorised over pixels via jax.vmap.
Use this when the OE prior causes spatial artefacts (uniform prior pulls
pixels toward the same mean across a spatially heterogeneous scene).
Tiling is delegated to Layer-2 engines (dask_engine, superpixel_engine).

Naming note
-----------
This module is named ``lsq_engine_optx`` to indicate that it is the
**LSQ** (pure least-squares, no prior) engine backed by **optimistix**,
symmetrically with ``oe_engine_optx`` which is the OE (with prior) engine
also backed by optimistix.
"""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
import optimistix as optx
import lineax as lx

from bio_optics.inversion.oe_engine import InversionSetup, to_physical, _build_noise_sqrt_inv


def invert_image(
    Rrs,
    setup: InversionSetup,
    noise,
    max_steps: int = 100,
    use_lm: bool = True,
    rtol: float = 1e-6,
    atol: float = 1e-6,
    x_a_image: 'np.ndarray | None' = None,
    store_chi2_spectral: bool = False,
    **kwargs,
) -> dict:
    """Pure weighted least-squares image inversion using JAX + optimistix.

    Layer-1 invert_fn: processes the full input batch via jax.vmap.  Tiling
    and parallelism are delegated to Layer-2 engines (``dask_engine``).
    ``tile_size`` is silently absorbed via ``**kwargs`` for backwards compat.

    Parameters
    ----------
    Rrs                 : (n_rows, n_cols, n_obs) or (n_pixels, n_obs)
    setup               : InversionSetup from oe_engine.build_inversion()
    noise               : scalar or (n_obs,) per-pixel noise in sr⁻¹
    max_steps           : maximum solver iterations per pixel
    use_lm              : True → LevenbergMarquardt (default); False → GaussNewton
    rtol/atol           : convergence thresholds (solver exits early when met)
    x_a_image           : (n_pixels, n_fit) or (n_rows, n_cols, n_fit) per-pixel
                          starting values in retrieval space.  When None every
                          pixel starts from setup.x_a.
    store_chi2_spectral : if True include ``chi2_spectral`` — mean squared
                          difference between observed and simulated spectrum,
                          no noise weighting (complements noise-normalised ``chi2``).

    Returns
    -------
    dict with:
        x_hat           (n_rows, n_cols, n_fit)  physical space
        chi2            (n_rows, n_cols)          noise-normalised spectral chi2
        n_steps         (n_rows, n_cols)          solver iterations per pixel
        fit_names       list[str]
        chi2_spectral   (n_rows, n_cols)          only when store_chi2_spectral=True
    """
    Rrs_arr = np.asarray(Rrs)
    if Rrs_arr.ndim == 3:
        n_rows, n_cols, n_obs = Rrs_arr.shape
        Rrs_flat = Rrs_arr.reshape(-1, n_obs)
    else:
        n_rows = n_cols = None
        Rrs_flat = Rrs_arr
        n_obs = Rrs_flat.shape[1]

    n_pixels = Rrs_flat.shape[0]
    n_fit    = len(setup.fit_names)

    eps_sqrt_inv = _build_noise_sqrt_inv(noise, n_obs)
    log_mask     = jnp.array(setup.log_mask, dtype=jnp.float64)

    if x_a_image is not None:
        x0_flat = np.asarray(x_a_image, dtype=np.float64).reshape(n_pixels, n_fit)
    else:
        x0_flat = np.tile(np.asarray(setup.x_a, dtype=np.float64), (n_pixels, 1))

    def residual_fn(x, y_obs):
        return eps_sqrt_inv * (y_obs - setup.f_fit(x))

    _lin   = lx.AutoLinearSolver(well_posed=False)
    solver = (
        optx.LevenbergMarquardt(rtol=rtol, atol=atol, linear_solver=_lin)
        if use_lm else
        optx.GaussNewton(rtol=rtol, atol=atol, linear_solver=_lin)
    )

    def invert_pixel(y_obs, x0_px):
        sol           = optx.least_squares(residual_fn, solver, x0_px,
                                           args=y_obs, max_steps=max_steps, throw=False)
        x_phys        = to_physical(sol.value, log_mask)
        res           = residual_fn(sol.value, y_obs)
        chi2          = jnp.sum(jnp.square(res)) / n_obs
        chi2_spectral = jnp.mean(jnp.square(res / eps_sqrt_inv))
        return x_phys, chi2, sol.stats['num_steps'], chi2_spectral

    x_hat_all   = np.full((n_pixels, n_fit), np.nan)
    chi2_all    = np.full(n_pixels, np.nan)
    n_steps_all = np.full(n_pixels, -1, dtype=np.int32)
    chi2_sp_all = np.full(n_pixels, np.nan)

    valid = np.isfinite(Rrs_flat).all(axis=-1)
    if valid.any():
        Rrs_padded = Rrs_flat.copy()
        if not valid.all():
            Rrs_padded[~valid] = Rrs_flat[valid][0]

        x_tile, chi2_tile, steps_tile, chi2_sp_tile = jax.jit(jax.vmap(invert_pixel))(
            jnp.asarray(Rrs_padded, dtype=jnp.float64),
            jnp.asarray(x0_flat,   dtype=jnp.float64),
        )
        x_hat_all[valid]   = np.array(x_tile)[valid]
        chi2_all[valid]    = np.array(chi2_tile)[valid]
        n_steps_all[valid] = np.array(steps_tile, dtype=np.int32)[valid]
        chi2_sp_all[valid] = np.array(chi2_sp_tile)[valid]

    if n_rows is not None:
        x_hat_all   = x_hat_all.reshape(n_rows, n_cols, n_fit)
        chi2_all    = chi2_all.reshape(n_rows, n_cols)
        n_steps_all = n_steps_all.reshape(n_rows, n_cols)
        chi2_sp_all = chi2_sp_all.reshape(n_rows, n_cols)

    out = {
        'x_hat':     x_hat_all,
        'chi2':      chi2_all,
        'n_steps':   n_steps_all,
        'fit_names': list(setup.fit_names),
    }
    if store_chi2_spectral:
        out['chi2_spectral'] = chi2_sp_all
    return out

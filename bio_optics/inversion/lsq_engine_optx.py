"""
Pure weighted least-squares image inversion using JAX + optimistix.

Minimises only the spectral residual — no OE prior term:

    J(x) = || σ_ε⁻¹ (y_obs − F(x)) ||²

Equivalent to lmfit.minimize but vectorised over pixels via jax.vmap.
Use this when the OE prior causes spatial artefacts (uniform prior pulls
pixels toward the same mean across a spatially heterogeneous scene).

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
    tile_size: int = 4096,
    use_lm: bool = True,
    rtol: float = 1e-6,
    atol: float = 1e-6,
    x_a_image: 'np.ndarray | None' = None,
) -> dict:
    """Pure weighted least-squares image inversion using JAX + optimistix.

    Parameters
    ----------
    Rrs       : (n_rows, n_cols, n_obs) or (n_pixels, n_obs)
    setup     : InversionSetup from oe_engine.build_inversion()
    noise     : scalar or (n_obs,) per-pixel noise in sr⁻¹
    max_steps : maximum solver iterations per pixel
    tile_size : pixels per JIT-compiled tile (controls memory)
    use_lm    : True → LevenbergMarquardt (default); False → GaussNewton
    rtol/atol : convergence thresholds (solver exits early when met)
    x_a_image : (n_pixels, n_fit) or (n_rows, n_cols, n_fit) per-pixel starting
                values in retrieval space (log-space for log-params).  When None
                every pixel starts from setup.x_a.  Analogous to the x_a_image
                parameter in oe_engine.invert_image — for OE it sets the prior mean, for
                LSQ it sets the solver starting point.

    Returns
    -------
    dict with:
        x_hat      (n_rows, n_cols, n_fit)  physical space
        chi2       (n_rows, n_cols)
        n_steps    (n_rows, n_cols)         solver iterations per pixel
        fit_names  list[str]
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

    # Per-pixel starting values: broadcast scalar x_a or use provided image
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
        sol    = optx.least_squares(residual_fn, solver, x0_px,
                                    args=y_obs, max_steps=max_steps, throw=False)
        x_phys = to_physical(sol.value, log_mask)
        res    = residual_fn(sol.value, y_obs)
        chi2   = jnp.sum(jnp.square(res)) / n_obs
        return x_phys, chi2, sol.stats['num_steps']

    @jax.jit
    def invert_tile(tile, x0_tile):
        return jax.vmap(invert_pixel)(tile, x0_tile)

    x_hat_all   = np.full((n_pixels, n_fit), np.nan)
    chi2_all    = np.full(n_pixels, np.nan)
    n_steps_all = np.full(n_pixels, -1, dtype=np.int32)

    for i, start in enumerate(range(0, n_pixels, tile_size)):
        end   = min(start + tile_size, n_pixels)
        tile  = Rrs_flat[start:end]
        valid = np.isfinite(tile).all(axis=-1)

        if not valid.any():
            continue

        tile_padded = tile.copy()
        if not valid.all():
            tile_padded[~valid] = tile[valid][0]

        x0_tile = jnp.asarray(x0_flat[start:end], dtype=jnp.float64)
        x_tile, chi2_tile, steps_tile = invert_tile(
            jnp.asarray(tile_padded, dtype=jnp.float64), x0_tile
        )
        x_hat_all[start:end][valid]   = np.array(x_tile)[valid]
        chi2_all[start:end][valid]    = np.array(chi2_tile)[valid]
        n_steps_all[start:end][valid] = np.array(steps_tile, dtype=np.int32)[valid]

        if (i + 1) % 10 == 0 or end == n_pixels:
            print(f'  {end}/{n_pixels} ({100*end/n_pixels:.0f}%)')

    if n_rows is not None:
        x_hat_all   = x_hat_all.reshape(n_rows, n_cols, n_fit)
        chi2_all    = chi2_all.reshape(n_rows, n_cols)
        n_steps_all = n_steps_all.reshape(n_rows, n_cols)

    return {
        'x_hat':     x_hat_all,
        'chi2':      chi2_all,
        'n_steps':   n_steps_all,
        'fit_names': list(setup.fit_names),
    }

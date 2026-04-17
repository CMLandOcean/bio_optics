"""
Dask-parallel OE inversion engine for full EO images.

Wraps ``oe_engine.invert_pixels()`` in a Dask tiling loop so that arbitrarily
large images can be inverted without holding all pixels in RAM simultaneously.

Architecture
------------
::

    Dask: splits the image into pixel tiles (e.g. 65 536 pixels = 256×256)
      ↓
    For each tile:
      JAX jax.jit(invert_pixels): vmapped Gauss-Newton over the tile pixels
      → returns x_hat (retrieval space), S_hat, A, chi2, y_hat
      ↓
    Convert to physical space: to_physical() / posterior_sigma_physical()
      ↓
    Stack tile results → full image arrays (numpy, optionally saved to Zarr)

The tile function ``invert_tile()`` is dispatched by Dask via cloudpickle,
which handles Python closures including the ``f_fit`` closure from
``InversionSetup``.  JAX's JIT cache is process-local, so the first tile
call in each worker process triggers one trace/compile step; subsequent tiles
on the same worker reuse the compiled XLA program.

NaN handling
------------
Pixels that contain NaN values in ``Rrs`` (e.g. cloud-masked pixels) will
produce NaN in ``x_hat`` and NaN chi2.  Filter them afterward using a validity
mask or the chi2 map.

Recommended output fields per pixel
------------------------------------
=============== ============= ======================================
 Key             Shape         Description
=============== ============= ======================================
 x_hat           (n_fit,)      Physical retrieved parameters
 sigma           (n_fit,)      Physical posterior σ (sqrt S_hat diag)
 A_diag          (n_fit,)      Averaging kernel diagonal (data fraction)
 chi2            scalar        Goodness of fit (≈1 = good)
 y_hat           (n_obs,)      Simulated spectrum at solution (optional)
=============== ============= ======================================

Store the full ``S_hat`` or Jacobian per pixel only for targeted offline
analysis (see the OE conceptual guide for the storage trade-off discussion).

Typical usage
-------------
::

    from bio_optics.inversion import oe_engine, dask_oe_engine

    # --- build InversionSetup once on the main process ----------------------
    setup = oe_engine.build_inversion(
        params_w, f_vec_w, sigma_a_w,
        log_params=['C_0', 'C_Y', 'C_Mie', 'zB'],
    )

    # --- invert full image --------------------------------------------------
    results = dask_oe_engine.invert_image(
        Rrs_image,        # shape (n_rows, n_cols, n_obs) numpy or dask array
        setup,
        noise=0.001,
        n_iter=10,
        tile_size=65536,  # pixels per Dask task (≈256×256)
        scheduler='synchronous',
    )

    x_hat_img  = results['x_hat']   # (n_rows, n_cols, n_fit) numpy array
    sigma_img  = results['sigma']   # (n_rows, n_cols, n_fit)
    A_diag_img = results['A_diag']  # (n_rows, n_cols, n_fit)
    chi2_img   = results['chi2']    # (n_rows, n_cols)

    # --- save to Zarr -------------------------------------------------------
    import zarr
    store = zarr.open('retrieval.zarr', mode='w')
    store['x_hat']  = x_hat_img
    store['sigma']  = sigma_img
    store['A_diag'] = A_diag_img
    store['chi2']   = chi2_img

Scheduler notes
---------------
* ``'synchronous'`` (default) — single-threaded, predictable, easiest to
  debug.  JAX parallelises the computation within each tile via XLA; no
  Dask-level parallelism.  Recommended for GPU-backed JAX.
* ``'threads'`` — each tile runs in its own Python thread.  JAX releases the
  GIL for XLA operations, so tiles can overlap on CPU.  Less useful when JAX
  is GPU-backed (one device serialises anyway).
* ``'synchronous'`` with a Dask Distributed cluster — submit tasks to a
  cluster; each worker process gets its own JAX/XLA session and compiles once.
"""

from __future__ import annotations

import numpy as np
import dask
from typing import Optional, Dict

from . import oe_engine
from .oe_engine import InversionSetup


# ---------------------------------------------------------------------------
# Module-level JIT-compiled invert_pixels
# ---------------------------------------------------------------------------
# f_fit is marked static (static_argnums=0) so JAX retraces only when the
# forward function changes, not for every tile.  n_iter and lm_damping are
# plain Python int/float values that are also specialised at trace time
# (different values produce different cached programs).
# ---------------------------------------------------------------------------

import jax as _jax

_invert_pixels_jit = _jax.jit(oe_engine.invert_pixels, static_argnums=(0,))


# ---------------------------------------------------------------------------
# Tile-level function — one Dask task per tile
# ---------------------------------------------------------------------------

def invert_tile(
    Rrs_tile: np.ndarray,
    f_fit,
    x_a: np.ndarray,
    S_a_inv: np.ndarray,
    log_mask: np.ndarray,
    noise,
    weights: Optional[np.ndarray] = None,
    n_iter: int = 10,
    lm_damping: float = 0.0,
    store_y_hat: bool = False,
):
    """
    Run Gauss-Newton OE inversion on a single pixel tile.

    This is the unit of work dispatched to each Dask task.  It calls the
    module-level JIT-compiled ``invert_pixels()`` on the tile, converts
    retrieval-space outputs to physical space, and returns plain NumPy arrays
    that Dask can serialise across processes.

    JAX traces and compiles ``invert_pixels`` on the first call in each
    process (or when the tile shape changes).  Subsequent calls in the same
    process reuse the compiled XLA program.

    Args:
        Rrs_tile:     observed spectra for this tile, shape (n_pixels, n_obs).
        f_fit:        projected forward function from ``InversionSetup.f_fit``.
                      Dask serialises this closure with cloudpickle.
        x_a:          prior mean in retrieval space, shape (n_fit,).
                      Pass ``np.array(setup.x_a)``.
        S_a_inv:      inverse prior covariance, shape (n_fit, n_fit).
                      Pass ``np.array(setup.S_a_inv)``.
        log_mask:     binary log-transform mask, shape (n_fit,).
                      Pass ``np.array(setup.log_mask)``.
        noise:        measurement uncertainty — scalar std, 1-D per-band std
                      (n_obs,), or pre-inverted covariance (n_obs, n_obs).
                      Applied identically to every pixel in the tile.
        weights:      optional per-band weight array, shape (n_obs,).
                      Pass ``np.array(setup.weights)`` or None.
        n_iter:       Gauss-Newton iterations, default 10.
        lm_damping:   LM damping factor, default 0.
        store_y_hat:  if True include the simulated spectra in the output.
                      Default False (saves memory and return bandwidth).

    Returns:
        Tuple of NumPy arrays:

        * ``x_hat_physical``  (n_pixels, n_fit)  — physical retrieved values
        * ``sigma_physical``  (n_pixels, n_fit)  — posterior σ in physical space
        * ``A_diagonal``      (n_pixels, n_fit)  — averaging kernel diagonal
        * ``chi2``            (n_pixels,)         — chi-squared per band
        * ``y_hat``           (n_pixels, n_obs)   — only if ``store_y_hat=True``
    """
    import jax.numpy as jnp

    Rrs_jax      = jnp.asarray(Rrs_tile,  dtype=jnp.float64)
    x_a_jax      = jnp.asarray(x_a,       dtype=jnp.float64)
    S_a_inv_jax  = jnp.asarray(S_a_inv,   dtype=jnp.float64)
    log_mask_jax = jnp.asarray(log_mask,  dtype=jnp.float64)
    weights_jax  = jnp.asarray(weights,   dtype=jnp.float64) if weights is not None else None

    results = _invert_pixels_jit(
        f_fit, Rrs_jax, noise, x_a_jax, S_a_inv_jax,
        n_iter=n_iter, lm_damping=lm_damping, weights=weights_jax,
    )

    x_hat_phys = oe_engine.to_physical(results.x_hat, log_mask_jax)
    sigma_phys = oe_engine.posterior_sigma_physical(results.S_hat, x_hat_phys, log_mask_jax)
    A_diag     = jnp.diagonal(results.A, axis1=-2, axis2=-1)

    out = (
        np.array(x_hat_phys),
        np.array(sigma_phys),
        np.array(A_diag),
        np.array(results.chi2),
    )
    if store_y_hat:
        out = out + (np.array(results.y_hat),)
    return out


# ---------------------------------------------------------------------------
# Image-level convenience function
# ---------------------------------------------------------------------------

def invert_image(
    Rrs: np.ndarray,
    setup: InversionSetup,
    noise,
    n_iter: int = 10,
    lm_damping: float = 0.0,
    tile_size: int = 65536,
    store_y_hat: bool = False,
    scheduler: str = 'synchronous',
) -> Dict[str, object]:
    """
    Tile-parallel OE inversion for a full EO image using Dask.

    Splits the image into pixel tiles, dispatches each tile as a Dask task
    calling ``invert_tile()``, and assembles the per-tile outputs into
    full image-shaped NumPy arrays.

    The ``InversionSetup`` is built once on the calling process.  Its
    ``f_fit`` closure is serialised by Dask via cloudpickle, so each worker
    or thread receives a picklable copy of the projected forward function.

    Args:
        Rrs:          image of observed spectra.  Accepted shapes:

                      * ``(n_rows, n_cols, n_obs)`` — spatial image array
                      * ``(n_pixels, n_obs)`` — pre-flattened pixel stack

                      Values should be in sr⁻¹ (remote sensing reflectance).
                      NaN pixels (e.g. cloud-masked) propagate through as NaN
                      in the output; filter them afterwards with chi2 or a
                      validity mask.
        setup:        ``InversionSetup`` from ``oe_engine.build_inversion()``.
                      Contains the projected forward function, retrieval-space
                      prior arrays, log_mask, and optional per-band weights.
        noise:        measurement uncertainty — scalar std, 1-D per-band std
                      array (n_obs,), or pre-inverted covariance matrix
                      (n_obs, n_obs).  Applied identically to every pixel.
        n_iter:       Gauss-Newton iterations per tile, default 10.
        lm_damping:   LM damping factor (see ``oe_engine.solve()``), default 0.
                      With log-transformed parameters this is often unnecessary.
        tile_size:    number of pixels per Dask task, default 65536 (≈256×256).
                      Larger tiles → fewer tasks, less scheduling overhead but
                      more peak memory per task.  Reduce if GPU RAM is limited.
                      The last tile may be smaller than ``tile_size``.
        store_y_hat:  if True, the returned dict includes a ``'y_hat'`` key
                      with the simulated spectra at the solution, shape
                      (…, n_obs).  Default False (saves memory).
        scheduler:    Dask scheduler passed to ``dask.compute()``.
                      ``'synchronous'`` (default) — single-threaded, easiest
                      to debug; JAX parallelises within each tile via XLA.
                      ``'threads'`` — overlaps tiles using Python threads
                      (JAX releases the GIL for XLA ops).
                      ``'distributed'`` — requires a running Dask cluster.

    Returns:
        dict with the following keys:

        ``'x_hat'``
            Physical retrieved parameters,
            shape (n_rows, n_cols, n_fit) or (n_pixels, n_fit).

        ``'sigma'``
            Physical posterior standard deviations (sqrt of S_hat diagonal,
            converted via delta method for log-params),
            shape (n_rows, n_cols, n_fit) or (n_pixels, n_fit).

        ``'A_diag'``
            Averaging kernel diagonal — fraction of each retrieved parameter
            that is data-driven (0 = prior, 1 = fully data-driven),
            shape (n_rows, n_cols, n_fit) or (n_pixels, n_fit).

        ``'chi2'``
            Chi-squared per band — goodness of fit; ≈1 = good fit; >>1
            indicates model–data mismatch,
            shape (n_rows, n_cols) or (n_pixels,).

        ``'y_hat'``
            Simulated spectrum at the solution (only present when
            ``store_y_hat=True``),
            shape (n_rows, n_cols, n_obs) or (n_pixels, n_obs).

        ``'fit_names'``
            List of free parameter names, length n_fit.  Maps the last axis
            of x_hat, sigma, and A_diag to parameter names.

    Raises:
        ValueError: if Rrs has an unsupported number of dimensions.

    Example::

        setup = oe_engine.build_inversion(
            params_w, f_vec_w, sigma_a_w,
            log_params=['C_0', 'C_Y', 'C_Mie', 'zB'],
        )
        results = dask_oe_engine.invert_image(
            Rrs_image, setup, noise=0.001, n_iter=10,
        )
        # save minimal outputs
        import zarr
        store = zarr.open('retrieval.zarr', mode='w')
        for key in ('x_hat', 'sigma', 'A_diag', 'chi2'):
            store[key] = results[key]
    """
    Rrs_arr = np.asarray(Rrs)
    spatial_shape = None

    if Rrs_arr.ndim == 3:
        n_rows, n_cols, n_obs = Rrs_arr.shape
        spatial_shape = (n_rows, n_cols)
        Rrs_flat = Rrs_arr.reshape(-1, n_obs)
    elif Rrs_arr.ndim == 2:
        Rrs_flat = Rrs_arr
        n_obs    = Rrs_flat.shape[1]
    else:
        raise ValueError(
            f"Rrs must be 2-D (n_pixels, n_obs) or 3-D (n_rows, n_cols, n_obs), "
            f"got shape {Rrs_arr.shape}"
        )

    n_pixels = Rrs_flat.shape[0]
    n_fit    = len(setup.fit_names)

    # Convert JAX/device arrays to plain NumPy for Dask serialisation
    x_a_np      = np.array(setup.x_a)
    S_a_inv_np  = np.array(setup.S_a_inv)
    log_mask_np = np.array(setup.log_mask)
    weights_np  = np.array(setup.weights) if setup.weights is not None else None

    # --- build one Dask delayed task per tile --------------------------------
    delayed_tasks = []
    for start in range(0, n_pixels, tile_size):
        end  = min(start + tile_size, n_pixels)
        tile = Rrs_flat[start:end]   # NumPy slice — no copy

        task = dask.delayed(invert_tile)(
            tile,
            setup.f_fit,
            x_a_np,
            S_a_inv_np,
            log_mask_np,
            noise,
            weights_np,
            n_iter,
            lm_damping,
            store_y_hat,
        )
        delayed_tasks.append(task)

    # --- execute all tiles ---------------------------------------------------
    tile_results = dask.compute(*delayed_tasks, scheduler=scheduler)

    # --- assemble tile outputs into full-image arrays ------------------------
    x_hat_all  = np.concatenate([r[0] for r in tile_results], axis=0)  # (n_pixels, n_fit)
    sigma_all  = np.concatenate([r[1] for r in tile_results], axis=0)
    A_diag_all = np.concatenate([r[2] for r in tile_results], axis=0)
    chi2_all   = np.concatenate([r[3] for r in tile_results], axis=0)  # (n_pixels,)

    # --- reshape to spatial dimensions if input was 3-D ---------------------
    if spatial_shape is not None:
        x_hat_all  = x_hat_all.reshape(*spatial_shape, n_fit)
        sigma_all  = sigma_all.reshape(*spatial_shape, n_fit)
        A_diag_all = A_diag_all.reshape(*spatial_shape, n_fit)
        chi2_all   = chi2_all.reshape(*spatial_shape)

    out: Dict[str, object] = {
        'x_hat':     x_hat_all,
        'sigma':     sigma_all,
        'A_diag':    A_diag_all,
        'chi2':      chi2_all,
        'fit_names': setup.fit_names,
    }

    if store_y_hat:
        y_hat_all = np.concatenate([r[4] for r in tile_results], axis=0)
        if spatial_shape is not None:
            y_hat_all = y_hat_all.reshape(*spatial_shape, n_obs)
        out['y_hat'] = y_hat_all

    return out

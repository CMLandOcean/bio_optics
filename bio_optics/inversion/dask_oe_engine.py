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
# f_fit      — static (static_argnums=0): JAX retraces only when the forward
#              function changes, not for every tile.
# n_iter     — static (static_argnames): used in range(n_iter), which is a
#              Python for-loop unrolled at trace time; must be a concrete int.
# lm_damping — static (static_argnames): used in `if lm_damping > 0.0`, a
#              Python branch that must be resolved at trace time.
# ---------------------------------------------------------------------------

import jax as _jax

_invert_pixels_jit = _jax.jit(
    oe_engine.invert_pixels,
    static_argnums=(0,),
    static_argnames=('n_iter', 'lm_damping'),
)


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
    aux_tile=None,
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
        x_a:          prior mean in retrieval space.  Two accepted shapes:

                      * ``(n_fit,)`` — same prior mean for every pixel in the tile.
                        Pass ``np.array(setup.x_a)`` for the default uniform prior.
                      * ``(n_pixels_tile, n_fit)`` — per-pixel prior mean for the
                        tile.  Slice from a full-image ``x_a_image`` array before
                        passing here.  For log-params, values should already be in
                        log-space (i.e. ``ln(depth_map)`` for a depth prior).

        S_a_inv:      inverse prior covariance in retrieval space.  Two accepted
                      shapes:

                      * ``(n_fit, n_fit)`` — same prior uncertainty for every pixel.
                        Pass ``np.array(setup.S_a_inv)`` for the default uniform case.
                      * ``(n_pixels_tile, n_fit, n_fit)`` — per-pixel inverse prior
                        covariance.  Use this to tighten the prior where an auxiliary
                        map (e.g. bathymetry survey) is more reliable.
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
        aux_tile:     optional per-pixel auxiliary data for this tile.  Any
                      pytree (dict or array) whose leaves have a leading
                      dimension of n_pixels_tile.  Passed to
                      ``invert_pixels(aux_pixels=...)`` and forwarded to
                      ``f_fit(x, aux)`` per pixel.  Typical use: per-pixel
                      bottom reflectance override::

                          aux_tile = {'R_b_i': R_b_i_tile}   # (n_tile, n_obs, 6)

                      Slice from a full-image array in ``invert_image()``
                      rather than building manually here.  Default None.

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

    # Convert aux_tile leaves to JAX float64 arrays (handles dict or plain array)
    if aux_tile is not None:
        if isinstance(aux_tile, dict):
            aux_jax = {k: jnp.asarray(v, dtype=jnp.float64) for k, v in aux_tile.items()}
        else:
            aux_jax = jnp.asarray(aux_tile, dtype=jnp.float64)
    else:
        aux_jax = None

    results = _invert_pixels_jit(
        f_fit, Rrs_jax, noise, x_a_jax, S_a_inv_jax,
        n_iter=n_iter, lm_damping=lm_damping, weights=weights_jax,
        aux_pixels=aux_jax,
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
    x_a_image: Optional[np.ndarray] = None,
    S_a_inv_image: Optional[np.ndarray] = None,
    aux_image=None,
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
        x_a_image:    optional per-pixel prior mean array in **retrieval space**.
                      Accepted shapes:

                      * ``(n_pixels, n_fit)`` — when Rrs is already flat 2-D.
                      * ``(n_rows, n_cols, n_fit)`` — when Rrs is 3-D image.

                      Each pixel row overrides ``setup.x_a`` for that pixel.
                      For log-transformed parameters (e.g. ``zB``, ``C_0``)
                      the values must already be in log-space:
                      ``x_a_image[..., zB_idx] = np.log(depth_map)``.
                      If None (default), ``setup.x_a`` is broadcast uniformly.
        S_a_inv_image: optional per-pixel inverse prior covariance in retrieval
                      space.  Accepted shapes:

                      * ``(n_pixels, n_fit, n_fit)`` or
                      * ``(n_rows, n_cols, n_fit, n_fit)``

                      Use this when your auxiliary map has spatially varying
                      confidence (e.g. tighter zB constraint in well-surveyed
                      shallow areas).  If None (default), ``setup.S_a_inv`` is
                      broadcast uniformly.
        aux_image:    optional per-pixel auxiliary data passed to the forward
                      model as ``f_fit(x, aux)``.  Can be a dict of arrays or
                      a single array; every leaf must share the same leading
                      spatial shape as Rrs.  Examples:

                      * Per-pixel bottom reflectance (dict)::

                            R_b_i_image = ...   # (n_rows, n_cols, n_obs, 6)
                            aux_image = {'R_b_i': R_b_i_image}

                      * Any other precomputed spectral quantity can be overridden
                        the same way (e.g. ``'a_w'``, ``'bb_w'``).

                      Requires the forward function to accept a second argument
                      ``aux``, as returned by
                      ``albert_mobley_jax.make_forward_vec()``.
                      If None (default) ``f_fit(x)`` is called without aux.

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

    **Using spatial prior maps** (e.g. bathymetry for zB, CDOM climatology)::

        fit_names = setup.fit_names
        zB_idx    = fit_names.index('zB')
        C_Y_idx   = fit_names.index('C_Y')

        # Build per-pixel x_a in retrieval space (log-space for log-params)
        x_a_image = np.tile(np.array(setup.x_a), (n_rows, n_cols, 1))
        x_a_image[..., zB_idx] = np.log(bathymetry_map)      # log(depth [m])
        x_a_image[..., C_Y_idx] = np.log(cdom_climatology)   # log(CDOM [1/m])

        # Optionally tighten S_a for depth where the bathymetry is reliable
        S_a_inv_image = np.tile(np.array(setup.S_a_inv), (n_rows, n_cols, 1, 1))
        trusted_mask = (bathymetry_map < 5)   # shallow pixels — survey reliable
        S_a_inv_image[trusted_mask, zB_idx, zB_idx] *= 4   # 2× tighter σ

        results = dask_oe_engine.invert_image(
            Rrs_image, setup, noise=0.001, n_iter=10,
            x_a_image=x_a_image,
            S_a_inv_image=S_a_inv_image,
        )

    **Using per-pixel bottom reflectance** (replaces per-tile manual loop)::

        pre = albert_mobley_jax.precompute(wavelengths)
        f_vec = albert_mobley_jax.make_forward_vec(all_names, pre)
        setup = oe_engine.build_inversion(params, f_vec, sigma_a, log_params=log_params)

        # Build R_b_i image: start from scene default, override first bottom type
        R_b_i_base   = np.array(pre['R_b_i'])                      # (n_obs, 6)
        R_b_i_image  = np.tile(R_b_i_base, (n_rows, n_cols, 1, 1)) # (n_rows, n_cols, n_obs, 6)
        R_b_i_image[..., 0] = albedo_image                          # per-pixel measured albedo

        results = dask_oe_engine.invert_image(
            Rrs_image, setup, noise=0.001, n_iter=10,
            aux_image={'R_b_i': R_b_i_image},
        )
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

    # Number of leading spatial dims (2 for image, 0 for flat pixel stack)
    n_spatial = len(spatial_shape) if spatial_shape is not None else 0

    # Flatten optional per-pixel prior maps to (n_pixels, ...) if provided
    x_a_flat      = None
    S_a_inv_flat  = None
    if x_a_image is not None:
        x_a_flat = np.asarray(x_a_image).reshape(n_pixels, n_fit)
    if S_a_inv_image is not None:
        S_a_inv_flat = np.asarray(S_a_inv_image).reshape(n_pixels, n_fit, n_fit)

    # Flatten optional per-pixel auxiliary data; keep dict structure intact
    def _flatten_leaf(a):
        a = np.asarray(a)
        return a.reshape(n_pixels, *a.shape[n_spatial:])

    aux_flat = None
    if aux_image is not None:
        if isinstance(aux_image, dict):
            aux_flat = {k: _flatten_leaf(v) for k, v in aux_image.items()}
        else:
            aux_flat = _flatten_leaf(aux_image)

    # --- build one Dask delayed task per tile --------------------------------
    delayed_tasks = []
    for start in range(0, n_pixels, tile_size):
        end  = min(start + tile_size, n_pixels)
        tile = Rrs_flat[start:end]   # NumPy slice — no copy

        # Per-tile prior: slice from image map if provided, else use uniform
        tile_x_a     = x_a_flat[start:end]     if x_a_flat     is not None else x_a_np
        tile_S_a_inv = S_a_inv_flat[start:end]  if S_a_inv_flat is not None else S_a_inv_np

        # Per-tile aux: slice from flattened aux image if provided
        if aux_flat is not None:
            if isinstance(aux_flat, dict):
                tile_aux = {k: v[start:end] for k, v in aux_flat.items()}
            else:
                tile_aux = aux_flat[start:end]
        else:
            tile_aux = None

        task = dask.delayed(invert_tile)(
            tile,
            setup.f_fit,
            tile_x_a,
            tile_S_a_inv,
            log_mask_np,
            noise,
            weights_np,
            n_iter,
            lm_damping,
            store_y_hat,
            tile_aux,
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


# ---------------------------------------------------------------------------
# xarray conversion helper
# ---------------------------------------------------------------------------

def to_dataset(results: Dict[str, object],
               spatial_dims=('y', 'x'),
               wavelengths=None,
               coords: dict = None):
    """
    Convert an ``invert_image()`` result dict to an ``xarray.Dataset``.

    Parameters with a ``param`` axis (``x_hat``, ``sigma``, ``A_diag``) become
    ``DataArray``s with a ``param`` coordinate populated from ``fit_names``.
    ``chi2`` becomes a scalar-per-pixel ``DataArray``.  If ``y_hat`` is present
    it gets a ``wavelength`` coordinate when ``wavelengths`` is provided.

    Args:
        results:       dict returned by ``invert_image()``.
        spatial_dims:  names of the spatial dimensions.  Use ``('y', 'x')``
                       for image input (default) or ``('pixel',)`` for
                       flat (n_pixels, …) input.
        wavelengths:   optional 1-D array of wavelengths [nm], length n_obs.
                       Used to label the ``wavelength`` axis of ``y_hat``.
                       If None and ``y_hat`` is present, the axis is labelled
                       ``'band'`` with integer indices.
        coords:        optional dict of additional coordinates to attach to
                       every variable, e.g.
                       ``{'y': y_arr, 'x': x_arr}`` for georeferenced data
                       or ``{'time': t}`` for time-series outputs.

    Returns:
        xr.Dataset with variables:

        ``x_hat``    — dims (*spatial_dims, 'param'), retrieved physical values
        ``sigma``    — dims (*spatial_dims, 'param'), posterior σ in physical space
        ``A_diag``   — dims (*spatial_dims, 'param'), averaging kernel diagonal
        ``chi2``     — dims (*spatial_dims,),          goodness of fit
        ``y_hat``    — dims (*spatial_dims, 'wavelength') if present

    Example::

        ds = dask_oe_engine.to_dataset(
            results,
            wavelengths=wavelengths,
            coords={'y': y_coords, 'x': x_coords},
        )
        ds['x_hat'].sel(param='C_0').plot()
        ds['chi2'].plot()
    """
    import xarray as xr

    fit_names = results['fit_names']
    base_coords = dict(coords or {})
    param_coords = {**base_coords, 'param': fit_names}

    def _make_da(arr, dims, da_coords):
        return xr.DataArray(arr, dims=dims, coords=da_coords)

    ds_vars = {
        'x_hat':  _make_da(results['x_hat'],  (*spatial_dims, 'param'), param_coords),
        'sigma':  _make_da(results['sigma'],   (*spatial_dims, 'param'), param_coords),
        'A_diag': _make_da(results['A_diag'],  (*spatial_dims, 'param'), param_coords),
        'chi2':   _make_da(results['chi2'],    spatial_dims,             base_coords),
    }

    if 'y_hat' in results:
        if wavelengths is not None:
            wl_dim = 'wavelength'
            wl_coords = {**base_coords, 'wavelength': np.asarray(wavelengths)}
        else:
            wl_dim = 'band'
            n_obs = results['y_hat'].shape[-1]
            wl_coords = {**base_coords, 'band': np.arange(n_obs)}
        ds_vars['y_hat'] = _make_da(results['y_hat'], (*spatial_dims, wl_dim), wl_coords)

    return xr.Dataset(ds_vars)

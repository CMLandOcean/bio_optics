"""
Dask-tiled image inversion engine.

Splits a full EO image into pixel tiles, dispatches each tile as a Dask
delayed task calling a pluggable Layer-1 invert_fn, and reassembles the
per-tile outputs into full image-shaped NumPy arrays.

``Rrs`` may be a plain NumPy array or a dask-backed array (e.g. opened from
a zarr store).  For dask-backed inputs, each tile is materialised inside its
delayed task so that only ``tile_size`` pixels are in RAM at a time.  For
plain NumPy arrays the extra ``np.asarray()`` call is a no-op.

Architecture
------------
::

    dask_engine.invert_image(Rrs, setup, noise, invert_fn=oe_engine.invert_image)
      ↓
    Tiles image into pixel chunks (default 65 536 pixels = 256×256)
      ↓
    For each tile → dask.delayed(invert_fn)(tile, setup, noise, **per_tile_kwargs)
      ↓
    invert_fn can be any Layer-1 engine:
        oe_engine.invert_image        → JAX vmap Gauss-Newton OE
        oe_engine_optx.invert_image   → JAX optimistix OE
        lsq_engine_optx.invert_image  → JAX optimistix pure LSQ
        lmfit_engine.invert_image     → Python loop, lmfit solver
        scipy_engine.invert_image     → Python loop, scipy solver
      ↓
    Concatenate tile dicts → reshape to (n_rows, n_cols, …)

Scheduler notes
---------------
* ``'synchronous'`` (default) — single-threaded, easiest to debug.  JAX
  parallelises within each tile via XLA.  Recommended for GPU-backed JAX.
* ``'threads'`` — overlaps tiles using Python threads (JAX releases the GIL
  for XLA ops).
* ``'distributed'`` — requires a running Dask cluster.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional

import numpy as np
import dask


def _tile_task(invert_fn, Rrs_slice, setup, noise, **kwargs):
    """Materialise one tile slice and run invert_fn.

    Module-level (not a closure) so it is picklable when
    scheduler='processes' is used on Windows.
    """
    return invert_fn(np.asarray(Rrs_slice), setup, noise, **kwargs)


def invert_image(
    Rrs,
    setup,
    noise,
    invert_fn: Optional[Callable] = None,
    tile_size: int = 65536,
    scheduler: str = 'synchronous',
    x_a_image=None,
    S_a_inv_image=None,
    aux_image=None,
    bounds_image=None,
    **invert_kwargs,
) -> Dict[str, object]:
    """
    Tile-parallel image inversion using Dask.

    Splits the image into pixel tiles, dispatches each tile as a Dask task
    calling ``invert_fn``, and assembles the per-tile dicts into full-image
    NumPy arrays.

    Args:
        Rrs:           image of observed spectra.  Accepted shapes:

                       * ``(n_rows, n_cols, n_obs)`` — spatial image
                       * ``(n_pixels, n_obs)`` — pre-flattened pixel stack

        setup:         engine setup object from the corresponding
                       ``build_inversion()`` call.  Passed unchanged to every
                       tile call of ``invert_fn``.
        noise:         measurement uncertainty (scalar, per-band array, or
                       pre-inverted covariance matrix).  Passed unchanged to
                       every tile.
        invert_fn:     Layer-1 inversion callable with signature::

                           fn(spectra, setup, noise, **kwargs) -> dict

                       Defaults to ``oe_engine.invert_image`` (JAX vmap OE).
                       Pass any engine's ``invert_image`` to switch solver.
        tile_size:     pixels per Dask task, default 65 536 (≈ 256×256).
        scheduler:     Dask scheduler — ``'synchronous'`` (default),
                       ``'threads'``, or ``'distributed'``.
        x_a_image:     per-pixel prior mean in retrieval space.  Accepted
                       shapes: ``(n_pixels, n_fit)`` or
                       ``(n_rows, n_cols, n_fit)``.  Sliced per tile and
                       forwarded to ``invert_fn`` as ``x_a_image``.  For
                       log-params, values must already be in log-space.
                       Ignored by non-OE invert_fn implementations.
        S_a_inv_image: per-pixel inverse prior covariance, shapes
                       ``(n_pixels, n_fit, n_fit)`` or
                       ``(n_rows, n_cols, n_fit, n_fit)``.  Sliced per tile
                       and forwarded to ``invert_fn`` as ``S_a_inv_image``.
                       Ignored by non-OE invert_fn implementations.
        aux_image:     per-pixel auxiliary data (array or dict of arrays).
                       Every leaf must share the leading spatial shape of
                       ``Rrs``.  Sliced per tile and forwarded as
                       ``aux_image``.
        bounds_image:  per-pixel parameter bounds dict, forwarded to
                       ``invert_fn`` as ``bounds_image``.  Structure::

                           {'param_name': {'min': ndarray, 'max': ndarray}, …}

                       Each array must be broadcastable to the image spatial
                       shape.  Sliced per tile.  Only consumed by
                       ``lmfit_engine.invert_image``; other engines emit a
                       UserWarning and ignore it.
        **invert_kwargs: forwarded unchanged to every ``invert_fn`` tile call
                       (e.g. ``n_iter``, ``lm_damping``, ``store_y_hat``).

    Returns:
        dict whose keys are the union of keys returned by ``invert_fn``.
        Array values are shaped ``(n_rows, n_cols, …)`` when ``Rrs`` was 3-D,
        or ``(n_pixels, …)`` when ``Rrs`` was 2-D.  ``fit_names`` is a list.

    Example::

        from bio_optics.inversion import oe_engine
        from bio_optics.image_processing import dask_engine

        setup   = oe_engine.build_inversion(params, f_vec, sigma_a,
                                            log_params=['C_0', 'zB'])
        oe_engine.warmup_jit(setup, noise, n_iter=10, tile_size=65536)

        results = dask_engine.invert_image(
            Rrs_image, setup, noise=0.001, n_iter=10,
        )
        ds = dask_engine.to_dataset(results, wavelengths=wavelengths)

    Using an alternative solver::

        from bio_optics.inversion import lmfit_engine
        setup_lm = lmfit_engine.build_inversion(params, wavelengths, forward_func)
        results  = dask_engine.invert_image(
            Rrs_image, setup_lm, noise=0.001,
            invert_fn=lmfit_engine.invert_image,
        )
    """
    if invert_fn is None:
        from bio_optics.inversion import oe_engine
        invert_fn = oe_engine.invert_image

    # Keep Rrs as-is (numpy or dask) — do NOT materialise the whole array here.
    # np.asarray() is deferred to each tile slice in the loop below so that
    # dask-backed zarr inputs are read one tile at a time.
    spatial_shape = None

    if Rrs.ndim == 3:
        n_rows, n_cols, n_obs = Rrs.shape
        spatial_shape = (n_rows, n_cols)
        Rrs_flat = Rrs.reshape(-1, n_obs)
    elif Rrs.ndim == 2:
        Rrs_flat = Rrs
        n_obs    = Rrs_flat.shape[1]
    else:
        raise ValueError(
            f"Rrs must be 2-D (n_pixels, n_obs) or 3-D (n_rows, n_cols, n_obs), "
            f"got shape {Rrs.shape}"
        )

    n_pixels  = Rrs_flat.shape[0]
    n_spatial = len(spatial_shape) if spatial_shape is not None else 0

    # Flatten optional per-pixel maps to (n_pixels, ...) for slicing
    x_a_flat     = np.asarray(x_a_image).reshape(n_pixels, -1) if x_a_image is not None else None
    n_fit        = len(setup.fit_names)
    Sa_inv_flat  = (np.asarray(S_a_inv_image).reshape(n_pixels, n_fit, n_fit)
                    if S_a_inv_image is not None else None)

    def _flatten_leaf(a):
        a = np.asarray(a)
        return a.reshape(n_pixels, *a.shape[n_spatial:])

    aux_flat = None
    if aux_image is not None:
        if isinstance(aux_image, dict):
            aux_flat = {k: _flatten_leaf(v) for k, v in aux_image.items()}
        else:
            aux_flat = _flatten_leaf(aux_image)

    bounds_flat = None
    if bounds_image is not None:
        bounds_flat = {
            name: {side: np.asarray(arr).ravel() for side, arr in bd.items()}
            for name, bd in bounds_image.items()
        }

    # Build one Dask delayed task per tile.
    # Rrs_flat[start:end] is passed as a raw slice (numpy view or dask slice) —
    # np.asarray() is deferred to inside _tile_task so that dask-backed zarr
    # inputs are only read from disk when the task actually executes, not here.
    delayed_tasks = []
    for start in range(0, n_pixels, tile_size):
        end  = min(start + tile_size, n_pixels)

        tile_x_a    = x_a_flat[start:end]   if x_a_flat    is not None else None
        tile_Sa_inv = Sa_inv_flat[start:end] if Sa_inv_flat is not None else None

        if aux_flat is not None:
            tile_aux = ({k: v[start:end] for k, v in aux_flat.items()}
                        if isinstance(aux_flat, dict) else aux_flat[start:end])
        else:
            tile_aux = None

        tile_bounds = None
        if bounds_flat is not None:
            tile_bounds = {
                name: {side: arr[start:end] for side, arr in bd.items()}
                for name, bd in bounds_flat.items()
            }

        task = dask.delayed(_tile_task)(
            invert_fn, Rrs_flat[start:end], setup, noise,
            x_a_image=tile_x_a,
            S_a_inv_image=tile_Sa_inv,
            aux_image=tile_aux,
            bounds_image=tile_bounds,
            **invert_kwargs,
        )
        delayed_tasks.append(task)

    tile_results = dask.compute(*delayed_tasks, scheduler=scheduler)

    # Concatenate tile dicts — every key that is a numpy array gets concatenated;
    # list keys (fit_names) are taken from the first tile unchanged.
    out: Dict[str, object] = {}
    for key in tile_results[0]:
        vals = [r[key] for r in tile_results]
        if isinstance(vals[0], list):
            out[key] = vals[0]
        else:
            out[key] = np.concatenate(vals, axis=0)

    # Reshape to spatial dimensions when input was 3-D
    if spatial_shape is not None:
        for key, val in out.items():
            if isinstance(val, list):
                continue
            if val.ndim == 1:
                out[key] = val.reshape(*spatial_shape)
            else:
                out[key] = val.reshape(*spatial_shape, *val.shape[1:])

    return out


def to_dataset(
    results: Dict[str, object],
    spatial_dims=('y', 'x'),
    wavelengths=None,
    coords: dict = None,
):
    """
    Convert an ``invert_image()`` result dict to an ``xarray.Dataset``.

    Parameters with a ``param`` axis (``x_hat``, ``sigma``, ``A_diag``)
    become DataArrays with a ``param`` coordinate from ``fit_names``.
    ``chi2`` and ``H_info`` are scalar-per-pixel DataArrays.  ``y_hat``
    gets a ``wavelength`` coordinate when ``wavelengths`` is provided.

    Args:
        results:      dict returned by ``invert_image()``.
        spatial_dims: names of the spatial dimensions — ``('y', 'x')`` for
                      image input (default) or ``('pixel',)`` for flat input.
        wavelengths:  optional 1-D array of wavelengths [nm].  Used to label
                      the wavelength axis of ``y_hat``; if None the axis is
                      labelled ``'band'`` with integer indices.
        coords:       optional dict of additional coordinates to attach to
                      every variable, e.g. ``{'y': y_arr, 'x': x_arr}``.

    Returns:
        xr.Dataset with variables x_hat, sigma, A_diag, chi2, H_info,
        and optionally y_hat, chi2_spectral.

    Example::

        ds = dask_engine.to_dataset(
            results,
            wavelengths=wavelengths,
            coords={'y': y_coords, 'x': x_coords},
        )
        ds['x_hat'].sel(param='C_0').plot()
    """
    import xarray as xr

    fit_names   = results['fit_names']
    base_coords = dict(coords or {})
    param_coords = {**base_coords, 'param': fit_names}

    def _da(arr, dims, da_coords):
        return xr.DataArray(arr, dims=dims, coords=da_coords)

    ds_vars = {
        'x_hat':  _da(results['x_hat'],  (*spatial_dims, 'param'), param_coords),
        'sigma':  _da(results['sigma'],   (*spatial_dims, 'param'), param_coords),
        'A_diag': _da(results['A_diag'],  (*spatial_dims, 'param'), param_coords),
        'chi2':   _da(results['chi2'],    spatial_dims,             base_coords),
    }
    if 'H_info' in results:
        ds_vars['H_info'] = _da(results['H_info'], spatial_dims, base_coords)

    if 'chi2_spectral' in results:
        ds_vars['chi2_spectral'] = _da(results['chi2_spectral'], spatial_dims, base_coords)

    if 'y_hat' in results:
        if wavelengths is not None:
            wl_dim    = 'wavelength'
            wl_coords = {**base_coords, 'wavelength': np.asarray(wavelengths)}
        else:
            wl_dim    = 'band'
            n_obs     = results['y_hat'].shape[-1]
            wl_coords = {**base_coords, 'band': np.arange(n_obs)}
        ds_vars['y_hat'] = _da(results['y_hat'], (*spatial_dims, wl_dim), wl_coords)

    return xr.Dataset(ds_vars)

"""
Dask-tiled image inversion engine.

Splits a full EO image into tiles, dispatches each tile as a Dask delayed
task calling a pluggable Layer-1 invert_fn, and reassembles the per-tile
outputs into full image-shaped NumPy arrays.

Two tiling strategies are used depending on the input type:

* **Block path** (3-D dask-backed input, e.g. zarr on S3): iterates over the
  array's native spatial chunks via ``Rrs[y0:y1, x0:x1, :]``.  Each zarr
  chunk maps to exactly one S3 GET and is read exactly once — no cross-chunk
  re-reads.  ``tile_size`` is not used; block shape comes from the dask chunks.
* **Flat tile path** (plain NumPy array or 2-D input): flattens to
  ``(n_pixels, n_obs)`` and tiles by flat pixel index using ``tile_size``.
  For dask-backed zarr inputs, each tile is materialised inside its delayed
  task so that only ``tile_size`` pixels are in RAM at a time.  For plain
  NumPy arrays the extra ``np.asarray()`` call is a no-op.

Architecture
------------
::

    dask_engine.invert_image(Rrs, setup, noise, invert_fn=oe_engine.invert_image)
      ↓
    Block path (3-D dask)          Flat tile path (numpy / 2-D)
    Iterate zarr spatial blocks    Tile by flat pixel index
      ↓                              ↓
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
import dask.array as da


def _tile_task(invert_fn, Rrs_slice, setup, noise, **kwargs):
    """Materialise one tile slice and run invert_fn.

    Module-level (not a closure) so it is picklable when
    scheduler='processes' is used on Windows.

    Dask-backed slices are materialised with ``scheduler='threads'`` so that
    multiple zarr variables (e.g. reflectance + cloud + cirrus + haze) that
    contribute to one spatial block are read concurrently from S3/disk.
    Plain numpy slices fall through to a zero-copy ``np.asarray``.
    """
    if isinstance(Rrs_slice, da.Array):
        data = Rrs_slice.compute(scheduler='threads')
    else:
        data = np.asarray(Rrs_slice)
    return invert_fn(data, setup, noise, **kwargs)


def _assemble_blocks(tile_results, block_shapes, n_y_blocks, n_x_blocks):
    """Assemble per-block result dicts into a (n_rows, n_cols, ...) array dict."""
    all_keys = set().union(*(r.keys() for r in tile_results))
    out: Dict[str, object] = {}
    for key in all_keys:
        first_val = next(r[key] for r in tile_results if key in r)
        if isinstance(first_val, list):
            out[key] = first_val
            continue
        if np.issubdtype(first_val.dtype, np.floating):
            fill = np.nan
        elif first_val.dtype == np.bool_:
            fill = False
        else:
            fill = -1

        rows = []
        idx  = 0
        for _iy in range(n_y_blocks):
            row = []
            for _ix in range(n_x_blocks):
                by, bx = block_shapes[idx]
                r = tile_results[idx]
                if key in r:
                    val = r[key]
                    tile = (val.reshape(by, bx)
                            if val.ndim == 1
                            else val.reshape(by, bx, *val.shape[1:]))
                else:
                    shape = ((by, bx) if first_val.ndim == 1
                             else (by, bx, *first_val.shape[1:]))
                    tile = np.full(shape, fill, dtype=first_val.dtype)
                row.append(tile)
                idx += 1
            rows.append(np.concatenate(row, axis=1))
        out[key] = np.concatenate(rows, axis=0)
    return out


def _assemble_flat(tile_results, tile_sizes):
    """Assemble per-tile result dicts into a (n_pixels, ...) array dict."""
    all_keys = set().union(*(r.keys() for r in tile_results))
    out: Dict[str, object] = {}
    for key in all_keys:
        first_val = next(r[key] for r in tile_results if key in r)
        if isinstance(first_val, list):
            out[key] = first_val
            continue
        if np.issubdtype(first_val.dtype, np.floating):
            fill = np.nan
        elif first_val.dtype == np.bool_:
            fill = False
        else:
            fill = -1
        vals = []
        for r, sz in zip(tile_results, tile_sizes):
            if key in r:
                vals.append(r[key])
            else:
                shape = (sz, *first_val.shape[1:]) if first_val.ndim > 1 else (sz,)
                vals.append(np.full(shape, fill, dtype=first_val.dtype))
        out[key] = np.concatenate(vals, axis=0)
    return out


def _invert_blocks(Rrs, n_rows, n_cols, n_obs, setup, noise, invert_fn, scheduler,
                   x_a_image, S_a_inv_image, aux_image, bounds_image, **invert_kwargs):
    """Block-aligned inversion for 3-D dask inputs.

    Iterates over native dask spatial blocks (one per zarr chunk) instead of
    flat pixel ranges.  Each S3 GET reads exactly one zarr chunk; no
    cross-chunk re-reads occur.
    """
    n_fit = len(setup.fit_names)

    # Convert per-pixel maps to (n_rows, n_cols, ...) for block slicing.
    # Flat (n_pixels, ...) inputs are also handled: n_pixels == n_rows * n_cols.
    xa_sp = (np.asarray(x_a_image).reshape(n_rows, n_cols, -1)
             if x_a_image is not None else None)
    sa_sp = (np.asarray(S_a_inv_image).reshape(n_rows, n_cols, n_fit, n_fit)
             if S_a_inv_image is not None else None)

    aux_sp = None
    if aux_image is not None:
        if isinstance(aux_image, dict):
            aux_sp = {}
            for k, v in aux_image.items():
                a = np.asarray(v)
                aux_sp[k] = a.reshape(n_rows, n_cols, *a.shape[2:])
        else:
            a = np.asarray(aux_image)
            aux_sp = a.reshape(n_rows, n_cols, *a.shape[2:])

    bnd_sp = None
    if bounds_image is not None:
        bnd_sp = {
            name: {side: np.asarray(arr).reshape(n_rows, n_cols)
                   for side, arr in bd.items()}
            for name, bd in bounds_image.items()
        }

    # Spatial chunk offsets (cumulative sum of chunk sizes along each axis).
    y_offsets  = [0] + list(np.cumsum(Rrs.chunks[0]))
    x_offsets  = [0] + list(np.cumsum(Rrs.chunks[1]))
    n_y_blocks = len(Rrs.chunks[0])
    n_x_blocks = len(Rrs.chunks[1])

    delayed_tasks = []
    block_shapes  = []

    for iy in range(n_y_blocks):
        y0, y1 = y_offsets[iy], y_offsets[iy + 1]
        for ix in range(n_x_blocks):
            x0, x1 = x_offsets[ix], x_offsets[ix + 1]
            by, bx = y1 - y0, x1 - x0
            block_shapes.append((by, bx))

            # Spatial dask slice — not yet materialised; _tile_task calls np.asarray()
            Rrs_block = Rrs[y0:y1, x0:x1, :].reshape(-1, n_obs)

            tile_xa = (xa_sp[y0:y1, x0:x1].reshape(-1, xa_sp.shape[-1])
                       if xa_sp is not None else None)
            tile_sa = (sa_sp[y0:y1, x0:x1].reshape(-1, n_fit, n_fit)
                       if sa_sp is not None else None)

            if aux_sp is not None:
                if isinstance(aux_sp, dict):
                    tile_aux = {k: v[y0:y1, x0:x1].reshape(by * bx, *v.shape[2:])
                                for k, v in aux_sp.items()}
                else:
                    tile_aux = aux_sp[y0:y1, x0:x1].reshape(by * bx, *aux_sp.shape[2:])
            else:
                tile_aux = None

            tile_bounds = None
            if bnd_sp is not None:
                tile_bounds = {
                    name: {side: arr[y0:y1, x0:x1].ravel()
                           for side, arr in bd.items()}
                    for name, bd in bnd_sp.items()
                }

            task = dask.delayed(_tile_task)(
                invert_fn, Rrs_block, setup, noise,
                x_a_image=tile_xa,
                S_a_inv_image=tile_sa,
                aux_image=tile_aux,
                bounds_image=tile_bounds,
                **invert_kwargs,
            )
            delayed_tasks.append(task)

    tile_results = dask.compute(*delayed_tasks, scheduler=scheduler)
    return _assemble_blocks(tile_results, block_shapes, n_y_blocks, n_x_blocks)


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

    Splits the image into tiles, dispatches each tile as a Dask task calling
    ``invert_fn``, and assembles the per-tile dicts into full-image NumPy arrays.

    For 3-D dask-backed inputs (e.g. zarr on S3) the **block path** is used:
    each native spatial chunk becomes one tile, so every zarr chunk is read
    exactly once.  For plain NumPy arrays or 2-D inputs the **flat tile path**
    is used, tiling by flat pixel index with ``tile_size``.

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
        tile_size:     pixels per Dask task for the flat tile path (numpy /
                       2-D inputs), default 65 536 (≈ 256×256).  Not used for
                       3-D dask inputs — block shape comes from dask chunks.
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

    spatial_shape = None
    if Rrs.ndim == 3:
        n_rows, n_cols, n_obs = Rrs.shape
        spatial_shape = (n_rows, n_cols)
    elif Rrs.ndim == 2:
        n_obs = Rrs.shape[1]
    else:
        raise ValueError(
            f"Rrs must be 2-D (n_pixels, n_obs) or 3-D (n_rows, n_cols, n_obs), "
            f"got shape {Rrs.shape}"
        )

    # Block path: 3-D dask inputs iterate over native spatial chunks so each
    # zarr chunk (S3 object) is read exactly once — no cross-chunk re-reads.
    if spatial_shape is not None and isinstance(Rrs, da.Array):
        return _invert_blocks(
            Rrs, n_rows, n_cols, n_obs, setup, noise, invert_fn, scheduler,
            x_a_image, S_a_inv_image, aux_image, bounds_image, **invert_kwargs,
        )

    # --- Flat tile path (numpy or 2-D input) ---------------------------------
    # Keep Rrs as-is; np.asarray() is deferred to each _tile_task call so that
    # any dask-backed 2-D inputs are materialised one tile at a time.
    Rrs_flat = Rrs.reshape(-1, n_obs) if spatial_shape is not None else Rrs
    n_pixels  = Rrs_flat.shape[0]
    n_spatial = len(spatial_shape) if spatial_shape is not None else 0
    n_fit     = len(setup.fit_names)

    x_a_flat    = np.asarray(x_a_image).reshape(n_pixels, -1) if x_a_image is not None else None
    Sa_inv_flat = (np.asarray(S_a_inv_image).reshape(n_pixels, n_fit, n_fit)
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

    delayed_tasks = []
    tile_sizes    = []
    for start in range(0, n_pixels, tile_size):
        end = min(start + tile_size, n_pixels)
        tile_sizes.append(end - start)

        tile_x_a    = x_a_flat[start:end]    if x_a_flat    is not None else None
        tile_Sa_inv = Sa_inv_flat[start:end]  if Sa_inv_flat is not None else None

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
    out = _assemble_flat(tile_results, tile_sizes)

    # Reshape to spatial dimensions when input was 3-D
    if spatial_shape is not None:
        for key, val in out.items():
            if isinstance(val, list):
                continue
            out[key] = (val.reshape(*spatial_shape)
                        if val.ndim == 1
                        else val.reshape(*spatial_shape, *val.shape[1:]))

    return out


def calibrate_noise(
    Rrs,
    ocean_mask,
    setup,
    noise_init: float,
    invert_fn: Callable,
    tile_size: int = 65536,
    tol: float = 0.05,
    max_iter: int = 5,
    verbose: bool = True,
    **invert_kwargs,
) -> float:
    """Find a noise scalar so that median chi² ≈ 1 over the most water-covered tile.

    Locates the tile with the highest ocean pixel count (using ``ocean_mask`` as
    a cheap proxy), materialises only that tile, and iterates::

        noise = noise * sqrt(median_chi2)

    until ``|median_chi2 - 1| < tol`` or ``max_iter`` is exhausted.  This
    formula is exact for linear forward models and converges in 2–3 iterations
    for weakly nonlinear OE problems.  The returned noise can be passed directly
    to ``invert_image()``.

    JAX must be warmed up for ``tile_size`` before calling this function — use
    the same warmup step as for the full inversion.

    Args:
        Rrs:           image of observed spectra, shape ``(n_rows, n_cols,
                       n_obs)``.  May be a numpy array or a dask array; only the
                       calibration tile is materialised.
        ocean_mask:    boolean array ``(n_rows, n_cols)`` — proxy for water
                       coverage used to select the calibration tile.
        setup:         engine setup from ``build_inversion()``.
        noise_init:    initial noise guess [sr⁻¹].
        invert_fn:     Layer-1 inversion callable with the same signature as
                       ``invert_image`` (e.g. ``oe_engine_optx.invert_image``).
        tile_size:     pixels per calibration tile.  Must match the value used
                       in ``invert_image()`` so the JIT-compiled kernel shape
                       is reused without recompilation.
        tol:           convergence tolerance on ``|median_chi2 - 1|``.
        max_iter:      maximum number of calibration iterations.
        verbose:       if True, print noise and median chi² per iteration.
        **invert_kwargs: forwarded to ``invert_fn`` (e.g. ``solver``,
                       ``max_steps``).  ``store_y_hat`` is forced to ``False``
                       to save memory during calibration.

    Returns:
        Calibrated noise as a Python float.

    Example::

        print('Calibrating NOISE from best tile ...')
        NOISE = dask_engine.calibrate_noise(
            Rrs_arr, ocean_mask.values, setup, NOISE,
            invert_fn=oe_engine_optx.invert_image,
            tile_size=TILE_SIZE, solver=SOLVER, max_steps=MAX_STEPS,
        )
        # → calibrated noise = 0.00060 sr⁻¹
    """
    n_rows, n_cols, n_obs = Rrs.shape
    n_pixels  = n_rows * n_cols
    mask_flat = np.asarray(ocean_mask).ravel()

    # Pick the tile with the most ocean pixels.
    starts = list(range(0, n_pixels, tile_size))
    counts = [int(mask_flat[s:min(s + tile_size, n_pixels)].sum()) for s in starts]
    best_s = starts[int(np.argmax(counts))]
    best_e = min(best_s + tile_size, n_pixels)

    # Materialise only the calibration tile (not the full scene).
    Rrs_flat = Rrs.reshape(n_pixels, n_obs)
    if isinstance(Rrs_flat, da.Array):
        tile = Rrs_flat[best_s:best_e].compute(scheduler='synchronous')
    else:
        tile = np.asarray(Rrs_flat[best_s:best_e])

    # Never store y_hat during calibration — wastes memory with no benefit.
    kwargs = {**invert_kwargs, 'store_y_hat': False}

    noise = float(noise_init)
    for i in range(max_iter):
        res    = invert_fn(tile, setup, noise, **kwargs)
        chi2   = res['chi2']
        valid  = chi2[np.isfinite(chi2)]
        if len(valid) == 0:
            break
        med   = float(np.median(valid))
        noise = noise * float(np.sqrt(med))
        if verbose:
            print(f'  iter {i + 1}: noise={noise / float(np.sqrt(med)):.5f}  '
                  f'median chi²={med:.3f}  → noise={noise:.5f} sr⁻¹')
        if abs(med - 1.0) < tol:
            break

    if verbose:
        print(f'Calibrated NOISE = {noise:.5f} sr⁻¹')
    return noise


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
        and optionally y_hat, chi2_spectral, n_steps.

    Example::

        ds = dask_engine.to_dataset(
            results,
            wavelengths=wavelengths,
            coords={'y': y_coords, 'x': x_coords},
        )
        ds['x_hat'].sel(param='C_0').plot()
    """
    import xarray as xr

    fit_names    = results['fit_names']
    base_coords  = dict(coords or {})
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

    if 'n_steps' in results:
        ds_vars['n_steps'] = _da(results['n_steps'], spatial_dims, base_coords)

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

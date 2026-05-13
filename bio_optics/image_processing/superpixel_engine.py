"""
SLIC superpixel-based image inversion.

Pipeline
--------
1. segment_image           — SLIC segmentation of the Rrs image
2. aggregate_superpixels   — per-segment mean spectrum + pixel counts
3. invert_superpixels      — inversion on mean spectra (any Layer-1 engine)
4. backinterp_pca_knn      — back-interpolate to full resolution via PCA+kNN IDW
5. invert_image_superpixel — end-to-end wrapper

Reference: Adams et al. (2021), Remote Sensing of Environment.

Chi-2 note
----------
The mean spectrum of a segment with N pixels has noise σ/√N, so inverting it
with the original pixel noise σ yields chi2_raw << 1 for large segments.
The calibrated chi2 is:  chi2_calibrated = chi2_raw × sp_counts
Both are included in the output.

invert_fn note
--------------
By default `invert_superpixels` calls `oe_engine.invert_image` (JAX vmap OE
with prior).  Pass any Layer-1 engine's `invert_image` to switch solver:
    invert_fn=lsq_engine_optx.invert_image   — pure weighted LSQ
    invert_fn=lmfit_engine.invert_image      — lmfit solver
    invert_fn=scipy_engine.invert_image      — scipy solver
All must have the signature: fn(spectra, setup, noise, **kwargs) → dict.
When the invert_fn returns no `sigma` or `A_diag` keys those fields are
omitted from the output.
"""
from __future__ import annotations

import numpy as np
from typing import Any, Optional

from skimage.segmentation import slic
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors


# ---------------------------------------------------------------------------
# 1. Segmentation
# ---------------------------------------------------------------------------

def segment_image(
    Rrs: np.ndarray,
    n_segments: int = 1000,
    compactness: float = 0.1,
    sigma: float = 2.0,
    band_slice: Optional[np.ndarray] = None,
) -> np.ndarray:
    """SLIC segmentation of an Rrs image.

    Parameters
    ----------
    Rrs         : (n_rows, n_cols, n_obs)
    n_segments  : target number of superpixels
    compactness : SLIC compactness (0.1 → shape follows spectral edges)
    sigma       : smoothing sigma before clustering
    band_slice  : optional index array selecting bands for segmentation
                  (e.g. indices of 420–690 nm bands); all bands used if None

    Returns
    -------
    labels : (n_rows, n_cols) int array of segment IDs starting at 0
    """
    Rrs_seg = Rrs[..., band_slice] if band_slice is not None else Rrs

    # slic rejects NaN/Inf values — fill invalid pixels with the per-band
    # nanmean so segmentation runs on the full spatial grid.  The original
    # NaN mask is preserved in Rrs and applied downstream (aggregation counts
    # only valid pixels; inversion engines skip NaN spectra).
    valid = np.isfinite(Rrs_seg).all(axis=-1)
    if not valid.all():
        fill = np.nanmean(Rrs_seg.reshape(-1, Rrs_seg.shape[-1]), axis=0)
        Rrs_seg = Rrs_seg.copy()
        Rrs_seg[~valid] = fill

    return slic(
        Rrs_seg,
        n_segments=n_segments,
        compactness=compactness,
        sigma=sigma,
        channel_axis=-1,
        start_label=0,
    )


# ---------------------------------------------------------------------------
# 2. Aggregation
# ---------------------------------------------------------------------------

def aggregate_superpixels(
    Rrs: np.ndarray,
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute mean spectrum and pixel count per superpixel.

    Parameters
    ----------
    Rrs    : (n_rows, n_cols, n_obs)
    labels : (n_rows, n_cols) int segment IDs

    Returns
    -------
    sp_spectra : (n_segs, n_obs)  mean Rrs per segment, ordered by segment ID
    sp_counts  : (n_segs,)        number of pixels per segment
    """
    Rrs_flat    = Rrs.reshape(-1, Rrs.shape[-1])
    labels_flat = labels.ravel()
    seg_ids     = np.unique(labels_flat)
    n_segs      = len(seg_ids)
    n_obs       = Rrs_flat.shape[-1]

    sp_spectra = np.empty((n_segs, n_obs), dtype=np.float64)
    sp_counts  = np.empty(n_segs, dtype=np.int64)

    for i, sid in enumerate(seg_ids):
        mask  = labels_flat == sid
        px    = Rrs_flat[mask]
        valid = np.isfinite(px).all(axis=-1)
        if valid.any():
            sp_spectra[i] = px[valid].mean(axis=0)
            sp_counts[i]  = int(valid.sum())
        else:
            sp_spectra[i] = np.nan   # all-land segment; inversion engine will skip
            sp_counts[i]  = 0

    return sp_spectra, sp_counts


# ---------------------------------------------------------------------------
# 3. Superpixel inversion
# ---------------------------------------------------------------------------

def invert_superpixels(
    sp_spectra: np.ndarray,
    sp_counts: np.ndarray,
    setup: Any,
    noise,
    invert_fn=None,
    **invert_kwargs,
) -> dict:
    """Inversion of superpixel mean spectra.

    Passes `noise` unchanged to the invert function.  The resulting chi2_raw
    will be < 1 for large segments (mean spectrum is smoother than individual
    pixels).  `invert_image_superpixel` adds a calibrated chi2:
    chi2_calibrated = chi2_raw × sp_counts.

    Parameters
    ----------
    sp_spectra  : (n_segs, n_obs)
    sp_counts   : (n_segs,)  pixel counts (used for chi2 calibration upstream)
    setup       : InversionSetup
    noise       : scalar or (n_obs,) — per-pixel noise level in sr⁻¹
    invert_fn   : callable(spectra, setup, noise, **kwargs) → dict
                  Defaults to dask_oe_engine.invert_image.  Pass
                  lmfit_engine.invert_image for lmfit-based LSQ.
    **invert_kwargs : forwarded to invert_fn (n_iter, tile_size, max_steps, …)

    Returns
    -------
    results dict with at least x_hat/chi2 shaped (n_segs, …); sigma/A_diag
    present only when invert_fn returns them.
    """
    if invert_fn is None:
        from bio_optics.inversion import oe_engine
        invert_fn = oe_engine.invert_image
    return invert_fn(
        sp_spectra,   # (n_segs, n_obs) — flat 2-D input accepted by both engines
        setup,
        noise,
        **invert_kwargs,
    )


# ---------------------------------------------------------------------------
# 4. PCA + kNN back-interpolation with uncertainty propagation
# ---------------------------------------------------------------------------

def backinterp_pca_knn(
    Rrs_pixels: np.ndarray,
    sp_spectra: np.ndarray,
    sp_x_hat: np.ndarray,
    sp_sigma: Optional[np.ndarray] = None,
    sp_A_diag: Optional[np.ndarray] = None,
    k: int = 4,
    n_components: int = 6,
) -> dict:
    """Back-interpolate superpixel inversion results to full pixel resolution.

    Uses PCA on brightness-normalised spectra + inverse-distance weighted
    average of k nearest superpixels in PCA space (Adams et al. 2021).

    Uncertainty propagation (when sp_sigma is provided)
    -----------------------
    σ²_pixel = Σ wᵢ · σ²_spᵢ  +  Σ wᵢ · (x_hat_spᵢ − x_hat_pixel)²

    First term is the IDW-weighted average of the superpixel posterior variances.
    Using wᵢ (not wᵢ²) is appropriate here because SLIC neighbours are selected
    for spectral similarity and are therefore correlated — the independent-sample
    formula (wᵢ²) would spuriously reduce sigma by ~1/sqrt(k), making propagated
    sigma ~2× lower than per-pixel sigma in smooth regions.
    Second term captures the spread among the k neighbours (interpolation
    uncertainty); zero when all k neighbours agree.

    chi2 is NOT back-interpolated here — it is copied via label array in
    invert_image_superpixel, since chi2 is a per-spectrum property.

    Parameters
    ----------
    Rrs_pixels  : (n_pixels, n_obs)
    sp_spectra  : (n_segs, n_obs)
    sp_x_hat    : (n_segs, n_fit)  physical space
    sp_sigma    : (n_segs, n_fit)  physical space; None for LSQ (no posterior sigma)
    sp_A_diag   : (n_segs, n_fit); None for LSQ (no averaging kernel)
    k           : number of nearest superpixel neighbours (paper default: 4)
    n_components: PCA components (paper default: 6)

    Returns
    -------
    dict with keys:
        x_hat  : (n_pixels, n_fit)
        sigma  : (n_pixels, n_fit)  only present when sp_sigma is not None
        A_diag : (n_pixels, n_fit)  only present when sp_A_diag is not None
    """
    # --- filter valid superpixels (all-land segments have NaN spectra) --------
    sp_valid = np.isfinite(sp_spectra).all(axis=-1)
    sp_spec  = sp_spectra[sp_valid]
    sp_xh    = sp_x_hat[sp_valid]
    sp_sig   = sp_sigma[sp_valid]  if sp_sigma  is not None else None
    sp_Ad    = sp_A_diag[sp_valid] if sp_A_diag is not None else None

    n_segs = sp_spec.shape[0]
    k_eff  = min(k, n_segs)

    # --- brightness normalise ------------------------------------------------
    def _norm(X):
        s = X.sum(axis=-1, keepdims=True)
        return X / np.where(s == 0, 1.0, s)

    sp_norm = _norm(sp_spec)

    # Fill NaN pixels with per-band nanmean before PCA transform.
    # Invalid pixels will be NaN-masked in invert_image_superpixel after
    # back-interpolation — we just need finite values here for sklearn.
    px_finite = np.isfinite(Rrs_pixels).all(axis=-1)
    if not px_finite.all():
        fill     = np.nanmean(Rrs_pixels, axis=0)
        Rrs_fill = Rrs_pixels.copy()
        Rrs_fill[~px_finite] = fill
    else:
        Rrs_fill = Rrs_pixels
    px_norm = _norm(Rrs_fill)

    # --- PCA on superpixel spectra -------------------------------------------
    n_comp = min(n_components, n_segs - 1, sp_norm.shape[-1])
    pca    = PCA(n_components=n_comp)
    sp_pca = pca.fit_transform(sp_norm)     # (n_segs, n_comp)
    px_pca = pca.transform(px_norm)         # (n_pixels, n_comp)

    # --- kNN in PCA space ----------------------------------------------------
    nn = NearestNeighbors(n_neighbors=k_eff, algorithm='auto')
    nn.fit(sp_pca)
    distances, indices = nn.kneighbors(px_pca)   # (n_pixels, k_eff)

    # IDW weights: 1/d normalised; exact match → full weight on that neighbour
    exact     = distances == 0
    any_exact = exact.any(axis=-1, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        inv_d = np.where(distances > 0, 1.0 / distances, 0.0)
    inv_d   = np.where(any_exact, exact.astype(float), inv_d)
    denom   = inv_d.sum(axis=-1, keepdims=True)
    weights = inv_d / np.where(denom == 0, 1.0, denom)    # (n_pixels, k_eff)

    w3 = weights[:, :, np.newaxis]                         # (n_pixels, k_eff, 1)

    # --- x_hat: IDW average --------------------------------------------------
    x_hat_nb  = sp_xh[indices]                             # (n_pixels, k_eff, n_fit)
    x_hat_out = (w3 * x_hat_nb).sum(axis=1)                # (n_pixels, n_fit)

    out = {'x_hat': x_hat_out}

    # --- A_diag: IDW average (OE only) ---------------------------------------
    if sp_Ad is not None:
        out['A_diag'] = (w3 * sp_Ad[indices]).sum(axis=1)  # (n_pixels, n_fit)

    # --- sigma: propagated through IDW + interpolation spread (OE only) ------
    if sp_sig is not None:
        sigma_nb   = sp_sig[indices]                        # (n_pixels, k_eff, n_fit)
        var_post   = (w3 * sigma_nb ** 2).sum(axis=1)
        diff2      = (x_hat_nb - x_hat_out[:, np.newaxis, :]) ** 2
        var_interp = (w3 * diff2).sum(axis=1)
        out['sigma'] = np.sqrt(var_post + var_interp)       # (n_pixels, n_fit)

    return out


# ---------------------------------------------------------------------------
# 5. End-to-end pipeline
# ---------------------------------------------------------------------------

def invert_image_superpixel(
    Rrs: np.ndarray,
    setup: Any,
    noise,
    n_segments: int = 1000,
    compactness: float = 0.1,
    sigma: float = 2.0,
    k: int = 4,
    n_components: int = 6,
    band_slice: Optional[np.ndarray] = None,
    store_sp_results: bool = False,
    invert_fn=None,
    x0_image: Optional[np.ndarray] = None,
    **invert_kwargs,
) -> dict:
    """SLIC superpixel inversion with PCA+kNN back-interpolation.

    Works with any inversion backend via `invert_fn`:
      - Default (None): oe_engine.invert_image         → JAX vmap OE with prior
      - Pass lsq_engine_optx.invert_image            → pure LSQ, no prior

    Output dict always contains:
      x_hat            (n_rows, n_cols, n_fit)  IDW back-interpolated
      chi2             (n_rows, n_cols)          label copy (superpixel chi2 at pixel noise)
      chi2_calibrated  (n_rows, n_cols)          chi2 × sp_counts (calibrated for mean-spectrum noise)
      fit_names        list[str]
      labels           (n_rows, n_cols) int      segment ID map
      sp_counts        (n_segs,)                 pixels per segment

    OE-only fields (present when invert_fn returns them):
      sigma            (n_rows, n_cols, n_fit)  propagated uncertainty
      A_diag           (n_rows, n_cols, n_fit)  IDW back-interpolated

    LSQ-only fields (present when invert_fn returns them):
      n_steps          (n_rows, n_cols)          label copy of solver iterations

    Parameters
    ----------
    Rrs         : (n_rows, n_cols, n_obs)
    setup       : InversionSetup from oe_engine.build_inversion()
    noise       : scalar or (n_obs,) per-pixel noise in sr⁻¹
    n_segments  : target SLIC superpixel count
    compactness : SLIC compactness (paper default: 0.1)
    sigma       : SLIC smoothing sigma (paper default: 2.0)
    k           : kNN neighbours for back-interpolation (paper default: 4)
    n_components: PCA components for spectral embedding (paper default: 6)
    band_slice  : band indices for SLIC segmentation; None = all bands
    store_sp_results : include raw superpixel-level results in output dict
    invert_fn   : callable(spectra, setup, noise, **kwargs) → dict
                  Defaults to dask_oe_engine.invert_image.
    x0_image    : (n_rows, n_cols, n_fit) array of per-pixel starting values in
                  retrieval space (log-space for log-params).  Aggregated to
                  per-superpixel means and injected as `x_a_image` into the OE
                  engine.  Ignored by LSQ engines (which have no prior term).
                  Typical use: set x0_image[..., zB_idx] = np.log(bathy_map) to
                  provide a spatially varying bathymetry prior.
    **invert_kwargs  : forwarded to invert_fn (n_iter, tile_size, max_steps, …)
    """
    if Rrs.ndim != 3:
        raise ValueError(f"Rrs must be (n_rows, n_cols, n_obs), got shape {Rrs.shape}")

    n_rows, n_cols, n_obs = Rrs.shape
    n_pixels = n_rows * n_cols

    # 1. Segment
    labels      = segment_image(Rrs, n_segments, compactness, sigma, band_slice)
    seg_ids     = np.unique(labels)
    n_segs      = len(seg_ids)
    labels_flat = labels.ravel()

    # 2. Aggregate
    sp_spectra, sp_counts = aggregate_superpixels(Rrs, labels)

    # 2b. Aggregate x0_image to per-superpixel means and inject as x_a_image.
    #     oe_engine.invert_image accepts x_a_image (n_segs, n_fit) to set
    #     a per-spectra prior mean in retrieval space.  LSQ engines ignore it.
    if x0_image is not None:
        if x0_image.shape[:2] != (n_rows, n_cols):
            raise ValueError(
                f"x0_image leading dims {x0_image.shape[:2]} must match "
                f"Rrs spatial dims ({n_rows}, {n_cols})"
            )
        n_fit_x0 = x0_image.shape[-1]
        x0_flat  = x0_image.reshape(-1, n_fit_x0)
        x0_sp    = np.full((n_segs, n_fit_x0), np.nan)
        for i, sid in enumerate(seg_ids):
            px    = x0_flat[labels_flat == sid]
            valid = np.isfinite(px).all(axis=-1)
            if valid.any():
                x0_sp[i] = px[valid].mean(axis=0)
        invert_kwargs = {**invert_kwargs, 'x_a_image': x0_sp}

    # 3. Invert superpixel mean spectra
    sp_results = invert_superpixels(
        sp_spectra, sp_counts, setup, noise,
        invert_fn=invert_fn, **invert_kwargs,
    )

    sp_x_hat  = sp_results['x_hat']            # (n_segs, n_fit) — physical space
    sp_sigma  = sp_results.get('sigma')         # None for LSQ
    sp_A_diag = sp_results.get('A_diag')        # None for LSQ
    sp_chi2    = sp_results['chi2']              # (n_segs,)
    sp_H_info  = sp_results.get('H_info')        # None for LSQ engines
    sp_chi2_sp = sp_results.get('chi2_spectral') # None unless store_chi2_spectral=True

    # 4. Back-interpolate x_hat (+ sigma/A_diag when available)
    bp = backinterp_pca_knn(
        Rrs.reshape(n_pixels, n_obs),
        sp_spectra, sp_x_hat,
        sp_sigma=sp_sigma, sp_A_diag=sp_A_diag,
        k=k, n_components=n_components,
    )

    # 5. chi2 / H_info: label copy — map segment ID → superpixel value
    seg_id_to_idx = {sid: i for i, sid in enumerate(seg_ids)}
    sp_idx_flat   = np.array([seg_id_to_idx[sid] for sid in labels_flat], dtype=np.int64)
    chi2_flat     = sp_chi2[sp_idx_flat]
    chi2_cal_flat = sp_chi2[sp_idx_flat] * sp_counts[sp_idx_flat]
    H_info_flat   = sp_H_info[sp_idx_flat] if sp_H_info is not None else None
    chi2_sp_flat  = sp_chi2_sp[sp_idx_flat] if sp_chi2_sp is not None else None

    # 6. Reshape to spatial dims
    n_fit = bp['x_hat'].shape[-1]
    out = {
        'x_hat':           bp['x_hat'].reshape(n_rows, n_cols, n_fit),
        'chi2':            chi2_flat.reshape(n_rows, n_cols),
        'chi2_calibrated': chi2_cal_flat.reshape(n_rows, n_cols),
        'fit_names':       sp_results['fit_names'],
        'labels':          labels,
        'sp_counts':       sp_counts,
    }
    if 'sigma' in bp:
        out['sigma']  = bp['sigma'].reshape(n_rows, n_cols, n_fit)
    if 'A_diag' in bp:
        out['A_diag'] = bp['A_diag'].reshape(n_rows, n_cols, n_fit)
    if H_info_flat is not None:
        out['H_info'] = H_info_flat.reshape(n_rows, n_cols)
    if chi2_sp_flat is not None:
        out['chi2_spectral'] = chi2_sp_flat.reshape(n_rows, n_cols)
    if 'n_steps' in sp_results:
        out['n_steps'] = sp_results['n_steps'][sp_idx_flat].reshape(n_rows, n_cols)
    if 'success' in sp_results:
        out['success'] = sp_results['success'][sp_idx_flat].reshape(n_rows, n_cols)
    if 'n_nfev' in sp_results:
        out['n_nfev'] = sp_results['n_nfev'][sp_idx_flat].reshape(n_rows, n_cols)
    if store_sp_results:
        out['sp_results'] = sp_results

    # Mask land/invalid pixels — back-interpolation fills them with extrapolated
    # values from nearest superpixels; NaN them out to match the input mask.
    pixel_valid = np.isfinite(Rrs).all(axis=-1)   # (n_rows, n_cols)
    inv = ~pixel_valid
    if inv.any():
        out['x_hat'][inv] = np.nan
        out['chi2'][inv] = np.nan
        out['chi2_calibrated'][inv] = np.nan
        if 'sigma'  in out: out['sigma'][inv]  = np.nan
        if 'A_diag' in out: out['A_diag'][inv] = np.nan
        if 'H_info' in out: out['H_info'][inv] = np.nan
        if 'chi2_spectral' in out: out['chi2_spectral'][inv] = np.nan
        if 'n_steps' in out: out['n_steps'][inv] = -1

    return out

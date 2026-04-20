"""
SLIC superpixel-based OE inversion.

Pipeline
--------
1. segment_image           — SLIC segmentation of the Rrs image
2. aggregate_superpixels   — per-segment mean spectrum + pixel counts
3. invert_superpixels      — OE inversion on mean spectra
4. backinterp_pca_knn      — back-interpolate to full resolution via PCA+kNN IDW
5. invert_image_superpixel — end-to-end wrapper (same dict format as dask_oe_engine)

Reference: Adams et al. (2021), Remote Sensing of Environment.

Chi-2 note
----------
`dask_oe_engine.invert_image` accepts a single noise level for all pixels.
The mean spectrum of a segment with N pixels has noise σ/√N, so inverting it
with the original pixel noise σ yields chi2_raw << 1 for large segments.
The calibrated chi2 is:  chi2_calibrated = chi2_raw × sp_counts
Both are included in the output.
"""
from __future__ import annotations

import numpy as np
from typing import Optional

from skimage.segmentation import slic
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

from bio_optics.inversion import dask_oe_engine
from bio_optics.inversion.oe_engine import InversionSetup


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
        mask          = labels_flat == sid
        sp_spectra[i] = Rrs_flat[mask].mean(axis=0)
        sp_counts[i]  = mask.sum()

    return sp_spectra, sp_counts


# ---------------------------------------------------------------------------
# 3. Superpixel inversion
# ---------------------------------------------------------------------------

def invert_superpixels(
    sp_spectra: np.ndarray,
    sp_counts: np.ndarray,
    setup: InversionSetup,
    noise,
    **invert_kwargs,
) -> dict:
    """OE inversion of superpixel mean spectra.

    Passes `noise` unchanged to `dask_oe_engine.invert_image`.  The resulting
    chi2_raw will be < 1 for large segments (mean spectrum is smoother than
    individual pixels).  `invert_image_superpixel` adds a calibrated chi2:
    chi2_calibrated = chi2_raw × sp_counts.

    Parameters
    ----------
    sp_spectra  : (n_segs, n_obs)
    sp_counts   : (n_segs,)  pixel counts (used for chi2 calibration upstream)
    setup       : InversionSetup
    noise       : scalar or (n_obs,) — per-pixel noise level in sr⁻¹
    **invert_kwargs : forwarded to dask_oe_engine.invert_image (n_iter, tile_size, …)

    Returns
    -------
    results dict with x_hat/sigma/A_diag/chi2 shaped (n_segs, …)
    """
    return dask_oe_engine.invert_image(
        sp_spectra,   # (n_segs, n_obs) — flat 2-D input accepted by invert_image
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
    sp_sigma: np.ndarray,
    sp_A_diag: np.ndarray,
    k: int = 4,
    n_components: int = 6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Back-interpolate superpixel OE results to full pixel resolution.

    Uses PCA on brightness-normalised spectra + inverse-distance weighted
    average of k nearest superpixels in PCA space (Adams et al. 2021).

    Uncertainty propagation
    -----------------------
    σ²_pixel = Σ wᵢ² · σ²_spᵢ  +  Σ wᵢ · (x_hat_spᵢ − x_hat_pixel)²

    First term propagates the formal OE posterior through the IDW weights.
    Second term captures the spread among the k neighbours (interpolation
    uncertainty); zero when all k neighbours agree.

    chi2 is NOT back-interpolated here — it is copied via label array in
    invert_image_superpixel, since chi2 is a per-spectrum property.

    Parameters
    ----------
    Rrs_pixels  : (n_pixels, n_obs)
    sp_spectra  : (n_segs, n_obs)
    sp_x_hat    : (n_segs, n_fit)  physical space
    sp_sigma    : (n_segs, n_fit)  physical space
    sp_A_diag   : (n_segs, n_fit)
    k           : number of nearest superpixel neighbours (paper default: 4)
    n_components: PCA components (paper default: 6)

    Returns
    -------
    x_hat  : (n_pixels, n_fit)
    sigma  : (n_pixels, n_fit)  propagated uncertainty
    A_diag : (n_pixels, n_fit)  IDW averaged
    """
    n_segs = sp_spectra.shape[0]
    k_eff  = min(k, n_segs)

    # --- brightness normalise ------------------------------------------------
    def _norm(X):
        s = X.sum(axis=-1, keepdims=True)
        return X / np.where(s == 0, 1.0, s)

    sp_norm = _norm(sp_spectra)
    px_norm = _norm(Rrs_pixels)

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
    x_hat_nb  = sp_x_hat[indices]                          # (n_pixels, k_eff, n_fit)
    x_hat_out = (w3 * x_hat_nb).sum(axis=1)                # (n_pixels, n_fit)

    # --- A_diag: IDW average -------------------------------------------------
    A_diag_out = (w3 * sp_A_diag[indices]).sum(axis=1)     # (n_pixels, n_fit)

    # --- sigma: propagated through IDW + interpolation spread ----------------
    sigma_nb  = sp_sigma[indices]                          # (n_pixels, k_eff, n_fit)
    w2_3      = (weights ** 2)[:, :, np.newaxis]

    var_post   = (w2_3 * sigma_nb ** 2).sum(axis=1)        # formal propagation
    diff2      = (x_hat_nb - x_hat_out[:, np.newaxis, :]) ** 2
    var_interp = (w3 * diff2).sum(axis=1)                  # interpolation spread

    sigma_out = np.sqrt(var_post + var_interp)             # (n_pixels, n_fit)

    return x_hat_out, sigma_out, A_diag_out


# ---------------------------------------------------------------------------
# 5. End-to-end pipeline
# ---------------------------------------------------------------------------

def invert_image_superpixel(
    Rrs: np.ndarray,
    setup: InversionSetup,
    noise,
    n_segments: int = 1000,
    compactness: float = 0.1,
    sigma: float = 2.0,
    k: int = 4,
    n_components: int = 6,
    band_slice: Optional[np.ndarray] = None,
    store_sp_results: bool = False,
    **invert_kwargs,
) -> dict:
    """SLIC superpixel OE inversion with PCA+kNN back-interpolation.

    Returns the same dict format as dask_oe_engine.invert_image() plus
    superpixel-specific diagnostics:

      x_hat            (n_rows, n_cols, n_fit)  IDW back-interpolated
      sigma            (n_rows, n_cols, n_fit)  propagated uncertainty
      A_diag           (n_rows, n_cols, n_fit)  IDW back-interpolated
      chi2             (n_rows, n_cols)          label copy (superpixel chi2 at pixel noise)
      chi2_calibrated  (n_rows, n_cols)          chi2 × sp_counts (calibrated for mean-spectrum noise)
      fit_names        list[str]
      labels           (n_rows, n_cols) int      segment ID map
      sp_counts        (n_segs,)                 pixels per segment
      sp_results       dict (if store_sp_results=True)

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
    **invert_kwargs  : forwarded to dask_oe_engine.invert_image (n_iter, tile_size, …)
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

    # 3. Invert superpixel mean spectra
    sp_results = invert_superpixels(sp_spectra, sp_counts, setup, noise, **invert_kwargs)

    sp_x_hat  = sp_results['x_hat']   # (n_segs, n_fit) — already physical space
    sp_sigma  = sp_results['sigma']   # (n_segs, n_fit)
    sp_A_diag = sp_results['A_diag']  # (n_segs, n_fit)
    sp_chi2   = sp_results['chi2']    # (n_segs,)

    # 4. Back-interpolate x_hat, sigma, A_diag
    x_hat_flat, sigma_flat, A_diag_flat = backinterp_pca_knn(
        Rrs.reshape(n_pixels, n_obs),
        sp_spectra, sp_x_hat, sp_sigma, sp_A_diag,
        k=k, n_components=n_components,
    )

    # 5. chi2: label copy — map segment ID → superpixel chi2
    # seg_ids[i] corresponds to sp_chi2[i] (same order as aggregate_superpixels)
    seg_id_to_idx = {sid: i for i, sid in enumerate(seg_ids)}
    sp_idx_flat   = np.array([seg_id_to_idx[sid] for sid in labels_flat], dtype=np.int64)
    chi2_flat           = sp_chi2[sp_idx_flat]
    chi2_cal_flat       = sp_chi2[sp_idx_flat] * sp_counts[sp_idx_flat]

    # 6. Reshape to spatial dims
    n_fit = x_hat_flat.shape[-1]
    out = {
        'x_hat':           x_hat_flat.reshape(n_rows, n_cols, n_fit),
        'sigma':           sigma_flat.reshape(n_rows, n_cols, n_fit),
        'A_diag':          A_diag_flat.reshape(n_rows, n_cols, n_fit),
        'chi2':            chi2_flat.reshape(n_rows, n_cols),
        'chi2_calibrated': chi2_cal_flat.reshape(n_rows, n_cols),
        'fit_names':       sp_results['fit_names'],
        'labels':          labels,
        'sp_counts':       sp_counts,
    }
    if store_sp_results:
        out['sp_results'] = sp_results

    return out

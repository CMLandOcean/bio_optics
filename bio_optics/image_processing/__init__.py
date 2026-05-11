# bio_optics.image_processing
#
# Image-level engines that tile or segment a full EO image and dispatch
# per-tile / per-segment inversion to a pluggable Layer-1 engine from
# bio_optics.inversion.
#
# Engines
# -------
# dask_engine      — Dask-tiled full-image inversion; accepts any invert_fn
# superpixel_engine — SLIC segmentation + PCA/kNN back-interpolation

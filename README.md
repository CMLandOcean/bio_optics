# bio_optics
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10246861.svg)](https://doi.org/10.5281/zenodo.10246861)

The bio_optics python package is an open-source framework for forward and inverse modelling of above-water multi- and hyperspectral observations of natural water bodies. It contains the bio-optical models of Albert & Mobley (2003)[^1] and couples them to the surface reflectance model of Gege (2012)[^2]. The intention of the authors was to build a modular and extendable software package that allows the combination and creation of different models to study optical properties of natural water bodies. It can be used for simulation and analysis of spectroscopy data through inversion of the above-mentioned models. Single parts of the models can also be run as single modules, e.g., for the analysis of spectral backscattering or absorption measurements, or the correction of sun and sky glint at the water surface. 

## HOW TO CONTRIBUTE
This framework was intentionally built to be a community project. If you want to contribute or have any questions please get in touch with Marcel [marcel.koenig AT brockmann-consult.de]. If you find errors, bugs or if you have other suggestions, you can also use GitHub's issue functionality.

## UPDATES
Here we list the major changes of every release. Please take a look at the commit history for all details.

<details>
<summary>dev branch (unreleased)</summary>

- JAX-native Optimal Estimation inversion engine (`bio_optics/inversion/oe_engine.py`, `dask_oe_engine.py`, `oe_engine_optx.py`) supporting image-scale retrieval via Dask tiling and the Optimistix solver
- Superpixel inversion engine (`superpixel_engine.py`) for spatially regularised retrievals
- Two-step retrieval pipeline: Step 1 (joint 3C glint+water fit) → Step 2 (water quality on corrected spectrum)
- Benthic mapping support with LUT-based bottom reflectance and softmax bottom-fraction parameterisation (NB13/NB14)
- Full Rodgers (2000) OE diagnostics: posterior covariance `S_hat`, averaging kernel `A`, DFS, χ², Jacobian `J`, fitted spectrum `y_hat`, information content `H_info` (nats), gain matrix `G`
- Log-transform support for strictly-positive parameters (lognormal prior, no negative-value failures)
- Spatially varying priors (`x_a_image`, `S_a_inv_image`) for bathymetry-assisted retrievals
- Restructured module layout: `bio_optics/water/reflectance/`, `bio_optics/coupled_models/`
- JAX-native forward models for all reflectance models: `albert_mobley_jax`, `bi_jax`, `hope_jax` (Lee 1998/1999 shallow water), `sbop_jax` (Li 2017 shallow water), plus coupled surface models `albert_mobley_3C_jax`, `bi_3C_jax`, `hope_3C_jax`, `sbop_3C_jax`; all share the same two-layer `precompute` / `make_forward_vec` API and are `jax.jit` / `jax.jacobian` / `jax.vmap` compatible
- JAX atmosphere layer: `transmittance_jax`, `downwelling_irradiance_jax` (dual-mode: pre-baked Ed for fixed scenes, or on-the-fly for atmosphere/geometry retrieval), `sky_radiance_jax`, `scattering_jax`
- Package-wide cleanup: replaced all mutable default `*_res=[]` arguments with `*_res=None` across 21 files
- See `docs/OE_PIPELINE.md` for the step-by-step inversion workflow

</details>

<details>
<summary>Version 0.0.2</summary>

- Integration of HEREON bio-optical model including a split of `a_d` into `a_md` and `a_bd`, and `b_d` into `b_bd` and `b_md` 
- Renaming of key variables and functions following the style `Quantity_Specification` to better resemble symbolism typically used in the bio-optical community (e.g., $b_{bw}$ is now represented as `bb_w` instead of `b_bw` and $R_{rs}$ is now represented as `Rrs` instead of `R_rs`). See the new GLOSSARY for details.
- Integration of OPSHAL for identification of optically shallow water
- Integration of multiple (semi-)empirical and semi-analytical algorithms for Chlorophyll-a, CDOM, SPM and Turbidity retrieval
</details>


## GLOSSARY
| Symbol            | Code representation(s) | Description | Unit |
| :---------------- | :------- | :---- | :---- |
| $a(\lambda)$       |   `a`   | Absorption coefficient of a natural water body as the sum of the absorption coefficients of pure water, phytoplankton, yellow substances/CDOM, and non-algal particles | $\text{m}^{-1}$ |
| $a_w(\lambda)$       |   `a_w`   | Absorption coefficient of pure water | $\text{m}^{-1}$ |
| $a_{ph}(\lambda)$       |   `a_ph`, `a_phy`,`a_Phi`   | Absorption coefficient of phytoplankton | $\text{m}^{-1}$ |
| $a_Y(\lambda)$       |   `a_Y`,`a_Y_pow`,`a_Y_gauss`,`a_Y_exp_gauss`   | Absorption coefficient of CDOM or yellow substances | $\text{m}^{-1}$ |
| $a_Y^{norm}(\lambda)$       |   `a_Y_norm`   | Normalized absorption coefficient of CDOM or yellow substances | $\text{m}^{-1}$ |
| $a_{NAP}(\lambda)$       |   `a_NAP`   | Absorption coefficient of non-algal particles | $\text{m}^{-1}$ |
| $a_{NAP}^{norm}(\lambda)$       |   `a_NAP_norm`   | Normalized absorption coefficient of non-algal particles | $\text{m}^{-1}$ |
| $b(\lambda)$       |   `b`   | Scattering coefficient of water | $\text{m}^{-1}$ |
| $b_{phy}(\lambda)$       |   `b_phy`   | Scattering coefficient of phytoplankton | $\text{m}^{-1}$ |
| $b_b(\lambda)$       |   `bb`   | Backscattering coefficient of a natural water body as the sum of the backscattering coefficients of pure water, phytoplankton and non-algal particles | $\text{m}^{-1}$ |
| $b_{bphy}(\lambda)$       |   `bb_phy`   | Backscattering coefficient of phytoplankton | $\text{m}^{-1}$ |
| $b_{bX}(\lambda)$       |   `bb_X`   | Backscattering coefficient of particles of type I with arbitrary wavelength dependency | $\text{m}^{-1}$ |
| $b_{bMie}(\lambda)$       |   `bb_Mie`   | Backscattering coefficient of particles of type II | $\text{m}^{-1}$ |
| $b_{bNAP}(\lambda)$       |   `bb_NAP`   | Backscattering coefficient of non-algal particles (NAP) as a mixture of two types (I and II) with spectrally different backscattering coefficients | $\text{m}^{-1}$ |
| $c(\lambda)$       |   `c`   | Attenuation coefficient of water | $\text{m}^{-1}$ |
| $c_d(\lambda)$       |   `c_d`   | Attenuation coefficient of detritus | $\text{m}^{-1}$ |
| $R(\lambda)$       |   `R`   | Reflectance | unitless |
| $R_b(\lambda)$       |   `R_b`   | Benthic reflectance | unitless |
| $r_{rs}(\lambda)$       |   `rrs`   | Radiance reflectance | $\text{sr}^{-1}$ |
| $R_{rs}(\lambda)$       |   `Rrs`   | Remote sensing reflectance | $\text{sr}^{-1}$ |


## PLEASE GIVE US CREDIT
This python package has initially been created by members of the CarbonMapper Land and Ocean Program at the Center for Global Discovery and Conservation Science at Arizona State University. When using this framework, please use the following citation:

> _König, M., Noel, P., Hondula. K.L., Jamalinia, E., Dai, J., Vaughn, N.R., Asner, G.P. (2023): bio_optics python package (Version x) [Software]. Available from https://github.com/CMLandOcean/bio_optics. [https://doi.org/10.5281/zenodo.10246860]_

## ACKNOWLEDGEMENTS
This repository is a framework for bio-optical modelling and builds on the extensive work of many researchers. It has mainly been inspired by the work of Peter Gege and the Water Color Simulator (WASI)[^3][^4][^5][^6]. Please give proper attribution when using this toolbox.

## LITERATURE
[^1]: Albert & Mobley (2003): An analytical model for subsurface irradiance and remote sensing reflectance in deep and shallow case-2 waters. [doi.org/10.1364/OE.11.002873](https://doi.org/10.1364/OE.11.002873)

[^2]: Gege (2012): Analytic model for the direct and diffuse components of downwelling spectral irradiance in water. [10.1364/AO.51.001407](https://doi.org/10.1364/AO.51.001407)

[^3]: Gege (2004): The water color simulator WASI: an integrating software tool for analysis and simulation of optical in situ spectra. [10.1016/j.cageo.2004.03.005](https://doi.org/10.1016/j.cageo.2004.03.005)

[^4]: Gege & Albert (2006): A Tool for Inverse Modeling of Spectral Measurements in Deep and Shallow Waters. [https://doi.org/10.1007/1-4020-3968-9_4](https://doi.org/10.1007/1-4020-3968-9_4)

[^5]: Gege (2014): WASI-2D: A software tool for regionally optimized analysis of imaging spectrometer data from deep and shallow waters. [https://doi.org/10.1016/j.cageo.2013.07.022](https://doi.org/10.1016/j.cageo.2013.07.022)

[^6]: Gege (2021): The Water Colour Simulator WASI. User manual for WASI version 6.

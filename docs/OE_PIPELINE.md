# OE Inversion Pipeline

Step-by-step workflow for running the bio_optics Optimal Estimation inversion.
Follow these phases in order for any new sensor/scene combination.

---

## Phase 1 — Setup (run once per sensor configuration)

### 1.1 Precompute spectral lookup tables

Choose a forward model based on your water type.  All models share the same two-layer API.

**Deep / optically deep water:**
```python
from bio_optics.water.reflectance import albert_mobley_jax
# or: from bio_optics.water.reflectance import bi_jax  (HEREON model)

pre = albert_mobley_jax.precompute(wavelengths)
# pre contains: a_w, bb_w, R_b_i, and other spectral tables
```

**Shallow water (Lee 1998/1999 HOPE model):**
```python
from bio_optics.coupled_models import hope_3C_jax
# or: from bio_optics.coupled_models import sbop_3C_jax  (Li 2017)

pre = hope_3C_jax.precompute(wavelengths, theta_sun=np.radians(30))
# pre contains: a_w, bb_w, A0, A1, R_b_i + irradiance arrays
```

**Atmosphere modes** (all 3C coupled models):
- **Mode A** (default): supply `theta_sun` → Ed arrays pre-baked, zero per-call overhead.  Use when geometry is fixed per scene.
- **Mode B**: pass `theta_sun=None` → Ed computed on-the-fly; `theta_sun`, `alpha`, `beta`, etc. can be free fit parameters.

```python
pre_A = albert_mobley_3C_jax.precompute(wavelengths, theta_sun=np.radians(30))  # Mode A
pre_B = albert_mobley_3C_jax.precompute(wavelengths, theta_sun=None)             # Mode B
```

For LUT-based bottom parameterisation, optionally subset `R_b_i` to the types you want:
```python
# e.g. keep only sand (index 1) and seagrass (index 5) from the 6-type LUT
selected = [1, 5]
R_b_i_sub = pre['R_b_i'][:, selected]                   # (n_obs, 2)
R_b_i_6   = np.zeros((len(wavelengths), 6))
R_b_i_6[:, :len(selected)] = R_b_i_sub
pre = {**pre, 'R_b_i': jnp.array(R_b_i_6)}
```

### 1.2 Define the parameter configuration
```python
fit_config = {
    # name:     vary    value   sigma_a  log
    'C_0':   (True,   0.5,    1.0,     True),   # phytoplankton [mg/m³], log-prior
    'C_Y':   (True,   0.1,    1.0,     True),   # CDOM [1/m], log-prior
    'C_Mie': (False,  0.01,   None,    True),   # SPM — fixed
    'zB':    (True,   5.0,    0.7,     True),   # depth [m], log-prior (±factor-of-2)
    'f_0':   (False,  1.0,    None,    False),  # bottom fraction type 0 — fixed
    # ... other geometry / IOP parameters at their expected values with vary=False
}
```

**sigma_a conventions:**
- For log-params: `sigma_a` is a **relative (fractional) uncertainty** in log-space.
  `ln(2) ≈ 0.69` = factor-of-2; `1.0` = factor-of-e (~2.7×). Use loose values (0.7–1.5) for Phase 1.
- For linear params: `sigma_a` is a physical std (same units as the parameter).
- Use `oe_engine.sigma_to_relative()` to convert physical stds to log-space sigmas.

### 1.3 Build the forward function and inversion setup
```python
from bio_optics.inversion import oe_engine

all_names = list(fit_config.keys())
f_vec     = albert_mobley_jax.make_forward_vec(all_names, pre)

params    = build_lmfit_params(fit_config)      # your helper to make lmfit.Parameters
sigma_a   = {k: v[2] for k, v in fit_config.items() if v[0]}
log_params = [k for k, v in fit_config.items() if v[3]]

setup = oe_engine.build_inversion(params, f_vec, sigma_a, log_params=log_params)
# setup.fit_names, setup.x_a, setup.S_a_inv, setup.log_mask, setup.f_fit
```

---

## Phase 2 — First run: generic parameterisation

Use loose priors and a conservative noise estimate. The goal is a convergent solution,
not a perfect one — diagnostics in Phase 3 will guide all adjustments.

```python
from bio_optics.image_processing import dask_engine

noise = 0.005   # sr⁻¹; ~1% of median Rrs is a safe starting point

results = dask_engine.invert_image(
    Rrs_image, setup,
    noise      = noise,
    n_iter     = 15,
    tile_size  = 4096,
    store_y_hat = True,    # keep simulated spectra for residual inspection
)
# results keys: x_hat, sigma, A_diag, chi2, H_info, y_hat (if store_y_hat), G (if store_gain), fit_names
```

For per-pixel bottom reflectance (measured albedo from low-tide image):
```python
R_b_i_image          = np.tile(np.array(pre['R_b_i']), (n_rows, n_cols, 1, 1))
R_b_i_image[..., 0]  = albedo_image   # per-pixel measured albedo → first bottom type
results = dask_engine.invert_image(..., aux_image={'R_b_i': R_b_i_image})
```

---

## Phase 3 — Noise calibration (chi2 check)

**Target:** `median(chi2) ≈ 1`. Values between 0.8 and 1.2 are acceptable.

```python
import numpy as np

median_chi2 = np.nanmedian(results['chi2'])
print(f"median chi2 = {median_chi2:.2f}")

# Implied correct noise for chi2 = 1:
noise_implied = noise * np.sqrt(median_chi2)
print(f"implied noise = {noise_implied:.4f} sr⁻¹")
```

| Symptom | Cause | Action |
|---|---|---|
| median chi2 >> 1 | noise too low | `noise = noise * sqrt(median_chi2)`, rerun |
| median chi2 << 1 | noise too high or model over-constrained | decrease noise, or free more parameters |
| chi2 elevated in patches | unmodelled glint / clouds / shallow substrate mismatch | mask those pixels, investigate |
| chi2 very low near coasts | forward model too flexible | check if parameters are hitting bounds |

Iterate: rerun `invert_image()` with `noise_implied` until median chi2 is stable near 1.

---

## Phase 4 — Parameter pruning (A_diag and H_info check)

For each free parameter `i`, compute the median averaging-kernel diagonal.
`A_diag[i] ∈ [0, 1]`: 1 = fully data-driven, 0 = prior-dominated (no information from data).

```python
for i, name in enumerate(results['fit_names']):
    a_med  = np.nanmedian(results['A_diag'][..., i])
    s_med  = np.nanmedian(results['sigma'][..., i])
    print(f"{name:12s}  A_diag={a_med:.2f}  posterior_sigma={s_med:.3f}")

# Scene-level information content (nats): how much total entropy was removed from the prior
print(f"Median H_info: {np.nanmedian(results['H_info']):.4f} nats")
```

**Interpreting H_info:**
- H_info is the Shannon information content (Rodgers 2000 eq. 2.80): total entropy removed from the prior by the measurement, summed across all retrieved parameters.
- For a **single uncorrelated parameter**: `H = −0.5·ln(1 − A[i,i])`, so H and A[i,i] carry identical information.
- For **multiple parameters**: H is the log-determinant analogue of DFS (trace). DFS counts *how many* params are constrained; H measures *how tightly* in total. H is dominated by the best-constrained parameters.
- Use H_info to compare scenes, sensor configurations, or retrieval setups — higher is better.
- Rule of thumb: H_info / n_fit gives the average information per parameter in nats; `exp(H_info/n_fit)` is the geometric-mean ratio of prior-to-posterior σ.

| Symptom | Diagnosis | Fix |
|---|---|---|
| `A_diag ≈ 0` | parameter not constrained | fix at prior mean (`vary=False`) |
| `posterior sigma ≈ prior sigma` | no information gained | fix parameter or tighten prior |
| `x_hat` uniform across scene | parameter unconstrained | fix it |
| `A_diag ≈ 1` and `posterior sigma << prior sigma` | well-constrained | keep free |

After fixing unconstrained parameters, rebuild `setup` with `vary=False` for those names and rerun.

---

## Phase 5 — Prior tuning (sigma_a adjustment)

Once the parameter set is stable, adjust the prior widths to reflect domain knowledge.

```python
# Example: tighten depth prior where bathymetry survey is available
fit_config['zB'] = (True, 5.0, 0.4, True)   # sigma_a = 0.4 ≈ ±50%

# Or build spatially varying priors from an auxiliary map
x_a_image                = np.tile(np.array(setup.x_a), (n_rows, n_cols, 1))
zB_idx                   = setup.fit_names.index('zB')
x_a_image[..., zB_idx]   = np.log(bathymetry_map)   # log-space for log-params

S_a_inv_image            = np.tile(np.array(setup.S_a_inv), (n_rows, n_cols, 1, 1))
reliable_mask            = bathymetry_map < 5
S_a_inv_image[reliable_mask, zB_idx, zB_idx] *= 4   # 2× tighter sigma

results = dask_engine.invert_image(
    Rrs_image, setup, noise=noise_final, n_iter=15,
    x_a_image=x_a_image, S_a_inv_image=S_a_inv_image,
)
```

Confirm chi2 is still ≈ 1 after prior changes. Tightening priors reduces effective degrees of freedom
and can pull chi2 below 1 if too aggressive.

---

## Phase 6 — Final run and interpretation

```python
results = dask_engine.invert_image(
    Rrs_image, setup, noise=noise_final, n_iter=15,
    store_y_hat=False,   # saves memory once diagnostics are done
)

# Convert to xarray
ds = dask_engine.to_dataset(results, spatial_dims=('y', 'x'), coords=coords)

# Physical-space results are already in results['x_hat'] (dask_engine applies to_physical)
# sigma is already in physical space (delta-method for log-params)

# Bottom type fractions (if using softmax f_mix_* parameterisation)
fracs = oe_engine.bottom_fractions(results['x_hat'], results['fit_names'])
if fracs is not None:
    ds['f_seagrass'] = (('y', 'x'), fracs[..., 0])
    ds['f_sand']     = (('y', 'x'), fracs[..., 1])
```

---

## Summary checklist

```
[ ] Phase 1: precompute + build_inversion with all candidate free params, loose priors
[ ] Phase 2: first invert_image run, noise ~0.005 sr⁻¹, n_iter=15, store_y_hat=True
[ ] Phase 3: check median chi2; adjust noise = noise * sqrt(median_chi2); rerun until chi2 ≈ 1
[ ] Phase 4: check A_diag per param; fix unconstrained ones; rebuild setup; rerun
[ ] Phase 5: tighten sigma_a using domain knowledge or auxiliary maps; confirm chi2 still ≈ 1
[ ] Phase 6: final run with store_y_hat=False; export to xarray; compute bottom fractions if needed
```

---

## Key functions

| Function | Location | Purpose |
|---|---|---|
| `precompute()` | `water/reflectance/albert_mobley_jax.py` | Deep water: spectral LUTs |
| `precompute()` | `water/reflectance/bi_jax.py` | HEREON deep water: spectral LUTs |
| `precompute()` | `water/reflectance/hope_jax.py` | Lee 1998/1999 shallow water: spectral LUTs |
| `precompute()` | `water/reflectance/sbop_jax.py` | Li 2017 shallow water: spectral LUTs |
| `precompute()` | `coupled_models/albert_mobley_3C_jax.py` | Deep water + surface (3C) |
| `precompute()` | `coupled_models/hope_3C_jax.py` | Shallow water + surface (3C, Lee) |
| `precompute()` | `coupled_models/sbop_3C_jax.py` | Shallow water + surface (3C, Li 2017) |
| `make_forward_vec()` | any `*_jax.py` model | Build `f(params_vec, aux=None)` for JIT/vmap |
| `build_inversion()` | `inversion/oe_engine.py` | Build `InversionSetup` from lmfit.Parameters |
| `sigma_to_relative()` | `inversion/oe_engine.py` | Convert physical σ to log-space fractional σ |
| `invert_image()` | `inversion/image_processing/dask_engine.py` | Tile-parallel image inversion |
| `to_dataset()` | `inversion/image_processing/dask_engine.py` | Convert result dict to xarray Dataset |
| `bottom_fractions()` | `inversion/oe_engine.py` | Convert `f_mix_*` logits to fractions |
| `posterior_sigma_physical()` | `inversion/oe_engine.py` | Delta-method σ for log-params |

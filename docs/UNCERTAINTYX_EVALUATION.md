# uncertaintyx — Integration Evaluation

`uncertaintyx` (2026.1.0) is a JAX-native framework for tensor-level uncertainty
propagation in Earth Observation algorithms, developed by a colleague as a sibling
repo (`../uncertaintyx`).  This document records the evaluation of whether and how
to integrate it into bio_optics.

## What it provides

- **Tensor-form Law of Propagation of Uncertainty (LPU)** — propagates input
  uncertainty through differentiable forward models without flattening spatial
  dimensions
- **JAX OE solver** (`retrieve/oe/jax.py`) — optimal estimation via optax/optimistix
- **Errors-in-variables fitting** (`fit/eiv/jax.py`) — not currently in bio_optics
- **Abstract interfaces** — `F`, `M`, `Fitting`, `Retrieving` ABCs for a
  backend-agnostic architecture (JAX, NumPy, PyTorch adapters)
- **Automatic Jacobians** via `jax.jacobian` (forward/reverse selectable)
- **Full vmap/jit support** throughout

## Pros

- JAX-native; aligns with the existing bio_optics JAX stack
- OE solver already implemented — less code to maintain ourselves
- Tensor-form LPU is genuinely new functionality not in bio_optics
- Errors-in-variables fitting adds a capability we currently lack
- Clean ABCs provide a principled structure for multi-backend support

## Cons

- **Different solver architecture** — their OE uses first-order optax/optimistix
  gradient optimizers; `oe_engine.py` uses explicit Gauss-Newton with exact
  Jacobians, log-space parameterization, and per-pixel prior maps.  Replacing
  would sacrifice significant tuning.
- **Wrapper boilerplate** — our `make_forward_vec` / `precompute` API does not
  fit the `F`/`M` interface without non-trivial adaptation.
- **Unversioned local dependency** — `../uncertaintyx` is a sibling repo, not a
  published package; maintenance is coupled to the colleague's development cycle.
- **Overlaps with existing code** — `oe_engine.py`, `image_processing/dask_engine.py`, and
  `optimistix_engine.py` already cover our OE needs.  Full adoption would mean
  replacing or running parallel systems.

## Comparison with our OE implementation

Both frameworks solve the same underlying linear-Gaussian inverse problem, but
differ in solver type and which diagnostics are returned as first-class outputs.

**Theoretical framing:**
`oe_engine.py` follows Rodgers (2000, Ch. 5), which defines the averaging kernel,
DFS, gain matrix, and information content as primary retrieval diagnostics.
`uncertaintyx` cites Tarantola (2005), which is the more general formulation:
arbitrary prior distributions, exact Hessian inversion at the MAP estimate, and
Quasi-Newton / Newton minimisation.  Rodgers' formalism is a linearised special
case of Tarantola valid for Gaussian priors and noise.

**Solver and posterior covariance accuracy:**

Both approaches compute a **Laplace approximation** to the posterior — a Gaussian
fitted at the MAP estimate.  They differ in how the Hessian at the MAP is obtained:

- `oe_engine.py`: Gauss-Newton drops the second-order residual term of the true Hessian:

  ```
  H_true = J^T S_ε^{-1} J + Sa^{-1}  −  Σ_k [y_k − f_k(x)] · ∂²f_k/∂x²
                                          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                                          GN drops this term
  ```

  `H_GN` equals the Fisher information matrix — the *expected* exact Hessian
  averaged over noise realisations.  When residuals at the solution are small
  (good fit) or the model is weakly nonlinear, `H_GN ≈ H_true`.

- uncertaintyx: computes `jax.hessian(cost)(x_opt)` after L-BFGS convergence —
  the exact Hessian of the cost function at the MAP estimate, including
  second-order residual contributions from the actual noise draw.

The colleague's exact Hessian gives a better Gaussian approximation for a *single
retrieval* when the forward model is strongly nonlinear.  The GN Hessian gives the
statistically correct posterior covariance *in expectation* across many noise
realisations (it is the Cramér-Rao bound).  For strongly non-Gaussian posteriors
neither approach is exact — MCMC or variational inference would be needed.

Our log-space parameterisation reduces effective nonlinearity significantly, so
the GN approximation error is smaller than in a naive linear parameterisation.
For typical water-quality forward models (moderately nonlinear, small residuals at
convergence) the practical difference is minor.

**Diagnostic outputs:**

| Output | `oe_engine.py` | uncertaintyx |
|---|---|---|
| Posterior mean `x_hat` | explicit | `xopt` |
| Posterior covariance `S_hat` | explicit | `xcov` |
| Averaging kernel `A` | explicit | derive post-hoc |
| Degrees of freedom `dfs` | explicit | derive post-hoc |
| Chi-squared `chi2` | explicit | only `zvar` (residual variance, not normalised) |
| Information content `H_info` | explicit | derive post-hoc |
| Gain matrix `G` | explicit | derive post-hoc |
| Jacobian `J` | explicit | not returned |

## Relationship to OE uncertainty

OE (`oe_engine.py`) and LPU (uncertaintyx) propagate uncertainty in **opposite
directions** — they are complementary, not overlapping.

**OE — inverse direction**

Given noisy observations, how uncertain is the retrieved state?

```
observation noise S_ε  →  [OE inversion]  →  posterior state uncertainty S_hat
prior uncertainty S_a  ↗
```

`S_hat = (J^T S_ε^{-1} J + S_a^{-1})^{-1}`

The inputs are `S_ε` (noise on Rrs) and `S_a` (prior on parameters).  The output
is `S_hat` — uncertainty on the retrieved `x_hat` (concentrations, depth, etc.).

**LPU — forward direction**

Given uncertain inputs or parameters, how uncertain is the model output?

```
parameter uncertainty S_p  →  [LPU: S_y ≈ J_p · S_p · J_p^T]  →  output uncertainty S_y
```

This propagates uncertainty *through* the forward model, not *from* observations
back to parameters.

**Full uncertainty chain**

A complete uncertainty budget uses both in sequence:

1. **Calibration / IOP uncertainty → Rrs uncertainty** (LPU, forward): given
   uncertain water absorption coefficients, how uncertain is the predicted Rrs?
   — currently missing from bio_optics
2. **Rrs noise → retrieved parameter uncertainty** (OE, inverse): already have
   this as `S_hat`
3. **Retrieved parameter uncertainty → derived product uncertainty** (LPU,
   forward): given uncertain retrieved C_0 / C_Y, how uncertain is the computed
   Kd? — also currently missing

LPU fills steps 1 and 3, which OE does not touch.  This is why cherry-picking
the tensor LPU is more valuable than replacing the OE solver.

## Recommendation

**Cherry-pick rather than full adoption.**

The highest-value piece is the **tensor LPU** (`m.lpu_p` / `yunc_t`) for
propagating instrument noise or model-parameter uncertainty through the forward
model to output uncertainty maps — that is genuinely new and complementary.

The OE solver replacement and the `F`/`M` ABC refactor are harder to justify:
the refactoring cost is high and the benefit over what `oe_engine.py` already
provides is low.

## Should we replace our implementations?

The F/M abstraction is aesthetically appealing, but replacing our engines with
uncertaintyx would require either accepting a weaker OE solver or porting
`oe_engine.py` into uncertaintyx — same code, different home.

**On the OE solver specifically:**
Gauss-Newton is the *correct* solver for Bayesian OE because it uses the exact
Fisher information matrix `J^T S_ε^{-1} J` as the Hessian, giving the
analytically exact posterior covariance `S_hat = H^{-1}`.  L-BFGS approximates
the Hessian numerically, so `S_hat`, `A`, `DFS`, and `H_info` are all less
accurate.  For a retrieval where the posterior covariance is a primary output
(not just a convergence aid), this is a meaningful regression.

Expanding uncertaintyx to match what we have would mean adding: Gauss-Newton
solver, log-space parameterization, per-pixel prior maps (`x_a_image`), Dask
tiling, JIT static-argname setup, and all explicit diagnostic outputs.  That is
most of `oe_engine.py` and `image_processing/dask_engine.py`.

**On LSQ:**
No equivalent in uncertaintyx.  The `fit/eiv/` module is errors-in-variables
for statistical model fitting, not forward-model inversion via
Levenberg-Marquardt.  `lsq_engine_optx.py` uses optimistix LM directly —
which uncertaintyx itself uses under the hood anyway.

| What to take from uncertaintyx | What to keep in bio_optics |
|---|---|
| Tensor LPU (genuinely new) | `oe_engine.py` — GN is the right OE solver |
| EIV fitting (genuinely new) | `image_processing/dask_engine.py` — image infrastructure |
| | `lsq_engine_optx.py` — no uncertaintyx equivalent |

## Should we adopt the F/M abstraction?

No.  The LPU computation itself is just:

```python
J  = jax.jacobian(f)(x)
S_y = J @ S_x @ J.T
```

This works on any JAX callable — no F/M wrapping required.  Our
`make_forward_vec` closures are already exactly what `jax.jacobian` expects.
Adopting F/M would mean subclassing ABCs across all 10+ forward models
(`albert_mobley_jax`, `hope_jax`, `lee_jax`, …) to gain the ability to call
code we could write in 3 lines ourselves.

The F/M abstraction earns its keep in uncertaintyx because it needs to support
multiple backends (JAX, NumPy, PyTorch) and provide unified dispatch.  We are
JAX-only and our `make_forward_vec` / `precompute` callable convention is already
consistent across all models — no new abstraction layer is needed.

### Suggested next step

Prototype `m.lpu_p` on one forward model (e.g. `albert_mobley_jax`) to see
whether the tensor LPU output integrates cleanly with the existing
`InversionSetup` / `dask_engine` workflow before committing to broader
adoption.

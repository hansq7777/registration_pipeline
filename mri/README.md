# MRI Registration Pipeline

Scope:

- resampling to a strict common grid
- ANTs-based rigid/affine/nonlinear registration
- multistart selection
- storyboard/QC generation
- nonlinear and linear benchmark sweeps
- experiment supervision/watchdog for long runs

Current recommended default from the `2026-02-26` brain-only benchmark:

- input contract: fixed/moving volumes already on one reference grid, with required brain masks
- preprocess: fixed/moving `N4`, moving `DenoiseImage`
- linear init: `translation_then_rigid`
- affine fallback: `on`
- nonlinear default: `syn` (`SyN[0.05,3,0]`)
- stage order: `rigid -> affine -> syn`
- keep QC gates on: affine sanity, Jacobian p01/p99, negative-Jacobian fraction, warp-energy advisory, storyboard/manifest outputs
- practical fallback: `bspline_syn + translation_init`

Main scripts:

- `scripts/resample_contract.py`
- `scripts/template_register.py`
- `scripts/template_register_multistart.py`
- `scripts/template_build.py`
- `scripts/registration_storyboard.py`
- `scripts/registration_nonlinear_benchmark.py`
- `scripts/registration_linear_benchmark.py`
- `scripts/registration_experiment_supervisor.py`
- `scripts/registration_experiment_watchdog.py`
- `scripts/orientation_qc_selector.py`

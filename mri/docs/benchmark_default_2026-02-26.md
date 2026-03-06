# MRI Registration Default (2026-02-26 Brain-Only Benchmark)

Source run:

- `mouse_mt_pipeline/test/registration_experiment_20260226_rerun_noopt`

Default to freeze for MRI-side work:

1. Inputs must already be resampled to one reference grid.
2. Fixed and moving brain masks are mandatory.
3. Preprocess with `N4` on both images; denoise the moving image before `N4`.
4. Use `translation_then_rigid` as the default linear initialization.
5. Keep affine fallback enabled.
6. Use `Affine -> SyN[0.05,3,0]` as the default nonlinear chain.
7. Keep stage order fixed as `rigid -> affine -> syn`.
8. Keep QC outputs and transform provenance on every run.

Best observed combo in the controlled benchmark:

- nonlinear: `syn`
- linear profile: `translation_init`
- Dice: `0.941233`
- CC: `0.928446`
- NMI: `1.357997`
- Jacobian negative fraction: `0.000000`

Close fallback:

- nonlinear: `bspline_syn`
- linear profile: `translation_init`
- Dice: `0.940677`
- CC: `0.930046`
- NMI: `1.353694`
- runtime was substantially shorter than the `syn` winner

Rejected/unstable families in the same benchmark:

- `bspline_displacement_field`
- `gaussian_displacement_field`
- `time_varying_bspline_velocity_field`

Reason:

- they failed either ANTs execution directly or downstream Jacobian hard-gate checks, so they should not be the frozen default.

# Myelin Mask Compute Profile Tuning 20260317

## Goal

Reduce `Step 2` mask runtime by switching from full exported crop computation to downsample-first computation, while keeping M3 mask quality close to the current baseline.

## Implementation

- Added `Mask Compute Profile` to the GUI:
  - `Standard 2048px`
  - `Fast 1600px`
  - `Full Export Resolution`
- `Step 2` batch prediction now:
  1. downsamples `crop_raw.png` to the selected working size,
  2. runs mask prediction on the working image,
  3. rescales masks back to the exported crop size with nearest-neighbor.
- `Step 3` single-section `Run/Refresh Auto Mask` now uses the same compute-profile mechanism.
- Current M3 kernels were made scale-aware before low-resolution inference:
  - entropy radius
  - candidate close/open
  - hysteresis close/open
  - fallback hybrid erode/dilate/close

## Result Summary

### Medium crops

Probe output:
- [20260317_myelin_mask_compute_profiles_probe_medium](/mnt/c/Users/Siqi/Desktop/REVIEW/20260317_myelin_mask_compute_profiles_probe_medium)

Sections:
- `2502_42`
- `2502_54`

Mean results:

| Profile | Mean Runtime (s) | Mean Dice | Mean Pred/GT |
|---|---:|---:|---:|
| `standard` | `95.28` | `0.9969` | `0.9986` |
| `fast` | `32.18` | `0.8281` | `0.7421` |

Interpretation:
- `Standard 2048px` keeps quality essentially intact on these medium-large crops.
- `Fast 1600px` is much faster, but under-segments heavily on at least one sampled section and is not safe as the default.

### Smaller crop

Probe output:
- [20260317_myelin_mask_compute_profiles_probe_smallfull](/mnt/c/Users/Siqi/Desktop/REVIEW/20260317_myelin_mask_compute_profiles_probe_smallfull)

Section:
- `2501_5`

Results:

| Profile | Runtime (s) | Dice | Pred/GT |
|---|---:|---:|---:|
| `full` | `99.43` | `0.9757` | `1.0493` |
| `standard` | `11.32` | `0.9698` | `1.0620` |
| `fast` | `21.89` | `0.9737` | `1.0535` |

Interpretation:
- Even on a relatively small crop, `Full` is still too slow for batch use.
- `Standard` gives an almost identical mask with a large runtime reduction.
- `Fast` can be acceptable on some small crops, but its behavior is not consistent across medium crops.

## Decision

- Keep `Standard 2048px` as the default compute profile for Step 2 and Step 3.
- Keep `Fast 1600px` as an optional speed mode only.
- Keep `Full Export Resolution` as a manual fallback only, not the default.

## Practical Conclusion

The current slowdown is not just a pathological single-slide issue. Full-resolution M3 on exported workspace crops is too slow for routine batch prediction. A downsample-first path is required, and `Standard 2048px` is the best current tradeoff between speed and mask stability.

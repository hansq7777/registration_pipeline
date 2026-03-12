# Nissl BBox GT Search 2026-03-11

## Scope

- GT source only:
  - `D:\Research\Image Analysis\Nanozoomer scans\20250424 Nissl cytoarchitectonic counterpart\Tissue&Masks\test`
- Evaluated sections:
  - `2507_73`
  - `2507_79`
  - `2507_85`
  - `2507_91`
  - `2507_97`
  - `2507_103`
- All sections include per-section `metadata.json` and can be mapped back to original slide coordinates.

## Evaluation Rule

Primary rule:

- generated bbox must cover the GT mask location in original slide space

Secondary rule:

- once coverage saturates, compare full proposal rectangle against the GT crop rectangle in slide space
- prefer higher proposal-vs-GT-crop IoU and smaller proposal area

This differs from the previous clipped metric that could not distinguish overly large proposals once GT coverage reached `1.0`.

## Outputs

- initial benchmark:
  - `/mnt/c/Users/Siqi/Desktop/REVIEW/20260311_bbox_test_nissl_2507`
- fine search around small uniform padding:
  - `/mnt/c/Users/Siqi/Desktop/REVIEW/20260311_bbox_test_nissl_2507_refine`

## Main Finding

Unlike the myelin case, the current Nissl bbox policy is already very strong on GT coverage.

Key result:

- many candidate policies achieve:
  - `mean_mask_coverage_recall = 1.0`
  - `full_coverage_rate_99 = 1.0`

So the Nissl problem is not "recover missing GT area".
The real optimization axis is:

- keep full GT coverage
- reduce unnecessary crop area if desired

## Coarse Search Result

From the first stain-aware benchmark:

- current default `baseline_uniform8` already gives full GT coverage on all 6 sections
- more aggressive projection/top-bias methods do not improve coverage
- many larger-padding methods only make the crop bigger

This is very different from myelin, where coverage-first expansion was necessary.

## Fine Search Result

The refined uniform-padding search showed:

- `raw_support_bbox`
  - not safe enough
  - mean coverage recall drops to `0.9991`
  - misses a small amount on some sections
- `uniform01_min0`
  - full GT coverage on all 6 sections
  - mean proposal-vs-GT-crop IoU: `0.7412`
  - mean proposal-area / GT-crop-area: `0.7412`
- `uniform02_min0`
  - full GT coverage on all 6 sections
  - mean proposal-vs-GT-crop IoU: `0.7764`
  - mean proposal-area / GT-crop-area: `0.7764`
- `uniform04_min0`
  - full GT coverage on all 6 sections
  - mean proposal-vs-GT-crop IoU: `0.8502`
  - mean proposal-area / GT-crop-area: `0.8502`
- `uniform06_min0`
  - full GT coverage on all 6 sections
  - mean proposal-vs-GT-crop IoU: `0.9222`
  - mean proposal-area / GT-crop-area: `0.9222`
- `baseline_uniform8`
  - full GT coverage on all 6 sections
  - mean proposal-vs-GT-crop IoU: `1.0000`
  - mean proposal-area / GT-crop-area: `1.0000`

## Interpretation

### What this means about the current default

`baseline_uniform8` is not failing on Nissl GT.

More than that:

- on this GT set, it matches the manual GT crop rectangle exactly
- so if the current manual crop rectangle is the desired reference output, the present default is already the best-matching policy

### What can still be improved

If the goal is not "match manual crop rectangle" but "keep full GT coverage with tighter crops", then smaller uniform padding is viable.

The smallest safe candidate on this GT set is:

- `uniform01_min0`

But that is likely too aggressive to promote directly as a new default from only one slide.

The safer efficiency-oriented candidates are:

- `uniform02_min0`
- `uniform04_min0`
- `uniform06_min0`

These all keep full GT coverage in the present GT set.

## Recommended Candidates

### Candidate 1: `baseline_uniform8`

Recommended if the target is:

- reproduce current hand-drawn Nissl crop boxes
- maximize safety margin
- keep behavior stable with existing outputs

### Candidate 2: `uniform06_min0`

Recommended balanced alternative if the target is:

- preserve full GT coverage
- reduce crop area moderately
- stay close to current manual crop geometry

### Candidate 3: `uniform02_min0`

Recommended only as an aggressive efficiency candidate if the target is:

- minimize crop size while still covering GT on the current slide

Risk:

- too little evidence yet to promote as default for future unseen Nissl slides

## Practical Conclusion

Current conclusion for Nissl:

- bbox is not the bottleneck
- current default does not show the myelin-style under-coverage problem
- changing to a more complex projection policy is unnecessary

Recommended next action:

- keep `baseline_uniform8` as the stable Nissl default for now
- optionally expose `uniform06_min0` as an efficiency-oriented alternative for future testing

Do not transfer the myelin bbox logic directly to Nissl. The GT-backed evidence here does not support that.

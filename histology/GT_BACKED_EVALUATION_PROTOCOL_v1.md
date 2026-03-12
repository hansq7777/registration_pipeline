# GT-Backed Evaluation Protocol v1

Date: 2026-03-11

## Canonical GT Sources

All future bbox and mask experiments should use only the following GT roots unless explicitly expanded with the same metadata contract:

- `D:\Research\Image Analysis\Nanozoomer scans\20250327 rat myelin quantification\Tissue&Masks\test`
- `D:\Research\Image Analysis\Nanozoomer scans\20250424 Nissl cytoarchitectonic counterpart\Tissue&Masks\test`

These folders are preferred because each section directory carries:

- `crop_raw.png`
- `tissue_mask_final.png`
- `artifact_mask_final.png`
- `usable_tissue_mask.png`
- `foreground_rgba.png`
- `metadata.json`

And `metadata.json` contains enough information to map crop and mask pixels back to original slide coordinates:

- `source_slide`
- `crop_bbox_level0`
- `crop_bbox_overview`
- `canvas_to_slide_level0`
- `proposal_context`

Do not use older extracted crops that lack this spatial metadata for benchmark claims.

## Evaluation Rules

### 1. BBox proposal validation

The validation target is not visual similarity to older extracted crops.
The target is: generated bbox must cover the original slide-space region occupied by the GT mask.

Procedure:

1. Read GT `crop_bbox_level0` from section metadata.
2. Project generated proposal bbox into the same slide level-0 coordinate system.
3. Intersect the generated bbox with the GT crop window.
4. Project that overlap back into the GT crop canvas.
5. Measure how much of the GT tissue mask is covered.

Primary metrics:

- `mask_coverage_recall`
- `full_coverage_rate_99`
- `coverage_rate_95`

Secondary metrics:

- `crop_area_to_gt_mask_ratio`
- `crop_efficiency`
- `proposal_rect_vs_gtbbox_iou`

Interpretation:

- Missing GT mask area is a failure.
- Larger crop area is acceptable if it materially improves GT coverage.

### 2. Mask method validation

Mask benchmarking must start from crops that already have adequate GT coverage.
Do not tune mask logic on obviously truncated crops.

Primary metrics:

- `Dice`
- `IoU`
- `Precision`
- `Recall`

Boundary-aware metrics:

- `boundary_f1_tol32`
- `boundary_f1_tol64`
- `ASSD`
- `HD95`

Interpretation:

- Region overlap alone is insufficient.
- Boundary metrics must be checked because downstream use includes foreground extraction and registration.

## Iteration Policy

Do not wait for a whole slide batch before checking quality.

Preferred loop:

1. Parse one slide filename and expected section count.
2. Generate proposal boxes only for that slide.
3. Compare proposal boxes against GT mask location in slide coordinates.
4. If proposal quality is acceptable, run mask evaluation on that same slide.
5. Keep the new strategy only if it improves metrics or clearly resolves observed failure modes.
6. Then move to the next slide.

This is intentionally single-slide-first to reduce wasted time.

## Current Results On GT `2507`

### BBox benchmark

Run directory:

- `C:\Users\Siqi\Desktop\REVIEW\20260311_bbox_test_gt_2507_win`

Best current bbox strategy on `gallyas_2507_42-72.ndpi`:

- `projection_full_topfloor35`

Key numbers:

- `mean_mask_coverage_recall = 0.9673`
- `full_coverage_rate_99 = 0.3333`
- `coverage_rate_95 = 0.6667`
- `mean_crop_area_to_gt_mask_ratio = 1.8349`

Baseline comparison:

- `baseline_uniform8 mean_mask_coverage_recall = 0.4374`

Conclusion:

- The old `uniform8` crop is far too tight on this GT set.
- Coverage-first bbox should remain the default direction even if crop area increases.

### Fast mask tuning on GT-covered crops

Run directory:

- `C:\Users\Siqi\Desktop\REVIEW\20260311_myelin_mask_gtcrop_2507_scale05`

This was a fast crop-only benchmark at `0.5x` working scale, used to isolate mask behavior after removing the bbox truncation problem.

Top methods:

1. `simple_conservative`
   - `mean_dice = 0.8782`
   - `mean_iou = 0.8000`
   - `mean_precision = 0.8062`
   - `mean_recall = 0.9918`

2. `legacy_simple`
   - `mean_dice = 0.8709`
   - `mean_iou = 0.7891`
   - `mean_precision = 0.7923`
   - `mean_recall = 0.9961`

3. `crop_center_loose2comp`
   - lower Dice, but stronger boundary metrics than legacy/simple

Conclusion:

- Once bbox coverage is no longer badly truncated, `simple_conservative` is a better crop-only baseline than `legacy_simple`.
- Stronger crop-center contextual variants become too conservative on GT crops unless supported by better slide-level context.

## Operational Notes

- Use Windows-native Python when benchmarking NDPI readers and proposal generation. WSL access to `D:\...` can significantly distort perceived IO cost.
- Intermediate experiment files may be written to `C:\Users\Siqi\Desktop\REVIEW`.
- Periodically clear stale or superseded result folders from `REVIEW` to preserve space.

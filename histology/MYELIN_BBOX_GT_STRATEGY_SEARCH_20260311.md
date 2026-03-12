# Myelin BBox GT Strategy Search 2026-03-11

Scope:

- stain: `gallyas`
- GT source: `D:\Research\Image Analysis\Nanozoomer scans\20250327 rat myelin quantification\Tissue&Masks\test`
- evaluation space: slide `level0`
- GT sections evaluated: `18`
  - `2503_113`, `2503_114`, `2503_119`, `2503_120`, `2503_126`, `2503_132`, `2503_138`, `2503_144`
  - `2504_5`, `2504_47`, `2504_161`, `2504_185`
  - `2507_42`, `2507_48`, `2507_54`, `2507_60`, `2507_66`, `2507_72`

Evaluation rule:

- compare all geometry in the same slide `level0` space
- auto proposal bbox:
  - generated on the overview
  - mapped to slide `level0`
- GT hand crop bbox:
  - read from each section folder `metadata.json -> crop_bbox_level0`
- GT tissue mask:
  - interpreted inside the GT crop canvas
  - projected back to the same slide `level0` crop footprint
- primary objective:
  - maximize `mask_coverage_recall`
  - the auto bbox should cover the GT mask footprint in slide space
- secondary objective:
  - reduce wasted area and neighbor intrusion
  - explicitly penalize overlap between the auto bbox and other GT masks on the same slide

Primary metrics:

- `mean_mask_coverage_recall`
- `full_coverage_rate_99`
- `coverage_rate_95`

Penalty / compactness metrics:

- `mean_neighbor_overlap_ratio_proposal`
- `mean_proposal_rect_vs_gtcrop_iou_level0`
- `mean_proposal_area_to_gt_crop_area_full`

Result directories:

- per-sample targeted search:
  - `C:\Users\Siqi\Desktop\REVIEW\20260311_bbox_seedbox_gt_myelin_2503`
  - `C:\Users\Siqi\Desktop\REVIEW\20260311_bbox_seedbox_gt_myelin_2504`
  - `C:\Users\Siqi\Desktop\REVIEW\20260311_bbox_seedbox_gt_myelin_2507`
- expanded GT refine:
  - `C:\Users\Siqi\Desktop\REVIEW\20260311_bbox_seedbox_gt_myelin_midrefine18`
- expanded GT control comparison:
  - `C:\Users\Siqi\Desktop\REVIEW\20260311_bbox_seedbox_gt_myelin_control18`

## Main finding

The old default `baseline_uniform8` is still clearly unusable on the expanded GT set.

- `mean_mask_coverage_recall = 0.7401`
- `full_coverage_rate_99 = 0.3333`
- `mean_proposal_rect_vs_gtcrop_iou_level0 = 0.4390`

The earlier relaxed crop strategy `hybrid_topfloor55_wide24` is much better than the old default, but it still misses too much tissue on several hard cases.

- `mean_mask_coverage_recall = 0.9944`
- `full_coverage_rate_99 = 0.8889`
- worst hard case:
  - `2503_144 coverage = 0.9110`

The best expanded-GT result is now the seed-focused family, with `seed_relaxed74_t026` at the top:

- `mean_mask_coverage_recall = 0.9981`
- `full_coverage_rate_99 = 0.9444`
- `coverage_rate_95 = 1.0000`
- `mean_neighbor_overlap_ratio_proposal = 0.1102`
- `mean_proposal_rect_vs_gtcrop_iou_level0 = 0.8640`
- `mean_proposal_area_to_gt_crop_area_full = 1.1153`

This means the best current direction is:

- make the seed box itself cover the section footprint
- keep later crop expansion minimal
- accept a moderate area premium to avoid truncating tissue

## Why the evaluation changed

Earlier experiments only used the `2507` GT slide. The expanded GT set exposed two failure modes that were underrepresented before:

- mixed-sample slides with nearby tissue from another sample mounted on the same physical slide
- hard lower-edge under-coverage on `2503_144`

Because of that, the benchmark now adds a same-slide neighbor penalty:

- restore each GT mask to the original slide `level0`
- compare the auto bbox not only against the target GT mask, but also against all other GT masks on that same slide
- penalize bboxes that unnecessarily swallow nearby sections

## Common failure modes

### 1. Lower / middle under-coverage on `2503_144`

This is still the hardest miss case.

For the current top candidates:

- `seed_relaxed74_t026`
  - coverage: `0.9661`
  - top / middle / bottom recall: `0.9996 / 0.9640 / 0.9358`
- `seed_relaxed72_t027`
  - coverage: `0.9627`
  - top / middle / bottom recall: `0.9988 / 0.9581 / 0.9334`
- `seed_relaxed65_t030`
  - coverage: `0.9488`
  - top / middle / bottom recall: `0.9927 / 0.9375 / 0.9207`

Interpretation:

- after the earlier top-bias fixes, the dominant miss is no longer dorsal-only
- on this case, the proposal support itself is still too conservative around the lower/middle footprint
- tightening the seed box further directly worsens this case

### 2. Neighbor intrusion on dense or mixed slides

The main penalty-heavy cases are:

- `2504_47`
- `2504_161`
- `2504_5`
- `2503_138`
- `2503_132`
- `2503_144`

These are the sections where the auto bbox overlaps non-target GT masks the most in slide space.

Interpretation:

- once the seed is relaxed enough to guarantee coverage, the next limiting factor is not target recall but neighborhood separation
- this is especially visible on mixed-sample or tightly mounted slides

## What information was not guiding bbox well

The experiments show that the missing guidance is not mainly stain identity or local grayscale thresholding.

The weak signals were:

- incomplete support from shallow or weak tissue regions
- insufficient geometric allowance around the detected support
- no explicit awareness of nearby mounted sections during earlier searches

After adding same-slide neighbor penalties, the best strategy remained a seed-based relaxed box, but the optimum shifted slightly tighter than the previous runtime setting.

## Parameter tuning conclusions

The parameters that mattered most were:

- top relaxation in the seed box
- side relaxation in the seed box
- projection threshold scale

The refined seed family behaved as follows:

- `seed_relaxed65_t030`
  - smallest area
  - least neighbor intrusion
  - too much loss on `2503_144`
- `seed_relaxed70_t028`
  - improved compactness
  - still slightly below the best coverage
- `seed_relaxed72_t027`
  - good compromise
  - slightly tighter than the best method
- `seed_relaxed74_t026`
  - best overall ranking
  - same full-coverage rate as `72/27`
  - slightly better hard-case recall
- `seed_relaxed75_t026`
  - nearly identical to `74/26`
  - slightly larger and slightly worse neighbor penalty

This is why the new optimum moved from `75/32` to `74/32`.

## Recommended candidates

### 1. `seed_relaxed74_t026`

Recommended default.

Why:

- best overall GT ranking on the expanded 18-section benchmark
- highest hard-case coverage among the top compact candidates
- same `full_coverage_rate_99` as the tighter variants
- slightly better than `seed_relaxed75_t026` on neighbor penalty and area

### 2. `seed_relaxed72_t027`

Tighter alternative.

Why:

- almost identical coverage to the default
- lower neighbor-overlap penalty
- slightly smaller area

Tradeoff:

- a bit worse on the hardest miss case `2503_144`

### 3. `seed_relaxed65_t030`

Compact fallback.

Why:

- best compactness and least neighbor intrusion in the refined seed family

Tradeoff:

- coverage degrades too much on the hard case, so it should not be the default

## Control comparison against previous baselines

On the full 18-section GT set:

- `seed_relaxed74_t026`
  - coverage: `0.9981`
  - neighbor overlap: `0.1102`
  - proposal / GT-crop IoU: `0.8640`
- `hybrid_topfloor55_wide24`
  - coverage: `0.9944`
  - neighbor overlap: `0.0736`
  - proposal / GT-crop IoU: `0.7904`
- `baseline_uniform8`
  - coverage: `0.7401`
  - neighbor overlap: `0.0009`
  - proposal / GT-crop IoU: `0.4390`

Interpretation:

- the old default was compact because it was missing tissue
- the hybrid wide strategy was safer than the old default, but still under-covered hard cases
- the new seed strategy is the first one that is both reliable enough on coverage and still acceptably compact

## Runtime update

Runtime `gallyas` seed-box generation has been updated to the new best default:

- seed box relaxation:
  - `left = 0.32`
  - `top = 0.74`
  - `right = 0.32`
  - `bottom = 0.28`
- projection:
  - `top_cap = 0.57`
  - `bottom_cap = 0.26`
  - `side_cap = 0.22`
  - `thresh_scale = 0.26`
  - `max_gap = 7`

This is now the working default in:

- `tools/run_ndpi_review_experiment.py`

GUI cache / export metadata versioning has also been moved to:

- `gallyas_seedcrop_relaxed74_wide32_t026_v3`

## Practical takeaway

For myelin, bbox proposal should continue to be optimized in slide-space, not crop-space.

The current best policy is:

- optimize seed-box coverage first
- penalize same-slide neighbor swallowing
- keep downstream crop padding light

This should remain the working default until a new method can:

- beat `seed_relaxed74_t026` on `2503_144`
- without increasing neighbor intrusion on the mixed / dense slides

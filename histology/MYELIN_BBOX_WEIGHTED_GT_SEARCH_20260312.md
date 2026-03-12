# Myelin BBox Weighted GT Search 2026-03-12

Goal:

- optimize Gallyas/myelin section proposal boxes directly in slide `level0` space
- prioritize:
  - `50%` covering the target section GT mask
  - `30%` avoiding overlap with non-target GT masks from the same slide
  - `20%` compactness / avoiding over-expansion

GT basis:

- `D:\Research\Image Analysis\Nanozoomer scans\20250327 rat myelin quantification\Tissue&Masks\test`
- expanded set: `30` sections with per-section `metadata.json`

Evaluation protocol:

- project each auto proposal rectangle into slide `level0`
- project the GT crop and GT mask into the same slide `level0`
- compute:
  - `mask_coverage_recall`
  - `neighbor_overlap_ratio_proposal`
  - `proposal_area_to_gt_crop_area_full`
- rank methods by:
  - `0.50 * coverage`
  - `0.30 * (1 - neighbor overlap)`
  - `0.20 * compactness`

Result:

## 1. `hybrid_topfloor55_wide24`

- `mean_weighted_priority_score = 0.9596`
- `mean_mask_coverage_recall = 0.9936`
- `full_coverage_rate_99 = 0.8333`
- `mean_neighbor_overlap_ratio_proposal = 0.1072`
- `mean_proposal_area_to_gt_crop_area_full = 0.8841`

Why it wins:

- it gives up a small amount of target coverage relative to the most relaxed seed-only variants
- in return it substantially reduces overlap with neighboring sections and reduces box size
- under the explicit `50/30/20` priority, this tradeoff is the best overall

Main failure mode:

- the hardest remaining case is `2503_144`
- the miss is no longer mainly dorsal/top
- it is concentrated in the lower/middle extent, so simply adding more top expansion is not the right fix

## 2. `seed_relaxed65_t030`

- `mean_weighted_priority_score = 0.9526`
- `mean_mask_coverage_recall = 0.9968`
- `full_coverage_rate_99 = 0.9000`
- `mean_neighbor_overlap_ratio_proposal = 0.1287`
- `mean_proposal_area_to_gt_crop_area_full = 0.9698`

Why keep it:

- this is the safer high-coverage fallback
- if the practical rule is “missing target tissue is worse than including a bit more neighbor”, this is the better alternative

## 3. `seed_relaxed70_side29_t028`

- `mean_weighted_priority_score = 0.9498`
- `mean_mask_coverage_recall = 0.9967`
- `full_coverage_rate_99 = 0.9000`
- `mean_neighbor_overlap_ratio_proposal = 0.1352`
- `mean_proposal_area_to_gt_crop_area_full = 0.9935`

Why keep it:

- middle compromise between the compact hybrid winner and the more coverage-heavy relaxed seed family

What did not remain best:

- `baseline_uniform8`
  - too many target masks are clipped
  - `mean_mask_coverage_recall = 0.8049`
- `seed_relaxed74_t026`
  - still covers target tissue well
  - but expanded GT shows it is now too large and overlaps neighbors too often for the weighted objective

Runtime choice:

- GUI and experiment runtime now use `hybrid_topfloor55_wide24` as the default Gallyas bbox strategy
- if later work shows that missing hard-case lower/middle tissue is still too costly, the first fallback to test should be `seed_relaxed65_t030`

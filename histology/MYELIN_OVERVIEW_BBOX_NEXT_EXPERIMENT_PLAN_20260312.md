# Myelin Overview BBox Next Experiment Plan 2026-03-12

Purpose:

- turn the bbox troubleshooting analysis into a short, executable experiment plan
- improve Gallyas / myelin whole-slide bbox proposal
- optimize in this priority order:
  - `50%` cover the target section GT mask
  - `30%` avoid covering non-target GT masks on the same slide
  - `20%` stay compact and avoid unnecessary expansion

Reference inputs:

- GT source:
  - `D:\Research\Image Analysis\Nanozoomer scans\20250327 rat myelin quantification\Tissue&Masks\test`
- canonical troubleshooting note:
  - [bbox_proposal_troubleshooting.md](/mnt/c/work/registration_pipeline/histology/bbox_proposal_troubleshooting.md)
- current weighted baseline:
  - [MYELIN_BBOX_WEIGHTED_GT_SEARCH_20260312.md](/mnt/c/work/registration_pipeline/histology/MYELIN_BBOX_WEIGHTED_GT_SEARCH_20260312.md)

Current baseline to beat:

- `hybrid_topfloor55_wide24`
- strengths:
  - best weighted score under the explicit `50/30/20` objective
  - relatively low neighbor overlap
  - relatively compact boxes
- weaknesses:
  - still clips hard cases such as `2503_144`
  - lower/middle footprint remains the main miss pattern

## Problem framing

This is not a simple tissue-detection problem. It is:

- multi-section slide
- instance-aware bbox proposal
- weak target fringe vs strong neighbor core discrimination

Current proposal still answers:

- `where is tissue-like signal?`

but not strongly enough:

- `which signal belongs to this target section, not the neighbor?`

## Main failure modes to target

1. `target truncation`
- shallow fringe or lower/middle footprint is not fully covered
- often caused by seed box being too centered on dark core tissue

2. `neighbor contamination`
- box grows into adjacent section because expansion sees tissue-like signal but not ownership

3. `seed mis-centering`
- candidate component is biased toward a deep-stained island
- later expansion compensates in the wrong geometry

## Hard rules for future evaluation

All bbox evaluation must stay in slide `level0` space.

For each GT section:

- project proposal bbox to slide `level0`
- project GT crop and GT mask to the same `level0`
- compute:
  - target mask coverage recall
  - neighbor overlap ratio against other GT masks from the same slide
  - proposal area relative to GT crop area

Primary ranking score:

- `0.50 * target_coverage`
- `0.30 * (1 - neighbor_overlap)`
- `0.20 * compactness`

Do not use mean score alone.
Always also inspect:

- worst-section target coverage
- `full_coverage_rate_99`
- hard-case list sorted by lowest target coverage

## Experiment sequence

### Experiment 1. Strong-core / weak-fringe split

Goal:

- stop using one merged score for both seed definition and outer growth

Change:

- build `strong_core_map`
  - higher-threshold residual
  - optional non-background gate
- build `weak_fringe_map`
  - lower-threshold residual
  - legacy darkness / nonwhite cue
  - optional entropy or gradient cue, but only as fringe support

Inference:

- seed must come from `strong_core_map`
- growth is allowed only inside `weak_fringe_map`

Success criterion:

- improves worst-case target coverage without strongly increasing neighbor overlap

Stop condition:

- if mean weighted score does not beat current baseline and hard-case coverage does not improve

### Experiment 2. Multi-seed competitive assignment

Goal:

- explicitly model section ownership on the same slide

Change:

- detect all strong cores on the slide first
- build a broad weak support mask
- assign weak support to cores using one of:
  - marker-controlled watershed
  - geodesic reconstruction from each seed
  - nearest-seed Voronoi partition as a minimal baseline

Inference:

- generate per-section pseudo-mask first
- derive bbox from that pseudo-mask

Why this matters:

- current pipeline has no real rival penalty during inference
- this is the cleanest way to reduce neighbor swallowing

Success criterion:

- lowers neighbor overlap on close-mount slides while preserving target coverage

Stop condition:

- if target coverage drops more than the neighbor-overlap gain justifies

### Experiment 3. Edge-wise objective optimization

Goal:

- move from rule-based expansion to objective-driven expansion

Change:

- start from a seed or pseudo-mask tight box
- optimize left/right/top/bottom independently
- at each step, only expand a side if:
  - target gain is positive enough
  - rival gain is small enough
  - area penalty is acceptable

Approximate objective per side:

- `delta_score = 0.50 * target_gain - 0.30 * rival_gain - 0.20 * excess_area_gain`

Success criterion:

- improves compactness and neighbor control without introducing new hard misses

Stop condition:

- if this only reproduces current best hybrid behavior with no hard-case improvement

### Experiment 4. Boundary-strip refinement at one higher resolution

Goal:

- use slightly richer information only near the candidate box border

Change:

- keep overview for global proposal
- read narrow strips around the four box boundaries at one higher pyramid level
- use the strips only to refine each side in a small search window

Why this is later, not first:

- more expensive
- should only be used if Experiments 1 to 3 cannot fix the lower/middle footprint misses

Success criterion:

- specifically rescues hard cases such as `2503_144`

Stop condition:

- if runtime cost is high and weighted score gain is marginal

## Minimal experiment matrix

Run in this order, not all at once:

1. `E1-A`
- strong core = high-threshold residual
- fringe = low-threshold residual

2. `E1-B`
- strong core = high-threshold residual
- fringe = max(low-residual, legacy, nonwhite)

3. `E2-A`
- watershed ownership

4. `E2-B`
- geodesic reconstruction ownership

5. `E3-A`
- side-wise objective optimization on current best seed

6. `E3-B`
- side-wise objective optimization on `E2` pseudo-mask box

Only if needed:

7. `E4-A`
- strip refinement on the lowest-coverage 5 hard cases

## Metrics to report every round

Global:

- mean weighted priority score
- mean target coverage recall
- `full_coverage_rate_99`
- mean neighbor overlap ratio
- mean proposal area / GT crop area

Tail / hard-case:

- min target coverage recall
- 5 worst sections by target coverage
- 5 worst sections by neighbor overlap

Segmented diagnostics:

- top / middle / bottom coverage recall
- left / center / right coverage recall

Use these diagnostics to answer:

- did the method fix dorsal/top-only issues?
- did it fix lower/middle footprint?
- did it reduce neighbor contamination or only enlarge the box?

## Decision rules

Promote a new default only if all are true:

- weighted score is better than current baseline
- worst-section coverage is not worse unless neighbor-overlap gain is substantial
- no new obvious catastrophic truncation cases appear

If a candidate is better for neighbor control but worse for hard-case coverage:

- keep it as a conservative alternative, not the default

## Expected outcomes

Most likely best next win:

- `multi-seed competitive assignment`

Most likely easiest quick win:

- `strong-core / weak-fringe split`

Most likely rescue for remaining hard tail:

- `boundary-strip refinement`

## Deliverables per round

For each experiment round, save:

- `proposal_summary.md`
- `proposal_summary.json`
- `proposal_metrics.csv`
- one short interpretation note:
  - what improved
  - what got worse
  - whether to continue or stop this branch

Intermediate outputs can go to `C:\Users\Siqi\Desktop\REVIEW`, but old failed branches should be cleaned regularly.

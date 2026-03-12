# NDPI Pipeline Efficiency Playbook v1

## Purpose

This note records the current best-practice workflow for reducing repeated NDPI reads,
duplicate crop extraction, and unnecessary high-resolution mask computation.

It is meant to be updated whenever a better efficiency strategy is validated.

## Current Bottlenecks

The current pipeline cost is dominated by these stages:

1. `load_ndpi`
   - opening a multi-GB NDPI
   - initializing reader state
   - worst on first open
2. `overview_proposal`
   - comparatively cheap
   - should remain slide-scope, not section-scope
3. `section_crop_cache_miss`
   - expensive after bbox expansion because crops are larger
4. `mask_generation`
   - current dominant per-section cost once crop coverage is corrected

## Canonical Efficient Workflow

1. Open one slide once per review session.
2. Build overview proposal once at slide scope.
3. Reuse a persistent slide handle for crop extraction whenever possible.
4. Route repeated preview and section crop requests through `SlideSessionCache`.
5. Use low-resolution preview crops for:
   - box proposal review
   - rapid method screening
   - UI interaction checks
6. Use higher-resolution section crops only for:
   - final mask generation
   - GT-backed validation
   - export-quality outputs
7. Avoid writing nonessential derived outputs during routine export.
8. Export only missing or changed section folders.

## What To Avoid

- Do not benchmark GUI NDPI behavior from WSL and treat it as Windows GUI truth.
- Do not reopen NDPI once per section when the session can hold one slide handle.
- Do not run heavy qualitative overlay export during every tuning run.
- Do not use full-resolution crops for early method screening by default.
- Do not recompute white/black background variants at export time if they are derivable later.

## Resolution Strategy

Use a staged policy:

- `overview / proposal`
  - use overview level only
- `preview crop`
  - use lower crop level such as level 5 when available
- `mask tuning`
  - start with lower crop level on a small GT subset
- `final mask validation`
  - raise to level 3 or better only for shortlisted methods
- `export`
  - write only canonical raw crop, masks, `foreground_rgba`, and metadata by default
  - map reviewed working-resolution masks back onto export-resolution crop at write time

## Cache Strategy

Cache at these layers:

- current slide handle
- overview thumbnail
- proposal preview crops
- section crops for current review pass
- tool module import singleton
- persistent backend hint cache
- persistent `tifffile_proxy` overview/label bundle cache
- persistent overview proposal cache
- persistent auto-mask result cache keyed by:
  - slide identity
  - stain
  - mask preset / algorithm version
  - bbox algorithm version
  - crop level
  - target proposal bbox
  - all-proposal snapshot

Future cache targets:

- auto-mask result cache keyed by:
  - `slide_path`
  - `bbox_level0`
  - `crop_level`
- `mask_algorithm_version`
- `mirror_enabled`

## Backend Strategy

Current runtime policy:

- first successful backend is persisted as a slide-level backend hint
- later opens of the same unchanged slide can skip doomed `OpenSlide` attempts
- fallback slides also persist a lightweight proxy bundle:
  - `overview_proxy.png`
  - `label_proxy.png`
  - reader metadata needed to reconstruct the same overview geometry

This turns the earlier reference-list idea into an actual runtime optimization.

Observed backend hint counts on the current known datasets:

- Nissl root:
  - `openslide`: 41
  - `tifffile_proxy`: 2
- Gallyas root:
  - `openslide`: 25
  - `tifffile_proxy`: 40

So the backend reference list is worth keeping.
It is especially useful for the Gallyas dataset, where fallback is common.

Validated timing gain on a representative fallback Gallyas slide (`gallyas_2501_114-144.ndpi`):

- cold `load_ndpi`: about `1.97s`
- cached `load_ndpi` after backend hint + proxy bundle: about `0.42s`

So backend hint + proxy bundle is now a real runtime optimization, not only an operator convenience.

## Default Export Strategy

Default export should contain only:

- `crop_raw.png`
- `tissue_mask_final.png`
- `artifact_mask_final.png`
- `usable_tissue_mask.png`
- `foreground_rgba.png`
- `metadata.json`

Derivable later:

- `foreground_rgb_white`
- `foreground_rgb_black`

## Experiment Template

For any new optimization, measure separately:

- `cold_start.load_ndpi`
- `cold_start.overview_proposal`
- `preview_crop.cache_miss`
- `preview_crop.cache_hit`
- `section_crop.cache_miss`
- `section_crop.cache_hit`
- `mask_generation`

Also record:

- crop coverage recall
- Dice / IoU
- contour metrics

## Current High-Value Optimization Directions

1. Keep the coverage-first bbox strategy.
   - correctness gain is larger than the crop-size cost
2. Keep persistent backend hint + proxy bundle enabled for fallback slides.
3. Reuse persistent overview proposal cache for unchanged slides.
4. Optimize mask generation before touching bbox again.
5. Add auto-mask result cache for unchanged sections.
6. Keep metadata rich enough to compare:
   - new bbox vs old GT location
   - new mask vs old GT location

## New Validated Timing Results

Small Windows-side timing on representative slides after the latest cache changes:

- fallback Gallyas `gallyas_2501_114-144.ndpi`
  - cold `load_ndpi`: about `1.97s`
  - cached `load_ndpi`: about `0.42s`
  - cold `overview_proposal`: about `1.02s`
  - cached `overview_proposal`: about `0.005s`
  - crop read remains the main cost:
    - reopen per request: about `7.94s`
    - persistent handle: about `7.79s`

- OpenSlide Gallyas `gallyas_2502_42-72.ndpi`
  - cold `load_ndpi`: about `7.11s`
  - cached `load_ndpi`: about `6.95s`
  - cold `overview_proposal`: about `1.30s`
  - cached `overview_proposal`: about `0.007s`
  - crop read:
    - reopen per request: about `1.90s`
    - persistent handle: about `1.79s`

Interpretation:

- for fallback slides, the biggest validated gain now comes from:
  - backend hint cache
  - persistent proxy bundle
  - persistent proposal cache
- auto-mask cache is also now worth using for repeated review on unchanged sections
- for OpenSlide slides, `load_ndpi` itself is still dominated by the library open cost
- after the new caches, the next dominant hotspot is section crop extraction, not slide open or proposal generation

## Auto-Mask Cache Result

A direct Windows-side timing on `gallyas_2502_42-72.ndpi`, section `2502_48`, `crop_level=4`, preset `hybrid_balanced` showed:

- uncached mask compute:
  - about `27.16s`
- cache write:
  - about `0.27s`
- memory cache hit:
  - about `0.010s`
- disk cache hit:
  - about `0.057s`

Practical takeaway:

- when bbox, crop level, preset, and neighboring proposal layout are unchanged, repeated `Run/Refresh Auto Mask` should now avoid full recomputation
- cache lookup overhead is negligible compared with actual mask generation

## Export Path Notes

An export regression was corrected:

- `ExportWorker` now opens a persistent slide handle for both:
  - `openslide`
  - `tifffile_proxy`

Previously it only persisted handles for `openslide`, which could force fallback export back into repeated per-section reopen behavior.

## Preview-Scale Mask Operations

A small mask-only fidelity test on 5 existing Gallyas `crop + final mask` items showed:

- pure mask downsample -> upsample fidelity:
  - `0.75x`: mean Dice about `0.99956`
  - `0.5x`: mean Dice about `0.99969`
  - `0.33x`: mean Dice about `0.99889`
- `shrink/expand` performed on downsampled mask and projected back to full-res:
  - `0.75x`: mean Dice about `0.99948 - 0.99950`
  - `0.5x`: mean Dice about `0.99960`
  - `0.33x`: mean Dice about `0.99886`

Practical takeaway:

- preview-scale morphology is safe enough for GUI-wide `shrink/expand`
- `0.5x` is the most attractive current working point
- `0.33x` still looks usable, but should be treated as a more aggressive speed mode
- this result applies to mask operations, not yet to full auto-mask generation from raw crop

Current GUI policy after this update:

- Step 1 proposal preview crops:
  - default to `level 5` when available
  - proposal preview cards are loaded lazily, not all at slide-open time
- Step 2 mask working crops:
  - default to `level 4`
- export:
  - use `level 0`
  - resize working masks to export resolution with nearest-neighbor mapping

## Update Policy

When a new optimization is tested:

1. write raw results to `REVIEW`
2. summarize deltas against the previous baseline
3. update this playbook with:
   - what changed
   - what got faster
   - what accuracy tradeoff was observed

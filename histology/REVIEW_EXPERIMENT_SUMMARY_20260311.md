# Review Experiment Summary

Date: 2026-03-11

This note condenses the main findings from the March 5 to March 11 `Desktop/REVIEW` experiment runs and supersedes the scattered top-level review notes that were previously kept outside the project tree.

## Whole-Slide Baseline

- The fixed whole-slide review baseline for Nissl is still `baseline_v1`.
- Whole-slide processing is now structured as:
  - filename parse
  - overview proposal
  - crop export
  - crop-level mask generation
  - human correction in GUI
  - traceable export with slide-coordinate metadata
- `soft_support_mgac` remains a failed prototype and should not be used as a default path.

## Gallyas / Myelin Status

- The old GUI crop-only Gallyas mask was too permissive and frequently expanded to almost the full crop.
- The current best automatic path is the contextual Gallyas branch:
  - proposal-aware crop ownership
  - support-constrained masking
  - residual-based myelin signal
  - conservative refine
- On the GT-backed `2502_78/84/90/96/102/108` subset:
  - old GUI baseline Dice/IoU: `0.869 / 0.771`
  - contextual GUI / experiment baseline Dice/IoU: `0.943 / 0.893`
- On hard qualitative cases such as `2503_102` and `2503_108`, contextual masking removed catastrophic full-crop over-expansion.

## BBox Proposal Status

- The old uniform `8%` padding proposal was too conservative for Gallyas.
- The best GT-coverage proposal strategy tested was `projection_full_topfloor20`.
- On the GT-backed `2502` subset:
  - old `uniform8` mean GT-mask coverage recall: `0.819`
  - `projection_full_topfloor20` mean GT-mask coverage recall: `0.902`
- The current GUI crop path was updated to a coverage-first bbox strategy so that crop clipping no longer dominates downstream mask failure.

## Resolution And GUI Working Policy

- Interactive GUI work should stay low-resolution:
  - Step 1 preview: very low-cost overview / preview crop
  - Step 2 mask work: lower pyramid level crop
  - export: lower-resolution working export with full slide-coordinate metadata
- The current recommended preview-scale morphology working point is around `0.5x`.
- Probe results showed no meaningful mask-fidelity loss at `0.5x`, while much lower scales become progressively more approximate.

## Export And Performance

- The most expensive export configuration was:
  - full-resolution `level 0` crop export
  - large coverage-first bbox
  - fallback slide reopening per section
- Export was corrected to:
  - default back to `level 3`
  - keep full coordinate traceability in metadata
  - reuse one `TiffFile + zarr` handle across the whole export session for `tifffile_proxy`
- This keeps future remapping possible without paying the full-resolution export cost by default.

## Current GUI Mask Presets

- `Latest Contextual`
  - current default for serious myelin work
  - uses stain-aware branching and proposal context
- `Legacy Simple`
  - explicit fallback for side-by-side testing
  - keeps the original simplified GUI crop-only logic

## Stain Routing

- GUI stain routing is filename-driven, not content-classification-driven.
- `nissl_*.ndpi` is routed to the Nissl branch.
- `gallyas_*.ndpi` is routed to the Gallyas / myelin branch.
- If a slide is misnamed, the wrong strategy will be selected.

## Retained Review Artifacts

Only two small machine-readable result folders are still intentionally retained under `Desktop/REVIEW` after cleanup:

- `20260310_myelin_gui_improved_gt_2502_78-108`
- `20260311_mask_resolution_fidelity_probe`

These were kept because they are small and still useful as compact numeric references for:

- GT-backed myelin mask quality
- resolution / preview-fidelity comparisons

All larger intermediate, duplicated, smoke-test, timing-test, and superseded review outputs were removed to reclaim disk space.

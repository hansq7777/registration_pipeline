# Current Histology Status And Decisions 2026-03-17

This note compresses the one-off experiment Markdown files under
`C:\work\registration_pipeline\histology` into a smaller active set.

It is the current "what is deployed / what is still experimental" summary for
the histology-side pipeline.

## Current Runtime Defaults

### Gallyas / myelin

- bbox proposal:
  - `gallyas_bbox_dr_localadaptive_compete_v2_sidepad`
  - ownership-aware competitive proposal with local-adaptive fringe support
  - includes small horizontal side-pad to avoid crop-edge truncation
- default mask preset:
  - `m3_hyst_entres_guard_v1`
- Step 2 mask compute profile:
  - default `Standard 2048px`
  - `Fast 1600px` remains optional
  - `Full` remains fallback only, not recommended for batch prediction

### Nissl

- bbox:
  - current default remains the stable GT-backed Nissl path
- mask:
  - current default remains the Nissl-specific tool baseline path

## Current Workflow State

- Step 1:
  - human review of bbox proposals
  - confirm accepted boxes
  - export section folders with crop + metadata
- Step 2:
  - prepare downsampled work images
  - batch mask prediction from prepared work images
  - prediction writes minimal outputs immediately per section
- Step 3:
  - review / adjust masks
  - save revisions inside each section folder
  - canonical saved mask files are now:
    - `mask_labels.png`
      - `0 = background`
      - `1 = tissue`
      - `2 = artifact`
    - `mask_preview.png`
      - RGB preview for folder thumbnails
      - tissue shown in red, artifact shown in cyan
    - `foreground_rgba.png`
      - crop RGB with transparent background outside tissue

## BBox Conclusions

### Myelin / Gallyas

- the old single-candidate expansion family was not the real long-term winner
- the decisive improvement came from ownership-aware competitive proposal
- later local-adaptive refinement improved coverage slightly and became the
  deployed default
- the current deployed choice is therefore:
  - `dr_localadaptive_compete_v2_sidepad`

### Nissl

- Nissl bbox was already close to saturated on GT
- no myelin-style aggressive proposal logic should be transferred into Nissl

## Mask Conclusions

### Myelin / Gallyas

- current deployed choice:
  - `m3_hyst_entres_guard_v1`
- why:
  - best balance of boundary adherence, leakage control, and collapse avoidance
- experiment status:
  - `M1 reconstruction-only`:
    - null result
  - `M2 entropy + residual joint candidate`:
    - useful direction, better than older hybrid baselines, but not better than
      the final M3 choice
  - `M4 multi-component set selection`:
    - did not beat M3
  - edge-fix / top-envelope / support-scoring after-the-fact corrections:
    - did not beat base M3 enough to replace it
  - weak-support `WS0`:
    - interesting but still unstable, not adopted

### Nissl

- Nissl mask path is stable and does not currently need the same level of
  rework as myelin

## Performance Conclusions

- Step 2 now uses explicit work-image preparation before mask prediction
- M3 was originally dominated by:
  - `binary_propagation`-like reconstruction behavior
  - then by `_retain_core_overlapping_components(...)`
- after optimization:
  - M3 `2503_150` work-image runtime dropped from minutes to about `3.2s`
  - no mask quality change on the benchmarked sample
- bbox proposal was originally dominated by:
  - `_competitive_support_bbox_runtime(...)`
  - specifically its ownership-assignment loop
- after batch ownership + slide-level score reuse:
  - `gallyas_2503_150-180.ndpi` uncached proposal time dropped from about `50s`
    to about `2.3s`
  - proposal boxes stayed identical on the benchmarked slide

## Active Documents To Keep

These are still worth keeping at the root as active references:

- [README.md](/mnt/c/work/registration_pipeline/histology/README.md)
- [WHOLESLIDE_TO_CYCLEGAN_AND_REGISTRATION_WORKFLOW_v1.md](/mnt/c/work/registration_pipeline/histology/WHOLESLIDE_TO_CYCLEGAN_AND_REGISTRATION_WORKFLOW_v1.md)
- [GT_BACKED_EVALUATION_PROTOCOL_v1.md](/mnt/c/work/registration_pipeline/histology/GT_BACKED_EVALUATION_PROTOCOL_v1.md)
- [HUMAN_IN_THE_LOOP_HISTOLOGY_GUI_DEV_SPEC_v1.md](/mnt/c/work/registration_pipeline/histology/HUMAN_IN_THE_LOOP_HISTOLOGY_GUI_DEV_SPEC_v1.md)
- [MYELIN_BBOX_WEIGHTED_GT_SEARCH_20260312.md](/mnt/c/work/registration_pipeline/histology/MYELIN_BBOX_WEIGHTED_GT_SEARCH_20260312.md)
- [MYELIN_MASK_GT_STRATEGY_SEARCH_20260311.md](/mnt/c/work/registration_pipeline/histology/MYELIN_MASK_GT_STRATEGY_SEARCH_20260311.md)
- [MYELIN_OVERVIEW_BBOX_NEXT_EXPERIMENT_PLAN_20260312.md](/mnt/c/work/registration_pipeline/histology/MYELIN_OVERVIEW_BBOX_NEXT_EXPERIMENT_PLAN_20260312.md)
- [MYELIN_MASK_NEXT_EXPERIMENT_PLAN_20260312.md](/mnt/c/work/registration_pipeline/histology/MYELIN_MASK_NEXT_EXPERIMENT_PLAN_20260312.md)
- [MYELIN_SUPPORT_WEAKSUP_BASELINE_PLAN_20260312.md](/mnt/c/work/registration_pipeline/histology/MYELIN_SUPPORT_WEAKSUP_BASELINE_PLAN_20260312.md)
- [NISSL_BBOX_GT_SEARCH_20260311.md](/mnt/c/work/registration_pipeline/histology/NISSL_BBOX_GT_SEARCH_20260311.md)
- [NISSL_MASK_GT_STRATEGY_SEARCH_20260311.md](/mnt/c/work/registration_pipeline/histology/NISSL_MASK_GT_STRATEGY_SEARCH_20260311.md)
- [NDPI_PIPELINE_EFFICIENCY_PLAYBOOK_v1.md](/mnt/c/work/registration_pipeline/histology/NDPI_PIPELINE_EFFICIENCY_PLAYBOOK_v1.md)
- [WINDOWS_TIMING_HARNESS_REFERENCE_v1.md](/mnt/c/work/registration_pipeline/histology/WINDOWS_TIMING_HARNESS_REFERENCE_v1.md)
- [bbox_proposal_troubleshooting.md](/mnt/c/work/registration_pipeline/histology/bbox_proposal_troubleshooting.md)
- [bboxdeep-research-report.md](/mnt/c/work/registration_pipeline/histology/bboxdeep-research-report.md)
- [Maskgen-deep-research-report.md](/mnt/c/work/registration_pipeline/histology/Maskgen-deep-research-report.md)

## Superseded One-Off Notes

The deleted one-off reports were historical experiment leaves whose conclusions
are now folded into this summary and the active protocol/plan documents.

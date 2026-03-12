# Histology Registration Pipeline

Scope:

- whole-slide NDPI review and candidate crop generation
- archived mask-extraction and section-level preprocessing notes

Contents:

- `tools/run_ndpi_review_experiment.py`
  - overview-based NDPI proposal workflow for candidate sections
  - supports stain-aware branches (`nissl`, `gallyas`)
  - supports optional `Nissl-guided proposal prior` for Gallyas runs via `--nissl-prior-root`
- `WHOLESLIDE_TO_CYCLEGAN_AND_REGISTRATION_WORKFLOW_v1.md`
  - end-to-end histology data-product design from NDPI to masks, cleaned crops, CycleGAN exports, and registration inputs
- `HUMAN_IN_THE_LOOP_HISTOLOGY_GUI_v1.md`
  - GUI design for human-guided proposal review, mask correction, orientation selection, pairing review, and export management
- `HUMAN_IN_THE_LOOP_HISTOLOGY_GUI_DEV_SPEC_v1.md`
  - implementation-oriented GUI specification: schema, modules, state machine, files, and export contracts
- `WINDOWS_TIMING_HARNESS_REFERENCE_v1.md`
  - reference for Windows-side GUI timing methodology, anti-patterns, and benchmark workflow
- `REVIEW_EXPERIMENT_SUMMARY_20260311.md`
  - condensed summary of the March whole-slide, myelin mask, bbox, low-resolution GUI, and export-efficiency experiments
- `GT_BACKED_EVALUATION_PROTOCOL_v1.md`
  - canonical GT sources, slide-space bbox evaluation rule, and current GT-backed bbox/mask benchmark conclusions
- `MYELIN_MASK_GT_STRATEGY_SEARCH_20260311.md`
  - GT-backed crop-mask strategy search for myelin sections, including boundary/leakage tradeoffs and current top 3 candidates
- `MYELIN_BBOX_GT_STRATEGY_SEARCH_20260311.md`
  - GT-backed myelin bbox proposal search comparing slide-space proposal rectangles against hand crop boxes and GT masks, with relaxed top-biased projection selected as the new best default
- `MYELIN_BBOX_WEIGHTED_GT_SEARCH_20260312.md`
  - weighted GT-backed myelin bbox search using `50%` target coverage, `30%` non-target avoidance, and `20%` compactness, with compact hybrid proposal selected as the new weighted-default candidate
- `NISSL_BBOX_GT_SEARCH_20260311.md`
  - GT-backed Nissl bbox search showing that full coverage is already saturated and that smaller uniform pads are the only meaningful optimization direction
- `NISSL_MASK_GT_STRATEGY_SEARCH_20260311.md`
  - GT-backed Nissl crop-mask strategy search comparing GUI and experiment-script paths, with the experiment baseline selected as the new best default
- `gui_mvp/`
  - SQLite schema, Python data models, directory conventions, and PySide6 GUI MVP skeleton
- `archive/2026-03-05_mask_extraction_v4/`
  - v4 fixed-policy mask extraction archive

Usage note:

- Treat this branch as the histology-side pre-registration layer.
- Output from here is expected to feed downstream section-level registration or MRI bridge steps, but the ANTs MRI registration stack lives under `../mri/`.
- Operational policy:
  - `baseline_v1` remains the fixed routine workflow.
  - `soft_support_mgac` remains experimental.
  - For Gallyas testing, prefer supplying a matching Nissl NDPI root as proposal prior when available.

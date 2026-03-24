# Histology Registration Pipeline

Scope:

- whole-slide NDPI review and candidate crop generation
- archived mask-extraction and section-level preprocessing notes

Contents:

- `CURRENT_STATUS_AND_DECISIONS_20260317.md`
  - compressed status note for deployed defaults, retained active docs, and
    one-off experiment conclusions
- `tools/run_ndpi_review_experiment.py`
  - overview-based NDPI proposal workflow for candidate sections
  - supports stain-aware branches (`nissl`, `gallyas`)
  - supports optional `Nissl-guided proposal prior` for Gallyas runs via `--nissl-prior-root`
- `WHOLESLIDE_TO_CYCLEGAN_AND_REGISTRATION_WORKFLOW_v1.md`
  - end-to-end histology data-product design from NDPI to masks, cleaned crops, CycleGAN exports, and registration inputs
- `HUMAN_IN_THE_LOOP_HISTOLOGY_GUI_DEV_SPEC_v1.md`
  - implementation-oriented GUI specification: schema, modules, state machine, files, and export contracts
- `WINDOWS_TIMING_HARNESS_REFERENCE_v1.md`
  - reference for Windows-side GUI timing methodology, anti-patterns, and benchmark workflow
- `GT_BACKED_EVALUATION_PROTOCOL_v1.md`
  - canonical GT sources, slide-space bbox evaluation rule, and current GT-backed bbox/mask benchmark conclusions
- `MYELIN_BBOX_WEIGHTED_GT_SEARCH_20260312.md`
  - current GT-backed myelin bbox decision reference
- `MYELIN_MASK_GT_STRATEGY_SEARCH_20260311.md`
  - early GT-backed myelin mask baseline reference; later one-off leaves are
    compressed into `CURRENT_STATUS_AND_DECISIONS_20260317.md`
- `NISSL_BBOX_GT_SEARCH_20260311.md`
  - GT-backed Nissl bbox search showing that full coverage is already saturated and that smaller uniform pads are the only meaningful optimization direction
- `NISSL_MASK_GT_STRATEGY_SEARCH_20260311.md`
  - GT-backed Nissl crop-mask strategy search comparing GUI and experiment-script paths, with the experiment baseline selected as the new best default
- `MYELIN_OVERVIEW_BBOX_NEXT_EXPERIMENT_PLAN_20260312.md`
  - forward bbox experiment plan and ownership-oriented troubleshooting path
- `MYELIN_MASK_NEXT_EXPERIMENT_PLAN_20260312.md`
  - forward myelin mask experiment plan based on deep-research review
- `MYELIN_SUPPORT_WEAKSUP_BASELINE_PLAN_20260312.md`
  - weak-supervision support-generation baseline plan; not deployed yet
- `gui_mvp/`
  - SQLite schema, Python data models, directory conventions, and PySide6 GUI MVP skeleton
  - current section-folder mask convention:
    - `mask_labels.png`
      - single-channel labels
      - `0 = background`
      - `1 = tissue`
      - `2 = artifact`
    - `mask_preview.png`
      - RGB thumbnail-friendly preview
      - tissue red, artifact cyan
    - `foreground_rgba.png`
      - RGB crop with alpha outside tissue
- `archive/2026-03-05_mask_extraction_v4/`
  - v4 fixed-policy mask extraction archive

Usage note:

- Treat this branch as the histology-side pre-registration layer.
- Output from here is expected to feed downstream section-level registration or MRI bridge steps, but the ANTs MRI registration stack lives under `../mri/`.
- Operational policy:
  - `baseline_v1` remains the fixed routine workflow.
  - `soft_support_mgac` remains experimental.
  - For Gallyas testing, prefer supplying a matching Nissl NDPI root as proposal prior when available.

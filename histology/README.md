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

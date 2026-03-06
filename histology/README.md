# Histology Registration Pipeline

Scope:

- whole-slide NDPI review and candidate crop generation
- archived mask-extraction and section-level preprocessing notes

Contents:

- `tools/run_ndpi_review_experiment.py`
  - overview-based NDPI proposal workflow for candidate sections
- `archive/2026-03-05_mask_extraction_v4/`
  - v4 fixed-policy mask extraction archive

Usage note:

- Treat this branch as the histology-side pre-registration layer.
- Output from here is expected to feed downstream section-level registration or MRI bridge steps, but the ANTs MRI registration stack lives under `../mri/`.

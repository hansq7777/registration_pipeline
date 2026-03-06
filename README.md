# Registration Pipelines

This repository now keeps two registration tracks:

- `histology/`: whole-slide and cropped-section registration support for histology workflows.
- `mri/`: ANTs-based MRI registration, benchmarking, QC, and transform-contract tooling.

Current split:

- `histology/archive/2026-03-05_mask_extraction_v4/`
  - archived v4 slice-mask pipeline design and notes
- `histology/tools/run_ndpi_review_experiment.py`
  - whole-slide NDPI review/proposal helper
- `mri/scripts/`
  - copied backup of the MRI registration stack from `mouse_mt_pipeline/scripts/`
- `mri/docs/`
  - benchmark-derived defaults and usage notes

Operational note:

- Histology and MRI code are intentionally separated so section-level mask/review experiments do not get mixed with ANTs-based MRI template and atlas registration code.

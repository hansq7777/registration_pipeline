# Directory And Naming Conventions v1

## 0. Principle

Raw crops, masks, cleaned images, exports, and revisions must be clearly
separated. No derived output may overwrite a canonical raw crop.

## 1. Workspace root

```text
project_root/
  db/
  manifests/
  logs/
  sections/
  exports/
  runs/
```

## 2. Project database

```text
project_root/db/project.sqlite
project_root/db/project.sqlite-shm
project_root/db/project.sqlite-wal
```

## 3. Run imports

Automatic proposal imports should be immutable snapshots.

```text
project_root/runs/
  auto_import/
    20260306_114837_ndpi_review_gallyas_2502_42-72/
  proposal_reruns/
    20260310_slide_2502_42-72_rerun_v002/
```

## 4. Section directory

Each section has one canonical directory:

```text
project_root/sections/<section_uid>/
  raw/
  masks/
  cleaned/
  exports/
  revisions/
```

`section_uid` format:

```text
<stain>_<sample_id>_<section_id>__<slide_short_id>__r<proposal_rank>
```

Example:

```text
gallyas_2502_48__slide2502_42_72__r02
```

## 5. Raw files

Canonical raw files:

```text
raw/crop_raw.png
raw/crop_raw_mirror.png
raw/crop_metadata.json
```

Rules:

- `crop_raw.png` is a direct crop from the NDPI pyramid
- `crop_raw_mirror.png` is only a mirrored view of the same crop
- no preprocessing may be written into either raw image

## 6. Mask files

```text
masks/tissue_mask_auto.png
masks/artifact_mask_auto.png
masks/tissue_mask_final_v001.png
masks/artifact_mask_final_v001.png
masks/usable_tissue_mask_v001.png
masks/mask_revision_manifest.json
```

Rules:

- `auto` files are immutable machine outputs
- `final` files are versioned human-approved outputs
- `usable_tissue_mask` is always derived from final tissue and artifact layers

## 7. Cleaned image files

```text
cleaned/foreground_rgba_v001.png
cleaned/cleaned_revision_manifest.json
```

Rules:

- `foreground_rgba` is the default cleaned-image export
- white-background and black-background RGB variants are derivable later from `crop_raw + usable_tissue_mask`
- do not export white/black RGB variants by default unless a specific experiment needs them

## 8. Export files

```text
exports/review_mask_v001/
exports/cyclegan_train_v001/
exports/registration_fullres_v001/
```

Within each export profile:

```text
image.png
tissue_mask.png
artifact_mask.png
usable_tissue_mask.png
foreground_rgba.png
metadata.json
```

`metadata.json` should include at minimum:

- `source_slide_identity`
  - `identity_method`
  - `path`
  - `source_slide_checksum` or `size_bytes` + `mtime`
- `source_slide`
  - source backend and fallback reason
  - overview/level0 geometry
  - crop level/downsample
  - optional physical calibration (`mpp_x`, `mpp_y`, `objective_power`)
- `algorithm_context`
  - export/profile version
  - bbox algorithm version
  - mask algorithm version
  - `git_commit`
- `proposal_context`
  - `expected_label`
  - `candidate_rank`
  - `row_index`
  - `all_candidate_boxes_snapshot`
- `crop_bbox_overview`
- `crop_bbox_level0`
- `canvas_to_slide_level0`
- `mask_qc_stats`
  - area
  - connected components
  - border touch ratio
  - neighbor occupancy ratio
- `manual_edit_summary`
- `physical_orientation`
- `reader_confidence`
- `output_files`

## 9. Revision files

```text
revisions/revision_log.jsonl
revisions/revision_0001.json
revisions/revision_0002.json
```

Each revision file should contain:

- author
- timestamp
- operation type
- changed layers
- note

## 10. Manifest files

Project-level manifests:

```text
manifests/slide_inventory.csv
manifests/section_inventory.csv
manifests/pair_inventory.csv
manifests/export_inventory.csv
```

## 11. Naming rules

Use ASCII only.

Use these suffixes consistently:

- `_auto`
- `_final_vNNN`
- `_usable`
- `_rgba`
- `_white`
- `_black`
- `_mirror`
- `_metadata`

## 12. Forbidden patterns

Do not:

- overwrite `crop_raw.png`
- overwrite `tissue_mask_auto.png`
- mix profile outputs into one flat folder
- write unstamped final files without a version suffix

# Human-In-The-Loop Histology GUI Development Spec v1

## 0. Scope

This document turns the conceptual GUI design into an implementation-oriented specification.

Target scope for the first engineering phase:

- proposal review
- section mask review and correction
- tissue/artifact layered editing
- orientation validation
- pair review
- export to:
  - `review_mask`
  - `cyclegan_train`
  - `registration_fullres`

Out of scope for the first build:

- model training orchestration
- cloud sync
- collaborative multi-user locking
- automated nonrigid registration inside the GUI

## 1. Technology choice

Recommended stack:

- GUI shell: `PySide6`
- image display: `QGraphicsView`/`QGraphicsScene` or `pyqtgraph`
- raster ops: `numpy`, `opencv-python`, `scikit-image`
- NDPI / WSI IO: `openslide-python`
- persistence: `SQLite`
- background tasks: `QThreadPool` + worker objects

Rationale:

- desktop-native image editing
- layered overlays
- stable brush interaction
- local file-heavy workflow
- easy integration with existing Python extraction scripts

## 2. High-level module map

### A. `project_core`

Responsibilities:

- project open/create
- path management
- config load/save
- schema migrations

### B. `db`

Responsibilities:

- SQLite schema
- query helpers
- transaction wrapper
- revision logging

### C. `proposal_service`

Responsibilities:

- import slide proposals from automatic runs
- rerun proposal for one slide
- update proposal bbox after manual edits

### D. `section_service`

Responsibilities:

- load section crop and masks
- generate derived masks
- recompute usable tissue
- prepare layer previews

### E. `pair_service`

Responsibilities:

- build Nissl/Myelin pair candidates
- score orientation options
- store pair/orientation decisions

### F. `export_service`

Responsibilities:

- build export jobs
- write profile-specific outputs
- maintain export manifests

### G. `ui`

Submodules:

- project window
- slide review window
- section review window
- pair review window
- export manager window

## 3. Database schema

Use SQLite with explicit migrations.

### Table: `projects`

Columns:

- `project_id` TEXT PRIMARY KEY
- `project_name` TEXT
- `created_at` TEXT
- `updated_at` TEXT
- `nissl_root` TEXT
- `gallyas_root` TEXT
- `workspace_root` TEXT
- `default_review_profile` TEXT
- `default_cyclegan_profile` TEXT
- `default_registration_profile` TEXT

### Table: `slides`

Columns:

- `slide_id` TEXT PRIMARY KEY
- `project_id` TEXT
- `stain` TEXT
- `sample_group` TEXT
- `source_path` TEXT
- `source_name` TEXT
- `readable` INTEGER
- `level_count` INTEGER
- `width_level0` INTEGER
- `height_level0` INTEGER
- `mpp_x` REAL
- `mpp_y` REAL
- `focal_metadata_json` TEXT
- `import_status` TEXT
- `created_at` TEXT
- `updated_at` TEXT

### Table: `sections`

Columns:

- `section_uid` TEXT PRIMARY KEY
- `project_id` TEXT
- `slide_id` TEXT
- `stain` TEXT
- `sample_id` TEXT
- `section_id` INTEGER
- `proposal_rank` INTEGER
- `proposal_method` TEXT
- `proposal_bbox_overview_json` TEXT
- `proposal_bbox_level0_json` TEXT
- `proposal_qc_flags_json` TEXT
- `crop_profile` TEXT
- `crop_bbox_level0_json` TEXT
- `crop_canvas_w` INTEGER
- `crop_canvas_h` INTEGER
- `crop_level` INTEGER
- `target_mpp` REAL
- `mirror_state` TEXT
- `orientation_method` TEXT
- `orientation_score_original` REAL
- `orientation_score_mirror` REAL
- `orientation_recommended` TEXT
- `orientation_ambiguous` INTEGER
- `pair_status` TEXT
- `review_status` TEXT
- `manual_review_status` TEXT
- `manual_mask_version` INTEGER
- `notes` TEXT
- `created_at` TEXT
- `updated_at` TEXT

### Table: `section_files`

One row per section artifact.

Columns:

- `file_id` TEXT PRIMARY KEY
- `section_uid` TEXT
- `file_role` TEXT
- `profile_name` TEXT
- `path` TEXT
- `checksum` TEXT
- `width_px` INTEGER
- `height_px` INTEGER
- `metadata_json` TEXT
- `created_at` TEXT

Expected `file_role` values:

- `raw_crop`
- `raw_crop_mirror`
- `tissue_mask_auto`
- `artifact_mask_auto`
- `tissue_mask_final`
- `artifact_mask_final`
- `usable_tissue_mask`
- `foreground_rgba`
- `foreground_rgb_white`
- `foreground_rgb_black`
- `registration_mask`
- `registration_outline`
- `cyclegan_train`

### Table: `pairs`

Columns:

- `pair_id` TEXT PRIMARY KEY
- `project_id` TEXT
- `nissl_section_uid` TEXT
- `gallyas_section_uid` TEXT
- `sample_id` TEXT
- `section_delta` INTEGER
- `pair_score_shape` REAL
- `pair_score_size` REAL
- `pair_score_orientation` REAL
- `pair_score_total` REAL
- `pair_status` TEXT
- `manual_override` INTEGER
- `notes` TEXT
- `created_at` TEXT
- `updated_at` TEXT

### Table: `revisions`

Columns:

- `revision_id` TEXT PRIMARY KEY
- `section_uid` TEXT
- `revision_type` TEXT
- `author` TEXT
- `timestamp` TEXT
- `base_revision_id` TEXT
- `delta_json` TEXT
- `note` TEXT

Revision types:

- `proposal_edit`
- `mask_edit`
- `artifact_edit`
- `orientation_edit`
- `pair_edit`
- `export_run`

## 4. Filesystem conventions

Project workspace:

```text
project_root/
  db/
    project.sqlite
  runs/
    auto_import/
    proposal_reruns/
  sections/
    <section_uid>/
      raw/
      masks/
      cleaned/
      exports/
      revisions/
  exports/
    review_mask/
    cyclegan_train/
    registration_fullres/
  manifests/
  logs/
```

Per section:

```text
sections/<section_uid>/
  raw/
    crop_raw.png
    crop_raw_mirror.png
  masks/
    tissue_mask_auto.png
    artifact_mask_auto.png
    tissue_mask_final_v001.png
    artifact_mask_final_v001.png
    usable_tissue_mask_v001.png
  cleaned/
    foreground_rgba_v001.png
    foreground_rgb_white_v001.png
    foreground_rgb_black_v001.png
  exports/
    review_mask_v001/
    cyclegan_train_v001/
    registration_fullres_v001/
  revisions/
    revision_log.jsonl
```

## 5. GUI view model objects

### `ProjectVM`

Fields:

- roots
- counts
- pending jobs
- open run summaries

### `SlideVM`

Fields:

- overview image
- proposal boxes
- proposal flags
- row grouping
- matched Nissl/Gallyas prior boxes

### `SectionVM`

Fields:

- current crop
- mirrored crop
- tissue/artifact/final/usable masks
- visible layers
- active tool
- undo stack
- export profile preview

### `PairVM`

Fields:

- left section
- right section
- current orientation combination
- scores
- pair decision

## 6. Interaction state machine

### Proposal state

States:

- `auto_proposed`
- `proposal_reviewed`
- `proposal_corrected`
- `crop_generated`

Transitions:

- `accept_proposal`
- `reject_proposal`
- `resize_proposal`
- `split_proposal`
- `merge_proposal`
- `rerun_proposal`

### Section mask state

States:

- `mask_auto_loaded`
- `mask_editing`
- `mask_saved`
- `mask_approved`

Transitions:

- `paint_tissue`
- `paint_artifact`
- `delete_region`
- `apply_morph_tool`
- `warp_local_region`
- `undo`
- `redo`
- `save_revision`
- `approve_mask`

### Orientation state

States:

- `orientation_unset`
- `orientation_suggested`
- `orientation_confirmed`
- `orientation_ambiguous`

Transitions:

- `toggle_mirror`
- `accept_orientation`
- `mark_ambiguous`
- `override_orientation`

### Pair state

States:

- `pair_suggested`
- `pair_confirmed`
- `pair_rejected`
- `pair_manual_override`

## 7. Canvas and tool behavior

### Coordinate model

The section review canvas must support:

- image pixel coordinates
- mask pixel coordinates
- display coordinates

All edits must be recorded in image pixel space.

### Brush semantics

Two explicit semantic modes:

- `paint tissue`
- `paint artifact`

Secondary actions:

- erase from active layer
- subtract from tissue using artifact

### Polygon semantics

Actions:

- add polygon to tissue
- subtract polygon from tissue
- add polygon to artifact
- subtract polygon from artifact

### Region tools

Required:

- keep connected component under cursor
- delete connected component under cursor
- delete border-connected component in selection

### Local warp tools

Required minimum:

- translate selected region
- rotate selected region
- scale selected region

Optional later:

- thin-plate spline local warp
- control-point nonrigid warp

## 8. Layer recomputation rules

Whenever `tissue_mask_final` or `artifact_mask_final` changes:

1. recompute `usable_tissue_mask`
2. refresh overlay
3. refresh cleaned foreground preview
4. mark section dirty

Whenever orientation changes:

1. refresh mirrored/raw view
2. invalidate stale export preview
3. update orientation fields in DB

## 9. Pair scoring contract

Minimum pair scoring inputs:

- mask area ratio
- bounding box aspect ratio
- contour similarity
- orientation score

Outputs:

- `pair_score_shape`
- `pair_score_size`
- `pair_score_orientation`
- `pair_score_total`

## 10. Orientation scoring contract

For each section pair, score:

- original/original
- original/mirrored

Store:

- `orientation_score_original`
- `orientation_score_mirror`
- `orientation_recommended`
- `orientation_ambiguous`

Ambiguity rule example:

- if absolute score gap < threshold, mark ambiguous

## 11. Export contract

### `review_mask`

Outputs:

- raw crop
- tissue/artifact/final/usable masks
- overlay storyboard

### `cyclegan_train`

Outputs:

- fixed-size cleaned RGB
- matching binary or layered masks
- orientation-fixed version
- manifest row

### `registration_fullres`

Outputs:

- highest approved crop
- final registration mask
- contour
- orientation choice
- pair metadata

## 12. Background task model

Long-running operations must not block the GUI.

Background jobs:

- NDPI crop export
- mask recompute
- pair scoring
- batch export

Each job should expose:

- status
- progress
- log text
- cancel

## 13. Undo/redo model

Undo/redo should be section-local and revision-aware.

Each command stores:

- target layer
- operation type
- affected bounding region
- before/after patch

Do not store whole-image snapshots for every step if avoidable.

## 14. Validation rules before export

Before a section is export-ready:

- raw crop exists
- final tissue mask exists
- final artifact mask exists
- usable tissue mask exists
- orientation is confirmed or explicitly marked ambiguous
- if pair-dependent export, pair is confirmed or manually overridden

## 15. Suggested implementation phases

### Phase 1. Data and review foundation

- SQLite schema
- import existing automatic runs
- project browser
- slide overview window
- section review with raster mask editing

### Phase 2. Pairing and orientation

- pair table
- pair review window
- mirror comparison
- orientation scoring

### Phase 3. Export system

- review export
- CycleGAN export
- registration export
- manifests

### Phase 4. Advanced editing

- topology tools
- local affine warps
- optional nonlinear warp tools

## 16. Recommended reuse

The existing `Myelin_anno_tool` already contains useful interaction patterns:

- mask overlay editing
- review status tracking
- export concepts

Recommended reuse approach:

- reuse interaction ideas and selected utility code
- do not force the new workflow into a z-stack-first data model
- keep this GUI section-centric and slide-aware from the start

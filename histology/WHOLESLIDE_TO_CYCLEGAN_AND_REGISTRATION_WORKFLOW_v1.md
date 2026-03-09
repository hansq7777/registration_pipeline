# Whole-Slide To CycleGAN And Registration Workflow v1

## 0. Goal

Build one histology-side workflow that turns whole-slide NDPI scans into reusable section-level data products for:

- tissue mask extraction and review
- manual mask correction
- Nissl-to-myelin CycleGAN training/inference
- downstream Nissl-to-myelin registration

This workflow must support both stains without one replacing the other:

- `nissl`
- `gallyas`

The current stable automatic mask baseline remains `baseline_v1`.

## 1. Design principles

1. Separate raw data, proposal data, masks, cleaned images, dataset exports, and registration inputs.
2. Never overwrite raw crops. Derived products must be versioned.
3. Keep stain-specific extraction logic inside one framework, but with explicit branch separation.
4. Store both `tissue` and `artifact` masks as first-class outputs.
5. Treat orientation as a reversible decision, not as a destructive preprocessing step.
6. Make every section exportable in multiple profiles for different downstream tasks.
7. Make manual correction part of the workflow, not an afterthought.

## 2. Core data model

Each extracted section should have one canonical section record with at least:

- `study_id`
- `sample_id`
- `stain`
- `section_id`
- `source_slide_path`
- `source_slide_name`
- `source_slide_stain`
- `proposal_rank`
- `proposal_method`
- `proposal_bbox_overview`
- `proposal_bbox_level0`
- `proposal_row_index`
- `proposal_qc_flags`
- `crop_profile`
- `crop_bbox_level0`
- `crop_canvas_px`
- `crop_target_mpp`
- `mirror_state`
- `orientation_method`
- `orientation_score`
- `tissue_mask_path`
- `artifact_mask_path`
- `usable_tissue_mask_path`
- `foreground_rgba_path`
- `foreground_rgb_white_path`
- `foreground_rgb_black_path`
- `registration_mask_path`
- `registration_contour_path`
- `cyclegan_export_path`
- `manual_review_status`
- `manual_mask_version`

## 3. End-to-end workflow

### Step A. Slide inventory

Input:

- Nissl NDPI root
- Gallyas NDPI root

Work:

- parse file names
- expand section IDs
- build a section-to-slide index
- build cross-stain candidate links by:
  - same `sample_id`
  - nearest `section_id`
  - allow `0`, `+1`, `-1` initially

Outputs:

- `slide_inventory.csv`
- `section_inventory.csv`
- `cross_stain_candidate_pairs.csv`

### Step B. Whole-slide proposal

Work:

- read overview image
- run stain-aware proposal:
  - `nissl`: saturation/nonwhite emphasis
  - `gallyas`: background-subtracted grayscale residual emphasis
- detect connected components
- assign row/column order
- build support mask and support bbox
- optionally use cross-stain prior:
  - `Nissl-guided size prior` for Gallyas
  - do not borrow absolute box position

Outputs per slide:

- `overview_level_last.png`
- `overview_component_mask.png`
- `overview_final_boxes.png`
- `candidate_summary.csv`

### Step C. Proposal QC and re-proposal

This is the first place to stop bad crops before downstream use.

Proposal QC flags should include:

- `row_bbox_height_outlier`
- `proposal_area_outlier`
- `proposal_touches_slide_border`
- `proposal_support_hole_risk`
- `proposal_uses_cross_stain_prior`

Re-proposal rules:

- if proposal is obviously too short or too small, expand before crop export
- if crop-level mask later touches top/bottom border too much, trigger crop-border re-proposal
- for Gallyas, allow stronger upward expansion on top-row dorsal cortex cases

### Step D. Section crop export

This stage should export the actual image cutout that later analyses will use.

Important rule:

- the exported `raw crop` must always be a direct crop from the original NDPI image pyramid
- no histogram normalization, denoising, background subtraction, or mask-based image modification should be written back into the canonical raw crop
- image preprocessing is allowed only for:
  - proposal generation
  - mask extraction
  - derived preview/export products
- all cleaned or processed outputs must be stored as derived products, never as replacements for the canonical raw crop

Do not keep only one crop style. Support multiple export profiles.

Recommended profiles:

1. `review_mask`
   - purpose: fast QC and manual mask review
   - example:
     - `crop_level=3` or `4`
     - variable canvas
     - moderate context

2. `cyclegan_train`
   - purpose: cross-stain model training/inference
   - requirements:
     - fixed output canvas size
     - fixed physical scale or near-fixed scale
     - identical export policy across stains
   - example defaults:
     - `target_long_edge_px=1024` or `1536`
     - `preserve_aspect=True`
     - `pad_to_fixed_canvas=True`
     - `context_pad_ratio=0.08 to 0.15`

3. `registration_fullres`
   - purpose: final registration
   - requirements:
     - higher detail
     - stable outline
     - preserve geometric fidelity
   - example defaults:
     - `crop_level=2` or level-0 resample to target MPP
     - larger context margin than CycleGAN export

Export controls that should be explicit parameters:

- `crop_level`
- `target_mpp`
- `target_long_edge_px`
- `fixed_canvas_w`
- `fixed_canvas_h`
- `context_pad_ratio`
- `context_pad_px`
- `pad_mode`
- `preserve_aspect`
- `background_fill`

### Step E. Mask extraction

For every crop, produce at least three mask layers:

1. `tissue_mask`
   - everything believed to be the main biological section

2. `artifact_mask`
   - border strip
   - detached stain dirt
   - cross marker / hardware shadow
   - neighboring tissue intrusion if manually marked

3. `usable_tissue_mask`
   - `tissue_mask AND NOT artifact_mask`

This separation is important because:

- CycleGAN may want one version that keeps more tissue and another that excludes ambiguous edges
- registration usually wants the cleanest geometry-focused mask

### Step F. Foreground cleaning

Derived image products should include:

1. `raw_crop_rgb`
   - unchanged crop export

2. `foreground_rgba`
   - RGB plus alpha from `usable_tissue_mask`

3. `foreground_rgb_white`
   - background removed, fill with white

4. `foreground_rgb_black`
   - background removed, fill with black

5. optional `artifact_removed_rgb`
   - remove or inpaint artifact regions only

For downstream use:

- CycleGAN usually wants `foreground_rgb_white` or `foreground_rgb_black`
- registration often wants both the clean mask and the raw crop

### Step G. Orientation handling

Do not hard-commit left-right orientation during crop export.

For every section, maintain:

- `original`
- `mirrored_lr`

Orientation selection should be a validation stage, not destructive preprocessing.

Recommended policy:

1. export both orientations logically
2. score both orientations against candidate cross-stain matches
3. store:
   - `orientation_recommended`
   - `orientation_score_original`
   - `orientation_score_mirror`
   - `orientation_ambiguous`
4. allow manual override

For ambiguous slices:

- keep both until registration review

### Step H. Cross-stain pairing layer

CycleGAN itself does not require exact pairs, but this project does benefit from a pairing table.

Build a section-pair table with:

- `sample_id`
- `nissl_section_id`
- `myelin_section_id`
- `section_delta`
- `pair_score_shape`
- `pair_score_size`
- `pair_score_orientation`
- `pair_status`

Pair classes:

- `exact`
- `adjacent_plus1`
- `adjacent_minus1`
- `weak_match`
- `manual_only`

### Step I. CycleGAN dataset packaging

Produce dataset exports in a dedicated packaging step.

Recommended outputs:

1. `domainA_nissl_train`
2. `domainB_myelin_train`
3. `domainA_nissl_val`
4. `domainB_myelin_val`

Optional weakly paired metadata:

- `cyclegan_pair_candidates.csv`

Recommended inclusion rules:

- exclude sections with failed proposal QC
- exclude sections with unresolved orientation ambiguity
- exclude masks requiring heavy manual correction from the first training pass
- keep a separate `hard_cases/` bucket for later curriculum-style training

### Step J. Registration preparation

Registration should not use the same image product as CycleGAN by default.

Prepare dedicated registration outputs:

1. `registration_mask`
   - clean binary mask
   - no background strip
   - no hardware marker

2. `registration_outline`
   - contour or spline-smoothed border

3. `registration_image`
   - raw or lightly cleaned image
   - not over-normalized

4. `distance_transform`
   - optional for mask-based prealignment

### Step K. Manual correction loop

Manual correction should operate on layered assets, not on flattened final PNGs.

Editable layers:

- `tissue_mask`
- `artifact_mask`
- `orientation_override`
- `pair_override`

Every correction should produce:

- new mask version
- reviewer ID
- timestamp
- note

## 4. Recommended directory structure

Per run:

```text
run_xxx/
  00_inventory/
  01_overview/
  02_candidates/
  03_crops_review/
  04_masks/
  05_cleaned/
  06_orientation/
  07_pairs/
  08_cyclegan/
  09_registration/
  10_qc/
  11_notes/
```

## 5. Minimum output products per section

Every accepted section should be able to provide all of the following:

- raw crop
- review crop
- CycleGAN crop
- registration crop
- tissue mask
- artifact mask
- usable tissue mask
- RGBA foreground
- white-background foreground
- orientation metadata
- pairing metadata
- QC flags

## 6. Current implementation priorities

Priority 1:

- stable proposal and crop export
- stable tissue/artifact mask separation
- manual review path

Priority 2:

- orientation scoring and override
- Nissl-guided prior for Gallyas
- re-proposal triggers

Priority 3:

- dataset packager for CycleGAN
- registration-ready contour exports

Priority 4:

- optional generative or completion experiments
- optional focus-metadata assisted QC

## 7. Recommended next implementation steps

1. Promote crop export profiles to first-class CLI settings.
2. Split current binary mask outputs into `tissue_mask` and `artifact_mask`.
3. Add `foreground_rgba`, `foreground_rgb_white`, `foreground_rgb_black` export.
4. Add orientation scoring and dual-orientation packaging.
5. Add a section-pair table linking Nissl and Gallyas candidates.
6. Add dedicated `cyclegan_train` and `registration_fullres` export profiles.

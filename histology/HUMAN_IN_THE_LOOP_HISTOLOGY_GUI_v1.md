# Human-In-The-Loop Histology GUI v1

## 0. Goal

Design one GUI that lets a human review and correct machine proposals for histology section extraction and prepare final outputs for:

- mask review and correction
- artifact labeling
- foreground-cleaned crop export
- CycleGAN dataset export
- Nissl-to-myelin registration preparation

The GUI is not only a viewer. It is the human-intervention layer that sits between automatic proposal and final export.

## 1. Design stance

The system is cooperative:

- machine provides:
  - candidate section boxes
  - tissue/artifact proposal masks
  - orientation suggestions
  - cross-stain pairing suggestions
  - export-ready derivatives
- human provides:
  - acceptance or rejection
  - mask correction
  - artifact editing
  - orientation choice
  - pair override
  - export decision

The GUI should make this cooperation explicit, not hidden.

## 2. Core user story

For each section:

1. machine proposes a crop region and mask layers
2. human inspects raw crop, mask, artifact layer, and paired-stain context
3. human edits with direct tools
4. human optionally compares original and mirrored orientation
5. human marks final section status
6. GUI exports:
   - corrected masks
   - cleaned foreground images
   - profile-specific crops
   - registration inputs
   - review metadata

## 3. Main objects in the GUI

The GUI must expose three nested levels:

1. `Slide level`
   - whole-slide overview
   - section proposals
   - row and layout context

2. `Section level`
   - one extracted crop
   - one section’s masks and exports

3. `Pair level`
   - one Nissl section and one candidate myelin partner
   - orientation comparison
   - registration preparation view

## 4. Main windows and panels

### A. Project/Home window

Purpose:

- choose workspace
- load review database
- select study roots

Show:

- Nissl source root
- Gallyas source root
- run directories
- total slides
- total sections
- unreviewed sections
- flagged sections
- export-ready sections

Actions:

- create/open project
- refresh inventory
- import machine proposal run
- rebuild pair table

### B. Slide Review window

Purpose:

- inspect whole-slide proposal quality
- accept/reject/resize section proposals before detailed editing

Main canvas:

- whole-slide overview
- proposal boxes
- proposal labels
- optional row guides
- optional matched Nissl/Gallyas reference boxes

Side panel:

- slide metadata
- filename-derived section list
- proposal QC flags
- Nissl-guided prior info
- focus/focal metadata summary if available

Slide-level actions:

- accept all proposals
- reject proposal
- split proposal
- merge proposals
- resize/move proposal box
- rerun proposal on one slide
- promote proposal to section-review queue

### C. Section Review window

Purpose:

- review one crop in detail and edit masks

Main center area:

- large image canvas with zoom/pan
- synchronized overlays

Required image layers:

- raw crop
- tissue proposal mask
- artifact mask
- usable tissue mask
- cleaned foreground preview
- optional cross-stain paired reference

Required display modes:

- raw only
- tissue overlay
- artifact overlay
- usable tissue overlay
- alpha-matted foreground
- white-background cleaned
- black-background cleaned
- outline-only view

Left panel:

- section metadata
- stain / sample / section ID
- crop profile
- current orientation
- pair suggestion
- QC flags
- export status

Right panel:

- layer list with visibility toggles
- opacity sliders
- tool settings
- brush size
- active class selector
- undo/redo history

Bottom strip:

- section queue navigation
- status buttons
- save/apply/export buttons

### D. Pair Review window

Purpose:

- compare one Nissl section with one myelin candidate
- validate orientation and pairing

Layout:

- left: Nissl
- right: Gallyas
- optional center: blended or contour-aligned preview

Modes:

- original/original
- original/mirrored
- mirrored/original
- mirrored/mirrored

Show:

- mask outlines
- bounding boxes
- shape similarity scores
- orientation scores
- section delta

Actions:

- accept current pair
- reject pair
- choose mirror state
- mark ambiguous
- choose alternate pair candidate

### E. Export Manager window

Purpose:

- define export profiles and generate final datasets

Profiles:

- `review_mask`
- `cyclegan_train`
- `registration_fullres`

Export preview should show:

- target canvas size
- target MPP or scale
- background fill mode
- orientation mode
- mask layer used
- output folder mapping

## 5. Editing tools

The GUI must support at least these tools:

### Raster tools

- brush add to `tissue_mask`
- brush add to `artifact_mask`
- eraser from active layer
- fill connected region
- lasso / freehand selection
- polygon add
- polygon subtract
- rectangle add/subtract

### Topology-aware tools

- keep component
- remove component
- remove border-touching component
- cut bridge
- fill holes
- smooth contour
- convexify local selection

### Geometry tools

- move mask region
- rotate local selection
- scale local selection
- affine warp local selection
- control-point nonrigid warp

These geometry tools are important because users may need to repair:

- slightly torn edges
- missing lobules
- neighboring-touch artifacts
- local shrink/expansion after wrong automatic proposal

### Orientation tools

- mirror left-right
- compare original vs mirrored
- lock orientation
- mark orientation uncertain

## 6. Mask layer model

The GUI should treat masks as editable layers, not one flattened binary image.

Required layers:

- `tissue_mask_auto`
- `artifact_mask_auto`
- `tissue_mask_manual_delta`
- `artifact_mask_manual_delta`
- `tissue_mask_final`
- `artifact_mask_final`
- `usable_tissue_mask`

Computation:

- `tissue_mask_final = tissue_mask_auto (+/- manual edits)`
- `artifact_mask_final = artifact_mask_auto (+/- manual edits)`
- `usable_tissue_mask = tissue_mask_final AND NOT artifact_mask_final`

This makes later reuse possible:

- one export may want `tissue_mask_final`
- another may want `usable_tissue_mask`

## 7. Export products per section

The GUI should export, per section:

### Raw section data

- `crop_raw.png` or TIFF
- `crop_mirror_raw.png` if mirrored version selected for retention

### Masks

- `tissue_mask_final.png`
- `artifact_mask_final.png`
- `usable_tissue_mask.png`
- optional vector contour JSON/SVG

### Foreground-cleaned images

- `foreground_rgba.png`
- `foreground_rgb_white.png`
- `foreground_rgb_black.png`

### CycleGAN profile

- fixed-size RGB image
- chosen orientation
- chosen foreground policy
- optional mask sidecar

### Registration profile

- full-resolution crop
- registration mask
- registration contour
- orientation metadata
- pair metadata

## 8. Human workflow states

Every section should move through explicit states:

1. `proposed`
2. `proposal_reviewed`
3. `mask_reviewed`
4. `pair_reviewed`
5. `orientation_locked`
6. `export_ready`
7. `exported`

Each state must have:

- timestamp
- reviewer
- notes

## 9. Manual intervention patterns the GUI must support

### Pattern 1. Proposal too small

User action:

- resize crop box
- rerun mask on updated crop

### Pattern 2. Artifact included in tissue

User action:

- paint artifact layer
- recompute usable tissue
- preview cleaned export

### Pattern 3. Missing tissue region

User action:

- paint tissue layer
- or warp/smooth local area

### Pattern 4. Orientation uncertain

User action:

- compare mirrored/original pair view
- choose one
- mark confidence

### Pattern 5. Cross-stain pair uncertain

User action:

- browse nearest candidate pairs
- inspect contour fit
- choose override

## 10. QC dashboard concepts

The GUI should expose fast triage views:

- sections with border-touch flags
- sections with neighbor-ownership flags
- sections where artifact area is high
- sections where orientation score gap is small
- sections where Nissl/Myelin pair score is poor
- sections not yet manually reviewed

## 11. Data versioning

Nothing should be overwritten in place without a version record.

Per section, record:

- `auto_version`
- `manual_version`
- `review_revision`
- `export_revision`

Suggested model:

- store mask edits as versioned raster files
- store edit commands optionally as JSON patch log

## 12. Suggested implementation architecture

To reduce development risk, build the GUI in layers:

### Layer 1. Review database

- SQLite or Parquet + filesystem
- one canonical section table
- one pair table
- one revision table

### Layer 2. Backend services

- proposal service
- crop export service
- mask recompute service
- export packaging service
- pair/orientation scoring service

### Layer 3. GUI shell

Prefer a desktop GUI with strong image interaction support.

Practical options:

- `PySide6 / Qt`
- or extend the existing `Myelin_anno_tool` if reuse is desired

Rationale:

- brush and polygon editing
- layered image display
- shortcut-rich workflow
- local filesystem integration

### Layer 4. Optional reuse from existing tooling

Existing annotation tooling already demonstrates useful concepts:

- mask review
- quick audit workflow
- tracker-driven review state
- brush/eraser and overlay display

Those ideas can be reused, but this new GUI needs broader histology-specific objects:

- slide proposal review
- pair/orientation review
- export profile manager

## 13. Minimum viable GUI

The smallest usable first release should include:

1. project browser
2. slide overview with editable proposal boxes
3. section review with tissue/artifact masks
4. orientation mirror toggle
5. save versioned masks
6. export `foreground_rgba`, `foreground_rgb_white`, `foreground_rgb_black`
7. export `cyclegan_train` and `registration_fullres`

## 14. Recommended next design step

Before implementation, define:

1. canonical section database schema
2. exact export folder conventions
3. mask-layer file naming rules
4. orientation score definition
5. pair score definition

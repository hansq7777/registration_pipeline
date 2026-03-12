# Histology Docs Inventory And Prune Recommendation 2026-03-12

## Purpose

This note compresses the Markdown documentation under
`C:\work\registration_pipeline\histology`
into a small number of functional categories, and marks which files are:

- `core keep`
- `keep but secondary`
- `archive candidate`
- `delete candidate`

No files were deleted by this note.

## Scope

Included:

- top-level project Markdown files
- `gui_mvp/*.md`
- archive manifest/workflow docs that are still part of project history

Excluded from review:

- `.venv` package license / third-party Markdown files
- generated dependency metadata

Those are environment noise, not project knowledge assets.

## Compression Summary

The current docs mostly collapse into five buckets:

1. `Canonical workflow / policy`
2. `GT-backed experiment decisions`
3. `GUI product / implementation docs`
4. `Operational performance / efficiency docs`
5. `Historical bridge / archive material`

The main redundancy is not that there are too many topics.

The redundancy is that some files are:

- one-off summaries of more detailed docs
- old conceptual docs now overshadowed by implementation specs
- pre-weighted search notes that have already been superseded by weighted selection

## Category A: Canonical Workflow / Policy

These are the documents that define how the project should currently operate.

### 1. [README.md](/mnt/c/work/registration_pipeline/histology/README.md)

Role:

- top-level index of the histology pipeline
- routing note for what this directory is responsible for
- quick pointer to core workflow and benchmark docs

Why keep:

- this is the only short entry document at the folder root
- useful even if all other docs remain

Recommendation:

- `core keep`

### 2. [WHOLESLIDE_TO_CYCLEGAN_AND_REGISTRATION_WORKFLOW_v1.md](/mnt/c/work/registration_pipeline/histology/WHOLESLIDE_TO_CYCLEGAN_AND_REGISTRATION_WORKFLOW_v1.md)

Role:

- end-to-end product workflow
- clearest high-level statement of the intended histology-side system
- bridges NDPI review, mask review, CycleGAN, and registration outputs

Why keep:

- this is the canonical workflow document
- other docs depend on it conceptually

Recommendation:

- `core keep`

### 3. [GT_BACKED_EVALUATION_PROTOCOL_v1.md](/mnt/c/work/registration_pipeline/histology/GT_BACKED_EVALUATION_PROTOCOL_v1.md)

Role:

- defines canonical GT sources
- defines evaluation rules
- locks which benchmarks are trusted

Why keep:

- this is a policy / methodology contract
- should remain stable and easy to find

Recommendation:

- `core keep`

### 4. [MYELIN_OVERVIEW_BBOX_NEXT_EXPERIMENT_PLAN_20260312.md](/mnt/c/work/registration_pipeline/histology/MYELIN_OVERVIEW_BBOX_NEXT_EXPERIMENT_PLAN_20260312.md)

Role:

- forward-looking experimental strategy
- redefines bbox work as ownership / rival suppression / compactness optimization

Why keep:

- this is the current planning document for the next iteration
- it is not redundant with past search notes

Recommendation:

- `core keep`

## Category B: GT-Backed Experiment Decisions

These files encode actual search outcomes and recommended defaults.

### 5. [MYELIN_BBOX_WEIGHTED_GT_SEARCH_20260312.md](/mnt/c/work/registration_pipeline/histology/MYELIN_BBOX_WEIGHTED_GT_SEARCH_20260312.md)

Role:

- weighted GT-backed myelin bbox decision
- introduces the current 50/30/20 objective
- states the current default runtime choice

Why keep:

- this appears to be the current canonical myelin bbox decision file

Recommendation:

- `core keep`

### 6. [MYELIN_MASK_GT_STRATEGY_SEARCH_20260311.md](/mnt/c/work/registration_pipeline/histology/MYELIN_MASK_GT_STRATEGY_SEARCH_20260311.md)

Role:

- current myelin crop-mask strategy comparison
- records best mask strategy and failure pattern

Why keep:

- still looks like an active decision source
- not replaced by a newer equivalent in the current tree

Recommendation:

- `core keep`

### 7. [NISSL_BBOX_GT_SEARCH_20260311.md](/mnt/c/work/registration_pipeline/histology/NISSL_BBOX_GT_SEARCH_20260311.md)

Role:

- GT-backed bbox conclusion for Nissl
- explicit claim that bbox is not the bottleneck for Nissl

Why keep:

- useful because it prevents wrong transfer of myelin logic into Nissl

Recommendation:

- `core keep`

### 8. [NISSL_MASK_GT_STRATEGY_SEARCH_20260311.md](/mnt/c/work/registration_pipeline/histology/NISSL_MASK_GT_STRATEGY_SEARCH_20260311.md)

Role:

- GT-backed mask strategy for Nissl
- states stable default and conservative fallback

Why keep:

- current Nissl default seems to depend on this

Recommendation:

- `core keep`

## Category C: GUI Product / Implementation Docs

These files define the human-in-the-loop system.

### 9. [HUMAN_IN_THE_LOOP_HISTOLOGY_GUI_DEV_SPEC_v1.md](/mnt/c/work/registration_pipeline/histology/HUMAN_IN_THE_LOOP_HISTOLOGY_GUI_DEV_SPEC_v1.md)

Role:

- implementation-oriented GUI spec
- most detailed engineering document in this subdomain

Why keep:

- this is the canonical engineering reference for the GUI

Recommendation:

- `core keep`

### 10. [HUMAN_IN_THE_LOOP_HISTOLOGY_GUI_v1.md](/mnt/c/work/registration_pipeline/histology/HUMAN_IN_THE_LOOP_HISTOLOGY_GUI_v1.md)

Role:

- conceptual GUI design doc
- higher-level product framing and UX intent

Problem:

- partially overlaps with the development spec
- much of the actionable content now exists in the dev spec

What still makes it useful:

- preserves original design stance and user story
- easier to read than the implementation spec

Recommendation:

- `keep but secondary`
- if you want to slim the root folder, this is a strong `archive candidate`

### 11. [gui_mvp/README.md](/mnt/c/work/registration_pipeline/histology/gui_mvp/README.md)

Role:

- local README for the MVP implementation subtree

Why keep:

- scoped to `gui_mvp/`, not redundant with the root README

Recommendation:

- `keep but secondary`

### 12. [gui_mvp/DIRECTORY_AND_NAMING_CONVENTIONS_v1.md](/mnt/c/work/registration_pipeline/histology/gui_mvp/DIRECTORY_AND_NAMING_CONVENTIONS_v1.md)

Role:

- storage layout and naming rules for the GUI workspace

Why keep:

- operational contract for GUI-generated data

Recommendation:

- `keep but secondary`

## Category D: Operational Performance / Efficiency

These are useful if performance and Windows-side IO remain active concerns.

### 13. [NDPI_PIPELINE_EFFICIENCY_PLAYBOOK_v1.md](/mnt/c/work/registration_pipeline/histology/NDPI_PIPELINE_EFFICIENCY_PLAYBOOK_v1.md)

Role:

- efficiency strategy for NDPI review and crop/mask staging

Why keep:

- useful operational note
- not duplicated elsewhere at the same detail level

Recommendation:

- `keep but secondary`

### 14. [WINDOWS_TIMING_HARNESS_REFERENCE_v1.md](/mnt/c/work/registration_pipeline/histology/WINDOWS_TIMING_HARNESS_REFERENCE_v1.md)

Role:

- measurement protocol for Windows-side timing
- avoids incorrect WSL-based conclusions

Why keep:

- important only if GUI / timing harness benchmarking is still active

Potential redundancy:

- some operational guidance overlaps with the efficiency playbook

Recommendation:

- `keep but secondary`
- if benchmarking is no longer active, this becomes an `archive candidate`

## Category E: Historical Bridge / Archive Material

These files are still informative, but they are less central now.

### 15. [REVIEW_EXPERIMENT_SUMMARY_20260311.md](/mnt/c/work/registration_pipeline/histology/REVIEW_EXPERIMENT_SUMMARY_20260311.md)

Role:

- bridge summary of March 5 to March 11 work
- condenses findings from multiple experiment notes

Problem:

- much of it is now duplicated by:
  - the root README
  - GT-backed search files
  - workflow docs

Why it may still help:

- useful as a narrative snapshot
- quick context for what changed during that week

Recommendation:

- `archive candidate`
- strongest top-level candidate to remove from the root if you want immediate doc slimming

### 16. [MYELIN_BBOX_GT_STRATEGY_SEARCH_20260311.md](/mnt/c/work/registration_pipeline/histology/MYELIN_BBOX_GT_STRATEGY_SEARCH_20260311.md)

Role:

- pre-weighted myelin bbox search note
- earlier search logic before weighted objective became the main decision rule

Problem:

- partly superseded by:
  - `MYELIN_BBOX_WEIGHTED_GT_SEARCH_20260312.md`
  - `MYELIN_OVERVIEW_BBOX_NEXT_EXPERIMENT_PLAN_20260312.md`

What still matters:

- contains historical rationale and failure analysis before weighted scoring

Recommendation:

- `archive candidate`
- not a strong delete target if provenance matters
- but not necessary as a top-level active doc

### 17. [archive/2026-03-05_mask_extraction_v4/MASK_EXTRACTION_EXPERIMENT_DESIGN_v1.md](/mnt/c/work/registration_pipeline/histology/archive/2026-03-05_mask_extraction_v4/MASK_EXTRACTION_EXPERIMENT_DESIGN_v1.md)

Role:

- archived v4 reproducible workflow for older Nissl mask extraction setup

Why keep:

- already properly archived
- does not clutter the root

Recommendation:

- `keep archived`

### 18. [archive/2026-03-05_mask_extraction_v4/ARCHIVE_MANIFEST.md](/mnt/c/work/registration_pipeline/histology/archive/2026-03-05_mask_extraction_v4/ARCHIVE_MANIFEST.md)

Role:

- minimal archive descriptor

Recommendation:

- `keep archived`

## Candidate Deletion / Archive Table

These are the files I would put in front of you first for pruning review.

| Priority | File | Recommendation | Why |
|---|---|---|---|
| 1 | [REVIEW_EXPERIMENT_SUMMARY_20260311.md](/mnt/c/work/registration_pipeline/histology/REVIEW_EXPERIMENT_SUMMARY_20260311.md) | archive or delete | mostly a bridge summary; content now distributed across more canonical docs |
| 2 | [HUMAN_IN_THE_LOOP_HISTOLOGY_GUI_v1.md](/mnt/c/work/registration_pipeline/histology/HUMAN_IN_THE_LOOP_HISTOLOGY_GUI_v1.md) | archive, not direct delete | conceptual doc now overshadowed by dev spec |
| 3 | [MYELIN_BBOX_GT_STRATEGY_SEARCH_20260311.md](/mnt/c/work/registration_pipeline/histology/MYELIN_BBOX_GT_STRATEGY_SEARCH_20260311.md) | archive | historical search note superseded by weighted search |
| 4 | [WINDOWS_TIMING_HARNESS_REFERENCE_v1.md](/mnt/c/work/registration_pipeline/histology/WINDOWS_TIMING_HARNESS_REFERENCE_v1.md) | conditional archive | only worth keeping active if timing harness work is still ongoing |

## Files I Do Not Recommend Deleting

These still look like the minimum useful active set:

- [README.md](/mnt/c/work/registration_pipeline/histology/README.md)
- [WHOLESLIDE_TO_CYCLEGAN_AND_REGISTRATION_WORKFLOW_v1.md](/mnt/c/work/registration_pipeline/histology/WHOLESLIDE_TO_CYCLEGAN_AND_REGISTRATION_WORKFLOW_v1.md)
- [GT_BACKED_EVALUATION_PROTOCOL_v1.md](/mnt/c/work/registration_pipeline/histology/GT_BACKED_EVALUATION_PROTOCOL_v1.md)
- [MYELIN_BBOX_WEIGHTED_GT_SEARCH_20260312.md](/mnt/c/work/registration_pipeline/histology/MYELIN_BBOX_WEIGHTED_GT_SEARCH_20260312.md)
- [MYELIN_MASK_GT_STRATEGY_SEARCH_20260311.md](/mnt/c/work/registration_pipeline/histology/MYELIN_MASK_GT_STRATEGY_SEARCH_20260311.md)
- [NISSL_BBOX_GT_SEARCH_20260311.md](/mnt/c/work/registration_pipeline/histology/NISSL_BBOX_GT_SEARCH_20260311.md)
- [NISSL_MASK_GT_STRATEGY_SEARCH_20260311.md](/mnt/c/work/registration_pipeline/histology/NISSL_MASK_GT_STRATEGY_SEARCH_20260311.md)
- [MYELIN_OVERVIEW_BBOX_NEXT_EXPERIMENT_PLAN_20260312.md](/mnt/c/work/registration_pipeline/histology/MYELIN_OVERVIEW_BBOX_NEXT_EXPERIMENT_PLAN_20260312.md)
- [HUMAN_IN_THE_LOOP_HISTOLOGY_GUI_DEV_SPEC_v1.md](/mnt/c/work/registration_pipeline/histology/HUMAN_IN_THE_LOOP_HISTOLOGY_GUI_DEV_SPEC_v1.md)
- [gui_mvp/DIRECTORY_AND_NAMING_CONVENTIONS_v1.md](/mnt/c/work/registration_pipeline/histology/gui_mvp/DIRECTORY_AND_NAMING_CONVENTIONS_v1.md)

## Suggested Minimal Active Root Set

If the goal is to make the root directory visibly cleaner without losing the main knowledge base, I would keep active in the root:

- `README.md`
- `WHOLESLIDE_TO_CYCLEGAN_AND_REGISTRATION_WORKFLOW_v1.md`
- `GT_BACKED_EVALUATION_PROTOCOL_v1.md`
- `MYELIN_BBOX_WEIGHTED_GT_SEARCH_20260312.md`
- `MYELIN_MASK_GT_STRATEGY_SEARCH_20260311.md`
- `NISSL_BBOX_GT_SEARCH_20260311.md`
- `NISSL_MASK_GT_STRATEGY_SEARCH_20260311.md`
- `MYELIN_OVERVIEW_BBOX_NEXT_EXPERIMENT_PLAN_20260312.md`
- `HUMAN_IN_THE_LOOP_HISTOLOGY_GUI_DEV_SPEC_v1.md`
- `NDPI_PIPELINE_EFFICIENCY_PLAYBOOK_v1.md`

And I would consider moving out of the active root:

- `REVIEW_EXPERIMENT_SUMMARY_20260311.md`
- `HUMAN_IN_THE_LOOP_HISTOLOGY_GUI_v1.md`
- `MYELIN_BBOX_GT_STRATEGY_SEARCH_20260311.md`
- optionally `WINDOWS_TIMING_HARNESS_REFERENCE_v1.md`

## Next Step Recommendation

The cleanest next move is not deletion first.

It is:

1. create an `archive/docs_20260312/` folder or similar
2. move the four candidate files there
3. leave this inventory note in the root
4. only delete later if you confirm nothing unique is still needed from them

That keeps provenance while still shrinking the active surface area of the folder.

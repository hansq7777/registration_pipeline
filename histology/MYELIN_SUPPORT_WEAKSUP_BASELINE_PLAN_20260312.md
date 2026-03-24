# Myelin Weak-Supervision Support Baseline Plan

Date: 2026-03-12

## Purpose

Define an executable weak-supervision baseline for **support generation** in the
current Gallyas / myelin crop-level mask pipeline.

This is not a replacement for the current M3 runtime yet. The goal is to test
whether a learned or weakly supervised **support map** can improve the part of
the pipeline that still fails most often:

- weak lateral cortical edge support
- structured background vs true tissue ambiguity
- under-coverage near low-OD outer cortex

The current target integration point is the support stage used before:

- `hysteresis_support_reconstruct(...)`

in:

- [search_myelin_mask_strategies.py](/mnt/c/work/registration_pipeline/histology/tools/search_myelin_mask_strategies.py)

## Current M3 Support Interface

The current best runtime family, `m3_hyst_entres_guard_v1`, is:

- support mask: `m2_candidate_union_v2(crop)`
- core score: `residual_score(crop)`
- structural core: `center_default(crop)`
- reconstruction:
  - strong core from high-quantile residual inside support
  - constrained propagation inside support
  - fallback to `m2_hybrid_entres_tight_v1` if result is too small

So the easiest way to test weak supervision is:

- keep the current M3 core logic
- replace or augment only `support_mask`

## Labeling Specification

Weak supervision should not try to annotate the final perfect tissue boundary
first. It should annotate **support semantics**:

### Classes

1. `support_tissue`
- pixels that should be allowed as valid tissue support
- can include weak but real cortical edge regions
- does **not** need to be an ultra-tight final contour

2. `non_support_background`
- clear glass / empty background
- obvious outside-of-tissue regions
- obvious whitespace or clearly non-brain area

3. `artifact_or_rival`
- neighboring section
- glass edge / strip artifact
- dark marks, dirt, fold-like junk, scanner edge contamination

### Annotation style

Use sparse scribbles or sparse brush strokes, not dense full masks.

For each crop:

- positive scribbles:
  - tissue interior
  - weak lateral/top edge that should still count as tissue
- background scribbles:
  - clean glass
  - empty crop corners
- artifact/rival scribbles:
  - neighboring section areas
  - structured junk that should not become support

### Annotation rules

Do not force uncertain boundary pixels into positive or negative if confidence is
low. Leave them unlabeled.

Positive labels should prefer:

- confident tissue interior
- known weak-edge tissue that current M3 tends to miss

Negative labels should prefer:

- obvious outside tissue
- obvious rival or artifact

## Training Input / Output

### Training unit

Use crop-level images already aligned to the current GUI default bbox policy.

Current source:

- `D:\\Research\\Image Analysis\\Nanozoomer scans\\20250327 rat myelin quantification\\Tissue&Masks\\test`

Each section already has:

- `crop_raw.png`
- `tissue_mask_final.png`
- `artifact_mask_final.png`
- `usable_tissue_mask.png`
- `metadata.json`

### Input features

The first weak baseline should not use a deep model.
Use a lightweight pixel classifier on handcrafted features:

- grayscale intensity
- residual score
- local entropy
- nonwhite score
- gradient magnitude
- local variance / local contrast
- normalized x/y position in crop
- distance-to-border

Optional later:

- small patch context features
- atlas or side prior if available later

### Output

Per-pixel `support_probability_map` in `[0, 1]`, same spatial size as crop.

From that probability map derive:

- `support_mask_strict = prob >= t_high`
- `support_mask_soft = prob >= t_low`

The first integration can use one binary support mask only.

## How To Integrate With Current M3

### Phase 1: drop-in support replacement

Current:

- `support_mask = m2_candidate_union_v2(crop)`

Experimental:

- `support_mask = weak_support_mask(crop)`

and keep everything else unchanged:

- same `core_score = residual_score(crop)`
- same `structural_core = center_default(crop)`
- same hysteresis reconstruction
- same fallback guard

This isolates the question:

- does better support alone improve M3?

### Phase 2: hybrid support

If pure replacement is unstable, test:

- `support_mask = weak_support OR m2_candidate_union_v2`

and separately:

- `support_mask = weak_support AND dilated(m2_candidate_union_v2)`

This gives:

- a recall-oriented hybrid
- a more conservative hybrid

### Phase 3: strict/soft dual support

If Phase 1 shows promise, use:

- `support_mask_soft` for propagation domain
- `support_mask_strict` as an extra gate for seed preservation or fallback

This would move M3 closer to a true weak-supervised hysteresis design.

## Minimal Experiment Design

### Experiment WS0: GT-derived simulation baseline

Purpose:

- test the integration wiring before collecting new scribble labels
- answer whether support-learning is even worth pursuing

Method:

- derive pseudo support labels from current GT masks
- train a lightweight pixel classifier
- use predicted support in place of `m2_candidate_union_v2`
- compare against current `m3_hyst_entres_guard_v1`

### Pseudo-label construction from current GT

For each GT mask:

- `positive_support`
  - eroded GT mask interior
- `hard_negative`
  - pixels well outside a dilated GT mask
- `ignore_band`
  - boundary ring between eroded and dilated GT

If `artifact_mask_final.png` is non-empty:

- map artifact pixels into `artifact_or_rival` negative class

Suggested initial morphology:

- erosion radius: `1%–2%` of min crop dimension
- dilation radius: `1%–2%` of min crop dimension

This gives conservative labels:

- positives are high-confidence tissue support
- negatives are high-confidence non-support
- ambiguous boundary pixels are ignored

### Experiment WS1: true weak-supervision pilot

After WS0 proves the pipeline can work:

- manually annotate `15–25` hard crops with scribbles
- train the same lightweight support classifier
- evaluate whether true weak labels outperform GT-derived pseudo-support

Recommended hard-case emphasis:

- `2503_144`
- `2504_161`
- `2504_185`
- `2504_47`
- `2504_5`
- `2507_66`

### Train / validation split

Never mix sections from the same slide across train and validation.

Split by slide:

- train slides
- validation slides
- optional held-out hard-case set

Otherwise support quality will look better than it really is.

## Can Current GT Masks Be Used Directly To Start Testing?

### Short answer

Yes, **for an initial offline proof-of-concept**, current GT masks are enough to
start testing.

### Why they are useful now

Current GT already gives:

- stable crop space
- final tissue mask
- artifact mask
- traceable metadata

That is enough to build:

- pseudo-positive support labels
- pseudo-negative background labels
- an initial support classifier experiment

This is the fastest way to test:

- whether replacing `m2_candidate_union_v2` with a learned support map helps M3

### Why GT masks are not sufficient as the final weak-supervision solution

Current GT masks are **final segmentation labels**, not true support labels.

That means:

- they do not explicitly encode:
  - uncertain fringe
  - rival section areas
  - weak-but-valid support beyond the final conservative contour
- they are too “clean” to represent what a weak support annotator would
  actually provide

So GT masks are suitable for:

- WS0 integration test
- feature ablation
- threshold/hybrid strategy search

But not enough to fully answer:

- what the best real weak-supervised support definition is

That requires scribble-based support labels later.

## Recommended Minimal Stack

### First baseline model

Use one of:

- scikit-learn RandomForest pixel classifier
- ilastik pixel classifier

Why:

- low engineering cost
- fast iteration
- naturally fits sparse weak labels
- more interpretable than immediately training U-Net

### Evaluation metrics

Do not judge only by support map quality.

Judge by downstream final mask quality after plugging support into M3:

- Dice
- BF64
- HD95
- FP / GT area
- left/right recall
- top boundary recall
- collapse count

### Promotion rule

Promote weak-support baseline only if:

- downstream `m3 + weak_support` beats current `m3_hyst_entres_guard_v1`
- especially on:
  - BF64
  - HD95
  - FP/GT
  - lateral boundary recall
- without introducing new collapse cases

## Immediate Next Step

Implement `WS0` first:

1. derive pseudo support labels from current GT masks
2. train a lightweight support classifier
3. plug predicted support into current M3
4. compare against:
   - `m3_hyst_entres_guard_v1`
   - optionally `m2_hybrid_entres_v1`

Only if `WS0` shows promise should the project spend time on:

- manual scribble collection
- ilastik interactive support labeling
- GUI-level integration

## Current Status

`WS0` implementation scaffold now exists in:

- [evaluate_myelin_support_ws0.py](/mnt/c/work/registration_pipeline/histology/tools/evaluate_myelin_support_ws0.py)

What is already done:

- GT-driven pseudo support label derivation
- slide-aware holdout split support
- three weak-support integration modes:
  - `ws0_rf_replace_v1`
  - `ws0_rf_or_v1`
  - `ws0_rf_gate_v1`
- downstream comparison against current `m3_hyst_entres_guard_v1`

What was learned from the first run attempt:

- the `WS0` idea is implementable with current GT assets
- but the current evaluation path is still too heavy for efficient iteration
- the bottleneck is not only the classifier, but the full downstream path:
  - feature extraction
  - support prediction over full crop
  - repeated `M3` reconstruction
  - contour / leakage metrics

Therefore the next practical step is:

1. keep the current `WS0` script as the canonical experiment entry
2. add a lighter `WS0 probe` path for:
   - selected slides
   - selected sections
   - reduced metric set
3. only after that rerun full confirm

In other words:

- `WS0` is started and wired into the project
- but it still needs a lightweight evaluation harness before large-scale confirm is worth the runtime cost

## WS0 Probe-Fast Initial Result

The first reduced probe has now been run on three representative sections:

- `2503_144`
- `2504_5`
- `2507_66`

using:

- reduced metric set
- lightweight logistic-regression support classifier
- current `M3` as the downstream mask generator
- comparison only against:
  - `m3_hyst_entres_guard_v1`
  - `ws0_lr_gate_v1`

Result summary:

- `WS0 gate` is **not yet strong enough** to justify full confirm
- it improved on `2507_66`
- but was clearly worse on `2503_144`
- and clearly worse on `2504_5`

So the current interpretation is:

- learned support has **some signal**
- but the present pseudo-label / feature / gate formulation is too unstable
- the next weak-supervision step should **not** be “run full confirm”
- it should be:
  1. improve pseudo-support target design
  2. improve gating logic
  3. only then rerun a broader probe

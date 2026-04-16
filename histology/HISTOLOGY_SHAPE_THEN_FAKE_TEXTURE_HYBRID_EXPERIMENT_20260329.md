# Histology Shape-Then-Fake-Texture Hybrid Registration Experiment

Date:
- `2026-03-29`

Branch:
- `experiment/histology_registration_preproc_20260326`

## Goal

Test whether a two-stage strategy can combine the strengths of:
- `shape-driven registration` for stable coarse geometry alignment
- `fake-myelin vs true-myelin local texture refinement` for local detail recovery after geometry is already correct

The main question is:
- after a geometry-first alignment is accepted, can a tightly constrained same-modality-like texture refinement improve `Dice / HD95` without reintroducing the regression problems seen in direct cross-stain MI-driven rigid/affine?

## Why This Experiment Is Needed

Current evidence supports three points:

1. `input-only` is already strong because current preprocessing removes much of the easy rigid error.
2. `mask-rigid` is more reliable than `MI-rigid` for coarse alignment.
3. `fake myelin` by itself does not clearly outperform `raw Nissl` when used as the primary coarse-registration moving image.

This suggests a different use for fake myelin:
- not as the first global registration input
- but as a second-stage local refinement image after a shape-based transform has already stabilized geometry

This is also conceptually consistent with the P4663-style interpretation:
- synthetic myelin is more useful as a registration helper than as a direct replacement for true myelin

## Core Hypothesis

The hybrid strategy may outperform both:
- `shape-only`
- and `direct intensity-driven cross-stain registration`

because it separates the problem into:

1. `Stage A: geometry`
- use tissue shape to solve global pose safely

2. `Stage B: local texture`
- use `fake myelin -> true myelin` similarity only after the global pose is already near-correct
- keep this refinement local, low-DOF, and strongly gated

## Fixed Evaluation Policy

Keep the current benchmark and bookkeeping frozen:

- benchmark set:
  - current `131` usable registration units
- same group handling:
  - existing `group 1/2` split
- same physical normalization:
  - `target_um_per_px = 10.0`
- same working scale:
  - `working_long_edge = 1024`
- same metric exclusion:
  - `mask mode = tissue_only`
- same runner logic:
  - `best accepted state runner`
- same gate:
  - geometry-based acceptance using `Dice` and `HD95`
- same reporting:
  - `input`
  - per-stage metrics
  - accepted/rejected bookkeeping
  - success rate
  - regression tail

## Inputs

### Stage A Inputs

- moving:
  - Nissl tissue mask
- fixed:
  - myelin tissue mask
- representation:
  - shape / boundary / distance-transform objective

### Stage B Inputs

- moving:
  - `epoch30 fake myelin`
- fixed:
  - true myelin
- representation:
  - same evaluation canvas already rebuilt for fake-myelin experiments
- preprocessing for first round:
  - `percentile clipping + fixed white background`

Rationale:
- `epoch30` was the strongest fake-myelin checkpoint so far on the full benchmark
- using only one fake checkpoint keeps this experiment interpretable

## Experimental Arms

### Baseline Arms

1. `input-only`
2. `MI-rigid`
   - current best rigid baseline:
   - `gradient_mag_blur_1.5`
3. `mask-rigid`

### Hybrid Arms

4. `mask-rigid -> fake-texture rigid-local`
5. `mask-rigid -> fake-texture weak-affine-local`

### Control Arm To Isolate Fake-Myelin Value

6. `mask-rigid -> raw-Nissl texture rigid-local`

This control is important.
Without it, an apparent gain could come from "any second-stage local refinement" rather than from `fake myelin` specifically.

## Transform Strategy

### Stage A

Use the existing best `mask-rigid` implementation:
- low-DOF
- geometry-driven
- accepted or rejected under the current gate

### Stage B

Stage B must start from the current best accepted Stage A state.

This means:
- if `mask-rigid` is accepted, Stage B starts from `mask-rigid`
- if `mask-rigid` is rejected, Stage B starts from `input`

This preserves the current monotonic runner logic.

### Local Texture Rigid

Allowed motion should be explicitly constrained around the Stage A solution.

Recommended first-round bounds:
- translation:
  - within `+/- 3%` of canvas width/height
- rotation:
  - within `+/- 4 degrees`
- isotropic scale:
  - disabled for rigid-local

### Local Texture Weak Affine

Only test if rigid-local is stable enough.

Recommended first-round constraints:
- translation:
  - same local window as above
- rotation:
  - same local window as above
- isotropic scale:
  - within `0.98 to 1.02`
- anisotropic scaling:
  - disabled or very tightly constrained
- shear:
  - disabled

This is not a full affine rescue path.
It is a deliberately weak local adjustment layer.

## Texture Objective

First-round objective for Stage B:
- `MI` or `CC` on `fake myelin` vs `true myelin`

but under these restrictions:
- same-modality-like image pair only
- local transform window only
- geometry gate still decides acceptance

This is the key design choice:
- allow intensity information to guide only local refinement
- but do not allow intensity scores to override geometric quality

## Acceptance Logic

### Stage A

Same as current:
- candidate accepted only if it improves geometry under the current gate

### Stage B

Same rule:
- texture-local candidate must beat the current best accepted state on geometry
- if not, reject it and keep the shape-only result

This is the main protection against "texture chasing."

## Primary Questions

1. Does `mask-rigid -> fake-texture rigid-local` improve success rate over `mask-rigid` alone?
2. Does it improve median `Dice` and reduce median `HD95`?
3. Does it reduce or worsen the bad-case regression tail?
4. Is `fake myelin` better than `raw Nissl` as the Stage B texture source?
5. Does weak-affine-local add value beyond rigid-local, or does it reopen regression risk?

## Success Criteria

Advance the hybrid route only if it meets at least one of these without worsening the others too much:

- higher success rate than `mask-rigid`
- positive median `Dice` delta over `mask-rigid`
- lower median `HD95` than `mask-rigid`
- smaller `HD95` regression tail than `mask-rigid`

Additionally:
- `mask-rigid -> fake-texture` should beat `mask-rigid -> raw-Nissl texture`
  - otherwise fake-myelin is not earning its complexity

## Failure Criteria

Deprioritize the hybrid route if:

- it improves average metrics only slightly but increases regression tail
- it helps only a tiny subset of cases while hurting already-good alignments
- `raw-Nissl texture refine` matches or beats `fake-myelin texture refine`
- local texture steps are accepted rarely enough that they do not justify maintenance cost

## Recommended Execution Order

### Round 1: Minimal informative run

Run on all `131` units:

1. `input-only`
2. `MI-rigid`
3. `mask-rigid`
4. `mask-rigid -> raw-Nissl texture rigid-local`
5. `mask-rigid -> fake-myelin(epoch30) texture rigid-local`

Why this first:
- it directly answers whether the hybrid idea is worth anything
- it isolates the value of fake-myelin
- it avoids opening weak-affine too early

### Round 2: Only if Round 1 is promising

Add:
- `mask-rigid -> fake-myelin weak-affine-local`

Only do this if rigid-local already shows stable gains.

## Outputs

Each run should write:

- `run_manifest.json`
- `storyboard.png`
- accepted/rejected bookkeeping
- `accepted path`
- geometry metrics at:
  - `input`
  - `mask-rigid`
  - `texture-rigid-local`
  - optional `texture-weak-affine-local`

Aggregate outputs should include:

- success rate
- mean / median `Dice` delta vs `mask-rigid`
- mean / median `HD95` delta vs `mask-rigid`
- regression tail summary
- per-arm accepted-stage histogram
- count of cases where Stage B is accepted

## Recommended Storyboard Structure

For this experiment, storyboard should explicitly show:

1. `input`
2. `mask-rigid`
3. `texture-local candidate`
4. `best accepted state`

This makes the role of Stage B easy to interpret.

## Expected Interpretation Possibilities

### Best-case outcome

- `mask-rigid` fixes global pose
- local fake-myelin refinement improves boundary fit on some cases
- regression tail stays controlled

This would support:
- `shape first, texture second`

### Neutral outcome

- `mask-rigid` still does most of the useful work
- local fake-myelin refinement is rarely accepted

This would mean:
- fake-myelin is not harmful if gated
- but not worth default deployment yet

### Negative outcome

- local texture stage reintroduces the same failure mode as direct MI
- accepted path rarely advances beyond `mask-rigid`
- or regression tail worsens

This would suggest:
- keep fake-myelin for segmentation-helper experiments only
- not for registration refinement

## One-Sentence Takeaway

The hybrid experiment asks whether fake myelin becomes useful only after geometry is already fixed: first align shape safely, then allow a tightly constrained local texture refinement to compete under the same geometry gate.

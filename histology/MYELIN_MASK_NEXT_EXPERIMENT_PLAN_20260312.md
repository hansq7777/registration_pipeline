# Myelin Mask Next Experiment Plan 2026-03-12

## Scope

- Deep-research reference:
  - `Maskgen-deep-research-report.md`
- Current project baselines:
  - `MYELIN_MASK_GT_STRATEGY_SEARCH_20260311.md`
  - GUI/runtime current Gallyas preset family:
    - `hybrid_tightcand_k7_o03`
    - `hybrid_default_k7_o03_posttight_v2`
    - `hybrid_default_k7_o03`
- GT source to be used for all next mask experiments:
  - `D:\Research\Image Analysis\Nanozoomer scans\20250327 rat myelin quantification\Tissue&Masks\test`

This plan is only for **crop-level myelin/Gallyas tissue mask generation**.
It does not revisit bbox proposal except where bbox quality affects crop validity.

## Reference-Pipeline Comparison And Local Status

The external reference pipeline is broader than the current local project. It
combines:

- whole-slide IO and normalization
- atlas registration and atlas-space priors
- tissue masking
- region / layer / nucleus / fiber segmentation
- QC automation
- downstream quantitative export

The current local project is **strong in GT-backed crop extraction, bbox
proposal, GUI review, metadata traceability, and crop-level tissue mask
evaluation**, but it is **not yet a full atlas-registered, multi-task histology
analysis platform**.

### Practical completion estimate

If the external reference workflow is split into 8 major modules, the local
project status is roughly:

- fully implemented: `4`
- partially implemented: `2`
- not implemented yet: `2`

That is approximately:

- `55%–60%` of the engineering workflow skeleton
- `30%–40%` of the full long-range research roadmap

This is not a failure signal. It mostly means the current project has focused
on the correct early bottlenecks first:

- robust NDPI reading
- reproducible bbox generation
- human-in-the-loop section review
- GT-backed mask optimization

### What is already implemented well

1. **Whole-slide reading and fallback**
- `OpenSlide` path works for readable slides
- `tifffile_proxy` fallback exists for difficult Hamamatsu NDPI cases
- low-resolution review proxies and cache paths already exist

2. **Stain-aware whole-slide proposal**
- `nissl` and `gallyas` already use different bbox / crop logic
- myelin bbox is no longer a naïve uniform-pad baseline
- proposal, crop, and GT can all be mapped to the same slide `level0` space

3. **GT-backed evaluation protocol**
- every serious bbox/mask experiment now uses only GT sets with spatial metadata
- metrics already include:
  - Dice / IoU / precision / recall
  - BF / HD95 / ASSD
  - local top/middle/bottom and left/center/right recall
  - bbox target coverage / non-target overlap

4. **Human-in-the-loop GUI**
- stepwise GUI exists
- proposal edit, crop review, tissue/artifact layers, revision save, export
- metadata is per-section and traceable back to slide space

5. **Cache and efficiency engineering**
- auto-mask cache
- proposal/proxy cache
- timing harness on Windows-side GUI-equivalent environment
- low-resolution working crops with mapped export

### What is partially implemented

1. **QC automation**
- current project has strong experiment-side QC images and metrics
- but it does not yet have a unified atlas-aware production QC layer comparable
  to the external pipeline vision

2. **Cross-stain / downstream readiness**
- metadata, pairing concepts, and export structure are already present
- but Nissl↔Gallyas registration and model-training datasets are not yet
  integrated into one complete production pipeline

### What is not implemented yet

1. **Atlas registration and atlas-prior segmentation**
- no current integration of:
  - DeepSlice
  - QuickNII
  - VisuAlign
  - QCAlign
- no Allen CCF-driven weak supervision in the current mask/bbox runtime

2. **Cell-level / layer-level / tract-level specialized models**
- no Cellpose / StarDist path for Nissl instance segmentation yet
- no nnU-Net / DeepLabv3+ / Attention U-Net benchmarking yet
- no fiber orientation / density field modeling for Gallyas yet

## Borrowable Ideas From The External Reference

The following ideas are the most useful for this project. They are listed in
priority order, not in reference-paper order.

### 1. Atlas-space priors for region-aware workflows

Most relevant long-term borrow:

- use DeepSlice / QuickNII / VisuAlign to place Nissl and Gallyas into Allen
  CCF space
- then use atlas labels as weak priors for:
  - section QC
  - region-level statistics
  - later layer / tract / nucleus workflows

Why this matters:

- the current project is already strong at crop/mask correctness
- the next large gain is likely to come from putting those masks into a common
  anatomical space, not from endlessly tuning one more morphology kernel

### 2. Weak-supervision tooling such as ilastik / QuPath

Most relevant near-term borrow:

- ilastik-style pixel classification can be used to generate:
  - tissue vs background
  - artifact vs tissue
  - support priors for hard Gallyas edge cases
- QuPath can help for fast WSI-scale ROI authoring and review

Why this matters:

- several current myelin failures still behave like support/candidate problems
- a weak-supervision support map may be more useful than more post hoc mask
  cleanup rules

### 3. Instance models for Nissl, but not yet for Gallyas fibers

Useful distinction from the external reference:

- Cellpose / StarDist are likely relevant for future **Nissl cell-body**
  workflows
- they are not the right next move for the current Gallyas tissue-mask problem

Why this matters:

- it prevents the project from jumping too early into the wrong model family
- Gallyas current bottleneck is still tissue/support boundary definition, not
  cell-instance segmentation

### 4. Fiber orientation / density representation for Gallyas

Important conceptual borrow:

- do not assume future Gallyas work must become instance segmentation
- for many myelin tasks, a better formulation is:
  - tissue mask
  - myelin density / coverage
  - orientation field

Why this matters:

- it aligns better with the stain physics and structure continuity
- it is a better long-term direction than trying to instance-segment fibers

### 5. Stronger standardized QC and report automation

The external workflow rightly emphasizes:

- data QC
- registration QC
- segmentation QC
- region-wise summaries

Current project should borrow this as a packaging discipline:

- make each experimental branch emit a fixed QC/report bundle
- make current GUI export and experiment outputs converge toward one report
  structure

## What Should Enter The Next-Step Plan

Based on the comparison above, the next-step plan should stay focused.

### Near-term priorities

1. keep current GT-backed bbox + mask optimization workflow as the core
2. finish selecting the best myelin mask strategy under the current GUI bbox
3. continue only experiments that change **candidate/support quality**, not
   post hoc patching that repeatedly fails to beat M3

### Mid-term additions worth planning now

1. add one weak-supervision baseline for support generation
   - likely ilastik-like or entropy/residual/objectness support map generation
2. add one atlas-registration baseline path
   - DeepSlice first
   - QuickNII/VisuAlign only when needed
3. add standardized QC bundles shared by:
   - GUI exports
   - experiment outputs
   - future registration/training datasets

### Long-term items to defer

1. Cellpose / StarDist benchmarking for Nissl instance segmentation
2. fiber orientation / density estimation for Gallyas
3. nnU-Net / DeepLabv3+ / attention U-Net benchmarking

These are worthwhile, but not before the current crop-level support/mask
problem is considered stable.

## Current Status

### What is already working

- bbox proposal is now good enough that mask experiments should be run on GT-backed crops rather than on obviously truncated crops
- the hybrid family is clearly better than the old GUI simple/contextual paths
- the current best practical baseline is:
  - `hybrid_tightcand_k7_o03`

### Current best baseline snapshot

From `MYELIN_MASK_GT_STRATEGY_SEARCH_20260311.md`:

- `hybrid_tightcand_k7_o03`
  - Dice `0.7504`
  - IoU `0.6902`
  - precision `0.6935`
  - recall `0.8294`
  - BF64 `0.5709`
  - HD95 `1366.2 px`
  - FP/GT `0.3584`
  - predicted/GT area `1.1878`

Interpretation:

- this is the best current compromise
- but it is still not good enough on:
  - boundary fit
  - leakage control
  - collapse-risk hard cases

## What The Deep-Research Report Adds

The report reframes the problem as a combination of:

1. foreground separability
2. target component selection
3. boundary localization

That matches the actual failure modes seen in current experiments:

- **boundary drift / outer spill**
  - masks sit one ring outside GT
- **collapse**
  - mask shrinks to a small deep-stained core
- **bridge / appendage leakage**
  - weak links pull in background or nearby structures
- **two-sided structure loss**
  - only one side survives after component selection
- **structured artifact confusion**
  - not random noise, but glass edge / strips / deep marks / nearby tissue

The report's main recommendation is not “just tune thresholds harder”, but:

- improve the representation used to build candidate/support masks
- separate high-confidence core from weaker fringe explicitly
- replace aggressive morphology with reconstruction-style constrained growth
- treat component-set selection and boundary refinement as separate steps

## Comparison To Current Project Strategy

### Current project strategy

Current hybrid family roughly does this:

1. residual-based or simple high-recall candidate
2. tighter candidate trimming
3. crop-center-derived structural prior
4. hybrid reconstruction inside candidate
5. slight conservative refine

This already solves:

- total over-expansion of `legacy_simple`
- catastrophic single-core collapse of crop-center-only methods

But it still has two structural weaknesses:

1. candidate/support generation is still too brightness-driven
   - weak outer tissue can be unstable
   - structured background can still enter candidate space

2. boundary control is still too morphology-driven
   - closing/opening and post-tightening do not explicitly optimize boundary placement

### Deep-research directions that align best with current codebase

The report suggests many methods, but the best near-term fit for the current project is:

1. **physical-scale-aware reconstruction morphology**
2. **entropy + intensity / residual joint candidate**
3. **hysteresis + constrained propagation**
4. **multi-component set selection**
5. optional later:
   - graph-based segmentation with automatic seeds
   - narrow-band contour refinement

These are attractive because they can be added incrementally on top of the current hybrid code, without replacing the whole mask pipeline at once.

## Next Experimental Goals

The next round should optimize in this order:

1. reduce obvious false-positive spill outside GT
2. reduce collapse on hard cases
3. improve boundary metrics without reintroducing leakage

More explicitly:

- primary goal:
  - better BF64 / HD95 / ASSD while keeping Dice from dropping
- secondary goal:
  - reduce `FP / GT area`
- hard constraint:
  - do not increase the number of near-zero masks / collapse cases

## Hard Cases To Always Track

These should be reported separately in every run:

- `2503_144`
  - difficult lower / boundary footprint
- `2507_66`
  - classic collapse-risk sample
- `2504_161`
- `2504_47`
- `2504_5`
- `2504_185`

These should be used as “easy / stable references”:

- `2507_42`
- `2507_48`
- `2507_54`

## Metrics To Prioritize

Do not rank methods by Dice alone.

Required metrics:

- overlap:
  - Dice
  - IoU
  - precision
  - recall
- boundary:
  - BF@32
  - BF@64
  - ASSD
  - HD95
- leakage:
  - FP / GT area
  - predicted / GT area
  - border-touch leakage
- local performance:
  - top / middle / bottom recall
  - left / center / right recall
  - boundary / core recall

Recommended ranking priority for this phase:

1. collapse count / catastrophic failures
2. BF64 + HD95 + ASSD
3. FP / GT area
4. Dice / IoU

## Proposed Next Experiments

### Experiment M1: Reconstruction Morphology In Physical Scale

Goal:

- reduce “one-ring outer spill”
- make postprocessing more stable across crop sizes

Core change:

- replace coarse closing/opening with:
  - opening by reconstruction
  - closing by reconstruction
- parameterize bridge width / component thresholds in physical scale where available, otherwise normalize by crop size

Compare against:

- `hybrid_tightcand_k7_o03`
- `hybrid_default_k7_o03_posttight_v2`

Parameters to sweep:

- reconstruction marker erosion radius
- reconstruction closing radius
- minimum surviving component area

Success criteria:

- FP / GT area down
- BF64 up
- no increase in collapse cases

Why first:

- lowest-risk replacement of the current morphology-heavy post-tightening
- very likely to help boundary overgrowth without rewriting candidate logic

### Experiment M2: Entropy + Residual Joint Candidate

Goal:

- recover weak tissue edge without dropping threshold too low
- reduce dependence on deep-stained structures only

Core change:

- build candidate/support from:
  - residual-based signal
  - local entropy / local texture signal
- candidate = union or weighted combination
- keep current hybrid reconstruction downstream

Compare against:

- current `hybrid_tightcand_k7_o03`
- M1 winner

Parameters to sweep:

- entropy window size
- entropy threshold quantile
- residual / entropy combination rule
- candidate tightening strength

Success criteria:

- boundary-band FN down
- `2507_66` and weak-edge samples do not collapse
- FP / GT area does not jump sharply

Why second:

- this targets the weak-edge problem directly
- it is the cleanest way to test the report's “better representation first” advice

### Experiment M3: Hysteresis Core + Support-Constrained Reconstruction

Goal:

- stabilize candidate growth
- reduce both collapse and leak

Core change:

- explicit high-threshold core
- explicit lower-threshold support
- final foreground = morphological reconstruction from core inside support
- optionally add crop-edge background barrier

Compare against:

- current hybrid baseline
- best from M1 / M2

Parameters to sweep:

- core threshold quantile
- support threshold quantile
- support scale factor
- barrier width

Success criteria:

- fewer collapse cases on `2507_66`, `2504_161`, `2504_47`
- no increase in remote FP
- BF64 or ASSD improves

Why third:

- this is the first experiment that explicitly separates core and fringe
- it operationalizes the deep-research report's main recommendation

### Experiment M4: Multi-Component Set Selection

Goal:

- stop losing one side when tissue splits into multiple components
- keep valid multi-part tissue without letting nearby junk in

Core change:

- after bridge-cutting / reconstruction, do not force a single winner
- allow top-K components
- score components by:
  - reachability from core
  - area
  - compactness
  - distance to target support
  - border-attachment penalty

Compare against:

- best from M1-M3

Parameters to sweep:

- K
- minimum area ratio
- border penalty
- component score weights

Success criteria:

- dual-structure cases improve
- no increase in neighbor/background leakage

Why fourth:

- this directly attacks the “only one side survives” failure mode
- but should only be tested after support generation is stabilized

## Optional Second-Tier Experiments

These should not be first.

### Experiment M5: Automatic-Seed Graph Segmentation

Candidate forms:

- random walker
- graph cut / GrabCut-like energy with automatic seeds

Reason to defer:

- potentially strong against structured artifacts
- but more implementation cost and harder debugging

### Experiment M6: Narrow-Band Boundary Refinement

Candidate forms:

- region-based active contour
- morphological snakes

Reason to defer:

- likely improves BF / ASSD
- but should be applied after candidate/support/component logic is already stable

## Execution Order

Recommended order:

1. M1
2. M2
3. M3
4. M4
5. only then consider M5 / M6

Practical iteration style:

- first run at `0.5x` on the GT set
- keep only clearly promising variants
- then run the top 2 to 3 at `1.0x` on:
  - all GT sections
  - plus the named hard cases

## Stop / Promote Rules

Promote a method to “next-best candidate” only if all are true:

- no new catastrophic collapse cases
- BF64 improves or stays effectively tied
- HD95 and/or ASSD improve
- FP / GT area decreases or stays effectively tied

Do not promote a method if:

- Dice goes up only because the mask got looser
- BF64 improves but collapse count rises
- one or two hard cases improve at the cost of many stable cases degrading

## Recommended Immediate Next Step

Start with:

- `Experiment M1`
- `Experiment M2`

Reason:

- they are the highest-yield / lowest-risk combination
- both are fully compatible with the current hybrid pipeline
- together they test the two strongest ideas from the deep-research report:
  - better candidate/support representation
  - less destructive morphology

If one of them clearly beats the current hybrid baseline, then move to:

- `Experiment M3`

and only after that:

- `Experiment M4`

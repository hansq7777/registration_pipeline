# Histology Cross-Stain Adjacent Section Registration Next Experiment Plan

Date: 2026-03-27

Branch:
- `experiment/histology_registration_preproc_20260326`

Source note:
- This document summarizes the ChatGPT Deep Research report returned on 2026-03-27 for `Nissl <-> Gallyas / myelin` adjacent-section 2D registration.
- The original literature citations remain in that source report; this file converts the report into an execution plan for this repository.

## Executive Summary

Current evidence points to a consistent failure mode:
- `input-only` often outperforms `rigid`, `affine`, and `syn`
- the current input state is already geometrically strong because scale normalization, tissue support cropping, half-brain grouping, and common-canvas centering remove most easy rigid error
- later optimization stages still maximize a mostly intensity-statistical objective, which is not well aligned with the real task objective for cross-stain adjacent sections
- therefore the optimizer can move a good geometric initialization toward a better intensity score but a worse anatomical alignment

The most important interpretation changes are:
- stop treating "more intensity preprocessing" as the main experimental axis
- move the experiment axis to `objective function`, `transform model`, and `acceptance policy`
- keep `identity / input-only` as an explicit candidate at every stage
- make every later stage pass a monotonic gate before it is accepted
- add at least a small structure-based validation set so we can distinguish "boundary got slightly worse but anatomy got better" from true geometric failure

## Main Conclusions From The Research Summary

### 1. Why input-only can beat rigid/affine/syn

The likely reason is objective mismatch, not lack of optimizer power:
- current preprocessing already solves much of the coarse geometry
- MI/CC can improve without real anatomical improvement
- nonlinear registration can overfit stain-specific texture, local defects, or repeated patterns
- high-DOF refinement is especially risky once baseline Dice is already high

### 2. Why MI/CC and Dice/HD95 decouple

The report's explanation can be operationalized as:
- MI measures statistical dependence, not direct geometric correctness
- cross-stain appearance is not stable enough for raw intensity similarity to be a reliable geometry target
- once the initial pose is already good, optimizer headroom on boundary overlap is small, so even slight texture chasing can cause net geometric regression

### 3. What should change in the next round

The next round should prioritize:
- shape-driven or structure-driven objectives
- rigid or similarity transforms before any affine or nonlinear step
- explicit "accept only if geometry improves" gating
- a robustness view of evaluation, not just mean score deltas

### 4. What should be deprioritized

These are no longer first-line directions:
- expanding blur / CLAHE / histogram preprocessing grids
- tuning default MI-driven `rigid -> affine -> syn` pipelines more aggressively
- treating stain normalization as a likely primary fix for `Nissl <-> myelin`

## Repository Touchpoints For Deployment

Likely implementation files in the current codebase:
- `histology/gui_mvp/hitl_gui/application/pair_registration.py`
- `histology/tools/run_usable_pair_registration_batch.py`
- `histology/tools/prepare_registration_preprocessing_and_eval.py`
- `histology/tools/compare_affine_profiles_on_selected_pairs.py`

Likely outputs to keep standardized:
- per-run `run_manifest.json`
- per-run `storyboard.png`
- aggregate batch summary JSON / Markdown
- robustness summary with success-rate and regression-tail statistics

## Required Evaluation Policy Before New Method Comparisons

Before comparing new registration methods, freeze a common evaluation policy.

### Required benchmark policy

- fixed benchmark set: current 131 usable units
- fixed geometry and current preprocessing entry path unless the method explicitly changes representation
- fixed reporting:
  - input-only metrics
  - best accepted stage metrics
  - per-stage delta vs previous accepted state
  - success rate
  - mean / median improvement
  - regression tail, especially bad-case HD95 degradation

### Required acceptance policy

Every candidate stage should compete against the current best state, not automatically replace it.

Recommended rule:
- baseline candidate is always `input-only`
- a new stage is accepted only if it improves geometry according to the agreed gate
- if it fails the gate, keep the previous best transform and continue from that best state only if the experiment design explicitly allows it

Decision still needed from the user:
- exact acceptance threshold

Suggested default for first implementation:
- accept if `Dice` improves by at least a small epsilon and `HD95` does not materially worsen
- or accept if `HD95` improves materially and `Dice` does not materially worsen
- keep threshold values conservative and explicit in config, not hard-coded in prose

## Next Experiment Program

## Phase 0: Build The Gated Benchmark Harness

Goal:
- convert the current batch experiments from "run fixed chain" to "evaluate candidate transforms under a monotonic gate"

Steps:
1. Freeze the benchmark unit list and evaluation outputs.
2. Add stage-level acceptance logic and best-state bookkeeping.
3. Add robustness reporting:
   - success rate
   - mean and median delta
   - worst-case and 95th percentile regressions
4. Save accepted-vs-rejected decisions into manifests and summary tables.
5. Keep storyboard generation aligned with the accepted state.

Codex can deploy:
- yes
- update batch harnesses and manifest schema
- add accepted / rejected stage tracking
- add summary exports and robustness tables

Needs user decision or manual data handling:
- confirm the acceptance rule and thresholds
- confirm whether "continue from last accepted stage" or "independent candidate evaluation from the same baseline" is preferred

## Phase 1: Shape-Driven Mask Similarity Baseline

Goal:
- test whether a geometry-only objective is more reliable than MI/CC for coarse alignment

Method:
- use tissue masks, boundaries, or distance transforms
- optimize `translation + rotation`, or `translation + rotation + isotropic scale`
- do not enable shear in the first round
- optionally add a small number of auxiliary shape constraints such as midline or ventricle masks if available

Steps:
1. Implement mask-boundary or distance-transform matching at current working scale.
2. Run identity vs mask-rigid vs mask-similarity on the full benchmark set.
3. Compare success rate against current MI-based rigid.
4. Review symmetry-failure cases and decide whether auxiliary constraints are needed.

Codex can deploy:
- yes
- implement the objective and optimizer wrapper
- integrate it into the existing batch harness
- generate batch summaries and storyboards

Needs user decision or manual data handling:
- decide whether first round should be `rigid` only or `similarity`
- decide whether auxiliary masks should be included immediately or only after baseline results
- if midline / ventricle masks are not already available, they require manual curation or a separate extraction pipeline

## Phase 2: NGF-Driven Rigid Or Similarity Refinement

Goal:
- replace raw intensity statistics with a structure-sensitive cross-stain metric

Method:
- use NGF or an engineering-equivalent normalized gradient metric
- compute only inside trusted tissue support
- exclude artifact regions whenever masks are available
- keep transform low-DOF in the first round

Steps:
1. Decide whether to implement NGF directly or use an existing metric backend such as elastix if already available locally.
2. Add NGF-rigid first.
3. If rigid improves reliably, add NGF-similarity.
4. Compare against Phase 1 on the same benchmark set under the same gate.

Codex can deploy:
- yes, with one caveat
- I can implement direct NGF or wire an existing backend into the harness if dependencies are already available
- I can add mask support and artifact exclusion to the metric computation

Needs user decision or manual data handling:
- choose whether to stay inside the current ANTs-based path or allow a new backend
- approve any dependency change if a new backend is required
- verify that artifact masks are complete enough to be trusted as metric exclusions

## Phase 3: MIND-Based Rigid Or Similarity Registration

Goal:
- test a descriptor designed for cross-modality structural matching rather than raw intensity similarity

Method:
- compute MIND descriptors at working scale
- optimize rigid or similarity transform on descriptor maps using SSD or NCC
- keep gating identical to Phase 1 and Phase 2

Steps:
1. Implement or integrate 2D MIND descriptor generation.
2. Benchmark MIND-rigid on the full set.
3. Add MIND-similarity only if rigid is stable.
4. Compare success rate and regression tail against mask-driven and NGF-driven methods.

Codex can deploy:
- yes
- descriptor generation
- optimizer wrapper
- batch integration
- reporting

Needs user decision or manual data handling:
- confirm compute budget if descriptor extraction becomes meaningfully slower
- decide whether MIND should be tested on all 131 units immediately or after a smaller pilot

## Phase 4: Weak Affine Only After A Clear Rigid Win

Goal:
- absorb mild global stretch from sectioning or mounting without reopening the full failure surface of unconstrained affine

Method:
- only enable weak affine after a method has already demonstrated a stable rigid or similarity gain
- prefer heavily regularized affine with constrained anisotropy and shear

Steps:
1. Pick the best-performing rigid or similarity candidate from Phases 1 to 3.
2. Add weak affine around that candidate only.
3. Compare "best rigid" vs "best rigid + weak affine" under the same gate.

Codex can deploy:
- yes
- implement constrained parameterization or an explicit affine penalty path
- batch and summary integration

Needs user decision or manual data handling:
- decide whether weak affine is worth testing at all if rigid already captures most gains
- decide acceptable shear / anisotropy range

## Phase 5: Triggered Nonlinear Only On Hard Cases

Goal:
- keep nonlinear registration as a rescue path, not the default path

Method:
- define a hard subset such as the worst 20 to 30 units by input Dice or HD95
- only run nonlinear after a low-DOF method already improves geometry
- use strong regularization and cross-stain-safe data terms, not default SyN settings

Steps:
1. Define the hard set.
2. Select the best low-DOF front-end from earlier phases.
3. Add a strongly regularized nonlinear candidate with explicit deformation limits.
4. Evaluate only on the hard set first.

Codex can deploy:
- yes
- hard-set selection scripts
- nonlinear experiment harness
- restricted deformation bookkeeping
- reporting and failure review outputs

Needs user decision or manual data handling:
- define the hard-set criterion
- choose whether nonlinear should be NGF-based, MIND-based, or another structure-driven formulation
- review visually whether any apparent structural gain justifies slight boundary loss

## Phase 6: Small Structure-Based Validation Set

Goal:
- add a direct structure-alignment signal so boundary overlap is not the only judge

Method:
- build a 15 to 20 pair mini-set with 10 to 20 cross-stain landmarks per pair
- use stable structures such as ventricle corners, hippocampal bends, major white-matter boundaries, large vessel holes, or other repeatably visible features

Steps:
1. Select representative easy, medium, and hard pairs.
2. Define the landmark schema and export format.
3. Annotate landmarks.
4. Add TRE / relative TRE reporting.
5. Compare whether any method that slightly loses on mask boundary actually wins structurally.

Codex can deploy:
- partially
- I can create the annotation template, validator, scorer, and summary plots
- I can integrate TRE into the batch reports once the points exist

Needs user decision or manual data handling:
- landmark annotation itself is human work
- deciding which anatomical structures count as reliable cross-stain correspondences is a human judgment call
- reviewing disputed or ambiguous points is also human work

## What Can Be Deployed By Codex vs What Needs Human Input

### Codex can deploy directly in this repo

- gated batch harness and best-state bookkeeping
- new objective functions and wrappers
- mask-driven rigid / similarity experiments
- NGF experiment plumbing
- MIND experiment plumbing
- weak affine constraints
- hard-set selection and triggered nonlinear harness
- summary JSON / Markdown exports
- robustness tables and risk-tail summaries
- manifest updates and storyboard updates
- landmark CSV schema, validation, and TRE scoring once annotations exist

### Human decisions or manual data work are still required

- final acceptance rule and threshold policy
- whether to prefer rigid or similarity in each phase
- whether to allow a new registration backend if needed
- whether existing artifact masks are trusted enough for metric exclusion
- whether auxiliary masks such as ventricle or midline should be added
- landmark definition and manual annotation
- interpretation of "boundary slightly worse but anatomy better" edge cases
- final go / no-go decisions after each phase

## Priority Order For Immediate Next Experiments

Important:
- the gating harness is not itself a registration method, but it should be treated as the mandatory first infrastructure change before trusting any new method comparison

### Priority 0: Mandatory infrastructure

1. Build monotonic gating and robustness reporting.

### Priority 1: First methods to try

1. Mask-driven rigid / similarity registration.
2. NGF-driven rigid refinement with mask and artifact support.

### Priority 2: Second-line methods

1. MIND-rigid or MIND-similarity.
2. Weak affine only after one of the above clearly wins.

### Priority 3: Selective high-risk methods

1. Triggered nonlinear on the hard set only.

### Start in parallel but do not block Phase 1

1. Build the small landmark validation set.

## Recommended Immediate Work Order

If we want the most informative next iteration with controlled effort, the recommended order is:

1. Implement the gated benchmark harness.
2. Run mask-driven rigid / similarity on all 131 usable units.
3. Run NGF-rigid on the same set under the same gate.
4. Start a small landmark annotation set in parallel.
5. Only then decide whether MIND or weak affine is the better next branch.

## Decision Criteria After The First New Round

Advance a method only if it improves at least one of these without unacceptable regression risk:
- success rate
- median Dice gain
- median HD95 reduction
- bad-case regression tail
- landmark TRE on the mini validation set once available

Stop or deprioritize a method if:
- mean gain is small but regression tail is large
- it improves MI/CC but not geometric metrics
- it requires too much tuning relative to its benchmark gain

## One-Sentence Takeaway

The next round should shift from "more intensity preprocessing and default multistage registration" to "shape/structure-aware low-DOF candidates under monotonic gating, with identity kept as a first-class baseline and nonlinear reserved for explicitly hard cases."

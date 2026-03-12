# Windows Timing Harness Reference v1

## Purpose

This reference locks the correct way to benchmark the histology GUI data path.

The main goal is to avoid repeating a common mistake:

- measuring GUI-related NDPI performance from WSL/Linux paths such as `/mnt/d/...`
- then drawing conclusions about a GUI that actually runs under Windows Python

For the current project, GUI timing must be treated as a **Windows-side measurement problem**.

## Canonical Measurement Environment

Use the same environment family as the GUI launcher:

- launcher: [run_timing_harness.bat](/mnt/c/work/registration_pipeline/histology/gui_mvp/run_timing_harness.bat)
- script: [timing_harness_windows.py](/mnt/c/work/registration_pipeline/histology/gui_mvp/timing_harness_windows.py)
- GUI launcher for reference: [start_gui.bat](/mnt/c/work/registration_pipeline/histology/gui_mvp/start_gui.bat)

This ensures:

- Windows Python
- the same dependency stack used by the GUI
- the same `PYTHONPATH` layout as the GUI

## What Must Be Measured Separately

Never collapse these into one “open slide time”.

Measure separately:

- `cold start`
  - first `load_slide_bundle()`
  - first `overview proposal`
  - first preview crop
  - first section crop
  - first mask generation
- `warm start`
  - second `load_slide_bundle()` in the same run
  - second `overview proposal`
  - first crop accesses in a new session after the first open has already warmed OS cache
- `cache hit`
  - repeated `preview crop` on the same proposal in the same `SlideSessionCache`
  - repeated `section crop` on the same proposal in the same `SlideSessionCache`

## Why This Split Matters

These timings answer different questions:

- `cold start`: what the user feels on the first open
- `warm start`: what repeated same-slide work costs once the file is no longer truly cold
- `cache hit`: whether GUI cache design is working

If these are mixed together, repeated runs can hide expensive first-open behavior.

## Current Anti-Patterns To Avoid

- Do not benchmark GUI performance from WSL and assume it matches Windows GUI behavior.
- Do not reopen NDPI once per section when the GUI can hold one slide handle per session.
- Do not benchmark at the highest crop level first.
- Do not write full qualitative overlays during tuning unless they are specifically needed.
- Do not mix `IO benchmarking` and `algorithm tuning` in the same heavy run by default.

## Recommended Benchmark Workflow

1. Run Windows timing harness on a representative `OpenSlide`-readable slide.
2. Record:
   - `load NDPI`
   - `overview proposal`
   - `preview crop direct`
   - `preview crop cache miss`
   - `preview crop cache hit`
   - `section crop cache miss`
   - `section crop cache hit`
   - `mask generation`
3. Run a small quality-speed tradeoff check on 2 to 4 GT sections.
4. Compare lower crop level vs higher crop level:
   - speed
   - Dice / IoU
   - contour metrics
5. Only after that, launch larger-scale experiments.

## Current Efficiency Strategy

When accuracy is the priority but repeated compute waste must be minimized:

- keep `load_slide_bundle()` at slide scope, not section scope
- keep one persistent slide handle per session
- route repeated crop requests through `SlideSessionCache`
- use lower crop levels for tuning and profiling
- validate the final candidate method on a small higher-resolution GT subset before scaling out

## Immediate Optimization Priorities

These are the safest current optimizations with low accuracy risk:

- Reuse one open slide handle for all proposal preview crops in a session
- Reuse one open slide handle for section crops in a review pass
- Cache preview crops and section crops by `(label, bbox, crop_level, mirror)`
- Use low-resolution crops for method screening
- Separate “metrics only” runs from “save many overlays” runs

## Update Policy

Whenever timing work is revisited:

- run the Windows harness first
- write JSON + markdown summary to `REVIEW`
- if a new optimization is adopted, append its rationale and observed effect to this reference

## Observed Baseline: 2026-03-10 Run2

Reference run:

- output dir: `C:\Users\Siqi\Desktop\REVIEW\20260310_windows_timing_harness_run2`
- harness output:
  - `timing_results.json`
  - `timing_summary.md`

Observed on `gallyas_2502_42-72.ndpi` under Windows Python:

- cold `load NDPI`: about `2.95s`
- warm `load NDPI`: about `2.50s`
- `overview proposal`: about `0.58-0.60s`
- `preview crop` cache miss at level 4: about `0.20s`
- `preview crop` cache hit at level 4: about `0.002s`
- `section crop` cache miss at level 3: about `0.49-0.53s`
- `section crop` cache hit at level 3: about `0.014s`
- `mask generation` at level 3: about `6.2-6.6s`

Interpretation:

- repeated `load NDPI` is still expensive enough that slide-scope reuse is mandatory
- `SlideSessionCache` is highly effective for repeated crop access
- crop retrieval is not the dominant cost after cache is in place
- current dominant per-section cost is `mask generation`, especially at higher crop level

Quality-speed comparison from the same Windows run:

- level 3:
  - mean crop time about `1.35s`
  - mean mask time about `11.51s`
  - better Dice / IoU / contour metrics than level 4 on the tested sections
- level 4:
  - mean crop time about `0.26s`
  - mean mask time about `1.16s`
  - substantially worse mask quality on the tested sections

Operational takeaway:

- use lower crop levels for profiling, UI responsiveness checks, and method screening
- do not substitute level 4 for final mask validation without a GT-backed quality check
- reserve level 3 or higher for final-quality mask generation and targeted validation

Important current caveat:

- the GUI crop path used in this benchmark still follows the GUI-side crop extraction logic
- this is not yet the same as the newer coverage-first bbox strategy tested in the tool experiments
- therefore, GUI timing is valid here, but crop coverage numbers should be interpreted as the **current GUI baseline**, not the newest tool-side bbox baseline

## Update: GUI Crop Path Switched To Coverage-First BBox

After the GUI crop path was updated to use the tool-side coverage-first bbox logic:

- review run:
  - `C:\Users\Siqi\Desktop\REVIEW\20260310_windows_timing_harness_run3_after_bbox`

Observed effect relative to the earlier GUI baseline run:

- mean crop coverage recall on the tested GT sections increased from about `0.5055` to about `0.9670`
- crop and mask workloads increased sharply because the actual crop became much larger
- level 3:
  - mean crop time rose from about `1.35s` to about `2.70s`
  - mean mask time rose from about `11.51s` to about `33.47s`
- level 4:
  - mean crop time rose from about `0.26s` to about `1.20s`
  - mean mask time rose from about `1.16s` to about `5.98s`

Interpretation:

- the new bbox strategy is doing what it was supposed to do: it prevents severe crop truncation
- but the larger crop shifts the performance bottleneck even more strongly into mask generation

Current recommendation after this update:

- keep the new coverage-first bbox path for correctness
- do not roll back to the old 8% pad crop logic just for speed
- instead, optimize the mask stage and use staged processing:
  - draft / fast review at lower crop level
  - final validation and export at higher crop level

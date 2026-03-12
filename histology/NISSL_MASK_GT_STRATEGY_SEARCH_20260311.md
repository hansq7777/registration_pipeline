# Nissl Mask GT Strategy Search 2026-03-11

## Scope

- GT source only:
  - `D:\Research\Image Analysis\Nanozoomer scans\20250424 Nissl cytoarchitectonic counterpart\Tissue&Masks\test`
- Evaluated sections:
  - `2507_73`
  - `2507_79`
  - `2507_85`
  - `2507_91`
  - `2507_97`
  - `2507_103`

All inputs include crop and mask metadata and can be traced back to original slide space, but this search itself is crop-level because Nissl bbox coverage is already stable.

## Goal

Systematically compare current GUI Nissl mask extraction, current experiment-script Nissl mask extraction, and parameterized tighter variants.

Main objectives:

- keep boundary close to GT
- suppress leakage beyond tissue edge
- avoid very small or collapsed masks
- explain why a method works better on certain local regions or sections

## Evaluation Design

### Stage 1: fast search

- script:
  - `tools/search_nissl_mask_strategies.py`
- working scale:
  - `0.5x`
- methods compared:
  - `gui_legacy_simple`
  - `gui_latest_contextual`
  - `exp_baseline_v1`
  - `exp_baseline_posttight_v1`
  - `exp_baseline_posttight_v2`
  - `exp_soft_support_mgac`
  - `param_default_match`
  - `param_balanced_v1`
  - `param_tight_v1`
  - `param_tight_v2`
  - `param_tight_v3`

Metrics:

- overlap:
  - Dice
  - IoU
  - precision
  - recall
- boundary:
  - boundary F1 @ 32 px
  - boundary F1 @ 64 px
  - ASSD
  - HD95
- leakage:
  - FP / GT area
  - border FP / GT area
  - predicted / GT area ratio
- local recall:
  - top / middle / bottom
  - left / center / right
  - boundary / core

Stage 1 output:

- `/mnt/c/Users/Siqi/Desktop/REVIEW/20260311_nissl_mask_strategy_search_2507_scale05`

### Stage 2: full-scale confirmation

Only the representative top candidates and current GUI baselines were re-run at `1.0x`.

Methods:

- `exp_baseline_v1`
- `exp_baseline_posttight_v2`
- `param_tight_v3`
- `gui_latest_contextual`
- `gui_legacy_simple`
- `exp_soft_support_mgac`

Stage 2 output:

- `/mnt/c/Users/Siqi/Desktop/REVIEW/20260311_nissl_mask_strategy_confirm_2507_scale10`

## Main Result

The current GUI Nissl path was clearly weaker than the experiment-script baseline.

Before correction:

- `gui_latest_contextual`
  - Dice `0.9662`
  - boundary F1 @ 64 `0.8451`
  - HD95 `550.1 px`
  - FP / GT area `0.0406`
- `gui_legacy_simple`
  - Dice `0.9665`
  - boundary F1 @ 64 `0.8476`
  - HD95 `549.5 px`
  - FP / GT area `0.0433`

By contrast, the experiment baseline family was much better:

- `exp_baseline_v1`
  - Dice `0.9862`
  - boundary F1 @ 64 `0.9784`
  - HD95 `49.4 px`
  - FP / GT area `0.0102`

So the main correction was not inventing a new Nissl method from scratch, but aligning the GUI Nissl path to the stronger experiment baseline.

## Best Candidates

### Candidate 1: `exp_baseline_v1`

Best overall full-scale strategy.

- Dice: `0.9862`
- IoU: `0.9729`
- precision: `0.9898`
- recall: `0.9828`
- boundary F1 @ 64: `0.9784`
- HD95: `49.37 px`
- FP / GT area: `0.0102`
- predicted / GT area ratio: `0.9930`

Why it is best:

- strongest full-scale composite score
- best balance of overlap, boundary accuracy, and leakage control
- slightly wider than the tightened variants, but the extra width is small and still close to GT
- strongest boundary/core retention among the top family

### Candidate 2: `exp_baseline_posttight_v2`

Best slightly more conservative alternative.

- Dice: `0.9863`
- IoU: `0.9731`
- precision: `0.9909`
- recall: `0.9819`
- boundary F1 @ 64: `0.9776`
- HD95: `50.22 px`
- FP / GT area: `0.0091`
- predicted / GT area ratio: `0.9910`

Why it is useful:

- trims leakage slightly more than the baseline
- keeps area ratio a bit tighter
- only a very small recall/boundary penalty

Best use case:

- when you want a slightly more conservative automatic mask before manual review

### Candidate 3: `param_tight_v3`

Best precision-heavy tuned variant.

- Dice: `0.9863`
- IoU: `0.9730`
- precision: `0.9918`
- recall: `0.9809`
- boundary F1 @ 64: `0.9739`
- HD95: `51.50 px`
- FP / GT area: `0.0082`
- predicted / GT area ratio: `0.9891`

Why it is useful:

- lowest leakage among the strong non-MGAC candidates
- best if the priority is to avoid slight outer spill even at the cost of a small recall drop

## Why Some Other Methods Lost

### `gui_legacy_simple` and old `gui_latest_contextual`

These were clearly too loose on the GT set.

Common pattern:

- larger HD95
- noticeably more FP area
- masks biased slightly larger than GT
- weaker boundary fit

Interpretation:

- the GUI simple residual/saturation path with coarse post-refine was not enough for Nissl edges
- it kept most tissue, but boundary placement was less accurate and leaked more

### `exp_soft_support_mgac`

This method had very high precision and low leakage, but it was too conservative overall.

- Dice `0.9647`
- recall `0.9359`
- boundary recall only `0.6327`

Interpretation:

- it over-penalizes boundary regions
- it is not appropriate as the main Nissl default

## Section-Level Interpretation

### Clean sections: `2507_103`, `2507_79`, `2507_85`, `2507_91`

The experiment baseline family consistently dominated.

Typical behavior:

- near-perfect Dice
- tight boundary fit
- low leakage

The GUI simple family was still acceptable in overlap, but much worse in boundary distance and spill.

### Harder section: `2507_97`

This section showed the most visible tradeoff.

- `param_tight_v3`
  - Dice `0.9810`
  - FP / GT area `0.0218`
- `exp_baseline_v1`
  - Dice `0.9803`
  - FP / GT area `0.0249`

Interpretation:

- tighter variants can slightly reduce spill on edge-sensitive sections
- but the gain is small
- the plain baseline remains safer as the default because it gives the strongest general balance

### Recall-sensitive section: `2507_73`

The plain baseline remained best.

- `exp_baseline_v1`
  - Dice `0.9798`
  - recall `0.9654`

Tighter variants reduced leakage a bit, but also trimmed valid tissue.

## Practical Conclusion

### Best default strategy

Use:

- `exp_baseline_v1`

### Best conservative alternative

Keep available:

- `exp_baseline_posttight_v2`

### Best leakage-first alternative

Keep available:

- `param_tight_v3`

## Implementation Change

The GUI Nissl `Latest Contextual` path has been updated to use the tool-side Nissl baseline logic instead of the older GUI-only simple heuristic path.

This means:

- GUI `Latest Contextual` for Nissl now matches `exp_baseline_v1`
- old `Legacy Simple` remains available for comparison

Regression check:

- `gui_latest_contextual` now matches `exp_baseline_v1` on the GT benchmark

## Recommendation

For Nissl:

1. use `exp_baseline_v1` as the stable default
2. keep `exp_baseline_posttight_v2` as a conservative manual-review preset
3. keep `Legacy Simple` only as a historical comparison path

Do not use `soft_support_mgac` as the main Nissl default.

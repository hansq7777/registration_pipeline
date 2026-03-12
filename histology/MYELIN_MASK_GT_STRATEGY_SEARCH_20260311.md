# Myelin Mask GT Strategy Search 2026-03-11

## Scope

- GT source only:
  - `D:\Research\Image Analysis\Nanozoomer scans\20250327 rat myelin quantification\Tissue&Masks\test`
- Evaluated sections:
  - `2503_113`, `2503_114`, `2503_119`, `2503_120`, `2503_126`, `2503_132`, `2503_138`, `2503_144`
  - `2504_5`, `2504_47`, `2504_161`, `2504_185`
  - `2507_42`, `2507_48`, `2507_54`, `2507_60`, `2507_66`, `2507_72`
- All crops/masks have per-section `metadata.json`, so results remain tied back to original slide space.

## Experimental Goal

Re-run the myelin crop-mask search after the newer seed-box coverage fix, with emphasis on:

- boundary fit to GT
- leakage suppression
- avoiding pathological tiny / collapsed masks
- understanding local failure modes instead of only looking at global Dice

## Search Design

Working benchmark:

- crop-only benchmark on GT crops
- scale: `0.5x`
- focused candidate set:
  - `legacy_simple`
  - `simple_conservative`
  - `crop_center_default2comp`
  - `hybrid_default_k7_o03`
  - `hybrid_default_k7_o03_posttight_v1`
  - `hybrid_default_k7_o03_posttight_v2`
  - `hybrid_tightcand_k7_o03`
  - `hybrid_guard65_tightfallback`
- additional targeted probe:
  - `candidate_center_default2comp`
  - `hybrid_candcenter_k7_o03`
  - `hybrid_candcenter_posttight_v1`

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
  - false-positive area over GT area
  - border-touch leakage
  - predicted-to-GT area ratio
- local-region recall:
  - top / middle / bottom
  - left / center / right
  - boundary / core

Search output:

- focused benchmark:
  - `C:\Users\Siqi\Desktop\REVIEW\20260311_myelin_strategy_search_expanded_focus05`
- candidate-center probe:
  - `C:\Users\Siqi\Desktop\REVIEW\20260311_myelin_strategy_search_candcenter05`

## Main Result

The best practical family is still the hybrid family, but the best variant is slightly tighter than the previous GUI default.

The new best overall candidate is:

- `hybrid_tightcand_k7_o03`

Its behavior:

- keep the high-recall simple residual path as the candidate
- tighten that candidate slightly before reconstruction
- use the crop-center support/core only as structural guidance
- reconstruct inside the candidate

This improves boundary fit and leakage slightly over the previous `Hybrid Balanced` without sacrificing the key robustness benefit of the hybrid family.

## Ranked Candidates

### Candidate 1: `hybrid_tightcand_k7_o03`

Recommended new default.

- composite score: `0.5720`
- Dice: `0.7504`
- IoU: `0.6902`
- precision: `0.6935`
- recall: `0.8294`
- boundary F1 @ 64: `0.5709`
- mean HD95: `1366.2 px`
- false-positive / GT area: `0.3584`
- predicted / GT area ratio: `1.1878`

Why it wins:

- best composite score on the expanded GT set
- best boundary F1 among the robust hybrid candidates
- slightly lower leakage than the previous default
- does not reintroduce the very high spill of the simple family

### Candidate 2: `hybrid_default_k7_o03_posttight_v2`

Best conservative hybrid alternative.

- composite score: `0.5696`
- Dice: `0.7495`
- IoU: `0.6887`
- precision: `0.6920`
- recall: `0.8294`
- boundary F1 @ 64: `0.5652`
- mean HD95: `1364.8 px`
- false-positive / GT area: `0.3609`
- predicted / GT area ratio: `1.1903`

Why it is useful:

- nearly identical to the new default
- explicit post-tightening keeps it slightly more conservative in practice

### Candidate 3: `hybrid_default_k7_o03`

Previous `Hybrid Balanced` baseline.

- composite score: `0.5686`
- Dice: `0.7495`
- IoU: `0.6888`
- precision: `0.6912`
- recall: `0.8305`
- boundary F1 @ 64: `0.5632`
- mean HD95: `1364.0 px`
- false-positive / GT area: `0.3630`
- predicted / GT area ratio: `1.1936`

Why it should still be kept:

- it remains a strong baseline
- the new winner is only a modest but real improvement, not a wholesale replacement

## Why Some Methods Failed

### 1. Simple family: too much spill

`legacy_simple` and `simple_conservative` still have near-perfect recall, but they stay far too wide.

For `simple_conservative`:

- recall: `0.9953`
- boundary F1 @ 64: `0.3483`
- false-positive / GT area: `1.1215`
- predicted / GT area ratio: `2.1167`

Interpretation:

- they cover almost everything
- but they do it by swallowing too much non-target tissue
- boundary quality is poor even when Dice looks passable

### 2. Crop-center-only core: catastrophic small-mask failures

`crop_center_default2comp` still has the lowest leakage among meaningful candidates, but it catastrophically collapses on some sections:

- `2507_66`: Dice `0.0153`
- `2504_161`: Dice `0.0000`
- `2504_47`: Dice `0.0022`
- `2504_5`: Dice `0.0034`

Interpretation:

- the crop-center anchor is too brittle
- when the true tissue is far from the crop center, the support mask locks onto the wrong region or nothing useful at all

### 3. Candidate-center probe: not enough on its own

I tested replacing the crop-center core with a candidate-centered core derived from the simple mask.

This did not solve the problem cleanly:

- it helped some cases such as `2504_5`
- but it still failed badly on `2504_161`, `2504_47`, and `2504_185`
- the hybrid candidate-center variants became more permissive and leaked more

Interpretation:

- the simple candidate itself is not stable enough on these hard sections to define a reliable anchor
- switching from crop-center to candidate-center is not the right main fix

## Why Different Methods Win on Different Sections

### Easy sections: `2507_42`, `2507_48`, `2507_54`

All hybrid variants are strong here.

Interpretation:

- the candidate mask is already close to correct
- the main job is trimming outer spill without losing tissue

### Hard boundary / lower-edge case: `2503_144`

This section remains a key discriminator.

- `hybrid_default_k7_o03`: Dice `0.8829`, BF64 `0.5880`, FP/GT `0.2654`
- `hybrid_tightcand_k7_o03`: Dice `0.8827`, BF64 `0.5900`, FP/GT `0.2649`

Interpretation:

- tighter candidate trimming helps slightly on boundary/leakage
- but this section is still fundamentally hard because the lower footprint is difficult to separate cleanly

### Collapse-risk case: `2507_66`

- `crop_center_default2comp`: Dice `0.0153`
- `hybrid_default_k7_o03`: Dice `0.9470`
- `hybrid_tightcand_k7_o03`: Dice `0.9469`

Interpretation:

- this remains the strongest argument for keeping the hybrid family as default
- the high-recall candidate path is necessary as a safety net

### Offset / awkward crop cases: `2504_161`, `2504_47`, `2504_5`, `2504_185`

These sections explain why the search cannot be reduced to a single metric.

- crop-center methods are too brittle
- candidate-center methods are not reliable enough
- simple methods overlap but leak badly
- hybrids remain the best compromise, even though these cases are still imperfect

## Practical Conclusion

The best current policy is:

- keep the hybrid family
- tighten the candidate slightly before reconstruction
- do not switch the anchor logic entirely to candidate-center

So the new recommended default is:

- `hybrid_tightcand_k7_o03`

## GUI Update

The GUI `Hybrid Balanced` preset has been updated to follow this improved tighter-candidate version.

That means current Gallyas default in the GUI now corresponds more closely to:

- simple candidate
- light candidate tightening
- crop-center core
- hybrid reconstruction

This is a small but real improvement over the earlier `Hybrid Balanced` baseline.

## Next Optimization Target

The remaining weak point is now clearer:

- not general recall
- not general boundary quality
- but specific offset / awkward-crop cases on the `2504` group

The next useful direction is likely:

- better target-component selection inside the candidate
- not a wider candidate
- and not a pure candidate-center anchor

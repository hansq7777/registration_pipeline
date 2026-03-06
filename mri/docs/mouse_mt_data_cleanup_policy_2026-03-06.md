# mouse_mt_pipeline Data Cleanup Policy (2026-03-06)

Context:

- `mouse_mt_pipeline` currently has local tracked changes under `Data/`.
- These changes mix final MRI outputs, format-only JSON churn, and obvious intermediate files.
- This note records how to separate them before any future versioning decision.

## 1. Keep as versionable final MRI outputs

These are reasonable to keep if the repository continues to store representative MRI-side results:

- `Data/mouse4test6_20251113/B1/B1_ph.nii.gz`
- `Data/mouse4test6_20251113/B1/B1_ph.json`
- `Data/mouse4test6_20251113/MToff_PDw/MToff_PDw.nii.gz`
- `Data/mouse4test6_20251113/MToff_T1/MToff_T1.nii.gz`
- `Data/mouse4test6_20251113/MTon/MTon.nii.gz`
- `Data/mouse4test6_20251113/MTon/MTon_mtc.nii.gz`
- `Data/mouse4test6_20251113/MTon/MTon_mtr.nii.gz`
- `Data/mouse4test6_20251113/MTon/MTon_mtsat.nii.gz`
- `Data/mouse4test6_20251113/RAREvfl/RAREvfl.nii.gz`

Operational note:

- `MTon_mask.nii.gz` is referenced by the UI presets and should not be treated as throwaway output without replacing those defaults.

## 2. Intermediate artifacts that should stay out of version control

These are clearly processing intermediates or retry artifacts:

- `Data/**/B1/*_vol2.nii.gz`
- `Data/**/B1/*_vol2_RS.nii.gz`
- `Data/**/B1/*_RS.nii.gz`
- `Data/**/B1/*_v2.nii.gz`
- `Data/**/*_recenter.nii.gz`

Current local examples:

- `Data/mouse4test6_20251113/B1/B1_ph_RS.nii.gz`
- `Data/mouse4test6_20251113/B1/B1_ph_v2.nii.gz`
- `Data/mouse4test6_20251113/B1/B1_ph_vol2.nii.gz`
- `Data/mouse4test6_20251113/B1/B1_ph_vol2_RS.nii.gz`
- `Data/mouse4test6_20251113/MTon/MTon_recenter.nii.gz`

## 3. Format-only JSON churn that should not be versioned as meaningful change

These files had no semantic key/value delta when compared against `HEAD`; they only changed by formatting/newline style:

- `Data/mouse3test5_20251110/MToff_PD_15/MToff_PD_15.json`
- `Data/mouse3test5_20251110/MToff_T1_21/MToff_T1_21.json`
- `Data/mouse3test5_20251110/MTon_24/MTon_24.json`
- `Data/mouse3test5_20251110/RAREvfl_3/RAREvfl_3.json`
- `Data/mouse4test6_20251113/MToff_PDw/MToff_PDw.json`
- `Data/mouse4test6_20251113/MToff_T1/MToff_T1.json`
- `Data/mouse4test6_20251113/MTon/MTon.json`
- `Data/mouse4test6_20251113/RAREvfl/RAREvfl.json`

## 4. Semantic JSON change detected

One local JSON change did alter content:

- `Data/mouse4test6_20251113/B1/B1_ph.json`
  - changed key: `B1Units`

This one requires domain review instead of automatic cleanup.

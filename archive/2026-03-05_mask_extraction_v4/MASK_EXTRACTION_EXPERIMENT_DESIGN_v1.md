# Nissl Tissue Mask Extraction (v4 Reproducible Workflow)

## 0. Purpose
Build reproducible coronal-section tissue masks from Nissl slides, with fixed v4 defaults, full traceability, and QC storyboard outputs.

## 1. Fixed v4 policy (do not change unless explicitly requested)
1) Gaussian branch only (`bilateral` disabled by default)
2) Threshold stage uses only `global_otsu` + `local_sauvola`
3) `global_triangle` disabled by default
4) Candidate fusion uses `union` only (`weighted` disabled)
5) Binary read threshold fixed to wide setting: `bin_thresh=0`
6) Final deliverables are:
- Step8 final masks
- Step9 storyboard overlays + summary CSV

## 2. Data and output layout
- Input raw slides: `C:/Users/Siqi/Desktop/test slices/*.jpg`
- Main run outputs:
  - `step1_v4_gaussian_wide`
  - `step2_v4_gaussian_wide`
  - `step3_v4_gaussian_wide`
  - `step4_v4_gaussian_wide`
  - `step5_v4_gaussian_wide`
  - `step6_v4_gaussian_wide`
  - `step7_v4_gaussian_wide`
  - `step8_v4_gaussian_wide`
  - `step9QC`

## 3. Script inventory (all scripts used in v4)
1. `step1白平衡/背景校正/run_step1_wb_bg.py`
2. `step2转换lab/run_step2_to_lab.py`
3. `step3轻度去噪_bchannel/run_step3_denoise_b.py`
4. `step4降采样阈值/run_step4_downsample_thresholds.py`
5. `step5候选tissue_mask/run_step5_candidate_masks.py`
6. `step6形态学清理_union/run_step6_union_morph.py`
7. `step7连通域_top1/run_step7_cc_top1.py`
8. `step8_v4_gaussian_wide/run_step8_final_mask.py`
9. `scripts/run_step9_qc_storyboard.py`
10. `scripts/run_v4_pipeline.sh` (new orchestration entry)

## 4. Step-by-step logic and parameters

### Step1: White balance + background correction
- Logic:
  - Grey-world white balance from border sampling
  - Large-scale Gaussian background estimation/subtraction
- Core params:
  - `border_ratio=0.06`
  - `border_min_px=10`
  - `gain_min=0.6`
  - `gain_max=1.8`
  - `sigma_min=80`
  - `sigma_scale=18.0`
- Output: `step1_v4_gaussian_wide/*_wb_bg.jpg`

### Step2: Lab conversion
- Logic:
  - Convert WB image to Lab
  - Export `L_channel`, `a_channel`, `b_channel`
- Output used downstream:
  - `step2_v4_gaussian_wide/b_channel/*.jpg`

### Step3: Light denoise on b-channel
- Logic:
  - Denoise to stabilize thresholding
  - v4 keeps only `gaussian`
- Core params:
  - `--branches gaussian`
  - Gaussian: `g_ksize=5`, `g_sigma=1.0`
- Output: `step3_v4_gaussian_wide/gaussian/*.jpg`

### Step4: Downsample + threshold candidates
- Logic:
  - 4x downsample for robust coarse thresholding
  - produce `global_otsu` and `local_sauvola`
  - `triangle` disabled by default
- Core params:
  - `ds_factor=4`
  - `sauvola_window=51`
  - `sauvola_k=0.2`
  - `sauvola_R=128`
  - no `--enable-triangle`
- Output:
  - `step4_v4_gaussian_wide/gaussian/global_otsu`
  - `step4_v4_gaussian_wide/gaussian/local_sauvola`
  - summary: `step4_v4_gaussian_wide/step4_threshold_summary.csv`

### Step5: Candidate mask fusion
- Logic:
  - fuse Otsu + Sauvola by OR
  - weighted voting disabled
- Core params:
  - `method_a=global_otsu`
  - `method_b=local_sauvola`
  - `--disable-weighted`
  - `bin_thresh=0` (default)
- Output:
  - `step5_v4_gaussian_wide/gaussian/union_global_otsu_plus_local_sauvola/*.png`
  - summary: `step5_v4_gaussian_wide/step5_candidate_summary.csv`

### Step6: Morphology cleanup + robust fill-holes
- Logic:
  - closing/opening smooths shape
  - robust padded flood-fill hole filling
  - guard rails prevent fill-explosion
- Core params:
  - `close_k=5`, `close_iter=1`
  - `open_k=3`, `open_iter=1`
  - `max_fill_gain=0.25`
  - `max_area_ratio=0.95`
  - `bin_thresh=0` (default)
- Output:
  - `step6_v4_gaussian_wide/gaussian/union_morph_filled/*.png`
  - summary: `step6_v4_gaussian_wide/step6_union_morph_summary.csv`

### Step7: Connected-component top-1
- Logic:
  - optional pre-CC opening to cut thin bridges
  - keep largest connected tissue component
- Core params:
  - `bridge_open_k=3`
  - `bin_thresh=0` (default)
- Output:
  - `step7_v4_gaussian_wide/gaussian/union_top1/*.png`
  - summary: `step7_v4_gaussian_wide/step7_top1_summary.csv`

### Step8: Final mask refinement at full resolution
- Logic:
  - upsample coarse mask to original grid (nearest)
  - one-pass boundary-band local refinement
  - post-close + fill-holes + top1
- Core params:
  - `boundary_k=9`
  - `adaptive_block=51`
  - `adaptive_c=2.0`
  - `post_close_k=3`
- Output:
  - `step8_v4_gaussian_wide/gaussian/final_coronal_tissue_mask/*.png`
  - summary: `step8_v4_gaussian_wide/step8_final_mask_summary.csv`
  - params: `step8_v4_gaussian_wide/step8_params_used.json`

### Step9: QC storyboard overlays
- Logic:
  - overlay stage masks on source image in semi-transparent red
  - stage sequence:
    - `step4_otsu` -> `step4_sauvola` -> `step5_union` -> `step6_morph` -> `step7_top1` -> `step8_final`
  - output per-sample storyboard for quick visual audit
- Core params:
  - `alpha=0.35`
  - `panel_max_w=840`
  - `panel_max_h=620`
- Output:
  - `step9QC/gaussian/storyboards/*.png`
  - `step9QC/gaussian/overlays/<stage>/*.png`
  - summary: `step9QC/step9_qc_storyboard_summary.csv`

## 5. Repro commands

### 5.1 One-command full run (recommended)
```bash
bash "/mnt/c/Users/Siqi/Desktop/test slices/scripts/run_v4_pipeline.sh"
```

### 5.2 Key deliverables after run
1. Final masks:
- `C:/Users/Siqi/Desktop/test slices/step8_v4_gaussian_wide/gaussian/final_coronal_tissue_mask`
2. Final mask summary:
- `C:/Users/Siqi/Desktop/test slices/step8_v4_gaussian_wide/step8_final_mask_summary.csv`
3. QC storyboards:
- `C:/Users/Siqi/Desktop/test slices/step9QC/gaussian/storyboards`
4. QC summary:
- `C:/Users/Siqi/Desktop/test slices/step9QC/step9_qc_storyboard_summary.csv`

## 6. Minimal QC acceptance checklist
1. Step8 summary row count == input image count
2. Step9 summary row count == input image count
3. Storyboard count == input image count
4. For known bridge-risk cases (e.g., `nissl_2501_145`), check step7/step8 overlay continuity
5. Inspect outliers by `step8_final_fg_ratio` in Step9 summary

## 7. Notes cleanup status
This document is now v4-only.
Removed from this note:
- historical v2/v3 branch strategy details
- deprecated bilateral default narrative
- deprecated triangle-in-default narrative
- obsolete pending-task notes

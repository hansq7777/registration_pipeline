# Nissl Tissue Mask Extraction (v4 Reproducible Workflow)

## 0. Purpose
Build reproducible coronal-section tissue masks from Nissl slides, with fixed v4 defaults, full traceability, and QC storyboard outputs.

## 0.5 Optional whole-slide NDPI extension
When the input is a NanoZoomer whole-slide (`.ndpi`) rather than an already cropped slice image, do not send the WSI directly into Step1-9. Use a pre-step whole-slide review workflow first:

1. Parse expected section IDs from the filename.
2. Read the smallest main pyramid level for whole-slide overview proposal.
3. Detect candidate slice regions on the overview image.
4. Map accepted boxes back to level-0 coordinates.
5. Export per-slice review crops, masks, RGBA cutouts, and QC panels.
6. If needed, re-export accepted ROIs and feed them into the original cropped-slice v4 workflow.

Current review script:
- `histology/tools/run_ndpi_review_experiment.py`

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
- Optional whole-slide input: `*.ndpi`
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
0. `histology/tools/run_ndpi_review_experiment.py` (optional whole-slide NDPI review/proposal)
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

### Step0 (optional): NDPI whole-slide proposal + review mask
- Scope:
  - use only when source data are whole-slide NanoZoomer scans
  - Step1-9 remain unchanged for accepted cropped ROIs
- Current status:
  - fixed baseline for routine review: `baseline_v1`
  - experimental prototype branch: `soft_support_mgac`
  - note: `baseline_v1` is stain-aware and contains separate Nissl and Gallyas branches; Gallyas adaptation does not replace the Nissl path
  - optional Gallyas enhancement: `Nissl-guided proposal prior` using matched or adjacent (`+/-1`) Nissl sections when available
- Logic:
  - parse stain/sample/section IDs from filename metadata
  - use the smallest main pyramid level as overview image
  - detect candidate slice components from overview saturation + nonwhite score
  - merge horizontally adjacent overview components when one slice is split by thresholding
  - assign section IDs by mount order: left-to-right on first row, then left-to-right on second row
  - map candidate boxes back to level-0 coordinates
  - export review crops plus per-crop review masks
- Baseline `baseline_v1` crop-mask logic:
  - detect border-touching artifacts on the crop
  - build a coarse large-blur silhouette from stain-aware foreground score
  - expand to lightly stained edge pixels by local score grow
  - keep the largest connected component and fill holes
- Stain-aware score note:
  - `nissl`: emphasize saturation + nonwhite tissue signal
  - `gallyas`: emphasize grayscale darkness/nonwhite signal because myelin slides are effectively low-saturation
  - implementation policy: keep these as separate stain-specific branches inside the same tool, with no cross-overwriting of thresholds/scores
- Gallyas proposal prior note:
  - if a corresponding Nissl section is available, the Gallyas proposal stage can borrow Nissl-derived bbox size priors
  - section matching uses same sample ID and nearest adjacent section number (`+1` preferred, then `-1`)
  - prior is used as a size prior, not as an absolute position prior, to avoid mounting-layout mismatch across slides
- Experimental `soft_support_mgac` logic:
  - start from the same overview proposal and crop extraction
  - create ownership-guided soft support
  - initialize a permissive crop candidate
  - run downsampled morphological geodesic active contour for shrink-to-boundary refinement
- Key outputs:
  - overview image
  - overview candidate boxes
  - per-crop RGB images
  - per-crop `score`, `artifact`, `blur`, `mask`, `overlay`, `RGBA`
  - candidate summary CSV with level-0 bbox coordinates

### Step0.1 Failure analysis summary
- Main failure mode is target-selection failure, not simple threshold failure.
- Close-contact artifacts include:
  - glass-edge bands touching the crop border
  - neighboring mounted sections entering the crop from top/bottom
  - dark cross/marker structures on slide hardware attracting the seed/component selection
  - dirty background structures with contrast similar to tissue edge
- This means:
  - heuristic morphology alone is insufficient in some crops
  - contour methods also fail if initialized from the wrong target support
  - parameter sweeps can shift under/over-segmentation, but do not fully solve wrong-target selection

### Step0.2 Registration-oriented requirements
- The segmentation target is used for downstream registration to corresponding myelin sections.
- Therefore:
  - edge quality and smoothness matter, not only foreground coverage
  - masks should preserve gross coronal outline without border tails or neighboring-section bleed
  - questionable cases should be explicitly flagged for manual review/intervention
- Recommended operational policy:
  - use `baseline_v1` for current batch review and manual correction triage
  - use QC flags and overlay review to identify slices needing manual mask cleanup before registration
  - treat orientation handling as a separate pre-registration step; left-right mirrored mounting should not be assumed consistent across slides

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

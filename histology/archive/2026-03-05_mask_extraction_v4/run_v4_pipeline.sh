#!/usr/bin/env bash
set -euo pipefail

BASE="/mnt/c/Users/Siqi/Desktop/test slices"

# v4 fixed outputs
S1="$BASE/step1_v4_gaussian_wide"
S2="$BASE/step2_v4_gaussian_wide"
S3="$BASE/step3_v4_gaussian_wide"
S4="$BASE/step4_v4_gaussian_wide"
S5="$BASE/step5_v4_gaussian_wide"
S6="$BASE/step6_v4_gaussian_wide"
S7="$BASE/step7_v4_gaussian_wide"
S8="$BASE/step8_v4_gaussian_wide"
S9="$BASE/step9QC"

mkdir -p "$S1" "$S2" "$S3" "$S4" "$S5" "$S6" "$S7" "$S8" "$S9"

# cleanup outputs (preserve scripts)
rm -rf "$S1"/* "$S2"/* "$S3"/* "$S4"/* "$S5"/* "$S6"/* "$S7"/*
rm -rf "$S8/gaussian" "$S8/step8_final_mask_summary.csv" "$S8/step8_params_used.json"
rm -rf "$S9/gaussian" "$S9/step9_qc_storyboard_summary.csv"

# Step1: white-balance + background correction
python3 "$BASE/step1白平衡/背景校正/run_step1_wb_bg.py" \
  --input "$BASE" \
  --output "$S1"

# Step2: Lab conversion
python3 "$BASE/step2转换lab/run_step2_to_lab.py" \
  --input "$S1" \
  --output "$S2"

# Step3: denoise (gaussian only)
python3 "$BASE/step3轻度去噪_bchannel/run_step3_denoise_b.py" \
  --input "$S2/b_channel" \
  --output "$S3" \
  --branches gaussian

# Step4: downsample + threshold (otsu + sauvola only; triangle disabled by default)
python3 "$BASE/step4降采样阈值/run_step4_downsample_thresholds.py" \
  --input-root "$S3" \
  --output-root "$S4" \
  --branches gaussian

# Step5: union(otsu+sauvola), weighted disabled, bin_thresh=0 (default)
python3 "$BASE/step5候选tissue_mask/run_step5_candidate_masks.py" \
  --input-root "$S4" \
  --output-root "$S5" \
  --branches gaussian \
  --method-a global_otsu \
  --method-b local_sauvola \
  --disable-weighted

# Step6: morphology + robust fill-holes guard
python3 "$BASE/step6形态学清理_union/run_step6_union_morph.py" \
  --step5-root "$S5" \
  --output-root "$S6" \
  --branches gaussian \
  --close-k 5 --open-k 3 --close-iter 1 --open-iter 1 \
  --max-fill-gain 0.25 --max-area-ratio 0.95

# Step7: top1 connected component with bridge cut
python3 "$BASE/step7连通域_top1/run_step7_cc_top1.py" \
  --step6-root "$S6" \
  --output-root "$S7" \
  --branches gaussian \
  --bridge-open-k 3

# Step8: upsample + boundary refinement + top1 final mask
python3 "$S8/run_step8_final_mask.py" \
  --step7-root "$S7" \
  --source-root "$S3" \
  --output-root "$S8" \
  --branch gaussian \
  --boundary-k 9 \
  --adaptive-block 51 \
  --adaptive-c 2.0 \
  --post-close-k 3

# Step9: QC storyboard overlays
python3 "$BASE/scripts/run_step9_qc_storyboard.py" \
  --step3-root "$S3" \
  --step4-root "$S4" \
  --step5-root "$S5" \
  --step6-root "$S6" \
  --step7-root "$S7" \
  --step8-root "$S8" \
  --output-root "$S9" \
  --branch gaussian \
  --alpha 0.35 \
  --panel-max-w 840 \
  --panel-max-h 620

echo "v4 pipeline done"
echo "step8 summary: $S8/step8_final_mask_summary.csv"
echo "step9 summary: $S9/step9_qc_storyboard_summary.csv"

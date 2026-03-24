#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from histology.tools.evaluate_myelin_mask_after_bbox_update import (  # noqa: E402
    build_section_to_slide_index,
    combined_overlay,
    collect_gt_sections,
    contour_metrics,
    extract_crop_from_overview_rect,
    finite_mean,
    open_slide_candidates,
    overlay_mask,
    overview_rect_to_level0,
    project_gt_to_new_crop,
    region_metrics,
)
from histology.tools.run_ndpi_review_experiment import proposal_crop_rect_overview  # noqa: E402
from histology.tools.search_myelin_mask_strategies import (  # noqa: E402
    leakage_metrics,
    local_recall,
    method_factory,
    region_slice_masks,
)


def _rescale_crop_and_mask(crop_rgb: np.ndarray, gt_mask: np.ndarray, scale: float) -> tuple[np.ndarray, np.ndarray]:
    if scale >= 0.999:
        return crop_rgb, gt_mask
    h, w = crop_rgb.shape[:2]
    out_w = max(1, int(round(w * scale)))
    out_h = max(1, int(round(h * scale)))
    crop_small = cv2.resize(crop_rgb, (out_w, out_h), interpolation=cv2.INTER_AREA)
    mask_small = cv2.resize(gt_mask.astype(np.uint8) * 255, (out_w, out_h), interpolation=cv2.INTER_NEAREST) > 0
    return crop_small, mask_small


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ndpi-root", required=True)
    parser.add_argument("--gt-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sample-ids", nargs="*", default=[])
    parser.add_argument("--sections", nargs="*", default=[])
    parser.add_argument("--crop-level", type=int, default=4)
    parser.add_argument("--scale", type=float, default=0.5)
    parser.add_argument("--methods", nargs="*", default=[])
    parser.add_argument("--save-per-section-images", action="store_true")
    args = parser.parse_args()

    ndpi_root = Path(args.ndpi_root)
    gt_root = Path(args.gt_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    per_section_dir = output_dir / "per_section"
    if args.save_per_section_images:
        per_section_dir.mkdir(exist_ok=True)

    gt_sections = collect_gt_sections(
        gt_root,
        set(args.sample_ids) if args.sample_ids else None,
        set(args.sections) if args.sections else None,
    )
    slide_index = build_section_to_slide_index(ndpi_root)
    chosen = set(args.methods) if args.methods else None

    slide_cache: dict[Path, tuple[object, np.ndarray, list[object], dict[str, object]]] = {}
    rows: list[dict] = []
    method_names = list(args.methods) if args.methods else list(method_factory().keys())
    aggregate: dict[str, list[dict]] = {name: [] for name in method_names}

    total = len(gt_sections)
    for idx_gt, gt in enumerate(gt_sections, start=1):
        print(f"[{idx_gt}/{total}] {gt.label}", flush=True)
        slide_path = slide_index.get(("gallyas", gt.sample_id, gt.section_id))
        if slide_path is None:
            continue
        if slide_path not in slide_cache:
            slide_cache[slide_path] = open_slide_candidates(slide_path)
        loaded, overview_rgb, candidates, candidate_map = slide_cache[slide_path]
        candidate = candidate_map.get(gt.label)
        if candidate is None:
            continue

        crop_rect_ov = proposal_crop_rect_overview(candidate, overview_rgb, "gallyas", candidates)
        crop_bbox_level0 = overview_rect_to_level0(loaded, crop_rect_ov)
        crop_rgb = extract_crop_from_overview_rect(loaded, crop_rect_ov, args.crop_level)
        gt_new, valid_window, crop_coverage = project_gt_to_new_crop(
            gt.gt_mask,
            gt.gt_crop_bbox_level0,
            crop_bbox_level0,
            crop_rgb.shape[:2],
        )
        crop_rgb_eval, gt_eval_full = _rescale_crop_and_mask(crop_rgb, gt_new, float(args.scale))
        valid_eval = (
            cv2.resize(valid_window.astype(np.uint8) * 255, (crop_rgb_eval.shape[1], crop_rgb_eval.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
            if float(args.scale) < 0.999
            else valid_window
        )
        local_regions = region_slice_masks(gt_eval_full)
        section_images: dict[str, np.ndarray] = {}
        methods = method_factory()
        if chosen:
            methods = {k: v for k, v in methods.items() if k in chosen}

        for method_name, fn in methods.items():
            pred = fn(crop_rgb_eval).astype(bool)
            gt_area = max(1, int(gt_eval_full.sum()))
            within_crop_gt_recall = float((pred & gt_eval_full).sum() / gt_area) if gt_eval_full.any() else 0.0
            overall_gt_recall = float(within_crop_gt_recall * crop_coverage)

            if valid_eval.any():
                pred_valid = pred[valid_eval]
                gt_valid = gt_eval_full[valid_eval]
                rm = region_metrics(pred_valid, gt_valid)
                ys, xs = np.where(valid_eval)
                sub = (slice(int(ys.min()), int(ys.max()) + 1), slice(int(xs.min()), int(xs.max()) + 1))
                cm = contour_metrics(pred[sub], gt_eval_full[sub])
            else:
                rm = {"dice": 0.0, "iou": 0.0, "precision": 0.0, "recall": 0.0}
                cm = {"boundary_f1_tol32": 0.0, "boundary_f1_tol64": 0.0, "assd_px": float("inf"), "hd95_px": float("inf")}

            leak = leakage_metrics(pred, gt_eval_full)
            row = {
                "section": gt.label,
                "slide_name": slide_path.name,
                "method": method_name,
                "crop_coverage_recall": crop_coverage,
                "within_crop_gt_recall": within_crop_gt_recall,
                "overall_gt_recall": overall_gt_recall,
                "pred_area_ratio": float(pred.mean()),
                "gt_area_ratio": float(gt_eval_full.mean()),
                "dice": rm["dice"],
                "iou": rm["iou"],
                "precision": rm["precision"],
                "recall": rm["recall"],
                "boundary_f1_tol32": cm["boundary_f1_tol32"],
                "boundary_f1_tol64": cm["boundary_f1_tol64"],
                "assd_px": cm["assd_px"],
                "hd95_px": cm["hd95_px"],
                **leak,
                "top_recall": local_recall(pred, local_regions["top"]),
                "middle_recall": local_recall(pred, local_regions["middle"]),
                "bottom_recall": local_recall(pred, local_regions["bottom"]),
                "left_recall": local_recall(pred, local_regions["left"]),
                "center_recall": local_recall(pred, local_regions["center"]),
                "right_recall": local_recall(pred, local_regions["right"]),
                "boundary_recall": local_recall(pred, local_regions["boundary"]),
                "core_recall": local_recall(pred, local_regions["core"]),
            }
            rows.append(row)
            aggregate[method_name].append(row)
            if args.save_per_section_images:
                section_images[method_name] = combined_overlay(crop_rgb_eval, pred, gt_eval_full, valid_eval)

        if args.save_per_section_images:
            sec_dir = per_section_dir / gt.label
            sec_dir.mkdir(exist_ok=True)
            cv2.imwrite(str(sec_dir / "crop_rgb.png"), cv2.cvtColor(crop_rgb_eval, cv2.COLOR_RGB2BGR))
            cv2.imwrite(
                str(sec_dir / "gt_overlay.png"),
                cv2.cvtColor(overlay_mask(crop_rgb_eval, gt_eval_full, (0, 255, 0)), cv2.COLOR_RGB2BGR),
            )
            for method_name, overlay in section_images.items():
                cv2.imwrite(str(sec_dir / f"{method_name}_overlay.png"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

            ordered = [m for m in methods.keys() if m in section_images]
            tile_h, tile_w = crop_rgb_eval.shape[:2]
            header_h = 36
            panel = np.zeros((header_h + tile_h, tile_w * (2 + len(ordered)), 3), dtype=np.uint8)
            panel[:header_h, :, :] = 18
            panel[header_h : header_h + tile_h, 0:tile_w, :] = crop_rgb_eval
            panel[header_h : header_h + tile_h, tile_w : 2 * tile_w, :] = overlay_mask(crop_rgb_eval, gt_eval_full, (0, 255, 0))
            cv2.putText(panel, "crop", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(panel, "gt", (tile_w + 8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
            for idx_m, method_name in enumerate(ordered, start=2):
                x0 = idx_m * tile_w
                panel[header_h : header_h + tile_h, x0 : x0 + tile_w, :] = section_images[method_name]
                cv2.putText(panel, method_name[:28], (x0 + 8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imwrite(str(sec_dir / f"{gt.label}_panel.png"), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))

    with (output_dir / "per_section_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {}
    for method_name, method_rows in aggregate.items():
        assd_vals = [r["assd_px"] for r in method_rows]
        hd95_vals = [r["hd95_px"] for r in method_rows]
        mean_dice = float(np.mean([r["dice"] for r in method_rows]))
        mean_iou = float(np.mean([r["iou"] for r in method_rows]))
        mean_precision = float(np.mean([r["precision"] for r in method_rows]))
        mean_boundary_f1 = float(np.mean([r["boundary_f1_tol64"] for r in method_rows]))
        mean_leak = float(np.mean([r["fp_over_gt_area"] for r in method_rows]))
        mean_area_ratio = float(np.mean([r["pred_to_gt_area_ratio"] for r in method_rows]))
        hd95_f = finite_mean(hd95_vals)
        hd95_term = 1.0 / (1.0 + ((hd95_f or 1e9) / 1000.0))
        composite = (
            0.35 * mean_dice
            + 0.15 * mean_iou
            + 0.15 * mean_precision
            + 0.20 * mean_boundary_f1
            + 0.10 * hd95_term
            - 0.10 * abs(mean_area_ratio - 1.0)
            - 0.10 * mean_leak
        )
        summary[method_name] = {
            "count": len(method_rows),
            "mean_crop_coverage_recall": float(np.mean([r["crop_coverage_recall"] for r in method_rows])),
            "mean_within_crop_gt_recall": float(np.mean([r["within_crop_gt_recall"] for r in method_rows])),
            "mean_overall_gt_recall": float(np.mean([r["overall_gt_recall"] for r in method_rows])),
            "mean_pred_area_ratio": float(np.mean([r["pred_area_ratio"] for r in method_rows])),
            "mean_gt_area_ratio": float(np.mean([r["gt_area_ratio"] for r in method_rows])),
            "mean_pred_to_gt_area_ratio": mean_area_ratio,
            "mean_dice": mean_dice,
            "mean_iou": mean_iou,
            "mean_precision": mean_precision,
            "mean_recall": float(np.mean([r["recall"] for r in method_rows])),
            "mean_boundary_f1_tol32": float(np.mean([r["boundary_f1_tol32"] for r in method_rows])),
            "mean_boundary_f1_tol64": mean_boundary_f1,
            "mean_assd_px_finite": finite_mean(assd_vals),
            "mean_hd95_px_finite": hd95_f,
            "mean_fp_over_gt_area": mean_leak,
            "mean_border_fp_over_gt_area": float(np.mean([r["border_fp_over_gt_area"] for r in method_rows])),
            "mean_boundary_recall": float(np.mean([r["boundary_recall"] for r in method_rows])),
            "mean_core_recall": float(np.mean([r["core_recall"] for r in method_rows])),
            "composite_score": composite,
        }

    (output_dir / "aggregate_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    ranked = sorted(
        summary.items(),
        key=lambda kv: (
            kv[1]["composite_score"],
            kv[1]["mean_dice"],
            kv[1]["mean_boundary_f1_tol64"],
            -kv[1]["mean_fp_over_gt_area"],
        ),
        reverse=True,
    )
    lines = [
        "# Myelin Mask On Current GUI BBox Crops",
        "",
        f"GT sections evaluated: {len(gt_sections)}",
        f"Crop level: {int(args.crop_level)}",
        f"Working scale: {float(args.scale):.3f}",
        "",
    ]
    for rank, (method_name, stats) in enumerate(ranked, start=1):
        lines.extend(
            [
                f"## {rank}. {method_name}",
                "",
                f"- composite_score: {stats['composite_score']:.4f}",
                f"- mean_crop_coverage_recall: {stats['mean_crop_coverage_recall']:.4f}",
                f"- mean_overall_gt_recall: {stats['mean_overall_gt_recall']:.4f}",
                f"- mean_dice: {stats['mean_dice']:.4f}",
                f"- mean_iou: {stats['mean_iou']:.4f}",
                f"- mean_precision: {stats['mean_precision']:.4f}",
                f"- mean_recall: {stats['mean_recall']:.4f}",
                f"- mean_boundary_f1_tol64: {stats['mean_boundary_f1_tol64']:.4f}",
                f"- mean_hd95_px_finite: {stats['mean_hd95_px_finite']}",
                f"- mean_fp_over_gt_area: {stats['mean_fp_over_gt_area']:.4f}",
                f"- mean_pred_to_gt_area_ratio: {stats['mean_pred_to_gt_area_ratio']:.4f}",
                "",
            ]
        )
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

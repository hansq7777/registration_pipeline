#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from histology.gui_mvp.hitl_gui.pipeline_adapters.segmentation_adapter import _make_mask_slightly_conservative, parse_slide_labels, propose_from_overview
from histology.gui_mvp.hitl_gui.pipeline_adapters.slide_io import extract_crop_for_preview, load_slide_bundle, open_slide_handle
from histology.gui_mvp.hitl_gui.pipeline_adapters.tool_bridge import load_histology_tool_module, proposal_bbox_level0_gui, proposal_to_tool_candidate
from histology.tools.evaluate_myelin_mask_after_bbox_update import (
    build_section_to_slide_index,
    collect_gt_sections,
    combined_overlay,
    contour_metrics,
    overview_rect_from_bbox_dict,
    overview_rect_to_level0,
    overlay_mask,
    project_gt_to_new_crop,
    region_metrics,
)


def build_context(loaded, proposal, proposals, crop_rgb, crop_level: int) -> dict:
    tool = load_histology_tool_module()
    overview_downsample = float(loaded.level_downsamples[loaded.overview_level])
    crop_level = min(crop_level, len(loaded.level_downsamples) - 1)
    crop_downsample = float(loaded.level_downsamples[crop_level])
    all_candidates = [proposal_to_tool_candidate(p, rank=idx + 1) for idx, p in enumerate(proposals)]
    target_rank = proposals.index(proposal) + 1
    target_candidate = proposal_to_tool_candidate(proposal, rank=target_rank)
    crop_bbox_level0 = proposal_bbox_level0_gui(loaded, proposal)
    ownership_strict, ownership_soft, support_mask = tool.build_crop_ownership_masks(
        target_candidate=target_candidate,
        all_candidates=all_candidates,
        crop_bbox_level0=crop_bbox_level0,
        crop_shape=crop_rgb.shape[:2],
        crop_downsample=crop_downsample,
        overview_downsample=overview_downsample,
    )
    target_center_px = tool.level0_point_to_crop(
        tool.candidate_center_level0(target_candidate, overview_downsample),
        crop_bbox_level0=crop_bbox_level0,
        crop_downsample=crop_downsample,
    )
    return {
        "tool": tool,
        "crop_bbox_level0": crop_bbox_level0,
        "ownership_strict": ownership_strict,
        "ownership_soft": ownership_soft,
        "support_mask": support_mask,
        "target_center_px": target_center_px,
    }


def old_contextual_mask(crop_rgb, ctx: dict) -> np.ndarray:
    result = ctx["tool"].build_crop_mask_baseline(
        crop_rgb,
        ownership_strict=ctx["ownership_strict"],
        ownership_soft=ctx["ownership_soft"],
        support_mask=ctx["support_mask"],
        target_center_px=ctx["target_center_px"],
        stain="gallyas",
    )
    return result["mask"].astype(np.uint8)


def tightened_contextual_mask(crop_rgb, ctx: dict) -> np.ndarray:
    result = ctx["tool"].build_crop_mask_baseline(
        crop_rgb,
        ownership_strict=ctx["ownership_strict"],
        ownership_soft=ctx["ownership_soft"],
        support_mask=ctx["support_mask"],
        target_center_px=ctx["target_center_px"],
        stain="gallyas",
        gallyas_support_soft_frac=0.016,
        gallyas_candidate_thresh_scale=1.05,
        gallyas_grow_quantile=0.22,
        gallyas_grow_scale=0.86,
    )
    return _make_mask_slightly_conservative(result["mask"].astype(np.uint8), "gallyas")


def mean_dict(rows: list[dict], keys: list[str]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key in keys:
        vals = [float(r[key]) for r in rows]
        out[key] = float(np.mean(vals)) if vals else 0.0
    return out


def write_overlay(path: Path, array: np.ndarray) -> None:
    Image.fromarray(array.astype(np.uint8), mode="RGB").save(path, format="PNG", compress_level=1, optimize=False)


def run_gt_benchmark(gallyas_root: Path, gt_root: Path, output_dir: Path, crop_level: int, labels: list[str]) -> dict:
    slide_index = build_section_to_slide_index(gallyas_root)
    gt_sections = collect_gt_sections(gt_root, sample_ids=None, section_labels=set(labels))
    rows: list[dict] = []
    comparison_dir = output_dir / "gt_comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)

    for gt in gt_sections:
        slide_path = slide_index.get(("gallyas", gt.sample_id, gt.section_id))
        if slide_path is None:
            continue

        loaded = load_slide_bundle(slide_path, "gallyas")
        overview_rgb = np.asarray(loaded.overview)
        proposals = propose_from_overview(slide_path, "gallyas", parse_slide_labels(slide_path.stem)[1], overview_rgb)
        proposal = next((p for p in proposals if p.label == gt.label), None)
        if proposal is None:
            continue

        handle = open_slide_handle(loaded)
        try:
            crop_rgb = extract_crop_for_preview(loaded, proposal, crop_level=crop_level, slide_handle=handle)
        finally:
            if handle is not None:
                handle.close()

        ctx = build_context(loaded, proposal, proposals, crop_rgb, crop_level)
        pred_old = old_contextual_mask(crop_rgb, ctx) > 0
        pred_new = tightened_contextual_mask(crop_rgb, ctx) > 0

        old_crop_rect_ov = overview_rect_from_bbox_dict(gt.proposal_bbox_overview, loaded.overview_size)
        old_crop_bbox_level0 = overview_rect_to_level0(loaded, old_crop_rect_ov)
        gt_new, valid, crop_coverage = project_gt_to_new_crop(
            gt.gt_mask,
            old_crop_bbox_level0,
            ctx["crop_bbox_level0"],
            crop_rgb.shape[:2],
        )

        old_reg = region_metrics(pred_old, gt_new)
        new_reg = region_metrics(pred_new, gt_new)
        old_contour = contour_metrics(pred_old, gt_new)
        new_contour = contour_metrics(pred_new, gt_new)

        rows.append(
            {
                "label": gt.label,
                "crop_coverage_recall": crop_coverage,
                "old_dice": old_reg["dice"],
                "new_dice": new_reg["dice"],
                "old_iou": old_reg["iou"],
                "new_iou": new_reg["iou"],
                "old_boundary_f1_tol32": old_contour["boundary_f1_tol32"],
                "new_boundary_f1_tol32": new_contour["boundary_f1_tol32"],
                "old_boundary_f1_tol64": old_contour["boundary_f1_tol64"],
                "new_boundary_f1_tol64": new_contour["boundary_f1_tol64"],
                "old_pred_area_ratio": float(pred_old.mean()),
                "new_pred_area_ratio": float(pred_new.mean()),
                "gt_area_ratio": float(gt_new.mean()),
            }
        )

        section_dir = comparison_dir / gt.label
        section_dir.mkdir(parents=True, exist_ok=True)
        write_overlay(section_dir / "old_overlay.png", combined_overlay(crop_rgb, pred_old, gt_new, valid))
        write_overlay(section_dir / "new_overlay.png", combined_overlay(crop_rgb, pred_new, gt_new, valid))
        write_overlay(section_dir / "gt_overlay.png", overlay_mask(crop_rgb, gt_new, (0, 255, 0)))

    csv_path = output_dir / "gt_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["label"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    keys = [
        "crop_coverage_recall",
        "old_dice",
        "new_dice",
        "old_iou",
        "new_iou",
        "old_boundary_f1_tol32",
        "new_boundary_f1_tol32",
        "old_boundary_f1_tol64",
        "new_boundary_f1_tol64",
        "old_pred_area_ratio",
        "new_pred_area_ratio",
        "gt_area_ratio",
    ]
    return {"rows": rows, "aggregate": mean_dict(rows, keys), "csv_path": str(csv_path)}


def run_qualitative(slide_path: Path, output_dir: Path, crop_level: int, labels: list[str]) -> dict:
    loaded = load_slide_bundle(slide_path, "gallyas")
    overview_rgb = np.asarray(loaded.overview)
    proposals = propose_from_overview(slide_path, "gallyas", parse_slide_labels(slide_path.stem)[1], overview_rgb)
    out_rows = []
    qual_dir = output_dir / "qualitative"
    qual_dir.mkdir(parents=True, exist_ok=True)

    handle = open_slide_handle(loaded)
    try:
        for label in labels:
            proposal = next((p for p in proposals if p.label == label), None)
            if proposal is None:
                continue
            crop_rgb = extract_crop_for_preview(loaded, proposal, crop_level=crop_level, slide_handle=handle)
            ctx = build_context(loaded, proposal, proposals, crop_rgb, crop_level)
            pred_old = old_contextual_mask(crop_rgb, ctx) > 0
            pred_new = tightened_contextual_mask(crop_rgb, ctx) > 0
            out_rows.append(
                {
                    "label": label,
                    "old_area_ratio": float(pred_old.mean()),
                    "new_area_ratio": float(pred_new.mean()),
                }
            )
            label_dir = qual_dir / label
            label_dir.mkdir(parents=True, exist_ok=True)
            write_overlay(label_dir / "old_overlay.png", overlay_mask(crop_rgb, pred_old, (255, 0, 0)))
            write_overlay(label_dir / "new_overlay.png", overlay_mask(crop_rgb, pred_new, (255, 0, 0)))
    finally:
        if handle is not None:
            handle.close()
    return {"rows": out_rows}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gallyas-root", default="/mnt/d/Research/Image Analysis/Nanozoomer scans/20250327 rat myelin quantification")
    parser.add_argument("--gt-root", default="/mnt/d/Research/Image Analysis/Nanozoomer scans/20250327 rat myelin quantification/Tissue&Masks")
    parser.add_argument("--gt-labels", nargs="*", default=["2502_78", "2502_84", "2502_90", "2502_96", "2502_102", "2502_108"])
    parser.add_argument("--qual-slide", default="/mnt/d/Research/Image Analysis/Nanozoomer scans/20250327 rat myelin quantification/gallyas_2503_78-108.ndpi")
    parser.add_argument("--qual-labels", nargs="*", default=["2503_102", "2503_108"])
    parser.add_argument("--crop-level", type=int, default=3)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    gt_summary = run_gt_benchmark(
        gallyas_root=Path(args.gallyas_root),
        gt_root=Path(args.gt_root),
        output_dir=output_dir,
        crop_level=args.crop_level,
        labels=list(args.gt_labels),
    )
    qual_summary = run_qualitative(
        slide_path=Path(args.qual_slide),
        output_dir=output_dir,
        crop_level=args.crop_level,
        labels=list(args.qual_labels),
    )

    summary = {
        "crop_level": args.crop_level,
        "gt_summary": gt_summary,
        "qualitative": qual_summary,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

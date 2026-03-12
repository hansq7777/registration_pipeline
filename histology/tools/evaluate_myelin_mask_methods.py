#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import importlib.util
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from histology.gui_mvp.hitl_gui.domain import ProposalBox
from histology.gui_mvp.hitl_gui.pipeline_adapters.segmentation_adapter import _simple_compute_auto_masks, compute_auto_masks
from histology.gui_mvp.hitl_gui.pipeline_adapters.slide_io import load_slide_bundle


def load_tool_module():
    tool_path = Path(__file__).resolve().parent / "run_ndpi_review_experiment.py"
    spec = importlib.util.spec_from_file_location("histology_ndpi_tool_eval", tool_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load tool module from {tool_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def region_metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    tp = int((pred & gt).sum())
    fp = int((pred & ~gt).sum())
    fn = int((~pred & gt).sum())
    tn = int((~pred & ~gt).sum())
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "dice": 2 * tp / max(1, 2 * tp + fp + fn),
        "iou": tp / max(1, tp + fp + fn),
        "precision": tp / max(1, tp + fp),
        "recall": tp / max(1, tp + fn),
        "pred_area_ratio": float(pred.mean()),
        "gt_area_ratio": float(gt.mean()),
    }


def boundary_mask(mask: np.ndarray) -> np.ndarray:
    mask_u8 = mask.astype(np.uint8)
    if mask_u8.max() == 0:
        return np.zeros_like(mask_u8, dtype=bool)
    eroded = cv2.erode(mask_u8 * 255, np.ones((3, 3), np.uint8), iterations=1) > 0
    return mask & (~eroded)


def contour_metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    pred_boundary = boundary_mask(pred)
    gt_boundary = boundary_mask(gt)
    if not pred_boundary.any() or not gt_boundary.any():
        return {
            "boundary_precision_tol16": 0.0,
            "boundary_recall_tol16": 0.0,
            "boundary_f1_tol16": 0.0,
            "boundary_precision_tol32": 0.0,
            "boundary_recall_tol32": 0.0,
            "boundary_f1_tol32": 0.0,
            "boundary_precision_tol64": 0.0,
            "boundary_recall_tol64": 0.0,
            "boundary_f1_tol64": 0.0,
            "assd_px": float("inf"),
            "hd95_px": float("inf"),
        }

    dt_to_gt = cv2.distanceTransform((~gt_boundary).astype(np.uint8), cv2.DIST_L2, 5)
    dt_to_pred = cv2.distanceTransform((~pred_boundary).astype(np.uint8), cv2.DIST_L2, 5)
    d_pred = dt_to_gt[pred_boundary]
    d_gt = dt_to_pred[gt_boundary]

    out = {}
    for tol_px in (16, 32, 64):
        b_prec = float((d_pred <= tol_px).mean()) if d_pred.size else 0.0
        b_rec = float((d_gt <= tol_px).mean()) if d_gt.size else 0.0
        b_f1 = 0.0 if (b_prec + b_rec) == 0 else 2 * b_prec * b_rec / (b_prec + b_rec)
        out[f"boundary_precision_tol{tol_px}"] = b_prec
        out[f"boundary_recall_tol{tol_px}"] = b_rec
        out[f"boundary_f1_tol{tol_px}"] = b_f1
    assd = float((d_pred.mean() + d_gt.mean()) / 2.0)
    hd95 = float(max(np.quantile(d_pred, 0.95), np.quantile(d_gt, 0.95)))
    out["assd_px"] = assd
    out["hd95_px"] = hd95
    return out


def draw_overlay(crop_rgb: np.ndarray, pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    overlay = crop_rgb.astype(np.float32).copy()
    overlay[gt] = 0.70 * overlay[gt] + 0.30 * np.array([0, 255, 0], dtype=np.float32)
    overlay[pred] = 0.70 * overlay[pred] + 0.30 * np.array([255, 0, 0], dtype=np.float32)
    return np.clip(overlay, 0, 255).astype(np.uint8)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--slide", required=True)
    parser.add_argument("--gt-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sections", nargs="+", required=True)
    parser.add_argument("--crop-level", type=int, default=3)
    args = parser.parse_args()

    tool = load_tool_module()
    slide_path = Path(args.slide)
    gt_root = Path(args.gt_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    per_section_dir = output_dir / "per_section"
    per_section_dir.mkdir(exist_ok=True)

    loaded = load_slide_bundle(slide_path, "gallyas")
    crop_level = min(args.crop_level, len(loaded.level_downsamples) - 1)
    overview_ds = float(loaded.level_downsamples[loaded.overview_level])
    crop_ds = float(loaded.level_downsamples[crop_level])
    slide_dims = loaded.level_dimensions[0]

    proposals: list[ProposalBox] = []
    tool_candidates = []
    for idx, sec in enumerate(args.sections, start=1):
        meta = json.loads((gt_root / sec / "metadata.json").read_text())
        x = meta["bbox_overview"]["x"]
        y = meta["bbox_overview"]["y"]
        w = meta["bbox_overview"]["w"]
        h = meta["bbox_overview"]["h"]
        proposals.append(
            ProposalBox(
                label=sec,
                stain="gallyas",
                sample_id=sec.split("_")[0],
                section_id=int(sec.split("_")[1]),
                proposal_rank=idx,
                x=x,
                y=y,
                w=w,
                h=h,
            )
        )
        tool_candidates.append(
            tool.CandidateBox(
                candidate_rank=idx,
                x=x,
                y=y,
                w=w,
                h=h,
                area=w * h,
                cx=x + w / 2.0,
                cy=y + h / 2.0,
                touches_border=False,
                section=tool.SectionLabel(stain="gallyas", sample_id=sec.split("_")[0], section_id=int(sec.split("_")[1])),
            )
        )
    loaded.proposals = proposals

    rows = []
    method_scores: dict[str, list[dict]] = {}
    for sec in args.sections:
        crop = np.asarray(Image.open(gt_root / sec / "crop_raw.png").convert("RGB"))
        gt = np.asarray(Image.open(gt_root / sec / "tissue_mask_final.png").convert("L")) > 0
        proposal = next(p for p in proposals if p.label == sec)
        cand = next(c for c in tool_candidates if c.section.short_label == sec)

        pad = max(24, int(round(max(cand.w, cand.h) * 0.08)))
        x1 = max(0, cand.x - pad)
        y1 = max(0, cand.y - pad)
        x2 = min(loaded.overview_size[0], cand.x + cand.w + pad)
        y2 = min(loaded.overview_size[1], cand.y + cand.h + pad)
        crop_bbox_level0 = (
            int(round(x1 * overview_ds)),
            int(round(y1 * overview_ds)),
            min(int(round((x2 - x1) * overview_ds)), slide_dims[0] - int(round(x1 * overview_ds))),
            min(int(round((y2 - y1) * overview_ds)), slide_dims[1] - int(round(y1 * overview_ds))),
        )
        ownership_strict, ownership_soft, support_mask = tool.build_crop_ownership_masks(
            target_candidate=cand,
            all_candidates=tool_candidates,
            crop_bbox_level0=crop_bbox_level0,
            crop_shape=crop.shape[:2],
            crop_downsample=crop_ds,
            overview_downsample=overview_ds,
        )
        target_center_px = tool.level0_point_to_crop(
            tool.candidate_center_level0(cand, overview_ds),
            crop_bbox_level0=crop_bbox_level0,
            crop_downsample=crop_ds,
        )

        methods = {
            "gui_old_baseline": _simple_compute_auto_masks(crop, "gallyas")[0] > 0,
            "experiment_single_component": tool.build_crop_mask_baseline(
                crop,
                ownership_strict,
                ownership_soft,
                support_mask,
                target_center_px,
                "gallyas",
                gallyas_max_components=1,
            )["mask"]
            > 0,
            "experiment_two_component": tool.build_crop_mask_baseline(
                crop,
                ownership_strict,
                ownership_soft,
                support_mask,
                target_center_px,
                "gallyas",
                gallyas_max_components=2,
            )["mask"]
            > 0,
            "gui_contextual_current": compute_auto_masks(
                crop,
                "gallyas",
                loaded_slide=loaded,
                target_proposal=proposal,
                all_proposals=proposals,
                crop_level=crop_level,
            )[0]
            > 0,
        }

        sec_dir = per_section_dir / sec
        sec_dir.mkdir(exist_ok=True)
        for method_name, pred in methods.items():
            row = {
                "section": sec,
                "method": method_name,
                **region_metrics(pred, gt),
                **contour_metrics(pred, gt),
            }
            rows.append(row)
            method_scores.setdefault(method_name, []).append(row)
            Image.fromarray((pred.astype(np.uint8) * 255)).save(sec_dir / f"{method_name}_mask.png")
            Image.fromarray(draw_overlay(crop, pred, gt)).save(sec_dir / f"{method_name}_overlay.png")

    with (output_dir / "comparison_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    aggregate = {}
    metric_names = [
        "dice",
        "iou",
        "precision",
        "recall",
        "pred_area_ratio",
        "gt_area_ratio",
        "boundary_precision_tol16",
        "boundary_recall_tol16",
        "boundary_f1_tol16",
        "boundary_precision_tol32",
        "boundary_recall_tol32",
        "boundary_f1_tol32",
        "boundary_precision_tol64",
        "boundary_recall_tol64",
        "boundary_f1_tol64",
        "assd_px",
        "hd95_px",
    ]
    for method_name, vals in method_scores.items():
        aggregate[method_name] = {
            metric: float(np.mean([v[metric] for v in vals]))
            for metric in metric_names
        }

    (output_dir / "aggregate_summary.json").write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "aggregate": aggregate}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

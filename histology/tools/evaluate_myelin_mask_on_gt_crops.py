#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from histology.gui_mvp.hitl_gui.pipeline_adapters.segmentation_adapter import (  # noqa: E402
    MASK_PRESET_LATEST_CONTEXTUAL,
    MASK_PRESET_LEGACY_SIMPLE,
    compute_auto_masks,
)
from histology.tools.run_ndpi_review_experiment import build_crop_mask_baseline  # noqa: E402


@dataclass
class GtCropItem:
    label: str
    sample_id: str
    section_id: int
    crop_rgb: np.ndarray
    gt_mask: np.ndarray
    metadata: dict


def _rescale_crop_and_mask(crop_rgb: np.ndarray, gt_mask: np.ndarray, scale: float) -> tuple[np.ndarray, np.ndarray]:
    if scale >= 0.999:
        return crop_rgb, gt_mask
    h, w = crop_rgb.shape[:2]
    out_w = max(1, int(round(w * scale)))
    out_h = max(1, int(round(h * scale)))
    crop_small = cv2.resize(crop_rgb, (out_w, out_h), interpolation=cv2.INTER_AREA)
    mask_small = cv2.resize(gt_mask.astype(np.uint8) * 255, (out_w, out_h), interpolation=cv2.INTER_NEAREST) > 0
    return crop_small, mask_small


def collect_gt_crops(
    gt_root: Path,
    *,
    sample_ids: set[str] | None = None,
    labels: set[str] | None = None,
    scale: float = 1.0,
) -> list[GtCropItem]:
    items: list[GtCropItem] = []
    for sec_dir in sorted(gt_root.iterdir()):
        if not sec_dir.is_dir():
            continue
        meta_path = sec_dir / "metadata.json"
        crop_path = sec_dir / "crop_raw.png"
        mask_path = sec_dir / "tissue_mask_final.png"
        if not meta_path.exists() or not crop_path.exists() or not mask_path.exists():
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        label = str(meta["label"])
        if labels and label not in labels:
            continue
        sample_id = str(meta["sample_id"])
        if sample_ids and sample_id not in sample_ids:
            continue
        crop_rgb = np.asarray(Image.open(crop_path).convert("RGB"))
        gt_mask = np.asarray(Image.open(mask_path).convert("L")) > 0
        crop_rgb, gt_mask = _rescale_crop_and_mask(crop_rgb, gt_mask, scale)
        items.append(
            GtCropItem(
                label=label,
                sample_id=sample_id,
                section_id=int(meta["section_id"]),
                crop_rgb=crop_rgb,
                gt_mask=gt_mask,
                metadata=meta,
            )
        )
    return items


def region_metrics(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    tp = int((pred & gt).sum())
    fp = int((pred & ~gt).sum())
    fn = int((~pred & gt).sum())
    return {
        "dice": 2 * tp / max(1, 2 * tp + fp + fn),
        "iou": tp / max(1, tp + fp + fn),
        "precision": tp / max(1, tp + fp),
        "recall": tp / max(1, tp + fn),
    }


def boundary_mask(mask: np.ndarray) -> np.ndarray:
    mask_u8 = mask.astype(np.uint8)
    if mask_u8.max() == 0:
        return np.zeros_like(mask_u8, dtype=bool)
    eroded = cv2.erode(mask_u8 * 255, np.ones((3, 3), np.uint8), iterations=1) > 0
    return mask & (~eroded)


def contour_metrics(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    pred_boundary = boundary_mask(pred)
    gt_boundary = boundary_mask(gt)
    if not pred_boundary.any() or not gt_boundary.any():
        return {
            "boundary_f1_tol32": 0.0,
            "boundary_f1_tol64": 0.0,
            "assd_px": float("inf"),
            "hd95_px": float("inf"),
        }
    dt_to_gt = cv2.distanceTransform((~gt_boundary).astype(np.uint8), cv2.DIST_L2, 5)
    dt_to_pred = cv2.distanceTransform((~pred_boundary).astype(np.uint8), cv2.DIST_L2, 5)
    d_pred = dt_to_gt[pred_boundary]
    d_gt = dt_to_pred[gt_boundary]

    def bf1(tol: int) -> float:
        b_prec = float((d_pred <= tol).mean()) if d_pred.size else 0.0
        b_rec = float((d_gt <= tol).mean()) if d_gt.size else 0.0
        return 0.0 if (b_prec + b_rec) == 0 else 2 * b_prec * b_rec / (b_prec + b_rec)

    return {
        "boundary_f1_tol32": bf1(32),
        "boundary_f1_tol64": bf1(64),
        "assd_px": float((d_pred.mean() + d_gt.mean()) / 2.0),
        "hd95_px": float(max(np.quantile(d_pred, 0.95), np.quantile(d_gt, 0.95))),
    }


def finite_mean(values: list[float]) -> float | None:
    finite = [float(v) for v in values if np.isfinite(v)]
    if not finite:
        return None
    return float(np.mean(finite))


def run_crop_center_baseline(crop_rgb: np.ndarray, **params) -> np.ndarray:
    h, w = crop_rgb.shape[:2]
    support = np.ones((h, w), dtype=bool)
    center = (float(w) / 2.0, float(h) / 2.0)
    result = build_crop_mask_baseline(
        crop_rgb,
        ownership_strict=support,
        ownership_soft=support,
        support_mask=support,
        target_center_px=center,
        stain="gallyas",
        **params,
    )
    return result["mask"] > 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sample-ids", nargs="*", default=[])
    parser.add_argument("--labels", nargs="*", default=[])
    parser.add_argument("--scale", type=float, default=1.0)
    args = parser.parse_args()

    gt_root = Path(args.gt_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    items = collect_gt_crops(
        gt_root,
        sample_ids=set(args.sample_ids) if args.sample_ids else None,
        labels=set(args.labels) if args.labels else None,
        scale=float(args.scale),
    )

    methods = {
        "legacy_simple": lambda crop: compute_auto_masks(crop, "gallyas", method=MASK_PRESET_LEGACY_SIMPLE)[0] > 0,
        "simple_conservative": lambda crop: compute_auto_masks(crop, "gallyas", method=MASK_PRESET_LATEST_CONTEXTUAL)[0] > 0,
        "crop_center_default2comp": lambda crop: run_crop_center_baseline(crop, gallyas_max_components=2),
        "crop_center_guiparams": lambda crop: run_crop_center_baseline(
            crop,
            gallyas_max_components=2,
            gallyas_support_soft_frac=0.016,
            gallyas_candidate_thresh_scale=1.05,
            gallyas_grow_quantile=0.22,
            gallyas_grow_scale=0.86,
        ),
        "crop_center_loose2comp": lambda crop: run_crop_center_baseline(
            crop,
            gallyas_max_components=2,
            gallyas_support_soft_frac=0.024,
            gallyas_candidate_thresh_scale=0.98,
            gallyas_grow_quantile=0.15,
            gallyas_grow_scale=0.81,
            gallyas_secondary_area_frac_primary=0.12,
            gallyas_secondary_area_frac_total=0.05,
            gallyas_secondary_support_overlap_min=0.50,
            gallyas_secondary_score_frac_primary=0.55,
        ),
    }

    rows: list[dict] = []
    aggregate: dict[str, list[dict]] = {name: [] for name in methods}

    for idx, item in enumerate(items, start=1):
        print(f"[{idx}/{len(items)}] {item.label}", flush=True)
        for method_name, fn in methods.items():
            pred = fn(item.crop_rgb)
            rm = region_metrics(pred, item.gt_mask)
            cm = contour_metrics(pred, item.gt_mask)
            row = {
                "section": item.label,
                "method": method_name,
                "pred_area_ratio": float(pred.mean()),
                "gt_area_ratio": float(item.gt_mask.mean()),
                "dice": rm["dice"],
                "iou": rm["iou"],
                "precision": rm["precision"],
                "recall": rm["recall"],
                "boundary_f1_tol32": cm["boundary_f1_tol32"],
                "boundary_f1_tol64": cm["boundary_f1_tol64"],
                "assd_px": cm["assd_px"],
                "hd95_px": cm["hd95_px"],
            }
            rows.append(row)
            aggregate[method_name].append(row)

    with (output_dir / "per_section_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {}
    for method_name, method_rows in aggregate.items():
        assd_vals = [r["assd_px"] for r in method_rows]
        hd95_vals = [r["hd95_px"] for r in method_rows]
        summary[method_name] = {
            "count": len(method_rows),
            "mean_pred_area_ratio": float(np.mean([r["pred_area_ratio"] for r in method_rows])),
            "mean_gt_area_ratio": float(np.mean([r["gt_area_ratio"] for r in method_rows])),
            "mean_dice": float(np.mean([r["dice"] for r in method_rows])),
            "mean_iou": float(np.mean([r["iou"] for r in method_rows])),
            "mean_precision": float(np.mean([r["precision"] for r in method_rows])),
            "mean_recall": float(np.mean([r["recall"] for r in method_rows])),
            "mean_boundary_f1_tol32": float(np.mean([r["boundary_f1_tol32"] for r in method_rows])),
            "mean_boundary_f1_tol64": float(np.mean([r["boundary_f1_tol64"] for r in method_rows])),
            "mean_assd_px_finite": finite_mean(assd_vals),
            "mean_hd95_px_finite": finite_mean(hd95_vals),
        }

    (output_dir / "aggregate_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    ranked = sorted(summary.items(), key=lambda kv: (kv[1]["mean_dice"], kv[1]["mean_boundary_f1_tol64"], kv[1]["mean_precision"]), reverse=True)
    lines = [
        "# Myelin Mask Benchmark On GT Crops",
        "",
        f"GT sections evaluated: {len(items)}",
        f"Working scale: {float(args.scale):.3f}",
        "",
        "Ranking priority:",
        "- mean Dice",
        "- mean boundary F1 @ 64 px",
        "- mean precision",
        "",
    ]
    for rank, (method_name, stats) in enumerate(ranked, start=1):
        lines.extend(
            [
                f"## {rank}. {method_name}",
                "",
                f"- mean_pred_area_ratio: {stats['mean_pred_area_ratio']:.4f}",
                f"- mean_gt_area_ratio: {stats['mean_gt_area_ratio']:.4f}",
                f"- mean_dice: {stats['mean_dice']:.4f}",
                f"- mean_iou: {stats['mean_iou']:.4f}",
                f"- mean_precision: {stats['mean_precision']:.4f}",
                f"- mean_recall: {stats['mean_recall']:.4f}",
                f"- mean_boundary_f1_tol32: {stats['mean_boundary_f1_tol32']:.4f}",
                f"- mean_boundary_f1_tol64: {stats['mean_boundary_f1_tol64']:.4f}",
                f"- mean_assd_px_finite: {stats['mean_assd_px_finite']}",
                f"- mean_hd95_px_finite: {stats['mean_hd95_px_finite']}",
                "",
            ]
        )
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

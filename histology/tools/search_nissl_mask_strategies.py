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
from scipy.ndimage import binary_fill_holes

Image.MAX_IMAGE_PIXELS = None

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from histology.gui_mvp.hitl_gui.pipeline_adapters.segmentation_adapter import (  # noqa: E402
    MASK_PRESET_LATEST_CONTEXTUAL,
    MASK_PRESET_LEGACY_SIMPLE,
    compute_auto_masks,
)
from histology.tools.run_ndpi_review_experiment import (  # noqa: E402
    build_crop_mask_baseline,
    build_crop_mask_soft_support_mgac,
    compute_stain_score,
    detect_border_artifacts,
    largest_component,
    odd_kernel,
)


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


def tight_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask)
    if xs.size == 0:
        return 0, 0, 0, 0
    x1 = int(xs.min())
    y1 = int(ys.min())
    x2 = int(xs.max()) + 1
    y2 = int(ys.max()) + 1
    return x1, y1, x2, y2


def region_slice_masks(gt: np.ndarray) -> dict[str, np.ndarray]:
    h, w = gt.shape[:2]
    x1, y1, x2, y2 = tight_bbox(gt)
    if x2 <= x1 or y2 <= y1:
        empty = np.zeros_like(gt, dtype=bool)
        return {name: empty for name in ["top", "middle", "bottom", "left", "center", "right", "boundary", "core"]}
    thirds_y = np.linspace(y1, y2, 4, dtype=int)
    thirds_x = np.linspace(x1, x2, 4, dtype=int)
    masks: dict[str, np.ndarray] = {}
    for name, a, b in [
        ("top", thirds_y[0], thirds_y[1]),
        ("middle", thirds_y[1], thirds_y[2]),
        ("bottom", thirds_y[2], thirds_y[3]),
    ]:
        m = np.zeros_like(gt, dtype=bool)
        m[a:b, x1:x2] = True
        masks[name] = m & gt
    for name, a, b in [
        ("left", thirds_x[0], thirds_x[1]),
        ("center", thirds_x[1], thirds_x[2]),
        ("right", thirds_x[2], thirds_x[3]),
    ]:
        m = np.zeros_like(gt, dtype=bool)
        m[y1:y2, a:b] = True
        masks[name] = m & gt
    band = max(3, int(round(min(y2 - y1, x2 - x1) * 0.03)))
    gt_u8 = gt.astype(np.uint8) * 255
    eroded = cv2.erode(gt_u8, np.ones((band * 2 + 1, band * 2 + 1), np.uint8), iterations=1) > 0
    masks["boundary"] = gt & (~eroded)
    masks["core"] = eroded
    return masks


def local_recall(pred: np.ndarray, gt_region: np.ndarray) -> float:
    denom = int(gt_region.sum())
    if denom == 0:
        return 0.0
    return float((pred & gt_region).sum() / denom)


def leakage_metrics(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    fp = pred & (~gt)
    gt_area = max(1, int(gt.sum()))
    pred_area = max(1, int(pred.sum()))
    h, w = gt.shape[:2]
    band = max(5, int(round(min(h, w) * 0.03)))
    border = np.zeros_like(gt, dtype=bool)
    border[:band, :] = True
    border[-band:, :] = True
    border[:, :band] = True
    border[:, -band:] = True
    x1, y1, x2, y2 = tight_bbox(gt)
    top = np.zeros_like(gt, dtype=bool)
    bottom = np.zeros_like(gt, dtype=bool)
    left = np.zeros_like(gt, dtype=bool)
    right = np.zeros_like(gt, dtype=bool)
    if x2 > x1 and y2 > y1:
        top[:y1, :] = True
        bottom[y2:, :] = True
        left[:, :x1] = True
        right[:, x2:] = True
    return {
        "fp_over_gt_area": float(fp.sum() / gt_area),
        "fp_over_pred_area": float(fp.sum() / pred_area),
        "border_fp_over_gt_area": float((fp & border).sum() / gt_area),
        "top_fp_over_gt_area": float((fp & top).sum() / gt_area),
        "bottom_fp_over_gt_area": float((fp & bottom).sum() / gt_area),
        "left_fp_over_gt_area": float((fp & left).sum() / gt_area),
        "right_fp_over_gt_area": float((fp & right).sum() / gt_area),
        "pred_to_gt_area_ratio": float(pred.sum() / gt_area),
    }


def finite_mean(values: list[float]) -> float | None:
    finite = [float(v) for v in values if np.isfinite(v)]
    if not finite:
        return None
    return float(np.mean(finite))


def _ones_support(crop_rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[float, float]]:
    h, w = crop_rgb.shape[:2]
    support = np.ones((h, w), dtype=bool)
    center = (float(w) / 2.0, float(h) / 2.0)
    return support, support, support, center


def tighten_with_area_guard(mask: np.ndarray, *, open_k: int, erode_k: int, min_keep_frac: float) -> np.ndarray:
    mask = mask.astype(bool)
    if not mask.any():
        return mask
    orig_area = int(mask.sum())
    out = cv2.morphologyEx(mask.astype(np.uint8) * 255, cv2.MORPH_OPEN, np.ones((open_k, open_k), np.uint8)) > 0
    eroded = cv2.erode(out.astype(np.uint8) * 255, np.ones((erode_k, erode_k), np.uint8), iterations=1) > 0
    if eroded.any() and int(eroded.sum()) >= int(round(orig_area * min_keep_frac)):
        out = eroded
    out = binary_fill_holes(out)
    return out


def nissl_parametric_mask(
    crop_rgb: np.ndarray,
    *,
    blur_sigma_frac: float = 0.015,
    open_frac: float = 0.003,
    close_frac: float = 0.012,
    local_sigma_frac: float = 0.004,
    grow_quantile: float = 0.08,
    grow_scale: float = 0.85,
    grow_frac: float = 0.004,
    final_close_frac: float = 0.006,
    post_open_k: int = 0,
    post_erode_k: int = 0,
    post_keep_frac: float = 0.92,
) -> np.ndarray:
    h, w = crop_rgb.shape[:2]
    score, channels = compute_stain_score(crop_rgb, "nissl")
    artifact = detect_border_artifacts(score, channels["nonwhite"], channels["sat"])
    score_clean = score.copy()
    score_clean[artifact > 0] = 0
    score_clean = score_clean.astype(np.uint8)

    sigma = max(11, int(round(min(h, w) * blur_sigma_frac)))
    blur = cv2.GaussianBlur(score_clean, (0, 0), sigmaX=sigma, sigmaY=sigma)
    blur_thresh = int(np.clip(cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0], 0, 255))
    candidate = (blur >= blur_thresh) & (artifact == 0)
    open_k = odd_kernel(int(round(min(h, w) * open_frac)), minimum=7)
    close_k = odd_kernel(int(round(min(h, w) * close_frac)), minimum=31)
    candidate = cv2.morphologyEx(candidate.astype(np.uint8) * 255, cv2.MORPH_OPEN, np.ones((open_k, open_k), np.uint8)) > 0
    candidate = cv2.morphologyEx(candidate.astype(np.uint8) * 255, cv2.MORPH_CLOSE, np.ones((close_k, close_k), np.uint8)) > 0
    candidate = binary_fill_holes(candidate)
    candidate = largest_component(candidate) > 0

    local_sigma = max(3, int(round(min(h, w) * local_sigma_frac)))
    local = cv2.GaussianBlur(score_clean, (0, 0), sigmaX=local_sigma, sigmaY=local_sigma)
    interior = local[candidate > 0]
    grow_thresh = max(8, int(np.quantile(interior, grow_quantile) * grow_scale)) if interior.size else 8
    grow_k = odd_kernel(int(round(min(h, w) * grow_frac)), minimum=7)
    grown = cv2.dilate(candidate.astype(np.uint8) * 255, np.ones((grow_k, grow_k), np.uint8)) > 0
    final = grown & (local >= grow_thresh) & (artifact == 0)
    final |= candidate
    final_close_k = odd_kernel(int(round(min(h, w) * final_close_frac)), minimum=11)
    final = cv2.morphologyEx(final.astype(np.uint8) * 255, cv2.MORPH_CLOSE, np.ones((final_close_k, final_close_k), np.uint8)) > 0
    final = binary_fill_holes(final)
    final = largest_component(final) > 0

    if post_open_k > 0 and post_erode_k > 0:
        final = tighten_with_area_guard(
            final,
            open_k=post_open_k,
            erode_k=post_erode_k,
            min_keep_frac=post_keep_frac,
        )
    return final > 0


def method_factory() -> dict[str, callable]:
    def gui_legacy(crop: np.ndarray) -> np.ndarray:
        return compute_auto_masks(crop, "nissl", method=MASK_PRESET_LEGACY_SIMPLE)[0] > 0

    def gui_latest(crop: np.ndarray) -> np.ndarray:
        return compute_auto_masks(crop, "nissl", method=MASK_PRESET_LATEST_CONTEXTUAL)[0] > 0

    def exp_baseline(crop: np.ndarray) -> np.ndarray:
        own_s, own_soft, support, center = _ones_support(crop)
        result = build_crop_mask_baseline(crop, own_s, own_soft, support, center, "nissl")
        return result["mask"] > 0

    def exp_soft_mgac(crop: np.ndarray) -> np.ndarray:
        own_s, own_soft, support, center = _ones_support(crop)
        result = build_crop_mask_soft_support_mgac(crop, own_s, own_soft, support, center, "nissl")
        return result["mask"] > 0

    def exp_baseline_posttight_v1(crop: np.ndarray) -> np.ndarray:
        return tighten_with_area_guard(exp_baseline(crop), open_k=3, erode_k=3, min_keep_frac=0.94)

    def exp_baseline_posttight_v2(crop: np.ndarray) -> np.ndarray:
        return tighten_with_area_guard(exp_baseline(crop), open_k=5, erode_k=3, min_keep_frac=0.92)

    def param_default(crop: np.ndarray) -> np.ndarray:
        return nissl_parametric_mask(crop)

    def param_tight_v1(crop: np.ndarray) -> np.ndarray:
        return nissl_parametric_mask(
            crop,
            open_frac=0.0035,
            close_frac=0.010,
            grow_scale=0.90,
            final_close_frac=0.0055,
            post_open_k=3,
            post_erode_k=3,
            post_keep_frac=0.94,
        )

    def param_tight_v2(crop: np.ndarray) -> np.ndarray:
        return nissl_parametric_mask(
            crop,
            open_frac=0.004,
            close_frac=0.009,
            grow_scale=0.92,
            final_close_frac=0.005,
            post_open_k=5,
            post_erode_k=3,
            post_keep_frac=0.92,
        )

    def param_tight_v3(crop: np.ndarray) -> np.ndarray:
        return nissl_parametric_mask(
            crop,
            blur_sigma_frac=0.014,
            open_frac=0.0035,
            close_frac=0.010,
            grow_quantile=0.10,
            grow_scale=0.90,
            final_close_frac=0.005,
            post_open_k=3,
            post_erode_k=3,
            post_keep_frac=0.95,
        )

    def param_balanced_v1(crop: np.ndarray) -> np.ndarray:
        return nissl_parametric_mask(
            crop,
            open_frac=0.0032,
            close_frac=0.0105,
            grow_scale=0.88,
            final_close_frac=0.0055,
            post_open_k=3,
            post_erode_k=3,
            post_keep_frac=0.95,
        )

    return {
        "gui_legacy_simple": gui_legacy,
        "gui_latest_contextual": gui_latest,
        "exp_baseline_v1": exp_baseline,
        "exp_baseline_posttight_v1": exp_baseline_posttight_v1,
        "exp_baseline_posttight_v2": exp_baseline_posttight_v2,
        "exp_soft_support_mgac": exp_soft_mgac,
        "param_default_match": param_default,
        "param_balanced_v1": param_balanced_v1,
        "param_tight_v1": param_tight_v1,
        "param_tight_v2": param_tight_v2,
        "param_tight_v3": param_tight_v3,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sample-ids", nargs="*", default=[])
    parser.add_argument("--labels", nargs="*", default=[])
    parser.add_argument("--scale", type=float, default=0.5)
    parser.add_argument("--methods", nargs="*", default=[])
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
    methods = method_factory()
    if args.methods:
        wanted = set(args.methods)
        methods = {k: v for k, v in methods.items() if k in wanted}

    rows: list[dict] = []
    aggregate: dict[str, list[dict]] = {name: [] for name in methods}
    for idx, item in enumerate(items, start=1):
        print(f"[{idx}/{len(items)}] {item.label}", flush=True)
        local_regions = region_slice_masks(item.gt_mask)
        for method_name, fn in methods.items():
            pred = fn(item.crop_rgb)
            rm = region_metrics(pred, item.gt_mask)
            cm = contour_metrics(pred, item.gt_mask)
            leak = leakage_metrics(pred, item.gt_mask)
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
            0.32 * mean_dice
            + 0.14 * mean_iou
            + 0.14 * mean_precision
            + 0.22 * mean_boundary_f1
            + 0.12 * hd95_term
            - 0.10 * abs(mean_area_ratio - 1.0)
            - 0.12 * mean_leak
        )
        summary[method_name] = {
            "count": len(method_rows),
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
            "mean_top_recall": float(np.mean([r["top_recall"] for r in method_rows])),
            "mean_middle_recall": float(np.mean([r["middle_recall"] for r in method_rows])),
            "mean_bottom_recall": float(np.mean([r["bottom_recall"] for r in method_rows])),
            "mean_left_recall": float(np.mean([r["left_recall"] for r in method_rows])),
            "mean_center_recall": float(np.mean([r["center_recall"] for r in method_rows])),
            "mean_right_recall": float(np.mean([r["right_recall"] for r in method_rows])),
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
        "# Nissl Mask Strategy Search",
        "",
        f"GT sections evaluated: {len(items)}",
        f"Working scale: {float(args.scale):.3f}",
        "",
        "Composite ranking emphasizes:",
        "- overlap quality (Dice/IoU)",
        "- boundary fit (boundary F1, HD95)",
        "- leakage suppression",
        "- area ratio staying near GT",
        "",
    ]
    for rank, (method_name, stats) in enumerate(ranked, start=1):
        lines.extend(
            [
                f"## {rank}. {method_name}",
                "",
                f"- composite_score: {stats['composite_score']:.4f}",
                f"- mean_dice: {stats['mean_dice']:.4f}",
                f"- mean_iou: {stats['mean_iou']:.4f}",
                f"- mean_precision: {stats['mean_precision']:.4f}",
                f"- mean_recall: {stats['mean_recall']:.4f}",
                f"- mean_boundary_f1_tol64: {stats['mean_boundary_f1_tol64']:.4f}",
                f"- mean_hd95_px_finite: {stats['mean_hd95_px_finite']}",
                f"- mean_fp_over_gt_area: {stats['mean_fp_over_gt_area']:.4f}",
                f"- mean_pred_to_gt_area_ratio: {stats['mean_pred_to_gt_area_ratio']:.4f}",
                f"- mean_top/middle/bottom_recall: {stats['mean_top_recall']:.4f} / {stats['mean_middle_recall']:.4f} / {stats['mean_bottom_recall']:.4f}",
                f"- mean_left/center/right_recall: {stats['mean_left_recall']:.4f} / {stats['mean_center_recall']:.4f} / {stats['mean_right_recall']:.4f}",
                f"- mean_boundary/core_recall: {stats['mean_boundary_recall']:.4f} / {stats['mean_core_recall']:.4f}",
                "",
            ]
        )
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import binary_fill_holes, binary_propagation

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
        "tp": tp,
        "fp": fp,
        "fn": fn,
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


def mask_centroid_xy(mask: np.ndarray) -> tuple[float, float] | None:
    ys, xs = np.where(mask)
    if xs.size == 0:
        return None
    return float(xs.mean()), float(ys.mean())


def largest_component(mask: np.ndarray) -> np.ndarray:
    mask_u8 = mask.astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, 8)
    if num <= 1:
        return mask.astype(bool)
    best_idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == best_idx


def run_candidate_center_baseline(crop_rgb: np.ndarray, *, candidate_mask: np.ndarray | None = None, **params) -> np.ndarray:
    h, w = crop_rgb.shape[:2]
    support = np.ones((h, w), dtype=bool)
    if candidate_mask is None:
        candidate_mask = _simple_compute_auto_masks(crop_rgb, "gallyas")[0] > 0
    center_mask = largest_component(candidate_mask)
    center = mask_centroid_xy(center_mask)
    if center is None:
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


def fallback_if_too_small(mask: np.ndarray, fallback: np.ndarray, *, min_frac_of_fallback: float) -> np.ndarray:
    mask = mask.astype(bool)
    fallback = fallback.astype(bool)
    if not mask.any():
        return fallback
    fallback_area = int(fallback.sum())
    if fallback_area <= 0:
        return mask
    if int(mask.sum()) < int(round(fallback_area * min_frac_of_fallback)):
        return fallback
    return mask


def retain_core_overlapping_components(mask: np.ndarray, core: np.ndarray, *, overlap_frac: float = 0.03) -> np.ndarray:
    mask_u8 = mask.astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, 8)
    if num <= 1:
        return mask
    keep = np.zeros_like(mask, dtype=bool)
    for idx in range(1, num):
        comp = labels == idx
        area = max(1, int(stats[idx, cv2.CC_STAT_AREA]))
        overlap = int((comp & core).sum())
        if overlap >= max(16, int(round(area * overlap_frac))):
            keep |= comp
    return keep


def hybrid_reconstruct(simple_mask: np.ndarray, core_mask: np.ndarray, *, erode_k: int, core_dilate_k: int, overlap_frac: float, final_close_k: int) -> np.ndarray:
    simple_mask = simple_mask.astype(bool)
    core_mask = core_mask.astype(bool)
    if not simple_mask.any():
        return simple_mask
    core_dil = cv2.dilate(core_mask.astype(np.uint8) * 255, np.ones((core_dilate_k, core_dilate_k), np.uint8), iterations=1) > 0
    shrunken = cv2.erode(simple_mask.astype(np.uint8) * 255, np.ones((erode_k, erode_k), np.uint8), iterations=1) > 0
    seed = retain_core_overlapping_components(shrunken, core_dil, overlap_frac=overlap_frac)
    if not seed.any():
        seed = core_mask & simple_mask
    if not seed.any():
        seed = core_mask
    if not seed.any():
        seed = simple_mask
    recon = binary_propagation(seed, mask=simple_mask)
    recon = cv2.morphologyEx(recon.astype(np.uint8) * 255, cv2.MORPH_CLOSE, np.ones((final_close_k, final_close_k), np.uint8)) > 0
    recon = binary_fill_holes(recon)
    return recon


def method_factory() -> dict[str, callable]:
    cache: dict[tuple[str, int], np.ndarray] = {}

    def key_for(crop: np.ndarray, name: str) -> tuple[str, int]:
        return (name, int(crop.__array_interface__["data"][0]))

    def cached(name: str, crop: np.ndarray, fn: callable) -> np.ndarray:
        key = key_for(crop, name)
        got = cache.get(key)
        if got is None:
            got = fn().astype(bool)
            cache[key] = got
        return got

    def legacy(crop: np.ndarray) -> np.ndarray:
        return cached(
            "legacy_simple",
            crop,
            lambda: compute_auto_masks(crop, "gallyas", method=MASK_PRESET_LEGACY_SIMPLE)[0] > 0,
        )

    def simple(crop: np.ndarray) -> np.ndarray:
        return cached(
            "simple_conservative",
            crop,
            lambda: compute_auto_masks(crop, "gallyas", method=MASK_PRESET_LATEST_CONTEXTUAL)[0] > 0,
        )

    def simple_tight_v1(crop: np.ndarray) -> np.ndarray:
        return cached(
            "simple_tight_v1",
            crop,
            lambda: tighten_with_area_guard(simple(crop), open_k=3, erode_k=3, min_keep_frac=0.90),
        )

    def simple_tight_v2(crop: np.ndarray) -> np.ndarray:
        return cached(
            "simple_tight_v2",
            crop,
            lambda: tighten_with_area_guard(simple(crop), open_k=5, erode_k=3, min_keep_frac=0.88),
        )

    def center_default(crop: np.ndarray) -> np.ndarray:
        return cached(
            "crop_center_default2comp",
            crop,
            lambda: run_crop_center_baseline(crop, gallyas_max_components=2),
        )

    def center_loose(crop: np.ndarray) -> np.ndarray:
        return cached(
            "crop_center_loose2comp",
            crop,
            lambda: run_crop_center_baseline(
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
        )

    def candidate_center_default(crop: np.ndarray) -> np.ndarray:
        return cached(
            "candidate_center_default2comp",
            crop,
            lambda: run_candidate_center_baseline(crop, candidate_mask=simple_tight_v1(crop), gallyas_max_components=2),
        )

    def hybrid_default_k5_o03(crop: np.ndarray) -> np.ndarray:
        return cached(
            "hybrid_default_k5_o03",
            crop,
            lambda: hybrid_reconstruct(simple(crop), center_default(crop), erode_k=5, core_dilate_k=21, overlap_frac=0.03, final_close_k=9),
        )

    def hybrid_default_k7_o03(crop: np.ndarray) -> np.ndarray:
        return cached(
            "hybrid_default_k7_o03",
            crop,
            lambda: hybrid_reconstruct(simple(crop), center_default(crop), erode_k=7, core_dilate_k=21, overlap_frac=0.03, final_close_k=9),
        )

    def hybrid_default_k7_o03_posttight_v1(crop: np.ndarray) -> np.ndarray:
        return cached(
            "hybrid_default_k7_o03_posttight_v1",
            crop,
            lambda: tighten_with_area_guard(hybrid_default_k7_o03(crop), open_k=3, erode_k=3, min_keep_frac=0.92),
        )

    def hybrid_default_k7_o03_posttight_v2(crop: np.ndarray) -> np.ndarray:
        return cached(
            "hybrid_default_k7_o03_posttight_v2",
            crop,
            lambda: tighten_with_area_guard(hybrid_default_k7_o03(crop), open_k=5, erode_k=3, min_keep_frac=0.90),
        )

    def hybrid_default_k5_o05(crop: np.ndarray) -> np.ndarray:
        return cached(
            "hybrid_default_k5_o05",
            crop,
            lambda: hybrid_reconstruct(simple(crop), center_default(crop), erode_k=5, core_dilate_k=21, overlap_frac=0.05, final_close_k=9),
        )

    def hybrid_tightcand_k7_o03(crop: np.ndarray) -> np.ndarray:
        return cached(
            "hybrid_tightcand_k7_o03",
            crop,
            lambda: hybrid_reconstruct(simple_tight_v1(crop), center_default(crop), erode_k=7, core_dilate_k=21, overlap_frac=0.03, final_close_k=9),
        )

    def hybrid_tightcand_k5_o05(crop: np.ndarray) -> np.ndarray:
        return cached(
            "hybrid_tightcand_k5_o05",
            crop,
            lambda: hybrid_reconstruct(simple_tight_v2(crop), center_default(crop), erode_k=5, core_dilate_k=17, overlap_frac=0.05, final_close_k=7),
        )

    def hybrid_loose_k5_o03(crop: np.ndarray) -> np.ndarray:
        return cached(
            "hybrid_loose_k5_o03",
            crop,
            lambda: hybrid_reconstruct(simple(crop), center_loose(crop), erode_k=5, core_dilate_k=21, overlap_frac=0.03, final_close_k=9),
        )

    def hybrid_loose_k7_o03(crop: np.ndarray) -> np.ndarray:
        return cached(
            "hybrid_loose_k7_o03",
            crop,
            lambda: hybrid_reconstruct(simple(crop), center_loose(crop), erode_k=7, core_dilate_k=21, overlap_frac=0.03, final_close_k=9),
        )

    def hybrid_loose_k5_o05(crop: np.ndarray) -> np.ndarray:
        return cached(
            "hybrid_loose_k5_o05",
            crop,
            lambda: hybrid_reconstruct(simple(crop), center_loose(crop), erode_k=5, core_dilate_k=21, overlap_frac=0.05, final_close_k=9),
        )

    def hybrid_guard65_tightfallback(crop: np.ndarray) -> np.ndarray:
        return cached(
            "hybrid_guard65_tightfallback",
            crop,
            lambda: fallback_if_too_small(hybrid_default_k7_o03(crop), simple_tight_v1(crop), min_frac_of_fallback=0.65),
        )

    def hybrid_guard55_tightfallback(crop: np.ndarray) -> np.ndarray:
        return cached(
            "hybrid_guard55_tightfallback",
            crop,
            lambda: fallback_if_too_small(hybrid_default_k7_o03(crop), simple_tight_v1(crop), min_frac_of_fallback=0.55),
        )

    def hybrid_candcenter_k7_o03(crop: np.ndarray) -> np.ndarray:
        return cached(
            "hybrid_candcenter_k7_o03",
            crop,
            lambda: hybrid_reconstruct(simple(crop), candidate_center_default(crop), erode_k=7, core_dilate_k=21, overlap_frac=0.03, final_close_k=9),
        )

    def hybrid_candcenter_posttight_v1(crop: np.ndarray) -> np.ndarray:
        return cached(
            "hybrid_candcenter_posttight_v1",
            crop,
            lambda: tighten_with_area_guard(hybrid_candcenter_k7_o03(crop), open_k=3, erode_k=3, min_keep_frac=0.92),
        )

    return {
        "legacy_simple": legacy,
        "simple_conservative": simple,
        "simple_tight_v1": simple_tight_v1,
        "simple_tight_v2": simple_tight_v2,
        "crop_center_default2comp": center_default,
        "crop_center_loose2comp": center_loose,
        "candidate_center_default2comp": candidate_center_default,
        "hybrid_default_k5_o03": hybrid_default_k5_o03,
        "hybrid_default_k7_o03": hybrid_default_k7_o03,
        "hybrid_default_k7_o03_posttight_v1": hybrid_default_k7_o03_posttight_v1,
        "hybrid_default_k7_o03_posttight_v2": hybrid_default_k7_o03_posttight_v2,
        "hybrid_default_k5_o05": hybrid_default_k5_o05,
        "hybrid_tightcand_k7_o03": hybrid_tightcand_k7_o03,
        "hybrid_tightcand_k5_o05": hybrid_tightcand_k5_o05,
        "hybrid_loose_k5_o03": hybrid_loose_k5_o03,
        "hybrid_loose_k7_o03": hybrid_loose_k7_o03,
        "hybrid_loose_k5_o05": hybrid_loose_k5_o05,
        "hybrid_guard65_tightfallback": hybrid_guard65_tightfallback,
        "hybrid_guard55_tightfallback": hybrid_guard55_tightfallback,
        "hybrid_candcenter_k7_o03": hybrid_candcenter_k7_o03,
        "hybrid_candcenter_posttight_v1": hybrid_candcenter_posttight_v1,
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
        methods = {k: v for k, v in methods.items() if k in set(args.methods)}

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
        "# Myelin Mask Strategy Search",
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

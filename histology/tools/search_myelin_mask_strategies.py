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
from skimage.filters import threshold_otsu
from skimage.filters.rank import entropy as rank_entropy
from skimage.morphology import disk

Image.MAX_IMAGE_PIXELS = None

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from histology.gui_mvp.hitl_gui.pipeline_adapters.segmentation_adapter import (  # noqa: E402
    MASK_PRESET_HYBRID_BALANCED,
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


def component_set_select(
    candidate_mask: np.ndarray,
    *,
    score_core: np.ndarray,
    support_mask: np.ndarray | None = None,
    max_components: int = 2,
    min_area_frac: float = 0.06,
    bridge_erode_k: int = 3,
    core_dilate_k: int = 15,
    border_band_frac: float = 0.03,
    border_penalty: float = 0.75,
    min_score: float = 0.08,
    final_close_k: int = 7,
) -> np.ndarray:
    candidate_mask = candidate_mask.astype(bool)
    score_core = score_core.astype(bool)
    support_mask = candidate_mask if support_mask is None else support_mask.astype(bool)
    if not candidate_mask.any():
        return candidate_mask

    h, w = candidate_mask.shape[:2]
    orig_area = max(1, int(candidate_mask.sum()))
    border_band = max(3, int(round(min(h, w) * border_band_frac)))
    border = np.zeros_like(candidate_mask, dtype=bool)
    border[:border_band, :] = True
    border[-border_band:, :] = True
    border[:, :border_band] = True
    border[:, -border_band:] = True

    core_dil = cv2.dilate(
        score_core.astype(np.uint8) * 255,
        np.ones((core_dilate_k, core_dilate_k), np.uint8),
        iterations=1,
    ) > 0

    if bridge_erode_k > 1:
        bridge_cut = cv2.erode(
            candidate_mask.astype(np.uint8) * 255,
            np.ones((bridge_erode_k, bridge_erode_k), np.uint8),
            iterations=1,
        ) > 0
    else:
        bridge_cut = candidate_mask.copy()
    if not bridge_cut.any():
        bridge_cut = candidate_mask.copy()

    num, labels, stats, _ = cv2.connectedComponentsWithStats(bridge_cut.astype(np.uint8), 8)
    components: list[tuple[float, np.ndarray]] = []
    core_cent = mask_centroid_xy(score_core)
    support_cent = mask_centroid_xy(support_mask)
    diag = float(math.hypot(h, w)) or 1.0

    for idx in range(1, num):
        seed = labels == idx
        comp = binary_propagation(seed, mask=candidate_mask)
        area = int(comp.sum())
        if area <= 0:
            continue
        area_frac = area / orig_area
        overlap_core = int((comp & core_dil).sum()) / area
        overlap_support = int((comp & support_mask).sum()) / area
        touch_border = int((comp & border).sum()) / area
        x1, y1, x2, y2 = tight_bbox(comp)
        bbox_area = max(1, (x2 - x1) * (y2 - y1))
        compactness = area / bbox_area
        cxcy = mask_centroid_xy(comp)
        if cxcy is None:
            dist_core = 1.0
            dist_support = 1.0
        else:
            dist_core = (
                math.hypot(cxcy[0] - core_cent[0], cxcy[1] - core_cent[1]) / diag
                if core_cent is not None
                else 1.0
            )
            dist_support = (
                math.hypot(cxcy[0] - support_cent[0], cxcy[1] - support_cent[1]) / diag
                if support_cent is not None
                else 1.0
            )
        score = (
            1.20 * overlap_core
            + 0.90 * overlap_support
            + 0.55 * area_frac
            + 0.25 * compactness
            - 0.80 * dist_core
            - 0.35 * dist_support
            - border_penalty * touch_border
        )
        if area_frac >= min_area_frac or overlap_core > 0.02 or score >= min_score:
            components.append((score, comp))

    if not components:
        return candidate_mask

    components.sort(key=lambda x: x[0], reverse=True)
    keep = np.zeros_like(candidate_mask, dtype=bool)
    taken = 0
    for score, comp in components:
        if taken == 0 or (taken < max_components and score >= min_score):
            keep |= comp
            taken += 1
        if taken >= max_components:
            break
    if not keep.any():
        keep = components[0][1]

    if final_close_k > 1:
        keep = cv2.morphologyEx(
            keep.astype(np.uint8) * 255,
            cv2.MORPH_CLOSE,
            np.ones((final_close_k, final_close_k), np.uint8),
        ) > 0
    keep = binary_fill_holes(keep)
    return keep


def _edge_touch_runs(mask: np.ndarray, side: str) -> list[tuple[int, int]]:
    if side == "left":
        vec = mask[:, 0]
    elif side == "right":
        vec = mask[:, -1]
    elif side == "top":
        vec = mask[0, :]
    else:
        vec = mask[-1, :]
    vec = np.asarray(vec, dtype=bool)
    if not vec.any():
        return []
    runs: list[tuple[int, int]] = []
    start = None
    for idx, val in enumerate(vec):
        if val and start is None:
            start = idx
        elif not val and start is not None:
            runs.append((start, idx))
            start = None
    if start is not None:
        runs.append((start, len(vec)))
    return runs


def _edge_touch_vector(mask: np.ndarray, side: str, *, depth: int = 1) -> np.ndarray:
    depth = max(1, int(depth))
    if side == "left":
        vec = np.any(mask[:, :depth], axis=1)
    elif side == "right":
        vec = np.any(mask[:, -depth:], axis=1)
    elif side == "top":
        vec = np.any(mask[:depth, :], axis=0)
    else:
        vec = np.any(mask[-depth:, :], axis=0)
    return np.asarray(vec, dtype=bool)


def _dilate_bool_1d(vec: np.ndarray, radius: int) -> np.ndarray:
    vec = np.asarray(vec, dtype=bool)
    if radius <= 0 or not vec.any():
        return vec
    kernel = np.ones(2 * radius + 1, dtype=np.uint8)
    dil = np.convolve(vec.astype(np.uint8), kernel, mode="same")
    return dil > 0


def fill_edge_touch_strips(
    mask: np.ndarray,
    support_mask: np.ndarray,
    *,
    strip_frac: float = 0.06,
    min_runs: int = 2,
    min_span_frac: float = 0.12,
    close_k: int = 5,
) -> np.ndarray:
    mask = mask.astype(bool)
    support_mask = support_mask.astype(bool)
    if not mask.any() or not support_mask.any():
        return mask

    h, w = mask.shape[:2]
    out = mask.copy()
    strip_w = max(4, int(round(w * strip_frac)))
    min_span = max(8, int(round(h * min_span_frac)))

    for side in ("left", "right"):
        runs = _edge_touch_runs(out, side)
        if len(runs) < min_runs:
            continue
        span_lo = min(r[0] for r in runs)
        span_hi = max(r[1] for r in runs)
        if (span_hi - span_lo) < min_span:
            continue

        if side == "left":
            sup = support_mask[:, :strip_w]
            seed = np.zeros_like(sup, dtype=bool)
            seed[:, 0] = sup[:, 0]
            edge_conn = binary_propagation(seed, mask=sup)
            fill = np.zeros_like(out, dtype=bool)
            fill[span_lo:span_hi, :strip_w] = edge_conn[span_lo:span_hi, :]
        else:
            sup = support_mask[:, w - strip_w :]
            seed = np.zeros_like(sup, dtype=bool)
            seed[:, -1] = sup[:, -1]
            edge_conn = binary_propagation(seed, mask=sup)
            fill = np.zeros_like(out, dtype=bool)
            fill[span_lo:span_hi, w - strip_w :] = edge_conn[span_lo:span_hi, :]

        out |= fill

    if close_k > 1:
        out = cv2.morphologyEx(
            out.astype(np.uint8) * 255,
            cv2.MORPH_CLOSE,
            np.ones((close_k, close_k), np.uint8),
        ) > 0
    out = binary_fill_holes(out)
    return out


def edge_support_augment(
    mask: np.ndarray,
    support_mask: np.ndarray,
    score_map: np.ndarray,
    *,
    strip_frac: float = 0.08,
    inner_frac: float = 0.16,
    score_q: float = 0.35,
    min_row_frac: float = 0.05,
    row_pad_frac: float = 0.03,
    close_k: int = 5,
) -> np.ndarray:
    mask = mask.astype(bool)
    support_mask = support_mask.astype(bool)
    if not mask.any() or not support_mask.any():
        return mask

    h, w = mask.shape[:2]
    strip_w = max(6, int(round(w * strip_frac)))
    inner_w = max(strip_w + 2, int(round(w * inner_frac)))
    min_rows = max(8, int(round(h * min_row_frac)))
    row_pad = max(2, int(round(h * row_pad_frac)))

    vals = score_map[support_mask]
    score_th = float(np.quantile(vals, score_q)) if vals.size else 0.0
    aug = mask.copy()

    for side in ("left", "right"):
        if side == "left":
            strip_support = support_mask[:, :strip_w]
            inner_mask = mask[:, :inner_w]
            strip_score = score_map[:, :strip_w]
            corridor = np.zeros_like(mask, dtype=bool)
            corridor[:, :inner_w] = True
        else:
            strip_support = support_mask[:, -strip_w:]
            inner_mask = mask[:, -inner_w:]
            strip_score = score_map[:, -strip_w:]
            corridor = np.zeros_like(mask, dtype=bool)
            corridor[:, -inner_w:] = True

        support_rows = np.any(strip_support & (strip_score >= score_th), axis=1)
        inner_rows = np.any(inner_mask, axis=1)
        active_rows = support_rows & _dilate_bool_1d(inner_rows, row_pad)
        if int(active_rows.sum()) < min_rows:
            continue

        row_mask = np.zeros_like(mask, dtype=bool)
        row_mask[active_rows, :] = True
        corridor_support = support_mask & corridor & row_mask
        if not corridor_support.any():
            continue
        aug |= corridor_support

    if close_k > 1:
        aug = cv2.morphologyEx(
            aug.astype(np.uint8) * 255,
            cv2.MORPH_CLOSE,
            np.ones((close_k, close_k), np.uint8),
        ) > 0
    aug = binary_fill_holes(aug)
    return aug


def edge_run_completion(
    mask: np.ndarray,
    support_mask: np.ndarray,
    score_map: np.ndarray,
    *,
    strip_frac: float = 0.08,
    depth: int = 2,
    min_runs: int = 2,
    min_span_frac: float = 0.12,
    row_pad_frac: float = 0.03,
    score_q: float = 0.20,
    close_k: int = 5,
) -> np.ndarray:
    mask = mask.astype(bool)
    support_mask = support_mask.astype(bool)
    if not mask.any() or not support_mask.any():
        return mask

    h, w = mask.shape[:2]
    strip_w = max(6, int(round(w * strip_frac)))
    min_span = max(8, int(round(h * min_span_frac)))
    row_pad = max(2, int(round(h * row_pad_frac)))
    vals = score_map[support_mask]
    score_th = float(np.quantile(vals, score_q)) if vals.size else 0.0

    out = mask.copy()
    support_score = support_mask & (score_map >= score_th)

    for side in ("left", "right"):
        touch_vec = _edge_touch_vector(support_score, side, depth=depth)
        runs = _edge_touch_runs(touch_vec[:, None] if touch_vec.ndim == 1 and side in ("left", "right") else touch_vec, "left") if False else None
        # direct 1D run extraction
        runs_1d: list[tuple[int, int]] = []
        start = None
        for idx, val in enumerate(touch_vec):
            if val and start is None:
                start = idx
            elif (not val) and start is not None:
                runs_1d.append((start, idx))
                start = None
        if start is not None:
            runs_1d.append((start, len(touch_vec)))
        if len(runs_1d) < min_runs:
            continue
        span_lo = min(r[0] for r in runs_1d)
        span_hi = max(r[1] for r in runs_1d)
        if (span_hi - span_lo) < min_span:
            continue
        span_lo = max(0, span_lo - row_pad)
        span_hi = min(h, span_hi + row_pad)

        for y in range(span_lo, span_hi):
            if side == "left":
                support_cols = np.where(support_score[y, :strip_w])[0]
                mask_cols = np.where(mask[y, :strip_w])[0]
                cols = support_cols if support_cols.size else mask_cols
                if cols.size:
                    x_end = int(cols.max()) + 1
                    out[y, :x_end] = True
            else:
                support_cols = np.where(support_score[y, w - strip_w :])[0]
                mask_cols = np.where(mask[y, w - strip_w :])[0]
                cols = support_cols if support_cols.size else mask_cols
                if cols.size:
                    x_start = w - strip_w + int(cols.min())
                    out[y, x_start:] = True

    if close_k > 1:
        out = cv2.morphologyEx(
            out.astype(np.uint8) * 255,
            cv2.MORPH_CLOSE,
            np.ones((close_k, close_k), np.uint8),
        ) > 0
    out = binary_fill_holes(out)
    return out


def top_envelope_from_mask(
    mask: np.ndarray,
    *,
    smooth_frac: float = 0.04,
    min_valid_frac: float = 0.20,
) -> tuple[np.ndarray | None, np.ndarray]:
    mask = mask.astype(bool)
    h, w = mask.shape[:2]
    env = np.full(w, np.nan, dtype=np.float32)
    for x in range(w):
        ys = np.where(mask[:, x])[0]
        if ys.size:
            env[x] = float(ys.min())
    valid = np.isfinite(env)
    if int(valid.sum()) < max(8, int(round(w * min_valid_frac))):
        return None, valid

    xs = np.arange(w, dtype=np.float32)
    env_interp = env.copy()
    env_interp[~valid] = np.interp(xs[~valid], xs[valid], env[valid])

    k = max(5, int(round(w * smooth_frac)))
    if k % 2 == 0:
        k += 1
    env_smooth = cv2.GaussianBlur(env_interp.reshape(1, -1), (k, 1), 0).reshape(-1)

    valid_idx = np.where(valid)[0]
    left_idx = valid_idx[: min(12, len(valid_idx))]
    right_idx = valid_idx[-min(12, len(valid_idx)) :]
    if left_idx.size >= 2:
        coef = np.polyfit(left_idx.astype(np.float32), env[left_idx], 1)
        left_x = np.arange(0, int(valid_idx[0]), dtype=np.float32)
        if left_x.size:
            env_smooth[left_x.astype(int)] = np.polyval(coef, left_x)
    if right_idx.size >= 2:
        coef = np.polyfit(right_idx.astype(np.float32), env[right_idx], 1)
        right_x = np.arange(int(valid_idx[-1]) + 1, w, dtype=np.float32)
        if right_x.size:
            env_smooth[right_x.astype(int)] = np.polyval(coef, right_x)

    env_smooth = np.clip(env_smooth, 0, h - 1).astype(np.float32)
    return env_smooth, valid


def top_envelope_corridor(
    envelope: np.ndarray,
    shape: tuple[int, int],
    *,
    band_frac: float = 0.08,
    lower_slack_frac: float = 0.05,
) -> np.ndarray:
    h, w = shape[:2]
    band = max(6, int(round(h * band_frac)))
    lower = max(4, int(round(h * lower_slack_frac)))
    corr = np.zeros((h, w), dtype=bool)
    for x in range(w):
        y = int(round(float(envelope[x])))
        y0 = max(0, y - band // 2)
        y1 = min(h, y + band // 2 + lower)
        corr[y0:y1, x] = True
    return corr


def top_envelope_lateral_completion(
    mask: np.ndarray,
    support_mask: np.ndarray,
    score_map: np.ndarray,
    nonwhite: np.ndarray,
    *,
    band_frac: float = 0.08,
    lower_slack_frac: float = 0.05,
    score_q: float = 0.22,
    nonwhite_min: int = 10,
    close_k: int = 5,
) -> np.ndarray:
    mask = mask.astype(bool)
    support_mask = support_mask.astype(bool)
    if not mask.any() or not support_mask.any():
        return mask

    env, valid = top_envelope_from_mask(mask)
    if env is None:
        return mask
    corridor = top_envelope_corridor(env, mask.shape, band_frac=band_frac, lower_slack_frac=lower_slack_frac)
    vals = score_map[support_mask]
    score_th = float(np.quantile(vals, score_q)) if vals.size else 0.0
    weak_support = (support_mask & corridor & (score_map >= score_th)) | (corridor & (nonwhite >= nonwhite_min) & mask)
    out = binary_propagation(mask, mask=(mask | weak_support))
    if close_k > 1:
        out = cv2.morphologyEx(
            out.astype(np.uint8) * 255,
            cv2.MORPH_CLOSE,
            np.ones((close_k, close_k), np.uint8),
        ) > 0
    out = binary_fill_holes(out)
    return out


def top_envelope_bridge_completion(
    mask: np.ndarray,
    support_mask: np.ndarray,
    score_map: np.ndarray,
    nonwhite: np.ndarray,
    *,
    band_frac: float = 0.08,
    lower_slack_frac: float = 0.05,
    score_q: float = 0.16,
    nonwhite_min: int = 8,
    min_bridge_cols_frac: float = 0.08,
    close_k: int = 7,
) -> np.ndarray:
    mask = mask.astype(bool)
    support_mask = support_mask.astype(bool)
    if not mask.any() or not support_mask.any():
        return mask

    env, valid = top_envelope_from_mask(mask)
    if env is None:
        return mask
    h, w = mask.shape[:2]
    corridor = top_envelope_corridor(env, mask.shape, band_frac=band_frac, lower_slack_frac=lower_slack_frac)
    vals = score_map[support_mask]
    score_th = float(np.quantile(vals, score_q)) if vals.size else 0.0
    support_corr = support_mask & corridor & ((score_map >= score_th) | (nonwhite >= nonwhite_min))

    col_has = np.any(mask & corridor, axis=0)
    first = int(np.argmax(col_has)) if col_has.any() else 0
    last = int(w - 1 - np.argmax(col_has[::-1])) if col_has.any() else w - 1
    min_bridge_cols = max(8, int(round(w * min_bridge_cols_frac)))
    if last - first >= min_bridge_cols:
        bridge_region = np.zeros_like(mask, dtype=bool)
        bridge_region[:, first : last + 1] = True
        support_corr |= corridor & bridge_region & (nonwhite >= nonwhite_min)

    out = binary_propagation(mask, mask=(mask | support_corr))
    if close_k > 1:
        out = cv2.morphologyEx(
            out.astype(np.uint8) * 255,
            cv2.MORPH_CLOSE,
            np.ones((close_k, close_k), np.uint8),
        ) > 0
    out = binary_fill_holes(out)
    return out


def edge_aware_support_mask(
    base_support: np.ndarray,
    reference_mask: np.ndarray,
    score_map: np.ndarray,
    nonwhite: np.ndarray,
    *,
    band_frac: float = 0.08,
    lower_slack_frac: float = 0.05,
    lateral_strip_frac: float = 0.22,
    inner_anchor_frac: float = 0.18,
    score_q: float = 0.18,
    nonwhite_min: int = 10,
    row_pad_frac: float = 0.03,
    close_k: int = 5,
) -> np.ndarray:
    base_support = base_support.astype(bool)
    reference_mask = reference_mask.astype(bool)
    if not base_support.any():
        return base_support

    h, w = base_support.shape[:2]
    env, _ = top_envelope_from_mask(reference_mask)
    if env is None:
        return base_support

    corridor = top_envelope_corridor(
        env,
        base_support.shape,
        band_frac=band_frac,
        lower_slack_frac=lower_slack_frac,
    )
    strip_w = max(6, int(round(w * lateral_strip_frac)))
    inner_w = max(strip_w + 2, int(round(w * inner_anchor_frac)))
    row_pad = max(2, int(round(h * row_pad_frac)))

    vals = score_map[base_support]
    score_th = float(np.quantile(vals, score_q)) if vals.size else 0.0
    out = base_support.copy()

    for side in ("left", "right"):
        if side == "left":
            outer = np.zeros_like(base_support, dtype=bool)
            outer[:, :strip_w] = True
            inner = np.zeros_like(base_support, dtype=bool)
            inner[:, :inner_w] = True
        else:
            outer = np.zeros_like(base_support, dtype=bool)
            outer[:, -strip_w:] = True
            inner = np.zeros_like(base_support, dtype=bool)
            inner[:, -inner_w:] = True

        inner_anchor_rows = np.any((base_support | reference_mask) & inner, axis=1)
        active_rows = _dilate_bool_1d(inner_anchor_rows, row_pad)
        if not active_rows.any():
            continue

        row_mask = np.zeros_like(base_support, dtype=bool)
        row_mask[active_rows, :] = True
        aug = outer & corridor & row_mask & (nonwhite >= nonwhite_min) & (score_map >= score_th)
        out |= aug

    if close_k > 1:
        out = cv2.morphologyEx(
            out.astype(np.uint8) * 255,
            cv2.MORPH_CLOSE,
            np.ones((close_k, close_k), np.uint8),
        ) > 0
    out = binary_fill_holes(out)
    return out


def edge_aware_support_bridge_mask(
    base_support: np.ndarray,
    reference_mask: np.ndarray,
    score_map: np.ndarray,
    nonwhite: np.ndarray,
    *,
    band_frac: float = 0.08,
    lower_slack_frac: float = 0.05,
    lateral_strip_frac: float = 0.22,
    inner_anchor_frac: float = 0.18,
    score_q: float = 0.14,
    nonwhite_min: int = 8,
    row_pad_frac: float = 0.03,
    min_bridge_span_frac: float = 0.10,
    close_k: int = 7,
) -> np.ndarray:
    base_support = base_support.astype(bool)
    reference_mask = reference_mask.astype(bool)
    if not base_support.any():
        return base_support

    h, w = base_support.shape[:2]
    env, _ = top_envelope_from_mask(reference_mask)
    if env is None:
        return base_support

    corridor = top_envelope_corridor(
        env,
        base_support.shape,
        band_frac=band_frac,
        lower_slack_frac=lower_slack_frac,
    )
    strip_w = max(6, int(round(w * lateral_strip_frac)))
    inner_w = max(strip_w + 2, int(round(w * inner_anchor_frac)))
    row_pad = max(2, int(round(h * row_pad_frac)))
    min_span = max(8, int(round(h * min_bridge_span_frac)))

    vals = score_map[base_support]
    score_th = float(np.quantile(vals, score_q)) if vals.size else 0.0
    support_score = (score_map >= score_th) & (nonwhite >= nonwhite_min)
    out = base_support.copy()

    for side in ("left", "right"):
        if side == "left":
            outer = np.zeros_like(base_support, dtype=bool)
            outer[:, :strip_w] = True
            inner = np.zeros_like(base_support, dtype=bool)
            inner[:, :inner_w] = True
        else:
            outer = np.zeros_like(base_support, dtype=bool)
            outer[:, -strip_w:] = True
            inner = np.zeros_like(base_support, dtype=bool)
            inner[:, -inner_w:] = True

        inner_anchor_rows = np.any((base_support | reference_mask) & inner, axis=1)
        active_rows = _dilate_bool_1d(inner_anchor_rows, row_pad)
        runs = []
        start = None
        for i, val in enumerate(active_rows):
            if val and start is None:
                start = i
            elif (not val) and start is not None:
                runs.append((start, i))
                start = None
        if start is not None:
            runs.append((start, len(active_rows)))

        for y0, y1 in runs:
            if (y1 - y0) < min_span:
                continue
            band = corridor[y0:y1, :] & outer[y0:y1, :]
            cand = support_score[y0:y1, :] & band
            if not cand.any():
                continue
            if side == "left":
                cols = np.where(np.any(cand, axis=0))[0]
                if cols.size:
                    x_end = int(cols.max()) + 1
                    out[y0:y1, :x_end] |= band[:, :x_end]
            else:
                cols = np.where(np.any(cand, axis=0))[0]
                if cols.size:
                    x_start = int(cols.min())
                    out[y0:y1, x_start:] |= band[:, x_start:]

    if close_k > 1:
        out = cv2.morphologyEx(
            out.astype(np.uint8) * 255,
            cv2.MORPH_CLOSE,
            np.ones((close_k, close_k), np.uint8),
        ) > 0
    out = binary_fill_holes(out)
    return out


def residual_score(crop_rgb: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)
    sigma = max(12, int(round(min(crop_rgb.shape[:2]) * 0.05)))
    bg = cv2.GaussianBlur(gray, (0, 0), sigmaX=float(sigma), sigmaY=float(sigma))
    return np.clip(bg.astype(np.int16) - gray.astype(np.int16), 0, 255).astype(np.uint8)


def entropy_score(crop_rgb: np.ndarray, radius: int = 5) -> np.ndarray:
    gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)
    ent = rank_entropy(gray, disk(max(1, int(radius))))
    ent = cv2.GaussianBlur(ent.astype(np.float32), (0, 0), sigmaX=1.0, sigmaY=1.0)
    if float(ent.max()) <= 1e-6:
        return np.zeros_like(gray, dtype=np.uint8)
    return cv2.normalize(ent, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def entropy_residual_candidate(
    crop_rgb: np.ndarray,
    *,
    ent_radius: int,
    ent_q: float,
    residual_scale: float,
    nonwhite_min: int,
    close_k: int,
    open_k: int,
) -> np.ndarray:
    resid = residual_score(crop_rgb)
    ent = entropy_score(crop_rgb, radius=ent_radius)
    nonwhite = (255 - crop_rgb.min(axis=2)).astype(np.uint8)

    resid_th = max(6, int(round(threshold_otsu(resid) * residual_scale)))
    resid_mask = resid >= resid_th

    ent_vals = ent[nonwhite > nonwhite_min]
    if ent_vals.size == 0:
        ent_mask = np.zeros_like(resid_mask, dtype=bool)
    else:
        ent_th = max(8, int(round(float(np.quantile(ent_vals, ent_q)))))
        ent_mask = ent >= ent_th
        ent_mask &= nonwhite >= nonwhite_min

    mask = resid_mask | ent_mask
    mask = cv2.morphologyEx(mask.astype(np.uint8) * 255, cv2.MORPH_CLOSE, np.ones((close_k, close_k), np.uint8)) > 0
    mask = cv2.morphologyEx(mask.astype(np.uint8) * 255, cv2.MORPH_OPEN, np.ones((open_k, open_k), np.uint8)) > 0
    mask = binary_fill_holes(mask)
    return mask


def hysteresis_support_reconstruct(
    support_mask: np.ndarray,
    core_score: np.ndarray,
    structural_core: np.ndarray,
    *,
    core_quantile: float,
    core_scale: float,
    overlap_frac: float,
    core_open_k: int,
    final_close_k: int,
) -> np.ndarray:
    support_mask = support_mask.astype(bool)
    structural_core = structural_core.astype(bool)
    if not support_mask.any():
        return support_mask

    vals = core_score[support_mask]
    if vals.size == 0:
        return support_mask
    core_th = max(6, int(round(float(np.quantile(vals, core_quantile)) * core_scale)))
    strong_core = (core_score >= core_th) & support_mask
    if core_open_k > 1:
        strong_core = cv2.morphologyEx(
            strong_core.astype(np.uint8) * 255,
            cv2.MORPH_OPEN,
            np.ones((core_open_k, core_open_k), np.uint8),
        ) > 0
    if strong_core.any():
        struct_dil = cv2.dilate(structural_core.astype(np.uint8) * 255, np.ones((9, 9), np.uint8), iterations=1) > 0
        seed = retain_core_overlapping_components(strong_core, struct_dil, overlap_frac=overlap_frac)
    else:
        seed = np.zeros_like(support_mask, dtype=bool)
    if not seed.any():
        seed = structural_core & support_mask
    if not seed.any():
        seed = support_mask
    recon = binary_propagation(seed, mask=support_mask)
    if final_close_k > 1:
        recon = cv2.morphologyEx(
            recon.astype(np.uint8) * 255,
            cv2.MORPH_CLOSE,
            np.ones((final_close_k, final_close_k), np.uint8),
        ) > 0
    recon = binary_fill_holes(recon)
    return recon


def _odd_kernel_from_frac(shape: tuple[int, int], frac: float, minimum: int = 3) -> int:
    k = max(minimum, int(round(min(shape[:2]) * frac)))
    return k if (k % 2 == 1) else (k + 1)


def opening_by_reconstruction(mask: np.ndarray, ksize: int) -> np.ndarray:
    mask_u8 = (mask.astype(np.uint8) * 255)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    eroded = cv2.erode(mask_u8, kernel, iterations=1)
    recon = binary_propagation(eroded > 0, mask=mask_u8 > 0)
    return np.asarray(recon, dtype=bool)


def closing_by_reconstruction(mask: np.ndarray, ksize: int) -> np.ndarray:
    inv = ~mask.astype(bool)
    opened_inv = opening_by_reconstruction(inv, ksize)
    return ~opened_inv


def reconstructive_cleanup(
    mask: np.ndarray,
    *,
    open_frac: float,
    close_frac: float,
    min_keep_frac: float,
) -> np.ndarray:
    mask = mask.astype(bool)
    if not mask.any():
        return mask
    orig_area = int(mask.sum())
    open_k = _odd_kernel_from_frac(mask.shape, open_frac, minimum=3)
    close_k = _odd_kernel_from_frac(mask.shape, close_frac, minimum=5)
    out = opening_by_reconstruction(mask, open_k)
    if out.any() and int(out.sum()) >= int(round(orig_area * min_keep_frac)):
        mask = out
    out = closing_by_reconstruction(mask, close_k)
    if out.any():
        mask = out
    mask = binary_fill_holes(mask)
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


def hybrid_reconstruct_m1(
    simple_mask: np.ndarray,
    core_mask: np.ndarray,
    *,
    erode_k: int,
    core_dilate_k: int,
    overlap_frac: float,
    candidate_open_frac: float,
    candidate_close_frac: float,
    final_open_frac: float,
    final_close_frac: float,
    min_keep_frac: float,
) -> np.ndarray:
    simple_mask = reconstructive_cleanup(
        simple_mask,
        open_frac=candidate_open_frac,
        close_frac=candidate_close_frac,
        min_keep_frac=min_keep_frac,
    )
    recon = hybrid_reconstruct(
        simple_mask,
        core_mask,
        erode_k=erode_k,
        core_dilate_k=core_dilate_k,
        overlap_frac=overlap_frac,
        final_close_k=3,
    )
    recon = reconstructive_cleanup(
        recon,
        open_frac=final_open_frac,
        close_frac=final_close_frac,
        min_keep_frac=min_keep_frac,
    )
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

    def gui_hybrid_balanced_production(crop: np.ndarray) -> np.ndarray:
        return cached(
            "gui_hybrid_balanced_production",
            crop,
            lambda: compute_auto_masks(crop, "gallyas", method=MASK_PRESET_HYBRID_BALANCED)[0] > 0,
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

    def m2_candidate_union_v1(crop: np.ndarray) -> np.ndarray:
        return cached(
            "m2_candidate_union_v1",
            crop,
            lambda: entropy_residual_candidate(
                crop,
                ent_radius=5,
                ent_q=0.68,
                residual_scale=0.96,
                nonwhite_min=18,
                close_k=9,
                open_k=3,
            ),
        )

    def m2_candidate_union_v2(crop: np.ndarray) -> np.ndarray:
        return cached(
            "m2_candidate_union_v2",
            crop,
            lambda: entropy_residual_candidate(
                crop,
                ent_radius=7,
                ent_q=0.64,
                residual_scale=0.92,
                nonwhite_min=16,
                close_k=11,
                open_k=3,
            ),
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

    def m2_hybrid_entres_v1(crop: np.ndarray) -> np.ndarray:
        return cached(
            "m2_hybrid_entres_v1",
            crop,
            lambda: hybrid_reconstruct(m2_candidate_union_v1(crop), center_default(crop), erode_k=7, core_dilate_k=21, overlap_frac=0.03, final_close_k=9),
        )

    def m2_hybrid_entres_tight_v1(crop: np.ndarray) -> np.ndarray:
        return cached(
            "m2_hybrid_entres_tight_v1",
            crop,
            lambda: hybrid_reconstruct(
                tighten_with_area_guard(m2_candidate_union_v1(crop), open_k=3, erode_k=3, min_keep_frac=0.92),
                center_default(crop),
                erode_k=7,
                core_dilate_k=21,
                overlap_frac=0.03,
                final_close_k=9,
            ),
        )

    def m2_hybrid_entres_guard_v1(crop: np.ndarray) -> np.ndarray:
        return cached(
            "m2_hybrid_entres_guard_v1",
            crop,
            lambda: fallback_if_too_small(
                hybrid_reconstruct(m2_candidate_union_v2(crop), center_default(crop), erode_k=7, core_dilate_k=21, overlap_frac=0.03, final_close_k=9),
                simple_tight_v1(crop),
                min_frac_of_fallback=0.72,
            ),
        )

    def m3_hyst_entres_v1(crop: np.ndarray) -> np.ndarray:
        return cached(
            "m3_hyst_entres_v1",
            crop,
            lambda: hysteresis_support_reconstruct(
                m2_candidate_union_v1(crop),
                residual_score(crop),
                center_default(crop),
                core_quantile=0.82,
                core_scale=1.00,
                overlap_frac=0.03,
                core_open_k=3,
                final_close_k=7,
            ),
        )

    def m3_hyst_entres_tight_v1(crop: np.ndarray) -> np.ndarray:
        return cached(
            "m3_hyst_entres_tight_v1",
            crop,
            lambda: tighten_with_area_guard(
                hysteresis_support_reconstruct(
                    m2_candidate_union_v1(crop),
                    residual_score(crop),
                    center_default(crop),
                    core_quantile=0.86,
                    core_scale=1.02,
                    overlap_frac=0.035,
                    core_open_k=3,
                    final_close_k=7,
                ),
                open_k=3,
                erode_k=3,
                min_keep_frac=0.92,
            ),
        )

    def m3_hyst_entres_guard_v1(crop: np.ndarray) -> np.ndarray:
        return cached(
            "m3_hyst_entres_guard_v1",
            crop,
            lambda: fallback_if_too_small(
                hysteresis_support_reconstruct(
                    m2_candidate_union_v2(crop),
                    residual_score(crop),
                    center_default(crop),
                    core_quantile=0.84,
                    core_scale=1.00,
                    overlap_frac=0.03,
                    core_open_k=3,
                    final_close_k=7,
                ),
                m2_hybrid_entres_tight_v1(crop),
                min_frac_of_fallback=0.70,
            ),
        )

    def m3_support_edgeaware_v1(crop: np.ndarray) -> np.ndarray:
        return cached(
            "m3_support_edgeaware_v1",
            crop,
            lambda: fallback_if_too_small(
                hysteresis_support_reconstruct(
                    edge_aware_support_mask(
                        m2_candidate_union_v2(crop),
                        binary_fill_holes(center_default(crop) | simple_tight_v1(crop)),
                        np.maximum(residual_score(crop), entropy_score(crop, radius=5)),
                        (255 - crop.min(axis=2)).astype(np.uint8),
                        band_frac=0.08,
                        lower_slack_frac=0.05,
                        lateral_strip_frac=0.22,
                        inner_anchor_frac=0.18,
                        score_q=0.18,
                        nonwhite_min=10,
                        row_pad_frac=0.03,
                        close_k=5,
                    ),
                    residual_score(crop),
                    center_default(crop),
                    core_quantile=0.84,
                    core_scale=1.00,
                    overlap_frac=0.03,
                    core_open_k=3,
                    final_close_k=7,
                ),
                m2_hybrid_entres_tight_v1(crop),
                min_frac_of_fallback=0.70,
            ),
        )

    def m3_support_edgeaware_bridge_v1(crop: np.ndarray) -> np.ndarray:
        return cached(
            "m3_support_edgeaware_bridge_v1",
            crop,
            lambda: fallback_if_too_small(
                hysteresis_support_reconstruct(
                    edge_aware_support_bridge_mask(
                        m2_candidate_union_v2(crop),
                        binary_fill_holes(center_default(crop) | simple_tight_v1(crop)),
                        np.maximum(residual_score(crop), entropy_score(crop, radius=5)),
                        (255 - crop.min(axis=2)).astype(np.uint8),
                        band_frac=0.08,
                        lower_slack_frac=0.05,
                        lateral_strip_frac=0.22,
                        inner_anchor_frac=0.18,
                        score_q=0.14,
                        nonwhite_min=8,
                        row_pad_frac=0.03,
                        min_bridge_span_frac=0.10,
                        close_k=7,
                    ),
                    residual_score(crop),
                    center_default(crop),
                    core_quantile=0.84,
                    core_scale=1.00,
                    overlap_frac=0.03,
                    core_open_k=3,
                    final_close_k=7,
                ),
                m2_hybrid_entres_tight_v1(crop),
                min_frac_of_fallback=0.70,
            ),
        )

    def m3_hyst_entres_guard_edgefill_v1(crop: np.ndarray) -> np.ndarray:
        return cached(
            "m3_hyst_entres_guard_edgefill_v1",
            crop,
            lambda: fill_edge_touch_strips(
                fallback_if_too_small(
                    hysteresis_support_reconstruct(
                        m2_candidate_union_v2(crop),
                        residual_score(crop),
                        center_default(crop),
                        core_quantile=0.84,
                        core_scale=1.00,
                        overlap_frac=0.03,
                        core_open_k=3,
                        final_close_k=7,
                    ),
                    m2_hybrid_entres_tight_v1(crop),
                    min_frac_of_fallback=0.70,
                ),
                m2_candidate_union_v2(crop),
                strip_frac=0.06,
                min_runs=2,
                min_span_frac=0.12,
                close_k=5,
            ),
        )

    def edge_support_aug_v1(crop: np.ndarray) -> np.ndarray:
        return cached(
            "edge_support_aug_v1",
            crop,
            lambda: binary_fill_holes(
                binary_propagation(
                    m3_hyst_entres_guard_v1(crop),
                    mask=edge_support_augment(
                        m3_hyst_entres_guard_v1(crop),
                        m2_candidate_union_v2(crop),
                        np.maximum(residual_score(crop), entropy_score(crop, radius=5)),
                        strip_frac=0.08,
                        inner_frac=0.16,
                        score_q=0.35,
                        min_row_frac=0.05,
                        row_pad_frac=0.03,
                        close_k=5,
                    ),
                )
            ),
        )

    def edge_run_completion_v1(crop: np.ndarray) -> np.ndarray:
        return cached(
            "edge_run_completion_v1",
            crop,
            lambda: edge_run_completion(
                m3_hyst_entres_guard_v1(crop),
                m2_candidate_union_v2(crop),
                np.maximum(residual_score(crop), entropy_score(crop, radius=5)),
                strip_frac=0.08,
                depth=2,
                min_runs=2,
                min_span_frac=0.12,
                row_pad_frac=0.03,
                score_q=0.20,
                close_k=5,
            ),
        )

    def edge_top_envelope_v1(crop: np.ndarray) -> np.ndarray:
        return cached(
            "edge_top_envelope_v1",
            crop,
            lambda: top_envelope_lateral_completion(
                m3_hyst_entres_guard_v1(crop),
                m2_candidate_union_v2(crop),
                np.maximum(residual_score(crop), entropy_score(crop, radius=5)),
                (255 - crop.min(axis=2)).astype(np.uint8),
                band_frac=0.08,
                lower_slack_frac=0.05,
                score_q=0.22,
                nonwhite_min=10,
                close_k=5,
            ),
        )

    def edge_top_envelope_bridge_v1(crop: np.ndarray) -> np.ndarray:
        return cached(
            "edge_top_envelope_bridge_v1",
            crop,
            lambda: top_envelope_bridge_completion(
                m3_hyst_entres_guard_v1(crop),
                m2_candidate_union_v2(crop),
                np.maximum(residual_score(crop), entropy_score(crop, radius=5)),
                (255 - crop.min(axis=2)).astype(np.uint8),
                band_frac=0.08,
                lower_slack_frac=0.05,
                score_q=0.16,
                nonwhite_min=8,
                min_bridge_cols_frac=0.08,
                close_k=7,
            ),
        )

    def m4_multicomp_entres_v1(crop: np.ndarray) -> np.ndarray:
        return cached(
            "m4_multicomp_entres_v1",
            crop,
            lambda: component_set_select(
                m2_candidate_union_v1(crop),
                score_core=center_default(crop),
                support_mask=m2_candidate_union_v1(crop),
                max_components=2,
                min_area_frac=0.08,
                bridge_erode_k=3,
                core_dilate_k=17,
                border_penalty=0.75,
                min_score=0.10,
                final_close_k=7,
            ),
        )

    def m4_multicomp_entres_v2(crop: np.ndarray) -> np.ndarray:
        return cached(
            "m4_multicomp_entres_v2",
            crop,
            lambda: component_set_select(
                m2_candidate_union_v2(crop),
                score_core=center_default(crop),
                support_mask=m2_candidate_union_v1(crop),
                max_components=3,
                min_area_frac=0.06,
                bridge_erode_k=3,
                core_dilate_k=17,
                border_penalty=0.85,
                min_score=0.06,
                final_close_k=7,
            ),
        )

    def m4_multicomp_guard_v1(crop: np.ndarray) -> np.ndarray:
        return cached(
            "m4_multicomp_guard_v1",
            crop,
            lambda: fallback_if_too_small(
                component_set_select(
                    m2_candidate_union_v2(crop),
                    score_core=m3_hyst_entres_guard_v1(crop),
                    support_mask=m2_candidate_union_v2(crop),
                    max_components=2,
                    min_area_frac=0.06,
                    bridge_erode_k=3,
                    core_dilate_k=13,
                    border_penalty=0.95,
                    min_score=0.05,
                    final_close_k=5,
                ),
                m3_hyst_entres_guard_v1(crop),
                min_frac_of_fallback=0.72,
            ),
        )

    def hybrid_tightcand_k5_o05(crop: np.ndarray) -> np.ndarray:
        return cached(
            "hybrid_tightcand_k5_o05",
            crop,
            lambda: hybrid_reconstruct(simple_tight_v2(crop), center_default(crop), erode_k=5, core_dilate_k=17, overlap_frac=0.05, final_close_k=7),
        )

    def m1_hybrid_tightcand_recon_v1(crop: np.ndarray) -> np.ndarray:
        return cached(
            "m1_hybrid_tightcand_recon_v1",
            crop,
            lambda: hybrid_reconstruct_m1(
                simple_tight_v1(crop),
                center_default(crop),
                erode_k=7,
                core_dilate_k=21,
                overlap_frac=0.03,
                candidate_open_frac=0.0045,
                candidate_close_frac=0.0065,
                final_open_frac=0.0035,
                final_close_frac=0.0055,
                min_keep_frac=0.92,
            ),
        )

    def m1_hybrid_tightcand_recon_v2(crop: np.ndarray) -> np.ndarray:
        return cached(
            "m1_hybrid_tightcand_recon_v2",
            crop,
            lambda: hybrid_reconstruct_m1(
                simple_tight_v1(crop),
                center_default(crop),
                erode_k=7,
                core_dilate_k=21,
                overlap_frac=0.03,
                candidate_open_frac=0.0055,
                candidate_close_frac=0.0075,
                final_open_frac=0.0040,
                final_close_frac=0.0060,
                min_keep_frac=0.90,
            ),
        )

    def m1_hybrid_default_recon_v1(crop: np.ndarray) -> np.ndarray:
        return cached(
            "m1_hybrid_default_recon_v1",
            crop,
            lambda: hybrid_reconstruct_m1(
                simple(crop),
                center_default(crop),
                erode_k=7,
                core_dilate_k=21,
                overlap_frac=0.03,
                candidate_open_frac=0.0045,
                candidate_close_frac=0.0065,
                final_open_frac=0.0035,
                final_close_frac=0.0055,
                min_keep_frac=0.92,
            ),
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
        "gui_hybrid_balanced_production": gui_hybrid_balanced_production,
        "simple_conservative": simple,
        "simple_tight_v1": simple_tight_v1,
        "simple_tight_v2": simple_tight_v2,
        "crop_center_default2comp": center_default,
        "crop_center_loose2comp": center_loose,
        "candidate_center_default2comp": candidate_center_default,
        "m2_candidate_union_v1": m2_candidate_union_v1,
        "m2_candidate_union_v2": m2_candidate_union_v2,
        "hybrid_default_k5_o03": hybrid_default_k5_o03,
        "hybrid_default_k7_o03": hybrid_default_k7_o03,
        "hybrid_default_k7_o03_posttight_v1": hybrid_default_k7_o03_posttight_v1,
        "hybrid_default_k7_o03_posttight_v2": hybrid_default_k7_o03_posttight_v2,
        "hybrid_default_k5_o05": hybrid_default_k5_o05,
        "hybrid_tightcand_k7_o03": hybrid_tightcand_k7_o03,
        "m2_hybrid_entres_v1": m2_hybrid_entres_v1,
        "m2_hybrid_entres_tight_v1": m2_hybrid_entres_tight_v1,
        "m2_hybrid_entres_guard_v1": m2_hybrid_entres_guard_v1,
        "m3_hyst_entres_v1": m3_hyst_entres_v1,
        "m3_hyst_entres_tight_v1": m3_hyst_entres_tight_v1,
        "m3_hyst_entres_guard_v1": m3_hyst_entres_guard_v1,
        "m3_support_edgeaware_v1": m3_support_edgeaware_v1,
        "m3_support_edgeaware_bridge_v1": m3_support_edgeaware_bridge_v1,
        "m3_hyst_entres_guard_edgefill_v1": m3_hyst_entres_guard_edgefill_v1,
        "edge_support_aug_v1": edge_support_aug_v1,
        "edge_run_completion_v1": edge_run_completion_v1,
        "edge_top_envelope_v1": edge_top_envelope_v1,
        "edge_top_envelope_bridge_v1": edge_top_envelope_bridge_v1,
        "m4_multicomp_entres_v1": m4_multicomp_entres_v1,
        "m4_multicomp_entres_v2": m4_multicomp_entres_v2,
        "m4_multicomp_guard_v1": m4_multicomp_guard_v1,
        "hybrid_tightcand_k5_o05": hybrid_tightcand_k5_o05,
        "m1_hybrid_tightcand_recon_v1": m1_hybrid_tightcand_recon_v1,
        "m1_hybrid_tightcand_recon_v2": m1_hybrid_tightcand_recon_v2,
        "m1_hybrid_default_recon_v1": m1_hybrid_default_recon_v1,
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

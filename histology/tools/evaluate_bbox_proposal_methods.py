#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import binary_fill_holes, binary_propagation, watershed_ift

Image.MAX_IMAGE_PIXELS = None

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from histology.gui_mvp.hitl_gui.pipeline_adapters.slide_io import load_slide_bundle, open_slide_handle
from histology.tools.run_ndpi_review_experiment import (
    CandidateBox,
    assign_sections,
    compute_stain_score,
    component_mask_from_overview,
    find_candidate_components,
    odd_kernel,
    parse_slide_stem,
)


@dataclass
class GtSection:
    label: str
    sample_id: str
    section_id: int
    gt_dir: Path
    proposal_bbox_overview: dict | None
    gt_crop_bbox_level0: tuple[int, int, int, int]
    mask: np.ndarray
    crop_shape: tuple[int, int]


def _parse_gt_crop_bbox_level0(meta: dict) -> tuple[int, int, int, int] | None:
    crop = meta.get("crop_bbox_level0")
    if isinstance(crop, dict):
        xywh = crop.get("xywh")
        if isinstance(xywh, dict):
            return (
                int(xywh["x"]),
                int(xywh["y"]),
                int(xywh["w"]),
                int(xywh["h"]),
            )
    return None


def _parse_gt_proposal_bbox_overview(meta: dict) -> dict | None:
    bbox = meta.get("proposal_bbox_overview_xywh")
    if isinstance(bbox, dict):
        return {
            "x": int(bbox["x"]),
            "y": int(bbox["y"]),
            "w": int(bbox["w"]),
            "h": int(bbox["h"]),
        }
    bbox = meta.get("bbox_overview")
    if isinstance(bbox, dict):
        return {
            "x": int(bbox["x"]),
            "y": int(bbox["y"]),
            "w": int(bbox["w"]),
            "h": int(bbox["h"]),
        }
    return None


def collect_gt_sections(gt_root: Path) -> list[GtSection]:
    sections: list[GtSection] = []
    for path in sorted(gt_root.iterdir()):
        if not path.is_dir():
            continue
        meta_path = path / "metadata.json"
        mask_path = path / "tissue_mask_final.png"
        if not meta_path.exists() or not mask_path.exists():
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        label = meta["label"]
        sample_id, section_blob = label.split("_", 1)
        mask = np.asarray(Image.open(mask_path).convert("L")) > 0
        crop_bbox_level0 = _parse_gt_crop_bbox_level0(meta)
        if crop_bbox_level0 is None:
            continue
        sections.append(
            GtSection(
                label=label,
                sample_id=sample_id,
                section_id=int(section_blob),
                gt_dir=path,
                proposal_bbox_overview=_parse_gt_proposal_bbox_overview(meta),
                gt_crop_bbox_level0=crop_bbox_level0,
                mask=mask,
                crop_shape=mask.shape[:2],
            )
        )
    return sections


def rect_overlap_gt_mask_level0_area(
    rect_level0_xywh: tuple[int, int, int, int],
    gt_section: GtSection,
) -> float:
    bounds = project_rect_to_gt_crop_bounds_level0(rect_level0_xywh, gt_section.gt_crop_bbox_level0, gt_section.crop_shape)
    if bounds is None:
        return 0.0
    px1, py1, px2, py2 = bounds
    covered = int(gt_section.mask[py1:py2, px1:px2].sum())
    gh, gw = gt_section.crop_shape
    _, _, gcw, gch = gt_section.gt_crop_bbox_level0
    pixel_area_level0 = (gcw / max(1.0, float(gw))) * (gch / max(1.0, float(gh)))
    return float(covered) * float(pixel_area_level0)


def build_section_to_slide_index(ndpi_root: Path) -> dict[tuple[str, str, int], Path]:
    index: dict[tuple[str, str, int], Path] = {}
    for slide_path in sorted(ndpi_root.glob("*.ndpi")):
        if slide_path.name.startswith("._"):
            continue
        stain, labels = parse_slide_stem(slide_path.stem)
        for label in labels:
            index[(stain.lower(), label.sample_id, label.section_id)] = slide_path
    return index


def gallyas_overview_residual(overview_rgb: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(overview_rgb, cv2.COLOR_RGB2GRAY)
    bg_sigma = max(25, int(round(min(overview_rgb.shape[:2]) * 0.03)))
    bg_sigma = float(odd_kernel(bg_sigma, minimum=25))
    bg = cv2.GaussianBlur(gray, (0, 0), sigmaX=bg_sigma, sigmaY=bg_sigma)
    return np.clip(bg.astype(np.int16) - gray.astype(np.int16), 0, 255).astype(np.uint8)


def overview_score_for_stain(overview_rgb: np.ndarray, stain: str) -> np.ndarray:
    if stain.lower() == "gallyas":
        return gallyas_overview_residual(overview_rgb)
    score, _ = compute_stain_score(overview_rgb, stain)
    score = cv2.GaussianBlur(score, (0, 0), sigmaX=1.2, sigmaY=1.2)
    return score.astype(np.uint8)


def bbox_score_maps_for_stain(overview_rgb: np.ndarray, stain: str) -> dict[str, np.ndarray]:
    stain_key = stain.lower()
    primary = overview_score_for_stain(overview_rgb, stain_key)
    legacy_score, info = compute_stain_score(overview_rgb, stain_key)
    legacy_score = cv2.GaussianBlur(legacy_score, (0, 0), sigmaX=1.2, sigmaY=1.2).astype(np.uint8)
    if stain_key == "gallyas":
        hybrid = np.maximum(primary, legacy_score).astype(np.uint8)
        nonwhite = cv2.GaussianBlur(info["nonwhite"], (0, 0), sigmaX=1.2, sigmaY=1.2).astype(np.uint8)
        fringe_hybrid_nonwhite = np.maximum(hybrid, nonwhite).astype(np.uint8)
        return {
            "primary": primary,
            "legacy": legacy_score,
            "hybrid": hybrid,
            "nonwhite": nonwhite,
            "fringe_hybrid_nonwhite": fringe_hybrid_nonwhite,
        }
    return {
        "primary": primary,
        "legacy": legacy_score,
        "hybrid": primary,
        "nonwhite": info["nonwhite"].astype(np.uint8),
        "fringe_hybrid_nonwhite": primary,
    }


def _best_component_near_center(mask: np.ndarray, center_xy: tuple[float, float]) -> np.ndarray:
    if mask.dtype != np.uint8:
        mask_u8 = (mask > 0).astype(np.uint8)
    else:
        mask_u8 = (mask > 0).astype(np.uint8)
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    if n <= 1:
        return mask_u8 > 0
    cx, cy = center_xy
    best_idx = 0
    best_score = -1e18
    for idx in range(1, n):
        area = float(stats[idx, cv2.CC_STAT_AREA])
        px, py = centroids[idx]
        dist = math.hypot(float(px) - cx, float(py) - cy)
        score = area - 2.5 * dist
        if score > best_score:
            best_score = score
            best_idx = idx
    return labels == best_idx


def _support_bbox_from_core_and_fringe(
    candidate: CandidateBox,
    overview_shape_wh: tuple[int, int],
    core_score: np.ndarray,
    fringe_score: np.ndarray,
    *,
    search_left_ratio: float = 0.24,
    search_top_ratio: float = 0.55,
    search_right_ratio: float = 0.24,
    search_bottom_ratio: float = 0.24,
    core_quantile: float = 0.84,
    fringe_quantile: float = 0.52,
    fringe_scale: float = 0.92,
    pad_ratio: float = 0.04,
) -> tuple[int, int, int, int]:
    search = expand_candidate_bbox(
        candidate,
        overview_shape_wh,
        left_ratio=search_left_ratio,
        top_ratio=search_top_ratio,
        right_ratio=search_right_ratio,
        bottom_ratio=search_bottom_ratio,
    )
    sx1, sy1, sx2, sy2 = search
    core_patch = core_score[sy1:sy2, sx1:sx2]
    fringe_patch = fringe_score[sy1:sy2, sx1:sx2]
    if core_patch.size == 0 or fringe_patch.size == 0:
        return search

    cx = float(candidate.x + candidate.w / 2.0 - sx1)
    cy = float(candidate.y + candidate.h / 2.0 - sy1)

    cand_core = core_score[candidate.y : candidate.y + candidate.h, candidate.x : candidate.x + candidate.w]
    core_vals = cand_core[cand_core > 0]
    if core_vals.size == 0:
        core_vals = core_patch[core_patch > 0]
    if core_vals.size == 0:
        return search
    core_thresh = max(10, int(round(float(np.quantile(core_vals, core_quantile)))))
    strong = (core_patch >= core_thresh).astype(np.uint8)
    strong = cv2.morphologyEx(strong, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    strong = cv2.morphologyEx(strong, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    if strong.max() == 0:
        return search
    strong = _best_component_near_center(strong, (cx, cy))
    if not strong.any():
        return search

    fringe_vals = fringe_patch[fringe_patch > 0]
    if fringe_vals.size == 0:
        return search
    fringe_thresh = max(6, int(round(float(np.quantile(fringe_vals, fringe_quantile)) * fringe_scale)))
    fringe = fringe_patch >= fringe_thresh
    fringe = cv2.morphologyEx((fringe.astype(np.uint8) * 255), cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8)) > 0
    seed = cv2.dilate(strong.astype(np.uint8) * 255, np.ones((3, 3), np.uint8), iterations=1) > 0
    support = binary_propagation(seed, mask=fringe)
    support = binary_fill_holes(support)
    support = cv2.morphologyEx((support.astype(np.uint8) * 255), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)) > 0
    if not support.any():
        support = strong
    ys, xs = np.where(support)
    if xs.size == 0:
        return search
    x1 = int(xs.min()) + sx1
    y1 = int(ys.min()) + sy1
    x2 = int(xs.max()) + 1 + sx1
    y2 = int(ys.max()) + 1 + sy1
    support_w = max(1, x2 - x1)
    support_h = max(1, y2 - y1)
    pad = max(6, int(round(max(support_w, support_h) * pad_ratio)))
    return clamp_crop_bbox(x1 - pad, y1 - pad, x2 + pad, y2 + pad, overview_shape_wh)


def clamp_crop_bbox(x1: int, y1: int, x2: int, y2: int, shape_wh: tuple[int, int]) -> tuple[int, int, int, int]:
    w, h = shape_wh
    x1 = max(0, min(w - 1, int(round(x1))))
    y1 = max(0, min(h - 1, int(round(y1))))
    x2 = max(x1 + 1, min(w, int(round(x2))))
    y2 = max(y1 + 1, min(h, int(round(y2))))
    return x1, y1, x2, y2


def rect_union(a: tuple[int, int, int, int], b: tuple[int, int, int, int], shape_wh: tuple[int, int]) -> tuple[int, int, int, int]:
    return clamp_crop_bbox(min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]), shape_wh)


def expand_candidate_bbox(
    candidate: CandidateBox,
    overview_shape_wh: tuple[int, int],
    *,
    left_ratio: float,
    top_ratio: float,
    right_ratio: float,
    bottom_ratio: float,
    min_pad: int = 24,
) -> tuple[int, int, int, int]:
    base = float(max(candidate.w, candidate.h))
    pl = max(min_pad, int(round(base * left_ratio)))
    pt = max(min_pad, int(round(base * top_ratio)))
    pr = max(min_pad, int(round(base * right_ratio)))
    pb = max(min_pad, int(round(base * bottom_ratio)))
    return clamp_crop_bbox(
        candidate.x - pl,
        candidate.y - pt,
        candidate.x + candidate.w + pr,
        candidate.y + candidate.h + pb,
        overview_shape_wh,
    )


def smooth1d(arr: np.ndarray, ksize: int = 9) -> np.ndarray:
    if arr.size == 0:
        return arr
    if ksize % 2 == 0:
        ksize += 1
    vec = arr.astype(np.float32)[None, :, None]
    out = cv2.GaussianBlur(vec, (1, ksize), 0).reshape(-1)
    return out


def contiguous_expand(signal: np.ndarray, threshold: float, max_gap: int = 3) -> int:
    if signal.size == 0:
        return 0
    expand = 0
    gap = 0
    for idx in range(signal.size - 1, -1, -1):
        if signal[idx] >= threshold:
            expand = signal.size - idx
            gap = 0
        else:
            gap += 1
            if gap >= max_gap:
                break
    return int(expand)


def region_slice_masks(gt: np.ndarray) -> dict[str, np.ndarray]:
    h, w = gt.shape[:2]
    x1, y1, x2, y2 = tight_bbox(gt)
    if x2 <= x1 or y2 <= y1:
        empty = np.zeros_like(gt, dtype=bool)
        return {name: empty for name in ["top", "middle", "bottom", "left", "center", "right"]}
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
    return masks


def projection_expand_bbox(
    candidate: CandidateBox,
    overview_score: np.ndarray,
    overview_shape_wh: tuple[int, int],
    *,
    top_cap_ratio: float = 0.30,
    bottom_cap_ratio: float = 0.14,
    side_cap_ratio: float = 0.10,
    span_margin_ratio: float = 0.08,
    top_only: bool = False,
    thresh_scale: float = 0.42,
    max_gap: int = 3,
) -> tuple[int, int, int, int]:
    h_img, w_img = overview_score.shape[:2]
    x1 = max(0, candidate.x)
    y1 = max(0, candidate.y)
    x2 = min(w_img, candidate.x + candidate.w)
    y2 = min(h_img, candidate.y + candidate.h)
    if x2 <= x1 or y2 <= y1:
        return expand_candidate_bbox(
            candidate,
            overview_shape_wh,
            left_ratio=0.08,
            top_ratio=0.08,
            right_ratio=0.08,
            bottom_ratio=0.08,
        )

    pad_base = max(24, int(round(max(candidate.w, candidate.h) * 0.08)))
    x1b, y1b, x2b, y2b = clamp_crop_bbox(x1 - pad_base, y1 - pad_base, x2 + pad_base, y2 + pad_base, overview_shape_wh)
    inside = overview_score[y1:y2, x1:x2]
    inside_vals = inside[inside > 0]
    if inside_vals.size == 0:
        return (x1b, y1b, x2b, y2b)
    thresh = max(6.0, float(np.quantile(inside_vals, 0.18)) * thresh_scale)

    span_x = max(12, int(round(candidate.w * span_margin_ratio)))
    span_y = max(12, int(round(candidate.h * span_margin_ratio)))
    col1 = max(0, x1 - span_x)
    col2 = min(w_img, x2 + span_x)
    row1 = max(0, y1 - span_y)
    row2 = min(h_img, y2 + span_y)

    top_cap = max(0, int(round(candidate.h * top_cap_ratio)))
    bottom_cap = max(0, int(round(candidate.h * bottom_cap_ratio)))
    side_cap = max(0, int(round(candidate.w * side_cap_ratio)))

    add_top = 0
    if top_cap > 0 and y1 > 0:
        patch = overview_score[max(0, y1 - top_cap) : y1, col1:col2]
        if patch.size > 0:
            signal = smooth1d(np.quantile(patch, 0.85, axis=1))
            add_top = contiguous_expand(signal, thresh, max_gap=max_gap)

    add_bottom = 0
    add_left = 0
    add_right = 0
    if not top_only:
        if bottom_cap > 0 and y2 < h_img:
            patch = overview_score[y2 : min(h_img, y2 + bottom_cap), col1:col2]
            if patch.size > 0:
                signal = smooth1d(np.quantile(patch, 0.85, axis=1))
                add_bottom = contiguous_expand(signal, thresh, max_gap=max_gap)
        vrow1 = max(0, y1 - span_y)
        vrow2 = min(h_img, y2 + span_y)
        if side_cap > 0 and x1 > 0:
            patch = overview_score[vrow1:vrow2, max(0, x1 - side_cap) : x1]
            if patch.size > 0:
                signal = smooth1d(np.quantile(patch, 0.85, axis=0))
                add_left = contiguous_expand(signal, thresh, max_gap=max_gap)
        if side_cap > 0 and x2 < w_img:
            patch = overview_score[vrow1:vrow2, x2 : min(w_img, x2 + side_cap)]
            if patch.size > 0:
                signal = smooth1d(np.quantile(patch, 0.85, axis=0))
                add_right = contiguous_expand(signal, thresh, max_gap=max_gap)

    return clamp_crop_bbox(x1b - add_left, y1b - add_top, x2b + add_right, y2b + add_bottom, overview_shape_wh)


def projection_full_topfloor(
    candidate: CandidateBox,
    overview_score: np.ndarray,
    overview_shape_wh: tuple[int, int],
    *,
    left_ratio: float,
    top_ratio: float,
    right_ratio: float,
    bottom_ratio: float,
    thresh_scale: float = 0.42,
    proj_top_cap_ratio: float = 0.30,
    proj_bottom_cap_ratio: float = 0.14,
    proj_side_cap_ratio: float = 0.10,
    proj_max_gap: int = 3,
) -> tuple[int, int, int, int]:
    top_bias = expand_candidate_bbox(
        candidate,
        overview_shape_wh,
        left_ratio=left_ratio,
        top_ratio=top_ratio,
        right_ratio=right_ratio,
        bottom_ratio=bottom_ratio,
    )
    proj = projection_expand_bbox(
        candidate,
        overview_score,
        overview_shape_wh,
        top_cap_ratio=proj_top_cap_ratio,
        bottom_cap_ratio=proj_bottom_cap_ratio,
        side_cap_ratio=proj_side_cap_ratio,
        top_only=False,
        thresh_scale=thresh_scale,
        max_gap=proj_max_gap,
    )
    return rect_union(top_bias, proj, overview_shape_wh)


def level0_rect_to_overview_rect(
    crop_bbox_level0: tuple[int, int, int, int],
    overview_downsample: float,
    overview_shape_wh: tuple[int, int],
) -> tuple[int, int, int, int]:
    x, y, w, h = crop_bbox_level0
    return clamp_crop_bbox(
        math.floor(x / overview_downsample),
        math.floor(y / overview_downsample),
        math.ceil((x + w) / overview_downsample),
        math.ceil((y + h) / overview_downsample),
        overview_shape_wh,
    )


def rect_to_gt_crop_mask_level0(
    rect_level0: tuple[int, int, int, int],
    gt_crop_level0: tuple[int, int, int, int],
    gt_shape: tuple[int, int],
) -> np.ndarray:
    gh, gw = gt_shape
    rx1, ry1, rx2, ry2 = rect_level0
    gx, gy, gw0, gh0 = gt_crop_level0
    gx1, gy1, gx2, gy2 = gx, gy, gx + gw0, gy + gh0
    ix1 = max(rx1, gx1)
    iy1 = max(ry1, gy1)
    ix2 = min(rx2, gx2)
    iy2 = min(ry2, gy2)
    out = np.zeros((gh, gw), dtype=bool)
    if ix1 >= ix2 or iy1 >= iy2:
        return out

    scale_x = gw / max(1.0, float(gw0))
    scale_y = gh / max(1.0, float(gh0))
    px1 = max(0, min(gw - 1, int(math.floor((ix1 - gx1) * scale_x))))
    py1 = max(0, min(gh - 1, int(math.floor((iy1 - gy1) * scale_y))))
    px2 = max(px1 + 1, min(gw, int(math.ceil((ix2 - gx1) * scale_x))))
    py2 = max(py1 + 1, min(gh, int(math.ceil((iy2 - gy1) * scale_y))))
    out[py1:py2, px1:px2] = True
    return out


def project_rect_to_gt_crop_bounds_level0(
    rect_level0: tuple[int, int, int, int],
    gt_crop_level0: tuple[int, int, int, int],
    gt_shape: tuple[int, int],
) -> tuple[int, int, int, int] | None:
    gh, gw = gt_shape
    rx, ry, rw0, rh0 = rect_level0
    gx, gy, gw0, gh0 = gt_crop_level0
    rx1, ry1, rx2, ry2 = rx, ry, rx + rw0, ry + rh0
    gx1, gy1, gx2, gy2 = gx, gy, gx + gw0, gy + gh0
    ix1 = max(rx1, gx1)
    iy1 = max(ry1, gy1)
    ix2 = min(rx2, gx2)
    iy2 = min(ry2, gy2)
    if ix1 >= ix2 or iy1 >= iy2:
        return None
    scale_x = gw / max(1.0, float(gw0))
    scale_y = gh / max(1.0, float(gh0))
    px1 = max(0, min(gw - 1, int(math.floor((ix1 - gx1) * scale_x))))
    py1 = max(0, min(gh - 1, int(math.floor((iy1 - gy1) * scale_y))))
    px2 = max(px1 + 1, min(gw, int(math.ceil((ix2 - gx1) * scale_x))))
    py2 = max(py1 + 1, min(gh, int(math.ceil((iy2 - gy1) * scale_y))))
    return px1, py1, px2, py2


def tight_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask)
    if xs.size == 0:
        return 0, 0, 0, 0
    x1 = int(xs.min())
    y1 = int(ys.min())
    x2 = int(xs.max()) + 1
    y2 = int(ys.max()) + 1
    return x1, y1, x2, y2


def tight_bbox_from_bounds(bounds: tuple[int, int, int, int] | None) -> tuple[int, int, int, int]:
    if bounds is None:
        return 0, 0, 0, 0
    x1, y1, x2, y2 = bounds
    return x1, y1, x2, y2


def iou_rect(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / max(1, union)


def coverage_metrics(
    gt_mask: np.ndarray,
    proposal_bounds_in_gt: tuple[int, int, int, int] | None,
    *,
    proposal_rect_level0_xywh: tuple[int, int, int, int],
    gt_crop_level0_xywh: tuple[int, int, int, int],
) -> dict[str, float]:
    gt_pixels = int(gt_mask.sum())
    gh, gw = gt_mask.shape[:2]
    gx, gy, gcw, gch = gt_crop_level0_xywh
    rx, ry, rw0, rh0 = proposal_rect_level0_xywh
    proposal_area_level0 = float(max(1, rw0 * rh0))
    gt_crop_area_level0 = float(max(1, gcw * gch))
    gt_mask_area_level0 = float(max(1.0, gt_pixels * (gcw / max(1.0, gw)) * (gch / max(1.0, gh))))
    proposal_rect_level0_xyxy = (rx, ry, rx + rw0, ry + rh0)
    gt_crop_level0_xyxy = (gx, gy, gx + gcw, gy + gch)
    if proposal_bounds_in_gt is None:
        covered = 0
        prop_pixels = 0
        prop_bbox = (0, 0, 0, 0)
        rect_mask = np.zeros_like(gt_mask, dtype=bool)
    else:
        px1, py1, px2, py2 = proposal_bounds_in_gt
        covered = int(gt_mask[py1:py2, px1:px2].sum())
        prop_pixels = int(max(0, px2 - px1) * max(0, py2 - py1))
        prop_bbox = tight_bbox_from_bounds(proposal_bounds_in_gt)
        rect_mask = np.zeros_like(gt_mask, dtype=bool)
        rect_mask[py1:py2, px1:px2] = True
    gt_bbox = tight_bbox(gt_mask)
    local_regions = region_slice_masks(gt_mask)

    def local_cov(region: np.ndarray) -> float:
        denom = int(region.sum())
        if denom == 0:
            return 0.0
        return float((region & rect_mask).sum() / denom)

    return {
        "mask_coverage_recall": covered / max(1, gt_pixels),
        "mask_miss_ratio": 1.0 - (covered / max(1, gt_pixels)),
        "crop_efficiency": covered / max(1, prop_pixels),
        "crop_area_to_gt_mask_ratio": prop_pixels / max(1, gt_pixels),
        "proposal_rect_vs_gtbbox_iou": iou_rect(prop_bbox, gt_bbox),
        "proposal_area_to_gt_crop_area_full": proposal_area_level0 / gt_crop_area_level0,
        "proposal_area_to_gt_mask_area_full": proposal_area_level0 / gt_mask_area_level0,
        "proposal_rect_vs_gtcrop_iou_level0": iou_rect(proposal_rect_level0_xyxy, gt_crop_level0_xyxy),
        "top_coverage_recall": local_cov(local_regions["top"]),
        "middle_coverage_recall": local_cov(local_regions["middle"]),
        "bottom_coverage_recall": local_cov(local_regions["bottom"]),
        "left_coverage_recall": local_cov(local_regions["left"]),
        "center_coverage_recall": local_cov(local_regions["center"]),
        "right_coverage_recall": local_cov(local_regions["right"]),
    }


def draw_overview_comparison(
    overview_rgb: np.ndarray,
    baseline_rect: tuple[int, int, int, int],
    best_rect: tuple[int, int, int, int],
    gt_crop_rect: tuple[int, int, int, int],
    label: str,
) -> np.ndarray:
    img = Image.fromarray(overview_rgb.astype(np.uint8)).convert("RGB")
    draw = ImageDraw.Draw(img)
    draw.rectangle((gt_crop_rect[0], gt_crop_rect[1], gt_crop_rect[2] - 1, gt_crop_rect[3] - 1), outline=(0, 255, 0), width=4)
    draw.rectangle((baseline_rect[0], baseline_rect[1], baseline_rect[2] - 1, baseline_rect[3] - 1), outline=(255, 0, 0), width=3)
    draw.rectangle((best_rect[0], best_rect[1], best_rect[2] - 1, best_rect[3] - 1), outline=(255, 255, 0), width=3)
    draw.text((20, 20), label, fill=(255, 255, 255))
    return np.asarray(img)


def _candidate_core_mask(
    candidate: CandidateBox,
    core_score: np.ndarray,
    *,
    core_quantile: float = 0.84,
    min_thresh: int = 10,
) -> np.ndarray:
    h, w = core_score.shape[:2]
    x1, y1, x2, y2 = clamp_crop_bbox(candidate.x, candidate.y, candidate.x + candidate.w, candidate.y + candidate.h, (w, h))
    patch = core_score[y1:y2, x1:x2]
    vals = patch[patch > 0]
    core_local = np.zeros_like(patch, dtype=bool)
    if vals.size > 0:
        thresh = max(min_thresh, int(round(float(np.quantile(vals, core_quantile)))))
        core_local = patch >= thresh
        core_local = cv2.morphologyEx((core_local.astype(np.uint8) * 255), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0
        core_local = cv2.morphologyEx((core_local.astype(np.uint8) * 255), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)) > 0
        if core_local.any():
            core_local = _best_component_near_center(core_local, (candidate.cx - x1, candidate.cy - y1))
    if (not core_local.any()) and candidate.overview_mask is not None:
        fallback = candidate.overview_mask[y1:y2, x1:x2].astype(bool)
        fallback = cv2.erode((fallback.astype(np.uint8) * 255), np.ones((3, 3), np.uint8), iterations=1) > 0
        if fallback.any():
            core_local = _best_component_near_center(fallback, (candidate.cx - x1, candidate.cy - y1))
    if not core_local.any():
        cx = int(round(min(max(0.0, candidate.cx - x1), max(0, x2 - x1 - 1))))
        cy = int(round(min(max(0.0, candidate.cy - y1), max(0, y2 - y1 - 1))))
        rr = max(2, int(round(max(candidate.w, candidate.h) * 0.03)))
        yy, xx = np.ogrid[: (y2 - y1), : (x2 - x1)]
        core_local = ((xx - cx) ** 2 + (yy - cy) ** 2) <= (rr * rr)
    full = np.zeros((h, w), dtype=bool)
    full[y1:y2, x1:x2] = core_local
    return full


def _weak_support_mask(
    fringe_score: np.ndarray,
    *,
    fringe_quantile: float = 0.52,
    fringe_scale: float = 0.92,
) -> np.ndarray:
    vals = fringe_score[fringe_score > 0]
    if vals.size == 0:
        return np.zeros_like(fringe_score, dtype=bool)
    thresh = max(6, int(round(float(np.quantile(vals, fringe_quantile)) * fringe_scale)))
    mask = fringe_score >= thresh
    mask = cv2.morphologyEx((mask.astype(np.uint8) * 255), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0
    mask = cv2.morphologyEx((mask.astype(np.uint8) * 255), cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8)) > 0
    return mask


def _local_adaptive_dark_mask(
    overview_rgb: np.ndarray,
    *,
    window_ratio: float = 0.09,
    k: float = 0.22,
) -> np.ndarray:
    gray = cv2.cvtColor(overview_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    h, w = gray.shape[:2]
    win = max(15, int(round(min(h, w) * window_ratio)))
    if win % 2 == 0:
        win += 1
    mean = cv2.boxFilter(gray, ddepth=cv2.CV_32F, ksize=(win, win), normalize=True)
    sqmean = cv2.boxFilter(gray * gray, ddepth=cv2.CV_32F, ksize=(win, win), normalize=True)
    var = np.maximum(sqmean - mean * mean, 0.0)
    std = np.sqrt(var)
    thresh = mean * (1.0 + k * ((std / 128.0) - 1.0))
    mask = gray <= thresh
    mask = cv2.morphologyEx((mask.astype(np.uint8) * 255), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0
    mask = cv2.morphologyEx((mask.astype(np.uint8) * 255), cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8)) > 0
    return mask


def _integral_sum(integral: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> float:
    return float(integral[y2, x2] - integral[y1, x2] - integral[y2, x1] + integral[y1, x1])


def _candidate_support_mask(
    candidate: CandidateBox,
    overview_shape_wh: tuple[int, int],
    *,
    dilate_px: int = 9,
) -> np.ndarray:
    w, h = overview_shape_wh
    if candidate.overview_mask is not None:
        support = candidate.overview_mask.astype(bool).copy()
    else:
        support = np.zeros((h, w), dtype=bool)
        x1, y1, x2, y2 = clamp_crop_bbox(candidate.x, candidate.y, candidate.x + candidate.w, candidate.y + candidate.h, overview_shape_wh)
        support[y1:y2, x1:x2] = True
    k = max(3, int(dilate_px))
    if k % 2 == 0:
        k += 1
    support = cv2.dilate((support.astype(np.uint8) * 255), np.ones((k, k), np.uint8), iterations=1) > 0
    support = cv2.morphologyEx((support.astype(np.uint8) * 255), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)) > 0
    return support


def _distance_map_to_mask(mask: np.ndarray) -> np.ndarray:
    inv = (~mask).astype(np.uint8)
    return cv2.distanceTransform(inv, cv2.DIST_L2, 3)


def _owned_masks_competitive(
    all_candidates: list[CandidateBox],
    overview_shape_wh: tuple[int, int],
    core_score: np.ndarray,
    fringe_score: np.ndarray,
    *,
    core_quantile: float = 0.84,
    fringe_quantile: float = 0.52,
    fringe_scale: float = 0.92,
) -> tuple[list[np.ndarray], np.ndarray]:
    if not all_candidates:
        h = overview_shape_wh[1]
        w = overview_shape_wh[0]
        return [], np.zeros((h, w), dtype=bool)

    seed_masks = [
        _candidate_core_mask(cand, core_score, core_quantile=core_quantile)
        for cand in all_candidates
    ]
    dilated_seeds = [
        cv2.dilate((seed.astype(np.uint8) * 255), np.ones((3, 3), np.uint8), iterations=1) > 0
        for seed in seed_masks
    ]
    weak_support = _weak_support_mask(fringe_score, fringe_quantile=fringe_quantile, fringe_scale=fringe_scale)
    if dilated_seeds:
        seed_union = np.zeros_like(weak_support, dtype=bool)
        for seed in dilated_seeds:
            seed_union |= seed
        weak_support |= seed_union

    n, labels, _, _ = cv2.connectedComponentsWithStats(weak_support.astype(np.uint8), connectivity=8)
    centers = np.array([(cand.cx, cand.cy) for cand in all_candidates], dtype=np.float32)
    owned_masks = [np.zeros_like(weak_support, dtype=bool) for _ in all_candidates]

    for comp_idx in range(1, n):
        comp = labels == comp_idx
        if not comp.any():
            continue
        contenders: list[int] = []
        for idx, seed in enumerate(dilated_seeds):
            if np.any(comp & seed):
                contenders.append(idx)
        if not contenders:
            continue
        if len(contenders) == 1:
            owned_masks[contenders[0]] |= comp
            continue
        ys, xs = np.where(comp)
        comp_centers = centers[np.asarray(contenders, dtype=np.int32)]
        d2 = (xs[:, None] - comp_centers[:, 0][None, :]) ** 2 + (ys[:, None] - comp_centers[:, 1][None, :]) ** 2
        winners = np.argmin(d2, axis=1)
        for local_idx, contender_idx in enumerate(contenders):
            keep = winners == local_idx
            if np.any(keep):
                owned_masks[contender_idx][ys[keep], xs[keep]] = True

    for idx, seed in enumerate(seed_masks):
        owned_masks[idx] |= seed
        owned_masks[idx] = cv2.morphologyEx((owned_masks[idx].astype(np.uint8) * 255), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)) > 0
        owned_masks[idx] = binary_fill_holes(owned_masks[idx])

    return owned_masks, weak_support


def _watershed_owned_masks(
    all_candidates: list[CandidateBox],
    overview_shape_wh: tuple[int, int],
    core_score: np.ndarray,
    fringe_score: np.ndarray,
    *,
    core_quantile: float = 0.86,
    fringe_quantile: float = 0.55,
    fringe_scale: float = 0.94,
) -> tuple[list[np.ndarray], np.ndarray]:
    if not all_candidates:
        h = overview_shape_wh[1]
        w = overview_shape_wh[0]
        return [], np.zeros((h, w), dtype=bool)

    seed_masks = [
        _candidate_core_mask(cand, core_score, core_quantile=core_quantile)
        for cand in all_candidates
    ]
    weak_support = _weak_support_mask(fringe_score, fringe_quantile=fringe_quantile, fringe_scale=fringe_scale)
    for seed in seed_masks:
        weak_support |= cv2.dilate((seed.astype(np.uint8) * 255), np.ones((3, 3), np.uint8), iterations=1) > 0

    markers = np.zeros_like(weak_support, dtype=np.int32)
    for idx, seed in enumerate(seed_masks, start=1):
        markers[seed] = idx
    if markers.max() == 0:
        return [np.zeros_like(weak_support, dtype=bool) for _ in all_candidates], weak_support

    fringe_u8 = fringe_score.astype(np.float32)
    if fringe_u8.max() > 0:
        fringe_u8 = fringe_u8 / fringe_u8.max()
    cost = np.clip(255.0 - 255.0 * fringe_u8, 0, 255).astype(np.uint8)
    cost[~weak_support] = 255
    ws = watershed_ift(cost, markers)
    owned = []
    for idx in range(1, len(all_candidates) + 1):
        mask = ws == idx
        mask |= seed_masks[idx - 1]
        mask = cv2.morphologyEx((mask.astype(np.uint8) * 255), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)) > 0
        mask = binary_fill_holes(mask)
        owned.append(mask)
    return owned, weak_support


def _bbox_from_mask(mask: np.ndarray, overview_shape_wh: tuple[int, int], *, pad_ratio: float = 0.035, min_pad: int = 6) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask)
    if xs.size == 0:
        return 0, 0, overview_shape_wh[0], overview_shape_wh[1]
    x1 = int(xs.min())
    y1 = int(ys.min())
    x2 = int(xs.max()) + 1
    y2 = int(ys.max()) + 1
    pad = max(min_pad, int(round(max(x2 - x1, y2 - y1) * pad_ratio)))
    return clamp_crop_bbox(x1 - pad, y1 - pad, x2 + pad, y2 + pad, overview_shape_wh)


def _watershed_bbox(
    target_candidate: CandidateBox,
    all_candidates: list[CandidateBox],
    overview_shape_wh: tuple[int, int],
    core_score: np.ndarray,
    fringe_score: np.ndarray,
    *,
    core_quantile: float = 0.86,
    fringe_quantile: float = 0.55,
    fringe_scale: float = 0.94,
    pad_ratio: float = 0.035,
) -> tuple[int, int, int, int]:
    owned_masks, _ = _watershed_owned_masks(
        all_candidates,
        overview_shape_wh,
        core_score,
        fringe_score,
        core_quantile=core_quantile,
        fringe_quantile=fringe_quantile,
        fringe_scale=fringe_scale,
    )
    target_idx = 0
    for idx, cand in enumerate(all_candidates):
        if cand.section is not None and target_candidate.section is not None and cand.section.short_label == target_candidate.section.short_label:
            target_idx = idx
            break
        if cand.candidate_rank == target_candidate.candidate_rank:
            target_idx = idx
    mask = owned_masks[target_idx] if target_idx < len(owned_masks) else np.zeros((overview_shape_wh[1], overview_shape_wh[0]), dtype=bool)
    if not mask.any():
        return clamp_crop_bbox(target_candidate.x, target_candidate.y, target_candidate.x + target_candidate.w, target_candidate.y + target_candidate.h, overview_shape_wh)
    return _bbox_from_mask(mask, overview_shape_wh, pad_ratio=pad_ratio)


def _edgewise_boxopt_bbox(
    target_candidate: CandidateBox,
    all_candidates: list[CandidateBox],
    overview_shape_wh: tuple[int, int],
    core_score: np.ndarray,
    fringe_score: np.ndarray,
    *,
    core_quantile: float = 0.86,
    fringe_quantile: float = 0.55,
    fringe_scale: float = 0.94,
    pad_ratio: float = 0.02,
    search_px: int = 16,
    step_px: int = 2,
) -> tuple[int, int, int, int]:
    owned_masks, _ = _owned_masks_competitive(
        all_candidates,
        overview_shape_wh,
        core_score,
        fringe_score,
        core_quantile=core_quantile,
        fringe_quantile=fringe_quantile,
        fringe_scale=fringe_scale,
    )
    target_idx = 0
    for idx, cand in enumerate(all_candidates):
        if cand.section is not None and target_candidate.section is not None and cand.section.short_label == target_candidate.section.short_label:
            target_idx = idx
            break
        if cand.candidate_rank == target_candidate.candidate_rank:
            target_idx = idx
    if target_idx >= len(owned_masks):
        return clamp_crop_bbox(target_candidate.x, target_candidate.y, target_candidate.x + target_candidate.w, target_candidate.y + target_candidate.h, overview_shape_wh)

    target_mask = owned_masks[target_idx]
    if not target_mask.any():
        return clamp_crop_bbox(target_candidate.x, target_candidate.y, target_candidate.x + target_candidate.w, target_candidate.y + target_candidate.h, overview_shape_wh)
    rival_mask = np.zeros_like(target_mask, dtype=bool)
    for idx, mask in enumerate(owned_masks):
        if idx != target_idx:
            rival_mask |= mask

    target_integral = cv2.integral(target_mask.astype(np.uint8))
    rival_integral = cv2.integral(rival_mask.astype(np.uint8))
    total_target = float(max(1, int(target_mask.sum())))
    tight = tight_bbox(target_mask)
    base = _bbox_from_mask(target_mask, overview_shape_wh, pad_ratio=pad_ratio)
    tight_area = float(max(1, (tight[2] - tight[0]) * (tight[3] - tight[1])))

    def score(rect: tuple[int, int, int, int]) -> float:
        x1, y1, x2, y2 = rect
        area = float(max(1, (x2 - x1) * (y2 - y1)))
        tgt = _integral_sum(target_integral, x1, y1, x2, y2)
        riv = _integral_sum(rival_integral, x1, y1, x2, y2)
        target_recall = tgt / total_target
        rival_ratio = riv / area
        compact = min(1.0, tight_area / area)
        return 0.50 * target_recall + 0.30 * (1.0 - rival_ratio) + 0.20 * compact

    best = base
    best_score = score(best)
    for _ in range(2):
        x1, y1, x2, y2 = best
        candidates = []
        for dx1 in range(-search_px, search_px + 1, step_px):
            candidates.append(clamp_crop_bbox(x1 + dx1, y1, x2, y2, overview_shape_wh))
        for dy1 in range(-search_px, search_px + 1, step_px):
            candidates.append(clamp_crop_bbox(x1, y1 + dy1, x2, y2, overview_shape_wh))
        for dx2 in range(-search_px, search_px + 1, step_px):
            candidates.append(clamp_crop_bbox(x1, y1, x2 + dx2, y2, overview_shape_wh))
        for dy2 in range(-search_px, search_px + 1, step_px):
            candidates.append(clamp_crop_bbox(x1, y1, x2, y2 + dy2, overview_shape_wh))
        for rect in candidates:
            s = score(rect)
            if s > best_score:
                best = rect
                best_score = s
    return best


def _competitive_support_bbox(
    target_candidate: CandidateBox,
    all_candidates: list[CandidateBox],
    overview_shape_wh: tuple[int, int],
    core_score: np.ndarray,
    fringe_score: np.ndarray,
    *,
    core_quantile: float = 0.84,
    fringe_quantile: float = 0.52,
    fringe_scale: float = 0.92,
    pad_ratio: float = 0.04,
) -> tuple[int, int, int, int]:
    if not all_candidates:
        return clamp_crop_bbox(
            target_candidate.x,
            target_candidate.y,
            target_candidate.x + target_candidate.w,
            target_candidate.y + target_candidate.h,
            overview_shape_wh,
        )

    seed_masks = [
        _candidate_core_mask(cand, core_score, core_quantile=core_quantile)
        for cand in all_candidates
    ]
    dilated_seeds = [
        cv2.dilate((seed.astype(np.uint8) * 255), np.ones((3, 3), np.uint8), iterations=1) > 0
        for seed in seed_masks
    ]
    weak_support = _weak_support_mask(fringe_score, fringe_quantile=fringe_quantile, fringe_scale=fringe_scale)
    if dilated_seeds:
        seed_union = np.zeros_like(weak_support, dtype=bool)
        for seed in dilated_seeds:
            seed_union |= seed
        weak_support |= seed_union

    n, labels, _, _ = cv2.connectedComponentsWithStats(weak_support.astype(np.uint8), connectivity=8)
    target_idx = 0
    for idx, cand in enumerate(all_candidates):
        if cand.section is not None and target_candidate.section is not None:
            if cand.section.short_label == target_candidate.section.short_label:
                target_idx = idx
                break
        elif cand.candidate_rank == target_candidate.candidate_rank:
            target_idx = idx
            break

    centers = np.array([(cand.cx, cand.cy) for cand in all_candidates], dtype=np.float32)
    owned = np.zeros_like(weak_support, dtype=bool)

    for comp_idx in range(1, n):
        comp = labels == comp_idx
        if not comp.any():
            continue
        contenders: list[int] = []
        for idx, seed in enumerate(dilated_seeds):
            if np.any(comp & seed):
                contenders.append(idx)
        if not contenders:
            continue
        if len(contenders) == 1:
            if contenders[0] == target_idx:
                owned |= comp
            continue
        ys, xs = np.where(comp)
        comp_centers = centers[np.array(contenders, dtype=np.int32)]
        d2 = (xs[:, None] - comp_centers[:, 0][None, :]) ** 2 + (ys[:, None] - comp_centers[:, 1][None, :]) ** 2
        winners = np.argmin(d2, axis=1)
        target_local = None
        for local_idx, contender_idx in enumerate(contenders):
            if contender_idx == target_idx:
                target_local = local_idx
                break
        if target_local is None:
            continue
        keep = winners == target_local
        if np.any(keep):
            owned[ys[keep], xs[keep]] = True

    owned |= seed_masks[target_idx]
    owned = cv2.morphologyEx((owned.astype(np.uint8) * 255), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)) > 0
    owned = binary_fill_holes(owned)
    ys, xs = np.where(owned)
    if xs.size == 0:
        return clamp_crop_bbox(
            target_candidate.x,
            target_candidate.y,
            target_candidate.x + target_candidate.w,
            target_candidate.y + target_candidate.h,
            overview_shape_wh,
        )
    x1 = int(xs.min())
    y1 = int(ys.min())
    x2 = int(xs.max()) + 1
    y2 = int(ys.max()) + 1
    pad = max(6, int(round(max(x2 - x1, y2 - y1) * pad_ratio)))
    return clamp_crop_bbox(x1 - pad, y1 - pad, x2 + pad, y2 + pad, overview_shape_wh)


def _competitive_support_bbox_cost(
    target_candidate: CandidateBox,
    all_candidates: list[CandidateBox],
    overview_shape_wh: tuple[int, int],
    core_score: np.ndarray,
    fringe_score: np.ndarray,
    *,
    core_quantile: float = 0.84,
    fringe_quantile: float = 0.52,
    fringe_scale: float = 0.92,
    pad_ratio: float = 0.04,
    use_unknown: bool = False,
    support_dilate_px: int = 9,
) -> tuple[int, int, int, int]:
    if not all_candidates:
        return clamp_crop_bbox(
            target_candidate.x,
            target_candidate.y,
            target_candidate.x + target_candidate.w,
            target_candidate.y + target_candidate.h,
            overview_shape_wh,
        )

    seed_masks = [
        _candidate_core_mask(cand, core_score, core_quantile=core_quantile)
        for cand in all_candidates
    ]
    dilated_seeds = [
        cv2.dilate((seed.astype(np.uint8) * 255), np.ones((3, 3), np.uint8), iterations=1) > 0
        for seed in seed_masks
    ]
    support_masks = [
        _candidate_support_mask(cand, overview_shape_wh, dilate_px=support_dilate_px)
        for cand in all_candidates
    ]
    support_dist_maps = [_distance_map_to_mask(mask) for mask in support_masks]
    core_dist_maps = [_distance_map_to_mask(seed) for seed in seed_masks]

    weak_support = _weak_support_mask(fringe_score, fringe_quantile=fringe_quantile, fringe_scale=fringe_scale)
    if dilated_seeds:
        seed_union = np.zeros_like(weak_support, dtype=bool)
        for seed in dilated_seeds:
            seed_union |= seed
        weak_support |= seed_union

    n, labels, _, _ = cv2.connectedComponentsWithStats(weak_support.astype(np.uint8), connectivity=8)
    target_idx = 0
    for idx, cand in enumerate(all_candidates):
        if cand.section is not None and target_candidate.section is not None:
            if cand.section.short_label == target_candidate.section.short_label:
                target_idx = idx
                break
        elif cand.candidate_rank == target_candidate.candidate_rank:
            target_idx = idx
            break

    owned = np.zeros_like(weak_support, dtype=bool)

    for comp_idx in range(1, n):
        comp = labels == comp_idx
        if not comp.any():
            continue
        ys, xs = np.where(comp)
        area = float(xs.size)
        cx = float(xs.mean())
        cy = float(ys.mean())
        scores: list[float] = []
        seed_overlaps: list[float] = []
        for idx, cand in enumerate(all_candidates):
            seed_overlap = float((comp & dilated_seeds[idx]).sum() / max(1.0, area))
            support_overlap = float((comp & support_masks[idx]).sum() / max(1.0, area))
            dist_support = float(support_dist_maps[idx][int(round(cy)), int(round(cx))])
            dist_core = float(core_dist_maps[idx][int(round(cy)), int(round(cx))])
            center_dist = math.hypot(cx - cand.cx, cy - cand.cy)
            score = (
                3.8 * support_overlap
                + 2.2 * seed_overlap
                - 0.030 * dist_support
                - 0.014 * dist_core
                - 0.006 * center_dist
            )
            scores.append(score)
            seed_overlaps.append(seed_overlap)

        order = np.argsort(np.asarray(scores))[::-1]
        best_idx = int(order[0])
        best_score = float(scores[best_idx])
        second_score = float(scores[int(order[1])]) if len(order) > 1 else -1e18
        best_seed_overlap = float(seed_overlaps[best_idx])
        if use_unknown:
            if best_score < 0.08:
                continue
            if (best_score - second_score) < 0.045 and best_seed_overlap < 0.02:
                continue
        if best_idx == target_idx:
            owned |= comp

    owned |= seed_masks[target_idx]
    owned = cv2.morphologyEx((owned.astype(np.uint8) * 255), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)) > 0
    owned = binary_fill_holes(owned)
    ys, xs = np.where(owned)
    if xs.size == 0:
        return clamp_crop_bbox(
            target_candidate.x,
            target_candidate.y,
            target_candidate.x + target_candidate.w,
            target_candidate.y + target_candidate.h,
            overview_shape_wh,
        )
    x1 = int(xs.min())
    y1 = int(ys.min())
    x2 = int(xs.max()) + 1
    y2 = int(ys.max()) + 1
    pad = max(6, int(round(max(x2 - x1, y2 - y1) * pad_ratio)))
    return clamp_crop_bbox(x1 - pad, y1 - pad, x2 + pad, y2 + pad, overview_shape_wh)


def _extract_overview_rect_at_level(
    loaded_slide,
    slide_handle,
    rect_ov: tuple[int, int, int, int],
    level: int,
) -> np.ndarray:
    level = min(level, loaded_slide.level_count - 1)
    x1_ov, y1_ov, x2_ov, y2_ov = rect_ov
    overview_downsample = float(loaded_slide.level_downsamples[loaded_slide.overview_level])
    target_downsample = float(loaded_slide.level_downsamples[level])

    if loaded_slide.backend == "openslide":
        x0 = int(round(x1_ov * overview_downsample))
        y0 = int(round(y1_ov * overview_downsample))
        w0 = int(round((x2_ov - x1_ov) * overview_downsample))
        h0 = int(round((y2_ov - y1_ov) * overview_downsample))
        out_w = max(1, int(round(w0 / target_downsample)))
        out_h = max(1, int(round(h0 / target_downsample)))
        return np.asarray(slide_handle.read_region((x0, y0), level, (out_w, out_h)).convert("RGB"))

    if target_downsample >= float(loaded_slide.tifffile_midres_downsample):
        overview_to_midres = float(loaded_slide.tifffile_overview_scale_from_midres)
        x1 = int(round(x1_ov * overview_to_midres))
        y1 = int(round(y1_ov * overview_to_midres))
        x2 = int(round(x2_ov * overview_to_midres))
        y2 = int(round(y2_ov * overview_to_midres))
        x1 = max(0, min(slide_handle.midres_arr.shape[1] - 1, x1))
        y1 = max(0, min(slide_handle.midres_arr.shape[0] - 1, y1))
        x2 = max(x1 + 1, min(slide_handle.midres_arr.shape[1], x2))
        y2 = max(y1 + 1, min(slide_handle.midres_arr.shape[0], y2))
        crop = np.asarray(slide_handle.midres_arr[y1:y2, x1:x2, :], dtype=np.uint8)
        if target_downsample > float(loaded_slide.tifffile_midres_downsample):
            scale = float(loaded_slide.tifffile_midres_downsample) / target_downsample
            out_w = max(1, int(round(crop.shape[1] * scale)))
            out_h = max(1, int(round(crop.shape[0] * scale)))
            crop = cv2.resize(crop, (out_w, out_h), interpolation=cv2.INTER_AREA)
        return crop

    x0 = int(round(x1_ov * overview_downsample))
    y0 = int(round(y1_ov * overview_downsample))
    x1 = max(0, min(slide_handle.full_w - 1, x0))
    y1 = max(0, min(slide_handle.full_h - 1, y0))
    x2 = max(x1 + 1, min(slide_handle.full_w, int(round(x2_ov * overview_downsample))))
    y2 = max(y1 + 1, min(slide_handle.full_h, int(round(y2_ov * overview_downsample))))
    crop0 = np.asarray(slide_handle.page0_arr[y1:y2, x1:x2, :], dtype=np.uint8)
    out_w = max(1, int(round((x2 - x1) / target_downsample)))
    out_h = max(1, int(round((y2 - y1) / target_downsample)))
    return cv2.resize(crop0, (out_w, out_h), interpolation=cv2.INTER_AREA)


def _boundary_strip_refine_bbox(
    base_rect_ov: tuple[int, int, int, int],
    loaded_slide,
    slide_handle,
    stain: str,
    overview_shape_wh: tuple[int, int],
    *,
    search_ratio: float = 0.10,
    strip_ratio: float = 0.12,
    occupancy_thresh: float = 0.05,
    pad_ratio: float = 0.012,
) -> tuple[int, int, int, int]:
    if slide_handle is None:
        return base_rect_ov
    refine_level = max(0, loaded_slide.overview_level - 1)
    if refine_level == loaded_slide.overview_level:
        return base_rect_ov

    x1, y1, x2, y2 = base_rect_ov
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    search_pad_ov = max(10, int(round(max(bw, bh) * search_ratio)))
    patch_rect_ov = clamp_crop_bbox(
        x1 - search_pad_ov,
        y1 - search_pad_ov,
        x2 + search_pad_ov,
        y2 + search_pad_ov,
        overview_shape_wh,
    )
    patch_rgb = _extract_overview_rect_at_level(loaded_slide, slide_handle, patch_rect_ov, refine_level)
    score_maps = bbox_score_maps_for_stain(patch_rgb, stain)
    score = score_maps["hybrid"] if stain.lower() == "gallyas" else score_maps["primary"]

    overview_ds = float(loaded_slide.level_downsamples[loaded_slide.overview_level])
    refine_ds = float(loaded_slide.level_downsamples[refine_level])
    scale = overview_ds / refine_ds

    px1 = int(round((x1 - patch_rect_ov[0]) * scale))
    py1 = int(round((y1 - patch_rect_ov[1]) * scale))
    px2 = int(round((x2 - patch_rect_ov[0]) * scale))
    py2 = int(round((y2 - patch_rect_ov[1]) * scale))
    px1 = max(0, min(score.shape[1] - 1, px1))
    py1 = max(0, min(score.shape[0] - 1, py1))
    px2 = max(px1 + 1, min(score.shape[1], px2))
    py2 = max(py1 + 1, min(score.shape[0], py2))

    inside = score[py1:py2, px1:px2]
    vals = inside[inside > 0]
    if vals.size == 0:
        return base_rect_ov
    thresh = max(6.0, float(np.quantile(vals, 0.18)) * 0.86)
    support = score >= thresh
    k = np.ones((5, 5), np.uint8)
    support = cv2.morphologyEx((support.astype(np.uint8) * 255), cv2.MORPH_CLOSE, k) > 0
    support = cv2.morphologyEx((support.astype(np.uint8) * 255), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0

    inset_x = max(4, int(round((px2 - px1) * 0.08)))
    inset_y = max(4, int(round((py2 - py1) * 0.08)))
    sx1 = min(px2 - 1, px1 + inset_x)
    sy1 = min(py2 - 1, py1 + inset_y)
    sx2 = max(sx1 + 1, px2 - inset_x)
    sy2 = max(sy1 + 1, py2 - inset_y)
    seed = np.zeros_like(support, dtype=bool)
    seed[sy1:sy2, sx1:sx2] = True
    seed &= support
    if not seed.any():
        seed = support[py1:py2, px1:px2].copy()
        full_seed = np.zeros_like(support, dtype=bool)
        full_seed[py1:py2, px1:px2] = seed
        seed = full_seed
    connected = binary_propagation(seed, mask=support)
    connected = binary_fill_holes(connected)
    connected = cv2.morphologyEx((connected.astype(np.uint8) * 255), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)) > 0

    side_margin_x = max(3, int(round((px2 - px1) * 0.08)))
    side_margin_y = max(3, int(round((py2 - py1) * 0.08)))
    xspan1 = max(0, px1 + side_margin_x)
    xspan2 = min(connected.shape[1], px2 - side_margin_x)
    yspan1 = max(0, py1 + side_margin_y)
    yspan2 = min(connected.shape[0], py2 - side_margin_y)
    if xspan2 <= xspan1 or yspan2 <= yspan1:
        return base_rect_ov

    row_occ = connected[:, xspan1:xspan2].mean(axis=1)
    col_occ = connected[yspan1:yspan2, :].mean(axis=0)
    row_occ = smooth1d(row_occ.astype(np.float32), ksize=9)
    col_occ = smooth1d(col_occ.astype(np.float32), ksize=9)

    search_ref = max(6, int(round(max(px2 - px1, py2 - py1) * strip_ratio)))

    top_lo = max(0, py1 - search_ref)
    top_hi = min(connected.shape[0], py1 + search_ref)
    bottom_lo = max(0, py2 - search_ref)
    bottom_hi = min(connected.shape[0], py2 + search_ref)
    left_lo = max(0, px1 - search_ref)
    left_hi = min(connected.shape[1], px1 + search_ref)
    right_lo = max(0, px2 - search_ref)
    right_hi = min(connected.shape[1], px2 + search_ref)

    top_new = py1
    idxs = np.where(row_occ[top_lo:top_hi] >= occupancy_thresh)[0]
    if idxs.size:
        top_new = top_lo + int(idxs[0])

    bottom_new = py2
    idxs = np.where(row_occ[bottom_lo:bottom_hi] >= occupancy_thresh)[0]
    if idxs.size:
        bottom_new = bottom_lo + int(idxs[-1]) + 1

    left_new = px1
    idxs = np.where(col_occ[left_lo:left_hi] >= occupancy_thresh)[0]
    if idxs.size:
        left_new = left_lo + int(idxs[0])

    right_new = px2
    idxs = np.where(col_occ[right_lo:right_hi] >= occupancy_thresh)[0]
    if idxs.size:
        right_new = right_lo + int(idxs[-1]) + 1

    rx1 = patch_rect_ov[0] + int(round(left_new / scale))
    ry1 = patch_rect_ov[1] + int(round(top_new / scale))
    rx2 = patch_rect_ov[0] + int(round(right_new / scale))
    ry2 = patch_rect_ov[1] + int(round(bottom_new / scale))

    pad = max(2, int(round(max(rx2 - rx1, ry2 - ry1) * pad_ratio)))
    return clamp_crop_bbox(rx1 - pad, ry1 - pad, rx2 + pad, ry2 + pad, overview_shape_wh)


def build_methods_for_scores(
    score_maps: dict[str, np.ndarray],
    slide_candidates: list[CandidateBox] | None = None,
    loaded_slide=None,
    slide_handle=None,
    overview_rgb: np.ndarray | None = None,
) -> dict[str, Callable[[CandidateBox, tuple[int, int]], tuple[int, int, int, int]]]:
    primary = score_maps["primary"]
    legacy = score_maps["legacy"]
    hybrid = score_maps["hybrid"]
    nonwhite = score_maps["nonwhite"]
    fringe_hybrid_nonwhite = score_maps["fringe_hybrid_nonwhite"]
    adaptive_dark = None
    if overview_rgb is not None:
        adaptive_dark = (_local_adaptive_dark_mask(overview_rgb).astype(np.uint8) * 255)
        fringe_adaptive = np.maximum(hybrid, adaptive_dark).astype(np.uint8)
    else:
        fringe_adaptive = hybrid
    all_candidates = list(slide_candidates or [])

    return {
        "raw_support_bbox": lambda cand, shape_wh: clamp_crop_bbox(
            cand.x, cand.y, cand.x + cand.w, cand.y + cand.h, shape_wh
        ),
        "uniform01_min0": lambda cand, shape_wh: expand_candidate_bbox(
            cand, shape_wh, left_ratio=0.01, top_ratio=0.01, right_ratio=0.01, bottom_ratio=0.01, min_pad=0
        ),
        "uniform015_min0": lambda cand, shape_wh: expand_candidate_bbox(
            cand, shape_wh, left_ratio=0.015, top_ratio=0.015, right_ratio=0.015, bottom_ratio=0.015, min_pad=0
        ),
        "uniform02_min0": lambda cand, shape_wh: expand_candidate_bbox(
            cand, shape_wh, left_ratio=0.02, top_ratio=0.02, right_ratio=0.02, bottom_ratio=0.02, min_pad=0
        ),
        "uniform025_min0": lambda cand, shape_wh: expand_candidate_bbox(
            cand, shape_wh, left_ratio=0.025, top_ratio=0.025, right_ratio=0.025, bottom_ratio=0.025, min_pad=0
        ),
        "uniform04_min0": lambda cand, shape_wh: expand_candidate_bbox(
            cand, shape_wh, left_ratio=0.04, top_ratio=0.04, right_ratio=0.04, bottom_ratio=0.04, min_pad=0
        ),
        "uniform06_min0": lambda cand, shape_wh: expand_candidate_bbox(
            cand, shape_wh, left_ratio=0.06, top_ratio=0.06, right_ratio=0.06, bottom_ratio=0.06, min_pad=0
        ),
        "uniform08_min0": lambda cand, shape_wh: expand_candidate_bbox(
            cand, shape_wh, left_ratio=0.08, top_ratio=0.08, right_ratio=0.08, bottom_ratio=0.08, min_pad=0
        ),
        "baseline_uniform8": lambda cand, shape_wh: expand_candidate_bbox(
            cand, shape_wh, left_ratio=0.08, top_ratio=0.08, right_ratio=0.08, bottom_ratio=0.08
        ),
        "uniform08_min8": lambda cand, shape_wh: expand_candidate_bbox(
            cand, shape_wh, left_ratio=0.08, top_ratio=0.08, right_ratio=0.08, bottom_ratio=0.08, min_pad=8
        ),
        "uniform12": lambda cand, shape_wh: expand_candidate_bbox(
            cand, shape_wh, left_ratio=0.12, top_ratio=0.12, right_ratio=0.12, bottom_ratio=0.12
        ),
        "uniform20": lambda cand, shape_wh: expand_candidate_bbox(
            cand, shape_wh, left_ratio=0.20, top_ratio=0.20, right_ratio=0.20, bottom_ratio=0.20
        ),
        "top_bias12_min0": lambda cand, shape_wh: expand_candidate_bbox(
            cand, shape_wh, left_ratio=0.04, top_ratio=0.12, right_ratio=0.04, bottom_ratio=0.04, min_pad=0
        ),
        "top_bias20": lambda cand, shape_wh: expand_candidate_bbox(
            cand, shape_wh, left_ratio=0.08, top_ratio=0.20, right_ratio=0.08, bottom_ratio=0.08
        ),
        "top_bias30": lambda cand, shape_wh: expand_candidate_bbox(
            cand, shape_wh, left_ratio=0.10, top_ratio=0.30, right_ratio=0.10, bottom_ratio=0.10
        ),
        "top_bias45_wide20": lambda cand, shape_wh: expand_candidate_bbox(
            cand, shape_wh, left_ratio=0.20, top_ratio=0.45, right_ratio=0.20, bottom_ratio=0.20
        ),
        "top_bias55_wide24": lambda cand, shape_wh: expand_candidate_bbox(
            cand, shape_wh, left_ratio=0.24, top_ratio=0.55, right_ratio=0.24, bottom_ratio=0.24
        ),
        "top_bias65_wide28": lambda cand, shape_wh: expand_candidate_bbox(
            cand, shape_wh, left_ratio=0.28, top_ratio=0.65, right_ratio=0.28, bottom_ratio=0.28
        ),
        "projection_top_v1": lambda cand, shape_wh: projection_expand_bbox(
            cand, primary, shape_wh, top_cap_ratio=0.30, top_only=True
        ),
        "projection_full_v1": lambda cand, shape_wh: projection_expand_bbox(
            cand, primary, shape_wh, top_cap_ratio=0.30, bottom_cap_ratio=0.14, side_cap_ratio=0.10, top_only=False
        ),
        "projection_full_topfloor20": lambda cand, shape_wh: projection_full_topfloor(
            cand, primary, shape_wh, left_ratio=0.08, top_ratio=0.20, right_ratio=0.08, bottom_ratio=0.08
        ),
        "projection_full_topfloor30": lambda cand, shape_wh: projection_full_topfloor(
            cand, primary, shape_wh, left_ratio=0.10, top_ratio=0.30, right_ratio=0.10, bottom_ratio=0.10
        ),
        "projection_full_topfloor35": lambda cand, shape_wh: projection_full_topfloor(
            cand, primary, shape_wh, left_ratio=0.12, top_ratio=0.35, right_ratio=0.12, bottom_ratio=0.12
        ),
        "projection_full_topfloor35_wide20": lambda cand, shape_wh: rect_union(
            projection_full_topfloor(cand, primary, shape_wh, left_ratio=0.12, top_ratio=0.35, right_ratio=0.12, bottom_ratio=0.12),
            expand_candidate_bbox(cand, shape_wh, left_ratio=0.20, top_ratio=0.20, right_ratio=0.20, bottom_ratio=0.20),
            shape_wh,
        ),
        "projection_full_topfloor45_wide20": lambda cand, shape_wh: rect_union(
            projection_full_topfloor(
                cand,
                primary,
                shape_wh,
                left_ratio=0.20,
                top_ratio=0.45,
                right_ratio=0.20,
                bottom_ratio=0.20,
            ),
            expand_candidate_bbox(cand, shape_wh, left_ratio=0.20, top_ratio=0.20, right_ratio=0.20, bottom_ratio=0.20),
            shape_wh,
        ),
        "projection_full_topfloor55_wide24": lambda cand, shape_wh: rect_union(
            projection_full_topfloor(
                cand,
                primary,
                shape_wh,
                left_ratio=0.24,
                top_ratio=0.55,
                right_ratio=0.24,
                bottom_ratio=0.24,
            ),
            expand_candidate_bbox(cand, shape_wh, left_ratio=0.24, top_ratio=0.24, right_ratio=0.24, bottom_ratio=0.24),
            shape_wh,
        ),
        "projection_full_topfloor65_wide28": lambda cand, shape_wh: rect_union(
            projection_full_topfloor(
                cand,
                primary,
                shape_wh,
                left_ratio=0.28,
                top_ratio=0.65,
                right_ratio=0.28,
                bottom_ratio=0.28,
            ),
            expand_candidate_bbox(cand, shape_wh, left_ratio=0.28, top_ratio=0.28, right_ratio=0.28, bottom_ratio=0.28),
            shape_wh,
        ),
        "projection_relaxed45_wide20_t036": lambda cand, shape_wh: rect_union(
            projection_full_topfloor(
                cand,
                primary,
                shape_wh,
                left_ratio=0.20,
                top_ratio=0.45,
                right_ratio=0.20,
                bottom_ratio=0.20,
                thresh_scale=0.36,
                proj_top_cap_ratio=0.45,
                proj_bottom_cap_ratio=0.20,
                proj_side_cap_ratio=0.15,
                proj_max_gap=5,
            ),
            expand_candidate_bbox(cand, shape_wh, left_ratio=0.20, top_ratio=0.20, right_ratio=0.20, bottom_ratio=0.20),
            shape_wh,
        ),
        "projection_relaxed55_wide24_t036": lambda cand, shape_wh: rect_union(
            projection_full_topfloor(
                cand,
                primary,
                shape_wh,
                left_ratio=0.24,
                top_ratio=0.55,
                right_ratio=0.24,
                bottom_ratio=0.24,
                thresh_scale=0.36,
                proj_top_cap_ratio=0.45,
                proj_bottom_cap_ratio=0.20,
                proj_side_cap_ratio=0.15,
                proj_max_gap=5,
            ),
            expand_candidate_bbox(cand, shape_wh, left_ratio=0.24, top_ratio=0.24, right_ratio=0.24, bottom_ratio=0.24),
            shape_wh,
        ),
        "projection_relaxed65_wide28_t030": lambda cand, shape_wh: rect_union(
            projection_full_topfloor(
                cand,
                primary,
                shape_wh,
                left_ratio=0.28,
                top_ratio=0.65,
                right_ratio=0.28,
                bottom_ratio=0.28,
                thresh_scale=0.30,
                proj_top_cap_ratio=0.50,
                proj_bottom_cap_ratio=0.24,
                proj_side_cap_ratio=0.18,
                proj_max_gap=6,
            ),
            expand_candidate_bbox(cand, shape_wh, left_ratio=0.28, top_ratio=0.28, right_ratio=0.28, bottom_ratio=0.28),
            shape_wh,
        ),
        "seed_relaxed65_t030": lambda cand, shape_wh: projection_full_topfloor(
            cand,
            primary,
            shape_wh,
            left_ratio=0.28,
            top_ratio=0.65,
            right_ratio=0.28,
            bottom_ratio=0.28,
            thresh_scale=0.30,
            proj_top_cap_ratio=0.50,
            proj_bottom_cap_ratio=0.24,
            proj_side_cap_ratio=0.18,
            proj_max_gap=6,
        ),
        "seed_relaxed70_t028": lambda cand, shape_wh: projection_full_topfloor(
            cand,
            primary,
            shape_wh,
            left_ratio=0.30,
            top_ratio=0.70,
            right_ratio=0.30,
            bottom_ratio=0.28,
            thresh_scale=0.28,
            proj_top_cap_ratio=0.54,
            proj_bottom_cap_ratio=0.25,
            proj_side_cap_ratio=0.20,
            proj_max_gap=6,
        ),
        "seed_relaxed70_side29_t028": lambda cand, shape_wh: projection_full_topfloor(
            cand,
            primary,
            shape_wh,
            left_ratio=0.29,
            top_ratio=0.70,
            right_ratio=0.29,
            bottom_ratio=0.27,
            thresh_scale=0.28,
            proj_top_cap_ratio=0.54,
            proj_bottom_cap_ratio=0.25,
            proj_side_cap_ratio=0.19,
            proj_max_gap=6,
        ),
        "seed_relaxed72_t027": lambda cand, shape_wh: projection_full_topfloor(
            cand,
            primary,
            shape_wh,
            left_ratio=0.31,
            top_ratio=0.72,
            right_ratio=0.31,
            bottom_ratio=0.28,
            thresh_scale=0.27,
            proj_top_cap_ratio=0.56,
            proj_bottom_cap_ratio=0.25,
            proj_side_cap_ratio=0.21,
            proj_max_gap=7,
        ),
        "seed_relaxed72_side30_t027": lambda cand, shape_wh: projection_full_topfloor(
            cand,
            primary,
            shape_wh,
            left_ratio=0.30,
            top_ratio=0.72,
            right_ratio=0.30,
            bottom_ratio=0.27,
            thresh_scale=0.27,
            proj_top_cap_ratio=0.56,
            proj_bottom_cap_ratio=0.25,
            proj_side_cap_ratio=0.20,
            proj_max_gap=7,
        ),
        "seed_relaxed74_t026": lambda cand, shape_wh: projection_full_topfloor(
            cand,
            primary,
            shape_wh,
            left_ratio=0.32,
            top_ratio=0.74,
            right_ratio=0.32,
            bottom_ratio=0.28,
            thresh_scale=0.26,
            proj_top_cap_ratio=0.57,
            proj_bottom_cap_ratio=0.26,
            proj_side_cap_ratio=0.22,
            proj_max_gap=7,
        ),
        "seed_relaxed74_side30_t026": lambda cand, shape_wh: projection_full_topfloor(
            cand,
            primary,
            shape_wh,
            left_ratio=0.30,
            top_ratio=0.74,
            right_ratio=0.30,
            bottom_ratio=0.27,
            thresh_scale=0.26,
            proj_top_cap_ratio=0.57,
            proj_bottom_cap_ratio=0.26,
            proj_side_cap_ratio=0.21,
            proj_max_gap=7,
        ),
        "seed_relaxed74_side28_t026": lambda cand, shape_wh: projection_full_topfloor(
            cand,
            primary,
            shape_wh,
            left_ratio=0.28,
            top_ratio=0.74,
            right_ratio=0.28,
            bottom_ratio=0.26,
            thresh_scale=0.26,
            proj_top_cap_ratio=0.57,
            proj_bottom_cap_ratio=0.25,
            proj_side_cap_ratio=0.20,
            proj_max_gap=7,
        ),
        "projection_relaxed75_wide32_t026": lambda cand, shape_wh: rect_union(
            projection_full_topfloor(
                cand,
                primary,
                shape_wh,
                left_ratio=0.32,
                top_ratio=0.75,
                right_ratio=0.32,
                bottom_ratio=0.28,
                thresh_scale=0.26,
                proj_top_cap_ratio=0.58,
                proj_bottom_cap_ratio=0.26,
                proj_side_cap_ratio=0.22,
                proj_max_gap=7,
            ),
            expand_candidate_bbox(cand, shape_wh, left_ratio=0.32, top_ratio=0.32, right_ratio=0.32, bottom_ratio=0.28),
            shape_wh,
        ),
        "seed_relaxed75_t026": lambda cand, shape_wh: projection_full_topfloor(
            cand,
            primary,
            shape_wh,
            left_ratio=0.32,
            top_ratio=0.75,
            right_ratio=0.32,
            bottom_ratio=0.28,
            thresh_scale=0.26,
            proj_top_cap_ratio=0.58,
            proj_bottom_cap_ratio=0.26,
            proj_side_cap_ratio=0.22,
            proj_max_gap=7,
        ),
        "seed_relaxed80_t024": lambda cand, shape_wh: projection_full_topfloor(
            cand,
            primary,
            shape_wh,
            left_ratio=0.34,
            top_ratio=0.80,
            right_ratio=0.34,
            bottom_ratio=0.29,
            thresh_scale=0.24,
            proj_top_cap_ratio=0.61,
            proj_bottom_cap_ratio=0.27,
            proj_side_cap_ratio=0.23,
            proj_max_gap=7,
        ),
        "projection_relaxed85_wide36_t022": lambda cand, shape_wh: rect_union(
            projection_full_topfloor(
                cand,
                primary,
                shape_wh,
                left_ratio=0.36,
                top_ratio=0.85,
                right_ratio=0.36,
                bottom_ratio=0.30,
                thresh_scale=0.22,
                proj_top_cap_ratio=0.64,
                proj_bottom_cap_ratio=0.28,
                proj_side_cap_ratio=0.24,
                proj_max_gap=8,
            ),
            expand_candidate_bbox(cand, shape_wh, left_ratio=0.36, top_ratio=0.36, right_ratio=0.36, bottom_ratio=0.30),
            shape_wh,
        ),
        "seed_relaxed85_t022": lambda cand, shape_wh: projection_full_topfloor(
            cand,
            primary,
            shape_wh,
            left_ratio=0.36,
            top_ratio=0.85,
            right_ratio=0.36,
            bottom_ratio=0.30,
            thresh_scale=0.22,
            proj_top_cap_ratio=0.64,
            proj_bottom_cap_ratio=0.28,
            proj_side_cap_ratio=0.24,
            proj_max_gap=8,
        ),
        "legacy_topfloor55_wide24": lambda cand, shape_wh: rect_union(
            projection_full_topfloor(
                cand,
                legacy,
                shape_wh,
                left_ratio=0.24,
                top_ratio=0.55,
                right_ratio=0.24,
                bottom_ratio=0.24,
                thresh_scale=0.38,
                proj_top_cap_ratio=0.45,
                proj_bottom_cap_ratio=0.20,
                proj_side_cap_ratio=0.16,
                proj_max_gap=5,
            ),
            expand_candidate_bbox(cand, shape_wh, left_ratio=0.24, top_ratio=0.24, right_ratio=0.24, bottom_ratio=0.24),
            shape_wh,
        ),
        "legacy_relaxed75_wide32_t026": lambda cand, shape_wh: rect_union(
            projection_full_topfloor(
                cand,
                legacy,
                shape_wh,
                left_ratio=0.32,
                top_ratio=0.75,
                right_ratio=0.32,
                bottom_ratio=0.28,
                thresh_scale=0.26,
                proj_top_cap_ratio=0.58,
                proj_bottom_cap_ratio=0.26,
                proj_side_cap_ratio=0.22,
                proj_max_gap=7,
            ),
            expand_candidate_bbox(cand, shape_wh, left_ratio=0.32, top_ratio=0.32, right_ratio=0.32, bottom_ratio=0.28),
            shape_wh,
        ),
        "legacy_relaxed85_wide36_t022": lambda cand, shape_wh: rect_union(
            projection_full_topfloor(
                cand,
                legacy,
                shape_wh,
                left_ratio=0.36,
                top_ratio=0.85,
                right_ratio=0.36,
                bottom_ratio=0.30,
                thresh_scale=0.22,
                proj_top_cap_ratio=0.64,
                proj_bottom_cap_ratio=0.28,
                proj_side_cap_ratio=0.24,
                proj_max_gap=8,
            ),
            expand_candidate_bbox(cand, shape_wh, left_ratio=0.36, top_ratio=0.36, right_ratio=0.36, bottom_ratio=0.30),
            shape_wh,
        ),
        "hybrid_topfloor45_wide20": lambda cand, shape_wh: rect_union(
            projection_full_topfloor(
                cand,
                hybrid,
                shape_wh,
                left_ratio=0.20,
                top_ratio=0.45,
                right_ratio=0.20,
                bottom_ratio=0.20,
                thresh_scale=0.34,
                proj_top_cap_ratio=0.45,
                proj_bottom_cap_ratio=0.20,
                proj_side_cap_ratio=0.16,
                proj_max_gap=5,
            ),
            expand_candidate_bbox(cand, shape_wh, left_ratio=0.20, top_ratio=0.20, right_ratio=0.20, bottom_ratio=0.20),
            shape_wh,
        ),
        "hybrid_topfloor55_wide24": lambda cand, shape_wh: rect_union(
            projection_full_topfloor(
                cand,
                hybrid,
                shape_wh,
                left_ratio=0.24,
                top_ratio=0.55,
                right_ratio=0.24,
                bottom_ratio=0.24,
                thresh_scale=0.32,
                proj_top_cap_ratio=0.48,
                proj_bottom_cap_ratio=0.22,
                proj_side_cap_ratio=0.18,
                proj_max_gap=6,
            ),
            expand_candidate_bbox(cand, shape_wh, left_ratio=0.24, top_ratio=0.24, right_ratio=0.24, bottom_ratio=0.24),
            shape_wh,
        ),
        "hybrid_topfloor58_wide24": lambda cand, shape_wh: rect_union(
            projection_full_topfloor(
                cand,
                hybrid,
                shape_wh,
                left_ratio=0.24,
                top_ratio=0.58,
                right_ratio=0.24,
                bottom_ratio=0.24,
                thresh_scale=0.31,
                proj_top_cap_ratio=0.50,
                proj_bottom_cap_ratio=0.22,
                proj_side_cap_ratio=0.18,
                proj_max_gap=6,
            ),
            expand_candidate_bbox(cand, shape_wh, left_ratio=0.24, top_ratio=0.24, right_ratio=0.24, bottom_ratio=0.24),
            shape_wh,
        ),
        "hybrid_topfloor60_wide24": lambda cand, shape_wh: rect_union(
            projection_full_topfloor(
                cand,
                hybrid,
                shape_wh,
                left_ratio=0.24,
                top_ratio=0.60,
                right_ratio=0.24,
                bottom_ratio=0.24,
                thresh_scale=0.30,
                proj_top_cap_ratio=0.51,
                proj_bottom_cap_ratio=0.22,
                proj_side_cap_ratio=0.18,
                proj_max_gap=6,
            ),
            expand_candidate_bbox(cand, shape_wh, left_ratio=0.24, top_ratio=0.24, right_ratio=0.24, bottom_ratio=0.24),
            shape_wh,
        ),
        "hybrid_topfloor60_wide26": lambda cand, shape_wh: rect_union(
            projection_full_topfloor(
                cand,
                hybrid,
                shape_wh,
                left_ratio=0.26,
                top_ratio=0.60,
                right_ratio=0.26,
                bottom_ratio=0.25,
                thresh_scale=0.30,
                proj_top_cap_ratio=0.51,
                proj_bottom_cap_ratio=0.23,
                proj_side_cap_ratio=0.19,
                proj_max_gap=6,
            ),
            expand_candidate_bbox(cand, shape_wh, left_ratio=0.26, top_ratio=0.25, right_ratio=0.26, bottom_ratio=0.25),
            shape_wh,
        ),
        "hybrid_relaxed65_wide28_t030": lambda cand, shape_wh: rect_union(
            projection_full_topfloor(
                cand,
                hybrid,
                shape_wh,
                left_ratio=0.28,
                top_ratio=0.65,
                right_ratio=0.28,
                bottom_ratio=0.28,
                thresh_scale=0.30,
                proj_top_cap_ratio=0.52,
                proj_bottom_cap_ratio=0.24,
                proj_side_cap_ratio=0.20,
                proj_max_gap=6,
            ),
            expand_candidate_bbox(cand, shape_wh, left_ratio=0.28, top_ratio=0.28, right_ratio=0.28, bottom_ratio=0.28),
            shape_wh,
        ),
        "hybrid_relaxed75_wide32_t026": lambda cand, shape_wh: rect_union(
            projection_full_topfloor(
                cand,
                hybrid,
                shape_wh,
                left_ratio=0.32,
                top_ratio=0.75,
                right_ratio=0.32,
                bottom_ratio=0.28,
                thresh_scale=0.26,
                proj_top_cap_ratio=0.58,
                proj_bottom_cap_ratio=0.26,
                proj_side_cap_ratio=0.22,
                proj_max_gap=7,
            ),
            expand_candidate_bbox(cand, shape_wh, left_ratio=0.32, top_ratio=0.32, right_ratio=0.32, bottom_ratio=0.28),
            shape_wh,
        ),
        "exp1_core84_fringe52_resid": lambda cand, shape_wh: _support_bbox_from_core_and_fringe(
            cand,
            shape_wh,
            primary,
            primary,
            core_quantile=0.84,
            fringe_quantile=0.52,
            fringe_scale=0.92,
            pad_ratio=0.04,
        ),
        "exp1_core84_fringe52_hybrid": lambda cand, shape_wh: _support_bbox_from_core_and_fringe(
            cand,
            shape_wh,
            primary,
            hybrid,
            core_quantile=0.84,
            fringe_quantile=0.52,
            fringe_scale=0.92,
            pad_ratio=0.04,
        ),
        "exp1_core82_fringe48_hybrid_nonwhite": lambda cand, shape_wh: _support_bbox_from_core_and_fringe(
            cand,
            shape_wh,
            primary,
            fringe_hybrid_nonwhite,
            core_quantile=0.82,
            fringe_quantile=0.48,
            fringe_scale=0.90,
            pad_ratio=0.04,
        ),
        "exp1_core86_fringe55_hybrid_tight": lambda cand, shape_wh: _support_bbox_from_core_and_fringe(
            cand,
            shape_wh,
            primary,
            hybrid,
            search_left_ratio=0.22,
            search_top_ratio=0.50,
            search_right_ratio=0.22,
            search_bottom_ratio=0.22,
            core_quantile=0.86,
            fringe_quantile=0.55,
            fringe_scale=0.94,
            pad_ratio=0.035,
        ),
        "exp2_compete_core84_fringe52_hybrid": lambda cand, shape_wh: _competitive_support_bbox(
            cand,
            all_candidates,
            shape_wh,
            primary,
            hybrid,
            core_quantile=0.84,
            fringe_quantile=0.52,
            fringe_scale=0.92,
            pad_ratio=0.04,
        ),
        "exp2_compete_core82_fringe48_hybrid_nonwhite": lambda cand, shape_wh: _competitive_support_bbox(
            cand,
            all_candidates,
            shape_wh,
            primary,
            fringe_hybrid_nonwhite,
            core_quantile=0.82,
            fringe_quantile=0.48,
            fringe_scale=0.90,
            pad_ratio=0.04,
        ),
        "exp2_compete_core86_fringe55_hybrid_tight": lambda cand, shape_wh: _competitive_support_bbox(
            cand,
            all_candidates,
            shape_wh,
            primary,
            hybrid,
            core_quantile=0.86,
            fringe_quantile=0.55,
            fringe_scale=0.94,
            pad_ratio=0.035,
        ),
        "dr_localadaptive_compete_v1": lambda cand, shape_wh: _competitive_support_bbox(
            cand,
            all_candidates,
            shape_wh,
            primary,
            fringe_adaptive,
            core_quantile=0.86,
            fringe_quantile=0.55,
            fringe_scale=0.94,
            pad_ratio=0.035,
        ),
        "dr_watershed_hybrid_v1": lambda cand, shape_wh: _watershed_bbox(
            cand,
            all_candidates,
            shape_wh,
            primary,
            hybrid,
            core_quantile=0.86,
            fringe_quantile=0.55,
            fringe_scale=0.94,
            pad_ratio=0.035,
        ),
        "dr_boxopt_exp2_v1": lambda cand, shape_wh: _edgewise_boxopt_bbox(
            cand,
            all_candidates,
            shape_wh,
            primary,
            hybrid,
            core_quantile=0.86,
            fringe_quantile=0.55,
            fringe_scale=0.94,
            pad_ratio=0.02,
            search_px=16,
            step_px=2,
        ),
        "exp2_cost_support_v1": lambda cand, shape_wh: _competitive_support_bbox_cost(
            cand,
            all_candidates,
            shape_wh,
            primary,
            hybrid,
            core_quantile=0.86,
            fringe_quantile=0.55,
            fringe_scale=0.94,
            pad_ratio=0.035,
            use_unknown=False,
            support_dilate_px=9,
        ),
        "exp2_cost_support_unknown_v1": lambda cand, shape_wh: _competitive_support_bbox_cost(
            cand,
            all_candidates,
            shape_wh,
            primary,
            hybrid,
            core_quantile=0.86,
            fringe_quantile=0.55,
            fringe_scale=0.94,
            pad_ratio=0.035,
            use_unknown=True,
            support_dilate_px=9,
        ),
        "exp2_cost_support_unknown_tight_v1": lambda cand, shape_wh: _competitive_support_bbox_cost(
            cand,
            all_candidates,
            shape_wh,
            primary,
            hybrid,
            core_quantile=0.88,
            fringe_quantile=0.56,
            fringe_scale=0.95,
            pad_ratio=0.032,
            use_unknown=True,
            support_dilate_px=7,
        ),
        "exp4_strip_refine_exp2_occ05": lambda cand, shape_wh: _boundary_strip_refine_bbox(
            _competitive_support_bbox(
                cand,
                all_candidates,
                shape_wh,
                primary,
                hybrid,
                core_quantile=0.86,
                fringe_quantile=0.55,
                fringe_scale=0.94,
                pad_ratio=0.035,
            ),
            loaded_slide,
            slide_handle,
            loaded_slide.stain if loaded_slide is not None else "gallyas",
            shape_wh,
            search_ratio=0.10,
            strip_ratio=0.12,
            occupancy_thresh=0.05,
            pad_ratio=0.012,
        ),
        "exp4_strip_refine_exp2_occ04": lambda cand, shape_wh: _boundary_strip_refine_bbox(
            _competitive_support_bbox(
                cand,
                all_candidates,
                shape_wh,
                primary,
                hybrid,
                core_quantile=0.86,
                fringe_quantile=0.55,
                fringe_scale=0.94,
                pad_ratio=0.035,
            ),
            loaded_slide,
            slide_handle,
            loaded_slide.stain if loaded_slide is not None else "gallyas",
            shape_wh,
            search_ratio=0.12,
            strip_ratio=0.14,
            occupancy_thresh=0.04,
            pad_ratio=0.012,
        ),
        "exp4_strip_refine_exp2_occ06_tight": lambda cand, shape_wh: _boundary_strip_refine_bbox(
            _competitive_support_bbox(
                cand,
                all_candidates,
                shape_wh,
                primary,
                hybrid,
                core_quantile=0.86,
                fringe_quantile=0.55,
                fringe_scale=0.94,
                pad_ratio=0.035,
            ),
            loaded_slide,
            slide_handle,
            loaded_slide.stain if loaded_slide is not None else "gallyas",
            shape_wh,
            search_ratio=0.08,
            strip_ratio=0.10,
            occupancy_thresh=0.06,
            pad_ratio=0.010,
        ),
        "hybrid_relaxed85_wide36_t022": lambda cand, shape_wh: rect_union(
            projection_full_topfloor(
                cand,
                hybrid,
                shape_wh,
                left_ratio=0.36,
                top_ratio=0.85,
                right_ratio=0.36,
                bottom_ratio=0.30,
                thresh_scale=0.22,
                proj_top_cap_ratio=0.64,
                proj_bottom_cap_ratio=0.28,
                proj_side_cap_ratio=0.24,
                proj_max_gap=8,
            ),
            expand_candidate_bbox(cand, shape_wh, left_ratio=0.36, top_ratio=0.36, right_ratio=0.36, bottom_ratio=0.30),
            shape_wh,
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ndpi-root", required=True)
    parser.add_argument("--gt-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--stain", required=True)
    parser.add_argument("--sample-ids", nargs="*", default=[])
    parser.add_argument("--methods", nargs="*", default=[])
    args = parser.parse_args()

    ndpi_root = Path(args.ndpi_root)
    gt_root = Path(args.gt_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    examples_dir = output_dir / "examples"
    examples_dir.mkdir(exist_ok=True)

    gt_sections = collect_gt_sections(gt_root)
    if args.sample_ids:
        keep = set(args.sample_ids)
        gt_sections = [sec for sec in gt_sections if sec.sample_id in keep]
    slide_index = build_section_to_slide_index(ndpi_root)
    stain_key = str(args.stain).lower()
    gt_sections_by_slide: dict[Path, list[GtSection]] = {}
    for gt in gt_sections:
        slide_path = slide_index.get((stain_key, gt.sample_id, gt.section_id))
        if slide_path is not None:
            gt_sections_by_slide.setdefault(slide_path, []).append(gt)

    dummy_scores = {
        "primary": np.zeros((8, 8), dtype=np.uint8),
        "legacy": np.zeros((8, 8), dtype=np.uint8),
        "hybrid": np.zeros((8, 8), dtype=np.uint8),
        "nonwhite": np.zeros((8, 8), dtype=np.uint8),
        "fringe_hybrid_nonwhite": np.zeros((8, 8), dtype=np.uint8),
    }
    method_names = list(build_methods_for_scores(dummy_scores).keys())
    if args.methods:
        requested = set(args.methods)
        method_names = [name for name in method_names if name in requested]
        missing = sorted(requested.difference(method_names))
        if missing:
            raise SystemExit(f"Unknown methods requested: {missing}")
    rows: list[dict] = []
    aggregate: dict[str, list[dict]] = {name: [] for name in method_names}
    slide_cache: dict[Path, tuple[object, object, np.ndarray, tuple[int, int], float, dict[str, CandidateBox], list[CandidateBox], dict[str, np.ndarray]]] = {}

    total = len(gt_sections)
    for idx_gt, gt in enumerate(gt_sections, start=1):
        print(f"[{idx_gt}/{total}] {gt.label}", flush=True)
        slide_path = slide_index.get((stain_key, gt.sample_id, gt.section_id))
        if slide_path is None:
            for method_name in method_names:
                row = {
                    "section": gt.label,
                    "slide_name": "",
                    "method": method_name,
                    "proposal_found": False,
                    "mask_coverage_recall": 0.0,
                    "mask_miss_ratio": 1.0,
                    "crop_efficiency": 0.0,
                    "crop_area_to_gt_mask_ratio": 0.0,
                    "proposal_rect_vs_gtbbox_iou": 0.0,
                    "proposal_area_to_gt_crop_area_full": 0.0,
                    "proposal_area_to_gt_mask_area_full": 0.0,
                    "proposal_rect_vs_gtcrop_iou_level0": 0.0,
                    "neighbor_mask_overlap_area_level0": 0.0,
                    "neighbor_overlap_ratio_proposal": 0.0,
                    "neighbor_overlap_ratio_targetmask": 0.0,
                }
                rows.append(row)
                aggregate[method_name].append(row)
            continue

        if slide_path not in slide_cache:
            loaded = load_slide_bundle(slide_path, stain_key)
            overview_rgb = np.asarray(loaded.overview)
            _, labels = parse_slide_stem(slide_path.stem)
            _, _, component_mask = component_mask_from_overview(overview_rgb, stain=stain_key)
            candidates = find_candidate_components(component_mask, len(labels))
            candidates = assign_sections(candidates, labels)
            candidate_map = {cand.section.short_label: cand for cand in candidates if cand.section is not None}
            score_maps = bbox_score_maps_for_stain(overview_rgb, stain_key)
            overview_downsample = float(loaded.level_downsamples[loaded.overview_level])
            slide_handle = open_slide_handle(loaded)
            slide_cache[slide_path] = (loaded, slide_handle, overview_rgb, loaded.overview_size, overview_downsample, candidate_map, candidates, score_maps)

        loaded_slide, slide_handle, overview_rgb, overview_size, overview_downsample, candidate_map, candidates, score_maps = slide_cache[slide_path]
        sibling_gt_sections = [sec for sec in gt_sections_by_slide.get(slide_path, []) if sec.label != gt.label]
        methods = build_methods_for_scores(
            score_maps,
            slide_candidates=candidates,
            loaded_slide=loaded_slide,
            slide_handle=slide_handle,
            overview_rgb=overview_rgb,
        )
        if args.methods:
            methods = {name: fn for name, fn in methods.items() if name in method_names}
        gt_crop_rect = level0_rect_to_overview_rect(gt.gt_crop_bbox_level0, overview_downsample, overview_size)
        candidate = candidate_map.get(gt.label)

        if candidate is None:
            for method_name in method_names:
                row = {
                    "section": gt.label,
                    "slide_name": slide_path.name,
                    "method": method_name,
                    "proposal_found": False,
                    "mask_coverage_recall": 0.0,
                    "mask_miss_ratio": 1.0,
                    "crop_efficiency": 0.0,
                    "crop_area_to_gt_mask_ratio": 0.0,
                    "proposal_rect_vs_gtbbox_iou": 0.0,
                    "proposal_area_to_gt_crop_area_full": 0.0,
                    "proposal_area_to_gt_mask_area_full": 0.0,
                    "proposal_rect_vs_gtcrop_iou_level0": 0.0,
                    "neighbor_mask_overlap_area_level0": 0.0,
                    "neighbor_overlap_ratio_proposal": 0.0,
                    "neighbor_overlap_ratio_targetmask": 0.0,
                }
                rows.append(row)
                aggregate[method_name].append(row)
            continue

        section_best_method = None
        section_best_score = (-1.0, -1.0, -1e18)
        section_best_rect = None
        baseline_rect = None
        for method_name, method_fn in methods.items():
            rect = method_fn(candidate, overview_size)
            rect_level0 = (
                int(round(rect[0] * overview_downsample)),
                int(round(rect[1] * overview_downsample)),
                int(round((rect[2] - rect[0]) * overview_downsample)),
                int(round((rect[3] - rect[1]) * overview_downsample)),
            )
            proposal_bounds_in_gt = project_rect_to_gt_crop_bounds_level0(rect_level0, gt.gt_crop_bbox_level0, gt.crop_shape)
            metrics = coverage_metrics(
                gt.mask,
                proposal_bounds_in_gt,
                proposal_rect_level0_xywh=rect_level0,
                gt_crop_level0_xywh=gt.gt_crop_bbox_level0,
            )
            neighbor_mask_overlap_area_level0 = float(
                sum(rect_overlap_gt_mask_level0_area(rect_level0, other_gt) for other_gt in sibling_gt_sections)
            )
            proposal_area_level0 = float(max(1, rect_level0[2] * rect_level0[3]))
            gh, gw = gt.crop_shape
            _, _, gcw, gch = gt.gt_crop_bbox_level0
            gt_mask_area_level0 = float(max(1.0, int(gt.mask.sum()) * (gcw / max(1.0, float(gw))) * (gch / max(1.0, float(gh)))))
            row = {
                "section": gt.label,
                "slide_name": slide_path.name,
                "method": method_name,
                "proposal_found": True,
                **metrics,
                "neighbor_mask_overlap_area_level0": neighbor_mask_overlap_area_level0,
                "neighbor_overlap_ratio_proposal": neighbor_mask_overlap_area_level0 / proposal_area_level0,
                "neighbor_overlap_ratio_targetmask": neighbor_mask_overlap_area_level0 / gt_mask_area_level0,
                "rect_x1": rect[0],
                "rect_y1": rect[1],
                "rect_x2": rect[2],
                "rect_y2": rect[3],
            }
            rows.append(row)
            aggregate[method_name].append(row)
            if method_name == "baseline_uniform8":
                baseline_rect = rect
            score_tuple = (
                metrics["mask_coverage_recall"],
                -row["neighbor_overlap_ratio_proposal"],
                metrics["proposal_rect_vs_gtcrop_iou_level0"],
                -metrics["proposal_area_to_gt_crop_area_full"],
            )
            if score_tuple > section_best_score:
                section_best_score = score_tuple
                section_best_method = method_name
                section_best_rect = rect

        if baseline_rect is not None and section_best_rect is not None and section_best_method != "baseline_uniform8":
            out = draw_overview_comparison(
                overview_rgb,
                baseline_rect=baseline_rect,
                best_rect=section_best_rect,
                gt_crop_rect=gt_crop_rect,
                label=f"{gt.label} | best={section_best_method}",
            )
            Image.fromarray(out).save(examples_dir / f"{gt.label}_overview_compare.png")

    with (output_dir / "proposal_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {}
    for method_name, method_rows in aggregate.items():
        recalls = np.array([float(r["mask_coverage_recall"]) for r in method_rows], dtype=np.float64)
        efficiencies = np.array([float(r["crop_efficiency"]) for r in method_rows], dtype=np.float64)
        area_ratios = np.array([float(r["crop_area_to_gt_mask_ratio"]) for r in method_rows], dtype=np.float64)
        bbox_ious = np.array([float(r["proposal_rect_vs_gtbbox_iou"]) for r in method_rows], dtype=np.float64)
        full_area_to_crop = np.array([float(r["proposal_area_to_gt_crop_area_full"]) for r in method_rows], dtype=np.float64)
        full_area_to_mask = np.array([float(r["proposal_area_to_gt_mask_area_full"]) for r in method_rows], dtype=np.float64)
        gtcrop_ious = np.array([float(r["proposal_rect_vs_gtcrop_iou_level0"]) for r in method_rows], dtype=np.float64)
        neighbor_overlap_prop = np.array([float(r.get("neighbor_overlap_ratio_proposal", 0.0)) for r in method_rows], dtype=np.float64)
        neighbor_overlap_target = np.array([float(r.get("neighbor_overlap_ratio_targetmask", 0.0)) for r in method_rows], dtype=np.float64)
        found = np.array([bool(r["proposal_found"]) for r in method_rows], dtype=bool)
        compactness_scores = np.minimum(1.0, 1.0 / np.maximum(full_area_to_crop, 1e-6)) if full_area_to_crop.size else np.array([], dtype=np.float64)
        non_neighbor_scores = np.clip(1.0 - neighbor_overlap_prop, 0.0, 1.0) if neighbor_overlap_prop.size else np.array([], dtype=np.float64)
        weighted_scores = (
            0.50 * recalls
            + 0.30 * non_neighbor_scores
            + 0.20 * compactness_scores
        ) if recalls.size else np.array([], dtype=np.float64)
        summary[method_name] = {
            "count": int(len(method_rows)),
            "proposal_found_rate": float(found.mean()) if found.size else 0.0,
            "mean_mask_coverage_recall": float(recalls.mean()) if recalls.size else 0.0,
            "median_mask_coverage_recall": float(np.median(recalls)) if recalls.size else 0.0,
            "full_coverage_rate_99": float((recalls >= 0.99).mean()) if recalls.size else 0.0,
            "coverage_rate_95": float((recalls >= 0.95).mean()) if recalls.size else 0.0,
            "mean_crop_efficiency": float(efficiencies.mean()) if efficiencies.size else 0.0,
            "mean_crop_area_to_gt_mask_ratio": float(area_ratios.mean()) if area_ratios.size else 0.0,
            "mean_proposal_rect_vs_gtbbox_iou": float(bbox_ious.mean()) if bbox_ious.size else 0.0,
            "mean_proposal_area_to_gt_crop_area_full": float(full_area_to_crop.mean()) if full_area_to_crop.size else 0.0,
            "mean_proposal_area_to_gt_mask_area_full": float(full_area_to_mask.mean()) if full_area_to_mask.size else 0.0,
            "mean_proposal_rect_vs_gtcrop_iou_level0": float(gtcrop_ious.mean()) if gtcrop_ious.size else 0.0,
            "mean_neighbor_overlap_ratio_proposal": float(neighbor_overlap_prop.mean()) if neighbor_overlap_prop.size else 0.0,
            "mean_neighbor_overlap_ratio_targetmask": float(neighbor_overlap_target.mean()) if neighbor_overlap_target.size else 0.0,
            "neighbor_overlap_lt_001_rate": float((neighbor_overlap_prop <= 0.01).mean()) if neighbor_overlap_prop.size else 0.0,
            "mean_non_neighbor_score": float(non_neighbor_scores.mean()) if non_neighbor_scores.size else 0.0,
            "mean_compactness_score": float(compactness_scores.mean()) if compactness_scores.size else 0.0,
            "mean_weighted_priority_score": float(weighted_scores.mean()) if weighted_scores.size else 0.0,
        }

    (output_dir / "proposal_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    ranked = sorted(
        summary.items(),
        key=lambda kv: (
            kv[1]["mean_weighted_priority_score"],
            kv[1]["mean_mask_coverage_recall"],
            kv[1]["full_coverage_rate_99"],
            -kv[1]["mean_neighbor_overlap_ratio_proposal"],
            kv[1]["mean_proposal_rect_vs_gtcrop_iou_level0"],
            -kv[1]["mean_proposal_area_to_gt_crop_area_full"],
        ),
        reverse=True,
    )
    md_lines = [
        "# BBox Proposal Benchmark",
        "",
        f"Stain: {stain_key}",
        f"GT sections evaluated: {len(gt_sections)}",
        "",
        "Weighted ranking priority:",
        "- 50% target-mask coverage recall",
        "- 30% avoiding overlap with non-target GT masks on the same slide",
        "- 20% compactness / avoiding over-expansion",
        "",
        "Tie-breakers:",
        "- full coverage rate @ 0.99",
        "- lower mean neighbor overlap ratio on the same slide",
        "- larger proposal-vs-GT-crop IoU in slide space",
        "",
    ]
    for rank, (method_name, stats) in enumerate(ranked, start=1):
        md_lines.extend(
            [
                f"## {rank}. {method_name}",
                "",
                f"- proposal_found_rate: {stats['proposal_found_rate']:.4f}",
                f"- mean_weighted_priority_score: {stats['mean_weighted_priority_score']:.4f}",
                f"- mean_mask_coverage_recall: {stats['mean_mask_coverage_recall']:.4f}",
                f"- median_mask_coverage_recall: {stats['median_mask_coverage_recall']:.4f}",
                f"- full_coverage_rate_99: {stats['full_coverage_rate_99']:.4f}",
                f"- coverage_rate_95: {stats['coverage_rate_95']:.4f}",
                f"- mean_crop_efficiency: {stats['mean_crop_efficiency']:.4f}",
                f"- mean_crop_area_to_gt_mask_ratio: {stats['mean_crop_area_to_gt_mask_ratio']:.4f}",
                f"- mean_proposal_rect_vs_gtbbox_iou: {stats['mean_proposal_rect_vs_gtbbox_iou']:.4f}",
                f"- mean_proposal_area_to_gt_crop_area_full: {stats['mean_proposal_area_to_gt_crop_area_full']:.4f}",
                f"- mean_proposal_area_to_gt_mask_area_full: {stats['mean_proposal_area_to_gt_mask_area_full']:.4f}",
                f"- mean_proposal_rect_vs_gtcrop_iou_level0: {stats['mean_proposal_rect_vs_gtcrop_iou_level0']:.4f}",
                f"- mean_neighbor_overlap_ratio_proposal: {stats['mean_neighbor_overlap_ratio_proposal']:.4f}",
                f"- mean_neighbor_overlap_ratio_targetmask: {stats['mean_neighbor_overlap_ratio_targetmask']:.4f}",
                f"- mean_non_neighbor_score: {stats['mean_non_neighbor_score']:.4f}",
                f"- mean_compactness_score: {stats['mean_compactness_score']:.4f}",
                f"- neighbor_overlap_lt_001_rate: {stats['neighbor_overlap_lt_001_rate']:.4f}",
                "",
            ]
        )
    (output_dir / "proposal_summary.md").write_text("\n".join(md_lines), encoding="utf-8")
    for entry in slide_cache.values():
        handle = entry[1]
        if handle is not None and hasattr(handle, "close"):
            try:
                handle.close()
            except Exception:
                pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

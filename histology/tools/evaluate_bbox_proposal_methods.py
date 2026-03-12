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

Image.MAX_IMAGE_PIXELS = None

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from histology.gui_mvp.hitl_gui.pipeline_adapters.slide_io import load_slide_bundle
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
        return {
            "primary": primary,
            "legacy": legacy_score,
            "hybrid": hybrid,
            "nonwhite": nonwhite,
        }
    return {
        "primary": primary,
        "legacy": legacy_score,
        "hybrid": primary,
        "nonwhite": info["nonwhite"].astype(np.uint8),
    }


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


def build_methods_for_scores(score_maps: dict[str, np.ndarray]) -> dict[str, Callable[[CandidateBox, tuple[int, int]], tuple[int, int, int, int]]]:
    primary = score_maps["primary"]
    legacy = score_maps["legacy"]
    hybrid = score_maps["hybrid"]

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
    slide_cache: dict[Path, tuple[np.ndarray, tuple[int, int], dict[str, CandidateBox], np.ndarray]] = {}

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
            slide_cache[slide_path] = (overview_rgb, loaded.overview_size, overview_downsample, candidate_map, score_maps)

        overview_rgb, overview_size, overview_downsample, candidate_map, score_maps = slide_cache[slide_path]
        sibling_gt_sections = [sec for sec in gt_sections_by_slide.get(slide_path, []) if sec.label != gt.label]
        methods = build_methods_for_scores(score_maps)
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

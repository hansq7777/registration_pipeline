#!/usr/bin/env python3
import argparse
import csv
import json
import math
import random
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
import openslide
from PIL import Image, ImageDraw
from scipy.ndimage import binary_fill_holes
from skimage.filters import gaussian, threshold_otsu
from skimage.segmentation import inverse_gaussian_gradient, morphological_geodesic_active_contour


@dataclass
class SectionLabel:
    stain: str
    sample_id: str
    section_id: int

    @property
    def short_label(self) -> str:
        return f"{self.sample_id}_{self.section_id}"

    @property
    def full_label(self) -> str:
        return f"{self.stain}_{self.sample_id}_{self.section_id}"


@dataclass
class CandidateBox:
    candidate_rank: int
    x: int
    y: int
    w: int
    h: int
    area: int
    cx: float
    cy: float
    touches_border: bool
    section: Optional[SectionLabel] = None
    overview_mask: Optional[np.ndarray] = None

    def as_bbox(self) -> Tuple[int, int, int, int]:
        return self.x, self.y, self.w, self.h


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def odd_kernel(value: int, minimum: int = 3) -> int:
    value = max(value, minimum)
    if value % 2 == 0:
        value += 1
    return value


def parse_sections(blob: str) -> List[int]:
    parts = [int(x) for x in blob.split("-") if x]
    if len(parts) == 2 and parts[1] > parts[0] and (parts[1] - parts[0]) % 6 == 0:
        return list(range(parts[0], parts[1] + 1, 6))
    return parts


def parse_slide_stem(stem: str) -> Tuple[str, List[SectionLabel]]:
    match = re.fullmatch(r"([A-Za-z]+)_(.+)", stem)
    if not match:
        raise ValueError(f"Unrecognized slide stem: {stem}")
    stain = match.group(1)
    rest = match.group(2)
    labels: List[SectionLabel] = []
    for group in rest.split(";"):
        group_match = re.fullmatch(r"(\d+)_(\d+(?:-\d+)*)", group)
        if not group_match:
            raise ValueError(f"Unrecognized group in slide stem: {group}")
        sample_id = group_match.group(1)
        for section_id in parse_sections(group_match.group(2)):
            labels.append(SectionLabel(stain=stain, sample_id=sample_id, section_id=section_id))
    return stain, labels


def collect_slide_inventory(input_dir: Path) -> List[dict]:
    rows = []
    for path in sorted(input_dir.glob("*.ndpi")):
        if path.name.startswith("._"):
            continue
        stain, labels = parse_slide_stem(path.stem)
        rows.append(
            {
                "filename": path.name,
                "slide_path": str(path),
                "stain": stain,
                "expected_count": len(labels),
                "section_labels": ";".join(label.short_label for label in labels),
            }
        )
    return rows


def overview_level(slide: openslide.OpenSlide) -> int:
    return slide.level_count - 1


def read_overview(slide: openslide.OpenSlide) -> np.ndarray:
    level = overview_level(slide)
    w, h = slide.level_dimensions[level]
    return np.asarray(slide.read_region((0, 0), level, (w, h)).convert("RGB"))


def overlay_mask(base_rgb: np.ndarray, mask: np.ndarray, alpha: float = 0.35) -> np.ndarray:
    out = base_rgb.astype(np.float32).copy()
    keep = mask > 0
    red = np.array([255.0, 0.0, 0.0], dtype=np.float32)
    out[keep] = (1.0 - alpha) * out[keep] + alpha * red
    return np.clip(out, 0, 255).astype(np.uint8)


def write_gray_png(path: Path, arr: np.ndarray) -> None:
    Image.fromarray(arr.astype(np.uint8), mode="L").save(path)


def write_rgb_png(path: Path, arr: np.ndarray) -> None:
    Image.fromarray(arr.astype(np.uint8), mode="RGB").save(path)


def compute_stain_score(rgb: np.ndarray, stain: str) -> Tuple[np.ndarray, dict]:
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    sat = hsv[:, :, 1].astype(np.uint8)
    val = hsv[:, :, 2].astype(np.uint8)
    nonwhite = (255 - rgb.min(axis=2)).astype(np.uint8)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    inv_gray = (255 - gray).astype(np.uint8)

    stain_key = stain.lower()
    if stain_key == "gallyas":
        # Myelin slides are effectively grayscale with dark tissue and weak
        # saturation. Emphasize darkness and broad tissue silhouette, not hue.
        score = np.maximum(inv_gray, nonwhite)
        score = cv2.GaussianBlur(score, (0, 0), sigmaX=1.2, sigmaY=1.2)
    else:
        score = np.maximum(sat, nonwhite)

    return score.astype(np.uint8), {
        "sat": sat,
        "val": val,
        "nonwhite": nonwhite,
        "gray": gray,
        "inv_gray": inv_gray,
    }


def component_mask_from_overview(overview_rgb: np.ndarray, stain: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    hsv = cv2.cvtColor(overview_rgb, cv2.COLOR_RGB2HSV)
    sat = hsv[:, :, 1].astype(np.uint8)
    nonwhite = (255 - overview_rgb.min(axis=2)).astype(np.uint8)
    gray = cv2.cvtColor(overview_rgb, cv2.COLOR_RGB2GRAY)
    inv_gray = (255 - gray).astype(np.uint8)

    if stain.lower() == "gallyas":
        score = np.maximum(inv_gray, nonwhite)
        score_thresh = max(int(threshold_otsu(score)), 18)
        raw = (score > score_thresh).astype(np.uint8) * 255
    else:
        sat_thresh = max(int(threshold_otsu(sat)), 12)
        nonwhite_thresh = max(int(threshold_otsu(nonwhite)), 20)
        raw = ((sat > sat_thresh) | (nonwhite > nonwhite_thresh)).astype(np.uint8) * 255

    opened = cv2.morphologyEx(raw, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    cleaned = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    return sat, nonwhite, cleaned


def bands_from_projection(proj: np.ndarray, thresh: float) -> List[Tuple[int, int]]:
    keep = proj > thresh
    bands: List[Tuple[int, int]] = []
    start: Optional[int] = None
    for idx, value in enumerate(keep):
        if value and start is None:
            start = idx
        elif (not value) and start is not None:
            bands.append((start, idx))
            start = None
    if start is not None:
        bands.append((start, len(keep)))
    return bands


def kmeans_1d(values: np.ndarray, k: int, iterations: int = 50) -> Tuple[np.ndarray, np.ndarray]:
    values = values.astype(np.float32)
    if values.size == 0:
        return np.array([], dtype=np.float32), np.array([], dtype=np.int32)
    if k <= 1:
        return np.array([float(values.mean())], dtype=np.float32), np.zeros(values.shape[0], dtype=np.int32)
    quantiles = np.linspace(0, 100, k + 2)[1:-1]
    centers = np.percentile(values, quantiles).astype(np.float32)
    for _ in range(iterations):
        labels = np.abs(values[:, None] - centers[None, :]).argmin(axis=1)
        new_centers = centers.copy()
        for idx in range(k):
            pts = values[labels == idx]
            if pts.size:
                new_centers[idx] = pts.mean()
        if np.allclose(new_centers, centers):
            break
        centers = new_centers
    return centers, labels.astype(np.int32)


def fallback_row_split_candidates(mask: np.ndarray, expected_count: int) -> List[CandidateBox]:
    if expected_count < 2:
        return []

    h, w = mask.shape[:2]
    proj_y = (mask > 0).sum(axis=1).astype(np.float32)
    bands: List[Tuple[int, int]] = []
    for frac in (0.18, 0.15, 0.12, 0.10):
        bands = [band for band in bands_from_projection(proj_y, float(proj_y.max()) * frac) if (band[1] - band[0]) > int(round(h * 0.05))]
        if len(bands) >= 2:
            break
    if not bands:
        return []

    if len(bands) >= 2 and expected_count > 2:
        band_rows = sorted(bands, key=lambda band: proj_y[band[0] : band[1]].sum(), reverse=True)[:2]
        band_rows = sorted(band_rows, key=lambda band: band[0])
        row_counts = [math.ceil(expected_count / 2), expected_count - math.ceil(expected_count / 2)]
    else:
        band_rows = [max(bands, key=lambda band: proj_y[band[0] : band[1]].sum())]
        row_counts = [expected_count]

    candidates: List[CandidateBox] = []
    for (y1, y2), per_row in zip(band_rows, row_counts):
        ys, xs = np.where(mask[y1:y2, :] > 0)
        if xs.size == 0:
            continue
        centers, labels = kmeans_1d(xs, per_row)
        for cluster_idx in np.argsort(centers):
            pts_x = xs[labels == cluster_idx]
            pts_y = ys[labels == cluster_idx]
            if pts_x.size == 0:
                continue
            cluster_mask = np.zeros_like(mask, dtype=np.uint8)
            cluster_mask[y1 + pts_y, pts_x] = 1
            cluster_mask = cv2.morphologyEx(cluster_mask * 255, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0
            cluster_mask = cv2.morphologyEx(cluster_mask.astype(np.uint8) * 255, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)) > 0
            cy_pts, cx_pts = np.where(cluster_mask)
            if cx_pts.size == 0:
                continue
            x_lo, x_hi = np.percentile(cx_pts, [0.5, 99.5])
            y_lo, y_hi = np.percentile(cy_pts, [0.5, 99.5])
            pad_x = max(4, int(round((x_hi - x_lo) * 0.03)))
            pad_y = max(4, int(round((y_hi - y_lo) * 0.03)))
            x_min = max(0, int(math.floor(x_lo)) - pad_x)
            x_max = min(w, int(math.ceil(x_hi)) + 1 + pad_x)
            y_min = max(0, int(math.floor(y_lo)) - pad_y)
            y_max = min(h, int(math.ceil(y_hi)) + 1 + pad_y)
            area = int(cx_pts.size)
            candidates.append(
                CandidateBox(
                    candidate_rank=len(candidates) + 1,
                    x=x_min,
                    y=y_min,
                    w=x_max - x_min,
                    h=y_max - y_min,
                    area=area,
                    cx=(x_min + x_max) / 2.0,
                    cy=(y_min + y_max) / 2.0,
                    touches_border=(x_min == 0 or y_min == 0 or x_max >= w or y_max >= h),
                    overview_mask=cluster_mask,
                )
            )
    return candidates


def find_candidate_components(mask: np.ndarray, expected_count: int) -> List[CandidateBox]:
    h, w = mask.shape[:2]
    border_margin = max(5, int(round(min(h, w) * 0.01)))
    min_area = max(5000, int(round(h * w * 0.004)))
    num, labels, stats, cents = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), 8)
    candidates: List[CandidateBox] = []
    rank = 1
    for idx in range(1, num):
        x, y, bw, bh, area = map(int, stats[idx])
        cx, cy = cents[idx]
        touches = x <= border_margin or y <= border_margin or (x + bw) >= (w - border_margin) or (y + bh) >= (h - border_margin)
        if area < min_area:
            continue
        if bw < int(round(w * 0.05)) or bh < int(round(h * 0.08)):
            continue
        candidates.append(
            CandidateBox(
                candidate_rank=rank,
                x=x,
                y=y,
                w=bw,
                h=bh,
                area=area,
                cx=float(cx),
                cy=float(cy),
                touches_border=touches,
                overview_mask=(labels == idx),
            )
        )
        rank += 1
    merged = merge_candidates(candidates, gap_px=max(40, int(round(w * 0.03))))
    merged = sorted(merged, key=lambda c: c.area, reverse=True)
    if len(merged) < expected_count:
        fallback = fallback_row_split_candidates(mask, expected_count)
        if len(fallback) >= expected_count:
            return fallback[:expected_count]
    if len(merged) > expected_count:
        merged = merged[:expected_count]
    return merged


def merge_candidates(candidates: Sequence[CandidateBox], gap_px: int) -> List[CandidateBox]:
    remaining = sorted(candidates, key=lambda c: (c.y, c.x))
    merged: List[CandidateBox] = []
    used = [False] * len(remaining)

    for i, current in enumerate(remaining):
        if used[i]:
            continue
        x1, y1, x2, y2 = current.x, current.y, current.x + current.w, current.y + current.h
        area = current.area
        overview_mask = current.overview_mask.copy() if current.overview_mask is not None else None
        changed = True
        while changed:
            changed = False
            for j, other in enumerate(remaining):
                if j == i or used[j]:
                    continue
                ox1, oy1, ox2, oy2 = other.x, other.y, other.x + other.w, other.y + other.h
                y_overlap = max(0, min(y2, oy2) - max(y1, oy1))
                min_height = min(y2 - y1, oy2 - oy1)
                x_gap = max(0, max(x1, ox1) - min(x2, ox2))
                if min_height > 0 and (y_overlap / min_height) > 0.6 and x_gap <= gap_px:
                    x1 = min(x1, ox1)
                    y1 = min(y1, oy1)
                    x2 = max(x2, ox2)
                    y2 = max(y2, oy2)
                    area += other.area
                    if other.overview_mask is not None:
                        overview_mask = other.overview_mask.copy() if overview_mask is None else (overview_mask | other.overview_mask)
                    used[j] = True
                    changed = True
        used[i] = True
        merged.append(
            CandidateBox(
                candidate_rank=len(merged) + 1,
                x=x1,
                y=y1,
                w=x2 - x1,
                h=y2 - y1,
                area=area,
                cx=(x1 + x2) / 2.0,
                cy=(y1 + y2) / 2.0,
                touches_border=False,
                overview_mask=overview_mask,
            )
        )
    return merged


def assign_sections(candidates: List[CandidateBox], labels: Sequence[SectionLabel]) -> List[CandidateBox]:
    ordered = sorted(candidates, key=lambda c: (c.cy, c.cx))
    if len(ordered) > 1:
        top_count = math.ceil(len(ordered) / 2)
        top_row = sorted(ordered[:top_count], key=lambda c: c.cx)
        bottom_row = sorted(ordered[top_count:], key=lambda c: c.cx)
        ordered = top_row + bottom_row
    for candidate, label in zip(ordered, labels):
        candidate.section = label
    return ordered


def draw_overview_boxes(overview_rgb: np.ndarray, candidates: Sequence[CandidateBox], labels: Sequence[SectionLabel]) -> np.ndarray:
    canvas = Image.fromarray(overview_rgb.copy())
    draw = ImageDraw.Draw(canvas)
    for idx, candidate in enumerate(candidates):
        label = labels[idx].short_label if idx < len(labels) else f"cand_{idx+1}"
        x1, y1 = candidate.x, candidate.y
        x2, y2 = candidate.x + candidate.w, candidate.y + candidate.h
        draw.rectangle((x1, y1, x2, y2), outline=(255, 0, 0), width=4)
        draw.rectangle((x1, max(0, y1 - 28), x1 + 180, y1), fill=(0, 0, 0))
        draw.text((x1 + 6, max(0, y1 - 24)), label, fill=(255, 255, 255))
    return np.asarray(canvas)


def convert_bbox_to_level0(slide: openslide.OpenSlide, candidate: CandidateBox, pad_overview: int) -> Tuple[int, int, int, int]:
    level = overview_level(slide)
    downsample = float(slide.level_downsamples[level])
    x1 = max(0, candidate.x - pad_overview)
    y1 = max(0, candidate.y - pad_overview)
    x2 = min(slide.level_dimensions[level][0], candidate.x + candidate.w + pad_overview)
    y2 = min(slide.level_dimensions[level][1], candidate.y + candidate.h + pad_overview)
    x0 = int(round(x1 * downsample))
    y0 = int(round(y1 * downsample))
    w0 = int(round((x2 - x1) * downsample))
    h0 = int(round((y2 - y1) * downsample))
    w0 = min(w0, slide.dimensions[0] - x0)
    h0 = min(h0, slide.dimensions[1] - y0)
    return x0, y0, w0, h0


def overview_bbox_to_level0(candidate: CandidateBox, overview_downsample: float) -> Tuple[int, int, int, int]:
    x0 = int(round(candidate.x * overview_downsample))
    y0 = int(round(candidate.y * overview_downsample))
    w0 = int(round(candidate.w * overview_downsample))
    h0 = int(round(candidate.h * overview_downsample))
    return x0, y0, w0, h0


def candidate_center_level0(candidate: CandidateBox, overview_downsample: float) -> Tuple[float, float]:
    return (
        float((candidate.x + 0.5 * candidate.w) * overview_downsample),
        float((candidate.y + 0.5 * candidate.h) * overview_downsample),
    )


def level0_point_to_crop(point_xy: Tuple[float, float], crop_bbox_level0: Tuple[int, int, int, int], crop_downsample: float) -> Tuple[float, float]:
    px = (point_xy[0] - crop_bbox_level0[0]) / crop_downsample
    py = (point_xy[1] - crop_bbox_level0[1]) / crop_downsample
    return float(px), float(py)


def rect_mask_in_crop(
    rect_level0: Tuple[int, int, int, int],
    crop_bbox_level0: Tuple[int, int, int, int],
    crop_shape: Tuple[int, int],
    crop_downsample: float,
    expand_frac_x: float = 0.12,
    expand_frac_y: float = 0.12,
    min_expand_px: int = 24,
) -> np.ndarray:
    h, w = crop_shape[:2]
    rx, ry, rw, rh = rect_level0
    px1 = int(math.floor((rx - crop_bbox_level0[0]) / crop_downsample))
    py1 = int(math.floor((ry - crop_bbox_level0[1]) / crop_downsample))
    px2 = int(math.ceil((rx + rw - crop_bbox_level0[0]) / crop_downsample))
    py2 = int(math.ceil((ry + rh - crop_bbox_level0[1]) / crop_downsample))

    mx = max(min_expand_px, int(round((px2 - px1) * expand_frac_x)))
    my = max(min_expand_px, int(round((py2 - py1) * expand_frac_y)))

    px1 = max(0, px1 - mx)
    py1 = max(0, py1 - my)
    px2 = min(w, px2 + mx)
    py2 = min(h, py2 + my)

    out = np.zeros((h, w), dtype=bool)
    if px1 < px2 and py1 < py2:
        out[py1:py2, px1:px2] = True
    return out


def overview_mask_to_crop_support(
    overview_mask: np.ndarray,
    crop_bbox_level0: Tuple[int, int, int, int],
    crop_shape: Tuple[int, int],
    crop_downsample: float,
    overview_downsample: float,
) -> np.ndarray:
    h, w = crop_shape[:2]
    oh, ow = overview_mask.shape[:2]
    ox1 = max(0, int(math.floor(crop_bbox_level0[0] / overview_downsample)))
    oy1 = max(0, int(math.floor(crop_bbox_level0[1] / overview_downsample)))
    ox2 = min(ow, int(math.ceil((crop_bbox_level0[0] + crop_bbox_level0[2]) / overview_downsample)))
    oy2 = min(oh, int(math.ceil((crop_bbox_level0[1] + crop_bbox_level0[3]) / overview_downsample)))
    if ox1 >= ox2 or oy1 >= oy2:
        return np.zeros((h, w), dtype=bool)

    patch = overview_mask[oy1:oy2, ox1:ox2].astype(np.uint8) * 255
    resized = cv2.resize(patch, (w, h), interpolation=cv2.INTER_NEAREST) > 0
    open_k = odd_kernel(int(round(min(h, w) * 0.004)), minimum=5)
    close_k = odd_kernel(int(round(min(h, w) * 0.008)), minimum=9)
    resized = cv2.morphologyEx(resized.astype(np.uint8) * 255, cv2.MORPH_OPEN, np.ones((open_k, open_k), np.uint8)) > 0
    resized = cv2.morphologyEx(resized.astype(np.uint8) * 255, cv2.MORPH_CLOSE, np.ones((close_k, close_k), np.uint8)) > 0
    return resized


def build_crop_ownership_masks(
    target_candidate: CandidateBox,
    all_candidates: Sequence[CandidateBox],
    crop_bbox_level0: Tuple[int, int, int, int],
    crop_shape: Tuple[int, int],
    crop_downsample: float,
    overview_downsample: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    h, w = crop_shape[:2]
    target_cx, target_cy = candidate_center_level0(target_candidate, overview_downsample)
    x_coords = crop_bbox_level0[0] + (np.arange(w, dtype=np.float32) + 0.5) * np.float32(crop_downsample)
    y_coords = crop_bbox_level0[1] + (np.arange(h, dtype=np.float32) + 0.5) * np.float32(crop_downsample)

    strict = np.ones((h, w), dtype=bool)
    for other in all_candidates:
        if other is target_candidate:
            continue
        other_cx, other_cy = candidate_center_level0(other, overview_downsample)
        rhs = (other_cx * other_cx + other_cy * other_cy) - (target_cx * target_cx + target_cy * target_cy)
        plane = (
            2.0 * (other_cx - target_cx) * x_coords[None, :]
            + 2.0 * (other_cy - target_cy) * y_coords[:, None]
        ) <= rhs
        strict &= plane

    support_rect = overview_bbox_to_level0(target_candidate, overview_downsample)
    support = rect_mask_in_crop(
        support_rect,
        crop_bbox_level0=crop_bbox_level0,
        crop_shape=crop_shape,
        crop_downsample=crop_downsample,
    )
    strict &= support
    if not strict.any():
        strict = support.copy()
    if not strict.any():
        strict = np.ones((h, w), dtype=bool)

    soft_k = odd_kernel(int(round(min(h, w) * 0.01)), minimum=15)
    soft = cv2.dilate(strict.astype(np.uint8) * 255, np.ones((soft_k, soft_k), np.uint8)) > 0
    return strict, soft, support


def extract_crop(slide: openslide.OpenSlide, bbox_level0: Tuple[int, int, int, int], level: int) -> np.ndarray:
    x0, y0, w0, h0 = bbox_level0
    downsample = float(slide.level_downsamples[level])
    size = (max(1, int(round(w0 / downsample))), max(1, int(round(h0 / downsample))))
    return np.asarray(slide.read_region((x0, y0), level, size).convert("RGB"))


def largest_component(mask: np.ndarray) -> np.ndarray:
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if num <= 1:
        return mask.astype(np.uint8)
    largest_idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (labels == largest_idx).astype(np.uint8)


def select_component_by_point(mask: np.ndarray, point_xy: Tuple[float, float]) -> np.ndarray:
    mask_u8 = mask.astype(np.uint8)
    num, labels, stats, cents = cv2.connectedComponentsWithStats(mask_u8, 8)
    if num <= 1:
        return mask_u8

    px = int(round(point_xy[0]))
    py = int(round(point_xy[1]))
    if 0 <= px < mask.shape[1] and 0 <= py < mask.shape[0]:
        hit = int(labels[py, px])
        if hit > 0:
            return (labels == hit).astype(np.uint8)

    best_idx = 1
    best_dist = float("inf")
    for idx in range(1, num):
        cx, cy = cents[idx]
        dist = (cx - point_xy[0]) ** 2 + (cy - point_xy[1]) ** 2
        if dist < best_dist:
            best_dist = dist
            best_idx = idx
    return (labels == best_idx).astype(np.uint8)


def reconstruct_from_core(candidate: np.ndarray, core: np.ndarray) -> np.ndarray:
    num, labels, _, _ = cv2.connectedComponentsWithStats(candidate.astype(np.uint8), 8)
    if num <= 1:
        return candidate.astype(np.uint8)
    keep = np.unique(labels[core > 0])
    keep = keep[keep > 0]
    if keep.size == 0:
        return np.zeros_like(candidate, dtype=np.uint8)
    return np.isin(labels, keep).astype(np.uint8)


def border_branch_prune(
    mask: np.ndarray,
    support_mask: np.ndarray,
    target_center_px: Tuple[float, float],
    blur_score: np.ndarray,
) -> np.ndarray:
    if not mask.any():
        return mask.astype(np.uint8)

    h, w = mask.shape[:2]
    support_tight_k = odd_kernel(int(round(min(h, w) * 0.01)), minimum=15)
    support_tight = cv2.erode(support_mask.astype(np.uint8) * 255, np.ones((support_tight_k, support_tight_k), np.uint8)) > 0
    if not support_tight.any():
        support_tight = support_mask > 0

    dist = cv2.distanceTransform(mask.astype(np.uint8) * 255, cv2.DIST_L2, 5)
    thin_thresh = max(4.0, float(min(h, w) * 0.006))
    border_band = mask & (~support_tight | (dist <= thin_thresh))

    num, labels, stats, _ = cv2.connectedComponentsWithStats(border_band.astype(np.uint8), 8)
    remove = np.zeros_like(mask, dtype=bool)
    mask_area = max(int(mask.sum()), 1)
    for idx in range(1, num):
        comp = labels == idx
        x, y, bw, bh, area = map(int, stats[idx])
        touches_border = x == 0 or y == 0 or (x + bw) >= w or (y + bh) >= h
        if not touches_border:
            continue
        if area > int(round(mask_area * 0.12)):
            continue
        if dist[comp].mean() > thin_thresh * 1.35:
            continue
        remove |= comp

    pruned = mask & (~remove)
    if pruned.any():
        score_inside = blur_score[pruned]
        interior_score = float(np.median(score_inside)) if score_inside.size else 0.0
        score_thresh = max(18.0, interior_score * 0.90)
        scan_top = max(12, int(round(h * 0.16)))
        scan_side = max(12, int(round(w * 0.10)))

        cut_top = 0
        for idx in range(scan_top):
            row = pruned[idx, :]
            occ = float(row.mean())
            row_score = float(blur_score[idx, :][row].mean()) if row.any() else 0.0
            if (occ == 0.0) or (occ < 0.32 and row_score < score_thresh):
                cut_top = idx + 1
                continue
            break

        cut_bottom = 0
        for idx in range(scan_top):
            row = pruned[h - 1 - idx, :]
            occ = float(row.mean())
            row_score = float(blur_score[h - 1 - idx, :][row].mean()) if row.any() else 0.0
            if (occ == 0.0) or (occ < 0.50 and row_score < score_thresh):
                cut_bottom = idx + 1
                continue
            break

        cut_left = 0
        for idx in range(scan_side):
            col = pruned[:, idx]
            occ = float(col.mean())
            col_score = float(blur_score[:, idx][col].mean()) if col.any() else 0.0
            if (occ == 0.0) or (occ < 0.22 and col_score < score_thresh):
                cut_left = idx + 1
                continue
            break

        cut_right = 0
        for idx in range(scan_side):
            col = pruned[:, w - 1 - idx]
            occ = float(col.mean())
            col_score = float(blur_score[:, w - 1 - idx][col].mean()) if col.any() else 0.0
            if (occ == 0.0) or (occ < 0.22 and col_score < score_thresh):
                cut_right = idx + 1
                continue
            break

        if cut_top > 0:
            pruned[:cut_top, :] = 0
        if cut_bottom > 0:
            pruned[h - cut_bottom :, :] = 0
        if cut_left > 0:
            pruned[:, :cut_left] = 0
        if cut_right > 0:
            pruned[:, w - cut_right :] = 0

    pruned = select_component_by_point(pruned, target_center_px)
    pruned = binary_fill_holes(pruned > 0)
    return pruned.astype(np.uint8)


def detect_border_artifacts(score: np.ndarray, nonwhite: np.ndarray, sat: np.ndarray) -> np.ndarray:
    h, w = score.shape
    raw = (
        (score > int(threshold_otsu(score)))
        | (nonwhite > int(threshold_otsu(nonwhite)))
        | (sat > int(threshold_otsu(sat)))
    ).astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(raw, 8)
    artifact = np.zeros_like(raw, dtype=np.uint8)

    for idx in range(1, num):
        x, y, bw, bh, area = map(int, stats[idx])
        touch_top = y == 0
        touch_bottom = (y + bh) >= h
        touch_left = x == 0
        touch_right = (x + bw) >= w
        if not (touch_top or touch_bottom or touch_left or touch_right):
            continue

        horizontal_strip = (touch_top or touch_bottom) and bh <= int(round(h * 0.10)) and bw >= int(round(w * 0.35))
        vertical_strip = (touch_left or touch_right) and bw <= int(round(w * 0.04)) and bh >= int(round(h * 0.25))
        corner_speck = (
            (touch_left or touch_right)
            and (touch_top or touch_bottom)
            and bw <= int(round(w * 0.18))
            and bh <= int(round(h * 0.18))
            and area <= int(round(h * w * 0.015))
        )
        if horizontal_strip or vertical_strip or corner_speck:
            artifact[labels == idx] = 1

    return artifact


def finalize_mask_metrics(
    final: np.ndarray,
    ownership_strict: np.ndarray,
    score: np.ndarray,
    artifact: np.ndarray,
    candidate: np.ndarray,
    core: np.ndarray,
    blur: np.ndarray,
) -> dict:
    mask_area = max(int(final.sum()), 1)
    band = max(12, int(round(min(final.shape[:2]) * 0.01)))
    border_band = np.zeros_like(final, dtype=bool)
    border_band[:band, :] = True
    border_band[-band:, :] = True
    border_band[:, :band] = True
    border_band[:, -band:] = True
    border_touch_ratio = float((final & border_band).sum() / mask_area)
    neighbor_occupancy_ratio = float((final & (~ownership_strict)).sum() / mask_area)
    return {
        "score": score.astype(np.uint8),
        "artifact": (artifact.astype(np.uint8) * 255),
        "ownership": (ownership_strict.astype(np.uint8) * 255),
        "candidate": (candidate.astype(np.uint8) * 255),
        "core": (core.astype(np.uint8) * 255),
        "blur": blur.astype(np.uint8),
        "mask": (final.astype(np.uint8) * 255),
        "border_touch_ratio": border_touch_ratio,
        "neighbor_occupancy_ratio": neighbor_occupancy_ratio,
        "flag_mask_touches_crop_border_too_much": border_touch_ratio > 0.015,
        "flag_mask_occupies_neighbor_ownership_zone": neighbor_occupancy_ratio > 0.01,
    }


def build_crop_mask_baseline(
    crop_rgb: np.ndarray,
    ownership_strict: np.ndarray,
    ownership_soft: np.ndarray,
    support_mask: np.ndarray,
    target_center_px: Tuple[float, float],
    stain: str,
) -> dict:
    score, channels = compute_stain_score(crop_rgb, stain)
    artifact = detect_border_artifacts(score, channels["nonwhite"], channels["sat"])

    score_clean = score.copy()
    score_clean[artifact > 0] = 0
    score_clean = score_clean.astype(np.uint8)

    sigma = max(11, int(round(min(crop_rgb.shape[:2]) * 0.015)))
    blur = cv2.GaussianBlur(score_clean, (0, 0), sigmaX=sigma, sigmaY=sigma)
    blur_thresh = int(threshold_otsu(blur))
    candidate = (blur >= blur_thresh) & (artifact == 0)
    open_k = odd_kernel(int(round(min(crop_rgb.shape[:2]) * 0.003)), minimum=7)
    close_k = odd_kernel(int(round(min(crop_rgb.shape[:2]) * 0.012)), minimum=31)
    candidate = cv2.morphologyEx(candidate.astype(np.uint8) * 255, cv2.MORPH_OPEN, np.ones((open_k, open_k), np.uint8)) > 0
    candidate = cv2.morphologyEx(candidate.astype(np.uint8) * 255, cv2.MORPH_CLOSE, np.ones((close_k, close_k), np.uint8)) > 0
    candidate = binary_fill_holes(candidate)
    candidate = largest_component(candidate)

    local_sigma = max(3, int(round(min(crop_rgb.shape[:2]) * 0.004)))
    local = cv2.GaussianBlur(score_clean, (0, 0), sigmaX=local_sigma, sigmaY=local_sigma)
    interior = local[candidate > 0]
    grow_thresh = max(8, int(np.quantile(interior, 0.08) * 0.85)) if interior.size else 8
    grow_k = odd_kernel(int(round(min(crop_rgb.shape[:2]) * 0.004)), minimum=7)
    grown = cv2.dilate(candidate.astype(np.uint8) * 255, np.ones((grow_k, grow_k), np.uint8)) > 0
    final = grown & (local >= grow_thresh) & (artifact == 0)
    final |= candidate
    final = cv2.morphologyEx(final.astype(np.uint8) * 255, cv2.MORPH_CLOSE, np.ones((odd_kernel(int(round(min(crop_rgb.shape[:2]) * 0.006)), minimum=11),) * 2, np.uint8)) > 0
    final = binary_fill_holes(final)
    final = largest_component(final)
    core = candidate.copy()
    return finalize_mask_metrics(final > 0, ownership_strict, score, artifact, candidate > 0, core > 0, blur)


def build_crop_mask_soft_support_mgac(
    crop_rgb: np.ndarray,
    ownership_strict: np.ndarray,
    ownership_soft: np.ndarray,
    support_mask: np.ndarray,
    target_center_px: Tuple[float, float],
    stain: str,
) -> dict:
    score, channels = compute_stain_score(crop_rgb, stain)
    artifact = detect_border_artifacts(score, channels["nonwhite"], channels["sat"])

    score_clean = score.copy()
    score_clean[artifact > 0] = 0
    score_clean[~ownership_soft] = 0

    sigma = max(11, int(round(min(crop_rgb.shape[:2]) * 0.015)))
    blur = cv2.GaussianBlur(score_clean, (0, 0), sigmaX=sigma, sigmaY=sigma)
    blur_vals = blur[ownership_soft]
    blur_thresh = int(threshold_otsu(blur_vals.astype(np.uint8))) if blur_vals.size else int(threshold_otsu(blur))
    candidate_thresh = max(6, int(round(blur_thresh * 0.92)))
    candidate = (blur >= candidate_thresh) & ownership_soft & (artifact == 0)
    open_k = odd_kernel(int(round(min(crop_rgb.shape[:2]) * 0.003)), minimum=7)
    close_k = odd_kernel(int(round(min(crop_rgb.shape[:2]) * 0.012)), minimum=31)
    candidate = cv2.morphologyEx(candidate.astype(np.uint8) * 255, cv2.MORPH_OPEN, np.ones((open_k, open_k), np.uint8)) > 0
    candidate = cv2.morphologyEx(candidate.astype(np.uint8) * 255, cv2.MORPH_CLOSE, np.ones((close_k, close_k), np.uint8)) > 0
    candidate &= ownership_soft
    candidate = binary_fill_holes(candidate)

    candidate_dt = cv2.distanceTransform(candidate.astype(np.uint8) * 255, cv2.DIST_L2, 5)
    core_dt_thresh = max(10.0, float(min(crop_rgb.shape[:2]) * 0.008))
    core = candidate & (candidate_dt >= core_dt_thresh)
    if not core.any():
        core_k = odd_kernel(int(round(min(crop_rgb.shape[:2]) * 0.01)), minimum=15)
        core = cv2.erode(candidate.astype(np.uint8) * 255, np.ones((core_k, core_k), np.uint8)) > 0
    if not core.any():
        core = candidate.copy()
    core = select_component_by_point(core, target_center_px) > 0

    h, w = score.shape
    yy, xx = np.mgrid[0:h, 0:w]
    tx, ty = target_center_px
    center_prior = np.exp(
        -0.5
        * (
            ((xx - tx) / max(1.0, 0.30 * w)) ** 2
            + ((yy - ty) / max(1.0, 0.28 * h)) ** 2
        )
    ).astype(np.float32)
    center_prior /= max(float(center_prior.max()), 1e-6)

    support_float = np.clip(
        0.50 * ownership_soft.astype(np.float32)
        + 0.25 * ownership_strict.astype(np.float32)
        + 0.20 * support_mask.astype(np.float32)
        + 0.35 * center_prior,
        0.0,
        1.0,
    )
    support_float *= ownership_soft.astype(np.float32)

    init = candidate & (support_float >= 0.30)
    if not init.any():
        init = candidate.copy()
    init |= core
    init = select_component_by_point(init, target_center_px) > 0

    work_max_dim = 1400
    scale = min(1.0, work_max_dim / max(h, w))
    work_w = max(1, int(round(w * scale)))
    work_h = max(1, int(round(h * scale)))

    score_small = cv2.resize(score_clean.astype(np.float32) / 255.0, (work_w, work_h), interpolation=cv2.INTER_AREA)
    init_small = cv2.resize(init.astype(np.uint8) * 255, (work_w, work_h), interpolation=cv2.INTER_NEAREST) > 0
    core_small = cv2.resize(core.astype(np.uint8) * 255, (work_w, work_h), interpolation=cv2.INTER_NEAREST) > 0
    support_small = cv2.resize(support_float, (work_w, work_h), interpolation=cv2.INTER_AREA)
    gate_small = support_small >= 0.18

    edge_image = gaussian(score_small, sigma=1.2)
    gimg = inverse_gaussian_gradient(edge_image, alpha=300.0, sigma=1.0)
    gimg *= 0.05 + 0.95 * support_small

    try:
        mgac = morphological_geodesic_active_contour(
            gimg,
            num_iter=80,
            init_level_set=init_small.astype(np.int8),
            smoothing=1,
            balloon=-1,
            threshold="auto",
        )
        final_small = binary_fill_holes(mgac > 0)
    except Exception:
        final_small = init_small.copy()

    final_small &= gate_small
    if core_small.any():
        final_small = reconstruct_from_core(final_small, core_small) > 0
    if not final_small.any():
        final_small = init_small.copy()

    final = cv2.resize(final_small.astype(np.uint8) * 255, (w, h), interpolation=cv2.INTER_NEAREST) > 0
    final &= ownership_soft
    final = binary_fill_holes(final)
    final = select_component_by_point(final, target_center_px) > 0
    return finalize_mask_metrics(final > 0, ownership_strict, score, artifact, candidate > 0, core > 0, blur)


def build_crop_mask(
    crop_rgb: np.ndarray,
    ownership_strict: np.ndarray,
    ownership_soft: np.ndarray,
    support_mask: np.ndarray,
    target_center_px: Tuple[float, float],
    mask_method: str,
    stain: str,
) -> dict:
    if mask_method == "baseline_v1":
        return build_crop_mask_baseline(crop_rgb, ownership_strict, ownership_soft, support_mask, target_center_px, stain)
    if mask_method == "soft_support_mgac":
        return build_crop_mask_soft_support_mgac(crop_rgb, ownership_strict, ownership_soft, support_mask, target_center_px, stain)
    raise ValueError(f"Unsupported mask method: {mask_method}")


def rgba_from_mask(crop_rgb: np.ndarray, mask: np.ndarray) -> Image.Image:
    alpha = mask.astype(np.uint8)
    rgba = np.dstack([crop_rgb, alpha])
    return Image.fromarray(rgba, mode="RGBA")


def write_contact_sheet(path: Path, panels: Sequence[Tuple[str, np.ndarray]]) -> None:
    rendered = []
    max_w = 0
    max_h = 0
    for label, image in panels:
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=2)
        header = np.full((56, image.shape[1], 3), 20, dtype=np.uint8)
        cv2.putText(header, label, (18, 38), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
        tile = np.vstack([header, image])
        rendered.append(tile)
        max_w = max(max_w, tile.shape[1])
        max_h = max(max_h, tile.shape[0])

    rows = []
    pad = 16
    cols = 3
    for start in range(0, len(rendered), cols):
        subset = rendered[start : start + cols]
        row = np.full((max_h, cols * max_w + (cols + 1) * pad, 3), 245, dtype=np.uint8)
        for idx, tile in enumerate(subset):
            x = pad + idx * (max_w + pad)
            row[: tile.shape[0], x : x + tile.shape[1]] = tile
        rows.append(row)
    canvas = np.full((len(rows) * max_h + (len(rows) + 1) * pad, rows[0].shape[1], 3), 245, dtype=np.uint8)
    for idx, row in enumerate(rows):
        y = pad + idx * max_h
        canvas[y : y + row.shape[0], : row.shape[1]] = row
    write_rgb_png(path, canvas)


def write_inventory_csv(path: Path, rows: Sequence[dict]) -> None:
    cols = ["filename", "slide_path", "stain", "expected_count", "section_labels"]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=cols)
        writer.writeheader()
        writer.writerows(rows)


def write_candidate_csv(path: Path, rows: Sequence[dict]) -> None:
    cols = [
        "candidate_rank",
        "assigned_label",
        "assigned_sample_id",
        "assigned_section_id",
        "overview_x",
        "overview_y",
        "overview_w",
        "overview_h",
        "overview_area",
        "level0_x",
        "level0_y",
        "level0_w",
        "level0_h",
        "crop_level",
        "mask_method",
        "crop_path",
        "artifact_path",
        "ownership_path",
        "candidate_mask_path",
        "core_mask_path",
        "mask_path",
        "rgba_path",
        "overlay_path",
        "qc_path",
        "fg_ratio",
        "border_touch_ratio",
        "neighbor_occupancy_ratio",
        "flag_mask_touches_crop_border_too_much",
        "flag_mask_occupies_neighbor_ownership_zone",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=cols)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in cols})


def write_experiment_note(
    note_path: Path,
    slide_path: Path,
    expected_labels: Sequence[SectionLabel],
    selected_seed: int,
    crop_level: int,
    detected_count: int,
    mask_method: str,
) -> None:
    lines = [
        "# NDPI Whole-Slide Review Experiment",
        "",
        f"- Generated: {datetime.now().isoformat(timespec='seconds')}",
        f"- Random seed: {selected_seed}",
        f"- Selected slide: `{slide_path.name}`",
        f"- Expected sections from filename: {', '.join(label.short_label for label in expected_labels)}",
        f"- Detected candidate count: {detected_count}",
        f"- Crop export level: `{crop_level}`",
        f"- Mask method: `{mask_method}`",
        "",
        "## Working Assumptions",
        "",
        "- Section indices in filenames are sampled at interval 6; two-number ranges are expanded by step 6.",
        "- Mount order is assigned left-to-right on the first row, then left-to-right on the second row.",
        "- Review crops are exported from pyramid level 3 for tractable file sizes; level-0 bounding boxes are preserved in CSV so the same ROIs can be re-extracted later.",
        "- Each crop now carries a slide-level ownership mask so adjacent slices can be suppressed before final mask generation.",
        "- `baseline_v1` is the fixed heuristic baseline for current review use.",
        "- `soft_support_mgac` remains an experimental prototype for future development.",
        "",
        "## Issues Encountered",
        "",
        "- The NDPI `macro` associated image is useful for visual review, but its aspect ratio does not match the main image pyramid. Candidate proposals therefore use the smallest main pyramid level instead of the macro image.",
        "- A simple foreground threshold on the overview can split one slice into multiple components. The current experiment merges horizontally adjacent components on the same row before section assignment.",
        "- Close-contact failure cases can still survive the prototype, especially when glass-edge strips or adjacent slices sit very near the target mask boundary.",
        "",
        "## Follow-up Questions",
        "",
        "- Whether every slide in this batch always uses exactly two rows when more than one slice is present.",
        "- Whether the preferred review export should move from level 3 to level 2 for sharper boundaries once the candidate boxes are confirmed.",
        "",
    ]
    note_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Whole-slide NDPI review experiment")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--review-root", required=True)
    parser.add_argument("--seed", type=int, default=20260306)
    parser.add_argument("--slide-name", default="")
    parser.add_argument("--crop-level", type=int, default=3)
    parser.add_argument("--mask-method", default="baseline_v1")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    review_root = Path(args.review_root)
    ensure_dir(review_root)

    inventory_rows = collect_slide_inventory(input_dir)
    if not inventory_rows:
        raise SystemExit(f"No NDPI files found in {input_dir}")

    slide_paths = [Path(row["slide_path"]) for row in inventory_rows]
    if args.slide_name:
        slide_path = input_dir / args.slide_name
        if not slide_path.exists():
            raise SystemExit(f"Slide not found: {slide_path}")
    else:
        rng = random.Random(args.seed)
        slide_path = rng.choice(slide_paths)

    stain, expected_labels = parse_slide_stem(slide_path.stem)
    run_dir = review_root / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_ndpi_review_{slide_path.stem}"
    ensure_dir(run_dir)

    dirs = {
        "inventory": run_dir / "00_inventory",
        "overview": run_dir / "01_overview",
        "candidates": run_dir / "02_candidates",
        "crops": run_dir / f"03_crops_level{args.crop_level}",
        "masks": run_dir / "04_masks",
        "rgba": run_dir / "05_rgba",
        "qc": run_dir / "06_qc",
        "notes": run_dir / "07_notes",
    }
    for path in dirs.values():
        ensure_dir(path)

    write_inventory_csv(dirs["inventory"] / "slide_inventory.csv", inventory_rows)
    print(f"[inventory] wrote {dirs['inventory'] / 'slide_inventory.csv'}", flush=True)

    slide = openslide.OpenSlide(str(slide_path))
    if args.crop_level >= slide.level_count:
        raise SystemExit(f"crop level {args.crop_level} >= level_count {slide.level_count}")

    config = {
        "slide_path": str(slide_path),
        "stain": stain,
        "method_version": args.mask_method,
        "expected_labels": [asdict(label) for label in expected_labels],
        "seed": args.seed,
        "crop_level": args.crop_level,
        "slide_dimensions": slide.dimensions,
        "level_dimensions": slide.level_dimensions,
        "level_downsamples": list(slide.level_downsamples),
        "mpp_x": slide.properties.get("openslide.mpp-x", ""),
        "mpp_y": slide.properties.get("openslide.mpp-y", ""),
    }
    (run_dir / "experiment_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    print(f"[slide] selected {slide_path.name}", flush=True)
    overview_rgb = read_overview(slide)
    write_rgb_png(dirs["overview"] / "overview_level_last.png", overview_rgb)
    if "macro" in slide.associated_images:
        Image.Image.save(slide.associated_images["macro"], dirs["overview"] / "overview_assoc_macro.png")
    print(f"[overview] exported level {overview_level(slide)} overview", flush=True)

    sat, nonwhite, component_mask = component_mask_from_overview(overview_rgb, stain=stain)
    write_gray_png(dirs["overview"] / "overview_saturation.png", sat)
    write_gray_png(dirs["overview"] / "overview_nonwhite.png", nonwhite)
    write_gray_png(dirs["overview"] / "overview_component_mask.png", component_mask)

    candidates = find_candidate_components(component_mask, len(expected_labels))
    candidates = assign_sections(candidates, expected_labels)
    print(f"[candidates] expected={len(expected_labels)} detected={len(candidates)}", flush=True)

    boxed = draw_overview_boxes(overview_rgb, candidates, expected_labels)
    write_rgb_png(dirs["overview"] / "overview_final_boxes.png", boxed)

    candidate_rows = []
    all_overlays = []
    overview_downsample = float(slide.level_downsamples[overview_level(slide)])
    crop_downsample = float(slide.level_downsamples[args.crop_level])
    for idx, candidate in enumerate(candidates, start=1):
        label = candidate.section.short_label if candidate.section else f"cand_{idx}"
        print(f"[crop] {idx}/{len(candidates)} {label}", flush=True)
        pad_overview = max(24, int(round(max(candidate.w, candidate.h) * 0.08)))
        bbox_level0 = convert_bbox_to_level0(slide, candidate, pad_overview=pad_overview)
        crop_rgb = extract_crop(slide, bbox_level0, args.crop_level)
        ownership_strict, ownership_soft, support_mask = build_crop_ownership_masks(
            target_candidate=candidate,
            all_candidates=candidates,
            crop_bbox_level0=bbox_level0,
            crop_shape=crop_rgb.shape[:2],
            crop_downsample=crop_downsample,
            overview_downsample=overview_downsample,
        )
        target_center_px = level0_point_to_crop(
            candidate_center_level0(candidate, overview_downsample),
            crop_bbox_level0=bbox_level0,
            crop_downsample=crop_downsample,
        )
        mask_result = build_crop_mask(
            crop_rgb,
            ownership_strict=ownership_strict,
            ownership_soft=ownership_soft,
            support_mask=support_mask,
            target_center_px=target_center_px,
            mask_method=args.mask_method,
            stain=stain,
        )
        score = mask_result["score"]
        artifact = mask_result["artifact"]
        ownership = mask_result["ownership"]
        candidate_mask = mask_result["candidate"]
        core_mask = mask_result["core"]
        blur = mask_result["blur"]
        mask = mask_result["mask"]
        fg_ratio = float((mask > 0).mean())
        overlay = overlay_mask(crop_rgb, mask)

        stem = f"{idx:02d}_{label}"
        crop_path = dirs["crops"] / f"{stem}_crop_rgb.png"
        score_path = dirs["masks"] / f"{stem}_score.png"
        artifact_path = dirs["masks"] / f"{stem}_artifact.png"
        ownership_path = dirs["masks"] / f"{stem}_ownership.png"
        candidate_mask_path = dirs["masks"] / f"{stem}_candidate.png"
        core_mask_path = dirs["masks"] / f"{stem}_core.png"
        blur_path = dirs["masks"] / f"{stem}_score_blur.png"
        mask_path = dirs["masks"] / f"{stem}_mask.png"
        overlay_path = dirs["qc"] / f"{stem}_overlay.png"
        rgba_path = dirs["rgba"] / f"{stem}_rgba.png"
        qc_path = dirs["qc"] / f"{stem}_qc_panel.png"

        write_rgb_png(crop_path, crop_rgb)
        write_gray_png(score_path, score)
        write_gray_png(artifact_path, artifact)
        write_gray_png(ownership_path, ownership)
        write_gray_png(candidate_mask_path, candidate_mask)
        write_gray_png(core_mask_path, core_mask)
        write_gray_png(blur_path, blur)
        write_gray_png(mask_path, mask)
        write_rgb_png(overlay_path, overlay)
        rgba_from_mask(crop_rgb, mask).save(rgba_path)
        write_contact_sheet(
            qc_path,
            [
                (f"{label} crop", crop_rgb),
                (f"{label} score", score),
                (f"{label} artifact", artifact),
                (f"{label} ownership", ownership),
                (f"{label} candidate", candidate_mask),
                (f"{label} core", core_mask),
                (f"{label} blur", blur),
                (f"{label} mask", mask),
                (f"{label} overlay", overlay),
            ],
        )

        thumb = cv2.resize(overlay, (700, max(1, int(round(700 * overlay.shape[0] / overlay.shape[1])))), interpolation=cv2.INTER_AREA)
        all_overlays.append((label, thumb))
        row = {
            "candidate_rank": idx,
            "assigned_label": label,
            "assigned_sample_id": candidate.section.sample_id if candidate.section else "",
            "assigned_section_id": candidate.section.section_id if candidate.section else "",
            "overview_x": candidate.x,
            "overview_y": candidate.y,
            "overview_w": candidate.w,
            "overview_h": candidate.h,
            "overview_area": candidate.area,
            "level0_x": bbox_level0[0],
            "level0_y": bbox_level0[1],
            "level0_w": bbox_level0[2],
            "level0_h": bbox_level0[3],
            "crop_level": args.crop_level,
            "mask_method": args.mask_method,
            "crop_path": str(crop_path),
            "artifact_path": str(artifact_path),
            "ownership_path": str(ownership_path),
            "candidate_mask_path": str(candidate_mask_path),
            "core_mask_path": str(core_mask_path),
            "mask_path": str(mask_path),
            "rgba_path": str(rgba_path),
            "overlay_path": str(overlay_path),
            "qc_path": str(qc_path),
            "fg_ratio": f"{fg_ratio:.6f}",
            "border_touch_ratio": f"{mask_result['border_touch_ratio']:.6f}",
            "neighbor_occupancy_ratio": f"{mask_result['neighbor_occupancy_ratio']:.6f}",
            "flag_mask_touches_crop_border_too_much": str(mask_result["flag_mask_touches_crop_border_too_much"]),
            "flag_mask_occupies_neighbor_ownership_zone": str(mask_result["flag_mask_occupies_neighbor_ownership_zone"]),
        }
        candidate_rows.append(row)

    write_candidate_csv(dirs["candidates"] / "candidate_summary.csv", candidate_rows)
    if all_overlays:
        write_contact_sheet(dirs["qc"] / "all_candidates_contact_sheet.png", all_overlays)

    write_experiment_note(
        dirs["notes"] / "experiment_notes.md",
        slide_path=slide_path,
        expected_labels=expected_labels,
        selected_seed=args.seed,
        crop_level=args.crop_level,
        detected_count=len(candidates),
        mask_method=args.mask_method,
    )

    manifest = {
        "run_dir": str(run_dir),
        "selected_slide": str(slide_path),
        "expected_count": len(expected_labels),
        "detected_count": len(candidates),
        "candidate_summary_csv": str(dirs["candidates"] / "candidate_summary.csv"),
        "overview_boxes_png": str(dirs["overview"] / "overview_final_boxes.png"),
        "contact_sheet": str(dirs["qc"] / "all_candidates_contact_sheet.png"),
        "notes": str(dirs["notes"] / "experiment_notes.md"),
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("[done] experiment complete", flush=True)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

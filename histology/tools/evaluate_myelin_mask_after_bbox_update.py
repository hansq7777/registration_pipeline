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
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from histology.gui_mvp.hitl_gui.pipeline_adapters.slide_io import load_slide_bundle
from histology.tools.run_ndpi_review_experiment import (
    CandidateBox,
    SectionLabel,
    assign_sections,
    build_crop_mask_baseline,
    build_crop_ownership_masks,
    candidate_center_level0,
    component_mask_from_overview,
    find_candidate_components,
    level0_point_to_crop,
    parse_slide_stem,
    proposal_crop_rect_overview,
    expand_candidate_rect,
)


@dataclass
class GtSection:
    label: str
    sample_id: str
    section_id: int
    gt_dir: Path
    proposal_bbox_overview: dict | None
    gt_crop_bbox_level0: tuple[int, int, int, int]
    gt_mask: np.ndarray
    gt_crop_shape: tuple[int, int]


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


def collect_gt_sections(
    gt_root: Path,
    sample_ids: set[str] | None = None,
    section_labels: set[str] | None = None,
) -> list[GtSection]:
    items: list[GtSection] = []
    for path in sorted(gt_root.iterdir()):
        if not path.is_dir():
            continue
        meta_path = path / "metadata.json"
        mask_path = path / "tissue_mask_final.png"
        if not meta_path.exists() or not mask_path.exists():
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        label = meta["label"]
        if section_labels and label not in section_labels:
            continue
        sample_id, sec_blob = label.split("_", 1)
        if sample_ids and sample_id not in sample_ids:
            continue
        gt_mask = np.asarray(Image.open(mask_path).convert("L")) > 0
        crop_bbox_level0 = _parse_gt_crop_bbox_level0(meta)
        if crop_bbox_level0 is None:
            continue
        items.append(
            GtSection(
                label=label,
                sample_id=sample_id,
                section_id=int(sec_blob),
                gt_dir=path,
                proposal_bbox_overview=_parse_gt_proposal_bbox_overview(meta),
                gt_crop_bbox_level0=crop_bbox_level0,
                gt_mask=gt_mask,
                gt_crop_shape=gt_mask.shape[:2],
            )
        )
    return items


def build_section_to_slide_index(input_dir: Path) -> dict[tuple[str, str, int], Path]:
    index: dict[tuple[str, str, int], Path] = {}
    for slide_path in sorted(input_dir.glob("*.ndpi")):
        if slide_path.name.startswith("._"):
            continue
        stain, labels = parse_slide_stem(slide_path.stem)
        for label in labels:
            index[(stain.lower(), label.sample_id, label.section_id)] = slide_path
    return index


def overview_rect_from_bbox_dict(bbox: dict, shape_wh: tuple[int, int]) -> tuple[int, int, int, int]:
    dummy = CandidateBox(
        candidate_rank=1,
        x=int(bbox["x"]),
        y=int(bbox["y"]),
        w=int(bbox["w"]),
        h=int(bbox["h"]),
        area=int(bbox["w"]) * int(bbox["h"]),
        cx=float(bbox["x"]) + float(bbox["w"]) / 2.0,
        cy=float(bbox["y"]) + float(bbox["h"]) / 2.0,
        touches_border=False,
    )
    return expand_candidate_rect(
        dummy,
        shape_wh,
        left_ratio=0.08,
        top_ratio=0.08,
        right_ratio=0.08,
        bottom_ratio=0.08,
    )


def open_slide_candidates(slide_path: Path) -> tuple[object, np.ndarray, list[CandidateBox], dict[str, CandidateBox]]:
    loaded = load_slide_bundle(slide_path, "gallyas")
    overview_rgb = np.asarray(loaded.overview)
    _, labels = parse_slide_stem(slide_path.stem)
    _, _, component_mask = component_mask_from_overview(overview_rgb, stain="gallyas")
    candidates = find_candidate_components(component_mask, len(labels))
    candidates = assign_sections(candidates, labels)
    candidate_map = {cand.section.short_label: cand for cand in candidates if cand.section is not None}
    return loaded, overview_rgb, candidates, candidate_map


def extract_crop_from_overview_rect(loaded, crop_rect_overview: tuple[int, int, int, int], crop_level: int) -> np.ndarray:
    crop_level = min(crop_level, len(loaded.level_downsamples) - 1)
    bbox_level0 = overview_rect_to_level0(loaded, crop_rect_overview)
    if loaded.backend == "openslide":
        import openslide

        slide = openslide.OpenSlide(str(loaded.slide_path))
        try:
            downsample = float(slide.level_downsamples[crop_level])
            out_w = max(1, int(round(bbox_level0[2] / downsample)))
            out_h = max(1, int(round(bbox_level0[3] / downsample)))
            return np.asarray(slide.read_region((bbox_level0[0], bbox_level0[1]), crop_level, (out_w, out_h)).convert("RGB"))
        finally:
            slide.close()

    import tifffile
    import zarr

    target_downsample = float(loaded.level_downsamples[crop_level])
    midres_downsample = float(loaded.tifffile_midres_downsample)
    overview_to_midres = float(loaded.tifffile_overview_scale_from_midres)
    x1_ov, y1_ov, x2_ov, y2_ov = crop_rect_overview

    with tifffile.TiffFile(str(loaded.slide_path)) as tf:
        if target_downsample >= midres_downsample:
            arr = zarr.open(tf.pages[loaded.tifffile_midres_page_index].aszarr(), mode="r")
            x1 = int(round(x1_ov * overview_to_midres))
            y1 = int(round(y1_ov * overview_to_midres))
            x2 = int(round(x2_ov * overview_to_midres))
            y2 = int(round(y2_ov * overview_to_midres))
            x1 = max(0, min(arr.shape[1] - 1, x1))
            y1 = max(0, min(arr.shape[0] - 1, y1))
            x2 = max(x1 + 1, min(arr.shape[1], x2))
            y2 = max(y1 + 1, min(arr.shape[0], y2))
            crop = np.asarray(arr[y1:y2, x1:x2, :], dtype=np.uint8)
            if target_downsample > midres_downsample:
                scale = midres_downsample / target_downsample
                out_w = max(1, int(round(crop.shape[1] * scale)))
                out_h = max(1, int(round(crop.shape[0] * scale)))
                crop = cv2.resize(crop, (out_w, out_h), interpolation=cv2.INTER_AREA)
            return crop

        arr0 = zarr.open(tf.pages[0].aszarr(), mode="r")
        overview_downsample = float(loaded.level_downsamples[loaded.overview_level])
        x0 = int(round(x1_ov * overview_downsample))
        y0 = int(round(y1_ov * overview_downsample))
        x1 = max(0, min(tf.pages[0].shape[1] - 1, x0))
        y1 = max(0, min(tf.pages[0].shape[0] - 1, y0))
        x2 = max(x1 + 1, min(tf.pages[0].shape[1], int(round(x2_ov * overview_downsample))))
        y2 = max(y1 + 1, min(tf.pages[0].shape[0], int(round(y2_ov * overview_downsample))))
        crop0 = np.asarray(arr0[y1:y2, x1:x2, :], dtype=np.uint8)
        out_w = max(1, int(round((x2 - x1) / target_downsample)))
        out_h = max(1, int(round((y2 - y1) / target_downsample)))
        return cv2.resize(crop0, (out_w, out_h), interpolation=cv2.INTER_AREA)


def overview_rect_to_level0(loaded, crop_rect_overview: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    overview_downsample = float(loaded.level_downsamples[loaded.overview_level])
    x1, y1, x2, y2 = crop_rect_overview
    x0 = int(round(x1 * overview_downsample))
    y0 = int(round(y1 * overview_downsample))
    w0 = int(round((x2 - x1) * overview_downsample))
    h0 = int(round((y2 - y1) * overview_downsample))
    w0 = min(w0, loaded.level_dimensions[0][0] - x0)
    h0 = min(h0, loaded.level_dimensions[0][1] - y0)
    return x0, y0, w0, h0


def overlap_rect(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> tuple[int, int, int, int] | None:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1 = max(ax, bx)
    y1 = max(ay, by)
    x2 = min(ax + aw, bx + bw)
    y2 = min(ay + ah, by + bh)
    if x1 >= x2 or y1 >= y2:
        return None
    return x1, y1, x2 - x1, y2 - y1


def project_gt_to_new_crop(
    gt_mask_old: np.ndarray,
    old_crop_bbox_level0: tuple[int, int, int, int],
    new_crop_bbox_level0: tuple[int, int, int, int],
    new_crop_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, float]:
    new_h, new_w = new_crop_shape
    gt_new = np.zeros((new_h, new_w), dtype=bool)
    valid = np.zeros((new_h, new_w), dtype=bool)

    overlap = overlap_rect(old_crop_bbox_level0, new_crop_bbox_level0)
    total_gt = int(gt_mask_old.sum())
    if overlap is None:
        return gt_new, valid, 0.0

    ox, oy, ow, oh = overlap
    old_x, old_y, old_w0, old_h0 = old_crop_bbox_level0
    new_x, new_y, new_w0, new_h0 = new_crop_bbox_level0
    old_h, old_w = gt_mask_old.shape[:2]

    src_x1 = int(np.floor((ox - old_x) * old_w / max(1.0, old_w0)))
    src_y1 = int(np.floor((oy - old_y) * old_h / max(1.0, old_h0)))
    src_x2 = int(np.ceil((ox + ow - old_x) * old_w / max(1.0, old_w0)))
    src_y2 = int(np.ceil((oy + oh - old_y) * old_h / max(1.0, old_h0)))
    src_x1 = max(0, min(old_w - 1, src_x1))
    src_y1 = max(0, min(old_h - 1, src_y1))
    src_x2 = max(src_x1 + 1, min(old_w, src_x2))
    src_y2 = max(src_y1 + 1, min(old_h, src_y2))
    src_mask_bool = gt_mask_old[src_y1:src_y2, src_x1:src_x2].astype(bool)
    src_mask = src_mask_bool.astype(np.uint8) * 255

    dst_x1 = int(np.floor((ox - new_x) * new_w / max(1.0, new_w0)))
    dst_y1 = int(np.floor((oy - new_y) * new_h / max(1.0, new_h0)))
    dst_x2 = int(np.ceil((ox + ow - new_x) * new_w / max(1.0, new_w0)))
    dst_y2 = int(np.ceil((oy + oh - new_y) * new_h / max(1.0, new_h0)))
    dst_x1 = max(0, min(new_w - 1, dst_x1))
    dst_y1 = max(0, min(new_h - 1, dst_y1))
    dst_x2 = max(dst_x1 + 1, min(new_w, dst_x2))
    dst_y2 = max(dst_y1 + 1, min(new_h, dst_y2))

    dst_w = max(1, dst_x2 - dst_x1)
    dst_h = max(1, dst_y2 - dst_y1)
    resized = cv2.resize(src_mask, (dst_w, dst_h), interpolation=cv2.INTER_NEAREST) > 0
    gt_new[dst_y1:dst_y2, dst_x1:dst_x2] = resized
    valid[dst_y1:dst_y2, dst_x1:dst_x2] = True
    crop_coverage = float(src_mask_bool.sum() / max(1, total_gt))
    return gt_new, valid, crop_coverage


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


def overlay_mask(crop_rgb: np.ndarray, mask: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    out = crop_rgb.astype(np.float32).copy()
    keep = mask > 0
    col = np.array(color, dtype=np.float32)
    out[keep] = 0.70 * out[keep] + 0.30 * col
    return np.clip(out, 0, 255).astype(np.uint8)


def combined_overlay(crop_rgb: np.ndarray, pred: np.ndarray, gt: np.ndarray, valid: np.ndarray) -> np.ndarray:
    out = crop_rgb.astype(np.float32).copy()
    out[valid & ~gt] = 0.85 * out[valid & ~gt] + 0.15 * np.array([255, 255, 0], dtype=np.float32)
    out[gt] = 0.70 * out[gt] + 0.30 * np.array([0, 255, 0], dtype=np.float32)
    out[pred] = 0.70 * out[pred] + 0.30 * np.array([255, 0, 0], dtype=np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)


def run_mask_method(
    crop_rgb: np.ndarray,
    crop_bbox_level0: tuple[int, int, int, int],
    target_candidate: CandidateBox,
    all_candidates: list[CandidateBox],
    crop_downsample: float,
    overview_downsample: float,
    *,
    params: dict,
) -> np.ndarray:
    ownership_strict, ownership_soft, support_mask = build_crop_ownership_masks(
        target_candidate=target_candidate,
        all_candidates=all_candidates,
        crop_bbox_level0=crop_bbox_level0,
        crop_shape=crop_rgb.shape[:2],
        crop_downsample=crop_downsample,
        overview_downsample=overview_downsample,
    )
    target_center_px = level0_point_to_crop(
        candidate_center_level0(target_candidate, overview_downsample),
        crop_bbox_level0=crop_bbox_level0,
        crop_downsample=crop_downsample,
    )
    result = build_crop_mask_baseline(
        crop_rgb,
        ownership_strict,
        ownership_soft,
        support_mask,
        target_center_px,
        "gallyas",
        **params,
    )
    return result["mask"] > 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ndpi-root", required=True)
    parser.add_argument("--gt-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sample-ids", nargs="*", default=["2502"])
    parser.add_argument("--sections", nargs="*", default=[])
    parser.add_argument("--crop-level", type=int, default=3)
    parser.add_argument("--methods", nargs="*", default=[])
    parser.add_argument("--qual-slide", default="")
    parser.add_argument("--qual-sections", nargs="*", default=["2503_102", "2503_108"])
    parser.add_argument("--save-per-section-images", action="store_true")
    args = parser.parse_args()

    ndpi_root = Path(args.ndpi_root)
    gt_root = Path(args.gt_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    per_section_dir = output_dir / "per_section"
    per_section_dir.mkdir(exist_ok=True)

    gt_sections = collect_gt_sections(
        gt_root,
        set(args.sample_ids) if args.sample_ids else None,
        set(args.sections) if args.sections else None,
    )
    slide_index = build_section_to_slide_index(ndpi_root)

    methods = {
        "legacy_bbox_defaultmask": {
            "bbox_mode": "legacy",
            "mask_params": {"gallyas_max_components": 2},
        },
        "updated_bbox_defaultmask": {
            "bbox_mode": "updated",
            "mask_params": {"gallyas_max_components": 2},
        },
        "updated_bbox_relaxed_v1": {
            "bbox_mode": "updated",
            "mask_params": {
                "gallyas_max_components": 2,
                "gallyas_support_soft_frac": 0.024,
                "gallyas_candidate_thresh_scale": 0.97,
                "gallyas_grow_quantile": 0.15,
                "gallyas_grow_scale": 0.80,
            },
        },
        "updated_bbox_relaxed_v2": {
            "bbox_mode": "updated",
            "mask_params": {
                "gallyas_max_components": 2,
                "gallyas_support_soft_frac": 0.028,
                "gallyas_candidate_thresh_scale": 0.95,
                "gallyas_grow_quantile": 0.12,
                "gallyas_grow_scale": 0.78,
            },
        },
        "updated_bbox_balanced2comp_v1": {
            "bbox_mode": "updated",
            "mask_params": {
                "gallyas_max_components": 2,
                "gallyas_support_soft_frac": 0.024,
                "gallyas_candidate_thresh_scale": 0.98,
                "gallyas_grow_quantile": 0.15,
                "gallyas_grow_scale": 0.81,
                "gallyas_secondary_area_frac_primary": 0.12,
                "gallyas_secondary_area_frac_total": 0.05,
                "gallyas_secondary_support_overlap_min": 0.50,
                "gallyas_secondary_score_frac_primary": 0.55,
            },
        },
        "updated_bbox_loose2comp_v1": {
            "bbox_mode": "updated",
            "mask_params": {
                "gallyas_max_components": 2,
                "gallyas_support_soft_frac": 0.026,
                "gallyas_candidate_thresh_scale": 0.96,
                "gallyas_grow_quantile": 0.14,
                "gallyas_grow_scale": 0.80,
                "gallyas_secondary_area_frac_primary": 0.10,
                "gallyas_secondary_area_frac_total": 0.04,
                "gallyas_secondary_support_overlap_min": 0.45,
                "gallyas_secondary_score_frac_primary": 0.50,
            },
        },
        "updated_bbox_loose2comp_v2": {
            "bbox_mode": "updated",
            "mask_params": {
                "gallyas_max_components": 2,
                "gallyas_support_soft_frac": 0.028,
                "gallyas_candidate_thresh_scale": 0.94,
                "gallyas_grow_quantile": 0.12,
                "gallyas_grow_scale": 0.78,
                "gallyas_secondary_area_frac_primary": 0.08,
                "gallyas_secondary_area_frac_total": 0.035,
                "gallyas_secondary_support_overlap_min": 0.40,
                "gallyas_secondary_score_frac_primary": 0.45,
            },
        },
    }
    if args.methods:
        methods = {k: v for k, v in methods.items() if k in set(args.methods)}

    slide_cache = {}
    rows: list[dict] = []
    aggregate: dict[str, list[dict]] = {name: [] for name in methods}

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
            for method_name in methods:
                row = {
                    "section": gt.label,
                    "slide_name": slide_path.name,
                    "method": method_name,
                    "proposal_found": False,
                    "crop_coverage_recall": 0.0,
                    "overall_gt_recall": 0.0,
                    "labeled_dice": 0.0,
                    "labeled_iou": 0.0,
                    "labeled_precision": 0.0,
                    "labeled_recall": 0.0,
                    "boundary_f1_tol32": 0.0,
                    "boundary_f1_tol64": 0.0,
                    "assd_px": float("inf"),
                    "hd95_px": float("inf"),
                }
                rows.append(row)
                aggregate[method_name].append(row)
            continue

        gt_old_crop_bbox_level0 = gt.gt_crop_bbox_level0
        crop_downsample = float(loaded.level_downsamples[min(args.crop_level, len(loaded.level_downsamples) - 1)])
        overview_downsample = float(loaded.level_downsamples[loaded.overview_level])
        crop_cache = {}

        sec_dir = per_section_dir / gt.label
        sec_dir.mkdir(exist_ok=True)
        for method_name, cfg in methods.items():
            bbox_mode = cfg["bbox_mode"]
            if bbox_mode not in crop_cache:
                if bbox_mode == "legacy":
                    crop_rect_ov = expand_candidate_rect(
                        candidate,
                        loaded.overview_size,
                        left_ratio=0.08,
                        top_ratio=0.08,
                        right_ratio=0.08,
                        bottom_ratio=0.08,
                    )
                else:
                    crop_rect_ov = proposal_crop_rect_overview(candidate, overview_rgb, "gallyas")
                crop_bbox_level0 = overview_rect_to_level0(loaded, crop_rect_ov)
                crop_rgb = extract_crop_from_overview_rect(loaded, crop_rect_ov, args.crop_level)
                gt_new, valid_window, crop_coverage = project_gt_to_new_crop(
                    gt.gt_mask,
                    gt_old_crop_bbox_level0,
                    crop_bbox_level0,
                    crop_rgb.shape[:2],
                )
                crop_cache[bbox_mode] = {
                    "crop_rect_ov": crop_rect_ov,
                    "crop_bbox_level0": crop_bbox_level0,
                    "crop_rgb": crop_rgb,
                    "gt_new": gt_new,
                    "valid_window": valid_window,
                    "crop_coverage": crop_coverage,
                }
            cached = crop_cache[bbox_mode]
            crop_rect_ov = cached["crop_rect_ov"]
            crop_bbox_level0 = cached["crop_bbox_level0"]
            crop_rgb = cached["crop_rgb"]
            gt_new = cached["gt_new"]
            valid_window = cached["valid_window"]
            crop_coverage = cached["crop_coverage"]
            pred_mask = run_mask_method(
                crop_rgb,
                crop_bbox_level0,
                candidate,
                candidates,
                crop_downsample,
                overview_downsample,
                params=cfg["mask_params"],
            )
            gt_new_area = max(1, int(gt_new.sum()))
            within_crop_gt_recall = float((pred_mask & gt_new).sum() / gt_new_area) if gt_new.any() else 0.0
            overall_gt_recall = float(within_crop_gt_recall * crop_coverage)

            if valid_window.any():
                pred_eval = pred_mask[valid_window]
                gt_eval = gt_new[valid_window]
                rm = region_metrics(pred_eval, gt_eval)
                ys, xs = np.where(valid_window)
                sub = (slice(int(ys.min()), int(ys.max()) + 1), slice(int(xs.min()), int(xs.max()) + 1))
                cm = contour_metrics(pred_mask[sub], gt_new[sub])
            else:
                rm = {"dice": 0.0, "iou": 0.0, "precision": 0.0, "recall": 0.0}
                cm = {"boundary_f1_tol32": 0.0, "boundary_f1_tol64": 0.0, "assd_px": float("inf"), "hd95_px": float("inf")}

            row = {
                "section": gt.label,
                "slide_name": slide_path.name,
                "method": method_name,
                "proposal_found": True,
                "crop_coverage_recall": crop_coverage,
                "within_crop_gt_recall": within_crop_gt_recall,
                "overall_gt_recall": overall_gt_recall,
                "pred_area_ratio_in_crop": float(pred_mask.mean()),
                "gt_area_ratio_in_crop": float(gt_new.mean()),
                "labeled_dice": rm["dice"],
                "labeled_iou": rm["iou"],
                "labeled_precision": rm["precision"],
                "labeled_recall": rm["recall"],
                "boundary_f1_tol32": cm["boundary_f1_tol32"],
                "boundary_f1_tol64": cm["boundary_f1_tol64"],
                "assd_px": cm["assd_px"],
                "hd95_px": cm["hd95_px"],
                "crop_overview_x1": crop_rect_ov[0],
                "crop_overview_y1": crop_rect_ov[1],
                "crop_overview_x2": crop_rect_ov[2],
                "crop_overview_y2": crop_rect_ov[3],
            }
            rows.append(row)
            aggregate[method_name].append(row)

            if args.save_per_section_images:
                Image.fromarray(combined_overlay(crop_rgb, pred_mask, gt_new, valid_window)).save(sec_dir / f"{method_name}_overlay.png")
                Image.fromarray((pred_mask.astype(np.uint8) * 255)).save(sec_dir / f"{method_name}_mask.png")
                Image.fromarray((gt_new.astype(np.uint8) * 255)).save(sec_dir / f"{method_name}_gt_mapped.png")

    with (output_dir / "comparison_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {}
    for method_name, method_rows in aggregate.items():
        assd_vals = [r["assd_px"] for r in method_rows]
        hd95_vals = [r["hd95_px"] for r in method_rows]
        summary[method_name] = {
            "count": len(method_rows),
            "proposal_found_rate": float(np.mean([bool(r["proposal_found"]) for r in method_rows])),
            "mean_crop_coverage_recall": float(np.mean([r["crop_coverage_recall"] for r in method_rows])),
            "mean_within_crop_gt_recall": float(np.mean([r["within_crop_gt_recall"] for r in method_rows])),
            "mean_overall_gt_recall": float(np.mean([r["overall_gt_recall"] for r in method_rows])),
            "mean_pred_area_ratio_in_crop": float(np.mean([r["pred_area_ratio_in_crop"] for r in method_rows])),
            "mean_gt_area_ratio_in_crop": float(np.mean([r["gt_area_ratio_in_crop"] for r in method_rows])),
            "mean_labeled_dice": float(np.mean([r["labeled_dice"] for r in method_rows])),
            "mean_labeled_iou": float(np.mean([r["labeled_iou"] for r in method_rows])),
            "mean_boundary_f1_tol32": float(np.mean([r["boundary_f1_tol32"] for r in method_rows])),
            "mean_boundary_f1_tol64": float(np.mean([r["boundary_f1_tol64"] for r in method_rows])),
            "mean_assd_px_finite": finite_mean(assd_vals),
            "mean_hd95_px_finite": finite_mean(hd95_vals),
            "assd_finite_count": int(sum(np.isfinite(v) for v in assd_vals)),
            "hd95_finite_count": int(sum(np.isfinite(v) for v in hd95_vals)),
        }
    (output_dir / "aggregate_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    qual_rows = []
    if args.qual_slide:
        qual_path = Path(args.qual_slide)
        loaded, overview_rgb, candidates, candidate_map = open_slide_candidates(qual_path)
        crop_downsample = float(loaded.level_downsamples[min(args.crop_level, len(loaded.level_downsamples) - 1)])
        overview_downsample = float(loaded.level_downsamples[loaded.overview_level])
        qual_dir = output_dir / "qualitative"
        qual_dir.mkdir(exist_ok=True)
        for label in args.qual_sections:
            candidate = candidate_map.get(label)
            if candidate is None:
                continue
            sec_dir = qual_dir / label
            sec_dir.mkdir(exist_ok=True)
            crop_cache = {}
            for method_name, cfg in methods.items():
                bbox_mode = cfg["bbox_mode"]
                if bbox_mode not in crop_cache:
                    if bbox_mode == "legacy":
                        crop_rect_ov = expand_candidate_rect(
                            candidate,
                            loaded.overview_size,
                            left_ratio=0.08,
                            top_ratio=0.08,
                            right_ratio=0.08,
                            bottom_ratio=0.08,
                        )
                    else:
                        crop_rect_ov = proposal_crop_rect_overview(candidate, overview_rgb, "gallyas")
                    crop_bbox_level0 = overview_rect_to_level0(loaded, crop_rect_ov)
                    crop_rgb = extract_crop_from_overview_rect(loaded, crop_rect_ov, args.crop_level)
                    crop_cache[bbox_mode] = {
                        "crop_rect_ov": crop_rect_ov,
                        "crop_bbox_level0": crop_bbox_level0,
                        "crop_rgb": crop_rgb,
                    }
                cached = crop_cache[bbox_mode]
                crop_rect_ov = cached["crop_rect_ov"]
                crop_bbox_level0 = cached["crop_bbox_level0"]
                crop_rgb = cached["crop_rgb"]
                pred_mask = run_mask_method(
                    crop_rgb,
                    crop_bbox_level0,
                    candidate,
                    candidates,
                    crop_downsample,
                    overview_downsample,
                    params=cfg["mask_params"],
                )
                overlay = overlay_mask(crop_rgb, pred_mask, (255, 0, 0))
                Image.fromarray(overlay).save(sec_dir / f"{method_name}_overlay.png")
                Image.fromarray((pred_mask.astype(np.uint8) * 255)).save(sec_dir / f"{method_name}_mask.png")
                qual_rows.append(
                    {
                        "section": label,
                        "method": method_name,
                        "mask_area_ratio": float(pred_mask.mean()),
                        "components": int(cv2.connectedComponents(pred_mask.astype(np.uint8), 8)[0] - 1),
                        "crop_overview_x1": crop_rect_ov[0],
                        "crop_overview_y1": crop_rect_ov[1],
                        "crop_overview_x2": crop_rect_ov[2],
                        "crop_overview_y2": crop_rect_ov[3],
                    }
                )
        (output_dir / "qualitative_summary.json").write_text(json.dumps(qual_rows, indent=2), encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

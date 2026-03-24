#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from histology.tools.evaluate_bbox_proposal_methods import (
    GtSection,
    bbox_score_maps_for_stain,
    build_methods_for_scores,
    build_section_to_slide_index,
    collect_gt_sections,
    coverage_metrics,
    level0_rect_to_overview_rect,
    load_slide_bundle,
    open_slide_handle,
    parse_slide_stem,
    component_mask_from_overview,
    find_candidate_components,
    assign_sections,
    project_rect_to_gt_crop_bounds_level0,
)


METHODS = [
    "exp2_compete_core86_fringe55_hybrid_tight",
    "dr_localadaptive_compete_v1",
]

METHOD_COLORS = {
    "gt": (0, 255, 0),
    "exp2_compete_core86_fringe55_hybrid_tight": (255, 80, 80),
    "dr_localadaptive_compete_v1": (255, 220, 0),
}


def _fit_rgb(path: Path, max_side: int = 1200) -> Image.Image:
    img = Image.open(path).convert("RGB")
    img.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    return img


def _draw_rect(draw: ImageDraw.ImageDraw, rect: tuple[int, int, int, int], color: tuple[int, int, int], width: int) -> None:
    x1, y1, x2, y2 = rect
    draw.rectangle((x1, y1, x2 - 1, y2 - 1), outline=color, width=width)


def _draw_mask_outline(base: Image.Image, mask: np.ndarray, color: tuple[int, int, int], width: int = 2) -> Image.Image:
    arr = np.asarray(base).copy()
    m = (mask.astype(np.uint8) * 255)
    import cv2

    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(arr, contours, -1, color, width)
    return Image.fromarray(arr)


def _bounds_to_rect(bounds: tuple[int, int, int, int] | None) -> tuple[int, int, int, int] | None:
    if bounds is None:
        return None
    x1, y1, x2, y2 = bounds
    return x1, y1, x2, y2


def export_qc(
    ndpi_root: Path,
    gt_root: Path,
    output_dir: Path,
    stain: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "panels").mkdir(exist_ok=True)
    (output_dir / "slides").mkdir(exist_ok=True)

    gt_sections = collect_gt_sections(gt_root)
    slide_index = build_section_to_slide_index(ndpi_root)
    gt_sections_by_slide: dict[Path, list[GtSection]] = {}
    for gt in gt_sections:
        slide_path = slide_index.get((stain.lower(), gt.sample_id, gt.section_id))
        if slide_path is not None:
            gt_sections_by_slide.setdefault(slide_path, []).append(gt)

    rows: list[dict] = []

    for slide_path, slide_gts in sorted(gt_sections_by_slide.items()):
        loaded = load_slide_bundle(slide_path, stain.lower())
        slide_handle = open_slide_handle(loaded)
        overview_rgb = np.asarray(loaded.overview)
        _, labels = parse_slide_stem(slide_path.stem)
        _, _, component_mask = component_mask_from_overview(overview_rgb, stain=stain.lower())
        candidates = find_candidate_components(component_mask, len(labels))
        candidates = assign_sections(candidates, labels)
        candidate_map = {cand.section.short_label: cand for cand in candidates if cand.section is not None}
        score_maps = bbox_score_maps_for_stain(overview_rgb, stain.lower())
        methods = build_methods_for_scores(
            score_maps,
            slide_candidates=candidates,
            loaded_slide=loaded,
            slide_handle=slide_handle,
            overview_rgb=overview_rgb,
        )

        overview_img = Image.fromarray(overview_rgb.astype(np.uint8)).convert("RGB")
        draw = ImageDraw.Draw(overview_img)
        for gt in slide_gts:
            gt_rect = level0_rect_to_overview_rect(gt.gt_crop_bbox_level0, float(loaded.level_downsamples[loaded.overview_level]), loaded.overview_size)
            _draw_rect(draw, gt_rect, METHOD_COLORS["gt"], 4)
            candidate = candidate_map.get(gt.label)
            if candidate is None:
                continue
            for method_name in METHODS:
                rect = methods[method_name](candidate, loaded.overview_size)
                _draw_rect(draw, rect, METHOD_COLORS[method_name], 2)
        overview_img.save(output_dir / "slides" / f"{slide_path.stem.replace(';', '_')}_overview_qc.png")

        for gt in slide_gts:
            candidate = candidate_map.get(gt.label)
            if candidate is None:
                continue
            gt_rect = level0_rect_to_overview_rect(gt.gt_crop_bbox_level0, float(loaded.level_downsamples[loaded.overview_level]), loaded.overview_size)

            overview_crop = overview_img.copy()
            d = ImageDraw.Draw(overview_crop)
            _draw_rect(d, gt_rect, METHOD_COLORS["gt"], 4)
            for method_name in METHODS:
                rect = methods[method_name](candidate, loaded.overview_size)
                _draw_rect(d, rect, METHOD_COLORS[method_name], 3)
                rect_level0 = (
                    int(round(rect[0] * loaded.level_downsamples[loaded.overview_level])),
                    int(round(rect[1] * loaded.level_downsamples[loaded.overview_level])),
                    int(round((rect[2] - rect[0]) * loaded.level_downsamples[loaded.overview_level])),
                    int(round((rect[3] - rect[1]) * loaded.level_downsamples[loaded.overview_level])),
                )
                bounds = project_rect_to_gt_crop_bounds_level0(rect_level0, gt.gt_crop_bbox_level0, gt.crop_shape)
                metrics = coverage_metrics(
                    gt.mask,
                    bounds,
                    proposal_rect_level0_xywh=rect_level0,
                    gt_crop_level0_xywh=gt.gt_crop_bbox_level0,
                )
                rows.append(
                    {
                        "section": gt.label,
                        "slide_name": slide_path.name,
                        "method": method_name,
                        "mask_coverage_recall": metrics["mask_coverage_recall"],
                        "neighbor_overlap_ratio_proposal": 0.0,
                        "proposal_area_to_gt_crop_area_full": metrics["proposal_area_to_gt_crop_area_full"],
                    }
                )

            crop_img = _fit_rgb(gt.gt_dir / "crop_raw.png", max_side=1200)
            crop_img = _draw_mask_outline(crop_img, gt.mask, METHOD_COLORS["gt"], width=3)
            draw_crop = ImageDraw.Draw(crop_img)
            for method_name in METHODS:
                rect = methods[method_name](candidate, loaded.overview_size)
                rect_level0 = (
                    int(round(rect[0] * loaded.level_downsamples[loaded.overview_level])),
                    int(round(rect[1] * loaded.level_downsamples[loaded.overview_level])),
                    int(round((rect[2] - rect[0]) * loaded.level_downsamples[loaded.overview_level])),
                    int(round((rect[3] - rect[1]) * loaded.level_downsamples[loaded.overview_level])),
                )
                bounds = project_rect_to_gt_crop_bounds_level0(rect_level0, gt.gt_crop_bbox_level0, gt.crop_shape)
                proj_rect = _bounds_to_rect(bounds)
                if proj_rect is None:
                    continue
                sx = crop_img.size[0] / float(gt.crop_shape[1])
                sy = crop_img.size[1] / float(gt.crop_shape[0])
                x1, y1, x2, y2 = proj_rect
                draw_crop.rectangle((x1 * sx, y1 * sy, x2 * sx - 1, y2 * sy - 1), outline=METHOD_COLORS[method_name], width=3)

            canvas = Image.new("RGB", (overview_crop.size[0] + crop_img.size[0], max(overview_crop.size[1], crop_img.size[1]) + 80), (24, 24, 24))
            canvas.paste(overview_crop, (0, 0))
            canvas.paste(crop_img, (overview_crop.size[0], 0))
            draw_panel = ImageDraw.Draw(canvas)
            draw_panel.text((16, max(overview_crop.size[1], crop_img.size[1]) + 12), f"{gt.label} | green=GT crop/mask | red=exp2_compete | yellow=dr_localadaptive", fill=(255, 255, 255))
            canvas.save(output_dir / "panels" / f"{gt.label}_qc_panel.png")

        if slide_handle is not None and hasattr(slide_handle, "close"):
            try:
                slide_handle.close()
            except Exception:
                pass

    with (output_dir / "qc_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ndpi-root", required=True)
    parser.add_argument("--gt-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--stain", default="gallyas")
    args = parser.parse_args()
    export_qc(Path(args.ndpi_root), Path(args.gt_root), Path(args.output_dir), args.stain)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

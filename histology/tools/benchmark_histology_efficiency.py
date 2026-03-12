#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from histology.gui_mvp.hitl_gui.pipeline_adapters.segmentation_adapter import parse_slide_labels, propose_from_overview
from histology.gui_mvp.hitl_gui.pipeline_adapters.slide_io import (
    extract_crop_for_preview,
    load_slide_bundle,
    open_slide_handle,
)
from histology.tools.evaluate_myelin_mask_after_bbox_update import (
    collect_gt_sections,
    contour_metrics,
    overview_rect_to_level0,
    project_gt_to_new_crop,
    region_metrics,
)
from histology.tools.run_ndpi_review_experiment import (
    build_crop_mask_baseline,
    build_crop_ownership_masks,
    candidate_center_level0,
    level0_point_to_crop,
    proposal_crop_rect_overview,
)


@dataclass
class TimerStat:
    name: str
    repeats: int
    mean_sec: float
    median_sec: float
    min_sec: float
    max_sec: float


def timed_repeat(name: str, repeats: int, fn) -> TimerStat:
    values: list[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        values.append(time.perf_counter() - t0)
    return TimerStat(
        name=name,
        repeats=repeats,
        mean_sec=float(statistics.mean(values)),
        median_sec=float(statistics.median(values)),
        min_sec=float(min(values)),
        max_sec=float(max(values)),
    )


def tool_candidates_for_loaded_slide(slide_path: Path, stain: str, overview_rgb: np.ndarray):
    tool_path = ROOT / "histology" / "tools" / "run_ndpi_review_experiment.py"
    import importlib.util

    spec = importlib.util.spec_from_file_location("histology_ndpi_tool_bench", tool_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load tool module from {tool_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    _, labels = parse_slide_labels(slide_path.stem)
    _, _, component_mask = module.component_mask_from_overview(overview_rgb, stain=stain)
    candidates = module.find_candidate_components(component_mask, len(labels))
    candidates = module.assign_sections(candidates, labels)
    candidate_map = {cand.section.short_label: cand for cand in candidates if cand.section is not None}
    return module, candidates, candidate_map


def benchmark_slide(slide_path: Path, stain: str, proposal_labels: list[str], crop_levels: list[int]) -> dict:
    print(f"[slide] loading {slide_path.name}", flush=True)
    load_stat = timed_repeat("load_slide_bundle", 3, lambda: load_slide_bundle(slide_path, stain))
    loaded = load_slide_bundle(slide_path, stain)
    overview_rgb = np.asarray(loaded.overview)

    print(f"[slide] proposal {slide_path.name}", flush=True)
    proposal_stat = timed_repeat(
        "overview_proposal",
        3,
        lambda: propose_from_overview(slide_path, stain, parse_slide_labels(slide_path.stem)[1], overview_rgb),
    )
    proposals = propose_from_overview(slide_path, stain, parse_slide_labels(slide_path.stem)[1], overview_rgb)
    proposal_map = {p.label: p for p in proposals}
    selected = [proposal_map[label] for label in proposal_labels if label in proposal_map]

    crop_stats: dict[str, dict] = {}
    if loaded.backend == "openslide":
        for level in crop_levels:
            print(f"[slide] crop benchmark {slide_path.name} level {level} reopen vs persistent", flush=True)
            reopen_stat = timed_repeat(
                f"crop_level{level}_reopen",
                len(selected),
                lambda lvl=level, idx=[0]: extract_crop_for_preview(
                    loaded, selected[(idx.__setitem__(0, (idx[0] + 1) % len(selected)) or idx[0] - 1)], crop_level=lvl
                ),
            )
            handle = open_slide_handle(loaded)
            try:
                handle_stat = timed_repeat(
                    f"crop_level{level}_persistent_handle",
                    len(selected),
                    lambda lvl=level, idx=[0]: extract_crop_for_preview(
                        loaded,
                        selected[(idx.__setitem__(0, (idx[0] + 1) % len(selected)) or idx[0] - 1)],
                        crop_level=lvl,
                        slide_handle=handle,
                    ),
                )
            finally:
                if handle is not None:
                    handle.close()
            crop_stats[f"level_{level}"] = {
                "reopen": asdict(reopen_stat),
                "persistent_handle": asdict(handle_stat),
            }
    else:
        for level in crop_levels:
            print(f"[slide] crop benchmark {slide_path.name} level {level} tifffile_proxy", flush=True)
            current_stat = timed_repeat(
                f"crop_level{level}_tifffile_proxy",
                max(1, len(selected)),
                lambda lvl=level, idx=[0]: extract_crop_for_preview(
                    loaded, selected[(idx.__setitem__(0, (idx[0] + 1) % len(selected)) or idx[0] - 1)], crop_level=lvl
                ),
            )
            crop_stats[f"level_{level}"] = {"tifffile_proxy": asdict(current_stat)}

    mask_stats: dict[str, dict] = {}
    for level in crop_levels:
        print(f"[slide] mask benchmark {slide_path.name} level {level}", flush=True)
        crops = []
        handle = open_slide_handle(loaded)
        try:
            for proposal in selected:
                crop = extract_crop_for_preview(loaded, proposal, crop_level=level, slide_handle=handle)
                crops.append((proposal, crop))
        finally:
            if handle is not None:
                handle.close()

        def _mask_once(idx=[0]):
            proposal, crop_rgb = crops[(idx.__setitem__(0, (idx[0] + 1) % len(crops)) or idx[0] - 1)]
            from histology.gui_mvp.hitl_gui.pipeline_adapters.segmentation_adapter import compute_auto_masks

            compute_auto_masks(
                crop_rgb,
                stain,
                loaded_slide=loaded,
                target_proposal=proposal,
                all_proposals=proposals,
                crop_level=level,
            )

        mask_stat = timed_repeat(f"mask_level{level}", len(crops), _mask_once)
        mask_stats[f"level_{level}"] = asdict(mask_stat)

    return {
        "slide_name": slide_path.name,
        "backend": loaded.backend,
        "overview_size": loaded.overview_size,
        "proposal_count": len(proposals),
        "load": asdict(load_stat),
        "proposal": asdict(proposal_stat),
        "crop": crop_stats,
        "mask": mask_stats,
    }


def quality_tradeoff(
    ndpi_root: Path,
    gt_root: Path,
    sections: list[str],
    crop_levels: list[int],
) -> dict:
    gt_sections = collect_gt_sections(gt_root, sample_ids=None, section_labels=set(sections))
    slide_index: dict[tuple[str, str, int], Path] = {}
    for slide_path in sorted(ndpi_root.glob("*.ndpi")):
        if slide_path.name.startswith("._"):
            continue
        stain, labels = parse_slide_labels(slide_path.stem)
        if stain.lower() != "gallyas":
            continue
        for label in labels:
            slide_index[("gallyas", label.sample_id, label.section_id)] = slide_path

    summary: dict[str, list[dict]] = {f"level_{lvl}": [] for lvl in crop_levels}
    for gt in gt_sections:
        print(f"[quality] {gt.label}", flush=True)
        slide_path = slide_index.get(("gallyas", gt.sample_id, gt.section_id))
        if slide_path is None:
            continue
        loaded = load_slide_bundle(slide_path, "gallyas")
        overview_rgb = np.asarray(loaded.overview)
        tool, candidates, candidate_map = tool_candidates_for_loaded_slide(slide_path, "gallyas", overview_rgb)
        candidate = candidate_map.get(gt.label)
        if candidate is None:
            continue
        crop_rect_ov = proposal_crop_rect_overview(candidate, overview_rgb, "gallyas")
        crop_bbox_level0 = overview_rect_to_level0(loaded, crop_rect_ov)
        gt_old_crop_rect_ov = (
            int(gt.proposal_bbox_overview["x"]),
            int(gt.proposal_bbox_overview["y"]),
            int(gt.proposal_bbox_overview["x"] + gt.proposal_bbox_overview["w"]),
            int(gt.proposal_bbox_overview["y"] + gt.proposal_bbox_overview["h"]),
        )
        gt_old_crop_bbox_level0 = overview_rect_to_level0(loaded, gt_old_crop_rect_ov)

        for level in crop_levels:
            t0 = time.perf_counter()
            handle = open_slide_handle(loaded)
            try:
                proposal_box = next(
                    p for p in propose_from_overview(slide_path, "gallyas", parse_slide_labels(slide_path.stem)[1], overview_rgb) if p.label == gt.label
                )
                crop_rgb = extract_crop_for_preview(loaded, proposal_box, crop_level=level, slide_handle=handle)
            finally:
                if handle is not None:
                    handle.close()
            crop_time = time.perf_counter() - t0

            crop_downsample = float(loaded.level_downsamples[min(level, len(loaded.level_downsamples) - 1)])
            overview_downsample = float(loaded.level_downsamples[loaded.overview_level])
            ownership_strict, ownership_soft, support_mask = build_crop_ownership_masks(
                target_candidate=candidate,
                all_candidates=candidates,
                crop_bbox_level0=crop_bbox_level0,
                crop_shape=crop_rgb.shape[:2],
                crop_downsample=crop_downsample,
                overview_downsample=overview_downsample,
            )
            target_center_px = level0_point_to_crop(
                candidate_center_level0(candidate, overview_downsample),
                crop_bbox_level0=crop_bbox_level0,
                crop_downsample=crop_downsample,
            )
            t1 = time.perf_counter()
            result = build_crop_mask_baseline(
                crop_rgb,
                ownership_strict=ownership_strict,
                ownership_soft=ownership_soft,
                support_mask=support_mask,
                target_center_px=target_center_px,
                stain="gallyas",
            )
            mask_time = time.perf_counter() - t1
            pred_mask = result["mask"] > 0

            gt_new, _, crop_coverage = project_gt_to_new_crop(
                gt.gt_mask,
                gt_old_crop_bbox_level0,
                crop_bbox_level0,
                crop_rgb.shape[:2],
            )
            reg = region_metrics(pred_mask, gt_new)
            contour = contour_metrics(pred_mask, gt_new)
            summary[f"level_{level}"].append(
                {
                    "section": gt.label,
                    "crop_time_sec": crop_time,
                    "mask_time_sec": mask_time,
                    "crop_coverage_recall": crop_coverage,
                    "dice": reg["dice"],
                    "iou": reg["iou"],
                    "boundary_f1_tol32": contour["boundary_f1_tol32"],
                    "boundary_f1_tol64": contour["boundary_f1_tol64"],
                }
            )

    aggregate = {}
    for level_key, rows in summary.items():
        if not rows:
            continue
        aggregate[level_key] = {
            "count": len(rows),
            "mean_crop_time_sec": float(np.mean([r["crop_time_sec"] for r in rows])),
            "mean_mask_time_sec": float(np.mean([r["mask_time_sec"] for r in rows])),
            "mean_crop_coverage_recall": float(np.mean([r["crop_coverage_recall"] for r in rows])),
            "mean_dice": float(np.mean([r["dice"] for r in rows])),
            "mean_iou": float(np.mean([r["iou"] for r in rows])),
            "mean_boundary_f1_tol32": float(np.mean([r["boundary_f1_tol32"] for r in rows])),
            "mean_boundary_f1_tol64": float(np.mean([r["boundary_f1_tol64"] for r in rows])),
        }
    return {"per_section": summary, "aggregate": aggregate}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--gallyas-root",
        default="/mnt/d/Research/Image Analysis/Nanozoomer scans/20250327 rat myelin quantification",
    )
    parser.add_argument(
        "--gallyas-gt-root",
        default="/mnt/d/Research/Image Analysis/Nanozoomer scans/20250327 rat myelin quantification/Tissue&Masks",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    gallyas_root = Path(args.gallyas_root)
    gt_root = Path(args.gallyas_gt_root)

    slide_cases = [
        {
            "slide_path": gallyas_root / "gallyas_2502_42-72.ndpi",
            "stain": "gallyas",
            "proposal_labels": ["2502_42", "2502_48", "2502_54"],
            "crop_levels": [3, 4],
        },
        {
            "slide_path": gallyas_root / "gallyas_2501_114-144.ndpi",
            "stain": "gallyas",
            "proposal_labels": ["2501_114"],
            "crop_levels": [5],
        },
    ]

    slide_results = []
    for case in slide_cases:
        slide_results.append(
            benchmark_slide(
                slide_path=case["slide_path"],
                stain=case["stain"],
                proposal_labels=case["proposal_labels"],
                crop_levels=case["crop_levels"],
            )
        )

    quality = quality_tradeoff(
        ndpi_root=gallyas_root,
        gt_root=gt_root,
        sections=["2502_78", "2502_102", "2502_108"],
        crop_levels=[3, 4],
    )

    result = {"slides": slide_results, "quality_tradeoff": quality}
    (output_dir / "benchmark_results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

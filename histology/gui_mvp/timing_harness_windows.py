from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
HISTOLOGY_ROOT = HERE.parent
REPO_ROOT = HISTOLOGY_ROOT.parent
WORK_ROOT = REPO_ROOT.parent

if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if str(WORK_ROOT) not in sys.path:
    sys.path.insert(0, str(WORK_ROOT))

from hitl_gui.application.session_cache import SlideSessionCache
from hitl_gui.pipeline_adapters import compute_auto_masks, load_slide_bundle, parse_slide_labels, propose_from_overview
from hitl_gui.pipeline_adapters.slide_io import clear_backend_hint, clear_proxy_cache, extract_crop_for_preview
from hitl_gui.pipeline_adapters.tool_bridge import proposal_bbox_level0_gui


@dataclass
class TimingStat:
    mean_sec: float
    median_sec: float
    min_sec: float
    max_sec: float
    repeats: int


def summarize(values: list[float]) -> TimingStat:
    return TimingStat(
        mean_sec=float(statistics.mean(values)),
        median_sec=float(statistics.median(values)),
        min_sec=float(min(values)),
        max_sec=float(max(values)),
        repeats=len(values),
    )


def time_once(fn):
    t0 = time.perf_counter()
    result = fn()
    return time.perf_counter() - t0, result


def time_repeat(repeats: int, fn) -> TimingStat:
    values = []
    for _ in range(repeats):
        t, _ = time_once(fn)
        values.append(t)
    return summarize(values)


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
) -> tuple[np.ndarray, float]:
    new_h, new_w = new_crop_shape
    gt_new = np.zeros((new_h, new_w), dtype=bool)

    overlap = overlap_rect(old_crop_bbox_level0, new_crop_bbox_level0)
    total_gt = int(gt_mask_old.sum())
    if overlap is None:
        return gt_new, 0.0

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
    crop_coverage = float(src_mask_bool.sum() / max(1, total_gt))
    return gt_new, crop_coverage


def region_metrics(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    tp = int((pred & gt).sum())
    fp = int((pred & ~gt).sum())
    fn = int((~pred & gt).sum())
    return {
        "dice": 2 * tp / max(1, 2 * tp + fp + fn),
        "iou": tp / max(1, tp + fp + fn),
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
        return {"boundary_f1_tol32": 0.0, "boundary_f1_tol64": 0.0}
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
    }


def read_gt_item(gt_root: Path, label: str) -> dict:
    item_dir = gt_root / label
    meta = json.loads((item_dir / "metadata.json").read_text(encoding="utf-8"))
    mask = np.asarray(Image.open(item_dir / "tissue_mask_final.png").convert("L")) > 0
    return {"label": label, "meta": meta, "mask": mask}


def proposal_by_label(proposals, label: str):
    for proposal in proposals:
        if proposal.label == label:
            return proposal
    raise KeyError(f"proposal label not found: {label}")

def benchmark_gui_like_session(
    slide_path: Path,
    stain: str,
    preview_label: str,
    section_label: str,
    preview_crop_level: int,
    section_crop_level: int,
) -> dict:
    labels = parse_slide_labels(slide_path.stem)[1]

    clear_backend_hint(slide_path)
    clear_proxy_cache(slide_path)
    cold_load_sec, loaded_cold = time_once(lambda: load_slide_bundle(slide_path, stain))
    cached_load_sec, loaded_cached = time_once(lambda: load_slide_bundle(slide_path, stain))
    cold_prop_sec, cold_proposals = time_once(
        lambda: propose_from_overview(slide_path, stain, labels, np.asarray(loaded_cold.overview))
    )

    preview_proposal = proposal_by_label(cold_proposals, preview_label)
    section_proposal = proposal_by_label(cold_proposals, section_label)

    session = SlideSessionCache()
    set_slide_sec, _ = time_once(lambda: session.set_slide(loaded_cold))
    preview_direct_sec, _ = time_once(lambda: extract_crop_for_preview(loaded_cold, preview_proposal, crop_level=preview_crop_level))
    preview_cache_miss_sec, _ = time_once(lambda: session.get_preview_crop(preview_proposal, preview_crop_level))
    preview_cache_hit_sec, _ = time_once(lambda: session.get_preview_crop(preview_proposal, preview_crop_level))
    section_cache_miss_sec, section_crop = time_once(lambda: session.get_section_crop(section_proposal, section_crop_level))
    section_cache_hit_sec, _ = time_once(lambda: session.get_section_crop(section_proposal, section_crop_level))
    mask_sec, (tissue_mask, artifact_mask) = time_once(
        lambda: compute_auto_masks(
            section_crop,
            stain,
            loaded_slide=loaded_cold,
            target_proposal=section_proposal,
            all_proposals=cold_proposals,
            crop_level=section_crop_level,
        )
    )
    session.close()

    warm_load_sec, loaded_warm = time_once(lambda: load_slide_bundle(slide_path, stain))
    warm_prop_sec, warm_proposals = time_once(
        lambda: propose_from_overview(slide_path, stain, labels, np.asarray(loaded_warm.overview))
    )
    warm_preview_proposal = proposal_by_label(warm_proposals, preview_label)
    warm_section_proposal = proposal_by_label(warm_proposals, section_label)
    warm_session = SlideSessionCache()
    warm_set_slide_sec, _ = time_once(lambda: warm_session.set_slide(loaded_warm))
    warm_preview_cache_miss_sec, _ = time_once(lambda: warm_session.get_preview_crop(warm_preview_proposal, preview_crop_level))
    warm_section_cache_miss_sec, warm_section_crop = time_once(
        lambda: warm_session.get_section_crop(warm_section_proposal, section_crop_level)
    )
    warm_mask_sec, _ = time_once(
        lambda: compute_auto_masks(
            warm_section_crop,
            stain,
            loaded_slide=loaded_warm,
            target_proposal=warm_section_proposal,
            all_proposals=warm_proposals,
            crop_level=section_crop_level,
        )
    )
    warm_session.close()

    return {
        "backend": loaded_cold.backend,
        "selected_labels": {"preview": preview_label, "section": section_label},
        "crop_levels": {"preview": preview_crop_level, "section": section_crop_level},
        "cold_start": {
            "load_ndpi_sec": cold_load_sec,
            "load_ndpi_cached_backend_sec": cached_load_sec,
            "overview_proposal_sec": cold_prop_sec,
            "session_set_slide_sec": set_slide_sec,
            "preview_crop_direct_sec": preview_direct_sec,
            "preview_crop_cache_miss_sec": preview_cache_miss_sec,
            "preview_crop_cache_hit_sec": preview_cache_hit_sec,
            "section_crop_cache_miss_sec": section_cache_miss_sec,
            "section_crop_cache_hit_sec": section_cache_hit_sec,
            "mask_generation_sec": mask_sec,
        },
        "warm_start": {
            "load_ndpi_sec": warm_load_sec,
            "overview_proposal_sec": warm_prop_sec,
            "session_set_slide_sec": warm_set_slide_sec,
            "preview_crop_cache_miss_sec": warm_preview_cache_miss_sec,
            "section_crop_cache_miss_sec": warm_section_cache_miss_sec,
            "mask_generation_sec": warm_mask_sec,
        },
        "mask_shapes": {
            "tissue": list(tissue_mask.shape),
            "artifact": list(artifact_mask.shape),
        },
        "backend_hint_effect": {
            "initial_backend": loaded_cold.backend,
            "cached_backend": loaded_cached.backend,
            "cold_load_sec": cold_load_sec,
            "cached_load_sec": cached_load_sec,
        },
    }


def benchmark_quality_tradeoff(
    slide_path: Path,
    stain: str,
    gt_root: Path,
    quality_sections: list[str],
    quality_crop_levels: list[int],
) -> dict:
    loaded = load_slide_bundle(slide_path, stain)
    overview_rgb = np.asarray(loaded.overview)
    labels = parse_slide_labels(slide_path.stem)[1]
    proposals = propose_from_overview(slide_path, stain, labels, overview_rgb)
    proposal_map = {proposal.label: proposal for proposal in proposals}

    rows_by_level: dict[str, list[dict]] = {}
    for level in quality_crop_levels:
        session = SlideSessionCache()
        session.set_slide(loaded)
        rows = []
        for label in quality_sections:
            gt_item = read_gt_item(gt_root, label)
            proposal = proposal_map[label]
            crop_bbox_level0 = proposal_bbox_level0_gui(loaded, proposal)
            meta = gt_item["meta"]
            gt_old_crop_rect_ov = (
                int(meta["bbox_overview"]["x"]),
                int(meta["bbox_overview"]["y"]),
                int(meta["bbox_overview"]["x"] + meta["bbox_overview"]["w"]),
                int(meta["bbox_overview"]["y"] + meta["bbox_overview"]["h"]),
            )
            gt_old_crop_bbox_level0 = overview_rect_to_level0(loaded, gt_old_crop_rect_ov)

            crop_t, crop_rgb = time_once(lambda: session.get_section_crop(proposal, level))
            # Measure the actual GUI auto-mask path, not a tool-only crop path.
            mask_t, (tissue_mask, _) = time_once(
                lambda: compute_auto_masks(
                    crop_rgb,
                    stain,
                    loaded_slide=loaded,
                    target_proposal=proposal,
                    all_proposals=proposals,
                    crop_level=level,
                )
            )
            pred = tissue_mask > 0
            gt_new, crop_cov = project_gt_to_new_crop(gt_item["mask"], gt_old_crop_bbox_level0, crop_bbox_level0, crop_rgb.shape[:2])
            rm = region_metrics(pred, gt_new)
            cm = contour_metrics(pred, gt_new)
            rows.append(
                {
                    "section": label,
                    "crop_time_sec": crop_t,
                    "mask_time_sec": mask_t,
                    "crop_coverage_recall": crop_cov,
                    "dice": rm["dice"],
                    "iou": rm["iou"],
                    "boundary_f1_tol32": cm["boundary_f1_tol32"],
                    "boundary_f1_tol64": cm["boundary_f1_tol64"],
                }
            )
        session.close()
        rows_by_level[f"level_{level}"] = rows

    aggregate = {}
    for level_key, rows in rows_by_level.items():
        aggregate[level_key] = {
            "count": len(rows),
            "mean_crop_time_sec": float(np.mean([row["crop_time_sec"] for row in rows])),
            "mean_mask_time_sec": float(np.mean([row["mask_time_sec"] for row in rows])),
            "mean_crop_coverage_recall": float(np.mean([row["crop_coverage_recall"] for row in rows])),
            "mean_dice": float(np.mean([row["dice"] for row in rows])),
            "mean_iou": float(np.mean([row["iou"] for row in rows])),
            "mean_boundary_f1_tol32": float(np.mean([row["boundary_f1_tol32"] for row in rows])),
            "mean_boundary_f1_tol64": float(np.mean([row["boundary_f1_tol64"] for row in rows])),
        }
    return {"aggregate": aggregate, "per_section": rows_by_level}


def render_markdown(results: dict, output_json: Path) -> str:
    gui = results["gui_like_session"]
    quality = results["quality_tradeoff"]["aggregate"]
    lines = [
        "# Windows Timing Harness Run",
        "",
        f"- `python`: `{results['environment']['python_executable']}`",
        f"- `platform`: `{results['environment']['platform']}`",
        f"- `output_json`: `{output_json}`",
        f"- `slide`: `{results['inputs']['slide_path']}`",
        "",
        "## GUI-Like Session Timing",
        "",
        f"- backend: `{gui['backend']}`",
        f"- cold `load NDPI`: `{gui['cold_start']['load_ndpi_sec']:.3f}s`",
        f"- cached backend/proxy `load NDPI`: `{gui['cold_start']['load_ndpi_cached_backend_sec']:.3f}s`",
        f"- warm `load NDPI`: `{gui['warm_start']['load_ndpi_sec']:.3f}s`",
        f"- cold `overview proposal`: `{gui['cold_start']['overview_proposal_sec']:.3f}s`",
        f"- warm `overview proposal`: `{gui['warm_start']['overview_proposal_sec']:.3f}s`",
        f"- cold `preview crop direct`: `{gui['cold_start']['preview_crop_direct_sec']:.3f}s`",
        f"- cold `preview crop cache miss`: `{gui['cold_start']['preview_crop_cache_miss_sec']:.3f}s`",
        f"- cold `preview crop cache hit`: `{gui['cold_start']['preview_crop_cache_hit_sec']:.6f}s`",
        f"- cold `section crop cache miss`: `{gui['cold_start']['section_crop_cache_miss_sec']:.3f}s`",
        f"- cold `section crop cache hit`: `{gui['cold_start']['section_crop_cache_hit_sec']:.6f}s`",
        f"- cold `mask generation`: `{gui['cold_start']['mask_generation_sec']:.3f}s`",
        "",
        "## Quality-Speed Tradeoff",
        "",
    ]
    for level_key in sorted(quality):
        agg = quality[level_key]
        lines.extend(
            [
                f"### {level_key}",
                f"- mean crop time: `{agg['mean_crop_time_sec']:.3f}s`",
                f"- mean mask time: `{agg['mean_mask_time_sec']:.3f}s`",
                f"- mean crop coverage recall: `{agg['mean_crop_coverage_recall']:.4f}`",
                f"- mean dice: `{agg['mean_dice']:.4f}`",
                f"- mean iou: `{agg['mean_iou']:.4f}`",
                f"- mean boundary_f1_tol32: `{agg['mean_boundary_f1_tol32']:.4f}`",
                f"- mean boundary_f1_tol64: `{agg['mean_boundary_f1_tol64']:.4f}`",
                "",
            ]
        )

    lines.extend(
        [
            "## Efficiency Notes",
            "",
            "- Benchmark in Windows Python when evaluating the GUI path; WSL `/mnt/d` timings are not representative.",
            "- Separate `cold start`, `warm start`, and `cache hit` so repeated runs are not confused with first-open cost.",
            "- Use `SlideSessionCache` and a persistent slide handle; avoid section-level re-open of NDPI.",
            "- Use lower crop levels for tuning and profiling first, then validate the best method at a higher level on a small GT subset.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--slide-path",
        default=r"D:\Research\Image Analysis\Nanozoomer scans\20250327 rat myelin quantification\gallyas_2502_42-72.ndpi",
    )
    parser.add_argument("--stain", default="gallyas")
    parser.add_argument("--preview-label", default="2502_42")
    parser.add_argument("--section-label", default="2502_48")
    parser.add_argument("--preview-crop-level", type=int, default=4)
    parser.add_argument("--section-crop-level", type=int, default=3)
    parser.add_argument(
        "--gt-root",
        default=r"D:\Research\Image Analysis\Nanozoomer scans\20250327 rat myelin quantification\Tissue&Masks",
    )
    parser.add_argument("--quality-sections", nargs="*", default=["2502_42", "2502_48", "2502_60"])
    parser.add_argument("--quality-crop-levels", nargs="*", type=int, default=[3, 4])
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_json = output_dir / "timing_results.json"
    output_md = output_dir / "timing_summary.md"

    gui_results = benchmark_gui_like_session(
        slide_path=Path(args.slide_path),
        stain=args.stain,
        preview_label=args.preview_label,
        section_label=args.section_label,
        preview_crop_level=args.preview_crop_level,
        section_crop_level=args.section_crop_level,
    )
    quality_results = benchmark_quality_tradeoff(
        slide_path=Path(args.slide_path),
        stain=args.stain,
        gt_root=Path(args.gt_root),
        quality_sections=list(args.quality_sections),
        quality_crop_levels=list(args.quality_crop_levels),
    )

    results = {
        "environment": {
            "python_executable": sys.executable,
            "python_version": sys.version,
            "platform": platform.platform(),
        },
        "inputs": {
            "slide_path": args.slide_path,
            "preview_label": args.preview_label,
            "section_label": args.section_label,
            "preview_crop_level": args.preview_crop_level,
            "section_crop_level": args.section_crop_level,
            "quality_sections": list(args.quality_sections),
            "quality_crop_levels": list(args.quality_crop_levels),
        },
        "gui_like_session": gui_results,
        "quality_tradeoff": quality_results,
    }
    output_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    output_md.write_text(render_markdown(results, output_json), encoding="utf-8")
    print(json.dumps(results, indent=2))
    print(f"\nWrote:\n- {output_json}\n- {output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

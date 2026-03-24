from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from time import perf_counter
from typing import Any

import cv2
import numpy as np
from scipy.ndimage import binary_fill_holes
from skimage.filters import threshold_otsu
from skimage.filters.rank import entropy as rank_entropy
from skimage.morphology import disk

from ..domain import LoadedSlide, ProposalBox
from .tool_bridge import load_histology_tool_module, proposal_bbox_level0_gui, proposal_to_tool_candidate

MASK_PRESET_LATEST_CONTEXTUAL = "latest_contextual"
MASK_PRESET_LEGACY_SIMPLE = "legacy_simple"
MASK_PRESET_HYBRID_BALANCED = "hybrid_balanced"
MASK_PRESET_M3_HYST_ENTRES_GUARD = "m3_hyst_entres_guard_v1"
MASK_COMPUTE_PROFILE_FULL = "full"
MASK_COMPUTE_PROFILE_STANDARD = "standard"
MASK_COMPUTE_PROFILE_FAST = "fast"
MASK_COMPUTE_PROFILE_MAX_LONG_EDGE = {
    MASK_COMPUTE_PROFILE_FULL: None,
    MASK_COMPUTE_PROFILE_STANDARD: 2048,
    MASK_COMPUTE_PROFILE_FAST: 1600,
}
MASK_COMPUTE_PROFILES = (
    MASK_COMPUTE_PROFILE_STANDARD,
    MASK_COMPUTE_PROFILE_FAST,
    MASK_COMPUTE_PROFILE_FULL,
)
MASK_PRESETS = (
    MASK_PRESET_M3_HYST_ENTRES_GUARD,
    MASK_PRESET_LATEST_CONTEXTUAL,
    MASK_PRESET_LEGACY_SIMPLE,
    MASK_PRESET_HYBRID_BALANCED,
)
PROPOSAL_CACHE_VERSION = "overview_proposal_v7"


def _persistent_cache_root() -> Path:
    local_appdata = os.environ.get("LOCALAPPDATA")
    if local_appdata:
        root = Path(local_appdata) / "histology_gui"
    else:
        root = Path.home() / ".cache" / "histology_gui"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _slide_identity(slide_path: Path) -> dict[str, object]:
    stat = slide_path.stat()
    return {
        "path": str(slide_path.resolve()),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(getattr(stat, "st_mtime_ns", int(stat.st_mtime * 1e9))),
    }


def _proposal_cache_path(slide_path: Path, stain: str) -> Path:
    ident = _slide_identity(slide_path)
    digest = hashlib.sha1(f"{ident['path']}|{stain}|{PROPOSAL_CACHE_VERSION}".encode("utf-8")).hexdigest()[:12]
    stem = slide_path.stem.replace(";", "_")
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in stem)
    root = _persistent_cache_root() / "proposal_cache_v1"
    root.mkdir(parents=True, exist_ok=True)
    return root / f"{safe}_{digest}.json"


def clear_proposal_cache(slide_path: Path, stain: str) -> None:
    path = _proposal_cache_path(slide_path, stain)
    if path.exists():
        path.unlink()


def default_mask_preset_for_stain(stain: str) -> str:
    return MASK_PRESET_M3_HYST_ENTRES_GUARD if stain.lower() == "gallyas" else MASK_PRESET_LATEST_CONTEXTUAL


def mask_compute_profile_max_long_edge(profile: str) -> int | None:
    return MASK_COMPUTE_PROFILE_MAX_LONG_EDGE.get(profile, MASK_COMPUTE_PROFILE_MAX_LONG_EDGE[MASK_COMPUTE_PROFILE_STANDARD])


def parse_slide_labels(slide_stem: str) -> tuple[str, list[Any]]:
    tool = load_histology_tool_module()
    return tool.parse_slide_stem(slide_stem)


def propose_from_overview(slide_path: Path, stain: str, labels: list[Any], overview_rgb: np.ndarray) -> list[ProposalBox]:
    cache_path = _proposal_cache_path(slide_path, stain)
    try:
        if cache_path.exists():
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            ident = _slide_identity(slide_path)
            if (
                cached.get("cache_version") == PROPOSAL_CACHE_VERSION
                and cached.get("stain") == stain
                and cached.get("source_slide_identity", {}) == ident
            ):
                out: list[ProposalBox] = []
                for item in cached.get("proposals", []):
                    out.append(
                        ProposalBox(
                            label=str(item["label"]),
                            stain=str(item["stain"]),
                            sample_id=str(item["sample_id"]),
                            section_id=int(item["section_id"]),
                            proposal_rank=int(item["proposal_rank"]),
                            x=int(item["x"]),
                            y=int(item["y"]),
                            w=int(item["w"]),
                            h=int(item["h"]),
                            mask_preset=default_mask_preset_for_stain(stain),
                        )
                    )
                if out:
                    return out
    except Exception:
        pass

    tool = load_histology_tool_module()
    _, _, component_mask = tool.component_mask_from_overview(overview_rgb, stain=stain)
    candidates = tool.find_candidate_components(component_mask, len(labels))
    candidates = tool.assign_sections(candidates, labels)
    gallyas_rects: list[tuple[int, int, int, int]] | None = None
    if stain.lower() == "gallyas":
        gallyas_rects = tool.gallyas_proposal_rects_overview(overview_rgb, candidates)
    proposals: list[ProposalBox] = []
    for idx, cand in enumerate(candidates, start=1):
        label = cand.section.short_label if getattr(cand, "section", None) else f"cand_{idx}"
        x = int(cand.x)
        y = int(cand.y)
        w = int(cand.w)
        h = int(cand.h)
        if stain.lower() == "gallyas":
            if gallyas_rects is not None and idx - 1 < len(gallyas_rects):
                x1, y1, x2, y2 = gallyas_rects[idx - 1]
            else:
                x1, y1, x2, y2 = tool.proposal_crop_rect_overview(cand, overview_rgb, stain, candidates)
            x = int(x1)
            y = int(y1)
            w = int(max(1, x2 - x1))
            h = int(max(1, y2 - y1))
        proposals.append(
            ProposalBox(
                label=label,
                stain=stain,
                sample_id=cand.section.sample_id if getattr(cand, "section", None) else "",
                section_id=cand.section.section_id if getattr(cand, "section", None) else idx,
                proposal_rank=idx,
                x=x,
                y=y,
                w=w,
                h=h,
                mask_preset=default_mask_preset_for_stain(stain),
            )
        )
    try:
        payload = {
            "cache_version": PROPOSAL_CACHE_VERSION,
            "stain": stain,
            "source_slide_identity": _slide_identity(slide_path),
            "proposals": [
                {
                    "label": p.label,
                    "stain": p.stain,
                    "sample_id": p.sample_id,
                    "section_id": int(p.section_id),
                    "proposal_rank": int(p.proposal_rank),
                    "x": int(p.x),
                    "y": int(p.y),
                    "w": int(p.w),
                    "h": int(p.h),
                }
                for p in proposals
            ],
        }
        tmp = cache_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp.replace(cache_path)
    except Exception:
        pass
    return proposals


def _simple_compute_auto_masks(crop_rgb: np.ndarray, stain: str) -> tuple[np.ndarray, np.ndarray]:
    if stain.lower() == "gallyas":
        gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)
        bg = cv2.GaussianBlur(gray, (0, 0), sigmaX=60, sigmaY=60)
        resid = np.clip(bg.astype(np.int16) - gray.astype(np.int16), 0, 255).astype(np.uint8)
        artifact = np.zeros_like(resid, dtype=np.uint8)
        th = int(threshold_otsu(resid))
        tissue = resid >= th
        tissue = cv2.morphologyEx(tissue.astype(np.uint8) * 255, cv2.MORPH_CLOSE, np.ones((31, 31), np.uint8)) > 0
        tissue = cv2.morphologyEx(tissue.astype(np.uint8) * 255, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8)) > 0
        tissue = binary_fill_holes(tissue)
        return (tissue.astype(np.uint8) * 255), artifact

    hsv = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2HSV)
    sat = hsv[:, :, 1].astype(np.uint8)
    nonwhite = (255 - crop_rgb.min(axis=2)).astype(np.uint8)
    score = np.maximum(sat, nonwhite)
    blur = cv2.GaussianBlur(score, (0, 0), sigmaX=25, sigmaY=25)
    th = int(threshold_otsu(blur))
    tissue = blur >= th
    tissue = cv2.morphologyEx(tissue.astype(np.uint8) * 255, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8)) > 0
    tissue = binary_fill_holes(tissue)
    artifact = np.zeros_like(score, dtype=np.uint8)
    return (tissue.astype(np.uint8) * 255), artifact


def _odd_kernel_size(value: int, minimum: int = 3) -> int:
    k = max(minimum, int(value))
    return k if (k % 2 == 1) else (k + 1)


def _scaled_int(value: int, scale: float, minimum: int = 1) -> int:
    safe_scale = max(0.2, min(1.0, float(scale)))
    return max(minimum, int(round(float(value) * safe_scale)))


def _scaled_odd_kernel(value: int, scale: float, minimum: int = 3) -> int:
    return _odd_kernel_size(_scaled_int(value, scale, minimum=minimum), minimum=minimum)


def _make_mask_slightly_conservative(mask_u8: np.ndarray, stain: str) -> np.ndarray:
    mask = mask_u8 > 0
    if not mask.any():
        return mask_u8.astype(np.uint8)

    h, w = mask.shape[:2]
    min_dim = min(h, w)
    if stain.lower() == "gallyas":
        open_k = _odd_kernel_size(int(round(min_dim * 0.0015)), minimum=3)
        erode_k = _odd_kernel_size(int(round(min_dim * 0.0018)), minimum=3)
        max_area_drop = 0.16
    else:
        open_k = _odd_kernel_size(int(round(min_dim * 0.0012)), minimum=3)
        erode_k = _odd_kernel_size(int(round(min_dim * 0.0014)), minimum=3)
        max_area_drop = 0.12

    original_area = int(mask.sum())
    refined = cv2.morphologyEx(mask.astype(np.uint8) * 255, cv2.MORPH_OPEN, np.ones((open_k, open_k), np.uint8)) > 0
    if refined.any():
        eroded = cv2.erode(refined.astype(np.uint8) * 255, np.ones((erode_k, erode_k), np.uint8), iterations=1) > 0
        if eroded.any() and int(eroded.sum()) >= int(round(original_area * (1.0 - max_area_drop))):
            refined = eroded
    if not refined.any():
        refined = mask
    refined = binary_fill_holes(refined)
    return refined.astype(np.uint8) * 255


def _tighten_with_area_guard(mask_u8: np.ndarray, *, open_k: int, erode_k: int, min_keep_frac: float) -> np.ndarray:
    mask = mask_u8 > 0
    if not mask.any():
        return mask_u8.astype(np.uint8)
    original_area = int(mask.sum())
    refined = cv2.morphologyEx(mask.astype(np.uint8) * 255, cv2.MORPH_OPEN, np.ones((open_k, open_k), np.uint8)) > 0
    if refined.any():
        eroded = cv2.erode(refined.astype(np.uint8) * 255, np.ones((erode_k, erode_k), np.uint8), iterations=1) > 0
        if eroded.any() and int(eroded.sum()) >= int(round(original_area * min_keep_frac)):
            refined = eroded
    if not refined.any():
        refined = mask
    refined = binary_fill_holes(refined)
    return refined.astype(np.uint8) * 255


def _fallback_if_too_small(mask: np.ndarray, fallback: np.ndarray, *, min_frac_of_fallback: float) -> np.ndarray:
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


def _crop_bbox_level0_from_proposal(loaded_slide: LoadedSlide, proposal: ProposalBox) -> tuple[int, int, int, int]:
    return proposal_bbox_level0_gui(loaded_slide, proposal)


def _retain_core_overlapping_components(mask: np.ndarray, core: np.ndarray, *, overlap_frac: float = 0.03) -> np.ndarray:
    mask_u8 = mask.astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, 8)
    if num <= 1:
        return mask
    core = core.astype(bool)
    overlap_counts = np.bincount(labels[core].ravel(), minlength=num)
    areas = stats[:, cv2.CC_STAT_AREA].astype(np.int32)
    thresholds = np.maximum(16, np.round(areas * float(overlap_frac)).astype(np.int32))
    keep_ids = np.flatnonzero((np.arange(num) > 0) & (overlap_counts >= thresholds))
    if keep_ids.size == 0:
        return np.zeros_like(mask, dtype=bool)
    keep_lut = np.zeros(num, dtype=bool)
    keep_lut[keep_ids] = True
    return keep_lut[labels]


def _mask_tight_roi(mask: np.ndarray, *, margin: int = 0) -> tuple[int, int, int, int] | None:
    mask = mask.astype(bool)
    if not mask.any():
        return None
    ys, xs = np.nonzero(mask)
    h, w = mask.shape[:2]
    y1 = max(0, int(ys.min()) - margin)
    y2 = min(h, int(ys.max()) + 1 + margin)
    x1 = max(0, int(xs.min()) - margin)
    x2 = min(w, int(xs.max()) + 1 + margin)
    if y2 <= y1 or x2 <= x1:
        return None
    return y1, y2, x1, x2


def _connected_seed_reconstruct(
    mask: np.ndarray,
    seed: np.ndarray,
    *,
    connectivity: int = 4,
    roi_margin: int = 0,
) -> tuple[np.ndarray, dict[str, float]]:
    timings = {
        "connected_components": 0.0,
        "seed_labels": 0.0,
        "isin_keep": 0.0,
    }
    mask = mask.astype(bool)
    if not mask.any():
        return mask, timings
    seed = seed.astype(bool) & mask
    if not seed.any():
        return mask, timings
    roi = _mask_tight_roi(mask | seed, margin=roi_margin)
    if roi is None:
        return mask, timings
    y1, y2, x1, x2 = roi
    mask_roi = mask[y1:y2, x1:x2]
    seed_roi = seed[y1:y2, x1:x2]
    stage_t0 = perf_counter()
    num, labels = cv2.connectedComponents(mask_roi.astype(np.uint8), connectivity=connectivity)
    timings["connected_components"] = round(perf_counter() - stage_t0, 4)
    if num <= 2:
        return mask, timings
    stage_t0 = perf_counter()
    seed_labels = np.unique(labels[seed_roi])
    seed_labels = seed_labels[seed_labels > 0]
    timings["seed_labels"] = round(perf_counter() - stage_t0, 4)
    if seed_labels.size == 0:
        return mask, timings
    stage_t0 = perf_counter()
    keep_roi = np.isin(labels, seed_labels)
    keep = np.zeros_like(mask, dtype=bool)
    keep[y1:y2, x1:x2] = keep_roi
    timings["isin_keep"] = round(perf_counter() - stage_t0, 4)
    return keep, timings


def _hybrid_reconstruct(
    candidate_mask: np.ndarray,
    core_mask: np.ndarray,
    *,
    erode_k: int = 7,
    core_dilate_k: int = 21,
    overlap_frac: float = 0.03,
    final_close_k: int = 9,
) -> np.ndarray:
    candidate_mask = candidate_mask.astype(bool)
    core_mask = core_mask.astype(bool)
    if not candidate_mask.any():
        return candidate_mask
    core_dil = cv2.dilate(core_mask.astype(np.uint8) * 255, np.ones((core_dilate_k, core_dilate_k), np.uint8), iterations=1) > 0
    shrunken = cv2.erode(candidate_mask.astype(np.uint8) * 255, np.ones((erode_k, erode_k), np.uint8), iterations=1) > 0
    seed = _retain_core_overlapping_components(shrunken, core_dil, overlap_frac=overlap_frac)
    if not seed.any():
        seed = core_mask & candidate_mask
    if not seed.any():
        seed = core_mask
    if not seed.any():
        seed = candidate_mask
    recon, _ = _connected_seed_reconstruct(
        candidate_mask,
        seed,
        connectivity=4,
        roi_margin=max(2, final_close_k),
    )
    recon = cv2.morphologyEx(
        recon.astype(np.uint8) * 255,
        cv2.MORPH_CLOSE,
        np.ones((final_close_k, final_close_k), np.uint8),
    ) > 0
    recon = binary_fill_holes(recon)
    return recon


def _crop_center_gallyas_masks(crop_rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    tool = load_histology_tool_module()
    h, w = crop_rgb.shape[:2]
    support = np.ones((h, w), dtype=bool)
    center = (float(w) / 2.0, float(h) / 2.0)
    result = tool.build_crop_mask_baseline(
        crop_rgb,
        ownership_strict=support,
        ownership_soft=support,
        support_mask=support,
        target_center_px=center,
        stain="gallyas",
        gallyas_max_components=2,
    )
    return result["mask"].astype(np.uint8), result["artifact"].astype(np.uint8)


def _nissl_tool_baseline_masks(crop_rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    tool = load_histology_tool_module()
    h, w = crop_rgb.shape[:2]
    support = np.ones((h, w), dtype=bool)
    center = (float(w) / 2.0, float(h) / 2.0)
    result = tool.build_crop_mask_baseline(
        crop_rgb,
        ownership_strict=support,
        ownership_soft=support,
        support_mask=support,
        target_center_px=center,
        stain="nissl",
    )
    return result["mask"].astype(np.uint8), result["artifact"].astype(np.uint8)


def _hybrid_balanced_gallyas_masks(crop_rgb: np.ndarray, *, input_scale: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    tighten_k = _scaled_odd_kernel(3, input_scale, minimum=3)
    hybrid_erode_k = _scaled_odd_kernel(7, input_scale, minimum=3)
    core_dilate_k = _scaled_odd_kernel(21, input_scale, minimum=5)
    final_close_k = _scaled_odd_kernel(9, input_scale, minimum=3)
    candidate_u8, _ = _simple_compute_auto_masks(crop_rgb, "gallyas")
    candidate = _tighten_with_area_guard(candidate_u8, open_k=tighten_k, erode_k=tighten_k, min_keep_frac=0.90) > 0
    core_u8, artifact = _crop_center_gallyas_masks(crop_rgb)
    core = core_u8 > 0
    hybrid = _hybrid_reconstruct(
        candidate,
        core,
        erode_k=hybrid_erode_k,
        core_dilate_k=core_dilate_k,
        overlap_frac=0.03,
        final_close_k=final_close_k,
    )
    return hybrid.astype(np.uint8) * 255, artifact


def _residual_score(crop_rgb: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)
    sigma = max(12, int(round(min(crop_rgb.shape[:2]) * 0.05)))
    bg = cv2.GaussianBlur(gray, (0, 0), sigmaX=float(sigma), sigmaY=float(sigma))
    return np.clip(bg.astype(np.int16) - gray.astype(np.int16), 0, 255).astype(np.uint8)


def _entropy_score(crop_rgb: np.ndarray, radius: int = 5) -> np.ndarray:
    gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)
    ent = rank_entropy(gray, disk(max(1, int(radius))))
    ent = cv2.GaussianBlur(ent.astype(np.float32), (0, 0), sigmaX=1.0, sigmaY=1.0)
    if float(ent.max()) <= 1e-6:
        return np.zeros_like(gray, dtype=np.uint8)
    return cv2.normalize(ent, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def _entropy_residual_candidate(
    crop_rgb: np.ndarray,
    *,
    ent_radius: int,
    ent_q: float,
    residual_scale: float,
    nonwhite_min: int,
    close_k: int,
    open_k: int,
) -> np.ndarray:
    resid = _residual_score(crop_rgb)
    ent = _entropy_score(crop_rgb, radius=ent_radius)
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


def _hysteresis_support_reconstruct(
    support_mask: np.ndarray,
    core_score: np.ndarray,
    structural_core: np.ndarray,
    *,
    core_quantile: float,
    core_scale: float,
    overlap_frac: float,
    core_open_k: int,
    final_close_k: int,
) -> tuple[np.ndarray, dict[str, float]]:
    timings = {
        "strong_core_prepare": 0.0,
        "seed_select": 0.0,
        "connected_components": 0.0,
        "seed_labels": 0.0,
        "isin_keep": 0.0,
        "final_close": 0.0,
        "fill_holes": 0.0,
    }
    support_mask = support_mask.astype(bool)
    structural_core = structural_core.astype(bool)
    if not support_mask.any():
        return support_mask, timings

    vals = core_score[support_mask]
    if vals.size == 0:
        return support_mask, timings
    stage_t0 = perf_counter()
    core_th = max(6, int(round(float(np.quantile(vals, core_quantile)) * core_scale)))
    strong_core = (core_score >= core_th) & support_mask
    if core_open_k > 1:
        strong_core = cv2.morphologyEx(
            strong_core.astype(np.uint8) * 255,
            cv2.MORPH_OPEN,
            np.ones((core_open_k, core_open_k), np.uint8),
        ) > 0
    timings["strong_core_prepare"] = round(perf_counter() - stage_t0, 4)
    stage_t0 = perf_counter()
    if strong_core.any():
        struct_dil = cv2.dilate(structural_core.astype(np.uint8) * 255, np.ones((9, 9), np.uint8), iterations=1) > 0
        seed = _retain_core_overlapping_components(strong_core, struct_dil, overlap_frac=overlap_frac)
    else:
        seed = np.zeros_like(support_mask, dtype=bool)
    if not seed.any():
        seed = structural_core & support_mask
    if not seed.any():
        seed = support_mask
    timings["seed_select"] = round(perf_counter() - stage_t0, 4)
    recon, cc_timings = _connected_seed_reconstruct(
        support_mask,
        seed,
        connectivity=4,
        roi_margin=max(2, final_close_k),
    )
    timings.update(cc_timings)
    if final_close_k > 1:
        stage_t0 = perf_counter()
        recon = cv2.morphologyEx(
            recon.astype(np.uint8) * 255,
            cv2.MORPH_CLOSE,
            np.ones((final_close_k, final_close_k), np.uint8),
        ) > 0
        timings["final_close"] = round(perf_counter() - stage_t0, 4)
    stage_t0 = perf_counter()
    recon = binary_fill_holes(recon)
    timings["fill_holes"] = round(perf_counter() - stage_t0, 4)
    return recon, timings


def _m3_hyst_entres_guard_gallyas_masks_with_info(
    crop_rgb: np.ndarray,
    *,
    input_scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    total_t0 = perf_counter()
    support_ent_radius = _scaled_int(7, input_scale, minimum=2)
    support_close_k = _scaled_odd_kernel(11, input_scale, minimum=3)
    support_open_k = _scaled_odd_kernel(3, input_scale, minimum=3)
    core_open_k = _scaled_odd_kernel(3, input_scale, minimum=1)
    guard_close_k = _scaled_odd_kernel(7, input_scale, minimum=3)
    fallback_ent_radius = _scaled_int(5, input_scale, minimum=2)
    fallback_close_k = _scaled_odd_kernel(9, input_scale, minimum=3)
    fallback_open_k = _scaled_odd_kernel(3, input_scale, minimum=3)
    fallback_tighten_k = _scaled_odd_kernel(3, input_scale, minimum=3)
    fallback_erode_k = _scaled_odd_kernel(7, input_scale, minimum=3)
    fallback_core_dilate_k = _scaled_odd_kernel(21, input_scale, minimum=5)
    fallback_final_close_k = _scaled_odd_kernel(9, input_scale, minimum=3)
    timings: dict[str, float] = {}
    stage_t0 = perf_counter()
    support = _entropy_residual_candidate(
        crop_rgb,
        ent_radius=support_ent_radius,
        ent_q=0.64,
        residual_scale=0.92,
        nonwhite_min=16,
        close_k=support_close_k,
        open_k=support_open_k,
    )
    timings["support_candidate"] = round(perf_counter() - stage_t0, 4)

    stage_t0 = perf_counter()
    core_score = _residual_score(crop_rgb)
    timings["core_score"] = round(perf_counter() - stage_t0, 4)

    stage_t0 = perf_counter()
    structural_core_u8, artifact = _crop_center_gallyas_masks(crop_rgb)
    timings["crop_center"] = round(perf_counter() - stage_t0, 4)
    structural_core = structural_core_u8 > 0

    stage_t0 = perf_counter()
    recon, hyst_timings = _hysteresis_support_reconstruct(
        support,
        core_score,
        structural_core,
        core_quantile=0.84,
        core_scale=1.00,
        overlap_frac=0.03,
        core_open_k=core_open_k,
        final_close_k=guard_close_k,
    )
    timings["hysteresis_reconstruct"] = round(perf_counter() - stage_t0, 4)
    timings["hysteresis_strong_core_prepare"] = hyst_timings["strong_core_prepare"]
    timings["hysteresis_seed_select"] = hyst_timings["seed_select"]
    timings["hysteresis_connected_components"] = hyst_timings["connected_components"]
    timings["hysteresis_seed_labels"] = hyst_timings["seed_labels"]
    timings["hysteresis_isin_keep"] = hyst_timings["isin_keep"]
    timings["hysteresis_final_close"] = hyst_timings["final_close"]
    timings["hysteresis_fill_holes"] = hyst_timings["fill_holes"]
    recon_area = int(recon.sum())
    support_area = int(support.sum())
    structural_core_area = int(structural_core.sum())
    fallback_needed = (
        recon_area <= 0
        or (structural_core_area > 0 and recon_area < int(round(structural_core_area * 1.10)))
        or (support_area > 0 and recon_area < int(round(support_area * 0.22)))
    )
    timings["fallback_gate"] = 0.0
    fallback = recon
    if fallback_needed:
        stage_t0 = perf_counter()
        fallback_candidate_u8 = _entropy_residual_candidate(
            crop_rgb,
            ent_radius=fallback_ent_radius,
            ent_q=0.68,
            residual_scale=0.96,
            nonwhite_min=18,
            close_k=fallback_close_k,
            open_k=fallback_open_k,
        ).astype(np.uint8) * 255
        timings["fallback_candidate"] = round(perf_counter() - stage_t0, 4)

        stage_t0 = perf_counter()
        fallback_candidate = _tighten_with_area_guard(
            fallback_candidate_u8,
            open_k=fallback_tighten_k,
            erode_k=fallback_tighten_k,
            min_keep_frac=0.92,
        ) > 0
        timings["fallback_tighten"] = round(perf_counter() - stage_t0, 4)

        stage_t0 = perf_counter()
        fallback = _hybrid_reconstruct(
            fallback_candidate,
            structural_core,
            erode_k=fallback_erode_k,
            core_dilate_k=fallback_core_dilate_k,
            overlap_frac=0.03,
            final_close_k=fallback_final_close_k,
        )
        timings["fallback_reconstruct"] = round(perf_counter() - stage_t0, 4)

        stage_t0 = perf_counter()
        tissue = _fallback_if_too_small(recon, fallback, min_frac_of_fallback=0.70)
        timings["fallback_decision"] = round(perf_counter() - stage_t0, 4)
    else:
        tissue = recon
        timings["fallback_candidate"] = 0.0
        timings["fallback_tighten"] = 0.0
        timings["fallback_reconstruct"] = 0.0
        timings["fallback_decision"] = 0.0
    timings["total_internal"] = round(perf_counter() - total_t0, 4)
    return (
        tissue.astype(np.uint8) * 255,
        artifact,
        {
            "internal_timing_s": timings,
            "internal_debug": {
                "support_ent_radius": int(support_ent_radius),
                "support_close_k": int(support_close_k),
                "support_open_k": int(support_open_k),
                "core_open_k": int(core_open_k),
                "guard_close_k": int(guard_close_k),
                "fallback_ent_radius": int(fallback_ent_radius),
                "fallback_close_k": int(fallback_close_k),
                "fallback_open_k": int(fallback_open_k),
                "fallback_tighten_k": int(fallback_tighten_k),
                "fallback_erode_k": int(fallback_erode_k),
                "fallback_core_dilate_k": int(fallback_core_dilate_k),
                "fallback_final_close_k": int(fallback_final_close_k),
                "working_shape_hw": [int(crop_rgb.shape[0]), int(crop_rgb.shape[1])],
                "input_scale": float(input_scale),
                "fallback_needed": bool(fallback_needed),
                "recon_area": int(recon_area),
                "support_area": int(support_area),
                "structural_core_area": int(structural_core_area),
            },
        },
    )


def _m3_hyst_entres_guard_gallyas_masks(crop_rgb: np.ndarray, *, input_scale: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    tissue, artifact, _ = _m3_hyst_entres_guard_gallyas_masks_with_info(crop_rgb, input_scale=input_scale)
    return tissue, artifact


def _prepare_mask_compute_input(
    crop_rgb: np.ndarray,
    *,
    compute_profile: str,
) -> tuple[np.ndarray, dict[str, object]]:
    profile = compute_profile if compute_profile in MASK_COMPUTE_PROFILE_MAX_LONG_EDGE else MASK_COMPUTE_PROFILE_STANDARD
    max_long_edge = mask_compute_profile_max_long_edge(profile)
    orig_h, orig_w = crop_rgb.shape[:2]
    long_edge = max(orig_h, orig_w)
    if max_long_edge is None or long_edge <= max_long_edge:
        return crop_rgb, {
            "compute_profile": profile,
            "working_max_long_edge": max_long_edge,
            "working_scale": 1.0,
            "original_shape_hw": [int(orig_h), int(orig_w)],
            "working_shape_hw": [int(orig_h), int(orig_w)],
        }
    scale = float(max_long_edge) / float(long_edge)
    out_w = max(1, int(round(orig_w * scale)))
    out_h = max(1, int(round(orig_h * scale)))
    working_rgb = cv2.resize(crop_rgb, (out_w, out_h), interpolation=cv2.INTER_AREA)
    return working_rgb, {
        "compute_profile": profile,
        "working_max_long_edge": int(max_long_edge),
        "working_scale": float(scale),
        "original_shape_hw": [int(orig_h), int(orig_w)],
        "working_shape_hw": [int(out_h), int(out_w)],
    }


def _contextual_gallyas_masks(
    crop_rgb: np.ndarray,
    loaded_slide: LoadedSlide,
    target_proposal: ProposalBox,
    all_proposals: list[ProposalBox],
    crop_level: int,
) -> tuple[np.ndarray, np.ndarray]:
    tool = load_histology_tool_module()
    overview_downsample = float(loaded_slide.level_downsamples[loaded_slide.overview_level])
    crop_level = min(crop_level, len(loaded_slide.level_downsamples) - 1)
    crop_downsample = float(loaded_slide.level_downsamples[crop_level])
    all_candidates = [proposal_to_tool_candidate(proposal, rank=idx + 1) for idx, proposal in enumerate(all_proposals)]
    target_rank = all_proposals.index(target_proposal) + 1
    target_candidate = proposal_to_tool_candidate(target_proposal, rank=target_rank)
    crop_bbox_level0 = _crop_bbox_level0_from_proposal(loaded_slide, target_proposal)
    ownership_strict, ownership_soft, support_mask = tool.build_crop_ownership_masks(
        target_candidate=target_candidate,
        all_candidates=all_candidates,
        crop_bbox_level0=crop_bbox_level0,
        crop_shape=crop_rgb.shape[:2],
        crop_downsample=crop_downsample,
        overview_downsample=overview_downsample,
    )
    target_center_px = tool.level0_point_to_crop(
        tool.candidate_center_level0(target_candidate, overview_downsample),
        crop_bbox_level0=crop_bbox_level0,
        crop_downsample=crop_downsample,
    )
    result = tool.build_crop_mask_baseline(
        crop_rgb,
        ownership_strict=ownership_strict,
        ownership_soft=ownership_soft,
        support_mask=support_mask,
        target_center_px=target_center_px,
        stain="gallyas",
        gallyas_support_soft_frac=0.016,
        gallyas_candidate_thresh_scale=1.05,
        gallyas_grow_quantile=0.22,
        gallyas_grow_scale=0.86,
    )
    tissue = _make_mask_slightly_conservative(result["mask"].astype(np.uint8), "gallyas")
    return tissue, result["artifact"].astype(np.uint8)


def compute_auto_masks_with_info(
    crop_rgb: np.ndarray,
    stain: str,
    *,
    method: str = MASK_PRESET_LATEST_CONTEXTUAL,
    loaded_slide: LoadedSlide | None = None,
    target_proposal: ProposalBox | None = None,
    all_proposals: list[ProposalBox] | None = None,
    crop_level: int = 3,
    input_scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    if method == MASK_PRESET_M3_HYST_ENTRES_GUARD and stain.lower() == "gallyas":
        return _m3_hyst_entres_guard_gallyas_masks_with_info(crop_rgb, input_scale=input_scale)

    if method == MASK_PRESET_HYBRID_BALANCED and stain.lower() == "gallyas":
        tissue, artifact = _hybrid_balanced_gallyas_masks(crop_rgb, input_scale=input_scale)
        return tissue, artifact, {}

    if method == MASK_PRESET_LEGACY_SIMPLE:
        tissue, artifact = _simple_compute_auto_masks(crop_rgb, stain)
        return tissue, artifact, {}

    if stain.lower() == "nissl":
        tissue, artifact = _nissl_tool_baseline_masks(crop_rgb)
        return tissue, artifact, {}

    if (
        stain.lower() == "gallyas"
        and loaded_slide is not None
        and target_proposal is not None
        and all_proposals is not None
    ):
        try:
            tissue, artifact = _contextual_gallyas_masks(
                crop_rgb,
                loaded_slide=loaded_slide,
                target_proposal=target_proposal,
                all_proposals=all_proposals,
                crop_level=crop_level,
            )
            return tissue, artifact, {}
        except Exception:
            pass
    tissue, artifact = _simple_compute_auto_masks(crop_rgb, stain)
    tissue = _make_mask_slightly_conservative(tissue, stain)
    return tissue, artifact, {}


def compute_auto_masks(
    crop_rgb: np.ndarray,
    stain: str,
    *,
    method: str = MASK_PRESET_LATEST_CONTEXTUAL,
    loaded_slide: LoadedSlide | None = None,
    target_proposal: ProposalBox | None = None,
    all_proposals: list[ProposalBox] | None = None,
    crop_level: int = 3,
    input_scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    tissue, artifact, _ = compute_auto_masks_with_info(
        crop_rgb,
        stain,
        method=method,
        loaded_slide=loaded_slide,
        target_proposal=target_proposal,
        all_proposals=all_proposals,
        crop_level=crop_level,
        input_scale=input_scale,
    )
    return tissue, artifact


def compute_auto_masks_resampled(
    crop_rgb: np.ndarray,
    stain: str,
    *,
    method: str = MASK_PRESET_LATEST_CONTEXTUAL,
    loaded_slide: LoadedSlide | None = None,
    target_proposal: ProposalBox | None = None,
    all_proposals: list[ProposalBox] | None = None,
    crop_level: int = 3,
    compute_profile: str = MASK_COMPUTE_PROFILE_STANDARD,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    working_rgb, info = _prepare_mask_compute_input(crop_rgb, compute_profile=compute_profile)
    tissue, artifact, algo_info = compute_auto_masks_with_info(
        working_rgb,
        stain,
        method=method,
        loaded_slide=loaded_slide,
        target_proposal=target_proposal,
        all_proposals=all_proposals,
        crop_level=crop_level,
        input_scale=float(info["working_scale"]),
    )
    if algo_info:
        info.update(dict(algo_info))
    orig_h, orig_w = crop_rgb.shape[:2]
    if working_rgb.shape[:2] != crop_rgb.shape[:2]:
        tissue = cv2.resize(tissue.astype(np.uint8), (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        artifact = cv2.resize(artifact.astype(np.uint8), (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    return tissue.astype(np.uint8), artifact.astype(np.uint8), info


def build_export_payload(
    crop_rgb: np.ndarray,
    tissue_mask: np.ndarray,
    artifact_mask: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    tissue = tissue_mask > 0
    if artifact_mask is None:
        artifact = np.zeros(tissue.shape, dtype=bool)
    else:
        artifact = artifact_mask > 0
    artifact &= ~tissue

    mask_labels = np.zeros(tissue.shape, dtype=np.uint8)
    mask_labels[tissue] = 1
    mask_labels[artifact] = 2

    preview = np.zeros((*tissue.shape, 3), dtype=np.uint8)
    preview[:, :, :] = np.array([24, 24, 24], dtype=np.uint8)
    preview[tissue] = np.array([255, 0, 0], dtype=np.uint8)
    preview[artifact] = np.array([0, 255, 255], dtype=np.uint8)

    rgba = np.dstack([crop_rgb, (tissue.astype(np.uint8) * 255)])
    return {
        "mask_labels": mask_labels,
        "mask_preview": preview.astype(np.uint8),
        "foreground_rgba": rgba.astype(np.uint8),
    }

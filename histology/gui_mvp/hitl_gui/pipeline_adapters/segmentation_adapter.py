from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scipy.ndimage import binary_fill_holes, binary_propagation
from skimage.filters import threshold_otsu

from ..domain import LoadedSlide, ProposalBox
from .tool_bridge import load_histology_tool_module, proposal_bbox_level0_gui, proposal_to_tool_candidate

MASK_PRESET_LATEST_CONTEXTUAL = "latest_contextual"
MASK_PRESET_LEGACY_SIMPLE = "legacy_simple"
MASK_PRESET_HYBRID_BALANCED = "hybrid_balanced"
MASK_PRESETS = (
    MASK_PRESET_LATEST_CONTEXTUAL,
    MASK_PRESET_LEGACY_SIMPLE,
    MASK_PRESET_HYBRID_BALANCED,
)
PROPOSAL_CACHE_VERSION = "overview_proposal_v4"


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
    return MASK_PRESET_HYBRID_BALANCED if stain.lower() == "gallyas" else MASK_PRESET_LATEST_CONTEXTUAL


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
    proposals: list[ProposalBox] = []
    for idx, cand in enumerate(candidates, start=1):
        label = cand.section.short_label if getattr(cand, "section", None) else f"cand_{idx}"
        x = int(cand.x)
        y = int(cand.y)
        w = int(cand.w)
        h = int(cand.h)
        if stain.lower() == "gallyas":
            x1, y1, x2, y2 = tool.proposal_crop_rect_overview(cand, overview_rgb, stain)
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


def _crop_bbox_level0_from_proposal(loaded_slide: LoadedSlide, proposal: ProposalBox) -> tuple[int, int, int, int]:
    return proposal_bbox_level0_gui(loaded_slide, proposal)


def _retain_core_overlapping_components(mask: np.ndarray, core: np.ndarray, *, overlap_frac: float = 0.03) -> np.ndarray:
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
    recon = binary_propagation(seed, mask=candidate_mask)
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


def _hybrid_balanced_gallyas_masks(crop_rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    candidate_u8, _ = _simple_compute_auto_masks(crop_rgb, "gallyas")
    candidate = _tighten_with_area_guard(candidate_u8, open_k=3, erode_k=3, min_keep_frac=0.90) > 0
    core_u8, artifact = _crop_center_gallyas_masks(crop_rgb)
    core = core_u8 > 0
    hybrid = _hybrid_reconstruct(candidate, core, erode_k=7, core_dilate_k=21, overlap_frac=0.03, final_close_k=9)
    return hybrid.astype(np.uint8) * 255, artifact


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


def compute_auto_masks(
    crop_rgb: np.ndarray,
    stain: str,
    *,
    method: str = MASK_PRESET_LATEST_CONTEXTUAL,
    loaded_slide: LoadedSlide | None = None,
    target_proposal: ProposalBox | None = None,
    all_proposals: list[ProposalBox] | None = None,
    crop_level: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    if method == MASK_PRESET_HYBRID_BALANCED and stain.lower() == "gallyas":
        return _hybrid_balanced_gallyas_masks(crop_rgb)

    if method == MASK_PRESET_LEGACY_SIMPLE:
        return _simple_compute_auto_masks(crop_rgb, stain)

    if stain.lower() == "nissl":
        return _nissl_tool_baseline_masks(crop_rgb)

    if (
        stain.lower() == "gallyas"
        and loaded_slide is not None
        and target_proposal is not None
        and all_proposals is not None
    ):
        try:
            return _contextual_gallyas_masks(
                crop_rgb,
                loaded_slide=loaded_slide,
                target_proposal=target_proposal,
                all_proposals=all_proposals,
                crop_level=crop_level,
            )
        except Exception:
            pass
    tissue, artifact = _simple_compute_auto_masks(crop_rgb, stain)
    tissue = _make_mask_slightly_conservative(tissue, stain)
    return tissue, artifact


def build_export_payload(crop_rgb: np.ndarray, tissue_mask: np.ndarray, artifact_mask: np.ndarray) -> dict[str, np.ndarray]:
    tissue = tissue_mask > 0
    artifact = artifact_mask > 0
    usable = tissue & ~artifact

    rgba = np.dstack([crop_rgb, (usable.astype(np.uint8) * 255)])
    return {
        "tissue_mask_final": tissue_mask.astype(np.uint8),
        "artifact_mask_final": artifact_mask.astype(np.uint8),
        "usable_tissue_mask": (usable.astype(np.uint8) * 255),
        "foreground_rgba": rgba.astype(np.uint8),
    }

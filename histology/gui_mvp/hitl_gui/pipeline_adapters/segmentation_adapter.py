from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scipy.ndimage import binary_fill_holes
from skimage.filters import threshold_otsu

from ..domain import ProposalBox


def _load_histology_tool_module() -> Any:
    here = Path(__file__).resolve()
    tool_path = here.parents[3] / "tools" / "run_ndpi_review_experiment.py"
    spec = importlib.util.spec_from_file_location("histology_ndpi_tool", tool_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load histology tool module from {tool_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_slide_labels(slide_stem: str) -> tuple[str, list[Any]]:
    tool = _load_histology_tool_module()
    return tool.parse_slide_stem(slide_stem)


def propose_from_overview(slide_path: Path, stain: str, labels: list[Any], overview_rgb: np.ndarray) -> list[ProposalBox]:
    tool = _load_histology_tool_module()
    _, _, component_mask = tool.component_mask_from_overview(overview_rgb, stain=stain)
    candidates = tool.find_candidate_components(component_mask, len(labels))
    candidates = tool.assign_sections(candidates, labels)
    proposals: list[ProposalBox] = []
    for idx, cand in enumerate(candidates, start=1):
        label = cand.section.short_label if getattr(cand, "section", None) else f"cand_{idx}"
        proposals.append(
            ProposalBox(
                label=label,
                stain=stain,
                sample_id=cand.section.sample_id if getattr(cand, "section", None) else "",
                section_id=cand.section.section_id if getattr(cand, "section", None) else idx,
                proposal_rank=idx,
                x=int(cand.x),
                y=int(cand.y),
                w=int(cand.w),
                h=int(cand.h),
            )
        )
    return proposals


def compute_auto_masks(crop_rgb: np.ndarray, stain: str) -> tuple[np.ndarray, np.ndarray]:
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


def build_export_payload(crop_rgb: np.ndarray, tissue_mask: np.ndarray, artifact_mask: np.ndarray) -> dict[str, np.ndarray]:
    tissue = tissue_mask > 0
    artifact = artifact_mask > 0
    usable = tissue & ~artifact

    rgba = np.dstack([crop_rgb, (usable.astype(np.uint8) * 255)])
    white = crop_rgb.copy()
    white[~usable] = 255
    black = crop_rgb.copy()
    black[~usable] = 0
    return {
        "tissue_mask_final": tissue_mask.astype(np.uint8),
        "artifact_mask_final": artifact_mask.astype(np.uint8),
        "usable_tissue_mask": (usable.astype(np.uint8) * 255),
        "foreground_rgba": rgba.astype(np.uint8),
        "foreground_rgb_white": white.astype(np.uint8),
        "foreground_rgb_black": black.astype(np.uint8),
    }

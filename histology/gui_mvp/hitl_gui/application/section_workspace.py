from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from time import perf_counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from ..pipeline_adapters import build_export_payload
from ..pipeline_adapters.slide_io import write_png_lossless_fast


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _read_png_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def _read_png_gray(path: Path, *, shape_hw: tuple[int, int]) -> np.ndarray:
    if not path.exists():
        return np.zeros(shape_hw, dtype=np.uint8)
    mask = np.asarray(Image.open(path).convert("L"))
    if mask.shape[:2] == shape_hw:
        return mask.astype(np.uint8)
    resized = cv2.resize(mask.astype(np.uint8), (shape_hw[1], shape_hw[0]), interpolation=cv2.INTER_NEAREST)
    return resized.astype(np.uint8)


def _metadata_original_shape_hw(metadata: dict[str, Any], item: "WorkspaceSection") -> tuple[int, int]:
    canvas = metadata.get("export_canvas") or {}
    width = int(canvas.get("width_px") or 0)
    height = int(canvas.get("height_px") or 0)
    if width > 0 and height > 0:
        return height, width
    with Image.open(item.crop_path) as im:
        return int(im.height), int(im.width)


def mask_labels_from_masks(tissue_mask: np.ndarray, artifact_mask: np.ndarray) -> np.ndarray:
    tissue = tissue_mask > 0
    artifact = (artifact_mask > 0) & ~tissue
    mask_labels = np.zeros(tissue.shape, dtype=np.uint8)
    mask_labels[tissue] = 1
    mask_labels[artifact] = 2
    return mask_labels


def load_masks_from_label_path(path: Path, *, shape_hw: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    labels = _read_png_gray(path, shape_hw=shape_hw)
    tissue = np.where(labels == 1, 255, 0).astype(np.uint8)
    artifact = np.where(labels == 2, 255, 0).astype(np.uint8)
    return tissue, artifact


def write_mask_labels_file(path: Path, tissue_mask: np.ndarray, artifact_mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_png_lossless_fast(path, mask_labels_from_masks(tissue_mask, artifact_mask), "L")


def _work_image_name(profile: str) -> str:
    return f"crop_work_{profile}.png"


def _backup_existing_outputs(section_dir: Path) -> Path | None:
    tracked = [
        "mask_labels.png",
        "mask_preview.png",
        "tissue_mask_final.png",
        "artifact_mask_final.png",
        "usable_tissue_mask.png",
        "foreground_rgba.png",
        "foreground_rgbwhite.png",
        "foreground_rgbblack.png",
        "foreground_rgb_white.png",
        "foreground_rgb_black.png",
        "metadata.json",
    ]
    existing = [section_dir / name for name in tracked if (section_dir / name).exists()]
    if not existing:
        return None
    rev_root = section_dir / "_mask_revisions"
    rev_root.mkdir(parents=True, exist_ok=True)
    backup_dir = rev_root / datetime.now().strftime("rev_%Y%m%d_%H%M%S")
    backup_dir.mkdir(parents=True, exist_ok=True)
    for src in existing:
        src.replace(backup_dir / src.name)
    return backup_dir


@dataclass
class WorkspaceSection:
    section_dir: Path
    label: str
    stain: str
    metadata_path: Path
    crop_path: Path
    has_masks: bool
    has_prepared_work: bool
    prepared_work_profiles: tuple[str, ...]


def external_mask_candidate_names(item: WorkspaceSection) -> list[str]:
    candidates: list[str] = []
    if item.stain == "nissl":
        candidates.append(f"nissl_{item.label}.png")
    else:
        candidates.append(f"myelin_{item.label}.png")
    candidates.append(f"{item.label}.png")
    return candidates


def find_external_prediction_mask(item: WorkspaceSection, mask_root: Path | None) -> Path | None:
    if mask_root is None or not mask_root.exists():
        return None
    for name in external_mask_candidate_names(item):
        candidate = mask_root / name
        if candidate.exists():
            return candidate
    return None


def list_workspace_sections(workspace_root: Path) -> list[WorkspaceSection]:
    items: list[WorkspaceSection] = []
    if not workspace_root.exists():
        return items
    for section_dir in sorted(p for p in workspace_root.iterdir() if p.is_dir()):
        metadata_path = section_dir / "metadata.json"
        crop_path = section_dir / "crop_raw.png"
        if not metadata_path.exists() or not crop_path.exists():
            continue
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        label = str(metadata.get("label") or section_dir.name)
        stain = str(metadata.get("stain") or "").lower()
        has_masks = (section_dir / "mask_labels.png").exists() or (section_dir / "tissue_mask_final.png").exists()
        prepared_work = dict(metadata.get("prepared_work_images") or {})
        prepared_profiles = tuple(sorted(str(k) for k in prepared_work.keys()))
        items.append(
            WorkspaceSection(
                section_dir=section_dir,
                label=label,
                stain=stain,
                metadata_path=metadata_path,
                crop_path=crop_path,
                has_masks=has_masks,
                has_prepared_work=bool(prepared_profiles),
                prepared_work_profiles=prepared_profiles,
            )
        )
    return items


def load_workspace_metadata(item: WorkspaceSection) -> dict[str, Any]:
    return json.loads(item.metadata_path.read_text(encoding="utf-8"))


def load_workspace_section(
    item: WorkspaceSection,
    *,
    external_mask_root: Path | None = None,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    metadata = load_workspace_metadata(item)
    crop_rgb = _read_png_rgb(item.crop_path)
    shape_hw = crop_rgb.shape[:2]
    external_mask = find_external_prediction_mask(item, external_mask_root)
    if external_mask is not None:
        tissue = _read_png_gray(external_mask, shape_hw=shape_hw)
        artifact = np.zeros(shape_hw, dtype=np.uint8)
        source_info = {
            "mask_source": "external_prediction_root",
            "mask_path": str(external_mask),
            "artifact_source": "none",
        }
    else:
        label_path = item.section_dir / "mask_labels.png"
        if label_path.exists():
            tissue, artifact = load_masks_from_label_path(label_path, shape_hw=shape_hw)
            source_info = {
                "mask_source": "workspace_mask_labels",
                "mask_path": str(label_path),
                "artifact_source": str(label_path),
            }
        else:
            tissue = _read_png_gray(item.section_dir / "tissue_mask_final.png", shape_hw=shape_hw)
            artifact = _read_png_gray(item.section_dir / "artifact_mask_final.png", shape_hw=shape_hw)
            source_info = {
                "mask_source": "workspace_section_folder_legacy",
                "mask_path": str(item.section_dir / "tissue_mask_final.png"),
                "artifact_source": str(item.section_dir / "artifact_mask_final.png"),
            }
    return metadata, crop_rgb, tissue, artifact, source_info


def prepare_workspace_work_image(
    item: WorkspaceSection,
    *,
    compute_profile: str,
    max_long_edge: int | None,
) -> dict[str, Any]:
    metadata = load_workspace_metadata(item)
    orig_h, orig_w = _metadata_original_shape_hw(metadata, item)
    long_edge = max(orig_h, orig_w)
    work_root = dict(metadata.get("prepared_work_images") or {})
    load_t0 = perf_counter()
    load_s = 0.0
    resize_s = 0.0
    write_s = 0.0

    if max_long_edge is None or long_edge <= max_long_edge:
        work_name = item.crop_path.name
        work_path = item.crop_path
        working_h, working_w = orig_h, orig_w
        working_scale = 1.0
    else:
        working_scale = float(max_long_edge) / float(long_edge)
        working_w = max(1, int(round(orig_w * working_scale)))
        working_h = max(1, int(round(orig_h * working_scale)))
        crop_rgb = _read_png_rgb(item.crop_path)
        load_s = perf_counter() - load_t0
        resize_t0 = perf_counter()
        work_rgb = cv2.resize(crop_rgb, (working_w, working_h), interpolation=cv2.INTER_AREA)
        resize_s = perf_counter() - resize_t0
        work_name = _work_image_name(compute_profile)
        work_path = item.section_dir / work_name
        write_t0 = perf_counter()
        write_png_lossless_fast(work_path, work_rgb, "RGB")
        write_s = perf_counter() - write_t0

    work_root[compute_profile] = {
        "prepared_at_utc": _utc_now_iso(),
        "path": work_name,
        "working_shape_hw": [int(working_h), int(working_w)],
        "original_shape_hw": [int(orig_h), int(orig_w)],
        "working_scale": float(working_scale),
        "working_max_long_edge": None if max_long_edge is None else int(max_long_edge),
    }
    metadata["prepared_work_images"] = work_root
    status = dict(metadata.get("workspace_status") or {})
    prepared_profiles = sorted(set(str(x) for x in status.get("prepared_work_profiles", [])) | {compute_profile})
    metadata["workspace_status"] = {
        "crop_exported": True,
        "mask_predicted": bool(status.get("mask_predicted", False)),
        "mask_reviewed": bool(status.get("mask_reviewed", False)),
        "prepared_work_profiles": prepared_profiles,
    }
    metadata["pipeline_stage"] = metadata.get("pipeline_stage") or "crop_exported"
    item.metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return {
        "path": str(work_path),
        "working_shape_hw": [int(working_h), int(working_w)],
        "original_shape_hw": [int(orig_h), int(orig_w)],
        "working_scale": float(working_scale),
        "timing_s": {
            "load": round(load_s, 4),
            "resize": round(resize_s, 4),
            "write": round(write_s, 4),
        },
    }


def load_workspace_prediction_input(
    item: WorkspaceSection,
    *,
    compute_profile: str,
) -> tuple[dict[str, Any], np.ndarray, dict[str, Any]]:
    metadata = load_workspace_metadata(item)
    prepared = dict(metadata.get("prepared_work_images") or {}).get(compute_profile)
    if not isinstance(prepared, dict):
        raise FileNotFoundError(
            f"Prepared work image for profile '{compute_profile}' not found in {item.section_dir}. "
            "Run the downsample preparation step first."
        )
    work_name = str(prepared.get("path") or "")
    work_path = item.crop_path if work_name == item.crop_path.name else (item.section_dir / work_name)
    if not work_path.exists():
        raise FileNotFoundError(f"Prepared work image missing: {work_path}")
    work_rgb = _read_png_rgb(work_path)
    info = {
        "work_image_path": str(work_path),
        "working_shape_hw": list(prepared.get("working_shape_hw") or [int(work_rgb.shape[0]), int(work_rgb.shape[1])]),
        "original_shape_hw": list(prepared.get("original_shape_hw") or list(_metadata_original_shape_hw(metadata, item))),
        "working_scale": float(prepared.get("working_scale") or 1.0),
        "working_max_long_edge": prepared.get("working_max_long_edge"),
    }
    return metadata, work_rgb, info


def write_workspace_prediction(
    item: WorkspaceSection,
    tissue_mask: np.ndarray,
    artifact_mask: np.ndarray,
    *,
    mask_preset: str,
    mask_algorithm_version: str,
    mask_compute_profile: str,
    compute_info: dict[str, Any] | None = None,
) -> None:
    metadata = load_workspace_metadata(item)
    payload = build_export_payload(
        np.zeros((*tissue_mask.shape[:2], 3), dtype=np.uint8),
        tissue_mask,
        artifact_mask,
    )
    write_png_lossless_fast(item.section_dir / "mask_labels.png", payload["mask_labels"], "L")
    for stale_name in (
        "mask_preview.png",
        "artifact_mask_final.png",
        "tissue_mask_final.png",
        "usable_tissue_mask.png",
        "foreground_rgba.png",
        "foreground_rgbwhite.png",
        "foreground_rgbblack.png",
        "foreground_rgb_white.png",
        "foreground_rgb_black.png",
    ):
        stale = item.section_dir / stale_name
        if stale.exists():
            stale.unlink()
    metadata["pipeline_stage"] = "mask_predicted"
    status = dict(metadata.get("workspace_status") or {})
    metadata["workspace_status"] = {
        "crop_exported": True,
        "mask_predicted": True,
        "mask_reviewed": bool(status.get("mask_reviewed", False)),
        "prepared_work_profiles": list(status.get("prepared_work_profiles", [])),
    }
    metadata["mask_prediction"] = {
        "predicted_at_utc": _utc_now_iso(),
        "mask_preset_selected": mask_preset,
        "mask_algorithm_version": mask_algorithm_version,
        "mask_compute_profile": mask_compute_profile,
        "generated_files": [
            "mask_labels.png",
            "metadata.json",
        ],
        "mask_compute_info": dict(compute_info or {}),
    }
    metadata["output_files"] = [
        "crop_raw.png",
        "mask_labels.png",
        "metadata.json",
    ]
    item.metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def save_workspace_review(
    item: WorkspaceSection,
    crop_rgb: np.ndarray,
    tissue_mask: np.ndarray,
    artifact_mask: np.ndarray,
    *,
    mask_preset: str,
    backup_existing: bool = True,
    write_preview: bool = True,
    write_rgba: bool = True,
) -> Path | None:
    metadata = load_workspace_metadata(item)
    backup_dir = _backup_existing_outputs(item.section_dir) if backup_existing else None
    write_png_lossless_fast(item.section_dir / "mask_labels.png", mask_labels_from_masks(tissue_mask, artifact_mask), "L")
    if write_preview or write_rgba:
        payload = build_export_payload(crop_rgb, tissue_mask, artifact_mask)
        if write_preview:
            write_png_lossless_fast(item.section_dir / "mask_preview.png", payload["mask_preview"], "RGB")
        if write_rgba:
            write_png_lossless_fast(item.section_dir / "foreground_rgba.png", payload["foreground_rgba"], "RGBA")
    for stale_name in (
        "artifact_mask_final.png",
        "tissue_mask_final.png",
        "usable_tissue_mask.png",
        "foreground_rgbwhite.png",
        "foreground_rgbblack.png",
        "foreground_rgb_white.png",
        "foreground_rgb_black.png",
    ):
        stale = item.section_dir / stale_name
        if stale.exists():
            stale.unlink()
    prev_summary = dict(metadata.get("manual_edit_summary") or {})
    metadata["pipeline_stage"] = "mask_reviewed"
    metadata["workspace_status"] = {
        "crop_exported": True,
        "mask_predicted": True,
        "mask_reviewed": True,
    }
    metadata["output_files"] = [
        "crop_raw.png",
        "mask_labels.png",
        "metadata.json",
    ]
    if write_preview:
        metadata["output_files"].insert(2, "mask_preview.png")
    if write_rgba:
        insert_at = 3 if write_preview else 2
        metadata["output_files"].insert(insert_at, "foreground_rgba.png")
    metadata["mask_label_semantics"] = {
        "background": 0,
        "tissue": 1,
        "artifact": 2,
        "artifact_never_overwrites_tissue": True,
    }
    metadata["manual_edit_summary"] = {
        **prev_summary,
        "manually_edited": True,
        "review_saved_at_utc": _utc_now_iso(),
        "mask_preset_selected": mask_preset,
        "backup_revision_dir": str(backup_dir) if backup_dir is not None else None,
    }
    item.metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return backup_dir

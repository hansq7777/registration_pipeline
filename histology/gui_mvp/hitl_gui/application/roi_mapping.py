from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import nibabel as nib
import numpy as np

from .pair_registration import (
    ants_cli_path,
    apply_group_flips,
    apply_whole_flip,
    keep_group,
    load_mask_labels,
    registration_support_mask,
)
from ..pipeline_adapters.slide_io import write_png_lossless_fast


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def default_pair_roi_root(myelin_root: Path | None, nissl_root: Path | None) -> Path | None:
    roots = [p for p in (myelin_root, nissl_root) if p is not None]
    if not roots:
        return None
    common = Path(os.path.commonpath([str(p.resolve()) for p in roots]))
    return common / "histology_pair_roi_annotations"


def _resolve_common_relpath(common_root: Path, rel_or_abs: str | None) -> Path | None:
    if rel_or_abs is None:
        return None
    raw = str(rel_or_abs).strip()
    if not raw:
        return None
    path = Path(raw)
    if path.is_absolute():
        return path
    return common_root / path


def _read_png_rgb(path: Path) -> np.ndarray:
    arr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if arr is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)


def _read_png_gray(path: Path, *, shape_hw: tuple[int, int] | None = None) -> np.ndarray:
    arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if arr is None:
        if shape_hw is None:
            raise FileNotFoundError(path)
        return np.zeros(shape_hw, dtype=np.uint8)
    if arr.ndim == 3:
        arr = arr[..., 0]
    gray = arr.astype(np.uint8)
    if shape_hw is not None and gray.shape[:2] != shape_hw:
        gray = cv2.resize(gray, (shape_hw[1], shape_hw[0]), interpolation=cv2.INTER_NEAREST)
    return gray


def _overlay_preview(rgb: np.ndarray, roi_mask: np.ndarray, *, color_rgb: tuple[int, int, int]) -> np.ndarray:
    overlay = rgb.copy()
    mask = roi_mask > 0
    if np.any(mask):
        tint = np.array(color_rgb, dtype=np.float32)
        overlay[mask] = np.clip(0.45 * overlay[mask].astype(np.float32) + 0.55 * tint, 0, 255).astype(np.uint8)
    return overlay


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _run_logged(cmd: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as f:
        proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(cmd)} | log={log_path}")


def _forward_transform_specs_for_stage(run_dir: Path, manifest: dict[str, Any], stage: str) -> list[str]:
    stage_data = dict((manifest.get("stages") or {}).get(stage) or {})
    transforms = [str(x) for x in stage_data.get("transforms") or [] if str(x).strip()]
    if not transforms:
        raise FileNotFoundError(f"No transforms found for stage={stage} in {run_dir}")
    return transforms


def _inverse_transform_specs_for_stage(run_dir: Path, manifest: dict[str, Any], stage: str) -> list[str]:
    forward = _forward_transform_specs_for_stage(run_dir, manifest, stage)
    inverse_specs: list[str] = []
    for raw in reversed(forward):
        path = Path(raw)
        name = path.name
        if name.endswith(".mat"):
            inverse_specs.append(f"[{ants_cli_path(path)},1]")
            continue
        if name.endswith("1Warp.nii.gz"):
            inv = path.with_name(name.replace("1Warp.nii.gz", "1InverseWarp.nii.gz"))
            if not inv.exists():
                raise FileNotFoundError(f"Missing inverse warp for {path}")
            inverse_specs.append(ants_cli_path(inv))
            continue
        raise ValueError(f"Unsupported transform for inversion: {path}")
    return inverse_specs


def _latest_completed_stage(manifest: dict[str, Any]) -> str:
    requested = [str(x).strip().lower() for x in manifest.get("run_stages") or [] if str(x).strip()]
    stages = dict(manifest.get("stages") or {})
    for stage in reversed(requested):
        if stage in stages:
            return stage
    for stage in ("syn", "affine", "rigid"):
        if stage in stages:
            return stage
    raise ValueError("No completed stage found in approved registration manifest.")


def _normalize_shape_hw(shape: Any, fallback_hw: tuple[int, int]) -> tuple[int, int]:
    if isinstance(shape, (list, tuple)) and len(shape) >= 2:
        try:
            return int(shape[0]), int(shape[1])
        except Exception:
            pass
    return int(fallback_hw[0]), int(fallback_hw[1])


def _bbox_tuple_from_manifest(bbox: dict[str, Any], fallback_hw: tuple[int, int]) -> tuple[int, int, int, int]:
    try:
        x = int(bbox.get("x", 0))
        y = int(bbox.get("y", 0))
        w = int(bbox.get("w", fallback_hw[1]))
        h = int(bbox.get("h", fallback_hw[0]))
        return x, y, w, h
    except Exception:
        return 0, 0, int(fallback_hw[1]), int(fallback_hw[0])


def _resample_mask(mask: np.ndarray, shape_hw: tuple[int, int]) -> np.ndarray:
    out_h, out_w = int(shape_hw[0]), int(shape_hw[1])
    if mask.shape[:2] == (out_h, out_w):
        return mask.copy()
    return cv2.resize(mask.astype(np.uint8), (out_w, out_h), interpolation=cv2.INTER_NEAREST)


def _apply_registration_view_transform(
    rgb: np.ndarray,
    labels: np.ndarray,
    preprocess: dict[str, Any],
    registration_mask_mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    rgb2, labels2 = apply_whole_flip(rgb, labels, bool(preprocess.get("whole_flip_lr", False)))
    rgb2, labels2 = apply_group_flips(
        rgb2,
        labels2,
        dict(preprocess.get("component_groups") or {}),
        dict(preprocess.get("group_flip_lr") or {}),
    )
    labels2 = keep_group(labels2, dict(preprocess.get("component_groups") or {}), str(preprocess.get("group_choice") or "all"))
    support = registration_support_mask(labels2, str(registration_mask_mode))
    rgb2 = rgb2.copy()
    rgb2[~support] = 255
    return rgb2, labels2


@dataclass
class ApprovedRegistrationContext:
    pair_key: str
    common_root: Path
    run_dir: Path
    manifest_path: Path
    manifest: dict[str, Any]
    approved_stage: str
    nissl_role: str
    myelin_role: str
    nissl_ref_nifti: Path
    myelin_ref_nifti: Path
    group_tag: str
    output_dir: Path
    myelin_root: Path
    nissl_root: Path
    myelin_label: str
    nissl_label: str
    nissl_preprocess: dict[str, Any]
    myelin_preprocess: dict[str, Any]
    registration_mask_mode: str


def load_approved_registration_context(
    pair_key: str,
    review: dict[str, Any],
    common_root: Path,
    roi_root: Path,
    myelin_root: Path,
    nissl_root: Path,
) -> ApprovedRegistrationContext | None:
    approved = dict(review.get("approved_registration") or {})
    run_dir = _resolve_common_relpath(common_root, approved.get("run_dir"))
    manifest_path = _resolve_common_relpath(common_root, approved.get("manifest_path"))
    if run_dir is None or manifest_path is None or not run_dir.exists() or not manifest_path.exists():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    approved_stage = str(approved.get("approved_stage") or _latest_completed_stage(manifest)).strip().lower()
    fixed_side = str(manifest.get("fixed_side") or "").strip().lower()
    moving_side = str(manifest.get("moving_side") or "").strip().lower()
    if {fixed_side, moving_side} != {"nissl", "myelin"}:
        raise ValueError(f"Approved registration must connect nissl<->myelin, got fixed={fixed_side}, moving={moving_side}")

    fixed_ref_nifti = _resolve_common_relpath(common_root, ((manifest.get("inputs") or {}).get("fixed_gray")))
    moving_ref_nifti = _resolve_common_relpath(common_root, ((manifest.get("inputs") or {}).get("moving_gray")))
    if fixed_ref_nifti is None or moving_ref_nifti is None:
        raise FileNotFoundError("Approved registration manifest is missing fixed/moving gray inputs.")

    nissl_role = "fixed" if fixed_side == "nissl" else "moving"
    myelin_role = "moving" if nissl_role == "fixed" else "fixed"
    nissl_ref_nifti = fixed_ref_nifti if nissl_role == "fixed" else moving_ref_nifti
    myelin_ref_nifti = moving_ref_nifti if nissl_role == "fixed" else fixed_ref_nifti
    group_tag = str(approved.get("group_tag") or approved.get("fixed_group") or approved.get("moving_group") or "all")
    output_dir = roi_root / pair_key / f"group_{group_tag}"

    inputs = dict(manifest.get("inputs") or {})
    fixed_pre = dict(inputs.get("fixed_preprocess") or {})
    moving_pre = dict(inputs.get("moving_preprocess") or {})
    nissl_pre = fixed_pre if nissl_role == "fixed" else moving_pre
    myelin_pre = moving_pre if nissl_role == "fixed" else fixed_pre

    return ApprovedRegistrationContext(
        pair_key=pair_key,
        common_root=common_root,
        run_dir=run_dir,
        manifest_path=manifest_path,
        manifest=manifest,
        approved_stage=approved_stage,
        nissl_role=nissl_role,
        myelin_role=myelin_role,
        nissl_ref_nifti=nissl_ref_nifti,
        myelin_ref_nifti=myelin_ref_nifti,
        group_tag=group_tag,
        output_dir=output_dir,
        myelin_root=myelin_root,
        nissl_root=nissl_root,
        myelin_label=str(review.get("myelin_label") or ""),
        nissl_label=str(review.get("nissl_label") or ""),
        nissl_preprocess=nissl_pre,
        myelin_preprocess=myelin_pre,
        registration_mask_mode=str(manifest.get("registration_mask_mode") or "union"),
    )


def _load_highres_registration_view(root: Path, label: str, preprocess: dict[str, Any], registration_mask_mode: str) -> tuple[np.ndarray, np.ndarray]:
    section_dir = root / label
    rgb = _read_png_rgb(section_dir / "crop_raw.png")
    labels = load_mask_labels(section_dir / "mask_labels.png")
    if labels.shape[:2] != rgb.shape[:2]:
        labels = cv2.resize(labels, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
    return _apply_registration_view_transform(rgb, labels, preprocess, registration_mask_mode)


def _load_saved_highres_roi(path: Path, shape_hw: tuple[int, int]) -> np.ndarray:
    return _read_png_gray(path, shape_hw=shape_hw) if path.exists() else np.zeros(shape_hw, dtype=np.uint8)


def current_step6_state(context: ApprovedRegistrationContext) -> dict[str, Any]:
    nissl_rgb, _nissl_labels = _load_highres_registration_view(
        context.nissl_root,
        context.nissl_label,
        context.nissl_preprocess,
        context.registration_mask_mode,
    )
    myelin_rgb, _myelin_labels = _load_highres_registration_view(
        context.myelin_root,
        context.myelin_label,
        context.myelin_preprocess,
        context.registration_mask_mode,
    )
    nissl_roi = _load_saved_highres_roi(context.output_dir / "roi_labels_nissl_highres.png", nissl_rgb.shape[:2])
    myelin_roi = _load_saved_highres_roi(context.output_dir / "roi_labels_myelin_mapped_highres.png", myelin_rgb.shape[:2])
    return {
        "nissl_rgb": nissl_rgb,
        "myelin_rgb": myelin_rgb,
        "nissl_roi": nissl_roi,
        "myelin_roi": myelin_roi,
        "nissl_overlay": _overlay_preview(nissl_rgb, nissl_roi, color_rgb=(255, 80, 80)),
        "myelin_overlay": _overlay_preview(myelin_rgb, myelin_roi, color_rgb=(80, 220, 255)),
    }


def _highres_roi_to_canvas(context: ApprovedRegistrationContext, roi_highres: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    pre = context.nissl_preprocess
    canonical_shape_hw = _normalize_shape_hw(pre.get("canonical_shape_hw"), roi_highres.shape[:2])
    if roi_highres.shape[:2] != canonical_shape_hw:
        roi_highres = _resample_mask(roi_highres, canonical_shape_hw)
    x, y, w, h = _bbox_tuple_from_manifest(pre.get("support_crop_bbox_xywh") or {}, canonical_shape_hw)
    roi_crop = roi_highres[y : y + h, x : x + w].copy()
    physical_shape_hw = _normalize_shape_hw(pre.get("physical_normalized_shape_hw"), roi_crop.shape[:2])
    working_shape_hw = _normalize_shape_hw(pre.get("working_shape_hw"), physical_shape_hw)
    roi_physical = _resample_mask(roi_crop, physical_shape_hw)
    roi_working = _resample_mask(roi_physical, working_shape_hw)
    canvas_shape_hw = _normalize_shape_hw(pre.get("common_canvas_shape_hw"), working_shape_hw)
    offset = dict(pre.get("common_canvas_offset_xy") or {})
    ox = int(offset.get("x", 0))
    oy = int(offset.get("y", 0))
    roi_canvas = np.zeros(canvas_shape_hw, dtype=np.uint8)
    roi_canvas[oy : oy + working_shape_hw[0], ox : ox + working_shape_hw[1]] = roi_working
    return roi_canvas, {
        "canonical_shape_hw": list(canonical_shape_hw),
        "support_crop_bbox_xywh": {"x": x, "y": y, "w": w, "h": h},
        "support_crop_shape_hw": [int(h), int(w)],
        "physical_normalized_shape_hw": list(physical_shape_hw),
        "working_shape_hw": list(working_shape_hw),
        "common_canvas_shape_hw": list(canvas_shape_hw),
        "common_canvas_offset_xy": {"x": ox, "y": oy},
    }


def _canvas_roi_to_myelin_highres(context: ApprovedRegistrationContext, roi_canvas: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    pre = context.myelin_preprocess
    canonical_shape_hw = _normalize_shape_hw(pre.get("canonical_shape_hw"), roi_canvas.shape[:2])
    physical_shape_hw = _normalize_shape_hw(pre.get("physical_normalized_shape_hw"), canonical_shape_hw)
    working_shape_hw = _normalize_shape_hw(pre.get("working_shape_hw"), physical_shape_hw)
    canvas_shape_hw = _normalize_shape_hw(pre.get("common_canvas_shape_hw"), working_shape_hw)
    if roi_canvas.shape[:2] != canvas_shape_hw:
        roi_canvas = _resample_mask(roi_canvas, canvas_shape_hw)
    offset = dict(pre.get("common_canvas_offset_xy") or {})
    ox = int(offset.get("x", 0))
    oy = int(offset.get("y", 0))
    roi_working = roi_canvas[oy : oy + working_shape_hw[0], ox : ox + working_shape_hw[1]].copy()
    roi_physical = _resample_mask(roi_working, physical_shape_hw)
    x, y, w, h = _bbox_tuple_from_manifest(pre.get("support_crop_bbox_xywh") or {}, canonical_shape_hw)
    roi_support = _resample_mask(roi_physical, (h, w))
    roi_highres = np.zeros(canonical_shape_hw, dtype=np.uint8)
    roi_highres[y : y + h, x : x + w] = roi_support
    return roi_highres, {
        "canonical_shape_hw": list(canonical_shape_hw),
        "support_crop_bbox_xywh": {"x": x, "y": y, "w": w, "h": h},
        "support_crop_shape_hw": [int(h), int(w)],
        "physical_normalized_shape_hw": list(physical_shape_hw),
        "working_shape_hw": list(working_shape_hw),
        "common_canvas_shape_hw": list(canvas_shape_hw),
        "common_canvas_offset_xy": {"x": ox, "y": oy},
    }


def _map_canvas_roi_to_myelin(
    context: ApprovedRegistrationContext,
    nissl_roi_canvas: np.ndarray,
    ants_bin: Path,
) -> np.ndarray:
    work_dir = context.output_dir / "_work"
    work_dir.mkdir(parents=True, exist_ok=True)
    nissl_roi_nifti = work_dir / "roi_labels_nissl_canvas.nii.gz"
    mapped_nifti = work_dir / "roi_labels_myelin_canvas_mapped.nii.gz"

    ref = nib.load(str(context.nissl_ref_nifti))
    nib.save(nib.Nifti1Image(nissl_roi_canvas.astype(np.float32), ref.affine, ref.header), str(nissl_roi_nifti))

    if context.nissl_role == "fixed":
        transforms = _inverse_transform_specs_for_stage(context.run_dir, context.manifest, context.approved_stage)
        reference = context.myelin_ref_nifti
    else:
        transforms = _forward_transform_specs_for_stage(context.run_dir, context.manifest, context.approved_stage)
        reference = context.myelin_ref_nifti

    cmd = [
        str(ants_bin / "antsApplyTransforms"),
        "-d",
        "2",
        "-i",
        ants_cli_path(nissl_roi_nifti),
        "-r",
        ants_cli_path(reference),
        "-o",
        ants_cli_path(mapped_nifti),
        "-n",
        "NearestNeighbor",
    ]
    for tfm in transforms:
        cmd.extend(["-t", tfm if tfm.startswith("[") else ants_cli_path(Path(tfm))])
    _run_logged(cmd, work_dir / "update_roi_mapping.log")
    arr = np.asarray(nib.load(str(mapped_nifti)).dataobj)
    if arr.ndim > 2:
        arr = np.squeeze(arr)
    return np.where(arr > 0.5, 255, 0).astype(np.uint8)


def update_step6_roi_mapping(
    context: ApprovedRegistrationContext,
    nissl_roi_highres: np.ndarray,
    ants_bin: Path,
) -> dict[str, Any]:
    nissl_canvas, nissl_debug = _highres_roi_to_canvas(context, nissl_roi_highres)
    myelin_canvas = _map_canvas_roi_to_myelin(context, nissl_canvas, ants_bin)
    myelin_highres, myelin_debug = _canvas_roi_to_myelin_highres(context, myelin_canvas)
    return {
        "nissl_roi_highres": nissl_roi_highres.astype(np.uint8),
        "nissl_roi_canvas": nissl_canvas.astype(np.uint8),
        "myelin_roi_canvas": myelin_canvas.astype(np.uint8),
        "myelin_roi_highres": myelin_highres.astype(np.uint8),
        "nissl_debug": nissl_debug,
        "myelin_debug": myelin_debug,
    }


def save_step6_roi(
    context: ApprovedRegistrationContext,
    mapping_result: dict[str, Any],
) -> dict[str, Any]:
    context.output_dir.mkdir(parents=True, exist_ok=True)
    state = current_step6_state(context)
    nissl_rgb = state["nissl_rgb"]
    myelin_rgb = state["myelin_rgb"]
    nissl_roi_highres = np.asarray(mapping_result["nissl_roi_highres"], dtype=np.uint8)
    nissl_roi_canvas = np.asarray(mapping_result["nissl_roi_canvas"], dtype=np.uint8)
    myelin_roi_canvas = np.asarray(mapping_result["myelin_roi_canvas"], dtype=np.uint8)
    myelin_roi_highres = np.asarray(mapping_result["myelin_roi_highres"], dtype=np.uint8)
    nissl_overlay = _overlay_preview(nissl_rgb, nissl_roi_highres, color_rgb=(255, 80, 80))
    myelin_overlay = _overlay_preview(myelin_rgb, myelin_roi_highres, color_rgb=(80, 220, 255))

    write_png_lossless_fast(context.output_dir / "roi_labels_nissl_highres.png", nissl_roi_highres, "L")
    write_png_lossless_fast(context.output_dir / "roi_labels_nissl_canvas.png", nissl_roi_canvas, "L")
    write_png_lossless_fast(context.output_dir / "roi_labels_myelin_canvas.png", myelin_roi_canvas, "L")
    write_png_lossless_fast(context.output_dir / "roi_labels_myelin_mapped_highres.png", myelin_roi_highres, "L")
    write_png_lossless_fast(context.output_dir / "roi_overlay_nissl.png", nissl_overlay.astype(np.uint8), "RGB")
    write_png_lossless_fast(context.output_dir / "roi_overlay_myelin.png", myelin_overlay.astype(np.uint8), "RGB")

    payload = {
        "pair_key": context.pair_key,
        "group_tag": context.group_tag,
        "approved_stage": context.approved_stage,
        "approved_run_dir": str(context.run_dir),
        "approved_manifest_path": str(context.manifest_path),
        "nissl_role": context.nissl_role,
        "myelin_role": context.myelin_role,
        "registration_mask_mode": context.registration_mask_mode,
        "saved_at_utc": _utc_now_iso(),
        "nissl_preprocess": context.nissl_preprocess,
        "myelin_preprocess": context.myelin_preprocess,
        "nissl_mapping_debug": mapping_result.get("nissl_debug", {}),
        "myelin_mapping_debug": mapping_result.get("myelin_debug", {}),
        "files": {
            "roi_labels_nissl_highres": str(context.output_dir / "roi_labels_nissl_highres.png"),
            "roi_labels_nissl_canvas": str(context.output_dir / "roi_labels_nissl_canvas.png"),
            "roi_labels_myelin_canvas": str(context.output_dir / "roi_labels_myelin_canvas.png"),
            "roi_labels_myelin_mapped_highres": str(context.output_dir / "roi_labels_myelin_mapped_highres.png"),
            "roi_overlay_nissl": str(context.output_dir / "roi_overlay_nissl.png"),
            "roi_overlay_myelin": str(context.output_dir / "roi_overlay_myelin.png"),
        },
    }
    _write_json(context.output_dir / "roi_manifest.json", payload)
    return payload

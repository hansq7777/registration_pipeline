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
    component_rank_map,
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


def _resolve_existing_path(raw: str | None) -> Path | None:
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    candidate = Path(text)
    if candidate.exists():
        return candidate
    if len(text) >= 3 and text[1] == ":" and text[2] in {"\\", "/"}:
        drive = text[0].lower()
        tail = text[3:].replace("\\", "/")
        alt = Path(f"/mnt/{drive}") / Path(tail)
        if alt.exists():
            return alt
    return None


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


def _best_accepted_stage(manifest: dict[str, Any]) -> str:
    best_stage = str(manifest.get("best_stage") or "").strip().lower()
    if best_stage and best_stage != "input":
        return best_stage
    accepted = [str(x).strip().lower() for x in manifest.get("accepted_stage_path") or [] if str(x).strip()]
    for stage in reversed(accepted):
        if stage != "input":
            return stage
    return ""


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
    registration_backend: str
    nissl_role: str
    myelin_role: str
    nissl_ref_nifti: Path | None
    myelin_ref_nifti: Path | None
    group_tag: str
    output_dir: Path
    myelin_root: Path
    nissl_root: Path
    myelin_label: str
    nissl_label: str
    nissl_preprocess: dict[str, Any]
    myelin_preprocess: dict[str, Any]
    registration_mask_mode: str
    approved_transform_matrix_2x3: list[list[float]] | None


def _normalize_matrix_2x3(raw: Any) -> list[list[float]] | None:
    if not isinstance(raw, (list, tuple)) or len(raw) != 2:
        return None
    rows: list[list[float]] = []
    for row in raw:
        if not isinstance(row, (list, tuple)) or len(row) != 3:
            return None
        try:
            rows.append([float(row[0]), float(row[1]), float(row[2])])
        except Exception:
            return None
    return rows


def _load_mask_shape_stage_matrix(common_root: Path, manifest: dict[str, Any], stage: str) -> list[list[float]] | None:
    stage_data = dict((manifest.get("stages") or {}).get(stage) or {})
    matrix = _normalize_matrix_2x3(stage_data.get("transform_matrix_2x3"))
    if matrix is not None:
        return matrix
    transforms = [str(x) for x in stage_data.get("transforms") or [] if str(x).strip()]
    for raw in transforms:
        path = _resolve_common_relpath(common_root, raw)
        if path is None or not path.exists() or path.suffix.lower() != ".json":
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        matrix = _normalize_matrix_2x3(payload.get("matrix_2x3") or payload.get("transform_matrix_2x3"))
        if matrix is not None:
            return matrix
    return None


def _invert_affine_matrix_2x3(matrix_2x3: np.ndarray) -> np.ndarray:
    mat = np.asarray(matrix_2x3, dtype=np.float64)
    if mat.shape != (2, 3):
        raise ValueError(f"Expected 2x3 affine matrix, got {mat.shape}")
    hom = np.vstack([mat, np.array([0.0, 0.0, 1.0], dtype=np.float64)])
    inv = np.linalg.inv(hom)
    return inv[:2, :].astype(np.float32)


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
    approved_stage = str(approved.get("approved_stage") or "").strip().lower()
    if not approved_stage or approved_stage == "input":
        approved_stage = _best_accepted_stage(manifest)
    if not approved_stage:
        return None
    registration_backend = str(manifest.get("registration_backend") or "ants").strip().lower() or "ants"
    fixed_side = str(manifest.get("fixed_side") or "").strip().lower()
    moving_side = str(manifest.get("moving_side") or "").strip().lower()
    if {fixed_side, moving_side} != {"nissl", "myelin"}:
        raise ValueError(f"Approved registration must connect nissl<->myelin, got fixed={fixed_side}, moving={moving_side}")

    fixed_ref_nifti: Path | None = None
    moving_ref_nifti: Path | None = None
    approved_transform_matrix_2x3: list[list[float]] | None = None
    if registration_backend == "ants":
        fixed_ref_nifti = _resolve_common_relpath(common_root, ((manifest.get("inputs") or {}).get("fixed_gray")))
        moving_ref_nifti = _resolve_common_relpath(common_root, ((manifest.get("inputs") or {}).get("moving_gray")))
        if fixed_ref_nifti is None or moving_ref_nifti is None:
            raise FileNotFoundError("Approved registration manifest is missing fixed/moving gray inputs.")
    elif registration_backend == "mask_shape":
        approved_transform_matrix_2x3 = _load_mask_shape_stage_matrix(common_root, manifest, approved_stage)
        if approved_transform_matrix_2x3 is None:
            raise FileNotFoundError(
                f"Approved mask-shape stage is missing transform_matrix_2x3: stage={approved_stage} run={run_dir}"
            )
    else:
        raise ValueError(f"Unsupported approved registration backend: {registration_backend}")

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
        registration_backend=registration_backend,
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
        approved_transform_matrix_2x3=approved_transform_matrix_2x3,
    )


def _normalize_step6_side(side: str) -> str:
    side_norm = str(side).strip().lower()
    if side_norm not in {"nissl", "myelin"}:
        raise ValueError(f"Unsupported Step 6 side: {side}")
    return side_norm


def _opposite_step6_side(side: str) -> str:
    side_norm = _normalize_step6_side(side)
    return "myelin" if side_norm == "nissl" else "nissl"


def _preprocess_for_side(context: ApprovedRegistrationContext, side: str) -> dict[str, Any]:
    return context.nissl_preprocess if _normalize_step6_side(side) == "nissl" else context.myelin_preprocess


def _role_for_side(context: ApprovedRegistrationContext, side: str) -> str:
    return context.nissl_role if _normalize_step6_side(side) == "nissl" else context.myelin_role


def _ref_nifti_for_side(context: ApprovedRegistrationContext, side: str) -> Path | None:
    return context.nissl_ref_nifti if _normalize_step6_side(side) == "nissl" else context.myelin_ref_nifti


def _roi_highres_path_for_side(context: ApprovedRegistrationContext, side: str) -> Path:
    return context.output_dir / f"roi_labels_{_normalize_step6_side(side)}_highres.png"


def _roi_canvas_path_for_side(context: ApprovedRegistrationContext, side: str) -> Path:
    return context.output_dir / f"roi_labels_{_normalize_step6_side(side)}_canvas.png"


def _roi_mapped_highres_path_for_side(context: ApprovedRegistrationContext, side: str) -> Path:
    return context.output_dir / f"roi_labels_{_normalize_step6_side(side)}_mapped_highres.png"


def _roi_overlay_path_for_side(context: ApprovedRegistrationContext, side: str) -> Path:
    return context.output_dir / f"roi_overlay_{_normalize_step6_side(side)}.png"


def _mask_labels_path_for_preprocess(section_dir: Path, preprocess: dict[str, Any]) -> Path:
    preferred = _resolve_existing_path(str(preprocess.get("mask_path") or ""))
    if preferred is not None:
        return preferred
    return section_dir / "mask_labels.png"


def _load_registration_mask_labels(section_dir: Path, preprocess: dict[str, Any], rgb_shape_hw: tuple[int, int]) -> np.ndarray:
    labels = load_mask_labels(_mask_labels_path_for_preprocess(section_dir, preprocess))
    if labels.shape[:2] != rgb_shape_hw:
        labels = cv2.resize(labels, (rgb_shape_hw[1], rgb_shape_hw[0]), interpolation=cv2.INTER_NEAREST)
    return labels


def _group_flip_bboxes_from_reference_labels(
    reference_labels: np.ndarray,
    preprocess: dict[str, Any],
) -> list[tuple[int, int, int, int]]:
    component_groups = dict(preprocess.get("component_groups") or {})
    group_flip_lr = dict(preprocess.get("group_flip_lr") or {})
    if not component_groups or not group_flip_lr:
        return []
    labels_cc, rank_to_label = component_rank_map(np.asarray(reference_labels, dtype=np.uint8))
    bboxes: list[tuple[int, int, int, int]] = []
    for raw_group_id, raw_enabled in group_flip_lr.items():
        if not bool(raw_enabled):
            continue
        ranks = component_groups.get(str(raw_group_id))
        if not isinstance(ranks, list):
            continue
        for raw_rank in ranks:
            try:
                rank = int(raw_rank)
            except Exception:
                continue
            label_idx = rank_to_label.get(rank)
            if label_idx is None:
                continue
            ys, xs = np.where(labels_cc == label_idx)
            if ys.size == 0 or xs.size == 0:
                continue
            bboxes.append((int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1))
    return bboxes


def _apply_group_flip_boxes_to_rgb_and_labels(
    rgb: np.ndarray,
    labels: np.ndarray,
    boxes: list[tuple[int, int, int, int]],
) -> tuple[np.ndarray, np.ndarray]:
    if not boxes:
        return rgb, labels
    out_rgb = np.asarray(rgb, dtype=np.uint8).copy()
    out_labels = np.asarray(labels, dtype=np.uint8).copy()
    for x1, y1, x2, y2 in list(boxes):
        out_rgb[y1:y2, x1:x2] = out_rgb[y1:y2, x1:x2][:, ::-1, :]
        out_labels[y1:y2, x1:x2] = out_labels[y1:y2, x1:x2][:, ::-1]
    return out_rgb, out_labels


def _load_highres_registration_view(root: Path, label: str, preprocess: dict[str, Any], registration_mask_mode: str) -> tuple[np.ndarray, np.ndarray]:
    section_dir = root / label
    rgb = _read_png_rgb(section_dir / "crop_raw.png")
    approved_labels = _load_registration_mask_labels(section_dir, preprocess, rgb.shape[:2])
    workspace_labels_path = section_dir / "mask_labels.png"
    if workspace_labels_path.exists():
        workspace_labels = load_mask_labels(workspace_labels_path)
        if workspace_labels.shape[:2] != rgb.shape[:2]:
            workspace_labels = cv2.resize(workspace_labels, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
    else:
        workspace_labels = approved_labels.copy()

    whole_flip = bool(preprocess.get("whole_flip_lr", False))
    rgb2, approved_after_whole = apply_whole_flip(rgb, approved_labels, whole_flip)
    if whole_flip:
        workspace_after_whole = workspace_labels[:, ::-1].copy()
    else:
        workspace_after_whole = workspace_labels.copy()

    flip_boxes = _group_flip_bboxes_from_reference_labels(approved_after_whole, preprocess)
    rgb2, workspace_labels2 = _apply_group_flip_boxes_to_rgb_and_labels(rgb2, workspace_after_whole, flip_boxes)
    return rgb2, workspace_labels2


def _apply_registration_view_transform_to_mask(
    mask_u8: np.ndarray,
    reference_labels: np.ndarray,
    preprocess: dict[str, Any],
    registration_mask_mode: str,
) -> np.ndarray:
    mask = np.where(np.asarray(mask_u8, dtype=np.uint8) > 0, 255, 0).astype(np.uint8)
    labels = np.asarray(reference_labels, dtype=np.uint8)
    if labels.shape[:2] != mask.shape[:2]:
        labels = cv2.resize(labels, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_NEAREST)
    mask_rgb = np.repeat(mask[..., None], 3, axis=2)
    mask_rgb, labels2 = apply_whole_flip(mask_rgb, labels, bool(preprocess.get("whole_flip_lr", False)))
    mask_rgb, labels2 = apply_group_flips(
        mask_rgb,
        labels2,
        dict(preprocess.get("component_groups") or {}),
        dict(preprocess.get("group_flip_lr") or {}),
    )
    labels2 = keep_group(labels2, dict(preprocess.get("component_groups") or {}), str(preprocess.get("group_choice") or "all"))
    support = registration_support_mask(labels2, str(registration_mask_mode))
    out = np.asarray(mask_rgb[..., 0], dtype=np.uint8)
    out[~support] = 0
    return np.where(out > 0, 255, 0).astype(np.uint8)


def _apply_affine_to_points(points_xy: np.ndarray, matrix_2x3: np.ndarray) -> np.ndarray:
    pts = np.asarray(points_xy, dtype=np.float32).reshape((-1, 2))
    if pts.size == 0:
        return pts.reshape((0, 2))
    aug = np.concatenate([pts, np.ones((pts.shape[0], 1), dtype=np.float32)], axis=1)
    mapped = aug @ np.asarray(matrix_2x3, dtype=np.float32).T
    return np.asarray(mapped, dtype=np.float32)


def _apply_registration_view_transform_to_points(
    points_xy: np.ndarray,
    reference_labels: np.ndarray,
    preprocess: dict[str, Any],
) -> np.ndarray:
    pts = np.asarray(points_xy, dtype=np.float32).reshape((-1, 2)).copy()
    if pts.size == 0:
        return pts.reshape((0, 2))
    labels = np.asarray(reference_labels, dtype=np.uint8).copy()
    h, w = labels.shape[:2]
    if bool(preprocess.get("whole_flip_lr", False)):
        pts[:, 0] = (float(w) - 1.0) - pts[:, 0]
        labels = labels[:, ::-1].copy()
    component_groups = dict(preprocess.get("component_groups") or {})
    group_flip_lr = dict(preprocess.get("group_flip_lr") or {})
    if component_groups and group_flip_lr:
        labels_cc, rank_to_label = component_rank_map(labels)
        for raw_group_id, raw_enabled in group_flip_lr.items():
            if not bool(raw_enabled):
                continue
            group_id = str(raw_group_id)
            ranks = component_groups.get(group_id)
            if not isinstance(ranks, list):
                continue
            for raw_rank in ranks:
                try:
                    rank = int(raw_rank)
                except Exception:
                    continue
                label_idx = rank_to_label.get(rank)
                if label_idx is None:
                    continue
                comp = labels_cc == label_idx
                ys, xs = np.where(comp)
                if ys.size == 0 or xs.size == 0:
                    continue
                y1, y2 = int(ys.min()), int(ys.max()) + 1
                x1, x2 = int(xs.min()), int(xs.max()) + 1
                inside = (
                    (pts[:, 0] >= float(x1))
                    & (pts[:, 0] <= float(x2 - 1))
                    & (pts[:, 1] >= float(y1))
                    & (pts[:, 1] <= float(y2 - 1))
                )
                pts[inside, 0] = (float(x1 + x2 - 1)) - pts[inside, 0]
                labels[y1:y2, x1:x2] = labels[y1:y2, x1:x2][:, ::-1]
                labels_cc[y1:y2, x1:x2] = labels_cc[y1:y2, x1:x2][:, ::-1]
    return pts


def _highres_points_to_canvas(
    context: ApprovedRegistrationContext,
    points_xy: np.ndarray,
    *,
    source_side: str,
) -> np.ndarray:
    pts = np.asarray(points_xy, dtype=np.float32).reshape((-1, 2)).copy()
    if pts.size == 0:
        return pts.reshape((0, 2))
    pre = _preprocess_for_side(context, source_side)
    canonical_shape_hw = _normalize_shape_hw(pre.get("canonical_shape_hw"), (1, 1))
    x, y, w, h = _bbox_tuple_from_manifest(pre.get("support_crop_bbox_xywh") or {}, canonical_shape_hw)
    physical_shape_hw = _normalize_shape_hw(pre.get("physical_normalized_shape_hw"), (h, w))
    working_shape_hw = _normalize_shape_hw(pre.get("working_shape_hw"), physical_shape_hw)
    offset = dict(pre.get("common_canvas_offset_xy") or {})
    ox = float(offset.get("x", 0))
    oy = float(offset.get("y", 0))
    pts[:, 0] -= float(x)
    pts[:, 1] -= float(y)
    scale_x_phys = float(physical_shape_hw[1]) / max(float(w), 1.0)
    scale_y_phys = float(physical_shape_hw[0]) / max(float(h), 1.0)
    pts[:, 0] *= scale_x_phys
    pts[:, 1] *= scale_y_phys
    scale_x_work = float(working_shape_hw[1]) / max(float(physical_shape_hw[1]), 1.0)
    scale_y_work = float(working_shape_hw[0]) / max(float(physical_shape_hw[0]), 1.0)
    pts[:, 0] = pts[:, 0] * scale_x_work + ox
    pts[:, 1] = pts[:, 1] * scale_y_work + oy
    return pts


def _canvas_points_to_highres(
    context: ApprovedRegistrationContext,
    points_xy: np.ndarray,
    *,
    target_side: str,
) -> np.ndarray:
    pts = np.asarray(points_xy, dtype=np.float32).reshape((-1, 2)).copy()
    if pts.size == 0:
        return pts.reshape((0, 2))
    pre = _preprocess_for_side(context, target_side)
    canonical_shape_hw = _normalize_shape_hw(pre.get("canonical_shape_hw"), (1, 1))
    physical_shape_hw = _normalize_shape_hw(pre.get("physical_normalized_shape_hw"), canonical_shape_hw)
    working_shape_hw = _normalize_shape_hw(pre.get("working_shape_hw"), physical_shape_hw)
    offset = dict(pre.get("common_canvas_offset_xy") or {})
    ox = float(offset.get("x", 0))
    oy = float(offset.get("y", 0))
    x, y, w, h = _bbox_tuple_from_manifest(pre.get("support_crop_bbox_xywh") or {}, canonical_shape_hw)
    pts[:, 0] -= ox
    pts[:, 1] -= oy
    scale_x_phys = float(physical_shape_hw[1]) / max(float(working_shape_hw[1]), 1.0)
    scale_y_phys = float(physical_shape_hw[0]) / max(float(working_shape_hw[0]), 1.0)
    pts[:, 0] *= scale_x_phys
    pts[:, 1] *= scale_y_phys
    scale_x_highres = float(w) / max(float(physical_shape_hw[1]), 1.0)
    scale_y_highres = float(h) / max(float(physical_shape_hw[0]), 1.0)
    pts[:, 0] = pts[:, 0] * scale_x_highres + float(x)
    pts[:, 1] = pts[:, 1] * scale_y_highres + float(y)
    return pts


def _map_highres_points_between_sides_mask_shape(
    context: ApprovedRegistrationContext,
    points_xy: np.ndarray,
    *,
    source_side: str,
) -> np.ndarray:
    source_side = _normalize_step6_side(source_side)
    target_side = _opposite_step6_side(source_side)
    matrix = context.approved_transform_matrix_2x3
    if matrix is None:
        raise FileNotFoundError("Approved mask-shape context is missing transform_matrix_2x3.")
    moving_to_fixed = np.asarray(matrix, dtype=np.float32)
    source_canvas = _highres_points_to_canvas(context, points_xy, source_side=source_side)
    if _role_for_side(context, source_side) == "moving":
        target_canvas = _apply_affine_to_points(source_canvas, moving_to_fixed)
    else:
        target_canvas = _apply_affine_to_points(source_canvas, _invert_affine_matrix_2x3(moving_to_fixed))
    return _canvas_points_to_highres(context, target_canvas, target_side=target_side)


def map_step7_scene_polygon_to_step6_side(
    context: ApprovedRegistrationContext,
    scene_polygon_xy: np.ndarray,
    *,
    step7_preview_shape_hw: tuple[int, int],
    step7_support_bbox_canvas_xywh: tuple[int, int, int, int],
    output_side: str,
    myelin_labels: np.ndarray | None = None,
) -> np.ndarray:
    output_side = _normalize_step6_side(output_side)
    poly = np.asarray(scene_polygon_xy, dtype=np.float32).reshape((-1, 2))
    if poly.shape[0] < 3:
        return np.zeros((0, 2), dtype=np.float32)
    preview_h = max(1, int(step7_preview_shape_hw[0]))
    preview_w = max(1, int(step7_preview_shape_hw[1]))
    x0, y0, w, h = [int(v) for v in step7_support_bbox_canvas_xywh]
    scale_x = float(w) / max(float(preview_w), 1.0)
    scale_y = float(h) / max(float(preview_h), 1.0)
    myelin_canvas_points = poly.copy()
    myelin_canvas_points[:, 0] = myelin_canvas_points[:, 0] * scale_x + float(x0)
    myelin_canvas_points[:, 1] = myelin_canvas_points[:, 1] * scale_y + float(y0)
    labels = (
        np.asarray(myelin_labels, dtype=np.uint8)
        if myelin_labels is not None
        else _load_registration_mask_labels(
            context.myelin_root / context.myelin_label,
            context.myelin_preprocess,
            _read_png_rgb(context.myelin_root / context.myelin_label / "crop_raw.png").shape[:2],
        )
    )
    myelin_points = _apply_registration_view_transform_to_points(
        myelin_canvas_points,
        labels,
        context.myelin_preprocess,
    )
    if output_side == "myelin":
        return myelin_points
    if context.registration_backend == "mask_shape":
        return _map_highres_points_between_sides_mask_shape(context, myelin_points, source_side="myelin")
    return np.zeros((0, 2), dtype=np.float32)


def map_step7_full_crop_polygon_to_step6_side(
    context: ApprovedRegistrationContext,
    full_crop_polygon_xy: np.ndarray,
    *,
    output_side: str,
    myelin_labels: np.ndarray | None = None,
) -> np.ndarray:
    output_side = _normalize_step6_side(output_side)
    poly = np.asarray(full_crop_polygon_xy, dtype=np.float32).reshape((-1, 2))
    if poly.shape[0] < 3:
        return np.zeros((0, 2), dtype=np.float32)
    labels = (
        np.asarray(myelin_labels, dtype=np.uint8)
        if myelin_labels is not None
        else _load_registration_mask_labels(
            context.myelin_root / context.myelin_label,
            context.myelin_preprocess,
            _read_png_rgb(context.myelin_root / context.myelin_label / "crop_raw.png").shape[:2],
        )
    )
    myelin_points = _apply_registration_view_transform_to_points(
        poly,
        labels,
        context.myelin_preprocess,
    )
    if output_side == "myelin":
        return myelin_points
    if context.registration_backend == "mask_shape":
        return _map_highres_points_between_sides_mask_shape(context, myelin_points, source_side="myelin")
    return np.zeros((0, 2), dtype=np.float32)


def map_step7_scene_mask_to_step6_side(
    context: ApprovedRegistrationContext,
    scene_mask_u8: np.ndarray,
    *,
    step7_preview_shape_hw: tuple[int, int],
    step7_support_bbox_canvas_xywh: tuple[int, int, int, int],
    output_side: str,
    ants_bin: Path,
) -> np.ndarray:
    output_side = _normalize_step6_side(output_side)
    scene_mask = np.where(np.asarray(scene_mask_u8, dtype=np.uint8) > 0, 255, 0).astype(np.uint8)
    preview_h = max(1, int(step7_preview_shape_hw[0]))
    preview_w = max(1, int(step7_preview_shape_hw[1]))
    if scene_mask.shape[:2] != (preview_h, preview_w):
        scene_mask = cv2.resize(scene_mask, (preview_w, preview_h), interpolation=cv2.INTER_NEAREST)

    myelin_section_dir = context.myelin_root / context.myelin_label
    myelin_labels = _load_registration_mask_labels(
        myelin_section_dir,
        context.myelin_preprocess,
        _read_png_rgb(myelin_section_dir / "crop_raw.png").shape[:2],
    )
    canvas_h, canvas_w = myelin_labels.shape[:2]
    x0, y0, w, h = [int(v) for v in step7_support_bbox_canvas_xywh]
    x0 = max(0, min(canvas_w, x0))
    y0 = max(0, min(canvas_h, y0))
    w = max(1, min(canvas_w - x0, w))
    h = max(1, min(canvas_h - y0, h))
    support_canvas_mask = cv2.resize(scene_mask, (w, h), interpolation=cv2.INTER_NEAREST)
    myelin_canvas_mask = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
    myelin_canvas_mask[y0 : y0 + h, x0 : x0 + w] = support_canvas_mask
    myelin_highres_mask = _apply_registration_view_transform_to_mask(
        myelin_canvas_mask,
        myelin_labels,
        context.myelin_preprocess,
        context.registration_mask_mode,
    )
    if output_side == "myelin":
        return myelin_highres_mask
    mapping = update_step6_roi_mapping(context, myelin_highres_mask, ants_bin, source_side="myelin")
    return np.asarray(mapping["target_roi_highres"], dtype=np.uint8)


def _load_saved_highres_roi(path: Path, shape_hw: tuple[int, int]) -> np.ndarray:
    return _read_png_gray(path, shape_hw=shape_hw) if path.exists() else np.zeros(shape_hw, dtype=np.uint8)


def current_step6_state(
    context: ApprovedRegistrationContext,
    *,
    source_side: str = "nissl",
) -> dict[str, Any]:
    source_side = _normalize_step6_side(source_side)
    target_side = _opposite_step6_side(source_side)
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
    source_rgb = nissl_rgb if source_side == "nissl" else myelin_rgb
    target_rgb = myelin_rgb if target_side == "myelin" else nissl_rgb
    source_roi = _load_saved_highres_roi(_roi_highres_path_for_side(context, source_side), source_rgb.shape[:2])
    target_roi = _load_saved_highres_roi(_roi_mapped_highres_path_for_side(context, target_side), target_rgb.shape[:2])
    source_overlay = _overlay_preview(source_rgb, source_roi, color_rgb=(255, 80, 80))
    target_overlay = _overlay_preview(target_rgb, target_roi, color_rgb=(80, 220, 255))
    return {
        "source_side": source_side,
        "target_side": target_side,
        "nissl_rgb": nissl_rgb,
        "myelin_rgb": myelin_rgb,
        "source_rgb": source_rgb,
        "target_rgb": target_rgb,
        "source_roi": source_roi,
        "target_roi": target_roi,
        "source_overlay": source_overlay,
        "target_overlay": target_overlay,
        "nissl_roi": source_roi if source_side == "nissl" else target_roi,
        "myelin_roi": source_roi if source_side == "myelin" else target_roi,
        "nissl_overlay": source_overlay if source_side == "nissl" else target_overlay,
        "myelin_overlay": source_overlay if source_side == "myelin" else target_overlay,
    }


def _highres_roi_to_canvas(
    context: ApprovedRegistrationContext,
    roi_highres: np.ndarray,
    *,
    source_side: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    pre = _preprocess_for_side(context, source_side)
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


def _canvas_roi_to_highres(
    context: ApprovedRegistrationContext,
    roi_canvas: np.ndarray,
    *,
    target_side: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    pre = _preprocess_for_side(context, target_side)
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


def _map_canvas_roi_between_sides(
    context: ApprovedRegistrationContext,
    source_roi_canvas: np.ndarray,
    ants_bin: Path,
    *,
    source_side: str,
) -> np.ndarray:
    source_side = _normalize_step6_side(source_side)
    target_side = _opposite_step6_side(source_side)
    if context.registration_backend == "mask_shape":
        matrix = context.approved_transform_matrix_2x3
        if matrix is None:
            raise FileNotFoundError("Approved mask-shape context is missing transform_matrix_2x3.")
        moving_to_fixed = np.asarray(matrix, dtype=np.float32)
        target_canvas_shape = _normalize_shape_hw(
            _preprocess_for_side(context, target_side).get("common_canvas_shape_hw"),
            source_roi_canvas.shape[:2],
        )
        if _role_for_side(context, source_side) == "moving":
            warp_mat = moving_to_fixed.astype(np.float32)
        else:
            warp_mat = _invert_affine_matrix_2x3(moving_to_fixed)
        warped = cv2.warpAffine(
            source_roi_canvas.astype(np.uint8),
            warp_mat,
            (int(target_canvas_shape[1]), int(target_canvas_shape[0])),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        return np.where(warped > 0, 255, 0).astype(np.uint8)

    work_dir = context.output_dir / "_work"
    work_dir.mkdir(parents=True, exist_ok=True)
    source_roi_nifti = work_dir / f"roi_labels_{source_side}_canvas.nii.gz"
    mapped_nifti = work_dir / f"roi_labels_{target_side}_canvas_mapped.nii.gz"

    source_ref = _ref_nifti_for_side(context, source_side)
    target_ref = _ref_nifti_for_side(context, target_side)
    if source_ref is None or target_ref is None:
        raise FileNotFoundError("Approved ANTs context is missing NIfTI references.")
    ref = nib.load(str(source_ref))
    nib.save(nib.Nifti1Image(source_roi_canvas.astype(np.float32), ref.affine, ref.header), str(source_roi_nifti))

    if _role_for_side(context, source_side) == "fixed":
        transforms = _inverse_transform_specs_for_stage(context.run_dir, context.manifest, context.approved_stage)
    else:
        transforms = _forward_transform_specs_for_stage(context.run_dir, context.manifest, context.approved_stage)

    cmd = [
        str(ants_bin / "antsApplyTransforms"),
        "-d",
        "2",
        "-i",
        ants_cli_path(source_roi_nifti),
        "-r",
        ants_cli_path(target_ref),
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
    source_roi_highres: np.ndarray,
    ants_bin: Path,
    *,
    source_side: str = "nissl",
) -> dict[str, Any]:
    source_side = _normalize_step6_side(source_side)
    target_side = _opposite_step6_side(source_side)
    source_canvas, source_debug = _highres_roi_to_canvas(context, source_roi_highres, source_side=source_side)
    target_canvas = _map_canvas_roi_between_sides(context, source_canvas, ants_bin, source_side=source_side)
    target_highres, target_debug = _canvas_roi_to_highres(context, target_canvas, target_side=target_side)
    return {
        "source_side": source_side,
        "target_side": target_side,
        "source_roi_highres": source_roi_highres.astype(np.uint8),
        "source_roi_canvas": source_canvas.astype(np.uint8),
        "target_roi_canvas": target_canvas.astype(np.uint8),
        "target_roi_highres": target_highres.astype(np.uint8),
        "source_debug": source_debug,
        "target_debug": target_debug,
    }


def save_step6_roi(
    context: ApprovedRegistrationContext,
    mapping_result: dict[str, Any],
) -> dict[str, Any]:
    source_side = _normalize_step6_side(str(mapping_result.get("source_side") or "nissl"))
    target_side = _opposite_step6_side(source_side)
    context.output_dir.mkdir(parents=True, exist_ok=True)
    state = current_step6_state(context, source_side=source_side)
    source_rgb = np.asarray(state["source_rgb"], dtype=np.uint8)
    target_rgb = np.asarray(state["target_rgb"], dtype=np.uint8)
    source_roi_highres = np.asarray(mapping_result["source_roi_highres"], dtype=np.uint8)
    source_roi_canvas = np.asarray(mapping_result["source_roi_canvas"], dtype=np.uint8)
    target_roi_canvas = np.asarray(mapping_result["target_roi_canvas"], dtype=np.uint8)
    target_roi_highres = np.asarray(mapping_result["target_roi_highres"], dtype=np.uint8)
    source_overlay = _overlay_preview(source_rgb, source_roi_highres, color_rgb=(255, 80, 80))
    target_overlay = _overlay_preview(target_rgb, target_roi_highres, color_rgb=(80, 220, 255))

    source_highres_path = _roi_highres_path_for_side(context, source_side)
    source_canvas_path = _roi_canvas_path_for_side(context, source_side)
    target_canvas_path = _roi_canvas_path_for_side(context, target_side)
    target_highres_path = _roi_mapped_highres_path_for_side(context, target_side)
    source_overlay_path = _roi_overlay_path_for_side(context, source_side)
    target_overlay_path = _roi_overlay_path_for_side(context, target_side)

    write_png_lossless_fast(source_highres_path, source_roi_highres, "L")
    write_png_lossless_fast(source_canvas_path, source_roi_canvas, "L")
    write_png_lossless_fast(target_canvas_path, target_roi_canvas, "L")
    write_png_lossless_fast(target_highres_path, target_roi_highres, "L")
    write_png_lossless_fast(source_overlay_path, source_overlay.astype(np.uint8), "RGB")
    write_png_lossless_fast(target_overlay_path, target_overlay.astype(np.uint8), "RGB")

    payload = {
        "pair_key": context.pair_key,
        "group_tag": context.group_tag,
        "approved_stage": context.approved_stage,
        "approved_run_dir": str(context.run_dir),
        "approved_manifest_path": str(context.manifest_path),
        "registration_backend": context.registration_backend,
        "source_side": source_side,
        "target_side": target_side,
        "nissl_role": context.nissl_role,
        "myelin_role": context.myelin_role,
        "registration_mask_mode": context.registration_mask_mode,
        "saved_at_utc": _utc_now_iso(),
        "nissl_preprocess": context.nissl_preprocess,
        "myelin_preprocess": context.myelin_preprocess,
        "source_mapping_debug": mapping_result.get("source_debug", {}),
        "target_mapping_debug": mapping_result.get("target_debug", {}),
        "files": {
            "source_roi_highres": str(source_highres_path),
            "source_roi_canvas": str(source_canvas_path),
            "target_roi_canvas": str(target_canvas_path),
            "target_roi_mapped_highres": str(target_highres_path),
            "source_overlay": str(source_overlay_path),
            "target_overlay": str(target_overlay_path),
            str(source_highres_path.name): str(source_highres_path),
            str(source_canvas_path.name): str(source_canvas_path),
            str(target_canvas_path.name): str(target_canvas_path),
            str(target_highres_path.name): str(target_highres_path),
            str(source_overlay_path.name): str(source_overlay_path),
            str(target_overlay_path.name): str(target_overlay_path),
        },
    }
    _write_json(context.output_dir / "roi_manifest.json", payload)
    return payload

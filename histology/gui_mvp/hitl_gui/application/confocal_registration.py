from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .pair_registration import (
    _compute_stage_heatmap,
    _run_logged,
    _write_coord_images,
    ants_cli_path,
    compute_registration_metrics,
    gray_preview_panel,
    metrics_note,
    overlay_preview,
    read_nifti_2d,
    render_storyboard,
    rgb_to_gray_float,
    write_nifti_2d,
)
from .section_workspace import WorkspaceSection, load_workspace_section


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def default_confocal_registration_root(myelin_root: Path | None) -> Path | None:
    if myelin_root is None:
        return None
    resolved = myelin_root.resolve()
    parents = resolved.parents
    common = parents[1] if len(parents) >= 2 else resolved.parent
    return common / "confocal_myelin_registration"


def _read_tiff_stack(path: Path) -> np.ndarray:
    try:
        import tifffile
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("tifffile is required for confocal z-stack loading.") from exc
    arr = tifffile.imread(str(path))
    if arr.ndim < 3:
        raise ValueError(f"Confocal stack must be 3D/4D, got shape={arr.shape}")
    return np.asarray(arr)


def infer_stack_channel_count(stack: np.ndarray) -> int:
    if stack.ndim == 3:
        return 1
    if stack.ndim != 4:
        return 1
    if stack.shape[-1] <= 4:
        return int(stack.shape[-1])
    if stack.shape[1] <= 4:
        return int(stack.shape[1])
    return 1


def extract_stack_channel(stack: np.ndarray, channel_index: int = 0) -> np.ndarray:
    if stack.ndim == 3:
        return stack.astype(np.float32)
    if stack.ndim != 4:
        raise ValueError(f"Unsupported confocal stack shape: {stack.shape}")
    if stack.shape[-1] <= 4:
        idx = max(0, min(int(channel_index), int(stack.shape[-1]) - 1))
        return stack[..., idx].astype(np.float32)
    if stack.shape[1] <= 4:
        idx = max(0, min(int(channel_index), int(stack.shape[1]) - 1))
        return stack[:, idx, ...].astype(np.float32)
    return stack[:, 0, ...].astype(np.float32)


def _normalize_u8(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    lo = float(arr.min())
    hi = float(arr.max())
    if hi <= lo:
        return np.zeros(arr.shape[:2], dtype=np.uint8)
    out = (arr - lo) / (hi - lo)
    return np.clip(np.round(out * 255.0), 0, 255).astype(np.uint8)


def project_confocal_stack(stack: np.ndarray, *, mode: str = "focus", channel_index: int = 0) -> np.ndarray:
    vol = extract_stack_channel(stack, channel_index=channel_index)
    if mode == "max":
        return _normalize_u8(np.max(vol, axis=0))
    if mode == "mean":
        return _normalize_u8(np.mean(vol, axis=0))
    if mode not in {"focus", "edf"}:
        raise ValueError(f"Unsupported confocal projection mode: {mode}")
    # Simple EDF: choose each pixel from the slice with the highest local Laplacian magnitude.
    focus_maps = []
    for z in range(vol.shape[0]):
        plane_u8 = _normalize_u8(vol[z])
        lap = cv2.Laplacian(plane_u8, cv2.CV_32F, ksize=3)
        focus_maps.append(cv2.GaussianBlur(np.abs(lap), (0, 0), sigmaX=1.0))
    focus_stack = np.stack(focus_maps, axis=0)
    best_idx = np.argmax(focus_stack, axis=0)
    yy, xx = np.indices(best_idx.shape)
    best = vol[best_idx, yy, xx]
    return _normalize_u8(best)


def prepare_myelin_confocal_fixed(item: WorkspaceSection) -> tuple[np.ndarray, np.ndarray]:
    _metadata, crop_rgb, tissue, artifact, _source = load_workspace_section(item)
    support = ((tissue > 0) | (artifact > 0))
    ys, xs = np.where(support)
    if ys.size == 0 or xs.size == 0:
        return crop_rgb, np.zeros(crop_rgb.shape[:2], dtype=np.uint8)
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    rgb = crop_rgb[y1:y2, x1:x2].copy()
    labels = np.where(tissue[y1:y2, x1:x2] > 0, 1, np.where(artifact[y1:y2, x1:x2] > 0, 2, 0)).astype(np.uint8)
    rgb[labels <= 0] = 255
    return rgb, labels


def build_manual_affine(
    moving_shape_hw: tuple[int, int],
    fixed_shape_hw: tuple[int, int],
    *,
    tx_px: float = 0.0,
    ty_px: float = 0.0,
    angle_deg: float = 0.0,
    scale: float = 1.0,
    flip_lr: bool = False,
    flip_ud: bool = False,
) -> np.ndarray:
    mh, mw = moving_shape_hw
    fh, fw = fixed_shape_hw
    src_center = np.array([mw / 2.0, mh / 2.0], dtype=np.float32)
    dst_center = np.array([fw / 2.0 + float(tx_px), fh / 2.0 + float(ty_px)], dtype=np.float32)

    theta = np.deg2rad(float(angle_deg))
    c = float(np.cos(theta) * scale)
    s = float(np.sin(theta) * scale)
    flip_x = -1.0 if flip_lr else 1.0
    flip_y = -1.0 if flip_ud else 1.0
    linear = np.array([[c * flip_x, s], [-s, c * flip_y]], dtype=np.float32)
    trans = dst_center - linear @ src_center
    mat = np.concatenate([linear, trans[:, None]], axis=1)
    return mat.astype(np.float32)


def apply_manual_transform(
    moving_gray_u8: np.ndarray,
    fixed_shape_hw: tuple[int, int],
    *,
    tx_px: float = 0.0,
    ty_px: float = 0.0,
    angle_deg: float = 0.0,
    scale: float = 1.0,
    flip_lr: bool = False,
    flip_ud: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mat = build_manual_affine(
        moving_gray_u8.shape[:2],
        fixed_shape_hw,
        tx_px=tx_px,
        ty_px=ty_px,
        angle_deg=angle_deg,
        scale=scale,
        flip_lr=flip_lr,
        flip_ud=flip_ud,
    )
    out_w = int(fixed_shape_hw[1])
    out_h = int(fixed_shape_hw[0])
    warped = cv2.warpAffine(
        moving_gray_u8,
        mat,
        (out_w, out_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    moving_mask = np.where(moving_gray_u8 > 0, 255, 0).astype(np.uint8)
    warped_mask = cv2.warpAffine(
        moving_mask,
        mat,
        (out_w, out_h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return warped, warped_mask, mat


def _stage_command_rigid(
    ants_bin: Path,
    fixed_path: Path,
    moving_path: Path,
    fixed_mask_path: Path,
    moving_mask_path: Path,
    prefix: Path,
) -> list[str]:
    return [
        str(ants_bin / "antsRegistration"),
        "-d",
        "2",
        "-o",
        ants_cli_path(prefix),
        "-r",
        f"[{ants_cli_path(fixed_path)},{ants_cli_path(moving_path)},1]",
        "-m",
        f"MI[{ants_cli_path(fixed_path)},{ants_cli_path(moving_path)},1,32,Regular,0.25]",
        "-t",
        "Rigid[0.1]",
        "-c",
        "[300x150x80,1e-6,10]",
        "-s",
        "3x2x1vox",
        "-f",
        "8x4x2",
        "-x",
        f"[{ants_cli_path(fixed_mask_path)},{ants_cli_path(moving_mask_path)}]",
    ]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


@dataclass
class ConfocalRigidConfig:
    myelin_label: str
    myelin_rgb: np.ndarray
    myelin_labels: np.ndarray
    confocal_projection_u8: np.ndarray
    ants_bin: Path
    out_root: Path
    confocal_source: Path
    projection_mode: str
    channel_index: int
    tx_px: float = 0.0
    ty_px: float = 0.0
    angle_deg: float = 0.0
    scale: float = 1.0
    flip_lr: bool = False
    flip_ud: bool = False


def run_confocal_rigid_registration(cfg: ConfocalRigidConfig) -> dict[str, Any]:
    session_id = f"{_utc_stamp()}_{cfg.projection_mode}_ch{cfg.channel_index}"
    run_dir = cfg.out_root / cfg.myelin_label / session_id
    inputs_dir = run_dir / "inputs"
    stage_dir = run_dir / "rigid"
    run_dir.mkdir(parents=True, exist_ok=True)
    inputs_dir.mkdir(parents=True, exist_ok=True)
    stage_dir.mkdir(parents=True, exist_ok=True)

    fixed_rgb = cfg.myelin_rgb.copy()
    fixed_gray = rgb_to_gray_float(fixed_rgb)
    fixed_mask = (cfg.myelin_labels == 1).astype(np.float32)
    if not np.any(fixed_mask > 0):
        fixed_mask = (cfg.myelin_labels > 0).astype(np.float32)

    manual_gray_u8, manual_mask_u8, manual_mat = apply_manual_transform(
        cfg.confocal_projection_u8,
        fixed_rgb.shape[:2],
        tx_px=cfg.tx_px,
        ty_px=cfg.ty_px,
        angle_deg=cfg.angle_deg,
        scale=cfg.scale,
        flip_lr=cfg.flip_lr,
        flip_ud=cfg.flip_ud,
    )
    moving_gray = manual_gray_u8.astype(np.float32) / 255.0
    moving_mask = (manual_mask_u8 > 0).astype(np.float32)

    input_metrics, input_metric_timings = compute_registration_metrics(fixed_gray, moving_gray, fixed_mask, moving_mask)
    input_note = metrics_note(input_metrics, input_metric_timings, "manual init")

    cv2.imwrite(str(inputs_dir / "myelin_fixed.png"), cv2.cvtColor(fixed_rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(inputs_dir / "confocal_projection.png"), cfg.confocal_projection_u8)
    cv2.imwrite(str(inputs_dir / "confocal_manual_warped.png"), manual_gray_u8)
    cv2.imwrite(str(inputs_dir / "myelin_labels.png"), cfg.myelin_labels)
    cv2.imwrite(str(inputs_dir / "confocal_manual_mask.png"), manual_mask_u8)

    fixed_img_path = inputs_dir / "fixed_gray.nii.gz"
    moving_img_path = inputs_dir / "moving_gray.nii.gz"
    fixed_mask_path = inputs_dir / "fixed_mask.nii.gz"
    moving_mask_path = inputs_dir / "moving_mask.nii.gz"
    write_nifti_2d(fixed_img_path, fixed_gray)
    write_nifti_2d(moving_img_path, moving_gray)
    write_nifti_2d(fixed_mask_path, fixed_mask)
    write_nifti_2d(moving_mask_path, moving_mask)
    moving_coord_x, moving_coord_y = _write_coord_images(inputs_dir, moving_gray.shape[:2])

    storyboard_path = run_dir / "storyboard.png"
    blank = np.full((*fixed_gray.shape, 3), 235, dtype=np.uint8)
    render_storyboard(
        [
            {
                "label": "Input",
                "note": input_note,
                "fixed": gray_preview_panel(fixed_gray),
                "moving": gray_preview_panel(moving_gray),
                "overlay": overlay_preview(fixed_gray, moving_gray, fixed_mask, moving_mask),
                "heatmap": blank,
                "col_titles": ("Moving", "Fixed", "Overlay", "Warp Field"),
            }
        ],
        storyboard_path,
    )

    t0 = time.perf_counter()
    prefix = stage_dir / "rigid_"
    cmd = _stage_command_rigid(cfg.ants_bin, fixed_img_path, moving_img_path, fixed_mask_path, moving_mask_path, prefix)
    ants_t0 = time.perf_counter()
    _run_logged(cmd, stage_dir / "rigid.log")
    ants_seconds = float(time.perf_counter() - ants_t0)

    rigid_mat = stage_dir / "rigid_0GenericAffine.mat"
    warped_img_path = stage_dir / "rigid_Warped.nii.gz"
    warped_mask_path = stage_dir / "rigid_warped_mask.nii.gz"
    subprocess.run(
        [
            str(cfg.ants_bin / "antsApplyTransforms"),
            "-d",
            "2",
            "-i",
            ants_cli_path(moving_mask_path),
            "-r",
            ants_cli_path(fixed_img_path),
            "-o",
            ants_cli_path(warped_mask_path),
            "-n",
            "NearestNeighbor",
            "-t",
            ants_cli_path(rigid_mat),
        ],
        check=True,
        stdout=(stage_dir / "rigid_warp_mask.log").open("w", encoding="utf-8"),
        stderr=subprocess.STDOUT,
    )

    warped_gray = read_nifti_2d(warped_img_path)
    warped_mask = read_nifti_2d(warped_mask_path)
    rigid_metrics, rigid_metric_timing = compute_registration_metrics(
        fixed_gray,
        np.clip(warped_gray, 0.0, 1.0),
        fixed_mask,
        (warped_mask > 0.5).astype(np.float32),
    )
    overlay = overlay_preview(
        fixed_gray,
        np.clip(warped_gray, 0.0, 1.0),
        fixed_mask,
        (warped_mask > 0.5).astype(np.float32),
    )
    heatmap_rgb, heatmap_png = _compute_stage_heatmap(
        cfg.ants_bin,
        stage_dir,
        "rigid",
        fixed_img_path,
        fixed_mask,
        moving_coord_x,
        moving_coord_y,
        rigid_mat,
        stage_dir / "unused_affine.mat",
        warped_mask_path,
    )
    total_seconds = float(time.perf_counter() - t0)
    render_storyboard(
        [
            {
                "label": "Input",
                "note": input_note,
                "fixed": gray_preview_panel(fixed_gray),
                "moving": gray_preview_panel(moving_gray),
                "overlay": overlay_preview(fixed_gray, moving_gray, fixed_mask, moving_mask),
                "heatmap": blank,
                "col_titles": ("Moving", "Fixed", "Overlay", "Warp Field"),
            },
            {
                "label": "Rigid",
                "note": metrics_note(rigid_metrics, rigid_metric_timing, "rigid finished"),
                "fixed": gray_preview_panel(fixed_gray),
                "moving": gray_preview_panel(np.clip(warped_gray, 0.0, 1.0)),
                "overlay": overlay,
                "heatmap": heatmap_rgb,
                "col_titles": ("Moving", "Fixed", "Overlay", "Warp Field"),
            },
        ],
        storyboard_path,
    )

    manifest = {
        "myelin_label": cfg.myelin_label,
        "confocal_source": str(cfg.confocal_source),
        "projection_mode": cfg.projection_mode,
        "channel_index": int(cfg.channel_index),
        "saved_at_utc": _utc_iso(),
        "manual_init": {
            "tx_px": float(cfg.tx_px),
            "ty_px": float(cfg.ty_px),
            "angle_deg": float(cfg.angle_deg),
            "scale": float(cfg.scale),
            "flip_lr": bool(cfg.flip_lr),
            "flip_ud": bool(cfg.flip_ud),
            "affine_matrix_2x3": manual_mat.tolist(),
        },
        "input_metrics": input_metrics,
        "input_metric_timing_seconds": input_metric_timings,
        "rigid_metrics": rigid_metrics,
        "rigid_metric_timing_seconds": rigid_metric_timing,
        "timing_seconds": {
            "ants_registration": ants_seconds,
            "total": total_seconds,
        },
        "files": {
            "myelin_fixed": str(inputs_dir / "myelin_fixed.png"),
            "confocal_projection": str(inputs_dir / "confocal_projection.png"),
            "confocal_manual_warped": str(inputs_dir / "confocal_manual_warped.png"),
            "storyboard": str(storyboard_path),
            "rigid_heatmap": str(heatmap_png),
            "rigid_transform": str(rigid_mat),
            "manifest": str(run_dir / "session_manifest.json"),
        },
    }
    _write_json(run_dir / "session_manifest.json", manifest)
    return manifest

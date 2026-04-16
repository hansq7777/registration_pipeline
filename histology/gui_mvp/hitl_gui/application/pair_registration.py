from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import cv2
import nibabel as nib
import numpy as np

from .physical_provenance import recover_or_load_section_physical_provenance


ProgressCallback = Callable[[dict[str, Any]], None]
REGISTRATION_MAX_LONG_EDGE = 1024
DEFAULT_TARGET_UM_PER_PX = 10.0
STRICT_PARETO_GATING_POLICY_NAME = "strict_pareto_geometry_v1"
STRICT_PARETO_MAXIMIZE_KEYS = ("dice",)
STRICT_PARETO_MINIMIZE_KEYS = ("hd95_px",)
STRICT_PARETO_DELTA_KEYS = ("dice", "hd95_px", "assd_px", "mi", "cc", "iou", "precision", "recall")
ANTS_REGISTRATION_BACKEND = "ants"
MASK_SHAPE_REGISTRATION_BACKEND = "mask_shape"
ANTS_STAGE_ORDER = ("rigid", "affine", "syn")
MASK_SHAPE_STAGE_ORDER = ("mask_rigid", "mask_similarity")
MASK_SHAPE_OPTIMIZATION_LONG_EDGE = 512
MASK_SHAPE_TRANSLATION_STEPS_FRACTION = (0.06, 0.03, 0.012, 0.004)
MASK_SHAPE_ANGLE_STEPS_DEG = (4.0, 2.0, 0.8, 0.3)
MASK_SHAPE_SCALE_STEPS = (0.03, 0.015, 0.0075, 0.003)
MASK_SHAPE_OBJECTIVE_NAME = "support_dice_plus_boundary_dt"
MASK_SHAPE_BOUNDARY_WEIGHT = 0.25
AFFINE_PROFILES: dict[str, dict[str, Any]] = {
    "current": {
        "metric_bins": 32,
        "metric_sampling": 0.25,
        "transform": "Affine[0.08]",
        "convergence": "[300x150x80,1e-6,10]",
        "smoothing_sigmas": "3x2x1vox",
        "shrink_factors": "8x4x2",
    },
    "stronger": {
        "metric_bins": 48,
        "metric_sampling": 0.35,
        "transform": "Affine[0.12]",
        "convergence": "[500x300x160x80,1e-7,15]",
        "smoothing_sigmas": "4x3x2x1vox",
        "shrink_factors": "12x8x4x2",
    },
}


def _utc_now_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def default_pair_registration_runs_root(myelin_root: Path | None, nissl_root: Path | None) -> Path | None:
    roots = [p for p in (myelin_root, nissl_root) if p is not None]
    if not roots:
        return None
    common = Path(os.path.commonpath([str(p.resolve()) for p in roots]))
    return common / "histology_pair_registration_runs"


def resolve_pair_registry_relpath(common_root: Path, relpath: str | None) -> Path | None:
    if relpath is None:
        return None
    rel_norm = str(relpath).replace("\\", "/").lstrip("/")
    if not rel_norm:
        return None
    return common_root / Path(rel_norm)


def find_ants_bin() -> Path | None:
    candidates = [
        Path("/mnt/c/tools/ANTs/ants-2.6.5/bin"),
        Path("C:/tools/ANTs/ants-2.6.5/bin"),
    ]
    for cand in candidates:
        for name in ("antsRegistration.exe", "antsRegistration"):
            try:
                if (cand / name).exists():
                    return cand
            except OSError:
                continue
    return None


def ants_cli_path(path: Path | str) -> str:
    p = Path(path)
    s = str(p)
    if s.startswith("/mnt/") and len(s) > 6:
        drive = s[5].upper()
        tail = s[6:].replace("/", "\\").lstrip("\\")
        return f"{drive}:\\{tail}"
    return s


def ants_binary_path(ants_bin: Path, stem: str) -> Path:
    for name in (f"{stem}.exe", stem):
        candidate = ants_bin / name
        try:
            if candidate.exists():
                return candidate
        except OSError:
            continue
    return ants_bin / stem


def latest_registration_run_dir(runs_root: Path | None, pair_key: str) -> Path | None:
    if runs_root is None:
        return None
    pair_dir = runs_root / pair_key
    if not pair_dir.exists():
        return None
    candidates = sorted(
        (p for p in pair_dir.iterdir() if p.is_dir()),
        key=lambda p: p.stat().st_mtime,
    )
    return candidates[-1] if candidates else None


def load_mask_labels(path: Path) -> np.ndarray:
    arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise FileNotFoundError(path)
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr.astype(np.uint8)


def component_rank_map(mask_labels: np.ndarray) -> tuple[np.ndarray, dict[int, int]]:
    union = (mask_labels > 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(union, connectivity=8)
    entries: list[tuple[int, int]] = []
    for label_idx in range(1, num_labels):
        area = int(stats[label_idx, cv2.CC_STAT_AREA])
        if area > 0:
            entries.append((label_idx, area))
    entries.sort(key=lambda x: x[1], reverse=True)
    rank_to_label = {rank: label for rank, (label, _) in enumerate(entries, start=1)}
    return labels, rank_to_label


def apply_whole_flip(rgb: np.ndarray, labels: np.ndarray, enabled: bool) -> tuple[np.ndarray, np.ndarray]:
    if not enabled:
        return rgb, labels
    return rgb[:, ::-1, :].copy(), labels[:, ::-1].copy()


def apply_group_flips(
    rgb: np.ndarray,
    labels: np.ndarray,
    component_groups: dict[str, list[int]] | None,
    group_flip_lr: dict[str, bool] | None,
) -> tuple[np.ndarray, np.ndarray]:
    if not component_groups or not group_flip_lr:
        return rgb, labels
    labels_cc, rank_to_label = component_rank_map(labels)
    out_rgb = rgb.copy()
    out_labels = labels.copy()
    for raw_group_id, raw_enabled in dict(group_flip_lr).items():
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
            out_rgb[y1:y2, x1:x2] = out_rgb[y1:y2, x1:x2][:, ::-1, :]
            out_labels[y1:y2, x1:x2] = out_labels[y1:y2, x1:x2][:, ::-1]
            labels_cc[y1:y2, x1:x2] = labels_cc[y1:y2, x1:x2][:, ::-1]
    return out_rgb, out_labels


def keep_group(mask_labels: np.ndarray, component_groups: dict[str, list[int]] | None, group_choice: str) -> np.ndarray:
    group_choice = str(group_choice).strip().lower()
    if group_choice in {"all", "", "0"}:
        return mask_labels.copy()

    labels_cc, rank_to_label = component_rank_map(mask_labels)
    keep_ranks: list[int]
    if component_groups and str(group_choice) in component_groups:
        keep_ranks = [int(x) for x in component_groups.get(str(group_choice), [])]
    elif group_choice == "1":
        keep_ranks = [1]
    else:
        keep_ranks = []
    keep_labels = {rank_to_label.get(rank) for rank in keep_ranks}
    keep_labels.discard(None)
    if not keep_labels:
        raise ValueError(f"Selected group {group_choice} is not available in the current registration mask.")
    keep_mask = np.isin(labels_cc, list(keep_labels))
    out = mask_labels.copy()
    out[~keep_mask] = 0
    return out


def crop_to_union(rgb: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    union = labels > 0
    ys, xs = np.where(union)
    if ys.size == 0 or xs.size == 0:
        return rgb.copy(), labels.copy()
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    rgb_crop = rgb[y1:y2, x1:x2].copy()
    labels_crop = labels[y1:y2, x1:x2].copy()
    rgb_crop[labels_crop <= 0] = 255
    return rgb_crop, labels_crop


def registration_support_mask(labels: np.ndarray, mode: str) -> np.ndarray:
    mode_norm = str(mode).strip().lower()
    if mode_norm == "union":
        return labels > 0
    if mode_norm in {"tissue", "tissue_only", "tissue-only"}:
        return labels == 1
    raise ValueError(f"Unknown registration_mask_mode: {mode}")


def _bbox_from_mask(mask: np.ndarray) -> dict[str, int]:
    ys, xs = np.where(mask > 0)
    if ys.size == 0 or xs.size == 0:
        return {"x": 0, "y": 0, "w": int(mask.shape[1]), "h": int(mask.shape[0])}
    x1 = int(xs.min())
    y1 = int(ys.min())
    x2 = int(xs.max()) + 1
    y2 = int(ys.max()) + 1
    return {"x": x1, "y": y1, "w": int(x2 - x1), "h": int(y2 - y1)}


def crop_to_support(rgb: np.ndarray, labels: np.ndarray, support_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ys, xs = np.where(support_mask > 0)
    if ys.size == 0 or xs.size == 0:
        return rgb.copy(), labels.copy()
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    rgb_crop = rgb[y1:y2, x1:x2].copy()
    labels_crop = labels[y1:y2, x1:x2].copy()
    support_crop = support_mask[y1:y2, x1:x2].astype(bool)
    rgb_crop[~support_crop] = 255
    return rgb_crop, labels_crop


def center_on_canvas(rgb: np.ndarray, labels: np.ndarray, out_shape_hw: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    out_h, out_w = out_shape_hw
    in_h, in_w = rgb.shape[:2]
    if in_h > out_h or in_w > out_w:
        raise ValueError(f"Input shape {rgb.shape[:2]} does not fit canvas {out_shape_hw}")
    y0 = (out_h - in_h) // 2
    x0 = (out_w - in_w) // 2
    out_rgb = np.full((out_h, out_w, 3), 255, dtype=np.uint8)
    out_labels = np.zeros((out_h, out_w), dtype=np.uint8)
    out_rgb[y0 : y0 + in_h, x0 : x0 + in_w] = rgb
    out_labels[y0 : y0 + in_h, x0 : x0 + in_w] = labels
    return out_rgb, out_labels


def _center_on_canvas_with_offset(
    rgb: np.ndarray,
    labels: np.ndarray,
    out_shape_hw: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    out_h, out_w = out_shape_hw
    in_h, in_w = rgb.shape[:2]
    if in_h > out_h or in_w > out_w:
        raise ValueError(f"Input shape {rgb.shape[:2]} does not fit canvas {out_shape_hw}")
    y0 = (out_h - in_h) // 2
    x0 = (out_w - in_w) // 2
    out_rgb, out_labels = center_on_canvas(rgb, labels, out_shape_hw)
    return out_rgb, out_labels, {"x": int(x0), "y": int(y0)}


def _apply_gaussian_blur_preserving_background(rgb: np.ndarray, labels: np.ndarray, sigma: float) -> np.ndarray:
    sigma = float(sigma)
    if sigma <= 0.0:
        return rgb
    blurred = cv2.GaussianBlur(rgb, (0, 0), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REPLICATE)
    out = blurred.copy()
    out[labels <= 0] = 255
    return out


def rgb_to_gray_float(rgb: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    if gray.max() > gray.min():
        gray = (gray - gray.min()) / float(gray.max() - gray.min())
    return gray.astype(np.float32)


def maybe_downsample_pair(
    fixed_rgb: np.ndarray,
    fixed_labels: np.ndarray,
    moving_rgb: np.ndarray,
    moving_labels: np.ndarray,
    *,
    max_long_edge: int = REGISTRATION_MAX_LONG_EDGE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    long_edge = max(fixed_rgb.shape[0], fixed_rgb.shape[1], moving_rgb.shape[0], moving_rgb.shape[1])
    if max_long_edge <= 0 or long_edge <= max_long_edge:
        return fixed_rgb, fixed_labels, moving_rgb, moving_labels, 1.0
    scale = float(max_long_edge) / float(long_edge)

    def _resize_rgb(arr: np.ndarray) -> np.ndarray:
        return cv2.resize(
            arr,
            (max(1, int(round(arr.shape[1] * scale))), max(1, int(round(arr.shape[0] * scale)))),
            interpolation=cv2.INTER_AREA,
        )

    def _resize_lab(arr: np.ndarray) -> np.ndarray:
        return cv2.resize(
            arr,
            (max(1, int(round(arr.shape[1] * scale))), max(1, int(round(arr.shape[0] * scale)))),
            interpolation=cv2.INTER_NEAREST,
        )

    return (
        _resize_rgb(fixed_rgb),
        _resize_lab(fixed_labels),
        _resize_rgb(moving_rgb),
        _resize_lab(moving_labels),
        scale,
    )


def stage_order_for_backend(registration_backend: str) -> tuple[str, ...]:
    backend = str(registration_backend).strip().lower() or ANTS_REGISTRATION_BACKEND
    if backend == ANTS_REGISTRATION_BACKEND:
        return ANTS_STAGE_ORDER
    if backend == MASK_SHAPE_REGISTRATION_BACKEND:
        return MASK_SHAPE_STAGE_ORDER
    raise ValueError(f"Unsupported registration_backend: {registration_backend}")


def stage_display_name(stage: str) -> str:
    labels = {
        "rigid": "Rigid",
        "affine": "Affine",
        "syn": "SyN",
        "mask_rigid": "Mask Rigid",
        "mask_similarity": "Mask Similarity",
    }
    return labels.get(str(stage), str(stage).replace("_", " ").title())


def build_affine_matrix(
    moving_shape_hw: tuple[int, int],
    fixed_shape_hw: tuple[int, int],
    *,
    tx_px: float = 0.0,
    ty_px: float = 0.0,
    angle_deg: float = 0.0,
    scale: float = 1.0,
) -> np.ndarray:
    mh, mw = moving_shape_hw
    fh, fw = fixed_shape_hw
    src_center = np.array([mw / 2.0, mh / 2.0], dtype=np.float32)
    dst_center = np.array([fw / 2.0 + float(tx_px), fh / 2.0 + float(ty_px)], dtype=np.float32)
    theta = np.deg2rad(float(angle_deg))
    c = float(np.cos(theta) * scale)
    s = float(np.sin(theta) * scale)
    linear = np.array([[c, s], [-s, c]], dtype=np.float32)
    trans = dst_center - linear @ src_center
    return np.concatenate([linear, trans[:, None]], axis=1).astype(np.float32)


def _warp_gray_affine(moving_gray: np.ndarray, mat: np.ndarray, fixed_shape_hw: tuple[int, int]) -> np.ndarray:
    out_h, out_w = fixed_shape_hw
    return cv2.warpAffine(
        moving_gray.astype(np.float32),
        mat.astype(np.float32),
        (int(out_w), int(out_h)),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=1.0,
    ).astype(np.float32)


def _warp_mask_affine(moving_mask: np.ndarray, mat: np.ndarray, fixed_shape_hw: tuple[int, int]) -> np.ndarray:
    out_h, out_w = fixed_shape_hw
    warped = cv2.warpAffine(
        np.where(moving_mask > 0, 255, 0).astype(np.uint8),
        mat.astype(np.float32),
        (int(out_w), int(out_h)),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return warped > 0


def _resize_mask_for_shape_optimization(mask: np.ndarray, max_long_edge: int) -> tuple[np.ndarray, float]:
    h, w = mask.shape[:2]
    long_edge = max(h, w)
    if max_long_edge <= 0 or long_edge <= max_long_edge:
        return (mask > 0), 1.0
    scale = float(max_long_edge) / float(long_edge)
    out_w = max(1, int(round(w * scale)))
    out_h = max(1, int(round(h * scale)))
    resized = cv2.resize(np.where(mask > 0, 255, 0).astype(np.uint8), (out_w, out_h), interpolation=cv2.INTER_NEAREST)
    return resized > 0, scale


def _wrap_angle_deg(angle_deg: float) -> float:
    wrapped = ((float(angle_deg) + 180.0) % 360.0) - 180.0
    if wrapped <= -180.0:
        wrapped += 360.0
    return float(wrapped)


def _wrap_line_angle_deg(angle_deg: float) -> float:
    wrapped = _wrap_angle_deg(angle_deg)
    if wrapped > 90.0:
        wrapped -= 180.0
    if wrapped <= -90.0:
        wrapped += 180.0
    return float(wrapped)


def _mask_centroid_xy(mask: np.ndarray) -> tuple[float, float]:
    ys, xs = np.where(mask > 0)
    if ys.size == 0 or xs.size == 0:
        h, w = mask.shape[:2]
        return float(w / 2.0), float(h / 2.0)
    return float(xs.mean()), float(ys.mean())


def _mask_principal_angle_deg(mask: np.ndarray) -> float:
    ys, xs = np.where(mask > 0)
    if ys.size < 3 or xs.size < 3:
        return 0.0
    pts = np.column_stack([xs.astype(np.float32), ys.astype(np.float32)])
    centered = pts - pts.mean(axis=0, keepdims=True)
    cov = centered.T @ centered
    eigvals, eigvecs = np.linalg.eigh(cov.astype(np.float64))
    axis = eigvecs[:, int(np.argmax(eigvals))]
    return _wrap_line_angle_deg(float(np.degrees(np.arctan2(axis[1], axis[0]))))


def _mask_shape_objective_fixed(fixed_mask: np.ndarray) -> dict[str, Any]:
    fixed_mask_bool = fixed_mask > 0
    fixed_boundary = _boundary_mask(fixed_mask_bool)
    if fixed_boundary.any():
        fixed_boundary_dt = cv2.distanceTransform((~fixed_boundary).astype(np.uint8), cv2.DIST_L2, 5).astype(np.float32)
    else:
        fixed_boundary_dt = np.full(fixed_mask_bool.shape, float("inf"), dtype=np.float32)
    diag = float(np.hypot(fixed_mask_bool.shape[0], fixed_mask_bool.shape[1]))
    return {
        "fixed_mask": fixed_mask_bool,
        "fixed_boundary_dt": fixed_boundary_dt,
        "diag_px": max(diag, 1.0),
    }


def _mask_shape_objective_eval(
    fixed_obj: dict[str, Any],
    warped_mask: np.ndarray,
) -> dict[str, float]:
    warped_mask_bool = warped_mask > 0
    if not warped_mask_bool.any():
        return {
            "loss": float("inf"),
            "support_dice": 0.0,
            "boundary_dt_px": float("inf"),
            "boundary_dt_norm": float("inf"),
        }
    fixed_mask_bool = fixed_obj["fixed_mask"]
    region = _region_metrics(warped_mask_bool, fixed_mask_bool)
    warped_boundary = _boundary_mask(warped_mask_bool)
    if warped_boundary.any():
        boundary_dt_px = float(np.mean(fixed_obj["fixed_boundary_dt"][warped_boundary]))
    else:
        boundary_dt_px = float("inf")
    boundary_dt_norm = float(boundary_dt_px / max(float(fixed_obj["diag_px"]), 1.0))
    loss = float((1.0 - float(region["dice"])) + MASK_SHAPE_BOUNDARY_WEIGHT * boundary_dt_norm)
    return {
        "loss": loss,
        "support_dice": float(region["dice"]),
        "boundary_dt_px": boundary_dt_px,
        "boundary_dt_norm": boundary_dt_norm,
    }


def _mask_shape_candidate_better(candidate: dict[str, Any], best: dict[str, Any] | None) -> bool:
    if best is None:
        return True
    cand_loss = float(candidate.get("loss", float("inf")))
    best_loss = float(best.get("loss", float("inf")))
    if cand_loss < best_loss - 1e-9:
        return True
    if cand_loss > best_loss + 1e-9:
        return False
    cand_dice = float(candidate.get("support_dice", float("-inf")))
    best_dice = float(best.get("support_dice", float("-inf")))
    if cand_dice > best_dice + 1e-9:
        return True
    if cand_dice < best_dice - 1e-9:
        return False
    return float(candidate.get("boundary_dt_px", float("inf"))) < float(best.get("boundary_dt_px", float("inf")))


def _evaluate_mask_shape_candidate(
    fixed_obj: dict[str, Any],
    moving_mask: np.ndarray,
    fixed_shape_hw: tuple[int, int],
    params: dict[str, float],
) -> dict[str, Any]:
    mat = build_affine_matrix(
        moving_mask.shape[:2],
        fixed_shape_hw,
        tx_px=float(params.get("tx_px", 0.0)),
        ty_px=float(params.get("ty_px", 0.0)),
        angle_deg=float(params.get("angle_deg", 0.0)),
        scale=float(params.get("scale", 1.0)),
    )
    warped_mask = _warp_mask_affine(moving_mask, mat, fixed_shape_hw)
    objective = _mask_shape_objective_eval(fixed_obj, warped_mask)
    return {
        "params": {
            "tx_px": float(params.get("tx_px", 0.0)),
            "ty_px": float(params.get("ty_px", 0.0)),
            "angle_deg": float(params.get("angle_deg", 0.0)),
            "scale": float(params.get("scale", 1.0)),
        },
        "matrix_2x3": mat.astype(np.float32),
        "warped_mask": warped_mask,
        **objective,
    }


def optimize_mask_shape_transform(
    fixed_mask: np.ndarray,
    moving_mask: np.ndarray,
    *,
    model: str,
    similarity_scale_percent: float = 5.0,
    optimization_long_edge: int = MASK_SHAPE_OPTIMIZATION_LONG_EDGE,
    init_params: dict[str, float] | None = None,
) -> dict[str, Any]:
    model_norm = str(model).strip().lower()
    if model_norm not in {"rigid", "similarity"}:
        raise ValueError(f"Unsupported mask shape model: {model}")

    fixed_opt, opt_scale = _resize_mask_for_shape_optimization(fixed_mask, optimization_long_edge)
    moving_opt, _ = _resize_mask_for_shape_optimization(moving_mask, optimization_long_edge)
    fixed_obj = _mask_shape_objective_fixed(fixed_opt)
    min_scale = max(0.5, 1.0 - float(similarity_scale_percent) / 100.0)
    max_scale = 1.0 + float(similarity_scale_percent) / 100.0

    fixed_cx, fixed_cy = _mask_centroid_xy(fixed_opt)
    moving_cx, moving_cy = _mask_centroid_xy(moving_opt)
    centroid_orient_init = {
        "tx_px": float(fixed_cx - moving_cx),
        "ty_px": float(fixed_cy - moving_cy),
        "angle_deg": float(_wrap_line_angle_deg(_mask_principal_angle_deg(fixed_opt) - _mask_principal_angle_deg(moving_opt))),
        "scale": 1.0,
    }
    initial_candidates = [
        {"tx_px": 0.0, "ty_px": 0.0, "angle_deg": 0.0, "scale": 1.0},
        centroid_orient_init,
    ]
    if init_params:
        initial_candidates.insert(
            0,
            {
                "tx_px": float(init_params.get("tx_px", 0.0)) * float(opt_scale),
                "ty_px": float(init_params.get("ty_px", 0.0)) * float(opt_scale),
                "angle_deg": float(init_params.get("angle_deg", 0.0)),
                "scale": float(init_params.get("scale", 1.0)),
            },
        )

    best: dict[str, Any] | None = None
    initial_evals: list[dict[str, Any]] = []
    for candidate in initial_candidates:
        if model_norm == "rigid":
            candidate["scale"] = 1.0
        else:
            candidate["scale"] = float(np.clip(candidate["scale"], min_scale, max_scale))
        evaluated = _evaluate_mask_shape_candidate(fixed_obj, moving_opt, fixed_opt.shape[:2], candidate)
        initial_evals.append(
            {
                "params": dict(evaluated["params"]),
                "loss": float(evaluated["loss"]),
                "support_dice": float(evaluated["support_dice"]),
                "boundary_dt_px": float(evaluated["boundary_dt_px"]),
            }
        )
        if _mask_shape_candidate_better(evaluated, best):
            best = evaluated

    assert best is not None
    trace: list[dict[str, Any]] = []
    max_opt_long_edge = float(max(fixed_opt.shape[0], fixed_opt.shape[1]))
    for trans_frac, angle_step, scale_step in zip(
        MASK_SHAPE_TRANSLATION_STEPS_FRACTION,
        MASK_SHAPE_ANGLE_STEPS_DEG,
        MASK_SHAPE_SCALE_STEPS,
    ):
        translation_step = max(1.0, float(max_opt_long_edge * float(trans_frac)))
        level_record: dict[str, Any] = {
            "translation_step_px": float(translation_step),
            "angle_step_deg": float(angle_step),
            "scale_step": float(scale_step if model_norm == "similarity" else 0.0),
            "start_params": dict(best["params"]),
            "passes": [],
        }
        for _ in range(2):
            pass_record: dict[str, Any] = {}
            current = dict(best["params"])

            translation_best = best
            for dx in (-2, -1, 0, 1, 2):
                for dy in (-2, -1, 0, 1, 2):
                    candidate = dict(current)
                    candidate["tx_px"] = float(current["tx_px"] + dx * translation_step)
                    candidate["ty_px"] = float(current["ty_px"] + dy * translation_step)
                    evaluated = _evaluate_mask_shape_candidate(fixed_obj, moving_opt, fixed_opt.shape[:2], candidate)
                    if _mask_shape_candidate_better(evaluated, translation_best):
                        translation_best = evaluated
            best = translation_best
            current = dict(best["params"])
            pass_record["after_translation"] = {
                "params": dict(best["params"]),
                "loss": float(best["loss"]),
                "support_dice": float(best["support_dice"]),
                "boundary_dt_px": float(best["boundary_dt_px"]),
            }

            angle_best = best
            for delta in (-2, -1, 0, 1, 2):
                candidate = dict(current)
                candidate["angle_deg"] = float(_wrap_angle_deg(current["angle_deg"] + delta * angle_step))
                evaluated = _evaluate_mask_shape_candidate(fixed_obj, moving_opt, fixed_opt.shape[:2], candidate)
                if _mask_shape_candidate_better(evaluated, angle_best):
                    angle_best = evaluated
            best = angle_best
            current = dict(best["params"])
            pass_record["after_angle"] = {
                "params": dict(best["params"]),
                "loss": float(best["loss"]),
                "support_dice": float(best["support_dice"]),
                "boundary_dt_px": float(best["boundary_dt_px"]),
            }

            if model_norm == "similarity":
                scale_best = best
                for delta in (-2, -1, 0, 1, 2):
                    candidate = dict(current)
                    candidate["scale"] = float(np.clip(current["scale"] + delta * scale_step, min_scale, max_scale))
                    evaluated = _evaluate_mask_shape_candidate(fixed_obj, moving_opt, fixed_opt.shape[:2], candidate)
                    if _mask_shape_candidate_better(evaluated, scale_best):
                        scale_best = evaluated
                best = scale_best
                pass_record["after_scale"] = {
                    "params": dict(best["params"]),
                    "loss": float(best["loss"]),
                    "support_dice": float(best["support_dice"]),
                    "boundary_dt_px": float(best["boundary_dt_px"]),
                }
            level_record["passes"].append(pass_record)
        level_record["best_after_level"] = {
            "params": dict(best["params"]),
            "loss": float(best["loss"]),
            "support_dice": float(best["support_dice"]),
            "boundary_dt_px": float(best["boundary_dt_px"]),
        }
        trace.append(level_record)

    full_params = {
        "tx_px": float(best["params"]["tx_px"] / max(opt_scale, 1e-12)),
        "ty_px": float(best["params"]["ty_px"] / max(opt_scale, 1e-12)),
        "angle_deg": float(best["params"]["angle_deg"]),
        "scale": float(best["params"]["scale"] if model_norm == "similarity" else 1.0),
    }
    full_matrix = build_affine_matrix(
        moving_mask.shape[:2],
        fixed_mask.shape[:2],
        tx_px=full_params["tx_px"],
        ty_px=full_params["ty_px"],
        angle_deg=full_params["angle_deg"],
        scale=full_params["scale"],
    )
    warped_mask_full = _warp_mask_affine(moving_mask, full_matrix, fixed_mask.shape[:2])
    full_objective = _mask_shape_objective_eval(_mask_shape_objective_fixed(fixed_mask), warped_mask_full)
    return {
        "model": model_norm,
        "objective_name": MASK_SHAPE_OBJECTIVE_NAME,
        "optimization_scale": float(opt_scale),
        "optimization_shape_hw": [int(fixed_opt.shape[0]), int(fixed_opt.shape[1])],
        "initial_candidates": initial_evals,
        "trace": trace,
        "params": full_params,
        "matrix_2x3": full_matrix.astype(np.float32),
        "objective": full_objective,
    }


def compute_affine_stage_heatmap(
    matrix_2x3: np.ndarray,
    fixed_shape_hw: tuple[int, int],
    support_mask: np.ndarray,
    out_path: Path,
) -> tuple[np.ndarray, Path]:
    matrix = np.asarray(matrix_2x3, dtype=np.float64)
    hom = np.vstack([matrix, np.array([0.0, 0.0, 1.0], dtype=np.float64)])
    inv = np.linalg.inv(hom)
    out_h, out_w = int(fixed_shape_hw[0]), int(fixed_shape_hw[1])
    yy, xx = np.indices((out_h, out_w), dtype=np.float32)
    src_x = (inv[0, 0] * xx) + (inv[0, 1] * yy) + inv[0, 2]
    src_y = (inv[1, 0] * xx) + (inv[1, 1] * yy) + inv[1, 2]
    mag = np.sqrt((src_x - xx) ** 2 + (src_y - yy) ** 2).astype(np.float32)
    heat_rgb = displacement_heatmap(mag, np.where(support_mask > 0, 1, 0).astype(np.uint8))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), cv2.cvtColor(heat_rgb, cv2.COLOR_RGB2BGR))
    return heat_rgb, out_path


def run_mask_shape_stage(
    stage: str,
    fixed_gray: np.ndarray,
    moving_gray: np.ndarray,
    fixed_mask: np.ndarray,
    moving_mask: np.ndarray,
    stage_dir: Path,
    *,
    similarity_scale_percent: float = 5.0,
    init_params: dict[str, float] | None = None,
    optimization_long_edge: int = MASK_SHAPE_OPTIMIZATION_LONG_EDGE,
) -> dict[str, Any]:
    model = "rigid" if str(stage).strip().lower() == "mask_rigid" else "similarity"
    stage_t0 = time.perf_counter()
    optimize_t0 = time.perf_counter()
    optimization = optimize_mask_shape_transform(
        fixed_mask,
        moving_mask,
        model=model,
        similarity_scale_percent=float(similarity_scale_percent),
        optimization_long_edge=int(optimization_long_edge),
        init_params=init_params,
    )
    optimize_seconds = float(time.perf_counter() - optimize_t0)
    matrix_2x3 = np.asarray(optimization["matrix_2x3"], dtype=np.float32)
    warped_gray = np.clip(_warp_gray_affine(moving_gray, matrix_2x3, fixed_gray.shape[:2]), 0.0, 1.0)
    warped_mask_bool = _warp_mask_affine(moving_mask, matrix_2x3, fixed_gray.shape[:2])
    stage_metrics, stage_metric_timings = compute_registration_metrics(
        fixed_gray,
        warped_gray,
        fixed_mask,
        warped_mask_bool.astype(np.float32),
    )
    overlay = overlay_preview(
        fixed_gray,
        warped_gray,
        fixed_mask,
        warped_mask_bool.astype(np.float32),
    )
    heatmap_png = stage_dir / f"{stage}_heatmap.png"
    heatmap_rgb, heatmap_png = compute_affine_stage_heatmap(
        matrix_2x3,
        fixed_gray.shape[:2],
        (fixed_mask > 0) | warped_mask_bool,
        heatmap_png,
    )
    warped_gray_path = stage_dir / f"{stage}_warped_gray.png"
    warped_mask_path = stage_dir / f"{stage}_warped_mask.png"
    write_gray_png(warped_gray_path, warped_gray)
    cv2.imwrite(str(warped_mask_path), np.where(warped_mask_bool, 255, 0).astype(np.uint8))
    transform_json = stage_dir / f"{stage}_transform.json"
    _write_json(
        transform_json,
        {
            "stage": str(stage),
            "backend": MASK_SHAPE_REGISTRATION_BACKEND,
            "objective_name": MASK_SHAPE_OBJECTIVE_NAME,
            "params": dict(optimization["params"]),
            "matrix_2x3": np.asarray(matrix_2x3, dtype=np.float64).tolist(),
            "optimization_scale": float(optimization["optimization_scale"]),
            "optimization_shape_hw": list(optimization["optimization_shape_hw"]),
            "initial_candidates": list(optimization["initial_candidates"]),
            "trace": list(optimization["trace"]),
            "objective": dict(optimization["objective"]),
        },
    )
    stage_seconds = float(time.perf_counter() - stage_t0)
    params = dict(optimization["params"])
    note = (
        f"{metrics_note(stage_metrics, stage_metric_timings, f'{stage_display_name(stage)} finished')} | "
        f"loss={float(optimization['objective']['loss']):.4f} "
        f"boundary={float(optimization['objective']['boundary_dt_px']):.2f}px "
        f"tx={float(params['tx_px']):.1f} ty={float(params['ty_px']):.1f} "
        f"ang={float(params['angle_deg']):.2f} scale={float(params['scale']):.4f}"
    )
    return {
        "moving": gray_preview_panel(warped_gray),
        "overlay": overlay,
        "heatmap": heatmap_rgb,
        "note": note,
        "warped_image": str(warped_gray_path),
        "warped_mask": str(warped_mask_path),
        "heatmap_png": str(heatmap_png),
        "transforms": [str(transform_json)],
        "metrics": stage_metrics,
        "metric_timing_seconds": stage_metric_timings,
        "timing_seconds": {
            "stage_total": stage_seconds,
            "optimizer": optimize_seconds,
            "postprocess": float(max(stage_seconds - optimize_seconds, 0.0)),
        },
        "optimizer": {
            "backend": MASK_SHAPE_REGISTRATION_BACKEND,
            "objective_name": MASK_SHAPE_OBJECTIVE_NAME,
            "optimization_scale": float(optimization["optimization_scale"]),
            "optimization_shape_hw": list(optimization["optimization_shape_hw"]),
            "initial_candidates": list(optimization["initial_candidates"]),
            "trace": list(optimization["trace"]),
            "objective": dict(optimization["objective"]),
        },
        "transform_params": params,
        "transform_matrix_2x3": np.asarray(matrix_2x3, dtype=np.float64).tolist(),
        "command": [
            "mask_shape_optimizer",
            stage,
            f"--model={model}",
            f"--similarity-scale-percent={float(similarity_scale_percent):.3f}",
        ],
    }


def write_nifti_2d(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = nib.Nifti1Image(arr.astype(np.float32), np.eye(4))
    nib.save(img, str(path))


def read_nifti_2d(path: Path) -> np.ndarray:
    data = np.asarray(nib.load(str(path)).get_fdata(), dtype=np.float32)
    while data.ndim > 2 and data.shape[-1] == 1:
        data = data[..., 0]
    return data.astype(np.float32)


def write_gray_png(path: Path, arr: np.ndarray) -> None:
    a = arr.astype(np.float32)
    if a.size == 0:
        out = np.zeros((1, 1), dtype=np.uint8)
    else:
        if np.nanmax(a) > np.nanmin(a):
            a = (a - np.nanmin(a)) / float(np.nanmax(a) - np.nanmin(a))
        out = np.clip(np.round(a * 255.0), 0, 255).astype(np.uint8)
    cv2.imwrite(str(path), out)


def _contours_from_mask(mask: np.ndarray) -> list[np.ndarray]:
    mask_u8 = np.where(mask > 0, 255, 0).astype(np.uint8)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def overlay_preview(fixed_gray: np.ndarray, moving_gray: np.ndarray, fixed_mask: np.ndarray, moving_mask: np.ndarray) -> np.ndarray:
    fixed_u8 = np.clip(np.round(fixed_gray * 255.0), 0, 255).astype(np.uint8)
    moving_u8 = np.clip(np.round(moving_gray * 255.0), 0, 255).astype(np.uint8)
    overlay = np.zeros((*fixed_u8.shape, 3), dtype=np.uint8)
    overlay[..., 1] = fixed_u8
    overlay[..., 0] = moving_u8
    overlay[..., 2] = moving_u8
    cv2.drawContours(overlay, _contours_from_mask(fixed_mask), -1, (0, 255, 0), 1)
    cv2.drawContours(overlay, _contours_from_mask(moving_mask), -1, (255, 255, 255), 1)
    return overlay


def gray_preview_panel(gray: np.ndarray) -> np.ndarray:
    gray_u8 = np.clip(np.round(gray * 255.0), 0, 255).astype(np.uint8)
    return cv2.cvtColor(gray_u8, cv2.COLOR_GRAY2RGB)


def displacement_heatmap(magnitude: np.ndarray, support_mask: np.ndarray) -> np.ndarray:
    mag = magnitude.astype(np.float32)
    valid = support_mask > 0
    if np.any(valid):
        vmax = float(np.percentile(mag[valid], 99.0))
        vmax = max(vmax, 1e-6)
    else:
        vmax = 1.0
    norm = np.clip(mag / vmax, 0.0, 1.0)
    heat = cv2.applyColorMap(np.clip(np.round(norm * 255.0), 0, 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    base = np.repeat((np.where(valid, 64, 240)).astype(np.uint8)[..., None], 3, axis=2)
    out = np.where(valid[..., None], ((0.35 * base) + (0.65 * heat)).astype(np.uint8), base)
    cv2.drawContours(out, _contours_from_mask(support_mask.astype(np.uint8)), -1, (255, 255, 255), 1)
    return out


def _region_metrics(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    tp = int((pred & gt).sum())
    fp = int((pred & ~gt).sum())
    fn = int((~pred & gt).sum())
    return {
        "dice": 2 * tp / max(1, 2 * tp + fp + fn),
        "iou": tp / max(1, tp + fp + fn),
        "precision": tp / max(1, tp + fp),
        "recall": tp / max(1, tp + fn),
    }


def _boundary_mask(mask: np.ndarray) -> np.ndarray:
    mask_u8 = (mask > 0).astype(np.uint8)
    if mask_u8.max() == 0:
        return np.zeros_like(mask_u8, dtype=bool)
    eroded = cv2.erode(mask_u8 * 255, np.ones((3, 3), np.uint8), iterations=1) > 0
    return mask_u8.astype(bool) & (~eroded)


def _contour_metrics(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    pred_boundary = _boundary_mask(pred)
    gt_boundary = _boundary_mask(gt)
    if not pred_boundary.any() or not gt_boundary.any():
        return {
            "assd_px": float("inf"),
            "hd95_px": float("inf"),
        }
    dt_to_gt = cv2.distanceTransform((~gt_boundary).astype(np.uint8), cv2.DIST_L2, 5)
    dt_to_pred = cv2.distanceTransform((~pred_boundary).astype(np.uint8), cv2.DIST_L2, 5)
    d_pred = dt_to_gt[pred_boundary]
    d_gt = dt_to_pred[gt_boundary]
    return {
        "assd_px": float((d_pred.mean() + d_gt.mean()) / 2.0),
        "hd95_px": float(max(np.quantile(d_pred, 0.95), np.quantile(d_gt, 0.95))),
    }


def _mutual_information(x: np.ndarray, y: np.ndarray, bins: int = 64) -> float:
    if x.size == 0 or y.size == 0:
        return float("nan")
    hist = np.histogram2d(x, y, bins=bins, range=[[0.0, 1.0], [0.0, 1.0]])[0].astype(np.float64)
    total = float(hist.sum())
    if total <= 0.0:
        return float("nan")
    pxy = hist / total
    px = pxy.sum(axis=1, keepdims=True)
    py = pxy.sum(axis=0, keepdims=True)
    denom = px @ py
    nz = pxy > 0
    return float(np.sum(pxy[nz] * np.log(pxy[nz] / np.maximum(denom[nz], 1e-12))))


def _cross_correlation(x: np.ndarray, y: np.ndarray) -> float:
    if x.size == 0 or y.size == 0:
        return float("nan")
    x = x.astype(np.float64)
    y = y.astype(np.float64)
    x_std = float(x.std())
    y_std = float(y.std())
    if x_std <= 1e-12 or y_std <= 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def compute_registration_metrics(
    fixed_gray: np.ndarray,
    moving_gray: np.ndarray,
    fixed_mask: np.ndarray,
    moving_mask: np.ndarray,
) -> tuple[dict[str, float], dict[str, float]]:
    timings: dict[str, float] = {}
    total_t0 = time.perf_counter()

    fixed_mask_bool = fixed_mask > 0
    moving_mask_bool = moving_mask > 0

    t0 = time.perf_counter()
    region = _region_metrics(moving_mask_bool, fixed_mask_bool)
    timings["region"] = float(time.perf_counter() - t0)

    t0 = time.perf_counter()
    contour = _contour_metrics(moving_mask_bool, fixed_mask_bool)
    timings["contour"] = float(time.perf_counter() - t0)

    valid = fixed_mask_bool | moving_mask_bool
    fx = fixed_gray[valid].astype(np.float32)
    mv = moving_gray[valid].astype(np.float32)

    t0 = time.perf_counter()
    mi = _mutual_information(fx, mv)
    timings["mi"] = float(time.perf_counter() - t0)

    t0 = time.perf_counter()
    cc = _cross_correlation(fx, mv)
    timings["cc"] = float(time.perf_counter() - t0)

    timings["total"] = float(time.perf_counter() - total_t0)
    metrics = {
        **region,
        **contour,
        "mi": mi,
        "cc": cc,
        "valid_pixels": int(valid.sum()),
    }
    return metrics, timings


def _fmt_metric(name: str, value: float, *, digits: int = 3) -> str:
    if not np.isfinite(value):
        return f"{name}=inf"
    return f"{name}={value:.{digits}f}"


def metrics_note(metrics: dict[str, float], timings: dict[str, float], prefix: str) -> str:
    return " | ".join(
        [
            prefix,
            _fmt_metric("MI", float(metrics.get("mi", float("nan")))),
            _fmt_metric("CC", float(metrics.get("cc", float("nan")))),
            _fmt_metric("Dice", float(metrics.get("dice", float("nan")))),
            _fmt_metric("HD95", float(metrics.get("hd95_px", float("nan"))), digits=1),
            f"metric_t={float(timings.get('total', 0.0)):.2f}s",
        ]
    )


def _safe_metric_value(metrics: dict[str, Any] | None, key: str, default: float = float("nan")) -> float:
    if not isinstance(metrics, dict):
        return default
    value = metrics.get(key, default)
    try:
        return float(value)
    except Exception:
        return default


def metric_delta_summary(candidate_metrics: dict[str, Any], reference_metrics: dict[str, Any]) -> dict[str, float]:
    keys = list(
        dict.fromkeys(
            [
                *STRICT_PARETO_DELTA_KEYS,
                *tuple(str(k) for k in candidate_metrics.keys()),
                *tuple(str(k) for k in reference_metrics.keys()),
            ]
        )
    )
    return {
        key: _safe_metric_value(candidate_metrics, key) - _safe_metric_value(reference_metrics, key)
        for key in keys
    }


def strict_pareto_gating_policy() -> dict[str, Any]:
    return {
        "policy_name": STRICT_PARETO_GATING_POLICY_NAME,
        "mode": "strict_pareto",
        "baseline_stage": "input",
        "reference_state": "current_best",
        "maximize_metrics": list(STRICT_PARETO_MAXIMIZE_KEYS),
        "minimize_metrics": list(STRICT_PARETO_MINIMIZE_KEYS),
        "epsilon": {key: 0.0 for key in (*STRICT_PARETO_MAXIMIZE_KEYS, *STRICT_PARETO_MINIMIZE_KEYS)},
        "acceptance_rule": (
            "Accept candidate only if all maximize metrics are >= current best, "
            "all minimize metrics are <= current best, and at least one tracked geometry metric improves strictly."
        ),
        "notes": [
            "MI/CC are diagnostic only in this policy and do not control acceptance.",
            "When supported by the runner, later stages initialize from the current best accepted state rather than a fixed chain.",
        ],
    }


def strict_pareto_gate_decision(best_metrics: dict[str, Any], candidate_metrics: dict[str, Any]) -> dict[str, Any]:
    checks: dict[str, dict[str, Any]] = {}
    nonfinite_keys: list[str] = []

    for key in STRICT_PARETO_MAXIMIZE_KEYS:
        best_val = _safe_metric_value(best_metrics, key)
        cand_val = _safe_metric_value(candidate_metrics, key)
        ok = bool(np.isfinite(best_val) and np.isfinite(cand_val) and cand_val >= best_val)
        strict = bool(np.isfinite(best_val) and np.isfinite(cand_val) and cand_val > best_val)
        checks[key] = {
            "direction": "maximize",
            "best": best_val,
            "candidate": cand_val,
            "pass": ok,
            "strict_improvement": strict,
        }
        if not (np.isfinite(best_val) and np.isfinite(cand_val)):
            nonfinite_keys.append(key)

    for key in STRICT_PARETO_MINIMIZE_KEYS:
        best_val = _safe_metric_value(best_metrics, key)
        cand_val = _safe_metric_value(candidate_metrics, key)
        ok = bool(np.isfinite(best_val) and np.isfinite(cand_val) and cand_val <= best_val)
        strict = bool(np.isfinite(best_val) and np.isfinite(cand_val) and cand_val < best_val)
        checks[key] = {
            "direction": "minimize",
            "best": best_val,
            "candidate": cand_val,
            "pass": ok,
            "strict_improvement": strict,
        }
        if not (np.isfinite(best_val) and np.isfinite(cand_val)):
            nonfinite_keys.append(key)

    all_pass = all(bool(item.get("pass")) for item in checks.values())
    any_strict = any(bool(item.get("strict_improvement")) for item in checks.values())
    accepted = bool(all_pass and any_strict and not nonfinite_keys)

    if nonfinite_keys:
        decision = "rejected_nonfinite_geometry_metrics"
    elif accepted:
        decision = "accepted_strict_pareto"
    else:
        failed = [key for key, item in checks.items() if not bool(item.get("pass"))]
        if failed:
            decision = f"rejected_{'_and_'.join(failed)}"
        else:
            decision = "rejected_no_strict_improvement"

    return {
        "accepted": accepted,
        "decision": decision,
        "all_constraints_pass": all_pass,
        "strict_improvement": any_strict,
        "failed_metrics": [key for key, item in checks.items() if not bool(item.get("pass"))],
        "nonfinite_metrics": nonfinite_keys,
        "checks": checks,
    }


def compute_monotonic_gating_summary(manifest: dict[str, Any]) -> dict[str, Any]:
    policy = strict_pareto_gating_policy()
    stages = manifest.get("stages", {}) if isinstance(manifest.get("stages"), dict) else {}
    requested_stages = [str(x) for x in list(manifest.get("run_stages") or [])]
    evaluated_stages = [stage for stage in requested_stages if isinstance(stages.get(stage), dict) and isinstance(stages[stage].get("metrics"), dict)]

    input_metrics = manifest.get("input_metrics", {}) if isinstance(manifest.get("input_metrics"), dict) else {}
    best_stage = "input"
    best_metrics = dict(input_metrics)
    accepted_stage_path = ["input"]

    input_gate = {
        "stage": "input",
        "accepted": True,
        "decision": "baseline",
        "best_stage_before": None,
        "best_stage_after": "input",
        "delta_vs_best_before": metric_delta_summary(input_metrics, input_metrics),
        "delta_vs_input": metric_delta_summary(input_metrics, input_metrics),
    }

    stage_gates: dict[str, dict[str, Any]] = {}
    for stage in evaluated_stages:
        candidate_metrics = stages[stage]["metrics"]
        best_stage_before = best_stage
        best_metrics_before = dict(best_metrics)
        gate = strict_pareto_gate_decision(best_metrics_before, candidate_metrics)
        gate["stage"] = stage
        gate["best_stage_before"] = best_stage_before
        gate["delta_vs_best_before"] = metric_delta_summary(candidate_metrics, best_metrics_before)
        gate["delta_vs_input"] = metric_delta_summary(candidate_metrics, input_metrics)
        if gate["accepted"]:
            best_stage = stage
            best_metrics = dict(candidate_metrics)
            accepted_stage_path.append(stage)
        gate["best_stage_after"] = best_stage
        stage_gates[stage] = gate

    return {
        "policy": policy,
        "input": input_gate,
        "stages": stage_gates,
        "evaluated_stage_order": evaluated_stages,
        "accepted_stage_path": accepted_stage_path,
        "best_stage": best_stage,
        "best_metrics": dict(best_metrics),
        "best_metrics_source": "input_metrics" if best_stage == "input" else f"stages.{best_stage}.metrics",
        "final_delta_vs_input": metric_delta_summary(best_metrics, input_metrics),
    }


def ensure_monotonic_gating_summary(manifest: dict[str, Any]) -> dict[str, Any]:
    existing_policy = manifest.get("gating_policy")
    existing_summary = manifest.get("gating_summary")
    if (
        isinstance(existing_policy, dict)
        and isinstance(existing_summary, dict)
        and str(existing_policy.get("policy_name", "")) == STRICT_PARETO_GATING_POLICY_NAME
        and str(existing_summary.get("best_stage", "")).strip()
    ):
        return {"policy": existing_policy, **existing_summary}
    return compute_monotonic_gating_summary(manifest)


def render_storyboard(rows: list[dict[str, Any]], out_path: Path) -> None:
    pad = 12
    title_h = 28
    row_gap = 18
    col_gap = 12
    font = cv2.FONT_HERSHEY_SIMPLEX
    default_panel_keys = ("moving", "fixed", "overlay", "heatmap")
    panel_keys = tuple(str(x) for x in rows[0].get("panel_keys", default_panel_keys)) if rows else default_panel_keys
    if not panel_keys:
        panel_keys = default_panel_keys
    default_title_map = {
        "moving": "Moving",
        "fixed": "Fixed",
        "overlay": "Overlay",
        "heatmap": "Warp Field",
        "scene_context": "Scene Context",
    }
    default_col_titles = tuple(default_title_map.get(key, key.replace("_", " ").title()) for key in panel_keys)
    raw_titles = rows[0].get("col_titles", default_col_titles) if rows else default_col_titles
    if not isinstance(raw_titles, (list, tuple)):
        raw_titles = default_col_titles
    col_titles = tuple(
        str(raw_titles[idx]) if idx < len(raw_titles) else default_col_titles[idx]
        for idx in range(len(panel_keys))
    )

    if not rows:
        canvas = np.full((240, 480, 3), 245, dtype=np.uint8)
        cv2.putText(canvas, "No registration stage output yet", (20, 120), font, 0.7, (20, 20, 20), 2, cv2.LINE_AA)
        cv2.imwrite(str(out_path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
        return

    example_panel = next(
        (
            row[key]
            for row in rows
            for key in panel_keys
            if isinstance(row.get(key), np.ndarray)
        ),
        None,
    )
    if example_panel is None:
        canvas = np.full((240, 480, 3), 245, dtype=np.uint8)
        cv2.putText(canvas, "No registration stage output yet", (20, 120), font, 0.7, (20, 20, 20), 2, cv2.LINE_AA)
        cv2.imwrite(str(out_path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
        return

    panel_h, panel_w = example_panel.shape[:2]
    blank_panel = np.full((panel_h, panel_w, 3), 235, dtype=np.uint8)
    total_w = pad * 2 + panel_w * len(panel_keys) + col_gap * max(0, len(panel_keys) - 1)
    total_h = pad * 2 + title_h + len(rows) * panel_h + max(0, len(rows) - 1) * row_gap
    canvas = np.full((total_h, total_w, 3), 245, dtype=np.uint8)

    for col_idx, title in enumerate(col_titles):
        x = pad + col_idx * (panel_w + col_gap)
        cv2.putText(canvas, title, (x, pad + 18), font, 0.55, (25, 25, 25), 1, cv2.LINE_AA)

    for row_idx, row in enumerate(rows):
        y = pad + title_h + row_idx * (panel_h + row_gap)
        x0 = pad
        for col_idx, key in enumerate(panel_keys):
            x = x0 + col_idx * (panel_w + col_gap)
            panel = row.get(key)
            if not isinstance(panel, np.ndarray):
                panel = blank_panel
            elif panel.ndim == 2:
                panel = cv2.cvtColor(panel.astype(np.uint8), cv2.COLOR_GRAY2RGB)
            else:
                panel = np.asarray(panel, dtype=np.uint8)
            if panel.shape[:2] != (panel_h, panel_w):
                panel = cv2.resize(panel, (panel_w, panel_h), interpolation=cv2.INTER_LINEAR)
            canvas[y : y + panel_h, x : x + panel_w] = panel
            cv2.rectangle(canvas, (x, y), (x + panel_w, y + panel_h), (200, 200, 200), 1)
        label = str(row.get("label", f"Row {row_idx + 1}"))
        qc = str(row.get("note", "")).strip()
        cv2.putText(canvas, label, (pad, y - 4), font, 0.55, (15, 15, 15), 1, cv2.LINE_AA)
        if qc:
            cv2.putText(canvas, qc[:120], (pad + 180, y - 4), font, 0.45, (80, 80, 80), 1, cv2.LINE_AA)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))


def _run_logged(cmd: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    log_path.write_text(proc.stdout or "", encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(cmd)}\n{proc.stdout}")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


@dataclass(frozen=True)
class PairRegistrationConfig:
    pair_key: str
    moving_side: str
    fixed_side: str
    moving_group: str
    fixed_group: str
    review: dict[str, Any]
    common_root: Path
    myelin_root: Path
    nissl_root: Path
    ants_bin: Path
    runs_root: Path
    target_um_per_px: float = DEFAULT_TARGET_UM_PER_PX
    working_long_edge: int = REGISTRATION_MAX_LONG_EDGE
    pre_blur_sigma: float = 0.0
    registration_mask_mode: str = "union"
    registration_backend: str = ANTS_REGISTRATION_BACKEND
    run_stages: tuple[str, ...] = ("rigid", "affine", "syn")
    affine_profile: str = "current"
    mask_similarity_scale_percent: float = 5.0


def _section_root_for_side(cfg: PairRegistrationConfig, side: str) -> Path:
    if side == "myelin":
        return cfg.myelin_root / str(cfg.review["myelin_label"])
    if side == "nissl":
        return cfg.nissl_root / str(cfg.review["nissl_label"])
    raise ValueError(f"Unknown side: {side}")


def _mask_path_for_side(cfg: PairRegistrationConfig, side: str) -> Path:
    reg_files = dict(cfg.review.get("registration_mask_files") or {})
    rel = reg_files.get(side)
    path = resolve_pair_registry_relpath(cfg.common_root, rel)
    if path is not None and not path.exists() and rel:
        rel_norm = str(rel).replace("\\", "/").lstrip("/")
        legacy_prefix = "histology_pair_registration_masks/"
        if rel_norm.startswith(legacy_prefix):
            suffix = rel_norm[len(legacy_prefix) :]
            candidates = sorted(
                p
                for p in cfg.common_root.glob("histology_pair_registration_masks*")
                if p.is_dir()
            )
            for cand in candidates:
                alt = cand / suffix
                if alt.exists():
                    path = alt
                    break
    if path is not None and path.exists():
        return path
    fallback = _section_root_for_side(cfg, side) / "mask_labels.png"
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f"Registration mask is missing for side {side}: {rel}")


def _current_canvas_um_per_px(provenance: dict[str, Any]) -> tuple[float, float]:
    info = dict(provenance.get("canvas_to_slide_um_per_px") or {})
    x = float(info.get("x_um_per_px") or 0.0)
    y = float(info.get("y_um_per_px") or 0.0)
    if x <= 0.0 or y <= 0.0:
        raise ValueError("Section physical_provenance is missing canvas_to_slide_um_per_px")
    return x, y


def _resample_to_target_um_per_px(
    rgb: np.ndarray,
    labels: np.ndarray,
    *,
    current_um_per_px_xy: tuple[float, float],
    target_um_per_px: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    cur_x, cur_y = current_um_per_px_xy
    if target_um_per_px <= 0:
        raise ValueError("target_um_per_px must be positive")
    scale_x = float(cur_x) / float(target_um_per_px)
    scale_y = float(cur_y) / float(target_um_per_px)
    out_w = max(1, int(round(rgb.shape[1] * scale_x)))
    out_h = max(1, int(round(rgb.shape[0] * scale_y)))
    rgb_out = cv2.resize(rgb, (out_w, out_h), interpolation=cv2.INTER_AREA)
    labels_out = cv2.resize(labels, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
    return rgb_out, labels_out, {
        "input_shape_hw": [int(rgb.shape[0]), int(rgb.shape[1])],
        "output_shape_hw": [int(out_h), int(out_w)],
        "current_um_per_px": {"x": float(cur_x), "y": float(cur_y)},
        "target_um_per_px": float(target_um_per_px),
        "scale_to_target": {"x": float(scale_x), "y": float(scale_y)},
    }


def _prepare_side(cfg: PairRegistrationConfig, side: str, group_choice: str) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    root = _section_root_for_side(cfg, side)
    _, provenance = recover_or_load_section_physical_provenance(root, write_back_if_missing=True)
    crop_path = root / "crop_raw.png"
    rgb_bgr = cv2.imread(str(crop_path), cv2.IMREAD_COLOR)
    if rgb_bgr is None:
        raise FileNotFoundError(crop_path)
    rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
    mask_path = _mask_path_for_side(cfg, side)
    labels = load_mask_labels(mask_path)
    mask_source = "pair_registration_mask" if "histology_pair_registration_masks" in str(mask_path).replace("\\", "/") else "canonical_section_mask"
    if labels.shape[:2] != rgb.shape[:2]:
        labels = cv2.resize(labels, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)

    canonical_shape_hw = [int(rgb.shape[0]), int(rgb.shape[1])]
    flip_lr = bool((cfg.review.get("flip_lr") or {}).get(side, False))
    component_groups = dict((cfg.review.get("component_groups") or {}).get(side) or {})
    group_flip_lr = dict((cfg.review.get("group_flip_lr") or {}).get(side) or {})

    rgb, labels = apply_whole_flip(rgb, labels, flip_lr)
    rgb, labels = apply_group_flips(
        rgb,
        labels,
        component_groups,
        group_flip_lr,
    )
    labels = keep_group(labels, component_groups, group_choice)
    support_mask = registration_support_mask(labels, cfg.registration_mask_mode)
    if not np.any(support_mask):
        raise ValueError(f"{side} group {group_choice} is empty after selection.")
    support_bbox = _bbox_from_mask(support_mask.astype(np.uint8))
    rgb, labels = crop_to_support(rgb, labels, support_mask)
    if str(cfg.registration_mask_mode).strip().lower() in {"tissue", "tissue_only", "tissue-only"}:
        labels = np.where(labels == 1, 1, 0).astype(np.uint8)
    current_um_per_px_xy = _current_canvas_um_per_px(provenance)
    rgb, labels, resample_info = _resample_to_target_um_per_px(
        rgb,
        labels,
        current_um_per_px_xy=current_um_per_px_xy,
        target_um_per_px=cfg.target_um_per_px,
    )
    return rgb, labels, {
        "section_root": str(root),
        "canonical_shape_hw": canonical_shape_hw,
        "whole_flip_lr": bool(flip_lr),
        "group_flip_lr": {str(k): bool(v) for k, v in sorted(group_flip_lr.items())},
        "group_choice": str(group_choice),
        "component_groups": component_groups,
        "support_crop_bbox_xywh": support_bbox,
        "support_crop_shape_hw": [int(support_bbox["h"]), int(support_bbox["w"])],
        "current_um_per_px_xy": {"x": float(current_um_per_px_xy[0]), "y": float(current_um_per_px_xy[1])},
        "target_um_per_px": float(cfg.target_um_per_px),
        "resample": resample_info,
        "physical_resample_scale_xy": dict(resample_info.get("scale_to_target") or {}),
        "physical_normalized_shape_hw": list(resample_info.get("output_shape_hw") or [int(rgb.shape[0]), int(rgb.shape[1])]),
        "source_slide_path": str((provenance.get("source_slide") or {}).get("path") or ""),
        "source_slide_backend": (provenance.get("source_slide") or {}).get("backend"),
        "mask_path": str(mask_path),
        "mask_source": mask_source,
        "registration_mask_mode": str(cfg.registration_mask_mode),
        "mpp_recovery_method": provenance.get("mpp_recovery_method"),
        "slide_resolution_method": provenance.get("slide_resolution_method"),
        "preprocess_chain": {
            "canonical_fullres_shape_hw": canonical_shape_hw,
            "support_crop_bbox_xywh": support_bbox,
            "support_crop_shape_hw": [int(support_bbox["h"]), int(support_bbox["w"])],
            "physical_normalized_shape_hw": list(resample_info.get("output_shape_hw") or [int(rgb.shape[0]), int(rgb.shape[1])]),
        },
    }


def _common_canvas_for_pair(fixed_rgb: np.ndarray, fixed_labels: np.ndarray, moving_rgb: np.ndarray, moving_labels: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    margin = 32
    target_h = max(fixed_rgb.shape[0], moving_rgb.shape[0]) + margin * 2
    target_w = max(fixed_rgb.shape[1], moving_rgb.shape[1]) + margin * 2
    fixed_rgb_c, fixed_labels_c = center_on_canvas(fixed_rgb, fixed_labels, (target_h, target_w))
    moving_rgb_c, moving_labels_c = center_on_canvas(moving_rgb, moving_labels, (target_h, target_w))
    return fixed_rgb_c, fixed_labels_c, moving_rgb_c, moving_labels_c


def _resize_pair_to_working_long_edge(
    fixed_rgb: np.ndarray,
    fixed_labels: np.ndarray,
    moving_rgb: np.ndarray,
    moving_labels: np.ndarray,
    *,
    working_long_edge: int,
    blur_sigma: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    long_edge = max(fixed_rgb.shape[0], fixed_rgb.shape[1], moving_rgb.shape[0], moving_rgb.shape[1])
    if working_long_edge <= 0 or long_edge <= working_long_edge:
        scale = 1.0
    else:
        scale = float(working_long_edge) / float(long_edge)

    def _resize_rgb(arr: np.ndarray) -> np.ndarray:
        if scale >= 0.999999:
            return arr.copy()
        return cv2.resize(
            arr,
            (max(1, int(round(arr.shape[1] * scale))), max(1, int(round(arr.shape[0] * scale)))),
            interpolation=cv2.INTER_AREA,
        )

    def _resize_lab(arr: np.ndarray) -> np.ndarray:
        if scale >= 0.999999:
            return arr.copy()
        return cv2.resize(
            arr,
            (max(1, int(round(arr.shape[1] * scale))), max(1, int(round(arr.shape[0] * scale)))),
            interpolation=cv2.INTER_NEAREST,
        )

    fixed_rgb_w = _resize_rgb(fixed_rgb)
    fixed_labels_w = _resize_lab(fixed_labels)
    moving_rgb_w = _resize_rgb(moving_rgb)
    moving_labels_w = _resize_lab(moving_labels)
    fixed_rgb_w = _apply_gaussian_blur_preserving_background(fixed_rgb_w, fixed_labels_w, blur_sigma)
    moving_rgb_w = _apply_gaussian_blur_preserving_background(moving_rgb_w, moving_labels_w, blur_sigma)
    return (
        fixed_rgb_w,
        fixed_labels_w,
        moving_rgb_w,
        moving_labels_w,
        {
            "working_long_edge": int(working_long_edge),
            "downsample_to_working_scale": float(scale),
            "pre_blur_sigma": float(blur_sigma),
        },
    )


def _common_canvas_for_pair_with_info(
    fixed_rgb: np.ndarray,
    fixed_labels: np.ndarray,
    moving_rgb: np.ndarray,
    moving_labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, int], dict[str, int], list[int]]:
    margin = 32
    target_h = max(fixed_rgb.shape[0], moving_rgb.shape[0]) + margin * 2
    target_w = max(fixed_rgb.shape[1], moving_rgb.shape[1]) + margin * 2
    fixed_rgb_c, fixed_labels_c, fixed_offset = _center_on_canvas_with_offset(fixed_rgb, fixed_labels, (target_h, target_w))
    moving_rgb_c, moving_labels_c, moving_offset = _center_on_canvas_with_offset(moving_rgb, moving_labels, (target_h, target_w))
    return fixed_rgb_c, fixed_labels_c, moving_rgb_c, moving_labels_c, fixed_offset, moving_offset, [int(target_h), int(target_w)]


def _ants_apply(
    ants_bin: Path,
    input_path: Path,
    ref_path: Path,
    output_path: Path,
    transforms: list[Path],
    *,
    interpolation: str,
    log_path: Path,
) -> None:
    cmd = [
        str(ants_binary_path(ants_bin, "antsApplyTransforms")),
        "-d",
        "2",
        "-i",
        ants_cli_path(input_path),
        "-r",
        ants_cli_path(ref_path),
        "-o",
        ants_cli_path(output_path),
        "-n",
        interpolation,
    ]
    for tfm in transforms:
        cmd.extend(["-t", ants_cli_path(tfm)])
    _run_logged(cmd, log_path)


def _stage_command(
    ants_bin: Path,
    stage: str,
    fixed_path: Path,
    moving_path: Path,
    fixed_mask_path: Path,
    moving_mask_path: Path,
    out_prefix: Path,
    initial_transforms: list[Path],
    affine_profile: str = "current",
) -> list[str]:
    cmd = [
        str(ants_binary_path(ants_bin, "antsRegistration")),
        "-d",
        "2",
        "--float",
        "1",
        "--output",
        f"[{ants_cli_path(out_prefix)},{ants_cli_path(out_prefix.parent / (out_prefix.name + 'Warped.nii.gz'))}]",
        "--interpolation",
        "Linear",
        "--winsorize-image-intensities",
        "[0.005,0.995]",
        "--use-histogram-matching",
        "0",
    ]
    for tfm in initial_transforms:
        cmd.extend(["-r", ants_cli_path(tfm)])
    if not initial_transforms:
        cmd.extend(["-r", f"[{ants_cli_path(fixed_path)},{ants_cli_path(moving_path)},1]"])

    if stage == "rigid":
        cmd.extend(
            [
                "-m",
                f"MI[{ants_cli_path(fixed_path)},{ants_cli_path(moving_path)},1,32,Regular,0.25]",
                "-t",
                "Rigid[0.1]",
                "-c",
                "[400x200x100,1e-6,10]",
                "-s",
                "3x2x1vox",
                "-f",
                "8x4x2",
            ]
        )
    elif stage == "affine":
        profile = AFFINE_PROFILES.get(str(affine_profile).strip().lower(), AFFINE_PROFILES["current"])
        cmd.extend(
            [
                "-m",
                f"MI[{ants_cli_path(fixed_path)},{ants_cli_path(moving_path)},1,{int(profile['metric_bins'])},Regular,{float(profile['metric_sampling'])}]",
                "-t",
                str(profile["transform"]),
                "-c",
                str(profile["convergence"]),
                "-s",
                str(profile["smoothing_sigmas"]),
                "-f",
                str(profile["shrink_factors"]),
            ]
        )
    elif stage == "syn":
        cmd.extend(
            [
                "-m",
                f"CC[{ants_cli_path(fixed_path)},{ants_cli_path(moving_path)},1,4]",
                "-t",
                "SyN[0.1,3,0]",
                "-c",
                "[120x80x40,1e-6,10]",
                "-s",
                "2x1x0vox",
                "-f",
                "4x2x1",
            ]
        )
    else:
        raise ValueError(stage)

    cmd.extend(["-x", f"[{ants_cli_path(fixed_mask_path)},{ants_cli_path(moving_mask_path)}]"])
    return cmd


def _stage_transforms(stage_dir: Path, stage: str, rigid_mat: Path, affine_mat: Path) -> list[Path]:
    if stage == "rigid":
        return [rigid_mat]
    if stage == "affine":
        return [affine_mat, rigid_mat]
    if stage == "syn":
        transforms: list[Path] = []
        syn_warp = stage_dir / "syn_1Warp.nii.gz"
        syn_aff = stage_dir / "syn_0GenericAffine.mat"
        if syn_warp.exists():
            transforms.append(syn_warp)
        if syn_aff.exists():
            transforms.append(syn_aff)
        transforms.extend([affine_mat, rigid_mat])
        return transforms
    raise ValueError(stage)


def _compose_stage_transforms_with_initial(
    stage_dir: Path,
    stage: str,
    initial_transforms: list[Path] | None = None,
) -> list[Path]:
    init = list(initial_transforms or [])
    if stage == "rigid":
        return [stage_dir / "rigid_0GenericAffine.mat", *init]
    if stage == "affine":
        return [stage_dir / "affine_0GenericAffine.mat", *init]
    if stage == "syn":
        transforms: list[Path] = []
        syn_warp = stage_dir / "syn_1Warp.nii.gz"
        syn_aff = stage_dir / "syn_0GenericAffine.mat"
        if syn_warp.exists():
            transforms.append(syn_warp)
        if syn_aff.exists():
            transforms.append(syn_aff)
        transforms.extend(init)
        return transforms
    raise ValueError(stage)


def _write_coord_images(inputs_dir: Path, moving_shape_hw: tuple[int, int]) -> tuple[Path, Path]:
    h, w = moving_shape_hw
    yy, xx = np.indices((h, w), dtype=np.float32)
    x_path = inputs_dir / "moving_x.nii.gz"
    y_path = inputs_dir / "moving_y.nii.gz"
    write_nifti_2d(x_path, xx)
    write_nifti_2d(y_path, yy)
    return x_path, y_path


def _compute_stage_heatmap(
    ants_bin: Path,
    stage_dir: Path,
    stage: str,
    fixed_img_path: Path,
    fixed_mask_arr: np.ndarray,
    moving_coord_x: Path,
    moving_coord_y: Path,
    rigid_mat: Path,
    affine_mat: Path,
    warped_mask_path: Path,
) -> tuple[np.ndarray, Path]:
    transforms = _stage_transforms(stage_dir, stage, rigid_mat, affine_mat)
    x_warped = stage_dir / f"{stage}_coord_x_warped.nii.gz"
    y_warped = stage_dir / f"{stage}_coord_y_warped.nii.gz"
    _ants_apply(
        ants_bin,
        moving_coord_x,
        fixed_img_path,
        x_warped,
        transforms,
        interpolation="Linear",
        log_path=stage_dir / f"{stage}_coord_x.log",
    )
    _ants_apply(
        ants_bin,
        moving_coord_y,
        fixed_img_path,
        y_warped,
        transforms,
        interpolation="Linear",
        log_path=stage_dir / f"{stage}_coord_y.log",
    )
    x_arr = read_nifti_2d(x_warped)
    y_arr = read_nifti_2d(y_warped)
    yy, xx = np.indices(fixed_mask_arr.shape, dtype=np.float32)
    mag = np.sqrt((x_arr - xx) ** 2 + (y_arr - yy) ** 2)
    warped_mask = read_nifti_2d(warped_mask_path)
    support = (fixed_mask_arr > 0) | (warped_mask > 0.5)
    heat_rgb = displacement_heatmap(mag, support.astype(np.uint8))
    heat_png = stage_dir / f"{stage}_heatmap.png"
    cv2.imwrite(str(heat_png), cv2.cvtColor(heat_rgb, cv2.COLOR_RGB2BGR))
    return heat_rgb, heat_png


def _compute_stage_heatmap_with_transforms(
    ants_bin: Path,
    stage_dir: Path,
    stage: str,
    fixed_img_path: Path,
    fixed_mask_arr: np.ndarray,
    moving_coord_x: Path,
    moving_coord_y: Path,
    transforms: list[Path],
    warped_mask_path: Path,
) -> tuple[np.ndarray, Path]:
    x_warped = stage_dir / f"{stage}_coord_x_warped.nii.gz"
    y_warped = stage_dir / f"{stage}_coord_y_warped.nii.gz"
    _ants_apply(
        ants_bin,
        moving_coord_x,
        fixed_img_path,
        x_warped,
        transforms,
        interpolation="Linear",
        log_path=stage_dir / f"{stage}_coord_x.log",
    )
    _ants_apply(
        ants_bin,
        moving_coord_y,
        fixed_img_path,
        y_warped,
        transforms,
        interpolation="Linear",
        log_path=stage_dir / f"{stage}_coord_y.log",
    )
    x_arr = read_nifti_2d(x_warped)
    y_arr = read_nifti_2d(y_warped)
    yy, xx = np.indices(fixed_mask_arr.shape, dtype=np.float32)
    mag = np.sqrt((x_arr - xx) ** 2 + (y_arr - yy) ** 2)
    warped_mask = read_nifti_2d(warped_mask_path)
    support = (fixed_mask_arr > 0) | (warped_mask > 0.5)
    heat_rgb = displacement_heatmap(mag, support.astype(np.uint8))
    heat_png = stage_dir / f"{stage}_heatmap.png"
    cv2.imwrite(str(heat_png), cv2.cvtColor(heat_rgb, cv2.COLOR_RGB2BGR))
    return heat_rgb, heat_png


def _storyboard_rows_from_stage_outputs(
    fixed_gray: np.ndarray,
    fixed_mask: np.ndarray,
    moving_gray: np.ndarray,
    moving_mask: np.ndarray,
    stage_records: dict[str, dict[str, Any]],
    input_note: str,
    stage_sequence: tuple[str, ...] = ("rigid", "affine", "syn"),
) -> list[dict[str, Any]]:
    fixed_panel = gray_preview_panel(fixed_gray)
    moving_input_panel = gray_preview_panel(moving_gray)
    rows: list[dict[str, Any]] = [
        {
            "label": "Input",
            "note": input_note,
            "fixed": fixed_panel,
            "moving": moving_input_panel,
            "overlay": overlay_preview(fixed_gray, moving_gray, fixed_mask, moving_mask),
            "heatmap": displacement_heatmap(np.zeros_like(fixed_gray, dtype=np.float32), (fixed_mask > 0) | (moving_mask > 0)),
        }
    ]
    for stage in stage_sequence:
        record = stage_records.get(stage)
        if not record:
            blank = np.full((*fixed_gray.shape, 3), 235, dtype=np.uint8)
            cv2.putText(blank, f"{stage}: pending", (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (60, 60, 60), 2, cv2.LINE_AA)
            rows.append(
                {
                    "label": stage_display_name(stage),
                    "note": "pending",
                    "fixed": fixed_panel,
                    "moving": moving_input_panel,
                    "overlay": blank,
                    "heatmap": blank.copy(),
                }
            )
            continue
        label = stage_display_name(stage)
        if str(record.get("gate_status", "")).strip():
            label = f"{label} [{str(record.get('gate_status')).strip()}]"
        rows.append(
            {
                "label": label,
                "note": str(record.get("note", "")),
                "fixed": fixed_panel,
                "moving": record["moving"],
                "overlay": record["overlay"],
                "heatmap": record["heatmap"],
            }
        )
    return rows


def run_pair_registration(cfg: PairRegistrationConfig, progress_cb: ProgressCallback | None = None) -> dict[str, Any]:
    total_t0 = time.perf_counter()
    registration_backend = str(cfg.registration_backend).strip().lower() or ANTS_REGISTRATION_BACKEND
    stage_order = stage_order_for_backend(registration_backend)
    requested_stages = tuple(str(x).strip().lower() for x in tuple(cfg.run_stages or stage_order) if str(x).strip())
    if not requested_stages:
        raise ValueError("run_stages must not be empty.")
    if any(stage not in stage_order for stage in requested_stages):
        raise ValueError(f"Unsupported run_stages: {requested_stages}")
    if requested_stages != stage_order[: len(requested_stages)]:
        raise ValueError(f"run_stages must be a prefix of {stage_order}.")
    affine_profile = str(cfg.affine_profile).strip().lower() or "current"
    if registration_backend == ANTS_REGISTRATION_BACKEND and affine_profile not in AFFINE_PROFILES:
        raise ValueError(f"Unknown affine_profile: {cfg.affine_profile}")
    storyboard_titles = ("Moving", "Fixed", "Overlay", "Warp Field")

    def emit(stage: str, **kwargs: Any) -> None:
        if progress_cb is None:
            return
        progress_cb({"stage": stage, **kwargs})

    if cfg.moving_side == cfg.fixed_side:
        raise ValueError("Moving side and fixed side must be different.")
    if registration_backend == ANTS_REGISTRATION_BACKEND and not ants_binary_path(cfg.ants_bin, "antsRegistration").exists():
        raise FileNotFoundError(f"ANTs not found in {cfg.ants_bin}")
    status = str(cfg.review.get("registration_status") or "").strip().lower()
    if status != "usable":
        raise ValueError("Only Step 4 usable pairs may enter registration.")

    run_id = f"{_utc_now_stamp()}_{cfg.moving_side}_{cfg.moving_group}_to_{cfg.fixed_side}_{cfg.fixed_group}"
    run_dir = cfg.runs_root / cfg.pair_key / run_id
    inputs_dir = run_dir / "inputs"
    stages_dir = run_dir / "stages"
    run_dir.mkdir(parents=True, exist_ok=True)
    inputs_dir.mkdir(parents=True, exist_ok=True)
    stages_dir.mkdir(parents=True, exist_ok=True)

    emit("prepare", progress_percent=5, message="Preparing moving/fixed crops at common physical scale")
    prepare_t0 = time.perf_counter()
    fixed_rgb, fixed_labels, fixed_pre = _prepare_side(cfg, cfg.fixed_side, cfg.fixed_group)
    moving_rgb, moving_labels, moving_pre = _prepare_side(cfg, cfg.moving_side, cfg.moving_group)
    (
        fixed_rgb,
        fixed_labels,
        moving_rgb,
        moving_labels,
        working_info,
    ) = _resize_pair_to_working_long_edge(
        fixed_rgb,
        fixed_labels,
        moving_rgb,
        moving_labels,
        working_long_edge=int(cfg.working_long_edge),
        blur_sigma=float(cfg.pre_blur_sigma),
    )
    fixed_work_shape_hw = [int(fixed_rgb.shape[0]), int(fixed_rgb.shape[1])]
    moving_work_shape_hw = [int(moving_rgb.shape[0]), int(moving_rgb.shape[1])]
    (
        fixed_rgb,
        fixed_labels,
        moving_rgb,
        moving_labels,
        fixed_offset,
        moving_offset,
        common_canvas_shape_hw,
    ) = _common_canvas_for_pair_with_info(
        fixed_rgb,
        fixed_labels,
        moving_rgb,
        moving_labels,
    )
    fixed_pre["working_shape_hw"] = fixed_work_shape_hw
    moving_pre["working_shape_hw"] = moving_work_shape_hw
    fixed_pre["downsample_to_working_scale"] = float(working_info["downsample_to_working_scale"])
    moving_pre["downsample_to_working_scale"] = float(working_info["downsample_to_working_scale"])
    fixed_pre["pre_blur_sigma"] = float(working_info["pre_blur_sigma"])
    moving_pre["pre_blur_sigma"] = float(working_info["pre_blur_sigma"])
    fixed_pre["common_canvas_offset_xy"] = {"x": int(fixed_offset["x"]), "y": int(fixed_offset["y"])}
    moving_pre["common_canvas_offset_xy"] = {"x": int(moving_offset["x"]), "y": int(moving_offset["y"])}
    fixed_pre["common_canvas_shape_hw"] = list(common_canvas_shape_hw)
    moving_pre["common_canvas_shape_hw"] = list(common_canvas_shape_hw)
    fixed_pre.setdefault("preprocess_chain", {}).update(
        {
            "working_shape_hw": fixed_work_shape_hw,
            "common_canvas_offset_xy": {"x": int(fixed_offset["x"]), "y": int(fixed_offset["y"])},
            "common_canvas_shape_hw": list(common_canvas_shape_hw),
        }
    )
    moving_pre.setdefault("preprocess_chain", {}).update(
        {
            "working_shape_hw": moving_work_shape_hw,
            "common_canvas_offset_xy": {"x": int(moving_offset["x"]), "y": int(moving_offset["y"])},
            "common_canvas_shape_hw": list(common_canvas_shape_hw),
        }
    )
    fixed_gray = rgb_to_gray_float(fixed_rgb)
    moving_gray = rgb_to_gray_float(moving_rgb)
    fixed_mask = (fixed_labels > 0).astype(np.float32)
    moving_mask = (moving_labels > 0).astype(np.float32)
    input_metrics, input_metric_timings = compute_registration_metrics(fixed_gray, moving_gray, fixed_mask, moving_mask)
    input_note = metrics_note(input_metrics, input_metric_timings, "before registration")

    cv2.imwrite(str(inputs_dir / "fixed_rgb.png"), cv2.cvtColor(fixed_rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(inputs_dir / "moving_rgb.png"), cv2.cvtColor(moving_rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(inputs_dir / "fixed_mask_labels.png"), fixed_labels)
    cv2.imwrite(str(inputs_dir / "moving_mask_labels.png"), moving_labels)

    fixed_img_path: Path | None = None
    moving_img_path: Path | None = None
    fixed_mask_path: Path | None = None
    moving_mask_path: Path | None = None
    moving_coord_x: Path | None = None
    moving_coord_y: Path | None = None
    if registration_backend == ANTS_REGISTRATION_BACKEND:
        fixed_img_path = inputs_dir / "fixed_gray.nii.gz"
        moving_img_path = inputs_dir / "moving_gray.nii.gz"
        fixed_mask_path = inputs_dir / "fixed_mask.nii.gz"
        moving_mask_path = inputs_dir / "moving_mask.nii.gz"
        write_nifti_2d(fixed_img_path, fixed_gray)
        write_nifti_2d(moving_img_path, moving_gray)
        write_nifti_2d(fixed_mask_path, fixed_mask)
        write_nifti_2d(moving_mask_path, moving_mask)
        moving_coord_x, moving_coord_y = _write_coord_images(inputs_dir, moving_gray.shape[:2])

    stage_records: dict[str, dict[str, Any]] = {}
    storyboard_path = run_dir / "storyboard.png"
    render_storyboard(
        [
            {**row, "col_titles": storyboard_titles}
            for row in _storyboard_rows_from_stage_outputs(
                fixed_gray,
                fixed_mask,
                moving_gray,
                moving_mask,
                stage_records,
                input_note,
                requested_stages,
            )
        ],
        storyboard_path,
    )
    prepare_seconds = float(time.perf_counter() - prepare_t0)
    emit("prepare", progress_percent=10, storyboard_path=str(storyboard_path), run_dir=str(run_dir))

    rigid_mat = stages_dir / "rigid" / "rigid_0GenericAffine.mat"
    affine_mat = stages_dir / "affine" / "affine_0GenericAffine.mat"
    if registration_backend == ANTS_REGISTRATION_BACKEND:
        full_stage_plan: list[tuple[str, Any, int]] = [
            ("rigid", [], 35),
            ("affine", [rigid_mat], 65),
            ("syn", [affine_mat, rigid_mat], 100),
        ]
    else:
        full_stage_plan = [
            ("mask_rigid", None, 55),
            ("mask_similarity", None, 100),
        ]
    stage_plan = [row for row in full_stage_plan if row[0] in requested_stages]
    input_payload: dict[str, Any] = {
        "fixed_rgb_png": str(inputs_dir / "fixed_rgb.png"),
        "moving_rgb_png": str(inputs_dir / "moving_rgb.png"),
        "fixed_mask_labels_png": str(inputs_dir / "fixed_mask_labels.png"),
        "moving_mask_labels_png": str(inputs_dir / "moving_mask_labels.png"),
        "target_um_per_px": float(cfg.target_um_per_px),
        "working_long_edge": int(cfg.working_long_edge),
        "pre_blur_sigma": float(cfg.pre_blur_sigma),
        "fixed_preprocess": fixed_pre,
        "moving_preprocess": moving_pre,
        "common_canvas_shape_hw": [int(fixed_rgb.shape[0]), int(fixed_rgb.shape[1])],
    }
    if registration_backend == ANTS_REGISTRATION_BACKEND:
        input_payload.update(
            {
                "fixed_gray": str(fixed_img_path),
                "moving_gray": str(moving_img_path),
                "fixed_mask": str(fixed_mask_path),
                "moving_mask": str(moving_mask_path),
            }
        )
    manifest: dict[str, Any] = {
        "pair_key": cfg.pair_key,
        "moving_side": cfg.moving_side,
        "fixed_side": cfg.fixed_side,
        "moving_group": cfg.moving_group,
        "fixed_group": cfg.fixed_group,
        "registration_backend": registration_backend,
        "registration_mask_mode": str(cfg.registration_mask_mode),
        "run_stages": list(requested_stages),
        "affine_profile": affine_profile,
        "affine_profile_params": AFFINE_PROFILES[affine_profile] if registration_backend == ANTS_REGISTRATION_BACKEND else {},
        "mask_similarity_scale_percent": float(cfg.mask_similarity_scale_percent),
        "run_id": run_id,
        "run_dir": str(run_dir),
        "ants_bin": str(cfg.ants_bin),
        "inputs": input_payload,
        "moving_mask_source": moving_pre.get("mask_source"),
        "fixed_mask_source": fixed_pre.get("mask_source"),
        "timing_seconds": {
            "prepare": prepare_seconds,
        },
        "input_metrics": input_metrics,
        "input_metric_timing_seconds": input_metric_timings,
        "stages": {},
        "storyboard": str(storyboard_path),
        "saved_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "runner_bookkeeping": {
            "runner_mode": "best_accepted_state_runner",
            "reference_state": "current_best",
            "accepted_execution_path": ["input"],
            "best_stage_runtime": "input",
        },
    }
    manifest_path = run_dir / "run_manifest.json"
    _write_json(manifest_path, manifest)

    mask_stage_params: dict[str, dict[str, float]] = {}
    current_best_stage_runtime = "input"
    current_best_metrics_runtime = dict(input_metrics)
    current_best_ants_transforms: list[Path] = []
    current_best_mask_params: dict[str, float] | None = None
    for stage, init_spec, progress_percent in stage_plan:
        emit(stage, progress_percent=max(12, progress_percent - 20), message=f"Running {stage_display_name(stage)}")
        stage_dir = stages_dir / stage
        stage_dir.mkdir(parents=True, exist_ok=True)
        runner_reference_stage = str(current_best_stage_runtime)
        best_metrics_before = dict(current_best_metrics_runtime)
        if registration_backend == ANTS_REGISTRATION_BACKEND:
            init_tfms = list(current_best_ants_transforms)
            prefix = stage_dir / f"{stage}_"
            cmd = _stage_command(
                cfg.ants_bin,
                stage,
                fixed_img_path,
                moving_img_path,
                fixed_mask_path,
                moving_mask_path,
                prefix,
                init_tfms,
                affine_profile,
            )
            stage_t0 = time.perf_counter()
            ants_stage_t0 = time.perf_counter()
            _run_logged(cmd, stage_dir / f"{stage}.log")
            ants_registration_seconds = float(time.perf_counter() - ants_stage_t0)

            warped_img_path = stage_dir / f"{stage}_Warped.nii.gz"
            effective_tfms = _compose_stage_transforms_with_initial(stage_dir, stage, init_tfms)
            warped_mask_path = stage_dir / f"{stage}_warped_mask.nii.gz"
            _ants_apply(
                cfg.ants_bin,
                moving_mask_path,
                fixed_img_path,
                warped_mask_path,
                effective_tfms,
                interpolation="NearestNeighbor",
                log_path=stage_dir / f"{stage}_warp_mask.log",
            )

            warped_gray = read_nifti_2d(warped_img_path)
            warped_mask = read_nifti_2d(warped_mask_path)
            stage_metrics, stage_metric_timings = compute_registration_metrics(
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
            heatmap_rgb, heatmap_png = _compute_stage_heatmap_with_transforms(
                cfg.ants_bin,
                stage_dir,
                stage,
                fixed_img_path,
                fixed_mask,
                moving_coord_x,
                moving_coord_y,
                effective_tfms,
                warped_mask_path,
            )
            stage_seconds = float(time.perf_counter() - stage_t0)

            gate = strict_pareto_gate_decision(best_metrics_before, stage_metrics)
            gate["stage"] = stage
            gate["best_stage_before"] = runner_reference_stage
            gate["delta_vs_best_before"] = metric_delta_summary(stage_metrics, best_metrics_before)
            gate["delta_vs_input"] = metric_delta_summary(stage_metrics, input_metrics)
            if gate["accepted"]:
                current_best_stage_runtime = stage
                current_best_metrics_runtime = dict(stage_metrics)
                current_best_ants_transforms = list(effective_tfms)
                manifest["runner_bookkeeping"]["accepted_execution_path"].append(stage)
            gate["best_stage_after"] = current_best_stage_runtime

            stage_records[stage] = {
                "moving": gray_preview_panel(np.clip(warped_gray, 0.0, 1.0)),
                "overlay": overlay,
                "heatmap": heatmap_rgb,
                "note": (
                    f"{metrics_note(stage_metrics, stage_metric_timings, f'{stage_display_name(stage)} finished')} | "
                    f"init={runner_reference_stage} | gate={'ACCEPTED' if gate['accepted'] else 'REJECTED'} vs {runner_reference_stage}"
                ),
                "warped_image": str(warped_img_path),
                "warped_mask": str(warped_mask_path),
                "heatmap_png": str(heatmap_png),
                "transforms": [str(p) for p in effective_tfms],
                "metrics": stage_metrics,
                "metric_timing_seconds": stage_metric_timings,
                "gate_status": "ACCEPTED" if gate["accepted"] else "REJECTED",
                "runner_reference_stage": runner_reference_stage,
            }
            manifest["stages"][stage] = {
                "backend": ANTS_REGISTRATION_BACKEND,
                "warped_image": str(warped_img_path),
                "warped_mask": str(warped_mask_path),
                "heatmap_png": str(heatmap_png),
                "transforms": [str(p) for p in effective_tfms],
                "initial_transforms": [str(p) for p in init_tfms],
                "runner_reference_stage": runner_reference_stage,
                "command": cmd,
                "metrics": stage_metrics,
                "metric_timing_seconds": stage_metric_timings,
                "timing_seconds": {
                    "stage_total": stage_seconds,
                    "ants_registration": ants_registration_seconds,
                    "postprocess": float(max(stage_seconds - ants_registration_seconds, 0.0)),
                },
                "gate": gate,
            }
            warped_image_for_emit = str(warped_img_path)
        else:
            init_params = dict(current_best_mask_params) if current_best_mask_params else None
            stage_result = run_mask_shape_stage(
                stage,
                fixed_gray,
                moving_gray,
                fixed_mask,
                moving_mask,
                stage_dir,
                similarity_scale_percent=float(cfg.mask_similarity_scale_percent),
                init_params=init_params,
            )
            gate = strict_pareto_gate_decision(best_metrics_before, dict(stage_result["metrics"]))
            gate["stage"] = stage
            gate["best_stage_before"] = runner_reference_stage
            gate["delta_vs_best_before"] = metric_delta_summary(dict(stage_result["metrics"]), best_metrics_before)
            gate["delta_vs_input"] = metric_delta_summary(dict(stage_result["metrics"]), input_metrics)
            if gate["accepted"]:
                current_best_stage_runtime = stage
                current_best_metrics_runtime = dict(stage_result["metrics"])
                current_best_mask_params = dict(stage_result["transform_params"])
                manifest["runner_bookkeeping"]["accepted_execution_path"].append(stage)
            gate["best_stage_after"] = current_best_stage_runtime
            stage_records[stage] = {
                "moving": stage_result["moving"],
                "overlay": stage_result["overlay"],
                "heatmap": stage_result["heatmap"],
                "note": (
                    f"{stage_result['note']} | init={runner_reference_stage} | "
                    f"gate={'ACCEPTED' if gate['accepted'] else 'REJECTED'} vs {runner_reference_stage}"
                ),
                "warped_image": stage_result["warped_image"],
                "warped_mask": stage_result["warped_mask"],
                "heatmap_png": stage_result["heatmap_png"],
                "transforms": list(stage_result["transforms"]),
                "metrics": dict(stage_result["metrics"]),
                "metric_timing_seconds": dict(stage_result["metric_timing_seconds"]),
                "gate_status": "ACCEPTED" if gate["accepted"] else "REJECTED",
                "runner_reference_stage": runner_reference_stage,
            }
            manifest["stages"][stage] = {
                "backend": MASK_SHAPE_REGISTRATION_BACKEND,
                "warped_image": stage_result["warped_image"],
                "warped_mask": stage_result["warped_mask"],
                "heatmap_png": stage_result["heatmap_png"],
                "transforms": list(stage_result["transforms"]),
                "initial_transform_params": dict(init_params or {}),
                "runner_reference_stage": runner_reference_stage,
                "command": list(stage_result["command"]),
                "metrics": dict(stage_result["metrics"]),
                "metric_timing_seconds": dict(stage_result["metric_timing_seconds"]),
                "timing_seconds": dict(stage_result["timing_seconds"]),
                "optimizer": dict(stage_result["optimizer"]),
                "transform_params": dict(stage_result["transform_params"]),
                "transform_matrix_2x3": list(stage_result["transform_matrix_2x3"]),
                "gate": gate,
            }
            mask_stage_params[stage] = dict(stage_result["transform_params"])
            warped_image_for_emit = str(stage_result["warped_image"])

        manifest["runner_bookkeeping"]["best_stage_runtime"] = str(current_best_stage_runtime)
        manifest["timing_seconds"][stage] = float(manifest["stages"][stage]["timing_seconds"]["stage_total"])
        manifest["timing_seconds"][f"{stage}_metrics"] = float(manifest["stages"][stage]["metric_timing_seconds"].get("total", 0.0))
        _write_json(manifest_path, manifest)
        render_storyboard(
            [
                {**row, "col_titles": storyboard_titles}
                for row in _storyboard_rows_from_stage_outputs(
                    fixed_gray,
                    fixed_mask,
                    moving_gray,
                    moving_mask,
                    stage_records,
                    input_note,
                    requested_stages,
                )
            ],
            storyboard_path,
        )
        emit(
            stage,
            progress_percent=progress_percent,
            storyboard_path=str(storyboard_path),
            run_dir=str(run_dir),
            warped_image=warped_image_for_emit,
            heatmap_path=str(manifest["stages"][stage]["heatmap_png"]),
            message=f"{stage_display_name(stage)} finished",
        )

    manifest["completed_at_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    manifest["timing_seconds"]["total"] = float(time.perf_counter() - total_t0)
    gating_summary = compute_monotonic_gating_summary(manifest)
    manifest["gating_policy"] = gating_summary["policy"]
    manifest["gating_summary"] = {k: v for k, v in gating_summary.items() if k != "policy"}
    manifest["best_stage"] = gating_summary["best_stage"]
    manifest["best_metrics"] = gating_summary["best_metrics"]
    manifest["accepted_stage_path"] = list(gating_summary["accepted_stage_path"])
    manifest["final_delta_vs_input"] = dict(gating_summary["final_delta_vs_input"])
    manifest["input_gate"] = gating_summary["input"]
    manifest["runner_bookkeeping"]["accepted_stage_path_summary"] = list(gating_summary["accepted_stage_path"])
    manifest["runner_bookkeeping"]["best_stage_summary"] = gating_summary["best_stage"]
    render_storyboard(
        [
            {**row, "col_titles": storyboard_titles}
            for row in _storyboard_rows_from_stage_outputs(
                fixed_gray,
                fixed_mask,
                moving_gray,
                moving_mask,
                stage_records,
                input_note,
                requested_stages,
            )
        ],
        storyboard_path,
    )
    _write_json(manifest_path, manifest)
    return {
        "pair_key": cfg.pair_key,
        "run_id": run_id,
        "run_dir": str(run_dir),
        "storyboard_path": str(storyboard_path),
        "manifest_path": str(manifest_path),
        "best_stage": gating_summary["best_stage"],
        "accepted_stage_path": list(gating_summary["accepted_stage_path"]),
        "final_delta_vs_input": dict(gating_summary["final_delta_vs_input"]),
    }

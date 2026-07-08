from __future__ import annotations

import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
from PIL import Image
from scipy.optimize import least_squares

from .pair_registration import (
    _cross_correlation,
    _mutual_information,
    _compute_stage_heatmap_with_transforms,
    _run_logged,
    _write_coord_images,
    ants_binary_path,
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
from .physical_provenance import recover_or_load_section_physical_provenance
from .section_workspace import WorkspaceSection, load_workspace_section
from ..pipeline_adapters.slide_io import load_slide_bundle, open_slide_handle


STEP7_TARGET_UM_PER_PX = 1.0
STEP7_REGISTRATION_INPUT_PROFILE = "paired_percentile_blur8"
STEP7_RELAXED_MIN_AREA = 6
STEP7_REFINE_LOCAL_RADIUS_PX = 4
STEP7_SEED_REFINE_LOCAL_RADIUS_PX = 3
STEP7_FRONTIER_REFINE_LOCAL_RADIUS_PX = 2
STEP7_COARSE_EARLY_EXIT_TEMPLATE_CC_MIN = 0.15
STEP7_COARSE_EARLY_EXIT_CC_GAIN_MIN = 0.01
STEP7_FAST_REFINE_TOPK = 3
STEP7_HYBRID_MI_WEIGHT = 0.5
STEP7_SPARSE_REFINE_OBJECTIVE = "cc"
STEP7_TILE_EVAL_DEFAULT_MAX_WORKERS = 16
STEP7_AUTO_SCALE_COARSE_COUNT = 5
STEP7_AUTO_SCALE_TOPK = 2
STEP7_AUTO_SCALE_PRUNE_MIN_TILES = 2
STEP7_AUTO_SCALE_PRUNE_COMPOSITE_MARGIN = 0.025
STEP7_AUTO_SCALE_PRUNE_MEAN_FINAL_CC_MARGIN = 0.015
STEP7_FRONTIER_COLUMN_SHARED_WEIGHT = 0.18
STEP7_FRONTIER_COLUMN_PRIOR_WEIGHT = 0.12
STEP7_FRONTIER_COLUMN_REL_WEIGHT = 0.10
STEP7_FRONTIER_GRAPH_CC_DROP_TOL = 0.003
STEP7_FRONTIER_GRAPH_LARGE_MOVE_PX = 2.0
STEP7_FRONTIER_GRAPH_LARGE_MOVE_GAIN_MIN = 0.005

Step7ProgressCallback = Callable[[dict[str, Any]], None]


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _safe_dir_component(text: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", str(text).strip())
    value = value.strip("._-")
    return value or "session"


def _emit_step7_progress(progress_cb: Step7ProgressCallback | None, payload: dict[str, Any]) -> None:
    if progress_cb is None:
        return
    try:
        progress_cb(dict(payload))
    except Exception:
        return


def _step7_progress_percent(base_percent: float, span_percent: float, fraction: float) -> int:
    frac = max(0.0, min(1.0, float(fraction)))
    percent = float(base_percent) + float(span_percent) * frac
    return max(0, min(100, int(round(percent))))


def _step7_tile_eval_worker_count(*, max_items: int) -> int:
    if max_items <= 1:
        return 1
    requested: int | None = None
    env_value = str(os.environ.get("STEP7_TILE_EVAL_WORKERS", "")).strip()
    if env_value:
        try:
            requested = max(1, int(env_value))
        except Exception:
            requested = None
    if requested is None:
        requested = min(int(os.cpu_count() or 1), int(STEP7_TILE_EVAL_DEFAULT_MAX_WORKERS))
    return max(1, min(int(max_items), int(requested)))


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


def _read_czi_stack(path: Path) -> np.ndarray:
    try:
        from czifile import CziFile
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("czifile is required for .czi z-stack loading.") from exc
    with CziFile(str(path)) as czi:
        arr = np.asarray(czi.asarray())
        axes = str(czi.axes)
    if arr.ndim != len(axes):
        raise ValueError(f"Unexpected CZI array shape {arr.shape} for axes {axes}")
    squeeze_axes = [idx for idx, ax in enumerate(axes) if ax not in {"Z", "C", "Y", "X"}]
    if any(arr.shape[idx] != 1 for idx in squeeze_axes):
        raise ValueError(f"Unsupported CZI axes for confocal projection: axes={axes} shape={arr.shape}")
    if squeeze_axes:
        arr = np.squeeze(arr, axis=tuple(squeeze_axes))
        axes = "".join(ax for idx, ax in enumerate(axes) if idx not in squeeze_axes)
    if axes == "ZYX":
        return np.asarray(arr)
    if "Z" not in axes or "Y" not in axes or "X" not in axes:
        raise ValueError(f"CZI stack must contain Z/Y/X axes, got axes={axes}")
    if "C" in axes:
        target_axes = "ZCYX"
    else:
        target_axes = "ZYX"
    arr = np.transpose(arr, [axes.index(ax) for ax in target_axes])
    return np.asarray(arr)


def _read_confocal_stack(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix in {".tif", ".tiff"}:
        return _read_tiff_stack(path)
    if suffix == ".czi":
        return _read_czi_stack(path)
    raise ValueError(f"Unsupported confocal stack format: {path.name}")


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


def _runtime_local_path(path: Path | str) -> Path:
    p = Path(path)
    s = str(p)
    if os.name == "nt" and s.startswith("/mnt/") and len(s) > 6:
        drive = s[5].upper()
        tail = s[6:].replace("/", "\\").lstrip("\\")
        return Path(f"{drive}:\\{tail}")
    if os.name != "nt" and len(s) >= 3 and s[1:3] == ":\\":
        drive = s[0].lower()
        tail = s[3:].replace("\\", "/").lstrip("/")
        return Path(f"/mnt/{drive}/{tail}")
    return p


def _portable_basename(path: Path | str) -> str:
    s = str(path).replace("\\", "/")
    return s.rsplit("/", 1)[-1]


def _invert_confocal_u8(arr: np.ndarray) -> np.ndarray:
    return (255 - np.asarray(arr, dtype=np.uint8)).astype(np.uint8)


def _masked_percentile_normalize_u8(
    image_u8: np.ndarray,
    mask_u8: np.ndarray,
    *,
    lo_pct: float = 1.0,
    hi_pct: float = 99.0,
) -> np.ndarray:
    arr = np.asarray(image_u8, dtype=np.float32)
    valid = np.asarray(mask_u8) > 0
    vals = arr[valid]
    if vals.size == 0:
        return np.full_like(np.asarray(image_u8, dtype=np.uint8), 255, dtype=np.uint8)
    lo = float(np.percentile(vals, lo_pct))
    hi = float(np.percentile(vals, hi_pct))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(vals.min())
        hi = float(vals.max())
    if hi <= lo:
        return np.full_like(np.asarray(image_u8, dtype=np.uint8), 255, dtype=np.uint8)
    scaled = np.clip((arr - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    out = np.clip(np.round(scaled * 255.0), 0, 255).astype(np.uint8)
    out[~valid] = 255
    return out


def _gaussian_blur_u8(image_u8: np.ndarray, mask_u8: np.ndarray, *, sigma: float) -> np.ndarray:
    blurred = cv2.GaussianBlur(np.asarray(image_u8, dtype=np.uint8), (0, 0), sigmaX=float(sigma), sigmaY=float(sigma))
    out = np.full_like(blurred, 255, dtype=np.uint8)
    inside = np.asarray(mask_u8) > 0
    out[inside] = blurred[inside]
    return out


def _apply_clahe_u8(image_u8: np.ndarray, mask_u8: np.ndarray, *, clip_limit: float = 2.5, tile_grid: int = 8) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=(int(tile_grid), int(tile_grid)))
    enhanced = clahe.apply(np.asarray(image_u8, dtype=np.uint8))
    out = np.full_like(enhanced, 255, dtype=np.uint8)
    inside = np.asarray(mask_u8) > 0
    out[inside] = enhanced[inside]
    return out


def _gamma_u8(image_u8: np.ndarray, mask_u8: np.ndarray, *, gamma: float) -> np.ndarray:
    arr = np.asarray(image_u8, dtype=np.float32) / 255.0
    out = np.full_like(arr, 1.0, dtype=np.float32)
    inside = np.asarray(mask_u8) > 0
    out[inside] = np.power(np.clip(arr[inside], 0.0, 1.0), float(gamma))
    return np.clip(np.round(out * 255.0), 0, 255).astype(np.uint8)


def _relaxed_binary_dark_u8(
    image_u8: np.ndarray,
    mask_u8: np.ndarray,
    *,
    keep_quantile: float = 60.0,
    min_area: int = STEP7_RELAXED_MIN_AREA,
) -> np.ndarray:
    inside = np.asarray(mask_u8) > 0
    vals = np.asarray(image_u8, dtype=np.uint8)[inside]
    out = np.full_like(np.asarray(image_u8, dtype=np.uint8), 255, dtype=np.uint8)
    if vals.size < 64:
        return out
    thr = float(np.percentile(vals, keep_quantile))
    fiber = np.zeros_like(out, dtype=np.uint8)
    fiber[inside & (np.asarray(image_u8, dtype=np.uint8) <= thr)] = 255
    num, labels, stats, _ = cv2.connectedComponentsWithStats(fiber, connectivity=8)
    keep = np.zeros_like(fiber, dtype=np.uint8)
    for idx in range(1, int(num)):
        if int(stats[idx, cv2.CC_STAT_AREA]) >= int(min_area):
            keep[labels == idx] = 255
    out[keep > 0] = 0
    return out


def _distance_transform_dark_u8(image_u8: np.ndarray, mask_u8: np.ndarray, *, keep_quantile: float = 60.0) -> np.ndarray:
    binary = _relaxed_binary_dark_u8(
        image_u8,
        mask_u8,
        keep_quantile=keep_quantile,
        min_area=STEP7_RELAXED_MIN_AREA,
    )
    fiber = (binary == 0).astype(np.uint8)
    dt = cv2.distanceTransform(fiber, cv2.DIST_L2, 3)
    inside = np.asarray(mask_u8) > 0
    vals = dt[inside]
    out = np.full_like(np.asarray(image_u8, dtype=np.uint8), 255, dtype=np.uint8)
    if vals.size == 0 or float(vals.max()) <= 1e-6:
        return out
    dt_norm = np.clip(dt / max(float(vals.max()), 1e-6), 0.0, 1.0)
    out[inside] = np.clip(np.round((1.0 - dt_norm[inside]) * 255.0), 0, 255).astype(np.uint8)
    return out


def _masked_histogram_match_u8(
    src_u8: np.ndarray,
    src_mask: np.ndarray,
    ref_u8: np.ndarray,
    ref_mask: np.ndarray,
) -> np.ndarray:
    src = np.asarray(src_u8, dtype=np.uint8)
    ref = np.asarray(ref_u8, dtype=np.uint8)
    src_valid = src[np.asarray(src_mask) > 0]
    ref_valid = ref[np.asarray(ref_mask) > 0]
    if src_valid.size < 64 or ref_valid.size < 64:
        return src.copy()
    s_values, s_counts = np.unique(src_valid, return_counts=True)
    r_values, r_counts = np.unique(ref_valid, return_counts=True)
    s_quantiles = np.cumsum(s_counts).astype(np.float64)
    s_quantiles /= max(s_quantiles[-1], 1.0)
    r_quantiles = np.cumsum(r_counts).astype(np.float64)
    r_quantiles /= max(r_quantiles[-1], 1.0)
    matched_values = np.interp(s_quantiles, r_quantiles, r_values).astype(np.float32)
    lut = np.arange(256, dtype=np.float32)
    lut[s_values] = matched_values
    return lut[src].astype(np.uint8)


def _prepare_step7_registration_inputs(
    fixed_gray: np.ndarray,
    moving_gray: np.ndarray,
    fixed_mask: np.ndarray,
    moving_mask: np.ndarray,
    *,
    profile: str = STEP7_REGISTRATION_INPUT_PROFILE,
) -> dict[str, Any]:
    fixed_u8 = np.clip(np.round(np.asarray(fixed_gray, dtype=np.float32) * 255.0), 0, 255).astype(np.uint8)
    moving_u8 = np.clip(np.round(np.asarray(moving_gray, dtype=np.float32) * 255.0), 0, 255).astype(np.uint8)
    fixed_mask_u8 = np.where(np.asarray(fixed_mask) > 0, 255, 0).astype(np.uint8)
    moving_mask_u8 = np.where(np.asarray(moving_mask) > 0, 255, 0).astype(np.uint8)
    profile_key = str(profile or STEP7_REGISTRATION_INPUT_PROFILE).strip().lower()

    if profile_key == "paired_percentile_raw":
        fixed_proc_u8 = _masked_percentile_normalize_u8(fixed_u8, fixed_mask_u8)
        moving_proc_u8 = _masked_percentile_normalize_u8(moving_u8, moving_mask_u8)
        description = "paired percentile normalization only"
    elif profile_key == "paired_percentile_blur1":
        sigma = 1.0
        fixed_proc_u8 = _gaussian_blur_u8(_masked_percentile_normalize_u8(fixed_u8, fixed_mask_u8), fixed_mask_u8, sigma=sigma)
        moving_proc_u8 = _gaussian_blur_u8(_masked_percentile_normalize_u8(moving_u8, moving_mask_u8), moving_mask_u8, sigma=sigma)
        description = "paired percentile normalization + Gaussian blur sigma=1"
    elif profile_key == "paired_percentile_blur2":
        sigma = 2.0
        fixed_proc_u8 = _gaussian_blur_u8(_masked_percentile_normalize_u8(fixed_u8, fixed_mask_u8), fixed_mask_u8, sigma=sigma)
        moving_proc_u8 = _gaussian_blur_u8(_masked_percentile_normalize_u8(moving_u8, moving_mask_u8), moving_mask_u8, sigma=sigma)
        description = "paired percentile normalization + Gaussian blur sigma=2"
    elif profile_key == "paired_percentile_blur4":
        sigma = 4.0
        fixed_proc_u8 = _gaussian_blur_u8(_masked_percentile_normalize_u8(fixed_u8, fixed_mask_u8), fixed_mask_u8, sigma=sigma)
        moving_proc_u8 = _gaussian_blur_u8(_masked_percentile_normalize_u8(moving_u8, moving_mask_u8), moving_mask_u8, sigma=sigma)
        description = "paired percentile normalization + Gaussian blur sigma=4"
    elif profile_key == "paired_percentile_blur6":
        sigma = 6.0
        fixed_proc_u8 = _gaussian_blur_u8(_masked_percentile_normalize_u8(fixed_u8, fixed_mask_u8), fixed_mask_u8, sigma=sigma)
        moving_proc_u8 = _gaussian_blur_u8(_masked_percentile_normalize_u8(moving_u8, moving_mask_u8), moving_mask_u8, sigma=sigma)
        description = "paired percentile normalization + Gaussian blur sigma=6"
    elif profile_key == "paired_percentile_clahe_blur2":
        sigma = 2.0
        fixed_pct_u8 = _masked_percentile_normalize_u8(fixed_u8, fixed_mask_u8)
        moving_pct_u8 = _masked_percentile_normalize_u8(moving_u8, moving_mask_u8)
        fixed_proc_u8 = _gaussian_blur_u8(_apply_clahe_u8(fixed_pct_u8, fixed_mask_u8, clip_limit=2.5, tile_grid=8), fixed_mask_u8, sigma=sigma)
        moving_proc_u8 = _gaussian_blur_u8(_apply_clahe_u8(moving_pct_u8, moving_mask_u8, clip_limit=2.5, tile_grid=8), moving_mask_u8, sigma=sigma)
        description = "paired percentile + CLAHE + Gaussian blur sigma=2"
    elif profile_key == "paired_percentile_clahe_blur3":
        sigma = 3.0
        fixed_pct_u8 = _masked_percentile_normalize_u8(fixed_u8, fixed_mask_u8)
        moving_pct_u8 = _masked_percentile_normalize_u8(moving_u8, moving_mask_u8)
        fixed_proc_u8 = _gaussian_blur_u8(_apply_clahe_u8(fixed_pct_u8, fixed_mask_u8, clip_limit=3.0, tile_grid=8), fixed_mask_u8, sigma=sigma)
        moving_proc_u8 = _gaussian_blur_u8(_apply_clahe_u8(moving_pct_u8, moving_mask_u8, clip_limit=3.0, tile_grid=8), moving_mask_u8, sigma=sigma)
        description = "paired percentile + CLAHE + Gaussian blur sigma=3"
    elif profile_key == "moving_percentile_hist_blur4":
        fixed_proc_u8 = _gaussian_blur_u8(_masked_percentile_normalize_u8(fixed_u8, fixed_mask_u8), fixed_mask_u8, sigma=4.0)
        moving_pct_u8 = _masked_percentile_normalize_u8(moving_u8, moving_mask_u8)
        moving_hist_u8 = _masked_histogram_match_u8(moving_pct_u8, moving_mask_u8, fixed_proc_u8, fixed_mask_u8)
        moving_proc_u8 = _gaussian_blur_u8(moving_hist_u8, moving_mask_u8, sigma=4.0)
        description = "fixed percentile+blur4; moving percentile + histmatch + blur4"
    elif profile_key == "moving_aggressive_clahe_hist_blur3":
        fixed_proc_u8 = _gaussian_blur_u8(_masked_percentile_normalize_u8(fixed_u8, fixed_mask_u8), fixed_mask_u8, sigma=3.0)
        moving_pct_u8 = _masked_percentile_normalize_u8(moving_u8, moving_mask_u8, lo_pct=0.5, hi_pct=99.5)
        moving_clahe_u8 = _apply_clahe_u8(moving_pct_u8, moving_mask_u8, clip_limit=5.0, tile_grid=8)
        moving_hist_u8 = _masked_histogram_match_u8(moving_clahe_u8, moving_mask_u8, fixed_proc_u8, fixed_mask_u8)
        moving_proc_u8 = _gaussian_blur_u8(moving_hist_u8, moving_mask_u8, sigma=3.0)
        description = "fixed percentile+blur3; moving aggressive CLAHE + histmatch + blur3"
    elif profile_key == "moving_gamma_clahe_hist_blur4":
        fixed_proc_u8 = _gaussian_blur_u8(_masked_percentile_normalize_u8(fixed_u8, fixed_mask_u8), fixed_mask_u8, sigma=4.0)
        moving_pct_u8 = _masked_percentile_normalize_u8(moving_u8, moving_mask_u8, lo_pct=0.5, hi_pct=99.5)
        moving_gamma_u8 = _gamma_u8(moving_pct_u8, moving_mask_u8, gamma=1.8)
        moving_clahe_u8 = _apply_clahe_u8(moving_gamma_u8, moving_mask_u8, clip_limit=6.0, tile_grid=8)
        moving_hist_u8 = _masked_histogram_match_u8(moving_clahe_u8, moving_mask_u8, fixed_proc_u8, fixed_mask_u8)
        moving_proc_u8 = _gaussian_blur_u8(moving_hist_u8, moving_mask_u8, sigma=4.0)
        description = "fixed percentile+blur4; moving gamma + CLAHE + histmatch + blur4"
    elif profile_key == "paired_relaxed_binary_q60":
        fixed_proc_u8 = _relaxed_binary_dark_u8(_masked_percentile_normalize_u8(fixed_u8, fixed_mask_u8), fixed_mask_u8, keep_quantile=60.0)
        moving_proc_u8 = _relaxed_binary_dark_u8(_masked_percentile_normalize_u8(moving_u8, moving_mask_u8), moving_mask_u8, keep_quantile=60.0)
        description = "paired relaxed binary q60"
    elif profile_key == "paired_relaxed_binary_q70":
        fixed_proc_u8 = _relaxed_binary_dark_u8(_masked_percentile_normalize_u8(fixed_u8, fixed_mask_u8), fixed_mask_u8, keep_quantile=70.0)
        moving_proc_u8 = _relaxed_binary_dark_u8(_masked_percentile_normalize_u8(moving_u8, moving_mask_u8), moving_mask_u8, keep_quantile=70.0)
        description = "paired relaxed binary q70"
    elif profile_key == "paired_relaxed_binary_dt_q60":
        fixed_proc_u8 = _distance_transform_dark_u8(_masked_percentile_normalize_u8(fixed_u8, fixed_mask_u8), fixed_mask_u8, keep_quantile=60.0)
        moving_proc_u8 = _distance_transform_dark_u8(_masked_percentile_normalize_u8(moving_u8, moving_mask_u8), moving_mask_u8, keep_quantile=60.0)
        description = "paired relaxed binary q60 + distance transform"
    elif profile_key == "paired_pct1_99_clahe2p5_blur6":
        sigma = 6.0
        fixed_pct_u8 = _masked_percentile_normalize_u8(fixed_u8, fixed_mask_u8)
        moving_pct_u8 = _masked_percentile_normalize_u8(moving_u8, moving_mask_u8)
        fixed_clahe_u8 = _apply_clahe_u8(fixed_pct_u8, fixed_mask_u8, clip_limit=2.5, tile_grid=8)
        moving_clahe_u8 = _apply_clahe_u8(moving_pct_u8, moving_mask_u8, clip_limit=2.5, tile_grid=8)
        fixed_proc_u8 = _gaussian_blur_u8(fixed_clahe_u8, fixed_mask_u8, sigma=sigma)
        moving_proc_u8 = _gaussian_blur_u8(moving_clahe_u8, moving_mask_u8, sigma=sigma)
        description = "paired percentile 1-99 + CLAHE clip=2.5 + Gaussian blur sigma=6"
    else:
        sigma = 8.0
        fixed_proc_u8 = _gaussian_blur_u8(_masked_percentile_normalize_u8(fixed_u8, fixed_mask_u8), fixed_mask_u8, sigma=sigma)
        moving_proc_u8 = _gaussian_blur_u8(_masked_percentile_normalize_u8(moving_u8, moving_mask_u8), moving_mask_u8, sigma=sigma)
        profile_key = STEP7_REGISTRATION_INPUT_PROFILE
        description = "paired percentile normalization + Gaussian blur sigma=8"

    return {
        "profile": profile_key,
        "description": description,
        "fixed_u8": fixed_proc_u8,
        "moving_u8": moving_proc_u8,
        "fixed_gray": fixed_proc_u8.astype(np.float32) / 255.0,
        "moving_gray": moving_proc_u8.astype(np.float32) / 255.0,
    }


def _prepare_step7_independent_side_u8(
    image_u8: np.ndarray,
    mask_u8: np.ndarray,
    *,
    profile: str,
) -> np.ndarray | None:
    profile_key = str(profile or STEP7_REGISTRATION_INPUT_PROFILE).strip().lower()
    if profile_key == "paired_percentile_raw":
        return _masked_percentile_normalize_u8(image_u8, mask_u8)
    if profile_key == "paired_percentile_blur1":
        return _gaussian_blur_u8(_masked_percentile_normalize_u8(image_u8, mask_u8), mask_u8, sigma=1.0)
    if profile_key == "paired_percentile_blur2":
        return _gaussian_blur_u8(_masked_percentile_normalize_u8(image_u8, mask_u8), mask_u8, sigma=2.0)
    if profile_key == "paired_percentile_blur4":
        return _gaussian_blur_u8(_masked_percentile_normalize_u8(image_u8, mask_u8), mask_u8, sigma=4.0)
    if profile_key == "paired_percentile_blur6":
        return _gaussian_blur_u8(_masked_percentile_normalize_u8(image_u8, mask_u8), mask_u8, sigma=6.0)
    if profile_key in {"paired_percentile_blur8", STEP7_REGISTRATION_INPUT_PROFILE.lower()}:
        return _gaussian_blur_u8(_masked_percentile_normalize_u8(image_u8, mask_u8), mask_u8, sigma=8.0)
    if profile_key == "paired_percentile_clahe_blur2":
        pct_u8 = _masked_percentile_normalize_u8(image_u8, mask_u8)
        return _gaussian_blur_u8(_apply_clahe_u8(pct_u8, mask_u8, clip_limit=2.5, tile_grid=8), mask_u8, sigma=2.0)
    if profile_key == "paired_percentile_clahe_blur3":
        pct_u8 = _masked_percentile_normalize_u8(image_u8, mask_u8)
        return _gaussian_blur_u8(_apply_clahe_u8(pct_u8, mask_u8, clip_limit=3.0, tile_grid=8), mask_u8, sigma=3.0)
    if profile_key == "paired_relaxed_binary_q60":
        return _relaxed_binary_dark_u8(_masked_percentile_normalize_u8(image_u8, mask_u8), mask_u8, keep_quantile=60.0)
    if profile_key == "paired_relaxed_binary_q70":
        return _relaxed_binary_dark_u8(_masked_percentile_normalize_u8(image_u8, mask_u8), mask_u8, keep_quantile=70.0)
    if profile_key == "paired_relaxed_binary_dt_q60":
        return _distance_transform_dark_u8(_masked_percentile_normalize_u8(image_u8, mask_u8), mask_u8, keep_quantile=60.0)
    if profile_key == "paired_pct1_99_clahe2p5_blur6":
        pct_u8 = _masked_percentile_normalize_u8(image_u8, mask_u8)
        clahe_u8 = _apply_clahe_u8(pct_u8, mask_u8, clip_limit=2.5, tile_grid=8)
        return _gaussian_blur_u8(clahe_u8, mask_u8, sigma=6.0)
    return None


def _get_step7_moving_local_profile_cache(
    *,
    ctx: dict[str, Any],
    registration_input_profile: str,
) -> dict[str, Any] | None:
    cache = ctx.setdefault("_moving_local_profile_cache", {})
    profile_key = str(registration_input_profile or STEP7_REGISTRATION_INPUT_PROFILE).strip().lower()
    if profile_key in cache:
        cached = cache.get(profile_key)
        return cached if isinstance(cached, dict) else None
    moving_u8 = np.asarray(ctx["tile_img_u8"], dtype=np.uint8)
    moving_mask_u8 = np.where(np.asarray(ctx["tile_local_mask"]) > 0, 255, 0).astype(np.uint8)
    moving_proc_u8 = _prepare_step7_independent_side_u8(
        moving_u8,
        moving_mask_u8,
        profile=profile_key,
    )
    if moving_proc_u8 is None:
        cache[profile_key] = None
        return None
    cached = {
        "profile": profile_key,
        "moving_u8": moving_proc_u8,
        "moving_gray": moving_proc_u8.astype(np.float32) / 255.0,
    }
    cache[profile_key] = cached
    return cached


def _sort_confocal_paths(paths: list[Path]) -> list[Path]:
    def _sort_key(path: Path) -> tuple[int, str]:
        match = re.search(r"_S(\d+)", path.stem, flags=re.IGNORECASE)
        if match:
            return (int(match.group(1)), path.name.lower())
        return (10**9, path.name.lower())

    return sorted(paths, key=_sort_key)


def _stack_series_index(path: Path) -> int | None:
    match = re.search(r"_S(\d+)(?:\.ome)?$", path.stem, flags=re.IGNORECASE)
    if not match:
        return None
    return int(match.group(1))


def _sibling_czi_for_tiff_tiles(paths: list[Path]) -> Path | None:
    if not paths:
        return None
    first = paths[0]
    base = re.sub(r"_S\d+(?:\.ome)?$", "", first.stem, flags=re.IGNORECASE)
    candidate = first.parent.parent / f"{base}.czi"
    return candidate if candidate.exists() else None


def _hash_confocal_stack_content(stack: np.ndarray) -> str:
    array = np.ascontiguousarray(stack)
    return hashlib.sha1(array.tobytes()).hexdigest()


def analyze_confocal_duplicate_stacks(
    paths: list[Path],
    *,
    max_report_groups: int = 6,
    max_group_names: int = 8,
) -> dict[str, Any]:
    clean_paths = _sort_confocal_paths([Path(p) for p in paths])
    report: dict[str, Any] = {
        "checked": False,
        "source_count": len(clean_paths),
        "unique_stack_count": len(clean_paths),
        "duplicate_group_count": 0,
        "duplicate_stack_count": 0,
        "all_tiles_identical": False,
        "duplicate_groups": [],
        "source_paths": [str(path) for path in clean_paths],
    }
    if len(clean_paths) <= 1:
        report["reason"] = "single_source"
        return report
    suffixes = {path.suffix.lower() for path in clean_paths}
    if suffixes - {".tif", ".tiff"}:
        report["reason"] = "non_tiff_sources"
        return report

    groups: dict[str, dict[str, Any]] = {}
    for path in clean_paths:
        stack = _read_confocal_stack(path)
        stack_hash = _hash_confocal_stack_content(stack)
        entry = groups.setdefault(
            stack_hash,
            {
                "hash": stack_hash,
                "shape": [int(v) for v in stack.shape],
                "names": [],
                "paths": [],
            },
        )
        entry["names"].append(path.name)
        entry["paths"].append(str(path))

    duplicate_groups = [entry for entry in groups.values() if len(entry["paths"]) > 1]
    duplicate_groups.sort(key=lambda entry: (-len(entry["paths"]), str(entry["names"][0]).lower()))
    limited_groups: list[dict[str, Any]] = []
    for idx, entry in enumerate(duplicate_groups[: max(0, int(max_report_groups))], start=1):
        names = list(entry["names"])
        preview_names = names[: max(1, int(max_group_names))]
        limited_groups.append(
            {
                "group_index": idx,
                "hash": str(entry["hash"]),
                "shape": list(entry["shape"]),
                "count": len(entry["paths"]),
                "names": preview_names,
                "truncated_name_count": max(0, len(names) - len(preview_names)),
                "paths": list(entry["paths"]),
            }
        )

    report.update(
        {
            "checked": True,
            "unique_stack_count": len(groups),
            "duplicate_group_count": len(duplicate_groups),
            "duplicate_stack_count": sum(max(0, len(entry["paths"]) - 1) for entry in duplicate_groups),
            "all_tiles_identical": len(groups) == 1 and len(clean_paths) > 1,
            "duplicate_groups": limited_groups,
        }
    )
    if report["duplicate_group_count"] <= 0:
        report["reason"] = "no_duplicates_detected"
    return report


def _czi_mosaic_layout_px(path: Path) -> dict[int, tuple[int, int]]:
    try:
        from czifile import CziFile
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("czifile is required to recover tile layout from .czi metadata.") from exc
    with CziFile(str(path)) as czi:
        entries = list(czi.filtered_subblock_directory)
    layout: dict[int, tuple[int, int]] = {}
    for entry in entries:
        idx = getattr(entry, "mosaic_index", None)
        if idx is None:
            continue
        x_px = None
        y_px = None
        dim_entries = getattr(entry, "dimension_entries", None)
        if dim_entries:
            for de in dim_entries:
                dim = str(getattr(de, "dimension", "")).upper()
                if dim == "X":
                    x_px = int(getattr(de, "start", 0))
                elif dim == "Y":
                    y_px = int(getattr(de, "start", 0))
        if x_px is None or y_px is None:
            axes = str(getattr(entry, "axes", ""))
            starts = tuple(getattr(entry, "start", ()))
            if axes and starts and len(axes) == len(starts):
                try:
                    x_px = int(starts[axes.index("X")])
                    y_px = int(starts[axes.index("Y")])
                except ValueError:
                    x_px = None
                    y_px = None
        if x_px is None or y_px is None:
            starts = tuple(getattr(entry, "start", ()))
            if len(starts) >= 3:
                # Observed Windows czifile builds may expose neither axes nor dimension_entries,
                # but still provide subblock starts ordered like ... Y, X, 0.
                try:
                    y_px = int(starts[-3])
                    x_px = int(starts[-2])
                except Exception:
                    x_px = None
                    y_px = None
        if x_px is None or y_px is None:
            continue
        layout.setdefault(int(idx), (x_px, y_px))
    if not layout:
        raise ValueError(f"Could not recover mosaic layout from {path.name}")
    min_x = min(x for x, _ in layout.values())
    min_y = min(y for _, y in layout.values())
    return {idx: (int(x - min_x), int(y - min_y)) for idx, (x, y) in layout.items()}


def _czi_physical_um_per_px(path: Path) -> tuple[float, float]:
    try:
        from czifile import CziFile
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("czifile is required to read .czi physical scale.") from exc
    with CziFile(str(path)) as czi:
        root = ET.fromstring(czi.metadata())
    x_m = None
    y_m = None
    for dist in root.findall(".//Distance"):
        axis = str(dist.attrib.get("Id", "")).upper()
        value_node = dist.find("./Value")
        if value_node is None or value_node.text is None:
            continue
        value = float(value_node.text)
        if axis == "X":
            x_m = value
        elif axis == "Y":
            y_m = value
    if x_m is None or y_m is None:
        raise ValueError(f"Missing X/Y physical scaling in {path.name}")
    return float(x_m * 1e6), float(y_m * 1e6)


def _ome_tiff_physical_um_per_px(path: Path) -> tuple[float, float]:
    try:
        import tifffile
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("tifffile and XML parser are required to read OME-TIFF physical scale.") from exc
    with tifffile.TiffFile(str(path)) as tf:
        ome = tf.ome_metadata
    root = ET.fromstring(ome)
    pixels = root.find(".//{*}Pixels")
    if pixels is None:
        raise ValueError(f"OME-TIFF is missing Pixels metadata: {path.name}")
    x_um = float(pixels.attrib["PhysicalSizeX"])
    y_um = float(pixels.attrib["PhysicalSizeY"])
    return x_um, y_um


def _resample_projection_to_target_um_per_px(
    projection_u8: np.ndarray,
    *,
    source_um_per_px_xy: tuple[float, float] | None,
    target_um_per_px_xy: tuple[float, float] | None,
) -> tuple[np.ndarray, dict[str, Any]]:
    if source_um_per_px_xy is None or target_um_per_px_xy is None:
        return projection_u8, {"resampled": False, "reason": "missing_physical_scale"}
    src_x, src_y = float(source_um_per_px_xy[0]), float(source_um_per_px_xy[1])
    dst_x, dst_y = float(target_um_per_px_xy[0]), float(target_um_per_px_xy[1])
    if min(src_x, src_y, dst_x, dst_y) <= 0:
        return projection_u8, {"resampled": False, "reason": "nonpositive_physical_scale"}
    scale_x = src_x / dst_x
    scale_y = src_y / dst_y
    if abs(scale_x - 1.0) < 1e-3 and abs(scale_y - 1.0) < 1e-3:
        return projection_u8, {
            "resampled": False,
            "reason": "already_matched",
            "source_um_per_px_xy": [src_x, src_y],
            "target_um_per_px_xy": [dst_x, dst_y],
        }
    h, w = projection_u8.shape[:2]
    new_w = max(1, int(round(w * scale_x)))
    new_h = max(1, int(round(h * scale_y)))
    resized = cv2.resize(projection_u8, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, {
        "resampled": True,
        "source_um_per_px_xy": [src_x, src_y],
        "target_um_per_px_xy": [dst_x, dst_y],
        "scale_xy": [scale_x, scale_y],
        "input_shape_hw": [int(h), int(w)],
        "output_shape_hw": [int(new_h), int(new_w)],
    }


def _resample_mask_to_target_um_per_px(
    mask_u8: np.ndarray,
    *,
    source_um_per_px_xy: tuple[float, float] | None,
    target_um_per_px_xy: tuple[float, float] | None,
) -> np.ndarray:
    if source_um_per_px_xy is None or target_um_per_px_xy is None:
        return np.asarray(mask_u8, dtype=np.uint8)
    src_x, src_y = float(source_um_per_px_xy[0]), float(source_um_per_px_xy[1])
    dst_x, dst_y = float(target_um_per_px_xy[0]), float(target_um_per_px_xy[1])
    if min(src_x, src_y, dst_x, dst_y) <= 0:
        return np.asarray(mask_u8, dtype=np.uint8)
    scale_x = src_x / dst_x
    scale_y = src_y / dst_y
    if abs(scale_x - 1.0) < 1e-3 and abs(scale_y - 1.0) < 1e-3:
        return np.asarray(mask_u8, dtype=np.uint8)
    h, w = mask_u8.shape[:2]
    new_w = max(1, int(round(w * scale_x)))
    new_h = max(1, int(round(h * scale_y)))
    resized = cv2.resize(np.asarray(mask_u8, dtype=np.uint8), (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    return np.where(resized > 0, 255, 0).astype(np.uint8)


def _estimate_vertical_tile_shift(left_u8: np.ndarray, right_u8: np.ndarray, overlap_px: int) -> tuple[int, float]:
    overlap_px = max(24, min(overlap_px, left_u8.shape[1], right_u8.shape[1]))
    left_strip = left_u8[:, -overlap_px:].astype(np.float32)
    right_strip = right_u8[:, :overlap_px].astype(np.float32)
    if left_strip.shape != right_strip.shape:
        common_h = min(left_strip.shape[0], right_strip.shape[0])
        left_strip = left_strip[:common_h]
        right_strip = right_strip[:common_h]
    shift, response = cv2.phaseCorrelate(left_strip, right_strip)
    dy = int(round(float(np.clip(shift[1], -64.0, 64.0))))
    return dy, float(response)


def _stitch_projected_tiles(tiles: list[np.ndarray], *, nominal_overlap_fraction: float = 0.10) -> tuple[np.ndarray, dict[str, Any]]:
    if not tiles:
        raise ValueError("No projected tiles provided for stitching.")
    if len(tiles) == 1:
        return tiles[0], {"mode": "single"}

    nominal_overlap_fraction = float(np.clip(nominal_overlap_fraction, 0.0, 0.45))
    positions: list[tuple[int, int]] = [(0, 0)]
    x_cursor = 0
    y_cursor = 0
    pairwise_info: list[dict[str, Any]] = []
    prev = tiles[0]
    for idx, tile in enumerate(tiles[1:], start=1):
        nominal_overlap_px = int(round(min(prev.shape[1], tile.shape[1]) * nominal_overlap_fraction))
        nominal_overlap_px = max(1, nominal_overlap_px)
        dy_px, response = _estimate_vertical_tile_shift(prev, tile, nominal_overlap_px)
        x_cursor += int(prev.shape[1] - nominal_overlap_px)
        y_cursor += int(dy_px)
        positions.append((x_cursor, y_cursor))
        pairwise_info.append(
            {
                "tile_index": idx,
                "nominal_overlap_px": nominal_overlap_px,
                "dy_px": int(dy_px),
                "phase_response": float(response),
            }
        )
        prev = tile

    min_x = min(x for x, _y in positions)
    min_y = min(y for _x, y in positions)
    norm_positions = [(x - min_x, y - min_y) for x, y in positions]
    canvas_w = max(x + tile.shape[1] for tile, (x, _y) in zip(tiles, norm_positions))
    canvas_h = max(y + tile.shape[0] for tile, (_x, y) in zip(tiles, norm_positions))

    accum = np.zeros((canvas_h, canvas_w), dtype=np.float32)
    counts = np.zeros((canvas_h, canvas_w), dtype=np.float32)
    for tile, (x0, y0) in zip(tiles, norm_positions):
        h, w = tile.shape[:2]
        accum[y0 : y0 + h, x0 : x0 + w] += tile.astype(np.float32)
        counts[y0 : y0 + h, x0 : x0 + w] += 1.0
    stitched = np.where(counts > 0, accum / np.maximum(counts, 1.0), 0.0).astype(np.uint8)
    info = {
        "mode": "multi_tiff_strip",
        "tile_count": len(tiles),
        "nominal_overlap_fraction": nominal_overlap_fraction,
        "tile_positions_xy": [[int(x), int(y)] for x, y in norm_positions],
        "pairwise_alignment": pairwise_info,
        "stitched_shape_hw": [int(stitched.shape[0]), int(stitched.shape[1])],
    }
    return stitched, info


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


@dataclass
class ConfocalProjectionBundle:
    projection_u8: np.ndarray
    channel_count: int
    source_mode: str
    source_paths: list[str]
    source_shapes: list[list[int]]
    stitch_info: dict[str, Any]
    physical_um_per_px_xy: tuple[float, float] | None


@dataclass
class MyelinFixedBundle:
    rgb: np.ndarray
    labels: np.ndarray
    preview_um_per_px_xy: tuple[float, float] | None
    source_um_per_px_xy: tuple[float, float] | None
    support_shape_hw: tuple[int, int]
    preview_shape_hw: tuple[int, int]
    support_bbox_canvas_xywh: tuple[int, int, int, int] | None = None
    fixed_working_mode: str = "workspace_support_crop"
    target_um_per_px_xy: tuple[float, float] | None = None


def _support_bbox_from_labels(labels: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.where(labels > 0)
    h, w = labels.shape[:2]
    if ys.size == 0 or xs.size == 0:
        return 0, 0, int(w), int(h)
    x0 = int(xs.min())
    y0 = int(ys.min())
    x1 = int(xs.max()) + 1
    y1 = int(ys.max()) + 1
    return x0, y0, int(x1 - x0), int(y1 - y0)


def _canvas_bbox_to_level0_bbox(
    bbox_xywh: tuple[int, int, int, int],
    *,
    provenance: dict[str, Any],
    canvas_shape_hw: tuple[int, int],
) -> tuple[int, int, int, int]:
    x0, y0, w, h = [int(v) for v in bbox_xywh]
    canvas_h, canvas_w = [int(v) for v in canvas_shape_hw]
    mapping = dict(provenance.get("canvas_to_slide_level0") or {})
    origin = dict(mapping.get("origin_level0_xy") or {})
    scale = dict(mapping.get("scale_level0_per_canvas_px") or {})
    origin_x = float(origin.get("x") or 0.0)
    origin_y = float(origin.get("y") or 0.0)
    scale_x = float(scale.get("x") or 0.0)
    scale_y = float(scale.get("y") or 0.0)
    mirror_x = bool(mapping.get("mirror_x_applied", False))
    if scale_x <= 0.0 or scale_y <= 0.0:
        raise ValueError("physical_provenance is missing canvas_to_slide_level0 scale")
    if mirror_x:
        level0_x = int(round(origin_x + (canvas_w - (x0 + w)) * scale_x))
    else:
        level0_x = int(round(origin_x + x0 * scale_x))
    level0_y = int(round(origin_y + y0 * scale_y))
    level0_w = max(1, int(round(w * scale_x)))
    level0_h = max(1, int(round(h * scale_y)))
    return level0_x, level0_y, level0_w, level0_h


def _resample_to_target_um_per_px_rgb_and_labels(
    rgb: np.ndarray,
    labels: np.ndarray,
    *,
    current_um_per_px_xy: tuple[float, float] | None,
    target_um_per_px: float | None,
) -> tuple[np.ndarray, np.ndarray, tuple[float, float] | None]:
    if current_um_per_px_xy is None or target_um_per_px is None or target_um_per_px <= 0:
        return rgb, labels, current_um_per_px_xy
    cur_x, cur_y = float(current_um_per_px_xy[0]), float(current_um_per_px_xy[1])
    if min(cur_x, cur_y) <= 0:
        return rgb, labels, current_um_per_px_xy
    scale_x = cur_x / float(target_um_per_px)
    scale_y = cur_y / float(target_um_per_px)
    if abs(scale_x - 1.0) < 1e-3 and abs(scale_y - 1.0) < 1e-3:
        return rgb, labels, (float(target_um_per_px), float(target_um_per_px))
    h, w = rgb.shape[:2]
    new_w = max(1, int(round(w * scale_x)))
    new_h = max(1, int(round(h * scale_y)))
    rgb_resized = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
    labels_resized = cv2.resize(labels, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    return rgb_resized, labels_resized.astype(np.uint8), (float(target_um_per_px), float(target_um_per_px))


def _extract_slide_bbox_to_target_um_per_px(
    *,
    slide_path: Path,
    stain: str,
    bbox_level0_xywh: tuple[int, int, int, int],
    target_um_per_px: float,
    fallback_mpp_xy: tuple[float, float] | None = None,
) -> tuple[np.ndarray, tuple[float, float] | None]:
    slide_path = _runtime_local_path(slide_path)
    loaded = load_slide_bundle(slide_path, stain)
    handle = open_slide_handle(loaded)
    if handle is None or not hasattr(handle, "read_region"):
        raise RuntimeError("Slide backend does not support read_region for local high-res extraction")
    try:
        mpp_x = float(loaded.mpp_x) if loaded.mpp_x is not None else None
        mpp_y = float(loaded.mpp_y) if loaded.mpp_y is not None else None
        if (mpp_x is None or mpp_y is None) and fallback_mpp_xy is not None:
            mpp_x = float(fallback_mpp_xy[0])
            mpp_y = float(fallback_mpp_xy[1])
        if mpp_x is None or mpp_y is None:
            raise RuntimeError("Slide MPP is unavailable for high-res local extraction")
        x0, y0, w0, h0 = [int(v) for v in bbox_level0_xywh]
        target = float(target_um_per_px)
        level_effective = []
        for idx, ds in enumerate(loaded.level_downsamples):
            eff_x = float(ds) * mpp_x
            eff_y = float(ds) * mpp_y
            level_effective.append((idx, eff_x, eff_y, (eff_x + eff_y) / 2.0))
        best_level = min(level_effective, key=lambda rec: abs(rec[3] - target))[0]
        downsample = float(loaded.level_downsamples[best_level])
        out_w = max(1, int(round(w0 / downsample)))
        out_h = max(1, int(round(h0 / downsample)))
        rgb = np.asarray(handle.read_region((x0, y0), int(best_level), (out_w, out_h)).convert("RGB"))
        current_um = (float(downsample * mpp_x), float(downsample * mpp_y))
        rgb, _dummy_labels, target_um = _resample_to_target_um_per_px_rgb_and_labels(
            rgb,
            np.ones(rgb.shape[:2], dtype=np.uint8),
            current_um_per_px_xy=current_um,
            target_um_per_px=target,
        )
        return rgb, target_um
    finally:
        try:
            handle.close()
        except Exception:
            pass


def _extract_fixed_local_patch_from_source_slide(
    *,
    section_dir: Path,
    stain: str,
    fixed_preview_rgb: np.ndarray,
    fixed_preview_labels: np.ndarray,
    fixed_info: dict[str, Any] | None,
    roi_bbox_preview_yxyx: tuple[int, int, int, int],
    target_um_per_px: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    y0, y1, x0, x1 = [int(v) for v in roi_bbox_preview_yxyx]
    preview_h, preview_w = fixed_preview_rgb.shape[:2]
    if fixed_info is None:
        raise RuntimeError("Missing Step 7 fixed_info for local slide extraction")
    support_shape = tuple(int(v) for v in (fixed_info.get("support_shape_hw") or [0, 0]))
    support_bbox_canvas = tuple(int(v) for v in (fixed_info.get("support_bbox_canvas_xywh") or [0, 0, preview_w, preview_h]))
    preview_um = tuple(float(v) for v in (fixed_info.get("preview_um_per_px_xy") or [target_um_per_px, target_um_per_px]))
    source_um = tuple(float(v) for v in (fixed_info.get("source_um_per_px_xy") or preview_um))
    scale_x_preview_to_support = float(preview_um[0]) / float(source_um[0]) if source_um[0] > 0 else 1.0
    scale_y_preview_to_support = float(preview_um[1]) / float(source_um[1]) if source_um[1] > 0 else 1.0
    x0_support = int(round(x0 * scale_x_preview_to_support))
    y0_support = int(round(y0 * scale_y_preview_to_support))
    x1_support = int(round(x1 * scale_x_preview_to_support))
    y1_support = int(round(y1 * scale_y_preview_to_support))
    x0_support = max(0, min(int(support_shape[1]), x0_support))
    y0_support = max(0, min(int(support_shape[0]), y0_support))
    x1_support = max(x0_support + 1, min(int(support_shape[1]), x1_support))
    y1_support = max(y0_support + 1, min(int(support_shape[0]), y1_support))
    preview_labels_local = fixed_preview_labels[y0:y1, x0:x1].copy()
    try:
        _meta, provenance = recover_or_load_section_physical_provenance(section_dir, write_back_if_missing=True)
        canvas_bbox_xywh = (
            int(support_bbox_canvas[0] + x0_support),
            int(support_bbox_canvas[1] + y0_support),
            int(x1_support - x0_support),
            int(y1_support - y0_support),
        )
        export_canvas = dict(provenance.get("export_canvas") or {})
        canvas_shape_hw = (
            int(export_canvas.get("height_px") or preview_h),
            int(export_canvas.get("width_px") or preview_w),
        )
        bbox_level0 = _canvas_bbox_to_level0_bbox(canvas_bbox_xywh, provenance=provenance, canvas_shape_hw=canvas_shape_hw)
        slide_info = dict(provenance.get("source_slide") or {})
        slide_path = slide_info.get("path")
        if not slide_path:
            raise RuntimeError("physical_provenance is missing source_slide.path")
        local_rgb, local_um = _extract_slide_bbox_to_target_um_per_px(
            slide_path=Path(str(slide_path)),
            stain=stain,
            bbox_level0_xywh=bbox_level0,
            target_um_per_px=float(target_um_per_px),
            fallback_mpp_xy=(
                float(slide_info.get("mpp_x") or source_um[0]),
                float(slide_info.get("mpp_y") or source_um[1]),
            ),
        )
        local_labels = cv2.resize(preview_labels_local, (local_rgb.shape[1], local_rgb.shape[0]), interpolation=cv2.INTER_NEAREST).astype(np.uint8)
        local_rgb[local_labels <= 0] = 255
        return local_rgb, local_labels, {
            "mode": "source_slide_local_patch",
            "preview_roi_bbox_yxyx": [int(y0), int(y1), int(x0), int(x1)],
            "support_roi_bbox_xyxy": [int(x0_support), int(y0_support), int(x1_support), int(y1_support)],
            "canvas_roi_bbox_xywh": [int(v) for v in canvas_bbox_xywh],
            "working_um_per_px_xy": list(local_um) if local_um is not None else [float(target_um_per_px), float(target_um_per_px)],
        }
    except Exception as exc:
        fallback_rgb = fixed_preview_rgb[y0:y1, x0:x1].copy()
        fallback_labels = preview_labels_local
        return fallback_rgb, fallback_labels, {
            "mode": "preview_fallback",
            "preview_roi_bbox_yxyx": [int(y0), int(y1), int(x0), int(x1)],
            "reason": f"{type(exc).__name__}: {exc}",
            "working_um_per_px_xy": [float(preview_um[0]), float(preview_um[1])],
        }


def load_confocal_projection(
    paths: list[Path],
    *,
    mode: str = "focus",
    channel_index: int = 0,
    nominal_overlap_fraction: float = 0.10,
) -> ConfocalProjectionBundle:
    clean_paths = _sort_confocal_paths([Path(p) for p in paths])
    if not clean_paths:
        raise ValueError("No confocal z-stack sources were provided.")
    suffixes = {path.suffix.lower() for path in clean_paths}
    if len(clean_paths) > 1 and suffixes != {".tif"} and suffixes != {".tiff"} and suffixes != {".tif", ".tiff"}:
        raise ValueError("Multiple-source stitching currently supports TIFF/OME-TIFF tiles only.")

    channel_count = 1
    physical_um_per_px_xy: tuple[float, float] | None = None
    source_shapes: list[list[int]] = []
    if len(clean_paths) == 1:
        stack = _read_confocal_stack(clean_paths[0])
        channel_count = infer_stack_channel_count(stack)
        source_shapes = [list(stack.shape)]
        projection = project_confocal_stack(stack, mode=mode, channel_index=channel_index)
        if clean_paths[0].suffix.lower() == ".czi":
            source_mode = "czi_whole"
            physical_um_per_px_xy = _czi_physical_um_per_px(clean_paths[0])
        else:
            source_mode = "single_tiff"
            physical_um_per_px_xy = _ome_tiff_physical_um_per_px(clean_paths[0])
        stitch_info: dict[str, Any] = {"mode": source_mode}
    else:
        projected_tiles: list[np.ndarray] = []
        series_indices = [_stack_series_index(path) for path in clean_paths]
        sibling_czi = _sibling_czi_for_tiff_tiles(clean_paths)
        layout_px: dict[int, tuple[int, int]] | None = None
        layout_error: str | None = None
        if sibling_czi is not None and all(idx is not None for idx in series_indices):
            try:
                layout_px = _czi_mosaic_layout_px(sibling_czi)
                physical_um_per_px_xy = _czi_physical_um_per_px(sibling_czi)
            except Exception as exc:
                layout_px = None
                layout_error = f"{type(exc).__name__}: {exc}"
        for path in clean_paths:
            stack = _read_confocal_stack(path)
            channel_count = min(channel_count, infer_stack_channel_count(stack)) if projected_tiles else infer_stack_channel_count(stack)
            source_shapes.append(list(stack.shape))
            projected_tiles.append(project_confocal_stack(stack, mode=mode, channel_index=channel_index))
        if physical_um_per_px_xy is None:
            physical_um_per_px_xy = _ome_tiff_physical_um_per_px(clean_paths[0])
        if layout_px is not None:
            placements: list[tuple[int, int]] = []
            for path, idx in zip(clean_paths, series_indices):
                if idx is None or idx not in layout_px:
                    raise ValueError(f"Could not map TIFF tile to CZI mosaic index: {path.name}")
                placements.append(layout_px[int(idx)])
            min_x = min(x for x, _ in placements)
            min_y = min(y for _, y in placements)
            placements = [(x - min_x, y - min_y) for x, y in placements]
            tile_h = int(projected_tiles[0].shape[0]) if projected_tiles else 0
            tile_w = int(projected_tiles[0].shape[1]) if projected_tiles else 0
            unique_xs = sorted({int(x) for x, _ in placements})
            unique_ys = sorted({int(y) for _, y in placements})
            x_steps = [int(unique_xs[i + 1] - unique_xs[i]) for i in range(len(unique_xs) - 1)]
            y_steps = [int(unique_ys[i + 1] - unique_ys[i]) for i in range(len(unique_ys) - 1)]
            median_x_step = float(np.median(x_steps)) if x_steps else float(tile_w)
            median_y_step = float(np.median(y_steps)) if y_steps else float(tile_h)
            overlap_x = None if tile_w <= 0 else max(0.0, 1.0 - (median_x_step / float(tile_w)))
            overlap_y = None if tile_h <= 0 else max(0.0, 1.0 - (median_y_step / float(tile_h)))
            canvas_w = max(x + tile.shape[1] for tile, (x, _y) in zip(projected_tiles, placements))
            canvas_h = max(y + tile.shape[0] for tile, (_x, y) in zip(projected_tiles, placements))
            accum = np.zeros((canvas_h, canvas_w), dtype=np.float32)
            counts = np.zeros((canvas_h, canvas_w), dtype=np.float32)
            for tile, (x0, y0) in zip(projected_tiles, placements):
                h, w = tile.shape[:2]
                accum[y0 : y0 + h, x0 : x0 + w] += tile.astype(np.float32)
                counts[y0 : y0 + h, x0 : x0 + w] += 1.0
            projection = np.where(counts > 0, accum / np.maximum(counts, 1.0), 0.0).astype(np.uint8)
            stitch_info = {
                "mode": "multi_tiff_czi_layout",
                "tile_count": len(projected_tiles),
                "source_czi": str(sibling_czi) if sibling_czi is not None else None,
                "tile_series_indices": [int(idx) for idx in series_indices if idx is not None],
                "tile_positions_xy": [[int(x), int(y)] for x, y in placements],
                "grid_shape_rc": [int(len(unique_ys)), int(len(unique_xs))],
                "tile_shape_hw": [tile_h, tile_w],
                "grid_unique_xs_px": unique_xs,
                "grid_unique_ys_px": unique_ys,
                "grid_unique_x_steps_px": x_steps,
                "grid_unique_y_steps_px": y_steps,
                "grid_step_xy_px": [median_x_step, median_y_step],
                "inferred_overlap_fraction_xy": [overlap_x, overlap_y],
                "stitched_shape_hw": [int(projection.shape[0]), int(projection.shape[1])],
            }
            source_mode = "multi_tiff_grid"
        else:
            projection, stitch_info = _stitch_projected_tiles(projected_tiles, nominal_overlap_fraction=nominal_overlap_fraction)
            if sibling_czi is not None:
                stitch_info["source_czi"] = str(sibling_czi)
            if layout_error is not None:
                stitch_info["layout_recovery_error"] = layout_error
            source_mode = "multi_tiff_strip"
        return ConfocalProjectionBundle(
            projection_u8=projection,
            channel_count=max(1, int(channel_count)),
            source_mode=source_mode,
            source_paths=[str(path) for path in clean_paths],
            source_shapes=source_shapes,
            stitch_info=stitch_info,
            physical_um_per_px_xy=physical_um_per_px_xy,
        )

    return ConfocalProjectionBundle(
        projection_u8=projection,
        channel_count=max(1, int(channel_count)),
        source_mode=source_mode,
        source_paths=[str(path) for path in clean_paths],
        source_shapes=source_shapes,
        stitch_info=stitch_info,
        physical_um_per_px_xy=physical_um_per_px_xy,
    )


def prepare_myelin_confocal_fixed_bundle(
    item: WorkspaceSection,
    *,
    max_long_edge: int | None = None,
    target_um_per_px: float | None = STEP7_TARGET_UM_PER_PX,
) -> MyelinFixedBundle:
    _metadata, crop_rgb, tissue, artifact, _source = load_workspace_section(item)
    support_mask = ((tissue > 0) | (artifact > 0))
    support_labels = np.where(tissue > 0, 1, np.where(artifact > 0, 2, 0)).astype(np.uint8)
    x0, y0, support_w, support_h = _support_bbox_from_labels(np.where(support_mask, 1, 0).astype(np.uint8))
    if support_w <= 0 or support_h <= 0:
        rgb = crop_rgb.copy()
        labels = np.zeros(crop_rgb.shape[:2], dtype=np.uint8)
        support_bbox_canvas_xywh = (0, 0, int(crop_rgb.shape[1]), int(crop_rgb.shape[0]))
    else:
        x1 = x0 + support_w
        y1 = y0 + support_h
        rgb = crop_rgb[y0:y1, x0:x1].copy()
        labels = support_labels[y0:y1, x0:x1].copy()
        rgb[labels <= 0] = 255
        support_bbox_canvas_xywh = (int(x0), int(y0), int(support_w), int(support_h))
    support_h, support_w = int(labels.shape[0]), int(labels.shape[1])
    source_um_per_px_xy: tuple[float, float] | None = None
    preview_um_per_px_xy: tuple[float, float] | None = None
    fixed_working_mode = "workspace_support_crop"
    try:
        _meta, provenance = recover_or_load_section_physical_provenance(item.section_dir, write_back_if_missing=True)
        info = dict(provenance.get("canvas_to_slide_um_per_px") or {})
        src_x = float(info.get("x_um_per_px"))
        src_y = float(info.get("y_um_per_px"))
        source_um_per_px_xy = (src_x, src_y)
        preview_um_per_px_xy = (src_x, src_y)
    except Exception:
        pass
    rgb, labels, preview_um_per_px_xy = _resample_to_target_um_per_px_rgb_and_labels(
        rgb,
        labels,
        current_um_per_px_xy=preview_um_per_px_xy,
        target_um_per_px=target_um_per_px,
    )
    fixed_working_mode = "workspace_support_crop_resampled"
    if max_long_edge is not None:
        h, w = rgb.shape[:2]
        long_edge = max(h, w)
        if long_edge > int(max_long_edge) > 0:
            scale = float(max_long_edge) / float(long_edge)
            new_w = max(1, int(round(w * scale)))
            new_h = max(1, int(round(h * scale)))
            rgb = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
            labels = cv2.resize(labels, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            if preview_um_per_px_xy is not None:
                preview_um_per_px_xy = (
                    float(preview_um_per_px_xy[0] * (w / float(new_w))),
                    float(preview_um_per_px_xy[1] * (h / float(new_h))),
                )
            fixed_working_mode = f"{fixed_working_mode}_downsampled"
    return MyelinFixedBundle(
        rgb=rgb,
        labels=labels,
        preview_um_per_px_xy=preview_um_per_px_xy,
        source_um_per_px_xy=source_um_per_px_xy,
        support_shape_hw=(support_h, support_w),
        preview_shape_hw=(int(rgb.shape[0]), int(rgb.shape[1])),
        support_bbox_canvas_xywh=support_bbox_canvas_xywh,
        fixed_working_mode=fixed_working_mode,
        target_um_per_px_xy=preview_um_per_px_xy,
    )


def prepare_myelin_confocal_fixed(item: WorkspaceSection, *, max_long_edge: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    bundle = prepare_myelin_confocal_fixed_bundle(item, max_long_edge=max_long_edge)
    return bundle.rgb, bundle.labels


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
    moving_mask_u8: np.ndarray | None = None,
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
    return apply_affine_matrix(
        moving_gray_u8,
        fixed_shape_hw,
        mat=mat,
        moving_mask_u8=moving_mask_u8,
    )


def apply_affine_matrix(
    moving_gray_u8: np.ndarray,
    fixed_shape_hw: tuple[int, int],
    *,
    mat: np.ndarray,
    moving_mask_u8: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mat = np.asarray(mat, dtype=np.float32).reshape(2, 3)
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
    moving_mask = np.where(moving_gray_u8 > 0, 255, 0).astype(np.uint8) if moving_mask_u8 is None else np.where(np.asarray(moving_mask_u8) > 0, 255, 0).astype(np.uint8)
    warped_mask = cv2.warpAffine(
        moving_mask,
        mat,
        (out_w, out_h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return warped, warped_mask, mat


def _display_local_from_raw_xy(
    raw_xy: tuple[float, float],
    shape_hw: tuple[int, int],
    *,
    flip_lr: bool,
    flip_ud: bool,
) -> tuple[float, float]:
    h, w = int(shape_hw[0]), int(shape_hw[1])
    x = float(w - 1 - raw_xy[0]) if flip_lr else float(raw_xy[0])
    y = float(h - 1 - raw_xy[1]) if flip_ud else float(raw_xy[1])
    return float(np.clip(x, 0.0, max(0.0, w - 1.0))), float(np.clip(y, 0.0, max(0.0, h - 1.0)))


def _fit_similarity_affine_matrix(src_xy: np.ndarray, dst_xy: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    src = np.asarray(src_xy, dtype=np.float64)
    dst = np.asarray(dst_xy, dtype=np.float64)
    src_ctr = src.mean(axis=0)
    dst_ctr = dst.mean(axis=0)
    src_c = src - src_ctr
    dst_c = dst - dst_ctr
    cov = src_c.T @ dst_c
    u, s, vt = np.linalg.svd(cov)
    rot = vt.T @ u.T
    if np.linalg.det(rot) < 0:
        vt[-1, :] *= -1.0
        rot = vt.T @ u.T
    denom = float(np.sum(src_c**2))
    scale = 1.0 if denom <= 1e-8 else float(np.sum(s) / denom)
    linear = scale * rot
    translation = dst_ctr - (src_ctr @ linear.T)
    mat = np.concatenate([linear, translation[:, None]], axis=1).astype(np.float32)
    pred = (src @ linear.T) + translation
    residual = pred - dst
    residual_norm = np.linalg.norm(residual, axis=1)
    info = {
        "model": "similarity",
        "scale": float(scale),
        "rotation_deg": float(np.degrees(np.arctan2(rot[1, 0], rot[0, 0]))),
        "translation_xy": translation.astype(float).tolist(),
        "rms_px": float(np.sqrt(np.mean(residual_norm**2))) if residual_norm.size else float("nan"),
        "max_px": float(np.max(residual_norm)) if residual_norm.size else float("nan"),
    }
    return mat, info


def _fit_affine_matrix(src_xy: np.ndarray, dst_xy: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    src = np.asarray(src_xy, dtype=np.float64)
    dst = np.asarray(dst_xy, dtype=np.float64)
    src_aug = np.concatenate([src, np.ones((src.shape[0], 1), dtype=np.float64)], axis=1)
    coeff, _, _, _ = np.linalg.lstsq(src_aug, dst, rcond=None)
    pred = src_aug @ coeff
    residual = pred - dst
    residual_norm = np.linalg.norm(residual, axis=1)
    mat = coeff.T.astype(np.float32)
    linear = mat[:, :2].astype(np.float64)
    scale_x = float(np.linalg.norm(linear[:, 0]))
    scale_y = float(np.linalg.norm(linear[:, 1]))
    shear = float(np.dot(linear[:, 0] / max(scale_x, 1e-8), linear[:, 1])) if scale_x > 1e-8 else 0.0
    info = {
        "model": "affine",
        "scale_x_like": scale_x,
        "scale_y_like": scale_y,
        "shear_like": shear,
        "rms_px": float(np.sqrt(np.mean(residual_norm**2))) if residual_norm.size else float("nan"),
        "max_px": float(np.max(residual_norm)) if residual_norm.size else float("nan"),
    }
    return mat, info


def _load_ants_affine_mat(path: Path) -> tuple[np.ndarray, np.ndarray]:
    from scipy.io import loadmat

    obj = loadmat(path)
    params = np.asarray(obj.get("AffineTransform_double_2_2"), dtype=np.float64).reshape(-1)
    fixed = np.asarray(obj.get("fixed"), dtype=np.float64).reshape(-1)
    if params.size != 6 or fixed.size != 2:
        raise ValueError(f"Unexpected ANTs affine contents in {path}")
    return params, fixed


def _write_ants_affine_mat(path: Path, params: np.ndarray, fixed: np.ndarray) -> None:
    from scipy.io import savemat

    payload = {
        "AffineTransform_double_2_2": np.asarray(params, dtype=np.float64).reshape(6, 1),
        "fixed": np.asarray(fixed, dtype=np.float64).reshape(2, 1),
    }
    savemat(path, payload, format="4")


def _apply_ants_affine_to_point(params: np.ndarray, fixed: np.ndarray, xy: tuple[float, float]) -> np.ndarray:
    p = np.asarray(params, dtype=np.float64).reshape(-1)
    c = np.asarray(fixed, dtype=np.float64).reshape(-1)
    linear = np.array([[p[0], p[1]], [p[2], p[3]]], dtype=np.float64)
    trans = np.array([p[4], p[5]], dtype=np.float64)
    x = np.asarray([float(xy[0]), float(xy[1])], dtype=np.float64)
    return linear @ (x - c) + c + trans


def _preserve_single_anchor_in_local_transform(
    mat_path: Path,
    *,
    anchor_scene_xy: tuple[float, float],
    local_roi_bbox_yxyx: tuple[int, int, int, int],
    out_path: Path,
) -> dict[str, Any]:
    params, fixed = _load_ants_affine_mat(mat_path)
    y0, y1, x0, x1 = [int(v) for v in local_roi_bbox_yxyx]
    anchor_local = np.array([float(anchor_scene_xy[0]) - float(x0), float(anchor_scene_xy[1]) - float(y0)], dtype=np.float64)
    pred = _apply_ants_affine_to_point(params, fixed, (float(anchor_local[0]), float(anchor_local[1])))
    delta = anchor_local - pred
    adjusted = params.copy()
    adjusted[4] += float(delta[0])
    adjusted[5] += float(delta[1])
    _write_ants_affine_mat(out_path, adjusted, fixed)
    return {
        "used": True,
        "mode": "single_anchor_translation_lock",
        "anchor_scene_xy": [float(anchor_scene_xy[0]), float(anchor_scene_xy[1])],
        "anchor_local_xy": [float(anchor_local[0]), float(anchor_local[1])],
        "pre_adjust_pred_local_xy": [float(pred[0]), float(pred[1])],
        "post_adjust_delta_xy": [float(delta[0]), float(delta[1])],
        "transform_path": str(out_path),
    }


def _anchor_guided_manual_affine(
    *,
    moving_shape_hw: tuple[int, int],
    fixed_shape_hw: tuple[int, int],
    current_mat: np.ndarray,
    anchor_pairs: list[dict[str, Any]],
    flip_lr: bool,
    flip_ud: bool,
) -> tuple[np.ndarray, dict[str, Any]]:
    valid_pairs: list[dict[str, Any]] = []
    src_pts: list[list[float]] = []
    dst_pts: list[list[float]] = []
    for pair in anchor_pairs:
        if not isinstance(pair, dict):
            continue
        raw_xy = pair.get("confocal_raw_xy")
        scene_xy = pair.get("section_scene_xy")
        if not isinstance(raw_xy, (list, tuple)) or len(raw_xy) != 2:
            continue
        if not isinstance(scene_xy, (list, tuple)) or len(scene_xy) != 2:
            continue
        # IMPORTANT: current_mat is applied directly to the raw moving raster in cv2.warpAffine,
        # so anchor fitting must use raw moving-image coordinates as the source space.
        # The display-space point is only for user-facing bookkeeping in the preview.
        raw_x = float(raw_xy[0])
        raw_y = float(raw_xy[1])
        disp_x, disp_y = _display_local_from_raw_xy((raw_x, raw_y), moving_shape_hw, flip_lr=flip_lr, flip_ud=flip_ud)
        src_pts.append([raw_x, raw_y])
        dst_pts.append([float(scene_xy[0]), float(scene_xy[1])])
        valid_pairs.append(
            {
                "index": int(pair.get("index", len(valid_pairs) + 1)),
                "confocal_raw_xy": [raw_x, raw_y],
                "confocal_display_xy": [disp_x, disp_y],
                "section_scene_xy": [float(scene_xy[0]), float(scene_xy[1])],
            }
        )
    if not valid_pairs:
        return current_mat.astype(np.float32), {"used": False, "pair_count": 0}
    src = np.asarray(src_pts, dtype=np.float64)
    dst = np.asarray(dst_pts, dtype=np.float64)
    if len(valid_pairs) == 1:
        src_aug = np.array([src[0, 0], src[0, 1], 1.0], dtype=np.float64)
        pred = np.asarray(current_mat, dtype=np.float64) @ src_aug
        delta = dst[0] - pred
        mat = np.asarray(current_mat, dtype=np.float32).copy()
        mat[0, 2] += float(delta[0])
        mat[1, 2] += float(delta[1])
        info = {
            "used": True,
            "model": "single_anchor_translation",
            "pair_count": 1,
            "translation_delta_xy": [float(delta[0]), float(delta[1])],
            "pairs": valid_pairs,
        }
        return mat, info
    if len(valid_pairs) >= 3:
        mat, fit_info = _fit_affine_matrix(src, dst)
    else:
        mat, fit_info = _fit_similarity_affine_matrix(src, dst)
    info = {
        "used": True,
        "pair_count": len(valid_pairs),
        "pairs": valid_pairs,
        **fit_info,
    }
    return mat.astype(np.float32), info


def _stage_command_local_refine(
    ants_bin: Path,
    fixed_path: Path,
    moving_path: Path,
    fixed_mask_path: Path,
    moving_mask_path: Path,
    prefix: Path,
    *,
    refine_model: str,
) -> list[str]:
    model_key = str(refine_model).strip().lower()
    transform = {
        "rigid": "Rigid[0.1]",
        "similarity": "Similarity[0.1]",
        "affine": "Affine[0.1]",
    }.get(model_key, "Similarity[0.1]")
    return [
        str(ants_binary_path(ants_bin, "antsRegistration")),
        "-d",
        "2",
        "-o",
        ants_cli_path(prefix),
        "-r",
        f"[{ants_cli_path(fixed_path)},{ants_cli_path(moving_path)},1]",
        "-m",
        f"CC[{ants_cli_path(fixed_path)},{ants_cli_path(moving_path)},1,4]",
        "-t",
        transform,
        "-c",
        "[300x150x80,1e-6,10]",
        "-s",
        "3x2x1vox",
        "-f",
        "8x4x2",
        "-x",
        f"[{ants_cli_path(fixed_mask_path)},{ants_cli_path(moving_mask_path)}]",
    ]


def _bbox_from_mask(mask: np.ndarray, *, margin_px: int = 96) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask > 0)
    h, w = mask.shape[:2]
    if ys.size == 0 or xs.size == 0:
        return 0, h, 0, w
    y0 = max(0, int(ys.min()) - margin_px)
    y1 = min(h, int(ys.max()) + 1 + margin_px)
    x0 = max(0, int(xs.min()) - margin_px)
    x1 = min(w, int(xs.max()) + 1 + margin_px)
    return y0, y1, x0, x1


def _paste_crop(full_shape_hw: tuple[int, int], crop: np.ndarray, bbox: tuple[int, int, int, int], *, fill_value: float = 0.0) -> np.ndarray:
    y0, y1, x0, x1 = bbox
    out = np.full(full_shape_hw, fill_value, dtype=np.float32)
    out[y0:y1, x0:x1] = crop.astype(np.float32)
    return out


def _local_zoom_overlay_panel(
    fixed_gray: np.ndarray,
    moving_gray: np.ndarray,
    fixed_mask: np.ndarray,
    moving_mask: np.ndarray,
    bbox: tuple[int, int, int, int],
) -> np.ndarray:
    y0, y1, x0, x1 = bbox
    overlay = overlay_preview(
        fixed_gray[y0:y1, x0:x1],
        moving_gray[y0:y1, x0:x1],
        fixed_mask[y0:y1, x0:x1],
        moving_mask[y0:y1, x0:x1],
    )
    return cv2.resize(overlay, (fixed_gray.shape[1], fixed_gray.shape[0]), interpolation=cv2.INTER_NEAREST)


def _bbox_from_union_masks(
    masks: list[np.ndarray],
    *,
    fallback_shape_hw: tuple[int, int],
    margin_px: int = 24,
) -> tuple[int, int, int, int]:
    union = np.zeros(fallback_shape_hw, dtype=np.uint8)
    for mask in masks:
        if mask is None:
            continue
        arr = np.asarray(mask)
        if arr.shape[:2] != fallback_shape_hw:
            continue
        union = np.maximum(union, (arr > 0).astype(np.uint8))
    if int(union.max()) <= 0:
        return 0, int(fallback_shape_hw[0]), 0, int(fallback_shape_hw[1])
    return _bbox_from_mask(union, margin_px=margin_px)


def _normalize_gray_for_panel(gray: np.ndarray, support_mask: np.ndarray | None = None) -> np.ndarray:
    arr = np.asarray(gray, dtype=np.float32)
    if arr.size == 0:
        return np.zeros_like(arr, dtype=np.float32)
    valid = np.isfinite(arr)
    if support_mask is not None and np.asarray(support_mask).shape[:2] == arr.shape[:2]:
        valid &= np.asarray(support_mask) > 0
    if not np.any(valid):
        valid = np.isfinite(arr)
    values = arr[valid]
    if values.size == 0:
        return np.zeros_like(arr, dtype=np.float32)
    lo = float(np.percentile(values, 1.0))
    hi = float(np.percentile(values, 99.0))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.min(values))
        hi = float(np.max(values))
    if hi <= lo:
        return np.zeros_like(arr, dtype=np.float32)
    norm = (arr - lo) / float(hi - lo)
    return np.clip(norm, 0.0, 1.0)


def _crop_gray_panel(
    gray: np.ndarray,
    bbox: tuple[int, int, int, int],
    *,
    support_mask: np.ndarray | None = None,
    fill_outside_support: float | None = None,
    out_shape_hw: tuple[int, int] | None = None,
) -> np.ndarray:
    y0, y1, x0, x1 = bbox
    crop = gray[y0:y1, x0:x1]
    crop_mask = support_mask[y0:y1, x0:x1] if support_mask is not None and support_mask.shape[:2] == gray.shape[:2] else None
    crop_norm = _normalize_gray_for_panel(crop, crop_mask)
    if crop_mask is not None and fill_outside_support is not None:
        crop_norm = np.where(crop_mask > 0, crop_norm, float(fill_outside_support)).astype(np.float32)
    panel = gray_preview_panel(crop_norm)
    if out_shape_hw is not None and panel.shape[:2] != out_shape_hw:
        panel = cv2.resize(panel, (int(out_shape_hw[1]), int(out_shape_hw[0])), interpolation=cv2.INTER_LINEAR)
    return panel


def _crop_overlay_panel(
    fixed_gray: np.ndarray,
    moving_gray: np.ndarray,
    fixed_mask: np.ndarray,
    moving_mask: np.ndarray,
    bbox: tuple[int, int, int, int],
    *,
    out_shape_hw: tuple[int, int] | None = None,
) -> np.ndarray:
    y0, y1, x0, x1 = bbox
    panel = overlay_preview(
        _normalize_gray_for_panel(fixed_gray[y0:y1, x0:x1]),
        _normalize_gray_for_panel(moving_gray[y0:y1, x0:x1]),
        fixed_mask[y0:y1, x0:x1],
        moving_mask[y0:y1, x0:x1],
    )
    if out_shape_hw is not None and panel.shape[:2] != out_shape_hw:
        panel = cv2.resize(panel, (int(out_shape_hw[1]), int(out_shape_hw[0])), interpolation=cv2.INTER_NEAREST)
    return panel


def _pattern_edge_panel(
    fixed_gray: np.ndarray,
    moving_gray: np.ndarray,
    bbox: tuple[int, int, int, int],
    *,
    out_shape_hw: tuple[int, int] | None = None,
) -> np.ndarray:
    y0, y1, x0, x1 = bbox
    fixed_crop = (_normalize_gray_for_panel(fixed_gray[y0:y1, x0:x1]) * 255.0).astype(np.uint8)
    moving_crop = (_normalize_gray_for_panel(moving_gray[y0:y1, x0:x1]) * 255.0).astype(np.uint8)
    fixed_edges = cv2.Canny(fixed_crop, 40, 120)
    moving_edges = cv2.Canny(moving_crop, 40, 120)
    panel = np.zeros((*fixed_edges.shape, 3), dtype=np.uint8)
    panel[..., 1] = fixed_edges
    panel[..., 0] = moving_edges
    panel[..., 2] = moving_edges
    if out_shape_hw is not None and panel.shape[:2] != out_shape_hw:
        panel = cv2.resize(panel, (int(out_shape_hw[1]), int(out_shape_hw[0])), interpolation=cv2.INTER_NEAREST)
    return panel


def _fiber_qc_row(
    *,
    label: str,
    note: str,
    fixed_gray: np.ndarray,
    moving_gray: np.ndarray,
    fixed_mask: np.ndarray,
    moving_mask: np.ndarray,
    zoom_bbox: tuple[int, int, int, int],
    out_shape_hw: tuple[int, int],
) -> dict[str, Any]:
    return {
        "label": label,
        "note": note,
        "moving": _crop_gray_panel(
            moving_gray,
            zoom_bbox,
            support_mask=moving_mask,
            fill_outside_support=1.0,
            out_shape_hw=out_shape_hw,
        ),
        "fixed": _crop_gray_panel(fixed_gray, zoom_bbox, support_mask=fixed_mask, out_shape_hw=out_shape_hw),
        "overlay": _crop_overlay_panel(fixed_gray, moving_gray, fixed_mask, moving_mask, zoom_bbox, out_shape_hw=out_shape_hw),
        "heatmap": _pattern_edge_panel(fixed_gray, moving_gray, zoom_bbox, out_shape_hw=out_shape_hw),
        "col_titles": ("Confocal local", "Myelin local", "Overlay", "Pattern edges"),
    }


def _read_gray_png_float(path: Path) -> np.ndarray:
    arr = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if arr is None:
        raise FileNotFoundError(f"Could not read grayscale PNG: {path}")
    return arr.astype(np.float32) / 255.0


def _read_mask_png_float(path: Path) -> np.ndarray:
    arr = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if arr is None:
        raise FileNotFoundError(f"Could not read mask PNG: {path}")
    return (arr > 127).astype(np.float32)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


@dataclass
class ConfocalRigidConfig:
    myelin_label: str
    myelin_section_dir: Path
    myelin_stain: str
    myelin_rgb: np.ndarray
    myelin_labels: np.ndarray
    myelin_fixed_info: dict[str, Any] | None
    confocal_projection_u8: np.ndarray
    confocal_signal_mask_u8: np.ndarray | None
    ants_bin: Path
    out_root: Path
    confocal_sources: list[Path]
    confocal_source_mode: str
    nominal_overlap_fraction: float
    projection_info: dict[str, Any] | None
    projection_mode: str
    channel_index: int
    local_refine_model: str = "similarity"
    registration_input_profile: str = STEP7_REGISTRATION_INPUT_PROFILE
    target_working_um_per_px: float = STEP7_TARGET_UM_PER_PX
    invert_confocal_for_registration: bool = True
    tx_px: float = 0.0
    ty_px: float = 0.0
    angle_deg: float = 0.0
    scale: float = 1.0
    flip_lr: bool = False
    flip_ud: bool = False
    anchor_pairs: list[dict[str, Any]] | None = None


@dataclass
class ConfocalSeedScreenConfig:
    myelin_label: str
    myelin_section_dir: Path
    myelin_rgb: np.ndarray
    myelin_labels: np.ndarray
    myelin_fixed_info: dict[str, Any] | None
    confocal_projection_u8: np.ndarray
    confocal_signal_mask_u8: np.ndarray | None
    out_root: Path
    confocal_sources: list[Path]
    confocal_source_mode: str
    nominal_overlap_fraction: float
    projection_info: dict[str, Any] | None
    projection_mode: str
    channel_index: int
    registration_input_profile: str = STEP7_REGISTRATION_INPUT_PROFILE
    target_working_um_per_px: float = STEP7_TARGET_UM_PER_PX
    invert_confocal_for_registration: bool = True
    tx_px: float = 0.0
    ty_px: float = 0.0
    angle_deg: float = 0.0
    scale: float = 1.0
    flip_lr: bool = False
    flip_ud: bool = False
    anchor_pairs: list[dict[str, Any]] | None = None
    search_radius_px: int = 32
    top_k_storyboard: int = 6


@dataclass
class ConfocalAutoScaleConfig:
    myelin_label: str
    myelin_section_dir: Path
    myelin_rgb: np.ndarray
    myelin_labels: np.ndarray
    myelin_fixed_info: dict[str, Any] | None
    confocal_projection_u8: np.ndarray
    confocal_signal_mask_u8: np.ndarray | None
    out_root: Path
    confocal_sources: list[Path]
    confocal_source_mode: str
    nominal_overlap_fraction: float
    projection_info: dict[str, Any] | None
    projection_mode: str
    channel_index: int
    registration_input_profile: str = STEP7_REGISTRATION_INPUT_PROFILE
    target_working_um_per_px: float = STEP7_TARGET_UM_PER_PX
    invert_confocal_for_registration: bool = True
    tx_px: float = 0.0
    ty_px: float = 0.0
    angle_deg: float = 0.0
    scale: float = 1.0
    flip_lr: bool = False
    flip_ud: bool = False
    anchor_pairs: list[dict[str, Any]] | None = None
    sweep_half_range: float = 0.020
    sweep_step: float = 0.002
    search_radius_px: int = 24
    local_refine_radius_px: int = 2
    sample_tile_limit: int = 3
    sample_strategy: str = "rowwise_uniform"
    coarse_scale_count: int = STEP7_AUTO_SCALE_COARSE_COUNT
    coarse_top_k: int = STEP7_AUTO_SCALE_TOPK
    prune_min_tiles: int = STEP7_AUTO_SCALE_PRUNE_MIN_TILES


@dataclass
class ConfocalFrontierConfig:
    myelin_label: str
    myelin_section_dir: Path
    myelin_rgb: np.ndarray
    myelin_labels: np.ndarray
    myelin_fixed_info: dict[str, Any] | None
    confocal_projection_u8: np.ndarray
    confocal_signal_mask_u8: np.ndarray | None
    out_root: Path
    confocal_sources: list[Path]
    confocal_source_mode: str
    nominal_overlap_fraction: float
    projection_info: dict[str, Any] | None
    projection_mode: str
    channel_index: int
    registration_input_profile: str = STEP7_REGISTRATION_INPUT_PROFILE
    target_working_um_per_px: float = STEP7_TARGET_UM_PER_PX
    invert_confocal_for_registration: bool = True
    tx_px: float = 0.0
    ty_px: float = 0.0
    angle_deg: float = 0.0
    scale: float = 1.0
    flip_lr: bool = False
    flip_ud: bool = False
    anchor_pairs: list[dict[str, Any]] | None = None
    selected_tile_index: int | None = None
    accepted_tile_indices: list[int] | None = None
    frozen_tile_indices: list[int] | None = None
    prior_rows: list[dict[str, Any]] | None = None
    search_radius_px: int = 20
    max_frontier_tiles: int = 6
    top_k_storyboard: int = 4


class TileState(str, Enum):
    UNSEEN = "unseen"
    FRONTIER = "frontier"
    ACCEPTED = "accepted"
    FROZEN = "frozen"
    HOLD = "hold"


@dataclass
class TileResultRow:
    tile_index: int
    label: str
    row_display: int
    col_display: int
    tile_state: str = TileState.UNSEEN.value
    pred_dx_px: float = 0.0
    pred_dy_px: float = 0.0
    meas_dx_px: float = 0.0
    meas_dy_px: float = 0.0
    final_dx_px: float = 0.0
    final_dy_px: float = 0.0
    current_cc: float = float("nan")
    current_mi: float = float("nan")
    meas_cc: float = float("nan")
    final_cc: float = float("nan")
    shift_gain_cc: float = float("nan")
    candidate_shift_dx_px: int = 0
    candidate_shift_dy_px: int = 0
    candidate_shifted_cc: float = float("nan")
    template_match_score: float = float("nan")
    proposal_gate: str = "kept_current"
    neighbor_count: int = 0
    neighbor_labels: list[str] = field(default_factory=list)
    neighbor_shift_spread_px: float = 0.0
    neighbor_agreement_score: float = float("nan")
    prior_deviation_px: float = float("nan")
    frontier_confidence: float = float("nan")
    graph_residual: float = float("nan")
    signal_coverage: float = float("nan")
    moving_edge_density: float = float("nan")
    fixed_edge_density: float = float("nan")
    edge_density_mean: float = float("nan")
    density_regime: str = ""
    profile_candidates: list[str] = field(default_factory=list)
    coarse_search_profiles: list[str] = field(default_factory=list)
    refine_profiles: list[str] = field(default_factory=list)
    refine_objective: str = "cc"
    registration_profile: str = ""
    eval_bbox_yxyx: list[int] = field(default_factory=list)
    tight_bbox_local_yxyx: list[int] = field(default_factory=list)
    metric_timing_total: float = 0.0
    moving: np.ndarray | None = None
    fixed: np.ndarray | None = None
    overlay: np.ndarray | None = None
    heatmap: np.ndarray | None = None
    col_titles: tuple[str, str, str, str] = (
        "Raw overlay current",
        "Raw overlay shifted",
        "Processed overlay current",
        "Processed overlay shifted",
    )

    def to_dict(self) -> dict[str, Any]:
        pred_dx = int(round(float(self.pred_dx_px)))
        pred_dy = int(round(float(self.pred_dy_px)))
        meas_dx = int(round(float(self.meas_dx_px)))
        meas_dy = int(round(float(self.meas_dy_px)))
        final_dx = int(round(float(self.final_dx_px)))
        final_dy = int(round(float(self.final_dy_px)))
        delta_pred_dx = final_dx - pred_dx
        delta_pred_dy = final_dy - pred_dy
        out = {
            "tile_index": int(self.tile_index),
            "label": str(self.label),
            "row_display": int(self.row_display),
            "col_display": int(self.col_display),
            "tile_state": str(self.tile_state),
            "pred_dx_px": pred_dx,
            "pred_dy_px": pred_dy,
            "meas_dx_px": meas_dx,
            "meas_dy_px": meas_dy,
            "final_dx_px": final_dx,
            "final_dy_px": final_dy,
            "current_cc": float(self.current_cc),
            "current_mi": float(self.current_mi),
            "meas_cc": float(self.meas_cc),
            "final_cc": float(self.final_cc),
            "shift_gain_cc": float(self.shift_gain_cc),
            "candidate_shift_dx_px": int(self.candidate_shift_dx_px),
            "candidate_shift_dy_px": int(self.candidate_shift_dy_px),
            "candidate_shifted_cc": float(self.candidate_shifted_cc),
            "template_match_score": float(self.template_match_score),
            "proposal_gate": str(self.proposal_gate),
            "neighbor_count": int(self.neighbor_count),
            "neighbor_labels": [str(v) for v in self.neighbor_labels],
            "neighbor_shift_spread_px": float(self.neighbor_shift_spread_px),
            "neighbor_agreement_score": float(self.neighbor_agreement_score),
            "prior_deviation_px": float(self.prior_deviation_px),
            "frontier_confidence": float(self.frontier_confidence),
            "graph_residual": float(self.graph_residual),
            "signal_coverage": float(self.signal_coverage),
            "moving_edge_density": float(self.moving_edge_density),
            "fixed_edge_density": float(self.fixed_edge_density),
            "edge_density_mean": float(self.edge_density_mean),
            "density_regime": str(self.density_regime),
            "profile_candidates": [str(v) for v in self.profile_candidates],
            "profile_candidates_str": "|".join(str(v) for v in self.profile_candidates),
            "coarse_search_profiles": [str(v) for v in self.coarse_search_profiles],
            "coarse_search_profiles_str": "|".join(str(v) for v in self.coarse_search_profiles),
            "refine_profiles": [str(v) for v in self.refine_profiles],
            "refine_profiles_str": "|".join(str(v) for v in self.refine_profiles),
            "refine_objective": str(self.refine_objective),
            "registration_profile": str(self.registration_profile),
            "eval_bbox_yxyx": [int(v) for v in self.eval_bbox_yxyx],
            "tight_bbox_local_yxyx": [int(v) for v in self.tight_bbox_local_yxyx],
            "metric_timing_total": float(self.metric_timing_total),
            "moving": self.moving,
            "fixed": self.fixed,
            "overlay": self.overlay,
            "heatmap": self.heatmap,
            "col_titles": tuple(str(v) for v in self.col_titles),
            # Legacy keys kept so the existing GUI and CSV wiring still works.
            "best_shift_dx_px": final_dx,
            "best_shift_dy_px": final_dy,
            "best_shift_cc": float(self.final_cc),
            "shifted_cc": float(self.final_cc),
            "prior_shift_dx_px": pred_dx,
            "prior_shift_dy_px": pred_dy,
            "delta_from_prior_dx_px": int(delta_pred_dx),
            "delta_from_prior_dy_px": int(delta_pred_dy),
            "delta_from_prior_norm_px": float(np.hypot(float(delta_pred_dx), float(delta_pred_dy))),
        }
        return out

    @classmethod
    def from_mapping(cls, row: dict[str, Any]) -> "TileResultRow":
        final_dx = float(row.get("final_dx_px", row.get("best_shift_dx_px", 0.0)))
        final_dy = float(row.get("final_dy_px", row.get("best_shift_dy_px", 0.0)))
        pred_dx = float(row.get("pred_dx_px", row.get("prior_shift_dx_px", 0.0)))
        pred_dy = float(row.get("pred_dy_px", row.get("prior_shift_dy_px", 0.0)))
        meas_dx = float(row.get("meas_dx_px", row.get("candidate_shift_dx_px", final_dx)))
        meas_dy = float(row.get("meas_dy_px", row.get("candidate_shift_dy_px", final_dy)))
        return cls(
            tile_index=int(row.get("tile_index", -1)),
            label=str(row.get("label") or f"T{int(row.get('tile_index', -1)):02d}"),
            row_display=int(row.get("row_display", 0)),
            col_display=int(row.get("col_display", 0)),
            tile_state=str(row.get("tile_state") or TileState.UNSEEN.value),
            pred_dx_px=pred_dx,
            pred_dy_px=pred_dy,
            meas_dx_px=meas_dx,
            meas_dy_px=meas_dy,
            final_dx_px=final_dx,
            final_dy_px=final_dy,
            current_cc=float(row.get("current_cc", float("nan"))),
            current_mi=float(row.get("current_mi", float("nan"))),
            meas_cc=float(row.get("meas_cc", row.get("candidate_shifted_cc", row.get("shifted_cc", float("nan"))))),
            final_cc=float(row.get("final_cc", row.get("shifted_cc", float("nan")))),
            shift_gain_cc=float(row.get("shift_gain_cc", float("nan"))),
            candidate_shift_dx_px=int(row.get("candidate_shift_dx_px", round(meas_dx))),
            candidate_shift_dy_px=int(row.get("candidate_shift_dy_px", round(meas_dy))),
            candidate_shifted_cc=float(row.get("candidate_shifted_cc", float("nan"))),
            template_match_score=float(row.get("template_match_score", float("nan"))),
            proposal_gate=str(row.get("proposal_gate") or "kept_current"),
            neighbor_count=int(row.get("neighbor_count", 0)),
            neighbor_labels=[str(v) for v in list(row.get("neighbor_labels", []) or [])],
            neighbor_shift_spread_px=float(row.get("neighbor_shift_spread_px", 0.0)),
            neighbor_agreement_score=float(row.get("neighbor_agreement_score", float("nan"))),
            prior_deviation_px=float(row.get("prior_deviation_px", row.get("delta_from_prior_norm_px", float("nan")))),
            frontier_confidence=float(row.get("frontier_confidence", row.get("seed_score", float("nan")))),
            graph_residual=float(row.get("graph_residual", float("nan"))),
            signal_coverage=float(row.get("signal_coverage", float("nan"))),
            moving_edge_density=float(row.get("moving_edge_density", float("nan"))),
            fixed_edge_density=float(row.get("fixed_edge_density", float("nan"))),
            edge_density_mean=float(row.get("edge_density_mean", float("nan"))),
            density_regime=str(row.get("density_regime") or ""),
            profile_candidates=[str(v) for v in list(row.get("profile_candidates", []) or [])],
            coarse_search_profiles=[str(v) for v in list(row.get("coarse_search_profiles", []) or [])],
            refine_profiles=[str(v) for v in list(row.get("refine_profiles", []) or [])],
            refine_objective=str(row.get("refine_objective") or "cc"),
            registration_profile=str(row.get("registration_profile") or STEP7_REGISTRATION_INPUT_PROFILE),
            eval_bbox_yxyx=[int(v) for v in list(row.get("eval_bbox_yxyx", []) or [])],
            tight_bbox_local_yxyx=[int(v) for v in list(row.get("tight_bbox_local_yxyx", []) or [])],
            metric_timing_total=float(row.get("metric_timing_total", 0.0)),
            moving=row.get("moving"),
            fixed=row.get("fixed"),
            overlay=row.get("overlay"),
            heatmap=row.get("heatmap"),
            col_titles=tuple(str(v) for v in list(row.get("col_titles", []))[:4]) if row.get("col_titles") else cls.col_titles,
        )


@dataclass
class GraphEdge:
    edge_type: str
    src_tile_index: int | None = None
    dst_tile_index: int | None = None
    src_column_index: int | None = None
    dst_column_index: int | None = None
    target_dx_px: float = 0.0
    target_dy_px: float = 0.0
    weight: float = 1.0
    robust_delta: float = 6.0
    meta: dict[str, Any] = field(default_factory=dict)


def build_confocal_tile_defs(
    stitch_info: dict[str, Any] | None,
    *,
    raw_shape_hw: tuple[int, int],
    scaled_shape_hw: tuple[int, int],
    flip_lr: bool = False,
    flip_ud: bool = False,
) -> list[dict[str, Any]]:
    info = dict(stitch_info or {})
    positions = info.get("tile_positions_xy") or []
    tile_shape = info.get("tile_shape_hw") or [0, 0]
    if not positions or len(tile_shape) != 2:
        return []
    raw_h, raw_w = int(raw_shape_hw[0]), int(raw_shape_hw[1])
    scaled_h, scaled_w = int(scaled_shape_hw[0]), int(scaled_shape_hw[1])
    tile_h_raw, tile_w_raw = int(tile_shape[0]), int(tile_shape[1])
    if min(raw_h, raw_w, scaled_h, scaled_w, tile_h_raw, tile_w_raw) <= 0:
        return []

    unique_xs = sorted({int(p[0]) for p in positions if isinstance(p, (list, tuple)) and len(p) == 2})
    unique_ys = sorted({int(p[1]) for p in positions if isinstance(p, (list, tuple)) and len(p) == 2})
    x_to_col = {
        x: (len(unique_xs) - 1 - idx if flip_lr else idx)
        for idx, x in enumerate(unique_xs)
    }
    y_to_row = {
        y: (len(unique_ys) - 1 - idx if flip_ud else idx)
        for idx, y in enumerate(unique_ys)
    }
    sx = float(scaled_w) / float(raw_w)
    sy = float(scaled_h) / float(raw_h)

    defs: list[dict[str, Any]] = []
    for tile_index, pos in enumerate(positions):
        if not isinstance(pos, (list, tuple)) or len(pos) != 2:
            continue
        raw_x = int(pos[0])
        raw_y = int(pos[1])
        x0 = max(0, min(scaled_w - 1, int(round(raw_x * sx))))
        y0 = max(0, min(scaled_h - 1, int(round(raw_y * sy))))
        x1 = max(x0 + 1, min(scaled_w, int(round((raw_x + tile_w_raw) * sx))))
        y1 = max(y0 + 1, min(scaled_h, int(round((raw_y + tile_h_raw) * sy))))
        disp_x0 = scaled_w - x1 if flip_lr else x0
        disp_x1 = scaled_w - x0 if flip_lr else x1
        disp_y0 = scaled_h - y1 if flip_ud else y0
        disp_y1 = scaled_h - y0 if flip_ud else y1
        row_display = int(y_to_row.get(raw_y, 0))
        col_display = int(x_to_col.get(raw_x, 0))
        defs.append(
            {
                "tile_index": int(tile_index),
                "row_display": row_display,
                "col_display": col_display,
                "label": f"T{int(tile_index):02d} r{row_display}c{col_display}",
                "raw_bbox_xyxy": [int(raw_x), int(raw_y), int(raw_x + tile_w_raw), int(raw_y + tile_h_raw)],
                "scaled_bbox_xyxy": [int(x0), int(y0), int(x1), int(y1)],
                "display_bbox_xyxy": [int(disp_x0), int(disp_y0), int(disp_x1), int(disp_y1)],
                "center_scaled_xy": [float((x0 + x1) / 2.0), float((y0 + y1) / 2.0)],
                "center_display_xy": [float((disp_x0 + disp_x1) / 2.0), float((disp_y0 + disp_y1) / 2.0)],
            }
        )
    defs.sort(key=lambda row: (int(row["row_display"]), int(row["col_display"])))
    return defs


@dataclass
class ConfocalOrientationProbeConfig:
    myelin_label: str
    myelin_rgb: np.ndarray
    myelin_labels: np.ndarray
    confocal_projection_u8: np.ndarray
    out_root: Path
    confocal_sources: list[Path]
    confocal_source_mode: str
    nominal_overlap_fraction: float
    projection_info: dict[str, Any] | None
    projection_mode: str
    channel_index: int
    tx_px: float = 0.0
    ty_px: float = 0.0
    angle_deg: float = 0.0
    scale: float = 1.0


def run_confocal_orientation_probe(cfg: ConfocalOrientationProbeConfig) -> dict[str, Any]:
    session_id = f"{_utc_stamp()}_{cfg.projection_mode}_ch{cfg.channel_index}_orientation_probe"
    run_dir = cfg.out_root / cfg.myelin_label / session_id
    inputs_dir = run_dir / "inputs"
    run_dir.mkdir(parents=True, exist_ok=True)
    inputs_dir.mkdir(parents=True, exist_ok=True)

    fixed_rgb = cfg.myelin_rgb.copy()
    fixed_gray_full = rgb_to_gray_float(fixed_rgb)
    fixed_mask_full = (cfg.myelin_labels == 1).astype(np.float32)
    if not np.any(fixed_mask_full > 0):
        fixed_mask_full = (cfg.myelin_labels > 0).astype(np.float32)

    variants = [
        ("none", "None", False, False),
        ("flip_lr", "Flip LR", True, False),
        ("flip_ud", "Flip UD", False, True),
        ("flip_lr_ud", "Flip LR+UD", True, True),
    ]

    variant_records: list[dict[str, Any]] = []
    union_mask = np.zeros(fixed_gray_full.shape, dtype=np.uint8)
    for key, label, flip_lr, flip_ud in variants:
        moving_u8, moving_mask_u8, manual_mat = apply_manual_transform(
            cfg.confocal_projection_u8,
            fixed_rgb.shape[:2],
            tx_px=cfg.tx_px,
            ty_px=cfg.ty_px,
            angle_deg=cfg.angle_deg,
            scale=cfg.scale,
            flip_lr=flip_lr,
            flip_ud=flip_ud,
        )
        union_mask = np.maximum(union_mask, np.where(moving_mask_u8 > 0, 255, 0).astype(np.uint8))
        variant_records.append(
            {
                "key": key,
                "label": label,
                "flip_lr": flip_lr,
                "flip_ud": flip_ud,
                "moving_u8_full": moving_u8,
                "moving_mask_u8_full": moving_mask_u8,
                "manual_mat": manual_mat,
            }
        )

    roi_bbox = _bbox_from_mask(union_mask, margin_px=max(96, min(fixed_rgb.shape[:2]) // 16))
    y0, y1, x0, x1 = roi_bbox
    fixed_gray = fixed_gray_full[y0:y1, x0:x1]
    fixed_mask = fixed_mask_full[y0:y1, x0:x1]
    cv2.imwrite(str(inputs_dir / "myelin_fixed_local_roi.png"), np.clip(np.round(fixed_gray * 255.0), 0, 255).astype(np.uint8))

    best_key = None
    best_cc = -float("inf")
    storyboard_rows: list[dict[str, Any]] = []

    for rec in variant_records:
        moving_gray = rec["moving_u8_full"][y0:y1, x0:x1].astype(np.float32) / 255.0
        moving_mask = (rec["moving_mask_u8_full"][y0:y1, x0:x1] > 0).astype(np.float32)
        metrics, timings = compute_registration_metrics(fixed_gray, moving_gray, fixed_mask, moving_mask)
        cc = float(metrics.get("cc", float("nan")))
        if np.isfinite(cc) and cc > best_cc:
            best_cc = cc
            best_key = str(rec["key"])

        local_bbox = _bbox_from_mask((moving_mask > 0).astype(np.uint8), margin_px=max(24, min(fixed_gray.shape[:2]) // 12))
        note = metrics_note(metrics, timings, rec["label"])
        rec["metrics"] = metrics
        rec["timings"] = timings
        rec["local_bbox_yxyx"] = [int(v) for v in local_bbox]
        if rec["key"] == best_key:
            note = f"{note} | BEST_CC_SO_FAR"
        storyboard_rows.append(
            {
                "label": rec["label"],
                "note": note,
                "moving": gray_preview_panel(moving_gray),
                "fixed": gray_preview_panel(fixed_gray),
                "overlay": overlay_preview(fixed_gray, moving_gray, fixed_mask, moving_mask),
                "heatmap": _local_zoom_overlay_panel(fixed_gray, moving_gray, fixed_mask, moving_mask, local_bbox),
                "col_titles": ("Moving", "Fixed", "Overlay", "Local zoom"),
            }
        )
        cv2.imwrite(str(inputs_dir / f"{rec['key']}_local_roi.png"), np.clip(np.round(moving_gray * 255.0), 0, 255).astype(np.uint8))

    # Rewrite BEST tag after final ranking.
    ranked = sorted(
        variant_records,
        key=lambda rec: float(rec.get("metrics", {}).get("cc", -float("inf"))),
        reverse=True,
    )
    rank_map = {str(rec["key"]): idx + 1 for idx, rec in enumerate(ranked)}
    storyboard_rows = []
    for rec in variant_records:
        moving_gray = rec["moving_u8_full"][y0:y1, x0:x1].astype(np.float32) / 255.0
        moving_mask = (rec["moving_mask_u8_full"][y0:y1, x0:x1] > 0).astype(np.float32)
        prefix = f"{rec['label']} | rank={rank_map[str(rec['key'])]}"
        if rec["key"] == best_key:
            prefix += " | BEST"
        storyboard_rows.append(
            {
                "label": rec["label"],
                "note": metrics_note(rec["metrics"], rec["timings"], prefix),
                "moving": gray_preview_panel(moving_gray),
                "fixed": gray_preview_panel(fixed_gray),
                "overlay": overlay_preview(fixed_gray, moving_gray, fixed_mask, moving_mask),
                "heatmap": _local_zoom_overlay_panel(
                    fixed_gray,
                    moving_gray,
                    fixed_mask,
                    moving_mask,
                    tuple(int(v) for v in rec["local_bbox_yxyx"]),
                ),
                "col_titles": ("Moving", "Fixed", "Overlay", "Local zoom"),
            }
        )

    storyboard_path = run_dir / "orientation_contact_sheet.png"
    render_storyboard(storyboard_rows, storyboard_path)

    manifest = {
        "myelin_label": cfg.myelin_label,
        "run_dir": str(run_dir),
        "storyboard_path": str(storyboard_path),
        "confocal_sources": [str(path) for path in cfg.confocal_sources],
        "confocal_source_mode": cfg.confocal_source_mode,
        "nominal_overlap_fraction": float(cfg.nominal_overlap_fraction),
        "projection_info": cfg.projection_info or {},
        "projection_mode": cfg.projection_mode,
        "channel_index": int(cfg.channel_index),
        "saved_at_utc": _utc_iso(),
        "manual_init": {
            "tx_px": float(cfg.tx_px),
            "ty_px": float(cfg.ty_px),
            "angle_deg": float(cfg.angle_deg),
            "scale": float(cfg.scale),
        },
        "probe_roi_bbox_yxyx": [int(y0), int(y1), int(x0), int(x1)],
        "probe_roi_shape_hw": [int(fixed_gray.shape[0]), int(fixed_gray.shape[1])],
        "variants": [
            {
                "key": str(rec["key"]),
                "label": str(rec["label"]),
                "flip_lr": bool(rec["flip_lr"]),
                "flip_ud": bool(rec["flip_ud"]),
                "rank": int(rank_map[str(rec["key"])]),
                "metrics": rec["metrics"],
                "metric_timings": rec["timings"],
                "local_bbox_yxyx": list(rec["local_bbox_yxyx"]),
            }
            for rec in variant_records
        ],
        "best_variant": next(
            (
                {
                    "key": str(rec["key"]),
                    "label": str(rec["label"]),
                    "flip_lr": bool(rec["flip_lr"]),
                    "flip_ud": bool(rec["flip_ud"]),
                    "metrics": rec["metrics"],
                }
                for rec in ranked[:1]
            ),
            None,
        ),
        "files": {
            "storyboard": str(storyboard_path),
            "manifest": str(run_dir / "orientation_probe_manifest.json"),
            "myelin_fixed_local_roi": str(inputs_dir / "myelin_fixed_local_roi.png"),
        },
    }
    _write_json(run_dir / "orientation_probe_manifest.json", manifest)
    return manifest


def _affine_apply_points(mat: np.ndarray, xy: np.ndarray) -> np.ndarray:
    arr = np.asarray(xy, dtype=np.float64)
    aug = np.concatenate([arr, np.ones((arr.shape[0], 1), dtype=np.float64)], axis=1)
    out = aug @ np.asarray(mat, dtype=np.float64).T
    return out[:, :2]


def _gray01_to_u8(arr: np.ndarray) -> np.ndarray:
    return np.clip(np.round(np.asarray(arr, dtype=np.float32) * 255.0), 0, 255).astype(np.uint8)


def _crop_to_bbox(arr: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    y0, y1, x0, x1 = bbox
    return np.asarray(arr)[y0:y1, x0:x1]


def _safe_match_template(
    fixed_search: np.ndarray,
    moving_template: np.ndarray,
    mask_template: np.ndarray,
) -> tuple[float, tuple[int, int]]:
    fixed_u8 = _gray01_to_u8(fixed_search)
    moving_u8 = _gray01_to_u8(moving_template)
    mask_u8 = np.where(np.asarray(mask_template) > 0, 255, 0).astype(np.uint8)
    if fixed_u8.shape[0] < moving_u8.shape[0] or fixed_u8.shape[1] < moving_u8.shape[1]:
        return float("nan"), (0, 0)
    if int(mask_u8.sum()) <= 0:
        return float("nan"), (0, 0)
    res = cv2.matchTemplate(fixed_u8, moving_u8, cv2.TM_CCORR_NORMED, mask=mask_u8)
    _min_val, max_val, _min_loc, max_loc = cv2.minMaxLoc(res)
    return float(max_val), (int(max_loc[0]), int(max_loc[1]))


def _safe_match_template_topk(
    fixed_search: np.ndarray,
    moving_template: np.ndarray,
    mask_template: np.ndarray,
    *,
    top_k: int,
    min_separation_px: int = 1,
) -> list[tuple[float, tuple[int, int]]]:
    fixed_u8 = _gray01_to_u8(fixed_search)
    moving_u8 = _gray01_to_u8(moving_template)
    mask_u8 = np.where(np.asarray(mask_template) > 0, 255, 0).astype(np.uint8)
    if fixed_u8.shape[0] < moving_u8.shape[0] or fixed_u8.shape[1] < moving_u8.shape[1]:
        return []
    if int(mask_u8.sum()) <= 0:
        return []
    res = cv2.matchTemplate(fixed_u8, moving_u8, cv2.TM_CCORR_NORMED, mask=mask_u8)
    score_map = np.asarray(res, dtype=np.float32)
    if score_map.size <= 0:
        return []
    flat = score_map.reshape(-1)
    valid_idx = np.flatnonzero(np.isfinite(flat))
    if valid_idx.size <= 0:
        return []
    want = max(1, int(top_k))
    pick_n = min(int(valid_idx.size), max(want, want * 12))
    local_vals = flat[valid_idx]
    top_sel = np.argpartition(local_vals, -pick_n)[-pick_n:]
    ranked = sorted(
        ((float(local_vals[int(i)]), int(valid_idx[int(i)])) for i in top_sel),
        key=lambda item: item[0],
        reverse=True,
    )
    out: list[tuple[float, tuple[int, int]]] = []
    chosen: list[tuple[int, int]] = []
    map_w = int(score_map.shape[1])
    sep = max(0, int(min_separation_px))
    for score, flat_idx in ranked:
        y = int(flat_idx // map_w)
        x = int(flat_idx % map_w)
        if any(max(abs(x - cx), abs(y - cy)) <= sep for cx, cy in chosen):
            continue
        out.append((float(score), (int(x), int(y))))
        chosen.append((int(x), int(y)))
        if len(out) >= want:
            break
    return out


def _get_step7_profile_search_context(
    *,
    ctx: dict[str, Any],
    registration_input_profile: str,
) -> dict[str, Any]:
    cache = ctx.setdefault("_profile_search_ctx", {})
    profile_key = str(registration_input_profile or STEP7_REGISTRATION_INPUT_PROFILE).strip().lower()
    cached = cache.get(profile_key)
    if isinstance(cached, dict):
        return cached
    reg_inputs = _prepare_step7_registration_inputs(
        np.asarray(ctx["fixed_crop"], dtype=np.float32),
        np.asarray(ctx["moving_global_crop"], dtype=np.float32),
        np.asarray(ctx["fixed_mask_crop"], dtype=np.float32),
        np.asarray(ctx["moving_global_mask_crop"], dtype=np.float32),
        profile=profile_key,
    )
    fixed_search_source = np.asarray(reg_inputs["fixed_gray"], dtype=np.float32)
    moving_search_source = np.asarray(reg_inputs["moving_gray"], dtype=np.float32)
    ty0, ty1, tx0, tx1 = [int(v) for v in ctx["tight_bbox"]]
    template = moving_search_source[ty0:ty1, tx0:tx1]
    template_mask = np.asarray(ctx["moving_global_mask_crop"], dtype=np.float32)[ty0:ty1, tx0:tx1]
    cached = {
        "profile": profile_key,
        "fixed_search_source": fixed_search_source,
        "fixed_mask_source": np.asarray(ctx["fixed_mask_crop"], dtype=np.float32),
        "moving_search_source": moving_search_source,
        "tight_bbox": (int(ty0), int(ty1), int(tx0), int(tx1)),
        "template": template,
        "template_mask": template_mask,
        "template_shape_hw": (int(template.shape[0]), int(template.shape[1])),
    }
    cache[profile_key] = cached
    return cached


def _template_match_candidates_for_profile(
    *,
    ctx: dict[str, Any],
    registration_input_profile: str,
    anchor_dx: int,
    anchor_dy: int,
    search_radius_px: int,
    top_k: int,
) -> list[dict[str, Any]]:
    profile_ctx = _get_step7_profile_search_context(
        ctx=ctx,
        registration_input_profile=registration_input_profile,
    )
    fixed_search_source = np.asarray(profile_ctx["fixed_search_source"], dtype=np.float32)
    template = np.asarray(profile_ctx["template"], dtype=np.float32)
    template_mask = np.asarray(profile_ctx["template_mask"], dtype=np.float32)
    ty0, ty1, tx0, tx1 = [int(v) for v in profile_ctx["tight_bbox"]]
    templ_h, templ_w = [int(v) for v in profile_ctx["template_shape_hw"]]
    prior_anchor_x = int(tx0 + int(anchor_dx))
    prior_anchor_y = int(ty0 + int(anchor_dy))
    search_radius = max(0, int(search_radius_px))
    sy0 = max(0, prior_anchor_y - search_radius)
    sy1 = min(int(fixed_search_source.shape[0]), prior_anchor_y + templ_h + search_radius)
    sx0 = max(0, prior_anchor_x - search_radius)
    sx1 = min(int(fixed_search_source.shape[1]), prior_anchor_x + templ_w + search_radius)
    fixed_search = fixed_search_source[sy0:sy1, sx0:sx1]
    raw_matches = _safe_match_template_topk(
        fixed_search,
        template,
        template_mask,
        top_k=max(1, int(top_k)),
        min_separation_px=1,
    )
    candidates: list[dict[str, Any]] = []
    for score, (loc_x, loc_y) in raw_matches:
        best_x = int(sx0 + int(loc_x))
        best_y = int(sy0 + int(loc_y))
        candidates.append(
            {
                "template_match_score": float(score),
                "dx": int(best_x - tx0),
                "dy": int(best_y - ty0),
            }
        )
    return candidates


def _score_cached_profile_shift_fast(
    *,
    profile_ctx: dict[str, Any],
    dx_px: int,
    dy_px: int,
    objective_name: str,
) -> dict[str, Any] | None:
    fixed_search_source = np.asarray(profile_ctx["fixed_search_source"], dtype=np.float32)
    fixed_mask_source = np.asarray(profile_ctx["fixed_mask_source"], dtype=np.float32)
    template = np.asarray(profile_ctx["template"], dtype=np.float32)
    template_mask = np.asarray(profile_ctx["template_mask"], dtype=np.float32)
    ty0, ty1, tx0, tx1 = [int(v) for v in profile_ctx["tight_bbox"]]
    y0 = int(ty0 + int(dy_px))
    y1 = int(ty1 + int(dy_px))
    x0 = int(tx0 + int(dx_px))
    x1 = int(tx1 + int(dx_px))
    if y0 < 0 or x0 < 0 or y1 > int(fixed_search_source.shape[0]) or x1 > int(fixed_search_source.shape[1]):
        return None
    fixed_patch = np.asarray(fixed_search_source[y0:y1, x0:x1], dtype=np.float32)
    fixed_mask_patch = np.asarray(fixed_mask_source[y0:y1, x0:x1], dtype=np.float32)
    if fixed_patch.shape != template.shape or fixed_mask_patch.shape != template.shape:
        return None
    moving_mask_bool = np.asarray(template_mask) > 0
    fixed_mask_bool = np.asarray(fixed_mask_patch) > 0
    valid = moving_mask_bool | fixed_mask_bool
    valid_count = int(np.count_nonzero(valid))
    if valid_count <= 0:
        return {
            "objective_score": float("nan"),
            "cc": float("nan"),
            "mi": float("nan"),
            "valid_pixels": 0,
        }
    fx = fixed_patch[valid].astype(np.float32)
    mv = template[valid].astype(np.float32)
    objective = str(objective_name or "cc").strip().lower()
    cc = float("nan")
    mi = float("nan")
    if objective in {"cc", "hybrid"}:
        cc = float(_cross_correlation(fx, mv))
    if objective in {"mi", "hybrid"}:
        mi = float(_mutual_information(fx, mv))
    objective_score = _objective_score_from_metrics(
        {"cc": cc, "mi": mi},
        objective_name=objective,
    )
    return {
        "objective_score": float(objective_score),
        "cc": cc,
        "mi": mi,
        "valid_pixels": valid_count,
    }


def _edge_density_from_gray(gray01: np.ndarray, mask: np.ndarray) -> float:
    mask_u8 = np.where(np.asarray(mask) > 0, 255, 0).astype(np.uint8)
    if int(mask_u8.sum()) <= 0:
        return 0.0
    gray_u8 = _gray01_to_u8(gray01)
    norm_u8 = _masked_percentile_normalize_u8(gray_u8, mask_u8)
    norm_u8 = cv2.GaussianBlur(norm_u8, (0, 0), sigmaX=1.0, sigmaY=1.0)
    edges = cv2.Canny(norm_u8, 40, 120)
    inside = mask_u8 > 0
    if not np.any(inside):
        return 0.0
    return float(np.mean(edges[inside] > 0))


def _step7_density_regime(
    *,
    signal_coverage: float,
    moving_edge_density: float,
    fixed_edge_density: float,
) -> tuple[str, float]:
    edge_mean = 0.5 * (float(moving_edge_density) + float(fixed_edge_density))
    if float(signal_coverage) >= 0.70 or (float(signal_coverage) >= 0.55 and edge_mean >= 0.10):
        return "high_density", edge_mean
    if float(signal_coverage) < 0.35:
        if edge_mean >= 0.07:
            return "sparse_feature", edge_mean
        return "sparse_weak", edge_mean
    return "mid_density", edge_mean


def _merge_profile_sequences(*groups: list[str] | tuple[str, ...]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for group in groups:
        for profile in group:
            key = str(profile or "").strip().lower()
            if not key or key in seen:
                continue
            seen.add(key)
            ordered.append(key)
    return ordered


def _objective_score_from_metrics(metrics: dict[str, Any] | None, *, objective_name: str) -> float:
    data = metrics if isinstance(metrics, dict) else {}
    cc = float(data.get("cc", float("nan")))
    mi = float(data.get("mi", float("nan")))
    objective = str(objective_name or "cc").strip().lower()
    if objective == "mi":
        return mi
    if objective == "hybrid":
        if np.isfinite(cc) and np.isfinite(mi):
            return float(cc + STEP7_HYBRID_MI_WEIGHT * mi)
        if np.isfinite(cc):
            return cc
        return mi
    return cc


def _density_aware_profile_policy(
    *,
    preferred_profile: str,
    signal_coverage: float,
    moving_edge_density: float,
    fixed_edge_density: float,
) -> tuple[str, float, list[str], list[str], str]:
    density_regime, edge_mean = _step7_density_regime(
        signal_coverage=signal_coverage,
        moving_edge_density=moving_edge_density,
        fixed_edge_density=fixed_edge_density,
    )
    if density_regime == "high_density":
        coarse_profiles = [
            "paired_percentile_blur8",
            "paired_percentile_blur6",
            "paired_percentile_blur4",
        ]
        refine_profiles = [
            "paired_percentile_blur6",
            "paired_percentile_blur4",
            "paired_percentile_blur2",
        ]
        refine_objective = "cc"
    elif density_regime == "mid_density":
        coarse_profiles = [
            "paired_percentile_blur8",
            "paired_percentile_blur4",
            "paired_percentile_clahe_blur3",
        ]
        refine_profiles = [
            "paired_percentile_blur4",
            "paired_percentile_blur2",
            "paired_percentile_clahe_blur3",
            "paired_percentile_blur1",
        ]
        refine_objective = "cc"
    elif density_regime == "sparse_feature":
        coarse_profiles = [
            "paired_percentile_blur6",
            "paired_percentile_blur4",
            "paired_percentile_clahe_blur3",
        ]
        refine_profiles = [
            "paired_percentile_blur2",
            "paired_percentile_blur1",
            "paired_percentile_clahe_blur3",
        ]
        refine_objective = str(STEP7_SPARSE_REFINE_OBJECTIVE or "cc")
    else:
        coarse_profiles = [
            "paired_percentile_blur4",
            "paired_percentile_clahe_blur3",
        ]
        refine_profiles = [
            "paired_percentile_blur2",
            "paired_percentile_blur1",
            "paired_percentile_clahe_blur3",
        ]
        refine_objective = str(STEP7_SPARSE_REFINE_OBJECTIVE or "cc")
    allowed_auto_profiles = {
        "paired_percentile_blur8",
        "paired_percentile_blur6",
        "paired_percentile_blur4",
        "paired_percentile_blur2",
        "paired_percentile_blur1",
        "paired_percentile_clahe_blur3",
    }
    preferred_key = str(preferred_profile or "").strip().lower()
    if preferred_key in allowed_auto_profiles and preferred_key not in {"auto", "density_aware", STEP7_REGISTRATION_INPUT_PROFILE.lower()}:
        coarse_profiles = [preferred_key, *coarse_profiles]
        refine_profiles = [preferred_key, *refine_profiles]
    coarse_ordered = _merge_profile_sequences(coarse_profiles)
    refine_ordered = _merge_profile_sequences(refine_profiles)
    return density_regime, float(edge_mean), coarse_ordered, refine_ordered, str(refine_objective or "cc")


def _profile_candidate_score(
    *,
    current_cc: float,
    final_cc: float,
    current_objective_score: float,
    final_objective_score: float,
    prior_dx: int,
    prior_dy: int,
    final_dx: int,
    final_dy: int,
    template_match_score: float,
    accepted: bool,
    profile_key: str,
    density_regime: str,
    objective_name: str,
) -> float:
    objective = str(objective_name or "cc").strip().lower()
    if objective == "cc" or not np.isfinite(final_objective_score):
        score = float(final_cc) if np.isfinite(final_cc) else -1.0
        if np.isfinite(current_cc) and np.isfinite(final_cc):
            score += 0.20 * max(0.0, float(final_cc) - float(current_cc))
    else:
        score = float(final_objective_score)
        if np.isfinite(current_objective_score) and np.isfinite(final_objective_score):
            score += 0.18 * max(0.0, float(final_objective_score) - float(current_objective_score))
        if np.isfinite(final_cc):
            score += 0.12 * float(final_cc)
    if accepted:
        score += 0.02
    if np.isfinite(template_match_score):
        score += 0.05 * float(template_match_score)
    score -= 0.005 * float(np.hypot(float(final_dx - prior_dx), float(final_dy - prior_dy)))
    profile = str(profile_key or "").strip().lower()
    regime = str(density_regime or "").strip().lower()
    bias = 0.0
    if regime.startswith("sparse"):
        if profile == "paired_percentile_raw":
            bias += 0.030
        elif profile == "paired_percentile_clahe_blur3":
            bias += 0.024
        elif profile == "paired_percentile_clahe_blur2":
            bias += 0.022
        elif profile == "paired_percentile_blur2":
            bias += 0.015
        elif profile == "paired_percentile_blur1":
            bias += 0.012
        elif profile == "paired_percentile_blur4":
            bias -= 0.008
        elif profile == "paired_percentile_blur6":
            bias -= 0.020
        elif profile == "paired_percentile_blur8":
            bias -= 0.035
    elif regime == "mid_density":
        if profile == "paired_percentile_clahe_blur3":
            bias += 0.014
        elif profile == "paired_percentile_blur4":
            bias += 0.010
        elif profile == "paired_percentile_blur2":
            bias += 0.006
        elif profile == "paired_percentile_blur8":
            bias -= 0.010
    elif regime == "high_density":
        if profile == "paired_percentile_blur6":
            bias += 0.012
        elif profile == "paired_percentile_blur8":
            bias += 0.010
        elif profile == "moving_gamma_clahe_hist_blur4":
            bias += 0.010
        elif profile == "paired_percentile_raw":
            bias -= 0.012
    score += bias
    return float(score)


def _local_refine_shift_for_profile(
    *,
    ctx: dict[str, Any],
    fixed_gray_full: np.ndarray,
    fixed_mask_full: np.ndarray,
    registration_input_profile: str,
    center_dx: int,
    center_dy: int,
    radius_px: int,
    objective_name: str,
) -> dict[str, Any]:
    best: dict[str, Any] | None = None
    radius = max(0, int(radius_px))
    profile_ctx = _get_step7_profile_search_context(
        ctx=ctx,
        registration_input_profile=registration_input_profile,
    )
    fast_candidates = _template_match_candidates_for_profile(
        ctx=ctx,
        registration_input_profile=registration_input_profile,
        anchor_dx=int(center_dx),
        anchor_dy=int(center_dy),
        search_radius_px=radius,
        top_k=max(1, int(STEP7_FAST_REFINE_TOPK)),
    )
    candidate_by_shift: dict[tuple[int, int], dict[str, Any]] = {
        (int(center_dx), int(center_dy)): {
            "dx": int(center_dx),
            "dy": int(center_dy),
            "template_match_score": float("nan"),
        }
    }
    for candidate in fast_candidates:
        key = (int(candidate.get("dx", center_dx)), int(candidate.get("dy", center_dy)))
        candidate_by_shift[key] = {
            "dx": int(candidate.get("dx", center_dx)),
            "dy": int(candidate.get("dy", center_dy)),
            "template_match_score": float(candidate.get("template_match_score", float("nan"))),
        }

    for candidate_seed in candidate_by_shift.values():
        dx = int(candidate_seed["dx"])
        dy = int(candidate_seed["dy"])
        fast_metrics = _score_cached_profile_shift_fast(
            profile_ctx=profile_ctx,
            dx_px=dx,
            dy_px=dy,
            objective_name=objective_name,
        )
        if fast_metrics is None:
            continue
        obj_score = float(fast_metrics.get("objective_score", float("nan")))
        cc_score = float(fast_metrics.get("cc", float("nan")))
        dist = float(np.hypot(float(dx - center_dx), float(dy - center_dy)))
        candidate = {
            "dx": int(dx),
            "dy": int(dy),
            "objective_score": float(obj_score),
            "cc_score": float(cc_score),
            "dist_from_center": dist,
            "template_match_score": float(candidate_seed.get("template_match_score", float("nan"))),
        }
        if best is None:
            best = candidate
            continue
        cand_obj = float(candidate["objective_score"])
        best_obj = float(best["objective_score"])
        if (np.isfinite(cand_obj) and not np.isfinite(best_obj)) or cand_obj > best_obj + 1e-6:
            best = candidate
            continue
        if abs(cand_obj - best_obj) <= 1e-6:
            cand_cc = float(candidate["cc_score"])
            best_cc = float(best["cc_score"])
            if (np.isfinite(cand_cc) and not np.isfinite(best_cc)) or cand_cc > best_cc + 1e-6:
                best = candidate
                continue
            if abs(cand_cc - best_cc) <= 1e-6 and float(candidate["dist_from_center"]) < float(best["dist_from_center"]):
                best = candidate
    if best is None:
        fallback_full = _evaluate_tile_shift_from_context(
            ctx=ctx,
            fixed_gray_full=fixed_gray_full,
            fixed_mask_full=fixed_mask_full,
            registration_input_profile=registration_input_profile,
            dx_px=int(center_dx),
            dy_px=int(center_dy),
            metrics_mode="full",
            objective_name=objective_name,
        )
        fallback_metrics = fallback_full.get("metrics") if isinstance(fallback_full, dict) else {}
        return {
            "eval": fallback_full,
            "dx": int(center_dx),
            "dy": int(center_dy),
            "objective_score": _objective_score_from_metrics(fallback_metrics, objective_name=objective_name),
            "cc_score": float(fallback_metrics.get("cc", float("nan"))) if isinstance(fallback_metrics, dict) else float("nan"),
            "dist_from_center": 0.0,
            "template_match_score": float("nan"),
        }

    winner_full = _evaluate_tile_shift_from_context(
        ctx=ctx,
        fixed_gray_full=fixed_gray_full,
        fixed_mask_full=fixed_mask_full,
        registration_input_profile=registration_input_profile,
        dx_px=int(best["dx"]),
        dy_px=int(best["dy"]),
        metrics_mode="full",
        objective_name=objective_name,
    )
    winner_metrics = winner_full.get("metrics") if isinstance(winner_full, dict) else {}
    return {
        "eval": winner_full,
        "dx": int(best["dx"]),
        "dy": int(best["dy"]),
        "objective_score": _objective_score_from_metrics(winner_metrics, objective_name=objective_name),
        "cc_score": float(winner_metrics.get("cc", float("nan"))) if isinstance(winner_metrics, dict) else float("nan"),
        "dist_from_center": float(best["dist_from_center"]),
        "template_match_score": float(best.get("template_match_score", float("nan"))),
    }


def _template_match_shift_for_profile(
    *,
    ctx: dict[str, Any],
    registration_input_profile: str,
    prior_dx: int,
    prior_dy: int,
    search_radius_px: int,
) -> tuple[float, int, int]:
    candidates = _template_match_candidates_for_profile(
        ctx=ctx,
        registration_input_profile=str(registration_input_profile or STEP7_REGISTRATION_INPUT_PROFILE),
        anchor_dx=int(prior_dx),
        anchor_dy=int(prior_dy),
        search_radius_px=int(search_radius_px),
        top_k=1,
    )
    if not candidates:
        return float("nan"), int(prior_dx), int(prior_dy)
    best = candidates[0]
    return (
        float(best.get("template_match_score", float("nan"))),
        int(best.get("dx", prior_dx)),
        int(best.get("dy", prior_dy)),
    )


def _shift_gray_and_mask(
    gray: np.ndarray,
    mask: np.ndarray,
    *,
    dx: int,
    dy: int,
) -> tuple[np.ndarray, np.ndarray]:
    h, w = gray.shape[:2]
    mat = np.array([[1.0, 0.0, float(dx)], [0.0, 1.0, float(dy)]], dtype=np.float32)
    shifted_gray = cv2.warpAffine(
        np.asarray(gray, dtype=np.float32),
        mat,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0,
    )
    shifted_mask_u8 = cv2.warpAffine(
        np.where(np.asarray(mask) > 0, 255, 0).astype(np.uint8),
        mat,
        (w, h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return shifted_gray.astype(np.float32), (shifted_mask_u8 > 0).astype(np.float32)


def _warp_screen_tile_from_scaled(
    scaled_reg_projection_u8: np.ndarray,
    scaled_mask_u8: np.ndarray,
    fixed_shape_hw: tuple[int, int],
    global_mat: np.ndarray,
    tile_def: dict[str, Any],
    *,
    margin_px: int,
    tile_static: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if isinstance(tile_static, dict) and tile_static:
        x0 = int(tile_static.get("x0", 0))
        y0 = int(tile_static.get("y0", 0))
        tile_img = np.asarray(tile_static.get("tile_img_u8"), dtype=np.uint8)
        tile_mask = np.asarray(tile_static.get("tile_mask_u8"), dtype=np.uint8)
    else:
        x0, y0, x1, y1 = [int(v) for v in tile_def["scaled_bbox_xyxy"]]
        tile_img = np.asarray(scaled_reg_projection_u8[y0:y1, x0:x1], dtype=np.uint8)
        tile_mask = np.asarray(scaled_mask_u8[y0:y1, x0:x1], dtype=np.uint8)
    linear = np.asarray(global_mat[:, :2], dtype=np.float32)
    trans = np.asarray(global_mat[:, 2], dtype=np.float32) + linear @ np.array([float(x0), float(y0)], dtype=np.float32)
    tile_mat = np.concatenate([linear, trans[:, None]], axis=1).astype(np.float32)
    tile_h, tile_w = tile_img.shape[:2]
    corners = np.asarray(
        [[0.0, 0.0], [float(tile_w), 0.0], [float(tile_w), float(tile_h)], [0.0, float(tile_h)]],
        dtype=np.float64,
    )
    warped_corners = _affine_apply_points(tile_mat, corners)
    bx0 = max(0, int(math.floor(float(np.min(warped_corners[:, 0]))) - margin_px))
    by0 = max(0, int(math.floor(float(np.min(warped_corners[:, 1]))) - margin_px))
    bx1 = min(int(fixed_shape_hw[1]), int(math.ceil(float(np.max(warped_corners[:, 0]))) + margin_px))
    by1 = min(int(fixed_shape_hw[0]), int(math.ceil(float(np.max(warped_corners[:, 1]))) + margin_px))
    local_w = max(1, bx1 - bx0)
    local_h = max(1, by1 - by0)
    local_mat = tile_mat.copy()
    local_mat[0, 2] -= float(bx0)
    local_mat[1, 2] -= float(by0)
    warped_img_u8, warped_mask_u8, _ = apply_affine_matrix(
        tile_img,
        (local_h, local_w),
        mat=local_mat,
        moving_mask_u8=tile_mask,
    )
    return {
        "tile_img_u8": tile_img,
        "tile_mask_u8": tile_mask,
        "tile_mat": tile_mat,
        "warped_img_u8": warped_img_u8,
        "warped_gray": warped_img_u8.astype(np.float32) / 255.0,
        "warped_mask": (warped_mask_u8 > 0).astype(np.float32),
        "eval_bbox_yxyx": (by0, by1, bx0, bx1),
    }


def _build_tile_eval_context(
    *,
    fixed_gray_full: np.ndarray,
    fixed_mask_full: np.ndarray,
    moving_reg_projection_u8: np.ndarray,
    moving_signal_mask_u8: np.ndarray,
    fixed_shape_hw: tuple[int, int],
    manual_mat: np.ndarray,
    tile: dict[str, Any],
    margin_px: int,
    tile_static: dict[str, Any] | None = None,
) -> dict[str, Any]:
    warped = _warp_screen_tile_from_scaled(
        moving_reg_projection_u8,
        moving_signal_mask_u8,
        fixed_shape_hw,
        manual_mat,
        tile,
        margin_px=margin_px,
        tile_static=tile_static,
    )
    eval_bbox = tuple(int(v) for v in warped["eval_bbox_yxyx"])
    moving_global_crop = np.asarray(warped["warped_gray"], dtype=np.float32)
    moving_global_mask_crop = np.asarray(warped["warped_mask"], dtype=np.float32)
    tile_img_u8 = np.asarray(
        tile_static.get("tile_img_u8") if isinstance(tile_static, dict) and tile_static.get("tile_img_u8") is not None else warped["tile_img_u8"],
        dtype=np.uint8,
    )
    tile_mask_u8 = np.asarray(
        tile_static.get("tile_mask_u8") if isinstance(tile_static, dict) and tile_static.get("tile_mask_u8") is not None else warped["tile_mask_u8"],
        dtype=np.uint8,
    )
    tight_bbox = _bbox_from_mask(moving_global_mask_crop, margin_px=0)
    return {
        "tile": dict(tile),
        "warped": warped,
        "eval_bbox": eval_bbox,
        "fixed_crop": _crop_to_bbox(fixed_gray_full, eval_bbox),
        "fixed_mask_crop": _crop_to_bbox(fixed_mask_full, eval_bbox),
        "moving_global_crop": moving_global_crop,
        "moving_global_mask_crop": moving_global_mask_crop,
        "tile_img_u8": tile_img_u8,
        "tile_mask_u8": tile_mask_u8,
        "tile_local_gray": np.asarray(
            tile_static.get("tile_local_gray") if isinstance(tile_static, dict) and tile_static.get("tile_local_gray") is not None else tile_img_u8.astype(np.float32) / 255.0,
            dtype=np.float32,
        ),
        "tile_local_mask": np.asarray(
            tile_static.get("tile_local_mask") if isinstance(tile_static, dict) and tile_static.get("tile_local_mask") is not None else (tile_mask_u8 > 0).astype(np.float32),
            dtype=np.float32,
        ),
        "tight_bbox": tuple(int(v) for v in tight_bbox),
    }


def _evaluate_tile_shift_from_context(
    *,
    ctx: dict[str, Any],
    fixed_gray_full: np.ndarray,
    fixed_mask_full: np.ndarray,
    registration_input_profile: str,
    dx_px: int,
    dy_px: int,
    metrics_mode: str = "full",
    objective_name: str = "cc",
) -> dict[str, Any]:
    tile_mat = np.asarray(ctx["warped"]["tile_mat"], dtype=np.float32).copy()
    tile_mat[0, 2] += float(dx_px)
    tile_mat[1, 2] += float(dy_px)
    tile_img_u8 = np.asarray(ctx["tile_img_u8"], dtype=np.uint8)
    fixed_tile_local, fixed_tile_local_mask = _backwarp_fixed_to_tile_local(
        fixed_gray_full,
        fixed_mask_full,
        tile_mat,
        tile_img_u8.shape[:2],
    )
    profile_key = str(registration_input_profile or STEP7_REGISTRATION_INPUT_PROFILE).strip().lower()
    moving_cache = _get_step7_moving_local_profile_cache(
        ctx=ctx,
        registration_input_profile=profile_key,
    )
    fixed_proc: np.ndarray
    moving_proc: np.ndarray
    if moving_cache is not None:
        fixed_u8 = _gray01_to_u8(fixed_tile_local)
        fixed_mask_u8 = np.where(np.asarray(fixed_tile_local_mask) > 0, 255, 0).astype(np.uint8)
        fixed_proc_u8 = _prepare_step7_independent_side_u8(
            fixed_u8,
            fixed_mask_u8,
            profile=profile_key,
        )
        if fixed_proc_u8 is not None:
            fixed_proc = fixed_proc_u8.astype(np.float32) / 255.0
            moving_proc = np.asarray(moving_cache["moving_gray"], dtype=np.float32)
        else:
            moving_cache = None
    if moving_cache is None:
        reg_inputs = _prepare_step7_registration_inputs(
            fixed_tile_local,
            np.asarray(ctx["tile_local_gray"], dtype=np.float32),
            fixed_tile_local_mask,
            np.asarray(ctx["tile_local_mask"], dtype=np.float32),
            profile=profile_key,
        )
        fixed_proc = np.asarray(reg_inputs["fixed_gray"], dtype=np.float32)
        moving_proc = np.asarray(reg_inputs["moving_gray"], dtype=np.float32)

    timings: dict[str, float] = {}
    metric_mode = str(metrics_mode or "full").strip().lower()
    objective = str(objective_name or "cc").strip().lower()
    if metric_mode == "full":
        metrics, timings = compute_registration_metrics(
            fixed_proc,
            moving_proc,
            fixed_tile_local_mask,
            np.asarray(ctx["tile_local_mask"], dtype=np.float32),
        )
    else:
        total_t0 = time.perf_counter()
        fixed_mask_bool = np.asarray(fixed_tile_local_mask) > 0
        moving_mask_bool = np.asarray(ctx["tile_local_mask"], dtype=np.float32) > 0
        valid = fixed_mask_bool | moving_mask_bool
        fx = fixed_proc[valid].astype(np.float32)
        mv = moving_proc[valid].astype(np.float32)
        metrics = {"valid_pixels": int(valid.sum())}
        if objective in {"mi", "hybrid"}:
            t0 = time.perf_counter()
            metrics["mi"] = _mutual_information(fx, mv)
            timings["mi"] = float(time.perf_counter() - t0)
        else:
            metrics["mi"] = float("nan")
        if objective in {"cc", "hybrid"}:
            t0 = time.perf_counter()
            metrics["cc"] = _cross_correlation(fx, mv)
            timings["cc"] = float(time.perf_counter() - t0)
        else:
            metrics["cc"] = float("nan")
        timings["total"] = float(time.perf_counter() - total_t0)
    return {
        "dx_px": int(dx_px),
        "dy_px": int(dy_px),
        "fixed_tile_local": fixed_tile_local,
        "fixed_tile_local_mask": fixed_tile_local_mask,
        "fixed_proc": fixed_proc,
        "moving_proc": moving_proc,
        "metrics": metrics,
        "timing": timings,
        "registration_profile": str(profile_key or STEP7_REGISTRATION_INPUT_PROFILE),
    }


def _build_tile_result_row_from_evals(
    *,
    tile: dict[str, Any],
    ctx: dict[str, Any],
    pred_eval: dict[str, Any],
    meas_eval: dict[str, Any],
    final_eval: dict[str, Any],
    template_match_score: float,
    neighbor_count: int,
    neighbor_labels: list[str] | None,
    neighbor_shift_spread_px: float,
    tile_state: str,
    proposal_gate: str,
    signal_coverage: float,
    moving_edge_density: float,
    fixed_edge_density: float,
    edge_density_mean: float,
    density_regime: str,
    profile_candidates: list[str],
    coarse_search_profiles: list[str],
    refine_profiles: list[str],
    refine_objective: str,
) -> TileResultRow:
    tile_local_gray = np.asarray(ctx["tile_local_gray"], dtype=np.float32)
    tile_local_mask = np.asarray(ctx["tile_local_mask"], dtype=np.float32)
    raw_current_overlay = overlay_preview(
        np.asarray(pred_eval["fixed_tile_local"], dtype=np.float32),
        tile_local_gray,
        np.asarray(pred_eval["fixed_tile_local_mask"], dtype=np.float32),
        tile_local_mask,
    )
    raw_final_overlay = overlay_preview(
        np.asarray(final_eval["fixed_tile_local"], dtype=np.float32),
        tile_local_gray,
        np.asarray(final_eval["fixed_tile_local_mask"], dtype=np.float32),
        tile_local_mask,
    )
    proc_current_overlay = overlay_preview(
        np.asarray(pred_eval["fixed_proc"], dtype=np.float32),
        np.asarray(pred_eval["moving_proc"], dtype=np.float32),
        np.asarray(pred_eval["fixed_tile_local_mask"], dtype=np.float32),
        tile_local_mask,
    )
    proc_final_overlay = overlay_preview(
        np.asarray(final_eval["fixed_proc"], dtype=np.float32),
        np.asarray(final_eval["moving_proc"], dtype=np.float32),
        np.asarray(final_eval["fixed_tile_local_mask"], dtype=np.float32),
        tile_local_mask,
    )
    moving_panel, fixed_panel, overlay_panel, shifted_panel = _match_panel_shapes(
        [
            raw_current_overlay,
            raw_final_overlay,
            proc_current_overlay,
            proc_final_overlay,
        ]
    )
    pred_dx = int(pred_eval["dx_px"])
    pred_dy = int(pred_eval["dy_px"])
    meas_dx = int(meas_eval["dx_px"])
    meas_dy = int(meas_eval["dy_px"])
    final_dx = int(final_eval["dx_px"])
    final_dy = int(final_eval["dy_px"])
    current_cc = float(pred_eval["metrics"].get("cc", float("nan")))
    final_cc = float(final_eval["metrics"].get("cc", float("nan")))
    delta_prior_norm = float(np.hypot(float(final_dx - pred_dx), float(final_dy - pred_dy)))
    return TileResultRow(
        tile_index=int(tile["tile_index"]),
        row_display=int(tile["row_display"]),
        col_display=int(tile["col_display"]),
        label=str(tile["label"]),
        tile_state=str(tile_state),
        pred_dx_px=float(pred_dx),
        pred_dy_px=float(pred_dy),
        meas_dx_px=float(meas_dx),
        meas_dy_px=float(meas_dy),
        final_dx_px=float(final_dx),
        final_dy_px=float(final_dy),
        current_cc=current_cc,
        current_mi=float(pred_eval["metrics"].get("mi", float("nan"))),
        meas_cc=float(meas_eval["metrics"].get("cc", float("nan"))),
        final_cc=final_cc,
        shift_gain_cc=final_cc - current_cc if np.isfinite(current_cc) and np.isfinite(final_cc) else float("nan"),
        candidate_shift_dx_px=int(meas_dx),
        candidate_shift_dy_px=int(meas_dy),
        candidate_shifted_cc=float(meas_eval["metrics"].get("cc", float("nan"))),
        template_match_score=float(template_match_score),
        proposal_gate=str(proposal_gate),
        neighbor_count=int(neighbor_count),
        neighbor_labels=[str(label) for label in (neighbor_labels or [])],
        neighbor_shift_spread_px=float(neighbor_shift_spread_px),
        prior_deviation_px=delta_prior_norm,
        signal_coverage=float(signal_coverage),
        moving_edge_density=float(moving_edge_density),
        fixed_edge_density=float(fixed_edge_density),
        edge_density_mean=float(edge_density_mean),
        density_regime=str(density_regime),
        profile_candidates=[str(v) for v in profile_candidates],
        coarse_search_profiles=[str(v) for v in coarse_search_profiles],
        refine_profiles=[str(v) for v in refine_profiles],
        refine_objective=str(refine_objective or "cc"),
        registration_profile=str(final_eval["registration_profile"] or STEP7_REGISTRATION_INPUT_PROFILE),
        eval_bbox_yxyx=[int(v) for v in ctx["eval_bbox"]],
        tight_bbox_local_yxyx=[int(v) for v in ctx["tight_bbox"]],
        metric_timing_total=float(pred_eval["timing"].get("total", 0.0)),
        moving=moving_panel,
        fixed=fixed_panel,
        overlay=overlay_panel,
        heatmap=shifted_panel,
    )


def _screen_seed_score(current_cc: float, best_shift_cc: float, dx: int, dy: int) -> float:
    cc = float(current_cc) if np.isfinite(current_cc) else -1.0
    best_cc = float(best_shift_cc) if np.isfinite(best_shift_cc) else cc
    improvement = max(0.0, best_cc - cc)
    shift_norm = float(np.hypot(float(dx), float(dy)))
    return cc + 0.20 * improvement - 0.005 * shift_norm


def _auto_scale_values(center_scale: float, *, half_range: float, step: float) -> list[float]:
    half = max(0.0, float(half_range))
    step_value = max(1e-4, float(step))
    lo = max(0.1, float(center_scale) - half)
    hi = max(lo, float(center_scale) + half)
    values: list[float] = []
    current = float(lo)
    while current <= hi + 1e-9:
        rounded = round(float(current), 5)
        if not values or abs(values[-1] - rounded) > 1e-6:
            values.append(float(rounded))
        current += step_value
    center_rounded = round(float(center_scale), 5)
    if all(abs(v - center_rounded) > 1e-6 for v in values):
        values.append(float(center_rounded))
        values.sort()
    return values or [float(center_rounded)]


def _auto_scale_unique_sorted_values(
    values: list[float] | tuple[float, ...],
    *,
    lo: float | None = None,
    hi: float | None = None,
) -> list[float]:
    ordered: list[float] = []
    for raw_value in sorted(float(v) for v in list(values or [])):
        if not np.isfinite(raw_value):
            continue
        if lo is not None and raw_value < float(lo) - 1e-9:
            continue
        if hi is not None and raw_value > float(hi) + 1e-9:
            continue
        rounded = round(float(raw_value), 5)
        if not ordered or abs(ordered[-1] - rounded) > 1e-6:
            ordered.append(float(rounded))
    return ordered


def _auto_scale_coarse_values(center_scale: float, *, half_range: float, coarse_count: int) -> list[float]:
    half = max(0.0, float(half_range))
    lo = max(0.1, float(center_scale) - half)
    hi = max(lo, float(center_scale) + half)
    count = max(1, int(coarse_count))
    if count <= 1 or hi <= lo + 1e-9:
        return [round(float(center_scale), 5)]
    grid = np.linspace(lo, hi, num=count, dtype=np.float64).tolist()
    return _auto_scale_unique_sorted_values(grid + [float(center_scale)], lo=lo, hi=hi)


def _prepare_tile_eval_static(
    moving_reg_projection_u8: np.ndarray,
    moving_signal_mask_u8: np.ndarray,
    tile: dict[str, Any],
) -> dict[str, Any]:
    x0, y0, x1, y1 = [int(v) for v in tile["scaled_bbox_xyxy"]]
    tile_img_u8 = np.asarray(moving_reg_projection_u8[y0:y1, x0:x1], dtype=np.uint8)
    tile_mask_u8 = np.asarray(moving_signal_mask_u8[y0:y1, x0:x1], dtype=np.uint8)
    return {
        "tile_index": int(tile.get("tile_index", -1)),
        "x0": int(x0),
        "y0": int(y0),
        "tile_img_u8": tile_img_u8,
        "tile_mask_u8": tile_mask_u8,
        "tile_local_gray": tile_img_u8.astype(np.float32) / 255.0,
        "tile_local_mask": (tile_mask_u8 > 0).astype(np.float32),
    }


def _select_auto_scale_sample_tiles(
    tile_defs: list[dict[str, Any]],
    *,
    max_tiles: int,
    strategy: str = "rowwise_uniform",
) -> list[dict[str, Any]]:
    ordered = sorted(
        [dict(tile) for tile in list(tile_defs or [])],
        key=lambda tile: (int(tile.get("row_display", 0)), int(tile.get("col_display", 0)), int(tile.get("tile_index", -1))),
    )
    limit = max(0, int(max_tiles))
    if limit <= 0 or len(ordered) <= limit:
        return ordered
    strategy_key = str(strategy or "rowwise_uniform").strip().lower()
    if strategy_key != "rowwise_uniform":
        return ordered[:limit]

    by_row: dict[int, list[dict[str, Any]]] = {}
    all_cols: list[int] = []
    for tile in ordered:
        row = int(tile.get("row_display", 0))
        by_row.setdefault(row, []).append(tile)
        all_cols.append(int(tile.get("col_display", 0)))
    row_keys = sorted(by_row.keys())
    if not row_keys:
        return ordered[:limit]

    selected_row_count = min(len(row_keys), limit)
    if selected_row_count == len(row_keys):
        selected_rows = list(row_keys)
    elif selected_row_count == 1:
        selected_rows = [row_keys[len(row_keys) // 2]]
    else:
        row_positions = np.linspace(0, len(row_keys) - 1, selected_row_count)
        selected_rows = []
        for pos in row_positions:
            idx = int(round(float(pos)))
            row_value = int(row_keys[max(0, min(len(row_keys) - 1, idx))])
            if row_value not in selected_rows:
                selected_rows.append(row_value)
        if len(selected_rows) < selected_row_count:
            for row_value in row_keys:
                if row_value not in selected_rows:
                    selected_rows.append(int(row_value))
                if len(selected_rows) >= selected_row_count:
                    break
    min_col = min(all_cols)
    max_col = max(all_cols)
    span = float(max_col - min_col)
    chosen: list[dict[str, Any]] = []
    used_tile_indices: set[int] = set()
    for order_idx, row_value in enumerate(selected_rows):
        row_tiles = sorted(by_row.get(int(row_value), []), key=lambda tile: int(tile.get("col_display", 0)))
        if not row_tiles:
            continue
        if len(selected_rows) <= 1 or span <= 0.0:
            target_col = float(np.mean([int(tile.get("col_display", 0)) for tile in row_tiles]))
        else:
            target_col = float(min_col) + span * (float(order_idx) / float(max(1, len(selected_rows) - 1)))
        best_tile = min(
            row_tiles,
            key=lambda tile: (
                abs(float(tile.get("col_display", 0)) - target_col),
                int(tile.get("col_display", 0)),
            ),
        )
        tile_index = int(best_tile.get("tile_index", -1))
        if tile_index not in used_tile_indices:
            chosen.append(best_tile)
            used_tile_indices.add(tile_index)
    if len(chosen) < limit:
        for tile in ordered:
            tile_index = int(tile.get("tile_index", -1))
            if tile_index in used_tile_indices:
                continue
            chosen.append(tile)
            used_tile_indices.add(tile_index)
            if len(chosen) >= limit:
                break
    return sorted(chosen[:limit], key=lambda tile: (int(tile.get("row_display", 0)), int(tile.get("col_display", 0))))


def _auto_scale_aggregate(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    if not rows:
        return {
            "mean_current_cc": float("nan"),
            "mean_final_cc": float("nan"),
            "mean_shift_gain_cc": float("nan"),
            "mean_abs_dx": float("nan"),
            "mean_abs_dy": float("nan"),
            "rightmost_abs_dx": float("nan"),
            "rightmost_mean_final_cc": float("nan"),
            "composite_score": float("nan"),
        }
    final_ccs = np.asarray([float(row.get("final_cc", float("nan"))) for row in rows], dtype=np.float64)
    current_ccs = np.asarray([float(row.get("current_cc", float("nan"))) for row in rows], dtype=np.float64)
    shift_gains = np.asarray([float(row.get("shift_gain_cc", float("nan"))) for row in rows], dtype=np.float64)
    dxs = np.asarray([abs(float(row.get("final_dx_px", 0.0))) for row in rows], dtype=np.float64)
    dys = np.asarray([abs(float(row.get("final_dy_px", 0.0))) for row in rows], dtype=np.float64)
    max_col = max(int(row.get("col_display", 0)) for row in rows)
    rightmost = [row for row in rows if int(row.get("col_display", 0)) == max_col]
    rightmost_abs_dx = float(np.nanmean([abs(float(row.get("final_dx_px", 0.0))) for row in rightmost])) if rightmost else float("nan")
    rightmost_mean_final_cc = float(np.nanmean([float(row.get("final_cc", float("nan"))) for row in rightmost])) if rightmost else float("nan")
    mean_final_cc = float(np.nanmean(final_ccs))
    mean_abs_dx = float(np.nanmean(dxs))
    composite = mean_final_cc - 0.01 * mean_abs_dx - 0.01 * (rightmost_abs_dx if np.isfinite(rightmost_abs_dx) else 0.0)
    return {
        "mean_current_cc": float(np.nanmean(current_ccs)),
        "mean_final_cc": mean_final_cc,
        "mean_shift_gain_cc": float(np.nanmean(shift_gains)),
        "mean_abs_dx": mean_abs_dx,
        "mean_abs_dy": float(np.nanmean(dys)),
        "rightmost_abs_dx": rightmost_abs_dx,
        "rightmost_mean_final_cc": rightmost_mean_final_cc,
        "composite_score": float(composite),
    }


def _auto_scale_tile_metric_row(scale_value: float, row: dict[str, Any]) -> dict[str, Any]:
    return {
        "scale": float(scale_value),
        "stage_name": str(row.get("stage_name") or ""),
        "scale_status": str(row.get("scale_status") or ""),
        "tile_index": int(row.get("tile_index", -1)),
        "label": str(row.get("label") or ""),
        "row_display": int(row.get("row_display", 0)),
        "col_display": int(row.get("col_display", 0)),
        "current_cc": float(row.get("current_cc", float("nan"))),
        "final_cc": float(row.get("final_cc", float("nan"))),
        "shift_gain_cc": float(row.get("shift_gain_cc", float("nan"))),
        "final_dx_px": float(row.get("final_dx_px", 0.0)),
        "final_dy_px": float(row.get("final_dy_px", 0.0)),
        "template_match_score": float(row.get("template_match_score", float("nan"))),
        "signal_coverage": float(row.get("signal_coverage", float("nan"))),
        "density_regime": str(row.get("density_regime") or ""),
        "registration_profile": str(row.get("registration_profile") or ""),
        "refine_objective": str(row.get("refine_objective") or "cc"),
    }


def _write_auto_scale_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = [
        "rank",
        "stage_name",
        "scale",
        "scale_status",
        "pruned",
        "prune_reason",
        "tile_count_evaluated",
        "tile_count_expected",
        "tile_count_skipped",
        "composite_score",
        "mean_current_cc",
        "mean_final_cc",
        "mean_shift_gain_cc",
        "mean_abs_dx",
        "mean_abs_dy",
        "rightmost_abs_dx",
        "rightmost_mean_final_cc",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def _auto_scale_summary_sort_key(row: dict[str, Any]) -> tuple[int, float, float, float]:
    expected = max(1, int(row.get("tile_count_expected") or 0))
    evaluated = int(row.get("tile_count_evaluated") or 0)
    complete_flag = 1 if evaluated >= expected else 0
    composite = float(row.get("composite_score", float("-inf")))
    if not np.isfinite(composite):
        composite = float("-inf")
    mean_final = float(row.get("mean_final_cc", float("-inf")))
    if not np.isfinite(mean_final):
        mean_final = float("-inf")
    mean_gain = float(row.get("mean_shift_gain_cc", float("-inf")))
    if not np.isfinite(mean_gain):
        mean_gain = float("-inf")
    return complete_flag, composite, mean_final, mean_gain


def _select_auto_scale_best_rows(summary_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any], dict[str, Any]]:
    ranked_rows = sorted(summary_rows, key=_auto_scale_summary_sort_key, reverse=True)
    complete_rows = [
        row
        for row in ranked_rows
        if int(row.get("tile_count_evaluated") or 0) >= max(1, int(row.get("tile_count_expected") or 0))
    ]
    candidates = complete_rows or ranked_rows
    if not candidates:
        return [], {}, {}, {}
    best_composite = dict(candidates[0])
    best_final = dict(max(candidates, key=lambda row: float(row.get("mean_final_cc", float("-inf")))))
    best_right = dict(
        min(
            candidates,
            key=lambda row: abs(float(row.get("rightmost_abs_dx", float("inf"))))
            if np.isfinite(float(row.get("rightmost_abs_dx", float("inf"))))
            else float("inf"),
        )
    )
    return ranked_rows, best_composite, best_final, best_right


def _auto_scale_should_prune(
    current_rows: list[dict[str, Any]],
    *,
    current_state_key: int,
    stage_states: dict[int, dict[str, Any]],
    total_tiles: int,
    min_tiles: int,
) -> tuple[bool, str]:
    if len(current_rows) < max(1, int(min_tiles)) or len(current_rows) >= max(1, int(total_tiles)):
        return False, ""
    current_agg = _auto_scale_aggregate(current_rows)
    current_composite = float(current_agg.get("composite_score", float("nan")))
    current_mean_final = float(current_agg.get("mean_final_cc", float("nan")))
    current_mean_gain = float(current_agg.get("mean_shift_gain_cc", float("nan")))
    if not np.isfinite(current_mean_final):
        return True, "nonfinite_mean_final_cc"

    best_complete_composite = float("-inf")
    best_complete_mean_final = float("-inf")
    best_partial_composite = float("-inf")
    best_partial_mean_final = float("-inf")
    for state_key, state in stage_states.items():
        if int(state_key) == int(current_state_key):
            continue
        state_rows = [row for row in list(state.get("ordered_rows") or []) if isinstance(row, dict)]
        if len(state_rows) < max(1, int(min_tiles)):
            continue
        state_agg = _auto_scale_aggregate(state_rows)
        state_composite = float(state_agg.get("composite_score", float("nan")))
        state_mean_final = float(state_agg.get("mean_final_cc", float("nan")))
        if np.isfinite(state_composite):
            if state_composite > best_partial_composite:
                best_partial_composite = float(state_composite)
                best_partial_mean_final = float(state_mean_final)
            if len(state_rows) >= max(1, int(total_tiles)) and state_composite > best_complete_composite:
                best_complete_composite = float(state_composite)
                best_complete_mean_final = float(state_mean_final)

    if (
        np.isfinite(current_composite)
        and np.isfinite(best_complete_composite)
        and current_composite < best_complete_composite - float(STEP7_AUTO_SCALE_PRUNE_COMPOSITE_MARGIN)
        and current_mean_final < best_complete_mean_final - float(STEP7_AUTO_SCALE_PRUNE_MEAN_FINAL_CC_MARGIN)
    ):
        return True, "behind_best_complete"
    if (
        np.isfinite(current_composite)
        and np.isfinite(best_partial_composite)
        and current_composite < best_partial_composite - float(STEP7_AUTO_SCALE_PRUNE_COMPOSITE_MARGIN * 1.25)
        and current_mean_final < best_partial_mean_final - float(STEP7_AUTO_SCALE_PRUNE_MEAN_FINAL_CC_MARGIN)
        and (not np.isfinite(current_mean_gain) or current_mean_gain <= 0.0)
    ):
        return True, "behind_best_partial"
    if current_mean_final < 0.02 and (not np.isfinite(current_mean_gain) or current_mean_gain <= 0.0):
        return True, "low_cc_no_gain"
    return False, ""


def _select_auto_scale_refine_values(
    coarse_rows: list[dict[str, Any]],
    *,
    center_scale: float,
    half_range: float,
    fine_step: float,
    coarse_values: list[float],
    top_k: int,
    already_evaluated: list[float],
) -> list[float]:
    ranked_coarse, _, _, _ = _select_auto_scale_best_rows(coarse_rows)
    selected_centers = [
        float(row.get("scale", float("nan")))
        for row in ranked_coarse[: max(1, int(top_k))]
        if np.isfinite(float(row.get("scale", float("nan"))))
    ]
    if not selected_centers:
        selected_centers = [float(center_scale)]
    coarse_sorted = _auto_scale_unique_sorted_values(list(coarse_values or []))
    diffs = [
        abs(float(coarse_sorted[idx + 1]) - float(coarse_sorted[idx]))
        for idx in range(max(0, len(coarse_sorted) - 1))
        if abs(float(coarse_sorted[idx + 1]) - float(coarse_sorted[idx])) > 1e-6
    ]
    coarse_window = max(float(fine_step), min(diffs) if diffs else float(half_range))
    lo = max(0.1, float(center_scale) - max(0.0, float(half_range)))
    hi = max(lo, float(center_scale) + max(0.0, float(half_range)))
    evaluated = _auto_scale_unique_sorted_values(list(already_evaluated or []), lo=lo, hi=hi)
    refine_values: list[float] = []
    for center in selected_centers:
        refine_values.extend(_auto_scale_values(float(center), half_range=float(coarse_window), step=float(fine_step)))
    refine_values = _auto_scale_unique_sorted_values(refine_values, lo=lo, hi=hi)
    return [value for value in refine_values if all(abs(float(value) - float(done)) > 1e-6 for done in evaluated)]


def _tile_neighbor_map(tile_defs: list[dict[str, Any]]) -> dict[int, list[int]]:
    by_rc: dict[tuple[int, int], int] = {}
    for tile in tile_defs:
        by_rc[(int(tile.get("row_display", 0)), int(tile.get("col_display", 0)))] = int(tile.get("tile_index", -1))
    neighbors: dict[int, list[int]] = {}
    for tile in tile_defs:
        idx = int(tile.get("tile_index", -1))
        row = int(tile.get("row_display", 0))
        col = int(tile.get("col_display", 0))
        tile_neighbors: list[int] = []
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            hit = by_rc.get((row + dr, col + dc))
            if hit is not None and hit >= 0:
                tile_neighbors.append(int(hit))
        neighbors[idx] = tile_neighbors
    return neighbors


def _tile_defs_by_index(tile_defs: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    return {int(tile.get("tile_index", -1)): tile for tile in tile_defs}


def _evaluate_tile_alignment_with_prior(
    *,
    fixed_gray_full: np.ndarray,
    fixed_mask_full: np.ndarray,
    moving_reg_projection_u8: np.ndarray,
    moving_signal_mask_u8: np.ndarray,
    fixed_shape_hw: tuple[int, int],
    manual_mat: np.ndarray,
    tile: dict[str, Any],
    registration_input_profile: str,
    search_radius_px: int,
    prior_shift_dx_px: int = 0,
    prior_shift_dy_px: int = 0,
    neighbor_count: int = 0,
    neighbor_labels: list[str] | None = None,
    neighbor_shift_spread_px: float = 0.0,
    local_refine_radius_px: int | None = None,
    tile_static: dict[str, Any] | None = None,
    progress_cb: Step7ProgressCallback | None = None,
    progress_mode: str = "step7",
    progress_item_index: int | None = None,
    progress_total_items: int | None = None,
    progress_base_percent: float = 0.0,
    progress_span_percent: float = 100.0,
) -> TileResultRow:
    prior_dx = int(prior_shift_dx_px)
    prior_dy = int(prior_shift_dy_px)
    margin_px = max(24, int(search_radius_px) + max(abs(prior_dx), abs(prior_dy)) + 8)
    ctx = _build_tile_eval_context(
        fixed_gray_full=fixed_gray_full,
        fixed_mask_full=fixed_mask_full,
        moving_reg_projection_u8=moving_reg_projection_u8,
        moving_signal_mask_u8=moving_signal_mask_u8,
        fixed_shape_hw=fixed_shape_hw,
        manual_mat=manual_mat,
        tile=tile,
        margin_px=margin_px,
        tile_static=tile_static,
    )

    tile_mat = np.asarray(ctx["warped"]["tile_mat"], dtype=np.float32).copy()
    tile_mat[0, 2] += float(prior_dx)
    tile_mat[1, 2] += float(prior_dy)
    fixed_prior_local, fixed_prior_mask = _backwarp_fixed_to_tile_local(
        fixed_gray_full,
        fixed_mask_full,
        tile_mat,
        np.asarray(ctx["tile_img_u8"], dtype=np.uint8).shape[:2],
    )
    tile_local_mask = np.asarray(ctx["tile_local_mask"], dtype=np.float32)
    signal_coverage = float(np.mean(tile_local_mask > 0))
    moving_edge_density = _edge_density_from_gray(np.asarray(ctx["tile_local_gray"], dtype=np.float32), tile_local_mask)
    fixed_edge_density = _edge_density_from_gray(fixed_prior_local, fixed_prior_mask)
    density_regime, edge_density_mean, coarse_search_profiles, refine_profiles, refine_objective = _density_aware_profile_policy(
        preferred_profile=str(registration_input_profile or STEP7_REGISTRATION_INPUT_PROFILE),
        signal_coverage=signal_coverage,
        moving_edge_density=moving_edge_density,
        fixed_edge_density=fixed_edge_density,
    )
    profile_candidates = _merge_profile_sequences(coarse_search_profiles, refine_profiles)
    tile_label = str(tile.get("label") or f"T{int(tile.get('tile_index', -1)):02d}")
    estimated_total_eval_steps = max(
        1,
        int(len(coarse_search_profiles) + len(_merge_profile_sequences(coarse_search_profiles[:1], refine_profiles))),
    )
    refine_radius_px = max(
        0,
        int(STEP7_REFINE_LOCAL_RADIUS_PX if local_refine_radius_px is None else local_refine_radius_px),
    )

    def _accepts_proposal(
        *,
        current_cc: float,
        proposal_cc: float,
        current_objective_score: float,
        proposal_objective_score: float,
    ) -> bool:
        cc_floor = 0.015 if str(refine_objective or "cc").strip().lower() == "hybrid" else 0.0
        return bool(
            np.isfinite(proposal_objective_score)
            and (
                not np.isfinite(current_objective_score)
                or proposal_objective_score >= current_objective_score
            )
            and (
                not np.isfinite(proposal_cc)
                or not np.isfinite(current_cc)
                or proposal_cc + cc_floor >= current_cc
            )
        )

    def _should_skip_refine_after_coarse(candidate: dict[str, Any]) -> tuple[bool, str]:
        template_cc = float(candidate.get("best_template_cc", float("nan")))
        current_cc = float(candidate.get("current_cc", float("nan")))
        proposal_cc = float(candidate.get("proposal_cc", float("nan")))
        template_bad = (not np.isfinite(template_cc)) or (template_cc < float(STEP7_COARSE_EARLY_EXIT_TEMPLATE_CC_MIN))
        if np.isfinite(current_cc) and np.isfinite(proposal_cc):
            coarse_gain_cc = float(proposal_cc - current_cc)
            gain_too_small = coarse_gain_cc < float(STEP7_COARSE_EARLY_EXIT_CC_GAIN_MIN)
        elif not np.isfinite(proposal_cc):
            gain_too_small = True
        else:
            gain_too_small = False
        if not template_bad and not gain_too_small:
            return False, ""
        reasons: list[str] = []
        if template_bad:
            reasons.append("template")
        if gain_too_small:
            reasons.append("cc_gain")
        return True, "coarse_early_exit_" + "_".join(reasons)

    best_coarse_candidate: dict[str, Any] | None = None
    for coarse_idx, coarse_profile in enumerate(coarse_search_profiles, start=1):
        _emit_step7_progress(
            progress_cb,
            {
                "mode": str(progress_mode),
                "stage": "coarse_eval",
                "tile_index": int(tile.get("tile_index", -1)),
                "tile_label": tile_label,
                "item_index": int(progress_item_index) if progress_item_index is not None else None,
                "total_items": int(progress_total_items) if progress_total_items is not None else None,
                "candidate_index": int(coarse_idx),
                "candidate_count": int(len(coarse_search_profiles)),
                "coarse_profile": str(coarse_profile),
                "refine_profile": "",
                "density_regime": str(density_regime),
                "progress_percent": _step7_progress_percent(
                    progress_base_percent,
                    progress_span_percent,
                    float(coarse_idx - 1) / float(max(1, estimated_total_eval_steps)),
                ),
                "message": (
                    f"{str(progress_mode)} | tile {int(progress_item_index) if progress_item_index is not None else '?'}"
                    f"/{int(progress_total_items) if progress_total_items is not None else '?'} "
                    f"{tile_label}"
                    f" | coarse {int(coarse_idx)}/{int(len(coarse_search_profiles))}"
                    f" | profile={str(coarse_profile)}"
                ),
            },
        )
        best_template_cc, coarse_dx, coarse_dy = _template_match_shift_for_profile(
            ctx=ctx,
            registration_input_profile=coarse_profile,
            prior_dx=prior_dx,
            prior_dy=prior_dy,
            search_radius_px=int(search_radius_px),
        )
        pred_eval = _evaluate_tile_shift_from_context(
            ctx=ctx,
            fixed_gray_full=fixed_gray_full,
            fixed_mask_full=fixed_mask_full,
            registration_input_profile=coarse_profile,
            dx_px=prior_dx,
            dy_px=prior_dy,
        )
        coarse_eval = _evaluate_tile_shift_from_context(
            ctx=ctx,
            fixed_gray_full=fixed_gray_full,
            fixed_mask_full=fixed_mask_full,
            registration_input_profile=coarse_profile,
            dx_px=int(coarse_dx),
            dy_px=int(coarse_dy),
        )
        current_cc = float(pred_eval["metrics"].get("cc", float("nan")))
        proposal_cc = float(coarse_eval["metrics"].get("cc", float("nan")))
        current_objective_score = _objective_score_from_metrics(
            pred_eval.get("metrics"),
            objective_name=str(refine_objective or "cc"),
        )
        proposal_objective_score = _objective_score_from_metrics(
            coarse_eval.get("metrics"),
            objective_name=str(refine_objective or "cc"),
        )
        accept_proposal = _accepts_proposal(
            current_cc=current_cc,
            proposal_cc=proposal_cc,
            current_objective_score=float(current_objective_score),
            proposal_objective_score=float(proposal_objective_score),
        )
        final_eval = coarse_eval if accept_proposal else pred_eval
        final_dx = int(coarse_dx if accept_proposal else prior_dx)
        final_dy = int(coarse_dy if accept_proposal else prior_dy)
        final_cc = float(proposal_cc if accept_proposal else current_cc)
        final_objective_score = float(proposal_objective_score if accept_proposal else current_objective_score)
        candidate = {
            "score": _profile_candidate_score(
                current_cc=current_cc,
                final_cc=final_cc,
                current_objective_score=current_objective_score,
                final_objective_score=final_objective_score,
                prior_dx=prior_dx,
                prior_dy=prior_dy,
                final_dx=final_dx,
                final_dy=final_dy,
                template_match_score=best_template_cc,
                accepted=accept_proposal,
                profile_key=coarse_profile,
                density_regime=density_regime,
                objective_name=str(refine_objective or "cc"),
            ),
            "pred_eval": pred_eval,
            "meas_eval": coarse_eval,
            "final_eval": final_eval,
            "current_cc": current_cc,
            "proposal_cc": proposal_cc,
            "final_cc": final_cc,
            "current_objective_score": float(current_objective_score),
            "proposal_objective_score": float(proposal_objective_score),
            "final_objective_score": float(final_objective_score),
            "best_dx": int(coarse_dx),
            "best_dy": int(coarse_dy),
            "final_dx": int(final_dx),
            "final_dy": int(final_dy),
            "best_template_cc": float(best_template_cc),
            "accept_proposal": bool(accept_proposal),
            "proposal_gate": "accepted" if accept_proposal else ("kept_prior" if abs(prior_dx) > 0 or abs(prior_dy) > 0 else "kept_current"),
            "coarse_profile": str(coarse_profile),
            "refine_profile": str(coarse_profile),
            "center_dx": int(coarse_dx),
            "center_dy": int(coarse_dy),
        }
        if best_coarse_candidate is None:
            best_coarse_candidate = candidate
            continue
        if float(candidate["score"]) > float(best_coarse_candidate["score"]) + 1e-6:
            best_coarse_candidate = candidate
            continue
        if abs(float(candidate["score"]) - float(best_coarse_candidate["score"])) <= 1e-6:
            cand_final = float(candidate["final_cc"])
            best_final = float(best_coarse_candidate["final_cc"])
            if (np.isfinite(cand_final) and not np.isfinite(best_final)) or cand_final > best_final:
                best_coarse_candidate = candidate

    best_candidate: dict[str, Any] | None = None
    if best_coarse_candidate is not None:
        best_coarse_profile = str(best_coarse_candidate["coarse_profile"])
        early_exit, early_exit_gate = _should_skip_refine_after_coarse(best_coarse_candidate)
        if early_exit:
            best_candidate = dict(best_coarse_candidate)
            best_candidate["accept_proposal"] = False
            best_candidate["proposal_gate"] = str(early_exit_gate or "coarse_early_exit")
            best_candidate["final_eval"] = best_candidate["pred_eval"]
            best_candidate["final_dx"] = int(prior_dx)
            best_candidate["final_dy"] = int(prior_dy)
            best_candidate["final_cc"] = float(best_candidate["current_cc"])
            best_candidate["final_objective_score"] = float(best_candidate["current_objective_score"])
        else:
            refine_candidates = _merge_profile_sequences([best_coarse_profile], refine_profiles)
            refine_center_dx = int(best_coarse_candidate.get("center_dx", prior_dx))
            refine_center_dy = int(best_coarse_candidate.get("center_dy", prior_dy))
            for refine_idx, refine_profile in enumerate(refine_candidates, start=1):
                eval_step = int(len(coarse_search_profiles) + refine_idx)
                _emit_step7_progress(
                    progress_cb,
                    {
                        "mode": str(progress_mode),
                        "stage": "refine_eval",
                        "tile_index": int(tile.get("tile_index", -1)),
                        "tile_label": tile_label,
                        "item_index": int(progress_item_index) if progress_item_index is not None else None,
                        "total_items": int(progress_total_items) if progress_total_items is not None else None,
                        "candidate_index": int(refine_idx),
                        "candidate_count": int(len(refine_candidates)),
                        "coarse_profile": best_coarse_profile,
                        "refine_profile": str(refine_profile),
                        "density_regime": str(density_regime),
                        "progress_percent": _step7_progress_percent(
                            progress_base_percent,
                            progress_span_percent,
                            float(eval_step - 1) / float(max(1, estimated_total_eval_steps)),
                        ),
                        "message": (
                            f"{str(progress_mode)} | tile {int(progress_item_index) if progress_item_index is not None else '?'}"
                            f"/{int(progress_total_items) if progress_total_items is not None else '?'} "
                            f"{tile_label}"
                            f" | refine {int(refine_idx)}/{int(len(refine_candidates))}"
                            f" | coarse={best_coarse_profile} refine={str(refine_profile)}"
                        ),
                    },
                )
                pred_eval = _evaluate_tile_shift_from_context(
                    ctx=ctx,
                    fixed_gray_full=fixed_gray_full,
                    fixed_mask_full=fixed_mask_full,
                    registration_input_profile=refine_profile,
                    dx_px=prior_dx,
                    dy_px=prior_dy,
                )
                refine_best = _local_refine_shift_for_profile(
                    ctx=ctx,
                    fixed_gray_full=fixed_gray_full,
                    fixed_mask_full=fixed_mask_full,
                    registration_input_profile=refine_profile,
                    center_dx=int(refine_center_dx),
                    center_dy=int(refine_center_dy),
                    radius_px=int(refine_radius_px),
                    objective_name=str(refine_objective or "cc"),
                )
                meas_eval = dict(refine_best["eval"])
                current_cc = float(pred_eval["metrics"].get("cc", float("nan")))
                proposal_cc = float(meas_eval["metrics"].get("cc", float("nan")))
                current_objective_score = _objective_score_from_metrics(
                    pred_eval.get("metrics"),
                    objective_name=str(refine_objective or "cc"),
                )
                proposal_objective_score = float(refine_best["objective_score"])
                accept_proposal = _accepts_proposal(
                    current_cc=current_cc,
                    proposal_cc=proposal_cc,
                    current_objective_score=float(current_objective_score),
                    proposal_objective_score=float(proposal_objective_score),
                )
                final_eval = meas_eval if accept_proposal else pred_eval
                final_dx = int(refine_best["dx"] if accept_proposal else prior_dx)
                final_dy = int(refine_best["dy"] if accept_proposal else prior_dy)
                final_cc = float(proposal_cc if accept_proposal else current_cc)
                final_objective_score = float(proposal_objective_score if accept_proposal else current_objective_score)
                candidate = {
                    "score": _profile_candidate_score(
                        current_cc=current_cc,
                        final_cc=final_cc,
                        current_objective_score=current_objective_score,
                        final_objective_score=final_objective_score,
                        prior_dx=prior_dx,
                        prior_dy=prior_dy,
                        final_dx=final_dx,
                        final_dy=final_dy,
                        template_match_score=float(best_coarse_candidate.get("best_template_cc", float("nan"))),
                        accepted=accept_proposal,
                        profile_key=refine_profile,
                        density_regime=density_regime,
                        objective_name=str(refine_objective or "cc"),
                    ),
                    "pred_eval": pred_eval,
                    "meas_eval": meas_eval,
                    "final_eval": final_eval,
                    "current_cc": current_cc,
                    "proposal_cc": proposal_cc,
                    "final_cc": final_cc,
                    "current_objective_score": float(current_objective_score),
                    "proposal_objective_score": float(proposal_objective_score),
                    "final_objective_score": float(final_objective_score),
                    "best_dx": int(refine_best["dx"]),
                    "best_dy": int(refine_best["dy"]),
                    "final_dx": int(final_dx),
                    "final_dy": int(final_dy),
                    "best_template_cc": float(best_coarse_candidate.get("best_template_cc", float("nan"))),
                    "accept_proposal": bool(accept_proposal),
                    "proposal_gate": "accepted" if accept_proposal else ("kept_prior" if abs(prior_dx) > 0 or abs(prior_dy) > 0 else "kept_current"),
                    "coarse_profile": best_coarse_profile,
                    "refine_profile": str(refine_profile),
                }
                if best_candidate is None:
                    best_candidate = candidate
                    continue
                if float(candidate["score"]) > float(best_candidate["score"]) + 1e-6:
                    best_candidate = candidate
                    continue
                if abs(float(candidate["score"]) - float(best_candidate["score"])) <= 1e-6:
                    cand_final = float(candidate["final_cc"])
                    best_final = float(best_candidate["final_cc"])
                    if (np.isfinite(cand_final) and not np.isfinite(best_final)) or cand_final > best_final:
                        best_candidate = candidate

    if best_candidate is None:
        pred_eval = _evaluate_tile_shift_from_context(
            ctx=ctx,
            fixed_gray_full=fixed_gray_full,
            fixed_mask_full=fixed_mask_full,
            registration_input_profile=str(registration_input_profile or STEP7_REGISTRATION_INPUT_PROFILE),
            dx_px=prior_dx,
            dy_px=prior_dy,
        )
        best_candidate = {
            "pred_eval": pred_eval,
            "meas_eval": pred_eval,
            "final_eval": pred_eval,
            "current_cc": float(pred_eval["metrics"].get("cc", float("nan"))),
            "proposal_cc": float(pred_eval["metrics"].get("cc", float("nan"))),
            "final_cc": float(pred_eval["metrics"].get("cc", float("nan"))),
            "best_dx": int(prior_dx),
            "best_dy": int(prior_dy),
            "final_dx": int(prior_dx),
            "final_dy": int(prior_dy),
            "best_template_cc": float("nan"),
            "accept_proposal": False,
            "proposal_gate": "kept_prior" if abs(prior_dx) > 0 or abs(prior_dy) > 0 else "kept_current",
        }

    return _build_tile_result_row_from_evals(
        tile=tile,
        ctx=ctx,
        pred_eval=best_candidate["pred_eval"],
        meas_eval=best_candidate["meas_eval"],
        final_eval=best_candidate["final_eval"],
        template_match_score=float(best_candidate["best_template_cc"]),
        neighbor_count=int(neighbor_count),
        neighbor_labels=neighbor_labels,
        neighbor_shift_spread_px=float(neighbor_shift_spread_px),
        tile_state=TileState.FRONTIER.value if bool(best_candidate["accept_proposal"]) else TileState.HOLD.value,
        proposal_gate=str(best_candidate["proposal_gate"]),
        signal_coverage=float(signal_coverage),
        moving_edge_density=float(moving_edge_density),
        fixed_edge_density=float(fixed_edge_density),
        edge_density_mean=float(edge_density_mean),
        density_regime=str(density_regime),
        profile_candidates=list(profile_candidates),
        coarse_search_profiles=list(coarse_search_profiles),
        refine_profiles=list(refine_profiles),
        refine_objective=str(refine_objective or "cc"),
    )


def _frontier_confidence_score(
    *,
    current_cc: float,
    proposal_cc: float,
    prior_dx: int,
    prior_dy: int,
    final_dx: int,
    final_dy: int,
    neighbor_count: int,
    neighbor_shift_spread_px: float,
) -> float:
    base = float(proposal_cc) if np.isfinite(proposal_cc) else -1.0
    gain = max(0.0, float(proposal_cc) - float(current_cc)) if np.isfinite(current_cc) and np.isfinite(proposal_cc) else 0.0
    delta_prior = float(np.hypot(float(final_dx - prior_dx), float(final_dy - prior_dy)))
    return base + 0.30 * gain + 0.02 * float(neighbor_count) - 0.010 * delta_prior - 0.006 * float(neighbor_shift_spread_px)


def _frontier_abs_match_weight(row: TileResultRow) -> float:
    if not np.isfinite(row.meas_cc):
        return 0.0
    gain = 0.0 if not np.isfinite(row.current_cc) else max(0.0, float(row.meas_cc) - float(row.current_cc))
    weight = 1.0 + 0.8 * gain
    if str(row.proposal_gate) != "accepted":
        weight *= 0.10
    return float(weight)


def _gate_frontier_graph_result(
    *,
    base_row: TileResultRow,
    graph_row: TileResultRow,
) -> TileResultRow:
    kept_row = TileResultRow.from_mapping(base_row.to_dict())
    graph_cc = float(graph_row.final_cc)
    local_cc = float(base_row.final_cc)
    current_cc = float(base_row.current_cc)
    graph_move_px = float(
        np.hypot(
            float(graph_row.final_dx_px) - float(base_row.final_dx_px),
            float(graph_row.final_dy_px) - float(base_row.final_dy_px),
        )
    )
    graph_gain_vs_local = (
        float(graph_cc) - float(local_cc)
        if np.isfinite(graph_cc) and np.isfinite(local_cc)
        else float("nan")
    )

    if str(base_row.proposal_gate) != "accepted":
        kept_row.proposal_gate = "graph_gated_local_reject"
        kept_row.tile_state = str(base_row.tile_state)
        return kept_row
    if not np.isfinite(graph_cc):
        kept_row.proposal_gate = "graph_gated_nonfinite"
        kept_row.tile_state = TileState.FRONTIER.value
        return kept_row
    if np.isfinite(local_cc) and graph_cc + float(STEP7_FRONTIER_GRAPH_CC_DROP_TOL) < local_cc:
        kept_row.proposal_gate = "graph_gated_cc_drop_vs_local"
        kept_row.tile_state = TileState.FRONTIER.value
        return kept_row
    if np.isfinite(current_cc) and graph_cc + float(STEP7_FRONTIER_GRAPH_CC_DROP_TOL) < current_cc:
        kept_row.proposal_gate = "graph_gated_cc_drop_vs_current"
        kept_row.tile_state = TileState.FRONTIER.value
        return kept_row
    if (
        graph_move_px > float(STEP7_FRONTIER_GRAPH_LARGE_MOVE_PX)
        and (not np.isfinite(graph_gain_vs_local) or graph_gain_vs_local < float(STEP7_FRONTIER_GRAPH_LARGE_MOVE_GAIN_MIN))
    ):
        kept_row.proposal_gate = "graph_gated_large_move_no_gain"
        kept_row.tile_state = TileState.FRONTIER.value
        return kept_row

    accepted_row = graph_row
    if (
        int(round(float(graph_row.final_dx_px))) != int(round(float(base_row.final_dx_px)))
        or int(round(float(graph_row.final_dy_px))) != int(round(float(base_row.final_dy_px)))
    ):
        accepted_row.proposal_gate = "graph_accepted"
    else:
        accepted_row.proposal_gate = str(base_row.proposal_gate)
    accepted_row.tile_state = TileState.FRONTIER.value
    return accepted_row


def _build_frontier_column_model(
    *,
    frontier_rows: list[TileResultRow],
    solved_rows_by_idx: dict[int, TileResultRow],
) -> dict[int, dict[str, Any]]:
    frontier_by_col: dict[int, list[TileResultRow]] = {}
    solved_by_col: dict[int, list[TileResultRow]] = {}
    for row in frontier_rows:
        frontier_by_col.setdefault(int(row.col_display), []).append(row)
    for row in solved_rows_by_idx.values():
        solved_by_col.setdefault(int(row.col_display), []).append(row)

    column_model: dict[int, dict[str, Any]] = {}
    for col, col_rows in sorted(frontier_by_col.items()):
        solved_col_rows = solved_by_col.get(int(col), [])
        if len(col_rows) < 2 and not solved_col_rows:
            continue
        if solved_col_rows:
            target_dx = float(np.mean([float(row.final_dx_px) for row in solved_col_rows]))
            target_dy = float(np.mean([float(row.final_dy_px) for row in solved_col_rows]))
            source = "solved_same_column"
        else:
            target_dx = float(np.mean([float(row.pred_dx_px) for row in col_rows]))
            target_dy = float(np.mean([float(row.pred_dy_px) for row in col_rows]))
            source = "frontier_pred_mean"
        column_model[int(col)] = {
            "column_index": int(col),
            "target_dx_px": float(target_dx),
            "target_dy_px": float(target_dy),
            "source": str(source),
            "frontier_tile_indices": [int(row.tile_index) for row in col_rows],
            "solved_tile_indices": [int(row.tile_index) for row in solved_col_rows],
        }
    return column_model


def _build_frontier_graph_edges(
    *,
    frontier_rows: list[TileResultRow],
    solved_rows_by_idx: dict[int, TileResultRow],
    neighbor_map: dict[int, list[int]],
) -> tuple[list[GraphEdge], dict[int, dict[str, Any]]]:
    edges: list[GraphEdge] = []
    frontier_idx_set = {int(row.tile_index) for row in frontier_rows}
    column_model = _build_frontier_column_model(
        frontier_rows=frontier_rows,
        solved_rows_by_idx=solved_rows_by_idx,
    )
    for row in frontier_rows:
        edges.append(
            GraphEdge(
                edge_type="abs_fixed",
                src_tile_index=int(row.tile_index),
                target_dx_px=float(row.meas_dx_px),
                target_dy_px=float(row.meas_dy_px),
                weight=_frontier_abs_match_weight(row),
            )
        )
        edges.append(
            GraphEdge(
                edge_type="prior",
                src_tile_index=int(row.tile_index),
                target_dx_px=float(row.pred_dx_px),
                target_dy_px=float(row.pred_dy_px),
                weight=0.22,
            )
        )
        for neighbor_idx in neighbor_map.get(int(row.tile_index), []):
            solved_row = solved_rows_by_idx.get(int(neighbor_idx))
            if solved_row is None:
                continue
            edges.append(
                GraphEdge(
                    edge_type="neighbor_anchor",
                    src_tile_index=int(row.tile_index),
                    dst_tile_index=int(neighbor_idx),
                    target_dx_px=float(solved_row.final_dx_px),
                    target_dy_px=float(solved_row.final_dy_px),
                    weight=0.18,
                    meta={"neighbor_label": solved_row.label},
                )
            )
        column_info = column_model.get(int(row.col_display))
        if column_info is not None:
            shared_weight = float(STEP7_FRONTIER_COLUMN_SHARED_WEIGHT)
            if len(list(column_info.get("frontier_tile_indices") or [])) <= 1:
                shared_weight *= 0.75
            edges.append(
                GraphEdge(
                    edge_type="column_shared",
                    src_tile_index=int(row.tile_index),
                    src_column_index=int(row.col_display),
                    target_dx_px=0.0,
                    target_dy_px=0.0,
                    weight=float(shared_weight),
                    meta={
                        "column_source": str(column_info.get("source") or ""),
                        "column_target_dx_px": float(column_info.get("target_dx_px", 0.0)),
                        "column_target_dy_px": float(column_info.get("target_dy_px", 0.0)),
                    },
                )
            )
    for col, info in sorted(column_model.items()):
        edges.append(
            GraphEdge(
                edge_type="column_prior",
                src_column_index=int(col),
                target_dx_px=float(info.get("target_dx_px", 0.0)),
                target_dy_px=float(info.get("target_dy_px", 0.0)),
                weight=float(STEP7_FRONTIER_COLUMN_PRIOR_WEIGHT),
                meta={"source": str(info.get("source") or "")},
            )
        )
    active_cols = sorted(column_model)
    for src_col, dst_col in zip(active_cols[:-1], active_cols[1:]):
        src_info = column_model[int(src_col)]
        dst_info = column_model[int(dst_col)]
        edges.append(
            GraphEdge(
                edge_type="column_rel",
                src_column_index=int(src_col),
                dst_column_index=int(dst_col),
                target_dx_px=float(dst_info.get("target_dx_px", 0.0)) - float(src_info.get("target_dx_px", 0.0)),
                target_dy_px=float(dst_info.get("target_dy_px", 0.0)) - float(src_info.get("target_dy_px", 0.0)),
                weight=float(STEP7_FRONTIER_COLUMN_REL_WEIGHT),
            )
        )
    seen_pairs: set[tuple[int, int]] = set()
    for row in frontier_rows:
        src_idx = int(row.tile_index)
        for neighbor_idx in neighbor_map.get(src_idx, []):
            dst_idx = int(neighbor_idx)
            if dst_idx not in frontier_idx_set or src_idx == dst_idx:
                continue
            pair = (min(src_idx, dst_idx), max(src_idx, dst_idx))
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            edges.append(
                GraphEdge(
                    edge_type="rel_neighbor",
                    src_tile_index=pair[0],
                    dst_tile_index=pair[1],
                    target_dx_px=0.0,
                    target_dy_px=0.0,
                    weight=0.16,
                )
            )
    return edges, column_model


def _solve_frontier_translation_subgraph(
    *,
    frontier_rows: list[TileResultRow],
    edges: list[GraphEdge],
) -> tuple[dict[int, tuple[float, float]], dict[int, float], list[dict[str, Any]], dict[int, tuple[float, float]]]:
    if not frontier_rows:
        return {}, {}, [], {}

    node_ids = sorted(int(row.tile_index) for row in frontier_rows)
    index_by_id = {tile_id: idx for idx, tile_id in enumerate(node_ids)}
    row_by_id = {int(row.tile_index): row for row in frontier_rows}
    column_ids = sorted(
        {
            int(col)
            for edge in edges
            for col in (edge.src_column_index, edge.dst_column_index)
            if col is not None
        }
    )
    column_index_by_id = {col_id: idx for idx, col_id in enumerate(column_ids)}
    x0 = np.zeros((2 * (len(node_ids) + len(column_ids)),), dtype=np.float64)
    for tile_id, idx in index_by_id.items():
        row = row_by_id[int(tile_id)]
        x0[2 * idx] = float(row.final_dx_px)
        x0[2 * idx + 1] = float(row.final_dy_px)
    column_prior_targets: dict[int, tuple[float, float]] = {}
    for edge in edges:
        if str(edge.edge_type) != "column_prior" or edge.src_column_index is None:
            continue
        column_prior_targets[int(edge.src_column_index)] = (
            float(edge.target_dx_px),
            float(edge.target_dy_px),
        )
    column_offset = 2 * len(node_ids)
    for col_id, idx in column_index_by_id.items():
        init_dx, init_dy = column_prior_targets.get(
            int(col_id),
            (
                float(np.mean([float(row.final_dx_px) for row in frontier_rows if int(row.col_display) == int(col_id)])),
                float(np.mean([float(row.final_dy_px) for row in frontier_rows if int(row.col_display) == int(col_id)])),
            ),
        )
        x0[column_offset + 2 * idx] = float(init_dx)
        x0[column_offset + 2 * idx + 1] = float(init_dy)

    def _shift_for_tile(params: np.ndarray, tile_id: int) -> tuple[float, float]:
        idx = index_by_id[int(tile_id)]
        return float(params[2 * idx]), float(params[2 * idx + 1])

    def _shift_for_column(params: np.ndarray, column_id: int) -> tuple[float, float]:
        idx = column_index_by_id[int(column_id)]
        base = column_offset + 2 * idx
        return float(params[base]), float(params[base + 1])

    def _residual_vector(params: np.ndarray) -> np.ndarray:
        out: list[float] = []
        for edge in edges:
            edge_type = str(edge.edge_type)
            if edge_type == "rel_neighbor" and edge.src_tile_index is not None and edge.dst_tile_index is not None:
                sx, sy = _shift_for_tile(params, int(edge.src_tile_index))
                dx, dy = _shift_for_tile(params, int(edge.dst_tile_index))
                out.extend(
                    [
                        (dx - sx - float(edge.target_dx_px)) * float(edge.weight),
                        (dy - sy - float(edge.target_dy_px)) * float(edge.weight),
                    ]
                )
            elif edge_type == "column_shared" and edge.src_tile_index is not None and edge.src_column_index is not None:
                sx, sy = _shift_for_tile(params, int(edge.src_tile_index))
                cx, cy = _shift_for_column(params, int(edge.src_column_index))
                out.extend(
                    [
                        (sx - cx - float(edge.target_dx_px)) * float(edge.weight),
                        (sy - cy - float(edge.target_dy_px)) * float(edge.weight),
                    ]
                )
            elif edge_type == "column_prior" and edge.src_column_index is not None:
                cx, cy = _shift_for_column(params, int(edge.src_column_index))
                out.extend(
                    [
                        (cx - float(edge.target_dx_px)) * float(edge.weight),
                        (cy - float(edge.target_dy_px)) * float(edge.weight),
                    ]
                )
            elif edge_type == "column_rel" and edge.src_column_index is not None and edge.dst_column_index is not None:
                sx, sy = _shift_for_column(params, int(edge.src_column_index))
                dx, dy = _shift_for_column(params, int(edge.dst_column_index))
                out.extend(
                    [
                        (dx - sx - float(edge.target_dx_px)) * float(edge.weight),
                        (dy - sy - float(edge.target_dy_px)) * float(edge.weight),
                    ]
                )
            else:
                assert edge.src_tile_index is not None
                sx, sy = _shift_for_tile(params, int(edge.src_tile_index))
                out.extend(
                    [
                        (sx - float(edge.target_dx_px)) * float(edge.weight),
                        (sy - float(edge.target_dy_px)) * float(edge.weight),
                    ]
                )
        return np.asarray(out, dtype=np.float64)

    result = least_squares(
        _residual_vector,
        x0=x0,
        loss="huber",
        f_scale=6.0,
        max_nfev=80,
    )
    params = np.asarray(result.x, dtype=np.float64)
    optimized = {
        int(tile_id): _shift_for_tile(params, int(tile_id))
        for tile_id in node_ids
    }
    optimized_columns = {
        int(col_id): _shift_for_column(params, int(col_id))
        for col_id in column_ids
    }
    edge_residuals: list[dict[str, Any]] = []
    node_bucket: dict[int, list[float]] = {int(tile_id): [] for tile_id in node_ids}
    column_bucket: dict[int, list[float]] = {int(col_id): [] for col_id in column_ids}
    for edge in edges:
        edge_type = str(edge.edge_type)
        if edge_type == "rel_neighbor" and edge.src_tile_index is not None and edge.dst_tile_index is not None:
            sx, sy = optimized[int(edge.src_tile_index)]
            dx, dy = optimized[int(edge.dst_tile_index)]
            rx = (dx - sx - float(edge.target_dx_px)) * float(edge.weight)
            ry = (dy - sy - float(edge.target_dy_px)) * float(edge.weight)
            node_bucket[int(edge.src_tile_index)].append(abs(rx) + abs(ry))
            node_bucket[int(edge.dst_tile_index)].append(abs(rx) + abs(ry))
        elif edge_type == "column_shared" and edge.src_tile_index is not None and edge.src_column_index is not None:
            sx, sy = optimized[int(edge.src_tile_index)]
            dx, dy = optimized_columns[int(edge.src_column_index)]
            rx = (sx - dx - float(edge.target_dx_px)) * float(edge.weight)
            ry = (sy - dy - float(edge.target_dy_px)) * float(edge.weight)
            node_bucket[int(edge.src_tile_index)].append(abs(rx) + abs(ry))
            column_bucket[int(edge.src_column_index)].append(abs(rx) + abs(ry))
        elif edge_type == "column_prior" and edge.src_column_index is not None:
            sx, sy = optimized_columns[int(edge.src_column_index)]
            rx = (sx - float(edge.target_dx_px)) * float(edge.weight)
            ry = (sy - float(edge.target_dy_px)) * float(edge.weight)
            column_bucket[int(edge.src_column_index)].append(abs(rx) + abs(ry))
        elif edge_type == "column_rel" and edge.src_column_index is not None and edge.dst_column_index is not None:
            sx, sy = optimized_columns[int(edge.src_column_index)]
            dx, dy = optimized_columns[int(edge.dst_column_index)]
            rx = (dx - sx - float(edge.target_dx_px)) * float(edge.weight)
            ry = (dy - sy - float(edge.target_dy_px)) * float(edge.weight)
            column_bucket[int(edge.src_column_index)].append(abs(rx) + abs(ry))
            column_bucket[int(edge.dst_column_index)].append(abs(rx) + abs(ry))
        else:
            assert edge.src_tile_index is not None
            sx, sy = optimized[int(edge.src_tile_index)]
            rx = (sx - float(edge.target_dx_px)) * float(edge.weight)
            ry = (sy - float(edge.target_dy_px)) * float(edge.weight)
            node_bucket[int(edge.src_tile_index)].append(abs(rx) + abs(ry))
        edge_residuals.append(
            {
                "edge_type": str(edge.edge_type),
                "src_tile_index": None if edge.src_tile_index is None else int(edge.src_tile_index),
                "dst_tile_index": None if edge.dst_tile_index is None else int(edge.dst_tile_index),
                "src_column_index": None if edge.src_column_index is None else int(edge.src_column_index),
                "dst_column_index": None if edge.dst_column_index is None else int(edge.dst_column_index),
                "residual_norm": float(np.hypot(float(rx), float(ry))),
                "weight": float(edge.weight),
            }
        )
    node_residuals = {
        int(tile_id): float(np.mean(vals + column_bucket.get(int(row_by_id[int(tile_id)].col_display), []))) if (vals or column_bucket.get(int(row_by_id[int(tile_id)].col_display), [])) else 0.0
        for tile_id, vals in node_bucket.items()
    }
    return optimized, node_residuals, edge_residuals, optimized_columns


def _match_panel_shapes(panels: list[np.ndarray]) -> list[np.ndarray]:
    valid = [np.asarray(panel) for panel in panels if isinstance(panel, np.ndarray)]
    if not valid:
        return panels
    target_h = max(int(panel.shape[0]) for panel in valid)
    target_w = max(int(panel.shape[1]) for panel in valid)
    matched: list[np.ndarray] = []
    for panel in panels:
        arr = np.asarray(panel)
        if arr.shape[:2] == (target_h, target_w):
            matched.append(arr)
            continue
        interp = cv2.INTER_NEAREST if arr.ndim == 3 else cv2.INTER_LINEAR
        matched.append(cv2.resize(arr, (target_w, target_h), interpolation=interp))
    return matched


def _backwarp_fixed_to_tile_local(
    fixed_gray_full: np.ndarray,
    fixed_mask_full: np.ndarray,
    tile_mat: np.ndarray,
    out_shape_hw: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    inv_mat = cv2.invertAffineTransform(np.asarray(tile_mat, dtype=np.float32))
    out_w = int(out_shape_hw[1])
    out_h = int(out_shape_hw[0])
    fixed_local = cv2.warpAffine(
        np.asarray(fixed_gray_full, dtype=np.float32),
        inv_mat,
        (out_w, out_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=1.0,
    )
    fixed_local_mask_u8 = cv2.warpAffine(
        np.where(np.asarray(fixed_mask_full) > 0, 255, 0).astype(np.uint8),
        inv_mat,
        (out_w, out_h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return fixed_local.astype(np.float32), (fixed_local_mask_u8 > 0).astype(np.float32)


def _run_confocal_auto_scale_stage(
    cfg: ConfocalAutoScaleConfig,
    *,
    stage_name: str,
    scale_values: list[float],
    sampled_tile_defs: list[dict[str, Any]],
    sampled_tile_static: dict[int, dict[str, Any]],
    fixed_gray_full: np.ndarray,
    fixed_mask_full: np.ndarray,
    moving_reg_projection_u8: np.ndarray,
    moving_signal_mask_u8: np.ndarray,
    fixed_shape_hw: tuple[int, int],
    progress_total_units: int,
    progress_done_offset: int,
    scale_index_offset: int,
    scale_count_total: int,
    progress_cb: Step7ProgressCallback | None = None,
) -> dict[str, Any]:
    ordered_scale_values = _auto_scale_unique_sorted_values(list(scale_values or []))
    total_tiles = int(len(sampled_tile_defs))
    if not ordered_scale_values or total_tiles <= 0:
        return {
            "summary_rows": [],
            "tile_metric_rows": [],
            "worker_count": 1,
            "progress_done_total": int(progress_done_offset),
            "scale_values": [],
        }
    worker_count = _step7_tile_eval_worker_count(max_items=max(1, len(ordered_scale_values) * total_tiles))
    _emit_step7_progress(
        progress_cb,
        {
            "mode": "auto_scale",
            "stage": "setup",
            "stage_name": str(stage_name),
            "tile_count": int(total_tiles),
            "scale_count": int(scale_count_total),
            "total_units": int(progress_total_units),
            "worker_count": int(worker_count),
            "progress_percent": max(1, int(round(92.0 * float(progress_done_offset) / float(max(1, progress_total_units))))),
            "message": (
                f"auto_scale | {str(stage_name)} setup | scales={int(len(ordered_scale_values))} "
                f"tiles={int(total_tiles)} | workers={int(worker_count)}"
            ),
        },
    )

    stage_states: dict[int, dict[str, Any]] = {}
    for local_scale_index, scale_value in enumerate(ordered_scale_values, start=1):
        global_scale_index = int(scale_index_offset) + int(local_scale_index)
        base_manual_mat = build_manual_affine(
            moving_reg_projection_u8.shape[:2],
            fixed_gray_full.shape[:2],
            tx_px=cfg.tx_px,
            ty_px=cfg.ty_px,
            angle_deg=cfg.angle_deg,
            scale=float(scale_value),
            flip_lr=cfg.flip_lr,
            flip_ud=cfg.flip_ud,
        )
        manual_mat, anchor_info = _anchor_guided_manual_affine(
            moving_shape_hw=moving_reg_projection_u8.shape[:2],
            fixed_shape_hw=fixed_gray_full.shape[:2],
            current_mat=base_manual_mat,
            anchor_pairs=list(cfg.anchor_pairs or []),
            flip_lr=cfg.flip_lr,
            flip_ud=cfg.flip_ud,
        )
        stage_states[int(global_scale_index)] = {
            "stage_name": str(stage_name),
            "scale_value": float(scale_value),
            "global_scale_index": int(global_scale_index),
            "manual_mat": np.asarray(manual_mat, dtype=np.float32),
            "anchor_info": dict(anchor_info or {}),
            "ordered_rows": [None] * int(total_tiles),
            "completed_count": 0,
            "scheduled_count": 0,
            "skipped_count": 0,
            "pruned": False,
            "prune_reason": "",
        }

    prune_min_tiles = min(int(total_tiles), max(1, int(cfg.prune_min_tiles)))

    def _scale_task(scale_key: int, tile_order: int) -> tuple[int, int, dict[str, Any]]:
        state = stage_states[int(scale_key)]
        tile_payload = dict(sampled_tile_defs[int(tile_order)])
        tile_index = int(tile_payload.get("tile_index", -1))
        row = _evaluate_tile_alignment_with_prior(
            fixed_gray_full=fixed_gray_full,
            fixed_mask_full=fixed_mask_full,
            moving_reg_projection_u8=moving_reg_projection_u8,
            moving_signal_mask_u8=moving_signal_mask_u8,
            fixed_shape_hw=fixed_shape_hw,
            manual_mat=np.asarray(state["manual_mat"], dtype=np.float32),
            tile=tile_payload,
            registration_input_profile=str(cfg.registration_input_profile or STEP7_REGISTRATION_INPUT_PROFILE),
            search_radius_px=int(cfg.search_radius_px),
            prior_shift_dx_px=0,
            prior_shift_dy_px=0,
            neighbor_count=0,
            neighbor_labels=[],
            neighbor_shift_spread_px=0.0,
            local_refine_radius_px=int(cfg.local_refine_radius_px),
            tile_static=dict(sampled_tile_static.get(tile_index) or {}),
            progress_cb=None,
        )
        row_dict = row.to_dict()
        row_dict["stage_name"] = str(stage_name)
        return int(scale_key), int(tile_order), row_dict

    progress_done_total = int(progress_done_offset)
    future_map: dict[Any, tuple[int, int]] = {}

    def _submit_tile(pool: ThreadPoolExecutor, scale_key: int, tile_order: int) -> None:
        future = pool.submit(_scale_task, int(scale_key), int(tile_order))
        future_map[future] = (int(scale_key), int(tile_order))
        stage_states[int(scale_key)]["scheduled_count"] = int(stage_states[int(scale_key)]["scheduled_count"]) + 1

    with ThreadPoolExecutor(max_workers=int(worker_count), thread_name_prefix=f"step7-scale-{str(stage_name)}") as pool:
        for scale_key in sorted(stage_states.keys()):
            warmup_count = min(int(total_tiles), int(prune_min_tiles))
            for tile_order in range(warmup_count):
                _submit_tile(pool, int(scale_key), int(tile_order))

        while future_map:
            future = next(as_completed(list(future_map.keys())))
            scale_key, tile_order = future_map.pop(future)
            done_scale_key, done_tile_order, row_dict = future.result()
            state = stage_states[int(done_scale_key)]
            state["ordered_rows"][int(done_tile_order)] = dict(row_dict)
            state["completed_count"] = int(state.get("completed_count", 0)) + 1
            progress_done_total += 1
            _emit_step7_progress(
                progress_cb,
                {
                    "mode": "auto_scale",
                    "stage": "tile_done",
                    "stage_name": str(stage_name),
                    "tile_index": int(row_dict.get("tile_index", -1)),
                    "tile_label": str(row_dict.get("label") or ""),
                    "scale": float(state.get("scale_value", float("nan"))),
                    "scale_index": int(done_scale_key),
                    "scale_count": int(scale_count_total),
                    "tile_count": int(total_tiles),
                    "total_units": int(progress_total_units),
                    "done_units_count": int(progress_done_total),
                    "progress_percent": 0,
                    "message": (
                        f"auto_scale | {str(stage_name)} | scale {int(done_scale_key)}/{int(scale_count_total)} "
                        f"| tile {int(done_tile_order) + 1}/{int(total_tiles)}"
                    ),
                },
            )

            if int(state.get("completed_count", 0)) < int(state.get("scheduled_count", 0)):
                continue
            if int(state.get("scheduled_count", 0)) >= int(total_tiles):
                continue

            completed_rows = [row for row in list(state.get("ordered_rows") or []) if isinstance(row, dict)]
            should_prune, prune_reason = _auto_scale_should_prune(
                completed_rows,
                current_state_key=int(done_scale_key),
                stage_states=stage_states,
                total_tiles=int(total_tiles),
                min_tiles=int(prune_min_tiles),
            )
            if should_prune:
                skipped_count = max(0, int(total_tiles) - int(state.get("scheduled_count", 0)))
                state["pruned"] = True
                state["prune_reason"] = str(prune_reason or "pruned")
                state["skipped_count"] = int(skipped_count)
                state["scheduled_count"] = int(total_tiles)
                if skipped_count > 0:
                    progress_done_total += int(skipped_count)
                    _emit_step7_progress(
                        progress_cb,
                        {
                            "mode": "auto_scale",
                            "stage": "tile_done",
                            "stage_name": str(stage_name),
                            "tile_index": -1,
                            "tile_label": "",
                            "scale": float(state.get("scale_value", float("nan"))),
                            "scale_index": int(done_scale_key),
                            "scale_count": int(scale_count_total),
                            "tile_count": int(total_tiles),
                            "total_units": int(progress_total_units),
                            "done_units_count": int(progress_done_total),
                            "progress_percent": 0,
                            "message": (
                                f"auto_scale | {str(stage_name)} pruned | scale {int(done_scale_key)}/{int(scale_count_total)} "
                                f"| skipped={int(skipped_count)} | reason={str(prune_reason or 'pruned')}"
                            ),
                        },
                    )
                continue

            for next_tile_order in range(int(state.get("scheduled_count", 0)), int(total_tiles)):
                _submit_tile(pool, int(done_scale_key), int(next_tile_order))

    summary_rows: list[dict[str, Any]] = []
    tile_metric_rows: list[dict[str, Any]] = []
    for scale_key in sorted(stage_states.keys()):
        state = stage_states[int(scale_key)]
        rows = [row for row in list(state.get("ordered_rows") or []) if isinstance(row, dict)]
        scale_status = "pruned" if bool(state.get("pruned")) else ("complete" if len(rows) >= int(total_tiles) else "partial")
        for row in rows:
            row["stage_name"] = str(stage_name)
            row["scale_status"] = str(scale_status)
            tile_metric_rows.append(_auto_scale_tile_metric_row(float(state.get("scale_value", float("nan"))), row))
        agg = _auto_scale_aggregate(rows)
        agg.update(
            {
                "stage_name": str(stage_name),
                "scale": float(state.get("scale_value", float("nan"))),
                "scale_index": int(scale_key),
                "scale_count": int(scale_count_total),
                "anchor_guided": dict(state.get("anchor_info") or {}),
                "tile_count_evaluated": int(len(rows)),
                "tile_count_expected": int(total_tiles),
                "tile_count_skipped": int(state.get("skipped_count") or 0),
                "pruned": bool(state.get("pruned")),
                "prune_reason": str(state.get("prune_reason") or ""),
                "scale_status": str(scale_status),
            }
        )
        summary_rows.append(agg)

    return {
        "summary_rows": summary_rows,
        "tile_metric_rows": tile_metric_rows,
        "worker_count": int(worker_count),
        "progress_done_total": int(progress_done_total),
        "scale_values": [float(v) for v in ordered_scale_values],
    }


def run_confocal_auto_scale_sweep(
    cfg: ConfocalAutoScaleConfig,
    *,
    progress_cb: Step7ProgressCallback | None = None,
) -> dict[str, Any]:
    session_id = f"{_utc_stamp()}_{cfg.projection_mode}_ch{cfg.channel_index}_auto_scale"
    run_dir = cfg.out_root / cfg.myelin_label / session_id
    process_dir = run_dir / "process"
    run_dir.mkdir(parents=True, exist_ok=True)
    process_dir.mkdir(parents=True, exist_ok=True)

    fixed_rgb = np.asarray(cfg.myelin_rgb).copy()
    fixed_gray_full = rgb_to_gray_float(fixed_rgb)
    fixed_mask_full = (np.asarray(cfg.myelin_labels) == 1).astype(np.float32)
    if not np.any(fixed_mask_full > 0):
        fixed_mask_full = (np.asarray(cfg.myelin_labels) > 0).astype(np.float32)

    moving_signal_mask_u8 = (
        np.where(np.asarray(cfg.confocal_signal_mask_u8) > 0, 255, 0).astype(np.uint8)
        if cfg.confocal_signal_mask_u8 is not None
        else np.where(np.asarray(cfg.confocal_projection_u8) > 0, 255, 0).astype(np.uint8)
    )
    moving_reg_projection_u8 = (
        _invert_confocal_u8(cfg.confocal_projection_u8)
        if bool(cfg.invert_confocal_for_registration)
        else np.asarray(cfg.confocal_projection_u8, dtype=np.uint8)
    )

    projection_info = dict(cfg.projection_info or {})
    tile_defs = build_confocal_tile_defs(
        projection_info.get("stitch_info") if isinstance(projection_info.get("stitch_info"), dict) else {},
        raw_shape_hw=tuple(np.asarray(projection_info.get("raw_projection_shape_hw") or cfg.confocal_projection_u8.shape[:2], dtype=np.int32).tolist()),
        scaled_shape_hw=tuple(np.asarray(cfg.confocal_projection_u8.shape[:2], dtype=np.int32).tolist()),
        flip_lr=bool(cfg.flip_lr),
        flip_ud=bool(cfg.flip_ud),
    )
    if not tile_defs:
        raise ValueError("Auto Scale Sweep requires multi-tile stitch info; no tile definitions are available.")
    sampled_tile_defs = _select_auto_scale_sample_tiles(
        tile_defs,
        max_tiles=int(cfg.sample_tile_limit),
        strategy=str(cfg.sample_strategy or "rowwise_uniform"),
    )
    if not sampled_tile_defs:
        sampled_tile_defs = list(tile_defs)

    full_scale_values = _auto_scale_values(
        float(cfg.scale),
        half_range=float(cfg.sweep_half_range),
        step=float(cfg.sweep_step),
    )
    coarse_scale_values = _auto_scale_coarse_values(
        float(cfg.scale),
        half_range=float(cfg.sweep_half_range),
        coarse_count=min(max(1, int(cfg.coarse_scale_count)), max(1, len(full_scale_values))),
    )
    fixed_shape_hw = fixed_gray_full.shape[:2]
    total_tiles = max(1, len(sampled_tile_defs))
    sampled_tile_static = {
        int(tile.get("tile_index", -1)): _prepare_tile_eval_static(moving_reg_projection_u8, moving_signal_mask_u8, tile)
        for tile in sampled_tile_defs
    }

    coarse_total_units = max(1, int(total_tiles) * int(len(coarse_scale_values)))
    coarse_stage = _run_confocal_auto_scale_stage(
        cfg,
        stage_name="coarse",
        scale_values=list(coarse_scale_values),
        sampled_tile_defs=sampled_tile_defs,
        sampled_tile_static=sampled_tile_static,
        fixed_gray_full=fixed_gray_full,
        fixed_mask_full=fixed_mask_full,
        moving_reg_projection_u8=moving_reg_projection_u8,
        moving_signal_mask_u8=moving_signal_mask_u8,
        fixed_shape_hw=fixed_shape_hw,
        progress_total_units=int(coarse_total_units),
        progress_done_offset=0,
        scale_index_offset=0,
        scale_count_total=int(len(coarse_scale_values)),
        progress_cb=progress_cb,
    )

    refine_scale_values = _select_auto_scale_refine_values(
        list(coarse_stage.get("summary_rows") or []),
        center_scale=float(cfg.scale),
        half_range=float(cfg.sweep_half_range),
        fine_step=float(cfg.sweep_step),
        coarse_values=list(coarse_scale_values),
        top_k=max(1, int(cfg.coarse_top_k)),
        already_evaluated=list(coarse_stage.get("scale_values") or []),
    )
    total_scale_values = list(coarse_stage.get("scale_values") or []) + list(refine_scale_values or [])
    progress_total_units = int(total_tiles) * max(1, len(total_scale_values))
    fine_stage = _run_confocal_auto_scale_stage(
        cfg,
        stage_name="fine",
        scale_values=list(refine_scale_values),
        sampled_tile_defs=sampled_tile_defs,
        sampled_tile_static=sampled_tile_static,
        fixed_gray_full=fixed_gray_full,
        fixed_mask_full=fixed_mask_full,
        moving_reg_projection_u8=moving_reg_projection_u8,
        moving_signal_mask_u8=moving_signal_mask_u8,
        fixed_shape_hw=fixed_shape_hw,
        progress_total_units=int(progress_total_units),
        progress_done_offset=int(coarse_stage.get("progress_done_total") or 0),
        scale_index_offset=int(len(coarse_stage.get("scale_values") or [])),
        scale_count_total=int(len(total_scale_values)),
        progress_cb=progress_cb,
    )

    summary_rows = sorted(
        [dict(row) for row in list(coarse_stage.get("summary_rows") or []) + list(fine_stage.get("summary_rows") or [])],
        key=lambda row: int(row.get("scale_index") or 0),
    )
    for row in summary_rows:
        row["scale_count"] = int(len(total_scale_values))

    tile_metric_rows = [
        dict(row)
        for row in list(coarse_stage.get("tile_metric_rows") or []) + list(fine_stage.get("tile_metric_rows") or [])
    ]

    _emit_step7_progress(
        progress_cb,
        {
            "mode": "auto_scale",
            "stage": "ranking",
            "scale_count": int(len(total_scale_values)),
            "tile_count": int(total_tiles),
            "total_units": int(progress_total_units),
            "done_units_count": int(progress_total_units),
            "progress_percent": 96,
            "message": "auto_scale | ranking scale candidates",
        },
    )

    ranked_rows, best_composite, best_final, best_right = _select_auto_scale_best_rows(summary_rows)
    for rank, row in enumerate(ranked_rows, start=1):
        row["rank"] = int(rank)

    summary_csv_path = process_dir / "scale_sweep_summary.csv"
    _write_auto_scale_summary_csv(summary_csv_path, ranked_rows)
    tile_csv_path = process_dir / "tile_metrics_long.csv"
    if tile_metric_rows:
        with tile_csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(tile_metric_rows[0].keys()))
            writer.writeheader()
            writer.writerows(tile_metric_rows)

    total_worker_count = max(int(coarse_stage.get("worker_count") or 1), int(fine_stage.get("worker_count") or 1))
    summary_lines = [
        "# Step 7 Auto Scale Sweep",
        "",
        f"- center scale: `{float(cfg.scale):.5f}`",
        f"- sweep half-range: `{float(cfg.sweep_half_range):.5f}`",
        f"- sweep step: `{float(cfg.sweep_step):.5f}`",
        f"- coarse scales: `{len(coarse_scale_values)}`",
        f"- refine scales: `{len(refine_scale_values)}`",
        f"- evaluated scales: `{len(ranked_rows)}`",
        f"- sample strategy: `{str(cfg.sample_strategy or 'rowwise_uniform')}`",
        f"- sample tiles used: `{len(sampled_tile_defs)}/{len(tile_defs)}`",
        f"- workers used: `{int(total_worker_count)}`",
        "- sampled tile labels: "
        + ", ".join(str(tile.get("label") or f"T{int(tile.get('tile_index', -1)):02d}") for tile in sampled_tile_defs),
        f"- chosen scale by composite score: `{float(best_composite.get('scale', float('nan'))):.5f}`",
        f"- best mean final CC scale: `{float(best_final.get('scale', float('nan'))):.5f}`",
        f"- best right-flatten scale: `{float(best_right.get('scale', float('nan'))):.5f}`",
        "",
        "## Chosen Scale Aggregate",
        f"- composite_score: `{float(best_composite.get('composite_score', float('nan'))):.5f}`",
        f"- mean_final_cc: `{float(best_composite.get('mean_final_cc', float('nan'))):.5f}`",
        f"- mean_abs_dx: `{float(best_composite.get('mean_abs_dx', float('nan'))):.3f}`",
        f"- rightmost_abs_dx: `{float(best_composite.get('rightmost_abs_dx', float('nan'))):.3f}`",
        "",
        "## Files",
        f"- scale_sweep_summary_csv: `{summary_csv_path}`",
        f"- tile_metrics_long_csv: `{tile_csv_path}`",
    ]
    summary_md_path = run_dir / "summary.md"
    summary_md_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    manifest = {
        "myelin_label": cfg.myelin_label,
        "section_dir": str(cfg.myelin_section_dir),
        "run_dir": str(run_dir),
        "files": {
            "summary_md": str(summary_md_path),
            "scale_sweep_summary_csv": str(summary_csv_path),
            "tile_metrics_long_csv": str(tile_csv_path),
            "manifest": str(run_dir / "auto_scale_manifest.json"),
        },
        "registration_input_profile": str(cfg.registration_input_profile or STEP7_REGISTRATION_INPUT_PROFILE),
        "projection_mode": str(cfg.projection_mode),
        "channel_index": int(cfg.channel_index),
        "confocal_source_mode": str(cfg.confocal_source_mode),
        "confocal_sources": [str(path) for path in cfg.confocal_sources],
        "manual_init": {
            "tx_px": float(cfg.tx_px),
            "ty_px": float(cfg.ty_px),
            "angle_deg": float(cfg.angle_deg),
            "scale": float(cfg.scale),
            "flip_lr": bool(cfg.flip_lr),
            "flip_ud": bool(cfg.flip_ud),
        },
        "sweep": {
            "scale_values_full_grid": [float(v) for v in full_scale_values],
            "scale_values_coarse": [float(v) for v in coarse_scale_values],
            "scale_values_refine": [float(v) for v in refine_scale_values],
            "scale_values_evaluated": [float(row.get("scale", float("nan"))) for row in ranked_rows if np.isfinite(float(row.get("scale", float("nan"))))],
            "search_radius_px": int(cfg.search_radius_px),
            "local_refine_radius_px": int(cfg.local_refine_radius_px),
            "tile_count_total": int(len(tile_defs)),
            "tile_count_sampled": int(len(sampled_tile_defs)),
            "sample_strategy": str(cfg.sample_strategy or "rowwise_uniform"),
            "sample_tile_limit": int(cfg.sample_tile_limit),
            "sampled_tile_indices": [int(tile.get("tile_index", -1)) for tile in sampled_tile_defs],
            "sampled_tile_labels": [str(tile.get("label") or f"T{int(tile.get('tile_index', -1)):02d}") for tile in sampled_tile_defs],
            "coarse_scale_count": int(cfg.coarse_scale_count),
            "coarse_top_k": int(cfg.coarse_top_k),
            "prune_min_tiles": int(cfg.prune_min_tiles),
            "worker_count": int(total_worker_count),
        },
        "best_by_composite": dict(best_composite),
        "best_by_mean_final_cc": dict(best_final),
        "best_by_rightmost_abs_dx": dict(best_right),
        "summary_rows": ranked_rows,
    }
    _write_json(run_dir / "auto_scale_manifest.json", manifest)

    _emit_step7_progress(
        progress_cb,
        {
            "mode": "auto_scale",
            "stage": "done",
            "scale_count": int(len(total_scale_values)),
            "tile_count": int(total_tiles),
            "chosen_scale": float(best_composite.get("scale", cfg.scale)),
            "progress_percent": 100,
            "message": f"auto_scale | done | chosen_scale={float(best_composite.get('scale', cfg.scale)):.5f}",
        },
    )
    return {
        "run_dir": str(run_dir),
        "chosen_scale": float(best_composite.get("scale", cfg.scale)),
        "manual_init": dict(manifest.get("manual_init") or {}),
        "sweep": dict(manifest.get("sweep") or {}),
        "best_by_composite": dict(best_composite),
        "best_by_mean_final_cc": dict(best_final),
        "best_by_rightmost_abs_dx": dict(best_right),
        "summary_rows": ranked_rows,
        "summary_md": str(summary_md_path),
        "scale_sweep_summary_csv": str(summary_csv_path),
        "tile_metrics_long_csv": str(tile_csv_path),
        "manifest": str(run_dir / "auto_scale_manifest.json"),
    }


def run_confocal_seed_tile_screening(
    cfg: ConfocalSeedScreenConfig,
    *,
    progress_cb: Step7ProgressCallback | None = None,
) -> dict[str, Any]:
    session_id = f"{_utc_stamp()}_{cfg.projection_mode}_ch{cfg.channel_index}_seed_screen"
    run_dir = cfg.out_root / cfg.myelin_label / session_id
    process_dir = run_dir / "process"
    run_dir.mkdir(parents=True, exist_ok=True)
    process_dir.mkdir(parents=True, exist_ok=True)

    fixed_rgb = np.asarray(cfg.myelin_rgb).copy()
    fixed_gray_full = rgb_to_gray_float(fixed_rgb)
    fixed_mask_full = (np.asarray(cfg.myelin_labels) == 1).astype(np.float32)
    if not np.any(fixed_mask_full > 0):
        fixed_mask_full = (np.asarray(cfg.myelin_labels) > 0).astype(np.float32)

    moving_signal_mask_u8 = (
        np.where(np.asarray(cfg.confocal_signal_mask_u8) > 0, 255, 0).astype(np.uint8)
        if cfg.confocal_signal_mask_u8 is not None
        else np.where(np.asarray(cfg.confocal_projection_u8) > 0, 255, 0).astype(np.uint8)
    )
    moving_reg_projection_u8 = (
        _invert_confocal_u8(cfg.confocal_projection_u8)
        if bool(cfg.invert_confocal_for_registration)
        else np.asarray(cfg.confocal_projection_u8, dtype=np.uint8)
    )

    base_manual_mat = build_manual_affine(
        moving_reg_projection_u8.shape[:2],
        fixed_rgb.shape[:2],
        tx_px=cfg.tx_px,
        ty_px=cfg.ty_px,
        angle_deg=cfg.angle_deg,
        scale=cfg.scale,
        flip_lr=cfg.flip_lr,
        flip_ud=cfg.flip_ud,
    )
    manual_mat, anchor_info = _anchor_guided_manual_affine(
        moving_shape_hw=moving_reg_projection_u8.shape[:2],
        fixed_shape_hw=fixed_rgb.shape[:2],
        current_mat=base_manual_mat,
        anchor_pairs=list(cfg.anchor_pairs or []),
        flip_lr=cfg.flip_lr,
        flip_ud=cfg.flip_ud,
    )

    projection_info = dict(cfg.projection_info or {})
    tile_defs = build_confocal_tile_defs(
        projection_info.get("stitch_info") if isinstance(projection_info.get("stitch_info"), dict) else {},
        raw_shape_hw=tuple(np.asarray(projection_info.get("raw_projection_shape_hw") or cfg.confocal_projection_u8.shape[:2], dtype=np.int32).tolist()),
        scaled_shape_hw=tuple(np.asarray(cfg.confocal_projection_u8.shape[:2], dtype=np.int32).tolist()),
        flip_lr=bool(cfg.flip_lr),
        flip_ud=bool(cfg.flip_ud),
    )

    if not tile_defs:
        raise ValueError("Seed screening requires multi-tile stitch info; no tile definitions are available.")

    rows: list[dict[str, Any]] = []
    fixed_shape_hw = fixed_gray_full.shape[:2]
    search_radius = max(8, int(cfg.search_radius_px))
    total_tiles = max(1, len(tile_defs))
    worker_count = _step7_tile_eval_worker_count(max_items=len(tile_defs))
    _emit_step7_progress(
        progress_cb,
        {
            "mode": "seed_screen",
            "stage": "setup",
            "tile_count": int(len(tile_defs)),
            "worker_count": int(worker_count),
            "progress_percent": 1,
            "message": f"seed_screen | setup | tiles={int(len(tile_defs))} | workers={int(worker_count)}",
        },
    )

    def _seed_task(item_index: int, tile_payload: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        tile_label = str(tile_payload.get("label") or f"T{int(tile_payload.get('tile_index', -1)):02d}")
        tile_base = (float(item_index - 1) / float(total_tiles)) * 88.0
        tile_span = 88.0 / float(total_tiles)
        row = _evaluate_tile_alignment_with_prior(
            fixed_gray_full=fixed_gray_full,
            fixed_mask_full=fixed_mask_full,
            moving_reg_projection_u8=moving_reg_projection_u8,
            moving_signal_mask_u8=moving_signal_mask_u8,
            fixed_shape_hw=fixed_shape_hw,
            manual_mat=manual_mat,
            tile=tile_payload,
            registration_input_profile=str(cfg.registration_input_profile or STEP7_REGISTRATION_INPUT_PROFILE),
            search_radius_px=int(search_radius),
            prior_shift_dx_px=0,
            prior_shift_dy_px=0,
            neighbor_count=0,
            neighbor_labels=[],
            neighbor_shift_spread_px=0.0,
            local_refine_radius_px=int(STEP7_SEED_REFINE_LOCAL_RADIUS_PX),
            progress_cb=progress_cb,
            progress_mode="seed_screen",
            progress_item_index=int(item_index),
            progress_total_items=int(total_tiles),
            progress_base_percent=float(tile_base),
            progress_span_percent=float(tile_span),
        )
        row_dict = row.to_dict()
        row_dict["seed_score"] = _screen_seed_score(
            float(row.current_cc),
            float(row.final_cc),
            int(round(row.final_dx_px)),
            int(round(row.final_dy_px)),
        )
        _emit_step7_progress(
            progress_cb,
            {
                "mode": "seed_screen",
                "stage": "tile_done",
                "tile_index": int(tile_payload.get("tile_index", -1)),
                "tile_label": tile_label,
                "item_index": int(item_index),
                "total_items": int(total_tiles),
                "progress_percent": _step7_progress_percent(0.0, 88.0, float(item_index) / float(total_tiles)),
                "message": (
                    f"seed_screen | tile {int(item_index)}/{int(total_tiles)} "
                    f"{tile_label}"
                    f" done | profile={str(row.registration_profile or '')}"
                ),
            },
        )
        return int(item_index), row_dict

    ordered_rows: list[dict[str, Any] | None] = [None] * int(len(tile_defs))
    if worker_count <= 1:
        for item_index, tile in enumerate(tile_defs, start=1):
            done_index, row_dict = _seed_task(int(item_index), tile)
            ordered_rows[int(done_index) - 1] = row_dict
    else:
        with ThreadPoolExecutor(max_workers=int(worker_count), thread_name_prefix="step7-seed") as pool:
            future_map = {
                pool.submit(_seed_task, int(item_index), tile): int(item_index)
                for item_index, tile in enumerate(tile_defs, start=1)
            }
            for future in as_completed(future_map):
                done_index, row_dict = future.result()
                ordered_rows[int(done_index) - 1] = row_dict
    rows = [row for row in ordered_rows if isinstance(row, dict)]

    _emit_step7_progress(
        progress_cb,
        {
            "mode": "seed_screen",
            "stage": "ranking",
            "tile_count": int(len(rows)),
            "progress_percent": 92,
            "message": f"seed_screen | ranking {int(len(rows))} tiles",
        },
    )
    ranked_rows = sorted(rows, key=lambda row: (float(row["seed_score"]), float(row["current_cc"])), reverse=True)
    for rank, row in enumerate(ranked_rows, start=1):
        row["rank"] = int(rank)

    panel_keys = ("moving", "fixed", "overlay", "heatmap")
    target_h = max(int(np.asarray(row[key]).shape[0]) for row in ranked_rows for key in panel_keys if isinstance(row.get(key), np.ndarray))
    target_w = max(int(np.asarray(row[key]).shape[1]) for row in ranked_rows for key in panel_keys if isinstance(row.get(key), np.ndarray))
    for row in ranked_rows:
        for key in panel_keys:
            panel = row.get(key)
            if not isinstance(panel, np.ndarray):
                continue
            if panel.shape[:2] != (target_h, target_w):
                row[key] = cv2.resize(panel, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

    storyboard_rows = []
    for row in ranked_rows[: max(1, int(cfg.top_k_storyboard))]:
        storyboard_rows.append(
            {
                "label": f"{row['label']} | rank={int(row['rank'])}",
                "note": (
                    f"score={float(row['seed_score']):.4f} | "
                    f"CC {float(row['current_cc']):.4f}->{float(row['shifted_cc']):.4f} | "
                    f"shift=({int(row['best_shift_dx_px'])},{int(row['best_shift_dy_px'])})"
                ),
                "moving": row["moving"],
                "fixed": row["fixed"],
                "overlay": row["overlay"],
                "heatmap": row["heatmap"],
                "col_titles": row["col_titles"],
            }
        )

    storyboard_path = run_dir / "seed_tile_storyboard.png"
    _emit_step7_progress(
        progress_cb,
        {
            "mode": "seed_screen",
            "stage": "storyboard",
            "progress_percent": 96,
            "message": "seed_screen | rendering storyboard",
        },
    )
    render_storyboard(storyboard_rows, storyboard_path)

    csv_path = process_dir / "seed_tile_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "rank",
                "tile_index",
                "label",
                "row_display",
                "col_display",
                "current_cc",
                "current_mi",
                "best_shift_dx_px",
                "best_shift_dy_px",
                "best_shift_cc",
                "shifted_cc",
                "shift_gain_cc",
                "seed_score",
                "density_regime",
                "signal_coverage",
                "moving_edge_density",
                "fixed_edge_density",
                "edge_density_mean",
                "registration_profile",
                "profile_candidates_str",
                "coarse_search_profiles_str",
                "refine_profiles_str",
                "refine_objective",
            ],
        )
        writer.writeheader()
        for row in ranked_rows:
            writer.writerow({key: row.get(key) for key in writer.fieldnames})

    top_seeds = [
        {
            "rank": int(row["rank"]),
            "tile_index": int(row["tile_index"]),
            "label": str(row["label"]),
            "row_display": int(row["row_display"]),
            "col_display": int(row["col_display"]),
            "current_cc": float(row["current_cc"]),
            "best_shift_dx_px": int(row["best_shift_dx_px"]),
            "best_shift_dy_px": int(row["best_shift_dy_px"]),
            "seed_score": float(row["seed_score"]),
            "density_regime": str(row.get("density_regime") or ""),
            "registration_profile": str(row.get("registration_profile") or ""),
            "refine_objective": str(row.get("refine_objective") or "cc"),
        }
        for row in ranked_rows[:5]
    ]
    manifest = {
        "myelin_label": cfg.myelin_label,
        "section_dir": str(cfg.myelin_section_dir),
        "run_dir": str(run_dir),
        "storyboard_path": str(storyboard_path),
        "registration_input_profile": str(cfg.registration_input_profile or STEP7_REGISTRATION_INPUT_PROFILE),
        "projection_mode": str(cfg.projection_mode),
        "channel_index": int(cfg.channel_index),
        "confocal_source_mode": str(cfg.confocal_source_mode),
        "confocal_sources": [str(path) for path in cfg.confocal_sources],
        "manual_init": {
            "tx_px": float(cfg.tx_px),
            "ty_px": float(cfg.ty_px),
            "angle_deg": float(cfg.angle_deg),
            "scale": float(cfg.scale),
            "flip_lr": bool(cfg.flip_lr),
            "flip_ud": bool(cfg.flip_ud),
            "anchor_guided": anchor_info,
        },
        "tile_defs": [
            {
                "tile_index": int(tile["tile_index"]),
                "label": str(tile["label"]),
                "row_display": int(tile["row_display"]),
                "col_display": int(tile["col_display"]),
                "scaled_bbox_xyxy": list(tile["scaled_bbox_xyxy"]),
                "display_bbox_xyxy": list(tile["display_bbox_xyxy"]),
            }
            for tile in tile_defs
        ],
        "top_seed_candidates": top_seeds,
        "files": {
            "storyboard": str(storyboard_path),
            "seed_tile_metrics_csv": str(csv_path),
            "manifest": str(run_dir / "seed_tile_manifest.json"),
        },
    }
    _write_json(run_dir / "seed_tile_manifest.json", manifest)
    _emit_step7_progress(
        progress_cb,
        {
            "mode": "seed_screen",
            "stage": "done",
            "progress_percent": 100,
            "message": f"seed_screen | finished | top seed={str(top_seeds[0]['label']) if top_seeds else 'none'}",
        },
    )
    return manifest | {"rows": ranked_rows}


def run_confocal_frontier_propagation(
    cfg: ConfocalFrontierConfig,
    *,
    progress_cb: Step7ProgressCallback | None = None,
) -> dict[str, Any]:
    session_id = f"{_utc_stamp()}_{cfg.projection_mode}_ch{cfg.channel_index}_frontier"
    run_dir = cfg.out_root / cfg.myelin_label / session_id
    process_dir = run_dir / "process"
    run_dir.mkdir(parents=True, exist_ok=True)
    process_dir.mkdir(parents=True, exist_ok=True)

    fixed_rgb = np.asarray(cfg.myelin_rgb).copy()
    fixed_gray_full = rgb_to_gray_float(fixed_rgb)
    fixed_mask_full = (np.asarray(cfg.myelin_labels) == 1).astype(np.float32)
    if not np.any(fixed_mask_full > 0):
        fixed_mask_full = (np.asarray(cfg.myelin_labels) > 0).astype(np.float32)

    moving_signal_mask_u8 = (
        np.where(np.asarray(cfg.confocal_signal_mask_u8) > 0, 255, 0).astype(np.uint8)
        if cfg.confocal_signal_mask_u8 is not None
        else np.where(np.asarray(cfg.confocal_projection_u8) > 0, 255, 0).astype(np.uint8)
    )
    moving_reg_projection_u8 = (
        _invert_confocal_u8(cfg.confocal_projection_u8)
        if bool(cfg.invert_confocal_for_registration)
        else np.asarray(cfg.confocal_projection_u8, dtype=np.uint8)
    )

    base_manual_mat = build_manual_affine(
        moving_reg_projection_u8.shape[:2],
        fixed_rgb.shape[:2],
        tx_px=cfg.tx_px,
        ty_px=cfg.ty_px,
        angle_deg=cfg.angle_deg,
        scale=cfg.scale,
        flip_lr=cfg.flip_lr,
        flip_ud=cfg.flip_ud,
    )
    manual_mat, anchor_info = _anchor_guided_manual_affine(
        moving_shape_hw=moving_reg_projection_u8.shape[:2],
        fixed_shape_hw=fixed_rgb.shape[:2],
        current_mat=base_manual_mat,
        anchor_pairs=list(cfg.anchor_pairs or []),
        flip_lr=cfg.flip_lr,
        flip_ud=cfg.flip_ud,
    )

    projection_info = dict(cfg.projection_info or {})
    tile_defs = build_confocal_tile_defs(
        projection_info.get("stitch_info") if isinstance(projection_info.get("stitch_info"), dict) else {},
        raw_shape_hw=tuple(np.asarray(projection_info.get("raw_projection_shape_hw") or cfg.confocal_projection_u8.shape[:2], dtype=np.int32).tolist()),
        scaled_shape_hw=tuple(np.asarray(cfg.confocal_projection_u8.shape[:2], dtype=np.int32).tolist()),
        flip_lr=bool(cfg.flip_lr),
        flip_ud=bool(cfg.flip_ud),
    )
    if not tile_defs:
        raise ValueError("Frontier propagation requires multi-tile stitch info; no tile definitions are available.")

    tile_by_idx = _tile_defs_by_index(tile_defs)
    neighbor_map = _tile_neighbor_map(tile_defs)
    fixed_shape_hw = fixed_gray_full.shape[:2]

    prior_rows_by_idx: dict[int, TileResultRow] = {}
    for row in list(cfg.prior_rows or []):
        if not isinstance(row, dict):
            continue
        tile_index = int(row.get("tile_index", -1))
        if tile_index >= 0:
            prior_rows_by_idx[tile_index] = TileResultRow.from_mapping(dict(row))

    frozen_tile_indices = {int(v) for v in list(cfg.frozen_tile_indices or []) if int(v) in tile_by_idx}
    accepted_tile_indices = {int(v) for v in list(cfg.accepted_tile_indices or []) if int(v) in tile_by_idx}
    for tile_index, row in list(prior_rows_by_idx.items()):
        state = str(row.tile_state or TileState.UNSEEN.value)
        if state == TileState.FROZEN.value:
            frozen_tile_indices.add(int(tile_index))
        elif state == TileState.ACCEPTED.value:
            accepted_tile_indices.add(int(tile_index))

    solved_tile_indices = set(frozen_tile_indices) | set(accepted_tile_indices)
    if not solved_tile_indices and prior_rows_by_idx:
        ranked_prior = sorted(
            prior_rows_by_idx.values(),
            key=lambda row: (
                float(row.frontier_confidence if np.isfinite(row.frontier_confidence) else -float("inf")),
                float(row.final_cc if np.isfinite(row.final_cc) else row.current_cc if np.isfinite(row.current_cc) else -float("inf")),
            ),
            reverse=True,
        )
        best = ranked_prior[0]
        solved_tile_indices.add(int(best.tile_index))

    if not solved_tile_indices:
        raise ValueError("Frontier propagation needs at least one accepted/frozen tile, or prior seed-screen rows.")

    _emit_step7_progress(
        progress_cb,
        {
            "mode": "frontier",
            "stage": "setup",
            "solved_tile_count": int(len(solved_tile_indices)),
            "progress_percent": 1,
            "message": (
                f"frontier | setup | solved={int(len(solved_tile_indices))} "
                f"| frozen={int(len(frozen_tile_indices))} | accepted={int(len(accepted_tile_indices))}"
            ),
        },
    )

    missing_solved_tile_indices = [int(tile_index) for tile_index in sorted(solved_tile_indices) if int(tile_index) not in prior_rows_by_idx]
    solved_worker_count = _step7_tile_eval_worker_count(max_items=max(1, len(missing_solved_tile_indices)))
    _emit_step7_progress(
        progress_cb,
        {
            "mode": "frontier",
            "stage": "solved_setup",
            "tile_count": int(len(missing_solved_tile_indices)),
            "worker_count": int(solved_worker_count),
            "progress_percent": 3,
            "message": (
                f"frontier | solved setup | missing={int(len(missing_solved_tile_indices))} "
                f"| workers={int(solved_worker_count)}"
            ),
        },
    )

    def _solve_seed_row(item_index: int, tile_index: int) -> tuple[int, TileResultRow]:
        tile_label = str(tile_by_idx[int(tile_index)].get("label") or f"T{int(tile_index):02d}")
        row = _evaluate_tile_alignment_with_prior(
            fixed_gray_full=fixed_gray_full,
            fixed_mask_full=fixed_mask_full,
            moving_reg_projection_u8=moving_reg_projection_u8,
            moving_signal_mask_u8=moving_signal_mask_u8,
            fixed_shape_hw=fixed_shape_hw,
            manual_mat=manual_mat,
            tile=tile_by_idx[int(tile_index)],
            registration_input_profile=str(cfg.registration_input_profile or STEP7_REGISTRATION_INPUT_PROFILE),
            search_radius_px=max(8, int(cfg.search_radius_px)),
            prior_shift_dx_px=0,
            prior_shift_dy_px=0,
            neighbor_count=0,
            neighbor_labels=[],
            neighbor_shift_spread_px=0.0,
            local_refine_radius_px=int(STEP7_FRONTIER_REFINE_LOCAL_RADIUS_PX),
            progress_cb=progress_cb,
            progress_mode="frontier_solved",
            progress_item_index=int(item_index),
            progress_total_items=max(1, int(len(missing_solved_tile_indices))),
            progress_base_percent=4.0 + (12.0 * float(item_index - 1) / float(max(1, len(missing_solved_tile_indices)))),
            progress_span_percent=12.0 / float(max(1, len(missing_solved_tile_indices))),
        )
        row.frontier_confidence = _screen_seed_score(
            float(row.current_cc),
            float(row.final_cc),
            int(round(row.final_dx_px)),
            int(round(row.final_dy_px)),
        )
        _emit_step7_progress(
            progress_cb,
            {
                "mode": "frontier",
                "stage": "solved_tile_done",
                "tile_index": int(tile_index),
                "tile_label": tile_label,
                "item_index": int(item_index),
                "total_items": int(max(1, len(missing_solved_tile_indices))),
                "progress_percent": _step7_progress_percent(
                    4.0,
                    12.0,
                    float(item_index) / float(max(1, len(missing_solved_tile_indices))),
                ),
                "message": (
                    f"frontier | solved tile {int(item_index)}/{int(max(1, len(missing_solved_tile_indices)))} "
                    f"{tile_label}"
                ),
            },
        )
        return int(tile_index), row

    if missing_solved_tile_indices:
        if solved_worker_count <= 1:
            for item_index, tile_index in enumerate(missing_solved_tile_indices, start=1):
                done_tile_index, row = _solve_seed_row(int(item_index), int(tile_index))
                prior_rows_by_idx[int(done_tile_index)] = row
        else:
            with ThreadPoolExecutor(max_workers=int(solved_worker_count), thread_name_prefix="step7-frontier-solved") as pool:
                future_map = {
                    pool.submit(_solve_seed_row, int(item_index), int(tile_index)): int(tile_index)
                    for item_index, tile_index in enumerate(missing_solved_tile_indices, start=1)
                }
                for future in as_completed(future_map):
                    done_tile_index, row = future.result()
                    prior_rows_by_idx[int(done_tile_index)] = row

    solved_rows: list[TileResultRow] = []
    for tile_index in sorted(solved_tile_indices):
        row = prior_rows_by_idx[int(tile_index)]
        row.tile_state = TileState.FROZEN.value if int(tile_index) in frozen_tile_indices else TileState.ACCEPTED.value
        solved_rows.append(row)

    solved_rows_by_idx = {int(row.tile_index): row for row in solved_rows}

    frontier_tile_indices: set[int] = set()
    for tile_index in solved_tile_indices:
        frontier_tile_indices.update(int(v) for v in neighbor_map.get(int(tile_index), []))
    frontier_tile_indices.difference_update(solved_tile_indices)

    frontier_candidates: list[tuple[int, list[int]]] = []
    for tile_index in sorted(frontier_tile_indices):
        solved_neighbors = [int(v) for v in neighbor_map.get(int(tile_index), []) if int(v) in solved_tile_indices and int(v) in solved_rows_by_idx]
        if solved_neighbors:
            frontier_candidates.append((int(tile_index), solved_neighbors))

    frontier_worker_count = _step7_tile_eval_worker_count(max_items=max(1, len(frontier_candidates)))
    _emit_step7_progress(
        progress_cb,
        {
            "mode": "frontier",
            "stage": "frontier_setup",
            "tile_count": int(len(frontier_candidates)),
            "worker_count": int(frontier_worker_count),
            "progress_percent": 18,
            "message": f"frontier | evaluating {int(len(frontier_candidates))} frontier tiles",
        },
    )

    def _frontier_task(item_index: int, tile_index: int, solved_neighbors: list[int]) -> tuple[int, TileResultRow]:
        tile_label = str(tile_by_idx[int(tile_index)].get("label") or f"T{int(tile_index):02d}")
        neighbor_rows = [solved_rows_by_idx[int(idx)] for idx in solved_neighbors]
        prior_dx = int(round(float(np.mean([float(row.final_dx_px) for row in neighbor_rows]))))
        prior_dy = int(round(float(np.mean([float(row.final_dy_px) for row in neighbor_rows]))))
        spread_x = float(np.std([float(row.final_dx_px) for row in neighbor_rows])) if len(neighbor_rows) > 1 else 0.0
        spread_y = float(np.std([float(row.final_dy_px) for row in neighbor_rows])) if len(neighbor_rows) > 1 else 0.0
        neighbor_shift_spread_px = float(np.hypot(spread_x, spread_y))
        row = _evaluate_tile_alignment_with_prior(
            fixed_gray_full=fixed_gray_full,
            fixed_mask_full=fixed_mask_full,
            moving_reg_projection_u8=moving_reg_projection_u8,
            moving_signal_mask_u8=moving_signal_mask_u8,
            fixed_shape_hw=fixed_shape_hw,
            manual_mat=manual_mat,
            tile=tile_by_idx[int(tile_index)],
            registration_input_profile=str(cfg.registration_input_profile or STEP7_REGISTRATION_INPUT_PROFILE),
            search_radius_px=max(8, int(cfg.search_radius_px)),
            prior_shift_dx_px=prior_dx,
            prior_shift_dy_px=prior_dy,
            neighbor_count=len(solved_neighbors),
            neighbor_labels=[str(tile_by_idx[int(idx)].get("label") or f"T{int(idx):02d}") for idx in solved_neighbors],
            neighbor_shift_spread_px=neighbor_shift_spread_px,
            local_refine_radius_px=int(STEP7_FRONTIER_REFINE_LOCAL_RADIUS_PX),
            progress_cb=progress_cb,
            progress_mode="frontier",
            progress_item_index=int(item_index),
            progress_total_items=max(1, int(len(frontier_candidates))),
            progress_base_percent=20.0 + (52.0 * float(item_index - 1) / float(max(1, len(frontier_candidates)))),
            progress_span_percent=52.0 / float(max(1, len(frontier_candidates))),
        )
        _emit_step7_progress(
            progress_cb,
            {
                "mode": "frontier",
                "stage": "frontier_tile_done",
                "tile_index": int(tile_index),
                "tile_label": tile_label,
                "item_index": int(item_index),
                "total_items": int(max(1, len(frontier_candidates))),
                "progress_percent": _step7_progress_percent(
                    20.0,
                    52.0,
                    float(item_index) / float(max(1, len(frontier_candidates))),
                ),
                "message": (
                    f"frontier | tile {int(item_index)}/{int(max(1, len(frontier_candidates)))} "
                    f"{tile_label}"
                    f" done | profile={str(row.registration_profile or '')}"
                ),
            },
        )
        return int(item_index), row

    ordered_frontier_rows: list[TileResultRow | None] = [None] * int(len(frontier_candidates))
    if frontier_worker_count <= 1:
        for item_index, (tile_index, solved_neighbors) in enumerate(frontier_candidates, start=1):
            done_index, row = _frontier_task(int(item_index), int(tile_index), list(solved_neighbors))
            ordered_frontier_rows[int(done_index) - 1] = row
    else:
        with ThreadPoolExecutor(max_workers=int(frontier_worker_count), thread_name_prefix="step7-frontier") as pool:
            future_map = {
                pool.submit(_frontier_task, int(item_index), int(tile_index), list(solved_neighbors)): int(item_index)
                for item_index, (tile_index, solved_neighbors) in enumerate(frontier_candidates, start=1)
            }
            for future in as_completed(future_map):
                done_index, row = future.result()
                ordered_frontier_rows[int(done_index) - 1] = row
    initial_frontier_rows = [row for row in ordered_frontier_rows if isinstance(row, TileResultRow)]

    _emit_step7_progress(
        progress_cb,
        {
            "mode": "frontier",
            "stage": "graph_solve",
            "tile_count": int(len(initial_frontier_rows)),
            "progress_percent": 74,
            "message": f"frontier | graph solve | frontier tiles={int(len(initial_frontier_rows))}",
        },
    )
    graph_edges, column_model = _build_frontier_graph_edges(
        frontier_rows=initial_frontier_rows,
        solved_rows_by_idx=solved_rows_by_idx,
        neighbor_map=neighbor_map,
    )
    optimized_shifts, node_residuals, edge_residuals, optimized_column_shifts = _solve_frontier_translation_subgraph(
        frontier_rows=initial_frontier_rows,
        edges=graph_edges,
    )

    _emit_step7_progress(
        progress_cb,
        {
            "mode": "frontier",
            "stage": "refresh_setup",
            "tile_count": int(len(initial_frontier_rows)),
            "progress_percent": 78,
            "message": "frontier | refreshing optimized tile QC",
        },
    )

    def _refresh_frontier_row(item_index: int, base_row: TileResultRow) -> tuple[int, TileResultRow]:
        tile_index = int(base_row.tile_index)
        final_dx, final_dy = optimized_shifts.get(tile_index, (float(base_row.final_dx_px), float(base_row.final_dy_px)))
        tile_profile = str(base_row.registration_profile or cfg.registration_input_profile or STEP7_REGISTRATION_INPUT_PROFILE)
        margin_px = max(
            24,
            int(cfg.search_radius_px)
            + max(
                abs(int(round(base_row.pred_dx_px))),
                abs(int(round(base_row.pred_dy_px))),
                abs(int(round(final_dx))),
                abs(int(round(final_dy))),
            )
            + 8,
        )
        ctx = _build_tile_eval_context(
            fixed_gray_full=fixed_gray_full,
            fixed_mask_full=fixed_mask_full,
            moving_reg_projection_u8=moving_reg_projection_u8,
            moving_signal_mask_u8=moving_signal_mask_u8,
            fixed_shape_hw=fixed_shape_hw,
            manual_mat=manual_mat,
            tile=tile_by_idx[tile_index],
            margin_px=margin_px,
        )
        pred_eval = _evaluate_tile_shift_from_context(
            ctx=ctx,
            fixed_gray_full=fixed_gray_full,
            fixed_mask_full=fixed_mask_full,
            registration_input_profile=tile_profile,
            dx_px=int(round(base_row.pred_dx_px)),
            dy_px=int(round(base_row.pred_dy_px)),
        )
        meas_eval = _evaluate_tile_shift_from_context(
            ctx=ctx,
            fixed_gray_full=fixed_gray_full,
            fixed_mask_full=fixed_mask_full,
            registration_input_profile=tile_profile,
            dx_px=int(round(base_row.meas_dx_px)),
            dy_px=int(round(base_row.meas_dy_px)),
        )
        final_eval = _evaluate_tile_shift_from_context(
            ctx=ctx,
            fixed_gray_full=fixed_gray_full,
            fixed_mask_full=fixed_mask_full,
            registration_input_profile=tile_profile,
            dx_px=int(round(final_dx)),
            dy_px=int(round(final_dy)),
        )
        current_cc = float(pred_eval["metrics"].get("cc", float("nan")))
        final_cc = float(final_eval["metrics"].get("cc", float("nan")))
        tile_state = TileState.FRONTIER.value if np.isfinite(final_cc) and (not np.isfinite(current_cc) or final_cc >= current_cc) else TileState.HOLD.value
        proposal_gate = str(base_row.proposal_gate)
        if int(round(final_dx)) != int(round(base_row.final_dx_px)) or int(round(final_dy)) != int(round(base_row.final_dy_px)):
            proposal_gate = "graph_candidate"
        graph_row = _build_tile_result_row_from_evals(
            tile=tile_by_idx[tile_index],
            ctx=ctx,
            pred_eval=pred_eval,
            meas_eval=meas_eval,
            final_eval=final_eval,
            template_match_score=float(base_row.template_match_score),
            neighbor_count=int(base_row.neighbor_count),
            neighbor_labels=list(base_row.neighbor_labels),
            neighbor_shift_spread_px=float(base_row.neighbor_shift_spread_px),
            tile_state=tile_state,
            proposal_gate=proposal_gate,
            signal_coverage=float(base_row.signal_coverage),
            moving_edge_density=float(base_row.moving_edge_density),
            fixed_edge_density=float(base_row.fixed_edge_density),
            edge_density_mean=float(base_row.edge_density_mean),
            density_regime=str(base_row.density_regime),
            profile_candidates=list(base_row.profile_candidates),
            coarse_search_profiles=list(base_row.coarse_search_profiles),
            refine_profiles=list(base_row.refine_profiles),
            refine_objective=str(base_row.refine_objective or "cc"),
        )
        row = _gate_frontier_graph_result(
            base_row=base_row,
            graph_row=graph_row,
        )
        row.graph_residual = float(node_residuals.get(tile_index, 0.0))
        _emit_step7_progress(
            progress_cb,
            {
                "mode": "frontier",
                "stage": "refresh_tile_done",
                "tile_index": int(tile_index),
                "tile_label": str(base_row.label),
                "item_index": int(item_index),
                "total_items": int(max(1, len(initial_frontier_rows))),
                "progress_percent": _step7_progress_percent(
                    80.0,
                    12.0,
                    float(item_index) / float(max(1, len(initial_frontier_rows))),
                ),
                "message": f"frontier | refresh {int(item_index)}/{int(max(1, len(initial_frontier_rows)))} {str(base_row.label)}",
            },
        )
        return int(item_index), row

    refresh_worker_count = _step7_tile_eval_worker_count(max_items=max(1, len(initial_frontier_rows)))
    ordered_ranked_rows: list[TileResultRow | None] = [None] * int(len(initial_frontier_rows))
    if refresh_worker_count <= 1:
        for item_index, base_row in enumerate(initial_frontier_rows, start=1):
            done_index, row = _refresh_frontier_row(int(item_index), base_row)
            ordered_ranked_rows[int(done_index) - 1] = row
    else:
        with ThreadPoolExecutor(max_workers=int(refresh_worker_count), thread_name_prefix="step7-frontier-refresh") as pool:
            future_map = {
                pool.submit(_refresh_frontier_row, int(item_index), base_row): int(item_index)
                for item_index, base_row in enumerate(initial_frontier_rows, start=1)
            }
            for future in as_completed(future_map):
                done_index, row = future.result()
                ordered_ranked_rows[int(done_index) - 1] = row
    ranked_rows = [row for row in ordered_ranked_rows if isinstance(row, TileResultRow)]

    refreshed_by_idx = {int(row.tile_index): row for row in ranked_rows}
    for row in ranked_rows:
        neighbor_diffs: list[float] = []
        for neighbor_idx in neighbor_map.get(int(row.tile_index), []):
            neighbor_row = solved_rows_by_idx.get(int(neighbor_idx)) or refreshed_by_idx.get(int(neighbor_idx))
            if neighbor_row is None:
                continue
            neighbor_diffs.append(
                float(
                    np.hypot(
                        float(row.final_dx_px) - float(neighbor_row.final_dx_px),
                        float(row.final_dy_px) - float(neighbor_row.final_dy_px),
                    )
                )
            )
        row.neighbor_agreement_score = float(1.0 / (1.0 + float(np.mean(neighbor_diffs)))) if neighbor_diffs else float("nan")
        row.frontier_confidence = _frontier_confidence_score(
            current_cc=float(row.current_cc),
            proposal_cc=float(row.final_cc),
            prior_dx=int(round(row.pred_dx_px)),
            prior_dy=int(round(row.pred_dy_px)),
            final_dx=int(round(row.final_dx_px)),
            final_dy=int(round(row.final_dy_px)),
            neighbor_count=int(row.neighbor_count),
            neighbor_shift_spread_px=float(row.neighbor_shift_spread_px),
        )
        if np.isfinite(row.neighbor_agreement_score):
            row.frontier_confidence += 0.25 * float(row.neighbor_agreement_score)
        row.frontier_confidence -= 0.10 * float(row.graph_residual)

    ranked_rows.sort(
        key=lambda row: (
            0 if str(row.tile_state) == TileState.FRONTIER.value else 1,
            -float(row.frontier_confidence if np.isfinite(row.frontier_confidence) else -float("inf")),
            -float(row.final_cc if np.isfinite(row.final_cc) else -float("inf")),
        )
    )

    panel_keys = ("moving", "fixed", "overlay", "heatmap")
    if ranked_rows:
        target_h = max(int(np.asarray(getattr(row, key)).shape[0]) for row in ranked_rows for key in panel_keys if isinstance(getattr(row, key), np.ndarray))
        target_w = max(int(np.asarray(getattr(row, key)).shape[1]) for row in ranked_rows for key in panel_keys if isinstance(getattr(row, key), np.ndarray))
        for row in ranked_rows:
            for key in panel_keys:
                panel = getattr(row, key)
                if not isinstance(panel, np.ndarray):
                    continue
                if panel.shape[:2] != (target_h, target_w):
                    setattr(row, key, cv2.resize(panel, (target_w, target_h), interpolation=cv2.INTER_LINEAR))

    storyboard_rows = []
    for rank, row in enumerate(ranked_rows[: max(1, min(int(cfg.top_k_storyboard), len(ranked_rows) if ranked_rows else 1))], start=1):
        storyboard_rows.append(
            {
                "label": f"{row.label} | rank={rank}",
                "note": (
                    f"state={row.tile_state} | "
                    f"frontier={float(row.frontier_confidence):.4f} | "
                    f"CC {float(row.current_cc):.4f}->{float(row.final_cc):.4f} | "
                    f"pred=({int(round(row.pred_dx_px))},{int(round(row.pred_dy_px))}) | "
                    f"shift=({int(round(row.final_dx_px))},{int(round(row.final_dy_px))}) | "
                    f"neighbors={int(row.neighbor_count)}"
                ),
                "moving": row.moving,
                "fixed": row.fixed,
                "overlay": row.overlay,
                "heatmap": row.heatmap,
                "col_titles": row.col_titles,
            }
        )
    storyboard_path = run_dir / "frontier_storyboard.png"
    if storyboard_rows:
        _emit_step7_progress(
            progress_cb,
            {
                "mode": "frontier",
                "stage": "storyboard",
                "progress_percent": 96,
                "message": "frontier | rendering storyboard",
            },
        )
        render_storyboard(storyboard_rows, storyboard_path)

    csv_path = process_dir / "frontier_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "rank",
                "tile_index",
                "label",
                "tile_state",
                "row_display",
                "col_display",
                "current_cc",
                "meas_cc",
                "shifted_cc",
                "shift_gain_cc",
                "pred_dx_px",
                "pred_dy_px",
                "meas_dx_px",
                "meas_dy_px",
                "prior_shift_dx_px",
                "prior_shift_dy_px",
                "best_shift_dx_px",
                "best_shift_dy_px",
                "delta_from_prior_dx_px",
                "delta_from_prior_dy_px",
                "delta_from_prior_norm_px",
                "neighbor_count",
                "neighbor_shift_spread_px",
                "neighbor_agreement_score",
                "graph_residual",
                "frontier_confidence",
                "density_regime",
                "signal_coverage",
                "moving_edge_density",
                "fixed_edge_density",
                "edge_density_mean",
                "registration_profile",
                "profile_candidates_str",
                "coarse_search_profiles_str",
                "refine_profiles_str",
                "refine_objective",
            ],
        )
        writer.writeheader()
        for rank, row in enumerate(ranked_rows, start=1):
            row_dict = row.to_dict()
            row_dict["rank"] = int(rank)
            writer.writerow({key: row_dict.get(key) for key in writer.fieldnames})

    top_frontier = [
        {
            "rank": int(rank),
            "tile_index": int(row.tile_index),
            "label": str(row.label),
            "tile_state": str(row.tile_state),
            "frontier_confidence": float(row.frontier_confidence),
            "current_cc": float(row.current_cc),
            "shifted_cc": float(row.final_cc),
            "prior_shift_dx_px": int(round(row.pred_dx_px)),
            "prior_shift_dy_px": int(round(row.pred_dy_px)),
            "best_shift_dx_px": int(round(row.final_dx_px)),
            "best_shift_dy_px": int(round(row.final_dy_px)),
            "neighbor_count": int(row.neighbor_count),
            "density_regime": str(row.density_regime),
            "registration_profile": str(row.registration_profile),
            "refine_objective": str(row.refine_objective),
        }
        for rank, row in enumerate(ranked_rows[: max(1, int(cfg.max_frontier_tiles))], start=1)
    ]
    manifest = {
        "myelin_label": cfg.myelin_label,
        "section_dir": str(cfg.myelin_section_dir),
        "run_dir": str(run_dir),
        "storyboard_path": str(storyboard_path) if storyboard_rows else "",
        "registration_input_profile": str(cfg.registration_input_profile or STEP7_REGISTRATION_INPUT_PROFILE),
        "projection_mode": str(cfg.projection_mode),
        "channel_index": int(cfg.channel_index),
        "confocal_source_mode": str(cfg.confocal_source_mode),
        "confocal_sources": [str(path) for path in cfg.confocal_sources],
        "manual_init": {
            "tx_px": float(cfg.tx_px),
            "ty_px": float(cfg.ty_px),
            "angle_deg": float(cfg.angle_deg),
            "scale": float(cfg.scale),
            "flip_lr": bool(cfg.flip_lr),
            "flip_ud": bool(cfg.flip_ud),
            "anchor_guided": anchor_info,
        },
        "graph_state": {
            "selected_tile_index": None if cfg.selected_tile_index is None else int(cfg.selected_tile_index),
            "accepted_tile_indices": sorted(int(v) for v in accepted_tile_indices),
            "frozen_tile_indices": sorted(int(v) for v in frozen_tile_indices),
            "solved_tile_indices": sorted(int(v) for v in solved_tile_indices),
            "frontier_tile_indices": sorted(int(v) for v in frontier_tile_indices),
            "residual_model": "translation_subgraph_ls_column_shared" if column_model else "translation_subgraph_ls",
            "search_radius_px": int(cfg.search_radius_px),
            "column_target_priors": [
                {
                    "column_index": int(col),
                    "target_dx_px": float(info.get("target_dx_px", 0.0)),
                    "target_dy_px": float(info.get("target_dy_px", 0.0)),
                    "source": str(info.get("source") or ""),
                    "frontier_tile_indices": [int(v) for v in list(info.get("frontier_tile_indices") or [])],
                    "solved_tile_indices": [int(v) for v in list(info.get("solved_tile_indices") or [])],
                }
                for col, info in sorted(column_model.items())
            ],
            "column_latent_shifts": [
                {
                    "column_index": int(col),
                    "dx_px": float(optimized_column_shifts[int(col)][0]),
                    "dy_px": float(optimized_column_shifts[int(col)][1]),
                }
                for col in sorted(optimized_column_shifts)
            ],
        },
        "top_frontier_candidates": top_frontier,
        "edge_residuals": edge_residuals,
        "files": {
            "storyboard": str(storyboard_path) if storyboard_rows else "",
            "frontier_metrics_csv": str(csv_path),
            "manifest": str(run_dir / "frontier_manifest.json"),
        },
    }
    _write_json(run_dir / "frontier_manifest.json", manifest)
    _emit_step7_progress(
        progress_cb,
        {
            "mode": "frontier",
            "stage": "done",
            "progress_percent": 100,
            "message": f"frontier | finished | candidates={int(len(ranked_rows))}",
        },
    )
    return manifest | {
        "rows": [row.to_dict() | {"rank": int(rank)} for rank, row in enumerate(ranked_rows, start=1)],
        "solved_rows": [row.to_dict() for row in solved_rows],
    }


def run_confocal_rigid_registration(cfg: ConfocalRigidConfig) -> dict[str, Any]:
    refine_model = str(getattr(cfg, "local_refine_model", "similarity") or "similarity").strip().lower()
    refine_model = refine_model if refine_model in {"rigid", "similarity", "affine"} else "similarity"
    refine_label = {"rigid": "Rigid", "similarity": "Similarity", "affine": "Affine"}[refine_model]
    session_id = f"{_utc_stamp()}_{cfg.projection_mode}_ch{cfg.channel_index}"
    run_dir = cfg.out_root / cfg.myelin_label / session_id
    inputs_dir = run_dir / "inputs"
    stage_dir = run_dir / "local_refine"
    run_dir.mkdir(parents=True, exist_ok=True)
    inputs_dir.mkdir(parents=True, exist_ok=True)
    stage_dir.mkdir(parents=True, exist_ok=True)

    fixed_rgb = cfg.myelin_rgb.copy()
    fixed_gray_full = rgb_to_gray_float(fixed_rgb)
    fixed_mask_full = (cfg.myelin_labels == 1).astype(np.float32)
    if not np.any(fixed_mask_full > 0):
        fixed_mask_full = (cfg.myelin_labels > 0).astype(np.float32)

    moving_signal_mask_u8 = (
        np.where(np.asarray(cfg.confocal_signal_mask_u8) > 0, 255, 0).astype(np.uint8)
        if cfg.confocal_signal_mask_u8 is not None
        else np.where(np.asarray(cfg.confocal_projection_u8) > 0, 255, 0).astype(np.uint8)
    )
    moving_reg_projection_u8 = (
        _invert_confocal_u8(cfg.confocal_projection_u8)
        if bool(cfg.invert_confocal_for_registration)
        else np.asarray(cfg.confocal_projection_u8, dtype=np.uint8)
    )

    base_manual_mat = build_manual_affine(
        moving_reg_projection_u8.shape[:2],
        fixed_rgb.shape[:2],
        tx_px=cfg.tx_px,
        ty_px=cfg.ty_px,
        angle_deg=cfg.angle_deg,
        scale=cfg.scale,
        flip_lr=cfg.flip_lr,
        flip_ud=cfg.flip_ud,
    )
    anchor_init_mat, anchor_init_info = _anchor_guided_manual_affine(
        moving_shape_hw=moving_reg_projection_u8.shape[:2],
        fixed_shape_hw=fixed_rgb.shape[:2],
        current_mat=base_manual_mat,
        anchor_pairs=list(cfg.anchor_pairs or []),
        flip_lr=cfg.flip_lr,
        flip_ud=cfg.flip_ud,
    )
    manual_gray_u8, manual_mask_u8, manual_mat = apply_affine_matrix(
        moving_reg_projection_u8,
        fixed_rgb.shape[:2],
        moving_mask_u8=moving_signal_mask_u8,
        mat=anchor_init_mat,
    )
    moving_gray_full = manual_gray_u8.astype(np.float32) / 255.0
    moving_mask_full = (manual_mask_u8 > 0).astype(np.float32)
    registration_inputs_full = _prepare_step7_registration_inputs(
        fixed_gray_full,
        moving_gray_full,
        fixed_mask_full,
        moving_mask_full,
        profile=str(getattr(cfg, "registration_input_profile", STEP7_REGISTRATION_INPUT_PROFILE) or STEP7_REGISTRATION_INPUT_PROFILE),
    )
    fixed_reg_gray_full = np.asarray(registration_inputs_full["fixed_gray"], dtype=np.float32)
    moving_reg_gray_full = np.asarray(registration_inputs_full["moving_gray"], dtype=np.float32)

    roi_bbox = _bbox_from_mask(manual_mask_u8, margin_px=max(96, min(fixed_rgb.shape[:2]) // 16))
    y0, y1, x0, x1 = roi_bbox
    fixed_local_rgb = fixed_rgb[y0:y1, x0:x1].copy()
    fixed_local_labels = cfg.myelin_labels[y0:y1, x0:x1].copy()
    fixed_local_info = {
        "mode": "preview_working_grid",
        "preview_roi_bbox_yxyx": [int(y0), int(y1), int(x0), int(x1)],
        "working_um_per_px_xy": list((cfg.myelin_fixed_info or {}).get("preview_um_per_px_xy") or [float(cfg.target_working_um_per_px), float(cfg.target_working_um_per_px)]),
    }
    fixed_gray = rgb_to_gray_float(fixed_local_rgb)
    fixed_mask = (fixed_local_labels == 1).astype(np.float32)
    if not np.any(fixed_mask > 0):
        fixed_mask = (fixed_local_labels > 0).astype(np.float32)
    moving_gray = moving_gray_full[y0:y1, x0:x1]
    moving_mask = moving_mask_full[y0:y1, x0:x1]
    fixed_reg_gray = fixed_reg_gray_full[y0:y1, x0:x1]
    moving_reg_gray = moving_reg_gray_full[y0:y1, x0:x1]
    if moving_gray.shape[:2] != fixed_gray.shape[:2]:
        moving_gray = cv2.resize(moving_gray, (fixed_gray.shape[1], fixed_gray.shape[0]), interpolation=cv2.INTER_LINEAR)
        moving_mask = cv2.resize(moving_mask, (fixed_gray.shape[1], fixed_gray.shape[0]), interpolation=cv2.INTER_NEAREST)
        moving_mask = (moving_mask > 0.5).astype(np.float32)
        moving_reg_gray = cv2.resize(moving_reg_gray, (fixed_gray.shape[1], fixed_gray.shape[0]), interpolation=cv2.INTER_LINEAR)
    if fixed_reg_gray.shape[:2] != fixed_gray.shape[:2]:
        fixed_reg_gray = cv2.resize(fixed_reg_gray, (fixed_gray.shape[1], fixed_gray.shape[0]), interpolation=cv2.INTER_LINEAR)

    input_metrics, input_metric_timings = compute_registration_metrics(fixed_reg_gray, moving_reg_gray, fixed_mask, moving_mask)
    input_note = metrics_note(
        input_metrics,
        input_metric_timings,
        (
            f"manual init local-ROI @ {float(cfg.target_working_um_per_px):.1f} um/px | "
            f"metric=CC | confocal={'inverted' if cfg.invert_confocal_for_registration else 'native'} | "
            f"reg_input={registration_inputs_full['profile']} | next={refine_label}"
        ),
    )

    cv2.imwrite(str(inputs_dir / "myelin_fixed.png"), cv2.cvtColor(fixed_rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(inputs_dir / "confocal_projection_native.png"), cfg.confocal_projection_u8)
    cv2.imwrite(str(inputs_dir / "confocal_projection_registration.png"), moving_reg_projection_u8)
    cv2.imwrite(str(inputs_dir / "confocal_manual_warped.png"), manual_gray_u8)
    cv2.imwrite(str(inputs_dir / "myelin_labels.png"), cfg.myelin_labels)
    cv2.imwrite(str(inputs_dir / "confocal_manual_mask.png"), manual_mask_u8)
    cv2.imwrite(str(inputs_dir / "myelin_fixed_local_roi.png"), np.clip(np.round(fixed_gray * 255.0), 0, 255).astype(np.uint8))
    cv2.imwrite(str(inputs_dir / "confocal_manual_local_roi.png"), np.clip(np.round(moving_gray * 255.0), 0, 255).astype(np.uint8))
    cv2.imwrite(str(inputs_dir / "myelin_fixed_local_roi_processed.png"), np.asarray(registration_inputs_full["fixed_u8"])[y0:y1, x0:x1])
    cv2.imwrite(str(inputs_dir / "confocal_manual_local_roi_processed.png"), np.asarray(registration_inputs_full["moving_u8"])[y0:y1, x0:x1])

    fixed_img_path = inputs_dir / "fixed_gray.nii.gz"
    moving_img_path = inputs_dir / "moving_gray.nii.gz"
    moving_raw_img_path = inputs_dir / "moving_gray_raw.nii.gz"
    fixed_mask_path = inputs_dir / "fixed_mask.nii.gz"
    moving_mask_path = inputs_dir / "moving_mask.nii.gz"
    write_nifti_2d(fixed_img_path, fixed_reg_gray)
    write_nifti_2d(moving_img_path, moving_reg_gray)
    write_nifti_2d(moving_raw_img_path, moving_gray)
    write_nifti_2d(fixed_mask_path, fixed_mask)
    write_nifti_2d(moving_mask_path, moving_mask)
    moving_coord_x, moving_coord_y = _write_coord_images(inputs_dir, moving_gray.shape[:2])

    initial_zoom_bbox = _bbox_from_union_masks(
        [moving_mask],
        fallback_shape_hw=fixed_gray.shape[:2],
        margin_px=max(24, min(fixed_gray.shape[:2]) // 10),
    )
    storyboard_path = run_dir / "quick_qc_storyboard.png"
    render_storyboard(
        [
            _fiber_qc_row(
                label="Input",
                note=input_note,
                fixed_gray=fixed_gray,
                moving_gray=moving_gray,
                fixed_mask=fixed_mask,
                moving_mask=moving_mask,
                zoom_bbox=initial_zoom_bbox,
                out_shape_hw=fixed_gray.shape[:2],
            )
        ],
        storyboard_path,
    )

    t0 = time.perf_counter()
    prefix = stage_dir / "local_"
    cmd = _stage_command_local_refine(
        cfg.ants_bin,
        fixed_img_path,
        moving_img_path,
        fixed_mask_path,
        moving_mask_path,
        prefix,
        refine_model=refine_model,
    )
    ants_t0 = time.perf_counter()
    _run_logged(cmd, stage_dir / "local_refine.log")
    ants_seconds = float(time.perf_counter() - ants_t0)

    local_mat = stage_dir / "local_0GenericAffine.mat"
    effective_local_mat = local_mat
    anchor_refine_info = {"used": False}
    if bool(anchor_init_info.get("used")) and int(anchor_init_info.get("pair_count") or 0) == 1:
        pair = (anchor_init_info.get("pairs") or [{}])[0]
        scene_xy = pair.get("section_scene_xy")
        if isinstance(scene_xy, (list, tuple)) and len(scene_xy) == 2:
            adjusted_mat = stage_dir / "local_anchor_preserved_0GenericAffine.mat"
            anchor_refine_info = _preserve_single_anchor_in_local_transform(
                local_mat,
                anchor_scene_xy=(float(scene_xy[0]), float(scene_xy[1])),
                local_roi_bbox_yxyx=roi_bbox,
                out_path=adjusted_mat,
            )
            effective_local_mat = adjusted_mat
    warped_img_path = stage_dir / "local_Warped.nii.gz"
    warped_raw_img_path = stage_dir / "local_Warped_raw.nii.gz"
    warped_mask_path = stage_dir / "local_warped_mask.nii.gz"
    subprocess.run(
        [
            str(ants_binary_path(cfg.ants_bin, "antsApplyTransforms")),
            "-d",
            "2",
            "-i",
            ants_cli_path(moving_img_path),
            "-r",
            ants_cli_path(fixed_img_path),
            "-o",
            ants_cli_path(warped_img_path),
            "-n",
            "Linear",
            "-t",
            ants_cli_path(effective_local_mat),
        ],
        check=True,
        stdout=(stage_dir / "local_warp_image.log").open("w", encoding="utf-8"),
        stderr=subprocess.STDOUT,
    )
    subprocess.run(
        [
            str(ants_binary_path(cfg.ants_bin, "antsApplyTransforms")),
            "-d",
            "2",
            "-i",
            ants_cli_path(moving_raw_img_path),
            "-r",
            ants_cli_path(fixed_img_path),
            "-o",
            ants_cli_path(warped_raw_img_path),
            "-n",
            "Linear",
            "-t",
            ants_cli_path(effective_local_mat),
        ],
        check=True,
        stdout=(stage_dir / "local_warp_raw_image.log").open("w", encoding="utf-8"),
        stderr=subprocess.STDOUT,
    )
    subprocess.run(
        [
            str(ants_binary_path(cfg.ants_bin, "antsApplyTransforms")),
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
            ants_cli_path(effective_local_mat),
        ],
        check=True,
        stdout=(stage_dir / "local_warp_mask.log").open("w", encoding="utf-8"),
        stderr=subprocess.STDOUT,
    )

    warped_gray = read_nifti_2d(warped_img_path)
    warped_raw_gray = read_nifti_2d(warped_raw_img_path)
    warped_mask = read_nifti_2d(warped_mask_path)
    warped_gray_for_full = np.clip(warped_gray, 0.0, 1.0)
    warped_raw_gray_for_full = np.clip(warped_raw_gray, 0.0, 1.0)
    warped_mask_for_full = (warped_mask > 0.5).astype(np.float32)
    preview_roi_h = max(1, int(y1 - y0))
    preview_roi_w = max(1, int(x1 - x0))
    if warped_gray_for_full.shape[:2] != (preview_roi_h, preview_roi_w):
        warped_gray_for_full = cv2.resize(warped_gray_for_full, (preview_roi_w, preview_roi_h), interpolation=cv2.INTER_LINEAR)
        warped_raw_gray_for_full = cv2.resize(warped_raw_gray_for_full, (preview_roi_w, preview_roi_h), interpolation=cv2.INTER_LINEAR)
        warped_mask_for_full = cv2.resize(warped_mask_for_full, (preview_roi_w, preview_roi_h), interpolation=cv2.INTER_NEAREST)
        warped_mask_for_full = (warped_mask_for_full > 0.5).astype(np.float32)
    warped_gray_full = _paste_crop(fixed_gray_full.shape[:2], warped_gray_for_full, roi_bbox, fill_value=1.0)
    warped_raw_gray_full = _paste_crop(fixed_gray_full.shape[:2], warped_raw_gray_for_full, roi_bbox, fill_value=0.0)
    warped_mask_full = _paste_crop(fixed_gray_full.shape[:2], warped_mask_for_full, roi_bbox, fill_value=0.0)
    full_input_metrics, full_input_metric_timings = compute_registration_metrics(
        fixed_reg_gray_full,
        moving_reg_gray_full,
        fixed_mask_full,
        moving_mask_full,
    )
    refine_metrics, refine_metric_timing = compute_registration_metrics(
        fixed_reg_gray,
        np.clip(warped_gray, 0.0, 1.0),
        fixed_mask,
        (warped_mask > 0.5).astype(np.float32),
    )
    full_refine_metrics, full_refine_metric_timing = compute_registration_metrics(
        fixed_reg_gray_full,
        np.clip(warped_gray_full, 0.0, 1.0),
        fixed_mask_full,
        (warped_mask_full > 0.5).astype(np.float32),
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
        refine_model,
        fixed_img_path,
        fixed_mask,
        moving_coord_x,
        moving_coord_y,
        [effective_local_mat],
        warped_mask_path,
    )
    total_seconds = float(time.perf_counter() - t0)

    fixed_gray_full_u8 = np.clip(np.round(fixed_gray_full * 255.0), 0, 255).astype(np.uint8)
    fixed_mask_full_u8 = np.where(fixed_mask_full > 0, 255, 0).astype(np.uint8)
    moving_gray_full_u8 = np.clip(np.round(moving_gray_full * 255.0), 0, 255).astype(np.uint8)
    fixed_reg_gray_full_u8 = np.clip(np.round(np.clip(fixed_reg_gray_full, 0.0, 1.0) * 255.0), 0, 255).astype(np.uint8)
    moving_reg_gray_full_u8 = np.clip(np.round(np.clip(moving_reg_gray_full, 0.0, 1.0) * 255.0), 0, 255).astype(np.uint8)
    warped_gray_full_u8 = np.clip(np.round(np.clip(warped_gray_full, 0.0, 1.0) * 255.0), 0, 255).astype(np.uint8)
    warped_raw_gray_full_u8 = np.clip(np.round(np.clip(warped_raw_gray_full, 0.0, 1.0) * 255.0), 0, 255).astype(np.uint8)
    moving_mask_full_u8 = np.where(moving_mask_full > 0, 255, 0).astype(np.uint8)
    warped_mask_full_u8 = np.where(warped_mask_full > 0.5, 255, 0).astype(np.uint8)
    cv2.imwrite(str(inputs_dir / "myelin_fixed_full_gray.png"), fixed_gray_full_u8)
    cv2.imwrite(str(inputs_dir / "myelin_fixed_full_reg_gray.png"), fixed_reg_gray_full_u8)
    cv2.imwrite(str(inputs_dir / "myelin_fixed_full_mask.png"), fixed_mask_full_u8)
    cv2.imwrite(str(inputs_dir / "confocal_manual_full_gray.png"), moving_gray_full_u8)
    cv2.imwrite(str(inputs_dir / "confocal_manual_full_reg_gray.png"), moving_reg_gray_full_u8)
    cv2.imwrite(str(inputs_dir / "confocal_manual_full_mask.png"), moving_mask_full_u8)
    cv2.imwrite(str(stage_dir / "local_warped_full_gray.png"), warped_gray_full_u8)
    cv2.imwrite(str(stage_dir / "local_warped_full_raw_gray.png"), warped_raw_gray_full_u8)
    cv2.imwrite(str(stage_dir / "local_warped_full_mask.png"), warped_mask_full_u8)

    coarse_alignment_overlay_full = overlay_preview(
        fixed_gray_full,
        moving_gray_full,
        fixed_mask_full,
        moving_mask_full,
    )
    coarse_alignment_overlay_local = overlay_preview(
        fixed_gray,
        moving_gray,
        fixed_mask,
        moving_mask,
    )
    cv2.imwrite(
        str(inputs_dir / "coarse_alignment_overlay_full.png"),
        cv2.cvtColor(coarse_alignment_overlay_full, cv2.COLOR_RGB2BGR),
    )
    cv2.imwrite(
        str(inputs_dir / "coarse_alignment_overlay_local.png"),
        cv2.cvtColor(coarse_alignment_overlay_local, cv2.COLOR_RGB2BGR),
    )

    zoom_bbox = _bbox_from_union_masks(
        [moving_mask, (warped_mask > 0.5).astype(np.float32)],
        fallback_shape_hw=fixed_gray.shape[:2],
        margin_px=max(24, min(fixed_gray.shape[:2]) // 10),
    )
    render_storyboard(
        [
            {
                "label": "Before",
                "note": input_note,
                "moving": _crop_gray_panel(
                    moving_gray,
                    zoom_bbox,
                    support_mask=moving_mask,
                    fill_outside_support=1.0,
                    out_shape_hw=fixed_gray.shape[:2],
                ),
                "fixed": _crop_gray_panel(fixed_gray, zoom_bbox, support_mask=fixed_mask, out_shape_hw=fixed_gray.shape[:2]),
                "overlay": _crop_overlay_panel(fixed_gray, moving_gray, fixed_mask, moving_mask, zoom_bbox, out_shape_hw=fixed_gray.shape[:2]),
                "heatmap": _crop_overlay_panel(fixed_reg_gray, moving_reg_gray, fixed_mask, moving_mask, zoom_bbox, out_shape_hw=fixed_gray.shape[:2]),
                "col_titles": ("Confocal raw", "Myelin raw", "Raw overlay", "Registration input overlay"),
            },
            {
                "label": f"After ({refine_label})",
                "note": metrics_note(
                    refine_metrics,
                    refine_metric_timing,
                    (
                        f"{refine_model} local-CC @ {float(cfg.target_working_um_per_px):.1f} um/px | "
                        f"confocal={'inverted' if cfg.invert_confocal_for_registration else 'native'} | "
                        f"reg_input={registration_inputs_full['profile']} | "
                        f"anchor_lock={'on' if bool(anchor_refine_info.get('used')) else 'off'}"
                    ),
                ),
                "moving": _crop_gray_panel(
                    np.clip(warped_raw_gray, 0.0, 1.0),
                    zoom_bbox,
                    support_mask=(warped_mask > 0.5).astype(np.float32),
                    fill_outside_support=1.0,
                    out_shape_hw=fixed_gray.shape[:2],
                ),
                "fixed": _crop_gray_panel(fixed_gray, zoom_bbox, support_mask=fixed_mask, out_shape_hw=fixed_gray.shape[:2]),
                "overlay": _crop_overlay_panel(
                    fixed_gray,
                    np.clip(warped_raw_gray, 0.0, 1.0),
                    fixed_mask,
                    (warped_mask > 0.5).astype(np.float32),
                    zoom_bbox,
                    out_shape_hw=fixed_gray.shape[:2],
                ),
                "heatmap": _crop_overlay_panel(
                    fixed_reg_gray,
                    np.clip(warped_gray, 0.0, 1.0),
                    fixed_mask,
                    (warped_mask > 0.5).astype(np.float32),
                    zoom_bbox,
                    out_shape_hw=fixed_gray.shape[:2],
                ),
                "col_titles": ("Confocal raw", "Myelin raw", "Raw overlay", "Registration input overlay"),
            },
        ],
        storyboard_path,
    )

    manifest = {
        "myelin_label": cfg.myelin_label,
        "run_dir": str(run_dir),
        "storyboard_path": str(storyboard_path),
        "confocal_sources": [str(path) for path in cfg.confocal_sources],
        "confocal_source_mode": cfg.confocal_source_mode,
        "nominal_overlap_fraction": float(cfg.nominal_overlap_fraction),
        "projection_info": cfg.projection_info or {},
        "projection_mode": cfg.projection_mode,
        "channel_index": int(cfg.channel_index),
        "local_refine_model": refine_model,
        "registration_input_profile": registration_inputs_full["profile"],
        "registration_input_description": registration_inputs_full["description"],
        "source_um_per_px_xy": list((cfg.projection_info or {}).get("source_um_per_px_xy") or []),
        "fixed_preview_um_per_px_xy": list((cfg.projection_info or {}).get("target_um_per_px_xy") or []),
        "confocal_projection_scaled_shape_hw": [int(cfg.confocal_projection_u8.shape[0]), int(cfg.confocal_projection_u8.shape[1])],
        "fixed_preview_shape_hw": [int(fixed_rgb.shape[0]), int(fixed_rgb.shape[1])],
        "target_working_um_per_px": float(cfg.target_working_um_per_px),
        "invert_confocal_for_registration": bool(cfg.invert_confocal_for_registration),
        "saved_at_utc": _utc_iso(),
        "manual_init": {
            "tx_px": float(cfg.tx_px),
            "ty_px": float(cfg.ty_px),
            "angle_deg": float(cfg.angle_deg),
            "scale": float(cfg.scale),
            "flip_lr": bool(cfg.flip_lr),
            "flip_ud": bool(cfg.flip_ud),
            "base_affine_matrix_2x3": base_manual_mat.tolist(),
            "affine_matrix_2x3": manual_mat.tolist(),
        },
        "manual_anchor_mode": anchor_init_info,
        "anchor_preserving_refine": anchor_refine_info,
        "local_registration": {
            "strategy": (
                f"manual-init ROI -> local {refine_model} refinement"
                + (" -> single-anchor translation lock" if bool(anchor_refine_info.get("used")) else "")
            ),
            "transform_model": refine_model,
            "metric": "CC",
            "registration_input_profile": registration_inputs_full["profile"],
            "registration_input_description": registration_inputs_full["description"],
            "working_um_per_px": float(cfg.target_working_um_per_px),
            "confocal_polarity_for_registration": "inverted" if cfg.invert_confocal_for_registration else "native",
            "roi_bbox_yxyx": [int(y0), int(y1), int(x0), int(x1)],
            "roi_shape_hw": [int(fixed_gray.shape[0]), int(fixed_gray.shape[1])],
            "fiber_qc_zoom_bbox_yxyx": [int(v) for v in zoom_bbox],
            "fixed_local_patch": fixed_local_info,
        },
        "coarse_alignment_record": {
            "reproducible_from_run_dir": True,
            "whole_slice_fixed_shape_hw": [int(fixed_gray_full.shape[0]), int(fixed_gray_full.shape[1])],
            "whole_slice_confocal_shape_hw": [int(moving_gray_full.shape[0]), int(moving_gray_full.shape[1])],
            "whole_slice_confocal_mask_pixels": int(np.count_nonzero(moving_mask_full > 0)),
            "local_roi_bbox_yxyx": [int(y0), int(y1), int(x0), int(x1)],
            "local_roi_shape_hw": [int(fixed_gray.shape[0]), int(fixed_gray.shape[1])],
            "fiber_qc_zoom_bbox_yxyx": [int(v) for v in zoom_bbox],
            "source_um_per_px_xy": list((cfg.projection_info or {}).get("source_um_per_px_xy") or []),
            "fixed_preview_um_per_px_xy": list((cfg.projection_info or {}).get("target_um_per_px_xy") or []),
            "manual_transform": {
                "tx_px": float(cfg.tx_px),
                "ty_px": float(cfg.ty_px),
                "angle_deg": float(cfg.angle_deg),
                "scale": float(cfg.scale),
                "flip_lr": bool(cfg.flip_lr),
                "flip_ud": bool(cfg.flip_ud),
            },
        },
        "input_metrics": input_metrics,
        "input_metric_timing_seconds": input_metric_timings,
        "full_input_metrics": full_input_metrics,
        "full_input_metric_timing_seconds": full_input_metric_timings,
        "refine_metrics": refine_metrics,
        "refine_metric_timing_seconds": refine_metric_timing,
        "full_refine_metrics": full_refine_metrics,
        "full_refine_metric_timing_seconds": full_refine_metric_timing,
        "rigid_metrics": refine_metrics,
        "rigid_metric_timing_seconds": refine_metric_timing,
        "full_rigid_metrics": full_refine_metrics,
        "full_rigid_metric_timing_seconds": full_refine_metric_timing,
        "timing_seconds": {
            "ants_registration": ants_seconds,
            "total": total_seconds,
        },
        "files": {
            "myelin_fixed": str(inputs_dir / "myelin_fixed.png"),
            "myelin_fixed_full_gray": str(inputs_dir / "myelin_fixed_full_gray.png"),
            "myelin_fixed_full_mask": str(inputs_dir / "myelin_fixed_full_mask.png"),
            "myelin_fixed_local_roi": str(inputs_dir / "myelin_fixed_local_roi.png"),
            "confocal_projection": str(inputs_dir / "confocal_projection_registration.png"),
            "confocal_projection_native": str(inputs_dir / "confocal_projection_native.png"),
            "confocal_projection_registration": str(inputs_dir / "confocal_projection_registration.png"),
            "confocal_manual_warped": str(inputs_dir / "confocal_manual_warped.png"),
            "confocal_manual_full_gray": str(inputs_dir / "confocal_manual_full_gray.png"),
            "confocal_manual_full_mask": str(inputs_dir / "confocal_manual_full_mask.png"),
            "confocal_manual_local_roi": str(inputs_dir / "confocal_manual_local_roi.png"),
            "coarse_alignment_overlay_full": str(inputs_dir / "coarse_alignment_overlay_full.png"),
            "coarse_alignment_overlay_local": str(inputs_dir / "coarse_alignment_overlay_local.png"),
            "quick_qc_storyboard": str(storyboard_path),
            "local_refine_heatmap": str(heatmap_png),
            "local_refine_transform": str(effective_local_mat),
            "local_refine_transform_raw": str(local_mat),
            "local_refine_warp_image_log": str(stage_dir / "local_warp_image.log"),
            "local_refine_warp_raw_image_log": str(stage_dir / "local_warp_raw_image.log"),
            "local_refine_warped_full_gray": str(stage_dir / "local_warped_full_gray.png"),
            "local_refine_warped_full_raw_gray": str(stage_dir / "local_warped_full_raw_gray.png"),
            "local_refine_warped_full_mask": str(stage_dir / "local_warped_full_mask.png"),
            "myelin_fixed_local_roi_processed": str(inputs_dir / "myelin_fixed_local_roi_processed.png"),
            "confocal_manual_local_roi_processed": str(inputs_dir / "confocal_manual_local_roi_processed.png"),
            "rigid_heatmap": str(heatmap_png),
            "rigid_transform": str(effective_local_mat),
            "rigid_warp_image_log": str(stage_dir / "local_warp_image.log"),
            "rigid_warped_full_gray": str(stage_dir / "local_warped_full_gray.png"),
            "rigid_warped_full_mask": str(stage_dir / "local_warped_full_mask.png"),
            "manifest": str(run_dir / "session_manifest.json"),
        },
    }
    repro_tracker = {
        "schema": "step7_confocal_repro_tracker_v1",
        "myelin_label": cfg.myelin_label,
        "myelin_section_dir": str(cfg.myelin_section_dir),
        "run_dir": str(run_dir),
        "saved_at_utc": manifest["saved_at_utc"],
        "confocal_sources": [str(path) for path in cfg.confocal_sources],
        "confocal_source_basenames": [path.name for path in cfg.confocal_sources],
        "confocal_source_mode": cfg.confocal_source_mode,
        "projection_mode": cfg.projection_mode,
        "channel_index": int(cfg.channel_index),
        "nominal_overlap_fraction": float(cfg.nominal_overlap_fraction),
        "target_working_um_per_px": float(cfg.target_working_um_per_px),
        "source_um_per_px_xy": manifest.get("source_um_per_px_xy", []),
        "fixed_preview_um_per_px_xy": manifest.get("fixed_preview_um_per_px_xy", []),
        "projection_info": cfg.projection_info or {},
        "fixed_preview_shape_hw": manifest.get("fixed_preview_shape_hw"),
        "confocal_projection_scaled_shape_hw": manifest.get("confocal_projection_scaled_shape_hw"),
        "manual_transform": {
            "tx_px": float(cfg.tx_px),
            "ty_px": float(cfg.ty_px),
            "angle_deg": float(cfg.angle_deg),
            "scale": float(cfg.scale),
            "flip_lr": bool(cfg.flip_lr),
            "flip_ud": bool(cfg.flip_ud),
            "base_affine_matrix_2x3": base_manual_mat.tolist(),
            "effective_manual_affine_matrix_2x3": manual_mat.tolist(),
        },
        "anchor_pairs_explicit": [
            {
                "index": int(pair.get("index", idx + 1)),
                "section_scene_xy": [float((pair.get("section_scene_xy") or [float("nan"), float("nan")])[0]), float((pair.get("section_scene_xy") or [float("nan"), float("nan")])[1])],
                "confocal_scene_xy": [float((pair.get("confocal_scene_xy") or [float("nan"), float("nan")])[0]), float((pair.get("confocal_scene_xy") or [float("nan"), float("nan")])[1])],
                "confocal_raw_xy": [float((pair.get("confocal_raw_xy") or [float("nan"), float("nan")])[0]), float((pair.get("confocal_raw_xy") or [float("nan"), float("nan")])[1])],
            }
            for idx, pair in enumerate(list(cfg.anchor_pairs or []))
            if isinstance(pair, dict)
        ],
        "anchor_init": anchor_init_info,
        "anchor_preserving_refine": anchor_refine_info,
        "local_roi_bbox_yxyx": [int(y0), int(y1), int(x0), int(x1)],
        "fiber_qc_zoom_bbox_yxyx": [int(v) for v in zoom_bbox],
        "registration_input": {
            "metric": "CC",
            "refine_model": refine_model,
            "confocal_polarity_for_registration": "inverted" if cfg.invert_confocal_for_registration else "native",
            "profile": registration_inputs_full["profile"],
            "description": registration_inputs_full["description"],
        },
        "files": {
            "coarse_alignment_overlay_full": str(inputs_dir / "coarse_alignment_overlay_full.png"),
            "coarse_alignment_overlay_local": str(inputs_dir / "coarse_alignment_overlay_local.png"),
            "quick_qc_storyboard": str(storyboard_path),
            "local_refine_transform": str(effective_local_mat),
            "local_refine_transform_raw": str(local_mat),
            "manifest": str(run_dir / "session_manifest.json"),
        },
        "notes": [
            "confocal_raw_xy is the raw moving-raster coordinate before display flips",
            "confocal_scene_xy is after current preview transform and flips",
            "section_scene_xy is in the Step 7 whole-section working scene at 1.0 um/px",
        ],
    }
    manifest["repro_tracker"] = repro_tracker
    manifest["files"]["repro_tracker"] = str(run_dir / "repro_tracker.json")
    _write_json(run_dir / "session_manifest.json", manifest)
    _write_json(run_dir / "repro_tracker.json", repro_tracker)
    return manifest


def export_confocal_full_report(run_dir: Path) -> dict[str, Any]:
    manifest_path = run_dir / "session_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing Step 7 manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = manifest.get("files") if isinstance(manifest.get("files"), dict) else {}
    fixed_gray = _read_gray_png_float(_runtime_local_path(files["myelin_fixed_full_gray"]))
    moving_input = _read_gray_png_float(_runtime_local_path(files["confocal_manual_full_gray"]))
    moving_input_mask = _read_mask_png_float(_runtime_local_path(files["confocal_manual_full_mask"]))
    refine_gray_path = files.get("local_refine_warped_full_gray") or files.get("rigid_warped_full_gray")
    refine_mask_path = files.get("local_refine_warped_full_mask") or files.get("rigid_warped_full_mask")
    moving_refine = _read_gray_png_float(_runtime_local_path(refine_gray_path))
    moving_refine_mask = _read_mask_png_float(_runtime_local_path(refine_mask_path))
    fixed_mask = _read_mask_png_float(_runtime_local_path(files["myelin_fixed_full_mask"]))
    refine_model = str(manifest.get("local_refine_model") or (manifest.get("local_registration") or {}).get("transform_model") or "rigid")
    refine_label = {"rigid": "Rigid", "similarity": "Similarity", "affine": "Affine"}.get(refine_model, refine_model.title())

    local_info = manifest.get("local_registration") if isinstance(manifest.get("local_registration"), dict) else {}
    roi_bbox = tuple(int(v) for v in (local_info.get("roi_bbox_yxyx") or [0, fixed_gray.shape[0], 0, fixed_gray.shape[1]]))
    full_storyboard_path = run_dir / "full_report_storyboard.png"
    render_storyboard(
        [
            {
                "label": "Input",
                "note": metrics_note(
                    dict(manifest.get("full_input_metrics") or {}),
                    dict(manifest.get("full_input_metric_timing_seconds") or {}),
                    "whole-slice input",
                ),
                "fixed": gray_preview_panel(fixed_gray),
                "moving": gray_preview_panel(moving_input),
                "overlay": overlay_preview(fixed_gray, moving_input, fixed_mask, moving_input_mask),
                "heatmap": _local_zoom_overlay_panel(fixed_gray, moving_input, fixed_mask, moving_input_mask, roi_bbox),
                "col_titles": ("Moving", "Fixed", "Whole overlay", "Local zoom"),
            },
            {
                "label": refine_label,
                "note": metrics_note(
                    dict(manifest.get("full_refine_metrics") or manifest.get("full_rigid_metrics") or {}),
                    dict(manifest.get("full_refine_metric_timing_seconds") or manifest.get("full_rigid_metric_timing_seconds") or {}),
                    f"whole-slice {refine_model}",
                ),
                "fixed": gray_preview_panel(fixed_gray),
                "moving": gray_preview_panel(moving_refine),
                "overlay": overlay_preview(fixed_gray, moving_refine, fixed_mask, moving_refine_mask),
                "heatmap": _local_zoom_overlay_panel(fixed_gray, moving_refine, fixed_mask, moving_refine_mask, roi_bbox),
                "col_titles": ("Moving", "Fixed", "Whole overlay", "Local zoom"),
            },
        ],
        full_storyboard_path,
    )
    report_json_path = run_dir / "full_metrics_report.json"
    report_md_path = run_dir / "full_metrics_report.md"
    report = {
        "myelin_label": manifest.get("myelin_label"),
        "run_dir": str(run_dir),
        "saved_at_utc": manifest.get("saved_at_utc"),
        "manual_init": manifest.get("manual_init"),
        "local_registration": manifest.get("local_registration"),
        "projection_info": manifest.get("projection_info"),
        "input_metrics_local_roi": manifest.get("input_metrics"),
        "refine_metrics_local_roi": manifest.get("refine_metrics") or manifest.get("rigid_metrics"),
        "input_metrics_whole_slice": manifest.get("full_input_metrics"),
        "refine_metrics_whole_slice": manifest.get("full_refine_metrics") or manifest.get("full_rigid_metrics"),
        "timing_seconds": manifest.get("timing_seconds"),
        "files": {
            "full_report_storyboard": str(full_storyboard_path),
            "quick_qc_storyboard": str(files.get("quick_qc_storyboard") or manifest.get("storyboard_path") or ""),
            "manifest": str(manifest_path),
        },
    }
    _write_json(report_json_path, report)
    report_md_path.write_text(
        "\n".join(
            [
                f"# Step 7 Full Report: {manifest.get('myelin_label', 'unknown')}",
                "",
                f"Run dir: `{run_dir}`",
                f"Saved at UTC: `{manifest.get('saved_at_utc', '')}`",
                "",
                "## Manual Init",
                f"- tx_px: `{manifest.get('manual_init', {}).get('tx_px', '')}`",
                f"- ty_px: `{manifest.get('manual_init', {}).get('ty_px', '')}`",
                f"- angle_deg: `{manifest.get('manual_init', {}).get('angle_deg', '')}`",
                f"- scale: `{manifest.get('manual_init', {}).get('scale', '')}`",
                f"- flip_lr: `{manifest.get('manual_init', {}).get('flip_lr', '')}`",
                f"- flip_ud: `{manifest.get('manual_init', {}).get('flip_ud', '')}`",
                "",
                "## Metrics",
                f"- local input: `{manifest.get('input_metrics', {})}`",
                f"- local {refine_model}: `{manifest.get('refine_metrics', manifest.get('rigid_metrics', {}))}`",
                f"- whole input: `{manifest.get('full_input_metrics', {})}`",
                f"- whole {refine_model}: `{manifest.get('full_refine_metrics', manifest.get('full_rigid_metrics', {}))}`",
                "",
                "## Files",
                f"- full_report_storyboard: `{full_storyboard_path}`",
                f"- quick_qc_storyboard: `{files.get('quick_qc_storyboard') or manifest.get('storyboard_path') or ''}`",
                f"- manifest: `{manifest_path}`",
            ]
        ),
        encoding="utf-8",
    )
    return {
        "run_dir": str(run_dir),
        "full_report_storyboard": str(full_storyboard_path),
        "full_metrics_report_json": str(report_json_path),
        "full_metrics_report_md": str(report_md_path),
    }


def _compose_affine_2x3(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    lhs3 = np.vstack([np.asarray(lhs, dtype=np.float32), np.array([[0.0, 0.0, 1.0]], dtype=np.float32)])
    rhs3 = np.vstack([np.asarray(rhs, dtype=np.float32), np.array([[0.0, 0.0, 1.0]], dtype=np.float32)])
    return (lhs3 @ rhs3)[:2, :].astype(np.float32)


def _scene_to_full_crop_affine(
    *,
    preview_shape_hw: tuple[int, int],
    support_bbox_canvas_xywh: tuple[int, int, int, int],
) -> np.ndarray:
    preview_h, preview_w = [max(1, int(v)) for v in preview_shape_hw]
    x0, y0, w, h = [int(v) for v in support_bbox_canvas_xywh]
    sx = float(w) / float(preview_w)
    sy = float(h) / float(preview_h)
    return np.array(
        [
            [sx, 0.0, float(x0)],
            [0.0, sy, float(y0)],
        ],
        dtype=np.float32,
    )


def _scene_polygon_and_bbox_from_raw_bbox(
    mat_2x3: np.ndarray,
    raw_bbox_xyxy: list[int],
) -> tuple[list[list[float]], list[float], list[float]]:
    x0, y0, x1, y1 = [float(v) for v in raw_bbox_xyxy]
    pts = np.asarray([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32)
    dst = _affine_apply_points(np.asarray(mat_2x3, dtype=np.float32), pts)
    polygon = [[float(v[0]), float(v[1])] for v in dst]
    if not polygon:
        return [], [float("nan")] * 4, [float("nan"), float("nan")]
    xs = [float(v[0]) for v in polygon]
    ys = [float(v[1]) for v in polygon]
    bbox = [float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))]
    center = [float(np.mean(xs)), float(np.mean(ys))]
    return polygon, bbox, center


def _augment_step7_tile_record_full_crop_geometry(
    record: dict[str, Any],
    *,
    scene_to_full_crop_mat: np.ndarray,
) -> dict[str, Any]:
    out = dict(record)
    raw_bbox_xyxy = [int(v) for v in list(out.get("raw_bbox_xyxy") or [0, 0, 0, 0])[:4]]
    final_scene_mat = np.asarray(
        out.get("final_affine_matrix_2x3") or np.eye(2, 3, dtype=np.float32),
        dtype=np.float32,
    ).reshape(2, 3)
    final_full_crop_mat = _compose_affine_2x3(np.asarray(scene_to_full_crop_mat, dtype=np.float32), final_scene_mat)
    final_full_poly, final_full_bbox, final_full_center = _scene_polygon_and_bbox_from_raw_bbox(
        final_full_crop_mat,
        raw_bbox_xyxy,
    )
    out.update(
        {
            "final_full_crop_affine_matrix_2x3": final_full_crop_mat.tolist(),
            "final_full_crop_polygon_xy": final_full_poly,
            "final_full_crop_bbox_xyxy": final_full_bbox,
            "final_full_crop_center_xy": final_full_center,
        }
    )
    return out


def _normalize_xywh4(raw: Any) -> tuple[int, int, int, int] | None:
    if not isinstance(raw, (list, tuple)) or len(raw) < 4:
        return None
    try:
        x0, y0, w, h = [int(raw[idx]) for idx in range(4)]
    except Exception:
        return None
    if w <= 0 or h <= 0:
        return None
    return int(x0), int(y0), int(w), int(h)


def _load_section_full_crop_shape_hw(section_dir: Path) -> tuple[int, int] | None:
    metadata_path = Path(section_dir) / "metadata.json"
    if metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            export_canvas = metadata.get("export_canvas") if isinstance(metadata.get("export_canvas"), dict) else {}
            width_px = int(export_canvas.get("width_px") or 0)
            height_px = int(export_canvas.get("height_px") or 0)
            if width_px > 0 and height_px > 0:
                return int(height_px), int(width_px)
        except Exception:
            pass
    crop_path = Path(section_dir) / "crop_raw.png"
    if not crop_path.exists():
        return None
    try:
        with Image.open(crop_path) as im:
            return int(im.height), int(im.width)
    except Exception:
        return None


def _load_workspace_support_bbox_canvas_xywh(section_dir: Path) -> tuple[int, int, int, int] | None:
    labels_path = Path(section_dir) / "mask_labels.png"
    labels = cv2.imread(str(labels_path), cv2.IMREAD_UNCHANGED)
    if labels is None:
        return None
    if labels.ndim == 3:
        labels = labels[..., 0]
    support = np.where(np.asarray(labels, dtype=np.uint8) > 0, 1, 0).astype(np.uint8)
    x0, y0, w, h = _support_bbox_from_labels(support)
    if w <= 0 or h <= 0:
        return None
    return int(x0), int(y0), int(w), int(h)


def _resolve_step7_scene_to_full_crop_contract(
    *,
    myelin_section_dir: Path | None,
    preview_shape_hw: tuple[int, int],
    support_bbox_canvas_xywh: Any = None,
) -> dict[str, Any]:
    bbox_xywh = _normalize_xywh4(support_bbox_canvas_xywh)
    if bbox_xywh is None and myelin_section_dir is not None:
        bbox_xywh = _load_workspace_support_bbox_canvas_xywh(Path(myelin_section_dir))
    if bbox_xywh is None:
        bbox_xywh = (0, 0, int(preview_shape_hw[1]), int(preview_shape_hw[0]))
    full_crop_shape_hw = None
    if myelin_section_dir is not None:
        full_crop_shape_hw = _load_section_full_crop_shape_hw(Path(myelin_section_dir))
    if full_crop_shape_hw is None:
        x0, y0, w, h = bbox_xywh
        full_crop_shape_hw = (max(int(preview_shape_hw[0]), y0 + h), max(int(preview_shape_hw[1]), x0 + w))
    scene_to_full_crop_mat = _scene_to_full_crop_affine(
        preview_shape_hw=preview_shape_hw,
        support_bbox_canvas_xywh=bbox_xywh,
    )
    return {
        "preview_shape_hw": [int(preview_shape_hw[0]), int(preview_shape_hw[1])],
        "fixed_support_bbox_canvas_xywh": [int(v) for v in bbox_xywh],
        "fixed_support_shape_hw": [int(bbox_xywh[3]), int(bbox_xywh[2])],
        "full_crop_shape_hw": [int(full_crop_shape_hw[0]), int(full_crop_shape_hw[1])],
        "scene_to_full_crop_affine_matrix_2x3": np.asarray(scene_to_full_crop_mat, dtype=np.float32).tolist(),
        "scene_to_full_crop_mat": np.asarray(scene_to_full_crop_mat, dtype=np.float32),
    }


def _raw_to_scaled_affine(
    *,
    raw_shape_hw: tuple[int, int],
    scaled_shape_hw: tuple[int, int],
) -> np.ndarray:
    raw_h, raw_w = [max(1, int(v)) for v in raw_shape_hw]
    scaled_h, scaled_w = [max(1, int(v)) for v in scaled_shape_hw]
    sx = float(scaled_w) / float(raw_w)
    sy = float(scaled_h) / float(raw_h)
    return np.array([[sx, 0.0, 0.0], [0.0, sy, 0.0]], dtype=np.float32)


def _tile_local_to_scene_affine(scene_from_raw: np.ndarray, raw_bbox_xyxy: list[int]) -> np.ndarray:
    x0, y0 = float(raw_bbox_xyxy[0]), float(raw_bbox_xyxy[1])
    mat = np.asarray(scene_from_raw, dtype=np.float32).copy()
    linear = mat[:, :2]
    trans = mat[:, 2]
    mat[:, 2] = (linear @ np.array([x0, y0], dtype=np.float32) + trans).astype(np.float32)
    return mat.astype(np.float32)


def _step7_tile_state_color_rgb(state: str) -> tuple[int, int, int]:
    key = str(state or "").strip().lower()
    if key == TileState.FROZEN.value:
        return (70, 120, 255)
    if key == TileState.ACCEPTED.value:
        return (0, 190, 210)
    if key == TileState.FRONTIER.value:
        return (40, 180, 80)
    if key == TileState.HOLD.value:
        return (245, 140, 40)
    return (150, 150, 150)


def _annotate_step7_storyboard_panel(
    panel: np.ndarray,
    *,
    row_display: int,
    col_display: int,
    row_count: int,
    col_count: int,
    shift_dx_px: float,
    shift_dy_px: float,
    tile_state: str,
) -> np.ndarray:
    arr = np.asarray(panel)
    if arr.ndim == 2:
        out = cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_GRAY2RGB)
    else:
        out = arr.astype(np.uint8).copy()
    h, w = out.shape[:2]
    color = _step7_tile_state_color_rgb(tile_state)
    cv2.rectangle(out, (1, 1), (max(1, w - 2), max(1, h - 2)), color, 3)

    mini_w = min(72, max(36, int(round(w * 0.22))))
    mini_h = min(42, max(24, int(round(h * 0.18))))
    gx0 = 8
    gy0 = 8
    cv2.rectangle(out, (gx0 - 2, gy0 - 2), (gx0 + mini_w + 2, gy0 + mini_h + 2), (255, 255, 255), -1)
    cv2.rectangle(out, (gx0 - 2, gy0 - 2), (gx0 + mini_w + 2, gy0 + mini_h + 2), color, 1)
    cell_w = float(mini_w) / float(max(1, col_count))
    cell_h = float(mini_h) / float(max(1, row_count))
    for rr in range(max(1, row_count)):
        for cc in range(max(1, col_count)):
            x0 = int(round(gx0 + cc * cell_w))
            y0 = int(round(gy0 + rr * cell_h))
            x1 = int(round(gx0 + (cc + 1) * cell_w))
            y1 = int(round(gy0 + (rr + 1) * cell_h))
            fill = (230, 230, 230)
            if rr == int(row_display) and cc == int(col_display):
                fill = color
            cv2.rectangle(out, (x0, y0), (max(x0 + 1, x1 - 1), max(y0 + 1, y1 - 1)), fill, -1)
            cv2.rectangle(out, (x0, y0), (max(x0 + 1, x1 - 1), max(y0 + 1, y1 - 1)), (120, 120, 120), 1)

    arrow_origin = (max(16, w - 30), max(16, h - 26))
    dx = float(np.clip(float(shift_dx_px) * 2.8, -18.0, 18.0))
    dy = float(np.clip(float(shift_dy_px) * 2.8, -18.0, 18.0))
    arrow_tip = (int(round(arrow_origin[0] + dx)), int(round(arrow_origin[1] + dy)))
    cv2.circle(out, arrow_origin, 3, (255, 255, 255), -1)
    cv2.arrowedLine(out, arrow_origin, arrow_tip, color, 2, tipLength=0.28)
    cv2.putText(
        out,
        f"r{int(row_display)}c{int(col_display)}",
        (gx0, min(h - 6, gy0 + mini_h + 14)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        (30, 30, 30),
        1,
        cv2.LINE_AA,
    )
    return out


def _build_step7_scene_context_panel(
    base_rgb: np.ndarray,
    *,
    nominal_scene_polygon_xy: list[list[float]] | None,
    final_scene_polygon_xy: list[list[float]] | None,
    nominal_scene_center_xy: list[float] | None,
    final_scene_center_xy: list[float] | None,
    tile_state: str,
    out_shape_hw: tuple[int, int],
    margin_px: int = 56,
) -> np.ndarray:
    base = np.asarray(base_rgb, dtype=np.uint8)
    out_h, out_w = [max(1, int(v)) for v in out_shape_hw]
    nominal_poly = np.asarray(nominal_scene_polygon_xy or [], dtype=np.float32)
    final_poly = np.asarray(final_scene_polygon_xy or [], dtype=np.float32)
    valid_polys = [poly for poly in (nominal_poly, final_poly) if poly.shape == (4, 2)]
    if not valid_polys:
        return np.full((out_h, out_w, 3), 245, dtype=np.uint8)

    xs = np.concatenate([poly[:, 0] for poly in valid_polys], axis=0)
    ys = np.concatenate([poly[:, 1] for poly in valid_polys], axis=0)
    crop_x0 = max(0, int(math.floor(float(np.min(xs))) - int(margin_px)))
    crop_y0 = max(0, int(math.floor(float(np.min(ys))) - int(margin_px)))
    crop_x1 = min(int(base.shape[1]), int(math.ceil(float(np.max(xs))) + int(margin_px)))
    crop_y1 = min(int(base.shape[0]), int(math.ceil(float(np.max(ys))) + int(margin_px)))
    if crop_x1 <= crop_x0 or crop_y1 <= crop_y0:
        return np.full((out_h, out_w, 3), 245, dtype=np.uint8)

    panel = base[crop_y0:crop_y1, crop_x0:crop_x1].copy()
    state_color = _step7_tile_state_color_rgb(tile_state)
    nominal_color = (150, 150, 150)

    if nominal_poly.shape == (4, 2):
        nominal_pts = np.round(nominal_poly - np.array([crop_x0, crop_y0], dtype=np.float32)).astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(panel, [nominal_pts], True, nominal_color, 1, cv2.LINE_AA)
    if final_poly.shape == (4, 2):
        final_pts = np.round(final_poly - np.array([crop_x0, crop_y0], dtype=np.float32)).astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(panel, [final_pts], True, state_color, 2, cv2.LINE_AA)

    nominal_center = np.asarray(nominal_scene_center_xy or [], dtype=np.float32)
    final_center = np.asarray(final_scene_center_xy or [], dtype=np.float32)
    if nominal_center.shape == (2,) and final_center.shape == (2,):
        start = tuple(int(round(v)) for v in (nominal_center - np.array([crop_x0, crop_y0], dtype=np.float32)))
        end = tuple(int(round(v)) for v in (final_center - np.array([crop_x0, crop_y0], dtype=np.float32)))
        cv2.circle(panel, start, 3, nominal_color, -1, cv2.LINE_AA)
        cv2.circle(panel, end, 3, state_color, -1, cv2.LINE_AA)
        if float(np.hypot(float(end[0] - start[0]), float(end[1] - start[1]))) >= 1.0:
            cv2.arrowedLine(panel, start, end, state_color, 2, cv2.LINE_AA, tipLength=0.22)

    cv2.putText(panel, "nominal", (8, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, nominal_color, 1, cv2.LINE_AA)
    cv2.putText(panel, "final", (8, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.42, state_color, 1, cv2.LINE_AA)
    cv2.rectangle(panel, (1, 1), (max(1, panel.shape[1] - 2), max(1, panel.shape[0] - 2)), state_color, 2)

    if panel.shape[:2] != (out_h, out_w):
        panel = cv2.resize(panel, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
    return panel


def _draw_step7_scene_geometry_overlay(
    base_rgb: np.ndarray,
    tile_records: list[dict[str, Any]],
) -> np.ndarray:
    out = np.asarray(base_rgb, dtype=np.uint8).copy()
    nominal_color = (150, 150, 150)
    for row in list(tile_records or []):
        state_color = _step7_tile_state_color_rgb(str(row.get("tile_state") or ""))
        nominal_poly = np.asarray(row.get("nominal_scene_polygon_xy") or [], dtype=np.float32)
        final_poly = np.asarray(row.get("final_scene_polygon_xy") or [], dtype=np.float32)
        if nominal_poly.shape == (4, 2):
            nominal_pts = np.round(nominal_poly).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(out, [nominal_pts], True, nominal_color, 1, cv2.LINE_AA)
        if final_poly.shape == (4, 2):
            final_pts = np.round(final_poly).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(out, [final_pts], True, state_color, 2, cv2.LINE_AA)
        nominal_center = np.asarray(row.get("nominal_scene_center_xy") or [], dtype=np.float32)
        final_center = np.asarray(row.get("final_scene_center_xy") or [], dtype=np.float32)
        if nominal_center.shape == (2,) and final_center.shape == (2,):
            start = tuple(int(round(v)) for v in nominal_center)
            end = tuple(int(round(v)) for v in final_center)
            cv2.circle(out, start, 3, nominal_color, -1, cv2.LINE_AA)
            cv2.circle(out, end, 3, state_color, -1, cv2.LINE_AA)
            if float(np.hypot(float(end[0] - start[0]), float(end[1] - start[1]))) >= 1.0:
                cv2.arrowedLine(out, start, end, state_color, 2, cv2.LINE_AA, tipLength=0.22)
        final_bbox = row.get("final_scene_bbox_xyxy") or row.get("nominal_scene_bbox_xyxy") or []
        if isinstance(final_bbox, (list, tuple)) and len(final_bbox) >= 4:
            bbox = _bbox_int_xyxy(list(final_bbox[:4]))
            cv2.putText(
                out,
                str(row.get("label") or f"T{int(row.get('tile_index', -1)):02d}"),
                (int(bbox[0]) + 4, max(12, int(bbox[1]) + 16)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.40,
                state_color,
                1,
                cv2.LINE_AA,
            )
    return out


def load_step7_handoff_payload(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid Step 7 handoff payload: {path}")
    return payload


def build_step7_scene_fov_masks(
    handoff: dict[str, Any],
    *,
    tile_states: tuple[str, ...] = (TileState.ACCEPTED.value, TileState.FROZEN.value),
    outline_thickness_px: int = 3,
) -> dict[str, Any]:
    scene_space = handoff.get("scene_space") if isinstance(handoff.get("scene_space"), dict) else {}
    fixed_shape_raw = scene_space.get("fixed_preview_shape_hw") or []
    if not isinstance(fixed_shape_raw, (list, tuple)) or len(fixed_shape_raw) < 2:
        raise ValueError("Step 7 handoff is missing scene_space.fixed_preview_shape_hw")
    scene_h = int(fixed_shape_raw[0])
    scene_w = int(fixed_shape_raw[1])
    if scene_h <= 0 or scene_w <= 0:
        raise ValueError("Invalid Step 7 scene shape")
    allowed_states = {str(v).strip().lower() for v in tile_states if str(v).strip()}
    support_mask = np.zeros((scene_h, scene_w), dtype=np.uint8)
    edge_mask = np.zeros((scene_h, scene_w), dtype=np.uint8)
    trusted_records: list[dict[str, Any]] = []
    for row in list(handoff.get("tile_records") or []):
        if not isinstance(row, dict):
            continue
        state = str(row.get("tile_state") or "").strip().lower()
        if state not in allowed_states:
            continue
        poly = np.asarray(row.get("final_scene_polygon_xy") or [], dtype=np.float32)
        if poly.shape != (4, 2):
            continue
        pts = np.round(poly).astype(np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(support_mask, [pts], 255, lineType=cv2.LINE_8)
        cv2.polylines(edge_mask, [pts], True, 255, max(1, int(outline_thickness_px)), cv2.LINE_AA)
        trusted_records.append(dict(row))
    return {
        "scene_shape_hw": [int(scene_h), int(scene_w)],
        "scene_support_mask_u8": support_mask,
        "scene_grid_edges_u8": edge_mask,
        "trusted_tile_count": int(len(trusted_records)),
        "trusted_states": sorted(allowed_states),
    }


def _build_step7_session_readme_lines(session_manifest: dict[str, Any]) -> list[str]:
    state_counts = session_manifest.get("state_counts") if isinstance(session_manifest.get("state_counts"), dict) else {}
    files = session_manifest.get("files") if isinstance(session_manifest.get("files"), dict) else {}
    seam_qc = session_manifest.get("seam_qc") if isinstance(session_manifest.get("seam_qc"), dict) else {}
    prediction_matching = (
        session_manifest.get("prediction_matching") if isinstance(session_manifest.get("prediction_matching"), dict) else {}
    )
    lines = [
        f"# Step 7 Session Export: {session_manifest.get('myelin_label', 'unknown')}",
        "",
        f"Export dir: `{session_manifest.get('run_dir', '')}`",
        f"Saved at UTC: `{session_manifest.get('saved_at_utc', '')}`",
        "",
        "## Scope",
        "- Whole-section myelin vs confocal preview relation",
        "- Current manual coarse transform",
        "- Tile-level states and absolute transforms",
        "- Seed/frontier run references",
        "- Step 8 handoff for fiber-density analysis",
        "",
        "## State Counts",
        f"- tile_count: `{state_counts.get('tile_count', '')}`",
        f"- frozen_count: `{state_counts.get('frozen_count', '')}`",
        f"- accepted_count: `{state_counts.get('accepted_count', '')}`",
        f"- hold_count: `{state_counts.get('hold_count', '')}`",
        f"- frontier_count: `{state_counts.get('frontier_count', '')}`",
        "",
        "## Files",
        f"- preview_scene_full: `{files.get('preview_scene_full', '')}`",
    ]
    if str(files.get("preview_scene_gui_snapshot") or "").strip():
        lines.append(f"- preview_scene_gui_snapshot: `{files.get('preview_scene_gui_snapshot', '')}`")
    if str(files.get("preview_scene_rebuilt") or "").strip():
        lines.append(f"- preview_scene_rebuilt: `{files.get('preview_scene_rebuilt', '')}`")
    lines.extend(
        [
            f"- tile_qc_storyboard: `{files.get('tile_qc_storyboard', '')}`",
            f"- tile_transforms_csv: `{files.get('tile_transforms_csv', '')}`",
            f"- tile_records_json: `{files.get('tile_records_json', '')}`",
            f"- seam_qc_csv: `{files.get('seam_qc_csv', '')}`",
            f"- stitched_confocal_raw_hard: `{files.get('stitched_confocal_raw_hard', '')}`",
            f"- stitched_confocal_raw_blended: `{files.get('stitched_confocal_raw_blended', '')}`",
            f"- prediction_match_json: `{files.get('prediction_match_json', '')}`",
            f"- stitched_prediction_probability: `{files.get('stitched_prediction_probability', '')}`",
            f"- step8_overlay_scene_probability: `{files.get('step8_overlay_scene_probability', '')}`",
            f"- step8_overlay_scene_mask: `{files.get('step8_overlay_scene_mask', '')}`",
            f"- step8_overlay_qc: `{files.get('step8_overlay_qc', '')}`",
            f"- session_manifest: `{session_manifest.get('run_dir', '')}/session_manifest.json`",
            f"- step8_handoff: `{session_manifest.get('run_dir', '')}/step8_handoff.json`",
            "",
            "## Seam QC",
            f"- summary: `{seam_qc}`",
            "",
            "## Prediction Matching",
            f"- summary: `{prediction_matching}`",
            "",
            "## Step 8 Handoff",
            "- Use `step8_handoff.json` as the primary geometry input",
            "- Use `tile_transforms.csv` for quick spreadsheet-level QC",
            "- Use `final_full_crop_polygon_xy` / `final_full_crop_affine_matrix_2x3` when bridging Step 7 tiles back onto canonical myelin full-crop / Step 6",
            "- Intersect downstream nnUNet prediction data with `final_scene_polygon_xy` / `final_affine_matrix_2x3` per tile when working in Step 7 preview-scene space",
        ]
    )
    return lines


def _upgrade_legacy_step7_storyboard_with_scene_context(
    *,
    storyboard_path: Path,
    fixed_rgb: np.ndarray,
    tile_records: list[dict[str, Any]],
) -> str:
    if not storyboard_path.exists():
        return ""
    storyboard_bgr = cv2.imread(str(storyboard_path), cv2.IMREAD_COLOR)
    if storyboard_bgr is None:
        return ""
    storyboard_rgb = cv2.cvtColor(storyboard_bgr, cv2.COLOR_BGR2RGB)
    height, width = storyboard_rgb.shape[:2]
    ordered_records = sorted(
        [dict(row) for row in list(tile_records or [])],
        key=lambda row: (int(row.get("row_display", 0)), int(row.get("col_display", 0))),
    )
    row_count = min(12, len(ordered_records))
    if row_count <= 0:
        return ""
    pad = 12
    title_h = 28
    row_gap = 18
    col_gap = 12
    panel_h_float = (float(height) - float(pad * 2 + title_h + max(0, row_count - 1) * row_gap)) / float(row_count)
    panel_h = int(round(panel_h_float))
    if panel_h <= 0:
        return ""
    expected_width_4col = int(round(pad * 2 + panel_h * 4 + col_gap * 3))
    expected_width_5col = int(round(pad * 2 + panel_h * 5 + col_gap * 4))
    if abs(int(width) - expected_width_5col) <= 2:
        return ""
    if abs(int(width) - expected_width_4col) > 4:
        return ""

    panel_w = int(panel_h)
    new_canvas = np.full((height, expected_width_5col, 3), 245, dtype=np.uint8)
    new_canvas[:, :width] = storyboard_rgb
    x_context = pad + 4 * (panel_w + col_gap)
    cv2.putText(new_canvas, "Scene context", (x_context, pad + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (25, 25, 25), 1, cv2.LINE_AA)
    for row_idx, record in enumerate(ordered_records[:row_count]):
        y = pad + title_h + row_idx * (panel_h + row_gap)
        panel = _build_step7_scene_context_panel(
            np.asarray(fixed_rgb, dtype=np.uint8),
            nominal_scene_polygon_xy=list(record.get("nominal_scene_polygon_xy") or []),
            final_scene_polygon_xy=list(record.get("final_scene_polygon_xy") or []),
            nominal_scene_center_xy=list(record.get("nominal_scene_center_xy") or []),
            final_scene_center_xy=list(record.get("final_scene_center_xy") or []),
            tile_state=str(record.get("tile_state") or ""),
            out_shape_hw=(panel_h, panel_w),
        )
        new_canvas[y : y + panel_h, x_context : x_context + panel_w] = panel
        cv2.rectangle(
            new_canvas,
            (x_context, y),
            (x_context + panel_w, y + panel_h),
            (200, 200, 200),
            1,
        )
    backup_path = storyboard_path.with_name(f"{storyboard_path.stem}_legacy4col{storyboard_path.suffix}")
    if not backup_path.exists():
        shutil.copyfile(storyboard_path, backup_path)
    cv2.imwrite(str(storyboard_path), cv2.cvtColor(new_canvas, cv2.COLOR_RGB2BGR))
    return str(backup_path)


def _build_step8_prediction_scene_overlays(
    fixed_rgb: np.ndarray,
    *,
    crop_bbox_xyxy: list[int],
    prediction_probability_u8: np.ndarray,
    prediction_mask_u8: np.ndarray,
    prediction_support_u8: np.ndarray,
) -> dict[str, np.ndarray]:
    base_full = np.asarray(fixed_rgb, dtype=np.uint8).copy()
    pred_prob = np.asarray(prediction_probability_u8, dtype=np.uint8)
    pred_mask = np.asarray(prediction_mask_u8, dtype=np.uint8)
    pred_support = np.asarray(prediction_support_u8, dtype=np.uint8)
    x0, y0, x1, y1 = [int(v) for v in crop_bbox_xyxy]
    x0 = max(0, min(int(base_full.shape[1]), x0))
    x1 = max(x0, min(int(base_full.shape[1]), x1))
    y0 = max(0, min(int(base_full.shape[0]), y0))
    y1 = max(y0, min(int(base_full.shape[0]), y1))
    crop_h = max(0, y1 - y0)
    crop_w = max(0, x1 - x0)
    if crop_h <= 0 or crop_w <= 0:
        blank = base_full.copy()
        return {
            "scene_probability": blank,
            "scene_mask": blank,
            "qc": blank,
        }

    base_prob = base_full.copy()
    base_mask = base_full.copy()
    crop_base_prob = base_prob[y0:y1, x0:x1].copy()
    crop_base_mask = base_mask[y0:y1, x0:x1].copy()
    prob_heat_bgr = cv2.applyColorMap(pred_prob, cv2.COLORMAP_TURBO)
    prob_heat_rgb = cv2.cvtColor(prob_heat_bgr, cv2.COLOR_BGR2RGB)
    support_bool = pred_support > 0
    alpha = np.zeros(pred_prob.shape, dtype=np.float32)
    alpha[support_bool] = np.clip(pred_prob[support_bool].astype(np.float32) / 255.0, 0.12, 0.72)
    alpha3 = alpha[..., None]
    crop_base_prob = np.where(
        support_bool[..., None],
        np.clip((1.0 - alpha3) * crop_base_prob.astype(np.float32) + alpha3 * prob_heat_rgb.astype(np.float32), 0, 255).astype(np.uint8),
        crop_base_prob,
    )
    base_prob[y0:y1, x0:x1] = crop_base_prob
    cv2.rectangle(base_prob, (x0, y0), (max(x0, x1 - 1), max(y0, y1 - 1)), (255, 140, 0), 2)

    red = np.zeros_like(crop_base_mask)
    red[..., 0] = 255
    mask_bool = pred_mask > 0
    mask_alpha3 = (mask_bool.astype(np.float32) * 0.42)[..., None]
    crop_base_mask = np.where(
        mask_bool[..., None],
        np.clip((1.0 - mask_alpha3) * crop_base_mask.astype(np.float32) + mask_alpha3 * red.astype(np.float32), 0, 255).astype(np.uint8),
        crop_base_mask,
    )
    base_mask[y0:y1, x0:x1] = crop_base_mask
    cv2.rectangle(base_mask, (x0, y0), (max(x0, x1 - 1), max(y0, y1 - 1)), (255, 80, 80), 2)

    crop_prob_zoom = crop_base_prob
    crop_mask_zoom = crop_base_mask
    zoom_h = 480
    zoom_w = max(1, int(round(float(zoom_h) * float(crop_w) / float(max(crop_h, 1)))))
    if zoom_w > 900:
        zoom_w = 900
        zoom_h = max(1, int(round(float(zoom_w) * float(crop_h) / float(max(crop_w, 1)))))
    crop_prob_zoom = cv2.resize(crop_prob_zoom, (zoom_w, zoom_h), interpolation=cv2.INTER_LINEAR)
    crop_mask_zoom = cv2.resize(crop_mask_zoom, (zoom_w, zoom_h), interpolation=cv2.INTER_LINEAR)

    full_h = 480
    full_w = max(1, int(round(float(full_h) * float(base_full.shape[1]) / float(max(base_full.shape[0], 1)))))
    full_scene_zoom = cv2.resize(base_mask, (full_w, full_h), interpolation=cv2.INTER_AREA)

    title_h = 26
    pad = 12
    panel_gap = 12
    panels = [full_scene_zoom, crop_prob_zoom, crop_mask_zoom]
    panel_titles = ["Full scene overlay", "Crop probability", "Crop mask"]
    total_w = pad * 2 + sum(panel.shape[1] for panel in panels) + panel_gap * (len(panels) - 1)
    total_h = pad * 2 + title_h + max(panel.shape[0] for panel in panels)
    qc = np.full((total_h, total_w, 3), 245, dtype=np.uint8)
    cursor_x = pad
    for title, panel in zip(panel_titles, panels):
        cv2.putText(qc, title, (cursor_x, pad + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (25, 25, 25), 1, cv2.LINE_AA)
        y = pad + title_h
        qc[y : y + panel.shape[0], cursor_x : cursor_x + panel.shape[1]] = panel
        cv2.rectangle(qc, (cursor_x, y), (cursor_x + panel.shape[1] - 1, y + panel.shape[0] - 1), (200, 200, 200), 1)
        cursor_x += panel.shape[1] + panel_gap

    return {
        "scene_probability": base_prob,
        "scene_mask": base_mask,
        "qc": qc,
    }


def _project_confocal_source_tile_u8(path: Path, *, mode: str, channel_index: int) -> np.ndarray:
    stack = _read_confocal_stack(_runtime_local_path(path))
    return np.asarray(project_confocal_stack(stack, mode=mode, channel_index=channel_index), dtype=np.uint8)


def _load_step7_projected_tile_images(
    confocal_sources: list[Path],
    *,
    projection_mode: str,
    channel_index: int,
) -> dict[int, np.ndarray]:
    if not confocal_sources:
        return {}
    ordered = [Path(p) for p in list(confocal_sources)]
    out: dict[int, np.ndarray] = {}
    worker_count = _step7_tile_eval_worker_count(max_items=len(ordered))
    if worker_count <= 1:
        for idx, path in enumerate(ordered):
            out[int(idx)] = _project_confocal_source_tile_u8(path, mode=str(projection_mode), channel_index=int(channel_index))
        return out
    with ThreadPoolExecutor(max_workers=int(worker_count), thread_name_prefix="step7-export-tiles") as pool:
        future_map = {
            pool.submit(_project_confocal_source_tile_u8, path, mode=str(projection_mode), channel_index=int(channel_index)): int(idx)
            for idx, path in enumerate(ordered)
        }
        for future in as_completed(future_map):
            out[int(future_map[future])] = np.asarray(future.result(), dtype=np.uint8)
    return out


def _tile_weight_map(shape_hw: tuple[int, int]) -> np.ndarray:
    h, w = [max(1, int(v)) for v in shape_hw]
    yy, xx = np.indices((h, w), dtype=np.float32)
    dist = np.minimum.reduce([xx, yy, float(w - 1) - xx, float(h - 1) - yy])
    dist = np.clip(dist, 0.0, None)
    max_dist = float(dist.max())
    if max_dist <= 1e-6:
        return np.ones((h, w), dtype=np.float32)
    norm = dist / max_dist
    return (0.10 + 0.90 * norm).astype(np.float32)


def _bbox_int_xyxy(values: list[float] | tuple[float, float, float, float]) -> list[int]:
    x0, y0, x1, y1 = [float(v) for v in values]
    return [
        int(math.floor(x0)),
        int(math.floor(y0)),
        int(math.ceil(x1)),
        int(math.ceil(y1)),
    ]


def _warp_tile_to_bbox(
    tile_u8: np.ndarray,
    *,
    local_affine_2x3: np.ndarray,
    scene_bbox_xyxy: list[int],
    local_mask_u8: np.ndarray | None = None,
    local_weight_f32: np.ndarray | None = None,
) -> dict[str, Any]:
    bbox = _bbox_int_xyxy(scene_bbox_xyxy)
    x0, y0, x1, y1 = bbox
    w = max(1, int(x1 - x0))
    h = max(1, int(y1 - y0))
    local_mat = np.asarray(local_affine_2x3, dtype=np.float32).copy()
    local_mat[:, 2] -= np.array([float(x0), float(y0)], dtype=np.float32)
    img = cv2.warpAffine(
        np.asarray(tile_u8, dtype=np.uint8),
        local_mat,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    mask_src = np.full(tile_u8.shape[:2], 255, dtype=np.uint8) if local_mask_u8 is None else np.asarray(local_mask_u8, dtype=np.uint8)
    mask = cv2.warpAffine(
        mask_src,
        local_mat,
        (w, h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    weight = None
    if local_weight_f32 is not None:
        weight = cv2.warpAffine(
            np.asarray(local_weight_f32, dtype=np.float32),
            local_mat,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0.0,
        ).astype(np.float32)
        weight[mask <= 0] = 0.0
    return {
        "bbox_xyxy": bbox,
        "image_u8": img.astype(np.uint8),
        "mask_u8": mask.astype(np.uint8),
        "weight_f32": weight,
    }


def _clip_warp_entry_to_crop(
    entry: dict[str, Any],
    crop_bbox_xyxy: list[int] | tuple[int, int, int, int],
) -> dict[str, Any] | None:
    crop_x0, crop_y0, crop_x1, crop_y1 = [int(v) for v in crop_bbox_xyxy]
    ex0, ey0, ex1, ey1 = [int(v) for v in list(entry.get("bbox_xyxy") or [0, 0, 0, 0])[:4]]
    ix0 = max(crop_x0, ex0)
    iy0 = max(crop_y0, ey0)
    ix1 = min(crop_x1, ex1)
    iy1 = min(crop_y1, ey1)
    if ix1 <= ix0 or iy1 <= iy0:
        return None
    src_x0 = int(ix0 - ex0)
    src_y0 = int(iy0 - ey0)
    src_x1 = src_x0 + int(ix1 - ix0)
    src_y1 = src_y0 + int(iy1 - iy0)
    image_u8 = np.asarray(entry["image_u8"], dtype=np.uint8)[src_y0:src_y1, src_x0:src_x1]
    mask_u8 = np.asarray(entry["mask_u8"], dtype=np.uint8)[src_y0:src_y1, src_x0:src_x1]
    weight_f32 = None
    if entry.get("weight_f32") is not None:
        weight_f32 = np.asarray(entry["weight_f32"], dtype=np.float32)[src_y0:src_y1, src_x0:src_x1]
    return {
        "scene_bbox_xyxy": [int(ix0), int(iy0), int(ix1), int(iy1)],
        "crop_bbox_xyxy": [int(ix0 - crop_x0), int(iy0 - crop_y0), int(ix1 - crop_x0), int(iy1 - crop_y0)],
        "image_u8": image_u8,
        "mask_u8": mask_u8,
        "weight_f32": weight_f32,
    }


def _prediction_expected_name_for_source(source_path: Path) -> str:
    name = _portable_basename(source_path)
    if name.lower().endswith(".ome.tif"):
        return name[:-8] + ".ome_pred.tif"
    if name.lower().endswith(".tif"):
        return name[:-4] + "_pred.tif"
    return Path(name).stem + "_pred.tif"


def _prediction_series_suffix(name: str) -> str:
    base = _portable_basename(name)
    match = re.search(r"(_S\d+)(?:\.ome(?:_pred)?\.tif|_pred\.tif|\.tif)?$", base, flags=re.IGNORECASE)
    return "" if match is None else str(match.group(1)).upper()


def _prediction_prefix_without_series(name: str) -> str:
    base = _portable_basename(name)
    match = re.search(r"^(.*?)(_S\d+)(?:\.ome(?:_pred)?\.tif|_pred\.tif|\.tif)?$", base, flags=re.IGNORECASE)
    if match is not None:
        return str(match.group(1))
    if base.lower().endswith(".ome_pred.tif"):
        return base[:-13]
    if base.lower().endswith(".ome.tif"):
        return base[:-8]
    if base.lower().endswith("_pred.tif"):
        return base[:-9]
    if base.lower().endswith(".tif"):
        return base[:-4]
    return Path(base).stem


def _prediction_sample_side_prefix(name: str) -> str:
    prefix = _prediction_prefix_without_series(name)
    return prefix.rsplit("_", 1)[0] if "_" in prefix else prefix


def _build_prediction_match_index(
    confocal_sources: list[Path],
    *,
    prediction_root: Path | None,
) -> dict[str, Any]:
    root = None if prediction_root is None else Path(prediction_root)
    rows: list[dict[str, Any]] = []
    exact_count = 0
    usable_match_count = 0
    same_stack_name_variant_only_count = 0
    same_sample_side_other_roi_count = 0
    ambiguous_suffix_only_count = 0
    missing_count = 0
    if root is None or not root.exists():
        return {
            "prediction_root": "" if root is None else str(root),
            "confocal_source_count": int(len(confocal_sources or [])),
            "prediction_file_count": 0,
            "exact_match_count": 0,
            "usable_match_count": 0,
            "same_stack_name_variant_only_count": 0,
            "same_sample_side_other_roi_count": 0,
            "ambiguous_suffix_only_count": 0,
            "missing_count": int(len(confocal_sources or [])),
            "rows": rows,
            "status": "missing_prediction_root",
        }
    files = [p for p in root.rglob("*") if p.is_file()]
    by_name = {p.name: p for p in files}
    by_suffix: dict[str, list[Path]] = {}
    for path in files:
        m = re.search(r"(_S\d+\.ome_pred\.tif)$", path.name, flags=re.IGNORECASE)
        if m:
            by_suffix.setdefault(m.group(1).lower(), []).append(path)
    for source in list(confocal_sources or []):
        source_name = _portable_basename(source)
        expected_name = _prediction_expected_name_for_source(source_name)
        exact_path = by_name.get(expected_name)
        relaxed_candidates: list[Path] = []
        suffix_match = re.search(r"(_S\d+\.ome(?:\.tif)?)$", source_name, flags=re.IGNORECASE)
        if suffix_match is not None:
            pred_suffix = suffix_match.group(1).replace(".ome.tif", ".ome_pred.tif").lower()
            relaxed_candidates = list(by_suffix.get(pred_suffix, []))
        same_sample_side_candidates = [
            p
            for p in relaxed_candidates
            if _prediction_sample_side_prefix(p.name).lower() == _prediction_sample_side_prefix(source_name).lower()
        ]
        same_stack_name_candidates = [
            p
            for p in relaxed_candidates
            if _prediction_prefix_without_series(p.name).lower() == _prediction_prefix_without_series(source_name).lower()
        ]
        resolved_path = exact_path if exact_path is not None else (same_stack_name_candidates[0] if len(same_stack_name_candidates) == 1 else None)
        if exact_path is not None:
            status = "exact"
        elif resolved_path is not None:
            status = "same_stack_name_variant_only"
        elif same_sample_side_candidates:
            status = "same_sample_side_other_roi_only"
        elif relaxed_candidates:
            status = "ambiguous_suffix_only"
        else:
            status = "missing"
        if exact_path is not None:
            exact_count += 1
        if resolved_path is not None:
            usable_match_count += 1
        if status == "same_stack_name_variant_only":
            same_stack_name_variant_only_count += 1
        elif status == "same_sample_side_other_roi_only":
            same_sample_side_other_roi_count += 1
        elif status == "ambiguous_suffix_only":
            ambiguous_suffix_only_count += 1
        elif status == "missing":
            missing_count += 1
        rows.append(
            {
                "confocal_source": str(source),
                "confocal_basename": source_name,
                "expected_prediction_basename": expected_name,
                "match_status": status,
                "prediction_path": "" if resolved_path is None else str(resolved_path),
                "prediction_basename": "" if resolved_path is None else resolved_path.name,
                "series_suffix": _prediction_series_suffix(source_name),
                "source_prefix_without_series": _prediction_prefix_without_series(source_name),
                "source_sample_side_prefix": _prediction_sample_side_prefix(source_name),
                "relaxed_candidates": [str(p) for p in relaxed_candidates[:10]],
                "same_sample_side_candidates": [str(p) for p in same_sample_side_candidates[:10]],
                "same_stack_name_candidates": [str(p) for p in same_stack_name_candidates[:10]],
            }
        )
    overall = "exact_available" if exact_count > 0 else ("usable_variant_available" if usable_match_count > 0 else "no_exact_matches")
    return {
        "prediction_root": str(root),
        "confocal_source_count": int(len(confocal_sources or [])),
        "prediction_file_count": int(len(files)),
        "exact_match_count": int(exact_count),
        "usable_match_count": int(usable_match_count),
        "same_stack_name_variant_only_count": int(same_stack_name_variant_only_count),
        "same_sample_side_other_roi_count": int(same_sample_side_other_roi_count),
        "ambiguous_suffix_only_count": int(ambiguous_suffix_only_count),
        "missing_count": int(missing_count),
        "rows": rows,
        "status": overall,
    }


def _rebuild_step7_export_tile_record_geometry(
    tile_records: list[dict[str, Any]],
    *,
    base_manual_scaled_mat: np.ndarray,
    raw_shape_hw: tuple[int, int],
    scaled_shape_hw: tuple[int, int],
) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray]:
    def _scene_polygon_and_bbox_local(mat_2x3: np.ndarray, raw_bbox_xyxy: list[int]) -> tuple[list[list[float]], list[float], list[float]]:
        x0, y0, x1, y1 = [float(v) for v in raw_bbox_xyxy]
        pts = np.asarray([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32)
        aug = np.concatenate([pts, np.ones((pts.shape[0], 1), dtype=np.float32)], axis=1)
        dst = aug @ np.asarray(mat_2x3, dtype=np.float32).T
        polygon = [[float(v[0]), float(v[1])] for v in dst]
        xs = [float(v[0]) for v in polygon]
        ys = [float(v[1]) for v in polygon]
        bbox = [float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))]
        center = [float(np.mean(xs)), float(np.mean(ys))]
        return polygon, bbox, center

    raw_to_scaled_mat = _raw_to_scaled_affine(
        raw_shape_hw=raw_shape_hw,
        scaled_shape_hw=scaled_shape_hw,
    )
    base_manual_raw_mat = _compose_affine_2x3(base_manual_scaled_mat, raw_to_scaled_mat)
    rebuilt: list[dict[str, Any]] = []
    for row in list(tile_records or []):
        record = dict(row)
        raw_bbox_xyxy = [int(v) for v in list(record.get("raw_bbox_xyxy") or [0, 0, 0, 0])[:4]]
        pred_dx = float(record.get("pred_dx_px", 0.0) or 0.0)
        pred_dy = float(record.get("pred_dy_px", 0.0) or 0.0)
        final_dx = float(record.get("final_dx_px", 0.0) or 0.0)
        final_dy = float(record.get("final_dy_px", 0.0) or 0.0)
        pred_mat = np.asarray(base_manual_raw_mat, dtype=np.float32).copy()
        pred_mat[:, 2] += np.array([pred_dx, pred_dy], dtype=np.float32)
        final_mat = np.asarray(base_manual_raw_mat, dtype=np.float32).copy()
        final_mat[:, 2] += np.array([final_dx, final_dy], dtype=np.float32)
        nominal_poly, nominal_bbox, nominal_center = _scene_polygon_and_bbox_local(
            np.asarray(base_manual_raw_mat, dtype=np.float32),
            raw_bbox_xyxy,
        )
        pred_poly, pred_bbox, pred_center = _scene_polygon_and_bbox_local(pred_mat, raw_bbox_xyxy)
        final_poly, final_bbox, final_center = _scene_polygon_and_bbox_local(final_mat, raw_bbox_xyxy)
        record.update(
            {
                "manual_affine_matrix_2x3": np.asarray(base_manual_raw_mat, dtype=np.float32).tolist(),
                "manual_affine_scaled_projection_matrix_2x3": np.asarray(base_manual_scaled_mat, dtype=np.float32).tolist(),
                "pred_affine_matrix_2x3": pred_mat.tolist(),
                "final_affine_matrix_2x3": final_mat.tolist(),
                "nominal_scene_polygon_xy": nominal_poly,
                "pred_scene_polygon_xy": pred_poly,
                "final_scene_polygon_xy": final_poly,
                "nominal_scene_bbox_xyxy": nominal_bbox,
                "pred_scene_bbox_xyxy": pred_bbox,
                "final_scene_bbox_xyxy": final_bbox,
                "nominal_scene_center_xy": nominal_center,
                "pred_scene_center_xy": pred_center,
                "final_scene_center_xy": final_center,
            }
        )
        rebuilt.append(record)
    return rebuilt, base_manual_raw_mat, raw_to_scaled_mat


def _load_prediction_projection_u8(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    try:
        import tifffile
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("tifffile is required to read exported prediction tiles.") from exc
    arr = np.asarray(tifffile.imread(str(path)))
    if arr.ndim >= 3:
        proj = np.max(arr.astype(np.float32), axis=0)
        source_mode = "z_max"
    else:
        proj = arr.astype(np.float32)
        source_mode = "planar"
    max_val = float(np.nanmax(proj)) if proj.size else 0.0
    if max_val <= 1.0:
        out = np.clip(np.round(np.clip(proj, 0.0, 1.0) * 255.0), 0, 255).astype(np.uint8)
        probability_like = True
    else:
        out = np.clip(np.round(np.clip(proj / max(max_val, 1e-6), 0.0, 1.0) * 255.0), 0, 255).astype(np.uint8)
        probability_like = False
    return out, {
        "source_mode": source_mode,
        "probability_like": bool(probability_like),
        "source_shape": [int(v) for v in arr.shape],
    }


def _extract_entry_strip(
    entry: dict[str, Any],
    strip_bbox_xyxy: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    sx0, sy0, sx1, sy1 = [int(v) for v in strip_bbox_xyxy]
    ex0, ey0, ex1, ey1 = [int(v) for v in entry["bbox_xyxy"]]
    w = max(1, sx1 - sx0)
    h = max(1, sy1 - sy0)
    patch_img = np.zeros((h, w), dtype=np.uint8)
    patch_mask = np.zeros((h, w), dtype=np.uint8)
    ix0 = max(sx0, ex0)
    iy0 = max(sy0, ey0)
    ix1 = min(sx1, ex1)
    iy1 = min(sy1, ey1)
    if ix1 <= ix0 or iy1 <= iy0:
        return patch_img, patch_mask
    src_x0 = ix0 - ex0
    src_y0 = iy0 - ey0
    dst_x0 = ix0 - sx0
    dst_y0 = iy0 - sy0
    hh = iy1 - iy0
    ww = ix1 - ix0
    patch_img[dst_y0 : dst_y0 + hh, dst_x0 : dst_x0 + ww] = entry["image_u8"][src_y0 : src_y0 + hh, src_x0 : src_x0 + ww]
    patch_mask[dst_y0 : dst_y0 + hh, dst_x0 : dst_x0 + ww] = entry["mask_u8"][src_y0 : src_y0 + hh, src_x0 : src_x0 + ww]
    return patch_img, patch_mask


def _cross_correlation_on_valid(a_u8: np.ndarray, b_u8: np.ndarray, valid_mask: np.ndarray) -> float:
    valid = np.asarray(valid_mask) > 0
    if int(np.count_nonzero(valid)) < 64:
        return float("nan")
    aa = np.asarray(a_u8, dtype=np.float32)[valid] / 255.0
    bb = np.asarray(b_u8, dtype=np.float32)[valid] / 255.0
    if aa.size < 64 or bb.size < 64:
        return float("nan")
    return float(_cross_correlation(aa, bb))


def _compute_step7_seam_qc(
    trusted_records: list[dict[str, Any]],
    warped_entries_by_tile: dict[int, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    trusted_by_pos = {(int(r["row_display"]), int(r["col_display"])): r for r in trusted_records}
    seam_rows: list[dict[str, Any]] = []
    for row in trusted_records:
        row_r = int(row["row_display"])
        row_c = int(row["col_display"])
        neighbors = [(row_r, row_c + 1, "vertical"), (row_r + 1, row_c, "horizontal")]
        for nr, nc, orientation in neighbors:
            neighbor = trusted_by_pos.get((nr, nc))
            if neighbor is None:
                continue
            left = row
            right = neighbor
            if orientation == "vertical" and int(left["col_display"]) > int(right["col_display"]):
                left, right = right, left
            if orientation == "horizontal" and int(left["row_display"]) > int(right["row_display"]):
                left, right = right, left
            bbox_a = _bbox_int_xyxy(left["final_scene_bbox_xyxy"])
            bbox_b = _bbox_int_xyxy(right["final_scene_bbox_xyxy"])
            if orientation == "vertical":
                gap_signed = float(bbox_b[0] - bbox_a[2])
                overlap_span = float(max(0, min(bbox_a[3], bbox_b[3]) - max(bbox_a[1], bbox_b[1])))
                seam_center = int(round((float(bbox_a[2]) + float(bbox_b[0])) / 2.0))
                strip_bbox = [
                    seam_center - 12,
                    max(bbox_a[1], bbox_b[1]),
                    seam_center + 12,
                    min(bbox_a[3], bbox_b[3]),
                ]
            else:
                gap_signed = float(bbox_b[1] - bbox_a[3])
                overlap_span = float(max(0, min(bbox_a[2], bbox_b[2]) - max(bbox_a[0], bbox_b[0])))
                seam_center = int(round((float(bbox_a[3]) + float(bbox_b[1])) / 2.0))
                strip_bbox = [
                    max(bbox_a[0], bbox_b[0]),
                    seam_center - 12,
                    min(bbox_a[2], bbox_b[2]),
                    seam_center + 12,
                ]
            entry_a = warped_entries_by_tile.get(int(left["tile_index"]))
            entry_b = warped_entries_by_tile.get(int(right["tile_index"]))
            strip_cc = float("nan")
            edge_cc = float("nan")
            mask_disagreement = float("nan")
            if entry_a is not None and entry_b is not None and strip_bbox[2] > strip_bbox[0] and strip_bbox[3] > strip_bbox[1]:
                patch_a, mask_a = _extract_entry_strip(entry_a, strip_bbox)
                patch_b, mask_b = _extract_entry_strip(entry_b, strip_bbox)
                both = np.where((mask_a > 0) & (mask_b > 0), 255, 0).astype(np.uint8)
                either = np.where((mask_a > 0) | (mask_b > 0), 255, 0).astype(np.uint8)
                strip_cc = _cross_correlation_on_valid(patch_a, patch_b, both)
                edge_a = cv2.Canny(patch_a, 32, 96)
                edge_b = cv2.Canny(patch_b, 32, 96)
                edge_cc = _cross_correlation_on_valid(edge_a, edge_b, both)
                denom = max(1, int(np.count_nonzero(either > 0)))
                mask_disagreement = float(np.count_nonzero(((mask_a > 0) ^ (mask_b > 0)) & (either > 0)) / float(denom))
            seam_status = "good"
            if overlap_span <= 0 or abs(gap_signed) > 8.0 or (np.isfinite(strip_cc) and strip_cc < 0.15) or (np.isfinite(mask_disagreement) and mask_disagreement > 0.35):
                seam_status = "bad"
            elif abs(gap_signed) > 3.0 or (np.isfinite(strip_cc) and strip_cc < 0.35) or (np.isfinite(mask_disagreement) and mask_disagreement > 0.15):
                seam_status = "warn"
            seam_rows.append(
                {
                    "tile_a_index": int(left["tile_index"]),
                    "tile_a_label": str(left["label"]),
                    "tile_b_index": int(right["tile_index"]),
                    "tile_b_label": str(right["label"]),
                    "orientation": str(orientation),
                    "scene_gap_signed_px": float(gap_signed),
                    "scene_gap_px": float(max(0.0, gap_signed)),
                    "scene_overlap_px": float(max(0.0, -gap_signed)),
                    "orth_overlap_span_px": float(overlap_span),
                    "strip_cc": float(strip_cc),
                    "edge_cc": float(edge_cc),
                    "mask_disagreement": float(mask_disagreement),
                    "seam_status": seam_status,
                }
            )
    summary = {
        "seam_count": int(len(seam_rows)),
        "good_count": int(sum(1 for row in seam_rows if str(row["seam_status"]) == "good")),
        "warn_count": int(sum(1 for row in seam_rows if str(row["seam_status"]) == "warn")),
        "bad_count": int(sum(1 for row in seam_rows if str(row["seam_status"]) == "bad")),
    }
    return seam_rows, summary


def _build_step7_session_export_products(
    *,
    run_dir: Path,
    fixed_rgb: np.ndarray,
    confocal_sources: list[Path],
    confocal_source_mode: str,
    projection_mode: str,
    channel_index: int,
    tile_records: list[dict[str, Any]],
    preview_scene_rgb: np.ndarray | None = None,
    prediction_root: Path | None = None,
) -> dict[str, Any]:
    trusted_records = [
        dict(row)
        for row in list(tile_records or [])
        if str(row.get("tile_state") or "").strip().lower() in {TileState.FROZEN.value, TileState.ACCEPTED.value}
    ]
    preview_scene_path = run_dir / "preview_scene_full.png"
    preview_scene_gui_path = run_dir / "preview_scene_gui_snapshot.png"
    preview_scene_rebuilt_path = run_dir / "preview_scene_rebuilt.png"
    seam_csv_path = run_dir / "seam_qc.csv"
    seam_json_path = run_dir / "seam_qc.json"
    confocal_hard_path = run_dir / "stitched_confocal_raw_hard.png"
    confocal_blended_path = run_dir / "stitched_confocal_raw_blended.png"
    confocal_support_path = run_dir / "stitched_confocal_support.png"
    confocal_tile_id_path = run_dir / "stitched_confocal_tile_id.png"
    prediction_match_path = run_dir / "prediction_match.json"
    prediction_prob_path = run_dir / "stitched_prediction_probability.png"
    prediction_mask_path = run_dir / "stitched_prediction_mask.png"
    prediction_support_path = run_dir / "stitched_prediction_support.png"
    step8_overlay_prob_scene_path = run_dir / "step8_overlay_scene_probability.png"
    step8_overlay_mask_scene_path = run_dir / "step8_overlay_scene_mask.png"
    step8_overlay_qc_path = run_dir / "step8_overlay_qc.png"

    prediction_match = _build_prediction_match_index(confocal_sources, prediction_root=prediction_root)
    _write_json(prediction_match_path, prediction_match)
    files = {
        "preview_scene_full": "",
        "preview_scene_gui_snapshot": "",
        "preview_scene_rebuilt": "",
        "seam_qc_csv": "",
        "seam_qc_json": "",
        "stitched_confocal_raw_hard": "",
        "stitched_confocal_raw_blended": "",
        "stitched_confocal_support": "",
        "stitched_confocal_tile_id": "",
        "prediction_match_json": str(prediction_match_path),
        "stitched_prediction_probability": "",
        "stitched_prediction_mask": "",
        "stitched_prediction_support": "",
        "step8_overlay_scene_probability": "",
        "step8_overlay_scene_mask": "",
        "step8_overlay_qc": "",
    }

    def _persist_preview_outputs(preview_rebuilt_rgb: np.ndarray) -> None:
        rebuilt_rgb = np.asarray(preview_rebuilt_rgb, dtype=np.uint8)
        cv2.imwrite(str(preview_scene_path), cv2.cvtColor(rebuilt_rgb, cv2.COLOR_RGB2BGR))
        for stale_path in (preview_scene_gui_path, preview_scene_rebuilt_path):
            try:
                if stale_path.exists():
                    stale_path.unlink()
            except Exception:
                pass
        files["preview_scene_full"] = str(preview_scene_path)

    if not trusted_records or str(confocal_source_mode) not in {"multi_tiff_grid", "multi_tiff_strip"}:
        preview_rebuilt = _draw_step7_scene_geometry_overlay(np.asarray(fixed_rgb, dtype=np.uint8), tile_records)
        _persist_preview_outputs(preview_rebuilt)
        return {
            "files": files,
            "trusted_tile_count": int(len(trusted_records)),
            "scene_crop_bbox_xyxy": [],
            "seam_summary": {"seam_count": 0, "good_count": 0, "warn_count": 0, "bad_count": 0},
            "prediction_summary": {
                "prediction_root": prediction_match.get("prediction_root", ""),
                "exact_match_count": int(prediction_match.get("exact_match_count", 0) or 0),
                "usable_match_count": int(prediction_match.get("usable_match_count", 0) or 0),
                "same_stack_name_variant_only_count": int(prediction_match.get("same_stack_name_variant_only_count", 0) or 0),
                "same_sample_side_other_roi_count": int(prediction_match.get("same_sample_side_other_roi_count", 0) or 0),
                "ambiguous_suffix_only_count": int(prediction_match.get("ambiguous_suffix_only_count", 0) or 0),
                "missing_count": int(prediction_match.get("missing_count", 0) or 0),
                "stitched_prediction_ready": False,
            },
        }

    projected_tiles = _load_step7_projected_tile_images(
        confocal_sources,
        projection_mode=str(projection_mode),
        channel_index=int(channel_index),
    )

    trusted_bboxes = [_bbox_int_xyxy(row.get("final_scene_bbox_xyxy") or [0, 0, 0, 0]) for row in trusted_records]
    crop_x0 = max(0, min(int(b[0]) for b in trusted_bboxes))
    crop_y0 = max(0, min(int(b[1]) for b in trusted_bboxes))
    crop_x1 = min(int(fixed_rgb.shape[1]), max(int(b[2]) for b in trusted_bboxes))
    crop_y1 = min(int(fixed_rgb.shape[0]), max(int(b[3]) for b in trusted_bboxes))
    crop_bbox = [int(crop_x0), int(crop_y0), int(crop_x1), int(crop_y1)]
    crop_w = max(1, crop_x1 - crop_x0)
    crop_h = max(1, crop_y1 - crop_y0)

    hard_img = np.zeros((crop_h, crop_w), dtype=np.uint8)
    hard_support = np.zeros((crop_h, crop_w), dtype=np.uint8)
    hard_priority = np.zeros((crop_h, crop_w), dtype=np.uint16)
    blend_accum = np.zeros((crop_h, crop_w), dtype=np.uint32)
    blend_weight = np.zeros((crop_h, crop_w), dtype=np.uint16)
    tile_id_map = np.zeros((crop_h, crop_w), dtype=np.uint16)
    prediction_accum = np.zeros((crop_h, crop_w), dtype=np.uint32)
    prediction_weight = np.zeros((crop_h, crop_w), dtype=np.uint16)
    prediction_support = np.zeros((crop_h, crop_w), dtype=np.uint8)
    prediction_resolved_count = 0

    warped_entries_by_tile: dict[int, dict[str, Any]] = {}
    prediction_resolved_by_basename = {
        Path(str(row.get("confocal_basename") or "")).name: Path(str(row.get("prediction_path")))
        for row in list(prediction_match.get("rows") or [])
        if str(row.get("match_status") or "") in {"exact", "same_stack_name_variant_only"} and str(row.get("prediction_path") or "").strip()
    }

    for record in trusted_records:
        tile_index = int(record["tile_index"])
        tile_img = projected_tiles.get(tile_index)
        raw_bbox = [int(v) for v in list(record.get("raw_bbox_xyxy") or [0, 0, 0, 0])[:4]]
        final_mat_raw = np.asarray(record.get("final_affine_matrix_2x3") or np.eye(2, 3, dtype=np.float32), dtype=np.float32).reshape(2, 3)
        local_affine = _tile_local_to_scene_affine(final_mat_raw, raw_bbox)
        weight_map = _tile_weight_map(tile_img.shape[:2]) if isinstance(tile_img, np.ndarray) else None
        if not isinstance(tile_img, np.ndarray):
            continue
        entry = _warp_tile_to_bbox(
            tile_img,
            local_affine_2x3=local_affine,
            scene_bbox_xyxy=record.get("final_scene_bbox_xyxy") or [0, 0, 1, 1],
            local_mask_u8=np.full(tile_img.shape[:2], 255, dtype=np.uint8),
            local_weight_f32=weight_map,
        )
        warped_entries_by_tile[int(tile_index)] = entry
        clipped_entry = _clip_warp_entry_to_crop(entry, crop_bbox)
        if clipped_entry is None:
            continue
        rx0, ry0, rx1, ry1 = [int(v) for v in clipped_entry["crop_bbox_xyxy"]]
        state = str(record.get("tile_state") or "")
        state_base = 20000 if state == TileState.FROZEN.value else 10000
        priority_value = int(state_base + round(max(0.0, min(1.0, float(record.get("final_cc", 0.0) or 0.0))) * 1000.0))
        mask_bool = np.asarray(clipped_entry["mask_u8"], dtype=np.uint8) > 0
        current_priority = hard_priority[ry0:ry1, rx0:rx1]
        replace = mask_bool & (np.uint16(priority_value) >= current_priority)
        hard_priority[ry0:ry1, rx0:rx1][replace] = np.uint16(priority_value)
        hard_img[ry0:ry1, rx0:rx1][replace] = np.asarray(clipped_entry["image_u8"], dtype=np.uint8)[replace]
        hard_support[ry0:ry1, rx0:rx1][mask_bool] = 255
        tile_id_map[ry0:ry1, rx0:rx1][replace] = np.uint16(tile_index + 1)

        if clipped_entry["weight_f32"] is not None:
            wq = np.clip(np.round(np.asarray(clipped_entry["weight_f32"], dtype=np.float32) * 255.0), 0, 65535).astype(np.uint16)
            wq[~mask_bool] = 0
            blend_accum[ry0:ry1, rx0:rx1] += np.asarray(clipped_entry["image_u8"], dtype=np.uint8).astype(np.uint32) * wq.astype(np.uint32)
            blend_weight[ry0:ry1, rx0:rx1] += wq

        basename = _portable_basename(confocal_sources[tile_index]) if tile_index < len(confocal_sources) else ""
        pred_path = prediction_resolved_by_basename.get(basename)
        if pred_path is not None and pred_path.exists():
            pred_u8, _pred_info = _load_prediction_projection_u8(pred_path)
            pred_entry = _warp_tile_to_bbox(
                pred_u8,
                local_affine_2x3=local_affine,
                scene_bbox_xyxy=record.get("final_scene_bbox_xyxy") or [0, 0, 1, 1],
                local_mask_u8=np.full(pred_u8.shape[:2], 255, dtype=np.uint8),
                local_weight_f32=weight_map if weight_map is not None else np.ones(pred_u8.shape[:2], dtype=np.float32),
            )
            clipped_pred_entry = _clip_warp_entry_to_crop(pred_entry, crop_bbox)
            if clipped_pred_entry is None:
                continue
            pred_mask_bool = np.asarray(clipped_pred_entry["mask_u8"], dtype=np.uint8) > 0
            pred_rx0, pred_ry0, pred_rx1, pred_ry1 = [int(v) for v in clipped_pred_entry["crop_bbox_xyxy"]]
            pred_weight_f32 = clipped_pred_entry["weight_f32"]
            if pred_weight_f32 is None:
                pred_weight_f32 = np.ones(np.asarray(clipped_pred_entry["image_u8"]).shape[:2], dtype=np.float32)
            pred_wq = np.clip(np.round(np.asarray(pred_weight_f32, dtype=np.float32) * 255.0), 0, 65535).astype(np.uint16)
            pred_wq[~pred_mask_bool] = 0
            prediction_accum[pred_ry0:pred_ry1, pred_rx0:pred_rx1] += np.asarray(clipped_pred_entry["image_u8"], dtype=np.uint8).astype(np.uint32) * pred_wq.astype(np.uint32)
            prediction_weight[pred_ry0:pred_ry1, pred_rx0:pred_rx1] += pred_wq
            prediction_support[pred_ry0:pred_ry1, pred_rx0:pred_rx1][pred_mask_bool] = 255
            prediction_resolved_count += 1

    seam_rows, seam_summary = _compute_step7_seam_qc(trusted_records, warped_entries_by_tile)
    with seam_csv_path.open("w", newline="", encoding="utf-8") as handle:
        if seam_rows:
            writer = csv.DictWriter(handle, fieldnames=list(seam_rows[0].keys()))
            writer.writeheader()
            writer.writerows(seam_rows)
    _write_json(seam_json_path, {"summary": seam_summary, "rows": seam_rows})
    files["seam_qc_csv"] = str(seam_csv_path)
    files["seam_qc_json"] = str(seam_json_path)

    blended_img = np.zeros_like(hard_img)
    valid = blend_weight > 0
    blended_img[valid] = np.clip(
        np.round(blend_accum[valid].astype(np.float32) / np.maximum(blend_weight[valid].astype(np.float32), 1.0)),
        0,
        255,
    ).astype(np.uint8)
    cv2.imwrite(str(confocal_hard_path), hard_img)
    cv2.imwrite(str(confocal_blended_path), blended_img)
    cv2.imwrite(str(confocal_support_path), hard_support)
    cv2.imwrite(str(confocal_tile_id_path), tile_id_map)
    files["stitched_confocal_raw_hard"] = str(confocal_hard_path)
    files["stitched_confocal_raw_blended"] = str(confocal_blended_path)
    files["stitched_confocal_support"] = str(confocal_support_path)
    files["stitched_confocal_tile_id"] = str(confocal_tile_id_path)

    if prediction_resolved_count > 0:
        pred_prob = np.zeros((crop_h, crop_w), dtype=np.uint8)
        pred_mask = np.zeros((crop_h, crop_w), dtype=np.uint8)
        pred_valid = prediction_weight > 0
        pred_prob[pred_valid] = np.clip(
            np.round(prediction_accum[pred_valid].astype(np.float32) / np.maximum(prediction_weight[pred_valid].astype(np.float32), 1.0)),
            0,
            255,
        ).astype(np.uint8)
        pred_mask[pred_prob >= 128] = 255
        cv2.imwrite(str(prediction_prob_path), pred_prob)
        cv2.imwrite(str(prediction_mask_path), pred_mask)
        cv2.imwrite(str(prediction_support_path), prediction_support)
        files["stitched_prediction_probability"] = str(prediction_prob_path)
        files["stitched_prediction_mask"] = str(prediction_mask_path)
        files["stitched_prediction_support"] = str(prediction_support_path)
        scene_overlays = _build_step8_prediction_scene_overlays(
            np.asarray(fixed_rgb, dtype=np.uint8),
            crop_bbox_xyxy=crop_bbox,
            prediction_probability_u8=pred_prob,
            prediction_mask_u8=pred_mask,
            prediction_support_u8=prediction_support,
        )
        cv2.imwrite(str(step8_overlay_prob_scene_path), cv2.cvtColor(np.asarray(scene_overlays["scene_probability"], dtype=np.uint8), cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(step8_overlay_mask_scene_path), cv2.cvtColor(np.asarray(scene_overlays["scene_mask"], dtype=np.uint8), cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(step8_overlay_qc_path), cv2.cvtColor(np.asarray(scene_overlays["qc"], dtype=np.uint8), cv2.COLOR_RGB2BGR))
        files["step8_overlay_scene_probability"] = str(step8_overlay_prob_scene_path)
        files["step8_overlay_scene_mask"] = str(step8_overlay_mask_scene_path)
        files["step8_overlay_qc"] = str(step8_overlay_qc_path)

    preview_rebuilt = np.asarray(fixed_rgb, dtype=np.uint8).copy()
    preview_crop = preview_rebuilt[crop_y0:crop_y1, crop_x0:crop_x1].copy()
    conf_rgb = cv2.cvtColor(blended_img, cv2.COLOR_GRAY2RGB)
    preview_rebuilt[crop_y0:crop_y1, crop_x0:crop_x1] = np.clip(
        0.68 * preview_crop.astype(np.float32) + 0.32 * conf_rgb.astype(np.float32),
        0,
        255,
    ).astype(np.uint8)
    preview_rebuilt = _draw_step7_scene_geometry_overlay(preview_rebuilt, tile_records)
    _persist_preview_outputs(preview_rebuilt)

    return {
        "files": files,
        "trusted_tile_count": int(len(trusted_records)),
        "scene_crop_bbox_xyxy": [int(v) for v in crop_bbox],
        "seam_summary": seam_summary,
        "prediction_summary": {
            "prediction_root": prediction_match.get("prediction_root", ""),
            "exact_match_count": int(prediction_match.get("exact_match_count", 0) or 0),
            "usable_match_count": int(prediction_match.get("usable_match_count", 0) or 0),
            "same_stack_name_variant_only_count": int(prediction_match.get("same_stack_name_variant_only_count", 0) or 0),
            "same_sample_side_other_roi_count": int(prediction_match.get("same_sample_side_other_roi_count", 0) or 0),
            "ambiguous_suffix_only_count": int(prediction_match.get("ambiguous_suffix_only_count", 0) or 0),
            "missing_count": int(prediction_match.get("missing_count", 0) or 0),
            "stitched_prediction_ready": bool(prediction_resolved_count > 0),
        },
        "prediction_match": prediction_match,
    }


def export_confocal_step7_session(
    *,
    myelin_label: str,
    myelin_section_dir: Path,
    out_root: Path,
    fixed_rgb: np.ndarray,
    fixed_info: dict[str, Any] | None,
    confocal_projection_u8: np.ndarray,
    confocal_sources: list[Path],
    confocal_source_mode: str,
    nominal_overlap_fraction: float,
    projection_info: dict[str, Any] | None,
    projection_mode: str,
    channel_index: int,
    registration_input_profile: str,
    target_working_um_per_px: float,
    tx_px: float,
    ty_px: float,
    angle_deg: float,
    scale: float,
    flip_lr: bool,
    flip_ud: bool,
    anchor_pairs: list[dict[str, Any]] | None,
    tile_defs: list[dict[str, Any]],
    tile_rows: list[dict[str, Any]],
    accepted_tile_indices: list[int] | None,
    frozen_tile_indices: list[int] | None,
    hold_tile_indices: list[int] | None,
    frontier_tile_indices: list[int] | None,
    selected_tile_indices: list[int] | None = None,
    preview_scene_rgb: np.ndarray | None = None,
    seed_screen_run_dir: Path | None = None,
    frontier_run_dir: Path | None = None,
    prediction_root: Path | None = None,
) -> dict[str, Any]:
    session_name = f"step7_session_export_{_safe_dir_component(myelin_label)}_{_utc_stamp()}"
    run_dir = Path(out_root) / session_name
    run_dir.mkdir(parents=True, exist_ok=True)
    refs_dir = run_dir / "reference"
    refs_dir.mkdir(parents=True, exist_ok=True)

    def _affine_points_xy(mat_2x3: np.ndarray, xy_points: list[list[float]]) -> list[list[float]]:
        pts = np.asarray(xy_points, dtype=np.float32)
        if pts.size == 0:
            return []
        aug = np.concatenate([pts, np.ones((pts.shape[0], 1), dtype=np.float32)], axis=1)
        dst = aug @ np.asarray(mat_2x3, dtype=np.float32).T
        return [[float(v[0]), float(v[1])] for v in dst]

    def _scene_polygon_and_bbox(mat_2x3: np.ndarray, raw_bbox_xyxy: list[int]) -> tuple[list[list[float]], list[float], list[float]]:
        x0, y0, x1, y1 = [float(v) for v in raw_bbox_xyxy]
        polygon = _affine_points_xy(
            mat_2x3,
            [[x0, y0], [x1, y0], [x1, y1], [x0, y1]],
        )
        if not polygon:
            return [], [float("nan")] * 4, [float("nan"), float("nan")]
        xs = [float(v[0]) for v in polygon]
        ys = [float(v[1]) for v in polygon]
        bbox = [float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))]
        center = [float(np.mean(xs)), float(np.mean(ys))]
        return polygon, bbox, center

    fixed_preview_path = refs_dir / "myelin_fixed_preview.png"
    confocal_native_path = refs_dir / "confocal_projection_native.png"
    confocal_inverted_path = refs_dir / "confocal_projection_inverted.png"
    cv2.imwrite(str(fixed_preview_path), cv2.cvtColor(np.asarray(fixed_rgb, dtype=np.uint8), cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(confocal_native_path), np.asarray(confocal_projection_u8, dtype=np.uint8))
    cv2.imwrite(str(confocal_inverted_path), _invert_confocal_u8(np.asarray(confocal_projection_u8, dtype=np.uint8)))

    fixed_shape_hw = [int(fixed_rgb.shape[0]), int(fixed_rgb.shape[1])]
    scene_contract = _resolve_step7_scene_to_full_crop_contract(
        myelin_section_dir=Path(myelin_section_dir),
        preview_shape_hw=(int(fixed_shape_hw[0]), int(fixed_shape_hw[1])),
        support_bbox_canvas_xywh=(fixed_info or {}).get("support_bbox_canvas_xywh"),
    )
    projection_shape_hw = [int(confocal_projection_u8.shape[0]), int(confocal_projection_u8.shape[1])]
    raw_shape_hw = tuple(
        np.asarray(
            (projection_info or {}).get("raw_projection_shape_hw") or projection_shape_hw,
            dtype=np.int32,
        ).tolist()
    )
    raw_to_scaled_mat = _raw_to_scaled_affine(
        raw_shape_hw=(int(raw_shape_hw[0]), int(raw_shape_hw[1])),
        scaled_shape_hw=(int(projection_shape_hw[0]), int(projection_shape_hw[1])),
    )
    base_manual_mat = build_manual_affine(
        tuple(projection_shape_hw),
        tuple(fixed_shape_hw),
        tx_px=float(tx_px),
        ty_px=float(ty_px),
        angle_deg=float(angle_deg),
        scale=float(scale),
        flip_lr=bool(flip_lr),
        flip_ud=bool(flip_ud),
    )
    base_manual_raw_mat = _compose_affine_2x3(np.asarray(base_manual_mat, dtype=np.float32), raw_to_scaled_mat)

    rows_by_index = {
        int(row.get("tile_index", -1)): dict(row)
        for row in list(tile_rows or [])
        if int(row.get("tile_index", -1)) >= 0
    }
    accepted_set = {int(v) for v in list(accepted_tile_indices or [])}
    frozen_set = {int(v) for v in list(frozen_tile_indices or [])}
    accepted_only_set = set(accepted_set) - set(frozen_set)
    hold_set = {int(v) for v in list(hold_tile_indices or [])}
    frontier_set = {int(v) for v in list(frontier_tile_indices or [])}
    selected_set = {int(v) for v in list(selected_tile_indices or [])}
    row_count = 1 + max((int(tile.get("row_display", 0)) for tile in list(tile_defs or [])), default=0)
    col_count = 1 + max((int(tile.get("col_display", 0)) for tile in list(tile_defs or [])), default=0)

    tile_records: list[dict[str, Any]] = []
    tile_csv_rows: list[dict[str, Any]] = []
    storyboard_rows: list[dict[str, Any]] = []
    for tile in list(tile_defs or []):
        tile_index = int(tile.get("tile_index", -1))
        label = str(tile.get("label") or f"T{tile_index:02d}")
        row = rows_by_index.get(tile_index, {})
        raw_bbox_xyxy = [int(v) for v in list(tile.get("raw_bbox_xyxy") or [0, 0, 0, 0])[:4]]
        display_bbox_xyxy = [int(v) for v in list(tile.get("display_bbox_xyxy") or [0, 0, 0, 0])[:4]]
        state = str(row.get("tile_state") or "").strip().lower()
        if not state:
            if tile_index in frozen_set:
                state = TileState.FROZEN.value
            elif tile_index in accepted_set:
                state = TileState.ACCEPTED.value
            elif tile_index in hold_set:
                state = TileState.HOLD.value
            elif tile_index in frontier_set:
                state = TileState.FRONTIER.value
            else:
                state = TileState.UNSEEN.value
        pred_dx = float(row.get("pred_dx_px", row.get("prior_shift_dx_px", 0.0)) or 0.0)
        pred_dy = float(row.get("pred_dy_px", row.get("prior_shift_dy_px", 0.0)) or 0.0)
        meas_dx = float(row.get("meas_dx_px", row.get("candidate_shift_dx_px", 0.0)) or 0.0)
        meas_dy = float(row.get("meas_dy_px", row.get("candidate_shift_dy_px", 0.0)) or 0.0)
        final_dx = float(row.get("final_dx_px", row.get("best_shift_dx_px", 0.0)) or 0.0)
        final_dy = float(row.get("final_dy_px", row.get("best_shift_dy_px", 0.0)) or 0.0)
        pred_mat = np.asarray(base_manual_raw_mat, dtype=np.float32).copy()
        pred_mat[:, 2] += np.array([pred_dx, pred_dy], dtype=np.float32)
        final_mat = np.asarray(base_manual_raw_mat, dtype=np.float32).copy()
        final_mat[:, 2] += np.array([final_dx, final_dy], dtype=np.float32)
        nominal_poly, nominal_bbox, nominal_center = _scene_polygon_and_bbox(np.asarray(base_manual_raw_mat, dtype=np.float32), raw_bbox_xyxy)
        final_poly, final_bbox, final_center = _scene_polygon_and_bbox(final_mat, raw_bbox_xyxy)
        pred_poly, pred_bbox, pred_center = _scene_polygon_and_bbox(pred_mat, raw_bbox_xyxy)
        record = {
            "tile_index": int(tile_index),
            "label": label,
            "row_display": int(tile.get("row_display", row.get("row_display", 0))),
            "col_display": int(tile.get("col_display", row.get("col_display", 0))),
            "tile_state": state,
            "selected": bool(tile_index in selected_set),
            "accepted": bool(tile_index in accepted_only_set),
            "frozen": bool(tile_index in frozen_set),
            "hold": bool(tile_index in hold_set),
            "frontier": bool(tile_index in frontier_set),
            "raw_bbox_xyxy": raw_bbox_xyxy,
            "display_bbox_xyxy": display_bbox_xyxy,
            "manual_affine_matrix_2x3": np.asarray(base_manual_raw_mat, dtype=np.float32).tolist(),
            "manual_affine_scaled_projection_matrix_2x3": np.asarray(base_manual_mat, dtype=np.float32).tolist(),
            "pred_affine_matrix_2x3": pred_mat.tolist(),
            "final_affine_matrix_2x3": final_mat.tolist(),
            "nominal_scene_polygon_xy": nominal_poly,
            "pred_scene_polygon_xy": pred_poly,
            "final_scene_polygon_xy": final_poly,
            "nominal_scene_bbox_xyxy": nominal_bbox,
            "pred_scene_bbox_xyxy": pred_bbox,
            "final_scene_bbox_xyxy": final_bbox,
            "nominal_scene_center_xy": nominal_center,
            "pred_scene_center_xy": pred_center,
            "final_scene_center_xy": final_center,
            "pred_dx_px": pred_dx,
            "pred_dy_px": pred_dy,
            "meas_dx_px": meas_dx,
            "meas_dy_px": meas_dy,
            "final_dx_px": final_dx,
            "final_dy_px": final_dy,
            "current_cc": float(row.get("current_cc", float("nan"))),
            "final_cc": float(row.get("final_cc", row.get("shifted_cc", float("nan")))),
            "shift_gain_cc": float(row.get("shift_gain_cc", float("nan"))),
            "candidate_shifted_cc": float(row.get("candidate_shifted_cc", float("nan"))),
            "current_mi": float(row.get("current_mi", float("nan"))),
            "template_match_score": float(row.get("template_match_score", float("nan"))),
            "proposal_gate": str(row.get("proposal_gate") or ""),
            "registration_profile": str(row.get("registration_profile") or registration_input_profile),
            "refine_objective": str(row.get("refine_objective") or "cc"),
            "density_regime": str(row.get("density_regime") or ""),
            "signal_coverage": float(row.get("signal_coverage", float("nan"))),
            "frontier_confidence": float(row.get("frontier_confidence", row.get("seed_score", float("nan")))),
            "graph_residual": float(row.get("graph_residual", float("nan"))),
            "source_path": str(confocal_sources[tile_index]) if tile_index < len(confocal_sources) else "",
        }
        record = _augment_step7_tile_record_full_crop_geometry(
            record,
            scene_to_full_crop_mat=np.asarray(scene_contract["scene_to_full_crop_mat"], dtype=np.float32),
        )
        tile_records.append(record)
        tile_csv_rows.append(
            {
                "tile_index": int(tile_index),
                "label": label,
                "row_display": int(record["row_display"]),
                "col_display": int(record["col_display"]),
                "tile_state": state,
                "selected": int(bool(record["selected"])),
                "accepted": int(bool(record["accepted"])),
                "frozen": int(bool(record["frozen"])),
                "hold": int(bool(record["hold"])),
                "frontier": int(bool(record["frontier"])),
                "raw_x0": int(raw_bbox_xyxy[0]),
                "raw_y0": int(raw_bbox_xyxy[1]),
                "raw_x1": int(raw_bbox_xyxy[2]),
                "raw_y1": int(raw_bbox_xyxy[3]),
                "final_dx_px": float(final_dx),
                "final_dy_px": float(final_dy),
                "pred_dx_px": float(pred_dx),
                "pred_dy_px": float(pred_dy),
                "scene_center_x": float(final_center[0]),
                "scene_center_y": float(final_center[1]),
                "scene_bbox_x0": float(final_bbox[0]),
                "scene_bbox_y0": float(final_bbox[1]),
                "scene_bbox_x1": float(final_bbox[2]),
                "scene_bbox_y1": float(final_bbox[3]),
                "full_crop_center_x": float(record["final_full_crop_center_xy"][0]),
                "full_crop_center_y": float(record["final_full_crop_center_xy"][1]),
                "full_crop_bbox_x0": float(record["final_full_crop_bbox_xyxy"][0]),
                "full_crop_bbox_y0": float(record["final_full_crop_bbox_xyxy"][1]),
                "full_crop_bbox_x1": float(record["final_full_crop_bbox_xyxy"][2]),
                "full_crop_bbox_y1": float(record["final_full_crop_bbox_xyxy"][3]),
                "current_cc": float(record["current_cc"]),
                "final_cc": float(record["final_cc"]),
                "shift_gain_cc": float(record["shift_gain_cc"]),
                "template_match_score": float(record["template_match_score"]),
                "proposal_gate": str(record["proposal_gate"]),
                "registration_profile": str(record["registration_profile"]),
                "refine_objective": str(record["refine_objective"]),
                "density_regime": str(record["density_regime"]),
                "signal_coverage": float(record["signal_coverage"]),
                "frontier_confidence": float(record["frontier_confidence"]),
                "graph_residual": float(record["graph_residual"]),
                "final_affine_matrix_2x3_json": json.dumps(final_mat.tolist()),
                "final_scene_polygon_xy_json": json.dumps(final_poly),
                "final_full_crop_affine_matrix_2x3_json": json.dumps(record["final_full_crop_affine_matrix_2x3"]),
                "final_full_crop_polygon_xy_json": json.dumps(record["final_full_crop_polygon_xy"]),
            }
        )
        if len(storyboard_rows) < 12 and isinstance(row.get("moving"), np.ndarray) and isinstance(row.get("fixed"), np.ndarray):
            moving_panel = _annotate_step7_storyboard_panel(
                np.asarray(row.get("moving")),
                row_display=int(record["row_display"]),
                col_display=int(record["col_display"]),
                row_count=int(row_count),
                col_count=int(col_count),
                shift_dx_px=float(final_dx),
                shift_dy_px=float(final_dy),
                tile_state=str(state),
            )
            fixed_panel = _annotate_step7_storyboard_panel(
                np.asarray(row.get("fixed")),
                row_display=int(record["row_display"]),
                col_display=int(record["col_display"]),
                row_count=int(row_count),
                col_count=int(col_count),
                shift_dx_px=float(final_dx),
                shift_dy_px=float(final_dy),
                tile_state=str(state),
            )
            overlay_panel = _annotate_step7_storyboard_panel(
                np.asarray(row.get("overlay")),
                row_display=int(record["row_display"]),
                col_display=int(record["col_display"]),
                row_count=int(row_count),
                col_count=int(col_count),
                shift_dx_px=float(final_dx),
                shift_dy_px=float(final_dy),
                tile_state=str(state),
            )
            heatmap_panel = _annotate_step7_storyboard_panel(
                np.asarray(row.get("heatmap")),
                row_display=int(record["row_display"]),
                col_display=int(record["col_display"]),
                row_count=int(row_count),
                col_count=int(col_count),
                shift_dx_px=float(final_dx),
                shift_dy_px=float(final_dy),
                tile_state=str(state),
            )
            scene_context_panel = _build_step7_scene_context_panel(
                np.asarray(fixed_rgb, dtype=np.uint8),
                nominal_scene_polygon_xy=list(record.get("nominal_scene_polygon_xy") or []),
                final_scene_polygon_xy=list(record.get("final_scene_polygon_xy") or []),
                nominal_scene_center_xy=list(record.get("nominal_scene_center_xy") or []),
                final_scene_center_xy=list(record.get("final_scene_center_xy") or []),
                tile_state=str(state),
                out_shape_hw=tuple(int(v) for v in moving_panel.shape[:2]),
            )
            storyboard_rows.append(
                {
                    "label": label,
                    "note": (
                        f"state={state} | rc=({int(record['row_display'])},{int(record['col_display'])}) | "
                        f"final_CC={float(record['final_cc']):.4f} | "
                        f"shift=({int(round(final_dx))},{int(round(final_dy))}) | "
                        f"profile={record['registration_profile']}"
                    ),
                    "moving": moving_panel,
                    "fixed": fixed_panel,
                    "overlay": overlay_panel,
                    "heatmap": heatmap_panel,
                    "scene_context": scene_context_panel,
                    "panel_keys": ("moving", "fixed", "overlay", "heatmap", "scene_context"),
                    "col_titles": (
                        tuple(str(v) for v in list(row.get("col_titles") or ())[:4]) + ("Scene context",)
                        if isinstance(row.get("col_titles"), (list, tuple))
                        else ("Raw overlay current", "Raw overlay shifted", "Processed overlay current", "Processed overlay shifted", "Scene context")
                    ),
                }
            )

    tile_records.sort(key=lambda r: (int({"frozen": 0, "accepted": 1, "frontier": 2, "hold": 3}.get(str(r["tile_state"]), 9)), int(r["row_display"]), int(r["col_display"])))
    tile_csv_rows.sort(key=lambda r: (int(r["row_display"]), int(r["col_display"])))

    tile_csv_path = run_dir / "tile_transforms.csv"
    if tile_csv_rows:
        with tile_csv_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(tile_csv_rows[0].keys()))
            writer.writeheader()
            writer.writerows(tile_csv_rows)
    tile_records_path = run_dir / "tile_records.json"
    _write_json(tile_records_path, {"rows": tile_records})

    storyboard_path = run_dir / "tile_qc_storyboard.png"
    if storyboard_rows:
        render_storyboard(storyboard_rows, storyboard_path)

    seed_manifest_path = (Path(seed_screen_run_dir) / "seed_tile_manifest.json") if seed_screen_run_dir is not None else None
    frontier_manifest_path = (Path(frontier_run_dir) / "frontier_manifest.json") if frontier_run_dir is not None else None
    source_runs = {
        "seed_screen_run_dir": str(seed_screen_run_dir) if seed_screen_run_dir is not None else "",
        "seed_screen_manifest": str(seed_manifest_path) if seed_manifest_path is not None and seed_manifest_path.exists() else "",
        "frontier_run_dir": str(frontier_run_dir) if frontier_run_dir is not None else "",
        "frontier_manifest": str(frontier_manifest_path) if frontier_manifest_path is not None and frontier_manifest_path.exists() else "",
    }

    state_counts = {
        "tile_count": int(len(tile_defs)),
        "accepted_count": int(len(accepted_only_set)),
        "frozen_count": int(len(frozen_set)),
        "hold_count": int(len(hold_set)),
        "frontier_count": int(len(frontier_set)),
        "selected_count": int(len(selected_set)),
    }
    export_products = _build_step7_session_export_products(
        run_dir=run_dir,
        fixed_rgb=np.asarray(fixed_rgb, dtype=np.uint8),
        confocal_sources=[Path(p) for p in list(confocal_sources or [])],
        confocal_source_mode=str(confocal_source_mode),
        projection_mode=str(projection_mode),
        channel_index=int(channel_index),
        tile_records=tile_records,
        preview_scene_rgb=preview_scene_rgb,
        prediction_root=prediction_root,
    )

    session_manifest = {
        "schema": "step7_tile_session_export_v1",
        "saved_at_utc": _utc_iso(),
        "myelin_label": str(myelin_label),
        "myelin_section_dir": str(myelin_section_dir),
        "run_dir": str(run_dir),
        "confocal_sources": [str(path) for path in list(confocal_sources or [])],
        "confocal_source_mode": str(confocal_source_mode),
        "nominal_overlap_fraction": float(nominal_overlap_fraction),
        "projection_mode": str(projection_mode),
        "channel_index": int(channel_index),
        "registration_input_profile": str(registration_input_profile),
        "projection_info": projection_info or {},
        "fixed_preview_shape_hw": fixed_shape_hw,
        "fixed_support_bbox_canvas_xywh": [int(v) for v in list(scene_contract["fixed_support_bbox_canvas_xywh"])],
        "fixed_support_shape_hw": [int(v) for v in list(scene_contract["fixed_support_shape_hw"])],
        "full_crop_shape_hw": [int(v) for v in list(scene_contract["full_crop_shape_hw"])],
        "scene_to_full_crop_affine_matrix_2x3": np.asarray(
            scene_contract["scene_to_full_crop_mat"],
            dtype=np.float32,
        ).tolist(),
        "confocal_projection_scaled_shape_hw": projection_shape_hw,
        "fixed_preview_um_per_px_xy": list((fixed_info or {}).get("preview_um_per_px_xy") or []),
        "source_um_per_px_xy": list((projection_info or {}).get("source_um_per_px_xy") or []),
        "target_working_um_per_px": float(target_working_um_per_px),
        "manual_init": {
            "tx_px": float(tx_px),
            "ty_px": float(ty_px),
            "angle_deg": float(angle_deg),
            "scale": float(scale),
            "flip_lr": bool(flip_lr),
            "flip_ud": bool(flip_ud),
            "base_affine_matrix_2x3": np.asarray(base_manual_raw_mat, dtype=np.float32).tolist(),
            "base_affine_scaled_projection_matrix_2x3": np.asarray(base_manual_mat, dtype=np.float32).tolist(),
            "raw_to_scaled_affine_matrix_2x3": np.asarray(raw_to_scaled_mat, dtype=np.float32).tolist(),
        },
        "anchor_pairs": list(anchor_pairs or []),
        "state_counts": state_counts,
        "accepted_tile_indices": sorted(int(v) for v in accepted_only_set),
        "frozen_tile_indices": sorted(int(v) for v in frozen_set),
        "hold_tile_indices": sorted(int(v) for v in hold_set),
        "frontier_tile_indices": sorted(int(v) for v in frontier_set),
        "selected_tile_indices": sorted(int(v) for v in selected_set),
        "source_runs": source_runs,
        "files": {
            "preview_scene_full": str(export_products["files"].get("preview_scene_full") or ""),
            "preview_scene_gui_snapshot": str(export_products["files"].get("preview_scene_gui_snapshot") or ""),
            "preview_scene_rebuilt": str(export_products["files"].get("preview_scene_rebuilt") or ""),
            "myelin_fixed_preview": str(fixed_preview_path),
            "confocal_projection_native": str(confocal_native_path),
            "confocal_projection_inverted": str(confocal_inverted_path),
            "tile_qc_storyboard": str(storyboard_path) if storyboard_path.exists() else "",
            "tile_qc_storyboard_legacy4col": "",
            "tile_transforms_csv": str(tile_csv_path),
            "tile_records_json": str(tile_records_path),
            "seam_qc_csv": str(export_products["files"].get("seam_qc_csv") or ""),
            "seam_qc_json": str(export_products["files"].get("seam_qc_json") or ""),
            "stitched_confocal_raw_hard": str(export_products["files"].get("stitched_confocal_raw_hard") or ""),
            "stitched_confocal_raw_blended": str(export_products["files"].get("stitched_confocal_raw_blended") or ""),
            "stitched_confocal_support": str(export_products["files"].get("stitched_confocal_support") or ""),
            "stitched_confocal_tile_id": str(export_products["files"].get("stitched_confocal_tile_id") or ""),
            "prediction_match_json": str(export_products["files"].get("prediction_match_json") or ""),
            "stitched_prediction_probability": str(export_products["files"].get("stitched_prediction_probability") or ""),
            "stitched_prediction_mask": str(export_products["files"].get("stitched_prediction_mask") or ""),
            "stitched_prediction_support": str(export_products["files"].get("stitched_prediction_support") or ""),
            "step8_overlay_scene_probability": str(export_products["files"].get("step8_overlay_scene_probability") or ""),
            "step8_overlay_scene_mask": str(export_products["files"].get("step8_overlay_scene_mask") or ""),
            "step8_overlay_qc": str(export_products["files"].get("step8_overlay_qc") or ""),
            "step8_handoff": str(run_dir / "step8_handoff.json"),
            "manifest": str(run_dir / "session_manifest.json"),
            "readme": str(run_dir / "README.md"),
        },
        "documentation_scope": {
            "whole_section_preview_relation": True,
            "manual_coarse_transform": True,
            "tile_states": True,
            "tile_absolute_transforms": True,
            "seed_frontier_source_refs": True,
            "step8_handoff_ready": True,
            "seam_qc_ready": True,
            "stitched_confocal_ready": True,
        },
        "stitched_scene": {
            "crop_bbox_xyxy": [int(v) for v in list(export_products.get("scene_crop_bbox_xyxy") or [])],
            "trusted_tile_count": int(export_products.get("trusted_tile_count", 0) or 0),
        },
        "seam_qc": dict(export_products.get("seam_summary") or {}),
        "prediction_matching": dict(export_products.get("prediction_summary") or {}),
    }
    _write_json(run_dir / "session_manifest.json", session_manifest)

    step8_handoff = {
        "schema": "step8_fiber_density_handoff_v1",
        "saved_at_utc": session_manifest["saved_at_utc"],
        "step7_export_dir": str(run_dir),
        "myelin_label": str(myelin_label),
        "myelin_section_dir": str(myelin_section_dir),
        "scene_space": {
            "name": "step7_preview_scene",
            "working_um_per_px": float(target_working_um_per_px),
            "fixed_preview_shape_hw": fixed_shape_hw,
            "fixed_support_bbox_canvas_xywh": [int(v) for v in list(scene_contract["fixed_support_bbox_canvas_xywh"])],
            "fixed_support_shape_hw": [int(v) for v in list(scene_contract["fixed_support_shape_hw"])],
            "full_crop_shape_hw": [int(v) for v in list(scene_contract["full_crop_shape_hw"])],
            "scene_to_full_crop_affine_matrix_2x3": np.asarray(
                scene_contract["scene_to_full_crop_mat"],
                dtype=np.float32,
            ).tolist(),
        },
        "fixed_preview_um_per_px_xy": session_manifest["fixed_preview_um_per_px_xy"],
        "source_um_per_px_xy": session_manifest["source_um_per_px_xy"],
        "manual_init": session_manifest["manual_init"],
        "projection_info": projection_info or {},
        "source_runs": source_runs,
        "tile_records": tile_records,
        "stitched_scene": {
            "crop_bbox_xyxy": [int(v) for v in list(export_products.get("scene_crop_bbox_xyxy") or [])],
            "files": {
                "stitched_confocal_raw_hard": str(export_products["files"].get("stitched_confocal_raw_hard") or ""),
                "stitched_confocal_raw_blended": str(export_products["files"].get("stitched_confocal_raw_blended") or ""),
                "stitched_confocal_support": str(export_products["files"].get("stitched_confocal_support") or ""),
                "stitched_confocal_tile_id": str(export_products["files"].get("stitched_confocal_tile_id") or ""),
            },
        },
        "seam_qc": {
            "summary": dict(export_products.get("seam_summary") or {}),
            "files": {
                "seam_qc_csv": str(export_products["files"].get("seam_qc_csv") or ""),
                "seam_qc_json": str(export_products["files"].get("seam_qc_json") or ""),
            },
        },
        "prediction_import": {
            "summary": dict(export_products.get("prediction_summary") or {}),
            "files": {
                "prediction_match_json": str(export_products["files"].get("prediction_match_json") or ""),
                "stitched_prediction_probability": str(export_products["files"].get("stitched_prediction_probability") or ""),
                "stitched_prediction_mask": str(export_products["files"].get("stitched_prediction_mask") or ""),
                "stitched_prediction_support": str(export_products["files"].get("stitched_prediction_support") or ""),
                "step8_overlay_scene_probability": str(export_products["files"].get("step8_overlay_scene_probability") or ""),
                "step8_overlay_scene_mask": str(export_products["files"].get("step8_overlay_scene_mask") or ""),
                "step8_overlay_qc": str(export_products["files"].get("step8_overlay_qc") or ""),
            },
        },
        "notes": [
            "final_affine_matrix_2x3 maps raw confocal projection coordinates into the Step 7 myelin preview scene",
            "final_full_crop_affine_matrix_2x3 maps raw confocal projection coordinates into the canonical myelin full-crop canvas",
            "final_scene_polygon_xy is the recommended geometry input for downstream tile-wise overlay and density analysis",
            "final_full_crop_polygon_xy is the recommended geometry input for Step 6 full-crop overlay and Step 5 bridge mapping",
            "tile_state can be used to restrict Step 8 to frozen/accepted tiles first",
            "stitched_scene.crop_bbox_xyxy defines the scene-space crop used by stitched confocal/prediction exports",
        ],
    }
    _write_json(run_dir / "step8_handoff.json", step8_handoff)
    (run_dir / "README.md").write_text("\n".join(_build_step7_session_readme_lines(session_manifest)), encoding="utf-8")

    return {
        "run_dir": str(run_dir),
        "preview_scene_full": str(session_manifest["files"].get("preview_scene_full") or ""),
        "preview_scene_rebuilt": str(session_manifest["files"].get("preview_scene_rebuilt") or ""),
        "tile_qc_storyboard": str(storyboard_path) if storyboard_path.exists() else "",
        "tile_transforms_csv": str(tile_csv_path),
        "tile_records_json": str(tile_records_path),
        "seam_qc_csv": str(session_manifest["files"].get("seam_qc_csv") or ""),
        "stitched_confocal_raw_hard": str(session_manifest["files"].get("stitched_confocal_raw_hard") or ""),
        "stitched_confocal_raw_blended": str(session_manifest["files"].get("stitched_confocal_raw_blended") or ""),
        "prediction_match_json": str(session_manifest["files"].get("prediction_match_json") or ""),
        "stitched_prediction_probability": str(session_manifest["files"].get("stitched_prediction_probability") or ""),
        "step8_overlay_qc": str(session_manifest["files"].get("step8_overlay_qc") or ""),
        "step8_handoff": str(run_dir / "step8_handoff.json"),
        "session_manifest": str(run_dir / "session_manifest.json"),
        "readme": str(run_dir / "README.md"),
    }


def repair_exported_step7_session(
    export_dir: Path,
    *,
    prediction_root: Path | None = None,
) -> dict[str, Any]:
    run_dir = Path(export_dir)
    manifest_path = run_dir / "session_manifest.json"
    tile_records_path = run_dir / "tile_records.json"
    fixed_preview_path = run_dir / "reference" / "myelin_fixed_preview.png"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing session manifest: {manifest_path}")
    if not tile_records_path.exists():
        raise FileNotFoundError(f"Missing tile records: {tile_records_path}")
    if not fixed_preview_path.exists():
        raise FileNotFoundError(f"Missing fixed preview: {fixed_preview_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    tile_records_payload = json.loads(tile_records_path.read_text(encoding="utf-8"))
    tile_records = [dict(row) for row in list(tile_records_payload.get("rows") or [])]
    step8_handoff_path = run_dir / "step8_handoff.json"
    handoff = json.loads(step8_handoff_path.read_text(encoding="utf-8")) if step8_handoff_path.exists() else {}
    fixed_bgr = cv2.imread(str(fixed_preview_path), cv2.IMREAD_COLOR)
    if fixed_bgr is None:
        raise RuntimeError(f"Could not read fixed preview image: {fixed_preview_path}")
    fixed_rgb = cv2.cvtColor(fixed_bgr, cv2.COLOR_BGR2RGB)
    preview_shape_hw = tuple(
        int(v)
        for v in list(
            manifest.get("fixed_preview_shape_hw")
            or ((handoff.get("scene_space") or {}).get("fixed_preview_shape_hw"))
            or list(fixed_rgb.shape[:2])
        )[:2]
    )
    myelin_section_dir_raw = str(
        manifest.get("myelin_section_dir")
        or handoff.get("myelin_section_dir")
        or ""
    ).strip()
    myelin_section_dir = _runtime_local_path(myelin_section_dir_raw) if myelin_section_dir_raw else None
    support_bbox_canvas_xywh = (
        manifest.get("fixed_support_bbox_canvas_xywh")
        or ((handoff.get("scene_space") or {}).get("fixed_support_bbox_canvas_xywh"))
    )
    scene_contract = _resolve_step7_scene_to_full_crop_contract(
        myelin_section_dir=myelin_section_dir,
        preview_shape_hw=(int(preview_shape_hw[0]), int(preview_shape_hw[1])),
        support_bbox_canvas_xywh=support_bbox_canvas_xywh,
    )
    manual_init = manifest.get("manual_init") if isinstance(manifest.get("manual_init"), dict) else {}
    projection_info = manifest.get("projection_info") if isinstance(manifest.get("projection_info"), dict) else {}
    base_manual_scaled_mat = np.asarray(
        manual_init.get("base_affine_scaled_projection_matrix_2x3")
        or manual_init.get("base_affine_matrix_2x3")
        or build_manual_affine(
            tuple(int(v) for v in list(manifest.get("confocal_projection_scaled_shape_hw") or [1, 1])[:2]),
            tuple(int(v) for v in list(manifest.get("fixed_preview_shape_hw") or list(fixed_rgb.shape[:2]))[:2]),
            tx_px=float(manual_init.get("tx_px", 0.0) or 0.0),
            ty_px=float(manual_init.get("ty_px", 0.0) or 0.0),
            angle_deg=float(manual_init.get("angle_deg", 0.0) or 0.0),
            scale=float(manual_init.get("scale", 1.0) or 1.0),
            flip_lr=bool(manual_init.get("flip_lr", False)),
            flip_ud=bool(manual_init.get("flip_ud", False)),
        ),
        dtype=np.float32,
    )
    raw_shape_hw = tuple(
        int(v)
        for v in list(
            projection_info.get("raw_projection_shape_hw")
            or ((projection_info.get("scale_to_section_preview") or {}).get("input_shape_hw"))
            or [1, 1]
        )[:2]
    )
    scaled_shape_hw = tuple(
        int(v)
        for v in list(
            manifest.get("confocal_projection_scaled_shape_hw")
            or projection_info.get("scaled_projection_shape_hw")
            or ((projection_info.get("scale_to_section_preview") or {}).get("output_shape_hw"))
            or [1, 1]
        )[:2]
    )
    tile_records, base_manual_raw_mat, raw_to_scaled_mat = _rebuild_step7_export_tile_record_geometry(
        tile_records,
        base_manual_scaled_mat=base_manual_scaled_mat,
        raw_shape_hw=raw_shape_hw,
        scaled_shape_hw=scaled_shape_hw,
    )
    tile_records = [
        _augment_step7_tile_record_full_crop_geometry(
            dict(row),
            scene_to_full_crop_mat=np.asarray(scene_contract["scene_to_full_crop_mat"], dtype=np.float32),
        )
        for row in list(tile_records or [])
    ]
    _write_json(tile_records_path, {"rows": tile_records})
    if not isinstance(manifest.get("manual_init"), dict):
        manifest["manual_init"] = {}
    manifest["manual_init"].update(
        {
            "base_affine_matrix_2x3": np.asarray(base_manual_raw_mat, dtype=np.float32).tolist(),
            "base_affine_scaled_projection_matrix_2x3": np.asarray(base_manual_scaled_mat, dtype=np.float32).tolist(),
            "raw_to_scaled_affine_matrix_2x3": np.asarray(raw_to_scaled_mat, dtype=np.float32).tolist(),
        }
    )
    manifest["fixed_preview_shape_hw"] = [int(v) for v in list(scene_contract["preview_shape_hw"])]
    manifest["fixed_support_bbox_canvas_xywh"] = [int(v) for v in list(scene_contract["fixed_support_bbox_canvas_xywh"])]
    manifest["fixed_support_shape_hw"] = [int(v) for v in list(scene_contract["fixed_support_shape_hw"])]
    manifest["full_crop_shape_hw"] = [int(v) for v in list(scene_contract["full_crop_shape_hw"])]
    manifest["scene_to_full_crop_affine_matrix_2x3"] = np.asarray(
        scene_contract["scene_to_full_crop_mat"],
        dtype=np.float32,
    ).tolist()

    export_products = _build_step7_session_export_products(
        run_dir=run_dir,
        fixed_rgb=fixed_rgb,
        confocal_sources=[Path(str(p)) for p in list(manifest.get("confocal_sources") or [])],
        confocal_source_mode=str(manifest.get("confocal_source_mode") or ""),
        projection_mode=str(manifest.get("projection_mode") or "focus"),
        channel_index=int(manifest.get("channel_index") or 0),
        tile_records=tile_records,
        prediction_root=prediction_root,
    )

    files = manifest.get("files") if isinstance(manifest.get("files"), dict) else {}
    files.update(
        {
            "preview_scene_full": str(export_products["files"].get("preview_scene_full") or ""),
            "preview_scene_gui_snapshot": "",
            "preview_scene_rebuilt": "",
            "seam_qc_csv": str(export_products["files"].get("seam_qc_csv") or ""),
            "seam_qc_json": str(export_products["files"].get("seam_qc_json") or ""),
            "stitched_confocal_raw_hard": str(export_products["files"].get("stitched_confocal_raw_hard") or ""),
            "stitched_confocal_raw_blended": str(export_products["files"].get("stitched_confocal_raw_blended") or ""),
            "stitched_confocal_support": str(export_products["files"].get("stitched_confocal_support") or ""),
            "stitched_confocal_tile_id": str(export_products["files"].get("stitched_confocal_tile_id") or ""),
            "prediction_match_json": str(export_products["files"].get("prediction_match_json") or ""),
            "stitched_prediction_probability": str(export_products["files"].get("stitched_prediction_probability") or ""),
            "stitched_prediction_mask": str(export_products["files"].get("stitched_prediction_mask") or ""),
            "stitched_prediction_support": str(export_products["files"].get("stitched_prediction_support") or ""),
            "step8_overlay_scene_probability": str(export_products["files"].get("step8_overlay_scene_probability") or ""),
            "step8_overlay_scene_mask": str(export_products["files"].get("step8_overlay_scene_mask") or ""),
            "step8_overlay_qc": str(export_products["files"].get("step8_overlay_qc") or ""),
        }
    )
    manifest["files"] = files
    storyboard_backup = _upgrade_legacy_step7_storyboard_with_scene_context(
        storyboard_path=run_dir / "tile_qc_storyboard.png",
        fixed_rgb=fixed_rgb,
        tile_records=tile_records,
    )
    if storyboard_backup:
        files["tile_qc_storyboard_legacy4col"] = str(storyboard_backup)
    manifest["stitched_scene"] = {
        "crop_bbox_xyxy": [int(v) for v in list(export_products.get("scene_crop_bbox_xyxy") or [])],
        "trusted_tile_count": int(export_products.get("trusted_tile_count", 0) or 0),
    }
    manifest["seam_qc"] = dict(export_products.get("seam_summary") or {})
    manifest["prediction_matching"] = dict(export_products.get("prediction_summary") or {})
    _write_json(manifest_path, manifest)

    if step8_handoff_path.exists():
        scene_space = handoff.get("scene_space") if isinstance(handoff.get("scene_space"), dict) else {}
        scene_space.update(
            {
                "name": str(scene_space.get("name") or "step7_preview_scene"),
                "fixed_preview_shape_hw": [int(v) for v in list(scene_contract["preview_shape_hw"])],
                "fixed_support_bbox_canvas_xywh": [int(v) for v in list(scene_contract["fixed_support_bbox_canvas_xywh"])],
                "fixed_support_shape_hw": [int(v) for v in list(scene_contract["fixed_support_shape_hw"])],
                "full_crop_shape_hw": [int(v) for v in list(scene_contract["full_crop_shape_hw"])],
                "scene_to_full_crop_affine_matrix_2x3": np.asarray(
                    scene_contract["scene_to_full_crop_mat"],
                    dtype=np.float32,
                ).tolist(),
            }
        )
        handoff["scene_space"] = scene_space
        handoff["tile_records"] = tile_records
        handoff["stitched_scene"] = {
            "crop_bbox_xyxy": [int(v) for v in list(export_products.get("scene_crop_bbox_xyxy") or [])],
            "files": {
                "stitched_confocal_raw_hard": str(export_products["files"].get("stitched_confocal_raw_hard") or ""),
                "stitched_confocal_raw_blended": str(export_products["files"].get("stitched_confocal_raw_blended") or ""),
                "stitched_confocal_support": str(export_products["files"].get("stitched_confocal_support") or ""),
                "stitched_confocal_tile_id": str(export_products["files"].get("stitched_confocal_tile_id") or ""),
            },
        }
        handoff["seam_qc"] = {
            "summary": dict(export_products.get("seam_summary") or {}),
            "files": {
                "seam_qc_csv": str(export_products["files"].get("seam_qc_csv") or ""),
                "seam_qc_json": str(export_products["files"].get("seam_qc_json") or ""),
            },
        }
        handoff["prediction_import"] = {
            "summary": dict(export_products.get("prediction_summary") or {}),
            "files": {
                "prediction_match_json": str(export_products["files"].get("prediction_match_json") or ""),
                "stitched_prediction_probability": str(export_products["files"].get("stitched_prediction_probability") or ""),
                "stitched_prediction_mask": str(export_products["files"].get("stitched_prediction_mask") or ""),
                "stitched_prediction_support": str(export_products["files"].get("stitched_prediction_support") or ""),
                "step8_overlay_scene_probability": str(export_products["files"].get("step8_overlay_scene_probability") or ""),
                "step8_overlay_scene_mask": str(export_products["files"].get("step8_overlay_scene_mask") or ""),
                "step8_overlay_qc": str(export_products["files"].get("step8_overlay_qc") or ""),
            },
        }
        notes = [str(v) for v in list(handoff.get("notes") or []) if str(v).strip()]
        if "final_full_crop_affine_matrix_2x3 maps raw confocal projection coordinates into the canonical myelin full-crop canvas" not in notes:
            notes.append("final_full_crop_affine_matrix_2x3 maps raw confocal projection coordinates into the canonical myelin full-crop canvas")
        if "final_full_crop_polygon_xy is the recommended geometry input for Step 6 full-crop overlay and Step 5 bridge mapping" not in notes:
            notes.append("final_full_crop_polygon_xy is the recommended geometry input for Step 6 full-crop overlay and Step 5 bridge mapping")
        handoff["notes"] = notes
        _write_json(step8_handoff_path, handoff)

    readme_path = run_dir / "README.md"
    readme_path.write_text("\n".join(_build_step7_session_readme_lines(manifest)) + "\n", encoding="utf-8")

    return {
        "run_dir": str(run_dir),
        "session_manifest": str(manifest_path),
        "step8_handoff": str(step8_handoff_path),
        "preview_scene_full": str(files.get("preview_scene_full") or ""),
        "preview_scene_rebuilt": str(files.get("preview_scene_rebuilt") or ""),
        "seam_qc_csv": str(files.get("seam_qc_csv") or ""),
        "stitched_confocal_raw_hard": str(files.get("stitched_confocal_raw_hard") or ""),
        "stitched_confocal_raw_blended": str(files.get("stitched_confocal_raw_blended") or ""),
        "prediction_match_json": str(files.get("prediction_match_json") or ""),
        "stitched_prediction_probability": str(files.get("stitched_prediction_probability") or ""),
        "step8_overlay_qc": str(files.get("step8_overlay_qc") or ""),
    }


def refresh_step8_products_from_handoff(
    handoff_path: Path,
    *,
    prediction_root: Path | None = None,
) -> dict[str, Any]:
    path = Path(handoff_path)
    if not path.exists():
        raise FileNotFoundError(f"Missing Step 8 handoff file: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    export_dir = Path(str(payload.get("step7_export_dir") or path.parent))
    summary = repair_exported_step7_session(export_dir, prediction_root=prediction_root)
    summary["step8_handoff"] = str(path)
    return summary

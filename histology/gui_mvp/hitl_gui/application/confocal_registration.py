from __future__ import annotations

import json
import os
import re
import subprocess
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .pair_registration import (
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
STEP7_REGISTRATION_INPUT_PROFILE = "paired_percentile_blur6"


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

    if profile_key == "paired_percentile_blur4":
        sigma = 4.0
        fixed_proc_u8 = _gaussian_blur_u8(_masked_percentile_normalize_u8(fixed_u8, fixed_mask_u8), fixed_mask_u8, sigma=sigma)
        moving_proc_u8 = _gaussian_blur_u8(_masked_percentile_normalize_u8(moving_u8, moving_mask_u8), moving_mask_u8, sigma=sigma)
        description = "paired percentile normalization + Gaussian blur sigma=4"
    else:
        sigma = 6.0
        fixed_proc_u8 = _gaussian_blur_u8(_masked_percentile_normalize_u8(fixed_u8, fixed_mask_u8), fixed_mask_u8, sigma=sigma)
        moving_proc_u8 = _gaussian_blur_u8(_masked_percentile_normalize_u8(moving_u8, moving_mask_u8), moving_mask_u8, sigma=sigma)
        profile_key = STEP7_REGISTRATION_INPUT_PROFILE
        description = "paired percentile normalization + Gaussian blur sigma=6"

    return {
        "profile": profile_key,
        "description": description,
        "fixed_u8": fixed_proc_u8,
        "moving_u8": moving_proc_u8,
        "fixed_gray": fixed_proc_u8.astype(np.float32) / 255.0,
        "moving_gray": moving_proc_u8.astype(np.float32) / 255.0,
    }


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

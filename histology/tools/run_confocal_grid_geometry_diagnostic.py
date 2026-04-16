from __future__ import annotations

import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception as exc:  # pragma: no cover
    raise RuntimeError("matplotlib is required for diagnostic plotting") from exc


REPO_ROOT = Path(__file__).resolve().parents[3]
GUI_MVP_ROOT = REPO_ROOT / "registration_pipeline" / "histology" / "gui_mvp"
if str(GUI_MVP_ROOT) not in sys.path:
    sys.path.insert(0, str(GUI_MVP_ROOT))

from hitl_gui.application import confocal_registration as confocal_registration_mod  # noqa: E402
from hitl_gui.application import pair_registration as pair_registration_mod  # noqa: E402
from hitl_gui.application.confocal_registration import (  # noqa: E402
    STEP7_TARGET_UM_PER_PX,
    ConfocalRigidConfig,
    _anchor_guided_manual_affine,
    _bbox_from_mask,
    _invert_confocal_u8,
    _load_ants_affine_mat,
    _resample_mask_to_target_um_per_px,
    _resample_projection_to_target_um_per_px,
    apply_affine_matrix,
    build_manual_affine,
    load_confocal_projection,
    prepare_myelin_confocal_fixed_bundle,
    run_confocal_rigid_registration,
)
from hitl_gui.application.pair_registration import (  # noqa: E402
    compute_registration_metrics,
    find_ants_bin,
    gray_preview_panel,
    overlay_preview,
    render_storyboard,
)
from hitl_gui.application.section_workspace import WorkspaceSection  # noqa: E402


LABEL = "2501_60"
SECTION_DIR = Path(
    "/mnt/d/Research/Image Analysis/Nanozoomer scans/20250327 rat myelin quantification/Tissue&Masks/2501_60"
)
CONFOCAL_TILE_DIR = Path(
    "/mnt/d/Research/Image Analysis/Confocal Myelin data/202512_8rats_3ROIs/2501_60_R_IL"
)
OUT_ROOT = Path("/mnt/c/Users/Siqi/Desktop/REVIEW/20260408_confocal_tile_diagnostic_2501_60")
MANUAL_STATE = {
    "tx_px": -155.0,
    "ty_px": -63.0,
    "angle_deg": 0.8,
    "scale": 0.943,
    "flip_lr": False,
    "flip_ud": True,
}
ANCHOR_PAIR = {
    "index": 1,
    "section_scene_xy": [4715.0, 3363.7],
    "confocal_raw_xy": [117.7, 554.3],
    "confocal_scene_xy": [4715.0, 3363.7],
}
SCALE_VALUES = [round(v, 3) for v in np.arange(0.930, 0.970 + 0.0001, 0.002)]
SEARCH_RADIUS = 72
FULL_RENDER_SCALE = 0.28


@dataclass
class TileDef:
    tile_index: int
    row: int
    col: int
    raw_bbox_xyxy: tuple[int, int, int, int]
    scaled_bbox_xyxy: tuple[int, int, int, int]
    center_scaled_xy: tuple[float, float]


@dataclass
class TileWarp:
    tile: TileDef
    source_gray_patch: np.ndarray
    source_mask_patch: np.ndarray
    tile_to_fixed_mat: np.ndarray
    warped_gray_full_patch: np.ndarray
    warped_mask_full_patch: np.ndarray
    warped_full_bbox_yxyx: tuple[int, int, int, int]
    warped_gray_patch: np.ndarray
    warped_mask_patch: np.ndarray
    warped_bbox_yxyx: tuple[int, int, int, int]


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _masked_percentile_normalize_u8(
    image_u8: np.ndarray,
    mask: np.ndarray,
    *,
    lo_pct: float = 1.0,
    hi_pct: float = 99.0,
) -> np.ndarray:
    arr = np.asarray(image_u8, dtype=np.float32)
    valid = np.asarray(mask) > 0
    vals = arr[valid]
    if vals.size == 0:
        return np.zeros(arr.shape[:2], dtype=np.uint8)
    lo = float(np.percentile(vals, lo_pct))
    hi = float(np.percentile(vals, hi_pct))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(vals.min())
        hi = float(vals.max())
    if hi <= lo:
        return np.zeros(arr.shape[:2], dtype=np.uint8)
    out = np.clip((arr - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    return np.clip(np.round(out * 255.0), 0, 255).astype(np.uint8)


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
    out = lut[src].astype(np.uint8)
    return out


def _workspace_item(section_dir: Path, label: str) -> WorkspaceSection:
    metadata_path = section_dir / "metadata.json"
    crop_path = section_dir / "crop_raw.png"
    return WorkspaceSection(
        section_dir=section_dir,
        label=label,
        stain="myelin",
        metadata_path=metadata_path,
        crop_path=crop_path,
        has_masks=True,
        has_prepared_work=False,
        prepared_work_profiles=(),
    )


def _load_inputs() -> dict[str, Any]:
    item = _workspace_item(SECTION_DIR, LABEL)
    fixed = prepare_myelin_confocal_fixed_bundle(item, max_long_edge=None, target_um_per_px=STEP7_TARGET_UM_PER_PX)
    confocal_paths = sorted(CONFOCAL_TILE_DIR.glob("*.ome.tif"))
    if not confocal_paths:
        raise FileNotFoundError(f"No confocal OME-TIFF tiles found under {CONFOCAL_TILE_DIR}")
    bundle = load_confocal_projection(confocal_paths, mode="focus", channel_index=0, nominal_overlap_fraction=0.10)
    raw_projection = np.asarray(bundle.projection_u8, dtype=np.uint8)
    raw_mask = np.where(raw_projection > 0, 255, 0).astype(np.uint8)
    scaled_projection, scale_info = _resample_projection_to_target_um_per_px(
        raw_projection,
        source_um_per_px_xy=bundle.physical_um_per_px_xy,
        target_um_per_px_xy=fixed.preview_um_per_px_xy,
    )
    scaled_mask = _resample_mask_to_target_um_per_px(
        raw_mask,
        source_um_per_px_xy=bundle.physical_um_per_px_xy,
        target_um_per_px_xy=fixed.preview_um_per_px_xy,
    )
    scaled_mask = np.where(np.asarray(scaled_mask) > 0, 255, 0).astype(np.uint8)
    return {
        "item": item,
        "fixed_bundle": fixed,
        "confocal_paths": confocal_paths,
        "projection_bundle": bundle,
        "raw_projection_u8": raw_projection,
        "scaled_projection_u8": np.asarray(scaled_projection, dtype=np.uint8),
        "scaled_signal_mask_u8": scaled_mask,
        "scale_info": scale_info,
    }


def _build_tile_defs(
    stitch_info: dict[str, Any],
    *,
    raw_shape_hw: tuple[int, int],
    scaled_shape_hw: tuple[int, int],
) -> list[TileDef]:
    positions = stitch_info.get("tile_positions_xy")
    tile_shape = stitch_info.get("tile_shape_hw")
    if not isinstance(positions, list) or not isinstance(tile_shape, (list, tuple)) or len(tile_shape) != 2:
        raise ValueError("Missing tile position metadata for multi_tiff_grid diagnostic")
    raw_h, raw_w = int(raw_shape_hw[0]), int(raw_shape_hw[1])
    scaled_h, scaled_w = int(scaled_shape_hw[0]), int(scaled_shape_hw[1])
    tile_h, tile_w = int(tile_shape[0]), int(tile_shape[1])
    unique_x = sorted({int(p[0]) for p in positions})
    unique_y = sorted({int(p[1]) for p in positions})
    x_to_col = {x: idx for idx, x in enumerate(unique_x)}
    y_to_row = {y: idx for idx, y in enumerate(unique_y)}
    sx = float(scaled_w) / float(raw_w)
    sy = float(scaled_h) / float(raw_h)
    defs: list[TileDef] = []
    for tile_index, pos in enumerate(positions):
        x0_raw = int(pos[0])
        y0_raw = int(pos[1])
        x1_raw = x0_raw + tile_w
        y1_raw = y0_raw + tile_h
        x0 = max(0, min(scaled_w - 1, int(round(x0_raw * sx))))
        y0 = max(0, min(scaled_h - 1, int(round(y0_raw * sy))))
        x1 = max(x0 + 1, min(scaled_w, int(round(x1_raw * sx))))
        y1 = max(y0 + 1, min(scaled_h, int(round(y1_raw * sy))))
        defs.append(
            TileDef(
                tile_index=tile_index,
                row=int(y_to_row[y0_raw]),
                col=int(x_to_col[x0_raw]),
                raw_bbox_xyxy=(x0_raw, y0_raw, x1_raw, y1_raw),
                scaled_bbox_xyxy=(x0, y0, x1, y1),
                center_scaled_xy=((x0 + x1) / 2.0, (y0 + y1) / 2.0),
            )
        )
    return defs


def _manual_affine_for_scale(
    moving_shape_hw: tuple[int, int],
    fixed_shape_hw: tuple[int, int],
    *,
    scale: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    base_mat = build_manual_affine(
        moving_shape_hw,
        fixed_shape_hw,
        tx_px=float(MANUAL_STATE["tx_px"]),
        ty_px=float(MANUAL_STATE["ty_px"]),
        angle_deg=float(MANUAL_STATE["angle_deg"]),
        scale=float(scale),
        flip_lr=bool(MANUAL_STATE["flip_lr"]),
        flip_ud=bool(MANUAL_STATE["flip_ud"]),
    )
    mat, info = _anchor_guided_manual_affine(
        moving_shape_hw=moving_shape_hw,
        fixed_shape_hw=fixed_shape_hw,
        current_mat=base_mat,
        anchor_pairs=[ANCHOR_PAIR],
        flip_lr=bool(MANUAL_STATE["flip_lr"]),
        flip_ud=bool(MANUAL_STATE["flip_ud"]),
    )
    return mat.astype(np.float32), info


def _tile_affine_from_global(full_mat: np.ndarray, tile_bbox_xyxy: tuple[int, int, int, int]) -> np.ndarray:
    x0, y0, _x1, _y1 = tile_bbox_xyxy
    mat = np.asarray(full_mat, dtype=np.float32).reshape(2, 3).copy()
    linear = mat[:, :2]
    trans = mat[:, 2] + linear @ np.asarray([float(x0), float(y0)], dtype=np.float32)
    return np.concatenate([linear, trans[:, None]], axis=1).astype(np.float32)


def _warp_one_tile(
    moving_u8: np.ndarray,
    moving_mask_u8: np.ndarray,
    tile: TileDef,
    full_mat: np.ndarray,
    fixed_shape_hw: tuple[int, int],
    *,
    extra_dx: float = 0.0,
    extra_dy: float = 0.0,
) -> TileWarp:
    x0, y0, x1, y1 = tile.scaled_bbox_xyxy
    crop = moving_u8[y0:y1, x0:x1]
    crop_mask = moving_mask_u8[y0:y1, x0:x1]
    tile_mat = _tile_affine_from_global(full_mat, tile.scaled_bbox_xyxy)
    tile_mat[0, 2] += float(extra_dx)
    tile_mat[1, 2] += float(extra_dy)
    crop_h, crop_w = crop.shape[:2]
    corners = np.array(
        [[0.0, 0.0], [float(crop_w), 0.0], [0.0, float(crop_h)], [float(crop_w), float(crop_h)]],
        dtype=np.float32,
    )
    warped_corners = corners @ tile_mat[:, :2].T + tile_mat[:, 2]
    fx0 = max(0, int(math.floor(float(warped_corners[:, 0].min()))) - 8)
    fy0 = max(0, int(math.floor(float(warped_corners[:, 1].min()))) - 8)
    fx1 = min(int(fixed_shape_hw[1]), int(math.ceil(float(warped_corners[:, 0].max()))) + 8)
    fy1 = min(int(fixed_shape_hw[0]), int(math.ceil(float(warped_corners[:, 1].max()))) + 8)
    fx1 = max(fx1, fx0 + 1)
    fy1 = max(fy1, fy0 + 1)
    local_mat = tile_mat.copy()
    local_mat[0, 2] -= float(fx0)
    local_mat[1, 2] -= float(fy0)
    warped_u8 = cv2.warpAffine(
        crop,
        local_mat,
        (fx1 - fx0, fy1 - fy0),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    warped_mask_u8 = cv2.warpAffine(
        crop_mask,
        local_mat,
        (fx1 - fx0, fy1 - fy0),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    full_gray_patch = warped_u8.astype(np.float32) / 255.0
    full_mask_patch = (warped_mask_u8 > 0).astype(np.float32)
    full_bbox = (fy0, fy1, fx0, fx1)
    local_bbox = _bbox_from_mask(np.where(warped_mask_u8 > 0, 1, 0).astype(np.uint8), margin_px=18)
    ly0, ly1, lx0, lx1 = [int(v) for v in local_bbox]
    bbox = (fy0 + ly0, fy0 + ly1, fx0 + lx0, fx0 + lx1)
    return TileWarp(
        tile=tile,
        source_gray_patch=crop.astype(np.float32) / 255.0,
        source_mask_patch=(crop_mask > 0).astype(np.float32),
        tile_to_fixed_mat=tile_mat.astype(np.float32),
        warped_gray_full_patch=full_gray_patch,
        warped_mask_full_patch=full_mask_patch,
        warped_full_bbox_yxyx=full_bbox,
        warped_gray_patch=warped_u8[ly0:ly1, lx0:lx1].astype(np.float32) / 255.0,
        warped_mask_patch=(warped_mask_u8[ly0:ly1, lx0:lx1] > 0).astype(np.float32),
        warped_bbox_yxyx=bbox,
    )


def _best_local_translation(
    fixed_gray_full_u8: np.ndarray,
    tile_warp: TileWarp,
    *,
    search_radius: int = SEARCH_RADIUS,
) -> dict[str, Any]:
    mask = tile_warp.warped_mask_patch > 0
    if not np.any(mask):
        return {"dx_px": float("nan"), "dy_px": float("nan"), "best_cc": float("nan"), "search_bbox_yxyx": None}
    y0, y1, x0, x1 = [int(v) for v in tile_warp.warped_bbox_yxyx]
    template = tile_warp.warped_gray_patch.astype(np.float32)
    template_mask_u8 = np.where(tile_warp.warped_mask_patch > 0, 255, 0).astype(np.uint8)
    search_y0 = max(0, y0 - search_radius)
    search_y1 = min(fixed_gray_full_u8.shape[0], y1 + search_radius)
    search_x0 = max(0, x0 - search_radius)
    search_x1 = min(fixed_gray_full_u8.shape[1], x1 + search_radius)
    fixed_search = fixed_gray_full_u8[search_y0:search_y1, search_x0:search_x1].astype(np.float32) / 255.0
    if (
        fixed_search.shape[0] < template.shape[0]
        or fixed_search.shape[1] < template.shape[1]
        or template_mask_u8.max() <= 0
    ):
        return {"dx_px": float("nan"), "dy_px": float("nan"), "best_cc": float("nan"), "search_bbox_yxyx": [search_y0, search_y1, search_x0, search_x1]}
    res = cv2.matchTemplate(fixed_search, template, cv2.TM_CCORR_NORMED, mask=template_mask_u8)
    _min_val, max_val, _min_loc, max_loc = cv2.minMaxLoc(res)
    best_x0 = int(search_x0 + max_loc[0])
    best_y0 = int(search_y0 + max_loc[1])
    return {
        "dx_px": float(best_x0 - x0),
        "dy_px": float(best_y0 - y0),
        "best_cc": float(max_val),
        "template_bbox_yxyx": [int(y0), int(y1), int(x0), int(x1)],
        "search_bbox_yxyx": [int(search_y0), int(search_y1), int(search_x0), int(search_x1)],
        "best_bbox_yxyx": [int(best_y0), int(best_y0 + template.shape[0]), int(best_x0), int(best_x0 + template.shape[1])],
    }


def _analyze_tile_warp(
    fixed_gray_full_u8: np.ndarray,
    fixed_mask_full: np.ndarray,
    tile_warp: TileWarp,
) -> dict[str, Any]:
    y0, y1, x0, x1 = tile_warp.warped_bbox_yxyx
    fixed_patch = fixed_gray_full_u8[y0:y1, x0:x1].astype(np.float32) / 255.0
    moving_patch = tile_warp.warped_gray_patch
    fixed_mask_patch = fixed_mask_full[y0:y1, x0:x1]
    moving_mask_patch = tile_warp.warped_mask_patch
    metrics, timings = compute_registration_metrics(fixed_patch, moving_patch, fixed_mask_patch, moving_mask_patch)
    best_shift = _best_local_translation(fixed_gray_full_u8, tile_warp)
    out = {
        "tile_index": int(tile_warp.tile.tile_index),
        "row": int(tile_warp.tile.row),
        "col": int(tile_warp.tile.col),
        "center_scaled_x": float(tile_warp.tile.center_scaled_xy[0]),
        "center_scaled_y": float(tile_warp.tile.center_scaled_xy[1]),
        "warped_bbox_yxyx": [int(v) for v in tile_warp.warped_bbox_yxyx],
        "current_cc": float(metrics.get("cc", float("nan"))),
        "current_mi": float(metrics.get("mi", float("nan"))),
        "current_dice": float(metrics.get("dice", float("nan"))),
        "current_hd95": float(metrics.get("hd95_px", float("nan"))),
        "metric_timing_s": float(timings.get("total", 0.0)),
        "dx_star_px": float(best_shift.get("dx_px", float("nan"))),
        "dy_star_px": float(best_shift.get("dy_px", float("nan"))),
        "best_translation_cc": float(best_shift.get("best_cc", float("nan"))),
        "template_bbox_yxyx": best_shift.get("template_bbox_yxyx"),
        "search_bbox_yxyx": best_shift.get("search_bbox_yxyx"),
        "best_bbox_yxyx": best_shift.get("best_bbox_yxyx"),
    }
    return out


def _compose_reduced_full_from_tiles(
    tile_warps: list[TileWarp],
    fixed_shape_hw: tuple[int, int],
    *,
    render_scale: float = FULL_RENDER_SCALE,
) -> tuple[np.ndarray, np.ndarray]:
    out_h = max(1, int(round(float(fixed_shape_hw[0]) * render_scale)))
    out_w = max(1, int(round(float(fixed_shape_hw[1]) * render_scale)))
    accum = np.zeros((out_h, out_w), dtype=np.float32)
    counts = np.zeros((out_h, out_w), dtype=np.float32)
    accum_mask = np.zeros((out_h, out_w), dtype=np.float32)
    for tw in tile_warps:
        y0, y1, x0, x1 = [int(v) for v in tw.warped_bbox_yxyx]
        sy0 = max(0, min(out_h - 1, int(round(y0 * render_scale))))
        sy1 = max(sy0 + 1, min(out_h, int(round(y1 * render_scale))))
        sx0 = max(0, min(out_w - 1, int(round(x0 * render_scale))))
        sx1 = max(sx0 + 1, min(out_w, int(round(x1 * render_scale))))
        patch = cv2.resize(tw.warped_gray_patch, (sx1 - sx0, sy1 - sy0), interpolation=cv2.INTER_LINEAR)
        mask_u8 = cv2.resize(
            np.where(tw.warped_mask_patch > 0, 255, 0).astype(np.uint8),
            (sx1 - sx0, sy1 - sy0),
            interpolation=cv2.INTER_NEAREST,
        )
        mask = mask_u8 > 0
        target_accum = accum[sy0:sy1, sx0:sx1]
        target_counts = counts[sy0:sy1, sx0:sx1]
        target_mask = accum_mask[sy0:sy1, sx0:sx1]
        target_accum[mask] += patch[mask]
        target_counts[mask] += 1.0
        target_mask[mask] += 1.0
    out_gray = np.where(counts > 0, accum / np.maximum(counts, 1.0), 1.0).astype(np.float32)
    out_mask = (accum_mask > 0).astype(np.float32)
    return out_gray, out_mask


def _collect_tile_results(
    *,
    label: str,
    moving_reg_projection_u8: np.ndarray,
    moving_signal_mask_u8: np.ndarray,
    fixed_gray_full: np.ndarray,
    fixed_mask_full: np.ndarray,
    tile_defs: list[TileDef],
    full_mat: np.ndarray,
    per_tile_shift: dict[tuple[int, int], tuple[float, float]] | None = None,
) -> tuple[list[dict[str, Any]], list[TileWarp]]:
    tile_warps: list[TileWarp] = []
    rows: list[dict[str, Any]] = []
    for tile in tile_defs:
        dx, dy = (0.0, 0.0)
        if per_tile_shift is not None:
            dx, dy = per_tile_shift.get((tile.row, tile.col), (0.0, 0.0))
        tw = _warp_one_tile(
            moving_reg_projection_u8,
            moving_signal_mask_u8,
            tile,
            full_mat,
            fixed_gray_full.shape[:2],
            extra_dx=float(dx),
            extra_dy=float(dy),
        )
        tile_warps.append(tw)
        rows.append(_analyze_tile_warp(fixed_gray_full, fixed_mask_full, tw))
    for row in rows:
        row["method"] = label
    return rows, tile_warps


def _panel_crop(gray: np.ndarray, mask: np.ndarray, bbox: tuple[int, int, int, int], *, fill: float | None = None) -> np.ndarray:
    y0, y1, x0, x1 = [int(v) for v in bbox]
    crop = gray[y0:y1, x0:x1]
    if crop.size == 0:
        return np.zeros((128, 128, 3), dtype=np.uint8)
    if crop.dtype == np.uint8:
        crop = crop.astype(np.float32) / 255.0
    if fill is not None:
        crop = np.where(mask[y0:y1, x0:x1] > 0, crop, float(fill)).astype(np.float32)
    return gray_preview_panel(crop)


def _square_panel(panel: np.ndarray, side: int = 320) -> np.ndarray:
    h, w = panel.shape[:2]
    if h <= 0 or w <= 0:
        return np.full((side, side, 3), 240, dtype=np.uint8)
    scale = min(float(side) / float(h), float(side) / float(w))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(panel, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.full((side, side, 3), 245, dtype=np.uint8)
    y0 = (side - new_h) // 2
    x0 = (side - new_w) // 2
    canvas[y0 : y0 + new_h, x0 : x0 + new_w] = resized
    return canvas


def _make_tile_storyboard(
    fixed_gray_full_u8: np.ndarray,
    fixed_mask_full: np.ndarray,
    tile_warps: list[TileWarp],
    tile_rows: list[dict[str, Any]],
    out_path: Path,
    *,
    title_prefix: str,
    moving_display_tile_warps: list[TileWarp] | None = None,
) -> None:
    rows = []
    row_map = {(int(r["row"]), int(r["col"])): r for r in tile_rows}
    display_map = None
    if moving_display_tile_warps is not None:
        display_map = {(int(tw.tile.row), int(tw.tile.col)): tw for tw in moving_display_tile_warps}

    def _warp_fixed_into_tile_frame(
        fixed_gray_full_u8: np.ndarray,
        fixed_mask_full: np.ndarray,
        tile_to_fixed_mat: np.ndarray,
        out_shape_hw: tuple[int, int],
        *,
        extra_dx: float = 0.0,
        extra_dy: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        mat = np.asarray(tile_to_fixed_mat, dtype=np.float32).copy()
        mat[0, 2] += float(extra_dx)
        mat[1, 2] += float(extra_dy)
        inv = cv2.invertAffineTransform(mat)
        out_h, out_w = int(out_shape_hw[0]), int(out_shape_hw[1])
        fixed_local = cv2.warpAffine(
            fixed_gray_full_u8,
            inv,
            (out_w, out_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=255,
        ).astype(np.float32) / 255.0
        fixed_local_mask = cv2.warpAffine(
            np.where(fixed_mask_full > 0, 255, 0).astype(np.uint8),
            inv,
            (out_w, out_h),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        return fixed_local, (fixed_local_mask > 0).astype(np.float32)

    for tw in tile_warps:
        rec = row_map[(tw.tile.row, tw.tile.col)]
        display_tw = display_map.get((tw.tile.row, tw.tile.col)) if display_map is not None else tw
        local_shape_hw = display_tw.source_gray_patch.shape[:2] if display_tw is not None else tw.source_gray_patch.shape[:2]
        fixed_local, fixed_local_mask = _warp_fixed_into_tile_frame(
            fixed_gray_full_u8,
            fixed_mask_full,
            tw.tile_to_fixed_mat,
            local_shape_hw,
        )
        moving_full_mask = np.ones_like(tw.source_gray_patch, dtype=np.float32)
        current_overlay = overlay_preview(fixed_local, tw.source_gray_patch, fixed_local_mask, moving_full_mask)
        dx_star = float(rec.get("dx_star_px", 0.0))
        dy_star = float(rec.get("dy_star_px", 0.0))
        fixed_local_shifted, fixed_local_shifted_mask = _warp_fixed_into_tile_frame(
            fixed_gray_full_u8,
            fixed_mask_full,
            tw.tile_to_fixed_mat,
            local_shape_hw,
            extra_dx=dx_star,
            extra_dy=dy_star,
        )
        best_overlay = overlay_preview(fixed_local_shifted, tw.source_gray_patch, fixed_local_shifted_mask, moving_full_mask)
        display_source = display_tw.source_gray_patch if display_tw is not None else tw.source_gray_patch
        moving_search_panel = gray_preview_panel(display_source)
        rows.append(
            {
                "label": f"T{tw.tile.tile_index:02d} r{tw.tile.row}c{tw.tile.col}",
                "note": (
                    f"{title_prefix} | CC={rec['current_cc']:.3f} MI={rec['current_mi']:.3f} "
                    f"dx*={rec['dx_star_px']:.1f} dy*={rec['dy_star_px']:.1f}"
                ),
                "moving": _square_panel(moving_search_panel),
                "fixed": _square_panel(gray_preview_panel(fixed_local)),
                "overlay": _square_panel(current_overlay),
                "heatmap": _square_panel(best_overlay),
                "col_titles": ("Confocal native tile FOV", "Myelin tile FOV", "Current overlay", "Best local shift"),
            }
        )
    render_storyboard(rows, out_path)


def _draw_full_overlay(
    fixed_gray_full_u8: np.ndarray,
    fixed_mask_full: np.ndarray,
    tile_rows: list[dict[str, Any]],
    tile_warps: list[TileWarp],
    *,
    title: str,
) -> np.ndarray:
    reduced_moving_gray, reduced_moving_mask = _compose_reduced_full_from_tiles(
        tile_warps,
        fixed_gray_full_u8.shape[:2],
        render_scale=FULL_RENDER_SCALE,
    )
    out_h, out_w = reduced_moving_gray.shape[:2]
    fixed_small_gray = cv2.resize(
        fixed_gray_full_u8,
        (out_w, out_h),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.float32) / 255.0
    fixed_small_mask = (
        cv2.resize(
            np.where(fixed_mask_full > 0, 255, 0).astype(np.uint8),
            (out_w, out_h),
            interpolation=cv2.INTER_NEAREST,
        )
        > 0
    ).astype(np.float32)
    overlay = overlay_preview(fixed_small_gray, reduced_moving_gray, fixed_small_mask, reduced_moving_mask)
    canvas = np.full((overlay.shape[0] + 80, overlay.shape[1], 3), 245, dtype=np.uint8)
    canvas[80:, :, :] = overlay
    cv2.putText(canvas, title, (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (25, 25, 25), 2, cv2.LINE_AA)
    cv2.putText(canvas, "tile boxes show row/col and dx*", (18, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (60, 60, 60), 2, cv2.LINE_AA)
    for rec in tile_rows:
        y0, y1, x0, x1 = [int(v) for v in rec["warped_bbox_yxyx"]]
        sx0 = int(round(x0 * FULL_RENDER_SCALE))
        sx1 = int(round(x1 * FULL_RENDER_SCALE))
        sy0 = int(round(y0 * FULL_RENDER_SCALE))
        sy1 = int(round(y1 * FULL_RENDER_SCALE))
        cv2.rectangle(canvas, (sx0, sy0 + 80), (sx1 - 1, sy1 - 1 + 80), (255, 210, 0), 1)
        cv2.putText(
            canvas,
            f"r{int(rec['row'])}c{int(rec['col'])} dx*={float(rec['dx_star_px']):.1f}",
            (sx0 + 4, sy0 + 94),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (255, 90, 90),
            1,
            cv2.LINE_AA,
        )
    return canvas


def _save_plot_dx_vs_x(rows: list[dict[str, Any]], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 4.8), dpi=160)
    for row_idx in sorted({int(r["row"]) for r in rows}):
        subset = sorted([r for r in rows if int(r["row"]) == row_idx], key=lambda r: int(r["col"]))
        ax.plot([float(r["center_scaled_x"]) for r in subset], [float(r["dx_star_px"]) for r in subset], marker="o", label=f"row {row_idx}")
    ax.axhline(0.0, color="black", linewidth=1, linestyle="--")
    ax.set_xlabel("tile center x in scaled confocal")
    ax.set_ylabel("best local translation dx* (px)")
    ax.set_title("dx* vs tile center x")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _save_plot_dx_vs_col(scale_rows: list[dict[str, Any]], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 4.8), dpi=160)
    for row_idx in sorted({int(r["row"]) for r in scale_rows}):
        subset = sorted([r for r in scale_rows if int(r["row"]) == row_idx], key=lambda r: int(r["col"]))
        ax.plot([int(r["col"]) for r in subset], [float(r["dx_star_px"]) for r in subset], marker="o", label=f"row {row_idx}")
    ax.axhline(0.0, color="black", linewidth=1, linestyle="--")
    ax.set_xlabel("column")
    ax.set_ylabel("best local translation dx* (px)")
    ax.set_title("dx* vs column")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _aggregate_scale_sweep(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    scales = sorted({float(r["scale"]) for r in rows})
    for scale in scales:
        subset = [r for r in rows if float(r["scale"]) == scale]
        by_col = {col: [r for r in subset if int(r["col"]) == col] for col in range(6)}
        dx = np.array([float(r["dx_star_px"]) for r in subset], dtype=np.float64)
        cc = np.array([float(r["current_cc"]) for r in subset], dtype=np.float64)
        row = {
            "scale": float(scale),
            "mean_tile_cc": float(np.nanmean(cc)),
            "mean_tile_mi": float(np.nanmean([float(r["current_mi"]) for r in subset])),
            "mean_abs_dx": float(np.nanmean(np.abs(dx))),
            "dx_slope_per_px": float(np.polyfit(
                np.asarray([float(r["center_scaled_x"]) for r in subset], dtype=np.float64),
                dx,
                1,
            )[0]),
            "rightmost_abs_dx": float(np.nanmean(np.abs([float(r["dx_star_px"]) for r in by_col.get(5, [])]))),
        }
        for col in range(6):
            col_subset = by_col.get(col, [])
            row[f"col{col}_mean_cc"] = float(np.nanmean([float(r["current_cc"]) for r in col_subset])) if col_subset else float("nan")
            row[f"col{col}_mean_dx"] = float(np.nanmean([float(r["dx_star_px"]) for r in col_subset])) if col_subset else float("nan")
            row[f"col{col}_mean_dy"] = float(np.nanmean([float(r["dy_star_px"]) for r in col_subset])) if col_subset else float("nan")
        summary.append(row)
    return summary


def _best_scale_from_summary(summary: list[dict[str, Any]]) -> tuple[float, float]:
    best_overall = max(summary, key=lambda r: float(r["mean_tile_cc"]))["scale"]
    best_right = min(summary, key=lambda r: float(r["rightmost_abs_dx"]))["scale"]
    return float(best_overall), float(best_right)


def _compose_affine(post_mat: np.ndarray, base_mat: np.ndarray) -> np.ndarray:
    post3 = np.eye(3, dtype=np.float64)
    post3[:2, :] = np.asarray(post_mat, dtype=np.float64)
    base3 = np.eye(3, dtype=np.float64)
    base3[:2, :] = np.asarray(base_mat, dtype=np.float64)
    return (post3 @ base3)[:2, :].astype(np.float32)


def _fit_anchor_locked_similarity(tile_rows: list[dict[str, Any]], anchor_xy: tuple[float, float]) -> np.ndarray:
    src = np.array([[float(r["center_scaled_x"]), float(r["center_scaled_y"])] for r in tile_rows], dtype=np.float64)
    residual = np.array([[float(r["dx_star_px"]), float(r["dy_star_px"])] for r in tile_rows], dtype=np.float64)
    dst = src + residual
    anchor = np.asarray(anchor_xy, dtype=np.float64).reshape(1, 2)
    src_c = src - anchor
    dst_c = dst - anchor
    cov = src_c.T @ dst_c
    U, S, Vt = np.linalg.svd(cov)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1.0
        R = Vt.T @ U.T
    denom = float(np.sum(src_c * src_c))
    scale = float(np.sum(S) / max(denom, 1e-8))
    linear = scale * R
    anchor_v = anchor.reshape(2)
    trans = anchor_v - linear @ anchor_v
    return np.concatenate([linear, trans[:, None]], axis=1).astype(np.float32)


def _fit_anchor_locked_affine(tile_rows: list[dict[str, Any]], anchor_xy: tuple[float, float]) -> np.ndarray:
    src = np.array([[float(r["center_scaled_x"]), float(r["center_scaled_y"])] for r in tile_rows], dtype=np.float64)
    residual = np.array([[float(r["dx_star_px"]), float(r["dy_star_px"])] for r in tile_rows], dtype=np.float64)
    dst = src + residual
    anchor = np.asarray(anchor_xy, dtype=np.float64)
    src_c = src - anchor
    dst_c = dst - anchor
    M_t, _resid, _rank, _s = np.linalg.lstsq(src_c, dst_c, rcond=None)
    linear = M_t.T
    trans = anchor - linear @ anchor
    return np.concatenate([linear, trans[:, None]], axis=1).astype(np.float32)


def _run_single_scale_diagnostic(
    *,
    scale: float,
    moving_reg_projection_u8: np.ndarray,
    moving_signal_mask_u8: np.ndarray,
    fixed_gray_full: np.ndarray,
    fixed_mask_full: np.ndarray,
    fixed_shape_hw: tuple[int, int],
    tile_defs: list[TileDef],
) -> tuple[np.ndarray, dict[str, Any], list[dict[str, Any]], list[TileWarp]]:
    full_mat, anchor_info = _manual_affine_for_scale(
        moving_reg_projection_u8.shape[:2],
        fixed_shape_hw,
        scale=scale,
    )
    rows, tile_warps = _collect_tile_results(
        label=f"coarse_scale_{scale:.3f}",
        moving_reg_projection_u8=moving_reg_projection_u8,
        moving_signal_mask_u8=moving_signal_mask_u8,
        fixed_gray_full=fixed_gray_full,
        fixed_mask_full=fixed_mask_full,
        tile_defs=tile_defs,
        full_mat=full_mat,
    )
    for rec in rows:
        rec["scale"] = float(scale)
    return full_mat, anchor_info, rows, tile_warps


def _column_smooth_offsets(tile_rows: list[dict[str, Any]]) -> dict[tuple[int, int], tuple[float, float]]:
    dx_mean = np.array([np.mean([float(r["dx_star_px"]) for r in tile_rows if int(r["col"]) == c]) for c in range(6)], dtype=np.float64)
    dy_mean = np.array([np.mean([float(r["dy_star_px"]) for r in tile_rows if int(r["col"]) == c]) for c in range(6)], dtype=np.float64)
    n = 6
    lam = 2.0
    D = np.zeros((n - 2, n), dtype=np.float64)
    for i in range(n - 2):
        D[i, i : i + 3] = np.array([1.0, -2.0, 1.0], dtype=np.float64)
    A = np.eye(n) + lam * (D.T @ D)
    for mean in (dx_mean, dy_mean):
        mean[0] = 0.0
    A[0, :] = 0.0
    A[0, 0] = 1.0
    dx_rhs = dx_mean.copy()
    dy_rhs = dy_mean.copy()
    dx_rhs[0] = 0.0
    dy_rhs[0] = 0.0
    dx_fit = np.linalg.solve(A, dx_rhs)
    dy_fit = np.linalg.solve(A, dy_rhs)
    out: dict[tuple[int, int], tuple[float, float]] = {}
    for row in range(3):
        for col in range(6):
            out[(row, col)] = (float(dx_fit[col]), float(dy_fit[col]))
    return out


def _tilewise_smooth_offsets(tile_rows: list[dict[str, Any]]) -> dict[tuple[int, int], tuple[float, float]]:
    dx = np.zeros((3, 6), dtype=np.float64)
    dy = np.zeros((3, 6), dtype=np.float64)
    for r in tile_rows:
        dx[int(r["row"]), int(r["col"])] = float(r["dx_star_px"])
        dy[int(r["row"]), int(r["col"])] = float(r["dy_star_px"])
    cur_dx = dx.copy()
    cur_dy = dy.copy()
    lam = 1.6
    for _ in range(30):
        nxt_dx = cur_dx.copy()
        nxt_dy = cur_dy.copy()
        for rr in range(3):
            for cc in range(6):
                if rr == 0 and cc == 0:
                    nxt_dx[rr, cc] = 0.0
                    nxt_dy[rr, cc] = 0.0
                    continue
                neigh: list[tuple[int, int]] = []
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    r2 = rr + dr
                    c2 = cc + dc
                    if 0 <= r2 < 3 and 0 <= c2 < 6:
                        neigh.append((r2, c2))
                avg_dx = float(np.mean([cur_dx[r2, c2] for r2, c2 in neigh])) if neigh else 0.0
                avg_dy = float(np.mean([cur_dy[r2, c2] for r2, c2 in neigh])) if neigh else 0.0
                nxt_dx[rr, cc] = (dx[rr, cc] + lam * avg_dx) / (1.0 + lam)
                nxt_dy[rr, cc] = (dy[rr, cc] + lam * avg_dy) / (1.0 + lam)
        nxt_dx -= nxt_dx[0, 0]
        nxt_dy -= nxt_dy[0, 0]
        cur_dx, cur_dy = nxt_dx, nxt_dy
    out: dict[tuple[int, int], tuple[float, float]] = {}
    for rr in range(3):
        for cc in range(6):
            out[(rr, cc)] = (float(cur_dx[rr, cc]), float(cur_dy[rr, cc]))
    return out


def _run_step7_refine(
    *,
    out_root: Path,
    item: WorkspaceSection,
    fixed_bundle: Any,
    confocal_paths: list[Path],
    moving_scaled_u8: np.ndarray,
    moving_scaled_mask_u8: np.ndarray,
    projection_bundle: Any,
    scale: float,
    refine_model: str,
) -> dict[str, Any]:
    ants_bin = find_ants_bin()
    if ants_bin is None:
        raise RuntimeError("Could not locate local ANTs binaries")
    def _ants_cli_posix(path: Path | str) -> str:
        return str(Path(path))

    def _ants_binary_posix(bin_dir: Path, stem: str) -> Path:
        for name in (stem, f"{stem}.exe"):
            candidate = Path(bin_dir) / name
            try:
                if candidate.exists():
                    return candidate
            except OSError:
                continue
        return Path(bin_dir) / stem

    cfg = ConfocalRigidConfig(
        myelin_label=item.label,
        myelin_section_dir=item.section_dir,
        myelin_stain=item.stain,
        myelin_rgb=fixed_bundle.rgb,
        myelin_labels=fixed_bundle.labels,
        myelin_fixed_info={
            "preview_um_per_px_xy": list(fixed_bundle.preview_um_per_px_xy or (STEP7_TARGET_UM_PER_PX, STEP7_TARGET_UM_PER_PX)),
            "source_um_per_px_xy": list(fixed_bundle.source_um_per_px_xy or fixed_bundle.preview_um_per_px_xy or (STEP7_TARGET_UM_PER_PX, STEP7_TARGET_UM_PER_PX)),
            "support_shape_hw": list(fixed_bundle.support_shape_hw),
            "preview_shape_hw": list(fixed_bundle.preview_shape_hw),
            "support_bbox_canvas_xywh": list(fixed_bundle.support_bbox_canvas_xywh) if fixed_bundle.support_bbox_canvas_xywh is not None else None,
            "fixed_working_mode": fixed_bundle.fixed_working_mode,
            "target_um_per_px_xy": list(fixed_bundle.target_um_per_px_xy) if fixed_bundle.target_um_per_px_xy is not None else None,
        },
        confocal_projection_u8=moving_scaled_u8,
        confocal_signal_mask_u8=moving_scaled_mask_u8,
        ants_bin=ants_bin,
        out_root=out_root,
        confocal_sources=confocal_paths,
        confocal_source_mode=projection_bundle.source_mode,
        nominal_overlap_fraction=0.10,
        projection_info={
            "source_um_per_px_xy": list(projection_bundle.physical_um_per_px_xy or ()),
            "target_um_per_px_xy": list(fixed_bundle.preview_um_per_px_xy or ()),
            "stitch_info": projection_bundle.stitch_info,
            "raw_projection_shape_hw": list(projection_bundle.projection_u8.shape[:2]),
            "scaled_projection_shape_hw": list(moving_scaled_u8.shape[:2]),
        },
        projection_mode="focus",
        channel_index=0,
        local_refine_model=refine_model,
        target_working_um_per_px=STEP7_TARGET_UM_PER_PX,
        invert_confocal_for_registration=True,
        tx_px=float(MANUAL_STATE["tx_px"]),
        ty_px=float(MANUAL_STATE["ty_px"]),
        angle_deg=float(MANUAL_STATE["angle_deg"]),
        scale=float(scale),
        flip_lr=bool(MANUAL_STATE["flip_lr"]),
        flip_ud=bool(MANUAL_STATE["flip_ud"]),
        anchor_pairs=[ANCHOR_PAIR],
    )
    old_pair_cli = pair_registration_mod.ants_cli_path
    old_pair_bin = pair_registration_mod.ants_binary_path
    old_conf_cli = confocal_registration_mod.ants_cli_path
    old_conf_bin = confocal_registration_mod.ants_binary_path
    pair_registration_mod.ants_cli_path = _ants_cli_posix
    pair_registration_mod.ants_binary_path = _ants_binary_posix
    confocal_registration_mod.ants_cli_path = _ants_cli_posix
    confocal_registration_mod.ants_binary_path = _ants_binary_posix
    try:
        return run_confocal_rigid_registration(cfg)
    finally:
        pair_registration_mod.ants_cli_path = old_pair_cli
        pair_registration_mod.ants_binary_path = old_pair_bin
        confocal_registration_mod.ants_cli_path = old_conf_cli
        confocal_registration_mod.ants_binary_path = old_conf_bin


def _ants_affine_mat_to_cv2(path: Path) -> np.ndarray:
    params, fixed = _load_ants_affine_mat(path)
    linear = np.array([[params[0], params[1]], [params[2], params[3]]], dtype=np.float32)
    c = np.asarray(fixed, dtype=np.float32).reshape(2)
    trans = np.asarray([params[4], params[5]], dtype=np.float32)
    offset = c + trans - linear @ c
    return np.concatenate([linear, offset[:, None]], axis=1).astype(np.float32)


def _refine_tile_warps_from_run(
    *,
    coarse_tile_warps: list[TileWarp],
    run_summary: dict[str, Any],
    fixed_shape_hw: tuple[int, int],
) -> list[TileWarp]:
    local_info = run_summary.get("local_registration") if isinstance(run_summary.get("local_registration"), dict) else {}
    roi_bbox = tuple(int(v) for v in (local_info.get("roi_bbox_yxyx") or [0, fixed_shape_hw[0], 0, fixed_shape_hw[1]]))
    files = run_summary.get("files") if isinstance(run_summary.get("files"), dict) else {}
    mat_path = Path(str(files.get("local_refine_transform") or files.get("rigid_transform")))
    cv2_mat = _ants_affine_mat_to_cv2(mat_path)
    y0, y1, x0, x1 = roi_bbox
    roi_h = y1 - y0
    roi_w = x1 - x0
    out: list[TileWarp] = []
    for coarse in coarse_tile_warps:
        gy0, gy1, gx0, gx1 = [int(v) for v in coarse.warped_bbox_yxyx]
        local_x0 = float(gx0 - x0)
        local_y0 = float(gy0 - y0)
        patch_mat = cv2_mat.copy()
        patch_mat[:, 2] += patch_mat[:, :2] @ np.asarray([local_x0, local_y0], dtype=np.float32)
        patch_h, patch_w = coarse.warped_gray_patch.shape[:2]
        corners = np.array(
            [[0.0, 0.0], [float(patch_w), 0.0], [0.0, float(patch_h)], [float(patch_w), float(patch_h)]],
            dtype=np.float32,
        )
        warped_corners = corners @ patch_mat[:, :2].T + patch_mat[:, 2]
        lx0 = max(0, int(math.floor(float(warped_corners[:, 0].min()))) - 6)
        ly0 = max(0, int(math.floor(float(warped_corners[:, 1].min()))) - 6)
        lx1 = min(roi_w, int(math.ceil(float(warped_corners[:, 0].max()))) + 6)
        ly1 = min(roi_h, int(math.ceil(float(warped_corners[:, 1].max()))) + 6)
        lx1 = max(lx1, lx0 + 1)
        ly1 = max(ly1, ly0 + 1)
        local_patch_mat = patch_mat.copy()
        local_patch_mat[0, 2] -= float(lx0)
        local_patch_mat[1, 2] -= float(ly0)
        refined_crop = cv2.warpAffine(
            coarse.warped_gray_patch.astype(np.float32),
            local_patch_mat,
            (lx1 - lx0, ly1 - ly0),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        refined_mask = cv2.warpAffine(
            np.where(coarse.warped_mask_patch > 0, 255, 0).astype(np.uint8),
            local_patch_mat,
            (lx1 - lx0, ly1 - ly0),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        local_bbox = _bbox_from_mask(np.where(refined_mask > 0, 1, 0).astype(np.uint8), margin_px=18)
        lby0, lby1, lbx0, lbx1 = [int(v) for v in local_bbox]
        bbox = (y0 + ly0 + lby0, y0 + ly0 + lby1, x0 + lx0 + lbx0, x0 + lx0 + lbx1)
        out.append(
            TileWarp(
                tile=coarse.tile,
                source_gray_patch=refined_crop[lby0:lby1, lbx0:lbx1].astype(np.float32),
                source_mask_patch=(refined_mask[lby0:lby1, lbx0:lbx1] > 0).astype(np.float32),
                tile_to_fixed_mat=np.array(
                    [
                        [1.0, 0.0, float(x0 + lx0 + lbx0)],
                        [0.0, 1.0, float(y0 + ly0 + lby0)],
                    ],
                    dtype=np.float32,
                ),
                warped_gray_full_patch=refined_crop.astype(np.float32),
                warped_mask_full_patch=(refined_mask > 0).astype(np.float32),
                warped_full_bbox_yxyx=(y0 + ly0, y0 + ly1, x0 + lx0, x0 + lx1),
                warped_gray_patch=refined_crop[lby0:lby1, lbx0:lbx1].astype(np.float32),
                warped_mask_patch=(refined_mask[lby0:lby1, lbx0:lbx1] > 0).astype(np.float32),
                warped_bbox_yxyx=bbox,
            )
        )
    return out


def _build_method_comparison_contact(
    panels: list[tuple[str, np.ndarray]],
    out_path: Path,
) -> None:
    if not panels:
        return
    max_w = max(p.shape[1] for _label, p in panels)
    total_h = sum(p.shape[0] + 24 for _label, p in panels) + 12
    canvas = np.full((total_h, max_w, 3), 248, dtype=np.uint8)
    y = 8
    for label, panel in panels:
        cv2.putText(canvas, label, (10, y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 30, 30), 2, cv2.LINE_AA)
        y += 24
        canvas[y : y + panel.shape[0], : panel.shape[1], :] = panel
        y += panel.shape[0]
    cv2.imwrite(str(out_path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))


def main() -> None:
    out_root = _ensure_dir(OUT_ROOT)
    process_dir = _ensure_dir(out_root / "process")
    figures_dir = _ensure_dir(out_root / "figures")
    runs_dir = _ensure_dir(out_root / "runs")

    inputs = _load_inputs()
    fixed_bundle = inputs["fixed_bundle"]
    fixed_mask_full = (fixed_bundle.labels == 1).astype(np.float32)
    if not np.any(fixed_mask_full > 0):
        fixed_mask_full = (fixed_bundle.labels > 0).astype(np.float32)
    fixed_gray_native_u8 = cv2.cvtColor(fixed_bundle.rgb, cv2.COLOR_RGB2GRAY)
    moving_scaled_mask_u8 = inputs["scaled_signal_mask_u8"]
    fixed_gray_full_u8 = _masked_percentile_normalize_u8(fixed_gray_native_u8, fixed_mask_full)
    moving_inverted_u8 = _invert_confocal_u8(inputs["scaled_projection_u8"])
    moving_scaled_u8 = _masked_percentile_normalize_u8(moving_inverted_u8, moving_scaled_mask_u8)
    moving_scaled_u8 = _masked_histogram_match_u8(
        moving_scaled_u8,
        moving_scaled_mask_u8,
        fixed_gray_full_u8,
        fixed_mask_full,
    )
    tile_defs = _build_tile_defs(
        inputs["projection_bundle"].stitch_info,
        raw_shape_hw=inputs["raw_projection_u8"].shape[:2],
        scaled_shape_hw=inputs["scaled_projection_u8"].shape[:2],
    )

    # Experiment 0
    coarse_mat, anchor_info, exp0_rows, exp0_tile_warps = _run_single_scale_diagnostic(
        scale=float(MANUAL_STATE["scale"]),
        moving_reg_projection_u8=moving_scaled_u8,
        moving_signal_mask_u8=moving_scaled_mask_u8,
        fixed_gray_full=fixed_gray_full_u8,
        fixed_mask_full=fixed_mask_full,
        fixed_shape_hw=fixed_bundle.rgb.shape[:2],
        tile_defs=tile_defs,
    )
    _write_csv(process_dir / "tile_metrics_exp0.csv", exp0_rows)
    _save_plot_dx_vs_x(exp0_rows, figures_dir / "dx_vs_x_plot.png")
    _save_plot_dx_vs_col(exp0_rows, figures_dir / "per_column_residual_plot.png")
    _make_tile_storyboard(
        fixed_gray_full_u8,
        fixed_mask_full,
        exp0_tile_warps,
        exp0_rows,
        figures_dir / "tile_zoom_storyboard.png",
        title_prefix="Experiment 0 coarse",
    )
    exp0_overlay = _draw_full_overlay(
        fixed_gray_full_u8,
        fixed_mask_full,
        exp0_rows,
        exp0_tile_warps,
        title="Experiment 0 coarse overlay",
    )
    cv2.imwrite(str(figures_dir / "full_overlay_contact_sheet.png"), cv2.cvtColor(exp0_overlay, cv2.COLOR_RGB2BGR))

    # Experiment 1
    sweep_rows: list[dict[str, Any]] = []
    for scale in SCALE_VALUES:
        _full_mat, _anchor_info, rows, _tile_warps = _run_single_scale_diagnostic(
            scale=scale,
            moving_reg_projection_u8=moving_scaled_u8,
            moving_signal_mask_u8=moving_scaled_mask_u8,
            fixed_gray_full=fixed_gray_full_u8,
            fixed_mask_full=fixed_mask_full,
            fixed_shape_hw=fixed_bundle.rgb.shape[:2],
            tile_defs=tile_defs,
        )
        sweep_rows.extend(rows)
    _write_csv(process_dir / "tile_metrics.csv", sweep_rows)
    sweep_summary = _aggregate_scale_sweep(sweep_rows)
    _write_csv(process_dir / "scale_sweep_summary.csv", sweep_summary)
    best_overall_scale, best_right_scale = _best_scale_from_summary(sweep_summary)

    fig, ax = plt.subplots(figsize=(8.4, 4.8), dpi=160)
    ax.plot([r["scale"] for r in sweep_summary], [r["mean_tile_cc"] for r in sweep_summary], marker="o", label="mean tile CC")
    ax.set_xlabel("scale")
    ax.set_ylabel("mean tile CC")
    ax.set_title("Scale sweep: mean local CC")
    ax.grid(alpha=0.25)
    ax.axvline(best_overall_scale, color="green", linestyle="--", label=f"best overall {best_overall_scale:.3f}")
    ax.axvline(best_right_scale, color="orange", linestyle="--", label=f"best right {best_right_scale:.3f}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "scale_sweep_cc_plot.png")
    plt.close(fig)

    compare_scale = float(best_overall_scale)
    best_full_mat, _best_anchor_info, best_scale_rows, best_scale_tile_warps = _run_single_scale_diagnostic(
        scale=compare_scale,
        moving_reg_projection_u8=moving_scaled_u8,
        moving_signal_mask_u8=moving_scaled_mask_u8,
        fixed_gray_full=fixed_gray_full_u8,
        fixed_mask_full=fixed_mask_full,
        fixed_shape_hw=fixed_bundle.rgb.shape[:2],
        tile_defs=tile_defs,
    )

    # Experiment 2: anchor-preserving geometric model comparison
    exp2_translation_rows = list(best_scale_rows)
    exp2_translation_tile_warps = list(best_scale_tile_warps)
    anchor_scene_xy = (float(ANCHOR_PAIR["section_scene_xy"][0]), float(ANCHOR_PAIR["section_scene_xy"][1]))
    similarity_correction = _fit_anchor_locked_similarity(exp2_translation_rows, anchor_scene_xy)
    similarity_full_mat = _compose_affine(similarity_correction, best_full_mat)
    similarity_rows, similarity_tile_warps = _collect_tile_results(
        label="anchor_locked_similarity",
        moving_reg_projection_u8=moving_scaled_u8,
        moving_signal_mask_u8=moving_scaled_mask_u8,
        fixed_gray_full=fixed_gray_full_u8,
        fixed_mask_full=fixed_mask_full,
        tile_defs=tile_defs,
        full_mat=similarity_full_mat,
    )
    for r in similarity_rows:
        r["method"] = "anchor_locked_similarity"
    affine_correction = _fit_anchor_locked_affine(exp2_translation_rows, anchor_scene_xy)
    affine_full_mat = _compose_affine(affine_correction, best_full_mat)
    affine_rows, affine_tile_warps = _collect_tile_results(
        label="anchor_locked_affine",
        moving_reg_projection_u8=moving_scaled_u8,
        moving_signal_mask_u8=moving_scaled_mask_u8,
        fixed_gray_full=fixed_gray_full_u8,
        fixed_mask_full=fixed_mask_full,
        tile_defs=tile_defs,
        full_mat=affine_full_mat,
    )
    for r in affine_rows:
        r["method"] = "anchor_locked_affine"

    _write_csv(process_dir / "tile_metrics_exp2_models.csv", exp2_translation_rows + similarity_rows + affine_rows)

    # Experiment 3
    column_offsets = _column_smooth_offsets(exp2_translation_rows)
    col_rows, col_tile_warps = _collect_tile_results(
        label="column_wise_correction",
        moving_reg_projection_u8=moving_scaled_u8,
        moving_signal_mask_u8=moving_scaled_mask_u8,
        fixed_gray_full=fixed_gray_full_u8,
        fixed_mask_full=fixed_mask_full,
        tile_defs=tile_defs,
        full_mat=best_full_mat,
        per_tile_shift=column_offsets,
    )
    smooth_offsets = _tilewise_smooth_offsets(exp2_translation_rows)
    smooth_rows, smooth_tile_warps = _collect_tile_results(
        label="tilewise_smooth_correction",
        moving_reg_projection_u8=moving_scaled_u8,
        moving_signal_mask_u8=moving_scaled_mask_u8,
        fixed_gray_full=fixed_gray_full_u8,
        fixed_mask_full=fixed_mask_full,
        tile_defs=tile_defs,
        full_mat=best_full_mat,
        per_tile_shift=smooth_offsets,
    )
    _write_csv(process_dir / "tile_metrics_exp3_piecewise.csv", col_rows + smooth_rows)

    # Comparison contact and summary
    comparison_panels = [
        ("Current coarse (scale=0.943)", exp0_overlay),
        (
            f"Best-scale translation only (scale={compare_scale:.3f})",
            _draw_full_overlay(
                fixed_gray_full_u8,
                fixed_mask_full,
                exp2_translation_rows,
                exp2_translation_tile_warps,
                title="Best-scale translation only",
            ),
        ),
        (
            "Anchor-locked similarity",
            _draw_full_overlay(
                fixed_gray_full_u8,
                fixed_mask_full,
                similarity_rows,
                similarity_tile_warps,
                title="Anchor-locked similarity",
            ),
        ),
        (
            "Anchor-locked affine",
            _draw_full_overlay(
                fixed_gray_full_u8,
                fixed_mask_full,
                affine_rows,
                affine_tile_warps,
                title="Anchor-locked affine",
            ),
        ),
        (
            "Column-wise correction",
            _draw_full_overlay(
                fixed_gray_full_u8,
                fixed_mask_full,
                col_rows,
                col_tile_warps,
                title="Column-wise correction",
            ),
        ),
        (
            "Tile-wise smooth correction",
            _draw_full_overlay(
                fixed_gray_full_u8,
                fixed_mask_full,
                smooth_rows,
                smooth_tile_warps,
                title="Tile-wise smooth correction",
            ),
        ),
    ]
    _build_method_comparison_contact(comparison_panels, figures_dir / "best_methods_comparison.png")

    def _agg(rows: list[dict[str, Any]]) -> dict[str, float]:
        return {
            "mean_cc": float(np.nanmean([float(r["current_cc"]) for r in rows])),
            "mean_mi": float(np.nanmean([float(r["current_mi"]) for r in rows])),
            "mean_abs_dx": float(np.nanmean(np.abs([float(r["dx_star_px"]) for r in rows]))),
            "mean_abs_dy": float(np.nanmean(np.abs([float(r["dy_star_px"]) for r in rows]))),
            "rightmost_mean_dx": float(np.nanmean([float(r["dx_star_px"]) for r in rows if int(r["col"]) == 5])),
        }

    coarse_firstcol = [r for r in exp0_rows if int(r["col"]) == 0]
    coarse_lastcol = [r for r in exp0_rows if int(r["col"]) == 5]
    summary_lines = [
        "# 2501_60 Confocal Grid Geometry Diagnostic",
        "",
        "## Input state",
        f"- section: `{SECTION_DIR}`",
        f"- confocal tile dir: `{CONFOCAL_TILE_DIR}`",
        f"- manual state: tx={MANUAL_STATE['tx_px']}, ty={MANUAL_STATE['ty_px']}, angle={MANUAL_STATE['angle_deg']}, scale={MANUAL_STATE['scale']}, flip_ud={MANUAL_STATE['flip_ud']}",
        f"- anchor: A1=B1 at section_scene=({ANCHOR_PAIR['section_scene_xy'][0]:.1f},{ANCHOR_PAIR['section_scene_xy'][1]:.1f}), confocal_raw=({ANCHOR_PAIR['confocal_raw_xy'][0]:.1f},{ANCHOR_PAIR['confocal_raw_xy'][1]:.1f})",
        "- registration input: inverted confocal gray + masked percentile normalization + masked histogram matching to myelin",
        "- anchor handling: all exp2 geometric models are constrained to keep A1/B1 fixed",
        "",
        "## Experiment 0",
        f"- first-column mean CC: {np.nanmean([float(r['current_cc']) for r in coarse_firstcol]):.4f}",
        f"- last-column mean CC: {np.nanmean([float(r['current_cc']) for r in coarse_lastcol]):.4f}",
        f"- first-column mean dx*: {np.nanmean([float(r['dx_star_px']) for r in coarse_firstcol]):.2f}px",
        f"- last-column mean dx*: {np.nanmean([float(r['dx_star_px']) for r in coarse_lastcol]):.2f}px",
        "- interpretation target: if dx* becomes more positive with x and is similar across rows, the dominant problem is global x-scale mismatch.",
        "",
        "## Experiment 1",
        f"- best overall scale by mean tile CC: {best_overall_scale:.3f}",
        f"- best right-column scale by |dx*|: {best_right_scale:.3f}",
        "",
        "## Experiment 2",
        f"- translation-only @ best scale: {json.dumps(_agg(exp2_translation_rows), ensure_ascii=True)}",
        f"- anchor-locked similarity: {json.dumps(_agg(similarity_rows), ensure_ascii=True)}",
        f"- anchor-locked affine: {json.dumps(_agg(affine_rows), ensure_ascii=True)}",
        "",
        "## Experiment 3",
        f"- column-wise correction: {json.dumps(_agg(col_rows), ensure_ascii=True)}",
        f"- tile-wise smooth correction: {json.dumps(_agg(smooth_rows), ensure_ascii=True)}",
        "",
        "## Files",
        "- `process/tile_metrics.csv`",
        "- `process/scale_sweep_summary.csv`",
        "- `figures/dx_vs_x_plot.png`",
        "- `figures/per_column_residual_plot.png`",
        "- `figures/full_overlay_contact_sheet.png`",
        "- `figures/tile_zoom_storyboard.png`",
        "- `figures/best_methods_comparison.png`",
    ]
    (out_root / "summary.md").write_text("\n".join(summary_lines), encoding="utf-8")

    _write_json(
        out_root / "run_manifest.json",
        {
            "manual_state": MANUAL_STATE,
            "anchor_pair": ANCHOR_PAIR,
            "scale_values": SCALE_VALUES,
            "search_radius": SEARCH_RADIUS,
            "best_overall_scale": best_overall_scale,
            "best_right_scale": best_right_scale,
            "anchor_init_info": anchor_info,
            "preprocessing": {
                "confocal_inverted": True,
                "fixed_normalization": "masked_percentile_1_99",
                "moving_normalization": "masked_percentile_1_99 + masked_histogram_match_to_fixed",
            },
            "paths": {
                "summary_md": str(out_root / "summary.md"),
                "tile_metrics_csv": str(process_dir / "tile_metrics.csv"),
                "scale_sweep_summary_csv": str(process_dir / "scale_sweep_summary.csv"),
                "dx_vs_x_plot": str(figures_dir / "dx_vs_x_plot.png"),
                "per_column_residual_plot": str(figures_dir / "per_column_residual_plot.png"),
                "full_overlay_contact_sheet": str(figures_dir / "full_overlay_contact_sheet.png"),
                "tile_zoom_storyboard": str(figures_dir / "tile_zoom_storyboard.png"),
                "best_methods_comparison": str(figures_dir / "best_methods_comparison.png"),
                "tile_metrics_exp2_models_csv": str(process_dir / "tile_metrics_exp2_models.csv"),
                "tile_metrics_exp3_piecewise_csv": str(process_dir / "tile_metrics_exp3_piecewise.csv"),
            },
        },
    )
    print(f"Diagnostics written to: {out_root}")


if __name__ == "__main__":
    main()

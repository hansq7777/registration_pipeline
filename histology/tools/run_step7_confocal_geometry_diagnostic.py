from __future__ import annotations

import csv
import gc
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None


REPO_ROOT = Path(__file__).resolve().parents[1]
GUI_MVP_ROOT = REPO_ROOT / "gui_mvp"
if str(GUI_MVP_ROOT) not in sys.path:
    sys.path.insert(0, str(GUI_MVP_ROOT))

from hitl_gui.application.confocal_registration import (  # noqa: E402
    STEP7_TARGET_UM_PER_PX,
    _anchor_guided_manual_affine,
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
    rgb_to_gray_float,
)
from hitl_gui.application.section_workspace import WorkspaceSection  # noqa: E402


SECTION_DIR = Path("/mnt/d/Research/Image Analysis/Nanozoomer scans/20250327 rat myelin quantification/Tissue&Masks/2501_60")
OUT_DIR = Path(
    os.environ.get(
        "STEP7_DIAG_OUT_DIR",
        "/mnt/c/Users/Siqi/Desktop/REVIEW/20260408_confocal_tile_diagnostic_2501_60_v2",
    )
)
CONFOCAL_TILE_DIR = Path("/mnt/d/Research/Image Analysis/Confocal Myelin data/202512_8rats_3ROIs/2501_60_R_IL")
CONFOCAL_TILE_PATHS = sorted(CONFOCAL_TILE_DIR.glob("*.ome.tif"))

MYELIN_LABEL = "2501_60"
PROJECTION_MODE = "focus"
CHANNEL_INDEX = 0
NOMINAL_OVERLAP = 0.10

BASE_TX = -155.0
BASE_TY = -63.0
BASE_ANGLE = 0.8
BASE_SCALE = 0.943
FLIP_LR = False
FLIP_UD = True
ANCHOR_A1 = (4715.0, 3363.7)
ANCHOR_B1_RAW = (117.7, 554.3)

SEARCH_RADIUS = 72
EXP0_TILE_MARGIN = 48
EXP0_SHIFTED_MARGIN = 32
SCALE_SWEEP = np.round(np.arange(0.930, 0.970 + 1e-9, 0.002), 3)
RUN_PHASE = os.environ.get("STEP7_DIAG_PHASE", "all").strip().lower()


@dataclass
class TileDef:
    tile_index: int
    row_display: int
    col: int
    raw_bbox_xyxy: tuple[int, int, int, int]
    scaled_bbox_xyxy: tuple[int, int, int, int]
    center_scaled_xy: tuple[float, float]


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _clip_u8(gray01: np.ndarray) -> np.ndarray:
    return np.clip(np.round(np.asarray(gray01, dtype=np.float32) * 255.0), 0, 255).astype(np.uint8)


def _affine_apply_points(mat: np.ndarray, xy: np.ndarray) -> np.ndarray:
    arr = np.asarray(xy, dtype=np.float64)
    aug = np.concatenate([arr, np.ones((arr.shape[0], 1), dtype=np.float64)], axis=1)
    out = aug @ np.asarray(mat, dtype=np.float64).T
    return out[:, :2]


def _ants_affine_to_cv2(mat_path: Path) -> np.ndarray:
    params, fixed = _load_ants_affine_mat(mat_path)
    p = np.asarray(params, dtype=np.float64).reshape(-1)
    c = np.asarray(fixed, dtype=np.float64).reshape(-1)
    linear = np.array([[p[0], p[1]], [p[2], p[3]]], dtype=np.float64)
    trans = np.array([p[4], p[5]], dtype=np.float64)
    offset = c + trans - linear @ c
    return np.concatenate([linear, offset[:, None]], axis=1).astype(np.float32)


def _bbox_from_mask(mask: np.ndarray, margin: int = 0) -> tuple[int, int, int, int]:
    ys, xs = np.where(np.asarray(mask) > 0)
    h, w = mask.shape[:2]
    if ys.size == 0 or xs.size == 0:
        return 0, h, 0, w
    y0 = max(0, int(ys.min()) - margin)
    y1 = min(h, int(ys.max()) + 1 + margin)
    x0 = max(0, int(xs.min()) - margin)
    x1 = min(w, int(xs.max()) + 1 + margin)
    return y0, y1, x0, x1


def _crop_to_bbox(arr: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    y0, y1, x0, x1 = bbox
    return arr[y0:y1, x0:x1]


def _compose_panel(tiles: list[np.ndarray], *, cols: int, bg: tuple[int, int, int] = (245, 245, 245)) -> np.ndarray:
    if not tiles:
        return np.full((256, 256, 3), bg, dtype=np.uint8)
    tile_h = max(tile.shape[0] for tile in tiles)
    tile_w = max(tile.shape[1] for tile in tiles)
    rows = int(math.ceil(len(tiles) / float(cols)))
    pad = 18
    canvas = np.full((pad + rows * (tile_h + pad), pad + cols * (tile_w + pad), 3), bg, dtype=np.uint8)
    for idx, tile in enumerate(tiles):
        r = idx // cols
        c = idx % cols
        y0 = pad + r * (tile_h + pad)
        x0 = pad + c * (tile_w + pad)
        canvas[y0 : y0 + tile.shape[0], x0 : x0 + tile.shape[1]] = tile
    return canvas


def _downscale_rgb(rgb: np.ndarray, *, max_long_edge: int = 2200) -> np.ndarray:
    arr = np.asarray(rgb)
    h, w = arr.shape[:2]
    long_edge = max(h, w)
    if long_edge <= int(max_long_edge):
        return arr
    scale = float(max_long_edge) / float(long_edge)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(arr, (new_w, new_h), interpolation=cv2.INTER_AREA)


def _tile_storyboard_cell(
    *,
    title: str,
    fixed_crop: np.ndarray,
    moving_crop: np.ndarray,
    fixed_mask_crop: np.ndarray,
    moving_mask_crop: np.ndarray,
    shifted_crop: np.ndarray | None,
    shifted_mask_crop: np.ndarray | None,
    note1: str,
    note2: str,
) -> np.ndarray:
    fixed_panel = gray_preview_panel(fixed_crop)
    moving_panel = gray_preview_panel(moving_crop)
    overlay_now = overlay_preview(fixed_crop, moving_crop, fixed_mask_crop, moving_mask_crop)
    if shifted_crop is not None and shifted_mask_crop is not None:
        overlay_best = overlay_preview(fixed_crop, shifted_crop, fixed_mask_crop, shifted_mask_crop)
    else:
        overlay_best = np.full_like(overlay_now, 255)
    panels = [fixed_panel, moving_panel, overlay_now, overlay_best]
    pad = 8
    title_h = 56
    note_h = 42
    h = max(p.shape[0] for p in panels)
    w = max(p.shape[1] for p in panels)
    canvas = np.full((title_h + h + note_h, pad * 5 + w * 4, 3), 248, dtype=np.uint8)
    cv2.putText(canvas, title, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (20, 20, 20), 2, cv2.LINE_AA)
    cv2.putText(canvas, note1, (12, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (55, 55, 55), 1, cv2.LINE_AA)
    xs = []
    for i, panel in enumerate(panels):
        x0 = pad + i * (w + pad)
        y0 = title_h
        xs.append(x0)
        canvas[y0 : y0 + panel.shape[0], x0 : x0 + panel.shape[1]] = panel
        cv2.rectangle(canvas, (x0, y0), (x0 + panel.shape[1] - 1, y0 + panel.shape[0] - 1), (120, 120, 120), 1)
    labels = ["Fixed", "Moving", "Overlay now", "Overlay best shift"]
    for x0, label in zip(xs, labels):
        cv2.putText(canvas, label, (x0 + 4, title_h + h + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (60, 60, 60), 1, cv2.LINE_AA)
    cv2.putText(canvas, note2, (12, title_h + h + 34), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (55, 55, 55), 1, cv2.LINE_AA)
    return canvas


_HEAVY_TILE_KEYS = {
    "fixed_crop",
    "fixed_mask_crop",
    "moving_crop",
    "moving_mask_crop",
    "shifted_crop",
    "shifted_mask_crop",
}


def _slim_row(row: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in row.items() if k not in _HEAVY_TILE_KEYS}


def _safe_match_template(
    fixed_search: np.ndarray,
    moving_template: np.ndarray,
    mask_template: np.ndarray,
) -> tuple[float, tuple[int, int]]:
    fixed_u8 = _clip_u8(fixed_search)
    moving_u8 = _clip_u8(moving_template)
    mask_u8 = np.where(mask_template > 0, 255, 0).astype(np.uint8)
    if fixed_u8.shape[0] < moving_u8.shape[0] or fixed_u8.shape[1] < moving_u8.shape[1]:
        return float("nan"), (0, 0)
    if int(mask_u8.sum()) <= 0:
        return float("nan"), (0, 0)
    res = cv2.matchTemplate(fixed_u8, moving_u8, cv2.TM_CCORR_NORMED, mask=mask_u8)
    _min_val, max_val, _min_loc, max_loc = cv2.minMaxLoc(res)
    return float(max_val), (int(max_loc[0]), int(max_loc[1]))


def _shift_crop(
    crop: np.ndarray,
    mask: np.ndarray,
    dx: int,
    dy: int,
) -> tuple[np.ndarray, np.ndarray]:
    h, w = crop.shape[:2]
    mat = np.array([[1.0, 0.0, float(dx)], [0.0, 1.0, float(dy)]], dtype=np.float32)
    shifted = cv2.warpAffine(crop.astype(np.float32), mat, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
    shifted_mask = cv2.warpAffine(np.where(mask > 0, 255, 0).astype(np.uint8), mat, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return shifted.astype(np.float32), (shifted_mask > 0).astype(np.float32)


def _build_tile_defs(
    stitch_info: dict[str, Any],
    *,
    raw_shape_hw: tuple[int, int],
    scaled_shape_hw: tuple[int, int],
    flip_ud: bool,
) -> list[TileDef]:
    positions = stitch_info.get("tile_positions_xy") or []
    tile_shape = stitch_info.get("tile_shape_hw") or [0, 0]
    if not positions or len(tile_shape) != 2:
        raise ValueError("Missing tile positions/tile shape in stitch info")
    raw_h, raw_w = int(raw_shape_hw[0]), int(raw_shape_hw[1])
    scaled_h, scaled_w = int(scaled_shape_hw[0]), int(scaled_shape_hw[1])
    tile_h_raw, tile_w_raw = int(tile_shape[0]), int(tile_shape[1])
    unique_xs = sorted({int(p[0]) for p in positions})
    unique_ys = sorted({int(p[1]) for p in positions})
    x_to_col = {x: idx for idx, x in enumerate(unique_xs)}
    if flip_ud:
        y_to_row = {y: (len(unique_ys) - 1 - idx) for idx, y in enumerate(unique_ys)}
    else:
        y_to_row = {y: idx for idx, y in enumerate(unique_ys)}
    sx = float(scaled_w) / float(raw_w)
    sy = float(scaled_h) / float(raw_h)
    defs: list[TileDef] = []
    for tile_index, pos in enumerate(positions):
        raw_x = int(pos[0])
        raw_y = int(pos[1])
        x0 = max(0, min(scaled_w - 1, int(round(raw_x * sx))))
        y0 = max(0, min(scaled_h - 1, int(round(raw_y * sy))))
        x1 = max(x0 + 1, min(scaled_w, int(round((raw_x + tile_w_raw) * sx))))
        y1 = max(y0 + 1, min(scaled_h, int(round((raw_y + tile_h_raw) * sy))))
        defs.append(
            TileDef(
                tile_index=int(tile_index),
                row_display=int(y_to_row[raw_y]),
                col=int(x_to_col[raw_x]),
                raw_bbox_xyxy=(raw_x, raw_y, raw_x + tile_w_raw, raw_y + tile_h_raw),
                scaled_bbox_xyxy=(x0, y0, x1, y1),
                center_scaled_xy=((x0 + x1) / 2.0, (y0 + y1) / 2.0),
            )
        )
    defs.sort(key=lambda t: (t.row_display, t.col))
    return defs


def _prepare_state(scale: float) -> dict[str, Any]:
    item = WorkspaceSection(
        section_dir=SECTION_DIR,
        label=MYELIN_LABEL,
        stain="myelin",
        metadata_path=SECTION_DIR / "metadata.json",
        crop_path=SECTION_DIR / "crop_raw.png",
        has_masks=True,
        has_prepared_work=False,
        prepared_work_profiles=(),
    )
    fixed_bundle = prepare_myelin_confocal_fixed_bundle(item, max_long_edge=None, target_um_per_px=STEP7_TARGET_UM_PER_PX)
    projection_bundle = load_confocal_projection(
        CONFOCAL_TILE_PATHS,
        mode=PROJECTION_MODE,
        channel_index=CHANNEL_INDEX,
        nominal_overlap_fraction=NOMINAL_OVERLAP,
    )
    raw_projection = projection_bundle.projection_u8
    raw_mask = np.where(raw_projection > 0, 255, 0).astype(np.uint8)
    scaled_projection, scale_info = _resample_projection_to_target_um_per_px(
        raw_projection,
        source_um_per_px_xy=projection_bundle.physical_um_per_px_xy,
        target_um_per_px_xy=fixed_bundle.preview_um_per_px_xy,
    )
    scaled_mask = _resample_mask_to_target_um_per_px(
        raw_mask,
        source_um_per_px_xy=projection_bundle.physical_um_per_px_xy,
        target_um_per_px_xy=fixed_bundle.preview_um_per_px_xy,
    )
    reg_projection = _invert_confocal_u8(scaled_projection)
    base_manual_mat = build_manual_affine(
        reg_projection.shape[:2],
        fixed_bundle.rgb.shape[:2],
        tx_px=BASE_TX,
        ty_px=BASE_TY,
        angle_deg=BASE_ANGLE,
        scale=float(scale),
        flip_lr=FLIP_LR,
        flip_ud=FLIP_UD,
    )
    anchor_pairs = [
        {
            "index": 1,
            "section_scene_xy": [float(ANCHOR_A1[0]), float(ANCHOR_A1[1])],
            "confocal_raw_xy": [float(ANCHOR_B1_RAW[0]), float(ANCHOR_B1_RAW[1])],
        }
    ]
    manual_mat, anchor_info = _anchor_guided_manual_affine(
        moving_shape_hw=reg_projection.shape[:2],
        fixed_shape_hw=fixed_bundle.rgb.shape[:2],
        current_mat=base_manual_mat,
        anchor_pairs=anchor_pairs,
        flip_lr=FLIP_LR,
        flip_ud=FLIP_UD,
    )
    moving_u8_full, moving_mask_u8_full, effective_manual_mat = apply_affine_matrix(
        reg_projection,
        fixed_bundle.rgb.shape[:2],
        mat=manual_mat,
        moving_mask_u8=scaled_mask,
    )
    fixed_gray_full = rgb_to_gray_float(fixed_bundle.rgb)
    fixed_mask_full = (fixed_bundle.labels == 1).astype(np.float32)
    if not np.any(fixed_mask_full > 0):
        fixed_mask_full = (fixed_bundle.labels > 0).astype(np.float32)
    moving_gray_full = moving_u8_full.astype(np.float32) / 255.0
    moving_mask_full = (moving_mask_u8_full > 0).astype(np.float32)
    tile_defs = _build_tile_defs(
        projection_bundle.stitch_info,
        raw_shape_hw=raw_projection.shape[:2],
        scaled_shape_hw=scaled_projection.shape[:2],
        flip_ud=FLIP_UD,
    )
    return {
        "item": item,
        "fixed_rgb": fixed_bundle.rgb,
        "fixed_gray_full": fixed_gray_full,
        "fixed_mask_full": fixed_mask_full,
        "fixed_labels": fixed_bundle.labels,
        "fixed_info": {
            "preview_um_per_px_xy": fixed_bundle.preview_um_per_px_xy,
            "source_um_per_px_xy": fixed_bundle.source_um_per_px_xy,
        },
        "projection_bundle": projection_bundle,
        "raw_projection": raw_projection,
        "scaled_projection": scaled_projection,
        "scaled_mask": scaled_mask,
        "reg_projection": reg_projection,
        "manual_mat": effective_manual_mat.astype(np.float32),
        "manual_gray_full_u8": moving_u8_full,
        "manual_gray_full": moving_gray_full,
        "manual_mask_full_u8": moving_mask_u8_full,
        "manual_mask_full": moving_mask_full,
        "tile_defs": tile_defs,
        "anchor_info": anchor_info,
        "scale_info": scale_info,
    }


def _warp_tile_from_scaled(
    scaled_reg_projection_u8: np.ndarray,
    scaled_mask_u8: np.ndarray,
    fixed_shape_hw: tuple[int, int],
    global_mat: np.ndarray,
    tile_def: TileDef,
    *,
    extra_dx: float = 0.0,
    extra_dy: float = 0.0,
) -> dict[str, Any]:
    x0, y0, x1, y1 = tile_def.scaled_bbox_xyxy
    tile_img = scaled_reg_projection_u8[y0:y1, x0:x1]
    tile_mask = scaled_mask_u8[y0:y1, x0:x1]
    linear = np.asarray(global_mat[:, :2], dtype=np.float32)
    trans = np.asarray(global_mat[:, 2], dtype=np.float32) + linear @ np.array([float(x0), float(y0)], dtype=np.float32)
    trans = trans + np.array([float(extra_dx), float(extra_dy)], dtype=np.float32)
    tile_mat = np.concatenate([linear, trans[:, None]], axis=1).astype(np.float32)
    tile_h, tile_w = tile_img.shape[:2]
    corners = np.asarray(
        [[0.0, 0.0], [float(tile_w), 0.0], [float(tile_w), float(tile_h)], [0.0, float(tile_h)]],
        dtype=np.float64,
    )
    warped_corners = _affine_apply_points(tile_mat, corners)
    bx0 = max(0, int(math.floor(float(np.min(warped_corners[:, 0]))) - EXP0_TILE_MARGIN))
    by0 = max(0, int(math.floor(float(np.min(warped_corners[:, 1]))) - EXP0_TILE_MARGIN))
    bx1 = min(int(fixed_shape_hw[1]), int(math.ceil(float(np.max(warped_corners[:, 0]))) + EXP0_TILE_MARGIN))
    by1 = min(int(fixed_shape_hw[0]), int(math.ceil(float(np.max(warped_corners[:, 1]))) + EXP0_TILE_MARGIN))
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
    center_scene = _affine_apply_points(
        tile_mat,
        np.asarray([[0.5 * tile_w, 0.5 * tile_h]], dtype=np.float64),
    )[0]
    return {
        "warped_img_u8": warped_img_u8,
        "warped_gray": warped_img_u8.astype(np.float32) / 255.0,
        "warped_mask_u8": warped_mask_u8,
        "warped_mask": (warped_mask_u8 > 0).astype(np.float32),
        "tile_mat": tile_mat,
        "local_mat": local_mat,
        "eval_bbox_yxyx": (by0, by1, bx0, bx1),
        "center_scene_xy": (float(center_scene[0]), float(center_scene[1])),
    }


def _tile_diagnostic(
    *,
    fixed_gray_full: np.ndarray,
    fixed_mask_full: np.ndarray,
    tile_warped_gray: np.ndarray,
    tile_warped_mask: np.ndarray,
    eval_bbox_yxyx: tuple[int, int, int, int],
    center_scene_xy: tuple[float, float],
    search_radius: int,
    tile_def: TileDef,
) -> dict[str, Any]:
    fy0, fy1, fx0, fx1 = eval_bbox_yxyx
    fixed_crop = _crop_to_bbox(fixed_gray_full, eval_bbox_yxyx)
    fixed_mask_crop = _crop_to_bbox(fixed_mask_full, eval_bbox_yxyx)
    moving_crop = tile_warped_gray
    moving_mask_crop = tile_warped_mask
    metrics, metric_timing = compute_registration_metrics(fixed_crop, moving_crop, fixed_mask_crop, moving_mask_crop)

    tight_bbox_local = _bbox_from_mask(tile_warped_mask, margin=0)
    ty0, ty1, tx0, tx1 = tight_bbox_local
    sy0 = max(0, ty0 - search_radius)
    sy1 = min(fixed_crop.shape[0], ty1 + search_radius)
    sx0 = max(0, tx0 - search_radius)
    sx1 = min(fixed_crop.shape[1], tx1 + search_radius)
    fixed_search = fixed_crop[sy0:sy1, sx0:sx1]
    templ = tile_warped_gray[ty0:ty1, tx0:tx1]
    templ_mask = tile_warped_mask[ty0:ty1, tx0:tx1]
    best_cc, best_loc = _safe_match_template(fixed_search, templ, templ_mask)
    best_x = sx0 + int(best_loc[0])
    best_y = sy0 + int(best_loc[1])
    dx = int(best_x - tx0)
    dy = int(best_y - ty0)

    shifted_crop, shifted_mask_crop = _shift_crop(moving_crop, moving_mask_crop, dx=dx, dy=dy)
    shifted_metrics, _ = compute_registration_metrics(fixed_crop, shifted_crop, fixed_mask_crop, shifted_mask_crop)

    return {
        "tile_index": tile_def.tile_index,
        "row_display": tile_def.row_display,
        "col": tile_def.col,
        "center_x_scene": float(center_scene_xy[0]),
        "center_y_scene": float(center_scene_xy[1]),
        "tight_bbox_yxyx": [int(fy0 + ty0), int(fy0 + ty1), int(fx0 + tx0), int(fx0 + tx1)],
        "eval_bbox_yxyx": [int(fy0), int(fy1), int(fx0), int(fx1)],
        "cc": float(metrics.get("cc", float("nan"))),
        "mi": float(metrics.get("mi", float("nan"))),
        "best_shift_dx_px": int(dx),
        "best_shift_dy_px": int(dy),
        "best_shift_cc": float(best_cc),
        "shifted_cc": float(shifted_metrics.get("cc", float("nan"))),
        "shifted_mi": float(shifted_metrics.get("mi", float("nan"))),
        "metric_t_total": float(metric_timing.get("total", 0.0)),
        "fixed_crop": fixed_crop,
        "fixed_mask_crop": fixed_mask_crop,
        "moving_crop": moving_crop,
        "moving_mask_crop": moving_mask_crop,
        "shifted_crop": shifted_crop,
        "shifted_mask_crop": shifted_mask_crop,
    }


def _run_tile_diagnostics(
    state: dict[str, Any],
    *,
    extra_dx_by_tile: dict[int, float] | None = None,
    extra_dy_by_tile: dict[int, float] | None = None,
) -> list[dict[str, Any]]:
    fixed_shape_hw = state["fixed_gray_full"].shape[:2]
    records: list[dict[str, Any]] = []
    extra_dx_by_tile = dict(extra_dx_by_tile or {})
    extra_dy_by_tile = dict(extra_dy_by_tile or {})
    for tile in state["tile_defs"]:
        warped = _warp_tile_from_scaled(
            state["reg_projection"],
            state["scaled_mask"],
            fixed_shape_hw,
            state["manual_mat"],
            tile,
            extra_dx=float(extra_dx_by_tile.get(tile.tile_index, 0.0)),
            extra_dy=float(extra_dy_by_tile.get(tile.tile_index, 0.0)),
        )
        rec = _tile_diagnostic(
            fixed_gray_full=state["fixed_gray_full"],
            fixed_mask_full=state["fixed_mask_full"],
            tile_warped_gray=warped["warped_gray"],
            tile_warped_mask=warped["warped_mask"],
            eval_bbox_yxyx=warped["eval_bbox_yxyx"],
            center_scene_xy=warped["center_scene_xy"],
            search_radius=SEARCH_RADIUS,
            tile_def=tile,
        )
        records.append(rec)
    records.sort(key=lambda r: (int(r["row_display"]), int(r["col"])))
    return records


def _write_tile_metrics_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = [
        "experiment",
        "arm",
        "tile_index",
        "row_display",
        "col",
        "center_x_scene",
        "center_y_scene",
        "cc",
        "mi",
        "best_shift_dx_px",
        "best_shift_dy_px",
        "best_shift_cc",
        "shifted_cc",
        "shifted_mi",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fields})


def _plot_dx_vs_x(rows: list[dict[str, Any]], out_path: Path, *, title: str) -> None:
    if plt is None:
        return
    fig, ax = plt.subplots(figsize=(9, 5), dpi=160)
    for row_id in sorted({int(r["row_display"]) for r in rows}):
        subset = [r for r in rows if int(r["row_display"]) == row_id]
        subset = sorted(subset, key=lambda r: float(r["center_x_scene"]))
        ax.plot(
            [float(r["center_x_scene"]) for r in subset],
            [float(r["best_shift_dx_px"]) for r in subset],
            marker="o",
            label=f"row {row_id}",
        )
    xs = np.asarray([float(r["center_x_scene"]) for r in rows], dtype=np.float64)
    ys = np.asarray([float(r["best_shift_dx_px"]) for r in rows], dtype=np.float64)
    if xs.size >= 2:
        coef = np.polyfit(xs, ys, 1)
        xfit = np.linspace(xs.min(), xs.max(), 128)
        yfit = np.polyval(coef, xfit)
        ax.plot(xfit, yfit, "--", color="black", alpha=0.6, label=f"overall slope={coef[0]:.4f}")
    ax.axhline(0.0, color="gray", linewidth=1)
    ax.set_title(title)
    ax.set_xlabel("tile center x (scene px)")
    ax.set_ylabel("best local dx* (px)")
    ax.legend()
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _plot_per_column(rows: list[dict[str, Any]], out_path: Path, *, title: str) -> None:
    if plt is None:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), dpi=160)
    cols = sorted({int(r["col"]) for r in rows})
    mean_dx = [float(np.mean([float(r["best_shift_dx_px"]) for r in rows if int(r["col"]) == col])) for col in cols]
    mean_cc = [float(np.mean([float(r["cc"]) for r in rows if int(r["col"]) == col])) for col in cols]
    axes[0].plot(cols, mean_dx, marker="o")
    axes[0].axhline(0.0, color="gray", linewidth=1)
    axes[0].set_title("mean dx* by column")
    axes[0].set_xlabel("column")
    axes[0].set_ylabel("dx* (px)")
    axes[0].grid(True, alpha=0.25)
    axes[1].plot(cols, mean_cc, marker="o", color="tab:red")
    axes[1].set_title("mean CC by column")
    axes[1].set_xlabel("column")
    axes[1].set_ylabel("local CC")
    axes[1].grid(True, alpha=0.25)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _linear_dx_summary(rows: list[dict[str, Any]]) -> dict[str, float]:
    xs = np.asarray([float(r["center_x_scene"]) for r in rows], dtype=np.float64)
    dx = np.asarray([float(r["best_shift_dx_px"]) for r in rows], dtype=np.float64)
    if xs.size < 2:
        return {"slope_dx_vs_x": float("nan"), "corr_dx_x": float("nan")}
    coef = np.polyfit(xs, dx, 1)
    corr = np.corrcoef(xs, dx)[0, 1]
    return {"slope_dx_vs_x": float(coef[0]), "corr_dx_x": float(corr)}


def _render_exp0_tiles(rows: list[dict[str, Any]], out_path: Path) -> None:
    cells: list[np.ndarray] = []
    for rec in rows:
        title = f"tile {rec['tile_index']} r{rec['row_display']} c{rec['col']}"
        note1 = f"CC={rec['cc']:.3f} MI={rec['mi']:.3f}"
        note2 = f"dx*={int(rec['best_shift_dx_px']):+d} dy*={int(rec['best_shift_dy_px']):+d} shiftedCC={rec['shifted_cc']:.3f}"
        cell = _tile_storyboard_cell(
            title=title,
            fixed_crop=rec["fixed_crop"],
            moving_crop=rec["moving_crop"],
            fixed_mask_crop=rec["fixed_mask_crop"],
            moving_mask_crop=rec["moving_mask_crop"],
            shifted_crop=rec["shifted_crop"],
            shifted_mask_crop=rec["shifted_mask_crop"],
            note1=note1,
            note2=note2,
        )
        cells.append(cell)
    panel = _compose_panel(cells, cols=2, bg=(250, 250, 250))
    cv2.imwrite(str(out_path), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))


def _render_full_overlay_grid(
    entries: list[tuple[str, np.ndarray]],
    out_path: Path,
) -> None:
    cells: list[np.ndarray] = []
    pad = 10
    title_h = 44
    for label, overlay in entries:
        panel = _downscale_rgb(overlay, max_long_edge=2200)
        canvas = np.full((title_h + panel.shape[0] + pad, panel.shape[1], 3), 248, dtype=np.uint8)
        cv2.putText(canvas, label, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (20, 20, 20), 2, cv2.LINE_AA)
        canvas[title_h : title_h + panel.shape[0], : panel.shape[1]] = panel
        cells.append(canvas)
    sheet = _compose_panel(cells, cols=2, bg=(250, 250, 250))
    cv2.imwrite(str(out_path), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))


def _aggregate_scale_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    cols = sorted({int(r["col"]) for r in rows})
    mean_cc = float(np.mean([float(r["cc"]) for r in rows]))
    mean_mi = float(np.mean([float(r["mi"]) for r in rows]))
    mean_abs_dx = float(np.mean([abs(float(r["best_shift_dx_px"])) for r in rows]))
    mean_abs_dy = float(np.mean([abs(float(r["best_shift_dy_px"])) for r in rows]))
    per_col_dx = [float(np.mean([float(r["best_shift_dx_px"]) for r in rows if int(r["col"]) == c])) for c in cols]
    per_col_cc = [float(np.mean([float(r["cc"]) for r in rows if int(r["col"]) == c])) for c in cols]
    summary = {
        "mean_tile_cc": mean_cc,
        "mean_tile_mi": mean_mi,
        "mean_abs_dx": mean_abs_dx,
        "mean_abs_dy": mean_abs_dy,
        "rightmost_mean_dx": float(per_col_dx[-1]) if per_col_dx else float("nan"),
        "per_col_dx": per_col_dx,
        "per_col_cc": per_col_cc,
    }
    summary.update(_linear_dx_summary(rows))
    return summary


def _write_scale_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    cols = sorted({int(c) for row in rows for c in row.get("cols", [])})
    fieldnames = [
        "scale",
        "mean_tile_cc",
        "mean_tile_mi",
        "mean_abs_dx",
        "mean_abs_dy",
        "slope_dx_vs_x",
        "corr_dx_x",
        "rightmost_mean_dx",
    ] + [f"col{c}_mean_dx" for c in cols] + [f"col{c}_mean_cc" for c in cols]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            record = {
                "scale": row["scale"],
                "mean_tile_cc": row["mean_tile_cc"],
                "mean_tile_mi": row["mean_tile_mi"],
                "mean_abs_dx": row["mean_abs_dx"],
                "mean_abs_dy": row["mean_abs_dy"],
                "slope_dx_vs_x": row["slope_dx_vs_x"],
                "corr_dx_x": row["corr_dx_x"],
                "rightmost_mean_dx": row["rightmost_mean_dx"],
            }
            per_col_dx = row["per_col_dx"]
            per_col_cc = row["per_col_cc"]
            for idx, c in enumerate(cols):
                record[f"col{c}_mean_dx"] = per_col_dx[idx]
                record[f"col{c}_mean_cc"] = per_col_cc[idx]
            writer.writerow(record)


def _read_scale_summary_csv(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed: dict[str, Any] = {}
            for k, v in row.items():
                if v is None or v == "":
                    parsed[k] = v
                    continue
                try:
                    parsed[k] = float(v)
                except Exception:
                    parsed[k] = v
            rows.append(parsed)
    return rows


def _make_similarity_cfg(state: dict[str, Any], *, ants_bin: Path, out_root: Path, refine_model: str, scale: float):
    fixed_info = {
        "preview_um_per_px_xy": list(state["fixed_info"]["preview_um_per_px_xy"] or [1.0, 1.0]),
        "source_um_per_px_xy": list(state["fixed_info"]["source_um_per_px_xy"] or [1.0, 1.0]),
    }
    projection_info = {
        "source_um_per_px_xy": list(state["projection_bundle"].physical_um_per_px_xy or [float("nan"), float("nan")]),
        "target_um_per_px_xy": list(state["fixed_info"]["preview_um_per_px_xy"] or [1.0, 1.0]),
        "stitch_info": state["projection_bundle"].stitch_info,
    }
    from hitl_gui.application.confocal_registration import ConfocalRigidConfig

    return ConfocalRigidConfig(
        myelin_label=MYELIN_LABEL,
        myelin_section_dir=SECTION_DIR,
        myelin_stain="myelin",
        myelin_rgb=state["fixed_rgb"],
        myelin_labels=state["fixed_labels"],
        myelin_fixed_info=fixed_info,
        confocal_projection_u8=state["scaled_projection"],
        confocal_signal_mask_u8=state["scaled_mask"],
        ants_bin=ants_bin,
        out_root=out_root,
        confocal_sources=list(CONFOCAL_TILE_PATHS),
        confocal_source_mode=state["projection_bundle"].source_mode,
        nominal_overlap_fraction=NOMINAL_OVERLAP,
        projection_info=projection_info,
        projection_mode=PROJECTION_MODE,
        channel_index=CHANNEL_INDEX,
        local_refine_model=refine_model,
        target_working_um_per_px=float(STEP7_TARGET_UM_PER_PX),
        invert_confocal_for_registration=True,
        tx_px=float(BASE_TX),
        ty_px=float(BASE_TY),
        angle_deg=float(BASE_ANGLE),
        scale=float(scale),
        flip_lr=FLIP_LR,
        flip_ud=FLIP_UD,
        anchor_pairs=[
            {
                "index": 1,
                "section_scene_xy": [float(ANCHOR_A1[0]), float(ANCHOR_A1[1])],
                "confocal_raw_xy": [float(ANCHOR_B1_RAW[0]), float(ANCHOR_B1_RAW[1])],
                "confocal_scene_xy": [float(ANCHOR_A1[0]), float(ANCHOR_A1[1])],
            }
        ],
    )


def _evaluate_run_manifest(
    manifest: dict[str, Any],
    state: dict[str, Any],
) -> list[dict[str, Any]]:
    files = manifest.get("files", {}) if isinstance(manifest.get("files"), dict) else {}
    refine_gray_path = Path(str(files.get("local_refine_warped_full_gray")))
    refine_mask_path = Path(str(files.get("local_refine_warped_full_mask")))
    refine_gray = cv2.imread(str(refine_gray_path), cv2.IMREAD_GRAYSCALE)
    refine_mask = cv2.imread(str(refine_mask_path), cv2.IMREAD_GRAYSCALE)
    if refine_gray is None or refine_mask is None:
        raise FileNotFoundError("Missing refined full gray/mask output")
    refine_gray = refine_gray.astype(np.float32) / 255.0
    refine_mask = (refine_mask > 127).astype(np.float32)

    records: list[dict[str, Any]] = []
    fixed_shape_hw = state["fixed_gray_full"].shape[:2]
    for tile in state["tile_defs"]:
        coarse = _warp_tile_from_scaled(
            state["reg_projection"],
            state["scaled_mask"],
            fixed_shape_hw,
            state["manual_mat"],
            tile,
        )
        y0, y1, x0, x1 = coarse["eval_bbox_yxyx"]
        tile_refined_gray = refine_gray[y0:y1, x0:x1]
        tile_refined_mask = refine_mask[y0:y1, x0:x1]
        rec = _tile_diagnostic(
            fixed_gray_full=state["fixed_gray_full"],
            fixed_mask_full=state["fixed_mask_full"],
            tile_warped_gray=tile_refined_gray,
            tile_warped_mask=tile_refined_mask,
            eval_bbox_yxyx=coarse["eval_bbox_yxyx"],
            center_scene_xy=coarse["center_scene_xy"],
            search_radius=SEARCH_RADIUS,
            tile_def=tile,
        )
        rec["refine_model"] = str(manifest.get("local_refine_model") or local_info.get("transform_model") or "unknown")
        records.append(rec)
    records.sort(key=lambda r: (int(r["row_display"]), int(r["col"])))
    return records


def _columnwise_offsets(rows: list[dict[str, Any]], *, anchor_tile_index: int) -> tuple[dict[int, float], dict[int, float]]:
    cols = sorted({int(r["col"]) for r in rows})
    anchor_rec = next((r for r in rows if int(r["tile_index"]) == int(anchor_tile_index)), None)
    anchor_dx = float(anchor_rec["best_shift_dx_px"]) if anchor_rec is not None else 0.0
    anchor_dy = float(anchor_rec["best_shift_dy_px"]) if anchor_rec is not None else 0.0
    dx_by_col = {col: float(np.mean([float(r["best_shift_dx_px"]) for r in rows if int(r["col"]) == col])) - anchor_dx for col in cols}
    dy_by_col = {col: float(np.mean([float(r["best_shift_dy_px"]) for r in rows if int(r["col"]) == col])) - anchor_dy for col in cols}
    return dx_by_col, dy_by_col


def _smooth_tile_offsets(rows: list[dict[str, Any]], *, anchor_tile_index: int) -> tuple[dict[int, float], dict[int, float]]:
    max_row = max(int(r["row_display"]) for r in rows)
    max_col = max(int(r["col"]) for r in rows)
    dx_grid = np.zeros((max_row + 1, max_col + 1), dtype=np.float32)
    dy_grid = np.zeros_like(dx_grid)
    tile_lookup: dict[tuple[int, int], dict[str, Any]] = {}
    anchor_row = 0
    anchor_col = 0
    for r in rows:
        rr = int(r["row_display"])
        cc = int(r["col"])
        tile_lookup[(rr, cc)] = r
        dx_grid[rr, cc] = float(r["best_shift_dx_px"])
        dy_grid[rr, cc] = float(r["best_shift_dy_px"])
        if int(r["tile_index"]) == int(anchor_tile_index):
            anchor_row, anchor_col = rr, cc
    dx_s = cv2.GaussianBlur(dx_grid, (0, 0), sigmaX=1.0, sigmaY=0.8, borderType=cv2.BORDER_REPLICATE)
    dy_s = cv2.GaussianBlur(dy_grid, (0, 0), sigmaX=1.0, sigmaY=0.8, borderType=cv2.BORDER_REPLICATE)
    dx_s = dx_s - float(dx_s[anchor_row, anchor_col])
    dy_s = dy_s - float(dy_s[anchor_row, anchor_col])
    dx_by_tile: dict[int, float] = {}
    dy_by_tile: dict[int, float] = {}
    for (rr, cc), rec in tile_lookup.items():
        dx_by_tile[int(rec["tile_index"])] = float(dx_s[rr, cc])
        dy_by_tile[int(rec["tile_index"])] = float(dy_s[rr, cc])
    return dx_by_tile, dy_by_tile


def _compose_full_overlay(state: dict[str, Any], *, extra_dx_by_tile: dict[int, float] | None = None, extra_dy_by_tile: dict[int, float] | None = None) -> np.ndarray:
    fixed_shape_hw = state["fixed_gray_full"].shape[:2]
    accum = np.zeros(fixed_shape_hw, dtype=np.float32)
    counts = np.zeros(fixed_shape_hw, dtype=np.float32)
    accum_mask = np.zeros(fixed_shape_hw, dtype=np.float32)
    extra_dx_by_tile = dict(extra_dx_by_tile or {})
    extra_dy_by_tile = dict(extra_dy_by_tile or {})
    for tile in state["tile_defs"]:
        warped = _warp_tile_from_scaled(
            state["reg_projection"],
            state["scaled_mask"],
            fixed_shape_hw,
            state["manual_mat"],
            tile,
            extra_dx=float(extra_dx_by_tile.get(tile.tile_index, 0.0)),
            extra_dy=float(extra_dy_by_tile.get(tile.tile_index, 0.0)),
        )
        y0, y1, x0, x1 = warped["eval_bbox_yxyx"]
        mask = warped["warped_mask"]
        accum[y0:y1, x0:x1] += warped["warped_gray"] * mask
        counts[y0:y1, x0:x1] += mask
        accum_mask[y0:y1, x0:x1] = np.maximum(accum_mask[y0:y1, x0:x1], mask)
    moving = np.where(counts > 0, accum / np.maximum(counts, 1.0), 0.0).astype(np.float32)
    return overlay_preview(state["fixed_gray_full"], moving, state["fixed_mask_full"], (accum_mask > 0).astype(np.float32))


def _overlay_from_manifest_full(manifest: dict[str, Any], state: dict[str, Any]) -> np.ndarray:
    files = manifest.get("files", {}) if isinstance(manifest.get("files"), dict) else {}
    gray_path = Path(str(files.get("local_refine_warped_full_gray") or ""))
    mask_path = Path(str(files.get("local_refine_warped_full_mask") or ""))
    gray = cv2.imread(str(gray_path), cv2.IMREAD_GRAYSCALE) if gray_path.exists() else None
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) if mask_path.exists() else None
    if gray is None or mask is None:
        return _compose_full_overlay(state)
    moving = gray.astype(np.float32) / 255.0
    moving_mask = (mask > 127).astype(np.float32)
    return overlay_preview(state["fixed_gray_full"], moving, state["fixed_mask_full"], moving_mask)


def _pick_anchor_tile_index(state: dict[str, Any]) -> int:
    ax, ay = ANCHOR_B1_RAW
    for tile in state["tile_defs"]:
        x0, y0, x1, y1 = tile.raw_bbox_xyxy
        if float(x0) <= ax <= float(x1) and float(y0) <= ay <= float(y1):
            return int(tile.tile_index)
    return int(state["tile_defs"][0].tile_index)


def _write_summary(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> None:
    out_root = _ensure_dir(OUT_DIR)
    exp0_dir = _ensure_dir(out_root / "exp0")
    exp1_dir = _ensure_dir(out_root / "exp1")
    exp2_dir = _ensure_dir(out_root / "exp2")
    exp3_dir = _ensure_dir(out_root / "exp3")
    figs_dir = _ensure_dir(out_root / "figures")
    runs_dir = _ensure_dir(out_root / "runs")

    if RUN_PHASE in {"all", "exp0_exp1"}:
        state0 = _prepare_state(BASE_SCALE)
        anchor_tile_index = _pick_anchor_tile_index(state0)
        print("[exp0] prepared base state", flush=True)
        exp0_rows = _run_tile_diagnostics(state0)
        for r in exp0_rows:
            r["experiment"] = "exp0"
            r["arm"] = "coarse_initial"
        _write_tile_metrics_csv(exp0_dir / "tile_metrics_exp0.csv", [_slim_row(r) for r in exp0_rows])
        _render_exp0_tiles(exp0_rows, exp0_dir / "tile_zoom_storyboard.png")
        _plot_dx_vs_x(exp0_rows, exp0_dir / "dx_vs_x_plot.png", title="Experiment 0: best local dx* vs x")
        _plot_per_column(exp0_rows, exp0_dir / "per_column_residual_plot.png", title="Experiment 0: tile residual summary")
        exp0_summary = _aggregate_scale_summary(exp0_rows)
        initial_overlay = _compose_full_overlay(state0)
        exp0_rows = [_slim_row(r) for r in exp0_rows]
        del state0
        gc.collect()
        print("[exp0] done", flush=True)

        sweep_rows: list[dict[str, Any]] = []
        sweep_tile_rows: list[dict[str, Any]] = []
        for scale in SCALE_SWEEP:
            print(f"[exp1] scale={scale:.3f}", flush=True)
            state = _prepare_state(float(scale))
            rows = _run_tile_diagnostics(state)
            for rec in rows:
                rec = _slim_row(rec)
                rec["experiment"] = "exp1"
                rec["arm"] = f"scale_{scale:.3f}"
                sweep_tile_rows.append(rec)
            agg = _aggregate_scale_summary(rows)
            agg["scale"] = float(scale)
            agg["cols"] = sorted({int(r["col"]) for r in rows})
            sweep_rows.append(agg)
        _write_tile_metrics_csv(exp1_dir / "tile_metrics_scale_sweep_long.csv", sweep_tile_rows)
        _write_scale_summary_csv(exp1_dir / "scale_sweep_summary.csv", sweep_rows)
        if plt is not None:
            fig, ax = plt.subplots(figsize=(9, 5), dpi=160)
            ax.plot([r["scale"] for r in sweep_rows], [r["mean_tile_cc"] for r in sweep_rows], marker="o", label="mean tile CC")
            ax2 = ax.twinx()
            ax2.plot([r["scale"] for r in sweep_rows], [r["rightmost_mean_dx"] for r in sweep_rows], marker="s", color="tab:red", label="rightmost mean dx")
            ax.set_xlabel("scale")
            ax.set_ylabel("mean tile CC")
            ax2.set_ylabel("rightmost mean dx (px)")
            ax.grid(True, alpha=0.2)
            fig.tight_layout()
            fig.savefig(exp1_dir / "scale_sweep_overview.png")
            plt.close(fig)

        best_overall = max(sweep_rows, key=lambda r: float(r["mean_tile_cc"]))
        best_right = min(sweep_rows, key=lambda r: abs(float(r["rightmost_mean_dx"])))
        best_scale = float(best_overall["scale"])
        gc.collect()
        print(f"[exp1] done best_overall={best_scale:.3f} best_right={float(best_right['scale']):.3f}", flush=True)
        if RUN_PHASE == "exp0_exp1":
            partial_lines = [
                "# 2501_60 Confocal Grid Geometry Diagnostic",
                "",
                "## Experiment 0",
                f"- mean tile CC: {exp0_summary['mean_tile_cc']:.4f}",
                f"- mean |dx*|: {exp0_summary['mean_abs_dx']:.2f} px",
                f"- dx vs x slope: {exp0_summary['slope_dx_vs_x']:.5f}",
                f"- corr(dx, x): {exp0_summary['corr_dx_x']:.4f}",
                "",
                "## Experiment 1",
                f"- best overall scale (max mean tile CC): {best_overall['scale']:.3f}",
                f"- best overall mean tile CC: {best_overall['mean_tile_cc']:.4f}",
                f"- best right-flatten scale (min |rightmost mean dx|): {best_right['scale']:.3f}",
                f"- rightmost mean dx at best-right scale: {best_right['rightmost_mean_dx']:.2f} px",
            ]
            _write_summary(out_root / "summary.md", partial_lines)
            print(f"OK: phase exp0_exp1 written to {out_root}", flush=True)
            return
    else:
        sweep_rows = _read_scale_summary_csv(exp1_dir / "scale_sweep_summary.csv")
        if not sweep_rows:
            raise FileNotFoundError(f"Missing scale sweep summary: {exp1_dir / 'scale_sweep_summary.csv'}")
        best_overall = max(sweep_rows, key=lambda r: float(r["mean_tile_cc"]))
        best_right = min(sweep_rows, key=lambda r: abs(float(r["rightmost_mean_dx"])))
        best_scale = float(best_overall["scale"])
        exp0_summary = _aggregate_scale_summary(list(csv.DictReader((exp0_dir / "tile_metrics_exp0.csv").open("r", encoding="utf-8"))))
        state0 = _prepare_state(BASE_SCALE)
        anchor_tile_index = _pick_anchor_tile_index(state0)
        initial_overlay = _compose_full_overlay(state0)
        del state0
        gc.collect()
        print(f"[resume] best_overall={best_scale:.3f} best_right={float(best_right['scale']):.3f}", flush=True)

    state_best = _prepare_state(best_scale)
    base_best_rows = _run_tile_diagnostics(state_best)
    for rec in base_best_rows:
        rec["experiment"] = "exp2"
        rec["arm"] = "translation_only"
    best_scale_overlay = _compose_full_overlay(state_best)

    ants_bin = find_ants_bin()
    if ants_bin is None:
        raise RuntimeError("Could not locate local ANTs binaries for experiment 2")

    sim_manifest = run_confocal_rigid_registration(
        _make_similarity_cfg(state_best, ants_bin=ants_bin, out_root=runs_dir, refine_model="similarity", scale=best_scale)
    )
    print("[exp2] similarity done", flush=True)
    aff_manifest = run_confocal_rigid_registration(
        _make_similarity_cfg(state_best, ants_bin=ants_bin, out_root=runs_dir, refine_model="affine", scale=best_scale)
    )
    print("[exp2] affine done", flush=True)
    sim_rows = _evaluate_run_manifest(sim_manifest, state_best)
    aff_rows = _evaluate_run_manifest(aff_manifest, state_best)
    for rec in sim_rows:
        rec["experiment"] = "exp2"
        rec["arm"] = "anchor_locked_similarity"
    for rec in aff_rows:
        rec["experiment"] = "exp2"
        rec["arm"] = "anchor_locked_affine"
    _write_tile_metrics_csv(
        exp2_dir / "tile_metrics_model_compare.csv",
        [_slim_row(r) for r in (base_best_rows + sim_rows + aff_rows)],
    )
    base_best_rows = [_slim_row(r) for r in base_best_rows]
    sim_rows = [_slim_row(r) for r in sim_rows]
    aff_rows = [_slim_row(r) for r in aff_rows]
    gc.collect()

    dx_by_col, dy_by_col = _columnwise_offsets(base_best_rows, anchor_tile_index=anchor_tile_index)
    col_dx_by_tile = {int(t.tile_index): float(dx_by_col[int(t.col)]) for t in state_best["tile_defs"]}
    col_dy_by_tile = {int(t.tile_index): float(dy_by_col[int(t.col)]) for t in state_best["tile_defs"]}
    col_rows = _run_tile_diagnostics(state_best, extra_dx_by_tile=col_dx_by_tile, extra_dy_by_tile=col_dy_by_tile)
    for rec in col_rows:
        rec["experiment"] = "exp3"
        rec["arm"] = "columnwise_correction"
    sm_dx_by_tile, sm_dy_by_tile = _smooth_tile_offsets(base_best_rows, anchor_tile_index=anchor_tile_index)
    smooth_rows = _run_tile_diagnostics(state_best, extra_dx_by_tile=sm_dx_by_tile, extra_dy_by_tile=sm_dy_by_tile)
    for rec in smooth_rows:
        rec["experiment"] = "exp3"
        rec["arm"] = "tilewise_smooth_correction"
    _write_tile_metrics_csv(
        exp3_dir / "tile_metrics_piecewise_compare.csv",
        [_slim_row(r) for r in (col_rows + smooth_rows)],
    )
    col_rows = [_slim_row(r) for r in col_rows]
    smooth_rows = [_slim_row(r) for r in smooth_rows]
    gc.collect()
    print("[exp3] piecewise diagnostics done", flush=True)

    overlays = [
        ("Initial coarse", initial_overlay),
        (f"Best-scale coarse ({best_scale:.3f})", best_scale_overlay),
        ("Anchor+Similarity", _overlay_from_manifest_full(sim_manifest, state_best)),
        ("Anchor+Affine", _overlay_from_manifest_full(aff_manifest, state_best)),
        ("Column-wise correction", _compose_full_overlay(state_best, extra_dx_by_tile=col_dx_by_tile, extra_dy_by_tile=col_dy_by_tile)),
        ("Tile-wise smooth correction", _compose_full_overlay(state_best, extra_dx_by_tile=sm_dx_by_tile, extra_dy_by_tile=sm_dy_by_tile)),
    ]
    _render_full_overlay_grid(overlays, figs_dir / "full_overlay_contact_sheet.png")

    comparison_rows = []
    for label, rows in [
        ("translation_only", base_best_rows),
        ("anchor_locked_similarity", sim_rows),
        ("anchor_locked_affine", aff_rows),
        ("columnwise_correction", col_rows),
        ("tilewise_smooth_correction", smooth_rows),
    ]:
        agg = _aggregate_scale_summary(rows)
        comparison_rows.append((label, agg))
    with (exp2_dir / "best_methods_comparison.json").open("w", encoding="utf-8") as f:
        json.dump({k: v for k, v in comparison_rows}, f, indent=2)
    if plt is not None:
        labels = [k for k, _ in comparison_rows]
        mean_cc = [float(v["mean_tile_cc"]) for _, v in comparison_rows]
        mean_abs_dx = [float(v["mean_abs_dx"]) for _, v in comparison_rows]
        fig, ax = plt.subplots(1, 2, figsize=(12, 4.5), dpi=160)
        ax[0].bar(labels, mean_cc)
        ax[0].set_title("Mean tile CC")
        ax[0].tick_params(axis="x", rotation=20)
        ax[1].bar(labels, mean_abs_dx, color="tab:red")
        ax[1].set_title("Mean |dx*|")
        ax[1].tick_params(axis="x", rotation=20)
        fig.tight_layout()
        fig.savefig(figs_dir / "best_methods_comparison.png")
        plt.close(fig)

    lines = [
        "# 2501_60 Confocal Grid Geometry Diagnostic",
        "",
        "## Fixed initial tracker state",
        f"- tx = {BASE_TX}",
        f"- ty = {BASE_TY}",
        f"- angle = {BASE_ANGLE}",
        f"- scale = {BASE_SCALE}",
        f"- flip_ud = {FLIP_UD}",
        f"- A1 = {ANCHOR_A1}",
        f"- B1_raw = {ANCHOR_B1_RAW}",
        "",
        "## Experiment 0",
        f"- mean tile CC: {exp0_summary['mean_tile_cc']:.4f}",
        f"- mean |dx*|: {exp0_summary['mean_abs_dx']:.2f} px",
        f"- dx vs x slope: {exp0_summary['slope_dx_vs_x']:.5f}",
        f"- corr(dx, x): {exp0_summary['corr_dx_x']:.4f}",
        "",
        "## Experiment 1",
        f"- best overall scale (max mean tile CC): {best_overall['scale']:.3f}",
        f"- best overall mean tile CC: {best_overall['mean_tile_cc']:.4f}",
        f"- best right-flatten scale (min |rightmost mean dx|): {best_right['scale']:.3f}",
        f"- rightmost mean dx at best-right scale: {best_right['rightmost_mean_dx']:.2f} px",
        "",
        "## Experiment 2",
    ]
    for label, agg in comparison_rows[:3]:
        lines.append(
            f"- {label}: meanCC={agg['mean_tile_cc']:.4f} mean|dx*|={agg['mean_abs_dx']:.2f} slope={agg['slope_dx_vs_x']:.5f}"
        )
    lines.extend(
        [
            "",
            "## Experiment 3",
            f"- columnwise_correction: meanCC={_aggregate_scale_summary(col_rows)['mean_tile_cc']:.4f} mean|dx*|={_aggregate_scale_summary(col_rows)['mean_abs_dx']:.2f}",
            f"- tilewise_smooth_correction: meanCC={_aggregate_scale_summary(smooth_rows)['mean_tile_cc']:.4f} mean|dx*|={_aggregate_scale_summary(smooth_rows)['mean_abs_dx']:.2f}",
            "",
            "## Outputs",
            f"- exp0 tile storyboard: `{(exp0_dir / 'tile_zoom_storyboard.png')}`",
            f"- exp1 scale summary: `{(exp1_dir / 'scale_sweep_summary.csv')}`",
            f"- exp2 ANTs runs: `{runs_dir}`",
            f"- figures: `{figs_dir}`",
        ]
    )
    _write_summary(out_root / "summary.md", lines)
    print(f"OK: results written to {out_root}", flush=True)


if __name__ == "__main__":
    main()

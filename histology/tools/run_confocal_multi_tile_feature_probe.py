from __future__ import annotations

import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
GUI_MVP_ROOT = REPO_ROOT / "registration_pipeline" / "histology" / "gui_mvp"
TOOLS_ROOT = REPO_ROOT / "registration_pipeline" / "histology" / "tools"
for p in (str(GUI_MVP_ROOT), str(TOOLS_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from hitl_gui.application.pair_registration import compute_registration_metrics, gray_preview_panel, overlay_preview  # noqa: E402
import run_confocal_grid_geometry_diagnostic as geom  # noqa: E402


OUT_ROOT = Path("/mnt/c/Users/Siqi/Desktop/REVIEW/20260408_confocal_multi_tile_feature_probe_2501_60")
SEARCH_RADIUS_SMALL = 8
RELAXED_MIN_AREA = 6
ANCHOR_TILE_INDEX = 12


@dataclass
class TileContext:
    tile_index: int
    row: int
    col: int
    center_scaled_xy: tuple[float, float]
    distance_from_anchor_px: float
    moving_native_u8: np.ndarray
    moving_signal_mask: np.ndarray
    moving_footprint_mask: np.ndarray
    fixed_native_u8: np.ndarray
    fixed_mask: np.ndarray


@dataclass
class MethodResult:
    name: str
    description: str
    tile_rows: list[dict[str, Any]]
    summary: dict[str, Any]


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _masked_percentile_normalize_u8(image_u8: np.ndarray, mask_u8: np.ndarray, lo_pct: float = 1.0, hi_pct: float = 99.0) -> np.ndarray:
    arr = np.asarray(image_u8, dtype=np.float32)
    valid = np.asarray(mask_u8) > 0
    vals = arr[valid]
    if vals.size == 0:
        out = np.full_like(image_u8, 255, dtype=np.uint8)
        return out
    lo = float(np.percentile(vals, lo_pct))
    hi = float(np.percentile(vals, hi_pct))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(vals.min())
        hi = float(vals.max())
    if hi <= lo:
        out = np.full_like(image_u8, 255, dtype=np.uint8)
        return out
    scaled = np.clip((arr - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    out = np.clip(np.round(scaled * 255.0), 0, 255).astype(np.uint8)
    out[~valid] = 255
    return out


def _apply_clahe_u8(image_u8: np.ndarray, mask_u8: np.ndarray, *, clip_limit: float = 2.5, tile_grid: int = 8) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=(tile_grid, tile_grid))
    enhanced = clahe.apply(np.asarray(image_u8, dtype=np.uint8))
    out = np.full_like(enhanced, 255, dtype=np.uint8)
    inside = np.asarray(mask_u8) > 0
    out[inside] = enhanced[inside]
    return out


def _gaussian_blur_u8(image_u8: np.ndarray, mask_u8: np.ndarray, *, sigma: float) -> np.ndarray:
    blurred = cv2.GaussianBlur(np.asarray(image_u8, dtype=np.uint8), (0, 0), sigmaX=float(sigma), sigmaY=float(sigma))
    out = np.full_like(blurred, 255, dtype=np.uint8)
    inside = np.asarray(mask_u8) > 0
    out[inside] = blurred[inside]
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
    min_area: int = RELAXED_MIN_AREA,
) -> np.ndarray:
    inside = np.asarray(mask_u8) > 0
    vals = image_u8[inside]
    out = np.full_like(image_u8, 255, dtype=np.uint8)
    if vals.size < 64:
        return out
    thr = float(np.percentile(vals, keep_quantile))
    fiber = np.zeros_like(image_u8, dtype=np.uint8)
    fiber[inside & (image_u8 <= thr)] = 255
    num, labels, stats, _ = cv2.connectedComponentsWithStats(fiber, connectivity=8)
    keep = np.zeros_like(fiber, dtype=np.uint8)
    for idx in range(1, num):
        if int(stats[idx, cv2.CC_STAT_AREA]) >= int(min_area):
            keep[labels == idx] = 255
    out[keep > 0] = 0
    return out


def _distance_transform_dark_u8(image_u8: np.ndarray, mask_u8: np.ndarray, *, keep_quantile: float = 60.0) -> np.ndarray:
    binary = _relaxed_binary_dark_u8(image_u8, mask_u8, keep_quantile=keep_quantile, min_area=RELAXED_MIN_AREA)
    fiber = (binary == 0).astype(np.uint8)
    dt = cv2.distanceTransform(fiber, cv2.DIST_L2, 3)
    inside = np.asarray(mask_u8) > 0
    vals = dt[inside]
    out = np.full_like(image_u8, 255, dtype=np.uint8)
    if vals.size == 0 or float(vals.max()) <= 1e-6:
        return out
    dt_norm = np.clip(dt / max(float(vals.max()), 1e-6), 0.0, 1.0)
    out[inside] = np.clip(np.round((1.0 - dt_norm[inside]) * 255.0), 0, 255).astype(np.uint8)
    return out


def _panel(image_u8: np.ndarray) -> np.ndarray:
    return gray_preview_panel(np.asarray(image_u8, dtype=np.float32) / 255.0)


def _square_panel(panel: np.ndarray, side: int = 180) -> np.ndarray:
    return geom._square_panel(panel, side=side)


def _shift_patch_within_canvas(gray_patch: np.ndarray, mask_patch: np.ndarray, dx: int, dy: int, *, fill_value: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    h, w = gray_patch.shape[:2]
    out = np.full((h, w), float(fill_value), dtype=np.float32)
    out_mask = np.zeros((h, w), dtype=np.float32)
    src_y0 = max(0, -dy)
    src_x0 = max(0, -dx)
    dst_y0 = max(0, dy)
    dst_x0 = max(0, dx)
    span_h = min(h - src_y0, h - dst_y0)
    span_w = min(w - src_x0, w - dst_x0)
    if span_h <= 0 or span_w <= 0:
        return out, out_mask
    src_y1 = src_y0 + span_h
    src_x1 = src_x0 + span_w
    dst_y1 = dst_y0 + span_h
    dst_x1 = dst_x0 + span_w
    patch_crop = gray_patch[src_y0:src_y1, src_x0:src_x1]
    mask_crop = mask_patch[src_y0:src_y1, src_x0:src_x1]
    out[dst_y0:dst_y1, dst_x0:dst_x1] = np.where(mask_crop > 0, patch_crop, out[dst_y0:dst_y1, dst_x0:dst_x1])
    out_mask[dst_y0:dst_y1, dst_x0:dst_x1] = mask_crop
    return out, out_mask


def _metrics_from_pair(fixed_u8: np.ndarray, moving_u8: np.ndarray, fixed_mask: np.ndarray, moving_mask: np.ndarray) -> dict[str, float]:
    metrics, _ = compute_registration_metrics(
        fixed_u8.astype(np.float32) / 255.0,
        moving_u8.astype(np.float32) / 255.0,
        fixed_mask.astype(np.float32),
        moving_mask.astype(np.float32),
    )
    return metrics


def _best_shift_cc(
    fixed_u8: np.ndarray,
    moving_u8: np.ndarray,
    fixed_mask: np.ndarray,
    moving_mask: np.ndarray,
    *,
    radius: int,
) -> dict[str, float]:
    best = {
        "dx": 0,
        "dy": 0,
        "cc": -1e9,
        "mi": float("nan"),
        "dice": float("nan"),
        "hd95": float("nan"),
    }
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            shifted_gray, shifted_mask = _shift_patch_within_canvas(
                moving_u8.astype(np.float32) / 255.0,
                moving_mask.astype(np.float32),
                dx,
                dy,
                fill_value=1.0,
            )
            metrics, _ = compute_registration_metrics(
                fixed_u8.astype(np.float32) / 255.0,
                shifted_gray,
                fixed_mask.astype(np.float32),
                shifted_mask.astype(np.float32),
            )
            cc = float(metrics.get("cc", float("nan")))
            if np.isfinite(cc) and cc > float(best["cc"]):
                best = {
                    "dx": int(dx),
                    "dy": int(dy),
                    "cc": cc,
                    "mi": float(metrics.get("mi", float("nan"))),
                    "dice": float(metrics.get("dice", float("nan"))),
                    "hd95": float(metrics.get("hd95_px", float("nan"))),
                }
    return best


def _compose_panels(
    *,
    fixed_native_u8: np.ndarray,
    moving_native_u8: np.ndarray,
    fixed_proc_u8: np.ndarray,
    moving_proc_u8: np.ndarray,
    fixed_mask: np.ndarray,
    moving_signal_mask: np.ndarray,
    moving_footprint_mask: np.ndarray,
    best_small_native: dict[str, float],
    best_small_proc: dict[str, float],
) -> dict[str, np.ndarray]:
    native_overlay_current = overlay_preview(
        fixed_native_u8.astype(np.float32) / 255.0,
        moving_native_u8.astype(np.float32) / 255.0,
        fixed_mask,
        moving_footprint_mask,
    )
    native_shifted_gray, native_shifted_mask = _shift_patch_within_canvas(
        moving_native_u8.astype(np.float32) / 255.0,
        moving_footprint_mask.astype(np.float32),
        int(best_small_native["dx"]),
        int(best_small_native["dy"]),
        fill_value=1.0,
    )
    native_overlay_shifted = overlay_preview(
        fixed_native_u8.astype(np.float32) / 255.0,
        native_shifted_gray,
        fixed_mask,
        np.where(native_shifted_mask > 0, 1.0, 0.0).astype(np.float32),
    )
    proc_overlay_current = overlay_preview(
        fixed_proc_u8.astype(np.float32) / 255.0,
        moving_proc_u8.astype(np.float32) / 255.0,
        fixed_mask,
        moving_footprint_mask,
    )
    proc_shifted_gray, proc_shifted_mask = _shift_patch_within_canvas(
        moving_proc_u8.astype(np.float32) / 255.0,
        moving_footprint_mask.astype(np.float32),
        int(best_small_proc["dx"]),
        int(best_small_proc["dy"]),
        fill_value=1.0,
    )
    proc_overlay_shifted = overlay_preview(
        fixed_proc_u8.astype(np.float32) / 255.0,
        proc_shifted_gray,
        fixed_mask,
        np.where(proc_shifted_mask > 0, 1.0, 0.0).astype(np.float32),
    )
    return {
        "confocal_native": _square_panel(_panel(moving_native_u8)),
        "myelin_native": _square_panel(_panel(fixed_native_u8)),
        "confocal_proc": _square_panel(_panel(moving_proc_u8)),
        "myelin_proc": _square_panel(_panel(fixed_proc_u8)),
        "native_overlay_current": _square_panel(native_overlay_current),
        "native_overlay_shifted": _square_panel(native_overlay_shifted),
        "proc_overlay_current": _square_panel(proc_overlay_current),
        "proc_overlay_shifted": _square_panel(proc_overlay_shifted),
    }


def _method_variants() -> list[tuple[str, str, Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]]]:
    def paired_percentile_raw(fixed_u8, moving_u8, fixed_mask, moving_mask):
        return _masked_percentile_normalize_u8(fixed_u8, fixed_mask), _masked_percentile_normalize_u8(moving_u8, moving_mask)

    def paired_percentile_blur4(fixed_u8, moving_u8, fixed_mask, moving_mask):
        return (
            _gaussian_blur_u8(_masked_percentile_normalize_u8(fixed_u8, fixed_mask), fixed_mask, sigma=4.0),
            _gaussian_blur_u8(_masked_percentile_normalize_u8(moving_u8, moving_mask), moving_mask, sigma=4.0),
        )

    def paired_percentile_blur6(fixed_u8, moving_u8, fixed_mask, moving_mask):
        return (
            _gaussian_blur_u8(_masked_percentile_normalize_u8(fixed_u8, fixed_mask), fixed_mask, sigma=6.0),
            _gaussian_blur_u8(_masked_percentile_normalize_u8(moving_u8, moving_mask), moving_mask, sigma=6.0),
        )

    def paired_percentile_clahe_blur3(fixed_u8, moving_u8, fixed_mask, moving_mask):
        return (
            _gaussian_blur_u8(_apply_clahe_u8(_masked_percentile_normalize_u8(fixed_u8, fixed_mask), fixed_mask, clip_limit=3.0, tile_grid=8), fixed_mask, sigma=3.0),
            _gaussian_blur_u8(_apply_clahe_u8(_masked_percentile_normalize_u8(moving_u8, moving_mask), moving_mask, clip_limit=3.0, tile_grid=8), moving_mask, sigma=3.0),
        )

    def moving_percentile_hist_blur4(fixed_u8, moving_u8, fixed_mask, moving_mask):
        fixed_proc = _gaussian_blur_u8(_masked_percentile_normalize_u8(fixed_u8, fixed_mask), fixed_mask, sigma=4.0)
        moving_pct = _masked_percentile_normalize_u8(moving_u8, moving_mask)
        moving_hist = geom._masked_histogram_match_u8(moving_pct, moving_mask, fixed_proc, fixed_mask)
        moving_proc = _gaussian_blur_u8(moving_hist, moving_mask, sigma=4.0)
        return fixed_proc, moving_proc

    def moving_aggressive_clahe_hist_blur3(fixed_u8, moving_u8, fixed_mask, moving_mask):
        fixed_proc = _gaussian_blur_u8(_masked_percentile_normalize_u8(fixed_u8, fixed_mask), fixed_mask, sigma=3.0)
        moving_pct = _masked_percentile_normalize_u8(moving_u8, moving_mask, lo_pct=0.5, hi_pct=99.5)
        moving_clahe = _apply_clahe_u8(moving_pct, moving_mask, clip_limit=5.0, tile_grid=8)
        moving_hist = geom._masked_histogram_match_u8(moving_clahe, moving_mask, fixed_proc, fixed_mask)
        moving_proc = _gaussian_blur_u8(moving_hist, moving_mask, sigma=3.0)
        return fixed_proc, moving_proc

    def moving_gamma_clahe_hist_blur4(fixed_u8, moving_u8, fixed_mask, moving_mask):
        fixed_proc = _gaussian_blur_u8(_masked_percentile_normalize_u8(fixed_u8, fixed_mask), fixed_mask, sigma=4.0)
        moving_pct = _masked_percentile_normalize_u8(moving_u8, moving_mask, lo_pct=0.5, hi_pct=99.5)
        moving_gamma = _gamma_u8(moving_pct, moving_mask, gamma=1.8)
        moving_clahe = _apply_clahe_u8(moving_gamma, moving_mask, clip_limit=6.0, tile_grid=8)
        moving_hist = geom._masked_histogram_match_u8(moving_clahe, moving_mask, fixed_proc, fixed_mask)
        moving_proc = _gaussian_blur_u8(moving_hist, moving_mask, sigma=4.0)
        return fixed_proc, moving_proc

    def paired_relaxed_binary_q60(fixed_u8, moving_u8, fixed_mask, moving_mask):
        return (
            _relaxed_binary_dark_u8(_masked_percentile_normalize_u8(fixed_u8, fixed_mask), fixed_mask, keep_quantile=60.0),
            _relaxed_binary_dark_u8(_masked_percentile_normalize_u8(moving_u8, moving_mask), moving_mask, keep_quantile=60.0),
        )

    def paired_relaxed_binary_q70(fixed_u8, moving_u8, fixed_mask, moving_mask):
        return (
            _relaxed_binary_dark_u8(_masked_percentile_normalize_u8(fixed_u8, fixed_mask), fixed_mask, keep_quantile=70.0),
            _relaxed_binary_dark_u8(_masked_percentile_normalize_u8(moving_u8, moving_mask), moving_mask, keep_quantile=70.0),
        )

    def paired_relaxed_binary_dt_q60(fixed_u8, moving_u8, fixed_mask, moving_mask):
        return (
            _distance_transform_dark_u8(_masked_percentile_normalize_u8(fixed_u8, fixed_mask), fixed_mask, keep_quantile=60.0),
            _distance_transform_dark_u8(_masked_percentile_normalize_u8(moving_u8, moving_mask), moving_mask, keep_quantile=60.0),
        )

    return [
        ("paired_percentile_raw", "Paired percentile normalization only", paired_percentile_raw),
        ("paired_percentile_blur4", "Paired percentile + blur sigma=4", paired_percentile_blur4),
        ("paired_percentile_blur6", "Paired percentile + blur sigma=6", paired_percentile_blur6),
        ("paired_percentile_clahe_blur3", "Paired percentile + CLAHE + blur sigma=3", paired_percentile_clahe_blur3),
        ("moving_percentile_hist_blur4", "Fixed percentile+blur4; moving percentile + histmatch + blur4", moving_percentile_hist_blur4),
        ("moving_aggressive_clahe_hist_blur3", "Fixed percentile+blur3; moving aggressive CLAHE + histmatch + blur3", moving_aggressive_clahe_hist_blur3),
        ("moving_gamma_clahe_hist_blur4", "Fixed percentile+blur4; moving gamma + CLAHE + histmatch + blur4", moving_gamma_clahe_hist_blur4),
        ("paired_relaxed_binary_q60", "Paired relaxed binary q60", paired_relaxed_binary_q60),
        ("paired_relaxed_binary_q70", "Paired relaxed binary q70", paired_relaxed_binary_q70),
        ("paired_relaxed_binary_dt_q60", "Paired relaxed binary q60 + distance transform", paired_relaxed_binary_dt_q60),
    ]


def _build_tile_contexts() -> tuple[list[TileContext], dict[str, Any]]:
    inputs = geom._load_inputs()
    fixed_bundle = inputs["fixed_bundle"]
    fixed_mask_full = (fixed_bundle.labels == 1).astype(np.float32)
    if not np.any(fixed_mask_full > 0):
        fixed_mask_full = (fixed_bundle.labels > 0).astype(np.float32)
    fixed_native_full_u8 = cv2.cvtColor(fixed_bundle.rgb, cv2.COLOR_RGB2GRAY)
    moving_native_scaled_u8 = np.asarray(inputs["scaled_projection_u8"], dtype=np.uint8)
    moving_signal_mask_u8 = np.asarray(inputs["scaled_signal_mask_u8"], dtype=np.uint8)
    moving_display_full_u8 = geom._invert_confocal_u8(moving_native_scaled_u8)
    tile_defs = geom._build_tile_defs(
        inputs["projection_bundle"].stitch_info,
        raw_shape_hw=inputs["raw_projection_u8"].shape[:2],
        scaled_shape_hw=inputs["scaled_projection_u8"].shape[:2],
    )
    full_mat, anchor_info = geom._manual_affine_for_scale(
        moving_native_scaled_u8.shape[:2],
        fixed_native_full_u8.shape[:2],
        scale=float(geom.MANUAL_STATE["scale"]),
    )
    _rows_display, display_tile_warps = geom._collect_tile_results(
        label="display_inverted",
        moving_reg_projection_u8=moving_display_full_u8,
        moving_signal_mask_u8=moving_signal_mask_u8,
        fixed_gray_full=fixed_native_full_u8,
        fixed_mask_full=fixed_mask_full,
        tile_defs=tile_defs,
        full_mat=full_mat,
    )
    anchor_tw = next(tw for tw in display_tile_warps if int(tw.tile.tile_index) == ANCHOR_TILE_INDEX)
    anchor_center = np.asarray(anchor_tw.tile.center_scaled_xy, dtype=np.float32)
    contexts: list[TileContext] = []
    for tw in display_tile_warps:
        local_h, local_w = tw.source_gray_patch.shape[:2]
        inv = cv2.invertAffineTransform(np.asarray(tw.tile_to_fixed_mat, dtype=np.float32))
        fixed_native_patch = cv2.warpAffine(
            fixed_native_full_u8,
            inv,
            (int(local_w), int(local_h)),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=255,
        )
        fixed_mask_patch_u8 = cv2.warpAffine(
            np.where(fixed_mask_full > 0, 255, 0).astype(np.uint8),
            inv,
            (int(local_w), int(local_h)),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        center = np.asarray(tw.tile.center_scaled_xy, dtype=np.float32)
        contexts.append(
            TileContext(
                tile_index=int(tw.tile.tile_index),
                row=int(tw.tile.row),
                col=int(tw.tile.col),
                center_scaled_xy=(float(center[0]), float(center[1])),
                distance_from_anchor_px=float(np.linalg.norm(center - anchor_center)),
                moving_native_u8=np.clip(tw.source_gray_patch * 255.0, 0, 255).astype(np.uint8),
                moving_signal_mask=(tw.source_mask_patch > 0).astype(np.float32),
                moving_footprint_mask=np.ones((int(local_h), int(local_w)), dtype=np.float32),
                fixed_native_u8=fixed_native_patch.astype(np.uint8),
                fixed_mask=(fixed_mask_patch_u8 > 0).astype(np.float32),
            )
        )
    return contexts, {"inputs": inputs, "anchor_info": anchor_info}


def _select_sample_tiles(contexts: list[TileContext]) -> list[int]:
    by_rc = {(ctx.row, ctx.col): ctx.tile_index for ctx in contexts}
    coverages = sorted(contexts, key=lambda c: float(np.mean(c.moving_signal_mask)))
    sample_ids = [
        ANCHOR_TILE_INDEX,
        by_rc.get((2, 5)),
        by_rc.get((1, 0)),
        by_rc.get((1, 3)),
        by_rc.get((0, 0)),
        by_rc.get((0, 5)),
        coverages[0].tile_index if coverages else None,
        coverages[-1].tile_index if coverages else None,
    ]
    dedup: list[int] = []
    for tid in sample_ids:
        if tid is not None and tid not in dedup:
            dedup.append(int(tid))
    return dedup


def _run_methods(contexts: list[TileContext]) -> list[MethodResult]:
    results: list[MethodResult] = []
    for name, description, fn in _method_variants():
        tile_rows: list[dict[str, Any]] = []
        for ctx in contexts:
            fixed_proc_u8, moving_proc_u8 = fn(
                ctx.fixed_native_u8,
                ctx.moving_native_u8,
                np.where(ctx.fixed_mask > 0, 255, 0).astype(np.uint8),
                np.where(ctx.moving_signal_mask > 0, 255, 0).astype(np.uint8),
            )
            current_native = _metrics_from_pair(ctx.fixed_native_u8, ctx.moving_native_u8, ctx.fixed_mask, ctx.moving_signal_mask)
            current_proc = _metrics_from_pair(fixed_proc_u8, moving_proc_u8, ctx.fixed_mask, ctx.moving_signal_mask)
            best_small_native = _best_shift_cc(ctx.fixed_native_u8, ctx.moving_native_u8, ctx.fixed_mask, ctx.moving_signal_mask, radius=SEARCH_RADIUS_SMALL)
            best_small_proc = _best_shift_cc(fixed_proc_u8, moving_proc_u8, ctx.fixed_mask, ctx.moving_signal_mask, radius=SEARCH_RADIUS_SMALL)
            panels = _compose_panels(
                fixed_native_u8=ctx.fixed_native_u8,
                moving_native_u8=ctx.moving_native_u8,
                fixed_proc_u8=fixed_proc_u8,
                moving_proc_u8=moving_proc_u8,
                fixed_mask=ctx.fixed_mask,
                moving_signal_mask=ctx.moving_signal_mask,
                moving_footprint_mask=ctx.moving_footprint_mask,
                best_small_native=best_small_native,
                best_small_proc=best_small_proc,
            )
            tile_rows.append(
                {
                    "method": name,
                    "description": description,
                    "tile_index": ctx.tile_index,
                    "row": ctx.row,
                    "col": ctx.col,
                    "center_scaled_x": ctx.center_scaled_xy[0],
                    "center_scaled_y": ctx.center_scaled_xy[1],
                    "distance_from_anchor_px": ctx.distance_from_anchor_px,
                    "signal_coverage": float(np.mean(ctx.moving_signal_mask)),
                    "current_native_cc": float(current_native["cc"]),
                    "current_native_mi": float(current_native["mi"]),
                    "current_proc_cc": float(current_proc["cc"]),
                    "current_proc_mi": float(current_proc["mi"]),
                    "best_small_proc_cc": float(best_small_proc["cc"]),
                    "best_small_proc_dx": int(best_small_proc["dx"]),
                    "best_small_proc_dy": int(best_small_proc["dy"]),
                    "best_small_proc_shift_mag": float(math.hypot(float(best_small_proc["dx"]), float(best_small_proc["dy"]))),
                    "delta_small_cc": float(best_small_proc["cc"] - current_proc["cc"]),
                    "panels": panels,
                }
            )
        current_vals = np.asarray([float(r["current_proc_cc"]) for r in tile_rows], dtype=np.float64)
        small_vals = np.asarray([float(r["best_small_proc_cc"]) for r in tile_rows], dtype=np.float64)
        delta_vals = np.asarray([float(r["delta_small_cc"]) for r in tile_rows], dtype=np.float64)
        shift_vals = np.asarray([float(r["best_small_proc_shift_mag"]) for r in tile_rows], dtype=np.float64)
        summary = {
            "mean_current_proc_cc": float(np.nanmean(current_vals)),
            "median_current_proc_cc": float(np.nanmedian(current_vals)),
            "mean_best_small_proc_cc": float(np.nanmean(small_vals)),
            "mean_delta_small_cc": float(np.nanmean(delta_vals)),
            "median_delta_small_cc": float(np.nanmedian(delta_vals)),
            "mean_small_shift_mag": float(np.nanmean(shift_vals)),
            "tile_count": len(tile_rows),
            "stable_tiles_shift_le_2px": int(sum(1 for r in tile_rows if float(r["best_small_proc_shift_mag"]) <= 2.0)),
            "strong_tiles_current_cc_ge_0_5": int(sum(1 for r in tile_rows if float(r["current_proc_cc"]) >= 0.5)),
        }
        results.append(MethodResult(name=name, description=description, tile_rows=tile_rows, summary=summary))
    return results


def _save_method_sample_sheet(method: MethodResult, sample_tile_ids: list[int], out_path: Path) -> None:
    cols = [
        ("confocal_native", "Confocal native\n(inverted)"),
        ("myelin_native", "Myelin native"),
        ("confocal_proc", "Confocal processed"),
        ("myelin_proc", "Myelin processed"),
        ("native_overlay_current", "Native overlay\ncurrent"),
        ("native_overlay_shifted", f"Native overlay\nshifted ±{SEARCH_RADIUS_SMALL}px"),
        ("proc_overlay_current", "Processed overlay\ncurrent"),
        ("proc_overlay_shifted", f"Processed overlay\nshifted ±{SEARCH_RADIUS_SMALL}px"),
    ]
    rows = [r for r in method.tile_rows if int(r["tile_index"]) in sample_tile_ids]
    rows.sort(key=lambda r: (int(r["row"]), int(r["col"])))
    side = 160
    left = 300
    top = 90
    gap = 10
    row_h = side + 56
    width = left + len(cols) * (side + gap) + gap
    height = top + len(rows) * row_h + gap
    canvas = np.full((height, width, 3), 246, dtype=np.uint8)
    cv2.putText(canvas, method.name, (18, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (25, 25, 25), 2, cv2.LINE_AA)
    cv2.putText(canvas, method.description, (18, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (85, 85, 85), 1, cv2.LINE_AA)
    x = left
    for _key, label in cols:
        for idx, line in enumerate(label.split("\n")):
            cv2.putText(canvas, line, (x + 4, 22 + idx * 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (50, 50, 50), 1, cv2.LINE_AA)
        x += side + gap
    for ridx, row in enumerate(rows):
        y = top + ridx * row_h
        cv2.putText(canvas, f"T{int(row['tile_index']):02d} r{int(row['row'])}c{int(row['col'])}", (18, y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (25, 25, 25), 2, cv2.LINE_AA)
        cv2.putText(
            canvas,
            f"dist={float(row['distance_from_anchor_px']):.0f}px cov={float(row['signal_coverage']):.2f}",
            (18, y + 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (85, 85, 85),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            f"CC {float(row['current_proc_cc']):.3f} -> {float(row['best_small_proc_cc']):.3f} @ ({int(row['best_small_proc_dx'])},{int(row['best_small_proc_dy'])})",
            (18, y + 56),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (85, 85, 85),
            1,
            cv2.LINE_AA,
        )
        x = left
        for key, _label in cols:
            panel = row["panels"][key]
            canvas[y : y + side, x : x + side] = cv2.resize(panel, (side, side), interpolation=cv2.INTER_AREA)
            x += side + gap
    cv2.imwrite(str(out_path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))


def _save_summary_plot(results: list[MethodResult], out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [r.name for r in results]
    current = [float(r.summary["mean_current_proc_cc"]) for r in results]
    best = [float(r.summary["mean_best_small_proc_cc"]) for r in results]
    delta = [float(r.summary["mean_delta_small_cc"]) for r in results]
    y = np.arange(len(results))
    fig, axes = plt.subplots(1, 2, figsize=(12, max(5, len(results) * 0.45)), dpi=160)
    axes[0].barh(y - 0.18, current, height=0.34, label="mean current CC")
    axes[0].barh(y + 0.18, best, height=0.34, label="mean best small-shift CC")
    axes[0].set_yticks(y, labels=names)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("CC")
    axes[0].set_title("Current vs best small-shift CC")
    axes[0].legend()
    axes[1].barh(y, delta, height=0.55, color="#b25dd9")
    axes[1].set_yticks(y, labels=names)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("best_small - current")
    axes[1].set_title("Registration improvement headroom")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _save_tile_winner_csv(results: list[MethodResult], out_path: Path) -> None:
    by_method = {r.name: r for r in results}
    all_tiles = sorted({int(row["tile_index"]) for r in results for row in r.tile_rows})
    rows: list[dict[str, Any]] = []
    for tile_index in all_tiles:
        tile_rows = [row for r in results for row in r.tile_rows if int(row["tile_index"]) == tile_index]
        best_current = max(tile_rows, key=lambda r: float(r["current_proc_cc"]))
        best_delta = max(tile_rows, key=lambda r: float(r["delta_small_cc"]))
        rows.append(
            {
                "tile_index": tile_index,
                "row": int(best_current["row"]),
                "col": int(best_current["col"]),
                "best_current_method": str(best_current["method"]),
                "best_current_cc": float(best_current["current_proc_cc"]),
                "best_delta_method": str(best_delta["method"]),
                "best_delta_cc": float(best_delta["delta_small_cc"]),
            }
        )
    _write_csv(out_path, rows)


def main() -> None:
    out_root = _ensure_dir(OUT_ROOT)
    figs_dir = _ensure_dir(out_root / "figures")
    process_dir = _ensure_dir(out_root / "process")
    methods_dir = _ensure_dir(out_root / "methods")

    contexts, common = _build_tile_contexts()
    sample_tile_ids = _select_sample_tiles(contexts)
    selected_contexts = [ctx for ctx in contexts if int(ctx.tile_index) in sample_tile_ids]
    results = _run_methods(selected_contexts)

    method_rows: list[dict[str, Any]] = []
    tile_rows: list[dict[str, Any]] = []
    for result in results:
        method_rows.append({"method": result.name, "description": result.description, **result.summary})
        method_dir = _ensure_dir(methods_dir / result.name)
        _save_method_sample_sheet(result, sample_tile_ids, method_dir / "sample_tile_qc.png")
        for row in result.tile_rows:
            out = {k: v for k, v in row.items() if k != "panels"}
            tile_rows.append(out)

    _write_csv(process_dir / "method_summary.csv", method_rows)
    _write_csv(process_dir / "tile_method_metrics.csv", tile_rows)
    _save_tile_winner_csv(results, process_dir / "tile_winner_methods.csv")
    _save_summary_plot(results, figs_dir / "method_ranking.png")

    ranked_by_current = sorted(results, key=lambda r: float(r.summary["mean_current_proc_cc"]), reverse=True)
    ranked_by_delta = sorted(results, key=lambda r: float(r.summary["mean_delta_small_cc"]), reverse=True)

    summary_lines = [
        "# Multi-tile Confocal Representation Probe",
        "",
        "Purpose:",
        "- keep current manual geometry fixed",
        "- compare stronger preprocessing/contrast enhancement strategies",
        "- evaluate both current alignment quality and how much local small-shift improvement remains",
        "- examine whether one method generalizes across a representative sample of tiles or only helps special cases",
        "",
        "Geometry / state held fixed:",
        f"- manual scale: {float(geom.MANUAL_STATE['scale']):.3f}",
        f"- manual angle: {float(geom.MANUAL_STATE['angle_deg']):.3f}",
        f"- flip_ud: {bool(geom.MANUAL_STATE['flip_ud'])}",
        f"- anchor tile: T{ANCHOR_TILE_INDEX:02d}",
        "- current local geometry is not changed by this probe",
        "",
        "Metrics:",
        f"- `current_proc_cc`: processed-image CC at current manual alignment",
        f"- `best_small_proc_cc`: best CC after small local search within ±{SEARCH_RADIUS_SMALL}px",
        "- `delta_small_cc`: improvement headroom; smaller is better if current alignment is already good",
        "",
        "Top methods by mean current_proc_cc:",
    ]
    for result in ranked_by_current[:5]:
        s = result.summary
        summary_lines.append(
            f"- `{result.name}`: current={float(s['mean_current_proc_cc']):.4f}, "
            f"best_small={float(s['mean_best_small_proc_cc']):.4f}, "
            f"delta={float(s['mean_delta_small_cc']):.4f}, "
            f"mean shift={float(s['mean_small_shift_mag']):.2f}px"
        )
    summary_lines.extend(["", "Top methods by improvement headroom (large delta means current alignment is still under-exploiting the representation):"])
    for result in ranked_by_delta[:5]:
        s = result.summary
        summary_lines.append(
            f"- `{result.name}`: delta={float(s['mean_delta_small_cc']):.4f}, current={float(s['mean_current_proc_cc']):.4f}"
        )
    summary_lines.extend(
        [
            "",
            "Selected sample tiles for QC:",
            "- " + ", ".join(f"T{int(tid):02d}" for tid in sample_tile_ids),
            "",
            "Important files:",
            f"- sample tile count: {len(selected_contexts)}",
            f"- method summary CSV: `{process_dir / 'method_summary.csv'}`",
            f"- per-tile metrics CSV: `{process_dir / 'tile_method_metrics.csv'}`",
            f"- tile winner CSV: `{process_dir / 'tile_winner_methods.csv'}`",
            f"- ranking figure: `{figs_dir / 'method_ranking.png'}`",
            "- per-method QC: `methods/<method>/sample_tile_qc.png`",
        ]
    )
    (out_root / "summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    _write_json(
        out_root / "run_manifest.json",
        {
            "manual_state": geom.MANUAL_STATE,
            "anchor_pair": geom.ANCHOR_PAIR,
            "anchor_tile_index": ANCHOR_TILE_INDEX,
            "search_radius_small_px": SEARCH_RADIUS_SMALL,
            "sample_tile_ids": sample_tile_ids,
            "method_count": len(results),
            "methods": method_rows,
            "files": {
                "summary_md": str(out_root / "summary.md"),
                "method_summary_csv": str(process_dir / "method_summary.csv"),
                "tile_method_metrics_csv": str(process_dir / "tile_method_metrics.csv"),
                "tile_winner_methods_csv": str(process_dir / "tile_winner_methods.csv"),
                "method_ranking_png": str(figs_dir / "method_ranking.png"),
            },
        },
    )

    print(f"Multi-tile feature probe written to: {out_root}")


if __name__ == "__main__":
    main()

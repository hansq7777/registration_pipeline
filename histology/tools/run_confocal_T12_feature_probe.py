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


OUT_ROOT = Path("/mnt/c/Users/Siqi/Desktop/REVIEW/20260408_T12_r2c0_feature_probe_2501_60")
TILE_INDEX = 12
SEARCH_RADIUS_SMALL = 8
SEARCH_RADIUS_WIDE = 24
MIN_COMPONENT_AREA = 20


@dataclass
class MethodResult:
    name: str
    description: str
    fixed_proc_u8: np.ndarray
    moving_proc_u8: np.ndarray
    current_metrics_native: dict[str, float]
    current_metrics_proc: dict[str, float]
    best_small_native: dict[str, float]
    best_small_proc: dict[str, float]
    best_wide_native: dict[str, float]
    best_wide_proc: dict[str, float]
    panels: dict[str, np.ndarray]


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
        return np.zeros_like(image_u8, dtype=np.uint8)
    lo = float(np.percentile(vals, lo_pct))
    hi = float(np.percentile(vals, hi_pct))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(vals.min())
        hi = float(vals.max())
    if hi <= lo:
        out = np.zeros_like(arr, dtype=np.uint8)
        out[~valid] = 255
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


def _binary_clean_dark_u8(image_u8: np.ndarray, mask_u8: np.ndarray, *, min_area: int = MIN_COMPONENT_AREA) -> np.ndarray:
    inside = np.asarray(mask_u8) > 0
    vals = image_u8[inside]
    if vals.size < 64:
        out = np.full_like(image_u8, 255, dtype=np.uint8)
        return out
    thr, _ = cv2.threshold(vals.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    fiber = np.zeros_like(image_u8, dtype=np.uint8)
    fiber[inside & (image_u8 <= thr)] = 255
    num, labels, stats, _ = cv2.connectedComponentsWithStats(fiber, connectivity=8)
    keep = np.zeros_like(fiber, dtype=np.uint8)
    for idx in range(1, num):
        if int(stats[idx, cv2.CC_STAT_AREA]) >= int(min_area):
            keep[labels == idx] = 255
    out = np.full_like(image_u8, 255, dtype=np.uint8)
    out[keep > 0] = 0
    out[~inside] = 255
    return out


def _distance_transform_dark_u8(image_u8: np.ndarray, mask_u8: np.ndarray, *, min_area: int = MIN_COMPONENT_AREA) -> np.ndarray:
    binary = _binary_clean_dark_u8(image_u8, mask_u8, min_area=min_area)
    fiber = (binary == 0).astype(np.uint8)
    dt = cv2.distanceTransform(fiber, cv2.DIST_L2, 3)
    inside = np.asarray(mask_u8) > 0
    vals = dt[inside]
    if vals.size == 0 or float(vals.max()) <= 1e-6:
        out = np.full_like(image_u8, 255, dtype=np.uint8)
        return out
    dt_norm = np.clip(dt / max(float(vals.max()), 1e-6), 0.0, 1.0)
    out = np.full_like(image_u8, 255, dtype=np.uint8)
    # dark thick fibers, white background
    out[inside] = np.clip(np.round((1.0 - dt_norm[inside]) * 255.0), 0, 255).astype(np.uint8)
    return out


def _panel(image_u8: np.ndarray) -> np.ndarray:
    return gray_preview_panel(np.asarray(image_u8, dtype=np.float32) / 255.0)


def _square_panel(panel: np.ndarray, side: int = 220) -> np.ndarray:
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
    metrics, _timings = compute_registration_metrics(
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


def _load_tile12_context() -> dict[str, Any]:
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
        label="display_native",
        moving_reg_projection_u8=moving_display_full_u8,
        moving_signal_mask_u8=moving_signal_mask_u8,
        fixed_gray_full=fixed_native_full_u8,
        fixed_mask_full=fixed_mask_full,
        tile_defs=tile_defs,
        full_mat=full_mat,
    )
    tile_warp = next(tw for tw in display_tile_warps if int(tw.tile.tile_index) == TILE_INDEX)
    local_h, local_w = tile_warp.source_gray_patch.shape[:2]
    inv = cv2.invertAffineTransform(np.asarray(tile_warp.tile_to_fixed_mat, dtype=np.float32))
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
    fixed_native_patch_u8 = fixed_native_patch.astype(np.uint8)
    fixed_mask_patch = (fixed_mask_patch_u8 > 0).astype(np.float32)
    moving_native_patch_u8 = np.clip(tile_warp.source_gray_patch * 255.0, 0, 255).astype(np.uint8)
    moving_mask_patch = tile_warp.source_mask_patch.astype(np.float32)
    moving_footprint_mask_patch = np.ones((int(local_h), int(local_w)), dtype=np.float32)
    return {
        "inputs": inputs,
        "anchor_info": anchor_info,
        "footprint_bbox_yxyx": list(int(v) for v in tile_warp.warped_full_bbox_yxyx),
        "fixed_native_patch_u8": fixed_native_patch_u8,
        "fixed_mask_patch": fixed_mask_patch,
        "moving_native_patch_u8": moving_native_patch_u8,
        "moving_mask_patch": moving_mask_patch,
        "moving_footprint_mask_patch": moving_footprint_mask_patch,
    }


def _method_variants() -> list[tuple[str, str, Callable[[np.ndarray, np.ndarray], np.ndarray]]]:
    return [
        ("paired_percentile", "Percentile normalization on both sides", lambda im, mk: _masked_percentile_normalize_u8(im, mk)),
        ("paired_percentile_blur2", "Percentile + Gaussian blur sigma=2 on both sides", lambda im, mk: _gaussian_blur_u8(_masked_percentile_normalize_u8(im, mk), mk, sigma=2.0)),
        ("paired_percentile_blur4", "Percentile + Gaussian blur sigma=4 on both sides", lambda im, mk: _gaussian_blur_u8(_masked_percentile_normalize_u8(im, mk), mk, sigma=4.0)),
        ("paired_percentile_clahe_blur2", "Percentile + CLAHE + blur sigma=2 on both sides", lambda im, mk: _gaussian_blur_u8(_apply_clahe_u8(_masked_percentile_normalize_u8(im, mk), mk, clip_limit=2.5, tile_grid=8), mk, sigma=2.0)),
        ("paired_binary_clean", "Percentile + Otsu dark-fiber binary + small-component cleanup on both sides", lambda im, mk: _binary_clean_dark_u8(_masked_percentile_normalize_u8(im, mk), mk, min_area=MIN_COMPONENT_AREA)),
        ("paired_distance_transform", "Percentile + cleaned binary + distance-transform emphasis on both sides", lambda im, mk: _distance_transform_dark_u8(_masked_percentile_normalize_u8(im, mk), mk, min_area=MIN_COMPONENT_AREA)),
    ]


def _compose_method_panels(
    *,
    fixed_native_u8: np.ndarray,
    moving_native_u8: np.ndarray,
    fixed_proc_u8: np.ndarray,
    moving_proc_u8: np.ndarray,
    fixed_mask: np.ndarray,
    moving_mask: np.ndarray,
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
        moving_mask.astype(np.float32),
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
        moving_mask.astype(np.float32),
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


def _run_methods(ctx: dict[str, Any]) -> list[MethodResult]:
    fixed_native_u8 = ctx["fixed_native_patch_u8"]
    moving_native_u8 = ctx["moving_native_patch_u8"]
    fixed_mask = ctx["fixed_mask_patch"]
    moving_mask = ctx["moving_mask_patch"]
    moving_footprint_mask = ctx["moving_footprint_mask_patch"]

    results: list[MethodResult] = []
    current_native_metrics = _metrics_from_pair(fixed_native_u8, moving_native_u8, fixed_mask, moving_mask)
    best_small_native = _best_shift_cc(fixed_native_u8, moving_native_u8, fixed_mask, moving_mask, radius=SEARCH_RADIUS_SMALL)
    best_wide_native = _best_shift_cc(fixed_native_u8, moving_native_u8, fixed_mask, moving_mask, radius=SEARCH_RADIUS_WIDE)

    for name, description, fn in _method_variants():
        fixed_proc_u8 = fn(fixed_native_u8, np.where(fixed_mask > 0, 255, 0).astype(np.uint8))
        moving_proc_u8 = fn(moving_native_u8, np.where(moving_mask > 0, 255, 0).astype(np.uint8))
        current_proc_metrics = _metrics_from_pair(fixed_proc_u8, moving_proc_u8, fixed_mask, moving_mask)
        best_small_proc = _best_shift_cc(fixed_proc_u8, moving_proc_u8, fixed_mask, moving_mask, radius=SEARCH_RADIUS_SMALL)
        best_wide_proc = _best_shift_cc(fixed_proc_u8, moving_proc_u8, fixed_mask, moving_mask, radius=SEARCH_RADIUS_WIDE)
        panels = _compose_method_panels(
            fixed_native_u8=fixed_native_u8,
            moving_native_u8=moving_native_u8,
            fixed_proc_u8=fixed_proc_u8,
            moving_proc_u8=moving_proc_u8,
            fixed_mask=fixed_mask,
            moving_mask=moving_mask,
            moving_footprint_mask=moving_footprint_mask,
            best_small_native=best_small_native,
            best_small_proc=best_small_proc,
        )
        results.append(
            MethodResult(
                name=name,
                description=description,
                fixed_proc_u8=fixed_proc_u8,
                moving_proc_u8=moving_proc_u8,
                current_metrics_native=current_native_metrics,
                current_metrics_proc=current_proc_metrics,
                best_small_native=best_small_native,
                best_small_proc=best_small_proc,
                best_wide_native=best_wide_native,
                best_wide_proc=best_wide_proc,
                panels=panels,
            )
        )
    return results


def _save_per_method(result: MethodResult, out_dir: Path) -> None:
    method_dir = _ensure_dir(out_dir / result.name)
    for key, panel in result.panels.items():
        cv2.imwrite(str(method_dir / f"{key}.png"), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(method_dir / "fixed_processed_u8.png"), result.fixed_proc_u8)
    cv2.imwrite(str(method_dir / "moving_processed_u8.png"), result.moving_proc_u8)


def _save_contact_sheet(results: list[MethodResult], out_path: Path) -> None:
    cols = [
        ("confocal_native", "Confocal native\n(inverted display)"),
        ("myelin_native", "Myelin native"),
        ("confocal_proc", "Confocal processed"),
        ("myelin_proc", "Myelin processed"),
        ("native_overlay_current", "Native overlay\ncurrent"),
        ("native_overlay_shifted", f"Native overlay\nshifted ±{SEARCH_RADIUS_SMALL}px"),
        ("proc_overlay_current", "Processed overlay\ncurrent"),
        ("proc_overlay_shifted", f"Processed overlay\nshifted ±{SEARCH_RADIUS_SMALL}px"),
    ]
    side = 180
    left = 360
    top = 92
    gap = 10
    row_h = side + 64
    width = left + len(cols) * (side + gap) + gap
    height = top + len(results) * row_h + gap
    canvas = np.full((height, width, 3), 246, dtype=np.uint8)
    cv2.putText(canvas, "T12 r2c0 feature / metric probe", (18, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.82, (25, 25, 25), 2, cv2.LINE_AA)
    cv2.putText(canvas, "Current geometry fixed to manual anchor state. Only representation / local metric behavior changes.", (18, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (80, 80, 80), 1, cv2.LINE_AA)
    x = left
    for _key, label in cols:
        for idx, line in enumerate(label.split("\n")):
            cv2.putText(canvas, line, (x + 8, 24 + idx * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (50, 50, 50), 1, cv2.LINE_AA)
        x += side + gap
    for ridx, result in enumerate(results):
        y = top + ridx * row_h
        cv2.putText(canvas, result.name, (18, y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (25, 25, 25), 2, cv2.LINE_AA)
        cv2.putText(canvas, result.description, (18, y + 44), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (90, 90, 90), 1, cv2.LINE_AA)
        metric_line1 = (
            f"proc current CC={result.current_metrics_proc['cc']:.3f} "
            f"small-> {result.best_small_proc['cc']:.3f} @ ({int(result.best_small_proc['dx'])},{int(result.best_small_proc['dy'])})"
        )
        metric_line2 = (
            f"wide-> {result.best_wide_proc['cc']:.3f} @ ({int(result.best_wide_proc['dx'])},{int(result.best_wide_proc['dy'])}) "
            f"| native current CC={result.current_metrics_native['cc']:.3f}"
        )
        cv2.putText(canvas, metric_line1, (18, y + 64), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (55, 55, 55), 1, cv2.LINE_AA)
        cv2.putText(canvas, metric_line2, (18, y + 82), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (55, 55, 55), 1, cv2.LINE_AA)
        x = left
        for key, _label in cols:
            panel = result.panels[key]
            canvas[y : y + side, x : x + side] = cv2.resize(panel, (side, side), interpolation=cv2.INTER_AREA)
            x += side + gap
    cv2.imwrite(str(out_path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))


def main() -> None:
    out_root = _ensure_dir(OUT_ROOT)
    ctx = _load_tile12_context()
    results = _run_methods(ctx)

    for result in results:
        _save_per_method(result, out_root)

    _save_contact_sheet(results, out_root / "T12_r2c0_contact_sheet.png")

    rows: list[dict[str, Any]] = []
    for result in results:
        rows.append(
            {
                "method": result.name,
                "description": result.description,
                "current_native_cc": float(result.current_metrics_native["cc"]),
                "current_native_mi": float(result.current_metrics_native["mi"]),
                "current_proc_cc": float(result.current_metrics_proc["cc"]),
                "current_proc_mi": float(result.current_metrics_proc["mi"]),
                "best_small_proc_cc": float(result.best_small_proc["cc"]),
                "best_small_proc_dx": int(result.best_small_proc["dx"]),
                "best_small_proc_dy": int(result.best_small_proc["dy"]),
                "best_wide_proc_cc": float(result.best_wide_proc["cc"]),
                "best_wide_proc_dx": int(result.best_wide_proc["dx"]),
                "best_wide_proc_dy": int(result.best_wide_proc["dy"]),
                "small_shift_mag": float(math.hypot(float(result.best_small_proc["dx"]), float(result.best_small_proc["dy"]))),
                "wide_shift_mag": float(math.hypot(float(result.best_wide_proc["dx"]), float(result.best_wide_proc["dy"]))),
            }
        )
    _write_csv(out_root / "method_metrics.csv", rows)

    summary_lines = [
        "# T12 r2c0 Feature Probe",
        "",
        "This probe keeps the current manual geometry fixed and tests whether image representation/metric behavior explains why CC may prefer a wrong shift.",
        "",
        "Rules:",
        f"- Tile focus: `T{TILE_INDEX} r2c0`",
        "- Geometry frozen to current manual anchor state.",
        f"- Small-shift search radius: ±{SEARCH_RADIUS_SMALL}px",
        f"- Wide-shift search radius: ±{SEARCH_RADIUS_WIDE}px",
        "- Confocal is displayed inverted in native panels.",
        "- Processed methods are applied symmetrically to both confocal and myelin patches.",
        "",
        "Ranking by `current_proc_cc`:",
    ]
    for row in sorted(rows, key=lambda r: float(r["current_proc_cc"]), reverse=True):
        summary_lines.append(
            f"- `{row['method']}`: current_proc_cc={row['current_proc_cc']:.4f}, "
            f"small_best={row['best_small_proc_cc']:.4f} @ ({row['best_small_proc_dx']},{row['best_small_proc_dy']}), "
            f"wide_best={row['best_wide_proc_cc']:.4f} @ ({row['best_wide_proc_dx']},{row['best_wide_proc_dy']})"
        )
    (out_root / "summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    _write_json(
        out_root / "run_manifest.json",
        {
            "tile_index": TILE_INDEX,
            "manual_state": geom.MANUAL_STATE,
            "anchor_pair": geom.ANCHOR_PAIR,
            "search_radius_small_px": SEARCH_RADIUS_SMALL,
            "search_radius_wide_px": SEARCH_RADIUS_WIDE,
            "footprint_bbox_yxyx": list(ctx["footprint_bbox_yxyx"]),
            "methods": rows,
            "contact_sheet": str(out_root / "T12_r2c0_contact_sheet.png"),
        },
    )

    print(f"T12 probe written to: {out_root}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
import math
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

import cv2
import numpy as np
from scipy.stats import mannwhitneyu

from .confocal_registration import refresh_step8_products_from_handoff


@dataclass
class Step8RibbonAnalysisResult:
    annotation_payload: dict[str, Any]
    analysis_payload: dict[str, Any]
    qc_rgb: np.ndarray
    depth_overlay_rgb: np.ndarray
    ribbon_heatmap_rgb: np.ndarray
    profile_rgb: np.ndarray


@dataclass
class Step8DepthProfileProbeResult:
    annotation_payload: dict[str, Any]
    profiles_rows: list[dict[str, Any]]
    windows_rows: list[dict[str, Any]]
    scene_overlay_rgb: np.ndarray
    crop_raw_rgb: np.ndarray
    crop_annotated_rgb: np.ndarray
    crop_original_bbox_rgb: np.ndarray
    profile_plot_rgb: np.ndarray


def _portable_basename(raw: str | Path) -> str:
    text = str(raw)
    if "\\" in text or (len(text) >= 2 and text[1] == ":"):
        return PureWindowsPath(text).name
    return PurePosixPath(text).name


def _portable_existing_path(raw: str | Path) -> Path | None:
    candidate = Path(str(raw))
    if candidate.exists():
        return candidate
    text = str(raw).strip()
    if text.startswith("/mnt/") and len(text) >= 7 and text[5].isalpha() and text[6] == "/":
        drive = text[5].upper()
        tail = text[7:].replace("/", "\\")
        alt = Path(f"{drive}:\\{tail}")
        if alt.exists():
            return alt
    if len(text) >= 3 and text[1] == ":" and text[2] in {"\\", "/"}:
        drive = text[0].lower()
        tail = text[3:].replace("\\", "/")
        alt = Path(f"/mnt/{drive}") / Path(tail)
        if alt.exists():
            return alt
    return None


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_csv_rows(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _prediction_expected_name_for_source(source_path: Path | str) -> str:
    name = _portable_basename(source_path)
    lower = name.lower()
    if lower.endswith(".ome.tif"):
        return name[:-8] + ".ome_pred.tif"
    if lower.endswith(".tif"):
        return name[:-4] + "_pred.tif"
    return Path(name).stem + "_pred.tif"


def _resample_polyline(points_xy: np.ndarray, count: int) -> np.ndarray:
    pts = np.asarray(points_xy, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[0] < 2 or pts.shape[1] != 2:
        raise ValueError("Polyline must be Nx2 with N>=2")
    count = max(2, int(count))
    deltas = np.diff(pts, axis=0)
    seg_lengths = np.sqrt(np.sum(deltas * deltas, axis=1))
    cum = np.concatenate([[0.0], np.cumsum(seg_lengths, dtype=np.float32)])
    total = float(cum[-1])
    if total <= 1e-6:
        return np.repeat(pts[:1], count, axis=0)
    targets = np.linspace(0.0, total, count, dtype=np.float32)
    out = np.empty((count, 2), dtype=np.float32)
    for i, t in enumerate(targets):
        seg = int(np.searchsorted(cum, t, side="right") - 1)
        seg = max(0, min(seg, len(seg_lengths) - 1))
        t0 = float(cum[seg])
        t1 = float(cum[seg + 1])
        alpha = 0.0 if t1 <= t0 else float((t - t0) / (t1 - t0))
        out[i] = pts[seg] * (1.0 - alpha) + pts[seg + 1] * alpha
    return out


def _draw_curve_mask(shape_hw: tuple[int, int], points_xy: np.ndarray, thickness: int = 3) -> np.ndarray:
    h, w = [max(1, int(v)) for v in shape_hw]
    canvas = np.zeros((h, w), dtype=np.uint8)
    pts = np.round(np.asarray(points_xy, dtype=np.float32)).astype(np.int32)
    if pts.shape[0] >= 2:
        cv2.polylines(canvas, [pts.reshape(-1, 1, 2)], False, 255, thickness=thickness, lineType=cv2.LINE_AA)
    return canvas


def _ribbon_polygon_mask(shape_hw: tuple[int, int], surface_xy: np.ndarray, interface_xy: np.ndarray) -> np.ndarray:
    h, w = [max(1, int(v)) for v in shape_hw]
    poly = np.vstack([surface_xy, interface_xy[::-1]])
    pts = np.round(np.asarray(poly, dtype=np.float32)).astype(np.int32)
    canvas = np.zeros((h, w), dtype=np.uint8)
    if pts.shape[0] >= 3:
        cv2.fillPoly(canvas, [pts.reshape(-1, 1, 2)], 255, lineType=cv2.LINE_AA)
    return canvas


def _solve_harmonic_depth_field(
    ribbon_mask_u8: np.ndarray,
    surface_xy: np.ndarray,
    interface_xy: np.ndarray,
    *,
    max_iter: int = 800,
    tol: float = 1e-4,
) -> np.ndarray:
    ribbon = np.asarray(ribbon_mask_u8, dtype=np.uint8) > 0
    h, w = ribbon.shape[:2]
    if not np.any(ribbon):
        return np.zeros((h, w), dtype=np.float32)
    ys, xs = np.nonzero(ribbon)
    x0 = max(0, int(xs.min()) - 2)
    x1 = min(w, int(xs.max()) + 3)
    y0 = max(0, int(ys.min()) - 2)
    y1 = min(h, int(ys.max()) + 3)
    local_ribbon = ribbon[y0:y1, x0:x1]
    surface_local = np.asarray(surface_xy, dtype=np.float32).copy()
    surface_local[:, 0] -= float(x0)
    surface_local[:, 1] -= float(y0)
    interface_local = np.asarray(interface_xy, dtype=np.float32).copy()
    interface_local[:, 0] -= float(x0)
    interface_local[:, 1] -= float(y0)
    surface_mask = _draw_curve_mask(local_ribbon.shape[:2], surface_local, thickness=3) > 0
    interface_mask = _draw_curve_mask(local_ribbon.shape[:2], interface_local, thickness=3) > 0
    surface_mask &= local_ribbon
    interface_mask &= local_ribbon
    ds = cv2.distanceTransform((~surface_mask).astype(np.uint8), cv2.DIST_L2, 3).astype(np.float32)
    di = cv2.distanceTransform((~interface_mask).astype(np.uint8), cv2.DIST_L2, 3).astype(np.float32)
    denom = np.maximum(ds + di, 1e-6)
    u = (ds / denom).astype(np.float32)
    u[~local_ribbon] = 0.0
    u[surface_mask] = 0.0
    u[interface_mask] = 1.0
    fixed = surface_mask | interface_mask
    work = u.copy()
    for _ in range(int(max_iter)):
        avg = 0.25 * (
            np.roll(work, 1, axis=0)
            + np.roll(work, -1, axis=0)
            + np.roll(work, 1, axis=1)
            + np.roll(work, -1, axis=1)
        )
        new_work = work.copy()
        update_mask = local_ribbon & (~fixed)
        new_work[update_mask] = avg[update_mask]
        new_work[surface_mask] = 0.0
        new_work[interface_mask] = 1.0
        diff = float(np.max(np.abs(new_work[update_mask] - work[update_mask]))) if np.any(update_mask) else 0.0
        work = new_work
        if diff <= float(tol):
            break
    full = np.zeros((h, w), dtype=np.float32)
    full[y0:y1, x0:x1] = work
    full[~ribbon] = 0.0
    return np.clip(full, 0.0, 1.0).astype(np.float32)


def _build_remap_grids(
    surface_xy: np.ndarray,
    interface_xy: np.ndarray,
    *,
    depth_samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    depth_samples = max(8, int(depth_samples))
    depth = np.linspace(0.0, 1.0, depth_samples, dtype=np.float32)[:, None]
    sx = surface_xy[:, 0][None, :]
    sy = surface_xy[:, 1][None, :]
    ix = interface_xy[:, 0][None, :]
    iy = interface_xy[:, 1][None, :]
    map_x = ((1.0 - depth) * sx + depth * ix).astype(np.float32)
    map_y = ((1.0 - depth) * sy + depth * iy).astype(np.float32)
    return map_x, map_y


def _render_profile_plot(
    profiles: list[tuple[str, np.ndarray, tuple[int, int, int]]],
    *,
    title: str,
    y_label: str,
    significance_depths: list[int] | None = None,
) -> np.ndarray:
    width = 900
    height = 360
    left = 72
    right = 24
    top = 46
    bottom = 48
    canvas = np.full((height, width, 3), 250, dtype=np.uint8)
    plot_w = width - left - right
    plot_h = height - top - bottom
    cv2.putText(canvas, title, (left, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (20, 20, 20), 2, cv2.LINE_AA)
    cv2.line(canvas, (left, top + plot_h), (left + plot_w, top + plot_h), (80, 80, 80), 1)
    cv2.line(canvas, (left, top), (left, top + plot_h), (80, 80, 80), 1)
    max_y = 1e-6
    for _name, profile, _color in profiles:
        arr = np.asarray(profile, dtype=np.float32)
        if arr.size:
            max_y = max(max_y, float(np.nanmax(arr)))
    max_y *= 1.08
    for tick in range(6):
        frac = tick / 5.0
        y = int(round(top + plot_h - frac * plot_h))
        value = frac * max_y
        cv2.line(canvas, (left - 4, y), (left + plot_w, y), (225, 225, 225), 1)
        cv2.putText(canvas, f"{value:.2f}", (8, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (60, 60, 60), 1, cv2.LINE_AA)
    if profiles:
        depth_n = max(2, int(np.asarray(profiles[0][1]).size))
        for series_idx, (name, profile, color) in enumerate(profiles):
            arr = np.asarray(profile, dtype=np.float32)
            if arr.size != depth_n:
                continue
            pts = []
            for i, v in enumerate(arr):
                x = int(round(left + (i / max(depth_n - 1, 1)) * plot_w))
                y = int(round(top + plot_h - (float(v) / max_y) * plot_h))
                pts.append((x, y))
            cv2.polylines(canvas, [np.asarray(pts, dtype=np.int32)], False, color, 2, cv2.LINE_AA)
            lx = left + 12
            ly = top + 18 + 22 * int(series_idx)
            cv2.line(canvas, (lx, ly - 5), (lx + 22, ly - 5), color, 2, cv2.LINE_AA)
            cv2.putText(canvas, name, (lx + 30, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (25, 25, 25), 1, cv2.LINE_AA)
        if significance_depths:
            sig_y = top + 8
            for i in significance_depths:
                x = int(round(left + (i / max(depth_n - 1, 1)) * plot_w))
                cv2.circle(canvas, (x, sig_y), 3, (200, 32, 32), -1, lineType=cv2.LINE_AA)
    cv2.putText(canvas, "Normalized cortical depth", (left + plot_w // 2 - 90, height - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 40, 40), 1, cv2.LINE_AA)
    cv2.putText(canvas, y_label, (10, top - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (40, 40, 40), 1, cv2.LINE_AA)
    return canvas


def _render_heatmap(
    values_f32: np.ndarray,
    valid_mask: np.ndarray,
    *,
    title: str,
) -> np.ndarray:
    vals = np.asarray(values_f32, dtype=np.float32)
    valid = np.asarray(valid_mask, dtype=bool)
    depth_h, tangential_w = vals.shape[:2]
    img_u8 = np.zeros((depth_h, tangential_w), dtype=np.uint8)
    if np.any(valid):
        img_u8[valid] = np.clip(vals[valid] * 255.0, 0.0, 255.0).astype(np.uint8)
    heat_bgr = cv2.applyColorMap(img_u8, cv2.COLORMAP_TURBO)
    heat = cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB)
    heat[~valid] = np.array([215, 215, 215], dtype=np.uint8)
    heat_big = cv2.resize(heat, (max(300, tangential_w * 2), max(300, depth_h * 2)), interpolation=cv2.INTER_NEAREST)
    canvas = np.full((heat_big.shape[0] + 46, heat_big.shape[1] + 24, 3), 245, dtype=np.uint8)
    cv2.putText(canvas, title, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (20, 20, 20), 2, cv2.LINE_AA)
    canvas[40 : 40 + heat_big.shape[0], 12 : 12 + heat_big.shape[1]] = heat_big
    cv2.rectangle(canvas, (12, 40), (11 + heat_big.shape[1], 39 + heat_big.shape[0]), (180, 180, 180), 1)
    cv2.putText(canvas, "Pial / surface", (12, 38 + heat_big.shape[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (30, 30, 30), 1, cv2.LINE_AA)
    cv2.putText(canvas, "WM/GM", (12, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (30, 30, 30), 1, cv2.LINE_AA)
    return canvas


def _coerce_probe_geometry(probe_payload: dict[str, Any]) -> dict[str, Any]:
    start = np.asarray(probe_payload.get("depth_start_scene_xy"), dtype=np.float32)
    end = np.asarray(probe_payload.get("depth_end_scene_xy"), dtype=np.float32)
    if start.shape != (2,) or end.shape != (2,):
        raise ValueError("Probe must include depth_start_scene_xy and depth_end_scene_xy")
    width_px = float(probe_payload.get("width_px") or 0.0)
    width_vec = np.asarray(probe_payload.get("width_vector_scene_xy") or [0.0, 0.0], dtype=np.float32)
    axis = end - start
    length = float(np.linalg.norm(axis))
    if length <= 1e-3:
        raise ValueError("Probe depth axis is too short")
    depth_unit = axis / length
    normal_unit = np.asarray([-depth_unit[1], depth_unit[0]], dtype=np.float32)
    if width_px <= 0.0 and width_vec.shape == (2,):
        width_px = float(2.0 * abs(float(np.dot(width_vec, normal_unit))))
    width_px = max(2.0, float(width_px))
    if width_vec.shape == (2,) and float(np.dot(width_vec, normal_unit)) < 0:
        normal_unit *= -1.0
    half = 0.5 * width_px
    corners = np.asarray(
        [
            start - normal_unit * half,
            start + normal_unit * half,
            end + normal_unit * half,
            end - normal_unit * half,
        ],
        dtype=np.float32,
    )
    return {
        "start": start,
        "end": end,
        "depth_unit": depth_unit,
        "normal_unit": normal_unit,
        "length_px": length,
        "width_px": width_px,
        "corners": corners,
    }


def _robust_iqr(values: np.ndarray, axis: int = 0) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    return (np.nanpercentile(arr, 75, axis=axis) - np.nanpercentile(arr, 25, axis=axis)).astype(np.float32)


def _trimmed_mean(values: np.ndarray, axis: int = 0, trim_frac: float = 0.1) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if arr.shape[axis] < 3:
        return np.nanmean(arr, axis=axis)
    sorted_arr = np.sort(arr, axis=axis)
    n = sorted_arr.shape[axis]
    lo = int(np.floor(n * trim_frac))
    hi = int(np.ceil(n * (1.0 - trim_frac)))
    if hi <= lo:
        return np.nanmean(sorted_arr, axis=axis)
    slicer = [slice(None)] * sorted_arr.ndim
    slicer[axis] = slice(lo, hi)
    return np.nanmean(sorted_arr[tuple(slicer)], axis=axis)


def _smooth_1d(values: np.ndarray, window: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    window = max(3, int(window) | 1)
    if arr.size < 3:
        return arr.copy()
    valid = np.isfinite(arr).astype(np.float32)
    filled = np.where(np.isfinite(arr), arr, 0.0).astype(np.float32)
    kernel = np.ones(window, dtype=np.float32)
    num = np.convolve(filled, kernel, mode="same")
    den = np.maximum(np.convolve(valid, kernel, mode="same"), 1e-6)
    out = (num / den).astype(np.float32)
    out[valid <= 0] = np.nan
    return out


def _render_probe_scene_overlay(
    scene_rgb: np.ndarray,
    corners_xy: np.ndarray,
    start_xy: np.ndarray,
    end_xy: np.ndarray,
    *,
    probe_id: str,
) -> np.ndarray:
    out = np.asarray(scene_rgb, dtype=np.uint8).copy()
    pts = np.round(np.asarray(corners_xy, dtype=np.float32)).astype(np.int32)
    cv2.polylines(out, [pts.reshape(-1, 1, 2)], True, (255, 208, 48), 3, cv2.LINE_AA)
    p0 = tuple(np.round(start_xy).astype(int).tolist())
    p1 = tuple(np.round(end_xy).astype(int).tolist())
    cv2.arrowedLine(out, p0, p1, (28, 120, 255), 3, cv2.LINE_AA, tipLength=0.05)
    for frac in np.linspace(0.0, 1.0, 6, dtype=np.float32):
        center = np.asarray(start_xy, dtype=np.float32) * (1.0 - float(frac)) + np.asarray(end_xy, dtype=np.float32) * float(frac)
        cv2.circle(out, tuple(np.round(center).astype(int).tolist()), 5, (28, 120, 255), -1, lineType=cv2.LINE_AA)
    label_pt = pts[0]
    cv2.putText(out, str(probe_id), (int(label_pt[0]) + 6, int(label_pt[1]) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 3, cv2.LINE_AA)
    cv2.putText(out, str(probe_id), (int(label_pt[0]) + 6, int(label_pt[1]) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 208, 48), 1, cv2.LINE_AA)
    return out


def _render_probe_crop_annotated(crop_rgb: np.ndarray, *, probe_id: str) -> np.ndarray:
    strip = np.asarray(crop_rgb, dtype=np.uint8)
    h, w = strip.shape[:2]
    margin_l = 70
    margin_t = 34
    margin_b = 34
    canvas = np.full((h + margin_t + margin_b, w + margin_l + 18, 3), 245, dtype=np.uint8)
    canvas[margin_t : margin_t + h, margin_l : margin_l + w] = strip
    cv2.rectangle(canvas, (margin_l, margin_t), (margin_l + w - 1, margin_t + h - 1), (120, 120, 120), 1)
    cv2.putText(canvas, f"{probe_id} raw strip", (margin_l, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (20, 20, 20), 2, cv2.LINE_AA)
    cv2.putText(canvas, "depth 0", (6, margin_t + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (40, 40, 40), 1, cv2.LINE_AA)
    cv2.putText(canvas, "depth 1", (6, margin_t + h - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (40, 40, 40), 1, cv2.LINE_AA)
    for frac in np.linspace(0.0, 1.0, 6, dtype=np.float32):
        y = int(round(margin_t + float(frac) * max(h - 1, 1)))
        cv2.line(canvas, (margin_l - 6, y), (margin_l + w - 1, y), (255, 208, 48), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"{float(frac):.1f}", (24, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (40, 40, 40), 1, cv2.LINE_AA)
    cv2.putText(canvas, "width axis", (margin_l + max(0, w // 2 - 34), margin_t + h + 23), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (40, 40, 40), 1, cv2.LINE_AA)
    return canvas


def _render_probe_original_bbox(
    scene_rgb: np.ndarray,
    corners_xy: np.ndarray,
    *,
    probe_id: str,
    pad_px: int = 40,
) -> np.ndarray:
    scene = np.asarray(scene_rgb, dtype=np.uint8)
    h, w = scene.shape[:2]
    pts = np.asarray(corners_xy, dtype=np.float32)
    x0 = max(0, int(np.floor(float(np.min(pts[:, 0])))) - int(pad_px))
    y0 = max(0, int(np.floor(float(np.min(pts[:, 1])))) - int(pad_px))
    x1 = min(w, int(np.ceil(float(np.max(pts[:, 0])))) + int(pad_px))
    y1 = min(h, int(np.ceil(float(np.max(pts[:, 1])))) + int(pad_px))
    crop = scene[y0:y1, x0:x1].copy()
    local = pts.copy()
    local[:, 0] -= float(x0)
    local[:, 1] -= float(y0)
    cv2.polylines(crop, [np.round(local).astype(np.int32).reshape(-1, 1, 2)], True, (255, 208, 48), 2, cv2.LINE_AA)
    cv2.putText(crop, str(probe_id), (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 3, cv2.LINE_AA)
    cv2.putText(crop, str(probe_id), (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 208, 48), 1, cv2.LINE_AA)
    return crop


def _render_probe_profile_plot(
    profiles: list[tuple[str, np.ndarray, tuple[int, int, int]]],
    *,
    title: str,
) -> np.ndarray:
    return _render_profile_plot(profiles, title=title, y_label="Profile value")


def _compose_two_panel(left_rgb: np.ndarray, right_rgb: np.ndarray, *, left_title: str, right_title: str) -> np.ndarray:
    left = np.asarray(left_rgb, dtype=np.uint8)
    right = np.asarray(right_rgb, dtype=np.uint8)
    height = max(left.shape[0], right.shape[0]) + 46
    width = left.shape[1] + right.shape[1] + 36
    canvas = np.full((height, width, 3), 245, dtype=np.uint8)
    cv2.putText(canvas, left_title, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (20, 20, 20), 2, cv2.LINE_AA)
    cv2.putText(canvas, right_title, (left.shape[1] + 24, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (20, 20, 20), 2, cv2.LINE_AA)
    canvas[40 : 40 + left.shape[0], 12 : 12 + left.shape[1]] = left
    canvas[40 : 40 + right.shape[0], left.shape[1] + 24 : left.shape[1] + 24 + right.shape[1]] = right
    return canvas


def _roi_suffix_from_label(label: str) -> str:
    clean = str(label).replace("(maybe_bad)", "").strip()
    return clean.rsplit("_", 1)[-1] if "_" in clean else clean


def _sample_group_from_label(label: str) -> str:
    try:
        value = int(str(label)[:4])
    except Exception:
        return "other"
    return "2501-2504" if value <= 2504 else "2505-2508"


def _render_depth_overlay(
    scene_rgb: np.ndarray,
    depth_field_f32: np.ndarray,
    ribbon_mask_u8: np.ndarray,
    coverage_mask_u8: np.ndarray,
    surface_xy: np.ndarray,
    interface_xy: np.ndarray,
) -> np.ndarray:
    base = np.asarray(scene_rgb, dtype=np.uint8).copy()
    depth_u8 = np.clip(np.asarray(depth_field_f32, dtype=np.float32) * 255.0, 0.0, 255.0).astype(np.uint8)
    heat = cv2.cvtColor(cv2.applyColorMap(depth_u8, cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)
    ribbon = np.asarray(ribbon_mask_u8, dtype=np.uint8) > 0
    coverage = np.asarray(coverage_mask_u8, dtype=np.uint8) > 0
    valid = ribbon & coverage
    alpha = np.zeros(base.shape[:2], dtype=np.float32)
    alpha[valid] = 0.48
    base = np.clip((1.0 - alpha[..., None]) * base.astype(np.float32) + alpha[..., None] * heat.astype(np.float32), 0, 255).astype(np.uint8)
    cv2.polylines(base, [np.round(surface_xy).astype(np.int32).reshape(-1, 1, 2)], False, (28, 120, 255), 3, cv2.LINE_AA)
    cv2.polylines(base, [np.round(interface_xy).astype(np.int32).reshape(-1, 1, 2)], False, (255, 96, 32), 3, cv2.LINE_AA)
    return base


def load_scene_prediction_arrays_from_handoff(
    handoff_path: Path,
    *,
    prediction_root: Path | None = None,
) -> dict[str, Any]:
    path = Path(handoff_path)
    payload = _read_json(path)
    if prediction_root is not None:
        files = ((payload.get("prediction_import") or {}).get("files") or {})
        has_scene_files = any(str(files.get(k) or "").strip() for k in ("stitched_prediction_mask", "stitched_prediction_probability", "stitched_prediction_support"))
        if not has_scene_files:
            refresh_step8_products_from_handoff(path, prediction_root=prediction_root)
            payload = _read_json(path)
    export_dir = _portable_existing_path(str(payload.get("step7_export_dir") or path.parent)) or Path(str(payload.get("step7_export_dir") or path.parent))
    scene_shape = tuple(payload.get("scene_space", {}).get("fixed_preview_shape_hw") or [512, 512])
    scene_h = max(1, int(scene_shape[0]))
    scene_w = max(1, int(scene_shape[1]))
    files = ((payload.get("prediction_import") or {}).get("files") or {})
    crop_bbox = list(payload.get("stitched_scene", {}).get("crop_bbox_xyxy") or [0, 0, 0, 0])
    prob_scene = np.zeros((scene_h, scene_w), dtype=np.uint8)
    mask_scene = np.zeros((scene_h, scene_w), dtype=np.uint8)
    support_scene = np.zeros((scene_h, scene_w), dtype=np.uint8)
    for key, target in (
        ("stitched_prediction_probability", prob_scene),
        ("stitched_prediction_mask", mask_scene),
        ("stitched_prediction_support", support_scene),
    ):
        raw = str(files.get(key) or "")
        img_path = _portable_existing_path(raw) if raw else None
        if img_path is None or not img_path.exists():
            continue
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        x0, y0, x1, y1 = [int(v) for v in crop_bbox]
        x0 = max(0, min(scene_w, x0))
        x1 = max(x0, min(scene_w, x1))
        y0 = max(0, min(scene_h, y0))
        y1 = max(y0, min(scene_h, y1))
        crop_h = min(img.shape[0], max(0, y1 - y0))
        crop_w = min(img.shape[1], max(0, x1 - x0))
        if crop_h > 0 and crop_w > 0:
            target[y0 : y0 + crop_h, x0 : x0 + crop_w] = img[:crop_h, :crop_w]
    if not np.any(support_scene):
        support_scene = (mask_scene > 0).astype(np.uint8) * 255
    if not np.any(prob_scene) and np.any(mask_scene):
        prob_scene = np.where(mask_scene > 0, 255, 0).astype(np.uint8)
    return {
        "handoff": payload,
        "export_dir": export_dir,
        "scene_shape_hw": (scene_h, scene_w),
        "crop_bbox_xyxy": crop_bbox,
        "probability_scene_u8": prob_scene,
        "mask_scene_u8": mask_scene,
        "support_scene_u8": support_scene,
    }


def compute_step8_ribbon_analysis(
    *,
    handoff_path: Path,
    scene_rgb: np.ndarray,
    tile_rows: list[dict[str, object]],
    include_map: dict[int, bool],
    surface_points_scene_xy: list[list[float]] | list[tuple[float, float]] | np.ndarray,
    interface_points_scene_xy: list[list[float]] | list[tuple[float, float]] | np.ndarray,
    prediction_root: Path | None = None,
    tangent_samples: int = 256,
    depth_samples: int = 128,
    max_scene_edge: int = 1024,
) -> Step8RibbonAnalysisResult:
    pred_bundle = load_scene_prediction_arrays_from_handoff(handoff_path, prediction_root=prediction_root)
    pred_scene = np.asarray(pred_bundle["probability_scene_u8"], dtype=np.uint8)
    scene_mask = np.asarray(pred_bundle["mask_scene_u8"], dtype=np.uint8)
    if pred_scene.shape != scene_rgb.shape[:2]:
        pred_scene = cv2.resize(pred_scene, (scene_rgb.shape[1], scene_rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
    if scene_mask.shape != scene_rgb.shape[:2]:
        scene_mask = cv2.resize(scene_mask, (scene_rgb.shape[1], scene_rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
    h, w = scene_rgb.shape[:2]
    scale = min(1.0, float(max_scene_edge) / float(max(h, w, 1)))
    scaled_hw = (max(1, int(round(h * scale))), max(1, int(round(w * scale))))
    scene_small = cv2.resize(np.asarray(scene_rgb, dtype=np.uint8), (scaled_hw[1], scaled_hw[0]), interpolation=cv2.INTER_AREA)
    burden_small = cv2.resize(pred_scene, (scaled_hw[1], scaled_hw[0]), interpolation=cv2.INTER_LINEAR)
    mask_small = cv2.resize(scene_mask, (scaled_hw[1], scaled_hw[0]), interpolation=cv2.INTER_NEAREST)
    coverage_full = np.zeros((h, w), dtype=np.uint8)
    for row in tile_rows:
        idx = int(row.get("tile_index", -1))
        if not bool(include_map.get(idx, False)):
            continue
        arr = np.asarray(row.get("final_scene_polygon_xy"), dtype=np.float32)
        if arr.ndim != 2 or arr.shape[0] < 3 or arr.shape[1] != 2:
            bbox = row.get("final_scene_bbox_xyxy") or row.get("pred_scene_bbox_xyxy") or row.get("nominal_scene_bbox_xyxy")
            if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4):
                continue
            x0, y0, x1, y1 = [float(v) for v in bbox]
            arr = np.asarray([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32)
        cv2.fillPoly(coverage_full, [np.round(arr).astype(np.int32).reshape(-1, 1, 2)], 255, lineType=cv2.LINE_AA)
    coverage_small = cv2.resize(coverage_full, (scaled_hw[1], scaled_hw[0]), interpolation=cv2.INTER_NEAREST)
    surface_xy = np.asarray(surface_points_scene_xy, dtype=np.float32) * float(scale)
    interface_xy = np.asarray(interface_points_scene_xy, dtype=np.float32) * float(scale)
    if surface_xy.ndim != 2 or surface_xy.shape[0] < 2 or surface_xy.shape[1] != 2:
        raise ValueError("Surface curve must contain at least two points")
    if interface_xy.ndim != 2 or interface_xy.shape[0] < 2 or interface_xy.shape[1] != 2:
        raise ValueError("WM/GM interface curve must contain at least two points")
    surface_res = _resample_polyline(surface_xy, int(tangent_samples))
    interface_res = _resample_polyline(interface_xy, int(tangent_samples))
    ribbon_mask = _ribbon_polygon_mask(scaled_hw, surface_res, interface_res)
    analysis_mask = np.where((ribbon_mask > 0) & (coverage_small > 0), 255, 0).astype(np.uint8)
    if not np.any(analysis_mask):
        raise ValueError("The ribbon does not intersect any included tiles. Adjust curves or tile QC.")
    depth_field = _solve_harmonic_depth_field(ribbon_mask, surface_res, interface_res)
    map_x, map_y = _build_remap_grids(surface_res, interface_res, depth_samples=int(depth_samples))
    ribbon_prob = cv2.remap(burden_small.astype(np.float32) / 255.0, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
    ribbon_mask_sampled = cv2.remap((analysis_mask > 0).astype(np.uint8), map_x, map_y, interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    ribbon_binary = cv2.remap((mask_small > 0).astype(np.uint8), map_x, map_y, interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    valid = ribbon_mask_sampled > 0
    ribbon_prob = np.where(valid, ribbon_prob, np.nan).astype(np.float32)
    ribbon_binary = np.where(valid, ribbon_binary.astype(np.float32), np.nan).astype(np.float32)
    depth_profile = np.nanmean(ribbon_prob, axis=1)
    tangential_profile = np.nanmean(ribbon_prob, axis=0)
    depth_overlay_rgb = _render_depth_overlay(scene_small, depth_field, ribbon_mask, coverage_small, surface_res, interface_res)
    heatmap_rgb = _render_heatmap(np.nan_to_num(ribbon_prob, nan=0.0), valid, title="Standardized ribbon myelin burden")
    profile_rgb = _render_profile_plot(
        [
            ("Probability burden", np.nan_to_num(depth_profile, nan=0.0), (32, 109, 214)),
            ("Binary burden", np.nan_to_num(np.nanmean(ribbon_binary, axis=1), nan=0.0), (210, 64, 64)),
        ],
        title="Depth-aligned myelin burden profile",
        y_label="Mean burden",
    )
    qc_rgb = _compose_two_panel(depth_overlay_rgb, heatmap_rgb, left_title="Depth field on scene", right_title="Straightened ribbon")
    handoff = pred_bundle["handoff"]
    export_dir = pred_bundle["export_dir"]
    saved_at = datetime.now(timezone.utc).isoformat()
    annotation_payload = {
        "schema": "step8_ribbon_annotation_v1",
        "saved_at_utc": saved_at,
        "step7_export_dir": str(export_dir),
        "step8_handoff_path": str(handoff_path),
        "myelin_label": str(handoff.get("myelin_label") or export_dir.parent.name),
        "surface_points_scene_xy": [[float(x), float(y)] for x, y in np.asarray(surface_points_scene_xy, dtype=np.float32)],
        "interface_points_scene_xy": [[float(x), float(y)] for x, y in np.asarray(interface_points_scene_xy, dtype=np.float32)],
        "analysis_params": {
            "tangent_samples": int(tangent_samples),
            "depth_samples": int(depth_samples),
            "max_scene_edge": int(max_scene_edge),
        },
    }
    analysis_payload = {
        "schema": "step8_ribbon_analysis_v1",
        "saved_at_utc": saved_at,
        "step7_export_dir": str(export_dir),
        "step8_handoff_path": str(handoff_path),
        "myelin_label": str(handoff.get("myelin_label") or export_dir.parent.name),
        "roi_suffix": _roi_suffix_from_label(str(handoff.get("myelin_label") or export_dir.parent.name)),
        "group": _sample_group_from_label(str(handoff.get("myelin_label") or export_dir.parent.name)),
        "selected_tile_count": int(sum(1 for v in include_map.values() if bool(v))),
        "scene_shape_hw": [int(h), int(w)],
        "analysis_shape_hw": [int(scaled_hw[0]), int(scaled_hw[1])],
        "analysis_params": annotation_payload["analysis_params"],
        "coverage_pixels": int(np.count_nonzero(analysis_mask)),
        "ribbon_pixels": int(np.count_nonzero(ribbon_mask)),
        "depth_profile_probability": [float(v) if np.isfinite(v) else None for v in depth_profile],
        "depth_profile_binary": [float(v) if np.isfinite(v) else None for v in np.nanmean(ribbon_binary, axis=1)],
        "tangential_profile_probability": [float(v) if np.isfinite(v) else None for v in tangential_profile],
    }
    return Step8RibbonAnalysisResult(
        annotation_payload=annotation_payload,
        analysis_payload=analysis_payload,
        qc_rgb=qc_rgb,
        depth_overlay_rgb=depth_overlay_rgb,
        ribbon_heatmap_rgb=heatmap_rgb,
        profile_rgb=profile_rgb,
    )


def save_step8_ribbon_result(
    export_dir: Path,
    result: Step8RibbonAnalysisResult,
) -> dict[str, str]:
    out_dir = Path(export_dir)
    annotation_path = out_dir / "step8_ribbon_annotation.json"
    analysis_path = out_dir / "step8_ribbon_analysis.json"
    qc_path = out_dir / "step8_ribbon_qc.png"
    depth_path = out_dir / "step8_ribbon_depth_field.png"
    heatmap_path = out_dir / "step8_ribbon_heatmap.png"
    profile_path = out_dir / "step8_ribbon_profile.png"
    _write_json(annotation_path, result.annotation_payload)
    _write_json(analysis_path, result.analysis_payload)
    cv2.imwrite(str(qc_path), cv2.cvtColor(np.asarray(result.qc_rgb, dtype=np.uint8), cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(depth_path), cv2.cvtColor(np.asarray(result.depth_overlay_rgb, dtype=np.uint8), cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(heatmap_path), cv2.cvtColor(np.asarray(result.ribbon_heatmap_rgb, dtype=np.uint8), cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(profile_path), cv2.cvtColor(np.asarray(result.profile_rgb, dtype=np.uint8), cv2.COLOR_RGB2BGR))
    return {
        "annotation_json": str(annotation_path),
        "analysis_json": str(analysis_path),
        "ribbon_qc_png": str(qc_path),
        "depth_field_png": str(depth_path),
        "ribbon_heatmap_png": str(heatmap_path),
        "profile_png": str(profile_path),
    }


def load_step8_saved_ribbon_annotation(export_dir: Path) -> dict[str, Any] | None:
    path = Path(export_dir) / "step8_ribbon_annotation.json"
    if not path.exists():
        return None
    try:
        return _read_json(path)
    except Exception:
        return None


def load_step8_saved_ribbon_analysis(export_dir: Path) -> dict[str, Any] | None:
    path = Path(export_dir) / "step8_ribbon_analysis.json"
    if not path.exists():
        return None
    try:
        return _read_json(path)
    except Exception:
        return None


def compute_groupwise_depth_aligned_comparison(
    analysis_rows: list[dict[str, Any]],
    *,
    roi_suffix: str,
) -> dict[str, Any]:
    rows = [dict(row) for row in analysis_rows if str(row.get("roi_suffix") or "").upper() == str(roi_suffix).upper()]
    if len(rows) < 2:
        raise ValueError("Need at least two saved ribbon analyses from the same ROI type")
    group_a = [row for row in rows if str(row.get("group") or "") == "2501-2504"]
    group_b = [row for row in rows if str(row.get("group") or "") == "2505-2508"]
    if not group_a or not group_b:
        raise ValueError("Need saved ribbon analyses from both groups")
    depth_n = max(len(list(row.get("depth_profile_probability") or [])) for row in rows)
    def _stack(items: list[dict[str, Any]]) -> np.ndarray:
        mats = []
        for row in items:
            arr = np.asarray([np.nan if v is None else float(v) for v in list(row.get("depth_profile_probability") or [])], dtype=np.float32)
            if arr.size < depth_n:
                arr = np.pad(arr, (0, depth_n - arr.size), constant_values=np.nan)
            mats.append(arr)
        return np.asarray(mats, dtype=np.float32)
    A = _stack(group_a)
    B = _stack(group_b)
    mean_a = np.nanmean(A, axis=0)
    mean_b = np.nanmean(B, axis=0)
    significant_depths: list[int] = []
    pvals: list[float | None] = []
    for i in range(depth_n):
        va = A[:, i]
        vb = B[:, i]
        va = va[np.isfinite(va)]
        vb = vb[np.isfinite(vb)]
        if va.size < 2 or vb.size < 2:
            pvals.append(None)
            continue
        res = mannwhitneyu(va, vb, alternative="two-sided")
        pvals.append(float(res.pvalue))
        if float(res.pvalue) < 0.05:
            significant_depths.append(int(i))
    plot_rgb = _render_profile_plot(
        [
            ("2501-2504", np.nan_to_num(mean_a, nan=0.0), (28, 120, 255)),
            ("2505-2508", np.nan_to_num(mean_b, nan=0.0), (222, 76, 76)),
        ],
        title=f"Groupwise depth-aligned comparison ({roi_suffix.upper()})",
        y_label="Mean burden",
        significance_depths=significant_depths,
    )
    payload = {
        "schema": "step8_groupwise_depth_comparison_v1",
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
        "roi_suffix": str(roi_suffix).upper(),
        "group_counts": {"2501-2504": int(len(group_a)), "2505-2508": int(len(group_b))},
        "depth_count": int(depth_n),
        "mean_profile_2501_2504": [float(v) if np.isfinite(v) else None for v in mean_a],
        "mean_profile_2505_2508": [float(v) if np.isfinite(v) else None for v in mean_b],
        "per_depth_pvalue": pvals,
        "significant_depth_indices_p_lt_0_05": significant_depths,
        "source_exports": [str(row.get("step7_export_dir") or "") for row in rows],
    }
    return {
        "payload": payload,
        "plot_rgb": plot_rgb,
    }


def compute_step8_depth_profile_probe(
    *,
    scene_rgb: np.ndarray,
    probe_payload: dict[str, Any],
    probe_id: str,
    source_image_path: str = "",
    scene_um_per_px_xy: tuple[float, float] | list[float] | None = None,
    depth_samples: int = 512,
    width_samples: int | None = None,
    window_fraction: float = 0.05,
) -> Step8DepthProfileProbeResult:
    scene = np.asarray(scene_rgb, dtype=np.uint8)
    if scene.ndim != 3 or scene.shape[2] != 3:
        raise ValueError("scene_rgb must be an RGB image")
    geom = _coerce_probe_geometry(probe_payload)
    depth_samples = max(16, int(depth_samples))
    if width_samples is None or int(width_samples) <= 0:
        width_samples = int(np.clip(round(float(geom["width_px"])), 16, 256))
    width_samples = max(4, int(width_samples))
    start = np.asarray(geom["start"], dtype=np.float32)
    end = np.asarray(geom["end"], dtype=np.float32)
    normal = np.asarray(geom["normal_unit"], dtype=np.float32)
    half = float(geom["width_px"]) * 0.5
    depth = np.linspace(0.0, 1.0, depth_samples, dtype=np.float32)[:, None]
    offsets = np.linspace(-half, half, width_samples, dtype=np.float32)[None, :]
    centers = start[None, :] * (1.0 - depth) + end[None, :] * depth
    map_x = (centers[:, 0:1] + offsets * normal[0]).astype(np.float32)
    map_y = (centers[:, 1:2] + offsets * normal[1]).astype(np.float32)
    strip = cv2.remap(scene, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(245, 245, 245))
    strip_f = strip.astype(np.float32)
    gray = (0.299 * strip_f[:, :, 0] + 0.587 * strip_f[:, :, 1] + 0.114 * strip_f[:, :, 2]).astype(np.float32)
    i0 = max(float(np.percentile(gray, 99.0)), 1.0)
    gray_od = -np.log((gray + 1.0) / (i0 + 1.0)).astype(np.float32)
    channel_i0 = np.maximum(np.percentile(strip_f, 99.0, axis=(0, 1)), 1.0).astype(np.float32)
    channel_od = -np.log((strip_f + 1.0) / (channel_i0[None, None, :] + 1.0)).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx * gx + gy * gy).astype(np.float32)
    jxx = np.nanmean(gx * gx, axis=1)
    jyy = np.nanmean(gy * gy, axis=1)
    jxy = np.nanmean(gx * gy, axis=1)
    coherency = (np.sqrt((jxx - jyy) ** 2 + 4.0 * jxy * jxy) / np.maximum(jxx + jyy, 1e-6)).astype(np.float32)
    gray_median = np.nanmedian(gray, axis=1).astype(np.float32)
    gray_trimmed = _trimmed_mean(gray, axis=1).astype(np.float32)
    gray_iqr = _robust_iqr(gray, axis=1)
    od_median = np.nanmedian(gray_od, axis=1).astype(np.float32)
    od_trimmed = _trimmed_mean(gray_od, axis=1).astype(np.float32)
    od_iqr = _robust_iqr(gray_od, axis=1)
    red_od = np.nanmedian(channel_od[:, :, 0], axis=1).astype(np.float32)
    green_od = np.nanmedian(channel_od[:, :, 1], axis=1).astype(np.float32)
    blue_od = np.nanmedian(channel_od[:, :, 2], axis=1).astype(np.float32)
    grad_median = np.nanmedian(grad_mag, axis=1).astype(np.float32)
    texture_width_var = np.nanvar(gray, axis=1).astype(np.float32)
    smooth_window = max(5, int(round(depth_samples * max(0.01, min(float(window_fraction), 0.2)))))
    if smooth_window % 2 == 0:
        smooth_window += 1
    od_smooth = _smooth_1d(od_median, smooth_window)
    gray_smooth = _smooth_1d(gray_median, smooth_window)
    x = np.linspace(0.0, 1.0, depth_samples, dtype=np.float32)
    od_d1 = np.gradient(od_smooth, x).astype(np.float32)
    od_d2 = np.gradient(od_d1, x).astype(np.float32)
    gray_d1 = np.gradient(gray_smooth, x).astype(np.float32)
    gray_d2 = np.gradient(gray_d1, x).astype(np.float32)
    center_xy = centers.reshape(depth_samples, 2)
    profiles_rows: list[dict[str, Any]] = []
    for i in range(depth_samples):
        profiles_rows.append(
            {
                "probe_id": str(probe_id),
                "depth_index": int(i),
                "depth_fraction": float(x[i]),
                "scene_center_x": float(center_xy[i, 0]),
                "scene_center_y": float(center_xy[i, 1]),
                "gray_median": float(gray_median[i]),
                "gray_trimmed_mean": float(gray_trimmed[i]),
                "gray_iqr": float(gray_iqr[i]),
                "od_median": float(od_median[i]),
                "od_trimmed_mean": float(od_trimmed[i]),
                "od_iqr": float(od_iqr[i]),
                "red_od_median": float(red_od[i]),
                "green_od_median": float(green_od[i]),
                "blue_od_median": float(blue_od[i]),
                "gradient_median": float(grad_median[i]),
                "texture_width_var": float(texture_width_var[i]),
                "coherency": float(coherency[i]),
                "od_smoothed": float(od_smooth[i]),
                "od_derivative1": float(od_d1[i]),
                "od_derivative2": float(od_d2[i]),
                "gray_smoothed": float(gray_smooth[i]),
                "gray_derivative1": float(gray_d1[i]),
                "gray_derivative2": float(gray_d2[i]),
            }
        )
    win = max(5, int(round(depth_samples * max(0.01, min(float(window_fraction), 0.5)))))
    if win % 2 == 0:
        win += 1
    half_win = win // 2
    windows_rows: list[dict[str, Any]] = []
    for i in range(depth_samples):
        lo = max(0, i - half_win)
        hi = min(depth_samples, i + half_win + 1)
        if hi - lo < 3:
            continue
        xx = x[lo:hi]
        od_seg = od_median[lo:hi]
        gray_seg = gray_median[lo:hi]
        try:
            od_slope = float(np.polyfit(xx, od_seg, 1)[0])
        except Exception:
            od_slope = float("nan")
        try:
            gray_slope = float(np.polyfit(xx, gray_seg, 1)[0])
        except Exception:
            gray_slope = float("nan")
        mid = lo + (hi - lo) // 2
        left = slice(lo, mid)
        right = slice(mid, hi)
        od_left = float(np.nanmedian(od_median[left])) if mid > lo else float("nan")
        od_right = float(np.nanmedian(od_median[right])) if hi > mid else float("nan")
        gray_left = float(np.nanmedian(gray_median[left])) if mid > lo else float("nan")
        gray_right = float(np.nanmedian(gray_median[right])) if hi > mid else float("nan")
        od_std = float(np.nanstd(od_seg))
        gray_std = float(np.nanstd(gray_seg))
        windows_rows.append(
            {
                "probe_id": str(probe_id),
                "window_size_samples": int(win),
                "center_depth_index": int(i),
                "center_depth_fraction": float(x[i]),
                "window_start_fraction": float(x[lo]),
                "window_end_fraction": float(x[hi - 1]),
                "od_local_median": float(np.nanmedian(od_seg)),
                "od_local_mean": float(np.nanmean(od_seg)),
                "od_local_slope": od_slope,
                "od_local_variance": float(np.nanvar(od_seg)),
                "od_left_right_delta": float(od_right - od_left),
                "od_left_right_effect_size": float((od_right - od_left) / max(od_std, 1e-6)),
                "gray_local_median": float(np.nanmedian(gray_seg)),
                "gray_local_mean": float(np.nanmean(gray_seg)),
                "gray_local_slope": gray_slope,
                "gray_local_variance": float(np.nanvar(gray_seg)),
                "gray_left_right_delta": float(gray_right - gray_left),
                "gray_left_right_effect_size": float((gray_right - gray_left) / max(gray_std, 1e-6)),
            }
        )
    corners = np.asarray(geom["corners"], dtype=np.float32)
    scene_um: tuple[float, float] | None = None
    if isinstance(scene_um_per_px_xy, (list, tuple)) and len(scene_um_per_px_xy) >= 2:
        try:
            sx = float(scene_um_per_px_xy[0])
            sy = float(scene_um_per_px_xy[1])
            if np.isfinite(sx) and np.isfinite(sy) and sx > 0.0 and sy > 0.0:
                scene_um = (sx, sy)
        except Exception:
            scene_um = None
    if scene_um is not None:
        depth_delta = end - start
        half_width_vec = np.asarray([normal[0] * half, normal[1] * half], dtype=np.float32)
        depth_length_um = float(np.hypot(float(depth_delta[0] * scene_um[0]), float(depth_delta[1] * scene_um[1])))
        width_um = float(2.0 * np.hypot(float(half_width_vec[0] * scene_um[0]), float(half_width_vec[1] * scene_um[1])))
    else:
        depth_length_um = None
        width_um = None
    scene_overlay = _render_probe_scene_overlay(scene, corners, start, end, probe_id=str(probe_id))
    crop_annotated = _render_probe_crop_annotated(strip, probe_id=str(probe_id))
    crop_bbox = _render_probe_original_bbox(scene, corners, probe_id=str(probe_id))
    plot = _render_probe_profile_plot(
        [
            ("OD median", od_median, (32, 109, 214)),
            ("OD smooth", od_smooth, (24, 160, 88)),
            ("Gradient median", grad_median / max(float(np.nanmax(grad_median)), 1e-6), (210, 64, 64)),
            ("Coherency", coherency, (128, 72, 190)),
        ],
        title=f"Depth profile probe {probe_id}",
    )
    payload = {
        "schema": "step8_depth_profile_probe_annotation_v1",
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
        "probe_id": str(probe_id),
        "source_image_path": str(source_image_path or ""),
        "scene_shape_hw": [int(scene.shape[0]), int(scene.shape[1])],
        "scene_um_per_px_xy": list(scene_um) if scene_um is not None else None,
        "depth_start_scene_xy": [float(start[0]), float(start[1])],
        "depth_end_scene_xy": [float(end[0]), float(end[1])],
        "width_vector_scene_xy": [float(normal[0] * half), float(normal[1] * half)],
        "width_px": float(geom["width_px"]),
        "width_um": width_um,
        "depth_length_px": float(geom["length_px"]),
        "depth_length_um": depth_length_um,
        "oriented_rectangle_corners_scene_xy": [[float(x0), float(y0)] for x0, y0 in corners],
        "sampling": {
            "depth_samples": int(depth_samples),
            "width_samples": int(width_samples),
            "window_fraction": float(window_fraction),
            "window_size_samples": int(win),
            "width_compression": "median_iqr",
            "depth_fraction_range": [0.0, 1.0],
            "width_offset_px_range": [-float(half), float(half)],
        },
    }
    return Step8DepthProfileProbeResult(
        annotation_payload=payload,
        profiles_rows=profiles_rows,
        windows_rows=windows_rows,
        scene_overlay_rgb=scene_overlay,
        crop_raw_rgb=strip,
        crop_annotated_rgb=crop_annotated,
        crop_original_bbox_rgb=crop_bbox,
        profile_plot_rgb=plot,
    )


def save_step8_depth_profile_probe_result(
    export_dir: Path,
    result: Step8DepthProfileProbeResult,
) -> dict[str, str]:
    out_dir = Path(export_dir)
    image_dir = out_dir / "step8_depth_profile_probe_images"
    image_dir.mkdir(parents=True, exist_ok=True)
    probe_id = str(result.annotation_payload.get("probe_id") or "probe")
    safe_id = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in probe_id)
    overlay_path = image_dir / f"{safe_id}_scene_overlay.png"
    raw_path = image_dir / f"{safe_id}_crop_raw.png"
    annotated_path = image_dir / f"{safe_id}_crop_annotated.png"
    bbox_path = image_dir / f"{safe_id}_crop_original_bbox.png"
    plot_path = image_dir / f"{safe_id}_profile_plot.png"
    for path, arr in (
        (overlay_path, result.scene_overlay_rgb),
        (raw_path, result.crop_raw_rgb),
        (annotated_path, result.crop_annotated_rgb),
        (bbox_path, result.crop_original_bbox_rgb),
        (plot_path, result.profile_plot_rgb),
    ):
        cv2.imwrite(str(path), cv2.cvtColor(np.asarray(arr, dtype=np.uint8), cv2.COLOR_RGB2BGR))
    annotation_path = out_dir / "step8_depth_profile_probe_annotation.json"
    profiles_path = out_dir / "step8_depth_profile_probe_profiles.csv"
    windows_path = out_dir / "step8_depth_profile_probe_windows.csv"
    manifest_path = out_dir / "step8_depth_profile_probe_manifest.json"
    existing_annotations: list[dict[str, Any]] = []
    if annotation_path.exists():
        try:
            payload = _read_json(annotation_path)
            existing_annotations = [dict(row) for row in list(payload.get("probes") or []) if isinstance(row, dict)]
        except Exception:
            existing_annotations = []
    annotation = dict(result.annotation_payload)
    annotation["outputs"] = {
        "scene_overlay_png": str(overlay_path),
        "crop_raw_png": str(raw_path),
        "crop_annotated_png": str(annotated_path),
        "crop_original_bbox_png": str(bbox_path),
        "profile_plot_png": str(plot_path),
        "profiles_csv": str(profiles_path),
        "windows_csv": str(windows_path),
    }
    existing_annotations = [row for row in existing_annotations if str(row.get("probe_id") or "") != probe_id]
    existing_annotations.append(annotation)
    _write_json(annotation_path, {"schema": "step8_depth_profile_probe_annotation_collection_v1", "probes": existing_annotations})
    profile_fields = list(result.profiles_rows[0].keys()) if result.profiles_rows else ["probe_id"]
    window_fields = list(result.windows_rows[0].keys()) if result.windows_rows else ["probe_id"]
    existing_profile_rows: list[dict[str, Any]] = []
    if profiles_path.exists():
        try:
            with profiles_path.open("r", newline="", encoding="utf-8") as f:
                existing_profile_rows = [dict(row) for row in csv.DictReader(f) if str(row.get("probe_id") or "") != probe_id]
        except Exception:
            existing_profile_rows = []
    existing_window_rows: list[dict[str, Any]] = []
    if windows_path.exists():
        try:
            with windows_path.open("r", newline="", encoding="utf-8") as f:
                existing_window_rows = [dict(row) for row in csv.DictReader(f) if str(row.get("probe_id") or "") != probe_id]
        except Exception:
            existing_window_rows = []
    _write_csv_rows(profiles_path, existing_profile_rows + result.profiles_rows, profile_fields)
    _write_csv_rows(windows_path, existing_window_rows + result.windows_rows, window_fields)
    manifest_entries = []
    for row in existing_annotations:
        outputs = dict(row.get("outputs") or {})
        manifest_entries.append(
            {
                "probe_id": str(row.get("probe_id") or ""),
                "saved_at_utc": str(row.get("saved_at_utc") or ""),
                "source_image_path": str(row.get("source_image_path") or ""),
                "scene_overlay_png": str(outputs.get("scene_overlay_png") or ""),
                "crop_raw_png": str(outputs.get("crop_raw_png") or ""),
                "crop_annotated_png": str(outputs.get("crop_annotated_png") or ""),
                "crop_original_bbox_png": str(outputs.get("crop_original_bbox_png") or ""),
                "profile_plot_png": str(outputs.get("profile_plot_png") or ""),
                "profiles_csv": str(outputs.get("profiles_csv") or profiles_path),
                "windows_csv": str(outputs.get("windows_csv") or windows_path),
            }
        )
    _write_json(
        manifest_path,
        {
            "schema": "step8_depth_profile_probe_manifest_v1",
            "saved_at_utc": datetime.now(timezone.utc).isoformat(),
            "probe_count": len(manifest_entries),
            "probes": manifest_entries,
        },
    )
    return {
        "annotation_json": str(annotation_path),
        "profiles_csv": str(profiles_path),
        "windows_csv": str(windows_path),
        "manifest_json": str(manifest_path),
        "scene_overlay_png": str(overlay_path),
        "crop_raw_png": str(raw_path),
        "crop_annotated_png": str(annotated_path),
        "crop_original_bbox_png": str(bbox_path),
        "profile_plot_png": str(plot_path),
    }

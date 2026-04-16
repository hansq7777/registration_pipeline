from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception as exc:  # pragma: no cover
    raise RuntimeError("matplotlib is required for representation probe plotting") from exc


REPO_ROOT = Path(__file__).resolve().parents[3]
TOOLS_ROOT = REPO_ROOT / "registration_pipeline" / "histology" / "tools"
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

import run_confocal_grid_geometry_diagnostic as geom  # noqa: E402


OUT_ROOT = Path("/mnt/c/Users/Siqi/Desktop/REVIEW/20260408_confocal_representation_probe_2501_60")
GEOM_RUN_ROOT = Path("/mnt/c/Users/Siqi/Desktop/REVIEW/20260408_confocal_tile_diagnostic_2501_60")


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


def _load_best_scale() -> float:
    manifest_path = GEOM_RUN_ROOT / "run_manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    return float(payload["best_overall_scale"])


def _apply_clahe_u8(image_u8: np.ndarray, mask_u8: np.ndarray, *, clip_limit: float = 2.0, tile_grid: int = 8) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=(tile_grid, tile_grid))
    enhanced = clahe.apply(np.asarray(image_u8, dtype=np.uint8))
    out = np.full_like(enhanced, 255, dtype=np.uint8)
    out[np.asarray(mask_u8) > 0] = enhanced[np.asarray(mask_u8) > 0]
    return out


def _binary_otsu_dark_fibers_u8(image_u8: np.ndarray, mask_u8: np.ndarray) -> np.ndarray:
    valid = image_u8[np.asarray(mask_u8) > 0]
    if valid.size < 64:
        return np.full_like(image_u8, 255, dtype=np.uint8)
    thr, _ = cv2.threshold(valid.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    out = np.full_like(image_u8, 255, dtype=np.uint8)
    inside = np.asarray(mask_u8) > 0
    out[inside & (image_u8 <= thr)] = 0
    out[inside & (image_u8 > thr)] = 255
    return out


def _downscale_panel_u8(image_u8: np.ndarray, *, max_long_edge: int = 1600) -> np.ndarray:
    h, w = image_u8.shape[:2]
    scale = min(1.0, float(max_long_edge) / float(max(h, w)))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    out = cv2.resize(image_u8, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return geom.gray_preview_panel(out.astype(np.float32) / 255.0)


def _hist_panel(fixed_u8: np.ndarray, fixed_mask: np.ndarray, moving_u8: np.ndarray, moving_mask: np.ndarray, *, title: str) -> np.ndarray:
    fig, ax = plt.subplots(figsize=(4.8, 3.6), dpi=160)
    fixed_vals = fixed_u8[np.asarray(fixed_mask) > 0]
    moving_vals = moving_u8[np.asarray(moving_mask) > 0]
    ax.hist(fixed_vals, bins=64, range=(0, 255), density=True, alpha=0.50, label="fixed")
    ax.hist(moving_vals, bins=64, range=(0, 255), density=True, alpha=0.50, label="moving")
    ax.set_title(title)
    ax.set_xlabel("intensity")
    ax.set_ylabel("density")
    ax.legend()
    fig.tight_layout()
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba()).copy()
    rgb = rgba[..., :3]
    plt.close(fig)
    return rgb


def _compose_method_overview(
    *,
    fixed_u8: np.ndarray,
    fixed_mask: np.ndarray,
    moving_u8: np.ndarray,
    moving_mask: np.ndarray,
    title: str,
) -> np.ndarray:
    fixed_panel = _downscale_panel_u8(fixed_u8)
    moving_panel = _downscale_panel_u8(moving_u8)
    hist_panel = _hist_panel(fixed_u8, fixed_mask, moving_u8, moving_mask, title=title)
    overlay_panel = np.full((320, 320, 3), 248, dtype=np.uint8)
    cv2.putText(overlay_panel, "Signal-only note", (72, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (40, 40, 40), 2, cv2.LINE_AA)
    cv2.putText(overlay_panel, "This overview is NOT", (48, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (180, 40, 40), 2, cv2.LINE_AA)
    cv2.putText(overlay_panel, "a geometric overlay.", (48, 146), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (180, 40, 40), 2, cv2.LINE_AA)
    cv2.putText(overlay_panel, "Use full_overlay.png", (42, 208), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (55, 55, 55), 2, cv2.LINE_AA)
    cv2.putText(overlay_panel, "and tile_zoom_storyboard.png", (18, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (55, 55, 55), 2, cv2.LINE_AA)
    cv2.putText(overlay_panel, "for spatial QC.", (82, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (55, 55, 55), 2, cv2.LINE_AA)
    panels = [moving_panel, fixed_panel, hist_panel, overlay_panel]
    max_h = max(p.shape[0] for p in panels)
    widths = [p.shape[1] for p in panels]
    pad = 14
    canvas = np.full((max_h + 52, sum(widths) + pad * (len(panels) + 1), 3), 245, dtype=np.uint8)
    cv2.putText(canvas, title, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (25, 25, 25), 2, cv2.LINE_AA)
    labels = ["Moving processed", "Fixed normalized", "Histogram", "QC note"]
    x = pad
    for label, panel in zip(labels, panels):
        y0 = 40 + (max_h - panel.shape[0]) // 2
        canvas[y0 : y0 + panel.shape[0], x : x + panel.shape[1]] = panel
        cv2.putText(canvas, label, (x, 40 + max_h + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (55, 55, 55), 1, cv2.LINE_AA)
        x += panel.shape[1] + pad
    return canvas


def _moving_contrast_stats(tile_warps: list[geom.TileWarp], fixed_u8: np.ndarray, fixed_mask: np.ndarray) -> dict[str, float]:
    moving_ranges: list[float] = []
    fixed_ranges: list[float] = []
    moving_stds: list[float] = []
    fixed_stds: list[float] = []
    for tw in tile_warps:
        mv = tw.warped_gray_patch[tw.warped_mask_patch > 0]
        y0, y1, x0, x1 = [int(v) for v in tw.warped_bbox_yxyx]
        fx = fixed_u8[y0:y1, x0:x1].astype(np.float32) / 255.0
        fm = fixed_mask[y0:y1, x0:x1] > 0
        fv = fx[fm]
        if mv.size >= 32:
            moving_ranges.append(float(np.percentile(mv, 95) - np.percentile(mv, 5)))
            moving_stds.append(float(np.std(mv)))
        if fv.size >= 32:
            fixed_ranges.append(float(np.percentile(fv, 95) - np.percentile(fv, 5)))
            fixed_stds.append(float(np.std(fv)))
    moving_range = float(np.mean(moving_ranges)) if moving_ranges else float("nan")
    fixed_range = float(np.mean(fixed_ranges)) if fixed_ranges else float("nan")
    moving_std = float(np.mean(moving_stds)) if moving_stds else float("nan")
    fixed_std = float(np.mean(fixed_stds)) if fixed_stds else float("nan")
    return {
        "moving_mean_p95_p5": moving_range,
        "fixed_mean_p95_p5": fixed_range,
        "contrast_ratio_range": float(moving_range / fixed_range) if np.isfinite(moving_range) and np.isfinite(fixed_range) and fixed_range > 1e-6 else float("nan"),
        "moving_mean_std": moving_std,
        "fixed_mean_std": fixed_std,
        "contrast_ratio_std": float(moving_std / fixed_std) if np.isfinite(moving_std) and np.isfinite(fixed_std) and fixed_std > 1e-6 else float("nan"),
    }


def _method_variants(
    moving_native_scaled_u8: np.ndarray,
    moving_mask_u8: np.ndarray,
    fixed_u8: np.ndarray,
    fixed_mask: np.ndarray,
) -> dict[str, np.ndarray]:
    inverted = geom._invert_confocal_u8(moving_native_scaled_u8)
    inv_pct = geom._masked_percentile_normalize_u8(inverted, moving_mask_u8)
    inv_pct_hist = geom._masked_histogram_match_u8(inv_pct, moving_mask_u8, fixed_u8, fixed_mask)
    inv_pct_clahe = _apply_clahe_u8(inv_pct, moving_mask_u8, clip_limit=2.5, tile_grid=8)
    inv_pct_clahe_hist = geom._masked_histogram_match_u8(inv_pct_clahe, moving_mask_u8, fixed_u8, fixed_mask)
    inv_binary = _binary_otsu_dark_fibers_u8(inv_pct, moving_mask_u8)
    return {
        "inverted_gray_percentile": inv_pct,
        "inverted_gray_percentile_histmatch": inv_pct_hist,
        "inverted_gray_percentile_clahe": inv_pct_clahe,
        "inverted_gray_percentile_clahe_histmatch": inv_pct_clahe_hist,
        "inverted_binary_otsu": inv_binary,
    }


def main() -> None:
    out_root = _ensure_dir(OUT_ROOT)
    figs_dir = _ensure_dir(out_root / "figures")
    process_dir = _ensure_dir(out_root / "process")

    inputs = geom._load_inputs()
    fixed_bundle = inputs["fixed_bundle"]
    fixed_mask = (fixed_bundle.labels == 1).astype(np.float32)
    if not np.any(fixed_mask > 0):
        fixed_mask = (fixed_bundle.labels > 0).astype(np.float32)
    fixed_native_u8 = cv2.cvtColor(fixed_bundle.rgb, cv2.COLOR_RGB2GRAY)
    fixed_norm_u8 = geom._masked_percentile_normalize_u8(fixed_native_u8, fixed_mask)
    moving_native_scaled_u8 = np.asarray(inputs["scaled_projection_u8"], dtype=np.uint8)
    moving_display_scaled_u8 = geom._invert_confocal_u8(moving_native_scaled_u8)
    moving_mask_u8 = np.asarray(inputs["scaled_signal_mask_u8"], dtype=np.uint8)
    tile_defs = geom._build_tile_defs(
        inputs["projection_bundle"].stitch_info,
        raw_shape_hw=inputs["raw_projection_u8"].shape[:2],
        scaled_shape_hw=inputs["scaled_projection_u8"].shape[:2],
    )
    best_scale = _load_best_scale()
    full_mat, anchor_info = geom._manual_affine_for_scale(
        moving_native_scaled_u8.shape[:2],
        fixed_bundle.rgb.shape[:2],
        scale=best_scale,
    )

    variants = _method_variants(moving_native_scaled_u8, moving_mask_u8, fixed_norm_u8, fixed_mask)
    summary_rows: list[dict[str, Any]] = []
    long_rows: list[dict[str, Any]] = []
    overview_panels: list[tuple[str, np.ndarray]] = []

    for method_name, moving_u8 in variants.items():
        method_dir = _ensure_dir(out_root / method_name)
        tile_rows, tile_warps = geom._collect_tile_results(
            label=method_name,
            moving_reg_projection_u8=moving_u8,
            moving_signal_mask_u8=moving_mask_u8,
            fixed_gray_full=fixed_norm_u8,
            fixed_mask_full=fixed_mask,
            tile_defs=tile_defs,
            full_mat=full_mat,
        )
        _display_rows, display_tile_warps = geom._collect_tile_results(
            label=f"{method_name}_display_native",
            moving_reg_projection_u8=moving_display_scaled_u8,
            moving_signal_mask_u8=moving_mask_u8,
            fixed_gray_full=fixed_norm_u8,
            fixed_mask_full=fixed_mask,
            tile_defs=tile_defs,
            full_mat=full_mat,
        )
        for rec in tile_rows:
            rec["representation"] = method_name
            long_rows.append(rec)

        geom._make_tile_storyboard(
            fixed_norm_u8,
            fixed_mask,
            tile_warps,
            tile_rows,
            method_dir / "tile_zoom_storyboard.png",
            title_prefix=method_name,
            moving_display_tile_warps=display_tile_warps,
        )
        full_overlay = geom._draw_full_overlay(
            fixed_norm_u8,
            fixed_mask,
            tile_rows,
            tile_warps,
            title=method_name,
        )
        cv2.imwrite(str(method_dir / "full_overlay.png"), cv2.cvtColor(full_overlay, cv2.COLOR_RGB2BGR))
        overview = _compose_method_overview(
            fixed_u8=fixed_norm_u8,
            fixed_mask=fixed_mask,
            moving_u8=moving_u8,
            moving_mask=moving_mask_u8,
            title=method_name,
        )
        cv2.imwrite(str(method_dir / "signal_overview.png"), cv2.cvtColor(overview, cv2.COLOR_RGB2BGR))
        overview_panels.append((method_name, overview))

        cc = np.array([float(r["current_cc"]) for r in tile_rows], dtype=np.float64)
        mi = np.array([float(r["current_mi"]) for r in tile_rows], dtype=np.float64)
        dx = np.array([float(r["dx_star_px"]) for r in tile_rows], dtype=np.float64)
        dy = np.array([float(r["dy_star_px"]) for r in tile_rows], dtype=np.float64)
        first_col = [r for r in tile_rows if int(r["col"]) == 0]
        last_col = [r for r in tile_rows if int(r["col"]) == 5]
        contrast = _moving_contrast_stats(tile_warps, fixed_norm_u8, fixed_mask)
        summary_rows.append(
            {
                "representation": method_name,
                "mean_tile_cc": float(np.nanmean(cc)),
                "mean_tile_mi": float(np.nanmean(mi)),
                "mean_abs_dx": float(np.nanmean(np.abs(dx))),
                "mean_abs_dy": float(np.nanmean(np.abs(dy))),
                "first_col_mean_cc": float(np.nanmean([float(r["current_cc"]) for r in first_col])),
                "last_col_mean_cc": float(np.nanmean([float(r["current_cc"]) for r in last_col])),
                "rightmost_mean_dx": float(np.nanmean([float(r["dx_star_px"]) for r in last_col])),
                **contrast,
            }
        )

    _write_csv(process_dir / "representation_tile_metrics.csv", long_rows)
    _write_csv(process_dir / "representation_summary.csv", summary_rows)

    # Summary plots
    methods = [r["representation"] for r in summary_rows]
    mean_cc = [float(r["mean_tile_cc"]) for r in summary_rows]
    right_cc = [float(r["last_col_mean_cc"]) for r in summary_rows]
    contrast_ratio = [float(r["contrast_ratio_range"]) for r in summary_rows]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.8), dpi=170)
    ax[0].bar(methods, mean_cc)
    ax[0].set_title("Mean tile CC")
    ax[0].tick_params(axis="x", rotation=20)
    ax[1].bar(methods, right_cc, color="tab:orange")
    ax[1].set_title("Rightmost column mean CC")
    ax[1].tick_params(axis="x", rotation=20)
    ax[2].bar(methods, contrast_ratio, color="tab:red")
    ax[2].set_title("Moving/fixed contrast ratio")
    ax[2].tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(figs_dir / "representation_comparison.png")
    plt.close(fig)

    max_w = max(p.shape[1] for _label, p in overview_panels)
    total_h = sum(p.shape[0] + 28 for _label, p in overview_panels) + 10
    canvas = np.full((total_h, max_w, 3), 248, dtype=np.uint8)
    y = 8
    for label, panel in overview_panels:
        cv2.putText(canvas, label, (10, y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (30, 30, 30), 2, cv2.LINE_AA)
        y += 24
        canvas[y : y + panel.shape[0], : panel.shape[1], :] = panel
        y += panel.shape[0] + 4
    cv2.imwrite(str(figs_dir / "signal_overview_contact_sheet.png"), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))

    ranked = sorted(summary_rows, key=lambda r: float(r["mean_tile_cc"]), reverse=True)
    lines = [
        "# 2501_60 Representation Probe",
        "",
        f"- geometry fixed at anchor-preserving coarse state with best scale `{best_scale:.3f}` from prior diagnostic",
        "- anchor A1/B1 remains fixed through this probe",
        "- compared only input representation; geometry was held constant",
        "",
        "## Preprocessing arms",
    ]
    for method in methods:
        lines.append(f"- `{method}`")
    lines.extend(["", "## Ranking by mean tile CC"])
    for row in ranked:
        lines.append(
            f"- `{row['representation']}`: mean_tile_cc={row['mean_tile_cc']:.4f}, "
            f"right_col_cc={row['last_col_mean_cc']:.4f}, "
            f"contrast_ratio_range={row['contrast_ratio_range']:.3f}, "
            f"mean_abs_dx={row['mean_abs_dx']:.2f}"
        )
    lines.extend(
        [
            "",
            "## Files",
            "- `process/representation_summary.csv`",
            "- `process/representation_tile_metrics.csv`",
            "- `figures/representation_comparison.png`",
            "- `figures/signal_overview_contact_sheet.png`",
            "- per-method `signal_overview.png`",
            "- per-method `full_overlay.png`",
            "- per-method `tile_zoom_storyboard.png`",
        ]
    )
    (out_root / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    _write_json(
        out_root / "run_manifest.json",
        {
            "best_scale": best_scale,
            "manual_state": geom.MANUAL_STATE,
            "anchor_pair": geom.ANCHOR_PAIR,
            "representations": methods,
            "paths": {
                "summary_md": str(out_root / "summary.md"),
                "representation_summary_csv": str(process_dir / "representation_summary.csv"),
                "representation_tile_metrics_csv": str(process_dir / "representation_tile_metrics.csv"),
                "representation_comparison_png": str(figs_dir / "representation_comparison.png"),
                "signal_overview_contact_sheet_png": str(figs_dir / "signal_overview_contact_sheet.png"),
            },
        },
    )
    print(f"Representation probe written to: {out_root}")


if __name__ == "__main__":
    main()

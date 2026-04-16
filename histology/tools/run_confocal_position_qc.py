from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

import run_confocal_grid_geometry_diagnostic as geom


OUT_ROOT = Path("/mnt/c/Users/Siqi/Desktop/REVIEW/20260408_confocal_position_qc_2501_60")
GEOM_RUN_ROOT = Path("/mnt/c/Users/Siqi/Desktop/REVIEW/20260408_confocal_tile_diagnostic_2501_60")


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _load_best_overall_scale() -> float:
    manifest_path = GEOM_RUN_ROOT / "run_manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    return float(payload["best_overall_scale"])


def _prepare_common_inputs() -> dict[str, Any]:
    inputs = geom._load_inputs()
    fixed_bundle = inputs["fixed_bundle"]
    fixed_mask = (fixed_bundle.labels == 1).astype(np.float32)
    if not np.any(fixed_mask > 0):
        fixed_mask = (fixed_bundle.labels > 0).astype(np.float32)
    fixed_native_u8 = cv2.cvtColor(fixed_bundle.rgb, cv2.COLOR_RGB2GRAY)
    fixed_norm_u8 = geom._masked_percentile_normalize_u8(fixed_native_u8, fixed_mask)

    moving_native_scaled_u8 = np.asarray(inputs["scaled_projection_u8"], dtype=np.uint8)
    moving_mask_u8 = np.asarray(inputs["scaled_signal_mask_u8"], dtype=np.uint8)
    moving_display_scaled_u8 = geom._invert_confocal_u8(moving_native_scaled_u8)
    moving_inverted = geom._invert_confocal_u8(moving_native_scaled_u8)
    moving_inv_pct = geom._masked_percentile_normalize_u8(moving_inverted, moving_mask_u8)

    tile_defs = geom._build_tile_defs(
        inputs["projection_bundle"].stitch_info,
        raw_shape_hw=inputs["raw_projection_u8"].shape[:2],
        scaled_shape_hw=inputs["scaled_projection_u8"].shape[:2],
    )
    return {
        "inputs": inputs,
        "fixed_mask": fixed_mask,
        "fixed_norm_u8": fixed_norm_u8,
        "moving_native_scaled_u8": moving_native_scaled_u8,
        "moving_display_scaled_u8": moving_display_scaled_u8,
        "moving_mask_u8": moving_mask_u8,
        "moving_inv_pct_u8": moving_inv_pct,
        "tile_defs": tile_defs,
    }


def _variant_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    dx = np.asarray([float(r["dx_star_px"]) for r in rows], dtype=np.float64)
    cc = np.asarray([float(r["current_cc"]) for r in rows], dtype=np.float64)
    right = [r for r in rows if int(r["col"]) == 5]
    return {
        "mean_tile_cc": float(np.nanmean(cc)),
        "mean_abs_dx": float(np.nanmean(np.abs(dx))),
        "rightmost_mean_dx": float(np.nanmean([float(r["dx_star_px"]) for r in right])),
        "row0_dx": [float(r["dx_star_px"]) for r in sorted([r for r in rows if int(r["row"]) == 0], key=lambda r: int(r["col"]))],
        "row1_dx": [float(r["dx_star_px"]) for r in sorted([r for r in rows if int(r["row"]) == 1], key=lambda r: int(r["col"]))],
        "row2_dx": [float(r["dx_star_px"]) for r in sorted([r for r in rows if int(r["row"]) == 2], key=lambda r: int(r["col"]))],
    }


def _collect_variant(
    *,
    name: str,
    scale: float,
    moving_reg_projection_u8: np.ndarray,
    moving_signal_mask_u8: np.ndarray,
    fixed_gray_full: np.ndarray,
    fixed_mask_full: np.ndarray,
    tile_defs: list[geom.TileDef],
    per_tile_shift: dict[tuple[int, int], tuple[float, float]] | None = None,
) -> dict[str, Any]:
    full_mat, anchor_info = geom._manual_affine_for_scale(
        moving_reg_projection_u8.shape[:2],
        fixed_gray_full.shape[:2],
        scale=scale,
    )
    rows, tile_warps = geom._collect_tile_results(
        label=name,
        moving_reg_projection_u8=moving_reg_projection_u8,
        moving_signal_mask_u8=moving_signal_mask_u8,
        fixed_gray_full=fixed_gray_full,
        fixed_mask_full=fixed_mask_full,
        tile_defs=tile_defs,
        full_mat=full_mat,
        per_tile_shift=per_tile_shift,
    )
    return {
        "name": name,
        "scale": float(scale),
        "full_mat": full_mat,
        "anchor_info": anchor_info,
        "rows": rows,
        "tile_warps": tile_warps,
        "summary": _variant_summary(rows),
    }


def _save_full_overlay(variant: dict[str, Any], fixed_gray_full_u8: np.ndarray, fixed_mask_full: np.ndarray, out_path: Path) -> None:
    overlay = geom._draw_full_overlay(
        fixed_gray_full_u8,
        fixed_mask_full,
        variant["rows"],
        variant["tile_warps"],
        title=variant["name"],
    )
    cv2.imwrite(str(out_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))


def _save_tile_storyboard(
    variant: dict[str, Any],
    fixed_gray_full_u8: np.ndarray,
    fixed_mask_full: np.ndarray,
    out_path: Path,
    *,
    moving_display_tile_warps: list[geom.TileWarp] | None = None,
) -> None:
    geom._make_tile_storyboard(
        fixed_gray_full_u8,
        fixed_mask_full,
        variant["tile_warps"],
        variant["rows"],
        out_path,
        title_prefix=variant["name"],
        moving_display_tile_warps=moving_display_tile_warps,
    )


def _current_overlay_square(
    fixed_gray_full_u8: np.ndarray,
    fixed_mask_full: np.ndarray,
    tw: geom.TileWarp,
    rec: dict[str, Any],
    *,
    side: int = 260,
) -> np.ndarray:
    full_y0, full_y1, full_x0, full_x1 = [int(v) for v in tw.warped_full_bbox_yxyx]
    rec_search = rec.get("search_bbox_yxyx")
    if isinstance(rec_search, list) and len(rec_search) == 4:
        search_bbox = (
            min(int(rec_search[0]), full_y0),
            max(int(rec_search[1]), full_y1),
            min(int(rec_search[2]), full_x0),
            max(int(rec_search[3]), full_x1),
        )
    else:
        search_bbox = (full_y0, full_y1, full_x0, full_x1)
    fixed_search = fixed_gray_full_u8[search_bbox[0] : search_bbox[1], search_bbox[2] : search_bbox[3]].astype(np.float32) / 255.0
    fixed_search_mask = fixed_mask_full[search_bbox[0] : search_bbox[1], search_bbox[2] : search_bbox[3]]
    current_canvas = np.ones_like(fixed_search, dtype=np.float32)
    current_mask = np.zeros_like(fixed_search_mask, dtype=np.float32)
    cy0, cy1, cx0, cx1 = [int(v) for v in tw.warped_full_bbox_yxyx]
    off_y0 = cy0 - search_bbox[0]
    off_y1 = off_y0 + tw.warped_gray_full_patch.shape[0]
    off_x0 = cx0 - search_bbox[2]
    off_x1 = off_x0 + tw.warped_gray_full_patch.shape[1]
    current_canvas[off_y0:off_y1, off_x0:off_x1] = np.where(tw.warped_mask_full_patch > 0, tw.warped_gray_full_patch, 1.0)
    current_mask[off_y0:off_y1, off_x0:off_x1] = tw.warped_mask_full_patch
    overlay = geom.overlay_preview(fixed_search, current_canvas, fixed_search_mask, current_mask)
    return geom._square_panel(overlay, side=side)


def _save_variant_contact_sheet(
    variants: list[dict[str, Any]],
    *,
    fixed_gray_full_u8: np.ndarray,
    fixed_mask_full: np.ndarray,
    tile_row: int,
    out_path: Path,
) -> None:
    cols = [0, 1, 2, 3, 4, 5]
    side = 220
    top_pad = 54
    left_pad = 170
    gap = 12
    width = left_pad + gap + len(cols) * (side + gap)
    height = top_pad + len(variants) * (side + 34 + gap)
    canvas = np.full((height, width, 3), 246, dtype=np.uint8)
    title = f"Geometry-faithful top-row QC (row {tile_row})"
    cv2.putText(canvas, title, (18, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (25, 25, 25), 2, cv2.LINE_AA)
    cv2.putText(canvas, "same fixed scene coordinates; only geometry variant changes", (18, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (80, 80, 80), 1, cv2.LINE_AA)
    x = left_pad
    for col in cols:
        cv2.putText(canvas, f"c{col}", (x + side // 2 - 16, top_pad - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (55, 55, 55), 2, cv2.LINE_AA)
        x += side + gap
    for ridx, variant in enumerate(variants):
        y = top_pad + ridx * (side + 34 + gap)
        cv2.putText(canvas, variant["name"], (18, y + side // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (35, 35, 35), 2, cv2.LINE_AA)
        cv2.putText(
            canvas,
            f"CC={variant['summary']['mean_tile_cc']:.3f} | right dx={variant['summary']['rightmost_mean_dx']:.2f}",
            (18, y + side // 2 + 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (95, 95, 95),
            1,
            cv2.LINE_AA,
        )
        row_map = {(int(r["row"]), int(r["col"])): r for r in variant["rows"]}
        warp_map = {(tw.tile.row, tw.tile.col): tw for tw in variant["tile_warps"]}
        x = left_pad
        for col in cols:
            rec = row_map[(tile_row, col)]
            tw = warp_map[(tile_row, col)]
            panel = _current_overlay_square(fixed_gray_full_u8, fixed_mask_full, tw, rec, side=side)
            canvas[y : y + side, x : x + side] = panel
            cv2.putText(
                canvas,
                f"dx*={float(rec['dx_star_px']):.1f}",
                (x + 8, y + side - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 215, 0),
                1,
                cv2.LINE_AA,
            )
            x += side + gap
    cv2.imwrite(str(out_path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))


def _save_full_contact_sheet(variants: list[dict[str, Any]], fixed_gray_full_u8: np.ndarray, fixed_mask_full: np.ndarray, out_path: Path) -> None:
    panels: list[np.ndarray] = []
    for variant in variants:
        overlay = geom._draw_full_overlay(
            fixed_gray_full_u8,
            fixed_mask_full,
            variant["rows"],
            variant["tile_warps"],
            title=variant["name"],
        )
        side_w = 900
        scale = min(1.0, float(side_w) / float(overlay.shape[1]))
        new_w = max(1, int(round(overlay.shape[1] * scale)))
        new_h = max(1, int(round(overlay.shape[0] * scale)))
        panels.append(cv2.resize(overlay, (new_w, new_h), interpolation=cv2.INTER_AREA))
    pad = 16
    width = max(p.shape[1] for p in panels) + pad * 2
    height = sum(p.shape[0] for p in panels) + pad * (len(panels) + 1)
    canvas = np.full((height, width, 3), 246, dtype=np.uint8)
    y = pad
    for panel in panels:
        x = (width - panel.shape[1]) // 2
        canvas[y : y + panel.shape[0], x : x + panel.shape[1]] = panel
        y += panel.shape[0] + pad
    cv2.imwrite(str(out_path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))


def _compose_local_overlay_crop(
    variant: dict[str, Any],
    fixed_gray_full_u8: np.ndarray,
    fixed_mask_full: np.ndarray,
    *,
    margin_px: int = 80,
) -> np.ndarray:
    y0 = min(int(r["warped_bbox_yxyx"][0]) for r in variant["rows"])
    y1 = max(int(r["warped_bbox_yxyx"][1]) for r in variant["rows"])
    x0 = min(int(r["warped_bbox_yxyx"][2]) for r in variant["rows"])
    x1 = max(int(r["warped_bbox_yxyx"][3]) for r in variant["rows"])
    y0 = max(0, y0 - margin_px)
    x0 = max(0, x0 - margin_px)
    y1 = min(fixed_gray_full_u8.shape[0], y1 + margin_px)
    x1 = min(fixed_gray_full_u8.shape[1], x1 + margin_px)
    fixed_crop = fixed_gray_full_u8[y0:y1, x0:x1].astype(np.float32) / 255.0
    fixed_mask_crop = fixed_mask_full[y0:y1, x0:x1]
    moving_canvas = np.ones_like(fixed_crop, dtype=np.float32)
    moving_mask = np.zeros_like(fixed_mask_crop, dtype=np.float32)
    for tw in variant["tile_warps"]:
        gy0, gy1, gx0, gx1 = [int(v) for v in tw.warped_full_bbox_yxyx]
        oy0 = gy0 - y0
        oy1 = oy0 + tw.warped_gray_full_patch.shape[0]
        ox0 = gx0 - x0
        ox1 = ox0 + tw.warped_gray_full_patch.shape[1]
        moving_canvas[oy0:oy1, ox0:ox1] = np.where(tw.warped_mask_full_patch > 0, tw.warped_gray_full_patch, moving_canvas[oy0:oy1, ox0:ox1])
        moving_mask[oy0:oy1, ox0:ox1] = np.maximum(moving_mask[oy0:oy1, ox0:ox1], tw.warped_mask_full_patch)
    overlay = geom.overlay_preview(fixed_crop, moving_canvas, fixed_mask_crop, moving_mask)
    return overlay


def _save_local_grid_neighborhood_contact_sheet(
    variants: list[dict[str, Any]],
    fixed_gray_full_u8: np.ndarray,
    fixed_mask_full: np.ndarray,
    out_path: Path,
) -> None:
    panels = [_compose_local_overlay_crop(v, fixed_gray_full_u8, fixed_mask_full) for v in variants]
    target_w = 1100
    resized: list[np.ndarray] = []
    for p in panels:
        scale = min(1.0, float(target_w) / float(p.shape[1]))
        new_w = max(1, int(round(p.shape[1] * scale)))
        new_h = max(1, int(round(p.shape[0] * scale)))
        resized.append(cv2.resize(p, (new_w, new_h), interpolation=cv2.INTER_AREA))
    pad = 18
    title_h = 58
    width = max(p.shape[1] for p in resized) + pad * 2
    height = title_h + sum(p.shape[0] for p in resized) + pad * (len(resized) + 1)
    canvas = np.full((height, width, 3), 246, dtype=np.uint8)
    cv2.putText(canvas, "Local 3x6 grid neighborhood overlays", (18, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.82, (25, 25, 25), 2, cv2.LINE_AA)
    cv2.putText(canvas, "geometry-faithful local scene crop around the whole grid", (18, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (80, 80, 80), 1, cv2.LINE_AA)
    y = title_h
    for variant, panel in zip(variants, resized):
        x = (width - panel.shape[1]) // 2
        canvas[y : y + panel.shape[0], x : x + panel.shape[1]] = panel
        cv2.putText(
            canvas,
            f"{variant['name']} | CC={variant['summary']['mean_tile_cc']:.3f} | right dx={variant['summary']['rightmost_mean_dx']:.2f}",
            (18, y + 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (35, 35, 35),
            2,
            cv2.LINE_AA,
        )
        y += panel.shape[0] + pad
    cv2.imwrite(str(out_path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))


def main() -> None:
    out_root = _ensure_dir(OUT_ROOT)
    figs_dir = _ensure_dir(out_root / "figures")
    process_dir = _ensure_dir(out_root / "process")

    common = _prepare_common_inputs()
    fixed_norm_u8 = common["fixed_norm_u8"]
    fixed_mask = common["fixed_mask"]
    moving_native_scaled_u8 = common["moving_native_scaled_u8"]
    moving_display_scaled_u8 = common["moving_display_scaled_u8"]
    moving_inv_pct_u8 = common["moving_inv_pct_u8"]
    moving_mask_u8 = common["moving_mask_u8"]
    tile_defs = common["tile_defs"]

    manual_scale = float(geom.MANUAL_STATE["scale"])
    best_scale = _load_best_overall_scale()

    manual_variant = _collect_variant(
        name="manual_exact_scale_0.943_anchor_fixed",
        scale=manual_scale,
        moving_reg_projection_u8=moving_inv_pct_u8,
        moving_signal_mask_u8=moving_mask_u8,
        fixed_gray_full=fixed_norm_u8,
        fixed_mask_full=fixed_mask,
        tile_defs=tile_defs,
    )
    best_scale_variant = _collect_variant(
        name=f"best_uniform_scale_{best_scale:.3f}_anchor_fixed",
        scale=best_scale,
        moving_reg_projection_u8=moving_inv_pct_u8,
        moving_signal_mask_u8=moving_mask_u8,
        fixed_gray_full=fixed_norm_u8,
        fixed_mask_full=fixed_mask,
        tile_defs=tile_defs,
    )
    col_shifts = geom._column_smooth_offsets(best_scale_variant["rows"])
    col_variant = _collect_variant(
        name=f"best_scale_{best_scale:.3f}_plus_columnwise",
        scale=best_scale,
        moving_reg_projection_u8=moving_inv_pct_u8,
        moving_signal_mask_u8=moving_mask_u8,
        fixed_gray_full=fixed_norm_u8,
        fixed_mask_full=fixed_mask,
        tile_defs=tile_defs,
        per_tile_shift=col_shifts,
    )
    smooth_shifts = geom._tilewise_smooth_offsets(best_scale_variant["rows"])
    smooth_variant = _collect_variant(
        name=f"best_scale_{best_scale:.3f}_plus_tilewise_smooth",
        scale=best_scale,
        moving_reg_projection_u8=moving_inv_pct_u8,
        moving_signal_mask_u8=moving_mask_u8,
        fixed_gray_full=fixed_norm_u8,
        fixed_mask_full=fixed_mask,
        tile_defs=tile_defs,
        per_tile_shift=smooth_shifts,
    )

    variants = [manual_variant, best_scale_variant, col_variant, smooth_variant]

    for variant in variants:
        variant_dir = _ensure_dir(out_root / variant["name"])
        _save_full_overlay(variant, fixed_norm_u8, fixed_mask, variant_dir / "full_overlay.png")
        _display_rows, display_tile_warps = geom._collect_tile_results(
            label=f"{variant['name']}_display_native",
            moving_reg_projection_u8=moving_display_scaled_u8,
            moving_signal_mask_u8=moving_mask_u8,
            fixed_gray_full=fixed_norm_u8,
            fixed_mask_full=fixed_mask,
            tile_defs=tile_defs,
            full_mat=variant["full_mat"],
        )
        _save_tile_storyboard(
            variant,
            fixed_norm_u8,
            fixed_mask,
            variant_dir / "tile_zoom_storyboard.png",
            moving_display_tile_warps=display_tile_warps,
        )
        _write_json(
            variant_dir / "variant_metrics.json",
            {
                "name": variant["name"],
                "scale": variant["scale"],
                "summary": variant["summary"],
                "anchor_info": variant["anchor_info"],
            },
        )

    _save_full_contact_sheet(variants, fixed_norm_u8, fixed_mask, figs_dir / "full_overlay_variants.png")
    _save_local_grid_neighborhood_contact_sheet(
        variants,
        fixed_gray_full_u8=fixed_norm_u8,
        fixed_mask_full=fixed_mask,
        out_path=figs_dir / "local_grid_neighborhood_variants.png",
    )
    _save_variant_contact_sheet(
        variants,
        fixed_gray_full_u8=fixed_norm_u8,
        fixed_mask_full=fixed_mask,
        tile_row=0,
        out_path=figs_dir / "top_row_left_to_right_comparison.png",
    )
    _save_variant_contact_sheet(
        variants,
        fixed_gray_full_u8=fixed_norm_u8,
        fixed_mask_full=fixed_mask,
        tile_row=1,
        out_path=figs_dir / "middle_row_left_to_right_comparison.png",
    )

    process_rows: list[dict[str, Any]] = []
    for variant in variants:
        for row in variant["rows"]:
            out = dict(row)
            out["variant"] = variant["name"]
            process_rows.append(out)
    geom._write_csv(process_dir / "tile_metrics_all_variants.csv", process_rows)

    summary_lines = [
        "# Position QC Summary",
        "",
        "This folder is intentionally geometry-only. It does not reuse the misleading signal-overview overlay from the representation probe.",
        "",
        "Key interpretation:",
        "- `signal_overview.png` in the representation probe is not scene-faithful. It resizes the fixed image to the moving image size for signal-distribution comparison only.",
        "- The true geometry-faithful views are the overlays and tile storyboards generated here.",
        "- The 10% overlap itself is not the likely cause of the apparent large mismatch. Grid recovery was already validated from CZI metadata and earlier raw-grid QC.",
        "",
        "Likely reasons the representation-probe folder looked wrong:",
        "- You may have looked at `signal_overview.png`, which is not a spatial overlay.",
        f"- The representation probe used `best uniform scale = {best_scale:.3f}`, not your exact manual `0.943`, so it was already a different geometry state.",
        "- There is a real rightward drift in the current geometry, but it is smaller and structured, not the catastrophic mismatch suggested by the signal overview panel.",
        "",
        "Variant summary:",
    ]
    for variant in variants:
        summary_lines.extend(
            [
                f"- `{variant['name']}`",
                f"  - mean tile CC: {variant['summary']['mean_tile_cc']:.4f}",
                f"  - mean |dx*|: {variant['summary']['mean_abs_dx']:.2f}",
                f"  - rightmost mean dx*: {variant['summary']['rightmost_mean_dx']:.2f}",
                f"  - row0 dx*: {variant['summary']['row0_dx']}",
            ]
        )
    (out_root / "summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    _write_json(
        out_root / "run_manifest.json",
        {
            "manual_state": geom.MANUAL_STATE,
            "anchor_pair": geom.ANCHOR_PAIR,
            "representation_used": "inverted_gray_percentile",
            "best_overall_scale_from_previous_geometry_diagnostic": best_scale,
            "variants": [
                {
                    "name": variant["name"],
                    "scale": variant["scale"],
                    "summary": variant["summary"],
                    "variant_dir": str(out_root / variant["name"]),
                }
                for variant in variants
            ],
            "figures": {
                "full_overlay_variants": str(figs_dir / "full_overlay_variants.png"),
                "local_grid_neighborhood_variants": str(figs_dir / "local_grid_neighborhood_variants.png"),
                "top_row_left_to_right_comparison": str(figs_dir / "top_row_left_to_right_comparison.png"),
                "middle_row_left_to_right_comparison": str(figs_dir / "middle_row_left_to_right_comparison.png"),
            },
            "process": {
                "tile_metrics_all_variants_csv": str(process_dir / "tile_metrics_all_variants.csv"),
            },
        },
    )


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
import math
import sys
import subprocess
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
TOOLS_ROOT = Path(__file__).resolve().parent
GUI_MVP_ROOT = REPO_ROOT / "registration_pipeline" / "histology" / "gui_mvp"
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))
if str(GUI_MVP_ROOT) not in sys.path:
    sys.path.insert(0, str(GUI_MVP_ROOT))

import run_confocal_grid_geometry_diagnostic as base
from hitl_gui.application.confocal_registration import ConfocalRigidConfig, STEP7_TARGET_UM_PER_PX, run_confocal_rigid_registration
from hitl_gui.application.pair_registration import find_ants_bin


OUT_ROOT = Path("/mnt/c/Users/Siqi/Desktop/REVIEW/20260409_confocal_column_similarity_post_probe_2501_60")


def _agg(rows: list[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        return {
            "mean_tile_cc": float("nan"),
            "mean_abs_dx": float("nan"),
            "mean_abs_dy": float("nan"),
            "rightmost_mean_cc": float("nan"),
            "rightmost_mean_abs_dx": float("nan"),
        }
    rightmost_col = max(int(r["col"]) for r in rows)
    rightmost = [r for r in rows if int(r["col"]) == rightmost_col]
    return {
        "mean_tile_cc": float(np.nanmean([float(r["current_cc"]) for r in rows])),
        "mean_abs_dx": float(np.nanmean(np.abs([float(r["dx_star_px"]) for r in rows]))),
        "mean_abs_dy": float(np.nanmean(np.abs([float(r["dy_star_px"]) for r in rows]))),
        "rightmost_mean_cc": float(np.nanmean([float(r["current_cc"]) for r in rightmost])) if rightmost else float("nan"),
        "rightmost_mean_abs_dx": float(np.nanmean(np.abs([float(r["dx_star_px"]) for r in rightmost]))) if rightmost else float("nan"),
    }


def _compose_full_from_tile_warps(
    tile_warps: list[base.TileWarp],
    *,
    fixed_shape_hw: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    accum = np.zeros(fixed_shape_hw, dtype=np.float32)
    counts = np.zeros(fixed_shape_hw, dtype=np.float32)
    support = np.zeros(fixed_shape_hw, dtype=np.float32)
    for tw in tile_warps:
        y0, y1, x0, x1 = [int(v) for v in tw.warped_full_bbox_yxyx]
        patch = np.asarray(tw.warped_gray_full_patch, dtype=np.float32)
        mask = np.asarray(tw.warped_mask_full_patch, dtype=np.float32) > 0
        target_accum = accum[y0:y1, x0:x1]
        target_counts = counts[y0:y1, x0:x1]
        target_support = support[y0:y1, x0:x1]
        target_accum[mask] += patch[mask]
        target_counts[mask] += 1.0
        target_support[mask] = 1.0
    full_gray = np.where(counts > 0, accum / np.maximum(counts, 1.0), 0.0).astype(np.float32)
    full_mask = (support > 0).astype(np.uint8) * 255
    full_u8 = np.clip(np.round(np.clip(full_gray, 0.0, 1.0) * 255.0), 0, 255).astype(np.uint8)
    return full_u8, full_mask.astype(np.uint8)


def _run_composite_weak_similarity_post(
    *,
    out_root: Path,
    inputs: dict[str, Any],
    moving_composite_u8: np.ndarray,
    moving_composite_mask_u8: np.ndarray,
) -> tuple[dict[str, Any], dict[str, float]]:
    ants_bin = find_ants_bin()
    if ants_bin is None:
        raise RuntimeError("Could not locate local ANTs binaries")
    fixed_bundle = inputs["fixed_bundle"]
    item = inputs["item"]
    anchor_scene = tuple(float(v) for v in base.ANCHOR_PAIR["section_scene_xy"])
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
        confocal_projection_u8=np.asarray(moving_composite_u8, dtype=np.uint8),
        confocal_signal_mask_u8=np.asarray(moving_composite_mask_u8, dtype=np.uint8),
        ants_bin=ants_bin,
        out_root=out_root,
        confocal_sources=list(inputs["confocal_paths"]),
        confocal_source_mode="column_shared_composite_proxy",
        nominal_overlap_fraction=0.0,
        projection_info={
            "source_um_per_px_xy": list(fixed_bundle.preview_um_per_px_xy or ()),
            "target_um_per_px_xy": list(fixed_bundle.preview_um_per_px_xy or ()),
            "stitch_info": inputs["projection_bundle"].stitch_info,
            "raw_projection_shape_hw": list(moving_composite_u8.shape[:2]),
            "scaled_projection_shape_hw": list(moving_composite_u8.shape[:2]),
            "proxy_mode": "column_shared_composite_then_similarity_post",
        },
        projection_mode="focus",
        channel_index=0,
        local_refine_model="similarity",
        target_working_um_per_px=STEP7_TARGET_UM_PER_PX,
        invert_confocal_for_registration=False,
        tx_px=0.0,
        ty_px=0.0,
        angle_deg=0.0,
        scale=1.0,
        flip_lr=False,
        flip_ud=False,
        anchor_pairs=[
            {
                "index": 1,
                "section_scene_xy": [float(anchor_scene[0]), float(anchor_scene[1])],
                "confocal_raw_xy": [float(anchor_scene[0]), float(anchor_scene[1])],
                "confocal_scene_xy": [float(anchor_scene[0]), float(anchor_scene[1])],
            }
        ],
    )
    def _ants_cli_posix(path: Path | str) -> str:
        try:
            return subprocess.check_output(
                ["wslpath", "-w", str(Path(path))],
                text=True,
            ).strip()
        except Exception:
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

    old_pair_cli = base.pair_registration_mod.ants_cli_path
    old_pair_bin = base.pair_registration_mod.ants_binary_path
    old_conf_cli = base.confocal_registration_mod.ants_cli_path
    old_conf_bin = base.confocal_registration_mod.ants_binary_path
    base.pair_registration_mod.ants_cli_path = _ants_cli_posix
    base.pair_registration_mod.ants_binary_path = _ants_binary_posix
    base.confocal_registration_mod.ants_cli_path = _ants_cli_posix
    base.confocal_registration_mod.ants_binary_path = _ants_binary_posix
    t0 = time.perf_counter()
    try:
        manifest = run_confocal_rigid_registration(cfg)
    finally:
        base.pair_registration_mod.ants_cli_path = old_pair_cli
        base.pair_registration_mod.ants_binary_path = old_pair_bin
        base.confocal_registration_mod.ants_cli_path = old_conf_cli
        base.confocal_registration_mod.ants_binary_path = old_conf_bin
    wall_seconds = float(time.perf_counter() - t0)
    timing = dict(manifest.get("timing_seconds") or {})
    timing["wall_total"] = wall_seconds
    return manifest, timing


def _risk_summary(
    *,
    baseline_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    base_by_idx = {int(r["tile_index"]): r for r in baseline_rows}
    records: list[dict[str, Any]] = []
    worsened_cc = 0
    worsened_abs_dx = 0
    for row in test_rows:
        idx = int(row["tile_index"])
        base_row = base_by_idx[idx]
        delta_cc = float(row["current_cc"]) - float(base_row["current_cc"])
        delta_abs_dx = abs(float(row["dx_star_px"])) - abs(float(base_row["dx_star_px"]))
        delta_abs_dy = abs(float(row["dy_star_px"])) - abs(float(base_row["dy_star_px"]))
        if delta_cc < -0.01:
            worsened_cc += 1
        if delta_abs_dx > 1.0:
            worsened_abs_dx += 1
        records.append(
            {
                "tile_index": idx,
                "row": int(row["row"]),
                "col": int(row["col"]),
                "baseline_cc": float(base_row["current_cc"]),
                "test_cc": float(row["current_cc"]),
                "delta_cc": float(delta_cc),
                "baseline_abs_dx": float(abs(float(base_row["dx_star_px"]))),
                "test_abs_dx": float(abs(float(row["dx_star_px"]))),
                "delta_abs_dx": float(delta_abs_dx),
                "baseline_abs_dy": float(abs(float(base_row["dy_star_px"]))),
                "test_abs_dy": float(abs(float(row["dy_star_px"]))),
                "delta_abs_dy": float(delta_abs_dy),
            }
        )
    return {
        "tiles_worse_cc_gt_0p01": int(worsened_cc),
        "tiles_worse_abs_dx_gt_1px": int(worsened_abs_dx),
        "mean_delta_cc": float(np.nanmean([float(r["delta_cc"]) for r in records])) if records else float("nan"),
        "mean_delta_abs_dx": float(np.nanmean([float(r["delta_abs_dx"]) for r in records])) if records else float("nan"),
        "per_tile": records,
    }


def main() -> None:
    out_root = base._ensure_dir(OUT_ROOT)
    process_dir = base._ensure_dir(out_root / "process")
    figures_dir = base._ensure_dir(out_root / "figures")
    runs_dir = base._ensure_dir(out_root / "runs")

    t_all = time.perf_counter()
    inputs = base._load_inputs()
    fixed_bundle = inputs["fixed_bundle"]
    fixed_mask_full = (fixed_bundle.labels == 1).astype(np.float32)
    if not np.any(fixed_mask_full > 0):
        fixed_mask_full = (fixed_bundle.labels > 0).astype(np.float32)
    fixed_gray_native_u8 = cv2.cvtColor(fixed_bundle.rgb, cv2.COLOR_RGB2GRAY)
    moving_scaled_mask_u8 = inputs["scaled_signal_mask_u8"]
    fixed_gray_full_u8 = base._masked_percentile_normalize_u8(fixed_gray_native_u8, fixed_mask_full)
    moving_inverted_u8 = base._invert_confocal_u8(inputs["scaled_projection_u8"])
    moving_scaled_u8 = base._masked_percentile_normalize_u8(moving_inverted_u8, moving_scaled_mask_u8)
    moving_scaled_u8 = base._masked_histogram_match_u8(
        moving_scaled_u8,
        moving_scaled_mask_u8,
        fixed_gray_full_u8,
        fixed_mask_full,
    )
    tile_defs = base._build_tile_defs(
        inputs["projection_bundle"].stitch_info,
        raw_shape_hw=inputs["raw_projection_u8"].shape[:2],
        scaled_shape_hw=inputs["scaled_projection_u8"].shape[:2],
    )

    sweep_t0 = time.perf_counter()
    sweep_rows: list[dict[str, Any]] = []
    for scale in base.SCALE_VALUES:
        _full_mat, _anchor_info, rows, _tile_warps = base._run_single_scale_diagnostic(
            scale=scale,
            moving_reg_projection_u8=moving_scaled_u8,
            moving_signal_mask_u8=moving_scaled_mask_u8,
            fixed_gray_full=fixed_gray_full_u8,
            fixed_mask_full=fixed_mask_full,
            fixed_shape_hw=fixed_bundle.rgb.shape[:2],
            tile_defs=tile_defs,
        )
        sweep_rows.extend(rows)
    sweep_summary = base._aggregate_scale_sweep(sweep_rows)
    best_overall_scale, best_right_scale = base._best_scale_from_summary(sweep_summary)
    sweep_seconds = float(time.perf_counter() - sweep_t0)

    translation_t0 = time.perf_counter()
    best_full_mat, _best_anchor_info, best_rows, best_tile_warps = base._run_single_scale_diagnostic(
        scale=float(best_overall_scale),
        moving_reg_projection_u8=moving_scaled_u8,
        moving_signal_mask_u8=moving_scaled_mask_u8,
        fixed_gray_full=fixed_gray_full_u8,
        fixed_mask_full=fixed_mask_full,
        fixed_shape_hw=fixed_bundle.rgb.shape[:2],
        tile_defs=tile_defs,
    )
    translation_seconds = float(time.perf_counter() - translation_t0)
    for r in best_rows:
        r["method"] = "translation_only"

    column_t0 = time.perf_counter()
    column_offsets = base._column_smooth_offsets(best_rows)
    col_rows, col_tile_warps = base._collect_tile_results(
        label="column_shared_residual",
        moving_reg_projection_u8=moving_scaled_u8,
        moving_signal_mask_u8=moving_scaled_mask_u8,
        fixed_gray_full=fixed_gray_full_u8,
        fixed_mask_full=fixed_mask_full,
        tile_defs=tile_defs,
        full_mat=best_full_mat,
        per_tile_shift=column_offsets,
    )
    column_seconds = float(time.perf_counter() - column_t0)

    compose_t0 = time.perf_counter()
    composite_u8, composite_mask_u8 = _compose_full_from_tile_warps(
        col_tile_warps,
        fixed_shape_hw=fixed_bundle.rgb.shape[:2],
    )
    compose_seconds = float(time.perf_counter() - compose_t0)

    post_manifest, post_timing = _run_composite_weak_similarity_post(
        out_root=runs_dir,
        inputs=inputs,
        moving_composite_u8=composite_u8,
        moving_composite_mask_u8=composite_mask_u8,
    )
    post_tile_warps = base._refine_tile_warps_from_run(
        coarse_tile_warps=col_tile_warps,
        run_summary=post_manifest,
        fixed_shape_hw=fixed_bundle.rgb.shape[:2],
    )
    post_rows = [base._analyze_tile_warp(fixed_gray_full_u8, fixed_mask_full, tw) for tw in post_tile_warps]
    for r in post_rows:
        r["method"] = "column_then_weak_similarity_post"

    base._write_csv(process_dir / "tile_metrics_compare.csv", best_rows + col_rows + post_rows)
    base._write_csv(process_dir / "scale_sweep_summary.csv", sweep_summary)

    translation_overlay = base._draw_full_overlay(
        fixed_gray_full_u8,
        fixed_mask_full,
        best_rows,
        best_tile_warps,
        title=f"Best-scale translation only ({best_overall_scale:.3f})",
    )
    column_overlay = base._draw_full_overlay(
        fixed_gray_full_u8,
        fixed_mask_full,
        col_rows,
        col_tile_warps,
        title="Column/shared residual",
    )
    post_overlay = base._draw_full_overlay(
        fixed_gray_full_u8,
        fixed_mask_full,
        post_rows,
        post_tile_warps,
        title="Column/shared residual + weak similarity post",
    )
    base._build_method_comparison_contact(
        [
            (f"Best-scale translation ({best_overall_scale:.3f})", translation_overlay),
            ("Column/shared residual", column_overlay),
            ("Column/shared residual + weak similarity post", post_overlay),
        ],
        figures_dir / "method_comparison_contact.png",
    )

    translation_agg = _agg(best_rows)
    column_agg = _agg(col_rows)
    post_agg = _agg(post_rows)
    risk = _risk_summary(baseline_rows=col_rows, test_rows=post_rows)
    timing = {
        "scale_sweep_seconds": sweep_seconds,
        "translation_best_scale_seconds": translation_seconds,
        "column_shared_residual_seconds": column_seconds,
        "compose_column_composite_seconds": compose_seconds,
        "weak_similarity_post_timing_seconds": post_timing,
        "total_probe_wall_seconds": float(time.perf_counter() - t_all),
    }
    summary = {
        "best_overall_scale": float(best_overall_scale),
        "best_right_scale": float(best_right_scale),
        "translation_only": translation_agg,
        "column_shared_residual": column_agg,
        "column_then_weak_similarity_post": post_agg,
        "risk_vs_column_shared": risk,
        "timing_seconds": timing,
        "weak_similarity_manifest": {
            "run_dir": str(post_manifest.get("run_dir") or ""),
            "timing_seconds": dict(post_manifest.get("timing_seconds") or {}),
            "local_registration": dict(post_manifest.get("local_registration") or {}),
        },
    }
    (process_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    lines = [
        "# Column/Shared Residual Then Weak Similarity Post Probe",
        "",
        f"- best overall scale from prior coarse environment: `{best_overall_scale:.3f}`",
        f"- best right-flatten scale from prior coarse environment: `{best_right_scale:.3f}`",
        "",
        "## Aggregate Comparison",
        f"- translation_only: meanCC={translation_agg['mean_tile_cc']:.4f} mean|dx*|={translation_agg['mean_abs_dx']:.2f} rightmost|dx*|={translation_agg['rightmost_mean_abs_dx']:.2f}",
        f"- column_shared_residual: meanCC={column_agg['mean_tile_cc']:.4f} mean|dx*|={column_agg['mean_abs_dx']:.2f} rightmost|dx*|={column_agg['rightmost_mean_abs_dx']:.2f}",
        f"- column_then_weak_similarity_post: meanCC={post_agg['mean_tile_cc']:.4f} mean|dx*|={post_agg['mean_abs_dx']:.2f} rightmost|dx*|={post_agg['rightmost_mean_abs_dx']:.2f}",
        "",
        "## Risk Against Column/Shared Residual",
        f"- tiles with CC drop > 0.01: {risk['tiles_worse_cc_gt_0p01']}",
        f"- tiles with |dx*| increase > 1 px: {risk['tiles_worse_abs_dx_gt_1px']}",
        f"- mean delta CC: {risk['mean_delta_cc']:.4f}",
        f"- mean delta |dx*|: {risk['mean_delta_abs_dx']:.2f}",
        "",
        "## Timing",
        f"- scale_sweep_seconds: {timing['scale_sweep_seconds']:.2f}",
        f"- translation_best_scale_seconds: {timing['translation_best_scale_seconds']:.2f}",
        f"- column_shared_residual_seconds: {timing['column_shared_residual_seconds']:.2f}",
        f"- compose_column_composite_seconds: {timing['compose_column_composite_seconds']:.2f}",
        f"- weak_similarity_post ants_registration seconds: {float((post_timing.get('ants_registration') or 0.0)):.2f}",
        f"- weak_similarity_post total seconds: {float((post_timing.get('total') or 0.0)):.2f}",
        f"- weak_similarity_post wall_total seconds: {float((post_timing.get('wall_total') or 0.0)):.2f}",
        f"- total_probe_wall_seconds: {timing['total_probe_wall_seconds']:.2f}",
        "",
        "## Outputs",
        f"- per-tile csv: `{process_dir / 'tile_metrics_compare.csv'}`",
        f"- sweep summary: `{process_dir / 'scale_sweep_summary.csv'}`",
        f"- method contact sheet: `{figures_dir / 'method_comparison_contact.png'}`",
        f"- weak similarity run dir: `{post_manifest.get('run_dir') or ''}`",
    ]
    (out_root / "summary.md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    print(f"OK: written to {out_root}")


if __name__ == "__main__":
    main()

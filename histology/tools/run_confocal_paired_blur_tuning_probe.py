from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
TOOLS_ROOT = REPO_ROOT / "registration_pipeline" / "histology" / "tools"
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

import run_confocal_multi_tile_feature_probe as base  # noqa: E402


OUT_ROOT = Path("/mnt/c/Users/Siqi/Desktop/REVIEW/20260408_confocal_paired_blur_tuning_probe_2501_60")
SEARCH_RADIUS_SMALL = base.SEARCH_RADIUS_SMALL


def _contrast_stats(fixed_u8: np.ndarray, moving_u8: np.ndarray, fixed_mask: np.ndarray, moving_mask: np.ndarray) -> dict[str, float]:
    fv = np.asarray(fixed_u8, dtype=np.float32)[np.asarray(fixed_mask) > 0]
    mv = np.asarray(moving_u8, dtype=np.float32)[np.asarray(moving_mask) > 0]
    if fv.size < 32 or mv.size < 32:
        return {
            "fixed_p95_p5": float("nan"),
            "moving_p95_p5": float("nan"),
            "contrast_ratio_range": float("nan"),
            "fixed_std": float("nan"),
            "moving_std": float("nan"),
            "contrast_ratio_std": float("nan"),
        }
    fixed_range = float(np.percentile(fv, 95.0) - np.percentile(fv, 5.0))
    moving_range = float(np.percentile(mv, 95.0) - np.percentile(mv, 5.0))
    fixed_std = float(np.std(fv))
    moving_std = float(np.std(mv))
    return {
        "fixed_p95_p5": fixed_range,
        "moving_p95_p5": moving_range,
        "contrast_ratio_range": float(moving_range / fixed_range) if fixed_range > 1e-6 else float("nan"),
        "fixed_std": fixed_std,
        "moving_std": moving_std,
        "contrast_ratio_std": float(moving_std / fixed_std) if fixed_std > 1e-6 else float("nan"),
    }


def _paired_variant(
    *,
    lo_pct: float,
    hi_pct: float,
    sigma: float,
    clahe_clip: float | None = None,
    clahe_grid: int = 8,
) -> Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    def _fn(fixed_u8, moving_u8, fixed_mask, moving_mask):
        fixed_proc = base._masked_percentile_normalize_u8(fixed_u8, fixed_mask, lo_pct=lo_pct, hi_pct=hi_pct)
        moving_proc = base._masked_percentile_normalize_u8(moving_u8, moving_mask, lo_pct=lo_pct, hi_pct=hi_pct)
        if clahe_clip is not None:
            fixed_proc = base._apply_clahe_u8(fixed_proc, fixed_mask, clip_limit=float(clahe_clip), tile_grid=int(clahe_grid))
            moving_proc = base._apply_clahe_u8(moving_proc, moving_mask, clip_limit=float(clahe_clip), tile_grid=int(clahe_grid))
        fixed_proc = base._gaussian_blur_u8(fixed_proc, fixed_mask, sigma=float(sigma))
        moving_proc = base._gaussian_blur_u8(moving_proc, moving_mask, sigma=float(sigma))
        return fixed_proc, moving_proc

    return _fn


def _method_variants() -> list[tuple[str, str, Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]]]:
    return [
        ("paired_pct1_99_blur4", "Paired percentile 1-99 + blur sigma=4", _paired_variant(lo_pct=1.0, hi_pct=99.0, sigma=4.0)),
        ("paired_pct1_99_blur6", "Paired percentile 1-99 + blur sigma=6 (current GUI baseline)", _paired_variant(lo_pct=1.0, hi_pct=99.0, sigma=6.0)),
        ("paired_pct1_99_blur8", "Paired percentile 1-99 + blur sigma=8", _paired_variant(lo_pct=1.0, hi_pct=99.0, sigma=8.0)),
        ("paired_pct0p5_99p5_blur6", "Paired percentile 0.5-99.5 + blur sigma=6", _paired_variant(lo_pct=0.5, hi_pct=99.5, sigma=6.0)),
        ("paired_pct2_98_blur6", "Paired percentile 2-98 + blur sigma=6", _paired_variant(lo_pct=2.0, hi_pct=98.0, sigma=6.0)),
        ("paired_pct0p5_99p5_blur8", "Paired percentile 0.5-99.5 + blur sigma=8", _paired_variant(lo_pct=0.5, hi_pct=99.5, sigma=8.0)),
        ("paired_pct1_99_clahe2p5_blur6", "Paired percentile 1-99 + CLAHE 2.5 + blur sigma=6", _paired_variant(lo_pct=1.0, hi_pct=99.0, sigma=6.0, clahe_clip=2.5)),
        ("paired_pct0p5_99p5_clahe4_blur6", "Paired percentile 0.5-99.5 + CLAHE 4.0 + blur sigma=6", _paired_variant(lo_pct=0.5, hi_pct=99.5, sigma=6.0, clahe_clip=4.0)),
    ]


def _run_methods(contexts: list[base.TileContext]) -> list[base.MethodResult]:
    results: list[base.MethodResult] = []
    for name, description, fn in _method_variants():
        tile_rows: list[dict[str, Any]] = []
        for ctx in contexts:
            fixed_mask_u8 = np.where(ctx.fixed_mask > 0, 255, 0).astype(np.uint8)
            moving_mask_u8 = np.where(ctx.moving_signal_mask > 0, 255, 0).astype(np.uint8)
            fixed_proc_u8, moving_proc_u8 = fn(ctx.fixed_native_u8, ctx.moving_native_u8, fixed_mask_u8, moving_mask_u8)
            current_native = base._metrics_from_pair(ctx.fixed_native_u8, ctx.moving_native_u8, ctx.fixed_mask, ctx.moving_signal_mask)
            current_proc = base._metrics_from_pair(fixed_proc_u8, moving_proc_u8, ctx.fixed_mask, ctx.moving_signal_mask)
            best_small_proc = base._best_shift_cc(fixed_proc_u8, moving_proc_u8, ctx.fixed_mask, ctx.moving_signal_mask, radius=SEARCH_RADIUS_SMALL)
            contrast = _contrast_stats(fixed_proc_u8, moving_proc_u8, ctx.fixed_mask, ctx.moving_signal_mask)
            panels = base._compose_panels(
                fixed_native_u8=ctx.fixed_native_u8,
                moving_native_u8=ctx.moving_native_u8,
                fixed_proc_u8=fixed_proc_u8,
                moving_proc_u8=moving_proc_u8,
                fixed_mask=ctx.fixed_mask,
                moving_signal_mask=ctx.moving_signal_mask,
                moving_footprint_mask=ctx.moving_footprint_mask,
                best_small_native=base._best_shift_cc(ctx.fixed_native_u8, ctx.moving_native_u8, ctx.fixed_mask, ctx.moving_signal_mask, radius=SEARCH_RADIUS_SMALL),
                best_small_proc=best_small_proc,
            )
            tile_rows.append(
                {
                    "method": name,
                    "description": description,
                    "tile_index": ctx.tile_index,
                    "row": ctx.row,
                    "col": ctx.col,
                    "distance_from_anchor_px": ctx.distance_from_anchor_px,
                    "signal_coverage": float(np.mean(ctx.moving_signal_mask)),
                    "current_native_cc": float(current_native["cc"]),
                    "current_proc_cc": float(current_proc["cc"]),
                    "current_proc_mi": float(current_proc["mi"]),
                    "best_small_proc_cc": float(best_small_proc["cc"]),
                    "best_small_proc_dx": int(best_small_proc["dx"]),
                    "best_small_proc_dy": int(best_small_proc["dy"]),
                    "best_small_proc_shift_mag": float(math.hypot(float(best_small_proc["dx"]), float(best_small_proc["dy"]))),
                    "delta_small_cc": float(best_small_proc["cc"] - current_proc["cc"]),
                    **contrast,
                    "panels": panels,
                }
            )
        current_vals = np.asarray([float(r["current_proc_cc"]) for r in tile_rows], dtype=np.float64)
        small_vals = np.asarray([float(r["best_small_proc_cc"]) for r in tile_rows], dtype=np.float64)
        delta_vals = np.asarray([float(r["delta_small_cc"]) for r in tile_rows], dtype=np.float64)
        shift_vals = np.asarray([float(r["best_small_proc_shift_mag"]) for r in tile_rows], dtype=np.float64)
        contrast_vals = np.asarray([float(r["contrast_ratio_range"]) for r in tile_rows], dtype=np.float64)
        contrast_dev = np.asarray([abs(float(v) - 1.0) if np.isfinite(v) else np.nan for v in contrast_vals], dtype=np.float64)
        summary = {
            "mean_current_proc_cc": float(np.nanmean(current_vals)),
            "median_current_proc_cc": float(np.nanmedian(current_vals)),
            "mean_best_small_proc_cc": float(np.nanmean(small_vals)),
            "mean_delta_small_cc": float(np.nanmean(delta_vals)),
            "median_delta_small_cc": float(np.nanmedian(delta_vals)),
            "mean_small_shift_mag": float(np.nanmean(shift_vals)),
            "mean_contrast_ratio_range": float(np.nanmean(contrast_vals)),
            "mean_contrast_abs_dev_from_1": float(np.nanmean(contrast_dev)),
            "tile_count": len(tile_rows),
            "stable_tiles_shift_le_2px": int(sum(1 for r in tile_rows if float(r["best_small_proc_shift_mag"]) <= 2.0)),
            "strong_tiles_current_cc_ge_0_5": int(sum(1 for r in tile_rows if float(r["current_proc_cc"]) >= 0.5)),
        }
        results.append(base.MethodResult(name=name, description=description, tile_rows=tile_rows, summary=summary))
    return results


def main() -> None:
    out_root = base._ensure_dir(OUT_ROOT)
    figs_dir = base._ensure_dir(out_root / "figures")
    process_dir = base._ensure_dir(out_root / "process")
    methods_dir = base._ensure_dir(out_root / "methods")

    contexts, _common = base._build_tile_contexts()
    sample_tile_ids = base._select_sample_tiles(contexts)
    results = _run_methods(contexts)

    method_rows: list[dict[str, Any]] = []
    tile_rows: list[dict[str, Any]] = []
    for result in results:
        method_rows.append({"method": result.name, "description": result.description, **result.summary})
        method_dir = base._ensure_dir(methods_dir / result.name)
        base._save_method_sample_sheet(result, sample_tile_ids, method_dir / "sample_tile_qc.png")
        for row in result.tile_rows:
            tile_rows.append({k: v for k, v in row.items() if k != "panels"})

    base._write_csv(process_dir / "method_summary.csv", method_rows)
    base._write_csv(process_dir / "tile_method_metrics.csv", tile_rows)
    base._save_tile_winner_csv(results, process_dir / "tile_winner_methods.csv")
    base._save_summary_plot(results, figs_dir / "method_ranking.png")

    ranked_by_current = sorted(results, key=lambda r: float(r.summary["mean_current_proc_cc"]), reverse=True)
    ranked_by_contrast = sorted(results, key=lambda r: float(r.summary["mean_contrast_abs_dev_from_1"]))
    summary_lines = [
        "# Paired Percentile/Blur Tuning Probe",
        "",
        "Purpose:",
        "- stress-test the current GUI default `paired_percentile_blur6`",
        "- compare nearby symmetric preprocessing settings on a full 3x6 tile set",
        "- quantify whether stronger percentile stretch, stronger blur, or mild CLAHE improves current alignment quality",
        "",
        "Held fixed:",
        f"- manual scale: {float(base.geom.MANUAL_STATE['scale']):.3f}",
        f"- manual angle: {float(base.geom.MANUAL_STATE['angle_deg']):.3f}",
        f"- flip_ud: {bool(base.geom.MANUAL_STATE['flip_ud'])}",
        f"- anchor tile reference: T{base.ANCHOR_TILE_INDEX:02d}",
        f"- tile count: {len(contexts)}",
        "",
        "Read these metrics as:",
        "- `mean_current_proc_cc`: how good the processed representation already is at the current manual alignment",
        f"- `mean_best_small_proc_cc`: best CC after a ±{SEARCH_RADIUS_SMALL}px local search",
        "- `mean_delta_small_cc`: remaining local optimization headroom; smaller is better if the current alignment is already close",
        "- `mean_contrast_ratio_range`: moving/fixed processed contrast ratio; values closer to 1 suggest more comparable dynamic range",
        "",
        "Top methods by mean_current_proc_cc:",
    ]
    for result in ranked_by_current[:6]:
        s = result.summary
        summary_lines.append(
            f"- `{result.name}`: current={float(s['mean_current_proc_cc']):.4f}, "
            f"best_small={float(s['mean_best_small_proc_cc']):.4f}, "
            f"delta={float(s['mean_delta_small_cc']):.4f}, "
            f"contrast_ratio={float(s['mean_contrast_ratio_range']):.3f}, "
            f"stable<=2px={int(s['stable_tiles_shift_le_2px'])}"
        )
    summary_lines.extend(["", "Top methods by contrast comparability (|ratio-1| smaller is better):"])
    for result in ranked_by_contrast[:6]:
        s = result.summary
        summary_lines.append(
            f"- `{result.name}`: |ratio-1|={float(s['mean_contrast_abs_dev_from_1']):.3f}, "
            f"contrast_ratio={float(s['mean_contrast_ratio_range']):.3f}, "
            f"current={float(s['mean_current_proc_cc']):.4f}"
        )
    summary_lines.extend(
        [
            "",
            "Sample-tile QC uses:",
            "- " + ", ".join(f"T{int(tid):02d}" for tid in sample_tile_ids),
            "",
            "Important files:",
            f"- summary CSV: `{process_dir / 'method_summary.csv'}`",
            f"- per-tile metrics CSV: `{process_dir / 'tile_method_metrics.csv'}`",
            f"- tile winner CSV: `{process_dir / 'tile_winner_methods.csv'}`",
            f"- ranking figure: `{figs_dir / 'method_ranking.png'}`",
            "- per-method QC: `methods/<method>/sample_tile_qc.png`",
        ]
    )
    (out_root / "summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    (out_root / "run_manifest.json").write_text(
        json.dumps(
            {
                "manual_state": base.geom.MANUAL_STATE,
                "anchor_tile_index": base.ANCHOR_TILE_INDEX,
                "search_radius_small_px": SEARCH_RADIUS_SMALL,
                "tile_count": len(contexts),
                "sample_tile_ids": sample_tile_ids,
                "methods": method_rows,
                "files": {
                    "summary_md": str(out_root / "summary.md"),
                    "method_summary_csv": str(process_dir / "method_summary.csv"),
                    "tile_method_metrics_csv": str(process_dir / "tile_method_metrics.csv"),
                    "tile_winner_methods_csv": str(process_dir / "tile_winner_methods.csv"),
                    "method_ranking_png": str(figs_dir / "method_ranking.png"),
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Paired blur tuning probe written to: {out_root}")


if __name__ == "__main__":
    main()

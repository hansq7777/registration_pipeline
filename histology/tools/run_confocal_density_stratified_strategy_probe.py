from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
GUI_MVP_ROOT = REPO_ROOT / "registration_pipeline" / "histology" / "gui_mvp"
TOOLS_ROOT = REPO_ROOT / "registration_pipeline" / "histology" / "tools"
for p in (str(GUI_MVP_ROOT), str(TOOLS_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from hitl_gui.application.confocal_registration import (  # noqa: E402
    _edge_density_from_gray,
    _objective_score_from_metrics,
    _step7_density_regime,
)
import run_confocal_multi_tile_feature_probe as base  # noqa: E402


OUT_ROOT = Path("/mnt/c/Users/Siqi/Desktop/REVIEW/20260409_confocal_density_stratified_strategy_probe_2501_60")
SEARCH_RADIUS_SMALL = base.SEARCH_RADIUS_SMALL
OBJECTIVES = ("cc", "mi", "hybrid")


def _paired_variant(
    *,
    lo_pct: float = 1.0,
    hi_pct: float = 99.0,
    sigma: float | None = None,
    clahe_clip: float | None = None,
    clahe_grid: int = 8,
) -> Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    def _fn(fixed_u8, moving_u8, fixed_mask, moving_mask):
        fixed_proc = base._masked_percentile_normalize_u8(fixed_u8, fixed_mask, lo_pct=lo_pct, hi_pct=hi_pct)
        moving_proc = base._masked_percentile_normalize_u8(moving_u8, moving_mask, lo_pct=lo_pct, hi_pct=hi_pct)
        if clahe_clip is not None:
            fixed_proc = base._apply_clahe_u8(fixed_proc, fixed_mask, clip_limit=float(clahe_clip), tile_grid=int(clahe_grid))
            moving_proc = base._apply_clahe_u8(moving_proc, moving_mask, clip_limit=float(clahe_clip), tile_grid=int(clahe_grid))
        if sigma is not None:
            fixed_proc = base._gaussian_blur_u8(fixed_proc, fixed_mask, sigma=float(sigma))
            moving_proc = base._gaussian_blur_u8(moving_proc, moving_mask, sigma=float(sigma))
        return fixed_proc, moving_proc

    return _fn


def _moving_hist_variant(
    *,
    fixed_sigma: float,
    moving_sigma: float,
    moving_lo_pct: float = 1.0,
    moving_hi_pct: float = 99.0,
    moving_gamma: float | None = None,
    moving_clahe_clip: float | None = None,
) -> Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    def _fn(fixed_u8, moving_u8, fixed_mask, moving_mask):
        fixed_proc = base._gaussian_blur_u8(
            base._masked_percentile_normalize_u8(fixed_u8, fixed_mask),
            fixed_mask,
            sigma=float(fixed_sigma),
        )
        moving_proc = base._masked_percentile_normalize_u8(
            moving_u8,
            moving_mask,
            lo_pct=float(moving_lo_pct),
            hi_pct=float(moving_hi_pct),
        )
        if moving_gamma is not None:
            moving_proc = base._gamma_u8(moving_proc, moving_mask, gamma=float(moving_gamma))
        if moving_clahe_clip is not None:
            moving_proc = base._apply_clahe_u8(moving_proc, moving_mask, clip_limit=float(moving_clahe_clip), tile_grid=8)
        moving_proc = base.geom._masked_histogram_match_u8(moving_proc, moving_mask, fixed_proc, fixed_mask)
        moving_proc = base._gaussian_blur_u8(moving_proc, moving_mask, sigma=float(moving_sigma))
        return fixed_proc, moving_proc

    return _fn


def _binary_dt_variant() -> Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    def _fn(fixed_u8, moving_u8, fixed_mask, moving_mask):
        return (
            base._distance_transform_dark_u8(base._masked_percentile_normalize_u8(fixed_u8, fixed_mask), fixed_mask, keep_quantile=60.0),
            base._distance_transform_dark_u8(base._masked_percentile_normalize_u8(moving_u8, moving_mask), moving_mask, keep_quantile=60.0),
        )

    return _fn


def _method_variants() -> list[tuple[str, str, Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]]]:
    return [
        ("paired_percentile_raw", "Paired percentile normalization only", _paired_variant()),
        ("paired_percentile_blur1", "Paired percentile + blur sigma=1", _paired_variant(sigma=1.0)),
        ("paired_percentile_blur2", "Paired percentile + blur sigma=2", _paired_variant(sigma=2.0)),
        ("paired_percentile_blur4", "Paired percentile + blur sigma=4", _paired_variant(sigma=4.0)),
        ("paired_percentile_blur6", "Paired percentile + blur sigma=6", _paired_variant(sigma=6.0)),
        ("paired_percentile_blur8", "Paired percentile + blur sigma=8", _paired_variant(sigma=8.0)),
        ("paired_percentile_clahe_blur2", "Paired percentile + CLAHE + blur sigma=2", _paired_variant(sigma=2.0, clahe_clip=2.5)),
        ("paired_percentile_clahe_blur3", "Paired percentile + CLAHE + blur sigma=3", _paired_variant(sigma=3.0, clahe_clip=3.0)),
        ("moving_percentile_hist_blur4", "Fixed percentile+blur4; moving percentile + histmatch + blur4", _moving_hist_variant(fixed_sigma=4.0, moving_sigma=4.0)),
        ("moving_gamma_clahe_hist_blur4", "Fixed percentile+blur4; moving gamma + CLAHE + histmatch + blur4", _moving_hist_variant(fixed_sigma=4.0, moving_sigma=4.0, moving_lo_pct=0.5, moving_hi_pct=99.5, moving_gamma=1.8, moving_clahe_clip=6.0)),
        ("paired_relaxed_binary_dt_q60", "Paired relaxed binary q60 + distance transform", _binary_dt_variant()),
    ]


def _context_density(ctx: base.TileContext) -> dict[str, Any]:
    signal_coverage = float(np.mean(ctx.moving_signal_mask > 0))
    moving_edge_density = _edge_density_from_gray(ctx.moving_native_u8.astype(np.float32) / 255.0, ctx.moving_signal_mask)
    fixed_edge_density = _edge_density_from_gray(ctx.fixed_native_u8.astype(np.float32) / 255.0, ctx.fixed_mask)
    density_regime, edge_mean = _step7_density_regime(
        signal_coverage=signal_coverage,
        moving_edge_density=moving_edge_density,
        fixed_edge_density=fixed_edge_density,
    )
    return {
        "signal_coverage": signal_coverage,
        "moving_edge_density": float(moving_edge_density),
        "fixed_edge_density": float(fixed_edge_density),
        "edge_density_mean": float(edge_mean),
        "density_regime": str(density_regime),
    }


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
            proc_shift_candidates = _enumerate_shift_candidates(
                fixed_proc_u8,
                moving_proc_u8,
                ctx.fixed_mask,
                ctx.moving_signal_mask,
                radius=SEARCH_RADIUS_SMALL,
            )
            objective_results = {
                objective: _best_shift_from_candidates(proc_shift_candidates, objective_name=objective)
                for objective in OBJECTIVES
            }
            best_small_proc = objective_results["cc"]
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
                    **_context_density(ctx),
                    "current_native_cc": float(current_native["cc"]),
                    "current_native_mi": float(current_native["mi"]),
                    "current_proc_cc": float(current_proc["cc"]),
                    "current_proc_mi": float(current_proc["mi"]),
                    "best_small_proc_cc": float(best_small_proc["cc"]),
                    "best_small_proc_mi": float(best_small_proc["mi"]),
                    "best_small_proc_dx": int(best_small_proc["dx"]),
                    "best_small_proc_dy": int(best_small_proc["dy"]),
                    "best_small_proc_shift_mag": float(math.hypot(float(best_small_proc["dx"]), float(best_small_proc["dy"]))),
                    "delta_small_cc": float(best_small_proc["cc"] - current_proc["cc"]),
                    "delta_small_mi": float(best_small_proc["mi"] - current_proc["mi"]) if np.isfinite(float(current_proc["mi"])) and np.isfinite(float(best_small_proc["mi"])) else float("nan"),
                    "objective_results": {
                        objective: {
                            "dx": int(objective_results[objective]["dx"]),
                            "dy": int(objective_results[objective]["dy"]),
                            "objective_score": float(objective_results[objective]["objective_score"]),
                            "cc": float(objective_results[objective]["cc"]),
                            "mi": float(objective_results[objective]["mi"]),
                            "dice": float(objective_results[objective]["dice"]),
                            "hd95": float(objective_results[objective]["hd95"]),
                        }
                        for objective in OBJECTIVES
                    },
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
        results.append(base.MethodResult(name=name, description=description, tile_rows=tile_rows, summary=summary))
    return results


def _enumerate_shift_candidates(
    fixed_u8: np.ndarray,
    moving_u8: np.ndarray,
    fixed_mask: np.ndarray,
    moving_mask: np.ndarray,
    *,
    radius: int,
) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            shifted_gray, shifted_mask = base._shift_patch_within_canvas(
                moving_u8.astype(np.float32) / 255.0,
                moving_mask.astype(np.float32),
                dx,
                dy,
                fill_value=1.0,
            )
            metrics, _ = base.compute_registration_metrics(
                fixed_u8.astype(np.float32) / 255.0,
                shifted_gray,
                fixed_mask.astype(np.float32),
                shifted_mask.astype(np.float32),
            )
            rows.append(
                {
                    "dx": int(dx),
                    "dy": int(dy),
                    "cc": float(metrics.get("cc", float("nan"))),
                    "mi": float(metrics.get("mi", float("nan"))),
                    "dice": float(metrics.get("dice", float("nan"))),
                    "hd95": float(metrics.get("hd95_px", float("nan"))),
                }
            )
    return rows


def _best_shift_from_candidates(candidates: list[dict[str, float]], *, objective_name: str) -> dict[str, float]:
    best: dict[str, float] | None = None
    objective = str(objective_name or "cc").strip().lower()
    for row in candidates:
        objective_score = _objective_score_from_metrics(row, objective_name=objective)
        candidate = {
            "dx": int(row.get("dx", 0)),
            "dy": int(row.get("dy", 0)),
            "objective_score": float(objective_score),
            "cc": float(row.get("cc", float("nan"))),
            "mi": float(row.get("mi", float("nan"))),
            "dice": float(row.get("dice", float("nan"))),
            "hd95": float(row.get("hd95", float("nan"))),
        }
        if best is None:
            best = candidate
            continue
        cand_score = float(candidate["objective_score"])
        best_score = float(best["objective_score"])
        if (np.isfinite(cand_score) and not np.isfinite(best_score)) or cand_score > best_score + 1e-6:
            best = candidate
            continue
        if abs(cand_score - best_score) <= 1e-6:
            cand_cc = float(candidate["cc"])
            best_cc = float(best["cc"])
            if (np.isfinite(cand_cc) and not np.isfinite(best_cc)) or cand_cc > best_cc + 1e-6:
                best = candidate
                continue
            if abs(cand_cc - best_cc) <= 1e-6 and math.hypot(float(candidate["dx"]), float(candidate["dy"])) < math.hypot(float(best["dx"]), float(best["dy"])):
                best = candidate
    if best is None:
        return {
            "dx": 0,
            "dy": 0,
            "objective_score": float("nan"),
            "cc": float("nan"),
            "mi": float("nan"),
            "dice": float("nan"),
            "hd95": float("nan"),
        }
    return best


def _density_summary_rows(results: list[base.MethodResult]) -> list[dict[str, Any]]:
    density_labels = ("sparse_feature", "sparse_weak", "mid_density", "high_density")
    rows: list[dict[str, Any]] = []
    for result in results:
        for regime in density_labels:
            subset = [row for row in result.tile_rows if str(row.get("density_regime")) == regime]
            if not subset:
                continue
            current_vals = np.asarray([float(r["current_proc_cc"]) for r in subset], dtype=np.float64)
            small_vals = np.asarray([float(r["best_small_proc_cc"]) for r in subset], dtype=np.float64)
            delta_vals = np.asarray([float(r["delta_small_cc"]) for r in subset], dtype=np.float64)
            shift_vals = np.asarray([float(r["best_small_proc_shift_mag"]) for r in subset], dtype=np.float64)
            rows.append(
                {
                    "density_regime": regime,
                    "method": result.name,
                    "description": result.description,
                    "tile_count": len(subset),
                    "mean_current_proc_cc": float(np.nanmean(current_vals)),
                    "median_current_proc_cc": float(np.nanmedian(current_vals)),
                    "mean_best_small_proc_cc": float(np.nanmean(small_vals)),
                    "mean_delta_small_cc": float(np.nanmean(delta_vals)),
                    "median_delta_small_cc": float(np.nanmedian(delta_vals)),
                    "mean_small_shift_mag": float(np.nanmean(shift_vals)),
                    "strong_tiles_current_cc_ge_0_5": int(sum(1 for r in subset if float(r["current_proc_cc"]) >= 0.5)),
                    "stable_tiles_shift_le_2px": int(sum(1 for r in subset if float(r["best_small_proc_shift_mag"]) <= 2.0)),
                }
            )
    return rows


def _top_rows_by_regime(summary_rows: list[dict[str, Any]], regime: str, field: str, *, limit: int = 4, reverse: bool = True) -> list[dict[str, Any]]:
    subset = [row for row in summary_rows if str(row.get("density_regime")) == regime]
    return sorted(subset, key=lambda row: float(row.get(field, float("nan"))), reverse=reverse)[:limit]


def _objective_tile_rows(results: list[base.MethodResult]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result in results:
        for row in result.tile_rows:
            objective_results = dict(row.get("objective_results") or {})
            current_cc = float(row.get("current_proc_cc", float("nan")))
            current_mi = float(row.get("current_proc_mi", float("nan")))
            for objective in OBJECTIVES:
                payload = dict(objective_results.get(objective) or {})
                if not payload:
                    continue
                rows.append(
                    {
                        "density_regime": str(row.get("density_regime") or ""),
                        "objective": str(objective),
                        "method": str(result.name),
                        "description": str(result.description),
                        "tile_index": int(row.get("tile_index", -1)),
                        "row": int(row.get("row", 0)),
                        "col": int(row.get("col", 0)),
                        "signal_coverage": float(row.get("signal_coverage", float("nan"))),
                        "moving_edge_density": float(row.get("moving_edge_density", float("nan"))),
                        "fixed_edge_density": float(row.get("fixed_edge_density", float("nan"))),
                        "edge_density_mean": float(row.get("edge_density_mean", float("nan"))),
                        "current_proc_cc": current_cc,
                        "current_proc_mi": current_mi,
                        "best_shift_dx_px": int(payload.get("dx", 0)),
                        "best_shift_dy_px": int(payload.get("dy", 0)),
                        "best_objective_score": float(payload.get("objective_score", float("nan"))),
                        "best_proc_cc": float(payload.get("cc", float("nan"))),
                        "best_proc_mi": float(payload.get("mi", float("nan"))),
                        "best_proc_shift_mag": float(math.hypot(float(payload.get("dx", 0)), float(payload.get("dy", 0)))),
                        "delta_proc_cc": float(payload.get("cc", float("nan")) - current_cc) if np.isfinite(current_cc) and np.isfinite(float(payload.get("cc", float("nan")))) else float("nan"),
                        "delta_proc_mi": float(payload.get("mi", float("nan")) - current_mi) if np.isfinite(current_mi) and np.isfinite(float(payload.get("mi", float("nan")))) else float("nan"),
                    }
                )
    return rows


def _density_objective_summary_rows(objective_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in objective_rows:
        key = (str(row.get("density_regime") or ""), str(row.get("objective") or ""), str(row.get("method") or ""))
        grouped.setdefault(key, []).append(row)
    for (density_regime, objective, method), subset in grouped.items():
        best_cc_vals = np.asarray([float(r["best_proc_cc"]) for r in subset], dtype=np.float64)
        best_mi_vals = np.asarray([float(r["best_proc_mi"]) for r in subset], dtype=np.float64)
        delta_cc_vals = np.asarray([float(r["delta_proc_cc"]) for r in subset], dtype=np.float64)
        obj_vals = np.asarray([float(r["best_objective_score"]) for r in subset], dtype=np.float64)
        shift_vals = np.asarray([float(r["best_proc_shift_mag"]) for r in subset], dtype=np.float64)
        rows.append(
            {
                "density_regime": density_regime,
                "objective": objective,
                "method": method,
                "description": str(subset[0].get("description") or ""),
                "tile_count": len(subset),
                "mean_best_proc_cc": float(np.nanmean(best_cc_vals)),
                "mean_best_proc_mi": float(np.nanmean(best_mi_vals)),
                "mean_delta_proc_cc": float(np.nanmean(delta_cc_vals)),
                "mean_best_objective_score": float(np.nanmean(obj_vals)),
                "mean_shift_mag": float(np.nanmean(shift_vals)),
            }
        )
    return rows


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
            tile_rows.append({k: v for k, v in row.items() if k not in {"panels", "objective_results"}})

    density_rows = _density_summary_rows(results)
    objective_tile_rows = _objective_tile_rows(results)
    density_objective_rows = _density_objective_summary_rows(objective_tile_rows)
    base._write_csv(process_dir / "method_summary.csv", method_rows)
    base._write_csv(process_dir / "tile_method_metrics.csv", tile_rows)
    base._write_csv(process_dir / "density_regime_method_summary.csv", density_rows)
    base._write_csv(process_dir / "tile_method_objective_metrics.csv", objective_tile_rows)
    base._write_csv(process_dir / "density_regime_objective_method_summary.csv", density_objective_rows)
    base._save_tile_winner_csv(results, process_dir / "tile_winner_methods.csv")
    base._save_summary_plot(results, figs_dir / "method_ranking.png")

    summary_lines = [
        "# Density-Stratified Confocal Strategy Probe",
        "",
        "Purpose:",
        "- compare preprocessing strategies across the full 3x6 tile set",
        "- stratify results by local signal density regime instead of only reporting one global winner",
        "- explicitly test the low-blur gap between `raw` and `blur4`",
        "- compare local shift objectives `cc`, `mi`, and CC-dominant `hybrid = cc + 0.5 * mi`",
        "",
        "Density regime features:",
        "- `signal_coverage`: moving tile foreground occupancy",
        "- `moving_edge_density` and `fixed_edge_density`: masked local edge occupancy",
        "- regimes follow the same helper logic now used by Step 7 backend",
        "",
        f"- tile count: {len(contexts)}",
        f"- method count: {len(results)}",
        f"- small-shift search radius: ±{SEARCH_RADIUS_SMALL}px",
        "",
        "Overall top methods by mean current CC:",
    ]
    overall_top = sorted(method_rows, key=lambda row: float(row["mean_current_proc_cc"]), reverse=True)[:5]
    for row in overall_top:
        summary_lines.append(
            f"- `{row['method']}`: current={float(row['mean_current_proc_cc']):.4f}, "
            f"best_small={float(row['mean_best_small_proc_cc']):.4f}, "
            f"delta={float(row['mean_delta_small_cc']):.4f}, "
            f"stable<=2px={int(row['stable_tiles_shift_le_2px'])}"
        )

    for regime in ("sparse_feature", "mid_density", "high_density"):
        subset = [row for row in density_rows if str(row.get("density_regime")) == regime]
        if not subset:
            continue
        summary_lines.extend(["", f"## {regime}"])
        summary_lines.append("Top by mean current CC:")
        for row in _top_rows_by_regime(density_rows, regime, "mean_current_proc_cc", limit=4, reverse=True):
            summary_lines.append(
                f"- `{row['method']}`: current={float(row['mean_current_proc_cc']):.4f}, "
                f"best_small={float(row['mean_best_small_proc_cc']):.4f}, "
                f"delta={float(row['mean_delta_small_cc']):.4f}, "
                f"n={int(row['tile_count'])}"
            )
        summary_lines.append("Top by improvement headroom:")
        for row in _top_rows_by_regime(density_rows, regime, "mean_delta_small_cc", limit=4, reverse=True):
            summary_lines.append(
                f"- `{row['method']}`: delta={float(row['mean_delta_small_cc']):.4f}, "
                f"current={float(row['mean_current_proc_cc']):.4f}, "
                f"mean shift={float(row['mean_small_shift_mag']):.2f}px"
            )
        obj_subset = [row for row in density_objective_rows if str(row.get("density_regime")) == regime]
        if obj_subset:
            summary_lines.append("Top objective/method pairs by final CC after objective-driven search:")
            top_obj_rows = sorted(obj_subset, key=lambda row: float(row.get("mean_best_proc_cc", float("nan"))), reverse=True)[:6]
            for row in top_obj_rows:
                summary_lines.append(
                    f"- `{row['objective']}` + `{row['method']}`: "
                    f"best_cc={float(row['mean_best_proc_cc']):.4f}, "
                    f"delta_cc={float(row['mean_delta_proc_cc']):.4f}, "
                    f"obj={float(row['mean_best_objective_score']):.4f}"
                )

    summary_lines.extend(
        [
            "",
            "Important files:",
            f"- overall summary CSV: `{process_dir / 'method_summary.csv'}`",
            f"- density-stratified summary CSV: `{process_dir / 'density_regime_method_summary.csv'}`",
            f"- per-tile metrics CSV: `{process_dir / 'tile_method_metrics.csv'}`",
            f"- per-tile objective metrics CSV: `{process_dir / 'tile_method_objective_metrics.csv'}`",
            f"- density/objective summary CSV: `{process_dir / 'density_regime_objective_method_summary.csv'}`",
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
                    "density_regime_method_summary_csv": str(process_dir / "density_regime_method_summary.csv"),
                    "tile_method_metrics_csv": str(process_dir / "tile_method_metrics.csv"),
                    "tile_method_objective_metrics_csv": str(process_dir / "tile_method_objective_metrics.csv"),
                    "density_regime_objective_method_summary_csv": str(process_dir / "density_regime_objective_method_summary.csv"),
                    "tile_winner_methods_csv": str(process_dir / "tile_winner_methods.csv"),
                    "method_ranking_png": str(figs_dir / "method_ranking.png"),
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Density-stratified strategy probe written to: {out_root}")


if __name__ == "__main__":
    main()

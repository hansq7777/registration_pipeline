from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
TOOLS_ROOT = REPO_ROOT / "registration_pipeline" / "histology" / "tools"
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

import run_confocal_multi_tile_feature_probe as mtfp  # noqa: E402


OUT_ROOT = Path("/mnt/c/Users/Siqi/Desktop/REVIEW/20260408_confocal_moving_binary_probe_2501_60")


def _custom_method_variants():
    def paired_percentile_blur4(fixed_u8, moving_u8, fixed_mask, moving_mask):
        return (
            mtfp._gaussian_blur_u8(mtfp._masked_percentile_normalize_u8(fixed_u8, fixed_mask), fixed_mask, sigma=4.0),
            mtfp._gaussian_blur_u8(mtfp._masked_percentile_normalize_u8(moving_u8, moving_mask), moving_mask, sigma=4.0),
        )

    def paired_percentile_blur6(fixed_u8, moving_u8, fixed_mask, moving_mask):
        return (
            mtfp._gaussian_blur_u8(mtfp._masked_percentile_normalize_u8(fixed_u8, fixed_mask), fixed_mask, sigma=6.0),
            mtfp._gaussian_blur_u8(mtfp._masked_percentile_normalize_u8(moving_u8, moving_mask), moving_mask, sigma=6.0),
        )

    def paired_percentile_clahe_blur3(fixed_u8, moving_u8, fixed_mask, moving_mask):
        return (
            mtfp._gaussian_blur_u8(
                mtfp._apply_clahe_u8(mtfp._masked_percentile_normalize_u8(fixed_u8, fixed_mask), fixed_mask, clip_limit=3.0, tile_grid=8),
                fixed_mask,
                sigma=3.0,
            ),
            mtfp._gaussian_blur_u8(
                mtfp._apply_clahe_u8(mtfp._masked_percentile_normalize_u8(moving_u8, moving_mask), moving_mask, clip_limit=3.0, tile_grid=8),
                moving_mask,
                sigma=3.0,
            ),
        )

    def moving_gamma_clahe_hist_blur4(fixed_u8, moving_u8, fixed_mask, moving_mask):
        fixed_proc = mtfp._gaussian_blur_u8(mtfp._masked_percentile_normalize_u8(fixed_u8, fixed_mask), fixed_mask, sigma=4.0)
        moving_pct = mtfp._masked_percentile_normalize_u8(moving_u8, moving_mask, lo_pct=0.5, hi_pct=99.5)
        moving_gamma = mtfp._gamma_u8(moving_pct, moving_mask, gamma=1.8)
        moving_clahe = mtfp._apply_clahe_u8(moving_gamma, moving_mask, clip_limit=6.0, tile_grid=8)
        moving_hist = mtfp.geom._masked_histogram_match_u8(moving_clahe, moving_mask, fixed_proc, fixed_mask)
        moving_proc = mtfp._gaussian_blur_u8(moving_hist, moving_mask, sigma=4.0)
        return fixed_proc, moving_proc

    def _moving_binary_variant(keep_quantile: float):
        def fn(fixed_u8, moving_u8, fixed_mask, moving_mask):
            fixed_proc = mtfp._gaussian_blur_u8(mtfp._masked_percentile_normalize_u8(fixed_u8, fixed_mask), fixed_mask, sigma=4.0)
            moving_pct = mtfp._masked_percentile_normalize_u8(moving_u8, moving_mask, lo_pct=0.5, hi_pct=99.7)
            moving_gamma = mtfp._gamma_u8(moving_pct, moving_mask, gamma=2.0)
            moving_clahe = mtfp._apply_clahe_u8(moving_gamma, moving_mask, clip_limit=7.0, tile_grid=8)
            moving_proc = mtfp._relaxed_binary_dark_u8(moving_clahe, moving_mask, keep_quantile=keep_quantile, min_area=4)
            return fixed_proc, moving_proc

        return fn

    def _moving_binary_dt_variant(keep_quantile: float):
        def fn(fixed_u8, moving_u8, fixed_mask, moving_mask):
            fixed_proc = mtfp._gaussian_blur_u8(mtfp._masked_percentile_normalize_u8(fixed_u8, fixed_mask), fixed_mask, sigma=4.0)
            moving_pct = mtfp._masked_percentile_normalize_u8(moving_u8, moving_mask, lo_pct=0.5, hi_pct=99.7)
            moving_gamma = mtfp._gamma_u8(moving_pct, moving_mask, gamma=2.0)
            moving_clahe = mtfp._apply_clahe_u8(moving_gamma, moving_mask, clip_limit=7.0, tile_grid=8)
            moving_proc = mtfp._distance_transform_dark_u8(moving_clahe, moving_mask, keep_quantile=keep_quantile)
            return fixed_proc, moving_proc

        return fn

    return [
        ("paired_percentile_blur6", "Remembered current best overall baseline: paired percentile + blur sigma=6", paired_percentile_blur6),
        ("paired_percentile_blur4", "Strong baseline: paired percentile + blur sigma=4", paired_percentile_blur4),
        ("paired_percentile_clahe_blur3", "Special-tile baseline: paired percentile + CLAHE + blur sigma=3", paired_percentile_clahe_blur3),
        ("moving_gamma_clahe_hist_blur4", "Competitive hard-tile baseline: moving gamma + CLAHE + histmatch + blur4", moving_gamma_clahe_hist_blur4),
        ("moving_binary_q55_fixed_blur4", "Only confocal binarized aggressively (q55), fixed stays grayscale blur4", _moving_binary_variant(55.0)),
        ("moving_binary_q65_fixed_blur4", "Only confocal binarized aggressively (q65), fixed stays grayscale blur4", _moving_binary_variant(65.0)),
        ("moving_binary_q75_fixed_blur4", "Only confocal binarized aggressively (q75), fixed stays grayscale blur4", _moving_binary_variant(75.0)),
        ("moving_binary_dt_q65_fixed_blur4", "Only confocal binarized+distance-transform (q65), fixed stays grayscale blur4", _moving_binary_dt_variant(65.0)),
    ]


def main() -> None:
    mtfp.OUT_ROOT = OUT_ROOT
    mtfp._method_variants = _custom_method_variants  # type: ignore[assignment]
    out_root = mtfp._ensure_dir(OUT_ROOT)
    figs_dir = mtfp._ensure_dir(out_root / "figures")
    process_dir = mtfp._ensure_dir(out_root / "process")
    methods_dir = mtfp._ensure_dir(out_root / "methods")

    contexts, _common = mtfp._build_tile_contexts()
    sample_tile_ids = mtfp._select_sample_tiles(contexts)
    selected_contexts = [ctx for ctx in contexts if int(ctx.tile_index) in sample_tile_ids]
    results = mtfp._run_methods(selected_contexts)

    method_rows = []
    tile_rows = []
    for result in results:
        method_rows.append({"method": result.name, "description": result.description, **result.summary})
        method_dir = mtfp._ensure_dir(methods_dir / result.name)
        mtfp._save_method_sample_sheet(result, sample_tile_ids, method_dir / "sample_tile_qc.png")
        for row in result.tile_rows:
            out = {k: v for k, v in row.items() if k != "panels"}
            tile_rows.append(out)

    mtfp._write_csv(process_dir / "method_summary.csv", method_rows)
    mtfp._write_csv(process_dir / "tile_method_metrics.csv", tile_rows)
    mtfp._save_tile_winner_csv(results, process_dir / "tile_winner_methods.csv")
    mtfp._save_summary_plot(results, figs_dir / "method_ranking.png")

    ranked_by_current = sorted(results, key=lambda r: float(r.summary["mean_current_proc_cc"]), reverse=True)
    ranked_by_delta = sorted(results, key=lambda r: float(r.summary["mean_delta_small_cc"]), reverse=True)

    summary_lines = [
        "# Moving-only Binary Probe",
        "",
        "Question:",
        "- keep myelin as grayscale processed",
        "- only binarize / aggressively enhance confocal",
        "- compare against the current strongest non-binary baselines",
        "",
        "Remembered best baseline from previous round:",
        "- `paired_percentile_blur6`",
        "",
        "Selected sample tiles:",
        "- " + ", ".join(f"T{int(tid):02d}" for tid in sample_tile_ids),
        "",
        "Top methods by mean current_proc_cc:",
    ]
    for result in ranked_by_current:
        s = result.summary
        summary_lines.append(
            f"- `{result.name}`: current={float(s['mean_current_proc_cc']):.4f}, "
            f"best_small={float(s['mean_best_small_proc_cc']):.4f}, "
            f"delta={float(s['mean_delta_small_cc']):.4f}, "
            f"mean shift={float(s['mean_small_shift_mag']):.2f}px"
        )
    summary_lines.extend(["", "Top methods by improvement headroom:"])
    for result in ranked_by_delta:
        s = result.summary
        summary_lines.append(
            f"- `{result.name}`: delta={float(s['mean_delta_small_cc']):.4f}, current={float(s['mean_current_proc_cc']):.4f}"
        )
    summary_lines.extend(
        [
            "",
            "Important files:",
            f"- `{process_dir / 'method_summary.csv'}`",
            f"- `{process_dir / 'tile_method_metrics.csv'}`",
            f"- `{process_dir / 'tile_winner_methods.csv'}`",
            f"- `{figs_dir / 'method_ranking.png'}`",
            "- `methods/<method>/sample_tile_qc.png`",
        ]
    )
    (out_root / "summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    mtfp._write_json(
        out_root / "run_manifest.json",
        {
            "remembered_best_previous_round": "paired_percentile_blur6",
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

    print(f"Moving-only binary probe written to: {out_root}")


if __name__ == "__main__":
    main()

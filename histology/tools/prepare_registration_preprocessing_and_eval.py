from __future__ import annotations

import json
import os
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

REPO_HISTOLOGY_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_HISTOLOGY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_HISTOLOGY_ROOT))

from gui_mvp.hitl_gui.application.pair_registration import (  # noqa: E402
    PairRegistrationConfig,
    _ants_apply,
    _common_canvas_for_pair,
    _compute_stage_heatmap,
    _prepare_side,
    _run_logged,
    _stage_command,
    _stage_transforms,
    _write_coord_images,
    compute_registration_metrics,
    default_pair_registration_runs_root,
    find_ants_bin,
    gray_preview_panel,
    metrics_note,
    overlay_preview,
    read_nifti_2d,
    render_storyboard,
    write_nifti_2d,
)
from gui_mvp.hitl_gui.application.pair_workspace import load_pair_registry  # noqa: E402


DEFAULT_MYELIN_ROOT = Path(r"/mnt/d/Research/Image Analysis/Nanozoomer scans/20250327 rat myelin quantification/Tissue&Masks")
DEFAULT_NISSL_ROOT = Path(r"/mnt/d/Research/Image Analysis/Nanozoomer scans/20250424 Nissl cytoarchitectonic counterpart/Tissue&Masks")
PAIR_MASKS_ROOT = Path(r"/mnt/d/Research/Image Analysis/Nanozoomer scans/histology_pair_registration_masks_20260324")
RUNS_ROOT = Path(r"/mnt/d/Research/Image Analysis/Nanozoomer scans/histology_pair_registration_runs")
STORYBOARD_COL_TITLES = ("Moving", "Fixed", "Overlay", "Warp Field")

VARIANTS = ("baseline", "clip", "clip_norm", "clip_norm_clahe")
EXPERIMENT_CASES = [
    ("2501_24__2501_25", "1"),
    ("2501_24__2501_25", "2"),
    ("2504_72__2504_73", "1"),
    ("2504_72__2504_73", "2"),
    ("2506_24__2506_25", "1"),
    ("2506_24__2506_25", "2"),
    ("2501_102__2501_103", "all"),
    ("2501_114__2501_115", "all"),
    ("2502_102__2502_79", "all"),
    ("2504_108__2504_109", "all"),
]


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _group_choices(review: dict[str, Any]) -> list[str]:
    if bool(review.get("multi_group_registration")):
        return ["1", "2"]
    return ["all"]


def _support_mask(labels: np.ndarray) -> np.ndarray:
    return labels > 0


def _clip_inside_mask(gray_u8: np.ndarray, support: np.ndarray, pct_low: float = 1.0, pct_high: float = 99.0) -> np.ndarray:
    out = gray_u8.copy()
    vals = gray_u8[support]
    if vals.size == 0:
        return out
    lo = float(np.percentile(vals, pct_low))
    hi = float(np.percentile(vals, pct_high))
    if hi <= lo:
        return out
    clipped = np.clip(vals.astype(np.float32), lo, hi)
    out[support] = np.round(clipped).astype(np.uint8)
    return out


def _normalize_inside_mask(gray_u8: np.ndarray, support: np.ndarray) -> np.ndarray:
    out = gray_u8.copy()
    vals = gray_u8[support].astype(np.float32)
    if vals.size == 0:
        return out
    lo = float(vals.min())
    hi = float(vals.max())
    if hi <= lo:
        return out
    norm = (vals - lo) / (hi - lo)
    out[support] = np.clip(np.round(norm * 255.0), 0, 255).astype(np.uint8)
    return out


def _clahe_inside_mask(gray_u8: np.ndarray, support: np.ndarray) -> np.ndarray:
    out = gray_u8.copy()
    ys, xs = np.where(support)
    if ys.size == 0 or xs.size == 0:
        return out
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    roi = out[y1:y2, x1:x2].copy()
    roi_support = support[y1:y2, x1:x2]
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    roi_eq = clahe.apply(roi)
    roi[roi_support] = roi_eq[roi_support]
    out[y1:y2, x1:x2] = roi
    return out


def preprocess_gray_variant(rgb: np.ndarray, labels: np.ndarray, variant: str) -> tuple[np.ndarray, np.ndarray]:
    support = _support_mask(labels)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gray[~support] = 255
    if variant == "baseline":
        return gray, labels.copy()
    if variant == "clip":
        out = _clip_inside_mask(gray, support)
        out[~support] = 255
        return out, labels.copy()
    if variant == "clip_norm":
        out = _clip_inside_mask(gray, support)
        out = _normalize_inside_mask(out, support)
        out[~support] = 255
        return out, labels.copy()
    if variant == "clip_norm_clahe":
        out = _clip_inside_mask(gray, support)
        out = _normalize_inside_mask(out, support)
        out = _clahe_inside_mask(out, support)
        out[~support] = 255
        return out, labels.copy()
    raise ValueError(variant)


def _gray_u8_to_float(gray_u8: np.ndarray) -> np.ndarray:
    return gray_u8.astype(np.float32) / 255.0


def _base_cfg(pair_key: str, review: dict[str, Any]) -> PairRegistrationConfig:
    common_root = Path(os.path.commonpath([str(DEFAULT_MYELIN_ROOT.resolve()), str(DEFAULT_NISSL_ROOT.resolve())]))
    runs_root = default_pair_registration_runs_root(DEFAULT_MYELIN_ROOT, DEFAULT_NISSL_ROOT)
    if runs_root is None:
        raise RuntimeError("Failed to resolve runs root.")
    ants_bin = find_ants_bin()
    if ants_bin is None:
        raise RuntimeError("ANTs not found.")
    return PairRegistrationConfig(
        pair_key=pair_key,
        moving_side="myelin",
        fixed_side="nissl",
        moving_group="all",
        fixed_group="all",
        review=review,
        common_root=common_root,
        myelin_root=DEFAULT_MYELIN_ROOT,
        nissl_root=DEFAULT_NISSL_ROOT,
        ants_bin=ants_bin,
        runs_root=runs_root,
        target_um_per_px=10.0,
        registration_mask_mode="union",
        run_stages=("rigid", "affine"),
        affine_profile="current",
    )


def _prepare_pair_group(pair_key: str, review: dict[str, Any], group: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cfg = _base_cfg(pair_key, review)
    fixed_rgb, fixed_labels, _ = _prepare_side(cfg, "nissl", group)
    moving_rgb, moving_labels, _ = _prepare_side(cfg, "myelin", group)
    return _common_canvas_for_pair(fixed_rgb, fixed_labels, moving_rgb, moving_labels)


def export_all_usable_preprocessed(registry: dict[str, Any]) -> dict[str, Any]:
    total_groups = 0
    total_images = 0
    usable_pairs = [
        (pair_key, review)
        for pair_key, review in sorted(registry.items())
        if isinstance(review, dict) and str(review.get("registration_status", "")).lower() == "usable"
    ]
    for pair_key, review in usable_pairs:
        pair_dir = PAIR_MASKS_ROOT / pair_key / "preprocessed_target10_union"
        for group in _group_choices(review):
            fixed_rgb, fixed_labels, moving_rgb, moving_labels = _prepare_pair_group(pair_key, review, group)
            group_dir = pair_dir / f"group_{group}"
            group_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(group_dir / "nissl_mask_labels.png"), fixed_labels)
            cv2.imwrite(str(group_dir / "myelin_mask_labels.png"), moving_labels)
            manifest = {
                "pair_key": pair_key,
                "group": group,
                "moving_side": "myelin",
                "fixed_side": "nissl",
                "target_um_per_px": 10.0,
                "registration_mask_mode": "union",
                "variants": {},
            }
            for side, rgb, labels in (
                ("nissl", fixed_rgb, fixed_labels),
                ("myelin", moving_rgb, moving_labels),
            ):
                for variant in VARIANTS:
                    gray_u8, _ = preprocess_gray_variant(rgb, labels, variant)
                    out_path = group_dir / f"{side}_{variant}_gray.png"
                    cv2.imwrite(str(out_path), gray_u8)
                    manifest["variants"].setdefault(side, {})[variant] = str(out_path)
                    total_images += 1
            _write_json(group_dir / "manifest.json", manifest)
            total_groups += 1
    return {
        "usable_pairs": len(usable_pairs),
        "prepared_groups": total_groups,
        "written_gray_images": total_images,
        "root": str(PAIR_MASKS_ROOT / "preprocessed_target10_union"),
    }


def _run_registration_variant(
    pair_key: str,
    review: dict[str, Any],
    group: str,
    variant: str,
) -> dict[str, Any]:
    cfg = _base_cfg(pair_key, review)
    ants_bin = cfg.ants_bin
    fixed_rgb, fixed_labels, moving_rgb, moving_labels = _prepare_pair_group(pair_key, review, group)
    fixed_gray_u8, fixed_labels = preprocess_gray_variant(fixed_rgb, fixed_labels, variant)
    moving_gray_u8, moving_labels = preprocess_gray_variant(moving_rgb, moving_labels, variant)
    fixed_gray = _gray_u8_to_float(fixed_gray_u8)
    moving_gray = _gray_u8_to_float(moving_gray_u8)
    fixed_mask = (fixed_labels > 0).astype(np.float32)
    moving_mask = (moving_labels > 0).astype(np.float32)

    run_id = f"{_utc_stamp()}_preproc_{variant}_myelin_{group}_to_nissl_{group}"
    run_dir = RUNS_ROOT / pair_key / run_id
    inputs_dir = run_dir / "inputs"
    stages_dir = run_dir / "stages"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    stages_dir.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(inputs_dir / "fixed_gray.png"), fixed_gray_u8)
    cv2.imwrite(str(inputs_dir / "moving_gray.png"), moving_gray_u8)
    cv2.imwrite(str(inputs_dir / "fixed_mask_labels.png"), fixed_labels)
    cv2.imwrite(str(inputs_dir / "moving_mask_labels.png"), moving_labels)

    fixed_img_path = inputs_dir / "fixed_gray.nii.gz"
    moving_img_path = inputs_dir / "moving_gray.nii.gz"
    fixed_mask_path = inputs_dir / "fixed_mask.nii.gz"
    moving_mask_path = inputs_dir / "moving_mask.nii.gz"
    write_nifti_2d(fixed_img_path, fixed_gray)
    write_nifti_2d(moving_img_path, moving_gray)
    write_nifti_2d(fixed_mask_path, fixed_mask)
    write_nifti_2d(moving_mask_path, moving_mask)
    moving_coord_x, moving_coord_y = _write_coord_images(inputs_dir, moving_gray.shape[:2])

    input_metrics, input_metric_timing = compute_registration_metrics(fixed_gray, moving_gray, fixed_mask, moving_mask)
    input_note = metrics_note(input_metrics, input_metric_timing, f"{variant} before registration")
    stage_records: dict[str, dict[str, Any]] = {}
    storyboard_path = run_dir / "storyboard.png"
    render_storyboard(
        [
            {
                "label": "Input",
                "note": input_note,
                "fixed": gray_preview_panel(fixed_gray),
                "moving": gray_preview_panel(moving_gray),
                "overlay": overlay_preview(fixed_gray, moving_gray, fixed_mask, moving_mask),
                "heatmap": np.full((*fixed_gray.shape, 3), 235, dtype=np.uint8),
                "col_titles": STORYBOARD_COL_TITLES,
            }
        ],
        storyboard_path,
    )

    rigid_mat = stages_dir / "rigid" / "rigid_0GenericAffine.mat"
    affine_mat = stages_dir / "affine" / "affine_0GenericAffine.mat"
    manifest: dict[str, Any] = {
        "pair_key": pair_key,
        "group": group,
        "variant": variant,
        "moving_side": "myelin",
        "fixed_side": "nissl",
        "moving_group": group,
        "fixed_group": group,
        "target_um_per_px": 10.0,
        "registration_mask_mode": "union",
        "run_stages": ["rigid", "affine"],
        "affine_profile": "current",
        "inputs": {
            "fixed_gray_png": str(inputs_dir / "fixed_gray.png"),
            "moving_gray_png": str(inputs_dir / "moving_gray.png"),
        },
        "input_metrics": input_metrics,
        "input_metric_timing_seconds": input_metric_timing,
        "timing_seconds": {},
        "stages": {},
    }

    total_t0 = time.perf_counter()
    for stage, init_tfms, progress_percent in (
        ("rigid", [], 50),
        ("affine", [rigid_mat], 100),
    ):
        stage_t0 = time.perf_counter()
        stage_dir = stages_dir / stage
        stage_dir.mkdir(parents=True, exist_ok=True)
        prefix = stage_dir / f"{stage}_"
        cmd = _stage_command(
            ants_bin,
            stage,
            fixed_img_path,
            moving_img_path,
            fixed_mask_path,
            moving_mask_path,
            prefix,
            init_tfms,
            "current",
        )
        ants_t0 = time.perf_counter()
        _run_logged(cmd, stage_dir / f"{stage}.log")
        ants_seconds = float(time.perf_counter() - ants_t0)

        warped_img_path = stage_dir / f"{stage}_Warped.nii.gz"
        tfms = _stage_transforms(stage_dir, stage, rigid_mat, affine_mat)
        warped_mask_path = stage_dir / f"{stage}_warped_mask.nii.gz"
        _ants_apply(
            ants_bin,
            moving_mask_path,
            fixed_img_path,
            warped_mask_path,
            tfms,
            interpolation="NearestNeighbor",
            log_path=stage_dir / f"{stage}_warp_mask.log",
        )
        warped_gray = read_nifti_2d(warped_img_path)
        warped_mask = read_nifti_2d(warped_mask_path)
        stage_metrics, stage_metric_timing = compute_registration_metrics(
            fixed_gray,
            np.clip(warped_gray, 0.0, 1.0),
            fixed_mask,
            (warped_mask > 0.5).astype(np.float32),
        )
        overlay = overlay_preview(
            fixed_gray,
            np.clip(warped_gray, 0.0, 1.0),
            fixed_mask,
            (warped_mask > 0.5).astype(np.float32),
        )
        heatmap_rgb, heatmap_png = _compute_stage_heatmap(
            ants_bin,
            stage_dir,
            stage,
            fixed_img_path,
            fixed_mask,
            moving_coord_x,
            moving_coord_y,
            rigid_mat,
            affine_mat,
            warped_mask_path,
        )
        stage_seconds = float(time.perf_counter() - stage_t0)
        stage_records[stage] = {
            "moving": gray_preview_panel(np.clip(warped_gray, 0.0, 1.0)),
            "overlay": overlay,
            "heatmap": heatmap_rgb,
            "note": metrics_note(stage_metrics, stage_metric_timing, f"{variant} {stage} finished"),
        }
        render_storyboard(
            [
                {
                    "label": "Input",
                    "note": input_note,
                    "fixed": gray_preview_panel(fixed_gray),
                    "moving": gray_preview_panel(moving_gray),
                    "overlay": overlay_preview(fixed_gray, moving_gray, fixed_mask, moving_mask),
                    "heatmap": np.full((*fixed_gray.shape, 3), 235, dtype=np.uint8),
                    "col_titles": STORYBOARD_COL_TITLES,
                },
                *[
                    {
                        "label": s.capitalize(),
                        "note": stage_records[s]["note"],
                        "fixed": gray_preview_panel(fixed_gray),
                        "moving": stage_records[s]["moving"],
                        "overlay": stage_records[s]["overlay"],
                        "heatmap": stage_records[s]["heatmap"],
                        "col_titles": STORYBOARD_COL_TITLES,
                    }
                    for s in ("rigid", "affine")
                    if s in stage_records
                ],
            ],
            storyboard_path,
        )
        manifest["stages"][stage] = {
            "warped_image": str(warped_img_path),
            "warped_mask": str(warped_mask_path),
            "heatmap_png": str(heatmap_png),
            "metrics": stage_metrics,
            "metric_timing_seconds": stage_metric_timing,
            "timing_seconds": {
                "stage_total": stage_seconds,
                "ants_registration": ants_seconds,
                "postprocess": float(max(stage_seconds - ants_seconds, 0.0)),
            },
        }
        manifest["timing_seconds"][stage] = stage_seconds
    manifest["timing_seconds"]["total"] = float(time.perf_counter() - total_t0)
    _write_json(run_dir / "run_manifest.json", manifest)
    return manifest


def _aggregate_variant(results: list[dict[str, Any]], variant: str) -> dict[str, float]:
    rows = [row["variants"][variant] for row in results if variant in row["variants"]]
    return {
        "mean_total_s": round(float(statistics.mean(r["timing_seconds"]["total"] for r in rows)), 3),
        "mean_affine_s": round(float(statistics.mean(r["timing_seconds"]["affine"] for r in rows)), 3),
        "mean_input_dice": round(float(statistics.mean(r["input_metrics"]["dice"] for r in rows)), 6),
        "mean_affine_dice": round(float(statistics.mean(r["stages"]["affine"]["metrics"]["dice"] for r in rows)), 6),
        "mean_affine_hd95_px": round(float(statistics.mean(r["stages"]["affine"]["metrics"]["hd95_px"] for r in rows)), 6),
        "mean_affine_mi": round(float(statistics.mean(r["stages"]["affine"]["metrics"]["mi"] for r in rows)), 6),
        "mean_affine_cc": round(float(statistics.mean(r["stages"]["affine"]["metrics"]["cc"] for r in rows)), 6),
    }


def _write_experiment_md(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Registration Preprocessing Experiment",
        "",
        f"- started_at_utc: `{payload['started_at_utc']}`",
        f"- completed_at_utc: `{payload.get('completed_at_utc', '')}`",
        f"- usable_pairs_preprocessed: `{payload['preprocess_export']['usable_pairs']}`",
        f"- prepared_groups: `{payload['preprocess_export']['prepared_groups']}`",
        f"- written_gray_images: `{payload['preprocess_export']['written_gray_images']}`",
        f"- preprocessed_root: `{payload['preprocess_export']['root']}`",
        "",
        "## Variants",
        "",
        "- `baseline`: current pair-specific/group-aware crop after physical scale normalization, background fixed to white.",
        "- `clip`: add mask-internal percentile clipping.",
        "- `clip_norm`: clipping + mask-internal min-max normalization.",
        "- `clip_norm_clahe`: clipping + normalization + local contrast enhancement (CLAHE).",
        "",
        "## Aggregate",
        "",
    ]
    for variant, stats in payload["aggregate"].items():
        lines.append(f"### {variant}")
        for k, v in stats.items():
            lines.append(f"- {k}: `{v}`")
        lines.append("")
    lines.extend(["## Conclusions", ""])
    conclusions = payload.get("conclusions", [])
    for line in conclusions:
        lines.append(f"- {line}")
    lines.extend(["", "## Per Case", ""])
    for row in payload["results"]:
        lines.append(f"### {row['pair_key']} | group {row['group']}")
        for variant in VARIANTS:
            rec = row["variants"][variant]
            aff = rec["stages"]["affine"]["metrics"]
            lines.extend(
                [
                    f"- {variant}:",
                    f"  - total_s: `{rec['timing_seconds']['total']:.2f}`",
                    f"  - affine_dice: `{aff['dice']:.4f}`",
                    f"  - affine_hd95_px: `{aff['hd95_px']:.2f}`",
                    f"  - affine_mi: `{aff['mi']:.4f}`",
                    f"  - affine_cc: `{aff['cc']:.4f}`",
                ]
            )
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    common_root = Path(os.path.commonpath([str(DEFAULT_MYELIN_ROOT.resolve()), str(DEFAULT_NISSL_ROOT.resolve())]))
    registry = load_pair_registry(common_root / "histology_pair_qc_registry.json")

    preprocess_export = export_all_usable_preprocessed(registry)
    experiment_id = f"{_utc_stamp()}_registration_preprocessing_probe"
    summary_json = PAIR_MASKS_ROOT / f"{experiment_id}.json"
    summary_md = PAIR_MASKS_ROOT / f"{experiment_id}.md"
    payload: dict[str, Any] = {
        "experiment_id": experiment_id,
        "started_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "preprocess_export": preprocess_export,
        "results": [],
        "aggregate": {},
        "conclusions": [],
    }
    _write_json(summary_json, payload)

    for idx, (pair_key, group) in enumerate(EXPERIMENT_CASES, start=1):
        review = registry[pair_key]
        print(f"[{idx}/{len(EXPERIMENT_CASES)}] {pair_key} group={group}", flush=True)
        row = {"pair_key": pair_key, "group": group, "variants": {}}
        for variant in VARIANTS:
            manifest = _run_registration_variant(pair_key, review, group, variant)
            row["variants"][variant] = manifest
            aff = manifest["stages"]["affine"]["metrics"]
            print(
                f"  {variant}: total={manifest['timing_seconds']['total']:.1f}s "
                f"dice={aff['dice']:.4f} hd95={aff['hd95_px']:.2f} mi={aff['mi']:.4f} cc={aff['cc']:.4f}",
                flush=True,
            )
        payload["results"].append(row)
        _write_json(summary_json, payload)

    payload["aggregate"] = {variant: _aggregate_variant(payload["results"], variant) for variant in VARIANTS}
    baseline = payload["aggregate"]["baseline"]
    conclusions = []
    for variant in ("clip", "clip_norm", "clip_norm_clahe"):
        stats = payload["aggregate"][variant]
        conclusions.append(
            f"{variant}: Dice delta vs baseline = {stats['mean_affine_dice'] - baseline['mean_affine_dice']:+.6f}, "
            f"HD95 delta = {stats['mean_affine_hd95_px'] - baseline['mean_affine_hd95_px']:+.6f}, "
            f"MI delta = {stats['mean_affine_mi'] - baseline['mean_affine_mi']:+.6f}, "
            f"CC delta = {stats['mean_affine_cc'] - baseline['mean_affine_cc']:+.6f}."
        )
    payload["conclusions"] = conclusions
    payload["completed_at_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    _write_json(summary_json, payload)
    _write_experiment_md(summary_md, payload)
    print(f"summary_json={summary_json}")
    print(f"summary_md={summary_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

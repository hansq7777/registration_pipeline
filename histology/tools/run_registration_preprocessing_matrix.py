from __future__ import annotations

import concurrent.futures as cf
import csv
import json
import os
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

REPO_HISTOLOGY_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_HISTOLOGY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_HISTOLOGY_ROOT))

from gui_mvp.hitl_gui.application.pair_registration import (  # noqa: E402
    _ants_apply,
    _common_canvas_for_pair_with_info,
    _prepare_side,
    _resize_pair_to_working_long_edge,
    _run_logged,
    _stage_command,
    _stage_transforms,
    compute_registration_metrics,
    find_ants_bin,
    read_nifti_2d,
    write_nifti_2d,
)
from gui_mvp.hitl_gui.application.pair_workspace import load_pair_registry  # noqa: E402
from gui_mvp.hitl_gui.application.pair_registration import PairRegistrationConfig  # noqa: E402


DEFAULT_MYELIN_ROOT = Path(r"/mnt/d/Research/Image Analysis/Nanozoomer scans/20250327 rat myelin quantification/Tissue&Masks")
DEFAULT_NISSL_ROOT = Path(r"/mnt/d/Research/Image Analysis/Nanozoomer scans/20250424 Nissl cytoarchitectonic counterpart/Tissue&Masks")
NANOZOOMER_ROOT = Path(r"/mnt/d/Research/Image Analysis/Nanozoomer scans")
DEFAULT_WORKERS = min(4, max(1, (os.cpu_count() or 1) // 2))
TARGET_UM_PER_PX = 10.0
WORKING_LONG_EDGE = 1024

VARIANT_SPECS: dict[str, dict[str, Any]] = {
    "clip_white": {"kind": "clip", "blur_sigma": 0.0, "background": 255},
    "clip_white_blur_0.5": {"kind": "clip", "blur_sigma": 0.5, "background": 255},
    "clip_white_blur_1.0": {"kind": "clip", "blur_sigma": 1.0, "background": 255},
    "clip_white_blur_1.5": {"kind": "clip", "blur_sigma": 1.5, "background": 255},
    "clip_white_blur_2.0": {"kind": "clip", "blur_sigma": 2.0, "background": 255},
    "clip_norm_white": {"kind": "clip_norm", "blur_sigma": 0.0, "background": 255},
    "clip_norm_white_blur_1.0": {"kind": "clip_norm", "blur_sigma": 1.0, "background": 255},
    "clip_norm_clahe_white": {"kind": "clip_norm_clahe", "blur_sigma": 0.0, "background": 255},
    "gradient_mag_blur_1.0": {"kind": "gradient_mag", "blur_sigma": 1.0, "background": 0},
    "gradient_mag_blur_1.5": {"kind": "gradient_mag", "blur_sigma": 1.5, "background": 0},
}


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _safe_float(v: Any) -> float:
    try:
        return float(v)
    except Exception:
        return float("nan")


def _clip_inside_mask(gray_u8: np.ndarray, tissue_mask: np.ndarray, pct_low: float = 1.0, pct_high: float = 99.0) -> np.ndarray:
    out = gray_u8.copy()
    vals = gray_u8[tissue_mask]
    if vals.size == 0:
        return out
    lo = float(np.percentile(vals, pct_low))
    hi = float(np.percentile(vals, pct_high))
    if hi <= lo:
        return out
    clipped = np.clip(vals.astype(np.float32), lo, hi)
    out[tissue_mask] = np.round(clipped).astype(np.uint8)
    return out


def _normalize_inside_mask(gray_u8: np.ndarray, tissue_mask: np.ndarray) -> np.ndarray:
    out = gray_u8.copy()
    vals = gray_u8[tissue_mask].astype(np.float32)
    if vals.size == 0:
        return out
    lo = float(vals.min())
    hi = float(vals.max())
    if hi <= lo:
        return out
    norm = (vals - lo) / (hi - lo)
    out[tissue_mask] = np.clip(np.round(norm * 255.0), 0, 255).astype(np.uint8)
    return out


def _clahe_inside_mask(gray_u8: np.ndarray, tissue_mask: np.ndarray) -> np.ndarray:
    out = gray_u8.copy()
    ys, xs = np.where(tissue_mask)
    if ys.size == 0 or xs.size == 0:
        return out
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    roi = out[y1:y2, x1:x2].copy()
    roi_tissue = tissue_mask[y1:y2, x1:x2]
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    roi_eq = clahe.apply(roi)
    roi[roi_tissue] = roi_eq[roi_tissue]
    out[y1:y2, x1:x2] = roi
    return out


def _gaussian_blur_gray(gray_u8: np.ndarray, sigma: float) -> np.ndarray:
    sigma = float(sigma)
    if sigma <= 0.0:
        return gray_u8.copy()
    return cv2.GaussianBlur(gray_u8, (0, 0), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REPLICATE)


def _gradient_mag(gray_u8: np.ndarray) -> np.ndarray:
    gray_f = gray_u8.astype(np.float32) / 255.0
    gx = cv2.Sobel(gray_f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_f, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    return mag


def _scale_float_inside_mask(arr: np.ndarray, tissue_mask: np.ndarray, background_value: int) -> np.ndarray:
    out = np.full(arr.shape, int(background_value), dtype=np.uint8)
    vals = arr[tissue_mask]
    if vals.size == 0:
        return out
    vmax = float(np.percentile(vals, 99.0))
    if vmax <= 1e-8:
        vmax = float(vals.max()) if vals.size else 0.0
    if vmax <= 1e-8:
        return out
    scaled = np.clip(np.round((vals / vmax) * 255.0), 0, 255).astype(np.uint8)
    out[tissue_mask] = scaled
    return out


def preprocess_variant(rgb: np.ndarray, labels: np.ndarray, variant: str) -> tuple[np.ndarray, np.ndarray]:
    spec = VARIANT_SPECS[variant]
    tissue = labels == 1
    background_value = int(spec["background"])
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    kind = str(spec["kind"])
    blur_sigma = float(spec["blur_sigma"])

    if kind == "clip":
        out = _clip_inside_mask(gray, tissue)
        out = _gaussian_blur_gray(out, blur_sigma)
        out[~tissue] = background_value
        return out, labels.copy()

    if kind == "clip_norm":
        out = _clip_inside_mask(gray, tissue)
        out = _normalize_inside_mask(out, tissue)
        out = _gaussian_blur_gray(out, blur_sigma)
        out[~tissue] = background_value
        return out, labels.copy()

    if kind == "clip_norm_clahe":
        out = _clip_inside_mask(gray, tissue)
        out = _normalize_inside_mask(out, tissue)
        out = _clahe_inside_mask(out, tissue)
        out = _gaussian_blur_gray(out, blur_sigma)
        out[~tissue] = background_value
        return out, labels.copy()

    if kind == "gradient_mag":
        base = _clip_inside_mask(gray, tissue)
        base = _gaussian_blur_gray(base, blur_sigma)
        mag = _gradient_mag(base)
        out = _scale_float_inside_mask(mag, tissue, background_value=background_value)
        return out, labels.copy()

    raise ValueError(f"Unknown variant: {variant}")


def _gray_u8_to_float(gray_u8: np.ndarray, labels: np.ndarray, background_value: int) -> np.ndarray:
    gray = gray_u8.astype(np.float32) / 255.0
    gray[labels != 1] = float(background_value) / 255.0
    return gray


def _group_choices(review: dict[str, Any]) -> list[str]:
    if bool(review.get("multi_group_registration")):
        return ["1", "2"]
    return ["all"]


@dataclass(frozen=True)
class RegistrationUnit:
    pair_key: str
    group: str

    @property
    def key(self) -> str:
        return f"{self.pair_key}__group_{self.group}"


def _base_cfg(pair_key: str, review: dict[str, Any], group: str) -> PairRegistrationConfig:
    common_root = Path(os.path.commonpath([str(DEFAULT_MYELIN_ROOT.resolve()), str(DEFAULT_NISSL_ROOT.resolve())]))
    ants_bin = find_ants_bin()
    if ants_bin is None:
        raise RuntimeError("ANTs not found.")
    return PairRegistrationConfig(
        pair_key=pair_key,
        moving_side="nissl",
        fixed_side="myelin",
        moving_group=group,
        fixed_group=group,
        review=review,
        common_root=common_root,
        myelin_root=DEFAULT_MYELIN_ROOT,
        nissl_root=DEFAULT_NISSL_ROOT,
        ants_bin=ants_bin,
        runs_root=NANOZOOMER_ROOT / "histology_pair_registration_runs",
        target_um_per_px=TARGET_UM_PER_PX,
        working_long_edge=WORKING_LONG_EDGE,
        pre_blur_sigma=0.0,
        registration_mask_mode="tissue_only",
        run_stages=("rigid", "affine"),
        affine_profile="current",
    )


def _build_geometry_for_unit(
    unit: RegistrationUnit,
    review: dict[str, Any],
    geom_root: Path,
) -> dict[str, Any]:
    unit_dir = geom_root / unit.key
    unit_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = unit_dir / "geometry_manifest.json"
    fixed_rgb_path = unit_dir / "fixed_myelin_rgb_1024.png"
    moving_rgb_path = unit_dir / "moving_nissl_rgb_1024.png"
    fixed_labels_path = unit_dir / "fixed_myelin_labels_1024.png"
    moving_labels_path = unit_dir / "moving_nissl_labels_1024.png"

    if all(p.exists() for p in (manifest_path, fixed_rgb_path, moving_rgb_path, fixed_labels_path, moving_labels_path)):
        return json.loads(manifest_path.read_text(encoding="utf-8"))

    cfg = _base_cfg(unit.pair_key, review, unit.group)
    fixed_rgb, fixed_labels, fixed_pre = _prepare_side(cfg, "myelin", unit.group)
    moving_rgb, moving_labels, moving_pre = _prepare_side(cfg, "nissl", unit.group)
    fixed_rgb, fixed_labels, moving_rgb, moving_labels, working_info = _resize_pair_to_working_long_edge(
        fixed_rgb,
        fixed_labels,
        moving_rgb,
        moving_labels,
        working_long_edge=WORKING_LONG_EDGE,
        blur_sigma=0.0,
    )
    (
        fixed_rgb,
        fixed_labels,
        moving_rgb,
        moving_labels,
        fixed_offset,
        moving_offset,
        common_canvas_shape_hw,
    ) = _common_canvas_for_pair_with_info(
        fixed_rgb,
        fixed_labels,
        moving_rgb,
        moving_labels,
    )

    cv2.imwrite(str(fixed_rgb_path), cv2.cvtColor(fixed_rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(moving_rgb_path), cv2.cvtColor(moving_rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(fixed_labels_path), fixed_labels)
    cv2.imwrite(str(moving_labels_path), moving_labels)

    manifest = {
        "unit_key": unit.key,
        "pair_key": unit.pair_key,
        "group": unit.group,
        "fixed_side": "myelin",
        "moving_side": "nissl",
        "target_um_per_px": TARGET_UM_PER_PX,
        "working_long_edge": WORKING_LONG_EDGE,
        "mask_mode": "tissue_only",
        "fixed_preprocess": fixed_pre,
        "moving_preprocess": moving_pre,
        "working_info": working_info,
        "fixed_offset_xy": fixed_offset,
        "moving_offset_xy": moving_offset,
        "common_canvas_shape_hw": common_canvas_shape_hw,
        "cache_files": {
            "fixed_rgb_png": str(fixed_rgb_path),
            "moving_rgb_png": str(moving_rgb_path),
            "fixed_labels_png": str(fixed_labels_path),
            "moving_labels_png": str(moving_labels_path),
        },
    }
    _write_json(manifest_path, manifest)
    return manifest


def _build_geometry_task(task: tuple[str, str, dict[str, Any], str]) -> str:
    pair_key, group, review, geom_root_s = task
    unit = RegistrationUnit(pair_key=pair_key, group=group)
    _build_geometry_for_unit(unit, review, Path(geom_root_s))
    return unit.key


def _stage_metrics_and_time(
    ants_bin: Path,
    stage: str,
    stage_dir: Path,
    fixed_img_path: Path,
    moving_img_path: Path,
    fixed_mask_path: Path,
    moving_mask_path: Path,
    rigid_mat: Path,
    affine_mat: Path,
    fixed_gray: np.ndarray,
    fixed_mask: np.ndarray,
    affine_profile: str,
) -> tuple[dict[str, Any], float]:
    stage_dir.mkdir(parents=True, exist_ok=True)
    prefix = stage_dir / f"{stage}_"
    init_tfms = [] if stage == "rigid" else [rigid_mat]
    cmd = _stage_command(
        ants_bin,
        stage,
        fixed_img_path,
        moving_img_path,
        fixed_mask_path,
        moving_mask_path,
        prefix,
        init_tfms,
        affine_profile,
    )
    t0 = time.perf_counter()
    ants_t0 = time.perf_counter()
    _run_logged(cmd, stage_dir / f"{stage}.log")
    ants_seconds = float(time.perf_counter() - ants_t0)

    tfms = _stage_transforms(stage_dir, stage, rigid_mat, affine_mat)
    warped_img_path = stage_dir / f"{stage}_Warped.nii.gz"
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
    metrics, metric_timing = compute_registration_metrics(
        fixed_gray,
        np.clip(warped_gray, 0.0, 1.0),
        fixed_mask,
        (warped_mask > 0.5).astype(np.float32),
    )
    total_seconds = float(time.perf_counter() - t0)
    return {
        "metrics": metrics,
        "metric_timing_seconds": metric_timing,
        "warped_image": str(warped_img_path),
        "warped_mask": str(warped_mask_path),
        "timing_seconds": {
            "stage_total": total_seconds,
            "ants_registration": ants_seconds,
            "postprocess": float(max(total_seconds - ants_seconds, 0.0)),
        },
    }, total_seconds


def _run_task(task: tuple[str, str, str, str, str, str, str]) -> dict[str, Any]:
    pair_key, group, variant, geom_root_s, out_root_s, ants_bin_s, affine_profile = task
    geom_root = Path(geom_root_s)
    out_root = Path(out_root_s)
    ants_bin = Path(ants_bin_s)
    unit_key = f"{pair_key}__group_{group}"
    unit_dir = geom_root / unit_key
    fixed_rgb = cv2.cvtColor(cv2.imread(str(unit_dir / "fixed_myelin_rgb_1024.png"), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    moving_rgb = cv2.cvtColor(cv2.imread(str(unit_dir / "moving_nissl_rgb_1024.png"), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    fixed_labels = cv2.imread(str(unit_dir / "fixed_myelin_labels_1024.png"), cv2.IMREAD_UNCHANGED).astype(np.uint8)
    moving_labels = cv2.imread(str(unit_dir / "moving_nissl_labels_1024.png"), cv2.IMREAD_UNCHANGED).astype(np.uint8)

    fixed_gray_u8, fixed_labels = preprocess_variant(fixed_rgb, fixed_labels, variant)
    moving_gray_u8, moving_labels = preprocess_variant(moving_rgb, moving_labels, variant)
    background_value = int(VARIANT_SPECS[variant]["background"])
    fixed_gray = _gray_u8_to_float(fixed_gray_u8, fixed_labels, background_value)
    moving_gray = _gray_u8_to_float(moving_gray_u8, moving_labels, background_value)
    fixed_mask = (fixed_labels == 1).astype(np.float32)
    moving_mask = (moving_labels == 1).astype(np.float32)

    input_metrics, input_metric_timing = compute_registration_metrics(fixed_gray, moving_gray, fixed_mask, moving_mask)
    task_dir = out_root / "tasks" / unit_key / variant
    result_path = task_dir / "result.json"
    if result_path.exists():
        return json.loads(result_path.read_text(encoding="utf-8"))

    task_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f"{unit_key}_{variant}_", dir=str(out_root / "tmp")) as tmpdir_s:
        tmpdir = Path(tmpdir_s)
        fixed_img_path = tmpdir / "fixed_gray.nii.gz"
        moving_img_path = tmpdir / "moving_gray.nii.gz"
        fixed_mask_path = tmpdir / "fixed_mask.nii.gz"
        moving_mask_path = tmpdir / "moving_mask.nii.gz"
        write_nifti_2d(fixed_img_path, fixed_gray)
        write_nifti_2d(moving_img_path, moving_gray)
        write_nifti_2d(fixed_mask_path, fixed_mask)
        write_nifti_2d(moving_mask_path, moving_mask)

        stages_dir = tmpdir / "stages"
        rigid_mat = stages_dir / "rigid" / "rigid_0GenericAffine.mat"
        affine_mat = stages_dir / "affine" / "affine_0GenericAffine.mat"

        rigid_record, rigid_seconds = _stage_metrics_and_time(
            ants_bin,
            "rigid",
            stages_dir / "rigid",
            fixed_img_path,
            moving_img_path,
            fixed_mask_path,
            moving_mask_path,
            rigid_mat,
            affine_mat,
            fixed_gray,
            fixed_mask,
            affine_profile,
        )
        affine_record, affine_seconds = _stage_metrics_and_time(
            ants_bin,
            "affine",
            stages_dir / "affine",
            fixed_img_path,
            moving_img_path,
            fixed_mask_path,
            moving_mask_path,
            rigid_mat,
            affine_mat,
            fixed_gray,
            fixed_mask,
            affine_profile,
        )

        result = {
            "pair_key": pair_key,
            "group": group,
            "variant": variant,
            "unit_key": unit_key,
            "config": {
                "moving_side": "nissl",
                "fixed_side": "myelin",
                "mask_mode": "tissue_only",
                "target_um_per_px": TARGET_UM_PER_PX,
                "working_long_edge": WORKING_LONG_EDGE,
                "affine_profile": affine_profile,
            },
            "input": input_metrics,
            "input_metric_timing_seconds": input_metric_timing,
            "rigid": rigid_record["metrics"],
            "rigid_metric_timing_seconds": rigid_record["metric_timing_seconds"],
            "affine": affine_record["metrics"],
            "affine_metric_timing_seconds": affine_record["metric_timing_seconds"],
            "timing_seconds": {
                "rigid": rigid_seconds,
                "affine": affine_seconds,
                "total": rigid_seconds + affine_seconds,
            },
        }
        _write_json(result_path, result)
        return result


def _summarize_variant(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def mean(field_getter) -> float:
        vals = [_safe_float(field_getter(row)) for row in rows]
        vals = [v for v in vals if np.isfinite(v)]
        return float(np.mean(vals)) if vals else float("nan")

    return {
        "unit_count": len(rows),
        "mean_input_dice": mean(lambda r: r["input"]["dice"]),
        "mean_rigid_dice": mean(lambda r: r["rigid"]["dice"]),
        "mean_affine_dice": mean(lambda r: r["affine"]["dice"]),
        "mean_input_hd95_px": mean(lambda r: r["input"]["hd95_px"]),
        "mean_rigid_hd95_px": mean(lambda r: r["rigid"]["hd95_px"]),
        "mean_affine_hd95_px": mean(lambda r: r["affine"]["hd95_px"]),
        "mean_input_mi": mean(lambda r: r["input"]["mi"]),
        "mean_rigid_mi": mean(lambda r: r["rigid"]["mi"]),
        "mean_affine_mi": mean(lambda r: r["affine"]["mi"]),
        "mean_input_cc": mean(lambda r: r["input"]["cc"]),
        "mean_rigid_cc": mean(lambda r: r["rigid"]["cc"]),
        "mean_affine_cc": mean(lambda r: r["affine"]["cc"]),
        "mean_rigid_time_s": mean(lambda r: r["timing_seconds"]["rigid"]),
        "mean_affine_time_s": mean(lambda r: r["timing_seconds"]["affine"]),
        "rigid_dice_gt_input_count": int(sum(_safe_float(r["rigid"]["dice"]) > _safe_float(r["input"]["dice"]) for r in rows)),
        "affine_dice_gt_rigid_count": int(sum(_safe_float(r["affine"]["dice"]) > _safe_float(r["rigid"]["dice"]) for r in rows)),
        "rigid_hd95_lt_input_count": int(sum(_safe_float(r["rigid"]["hd95_px"]) < _safe_float(r["input"]["hd95_px"]) for r in rows)),
        "affine_hd95_lt_rigid_count": int(sum(_safe_float(r["affine"]["hd95_px"]) < _safe_float(r["rigid"]["hd95_px"]) for r in rows)),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "pair_key",
        "group",
        "variant",
        "input_dice",
        "rigid_dice",
        "affine_dice",
        "input_hd95_px",
        "rigid_hd95_px",
        "affine_hd95_px",
        "input_mi",
        "rigid_mi",
        "affine_mi",
        "input_cc",
        "rigid_cc",
        "affine_cc",
        "rigid_time_s",
        "affine_time_s",
        "total_time_s",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "pair_key": row["pair_key"],
                    "group": row["group"],
                    "variant": row["variant"],
                    "input_dice": row["input"]["dice"],
                    "rigid_dice": row["rigid"]["dice"],
                    "affine_dice": row["affine"]["dice"],
                    "input_hd95_px": row["input"]["hd95_px"],
                    "rigid_hd95_px": row["rigid"]["hd95_px"],
                    "affine_hd95_px": row["affine"]["hd95_px"],
                    "input_mi": row["input"]["mi"],
                    "rigid_mi": row["rigid"]["mi"],
                    "affine_mi": row["affine"]["mi"],
                    "input_cc": row["input"]["cc"],
                    "rigid_cc": row["rigid"]["cc"],
                    "affine_cc": row["affine"]["cc"],
                    "rigid_time_s": row["timing_seconds"]["rigid"],
                    "affine_time_s": row["timing_seconds"]["affine"],
                    "total_time_s": row["timing_seconds"]["total"],
                }
            )


def _write_md(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Histology Registration Preprocessing Matrix",
        "",
        f"- started_at_utc: `{payload['started_at_utc']}`",
        f"- completed_at_utc: `{payload.get('completed_at_utc', '')}`",
        f"- usable_pairs: `{payload['usable_pairs']}`",
        f"- registration_units: `{payload['registration_units']}`",
        f"- workers: `{payload['workers']}`",
        f"- working_long_edge: `{payload['config']['working_long_edge']}`",
        f"- target_um_per_px: `{payload['config']['target_um_per_px']}`",
        f"- mask_mode: `{payload['config']['mask_mode']}`",
        f"- affine_profile: `{payload['config']['affine_profile']}`",
        "",
        "## Variant Summary",
        "",
    ]
    for variant, stats in payload["aggregate"].items():
        lines.append(f"### {variant}")
        for k, v in stats.items():
            if isinstance(v, float):
                lines.append(f"- {k}: `{v:.6f}`")
            else:
                lines.append(f"- {k}: `{v}`")
        lines.append("")
    lines.extend(
        [
            "## Recommended Reading",
            "",
            "- ANHIR challenge summary: https://pmc.ncbi.nlm.nih.gov/articles/PMC7584382/",
            "- NGF / differently stained histology registration: https://arxiv.org/abs/1903.12063",
            "- Macenko stain normalization: https://www.cs.unc.edu/~mn/sites/default/files/macenko2009.pdf",
            "- MIND multimodal descriptor: https://doi.org/10.1016/j.media.2012.05.008",
            "",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    out_root = NANOZOOMER_ROOT / f"histology_registration_preproc_matrix_{_utc_stamp()}"
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "tmp").mkdir(parents=True, exist_ok=True)
    geom_root = out_root / "geometry"
    ants_bin = find_ants_bin()
    if ants_bin is None:
        raise RuntimeError("ANTs not found.")

    common_root = Path(os.path.commonpath([str(DEFAULT_MYELIN_ROOT.resolve()), str(DEFAULT_NISSL_ROOT.resolve())]))
    registry = load_pair_registry(common_root / "histology_pair_qc_registry.json")
    usable_reviews = [
        (pair_key, review)
        for pair_key, review in sorted(registry.items())
        if isinstance(review, dict) and str(review.get("registration_status", "")).lower() == "usable"
    ]
    units: list[RegistrationUnit] = []
    for pair_key, review in usable_reviews:
        for group in _group_choices(review):
            units.append(RegistrationUnit(pair_key=pair_key, group=group))

    summary = {
        "started_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "usable_pairs": len(usable_reviews),
        "registration_units": len(units),
        "workers": DEFAULT_WORKERS,
        "config": {
            "moving_side": "nissl",
            "fixed_side": "myelin",
            "working_long_edge": WORKING_LONG_EDGE,
            "target_um_per_px": TARGET_UM_PER_PX,
            "mask_mode": "tissue_only",
            "affine_profile": "current",
            "variants": list(VARIANT_SPECS),
        },
        "geometry_root": str(geom_root),
        "results_csv": str(out_root / "matrix_results.csv"),
        "aggregate_json": str(out_root / "matrix_aggregate.json"),
        "summary_md": str(out_root / "matrix_summary.md"),
    }
    _write_json(out_root / "experiment_config.json", summary)

    geom_tasks = [
        (unit.pair_key, unit.group, dict(registry[unit.pair_key]), str(geom_root))
        for unit in units
    ]
    with cf.ProcessPoolExecutor(max_workers=DEFAULT_WORKERS) as ex:
        future_map = {ex.submit(_build_geometry_task, task): task for task in geom_tasks}
        geom_done = 0
        for fut in cf.as_completed(future_map):
            unit_key = fut.result()
            geom_done += 1
            print(f"[geom {geom_done}/{len(geom_tasks)}] {unit_key}", flush=True)

    tasks = [
        (unit.pair_key, unit.group, variant, str(geom_root), str(out_root), str(ants_bin), "current")
        for unit in units
        for variant in VARIANT_SPECS
    ]
    print(f"tasks={len(tasks)} workers={DEFAULT_WORKERS}", flush=True)

    results: list[dict[str, Any]] = []
    with cf.ProcessPoolExecutor(max_workers=DEFAULT_WORKERS) as ex:
        future_map = {ex.submit(_run_task, task): task for task in tasks}
        done_count = 0
        for fut in cf.as_completed(future_map):
            task = future_map[fut]
            pair_key, group, variant = task[:3]
            result = fut.result()
            results.append(result)
            done_count += 1
            print(
                f"[{done_count}/{len(tasks)}] {pair_key} group={group} variant={variant} "
                f"input_dice={result['input']['dice']:.4f} rigid_dice={result['rigid']['dice']:.4f} "
                f"affine_dice={result['affine']['dice']:.4f}",
                flush=True,
            )

    results.sort(key=lambda r: (r["variant"], r["pair_key"], str(r["group"])))
    aggregate = {
        variant: _summarize_variant([r for r in results if r["variant"] == variant])
        for variant in VARIANT_SPECS
    }
    payload = {
        **summary,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "aggregate": aggregate,
    }
    _write_csv(out_root / "matrix_results.csv", results)
    _write_json(out_root / "matrix_aggregate.json", payload)
    _write_md(out_root / "matrix_summary.md", payload)

    shutil.rmtree(out_root / "tmp", ignore_errors=True)
    print(f"out_root={out_root}")
    print(f"summary_md={out_root / 'matrix_summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

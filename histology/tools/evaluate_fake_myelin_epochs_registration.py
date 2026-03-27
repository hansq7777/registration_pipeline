from __future__ import annotations

import argparse
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
    _run_logged,
    _stage_command,
    compute_registration_metrics,
    find_ants_bin,
    read_nifti_2d,
    write_nifti_2d,
)


NANOZOOMER_ROOT = Path(r"/mnt/d/Research/Image Analysis/Nanozoomer scans")
CYCLEGAN_ROOT = NANOZOOMER_ROOT / "cyclegan_usable_pairs_20260324"
EPOCH_EXPORT_ROOT = (
    CYCLEGAN_ROOT / "cyclegan_h1_masked_ds1536_crop512_resnet9_2026-03-25_032614_epoch_exports"
)
DEFAULT_EPOCHS = ("epoch25", "epoch30", "epoch35", "epoch40", "epoch45", "epoch50", "epoch55", "epoch60")
TARGET_EVAL_CANVAS = 1536
WORKING_LONG_EDGE = 1024
DEFAULT_WORKERS = min(4, max(1, (os.cpu_count() or 1) // 2))
SHORT_OUT_PREFIX = "fake_my_eval1024"
RAW_NISSL_BASELINE = "raw_nissl"


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _read_rgb(path: Path) -> np.ndarray:
    arr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if arr is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)


def _read_mask(path: Path) -> np.ndarray:
    arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise FileNotFoundError(path)
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr.astype(np.uint8)


def _resize_and_center_to_square(
    rgb: np.ndarray,
    labels: np.ndarray,
    *,
    canvas_size: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    in_h, in_w = rgb.shape[:2]
    long_edge = max(in_h, in_w)
    if long_edge <= 0:
        raise ValueError("Input image has invalid shape.")
    scale = float(canvas_size) / float(long_edge)
    out_w = max(1, int(round(in_w * scale)))
    out_h = max(1, int(round(in_h * scale)))
    rgb_res = cv2.resize(rgb, (out_w, out_h), interpolation=cv2.INTER_AREA)
    labels_res = cv2.resize(labels, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
    x0 = (canvas_size - out_w) // 2
    y0 = (canvas_size - out_h) // 2
    out_rgb = np.full((canvas_size, canvas_size, 3), 255, dtype=np.uint8)
    out_labels = np.zeros((canvas_size, canvas_size), dtype=np.uint8)
    out_rgb[y0 : y0 + out_h, x0 : x0 + out_w] = rgb_res
    out_labels[y0 : y0 + out_h, x0 : x0 + out_w] = labels_res
    return out_rgb, out_labels, {
        "input_shape_hw": [int(in_h), int(in_w)],
        "canvas_shape_hw": [int(canvas_size), int(canvas_size)],
        "scale_to_canvas": float(scale),
        "resized_shape_hw": [int(out_h), int(out_w)],
        "canvas_offset_xy": {"x": int(x0), "y": int(y0)},
    }


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


def _preprocess_clip_gray(rgb: np.ndarray, mask_labels: np.ndarray) -> np.ndarray:
    tissue = mask_labels == 1
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gray = _clip_inside_mask(gray, tissue)
    gray[~tissue] = 255
    return gray


def _resize_gray_and_mask_to_working(
    gray_u8: np.ndarray,
    mask_labels: np.ndarray,
    *,
    working_long_edge: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    in_h, in_w = gray_u8.shape[:2]
    long_edge = max(in_h, in_w)
    if long_edge <= 0:
        raise ValueError("Invalid input shape for working resize.")
    if long_edge <= working_long_edge:
        out_gray = gray_u8.copy()
        out_mask = mask_labels.copy()
        scale = 1.0
    else:
        scale = float(working_long_edge) / float(long_edge)
        out_w = max(1, int(round(in_w * scale)))
        out_h = max(1, int(round(in_h * scale)))
        out_gray = cv2.resize(gray_u8, (out_w, out_h), interpolation=cv2.INTER_AREA)
        out_mask = cv2.resize(mask_labels, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
    tissue = out_mask == 1
    out_gray = out_gray.copy()
    out_gray[~tissue] = 255
    return out_gray, out_mask, {
        "input_shape_hw": [int(in_h), int(in_w)],
        "output_shape_hw": [int(out_gray.shape[0]), int(out_gray.shape[1])],
        "scale_to_working": float(scale),
        "working_long_edge": int(working_long_edge),
    }


def _gray_u8_to_float(gray_u8: np.ndarray) -> np.ndarray:
    return gray_u8.astype(np.float32) / 255.0


def _safe_float(v: Any) -> float:
    try:
        return float(v)
    except Exception:
        return float("nan")


def _aggregate_epoch(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def _vals(stage: str, field: str) -> list[float]:
        return [_safe_float(r[stage][field]) for r in rows]

    def _mean(vals: list[float]) -> float:
        finite = [float(v) for v in vals if np.isfinite(v)]
        return float(np.mean(finite)) if finite else float("nan")

    def _median(vals: list[float]) -> float:
        finite = [float(v) for v in vals if np.isfinite(v)]
        return float(np.median(finite)) if finite else float("nan")

    input_dice = _vals("input", "dice")
    rigid_dice = _vals("rigid", "dice")
    affine_dice = _vals("affine", "dice")
    input_hd95 = _vals("input", "hd95_px")
    rigid_hd95 = _vals("rigid", "hd95_px")
    affine_hd95 = _vals("affine", "hd95_px")

    return {
        "pair_count": len(rows),
        "mean_input_dice": _mean(input_dice),
        "mean_rigid_dice": _mean(rigid_dice),
        "mean_affine_dice": _mean(affine_dice),
        "median_affine_dice": _median(affine_dice),
        "mean_input_hd95_px": _mean(input_hd95),
        "mean_rigid_hd95_px": _mean(rigid_hd95),
        "mean_affine_hd95_px": _mean(affine_hd95),
        "median_affine_hd95_px": _median(affine_hd95),
        "mean_affine_mi": _mean(_vals("affine", "mi")),
        "mean_affine_cc": _mean(_vals("affine", "cc")),
        "mean_rigid_time_s": _mean([_safe_float(r["timing_seconds"]["rigid"]) for r in rows]),
        "mean_affine_time_s": _mean([_safe_float(r["timing_seconds"]["affine"]) for r in rows]),
        "affine_dice_gt_input_count": int(sum(_safe_float(a) > _safe_float(i) for a, i in zip(affine_dice, input_dice))),
        "affine_hd95_lt_input_count": int(sum(_safe_float(a) < _safe_float(i) for a, i in zip(affine_hd95, input_hd95))),
        "affine_dice_gt_rigid_count": int(sum(_safe_float(a) > _safe_float(r) for a, r in zip(affine_dice, rigid_dice))),
        "affine_hd95_lt_rigid_count": int(sum(_safe_float(a) < _safe_float(r) for a, r in zip(affine_hd95, rigid_hd95))),
    }


@dataclass(frozen=True)
class PairGeometry:
    pair_key: str
    stem: str
    myelin_image_path: Path
    nissl_image_path: Path
    myelin_mask_path: Path
    nissl_mask_path: Path
    fixed_gray_1024_path: Path
    fixed_mask_1024_path: Path
    moving_mask_1536_path: Path
    moving_mask_1024_path: Path
    geometry_manifest_path: Path


def _build_geometry_for_row(row: dict[str, str], out_root: Path) -> PairGeometry:
    stem = Path(str(row["myelin_image"])).stem.removesuffix("_myelin")
    pair_key = str(row["pair_key"])
    myelin_image_path = CYCLEGAN_ROOT / str(row["myelin_image"])
    nissl_image_path = CYCLEGAN_ROOT / str(row["nissl_image"])
    myelin_mask_path = CYCLEGAN_ROOT / str(row["myelin_mask"])
    nissl_mask_path = CYCLEGAN_ROOT / str(row["nissl_mask"])

    geometry_dir = out_root / "geom"
    geometry_dir.mkdir(parents=True, exist_ok=True)
    fixed_gray_1024_path = geometry_dir / f"{stem}_fixed_gray_clip_1024.png"
    fixed_mask_1024_path = geometry_dir / f"{stem}_fixed_mask_tissue_1024.png"
    moving_mask_1536_path = geometry_dir / f"{stem}_moving_mask_tissue_1536.png"
    moving_mask_1024_path = geometry_dir / f"{stem}_moving_mask_tissue_1024.png"
    geometry_manifest_path = geometry_dir / f"{stem}_geometry.json"

    if (
        fixed_gray_1024_path.exists()
        and fixed_mask_1024_path.exists()
        and moving_mask_1536_path.exists()
        and moving_mask_1024_path.exists()
        and geometry_manifest_path.exists()
    ):
        return PairGeometry(
            pair_key=pair_key,
            stem=stem,
            myelin_image_path=myelin_image_path,
            nissl_image_path=nissl_image_path,
            myelin_mask_path=myelin_mask_path,
            nissl_mask_path=nissl_mask_path,
            fixed_gray_1024_path=fixed_gray_1024_path,
            fixed_mask_1024_path=fixed_mask_1024_path,
            moving_mask_1536_path=moving_mask_1536_path,
            moving_mask_1024_path=moving_mask_1024_path,
            geometry_manifest_path=geometry_manifest_path,
        )

    fixed_rgb = _read_rgb(myelin_image_path)
    fixed_labels = _read_mask(myelin_mask_path)
    nissl_rgb = _read_rgb(nissl_image_path)
    moving_labels_src = _read_mask(nissl_mask_path)

    fixed_rgb_1536, fixed_labels_1536, fixed_canvas_info = _resize_and_center_to_square(
        fixed_rgb,
        fixed_labels,
        canvas_size=TARGET_EVAL_CANVAS,
    )
    _moving_rgb_1536, moving_labels_1536, moving_canvas_info = _resize_and_center_to_square(
        nissl_rgb,
        moving_labels_src,
        canvas_size=TARGET_EVAL_CANVAS,
    )
    fixed_gray_1024, fixed_labels_1024, fixed_work_info = _resize_gray_and_mask_to_working(
        _preprocess_clip_gray(fixed_rgb_1536, fixed_labels_1536),
        fixed_labels_1536,
        working_long_edge=WORKING_LONG_EDGE,
    )
    _moving_gray_placeholder_1024, moving_labels_1024, moving_work_info = _resize_gray_and_mask_to_working(
        _preprocess_clip_gray(_moving_rgb_1536, moving_labels_1536),
        moving_labels_1536,
        working_long_edge=WORKING_LONG_EDGE,
    )

    cv2.imwrite(str(fixed_gray_1024_path), fixed_gray_1024)
    cv2.imwrite(str(fixed_mask_1024_path), np.where(fixed_labels_1024 == 1, 255, 0).astype(np.uint8))
    cv2.imwrite(str(moving_mask_1536_path), np.where(moving_labels_1536 == 1, 255, 0).astype(np.uint8))
    cv2.imwrite(str(moving_mask_1024_path), np.where(moving_labels_1024 == 1, 255, 0).astype(np.uint8))

    _write_json(
        geometry_manifest_path,
        {
            "pair_key": pair_key,
            "stem": stem,
            "myelin_image": str(myelin_image_path),
            "nissl_image": str(nissl_image_path),
            "myelin_mask": str(myelin_mask_path),
            "nissl_mask": str(nissl_mask_path),
            "evaluation_canvas": {
                "canvas_size": TARGET_EVAL_CANVAS,
                "working_long_edge": WORKING_LONG_EDGE,
                "mask_mode": "tissue_only",
                "preprocessing": "percentile_clipping_plus_fixed_background",
            },
            "fixed_myelin_canvas_1536": fixed_canvas_info,
            "moving_nissl_canvas_1536": moving_canvas_info,
            "fixed_myelin_working_1024": fixed_work_info,
            "moving_nissl_working_1024": moving_work_info,
            "cache_files": {
                "fixed_gray_1024": str(fixed_gray_1024_path),
                "fixed_mask_1024": str(fixed_mask_1024_path),
                "moving_mask_1536": str(moving_mask_1536_path),
                "moving_mask_1024": str(moving_mask_1024_path),
            },
        },
    )

    return PairGeometry(
        pair_key=pair_key,
        stem=stem,
        myelin_image_path=myelin_image_path,
        nissl_image_path=nissl_image_path,
        myelin_mask_path=myelin_mask_path,
        nissl_mask_path=nissl_mask_path,
        fixed_gray_1024_path=fixed_gray_1024_path,
        fixed_mask_1024_path=fixed_mask_1024_path,
        moving_mask_1536_path=moving_mask_1536_path,
        moving_mask_1024_path=moving_mask_1024_path,
        geometry_manifest_path=geometry_manifest_path,
    )


def _load_binary_mask(path: Path) -> np.ndarray:
    arr = _read_mask(path)
    return np.where(arr > 0, 1.0, 0.0).astype(np.float32)


def _moving_gray_for_source(epoch_name: str, geometry: PairGeometry, moving_mask_1536: np.ndarray, moving_mask_1024: np.ndarray) -> tuple[np.ndarray, str]:
    if epoch_name == RAW_NISSL_BASELINE:
        nissl_rgb = _read_rgb(geometry.nissl_image_path)
        nissl_labels = _read_mask(geometry.nissl_mask_path)
        moving_rgb_1536, moving_labels_1536, _ = _resize_and_center_to_square(
            nissl_rgb,
            nissl_labels,
            canvas_size=TARGET_EVAL_CANVAS,
        )
        moving_gray_1536_u8 = _preprocess_clip_gray(moving_rgb_1536, moving_labels_1536)
        source_path = str(geometry.nissl_image_path)
    else:
        fake_path = EPOCH_EXPORT_ROOT / epoch_name / "fake_myelin_pngs" / f"{geometry.stem}_fake_myelin.png"
        if not fake_path.exists():
            raise FileNotFoundError(fake_path)
        fake_rgb = _read_rgb(fake_path)
        if fake_rgb.shape[:2] != (TARGET_EVAL_CANVAS, TARGET_EVAL_CANVAS):
            raise ValueError(f"Unexpected fake image shape for {fake_path.name}: {fake_rgb.shape[:2]}")
        moving_gray_1536_u8 = _preprocess_clip_gray(fake_rgb, np.where(moving_mask_1536 > 0, 1, 0).astype(np.uint8))
        source_path = str(fake_path)

    moving_gray_1024_u8 = cv2.resize(
        moving_gray_1536_u8,
        (int(moving_mask_1024.shape[1]), int(moving_mask_1024.shape[0])),
        interpolation=cv2.INTER_AREA,
    )
    moving_gray_1024_u8[moving_mask_1024 <= 0] = 255
    return _gray_u8_to_float(moving_gray_1024_u8), source_path


def _run_one_epoch_pair(
    epoch_name: str,
    geometry: PairGeometry,
    ants_bin: Path,
    tmp_root: Path,
) -> dict[str, Any]:
    fixed_gray_u8 = _read_mask(geometry.fixed_gray_1024_path)
    fixed_gray = _gray_u8_to_float(fixed_gray_u8)
    fixed_mask = _load_binary_mask(geometry.fixed_mask_1024_path)
    moving_mask_1536 = _load_binary_mask(geometry.moving_mask_1536_path)
    moving_mask_1024 = _load_binary_mask(geometry.moving_mask_1024_path)

    moving_gray, source_image_path = _moving_gray_for_source(epoch_name, geometry, moving_mask_1536, moving_mask_1024)

    input_metrics, input_metric_timing = compute_registration_metrics(fixed_gray, moving_gray, fixed_mask, moving_mask_1024)

    short_epoch = epoch_name.replace("epoch", "e")
    short_pair = geometry.stem.split("_", 1)[0]
    work_dir = Path(tempfile.mkdtemp(prefix=f"{short_epoch}_{short_pair}_", dir=str(tmp_root)))
    inputs_dir = work_dir / "i"
    rigid_dir = work_dir / "r"
    affine_dir = work_dir / "a"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    rigid_dir.mkdir(parents=True, exist_ok=True)
    affine_dir.mkdir(parents=True, exist_ok=True)

    fixed_img_path = inputs_dir / "f.nii.gz"
    moving_img_path = inputs_dir / "m.nii.gz"
    fixed_mask_path = inputs_dir / "fm.nii.gz"
    moving_mask_path = inputs_dir / "mm.nii.gz"
    write_nifti_2d(fixed_img_path, fixed_gray)
    write_nifti_2d(moving_img_path, moving_gray)
    write_nifti_2d(fixed_mask_path, fixed_mask)
    write_nifti_2d(moving_mask_path, moving_mask_1024)

    result: dict[str, Any] = {
        "pair_key": geometry.pair_key,
        "stem": geometry.stem,
        "epoch": epoch_name,
        "source_image_path": source_image_path,
        "geometry_manifest_path": str(geometry.geometry_manifest_path),
        "config": {
            "evaluation_canvas": TARGET_EVAL_CANVAS,
            "working_long_edge": WORKING_LONG_EDGE,
            "mask_mode": "tissue_only",
            "preprocessing": "percentile_clipping_plus_fixed_background",
            "coarse_registration": "rigid_plus_affine",
            "affine_profile": "current",
        },
        "input": input_metrics,
        "input_metric_timing_seconds": input_metric_timing,
        "timing_seconds": {},
    }

    try:
        rigid_prefix = rigid_dir / "r_"
        rigid_mat = rigid_dir / "r_0GenericAffine.mat"
        t0 = time.perf_counter()
        ants_t0 = time.perf_counter()
        _run_logged(
            _stage_command(
                ants_bin,
                "rigid",
                fixed_img_path,
                moving_img_path,
                fixed_mask_path,
                moving_mask_path,
                rigid_prefix,
                [],
                "current",
            ),
            rigid_dir / "r.log",
        )
        rigid_ants = float(time.perf_counter() - ants_t0)
        rigid_warped_mask_path = rigid_dir / "r_mask.nii.gz"
        _ants_apply(
            ants_bin,
            moving_mask_path,
            fixed_img_path,
            rigid_warped_mask_path,
            [rigid_mat],
            interpolation="NearestNeighbor",
            log_path=rigid_dir / "r_mask.log",
        )
        rigid_gray = read_nifti_2d(rigid_dir / "r_Warped.nii.gz")
        rigid_mask = read_nifti_2d(rigid_warped_mask_path)
        rigid_metrics, rigid_metric_timing = compute_registration_metrics(
            fixed_gray,
            np.clip(rigid_gray, 0.0, 1.0),
            fixed_mask,
            (rigid_mask > 0.5).astype(np.float32),
        )
        rigid_total = float(time.perf_counter() - t0)
        result["rigid"] = rigid_metrics
        result["rigid_metric_timing_seconds"] = rigid_metric_timing
        result["timing_seconds"]["rigid"] = rigid_total
        result["timing_seconds"]["rigid_ants_registration"] = rigid_ants

        affine_prefix = affine_dir / "a_"
        affine_mat = affine_dir / "a_0GenericAffine.mat"
        t0 = time.perf_counter()
        ants_t0 = time.perf_counter()
        _run_logged(
            _stage_command(
                ants_bin,
                "affine",
                fixed_img_path,
                moving_img_path,
                fixed_mask_path,
                moving_mask_path,
                affine_prefix,
                [rigid_mat],
                "current",
            ),
            affine_dir / "a.log",
        )
        affine_ants = float(time.perf_counter() - ants_t0)
        affine_warped_mask_path = affine_dir / "a_mask.nii.gz"
        _ants_apply(
            ants_bin,
            moving_mask_path,
            fixed_img_path,
            affine_warped_mask_path,
            [affine_mat, rigid_mat],
            interpolation="NearestNeighbor",
            log_path=affine_dir / "a_mask.log",
        )
        affine_gray = read_nifti_2d(affine_dir / "a_Warped.nii.gz")
        affine_mask = read_nifti_2d(affine_warped_mask_path)
        affine_metrics, affine_metric_timing = compute_registration_metrics(
            fixed_gray,
            np.clip(affine_gray, 0.0, 1.0),
            fixed_mask,
            (affine_mask > 0.5).astype(np.float32),
        )
        affine_total = float(time.perf_counter() - t0)
        result["affine"] = affine_metrics
        result["affine_metric_timing_seconds"] = affine_metric_timing
        result["timing_seconds"]["affine"] = affine_total
        result["timing_seconds"]["affine_ants_registration"] = affine_ants
        result["timing_seconds"]["total"] = float(rigid_total + affine_total)
        result["status"] = "ok"
    except Exception as exc:
        result["status"] = "failed"
        result["error"] = str(exc)
        result["failed_work_dir"] = str(work_dir)
        return result
    finally:
        if result.get("status") == "ok":
            shutil.rmtree(work_dir, ignore_errors=True)
    return result


def _flatten_result_for_csv(result: dict[str, Any]) -> dict[str, Any]:
    row = {
        "pair_key": result["pair_key"],
        "stem": result["stem"],
        "epoch": result["epoch"],
        "status": result.get("status", "ok"),
        "source_image_path": result.get("source_image_path", ""),
        "geometry_manifest_path": result.get("geometry_manifest_path", ""),
        "rigid_time_s": result.get("timing_seconds", {}).get("rigid", float("nan")),
        "affine_time_s": result.get("timing_seconds", {}).get("affine", float("nan")),
        "rigid_ants_time_s": result.get("timing_seconds", {}).get("rigid_ants_registration", float("nan")),
        "affine_ants_time_s": result.get("timing_seconds", {}).get("affine_ants_registration", float("nan")),
        "input_mi": result.get("input", {}).get("mi", float("nan")),
        "input_cc": result.get("input", {}).get("cc", float("nan")),
        "input_dice": result.get("input", {}).get("dice", float("nan")),
        "input_hd95_px": result.get("input", {}).get("hd95_px", float("nan")),
        "rigid_mi": result.get("rigid", {}).get("mi", float("nan")),
        "rigid_cc": result.get("rigid", {}).get("cc", float("nan")),
        "rigid_dice": result.get("rigid", {}).get("dice", float("nan")),
        "rigid_hd95_px": result.get("rigid", {}).get("hd95_px", float("nan")),
        "affine_mi": result.get("affine", {}).get("mi", float("nan")),
        "affine_cc": result.get("affine", {}).get("cc", float("nan")),
        "affine_dice": result.get("affine", {}).get("dice", float("nan")),
        "affine_hd95_px": result.get("affine", {}).get("hd95_px", float("nan")),
    }
    if result.get("status") != "ok":
        row["error"] = result.get("error", "")
        row["failed_work_dir"] = result.get("failed_work_dir", "")
    return row


def _write_summary_md(path: Path, aggregate: dict[str, Any], config: dict[str, Any], pair_count: int) -> None:
    ranking = sorted(
        aggregate.items(),
        key=lambda kv: (
            -_safe_float(kv[1].get("mean_affine_dice")),
            _safe_float(kv[1].get("mean_affine_hd95_px")),
        ),
    )
    lines = [
        "# Fake Myelin Epoch Registration Evaluation",
        "",
        f"- evaluated_pairs: `{pair_count}`",
        f"- epochs: `{', '.join(sorted(aggregate.keys()))}`",
        f"- evaluation_canvas: `{config['evaluation_canvas']}`",
        f"- working_long_edge: `{config['working_long_edge']}`",
        f"- mask_mode: `{config['mask_mode']}`",
        f"- preprocessing: `{config['preprocessing']}`",
        f"- coarse_registration: `{config['coarse_registration']}`",
        f"- affine_profile: `{config['affine_profile']}`",
        "",
        "## Epoch Ranking",
        "",
    ]
    for rank, (epoch_name, stats) in enumerate(ranking, start=1):
        lines.extend(
            [
                f"### {rank}. {epoch_name}",
                f"- mean_affine_dice: `{stats['mean_affine_dice']:.6f}`",
                f"- mean_affine_hd95_px: `{stats['mean_affine_hd95_px']:.3f}`",
                f"- mean_affine_mi: `{stats['mean_affine_mi']:.6f}`",
                f"- mean_affine_cc: `{stats['mean_affine_cc']:.6f}`",
                f"- mean_rigid_time_s: `{stats['mean_rigid_time_s']:.3f}`",
                f"- mean_affine_time_s: `{stats['mean_affine_time_s']:.3f}`",
                f"- affine_dice_gt_input_count: `{stats['affine_dice_gt_input_count']}`",
                f"- affine_hd95_lt_input_count: `{stats['affine_hd95_lt_input_count']}`",
                "",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", nargs="*", default=list(DEFAULT_EPOCHS))
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--limit-pairs", type=int, default=0)
    parser.add_argument("--out-root", default="")
    args = parser.parse_args()

    os.environ.setdefault("ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")

    ants_bin = find_ants_bin()
    if ants_bin is None:
        raise RuntimeError("ANTs not found.")

    selected_epochs = [str(x).strip() for x in args.epochs if str(x).strip()]
    if not selected_epochs:
        raise RuntimeError("No epochs selected.")
    for epoch in selected_epochs:
        if epoch == RAW_NISSL_BASELINE:
            continue
        fake_dir = EPOCH_EXPORT_ROOT / epoch / "fake_myelin_pngs"
        if not fake_dir.exists():
            raise FileNotFoundError(fake_dir)

    out_root = Path(args.out_root) if str(args.out_root).strip() else NANOZOOMER_ROOT / f"{SHORT_OUT_PREFIX}_{_utc_stamp()}"
    out_root.mkdir(parents=True, exist_ok=True)
    tmp_root = out_root / "_t"
    tmp_root.mkdir(parents=True, exist_ok=True)

    manifest_path = CYCLEGAN_ROOT / "manifest.csv"
    with manifest_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if int(args.limit_pairs) > 0:
        rows = rows[: int(args.limit_pairs)]

    geometry_items: list[PairGeometry] = []
    geom_workers = max(1, min(8, int(args.workers)))
    with cf.ThreadPoolExecutor(max_workers=geom_workers) as ex:
        future_map = {ex.submit(_build_geometry_for_row, row, out_root): row for row in rows}
        total_geom = len(future_map)
        for idx, fut in enumerate(cf.as_completed(future_map), start=1):
            geometry_items.append(fut.result())
            if idx % 10 == 0 or idx == total_geom:
                print(f"[geom {idx}/{total_geom}]", flush=True)
    geometry_items.sort(key=lambda g: g.stem)
    geometry_csv_path = out_root / "evaluation_geometry_manifest.csv"
    with geometry_csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "pair_key",
                "stem",
                "myelin_image_path",
                "nissl_image_path",
                "myelin_mask_path",
                "nissl_mask_path",
                "fixed_gray_1024_path",
                "fixed_mask_1024_path",
                "moving_mask_1536_path",
                "moving_mask_1024_path",
                "geometry_manifest_path",
            ],
        )
        writer.writeheader()
        for g in geometry_items:
            writer.writerow(
                {
                    "pair_key": g.pair_key,
                    "stem": g.stem,
                    "myelin_image_path": str(g.myelin_image_path),
                    "nissl_image_path": str(g.nissl_image_path),
                    "myelin_mask_path": str(g.myelin_mask_path),
                    "nissl_mask_path": str(g.nissl_mask_path),
                    "fixed_gray_1024_path": str(g.fixed_gray_1024_path),
                    "fixed_mask_1024_path": str(g.fixed_mask_1024_path),
                    "moving_mask_1536_path": str(g.moving_mask_1536_path),
                    "moving_mask_1024_path": str(g.moving_mask_1024_path),
                    "geometry_manifest_path": str(g.geometry_manifest_path),
                }
            )

    config = {
        "evaluation_canvas": TARGET_EVAL_CANVAS,
        "working_long_edge": WORKING_LONG_EDGE,
        "mask_mode": "tissue_only",
        "preprocessing": "percentile_clipping_plus_fixed_background",
        "coarse_registration": "rigid_plus_affine",
        "affine_profile": "current",
        "epochs": selected_epochs,
        "workers": int(args.workers),
        "pair_count": len(geometry_items),
        "geometry_manifest_csv": str(geometry_csv_path),
    }
    _write_json(out_root / "experiment_config.json", config)

    tasks = [(epoch, geometry) for epoch in selected_epochs for geometry in geometry_items]
    results_jsonl = out_root / "epoch_results.jsonl"
    results: list[dict[str, Any]] = []
    with results_jsonl.open("w", encoding="utf-8") as jsonl, cf.ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as ex:
        future_map = {
            ex.submit(_run_one_epoch_pair, epoch, geometry, ants_bin, tmp_root): (epoch, geometry.pair_key)
            for epoch, geometry in tasks
        }
        total = len(future_map)
        for idx, fut in enumerate(cf.as_completed(future_map), start=1):
            epoch, pair_key = future_map[fut]
            result = fut.result()
            results.append(result)
            jsonl.write(json.dumps(result) + "\n")
            jsonl.flush()
            status = result.get("status", "ok")
            aff = result.get("affine", {})
            print(
                f"[{idx}/{total}] {epoch} {pair_key} {status} "
                f"dice={_safe_float(aff.get('dice')):.4f} hd95={_safe_float(aff.get('hd95_px')):.2f}",
                flush=True,
            )

    flat_rows = [_flatten_result_for_csv(r) for r in results]
    results_csv_path = out_root / "epoch_results.csv"
    with results_csv_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = list(flat_rows[0].keys()) if flat_rows else []
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(flat_rows)

    aggregate = {
        epoch: _aggregate_epoch([r for r in results if r.get("epoch") == epoch and r.get("status") == "ok"])
        for epoch in selected_epochs
    }
    _write_json(
        out_root / "epoch_aggregate.json",
        {
            "config": config,
            "aggregate": aggregate,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "result_count": len(results),
            "failure_count": int(sum(r.get("status") != "ok" for r in results)),
        },
    )
    _write_summary_md(out_root / "epoch_aggregate.md", aggregate, config, len(geometry_items))
    print(f"out_root={out_root}")
    print(f"results_csv={results_csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

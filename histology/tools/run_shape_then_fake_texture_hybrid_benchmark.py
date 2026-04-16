from __future__ import annotations

import argparse
import concurrent.futures as cf
import csv
import glob
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

import numpy as np

REPO_HISTOLOGY_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_HISTOLOGY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_HISTOLOGY_ROOT))

from gui_mvp.hitl_gui.application.pair_registration import (  # noqa: E402
    _ants_apply,
    _run_logged,
    _stage_command,
    _warp_gray_affine,
    _warp_mask_affine,
    build_affine_matrix,
    compute_registration_metrics,
    find_ants_bin,
    optimize_mask_shape_transform,
    read_nifti_2d,
    strict_pareto_gate_decision,
    write_nifti_2d,
)
from evaluate_fake_myelin_epochs_registration import (  # noqa: E402
    CYCLEGAN_ROOT,
    RAW_NISSL_BASELINE,
    PairGeometry,
    TARGET_EVAL_CANVAS,
    WORKING_LONG_EDGE,
    _build_geometry_for_row,
    _gray_u8_to_float,
    _load_binary_mask,
    _moving_gray_for_source,
    _read_mask,
)


NANOZOOMER_ROOT = Path(r"/mnt/d/Research/Image Analysis/Nanozoomer scans")
DEFAULT_EPOCH = "epoch30"
DEFAULT_TARGET_UM_PER_PX = 10.0
DEFAULT_WORKING_LONG_EDGE = 1024
DEFAULT_WORKERS = min(4, max(1, (os.cpu_count() or 1) // 2))
SHORT_OUT_PREFIX = "histology_shape_fake_hybrid"


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def _round_float(value: float, digits: int = 6) -> float:
    return round(float(value), digits)


def _latest_fake_eval_root() -> Path | None:
    matches = sorted(glob.glob("/mnt/d/Research/Image Analysis/Nanozoomer scans/fake_my_eval1024_*/geom"))
    if not matches:
        return None
    return Path(matches[-1]).parent


def _load_manifest_rows() -> list[dict[str, str]]:
    manifest_csv = CYCLEGAN_ROOT / "manifest.csv"
    with manifest_csv.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _geometry_from_manifest(manifest_path: Path) -> PairGeometry:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    cache = dict(payload.get("cache_files") or {})
    return PairGeometry(
        pair_key=str(payload["pair_key"]),
        stem=str(payload["stem"]),
        myelin_image_path=Path(str(payload["myelin_image"])),
        nissl_image_path=Path(str(payload["nissl_image"])),
        myelin_mask_path=Path(str(payload["myelin_mask"])),
        nissl_mask_path=Path(str(payload["nissl_mask"])),
        fixed_gray_1024_path=Path(str(cache["fixed_gray_1024"])),
        fixed_mask_1024_path=Path(str(cache["fixed_mask_1024"])),
        moving_mask_1536_path=Path(str(cache["moving_mask_1536"])),
        moving_mask_1024_path=Path(str(cache["moving_mask_1024"])),
        geometry_manifest_path=manifest_path,
    )


def _load_or_build_geometry(row: dict[str, str], out_root: Path) -> PairGeometry:
    stem = Path(str(row["myelin_image"])).stem.removesuffix("_myelin")
    latest_root = _latest_fake_eval_root()
    if latest_root is not None:
        manifest_path = latest_root / "geom" / f"{stem}_geometry.json"
        if manifest_path.exists():
            return _geometry_from_manifest(manifest_path)
    return _build_geometry_for_row(row, out_root)


def _metric_delta(candidate: dict[str, Any], reference: dict[str, Any]) -> dict[str, float]:
    return {
        "dice": _safe_float(candidate.get("dice")) - _safe_float(reference.get("dice")),
        "hd95_px": _safe_float(candidate.get("hd95_px")) - _safe_float(reference.get("hd95_px")),
    }


def _run_rigid_candidate(
    *,
    fixed_gray: np.ndarray,
    moving_gray: np.ndarray,
    fixed_mask: np.ndarray,
    moving_mask: np.ndarray,
    ants_bin: Path,
    tmp_root: Path,
    tag: str,
) -> dict[str, Any]:
    work_dir = Path(tempfile.mkdtemp(prefix=f"{tag}_", dir=str(tmp_root)))
    inputs_dir = work_dir / "i"
    rigid_dir = work_dir / "r"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    rigid_dir.mkdir(parents=True, exist_ok=True)

    fixed_img_path = inputs_dir / "f.nii.gz"
    moving_img_path = inputs_dir / "m.nii.gz"
    fixed_mask_path = inputs_dir / "fm.nii.gz"
    moving_mask_path = inputs_dir / "mm.nii.gz"
    write_nifti_2d(fixed_img_path, fixed_gray)
    write_nifti_2d(moving_img_path, moving_gray)
    write_nifti_2d(fixed_mask_path, fixed_mask)
    write_nifti_2d(moving_mask_path, moving_mask)

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
    ants_seconds = float(time.perf_counter() - ants_t0)

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
    metrics, metric_timing = compute_registration_metrics(
        fixed_gray,
        np.clip(rigid_gray, 0.0, 1.0),
        fixed_mask,
        (rigid_mask > 0.5).astype(np.float32),
    )
    total_seconds = float(time.perf_counter() - t0)
    shutil.rmtree(work_dir, ignore_errors=True)
    return {
        "metrics": metrics,
        "metric_timing_seconds": metric_timing,
        "timing_seconds": {
            "stage_total": total_seconds,
            "ants_registration": ants_seconds,
        },
    }


def _summarize(values: list[float]) -> dict[str, Any]:
    finite = [float(v) for v in values if np.isfinite(v)]
    if not finite:
        return {"count": 0}
    arr = np.asarray(finite, dtype=np.float64)
    return {
        "count": int(arr.size),
        "mean": _round_float(float(arr.mean())),
        "median": _round_float(float(np.median(arr))),
        "min": _round_float(float(arr.min())),
        "max": _round_float(float(arr.max())),
        "p05": _round_float(float(np.percentile(arr, 5))),
        "p95": _round_float(float(np.percentile(arr, 95))),
    }


def _collect_final_metrics(rows: list[dict[str, Any]], arm: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        final = dict((row.get("arms") or {}).get(arm) or {})
        metrics = dict(final.get("final_metrics") or {})
        if metrics:
            out.append({"best_stage": final.get("best_stage"), "metrics": metrics, "arm": arm})
    return out


def _arm_summary(rows: list[dict[str, Any]], arm: str, reference_arm: str | None = None) -> dict[str, Any]:
    finals = _collect_final_metrics(rows, arm)
    if not finals:
        return {"arm": arm, "count": 0}
    dice = [_safe_float(r["metrics"].get("dice")) for r in finals]
    hd95 = [_safe_float(r["metrics"].get("hd95_px")) for r in finals]
    mi = [_safe_float(r["metrics"].get("mi")) for r in finals]
    cc = [_safe_float(r["metrics"].get("cc")) for r in finals]
    best_stage_hist: dict[str, int] = {}
    success_count = 0
    texture_accept_count = 0
    for row in rows:
        arm_rec = dict((row.get("arms") or {}).get(arm) or {})
        best_stage = str(arm_rec.get("best_stage") or "input")
        best_stage_hist[best_stage] = best_stage_hist.get(best_stage, 0) + 1
        if best_stage != "input":
            success_count += 1
        if str(arm_rec.get("texture_stage_gate", {}).get("accepted")).lower() == "true":
            texture_accept_count += 1

    summary = {
        "arm": arm,
        "count": len(finals),
        "success_rate": _round_float(float(success_count) / float(len(finals))),
        "texture_accept_count": int(texture_accept_count),
        "best_stage_histogram": dict(sorted(best_stage_hist.items())),
        "final_dice": _summarize(dice),
        "final_hd95_px": _summarize(hd95),
        "final_mi": _summarize(mi),
        "final_cc": _summarize(cc),
    }
    if reference_arm:
        d_dice: list[float] = []
        d_hd95: list[float] = []
        better_dice = 0
        better_hd95 = 0
        ref_tail: list[float] = []
        for row in rows:
            arm_rec = dict((row.get("arms") or {}).get(arm) or {})
            ref_rec = dict((row.get("arms") or {}).get(reference_arm) or {})
            arm_metrics = dict(arm_rec.get("final_metrics") or {})
            ref_metrics = dict(ref_rec.get("final_metrics") or {})
            if arm_metrics and ref_metrics:
                dd = _safe_float(arm_metrics.get("dice")) - _safe_float(ref_metrics.get("dice"))
                dh = _safe_float(arm_metrics.get("hd95_px")) - _safe_float(ref_metrics.get("hd95_px"))
                if np.isfinite(dd):
                    d_dice.append(dd)
                    if dd > 0:
                        better_dice += 1
                if np.isfinite(dh):
                    d_hd95.append(dh)
                    if dh < 0:
                        better_hd95 += 1
                    if dh > 0:
                        ref_tail.append(dh)
        summary["delta_vs_" + reference_arm] = {
            "dice": _summarize(d_dice),
            "hd95_px": _summarize(d_hd95),
            "better_dice_count": int(better_dice),
            "better_hd95_count": int(better_hd95),
            "hd95_regression_tail": _summarize(ref_tail),
        }
    return summary


def _markdown_summary(config: dict[str, Any], summary: dict[str, Any]) -> str:
    lines = [
        "# Shape-Then-Fake-Texture Hybrid Benchmark",
        "",
        "## Config",
        "",
        f"- benchmark_pairs: `{config['pair_count']}`",
        f"- epoch: `{config['epoch']}`",
        f"- target_um_per_px: `{config['target_um_per_px']}`",
        f"- working_long_edge: `{config['working_long_edge']}`",
        f"- mask_mode: `{config['mask_mode']}`",
        f"- preprocessing: `{config['preprocessing']}`",
        "",
        "## Arms",
        "",
        "- `input_only`",
        "- `mi_rigid`",
        "- `mask_rigid`",
        "- `shape_then_raw_texture_rigid`",
        "- `shape_then_fake_texture_rigid`",
        "",
        "## Aggregate",
        "",
    ]
    for arm in (
        "input_only",
        "mi_rigid",
        "mask_rigid",
        "shape_then_raw_texture_rigid",
        "shape_then_fake_texture_rigid",
    ):
        stats = summary.get(arm) or {}
        lines.extend(
            [
                f"### {arm}",
                "",
                f"- success_rate: `{stats.get('success_rate')}`",
                f"- texture_accept_count: `{stats.get('texture_accept_count', 0)}`",
                f"- final_dice mean/median: `{(stats.get('final_dice') or {}).get('mean')}` / `{(stats.get('final_dice') or {}).get('median')}`",
                f"- final_hd95 mean/median: `{(stats.get('final_hd95_px') or {}).get('mean')}` / `{(stats.get('final_hd95_px') or {}).get('median')}`",
            ]
        )
        ref = stats.get("delta_vs_mask_rigid")
        if isinstance(ref, dict):
            lines.extend(
                [
                    f"- delta_vs_mask_rigid Dice mean/median: `{(ref.get('dice') or {}).get('mean')}` / `{(ref.get('dice') or {}).get('median')}`",
                    f"- delta_vs_mask_rigid HD95 mean/median: `{(ref.get('hd95_px') or {}).get('mean')}` / `{(ref.get('hd95_px') or {}).get('median')}`",
                    f"- better_than_mask_rigid Dice count: `{ref.get('better_dice_count')}`",
                    f"- better_than_mask_rigid HD95 count: `{ref.get('better_hd95_count')}`",
                ]
            )
        lines.append("")
    return "\n".join(lines)


def _run_pair_task(task: tuple[dict[str, str], str, str]) -> dict[str, Any]:
    row, out_root_s, epoch_name = task
    out_root = Path(out_root_s)
    geometry = _load_or_build_geometry(row, out_root)
    ants_bin = find_ants_bin()
    if ants_bin is None:
        raise RuntimeError("Could not locate local ANTs binaries.")
    tmp_root = out_root / "tmp"
    tmp_root.mkdir(parents=True, exist_ok=True)

    fixed_gray_u8 = _read_mask(geometry.fixed_gray_1024_path)
    fixed_gray = _gray_u8_to_float(fixed_gray_u8)
    fixed_mask = _load_binary_mask(geometry.fixed_mask_1024_path)
    moving_mask_1536 = _load_binary_mask(geometry.moving_mask_1536_path)
    moving_mask_1024 = _load_binary_mask(geometry.moving_mask_1024_path)
    raw_gray, raw_source = _moving_gray_for_source(RAW_NISSL_BASELINE, geometry, moving_mask_1536, moving_mask_1024)
    fake_gray, fake_source = _moving_gray_for_source(epoch_name, geometry, moving_mask_1536, moving_mask_1024)

    input_metrics, input_metric_timing = compute_registration_metrics(
        fixed_gray,
        raw_gray,
        fixed_mask,
        moving_mask_1024,
    )

    mi_rigid = _run_rigid_candidate(
        fixed_gray=fixed_gray,
        moving_gray=raw_gray,
        fixed_mask=fixed_mask,
        moving_mask=moving_mask_1024,
        ants_bin=ants_bin,
        tmp_root=tmp_root,
        tag=f"mi_{geometry.stem}",
    )
    mi_gate = strict_pareto_gate_decision(input_metrics, dict(mi_rigid["metrics"]))
    mi_best_stage = "rigid" if bool(mi_gate.get("accepted")) else "input"
    mi_final_metrics = dict(mi_rigid["metrics"]) if mi_best_stage == "rigid" else dict(input_metrics)

    shape_opt = optimize_mask_shape_transform(
        fixed_mask,
        moving_mask_1024,
        model="rigid",
    )
    shape_mat = np.asarray(shape_opt["matrix_2x3"], dtype=np.float32)
    shape_mask = _warp_mask_affine(moving_mask_1024, shape_mat, fixed_gray.shape[:2]).astype(np.float32)
    shape_gray_raw = np.clip(_warp_gray_affine(raw_gray, shape_mat, fixed_gray.shape[:2]), 0.0, 1.0)
    shape_metrics, shape_metric_timing = compute_registration_metrics(
        fixed_gray,
        shape_gray_raw,
        fixed_mask,
        shape_mask,
    )
    shape_gate = strict_pareto_gate_decision(input_metrics, shape_metrics)
    shape_best_stage = "mask_rigid" if bool(shape_gate.get("accepted")) else "input"
    best_before_texture_metrics = dict(shape_metrics) if shape_best_stage == "mask_rigid" else dict(input_metrics)

    best_raw_gray = shape_gray_raw if shape_best_stage == "mask_rigid" else raw_gray
    best_fake_gray = np.clip(_warp_gray_affine(fake_gray, shape_mat, fixed_gray.shape[:2]), 0.0, 1.0) if shape_best_stage == "mask_rigid" else fake_gray
    best_mask = shape_mask if shape_best_stage == "mask_rigid" else moving_mask_1024

    raw_local = _run_rigid_candidate(
        fixed_gray=fixed_gray,
        moving_gray=best_raw_gray,
        fixed_mask=fixed_mask,
        moving_mask=best_mask,
        ants_bin=ants_bin,
        tmp_root=tmp_root,
        tag=f"rawloc_{geometry.stem}",
    )
    raw_local_gate = strict_pareto_gate_decision(best_before_texture_metrics, dict(raw_local["metrics"]))
    raw_local_best_stage = "texture_rigid_local" if bool(raw_local_gate.get("accepted")) else shape_best_stage
    raw_local_final_metrics = dict(raw_local["metrics"]) if raw_local_best_stage == "texture_rigid_local" else dict(best_before_texture_metrics)

    fake_local = _run_rigid_candidate(
        fixed_gray=fixed_gray,
        moving_gray=best_fake_gray,
        fixed_mask=fixed_mask,
        moving_mask=best_mask,
        ants_bin=ants_bin,
        tmp_root=tmp_root,
        tag=f"fakeloc_{geometry.stem}",
    )
    fake_local_gate = strict_pareto_gate_decision(best_before_texture_metrics, dict(fake_local["metrics"]))
    fake_local_best_stage = "texture_rigid_local" if bool(fake_local_gate.get("accepted")) else shape_best_stage
    fake_local_final_metrics = dict(fake_local["metrics"]) if fake_local_best_stage == "texture_rigid_local" else dict(best_before_texture_metrics)

    return {
        "pair_key": geometry.pair_key,
        "stem": geometry.stem,
        "geometry_manifest_path": str(geometry.geometry_manifest_path),
        "sources": {
            "raw_nissl": raw_source,
            "fake_myelin": fake_source,
            "true_myelin": str(geometry.myelin_image_path),
        },
        "input_metrics": input_metrics,
        "input_metric_timing_seconds": input_metric_timing,
        "arms": {
            "input_only": {
                "best_stage": "input",
                "final_metrics": dict(input_metrics),
                "accepted_path": ["input"],
            },
            "mi_rigid": {
                "rigid_metrics": dict(mi_rigid["metrics"]),
                "rigid_gate": dict(mi_gate),
                "best_stage": mi_best_stage,
                "final_metrics": dict(mi_final_metrics),
                "timing_seconds": dict(mi_rigid["timing_seconds"]),
                "accepted_path": ["input"] + (["rigid"] if mi_best_stage == "rigid" else []),
            },
            "mask_rigid": {
                "mask_rigid_metrics": dict(shape_metrics),
                "mask_rigid_gate": dict(shape_gate),
                "mask_rigid_transform_params": dict(shape_opt["params"]),
                "best_stage": shape_best_stage,
                "final_metrics": dict(best_before_texture_metrics),
                "metric_timing_seconds": dict(shape_metric_timing),
                "accepted_path": ["input"] + (["mask_rigid"] if shape_best_stage == "mask_rigid" else []),
            },
            "shape_then_raw_texture_rigid": {
                "mask_rigid_metrics": dict(shape_metrics),
                "mask_rigid_gate": dict(shape_gate),
                "texture_stage_metrics": dict(raw_local["metrics"]),
                "texture_stage_gate": dict(raw_local_gate),
                "best_stage": raw_local_best_stage,
                "final_metrics": dict(raw_local_final_metrics),
                "timing_seconds": {
                    "texture_rigid": _safe_float(dict(raw_local["timing_seconds"]).get("stage_total")),
                },
                "accepted_path": ["input"]
                + (["mask_rigid"] if shape_best_stage == "mask_rigid" else [])
                + (["texture_rigid_local"] if raw_local_best_stage == "texture_rigid_local" else []),
            },
            "shape_then_fake_texture_rigid": {
                "mask_rigid_metrics": dict(shape_metrics),
                "mask_rigid_gate": dict(shape_gate),
                "texture_stage_metrics": dict(fake_local["metrics"]),
                "texture_stage_gate": dict(fake_local_gate),
                "best_stage": fake_local_best_stage,
                "final_metrics": dict(fake_local_final_metrics),
                "timing_seconds": {
                    "texture_rigid": _safe_float(dict(fake_local["timing_seconds"]).get("stage_total")),
                },
                "accepted_path": ["input"]
                + (["mask_rigid"] if shape_best_stage == "mask_rigid" else [])
                + (["texture_rigid_local"] if fake_local_best_stage == "texture_rigid_local" else []),
            },
        },
    }


def _result_row(case: dict[str, Any], arm: str) -> dict[str, Any]:
    arm_rec = dict((case.get("arms") or {}).get(arm) or {})
    out = {
        "pair_key": case.get("pair_key"),
        "stem": case.get("stem"),
        "arm": arm,
        "best_stage": arm_rec.get("best_stage"),
        "accepted_path": " -> ".join(list(arm_rec.get("accepted_path") or [])),
        "input_dice": _safe_float(dict(case.get("input_metrics") or {}).get("dice")),
        "input_hd95_px": _safe_float(dict(case.get("input_metrics") or {}).get("hd95_px")),
        "final_dice": _safe_float(dict(arm_rec.get("final_metrics") or {}).get("dice")),
        "final_hd95_px": _safe_float(dict(arm_rec.get("final_metrics") or {}).get("hd95_px")),
        "final_mi": _safe_float(dict(arm_rec.get("final_metrics") or {}).get("mi")),
        "final_cc": _safe_float(dict(arm_rec.get("final_metrics") or {}).get("cc")),
    }
    if "rigid_metrics" in arm_rec:
        out["rigid_gate_accepted"] = bool(dict(arm_rec.get("rigid_gate") or {}).get("accepted"))
        out["rigid_dice"] = _safe_float(dict(arm_rec.get("rigid_metrics") or {}).get("dice"))
        out["rigid_hd95_px"] = _safe_float(dict(arm_rec.get("rigid_metrics") or {}).get("hd95_px"))
    if "mask_rigid_metrics" in arm_rec:
        out["mask_rigid_gate_accepted"] = bool(dict(arm_rec.get("mask_rigid_gate") or {}).get("accepted"))
        out["mask_rigid_dice"] = _safe_float(dict(arm_rec.get("mask_rigid_metrics") or {}).get("dice"))
        out["mask_rigid_hd95_px"] = _safe_float(dict(arm_rec.get("mask_rigid_metrics") or {}).get("hd95_px"))
    if "texture_stage_metrics" in arm_rec:
        out["texture_gate_accepted"] = bool(dict(arm_rec.get("texture_stage_gate") or {}).get("accepted"))
        out["texture_dice"] = _safe_float(dict(arm_rec.get("texture_stage_metrics") or {}).get("dice"))
        out["texture_hd95_px"] = _safe_float(dict(arm_rec.get("texture_stage_metrics") or {}).get("hd95_px"))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Run shape-then-fake-texture hybrid registration benchmark.")
    ap.add_argument("--epoch", default=DEFAULT_EPOCH)
    ap.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    args = ap.parse_args()

    out_root = NANOZOOMER_ROOT / f"{SHORT_OUT_PREFIX}_{_utc_stamp()}"
    out_root.mkdir(parents=True, exist_ok=True)
    results_dir = out_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    rows = _load_manifest_rows()
    config = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "pair_count": len(rows),
        "epoch": str(args.epoch),
        "target_um_per_px": float(DEFAULT_TARGET_UM_PER_PX),
        "working_long_edge": int(DEFAULT_WORKING_LONG_EDGE),
        "evaluation_canvas": int(TARGET_EVAL_CANVAS),
        "mask_mode": "tissue_only",
        "preprocessing": "percentile_clipping_plus_fixed_background",
        "runner_mode": "best_accepted_state_runner",
        "stage_a": "mask_rigid",
        "stage_b_candidates": [
            "raw_nissl_texture_rigid_local",
            f"{args.epoch}_fake_myelin_texture_rigid_local",
        ],
        "notes": [
            "Stage B local refinement is implemented as rigid-from-identity after shape-prewarping into the current best accepted state.",
            "This first-pass benchmark is pair-level because fake-myelin exports are pair-level assets.",
        ],
    }
    _write_json(out_root / "experiment_config.json", config)

    tasks = [(row, str(out_root), str(args.epoch)) for row in rows]
    pair_results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    with cf.ProcessPoolExecutor(max_workers=max(1, int(args.workers))) as ex:
        future_map = {ex.submit(_run_pair_task, task): task for task in tasks}
        for fut in cf.as_completed(future_map):
            row = future_map[fut][0]
            try:
                result = fut.result()
                pair_results.append(result)
                _write_json(results_dir / f"{result['stem']}.json", result)
            except Exception as exc:
                failures.append({"pair_key": row.get("pair_key"), "error": str(exc)})
    elapsed = float(time.perf_counter() - t0)

    pair_results.sort(key=lambda r: str(r.get("pair_key") or ""))

    summary = {
        "config": config,
        "pair_count": len(rows),
        "completed_pair_count": len(pair_results),
        "failure_count": len(failures),
        "elapsed_seconds": _round_float(elapsed, 3),
        "failures": failures,
        "input_only": _arm_summary(pair_results, "input_only"),
        "mi_rigid": _arm_summary(pair_results, "mi_rigid"),
        "mask_rigid": _arm_summary(pair_results, "mask_rigid"),
        "shape_then_raw_texture_rigid": _arm_summary(pair_results, "shape_then_raw_texture_rigid", reference_arm="mask_rigid"),
        "shape_then_fake_texture_rigid": _arm_summary(pair_results, "shape_then_fake_texture_rigid", reference_arm="mask_rigid"),
    }
    _write_json(out_root / "hybrid_summary.json", summary)
    (out_root / "hybrid_summary.md").write_text(_markdown_summary(config, summary), encoding="utf-8")

    fieldnames: list[str] = []
    rows_out: list[dict[str, Any]] = []
    for case in pair_results:
        for arm in (
            "input_only",
            "mi_rigid",
            "mask_rigid",
            "shape_then_raw_texture_rigid",
            "shape_then_fake_texture_rigid",
        ):
            row = _result_row(case, arm)
            rows_out.append(row)
            for key in row.keys():
                if key not in fieldnames:
                    fieldnames.append(key)
    with (out_root / "hybrid_results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)

    print(f"Hybrid benchmark written to {out_root}")


if __name__ == "__main__":
    main()

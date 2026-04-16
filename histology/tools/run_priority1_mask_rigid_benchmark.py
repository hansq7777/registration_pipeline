from __future__ import annotations

import argparse
import concurrent.futures as cf
import csv
import glob
import json
import os
import sys
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
    MASK_SHAPE_REGISTRATION_BACKEND,
    PairRegistrationConfig,
    ensure_monotonic_gating_summary,
    run_pair_registration,
    strict_pareto_gate_decision,
)
from gui_mvp.hitl_gui.application.pair_workspace import load_pair_registry  # noqa: E402


DEFAULT_MYELIN_ROOT = Path(r"/mnt/d/Research/Image Analysis/Nanozoomer scans/20250327 rat myelin quantification/Tissue&Masks")
DEFAULT_NISSL_ROOT = Path(r"/mnt/d/Research/Image Analysis/Nanozoomer scans/20250424 Nissl cytoarchitectonic counterpart/Tissue&Masks")
DEFAULT_VARIANT = "gradient_mag_blur_1.5"
DEFAULT_TARGET_UM_PER_PX = 10.0
DEFAULT_WORKING_LONG_EDGE = 1024
DEFAULT_WORKERS = min(4, max(1, (os.cpu_count() or 1) // 2))


@dataclass(frozen=True)
class RegistrationUnit:
    pair_key: str
    group: str

    @property
    def key(self) -> str:
        return f"{self.pair_key}__group_{self.group}"


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


def _latest_matrix_results_csv() -> Path:
    matches = sorted(
        glob.glob("/mnt/d/Research/Image Analysis/Nanozoomer scans/histology_registration_preproc_matrix_*/matrix_results.csv")
    )
    if not matches:
        raise FileNotFoundError("Could not find any histology_registration_preproc_matrix_*/matrix_results.csv")
    return Path(matches[-1])


def _group_choices(review: dict[str, Any]) -> list[str]:
    if bool(review.get("multi_group_registration")):
        return ["1", "2"]
    return ["all"]


def _load_units(registry_path: Path) -> tuple[list[RegistrationUnit], dict[str, dict[str, Any]]]:
    registry = load_pair_registry(registry_path)
    usable_reviews = {
        pair_key: review
        for pair_key, review in sorted(registry.items())
        if isinstance(review, dict) and str(review.get("registration_status", "")).lower() == "usable"
    }
    units: list[RegistrationUnit] = []
    for pair_key, review in usable_reviews.items():
        for group in _group_choices(review):
            units.append(RegistrationUnit(pair_key=pair_key, group=group))
    return units, usable_reviews


def _load_mi_baseline_rows(matrix_csv: Path, variant: str) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with matrix_csv.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if str(row.get("variant") or "") != variant:
                continue
            pair_key = str(row.get("pair_key") or "")
            group = str(row.get("group") or "all")
            unit_key = f"{pair_key}__group_{group}"
            input_metrics = {
                "dice": _safe_float(row.get("input_dice")),
                "hd95_px": _safe_float(row.get("input_hd95_px")),
            }
            rigid_metrics = {
                "dice": _safe_float(row.get("rigid_dice")),
                "hd95_px": _safe_float(row.get("rigid_hd95_px")),
            }
            gate = strict_pareto_gate_decision(input_metrics, rigid_metrics)
            gate["stage"] = "rigid"
            gate["best_stage_before"] = "input"
            gate["best_stage_after"] = "rigid" if bool(gate.get("accepted")) else "input"
            rows[unit_key] = {
                "unit_key": unit_key,
                "pair_key": pair_key,
                "group": group,
                "variant": variant,
                "input_metrics": input_metrics,
                "candidate_metrics": rigid_metrics,
                "gate": gate,
                "timing_seconds": {
                    "rigid": _safe_float(row.get("rigid_time_s")),
                },
            }
    if not rows:
        raise RuntimeError(f"No rows found for variant {variant} in {matrix_csv}")
    return rows


def _mask_cfg(
    unit: RegistrationUnit,
    review: dict[str, Any],
    *,
    common_root: Path,
    myelin_root: Path,
    nissl_root: Path,
    runs_root: Path,
    target_um_per_px: float,
    working_long_edge: int,
) -> PairRegistrationConfig:
    return PairRegistrationConfig(
        pair_key=unit.pair_key,
        moving_side="nissl",
        fixed_side="myelin",
        moving_group=unit.group,
        fixed_group=unit.group,
        review=review,
        common_root=common_root,
        myelin_root=myelin_root,
        nissl_root=nissl_root,
        ants_bin=Path("."),
        runs_root=runs_root,
        target_um_per_px=float(target_um_per_px),
        working_long_edge=int(working_long_edge),
        pre_blur_sigma=0.0,
        registration_mask_mode="tissue_only",
        registration_backend=MASK_SHAPE_REGISTRATION_BACKEND,
        run_stages=("mask_rigid",),
    )


def _run_mask_rigid_unit(
    unit: RegistrationUnit,
    review: dict[str, Any],
    *,
    common_root: Path,
    myelin_root: Path,
    nissl_root: Path,
    runs_root: Path,
    target_um_per_px: float,
    working_long_edge: int,
) -> dict[str, Any]:
    cfg = _mask_cfg(
        unit,
        review,
        common_root=common_root,
        myelin_root=myelin_root,
        nissl_root=nissl_root,
        runs_root=runs_root,
        target_um_per_px=target_um_per_px,
        working_long_edge=working_long_edge,
    )
    result = run_pair_registration(cfg)
    manifest = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))
    gating = ensure_monotonic_gating_summary(manifest)
    stage = dict(manifest.get("stages", {})).get("mask_rigid", {})
    return {
        "pair_key": unit.pair_key,
        "group": unit.group,
        "unit_key": unit.key,
        "manifest_path": str(result["manifest_path"]),
        "run_dir": str(result["run_dir"]),
        "input_metrics": dict(manifest.get("input_metrics") or {}),
        "candidate_metrics": dict(stage.get("metrics") or {}),
        "gate": dict((gating.get("stages") or {}).get("mask_rigid") or {}),
        "timing_seconds": {
            "mask_rigid": _safe_float(dict(stage.get("timing_seconds") or {}).get("stage_total")),
        },
    }


def _run_mask_rigid_task(task: tuple[str, str, dict[str, Any], str, str, str, str, float, int]) -> dict[str, Any]:
    pair_key, group, review, common_root_s, myelin_root_s, nissl_root_s, runs_root_s, target_um_per_px, working_long_edge = task
    return _run_mask_rigid_unit(
        RegistrationUnit(pair_key=pair_key, group=group),
        review,
        common_root=Path(common_root_s),
        myelin_root=Path(myelin_root_s),
        nissl_root=Path(nissl_root_s),
        runs_root=Path(runs_root_s),
        target_um_per_px=float(target_um_per_px),
        working_long_edge=int(working_long_edge),
    )


def _method_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    success_count = 0
    dice_delta: list[float] = []
    hd95_delta: list[float] = []
    regression_hd95: list[float] = []
    strict_regression_count = 0
    dice_improved_count = 0
    hd95_improved_count = 0
    for row in rows:
        gate = dict(row.get("gate") or {})
        if bool(gate.get("accepted")):
            success_count += 1
        input_metrics = dict(row.get("input_metrics") or {})
        cand_metrics = dict(row.get("candidate_metrics") or {})
        d_dice = _safe_float(cand_metrics.get("dice")) - _safe_float(input_metrics.get("dice"))
        d_hd95 = _safe_float(cand_metrics.get("hd95_px")) - _safe_float(input_metrics.get("hd95_px"))
        if np.isfinite(d_dice):
            dice_delta.append(d_dice)
            if d_dice > 0:
                dice_improved_count += 1
        if np.isfinite(d_hd95):
            hd95_delta.append(d_hd95)
            if d_hd95 < 0:
                hd95_improved_count += 1
            if d_hd95 > 0:
                regression_hd95.append(d_hd95)
        if np.isfinite(d_dice) and np.isfinite(d_hd95) and d_dice < 0 and d_hd95 > 0:
            strict_regression_count += 1
    return {
        "unit_count": len(rows),
        "success_count": int(success_count),
        "success_rate": _round_float(success_count / len(rows)) if rows else 0.0,
        "dice_improved_count": int(dice_improved_count),
        "hd95_improved_count": int(hd95_improved_count),
        "strict_regression_count": int(strict_regression_count),
        "delta_dice": _summarize(dice_delta),
        "delta_hd95_px": _summarize(hd95_delta),
        "hd95_regression_tail_px": _summarize(regression_hd95),
    }


def _compare_methods(mask_rows: list[dict[str, Any]], mi_rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_unit_mask = {row["unit_key"]: row for row in mask_rows}
    by_unit_mi = {row["unit_key"]: row for row in mi_rows}
    common_keys = sorted(set(by_unit_mask) & set(by_unit_mi))
    mask_better_dice = 0
    mask_better_hd95 = 0
    mask_lower_strict_regression = 0
    per_unit: list[dict[str, Any]] = []
    for key in common_keys:
        mask = by_unit_mask[key]
        mi = by_unit_mi[key]
        mask_dice = _safe_float(dict(mask.get("candidate_metrics") or {}).get("dice"))
        mi_dice = _safe_float(dict(mi.get("candidate_metrics") or {}).get("dice"))
        mask_hd95 = _safe_float(dict(mask.get("candidate_metrics") or {}).get("hd95_px"))
        mi_hd95 = _safe_float(dict(mi.get("candidate_metrics") or {}).get("hd95_px"))
        input_dice = _safe_float(dict(mask.get("input_metrics") or {}).get("dice"))
        input_hd95 = _safe_float(dict(mask.get("input_metrics") or {}).get("hd95_px"))
        mask_strict_reg = int(mask_dice < input_dice and mask_hd95 > input_hd95)
        mi_strict_reg = int(mi_dice < input_dice and mi_hd95 > input_hd95)
        if np.isfinite(mask_dice) and np.isfinite(mi_dice) and mask_dice > mi_dice:
            mask_better_dice += 1
        if np.isfinite(mask_hd95) and np.isfinite(mi_hd95) and mask_hd95 < mi_hd95:
            mask_better_hd95 += 1
        if mask_strict_reg < mi_strict_reg:
            mask_lower_strict_regression += 1
        per_unit.append(
            {
                "unit_key": key,
                "pair_key": mask["pair_key"],
                "group": mask["group"],
                "input_dice": input_dice,
                "input_hd95_px": input_hd95,
                "mask_rigid_dice": mask_dice,
                "mask_rigid_hd95_px": mask_hd95,
                "mask_rigid_accepted": bool(dict(mask.get("gate") or {}).get("accepted")),
                "mi_rigid_dice": mi_dice,
                "mi_rigid_hd95_px": mi_hd95,
                "mi_rigid_accepted": bool(dict(mi.get("gate") or {}).get("accepted")),
            }
        )
    return {
        "common_unit_count": len(common_keys),
        "mask_better_dice_count": int(mask_better_dice),
        "mask_better_hd95_count": int(mask_better_hd95),
        "mask_lower_strict_regression_count": int(mask_lower_strict_regression),
        "per_unit": per_unit,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "unit_key",
        "pair_key",
        "group",
        "input_dice",
        "input_hd95_px",
        "mask_rigid_dice",
        "mask_rigid_hd95_px",
        "mask_rigid_accepted",
        "mi_rigid_dice",
        "mi_rigid_hd95_px",
        "mi_rigid_accepted",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def _write_summary_md(
    path: Path,
    *,
    payload: dict[str, Any],
    mask_summary: dict[str, Any],
    mi_summary: dict[str, Any],
    comparison: dict[str, Any],
) -> None:
    lines = [
        "# Priority 1 Mask-Rigid vs MI-Rigid Benchmark",
        "",
        f"- started_at_utc: `{payload['started_at_utc']}`",
        f"- completed_at_utc: `{payload['completed_at_utc']}`",
        f"- matrix_baseline_csv: `{payload['matrix_baseline_csv']}`",
        f"- mi_rigid_variant: `{payload['mi_rigid_variant']}`",
        f"- usable_pairs: `{payload['usable_pairs']}`",
        f"- registration_units: `{payload['registration_units']}`",
        f"- target_um_per_px: `{payload['target_um_per_px']}`",
        f"- working_long_edge: `{payload['working_long_edge']}`",
        "",
        "## Method Summary",
        "",
        f"- `mask_rigid` success_rate: `{mask_summary['success_rate']}`",
        f"- `MI-rigid` success_rate: `{mi_summary['success_rate']}`",
        f"- `mask_rigid` strict_regression_count: `{mask_summary['strict_regression_count']}`",
        f"- `MI-rigid` strict_regression_count: `{mi_summary['strict_regression_count']}`",
        f"- `mask_rigid` HD95 regression tail p95/max: `{mask_summary['hd95_regression_tail_px'].get('p95', 'na')}` / `{mask_summary['hd95_regression_tail_px'].get('max', 'na')}`",
        f"- `MI-rigid` HD95 regression tail p95/max: `{mi_summary['hd95_regression_tail_px'].get('p95', 'na')}` / `{mi_summary['hd95_regression_tail_px'].get('max', 'na')}`",
        "",
        "## Direct Head-to-Head",
        "",
        f"- common_unit_count: `{comparison['common_unit_count']}`",
        f"- mask better Dice count: `{comparison['mask_better_dice_count']}`",
        f"- mask better HD95 count: `{comparison['mask_better_hd95_count']}`",
        f"- mask lower strict regression count on unit-wise comparison: `{comparison['mask_lower_strict_regression_count']}`",
        "",
        "## Interpretation",
        "",
        "- `success_rate` is the strict Pareto acceptance rate against `input-only` under the current gate (`Dice` must not worsen, `HD95` must not worsen, and one must improve strictly).",
        "- `strict_regression_count` means both `Dice` worsened and `HD95` worsened against input.",
        "- `HD95 regression tail` summarizes only positive HD95 regressions, because that is the failure mode we want to shrink.",
        "",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Priority 1 mask-rigid full benchmark against the strongest MI-rigid baseline.")
    parser.add_argument("--myelin-root", type=Path, default=DEFAULT_MYELIN_ROOT)
    parser.add_argument("--nissl-root", type=Path, default=DEFAULT_NISSL_ROOT)
    parser.add_argument("--matrix-baseline-csv", type=Path, default=None)
    parser.add_argument("--mi-rigid-variant", default=DEFAULT_VARIANT)
    parser.add_argument("--target-um-per-px", type=float, default=DEFAULT_TARGET_UM_PER_PX)
    parser.add_argument("--working-long-edge", type=int, default=DEFAULT_WORKING_LONG_EDGE)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    common_root = Path(os.path.commonpath([str(args.myelin_root.resolve()), str(args.nissl_root.resolve())]))
    registry_path = common_root / "histology_pair_qc_registry.json"
    matrix_csv = args.matrix_baseline_csv or _latest_matrix_results_csv()
    units, usable_reviews = _load_units(registry_path)
    if args.limit and args.limit > 0:
        units = units[: args.limit]

    out_root = common_root / f"histology_priority1_mask_rigid_benchmark_{_utc_stamp()}"
    runs_root = out_root / "mask_rigid_runs"
    out_root.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "started_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "myelin_root": str(args.myelin_root),
        "nissl_root": str(args.nissl_root),
        "registry_path": str(registry_path),
        "matrix_baseline_csv": str(matrix_csv),
        "mi_rigid_variant": str(args.mi_rigid_variant),
        "usable_pairs": len(usable_reviews),
        "registration_units": len(units),
        "target_um_per_px": float(args.target_um_per_px),
        "working_long_edge": int(args.working_long_edge),
        "workers": int(max(1, int(args.workers))),
        "out_root": str(out_root),
        "runs_root": str(runs_root),
    }
    _write_json(out_root / "experiment_config.json", payload)

    mi_rows = _load_mi_baseline_rows(matrix_csv, str(args.mi_rigid_variant))

    mask_rows: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    total = len(units)
    tasks = [
        (
            unit.pair_key,
            unit.group,
            usable_reviews[unit.pair_key],
            str(common_root),
            str(args.myelin_root),
            str(args.nissl_root),
            str(runs_root),
            float(args.target_um_per_px),
            int(args.working_long_edge),
        )
        for unit in units
    ]
    workers = max(1, int(args.workers))
    if workers == 1:
        for idx, task in enumerate(tasks, start=1):
            print(f"[{idx}/{total}] mask_rigid {task[0]}__group_{task[1]} ...", flush=True)
            mask_rows.append(_run_mask_rigid_task(task))
    else:
        with cf.ProcessPoolExecutor(max_workers=workers) as ex:
            future_to_meta = {
                ex.submit(_run_mask_rigid_task, task): (idx, task[0], task[1])
                for idx, task in enumerate(tasks, start=1)
            }
            for fut in cf.as_completed(future_to_meta):
                idx, pair_key, group = future_to_meta[fut]
                print(f"[{idx}/{total}] mask_rigid {pair_key}__group_{group} finished", flush=True)
                mask_rows.append(fut.result())
    mask_rows.sort(key=lambda row: (str(row["pair_key"]), str(row["group"])))

    mask_summary = _method_summary(mask_rows)
    mi_summary = _method_summary(list(mi_rows.values()))
    comparison = _compare_methods(mask_rows, list(mi_rows.values()))

    payload["completed_at_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    payload["wall_seconds"] = _round_float(time.perf_counter() - t0)
    payload["mask_rigid_summary"] = mask_summary
    payload["mi_rigid_summary"] = mi_summary
    payload["comparison"] = {k: v for k, v in comparison.items() if k != "per_unit"}

    _write_json(out_root / "benchmark_summary.json", payload)
    _write_csv(out_root / "head_to_head.csv", comparison["per_unit"])
    _write_summary_md(
        out_root / "benchmark_summary.md",
        payload=payload,
        mask_summary=mask_summary,
        mi_summary=mi_summary,
        comparison=comparison,
    )
    print(f"Done. Summary: {out_root / 'benchmark_summary.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

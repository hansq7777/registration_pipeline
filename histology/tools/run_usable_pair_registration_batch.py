from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

REPO_HISTOLOGY_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_HISTOLOGY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_HISTOLOGY_ROOT))

from gui_mvp.hitl_gui.application.pair_registration import (  # noqa: E402
    ANTS_REGISTRATION_BACKEND,
    MASK_SHAPE_REGISTRATION_BACKEND,
    PairRegistrationConfig,
    default_pair_registration_runs_root,
    ensure_monotonic_gating_summary,
    find_ants_bin,
    run_pair_registration,
    stage_order_for_backend,
)
from gui_mvp.hitl_gui.application.pair_workspace import load_pair_registry  # noqa: E402


DEFAULT_MYELIN_ROOT = Path(r"/mnt/d/Research/Image Analysis/Nanozoomer scans/20250327 rat myelin quantification/Tissue&Masks")
DEFAULT_NISSL_ROOT = Path(r"/mnt/d/Research/Image Analysis/Nanozoomer scans/20250424 Nissl cytoarchitectonic counterpart/Tissue&Masks")


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _round_float(value: float, digits: int = 6) -> float:
    return round(float(value), digits)


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def _summarize_values(values: list[float]) -> dict[str, float | int]:
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


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _compute_robustness_summary(manifest_paths: list[str]) -> dict[str, Any]:
    unique_paths = []
    seen: set[str] = set()
    for item in manifest_paths:
        path_str = str(item).strip()
        if not path_str or path_str in seen:
            continue
        seen.add(path_str)
        unique_paths.append(path_str)

    final_best_stage_counts: dict[str, int] = {}
    final_delta_vs_input_raw: dict[str, list[float]] = {key: [] for key in ("dice", "hd95_px", "mi", "cc")}
    per_stage_raw: dict[str, dict[str, Any]] = {}

    analyzed_runs = 0
    for path_str in unique_paths:
        manifest_path = Path(path_str)
        if not manifest_path.exists():
            continue
        manifest = _load_json(manifest_path)
        gating = ensure_monotonic_gating_summary(manifest)
        analyzed_runs += 1

        best_stage = str(gating.get("best_stage", "input"))
        final_best_stage_counts[best_stage] = final_best_stage_counts.get(best_stage, 0) + 1

        final_delta = gating.get("final_delta_vs_input", {})
        for key in final_delta_vs_input_raw:
            value = _safe_float(final_delta.get(key))
            if np.isfinite(value):
                final_delta_vs_input_raw[key].append(value)

        for stage, gate in dict(gating.get("stages", {})).items():
            bucket = per_stage_raw.setdefault(
                stage,
                {
                    "candidate_count": 0,
                    "accepted_count": 0,
                    "rejected_count": 0,
                    "decision_counts": {},
                    "delta_vs_best_before": {key: [] for key in ("dice", "hd95_px", "mi", "cc")},
                    "delta_vs_input": {key: [] for key in ("dice", "hd95_px", "mi", "cc")},
                    "dice_regression_magnitude": [],
                    "hd95_regression_magnitude": [],
                    "mi_improved_rejected_count": 0,
                    "cc_improved_rejected_count": 0,
                    "intensity_improved_rejected_count": 0,
                },
            )
            bucket["candidate_count"] += 1
            accepted = bool(gate.get("accepted"))
            if accepted:
                bucket["accepted_count"] += 1
            else:
                bucket["rejected_count"] += 1
            decision = str(gate.get("decision", "unknown"))
            bucket["decision_counts"][decision] = int(bucket["decision_counts"].get(decision, 0)) + 1

            delta_best = dict(gate.get("delta_vs_best_before", {}))
            delta_input = dict(gate.get("delta_vs_input", {}))
            for key in ("dice", "hd95_px", "mi", "cc"):
                value_best = _safe_float(delta_best.get(key))
                if np.isfinite(value_best):
                    bucket["delta_vs_best_before"][key].append(value_best)
                value_input = _safe_float(delta_input.get(key))
                if np.isfinite(value_input):
                    bucket["delta_vs_input"][key].append(value_input)

            if not accepted:
                dice_delta = _safe_float(delta_best.get("dice"))
                hd95_delta = _safe_float(delta_best.get("hd95_px"))
                mi_delta = _safe_float(delta_best.get("mi"))
                cc_delta = _safe_float(delta_best.get("cc"))
                if np.isfinite(dice_delta) and dice_delta < 0:
                    bucket["dice_regression_magnitude"].append(-dice_delta)
                if np.isfinite(hd95_delta) and hd95_delta > 0:
                    bucket["hd95_regression_magnitude"].append(hd95_delta)
                if np.isfinite(mi_delta) and mi_delta > 0:
                    bucket["mi_improved_rejected_count"] += 1
                if np.isfinite(cc_delta) and cc_delta > 0:
                    bucket["cc_improved_rejected_count"] += 1
                if (np.isfinite(mi_delta) and mi_delta > 0) or (np.isfinite(cc_delta) and cc_delta > 0):
                    bucket["intensity_improved_rejected_count"] += 1

    per_stage: dict[str, Any] = {}
    stage_order = ("mask_rigid", "mask_similarity", "rigid", "affine", "syn")
    for stage in sorted(per_stage_raw.keys(), key=lambda s: stage_order.index(s) if s in stage_order else 99):
        bucket = per_stage_raw[stage]
        candidate_count = int(bucket["candidate_count"])
        per_stage[stage] = {
            "candidate_count": candidate_count,
            "accepted_count": int(bucket["accepted_count"]),
            "rejected_count": int(bucket["rejected_count"]),
            "accept_rate": _round_float(bucket["accepted_count"] / candidate_count) if candidate_count else 0.0,
            "decision_counts": dict(bucket["decision_counts"]),
            "delta_vs_best_before": {key: _summarize_values(vals) for key, vals in bucket["delta_vs_best_before"].items()},
            "delta_vs_input": {key: _summarize_values(vals) for key, vals in bucket["delta_vs_input"].items()},
            "dice_regression_magnitude": _summarize_values(bucket["dice_regression_magnitude"]),
            "hd95_regression_magnitude": _summarize_values(bucket["hd95_regression_magnitude"]),
            "mi_improved_rejected_count": int(bucket["mi_improved_rejected_count"]),
            "cc_improved_rejected_count": int(bucket["cc_improved_rejected_count"]),
            "intensity_improved_rejected_count": int(bucket["intensity_improved_rejected_count"]),
        }

    final_best_stage_fraction = {
        stage: _round_float(count / analyzed_runs) if analyzed_runs else 0.0
        for stage, count in sorted(final_best_stage_counts.items())
    }
    return {
        "policy_name": "strict_pareto_geometry_v1",
        "analyzed_run_count": analyzed_runs,
        "final_best_stage_counts": dict(sorted(final_best_stage_counts.items())),
        "final_best_stage_fraction": final_best_stage_fraction,
        "final_delta_vs_input": {key: _summarize_values(vals) for key, vals in final_delta_vs_input_raw.items()},
        "per_stage": per_stage,
    }


def _write_md(path: Path, payload: dict) -> None:
    lines = [
        f"# Usable Pair Registration Batch {payload['batch_id']}",
        "",
        f"- started_at_utc: `{payload['started_at_utc']}`",
        f"- completed_at_utc: `{payload.get('completed_at_utc', '')}`",
        f"- myelin_root: `{payload['myelin_root']}`",
        f"- nissl_root: `{payload['nissl_root']}`",
        f"- ants_bin: `{payload['ants_bin']}`",
        f"- registration_backend: `{payload.get('registration_backend', '')}`",
        f"- moving_side: `{payload['moving_side']}`",
        f"- fixed_side: `{payload['fixed_side']}`",
        f"- moving_group: `{payload['moving_group']}`",
        f"- fixed_group: `{payload['fixed_group']}`",
        f"- registration_mask_mode: `{payload['registration_mask_mode']}`",
        f"- mask_similarity_scale_percent: `{payload.get('mask_similarity_scale_percent', 0.0)}`",
        f"- target_um_per_px: `{payload['target_um_per_px']}`",
        f"- run_stages: `{','.join(payload['run_stages'])}`",
        f"- usable_pairs_total: `{payload['usable_pairs_total']}`",
        f"- skipped_count: `{payload['skipped_count']}`",
        f"- success_count: `{payload['success_count']}`",
        f"- failure_count: `{payload['failure_count']}`",
        f"- analyzed_manifest_count: `{payload.get('analyzed_manifest_count', 0)}`",
        f"- wall_seconds: `{payload.get('wall_seconds', 0.0):.1f}`",
        "",
    ]
    robustness = payload.get("robustness_summary") or {}
    if robustness:
        lines.extend(
            [
                "## Robustness Summary",
                "",
                f"- policy_name: `{robustness.get('policy_name', '')}`",
                f"- analyzed_run_count: `{robustness.get('analyzed_run_count', 0)}`",
                "",
                "### Final Best Stage Distribution",
                "",
            ]
        )
        for stage, count in dict(robustness.get("final_best_stage_counts", {})).items():
            frac = dict(robustness.get("final_best_stage_fraction", {})).get(stage, 0.0)
            lines.append(f"- {stage}: `{count}` ({frac:.3f})")
        lines.extend(["", "### Stage Acceptance", ""])
        for stage, stats in dict(robustness.get("per_stage", {})).items():
            lines.extend(
                [
                    f"- {stage}:",
                    f"  - candidate_count: `{stats.get('candidate_count', 0)}`",
                    f"  - accepted_count: `{stats.get('accepted_count', 0)}`",
                    f"  - rejected_count: `{stats.get('rejected_count', 0)}`",
                    f"  - accept_rate: `{float(stats.get('accept_rate', 0.0)):.3f}`",
                    f"  - intensity_improved_rejected_count: `{stats.get('intensity_improved_rejected_count', 0)}`",
                ]
            )
        lines.extend(["", "## Successes", ""])
    else:
        lines.extend(["## Successes", ""])
    for row in payload["successes"]:
        lines.extend(
            [
                f"- `{row['pair_key']}`",
                f"  - seconds: `{row['seconds']:.1f}`",
                f"  - manifest: `{row['manifest_path']}`",
                f"  - best_stage: `{row.get('best_stage', '')}`",
            ]
        )
    if payload["failures"]:
        lines.extend(["", "## Failures", ""])
        for row in payload["failures"]:
            lines.extend(
                [
                    f"- `{row['pair_key']}`",
                    f"  - error: `{row['error']}`",
                ]
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _has_matching_run(
    runs_root: Path,
    pair_key: str,
    *,
    registration_backend: str,
    moving_side: str,
    fixed_side: str,
    moving_group: str,
    fixed_group: str,
    target_um_per_px: float,
    registration_mask_mode: str,
    mask_similarity_scale_percent: float,
    run_stages: list[str],
) -> str | None:
    pair_dir = runs_root / pair_key
    if not pair_dir.exists():
        return None
    for manifest_path in sorted(pair_dir.glob("*/run_manifest.json")):
        try:
            obj = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if (
            str(obj.get("registration_backend") or ANTS_REGISTRATION_BACKEND) == str(registration_backend)
            and
            str(obj.get("moving_side")) == moving_side
            and str(obj.get("fixed_side")) == fixed_side
            and str(obj.get("moving_group")) == moving_group
            and str(obj.get("fixed_group")) == fixed_group
            and float(obj.get("inputs", {}).get("target_um_per_px") or 0.0) == float(target_um_per_px)
            and str(obj.get("registration_mask_mode")) == registration_mask_mode
            and float(obj.get("mask_similarity_scale_percent") or 5.0) == float(mask_similarity_scale_percent)
            and list(obj.get("run_stages") or []) == list(run_stages)
        ):
            return str(manifest_path)
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Batch-run histology pair registration for all usable pairs.")
    parser.add_argument("--myelin-root", type=Path, default=DEFAULT_MYELIN_ROOT)
    parser.add_argument("--nissl-root", type=Path, default=DEFAULT_NISSL_ROOT)
    parser.add_argument("--registration-backend", choices=[ANTS_REGISTRATION_BACKEND, MASK_SHAPE_REGISTRATION_BACKEND], default=ANTS_REGISTRATION_BACKEND)
    parser.add_argument("--moving-side", choices=["myelin", "nissl"], default="myelin")
    parser.add_argument("--fixed-side", choices=["myelin", "nissl"], default="nissl")
    parser.add_argument("--moving-group", default="all")
    parser.add_argument("--fixed-group", default="all")
    parser.add_argument("--registration-mask-mode", default="union")
    parser.add_argument("--mask-similarity-scale-percent", type=float, default=5.0)
    parser.add_argument("--target-um-per-px", type=float, default=10.0)
    parser.add_argument("--stages", nargs="+", default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--skip-finished", action="store_true")
    args = parser.parse_args()
    if not args.stages:
        default_stage_order = stage_order_for_backend(args.registration_backend)
        if args.registration_backend == ANTS_REGISTRATION_BACKEND:
            args.stages = list(default_stage_order[:2])
        else:
            args.stages = list(default_stage_order)

    common_root = Path(os.path.commonpath([str(args.myelin_root.resolve()), str(args.nissl_root.resolve())]))
    registry_path = common_root / "histology_pair_qc_registry.json"
    runs_root = default_pair_registration_runs_root(args.myelin_root, args.nissl_root)
    if runs_root is None:
        raise RuntimeError("Failed to resolve runs root.")
    runs_root.mkdir(parents=True, exist_ok=True)

    ants_bin = find_ants_bin() if args.registration_backend == ANTS_REGISTRATION_BACKEND else None
    if args.registration_backend == ANTS_REGISTRATION_BACKEND and ants_bin is None and not args.skip_finished:
        raise RuntimeError("ANTs binary directory not found.")

    pairs = load_pair_registry(registry_path)
    usable_items = [
        (pair_key, review)
        for pair_key, review in sorted(pairs.items())
        if isinstance(review, dict) and str(review.get("registration_status", "")).lower() == "usable"
    ]
    if args.limit and args.limit > 0:
        usable_items = usable_items[: args.limit]

    batch_id = f"{_utc_stamp()}_{args.moving_side}_{args.fixed_side}_{'-'.join(args.stages)}"
    payload: dict[str, object] = {
        "batch_id": batch_id,
        "started_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "myelin_root": str(args.myelin_root),
        "nissl_root": str(args.nissl_root),
        "registry_path": str(registry_path),
        "runs_root": str(runs_root),
        "ants_bin": str(ants_bin or ""),
        "registration_backend": args.registration_backend,
        "moving_side": args.moving_side,
        "fixed_side": args.fixed_side,
        "moving_group": args.moving_group,
        "fixed_group": args.fixed_group,
        "registration_mask_mode": args.registration_mask_mode,
        "mask_similarity_scale_percent": float(args.mask_similarity_scale_percent),
        "target_um_per_px": float(args.target_um_per_px),
        "run_stages": list(args.stages),
        "usable_pairs_total": len(usable_items),
        "skipped_count": 0,
        "success_count": 0,
        "failure_count": 0,
        "skipped": [],
        "successes": [],
        "failures": [],
        "analyzed_manifest_count": 0,
        "robustness_summary": {},
    }

    summary_json = runs_root / f"usable_pair_batch_{batch_id}.json"
    summary_md = runs_root / f"usable_pair_batch_{batch_id}.md"
    _write_json(summary_json, payload)

    batch_t0 = time.perf_counter()
    total = len(usable_items)
    for idx, (pair_key, review) in enumerate(usable_items, start=1):
        t0 = time.perf_counter()
        print(f"[{idx}/{total}] {pair_key} ...", flush=True)
        if args.skip_finished:
            existing_manifest = _has_matching_run(
                runs_root,
                pair_key,
                registration_backend=args.registration_backend,
                moving_side=args.moving_side,
                fixed_side=args.fixed_side,
                moving_group=args.moving_group,
                fixed_group=args.fixed_group,
                target_um_per_px=float(args.target_um_per_px),
                registration_mask_mode=str(args.registration_mask_mode),
                mask_similarity_scale_percent=float(args.mask_similarity_scale_percent),
                run_stages=[str(x).strip().lower() for x in args.stages],
            )
            if existing_manifest:
                payload["skipped_count"] = int(payload["skipped_count"]) + 1
                payload["skipped"].append({"pair_key": pair_key, "manifest_path": existing_manifest})
                print(f"  skipped existing: {existing_manifest}", flush=True)
                _write_json(summary_json, payload)
                continue
        cfg = PairRegistrationConfig(
            pair_key=pair_key,
            moving_side=args.moving_side,
            fixed_side=args.fixed_side,
            moving_group=args.moving_group,
            fixed_group=args.fixed_group,
            review=review,
            common_root=common_root,
            myelin_root=args.myelin_root,
            nissl_root=args.nissl_root,
            ants_bin=ants_bin if ants_bin is not None else Path("."),
            runs_root=runs_root,
            target_um_per_px=float(args.target_um_per_px),
            registration_mask_mode=str(args.registration_mask_mode),
            registration_backend=str(args.registration_backend),
            run_stages=tuple(str(x).strip().lower() for x in args.stages),
            mask_similarity_scale_percent=float(args.mask_similarity_scale_percent),
        )
        try:
            if args.registration_backend == ANTS_REGISTRATION_BACKEND and ants_bin is None:
                ants_bin = find_ants_bin()
                if ants_bin is None:
                    raise RuntimeError("ANTs binary directory not found for unfinished pairs.")
                cfg = PairRegistrationConfig(
                    pair_key=pair_key,
                    moving_side=args.moving_side,
                    fixed_side=args.fixed_side,
                    moving_group=args.moving_group,
                    fixed_group=args.fixed_group,
                    review=review,
                    common_root=common_root,
                    myelin_root=args.myelin_root,
                    nissl_root=args.nissl_root,
                    ants_bin=ants_bin,
                    runs_root=runs_root,
                    target_um_per_px=float(args.target_um_per_px),
                    registration_mask_mode=str(args.registration_mask_mode),
                    registration_backend=str(args.registration_backend),
                    run_stages=tuple(str(x).strip().lower() for x in args.stages),
                    mask_similarity_scale_percent=float(args.mask_similarity_scale_percent),
                )
            result = run_pair_registration(cfg)
            seconds = float(time.perf_counter() - t0)
            payload["success_count"] = int(payload["success_count"]) + 1
            payload["successes"].append(
                {
                    "pair_key": pair_key,
                    "seconds": seconds,
                    "run_dir": result["run_dir"],
                    "manifest_path": result["manifest_path"],
                    "storyboard_path": result["storyboard_path"],
                    "best_stage": result.get("best_stage", ""),
                    "accepted_stage_path": result.get("accepted_stage_path", []),
                    "final_delta_vs_input": result.get("final_delta_vs_input", {}),
                }
            )
            print(f"  ok {seconds:.1f}s", flush=True)
        except Exception as exc:
            payload["failure_count"] = int(payload["failure_count"]) + 1
            payload["failures"].append({"pair_key": pair_key, "error": str(exc)})
            print(f"  failed: {exc}", flush=True)
        _write_json(summary_json, payload)

    analyzed_manifest_paths = [str(row.get("manifest_path", "")) for row in payload["successes"]]
    analyzed_manifest_paths.extend(str(row.get("manifest_path", "")) for row in payload["skipped"])
    payload["analyzed_manifest_count"] = len({p for p in analyzed_manifest_paths if str(p).strip()})
    payload["robustness_summary"] = _compute_robustness_summary(analyzed_manifest_paths)
    payload["completed_at_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    payload["wall_seconds"] = float(time.perf_counter() - batch_t0)
    _write_json(summary_json, payload)
    _write_md(summary_md, payload)
    print(f"summary_json={summary_json}")
    print(f"summary_md={summary_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

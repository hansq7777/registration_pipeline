#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any


NONLINEAR_ALGOS_DEFAULT = [
    "syn",
    "bspline_syn",
    "bspline_displacement_field",
    "gaussian_displacement_field",
    "time_varying_velocity_field",
    "time_varying_bspline_velocity_field",
    "exponential",
    "bspline_exponential",
]

NONLINEAR_FAMILY = {
    "syn": "diffeomorphic",
    "bspline_syn": "diffeomorphic",
    "bspline_displacement_field": "non_diffeomorphic_displacement",
    "gaussian_displacement_field": "non_diffeomorphic_displacement",
    "time_varying_velocity_field": "diffeomorphic_velocity",
    "time_varying_bspline_velocity_field": "diffeomorphic_velocity",
    "exponential": "diffeomorphic_velocity",
    "bspline_exponential": "diffeomorphic_velocity",
    "custom": "custom",
}


def _now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _finite(value: float) -> bool:
    return isinstance(value, float) and math.isfinite(value)


def _higher(value: Any) -> float:
    x = _safe_float(value, float("nan"))
    return x if _finite(x) else -1e30


def _lower(value: Any) -> float:
    x = _safe_float(value, float("nan"))
    return x if _finite(x) else 1e30


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _tail_text(text: str, lines: int = 30) -> str:
    rows = [ln for ln in text.splitlines() if ln.strip()]
    return "\n".join(rows[-lines:])


def _manifest_for_run(out_dir: Path, run_id: str) -> Path | None:
    preferred = out_dir / f"run_manifest_{run_id}.json"
    if preferred.exists():
        return preferred
    cands = sorted(out_dir.glob("run_manifest_*.json"))
    return cands[-1] if cands else None


def _resolve_storyboard_script(user_path: str) -> Path:
    if user_path.strip():
        p = Path(user_path).expanduser().resolve()
    else:
        p = Path(__file__).resolve().parent / "registration_storyboard.py"
    if not p.exists():
        raise FileNotFoundError(f"registration_storyboard.py not found: {p}")
    return p


def _run_storyboard(
    *,
    storyboard_script: Path,
    run_dir: Path,
    prefix: str,
    dpi: int,
) -> dict[str, Any]:
    log_path = run_dir / f"{prefix}storyboard.log.txt"
    cmd = [
        sys.executable,
        str(storyboard_script),
        "--run-dir",
        str(run_dir),
        "--prefix",
        prefix,
        "--dpi",
        str(max(int(dpi), 72)),
    ]
    done = subprocess.run(cmd, text=True, capture_output=True)
    log_text = (
        "$ " + " ".join(cmd) + "\n"
        + f"[return_code] {done.returncode}\n"
        + "[stdout]\n" + (done.stdout or "") + "\n"
        + "[stderr]\n" + (done.stderr or "") + "\n"
    )
    log_path.write_text(log_text, encoding="utf-8")
    storyboard_png = run_dir / "registration_storyboard.png"
    storyboard_manifest = run_dir / "registration_storyboard_manifest.json"
    return {
        "storyboard_status": "success" if done.returncode == 0 and storyboard_png.exists() else "failed",
        "storyboard_return_code": int(done.returncode),
        "storyboard_png": str(storyboard_png) if storyboard_png.exists() else "",
        "storyboard_manifest": str(storyboard_manifest) if storyboard_manifest.exists() else "",
        "storyboard_log": str(log_path),
        "storyboard_error_tail": _tail_text((done.stderr or "") + "\n" + (done.stdout or ""), lines=30)
        if done.returncode != 0
        else "",
    }


def _score_row(row: dict[str, Any]) -> tuple[float, float, float, float, float]:
    return (
        _higher(row.get("dice")),
        -_lower(row.get("jacobian_negative_fraction")),
        -_lower(row.get("warp_l2_energy_mean")),
        _higher(row.get("cc")),
        -_lower(row.get("runtime_seconds")),
    )


def _resolve_parallel_settings(
    *,
    requested_threads: int,
    requested_max_parallel: int,
    continue_on_fail: str,
) -> dict[str, Any]:
    cpu_count = max(int(os.cpu_count() or 1), 1)
    threads_req = max(int(requested_threads), 1)
    parallel_req = max(int(requested_max_parallel), 1)
    parallel_eff = min(parallel_req, cpu_count)
    threads_eff = threads_req
    oversubscribe_guard_applied = False

    if continue_on_fail == "off" and parallel_eff > 1:
        parallel_eff = 1

    if threads_eff * parallel_eff > cpu_count:
        oversubscribe_guard_applied = True
        threads_eff = max(1, cpu_count // parallel_eff)

    return {
        "cpu_count": cpu_count,
        "threads_requested": threads_req,
        "threads_effective": threads_eff,
        "max_parallel_requested": parallel_req,
        "max_parallel_effective": parallel_eff,
        "oversubscribe_guard_applied": oversubscribe_guard_applied,
    }


def _preproc_cache_ready(run_dir: Path, prefix: str) -> bool:
    pre = run_dir / "preproc"
    return (pre / f"{prefix}fixed_n4.nii.gz").exists() and (pre / f"{prefix}moving_n4.nii.gz").exists()


def _linear_cache_ready(run_dir: Path, prefix: str) -> bool:
    stages = run_dir / "stages"
    return (
        (stages / "rigid").exists()
        and (stages / "affine").exists()
        and (stages / "affine" / f"{prefix}affine_0GenericAffine.mat").exists()
    )


def _upstream_cache_ready(run_dir: Path, prefix: str) -> bool:
    return _preproc_cache_ready(run_dir, prefix) and _linear_cache_ready(run_dir, prefix)


def _resolve_existing_upstream_source(
    *,
    algos: list[str],
    records_by_algo: dict[str, dict[str, Any]],
    prefix: str,
) -> Path | None:
    for algo in algos:
        row = records_by_algo.get(algo)
        if not row:
            continue
        if str(row.get("status", "")).strip().lower() != "success":
            continue
        out_dir = Path(str(row.get("output_dir", "")))
        if out_dir.exists() and _upstream_cache_ready(out_dir, prefix):
            return out_dir
    return None


def _collect_row_from_outputs(
    *,
    row: dict[str, Any],
    algo_dir: Path,
    run_id: str,
    prefix: str,
) -> dict[str, Any]:
    manifest_path = _manifest_for_run(algo_dir, run_id)
    if manifest_path and manifest_path.exists():
        try:
            manifest = _read_json(manifest_path)
            row["manifest_path"] = str(manifest_path)
            row["selected_init_method"] = str((manifest.get("selected_init_attempt") or {}).get("init_method", ""))
            row["selected_attempt_name"] = str((manifest.get("selected_init_attempt") or {}).get("attempt_name", ""))
            gate = (manifest.get("selected_init_attempt") or {}).get("attempt_gate", {}) or {}
            if isinstance(gate, dict):
                row["hard_gate_passed"] = bool(gate.get("passed")) if "passed" in gate else None
                row["hard_gate_reasons"] = list(gate.get("reasons", []) or [])
                row["hard_gate_warnings"] = list(gate.get("warnings", []) or [])
                jac = dict(gate.get("jacobian", {}) or {})
                warp = dict(gate.get("warp_energy", {}) or {})
                row["jacobian_min"] = _safe_float(jac.get("min"), float("nan"))
                row["jacobian_p01"] = _safe_float(jac.get("p01"), float("nan"))
                row["jacobian_p99"] = _safe_float(jac.get("p99"), float("nan"))
                row["jacobian_negative_fraction"] = _safe_float(jac.get("negative_fraction"), float("nan"))
                row["warp_l2_energy_mean"] = _safe_float(warp.get("l2_energy_mean"), float("nan"))
        except Exception:
            pass

    qc_path = algo_dir / f"{prefix}qc_metrics.json"
    if qc_path.exists():
        try:
            qc = _read_json(qc_path)
            row["qc_path"] = str(qc_path)
            row["dice"] = _safe_float(qc.get("dice"), float("nan"))
            row["cc"] = _safe_float(qc.get("cc"), float("nan"))
            row["mi"] = _safe_float(qc.get("mi"), float("nan"))
            row["nmi"] = _safe_float(qc.get("nmi"), float("nan"))
        except Exception:
            pass

    jac_path = algo_dir / f"{prefix}jacobian_metrics.json"
    if jac_path.exists():
        try:
            jac_m = _read_json(jac_path)
            jac = dict(jac_m.get("jacobian", {}) or {})
            warp = dict(jac_m.get("warp_energy", {}) or {})
            row["jacobian_min"] = _safe_float(jac.get("min"), row["jacobian_min"])
            row["jacobian_p01"] = _safe_float(jac.get("p01"), row["jacobian_p01"])
            row["jacobian_p99"] = _safe_float(jac.get("p99"), row["jacobian_p99"])
            row["jacobian_negative_fraction"] = _safe_float(
                jac.get("negative_fraction"),
                row["jacobian_negative_fraction"],
            )
            row["warp_l2_energy_mean"] = _safe_float(
                warp.get("l2_energy_mean"),
                row["warp_l2_energy_mean"],
            )
        except Exception:
            pass

    return row


def _build_row_template(*, algo: str, run_id: str, algo_dir: Path, log_path: Path, family: str) -> dict[str, Any]:
    return {
        "algo": algo,
        "family": family,
        "run_id": run_id,
        "status": "failed",
        "return_code": -1,
        "runtime_seconds": float("nan"),
        "output_dir": str(algo_dir),
        "log_path": str(log_path),
        "dice": float("nan"),
        "cc": float("nan"),
        "mi": float("nan"),
        "nmi": float("nan"),
        "jacobian_min": float("nan"),
        "jacobian_p01": float("nan"),
        "jacobian_p99": float("nan"),
        "jacobian_negative_fraction": float("nan"),
        "warp_l2_energy_mean": float("nan"),
        "hard_gate_passed": None,
        "hard_gate_reasons": [],
        "hard_gate_warnings": [],
        "selected_init_method": "",
        "selected_attempt_name": "",
        "error_tail": "",
        "storyboard_status": "skipped",
        "storyboard_return_code": None,
        "storyboard_png": "",
        "storyboard_manifest": "",
        "storyboard_log": "",
        "storyboard_error_tail": "",
        "used_reuse_preproc": False,
        "used_reuse_linear": False,
        "upstream_source_dir": "",
        "task_index": 0,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Benchmark nonlinear ANTs transforms with fixed Rigid/Affine stages "
            "using template_register.py, then generate JSON/CSV/Markdown report."
        )
    )
    p.add_argument("--fixed", required=True, help="Fixed image (on reference grid).")
    p.add_argument("--moving", required=True, help="Moving image (on reference grid).")
    p.add_argument("--fixed-mask", required=True, help="Fixed brain mask.")
    p.add_argument("--moving-mask", required=True, help="Moving brain mask.")
    p.add_argument("--output-root", required=True, help="Experiment output root directory.")
    p.add_argument("--prefix", default="nlbench_", help="Output prefix for each registration run.")
    p.add_argument(
        "--algorithms",
        default=",".join(NONLINEAR_ALGOS_DEFAULT),
        help="Comma-separated nonlinear algorithms, e.g. syn,bspline_syn,bspline_displacement_field",
    )
    p.add_argument(
        "--nonlinear-transform-spec",
        default="",
        help="Custom transform spec string. Used only when algorithm list includes `custom`.",
    )
    p.add_argument("--threads", type=int, default=1, help="Threads passed to template_register.py.")
    p.add_argument(
        "--max-parallel",
        type=int,
        default=1,
        help="Maximum parallel registration jobs. Oversubscription guard may lower effective threads per job.",
    )
    p.add_argument("--random-seed", default="42", help="Fixed random seed for comparability.")
    p.add_argument("--ants-bin", default="C:/tools/ANTs/ants-2.6.5/bin")
    p.add_argument(
        "--affine-min-dice",
        type=float,
        default=0.55,
        help="Pass-through to template_register.py affine minimum Dice threshold.",
    )
    p.add_argument(
        "--affine-min-nmi",
        type=float,
        default=1.0,
        help="Pass-through to template_register.py affine minimum NMI threshold.",
    )
    p.add_argument(
        "--affine-min-cc",
        type=float,
        default=0.0,
        help="Pass-through to template_register.py affine minimum CC threshold.",
    )
    p.add_argument(
        "--coverage-margin-min-vox",
        type=int,
        default=0,
        help="Pass-through to template_register.py coverage hard gate (default 0 for comparative benchmark runs).",
    )
    p.add_argument(
        "--coverage-margin-min-mm",
        type=float,
        default=0.01,
        help="Pass-through to template_register.py coverage hard gate (default 0.01 mm for comparative benchmark runs).",
    )
    p.add_argument("--template-register-script", default="")
    p.add_argument("--generate-storyboard", default="on", choices=["on", "off"])
    p.add_argument("--storyboard-script", default="", help="Optional path to registration_storyboard.py.")
    p.add_argument("--storyboard-dpi", type=int, default=180)
    p.add_argument("--continue-on-fail", default="on", choices=["on", "off"])
    p.add_argument(
        "--resume",
        default="on",
        choices=["on", "off"],
        help="Resume from existing summary/output-root if present.",
    )
    p.add_argument(
        "--rerun-failed",
        default="on",
        choices=["on", "off"],
        help="When --resume=on, rerun previously failed algorithms (default on).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    fixed = str(Path(args.fixed).expanduser().resolve())
    moving = str(Path(args.moving).expanduser().resolve())
    fixed_mask = str(Path(args.fixed_mask).expanduser().resolve())
    moving_mask = str(Path(args.moving_mask).expanduser().resolve())

    if args.template_register_script.strip():
        template_register = Path(args.template_register_script).expanduser().resolve()
    else:
        template_register = Path(__file__).resolve().parent / "template_register.py"
    if not template_register.exists():
        raise FileNotFoundError(f"template_register.py not found: {template_register}")

    storyboard_script = None
    if args.generate_storyboard == "on":
        storyboard_script = _resolve_storyboard_script(args.storyboard_script)

    algos: list[str] = []
    seen: set[str] = set()
    for tok in args.algorithms.replace(";", ",").split(","):
        name = tok.strip().lower()
        if not name:
            continue
        if name in seen:
            continue
        seen.add(name)
        algos.append(name)
    if not algos:
        raise RuntimeError("No nonlinear algorithms parsed from --algorithms.")

    runtime = _resolve_parallel_settings(
        requested_threads=args.threads,
        requested_max_parallel=args.max_parallel,
        continue_on_fail=args.continue_on_fail,
    )

    summary_json_path = output_root / f"{args.prefix}nonlinear_benchmark_summary.json"
    state_json_path = output_root / f"{args.prefix}nonlinear_benchmark_state.json"
    previous_summary: dict[str, Any] = {}
    previous_rows_by_algo: dict[str, dict[str, Any]] = {}
    if args.resume == "on" and summary_json_path.exists():
        try:
            previous_summary = _read_json(summary_json_path)
            for row in list(previous_summary.get("records", []) or []):
                algo_name = str(row.get("algo", "")).strip().lower()
                if algo_name:
                    previous_rows_by_algo[algo_name] = row
        except Exception:
            previous_summary = {}
            previous_rows_by_algo = {}

    run_tag = str(previous_summary.get("run_tag", "")).strip() or _now_tag()
    records_by_algo: dict[str, dict[str, Any]] = {}
    pending_tasks: list[dict[str, Any]] = []

    def _ordered_records() -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for a in algos:
            row = records_by_algo.get(a)
            if row is not None:
                out.append(row)
        return out

    def _write_state(payload: dict[str, Any]) -> None:
        state_json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _write_partial_summary() -> None:
        partial = {
            "run_tag": run_tag,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "fixed": fixed,
            "moving": moving,
            "fixed_mask": fixed_mask,
            "moving_mask": moving_mask,
            "template_register_script": str(template_register),
            "random_seed": args.random_seed.strip(),
            "threads": runtime["threads_effective"],
            "threads_requested": runtime["threads_requested"],
            "max_parallel_requested": runtime["max_parallel_requested"],
            "max_parallel_effective": runtime["max_parallel_effective"],
            "cpu_count": runtime["cpu_count"],
            "oversubscribe_guard_applied": runtime["oversubscribe_guard_applied"],
            "affine_min_dice": float(args.affine_min_dice),
            "affine_min_nmi": float(args.affine_min_nmi),
            "affine_min_cc": float(args.affine_min_cc),
            "algorithms": algos,
            "records": _ordered_records(),
            "resume": args.resume,
            "rerun_failed": args.rerun_failed,
        }
        summary_json_path.write_text(json.dumps(partial, indent=2), encoding="utf-8")

    for idx, algo in enumerate(algos, start=1):
        algo_dir = output_root / f"{idx:02d}_{algo}"
        algo_dir.mkdir(parents=True, exist_ok=True)
        prev_row = previous_rows_by_algo.get(algo)
        if args.resume == "on" and prev_row is not None:
            prev_status = str(prev_row.get("status", "")).strip().lower()
            qc_exists = (algo_dir / f"{args.prefix}qc_metrics.json").exists()
            if prev_status == "success" and qc_exists:
                reused = dict(prev_row)
                reused["resume_action"] = "reused_success"
                reused["task_index"] = idx
                records_by_algo[algo] = reused
                _write_state(
                    {
                        "status": "resume_skip",
                        "updated_at": datetime.now().isoformat(timespec="seconds"),
                        "algorithm": algo,
                        "algorithm_index": idx,
                        "reason": "existing_success_row_and_qc_present",
                    }
                )
                _write_partial_summary()
                continue
            if prev_status != "success" and args.rerun_failed == "off":
                reused = dict(prev_row)
                reused["resume_action"] = "kept_previous_failure"
                reused["task_index"] = idx
                records_by_algo[algo] = reused
                _write_state(
                    {
                        "status": "resume_skip",
                        "updated_at": datetime.now().isoformat(timespec="seconds"),
                        "algorithm": algo,
                        "algorithm_index": idx,
                        "reason": "previous_failure_kept",
                    }
                )
                _write_partial_summary()
                continue

        run_id = f"RUN_NLBENCH_{run_tag}_{idx:02d}_{algo}"
        pending_tasks.append(
            {
                "idx": idx,
                "algo": algo,
                "algo_dir": algo_dir,
                "run_id": run_id,
            }
        )

    shared_upstream_source = _resolve_existing_upstream_source(
        algos=algos,
        records_by_algo=records_by_algo,
        prefix=args.prefix,
    )

    def _run_single_task(task: dict[str, Any], *, upstream_source: Path | None) -> dict[str, Any]:
        idx = int(task["idx"])
        algo = str(task["algo"])
        algo_dir = Path(task["algo_dir"])
        run_id = str(task["run_id"])
        log_path = algo_dir / f"{args.prefix}benchmark.log.txt"

        row = _build_row_template(
            algo=algo,
            run_id=run_id,
            algo_dir=algo_dir,
            log_path=log_path,
            family=NONLINEAR_FAMILY.get(algo, "unknown"),
        )
        row["task_index"] = idx

        cmd = [
            sys.executable,
            str(template_register),
            "--fixed",
            fixed,
            "--moving",
            moving,
            "--output-dir",
            str(algo_dir),
            "--prefix",
            args.prefix,
            "--run-id",
            run_id,
            "--trace-dir",
            str(algo_dir),
            "--fixed-mask",
            fixed_mask,
            "--moving-mask",
            moving_mask,
            "--preprocess-mode",
            "n4_locked",
            "--moving-denoise",
            "on",
            "--use-mask-in-optimization",
            "on",
            "--init-strategy",
            "com_only",
            "--affine-fallback",
            "on",
            "--nonlinear-transform",
            algo,
            "--coverage-margin-min-vox",
            str(max(args.coverage_margin_min_vox, 0)),
            "--coverage-margin-min-mm",
            str(max(args.coverage_margin_min_mm, 0.001)),
            "--ants-bin",
            args.ants_bin,
            "--affine-min-dice",
            str(float(args.affine_min_dice)),
            "--affine-min-nmi",
            str(float(args.affine_min_nmi)),
            "--affine-min-cc",
            str(float(args.affine_min_cc)),
            "--threads",
            str(runtime["threads_effective"]),
            "--random-seed",
            args.random_seed.strip(),
        ]

        if upstream_source is not None and _upstream_cache_ready(upstream_source, args.prefix):
            cmd.extend(["--reuse-preproc-dir", str(upstream_source / "preproc")])
            cmd.extend(["--reuse-linear-stages-dir", str(upstream_source / "stages")])
            row["used_reuse_preproc"] = True
            row["used_reuse_linear"] = True
            row["upstream_source_dir"] = str(upstream_source)

        if algo == "custom":
            if not args.nonlinear_transform_spec.strip():
                raise RuntimeError("Algorithm list includes `custom` but --nonlinear-transform-spec is empty.")
            cmd.extend(["--nonlinear-transform-spec", args.nonlinear_transform_spec.strip()])

        t0 = time.monotonic()
        done = subprocess.run(cmd, text=True, capture_output=True)
        elapsed = time.monotonic() - t0
        log_text = (
            "$ " + " ".join(cmd) + "\n"
            + f"[return_code] {done.returncode}\n"
            + "[stdout]\n" + (done.stdout or "") + "\n"
            + "[stderr]\n" + (done.stderr or "") + "\n"
        )
        log_path.write_text(log_text, encoding="utf-8")

        row["status"] = "success" if done.returncode == 0 else "failed"
        row["return_code"] = int(done.returncode)
        row["runtime_seconds"] = float(elapsed)

        row = _collect_row_from_outputs(
            row=row,
            algo_dir=algo_dir,
            run_id=run_id,
            prefix=args.prefix,
        )

        if done.returncode != 0:
            row["error_tail"] = _tail_text((done.stderr or "") + "\n" + (done.stdout or ""), lines=40)

        if storyboard_script is not None:
            row.update(
                _run_storyboard(
                    storyboard_script=storyboard_script,
                    run_dir=algo_dir,
                    prefix=args.prefix,
                    dpi=args.storyboard_dpi,
                )
            )

        row["command"] = cmd
        return row

    if pending_tasks and shared_upstream_source is None:
        bootstrap = pending_tasks.pop(0)
        _write_state(
            {
                "status": "bootstrap_running",
                "started_at": datetime.now().isoformat(timespec="seconds"),
                "algorithm": bootstrap["algo"],
                "algorithm_index": bootstrap["idx"],
                "algorithm_total": len(algos),
                "reason": "shared_upstream_source_missing",
            }
        )
        boot_row = _run_single_task(bootstrap, upstream_source=None)
        records_by_algo[str(bootstrap["algo"])] = boot_row
        _write_partial_summary()
        if boot_row.get("status") == "success":
            boot_dir = Path(str(boot_row.get("output_dir", "")))
            if _upstream_cache_ready(boot_dir, args.prefix):
                shared_upstream_source = boot_dir

        if args.continue_on_fail == "off" and boot_row.get("status") != "success":
            pending_tasks = []

    if pending_tasks:
        _write_state(
            {
                "status": "dispatching",
                "updated_at": datetime.now().isoformat(timespec="seconds"),
                "pending_count": len(pending_tasks),
                "max_parallel_effective": runtime["max_parallel_effective"],
                "threads_effective": runtime["threads_effective"],
                "shared_upstream_source": str(shared_upstream_source) if shared_upstream_source else "",
            }
        )

        if runtime["max_parallel_effective"] <= 1:
            for task in pending_tasks:
                row = _run_single_task(task, upstream_source=shared_upstream_source)
                records_by_algo[str(task["algo"])] = row
                _write_state(
                    {
                        "status": "algorithm_done",
                        "updated_at": datetime.now().isoformat(timespec="seconds"),
                        "algorithm": task["algo"],
                        "algorithm_index": task["idx"],
                        "algorithm_total": len(algos),
                        "run_id": task["run_id"],
                        "return_code": int(row.get("return_code", -1)),
                        "runtime_seconds": float(row.get("runtime_seconds", float("nan"))),
                        "row_status": row.get("status"),
                    }
                )
                _write_partial_summary()
                if args.continue_on_fail == "off" and row.get("status") != "success":
                    break
        else:
            with ThreadPoolExecutor(max_workers=runtime["max_parallel_effective"]) as ex:
                futures = {
                    ex.submit(_run_single_task, task, upstream_source=shared_upstream_source): task
                    for task in pending_tasks
                }
                for fut in as_completed(futures):
                    task = futures[fut]
                    try:
                        row = fut.result()
                    except Exception as exc:
                        row = _build_row_template(
                            algo=str(task["algo"]),
                            run_id=str(task["run_id"]),
                            algo_dir=Path(task["algo_dir"]),
                            log_path=Path(task["algo_dir"]) / f"{args.prefix}benchmark.log.txt",
                            family=NONLINEAR_FAMILY.get(str(task["algo"]), "unknown"),
                        )
                        row["status"] = "failed"
                        row["error_tail"] = f"parallel worker exception: {exc}"
                        row["task_index"] = int(task["idx"])
                    records_by_algo[str(task["algo"])] = row
                    _write_state(
                        {
                            "status": "algorithm_done",
                            "updated_at": datetime.now().isoformat(timespec="seconds"),
                            "algorithm": task["algo"],
                            "algorithm_index": task["idx"],
                            "algorithm_total": len(algos),
                            "run_id": task["run_id"],
                            "return_code": int(row.get("return_code", -1)),
                            "runtime_seconds": float(row.get("runtime_seconds", float("nan"))),
                            "row_status": row.get("status"),
                            "remaining_uncollected": sum(1 for x in futures if not x.done()),
                        }
                    )
                    _write_partial_summary()

    records = _ordered_records()
    successful = [r for r in records if r.get("status") == "success"]
    ranked = sorted(successful, key=_score_row, reverse=True)
    rank_map = {r["algo"]: i + 1 for i, r in enumerate(ranked)}
    for r in records:
        r["rank"] = int(rank_map.get(r["algo"], 0))

    summary = {
        "run_tag": run_tag,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "fixed": fixed,
        "moving": moving,
        "fixed_mask": fixed_mask,
        "moving_mask": moving_mask,
        "template_register_script": str(template_register),
        "random_seed": args.random_seed.strip(),
        "threads": runtime["threads_effective"],
        "threads_requested": runtime["threads_requested"],
        "max_parallel_requested": runtime["max_parallel_requested"],
        "max_parallel_effective": runtime["max_parallel_effective"],
        "cpu_count": runtime["cpu_count"],
        "oversubscribe_guard_applied": runtime["oversubscribe_guard_applied"],
        "shared_upstream_source": str(shared_upstream_source) if shared_upstream_source else "",
        "affine_min_dice": float(args.affine_min_dice),
        "affine_min_nmi": float(args.affine_min_nmi),
        "affine_min_cc": float(args.affine_min_cc),
        "algorithms": algos,
        "records": records,
        "resume": args.resume,
        "rerun_failed": args.rerun_failed,
    }

    json_path = output_root / f"{args.prefix}nonlinear_benchmark_summary.json"
    csv_path = output_root / f"{args.prefix}nonlinear_benchmark_summary.csv"
    md_path = output_root / f"{args.prefix}nonlinear_benchmark_report.md"

    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    csv_cols = [
        "rank",
        "algo",
        "family",
        "status",
        "return_code",
        "runtime_seconds",
        "dice",
        "cc",
        "mi",
        "nmi",
        "jacobian_min",
        "jacobian_p01",
        "jacobian_p99",
        "jacobian_negative_fraction",
        "warp_l2_energy_mean",
        "hard_gate_passed",
        "selected_init_method",
        "selected_attempt_name",
        "used_reuse_preproc",
        "used_reuse_linear",
        "upstream_source_dir",
        "storyboard_status",
        "storyboard_png",
        "output_dir",
        "log_path",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_cols)
        writer.writeheader()
        for r in records:
            writer.writerow({k: r.get(k, "") for k in csv_cols})

    md: list[str] = []
    md.append("# Nonlinear Registration Benchmark Report")
    md.append("")
    md.append(f"- generated_at: `{summary['generated_at']}`")
    md.append(f"- fixed: `{fixed}`")
    md.append(f"- moving: `{moving}`")
    md.append(f"- fixed_mask: `{fixed_mask}`")
    md.append(f"- moving_mask: `{moving_mask}`")
    md.append(f"- random_seed: `{args.random_seed.strip()}`")
    md.append(
        f"- threads_requested/effective: `{runtime['threads_requested']}/{runtime['threads_effective']}`"
    )
    md.append(
        f"- max_parallel_requested/effective: `{runtime['max_parallel_requested']}/{runtime['max_parallel_effective']}`"
    )
    md.append(f"- cpu_count: `{runtime['cpu_count']}`")
    md.append(f"- oversubscribe_guard_applied: `{runtime['oversubscribe_guard_applied']}`")
    md.append(f"- shared_upstream_source: `{summary.get('shared_upstream_source','')}`")
    md.append(f"- affine_min_dice: `{float(args.affine_min_dice)}`")
    md.append(f"- affine_min_nmi: `{float(args.affine_min_nmi)}`")
    md.append(f"- affine_min_cc: `{float(args.affine_min_cc)}`")
    md.append(
        "- control policy: `Rigid/Affine schedule unchanged, only nonlinear transform replaced`"
    )
    md.append("")
    md.append("## Results")
    md.append("")
    md.append(
        "| rank | algo | family | status | runtime(s) | Dice | CC | NMI | JacNegFrac | WarpL2Mean | reuse_preproc | reuse_linear | hard_gate |"
    )
    md.append("|---:|---|---|---|---:|---:|---:|---:|---:|---:|---|---|---|")
    for r in records:
        md.append(
            "| "
            + f"{r.get('rank', 0) or ''} | {r.get('algo', '')} | {r.get('family', '')} | "
            + f"{r.get('status', '')} | "
            + f"{_safe_float(r.get('runtime_seconds'), float('nan')):.2f} | "
            + f"{_safe_float(r.get('dice'), float('nan')):.6f} | "
            + f"{_safe_float(r.get('cc'), float('nan')):.6f} | "
            + f"{_safe_float(r.get('nmi'), float('nan')):.6f} | "
            + f"{_safe_float(r.get('jacobian_negative_fraction'), float('nan')):.6f} | "
            + f"{_safe_float(r.get('warp_l2_energy_mean'), float('nan')):.6f} | "
            + f"{bool(r.get('used_reuse_preproc', False))} | "
            + f"{bool(r.get('used_reuse_linear', False))} | "
            + f"{r.get('hard_gate_passed')} |"
        )

    md.append("")
    md.append("## Failures")
    md.append("")
    failed = [r for r in records if r.get("status") != "success"]
    if not failed:
        md.append("- none")
    else:
        for r in failed:
            md.append(f"- `{r.get('algo')}` return_code={r.get('return_code')}")
            tail = str(r.get("error_tail", "")).strip()
            if tail:
                md.append("```text")
                md.append(tail)
                md.append("```")

    md.append("")
    md.append("## Artifacts")
    md.append("")
    md.append(f"- summary_json: `{json_path}`")
    md.append(f"- summary_csv: `{csv_path}`")
    md.append(f"- output_root: `{output_root}`")
    md.append("")

    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")

    print(f"[OK] summary_json: {json_path}")
    print(f"[OK] summary_csv: {csv_path}")
    print(f"[OK] report_md: {md_path}")
    _write_state(
        {
            "status": "completed",
            "finished_at": datetime.now().isoformat(timespec="seconds"),
            "algorithms_total": len(algos),
            "successful_count": len(successful),
            "failed_count": len(records) - len(successful),
            "summary_json": str(json_path),
            "summary_csv": str(csv_path),
            "report_md": str(md_path),
            "shared_upstream_source": str(shared_upstream_source) if shared_upstream_source else "",
            "threads_effective": runtime["threads_effective"],
            "max_parallel_effective": runtime["max_parallel_effective"],
        }
    )


if __name__ == "__main__":
    main()

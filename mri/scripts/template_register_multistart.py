#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

DEFAULT_TIE_BREAK_DICE_EPS = 0.005


def now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _is_finite(value: float) -> bool:
    return isinstance(value, float) and math.isfinite(value)


def _seed_list(seed_text: str) -> list[int]:
    out: list[int] = []
    seen: set[int] = set()
    for token in seed_text.replace(";", ",").split(","):
        token = token.strip()
        if not token:
            continue
        try:
            seed = int(token)
        except ValueError:
            continue
        if seed in seen:
            continue
        out.append(seed)
        seen.add(seed)
    return out


def _ensure_default_seeds(seeds: list[int]) -> list[int]:
    defaults = [42, 17, 73]
    merged: list[int] = []
    seen: set[int] = set()
    for seed in defaults + seeds:
        if seed in seen:
            continue
        merged.append(seed)
        seen.add(seed)
    return merged[:3]


def _run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    print("$ " + " ".join(cmd), flush=True)
    done = subprocess.run(cmd, text=True, capture_output=True)
    if done.stdout:
        print(done.stdout.strip(), flush=True)
    if done.stderr:
        print(done.stderr.strip(), flush=True)
    return done


def _higher_better(value: Any) -> float:
    x = _safe_float(value, float("nan"))
    return x if _is_finite(x) else -1e30


def _lower_better(value: Any) -> float:
    x = _safe_float(value, float("nan"))
    return x if _is_finite(x) else 1e30


def _build_tie_break_metrics(
    *, seed: int, qc: dict[str, Any], attempt_gate: dict[str, Any], manifest: dict[str, Any]
) -> dict[str, Any]:
    jac = dict(attempt_gate.get("jacobian", {}))
    warp = dict(attempt_gate.get("warp_energy", {}))
    if not jac:
        jac = dict(manifest.get("jacobian_stats", {}))
    if not warp:
        warp = dict(manifest.get("warp_energy_stats", {}))
    return {
        "dice": _safe_float(qc.get("dice"), float("nan")),
        "jacobian_negative_fraction": _safe_float(jac.get("negative_fraction"), float("nan")),
        "warp_l2_energy_mean": _safe_float(warp.get("l2_energy_mean"), float("nan")),
        "cc": _safe_float(qc.get("cc"), float("nan")),
        "seed": int(seed),
    }


def _build_score_from_tie_break(metrics: dict[str, Any]) -> tuple[float, float, float, float, int]:
    # Legacy display field: higher is better in lexicographic order.
    dice = _higher_better(metrics.get("dice"))
    jac_neg = _lower_better(metrics.get("jacobian_negative_fraction"))
    warp_e = _lower_better(metrics.get("warp_l2_energy_mean"))
    cc = _higher_better(metrics.get("cc"))
    seed = int(metrics.get("seed", 0))
    return (dice, -jac_neg, -warp_e, cc, -seed)


def _is_seed_better(candidate: dict[str, Any], current: dict[str, Any], *, dice_eps: float) -> bool:
    cand_tb = dict(candidate.get("tie_break", {}))
    curr_tb = dict(current.get("tie_break", {}))

    cand_dice = _higher_better(cand_tb.get("dice"))
    curr_dice = _higher_better(curr_tb.get("dice"))
    if abs(cand_dice - curr_dice) >= dice_eps:
        return cand_dice > curr_dice

    cand_jneg = _lower_better(cand_tb.get("jacobian_negative_fraction"))
    curr_jneg = _lower_better(curr_tb.get("jacobian_negative_fraction"))
    if abs(cand_jneg - curr_jneg) > 1e-12:
        return cand_jneg < curr_jneg

    cand_warp = _lower_better(cand_tb.get("warp_l2_energy_mean"))
    curr_warp = _lower_better(curr_tb.get("warp_l2_energy_mean"))
    if abs(cand_warp - curr_warp) > 1e-12:
        return cand_warp < curr_warp

    cand_cc = _higher_better(cand_tb.get("cc"))
    curr_cc = _higher_better(curr_tb.get("cc"))
    if abs(cand_cc - curr_cc) > 1e-12:
        return cand_cc > curr_cc

    cand_seed = int(candidate.get("seed", 0))
    curr_seed = int(current.get("seed", 0))
    if cand_seed != curr_seed:
        return cand_seed < curr_seed

    cand_run = str(candidate.get("run_id", ""))
    curr_run = str(current.get("run_id", ""))
    if cand_run != curr_run:
        return cand_run < curr_run

    return False


def _select_best_seed(results: list[dict[str, Any]], *, dice_eps: float) -> dict[str, Any]:
    if not results:
        raise RuntimeError("Cannot select best seed from empty results.")
    best = results[0]
    for cand in results[1:]:
        if _is_seed_better(cand, best, dice_eps=dice_eps):
            best = cand
    return best


def _copy_if_exists(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run template registration with 3-seed multi-start and promote best QC run."
    )
    p.add_argument("--fixed", required=True)
    p.add_argument("--moving", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--prefix", default="template_reg_")
    p.add_argument("--from-space", default="moving_template")
    p.add_argument("--to-space", default="fixed_template")
    p.add_argument("--edge-id", default="")
    p.add_argument("--run-id", default="")
    p.add_argument("--trace-dir", default="")
    p.add_argument(
        "--fixed-mask",
        required=True,
        help="Required fixed-space brain mask. Multi-start registration fails without it.",
    )
    p.add_argument(
        "--moving-mask",
        required=True,
        help="Required moving-space brain mask. Multi-start registration fails without it.",
    )
    p.add_argument("--preprocess-mode", default="n4_locked", choices=["n4_locked", "n4"])
    p.add_argument(
        "--moving-denoise",
        default="on",
        choices=["on", "off"],
        help="Moving denoise switch before N4 (default on).",
    )
    p.add_argument(
        "--use-mask-in-optimization",
        default="on",
        choices=["on"],
        help="Registration is mask-gated; optimization always uses masks.",
    )
    p.add_argument(
        "--init-strategy",
        default="com_only",
        choices=["com_only", "translation_then_rigid"],
    )
    p.add_argument(
        "--nonlinear-transform",
        default="syn",
        choices=[
            "syn",
            "bspline_syn",
            "bspline_displacement_field",
            "gaussian_displacement_field",
            "time_varying_velocity_field",
            "time_varying_bspline_velocity_field",
            "exponential",
            "bspline_exponential",
            "custom",
        ],
        help="Nonlinear transform family passed through to template_register.py.",
    )
    p.add_argument(
        "--nonlinear-transform-spec",
        default="",
        help="Optional explicit ANTs transform spec string passed through to template_register.py.",
    )
    p.add_argument("--affine-fallback", default="on", choices=["on", "off"])
    p.add_argument("--affine-fallback-order", default="moments,geometry,antsai")
    p.add_argument("--affine-min-dice", type=float, default=0.55)
    p.add_argument("--affine-min-nmi", type=float, default=1.0)
    p.add_argument("--affine-min-cc", type=float, default=0.0)
    p.add_argument("--affine-det-min", type=float, default=0.25)
    p.add_argument("--affine-det-max", type=float, default=4.0)
    p.add_argument("--affine-sv-min", type=float, default=0.25)
    p.add_argument("--affine-sv-max", type=float, default=4.0)
    p.add_argument("--jac-min", type=float, default=0.05)
    p.add_argument("--jac-min-hard-gate", default="off", choices=["on", "off"])
    p.add_argument("--jac-mask-erosion-iters", type=int, default=1)
    p.add_argument("--jac-p01-min", type=float, default=0.20)
    p.add_argument("--jac-p99-max", type=float, default=5.0)
    p.add_argument("--jac-neg-frac-max", type=float, default=0.001)
    p.add_argument("--warp-l2-energy-max", type=float, default=50.0)
    p.add_argument("--warp-l2-hard-gate", default="off", choices=["on", "off"])
    p.add_argument("--coverage-margin-min-vox", type=int, default=10)
    p.add_argument("--coverage-margin-min-mm", type=float, default=2.5)
    p.add_argument(
        "--resample-interpolation",
        default="Linear",
        choices=[
            "Linear",
            "NearestNeighbor",
            "BSpline",
            "Gaussian",
            "CosineWindowedSinc",
            "WelchWindowedSinc",
            "HammingWindowedSinc",
            "LanczosWindowedSinc",
            "MultiLabel",
            "GenericLabel",
        ],
    )
    p.add_argument("--ants-bin", default="C:/tools/ANTs/ants-2.6.5/bin")
    p.add_argument("--threads", type=int, default=1, help="Threads per seed run.")
    p.add_argument(
        "--seed-list",
        default="42,17,73",
        help="Comma-separated seeds. Default enforces 3 runs and always includes 42.",
    )
    p.add_argument(
        "--tie-break-dice-eps",
        type=float,
        default=DEFAULT_TIE_BREAK_DICE_EPS,
        help=(
            "Deterministic seed tie-break epsilon on Dice. "
            "If |Dice_a-Dice_b| < eps, compare Jacobian negative fraction, "
            "warp L2 energy, CC, then seed."
        ),
    )
    p.add_argument(
        "--template-register-script",
        default="",
        help="Optional explicit path to template_register.py.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    multi_root = output_dir / "multistart_runs"
    multi_root.mkdir(parents=True, exist_ok=True)

    user_seeds = _seed_list(args.seed_list)
    seeds = _ensure_default_seeds(user_seeds)
    if len(seeds) < 3:
        raise RuntimeError("Failed to construct 3 seeds for multi-start.")
    if args.tie_break_dice_eps <= 0:
        raise RuntimeError("--tie-break-dice-eps must be > 0.")
    if args.jac_mask_erosion_iters < 0:
        raise RuntimeError("--jac-mask-erosion-iters must be >= 0.")
    if args.coverage_margin_min_vox < 0:
        raise RuntimeError("--coverage-margin-min-vox must be >= 0.")
    if args.coverage_margin_min_mm <= 0:
        raise RuntimeError("--coverage-margin-min-mm must be > 0.")

    this_script = Path(__file__).resolve()
    template_script = (
        Path(args.template_register_script).expanduser().resolve()
        if args.template_register_script.strip()
        else this_script.parent / "template_register.py"
    )
    if not template_script.exists():
        raise FileNotFoundError(f"template_register.py not found: {template_script}")

    base_run_id = args.run_id.strip() or f"RUN_MS_{now_tag()}"
    trace_root = (
        Path(args.trace_dir).expanduser().resolve()
        if args.trace_dir.strip()
        else output_dir
    )
    trace_root.mkdir(parents=True, exist_ok=True)

    fixed_mask_path = Path(args.fixed_mask).expanduser().resolve()
    moving_mask_path = Path(args.moving_mask).expanduser().resolve()
    if not fixed_mask_path.exists():
        raise FileNotFoundError(f"Fixed brain mask not found: {fixed_mask_path}")
    if not moving_mask_path.exists():
        raise FileNotFoundError(f"Moving brain mask not found: {moving_mask_path}")

    results: list[dict[str, Any]] = []
    for seed in seeds:
        seed_dir = multi_root / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        seed_run_id = f"{base_run_id}_S{seed}"
        seed_trace_dir = trace_root / "multistart_runs" / f"seed_{seed}"
        seed_trace_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable,
            str(template_script),
            "--fixed",
            args.fixed,
            "--moving",
            args.moving,
            "--output-dir",
            str(seed_dir),
            "--prefix",
            args.prefix,
            "--from-space",
            args.from_space,
            "--to-space",
            args.to_space,
            "--run-id",
            seed_run_id,
            "--trace-dir",
            str(seed_trace_dir),
            "--resample-interpolation",
            args.resample_interpolation,
            "--preprocess-mode",
            args.preprocess_mode,
            "--moving-denoise",
            args.moving_denoise,
            "--use-mask-in-optimization",
            args.use_mask_in_optimization,
            "--init-strategy",
            args.init_strategy,
            "--nonlinear-transform",
            args.nonlinear_transform,
            "--affine-fallback",
            args.affine_fallback,
            "--affine-fallback-order",
            args.affine_fallback_order,
            "--affine-min-dice",
            str(args.affine_min_dice),
            "--affine-min-nmi",
            str(args.affine_min_nmi),
            "--affine-min-cc",
            str(args.affine_min_cc),
            "--affine-det-min",
            str(args.affine_det_min),
            "--affine-det-max",
            str(args.affine_det_max),
            "--affine-sv-min",
            str(args.affine_sv_min),
            "--affine-sv-max",
            str(args.affine_sv_max),
            "--jac-min",
            str(args.jac_min),
            "--jac-min-hard-gate",
            args.jac_min_hard_gate,
            "--jac-mask-erosion-iters",
            str(args.jac_mask_erosion_iters),
            "--jac-p01-min",
            str(args.jac_p01_min),
            "--jac-p99-max",
            str(args.jac_p99_max),
            "--jac-neg-frac-max",
            str(args.jac_neg_frac_max),
            "--warp-l2-energy-max",
            str(args.warp_l2_energy_max),
            "--warp-l2-hard-gate",
            args.warp_l2_hard_gate,
            "--coverage-margin-min-vox",
            str(args.coverage_margin_min_vox),
            "--coverage-margin-min-mm",
            str(args.coverage_margin_min_mm),
            "--tie-break-dice-eps",
            str(args.tie_break_dice_eps),
            "--ants-bin",
            args.ants_bin,
            "--threads",
            str(max(args.threads, 1)),
            "--random-seed",
            str(seed),
        ]
        if args.nonlinear_transform_spec.strip():
            cmd.extend(["--nonlinear-transform-spec", args.nonlinear_transform_spec.strip()])
        if args.edge_id.strip():
            cmd.extend(["--edge-id", args.edge_id.strip()])
        cmd.extend(["--fixed-mask", str(fixed_mask_path)])
        cmd.extend(["--moving-mask", str(moving_mask_path)])

        done = _run(cmd)
        seed_result: dict[str, Any] = {
            "seed": seed,
            "status": "failed" if done.returncode != 0 else "success",
            "return_code": done.returncode,
            "seed_dir": str(seed_dir),
            "trace_dir": str(seed_trace_dir),
            "run_id": seed_run_id,
            "qc_metrics": {},
            "attempt_gate": {},
            "tie_break": {
                "dice": float("nan"),
                "jacobian_negative_fraction": float("nan"),
                "warp_l2_energy_mean": float("nan"),
                "cc": float("nan"),
                "seed": seed,
            },
            "score": (-1.0, -1.0, -1.0, -1.0, -seed),
        }

        qc_path = seed_dir / f"{args.prefix}qc_metrics.json"
        if done.returncode == 0 and qc_path.exists():
            qc = json.loads(qc_path.read_text(encoding="utf-8"))
            seed_result["qc_metrics"] = qc
            seed_result["qc_path"] = str(qc_path)

        # Best manifest from this seed trace dir (run_id already known).
        manifest_path = seed_trace_dir / f"run_manifest_{seed_run_id}.json"
        if manifest_path.exists():
            seed_result["manifest_path"] = str(manifest_path)
        else:
            candidates = sorted(seed_trace_dir.glob("run_manifest_*.json"))
            if candidates:
                seed_result["manifest_path"] = str(candidates[-1])
        manifest_for_seed = Path(seed_result["manifest_path"]) if seed_result.get("manifest_path") else None
        manifest: dict[str, Any] = {}
        if manifest_for_seed and manifest_for_seed.exists():
            try:
                manifest = json.loads(manifest_for_seed.read_text(encoding="utf-8"))
                seed_result["registration_stage_order"] = manifest.get("registration_stage_order", [])
                seed_result["stage_qc_summary_path"] = manifest.get("stage_qc_summary_path", "")
                seed_result["stages_dir"] = manifest.get("stages_dir", "")
                seed_result["storyboard_contract_note"] = manifest.get("storyboard_contract_note", "")
                seed_result["attempt_gate"] = (
                    dict(manifest.get("selected_init_attempt", {}).get("attempt_gate", {}))
                    if isinstance(manifest, dict)
                    else {}
                )
            except Exception:
                seed_result["registration_stage_order"] = []
                manifest = {}

        seed_result["tie_break"] = _build_tie_break_metrics(
            seed=seed,
            qc=dict(seed_result.get("qc_metrics", {})),
            attempt_gate=dict(seed_result.get("attempt_gate", {})),
            manifest=manifest,
        )
        seed_result["score"] = _build_score_from_tie_break(seed_result["tie_break"])
        results.append(seed_result)

    successes = [
        r
        for r in results
        if r["status"] == "success"
        and _is_finite(_safe_float(dict(r.get("tie_break", {})).get("dice"), float("nan")))
    ]
    if not successes:
        summary = {
            "status": "failed",
            "message": "No successful seed run with valid QC metrics.",
            "seeds": seeds,
            "results": results,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }
        (output_dir / f"{args.prefix}multistart_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        raise RuntimeError("All multi-start seed runs failed.")

    best = _select_best_seed(successes, dice_eps=args.tie_break_dice_eps)
    best_seed = int(best["seed"])
    best_dir = Path(best["seed_dir"])

    promoted_files = {
        "warped": f"{args.prefix}Warped.nii.gz",
        "warped_full": f"{args.prefix}WarpedFull.nii.gz",
        "inverse_warped": f"{args.prefix}InverseWarped.nii.gz",
        "deformation_grid": f"{args.prefix}deformationGrid.nii.gz",
        "deformation_grid_log": f"{args.prefix}deformationGrid.log.txt",
        "jacobian_det": f"{args.prefix}JacobianDet.nii.gz",
        "jacobian_log": f"{args.prefix}jacobian.log.txt",
        "jacobian_metrics": f"{args.prefix}jacobian_metrics.json",
        "affine": f"{args.prefix}0GenericAffine.mat",
        "warp_field": f"{args.prefix}1Warp.nii.gz",
        "inverse_warp_field": f"{args.prefix}1InverseWarp.nii.gz",
        "moving_mask_in_fixed": f"{args.prefix}movingMaskInFixed.nii.gz",
        "qc_metrics": f"{args.prefix}qc_metrics.json",
        "stage_qc_summary": f"{args.prefix}stage_qc_summary.json",
    }

    promoted: dict[str, str] = {}
    for key, name in promoted_files.items():
        src = best_dir / name
        dst = output_dir / name
        if _copy_if_exists(src, dst):
            promoted[key] = str(dst)

    best_trace_dir = Path(best["trace_dir"])
    _copy_if_exists(best_trace_dir / "transform_trace.jsonl", output_dir / "transform_trace.jsonl")
    if best.get("manifest_path"):
        manifest_src = Path(best["manifest_path"])
        _copy_if_exists(manifest_src, output_dir / f"run_manifest_{base_run_id}.json")
    best_stages_dir = best_dir / "stages"
    promoted_stages_dir = output_dir / "stages"
    if best_stages_dir.exists():
        if promoted_stages_dir.exists():
            shutil.rmtree(promoted_stages_dir)
        shutil.copytree(best_stages_dir, promoted_stages_dir)
        promoted["stages_dir"] = str(promoted_stages_dir)

    best_preproc_dir = best_dir / "preproc"
    promoted_preproc_dir = output_dir / "preproc"
    if best_preproc_dir.exists():
        if promoted_preproc_dir.exists():
            shutil.rmtree(promoted_preproc_dir)
        shutil.copytree(best_preproc_dir, promoted_preproc_dir)
        promoted["preproc_dir"] = str(promoted_preproc_dir)

    summary = {
        "status": "success",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "strategy": "multi_start_3seeds_tie_break",
        "selection_rule": (
            "dice_primary_with_eps_then_jacobian_negative_fraction_then_"
            "warp_l2_energy_then_cc_then_seed"
        ),
        "tie_break_dice_eps": args.tie_break_dice_eps,
        "threads_per_seed": max(args.threads, 1),
        "preprocess_mode": args.preprocess_mode,
        "moving_denoise": args.moving_denoise,
        "nonlinear_transform": args.nonlinear_transform,
        "nonlinear_transform_spec": args.nonlinear_transform_spec,
        "affine_fallback": {
            "enabled": args.affine_fallback,
            "order": args.affine_fallback_order,
            "min_dice": args.affine_min_dice,
            "min_nmi": args.affine_min_nmi,
            "min_cc": args.affine_min_cc,
            "det_min": args.affine_det_min,
            "det_max": args.affine_det_max,
            "sv_min": args.affine_sv_min,
            "sv_max": args.affine_sv_max,
            "jac_min": args.jac_min,
            "jac_min_hard_gate": args.jac_min_hard_gate,
            "jac_mask_erosion_iters": args.jac_mask_erosion_iters,
            "jac_p01_min": args.jac_p01_min,
            "jac_p99_max": args.jac_p99_max,
            "jac_neg_frac_max": args.jac_neg_frac_max,
            "warp_l2_energy_max": args.warp_l2_energy_max,
            "warp_l2_hard_gate": args.warp_l2_hard_gate,
        },
        "coverage_check": {
            "min_margin_vox": args.coverage_margin_min_vox,
            "min_margin_mm": args.coverage_margin_min_mm,
            "logic": "pass_if(min_margin_vox >= threshold_vox OR min_margin_mm >= threshold_mm)",
        },
        "seeds": seeds,
        "selected_seed": best_seed,
        "selected_score": best["score"],
        "selected_tie_break": best.get("tie_break", {}),
        "selected_qc_metrics": best.get("qc_metrics", {}),
        "selected_registration_stage_order": best.get("registration_stage_order", []),
        "results": results,
        "promoted_outputs": promoted,
        "output_dir": str(output_dir),
    }
    summary_json = output_dir / f"{args.prefix}multistart_summary.json"
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    summary_md = output_dir / f"{args.prefix}multistart_summary.md"
    md_lines = [
        "# Template Registration Multi-start Summary",
        "",
        f"- strategy: `multi_start_3seeds_tie_break`",
        (
            "- selection_rule: "
            "`Dice primary; if |Dice diff| < eps then lower Jacobian negative fraction, "
            "lower warp L2 energy, higher CC, smaller seed`"
        ),
        (
            "- gate_policy: "
            f"`jac_min_hard_gate={args.jac_min_hard_gate}, "
            f"warp_l2_hard_gate={args.warp_l2_hard_gate}, "
            f"jac_mask_erosion_iters={args.jac_mask_erosion_iters}`"
        ),
        f"- tie_break_dice_eps: `{args.tie_break_dice_eps}`",
        (
            "- coverage_hard_gate: "
            f"`min_margin_vox={args.coverage_margin_min_vox} OR "
            f"min_margin_mm={args.coverage_margin_min_mm}`"
        ),
        f"- selected_seed: `{best_seed}`",
        f"- threads_per_seed: `{max(args.threads, 1)}`",
        f"- nonlinear_transform: `{args.nonlinear_transform}`",
        f"- nonlinear_transform_spec: `{args.nonlinear_transform_spec or '(profile default)'}`",
        f"- summary_json: `{summary_json}`",
        "",
        "## Seed Results",
        "",
        "| seed | status | Dice | JacNegFrac | WarpL2Mean | CC | MI | NMI |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in results:
        qc = row.get("qc_metrics", {}) or {}
        tb = row.get("tie_break", {}) or {}
        md_lines.append(
            "| "
            + f"{row.get('seed')} | {row.get('status')} | "
            + f"{_safe_float(qc.get('dice'), float('nan')):.6f} | "
            + f"{_safe_float(tb.get('jacobian_negative_fraction'), float('nan')):.6f} | "
            + f"{_safe_float(tb.get('warp_l2_energy_mean'), float('nan')):.6f} | "
            + f"{_safe_float(qc.get('cc'), float('nan')):.6f} | "
            + f"{_safe_float(qc.get('mi'), float('nan')):.6f} | "
            + f"{_safe_float(qc.get('nmi'), float('nan')):.6f} |"
        )
    md_lines.append("")
    md_lines.append("## Promoted Outputs")
    md_lines.append("")
    for key, value in promoted.items():
        md_lines.append(f"- {key}: `{value}`")
    summary_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(f"[OK] selected seed: {best_seed}")
    print(f"[OK] summary: {summary_json}")
    print(f"[OK] promoted outputs in: {output_dir}")


if __name__ == "__main__":
    main()

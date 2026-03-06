#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


NONLINEAR_ALGOS_DEFAULT = ",".join(
    [
        "syn",
        "bspline_syn",
        "bspline_displacement_field",
        "gaussian_displacement_field",
        "time_varying_velocity_field",
        "time_varying_bspline_velocity_field",
        "exponential",
        "bspline_exponential",
    ]
)


def _run_cmd(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, text=True, capture_output=True, encoding="utf-8", errors="replace")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_float(v: Any) -> float:
    try:
        return float(v)
    except Exception:
        return float("nan")


def _powershell_json(script: str) -> Any:
    wrapped = (
        "[Console]::OutputEncoding=[System.Text.Encoding]::UTF8; "
        "$OutputEncoding=[System.Text.Encoding]::UTF8; "
        + script
    )
    done = _run_cmd(["powershell.exe", "-NoProfile", "-Command", wrapped])
    if done.returncode != 0:
        raise RuntimeError(f"powershell failed ({done.returncode}): {done.stderr.strip()}")
    text = (done.stdout or "").strip()
    if not text:
        return {}
    return json.loads(text)


def _query_reboot_audit(max_events: int = 80) -> dict[str, Any]:
    events_script = (
        "$ids=41,1074,6008,6005,6006; "
        f"Get-WinEvent -FilterHashtable @{{LogName='System'; Id=$ids}} -MaxEvents {max_events} | "
        "ForEach-Object { "
        "$proc=''; $reason=''; "
        "if ($_.Id -eq 1074 -and $_.Properties.Count -ge 4) { $proc=[string]$_.Properties[0].Value; $reason=[string]$_.Properties[3].Value }; "
        "[PSCustomObject]@{TimeCreated=$_.TimeCreated.ToString('yyyy-MM-dd HH:mm:ss'); Id=$_.Id; Provider=$_.ProviderName; Process=$proc; ReasonCode=$reason; Message=[string]$_.Message} "
        "} | ConvertTo-Json -Depth 5"
    )
    pending_script = (
        "$p=[PSCustomObject]@{ "
        "windows_update=(Test-Path 'HKLM:\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\WindowsUpdate\\Auto Update\\RebootRequired'); "
        "component_based_servicing=(Test-Path 'HKLM:\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\Component Based Servicing\\RebootPending'); "
        "pending_file_rename=((Get-ItemProperty -Path 'HKLM:\\SYSTEM\\CurrentControlSet\\Control\\Session Manager' -Name PendingFileRenameOperations -ErrorAction SilentlyContinue) -ne $null) "
        "}; $p | ConvertTo-Json -Depth 4"
    )
    events = _powershell_json(events_script)
    pending = _powershell_json(pending_script)
    if isinstance(events, dict):
        events = [events]
    latest_1074 = None
    for e in events:
        if int(e.get("Id", 0)) == 1074:
            latest_1074 = e
            break
    likely_cause = "unknown"
    if latest_1074:
        proc = str(latest_1074.get("Process", ""))
        if "MoUsoCoreWorker.exe" in proc:
            likely_cause = "windows_update_orchestrated_restart"
        elif proc:
            likely_cause = f"user_or_process_initiated_restart: {proc}"
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "pending_reboot": pending,
        "events": events,
        "latest_event_1074": latest_1074,
        "likely_cause": likely_cause,
    }


def _write_report(
    report_path: Path,
    audit: dict[str, Any],
    phase1_summary: dict[str, Any],
    phase2_summary: dict[str, Any],
) -> None:
    p1_rows = [dict(x) for x in list(phase1_summary.get("records", []) or [])]
    p2_rows = [dict(x) for x in list(phase2_summary.get("records", []) or [])]
    p1_ok = [r for r in p1_rows if str(r.get("status", "")).lower() == "success"]
    p2_ok = [r for r in p2_rows if str(r.get("status", "")).lower() == "success"]
    p1_ok = sorted(p1_ok, key=lambda x: int(x.get("rank", 0) or 999999))
    p2_ok = sorted(p2_ok, key=lambda x: int(x.get("rank", 0) or 999999))
    top_nonlinear = [str(r.get("algo", "")) for r in p1_ok[:5] if str(r.get("algo", "")).strip()]

    md: list[str] = []
    md.append("# Registration Experiment Comparison Report")
    md.append("")
    md.append(f"- generated_at: `{datetime.now().isoformat(timespec='seconds')}`")
    md.append(f"- reboot_likely_cause: `{audit.get('likely_cause', 'unknown')}`")
    md.append(f"- pending_reboot_flags: `{json.dumps(audit.get('pending_reboot', {}), ensure_ascii=False)}`")
    latest1074 = audit.get("latest_event_1074") or {}
    if latest1074:
        md.append(
            f"- latest_1074: time=`{latest1074.get('TimeCreated','')}` process=`{latest1074.get('Process','')}` reason_code=`{latest1074.get('ReasonCode','')}`"
        )
    md.append("")
    md.append("## Phase 1: Nonlinear Benchmark (Rigid/Affine fixed)")
    md.append("")
    md.append("| rank | nonlinear | status | runtime(s) | Dice | CC | NMI | JacNegFrac | WarpL2Mean |")
    md.append("|---:|---|---|---:|---:|---:|---:|---:|---:|")
    for r in p1_rows:
        md.append(
            "| "
            + f"{r.get('rank', '') or ''} | {r.get('algo', '')} | {r.get('status', '')} | "
            + f"{_safe_float(r.get('runtime_seconds')):.2f} | "
            + f"{_safe_float(r.get('dice')):.6f} | "
            + f"{_safe_float(r.get('cc')):.6f} | "
            + f"{_safe_float(r.get('nmi')):.6f} | "
            + f"{_safe_float(r.get('jacobian_negative_fraction')):.6f} | "
            + f"{_safe_float(r.get('warp_l2_energy_mean')):.6f} |"
        )
    md.append("")
    md.append(f"- selected_top5_for_phase2: `{','.join(top_nonlinear)}`")
    md.append("")
    md.append("## Phase 2: Linear Benchmark (Top5 nonlinear fixed per run)")
    md.append("")
    md.append("| rank | nonlinear | linear_profile | status | runtime(s) | Dice | CC | NMI | JacNegFrac | WarpL2Mean |")
    md.append("|---:|---|---|---|---:|---:|---:|---:|---:|---:|")
    for r in p2_rows:
        md.append(
            "| "
            + f"{r.get('rank', '') or ''} | {r.get('nonlinear_algo', '')} | {r.get('linear_profile', '')} | "
            + f"{r.get('status', '')} | "
            + f"{_safe_float(r.get('runtime_seconds')):.2f} | "
            + f"{_safe_float(r.get('dice')):.6f} | "
            + f"{_safe_float(r.get('cc')):.6f} | "
            + f"{_safe_float(r.get('nmi')):.6f} | "
            + f"{_safe_float(r.get('jacobian_negative_fraction')):.6f} | "
            + f"{_safe_float(r.get('warp_l2_energy_mean')):.6f} |"
        )
    md.append("")
    md.append("## Summary")
    md.append("")
    if p2_ok:
        best = p2_ok[0]
        md.append(
            f"- overall_best_combo: nonlinear=`{best.get('nonlinear_algo','')}` linear_profile=`{best.get('linear_profile','')}` "
            f"Dice=`{_safe_float(best.get('dice')):.6f}` CC=`{_safe_float(best.get('cc')):.6f}` "
            f"JacNegFrac=`{_safe_float(best.get('jacobian_negative_fraction')):.6f}`"
        )
    elif p1_ok:
        best = p1_ok[0]
        md.append(
            f"- phase2_has_no_success; best_phase1_nonlinear=`{best.get('algo','')}` "
            f"Dice=`{_safe_float(best.get('dice')):.6f}` CC=`{_safe_float(best.get('cc')):.6f}`"
        )
    else:
        md.append("- no successful runs found in phase1/phase2.")
    md.append("")
    md.append("## Artifacts")
    md.append("")
    md.append(f"- phase1_summary: `{phase1_summary.get('_path','')}`")
    md.append(f"- phase2_summary: `{phase2_summary.get('_path','')}`")
    md.append(f"- reboot_audit: `{audit.get('_path','')}`")
    md.append("")
    report_path.write_text("\n".join(md) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Supervise full registration comparison experiment with resume support: "
            "phase1 nonlinear benchmark + phase2 linear benchmark on top5 nonlinear results."
        )
    )
    p.add_argument("--fixed", required=True)
    p.add_argument("--moving", required=True)
    p.add_argument("--fixed-mask", required=True)
    p.add_argument("--moving-mask", required=True)
    p.add_argument("--output-root", required=True)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--phase1-max-parallel", type=int, default=1)
    p.add_argument("--phase2-max-parallel", type=int, default=1)
    p.add_argument("--random-seed", default="42")
    p.add_argument("--ants-bin", default="C:/tools/ANTs/ants-2.6.5/bin")
    p.add_argument("--affine-min-dice", type=float, default=0.55)
    p.add_argument("--affine-min-nmi", type=float, default=1.0)
    p.add_argument("--affine-min-cc", type=float, default=0.0)
    p.add_argument("--phase1-algorithms", default=NONLINEAR_ALGOS_DEFAULT)
    p.add_argument("--phase2-top-k", type=int, default=5)
    p.add_argument(
        "--phase2-linear-profiles",
        default="baseline,translation_init,no_fallback,dense_linear,fast_linear",
    )
    p.add_argument("--coverage-margin-min-vox", type=int, default=0)
    p.add_argument("--coverage-margin-min-mm", type=float, default=0.01)
    p.add_argument("--continue-on-fail", default="on", choices=["on", "off"])
    p.add_argument("--resume", default="on", choices=["on", "off"])
    p.add_argument("--rerun-failed", default="on", choices=["on", "off"])
    p.add_argument("--allow-pending-reboot", default="off", choices=["on", "off"])
    p.add_argument("--skip-phase2", default="off", choices=["on", "off"])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    scripts_dir = Path(__file__).resolve().parent
    phase1_script = scripts_dir / "registration_nonlinear_benchmark.py"
    phase2_script = scripts_dir / "registration_linear_benchmark.py"
    if not phase1_script.exists():
        raise FileNotFoundError(f"Missing phase1 script: {phase1_script}")
    if not phase2_script.exists():
        raise FileNotFoundError(f"Missing phase2 script: {phase2_script}")

    output_root = Path(args.output_root).expanduser().resolve()
    phase1_root = output_root / "phase1_nonlinear"
    phase2_root = output_root / "phase2_linear"
    output_root.mkdir(parents=True, exist_ok=True)
    phase1_root.mkdir(parents=True, exist_ok=True)
    phase2_root.mkdir(parents=True, exist_ok=True)

    state_path = output_root / "experiment_supervisor_state.json"
    audit_json = output_root / "reboot_audit.json"
    report_path = output_root / "experiment_comparison_report.md"

    def write_state(payload: dict[str, Any]) -> None:
        state_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    audit = _query_reboot_audit(max_events=80)
    audit["_path"] = str(audit_json)
    audit_json.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    pending = audit.get("pending_reboot", {}) or {}
    pending_true = any(bool(v) for v in pending.values())
    if pending_true and args.allow_pending_reboot == "off":
        write_state(
            {
                "status": "blocked_pending_reboot",
                "updated_at": datetime.now().isoformat(timespec="seconds"),
                "pending_reboot": pending,
                "audit_json": str(audit_json),
            }
        )
        raise RuntimeError(
            f"Pending reboot detected {pending}. Resolve reboot/update first or rerun with --allow-pending-reboot on."
        )

    phase1_cmd = [
        sys.executable,
        str(phase1_script),
        "--fixed",
        str(Path(args.fixed).expanduser().resolve()),
        "--moving",
        str(Path(args.moving).expanduser().resolve()),
        "--fixed-mask",
        str(Path(args.fixed_mask).expanduser().resolve()),
        "--moving-mask",
        str(Path(args.moving_mask).expanduser().resolve()),
        "--output-root",
        str(phase1_root),
        "--prefix",
        "phase1_",
        "--algorithms",
        args.phase1_algorithms,
        "--threads",
        str(max(args.threads, 1)),
        "--max-parallel",
        str(max(args.phase1_max_parallel, 1)),
        "--random-seed",
        args.random_seed.strip(),
        "--ants-bin",
        args.ants_bin,
        "--affine-min-dice",
        str(float(args.affine_min_dice)),
        "--affine-min-nmi",
        str(float(args.affine_min_nmi)),
        "--affine-min-cc",
        str(float(args.affine_min_cc)),
        "--coverage-margin-min-vox",
        str(max(args.coverage_margin_min_vox, 0)),
        "--coverage-margin-min-mm",
        str(max(args.coverage_margin_min_mm, 0.001)),
        "--continue-on-fail",
        args.continue_on_fail,
        "--resume",
        args.resume,
        "--rerun-failed",
        args.rerun_failed,
    ]
    write_state(
        {
            "status": "phase1_running",
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "phase1_cmd": phase1_cmd,
            "audit_json": str(audit_json),
        }
    )
    phase1_done = _run_cmd(phase1_cmd)
    (phase1_root / "supervisor_phase1.log.txt").write_text(
        "$ " + " ".join(phase1_cmd) + "\n"
        + f"[return_code] {phase1_done.returncode}\n"
        + "[stdout]\n" + (phase1_done.stdout or "") + "\n"
        + "[stderr]\n" + (phase1_done.stderr or "") + "\n",
        encoding="utf-8",
    )
    if phase1_done.returncode != 0:
        write_state(
            {
                "status": "phase1_failed",
                "updated_at": datetime.now().isoformat(timespec="seconds"),
                "return_code": phase1_done.returncode,
                "phase1_log": str(phase1_root / "supervisor_phase1.log.txt"),
            }
        )
        raise RuntimeError(f"Phase1 failed with return code {phase1_done.returncode}")

    phase1_summary_path = phase1_root / "phase1_nonlinear_benchmark_summary.json"
    if not phase1_summary_path.exists():
        raise FileNotFoundError(f"Phase1 summary missing: {phase1_summary_path}")

    phase2_summary_path = phase2_root / "phase2_linear_benchmark_summary.json"
    if args.skip_phase2 == "off":
        phase2_cmd = [
            sys.executable,
            str(phase2_script),
            "--fixed",
            str(Path(args.fixed).expanduser().resolve()),
            "--moving",
            str(Path(args.moving).expanduser().resolve()),
            "--fixed-mask",
            str(Path(args.fixed_mask).expanduser().resolve()),
            "--moving-mask",
            str(Path(args.moving_mask).expanduser().resolve()),
            "--output-root",
            str(phase2_root),
            "--prefix",
            "phase2_",
            "--nonlinear-summary",
            str(phase1_summary_path),
            "--top-k",
            str(max(args.phase2_top_k, 1)),
            "--linear-profiles",
            args.phase2_linear_profiles,
            "--threads",
            str(max(args.threads, 1)),
            "--max-parallel",
            str(max(args.phase2_max_parallel, 1)),
            "--random-seed",
            args.random_seed.strip(),
            "--ants-bin",
            args.ants_bin,
            "--affine-min-dice",
            str(float(args.affine_min_dice)),
            "--affine-min-nmi",
            str(float(args.affine_min_nmi)),
            "--affine-min-cc",
            str(float(args.affine_min_cc)),
            "--coverage-margin-min-vox",
            str(max(args.coverage_margin_min_vox, 0)),
            "--coverage-margin-min-mm",
            str(max(args.coverage_margin_min_mm, 0.001)),
            "--continue-on-fail",
            args.continue_on_fail,
            "--resume",
            args.resume,
            "--rerun-failed",
            args.rerun_failed,
        ]
        write_state(
            {
                "status": "phase2_running",
                "updated_at": datetime.now().isoformat(timespec="seconds"),
                "phase2_cmd": phase2_cmd,
                "phase1_summary": str(phase1_summary_path),
            }
        )
        phase2_done = _run_cmd(phase2_cmd)
        (phase2_root / "supervisor_phase2.log.txt").write_text(
            "$ " + " ".join(phase2_cmd) + "\n"
            + f"[return_code] {phase2_done.returncode}\n"
            + "[stdout]\n" + (phase2_done.stdout or "") + "\n"
            + "[stderr]\n" + (phase2_done.stderr or "") + "\n",
            encoding="utf-8",
        )
        if phase2_done.returncode != 0:
            write_state(
                {
                    "status": "phase2_failed",
                    "updated_at": datetime.now().isoformat(timespec="seconds"),
                    "return_code": phase2_done.returncode,
                    "phase2_log": str(phase2_root / "supervisor_phase2.log.txt"),
                }
            )
            raise RuntimeError(f"Phase2 failed with return code {phase2_done.returncode}")
        if not phase2_summary_path.exists():
            raise FileNotFoundError(f"Phase2 summary missing: {phase2_summary_path}")
    else:
        if not phase2_summary_path.exists():
            phase2_summary_path.write_text(
                json.dumps(
                    {
                        "run_tag": "",
                        "generated_at": datetime.now().isoformat(timespec="seconds"),
                        "records": [],
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

    p1 = _read_json(phase1_summary_path)
    p1["_path"] = str(phase1_summary_path)
    p2 = _read_json(phase2_summary_path)
    p2["_path"] = str(phase2_summary_path)
    _write_report(report_path, audit, p1, p2)

    write_state(
        {
            "status": "completed",
            "finished_at": datetime.now().isoformat(timespec="seconds"),
            "audit_json": str(audit_json),
            "phase1_summary": str(phase1_summary_path),
            "phase2_summary": str(phase2_summary_path),
            "final_report": str(report_path),
        }
    )
    print(f"[OK] audit_json: {audit_json}")
    print(f"[OK] phase1_summary: {phase1_summary_path}")
    print(f"[OK] phase2_summary: {phase2_summary_path}")
    print(f"[OK] final_report: {report_path}")


if __name__ == "__main__":
    main()

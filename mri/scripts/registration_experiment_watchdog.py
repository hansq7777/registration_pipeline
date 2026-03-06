#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _query_pending_reboot() -> dict[str, Any]:
    ps = (
        "[Console]::OutputEncoding=[System.Text.Encoding]::UTF8; "
        "$OutputEncoding=[System.Text.Encoding]::UTF8; "
        "$p=[PSCustomObject]@{ "
        "windows_update=(Test-Path 'HKLM:\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\WindowsUpdate\\Auto Update\\RebootRequired'); "
        "component_based_servicing=(Test-Path 'HKLM:\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\Component Based Servicing\\RebootPending'); "
        "pending_file_rename=((Get-ItemProperty -Path 'HKLM:\\SYSTEM\\CurrentControlSet\\Control\\Session Manager' -Name PendingFileRenameOperations -ErrorAction SilentlyContinue) -ne $null) "
        "}; $p | ConvertTo-Json -Depth 4"
    )
    done = subprocess.run(
        ["powershell.exe", "-NoProfile", "-Command", ps],
        text=True,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
    )
    if done.returncode != 0:
        return {
            "query_ok": False,
            "error": (done.stderr or done.stdout or "").strip(),
        }
    text = (done.stdout or "").strip()
    if not text:
        return {"query_ok": False, "error": "empty powershell output"}
    try:
        out = json.loads(text)
        out["query_ok"] = True
        return out
    except Exception:
        return {"query_ok": False, "error": f"bad json: {text[:300]}"}


def _query_latest_reboot_event() -> dict[str, Any]:
    ps = (
        "[Console]::OutputEncoding=[System.Text.Encoding]::UTF8; "
        "$OutputEncoding=[System.Text.Encoding]::UTF8; "
        "$ids=1074,6008,41; "
        "Get-WinEvent -FilterHashtable @{LogName='System'; Id=$ids} -MaxEvents 8 | "
        "ForEach-Object { "
        "  $proc=''; $reason=''; "
        "  if ($_.Id -eq 1074 -and $_.Properties.Count -ge 4) { "
        "    $proc=[string]$_.Properties[0].Value; "
        "    $reason=[string]$_.Properties[3].Value "
        "  }; "
        "  [PSCustomObject]@{ "
        "    TimeCreated=$_.TimeCreated.ToString('yyyy-MM-dd HH:mm:ss'); "
        "    Id=$_.Id; Provider=$_.ProviderName; Process=$proc; ReasonCode=$reason "
        "  } "
        "} | ConvertTo-Json -Depth 5"
    )
    done = subprocess.run(
        ["powershell.exe", "-NoProfile", "-Command", ps],
        text=True,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
    )
    if done.returncode != 0:
        return {"query_ok": False, "error": (done.stderr or "").strip()}
    text = (done.stdout or "").strip()
    if not text:
        return {"query_ok": False, "error": "empty event output"}
    try:
        rows = json.loads(text)
        if isinstance(rows, dict):
            rows = [rows]
        return {"query_ok": True, "events": rows}
    except Exception:
        return {"query_ok": False, "error": f"bad json: {text[:300]}"}


def _summary_stats(path: Path) -> dict[str, int]:
    d = _read_json(path)
    rows = list(d.get("records", []) or [])
    ok = sum(1 for r in rows if str(r.get("status", "")).lower() == "success")
    return {"records": len(rows), "success": ok, "failed": len(rows) - ok}


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Watchdog for registration experiment: periodically log progress and reboot-risk signals "
            "for resume/forensics."
        )
    )
    p.add_argument("--experiment-root", required=True)
    p.add_argument("--interval-sec", type=int, default=60)
    p.add_argument("--max-iterations", type=int, default=0, help="0 means run forever until stop-file exists.")
    p.add_argument("--stop-file", default="")
    p.add_argument("--heartbeat-file", default="watchdog_heartbeat.jsonl")
    p.add_argument("--alerts-file", default="watchdog_alerts.jsonl")
    args = p.parse_args()

    root = Path(args.experiment_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    heartbeat_path = root / args.heartbeat_file
    alerts_path = root / args.alerts_file
    stop_file = Path(args.stop_file).expanduser().resolve() if args.stop_file.strip() else (root / "watchdog.stop")

    supervisor_state = root / "experiment_supervisor_state.json"
    phase1_state = root / "phase1_nonlinear" / "phase1_nonlinear_benchmark_state.json"
    phase2_state = root / "phase2_linear" / "phase2_linear_benchmark_state.json"
    phase1_summary = root / "phase1_nonlinear" / "phase1_nonlinear_benchmark_summary.json"
    phase2_summary = root / "phase2_linear" / "phase2_linear_benchmark_summary.json"

    iteration = 0
    while True:
        iteration += 1
        now = _now_iso()
        sup = _read_json(supervisor_state)
        p1s = _read_json(phase1_state)
        p2s = _read_json(phase2_state)
        p1stats = _summary_stats(phase1_summary) if phase1_summary.exists() else {"records": 0, "success": 0, "failed": 0}
        p2stats = _summary_stats(phase2_summary) if phase2_summary.exists() else {"records": 0, "success": 0, "failed": 0}
        pending = _query_pending_reboot()
        events = _query_latest_reboot_event()

        payload = {
            "ts": now,
            "iteration": iteration,
            "supervisor_status": sup.get("status", ""),
            "phase1_state": {
                "status": p1s.get("status", ""),
                "algorithm": p1s.get("algorithm", ""),
                "algorithm_index": p1s.get("algorithm_index", 0),
                "algorithm_total": p1s.get("algorithm_total", 0),
                "updated_at": p1s.get("updated_at", p1s.get("started_at", "")),
            },
            "phase2_state": {
                "status": p2s.get("status", ""),
                "combo_key": p2s.get("combo_key", ""),
                "combo_index": p2s.get("combo_index", 0),
                "combo_total": p2s.get("combo_total", 0),
                "updated_at": p2s.get("updated_at", p2s.get("started_at", "")),
            },
            "phase1_progress": p1stats,
            "phase2_progress": p2stats,
            "pending_reboot": pending,
            "latest_reboot_events": (events.get("events", []) or [])[:3],
        }
        with heartbeat_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

        pending_flags = pending if isinstance(pending, dict) else {}
        pending_true = any(bool(pending_flags.get(k)) for k in ("windows_update", "component_based_servicing", "pending_file_rename"))
        if pending_true:
            with alerts_path.open("a", encoding="utf-8") as f:
                f.write(
                    json.dumps(
                        {
                            "ts": now,
                            "type": "pending_reboot_detected",
                            "pending_reboot": pending_flags,
                            "supervisor_status": sup.get("status", ""),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

        print(
            f"[watchdog] {now} sup={sup.get('status','')} "
            f"p1={p1stats['records']}({p1stats['success']} ok/{p1stats['failed']} fail) "
            f"p2={p2stats['records']}({p2stats['success']} ok/{p2stats['failed']} fail) "
            f"pending={pending_true}",
            flush=True,
        )

        if stop_file.exists():
            print(f"[watchdog] stop-file found: {stop_file}", flush=True)
            break
        if args.max_iterations > 0 and iteration >= args.max_iterations:
            break
        time.sleep(max(args.interval_sec, 5))


if __name__ == "__main__":
    main()

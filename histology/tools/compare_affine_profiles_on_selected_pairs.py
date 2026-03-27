from __future__ import annotations

import json
import os
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_HISTOLOGY_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_HISTOLOGY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_HISTOLOGY_ROOT))

from gui_mvp.hitl_gui.application.pair_registration import (  # noqa: E402
    PairRegistrationConfig,
    default_pair_registration_runs_root,
    find_ants_bin,
    run_pair_registration,
)
from gui_mvp.hitl_gui.application.pair_workspace import load_pair_registry  # noqa: E402


DEFAULT_MYELIN_ROOT = Path(r"/mnt/d/Research/Image Analysis/Nanozoomer scans/20250327 rat myelin quantification/Tissue&Masks")
DEFAULT_NISSL_ROOT = Path(r"/mnt/d/Research/Image Analysis/Nanozoomer scans/20250424 Nissl cytoarchitectonic counterpart/Tissue&Masks")
RUNS_ROOT = Path(r"/mnt/d/Research/Image Analysis/Nanozoomer scans/histology_pair_registration_runs")

CASES: list[dict[str, str]] = [
    {"pair_key": "2501_24__2501_25", "group": "1"},
    {"pair_key": "2501_24__2501_25", "group": "2"},
    {"pair_key": "2504_72__2504_73", "group": "1"},
    {"pair_key": "2504_72__2504_73", "group": "2"},
    {"pair_key": "2506_24__2506_25", "group": "1"},
    {"pair_key": "2506_24__2506_25", "group": "2"},
    {"pair_key": "2501_102__2501_103", "group": "all"},
    {"pair_key": "2501_114__2501_115", "group": "all"},
    {"pair_key": "2502_102__2502_79", "group": "all"},
    {"pair_key": "2504_108__2504_109", "group": "all"},
]


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_md(path: Path, payload: dict) -> None:
    lines = [
        f"# Affine Profile Compare {payload['experiment_id']}",
        "",
        f"- started_at_utc: `{payload['started_at_utc']}`",
        f"- completed_at_utc: `{payload.get('completed_at_utc', '')}`",
        f"- moving_side: `{payload['moving_side']}`",
        f"- fixed_side: `{payload['fixed_side']}`",
        f"- target_um_per_px: `{payload['target_um_per_px']}`",
        f"- registration_mask_mode: `{payload['registration_mask_mode']}`",
        f"- run_stages: `{','.join(payload['run_stages'])}`",
        f"- cases: `{len(payload['cases'])}`",
        "",
        "## Aggregate",
        "",
    ]
    agg = payload.get("aggregate", {})
    for prof, stats in agg.items():
        lines.append(f"### {prof}")
        for k, v in stats.items():
            lines.append(f"- {k}: `{v}`")
        lines.append("")
    lines.extend(["## Per Case", ""])
    for case in payload.get("results", []):
        lines.append(f"### {case['pair_key']} | group {case['group']}")
        for prof, rec in case["profiles"].items():
            if rec.get("status") != "ok":
                lines.append(f"- {prof}: failed `{rec.get('error')}`")
                continue
            aff = rec["manifest"]["stages"]["affine"]["metrics"]
            t = rec["manifest"]["timing_seconds"]
            lines.extend(
                [
                    f"- {prof}:",
                    f"  - total_s: `{t.get('total', 0.0):.2f}`",
                    f"  - affine_s: `{t.get('affine', 0.0):.2f}`",
                    f"  - Dice: `{aff.get('dice', 0.0):.4f}`",
                    f"  - HD95: `{aff.get('hd95_px', 0.0):.2f}`",
                    f"  - MI: `{aff.get('mi', 0.0):.4f}`",
                    f"  - CC: `{aff.get('cc', 0.0):.4f}`",
                    f"  - manifest: `{rec['manifest_path']}`",
                ]
            )
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _profile_stats(results: list[dict], profile: str) -> dict[str, float]:
    rows = [r["profiles"][profile] for r in results if r["profiles"][profile].get("status") == "ok"]
    if not rows:
        return {}
    def mean_of(path: tuple[str, ...]) -> float:
        vals = []
        for row in rows:
            cur = row
            for key in path:
                cur = cur[key]
            vals.append(float(cur))
        return float(statistics.mean(vals))
    return {
        "mean_total_s": round(mean_of(("manifest", "timing_seconds", "total")), 3),
        "mean_affine_s": round(mean_of(("manifest", "timing_seconds", "affine")), 3),
        "mean_affine_dice": round(mean_of(("manifest", "stages", "affine", "metrics", "dice")), 6),
        "mean_affine_hd95_px": round(mean_of(("manifest", "stages", "affine", "metrics", "hd95_px")), 6),
        "mean_affine_mi": round(mean_of(("manifest", "stages", "affine", "metrics", "mi")), 6),
        "mean_affine_cc": round(mean_of(("manifest", "stages", "affine", "metrics", "cc")), 6),
    }


def main() -> int:
    ants_bin = find_ants_bin()
    if ants_bin is None:
        raise RuntimeError("ANTs not found")
    common_root = Path(os.path.commonpath([str(DEFAULT_MYELIN_ROOT.resolve()), str(DEFAULT_NISSL_ROOT.resolve())]))
    registry = load_pair_registry(common_root / "histology_pair_qc_registry.json")
    runs_root = default_pair_registration_runs_root(DEFAULT_MYELIN_ROOT, DEFAULT_NISSL_ROOT)
    if runs_root is None:
        raise RuntimeError("Failed to resolve runs root")

    experiment_id = f"{_utc_stamp()}_affine_compare_selected10"
    out_json = RUNS_ROOT / f"{experiment_id}.json"
    out_md = RUNS_ROOT / f"{experiment_id}.md"
    payload: dict[str, object] = {
        "experiment_id": experiment_id,
        "started_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "moving_side": "myelin",
        "fixed_side": "nissl",
        "target_um_per_px": 10.0,
        "registration_mask_mode": "union",
        "run_stages": ["rigid", "affine"],
        "cases": CASES,
        "results": [],
        "aggregate": {},
    }
    _write_json(out_json, payload)

    profiles = ["current", "stronger"]
    t0 = time.perf_counter()
    for idx, case in enumerate(CASES, start=1):
        pair_key = case["pair_key"]
        group = case["group"]
        review = registry.get(pair_key)
        row = {"pair_key": pair_key, "group": group, "profiles": {}}
        print(f"[{idx}/{len(CASES)}] {pair_key} group={group}", flush=True)
        if not isinstance(review, dict):
            for profile in profiles:
                row["profiles"][profile] = {"status": "failed", "error": "missing review"}
            payload["results"].append(row)
            _write_json(out_json, payload)
            continue
        for profile in profiles:
            cfg = PairRegistrationConfig(
                pair_key=pair_key,
                moving_side="myelin",
                fixed_side="nissl",
                moving_group=group,
                fixed_group=group,
                review=review,
                common_root=common_root,
                myelin_root=DEFAULT_MYELIN_ROOT,
                nissl_root=DEFAULT_NISSL_ROOT,
                ants_bin=ants_bin,
                runs_root=runs_root,
                target_um_per_px=10.0,
                registration_mask_mode="union",
                run_stages=("rigid", "affine"),
                affine_profile=profile,
            )
            try:
                result = run_pair_registration(cfg)
                manifest = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))
                row["profiles"][profile] = {
                    "status": "ok",
                    "manifest_path": result["manifest_path"],
                    "storyboard_path": result["storyboard_path"],
                    "manifest": manifest,
                }
                print(
                    f"  {profile}: total={manifest['timing_seconds']['total']:.1f}s "
                    f"dice={manifest['stages']['affine']['metrics']['dice']:.4f} "
                    f"hd95={manifest['stages']['affine']['metrics']['hd95_px']:.2f}",
                    flush=True,
                )
            except Exception as exc:
                row["profiles"][profile] = {"status": "failed", "error": str(exc)}
                print(f"  {profile}: failed {exc}", flush=True)
        payload["results"].append(row)
        _write_json(out_json, payload)

    results = payload["results"]
    payload["aggregate"] = {profile: _profile_stats(results, profile) for profile in profiles}
    payload["completed_at_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    payload["wall_seconds"] = float(time.perf_counter() - t0)
    _write_json(out_json, payload)
    _write_md(out_md, payload)
    print(f"summary_json={out_json}")
    print(f"summary_md={out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

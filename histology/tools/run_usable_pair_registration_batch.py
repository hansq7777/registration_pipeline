from __future__ import annotations

import argparse
import json
import os
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


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_md(path: Path, payload: dict) -> None:
    lines = [
        f"# Usable Pair Registration Batch {payload['batch_id']}",
        "",
        f"- started_at_utc: `{payload['started_at_utc']}`",
        f"- completed_at_utc: `{payload.get('completed_at_utc', '')}`",
        f"- myelin_root: `{payload['myelin_root']}`",
        f"- nissl_root: `{payload['nissl_root']}`",
        f"- ants_bin: `{payload['ants_bin']}`",
        f"- moving_side: `{payload['moving_side']}`",
        f"- fixed_side: `{payload['fixed_side']}`",
        f"- moving_group: `{payload['moving_group']}`",
        f"- fixed_group: `{payload['fixed_group']}`",
        f"- registration_mask_mode: `{payload['registration_mask_mode']}`",
        f"- target_um_per_px: `{payload['target_um_per_px']}`",
        f"- run_stages: `{','.join(payload['run_stages'])}`",
        f"- usable_pairs_total: `{payload['usable_pairs_total']}`",
        f"- skipped_count: `{payload['skipped_count']}`",
        f"- success_count: `{payload['success_count']}`",
        f"- failure_count: `{payload['failure_count']}`",
        f"- wall_seconds: `{payload.get('wall_seconds', 0.0):.1f}`",
        "",
        "## Successes",
        "",
    ]
    for row in payload["successes"]:
        lines.extend(
            [
                f"- `{row['pair_key']}`",
                f"  - seconds: `{row['seconds']:.1f}`",
                f"  - manifest: `{row['manifest_path']}`",
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
    moving_side: str,
    fixed_side: str,
    moving_group: str,
    fixed_group: str,
    target_um_per_px: float,
    registration_mask_mode: str,
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
            str(obj.get("moving_side")) == moving_side
            and str(obj.get("fixed_side")) == fixed_side
            and str(obj.get("moving_group")) == moving_group
            and str(obj.get("fixed_group")) == fixed_group
            and float(obj.get("inputs", {}).get("target_um_per_px") or 0.0) == float(target_um_per_px)
            and str(obj.get("registration_mask_mode")) == registration_mask_mode
            and list(obj.get("run_stages") or []) == list(run_stages)
        ):
            return str(manifest_path)
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Batch-run rigid/affine registration for all usable histology pairs.")
    parser.add_argument("--myelin-root", type=Path, default=DEFAULT_MYELIN_ROOT)
    parser.add_argument("--nissl-root", type=Path, default=DEFAULT_NISSL_ROOT)
    parser.add_argument("--moving-side", choices=["myelin", "nissl"], default="myelin")
    parser.add_argument("--fixed-side", choices=["myelin", "nissl"], default="nissl")
    parser.add_argument("--moving-group", default="all")
    parser.add_argument("--fixed-group", default="all")
    parser.add_argument("--registration-mask-mode", default="union")
    parser.add_argument("--target-um-per-px", type=float, default=10.0)
    parser.add_argument("--stages", nargs="+", default=["rigid", "affine"])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--skip-finished", action="store_true")
    args = parser.parse_args()

    common_root = Path(os.path.commonpath([str(args.myelin_root.resolve()), str(args.nissl_root.resolve())]))
    registry_path = common_root / "histology_pair_qc_registry.json"
    runs_root = default_pair_registration_runs_root(args.myelin_root, args.nissl_root)
    if runs_root is None:
        raise RuntimeError("Failed to resolve runs root.")
    runs_root.mkdir(parents=True, exist_ok=True)

    ants_bin = find_ants_bin()
    if ants_bin is None:
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
        "ants_bin": str(ants_bin),
        "moving_side": args.moving_side,
        "fixed_side": args.fixed_side,
        "moving_group": args.moving_group,
        "fixed_group": args.fixed_group,
        "registration_mask_mode": args.registration_mask_mode,
        "target_um_per_px": float(args.target_um_per_px),
        "run_stages": list(args.stages),
        "usable_pairs_total": len(usable_items),
        "skipped_count": 0,
        "success_count": 0,
        "failure_count": 0,
        "skipped": [],
        "successes": [],
        "failures": [],
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
                moving_side=args.moving_side,
                fixed_side=args.fixed_side,
                moving_group=args.moving_group,
                fixed_group=args.fixed_group,
                target_um_per_px=float(args.target_um_per_px),
                registration_mask_mode=str(args.registration_mask_mode),
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
            ants_bin=ants_bin,
            runs_root=runs_root,
            target_um_per_px=float(args.target_um_per_px),
            registration_mask_mode=str(args.registration_mask_mode),
            run_stages=tuple(str(x).strip().lower() for x in args.stages),
        )
        try:
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
                }
            )
            print(f"  ok {seconds:.1f}s", flush=True)
        except Exception as exc:
            payload["failure_count"] = int(payload["failure_count"]) + 1
            payload["failures"].append({"pair_key": pair_key, "error": str(exc)})
            print(f"  failed: {exc}", flush=True)
        _write_json(summary_json, payload)

    payload["completed_at_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    payload["wall_seconds"] = float(time.perf_counter() - batch_t0)
    _write_json(summary_json, payload)
    _write_md(summary_md, payload)
    print(f"summary_json={summary_json}")
    print(f"summary_md={summary_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

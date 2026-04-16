#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from registration_pipeline.histology.gui_mvp.hitl_gui.application.confocal_registration import (  # noqa: E402
    refresh_step8_products_from_handoff,
    repair_exported_step7_session,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rebuild Step 8 stitched confocal/prediction products from a Step 7 export.")
    parser.add_argument("--handoff", type=Path, help="Path to step8_handoff.json")
    parser.add_argument("--export-dir", type=Path, help="Path to step7_session_export_* directory")
    parser.add_argument("--prediction-root", type=Path, help="Root directory containing tile-native nnUNet predictions")
    parser.add_argument("--summary-json", type=Path, help="Optional path for a JSON summary output")
    args = parser.parse_args()
    if args.handoff is None and args.export_dir is None:
        parser.error("provide either --handoff or --export-dir")
    return args


def main() -> int:
    args = _parse_args()
    prediction_root = None if args.prediction_root is None else Path(args.prediction_root)
    if args.handoff is not None:
        summary = refresh_step8_products_from_handoff(Path(args.handoff), prediction_root=prediction_root)
        export_dir = Path(str(summary.get("run_dir") or Path(args.handoff).parent))
    else:
        export_dir = Path(args.export_dir)
        summary = repair_exported_step7_session(export_dir, prediction_root=prediction_root)
    summary_path = Path(args.summary_json) if args.summary_json is not None else (export_dir / "step8_run_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

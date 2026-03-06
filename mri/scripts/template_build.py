#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, List

import nibabel as nib
import numpy as np


def run(cmd: List[str], env: dict[str, str] | None = None) -> None:
    print("$ " + " ".join(cmd), flush=True)
    completed = subprocess.run(cmd, text=True, capture_output=True, env=env)
    if completed.stdout:
        print(completed.stdout.strip(), flush=True)
    if completed.stderr:
        print(completed.stderr.strip(), flush=True)
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed ({completed.returncode}): {' '.join(cmd)}")


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def ensure_exec(ants_bin: Path, name: str) -> str:
    candidates = [
        ants_bin / name,
        ants_bin / f"{name}.exe",
    ]
    for exe in candidates:
        if exe.exists():
            return str(exe)
    raise FileNotFoundError(f"Missing executable: {candidates[-1]}")


def normalize_path(path_str: str) -> Path:
    raw = path_str.strip().strip('"')
    if os.name != "nt":
        m = re.match(r"^([A-Za-z]):[\\/](.*)$", raw)
        if m:
            drive = m.group(1).lower()
            tail = m.group(2).replace("\\", "/")
            return Path(f"/mnt/{drive}/{tail}").resolve()
    return Path(raw).expanduser().resolve()


def to_ants_path(path: Path, windows_mode: bool) -> str:
    if windows_mode:
        posix = path.as_posix()
        if posix.startswith("/mnt/"):
            parts = posix.split("/")
            if len(parts) >= 4:
                drive = parts[2].upper()
                tail = "/".join(parts[3:])
                return f"{drive}:/{tail}"
    return str(path)


def append_trace(trace_path: Path, event: dict[str, Any]) -> None:
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    with trace_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")


def sanitize_id(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_\\-]+", "_", text).strip("_")


def compute_mi_cc(fixed_path: Path, moving_path: Path, bins: int = 64) -> dict[str, float]:
    fixed = np.asarray(nib.load(str(fixed_path)).get_fdata(), dtype=np.float32)
    moving = np.asarray(nib.load(str(moving_path)).get_fdata(), dtype=np.float32)

    if fixed.shape != moving.shape:
        raise RuntimeError(f"QC shape mismatch: {fixed.shape} vs {moving.shape}")

    mask = np.isfinite(fixed) & np.isfinite(moving)
    if np.count_nonzero(mask) == 0:
        return {"cc": float("nan"), "mi": float("nan"), "nmi": float("nan")}

    x = fixed[mask].astype(np.float64)
    y = moving[mask].astype(np.float64)

    sx = np.std(x)
    sy = np.std(y)
    if sx <= 1e-12 or sy <= 1e-12:
        cc = float("nan")
    else:
        cc = float(np.corrcoef(x, y)[0, 1])

    x1, x99 = np.percentile(x, [1, 99])
    y1, y99 = np.percentile(y, [1, 99])
    x = np.clip(x, x1, x99)
    y = np.clip(y, y1, y99)

    h2d, _, _ = np.histogram2d(x, y, bins=bins)
    pxy = h2d / np.maximum(h2d.sum(), 1e-12)
    px = pxy.sum(axis=1)
    py = pxy.sum(axis=0)

    pxpy = np.outer(px, py)
    nz = pxy > 0
    mi = float(np.sum(pxy[nz] * np.log(np.maximum(pxy[nz], 1e-12) / np.maximum(pxpy[nz], 1e-12))))

    hx = -float(np.sum(px[px > 0] * np.log(px[px > 0])))
    hy = -float(np.sum(py[py > 0] * np.log(py[py > 0])))
    hxy = -float(np.sum(pxy[nz] * np.log(np.maximum(pxy[nz], 1e-12))))
    nmi = float((hx + hy) / np.maximum(hxy, 1e-12))

    return {"cc": cc, "mi": mi, "nmi": nmi}


def safe_mean(values: list[float]) -> float:
    valid = [v for v in values if np.isfinite(v)]
    if not valid:
        return float("nan")
    return float(np.mean(valid))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a small group template using ANTs pairwise registration + averaging."
    )
    parser.add_argument("--images", nargs="+", required=True, help="Input NIfTI images.")
    parser.add_argument("--output-dir", required=True, help="Output directory.")
    parser.add_argument("--template-name", default="template", help="Template base name.")
    parser.add_argument(
        "--from-space",
        default="subject_space",
        help="Logical source space name.",
    )
    parser.add_argument(
        "--to-space",
        default="group_template_space",
        help="Logical target space name.",
    )
    parser.add_argument("--run-id", default="", help="Optional run ID.")
    parser.add_argument("--trace-dir", default="", help="Optional trace output folder.")
    parser.add_argument(
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
        help="Resample interpolation for moving image output.",
    )
    parser.add_argument(
        "--ants-bin",
        default="C:/tools/ANTs/ants-2.6.5/bin",
        help="ANTs bin directory (contains antsRegistration.exe).",
    )
    parser.add_argument("--threads", type=int, default=4, help="ITK/OMP threads.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    images = [normalize_path(p) for p in args.images]
    if len(images) < 2:
        raise ValueError("Need at least 2 images to build a group template.")
    for img in images:
        if not img.exists():
            raise FileNotFoundError(f"Input image not found: {img}")

    output_dir = normalize_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    work_dir = output_dir / f"{args.template_name}_work"
    work_dir.mkdir(parents=True, exist_ok=True)

    trace_dir = normalize_path(args.trace_dir) if args.trace_dir.strip() else output_dir
    trace_dir.mkdir(parents=True, exist_ok=True)
    trace_path = trace_dir / "transform_trace.jsonl"

    run_id = args.run_id.strip() or datetime.now().strftime("RUN_%Y%m%d_%H%M%S")
    run_manifest_path = trace_dir / f"run_manifest_{run_id}.json"
    manifest_path = output_dir / f"{args.template_name}_manifest.json"

    run_manifest: dict[str, Any] = {
        "run_id": run_id,
        "status": "running",
        "step_name": "template_build",
        "started_at": now_iso(),
        "template_name": args.template_name,
        "from_space": args.from_space,
        "to_space": args.to_space,
        "trace_path": str(trace_path),
        "images": [str(p) for p in images],
        "resample_interpolation": args.resample_interpolation,
    }

    append_trace(
        trace_path,
        {
            "ts": now_iso(),
            "run_id": run_id,
            "step_name": "RUN_START",
            "status": "started",
            "workflow": "template_build",
            "from_space": args.from_space,
            "to_space": args.to_space,
            "n_images": len(images),
        },
    )

    try:
        ants_bin = normalize_path(args.ants_bin)
        ants_reg = ensure_exec(ants_bin, "antsRegistration")
        ants_apply = ensure_exec(ants_bin, "antsApplyTransforms")
        avg_img = ensure_exec(ants_bin, "AverageImages")
        windows_mode = ants_reg.lower().endswith(".exe")

        env = os.environ.copy()
        env["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = str(max(args.threads, 1))
        env["OMP_NUM_THREADS"] = str(max(args.threads, 1))

        fixed = images[0]
        fixed_ants = to_ants_path(fixed, windows_mode)
        warped_paths: list[Path] = []
        registrations: list[dict[str, Any]] = []
        edge_qc: list[dict[str, float]] = []

        anchor_copy = work_dir / "warped_000_anchor.nii.gz"
        shutil.copy2(fixed, anchor_copy)
        warped_paths.append(anchor_copy)

        for idx, moving in enumerate(images[1:], start=1):
            edge_id = sanitize_id(f"{args.from_space}_{idx:03d}_to_{args.to_space}_anchor")
            moving_ants = to_ants_path(moving, windows_mode)
            prefix = work_dir / f"img{idx:03d}_to_anchor_"
            prefix_ants = to_ants_path(prefix, windows_mode)
            warped = work_dir / f"img{idx:03d}_to_anchor_Warped.nii.gz"
            warped_ants = to_ants_path(warped, windows_mode)
            inverse = work_dir / f"img{idx:03d}_to_anchor_InverseWarped.nii.gz"
            inverse_ants = to_ants_path(inverse, windows_mode)

            append_trace(
                trace_path,
                {
                    "ts": now_iso(),
                    "run_id": run_id,
                    "step_name": "EDGE_STARTED",
                    "status": "started",
                    "edge_id": edge_id,
                    "from_space": args.from_space,
                    "to_space": args.to_space,
                    "fixed_image": str(fixed),
                    "moving_image": str(moving),
                },
            )

            cmd_reg = [
                ants_reg,
                "-d",
                "3",
                "--float",
                "0",
                "--interpolation",
                args.resample_interpolation,
                "--winsorize-image-intensities",
                "[0.005,0.995]",
                "--use-histogram-matching",
                "0",
                "-r",
                f"[{fixed_ants},{moving_ants},1]",
                "-m",
                f"MI[{fixed_ants},{moving_ants},1,32,Regular,0.25]",
                "-t",
                "Rigid[0.1]",
                "-c",
                "[100x50x20,1e-6,10]",
                "-s",
                "4x2x1vox",
                "-f",
                "4x2x1",
                "-m",
                f"MI[{fixed_ants},{moving_ants},1,32,Regular,0.25]",
                "-t",
                "Affine[0.1]",
                "-c",
                "[100x50x20,1e-6,10]",
                "-s",
                "4x2x1vox",
                "-f",
                "4x2x1",
                "-m",
                f"CC[{fixed_ants},{moving_ants},1,4]",
                "-t",
                "SyN[0.05,3,0]",
                "-c",
                "[60x30x10,1e-6,10]",
                "-s",
                "3x2x1vox",
                "-f",
                "4x2x1",
                "-o",
                f"[{prefix_ants},{warped_ants},{inverse_ants}]",
            ]
            run(cmd_reg, env=env)

            affine_mat = work_dir / f"img{idx:03d}_to_anchor_0GenericAffine.mat"
            warp_field = work_dir / f"img{idx:03d}_to_anchor_1Warp.nii.gz"
            inverse_warp = work_dir / f"img{idx:03d}_to_anchor_1InverseWarp.nii.gz"

            # Re-apply transforms explicitly for deterministic output.
            cmd_apply = [
                ants_apply,
                "-d",
                "3",
                "-i",
                moving_ants,
                "-r",
                fixed_ants,
                "-o",
                warped_ants,
                "-t",
                to_ants_path(warp_field, windows_mode),
                "-t",
                to_ants_path(affine_mat, windows_mode),
                "-n",
                args.resample_interpolation,
            ]
            run(cmd_apply, env=env)

            qc = compute_mi_cc(fixed, warped)
            edge_qc.append(qc)
            qc_json = work_dir / f"img{idx:03d}_qc_metrics.json"
            qc_json.write_text(json.dumps(qc, indent=2), encoding="utf-8")

            transform_files = [
                str(affine_mat),
                str(warp_field),
                str(inverse_warp),
            ]
            registration = {
                "edge_id": edge_id,
                "moving": str(moving),
                "fixed": str(fixed),
                "from_space": args.from_space,
                "to_space": args.to_space,
                "prefix": str(prefix),
                "warped": str(warped),
                "affine_mat": str(affine_mat),
                "warp_field": str(warp_field),
                "inverse_warp_field": str(inverse_warp),
                "transform_files": transform_files,
                "qc_metrics": qc,
                "qc_path": str(qc_json),
            }
            registrations.append(registration)
            warped_paths.append(warped)

            append_trace(
                trace_path,
                {
                    "ts": now_iso(),
                    "run_id": run_id,
                    "step_name": "EDGE_DONE",
                    "status": "success",
                    "edge_id": edge_id,
                    "from_space": args.from_space,
                    "to_space": args.to_space,
                    "transform_files": transform_files,
                    "interpolation": args.resample_interpolation,
                    "output_path": str(warped),
                    "qc_metrics": qc,
                },
            )

        template_path = output_dir / f"{args.template_name}.nii.gz"
        cmd_avg = [
            avg_img,
            "3",
            to_ants_path(template_path, windows_mode),
            "0",
            *[to_ants_path(p, windows_mode) for p in warped_paths],
        ]
        run(cmd_avg, env=env)

        template_vs_anchor_qc = compute_mi_cc(fixed, template_path)
        qc_summary = {
            "n_edges": len(edge_qc),
            "cc_mean": safe_mean([x.get("cc", float("nan")) for x in edge_qc]),
            "mi_mean": safe_mean([x.get("mi", float("nan")) for x in edge_qc]),
            "nmi_mean": safe_mean([x.get("nmi", float("nan")) for x in edge_qc]),
            "template_vs_anchor": template_vs_anchor_qc,
        }

        template_qc_path = output_dir / f"{args.template_name}_qc_metrics.json"
        template_qc_path.write_text(json.dumps(qc_summary, indent=2), encoding="utf-8")

        manifest = {
            "run_at": now_iso(),
            "run_id": run_id,
            "template_name": args.template_name,
            "template_path": str(template_path),
            "template_qc_path": str(template_qc_path),
            "fixed_anchor": str(fixed),
            "images": [str(p) for p in images],
            "warped_images": [str(p) for p in warped_paths],
            "registrations": registrations,
            "from_space": args.from_space,
            "to_space": args.to_space,
            "resample_interpolation": args.resample_interpolation,
            "trace_path": str(trace_path),
            "ants_bin": str(ants_bin),
            "threads": max(args.threads, 1),
            "qc_summary": qc_summary,
        }
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        run_manifest.update(
            {
                "status": "success",
                "finished_at": now_iso(),
                "output_dir": str(output_dir),
                "template_path": str(template_path),
                "template_manifest_path": str(manifest_path),
                "template_qc_path": str(template_qc_path),
                "registrations": registrations,
                "qc_summary": qc_summary,
            }
        )
        run_manifest_path.write_text(json.dumps(run_manifest, indent=2), encoding="utf-8")

        append_trace(
            trace_path,
            {
                "ts": now_iso(),
                "run_id": run_id,
                "step_name": "RUN_END",
                "status": "success",
                "run_manifest": str(run_manifest_path),
                "template_path": str(template_path),
                "template_qc_path": str(template_qc_path),
            },
        )

        print(f"[OK] template: {template_path}")
        print(f"[OK] manifest: {manifest_path}")
        print(f"[OK] qc: {template_qc_path}")
        print(f"[OK] trace: {trace_path}")
        print(f"[OK] run manifest: {run_manifest_path}")

    except Exception as exc:
        run_manifest.update(
            {
                "status": "failed",
                "finished_at": now_iso(),
                "error_message": str(exc),
            }
        )
        run_manifest_path.write_text(json.dumps(run_manifest, indent=2), encoding="utf-8")
        append_trace(
            trace_path,
            {
                "ts": now_iso(),
                "run_id": run_id,
                "step_name": "RUN_END",
                "status": "fail",
                "error_message": str(exc),
                "run_manifest": str(run_manifest_path),
            },
        )
        raise


if __name__ == "__main__":
    main()

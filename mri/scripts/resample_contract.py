#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import nibabel as nib
import numpy as np


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def run(cmd: list[str], env: dict[str, str] | None = None) -> None:
    print("$ " + " ".join(cmd), flush=True)
    completed = subprocess.run(cmd, text=True, capture_output=True, env=env)
    if completed.stdout:
        print(completed.stdout.strip(), flush=True)
    if completed.stderr:
        print(completed.stderr.strip(), flush=True)
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed ({completed.returncode}): {' '.join(cmd)}")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


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


def ensure_exec(ants_bin: Path, name: str) -> str:
    candidates = [ants_bin / name, ants_bin / f"{name}.exe"]
    for exe in candidates:
        if exe.exists():
            return str(exe)
    raise FileNotFoundError(f"Missing executable: {candidates[-1]}")


def header_snapshot(img: nib.Nifti1Image) -> dict[str, Any]:
    h = img.header
    spatial_unit, time_unit = h.get_xyzt_units()
    try:
        axcodes_raw = nib.aff2axcodes(img.affine)
    except Exception:
        axcodes_raw = (None, None, None)
    axcodes = [str(x) if x is not None else "?" for x in axcodes_raw]
    det = float(np.linalg.det(np.asarray(img.affine[:3, :3], dtype=np.float64)))
    return {
        "dtype": str(h.get_data_dtype()),
        "shape": [int(x) for x in img.shape[:3]],
        "zooms": [float(x) for x in h.get_zooms()[:3]],
        "qform_code": int(h["qform_code"]),
        "sform_code": int(h["sform_code"]),
        "xyzt_units": int(h["xyzt_units"]),
        "spatial_unit": spatial_unit or "unknown",
        "time_unit": time_unit or "unknown",
        "axcodes": axcodes,
        "affine_det_3x3": det,
        "pixdim_4_7": [float(x) for x in h["pixdim"][4:8]],
    }


def canonicalize_reference_grid(reference_path: Path) -> dict[str, Any]:
    ref = nib.load(str(reference_path))
    arr = np.asarray(ref.get_fdata(), dtype=np.float32)
    affine = ref.affine.copy()
    hdr = ref.header.copy()
    hdr.set_data_dtype(np.float32)
    hdr.set_xyzt_units("mm", "sec")
    hdr["pixdim"][1:4] = np.asarray(ref.header.get_zooms()[:3], dtype=np.float32)
    hdr["pixdim"][4:8] = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)

    out = nib.Nifti1Image(arr, affine, hdr)
    out.set_qform(affine, code=1)
    out.set_sform(affine, code=1)
    nib.save(out, str(reference_path))

    return {
        "path": str(reference_path),
        "header": header_snapshot(nib.load(str(reference_path))),
    }


def world_bounds(img: nib.Nifti1Image) -> tuple[np.ndarray, np.ndarray]:
    shape = np.asarray(img.shape[:3], dtype=np.float64)
    corners_idx = np.array(
        [[i, j, k] for i in (0, shape[0] - 1) for j in (0, shape[1] - 1) for k in (0, shape[2] - 1)],
        dtype=np.float64,
    )
    homo = np.concatenate([corners_idx, np.ones((8, 1), dtype=np.float64)], axis=1)
    xyz = (img.affine @ homo.T).T[:, :3]
    return xyz.min(axis=0), xyz.max(axis=0)


def parse_spacing(spacing_text: str | None, img_a: nib.Nifti1Image, img_b: nib.Nifti1Image) -> np.ndarray:
    if spacing_text and spacing_text.strip():
        txt = spacing_text.strip().replace("x", ",")
        vals = [float(x) for x in txt.split(",") if x.strip()]
        if len(vals) == 1:
            return np.array([vals[0], vals[0], vals[0]], dtype=np.float64)
        if len(vals) == 3:
            return np.array(vals, dtype=np.float64)
        raise ValueError("--spacing must be scalar or three comma-separated values, e.g. 0.2 or 0.2,0.2,0.2")

    s1 = np.asarray(img_a.header.get_zooms()[:3], dtype=np.float64)
    s2 = np.asarray(img_b.header.get_zooms()[:3], dtype=np.float64)
    return np.minimum(s1, s2)


def build_union_reference_grid(
    moving_path: Path,
    fixed_path: Path,
    out_path: Path,
    spacing_text: str | None,
) -> dict[str, Any]:
    moving_img = nib.load(str(moving_path))
    fixed_img = nib.load(str(fixed_path))

    spacing = parse_spacing(spacing_text, moving_img, fixed_img)

    mn1, mx1 = world_bounds(moving_img)
    mn2, mx2 = world_bounds(fixed_img)
    xyz_min = np.minimum(mn1, mn2)
    xyz_max = np.maximum(mx1, mx2)

    # Inclusive grid size on voxel centers.
    size = xyz_max - xyz_min
    shape = np.floor(size / spacing + 0.5).astype(np.int64) + 1
    shape = np.maximum(shape, 1)

    affine = np.eye(4, dtype=np.float64)
    affine[0, 0] = spacing[0]
    affine[1, 1] = spacing[1]
    affine[2, 2] = spacing[2]
    affine[:3, 3] = xyz_min

    hdr = nib.Nifti1Header()
    hdr.set_data_dtype(np.float32)
    hdr.set_xyzt_units("mm", "sec")
    hdr["pixdim"][1:4] = spacing.astype(np.float32)
    hdr["pixdim"][4:8] = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)

    ref = nib.Nifti1Image(np.zeros(shape.tolist(), dtype=np.float32), affine, hdr)
    ref.set_qform(affine, code=1)
    ref.set_sform(affine, code=1)
    nib.save(ref, str(out_path))

    return {
        "path": str(out_path),
        "shape": [int(x) for x in shape.tolist()],
        "spacing": [float(x) for x in spacing.tolist()],
        "xyz_min": [float(x) for x in xyz_min.tolist()],
        "xyz_max": [float(x) for x in xyz_max.tolist()],
        "source": "built_union_grid",
    }


def save_binary_mask_from_labels(labels_path: Path, out_mask_path: Path) -> None:
    labels = nib.load(str(labels_path))
    data = np.asarray(labels.get_fdata(), dtype=np.float32)
    mask = (data > 0).astype(np.uint8)
    hdr = labels.header.copy()
    hdr.set_data_dtype(np.uint8)
    out = nib.Nifti1Image(mask, labels.affine, hdr)
    out.set_qform(*labels.get_qform(coded=True))
    out.set_sform(*labels.get_sform(coded=True))
    nib.save(out, str(out_mask_path))


def expected_np_dtype(dtype_name: str) -> np.dtype:
    return np.dtype(dtype_name)


def canonicalize_to_reference(out_path: Path, reference_path: Path, dtype_name: str) -> None:
    out_img = nib.load(str(out_path))
    ref_img = nib.load(str(reference_path))

    dtype = expected_np_dtype(dtype_name)

    data = out_img.get_fdata()
    if np.issubdtype(dtype, np.integer):
        data = np.rint(data).astype(dtype)
    else:
        data = data.astype(dtype)

    ref_hdr = ref_img.header.copy()
    ref_hdr.set_data_dtype(dtype)
    ref_hdr["pixdim"][4:8] = ref_img.header["pixdim"][4:8]

    canonical = nib.Nifti1Image(data, ref_img.affine, ref_hdr)
    qf, qcode = ref_img.get_qform(coded=True)
    sf, scode = ref_img.get_sform(coded=True)
    canonical.set_qform(qf if qf is not None else ref_img.affine, code=int(qcode))
    canonical.set_sform(sf if sf is not None else ref_img.affine, code=int(scode))
    nib.save(canonical, str(out_path))


def check_contract(out_path: Path, reference_path: Path, dtype_name: str) -> dict[str, Any]:
    out_img = nib.load(str(out_path))
    ref_img = nib.load(str(reference_path))
    out_h = out_img.header
    ref_h = ref_img.header

    errors: list[str] = []
    if tuple(out_img.shape[:3]) != tuple(ref_img.shape[:3]):
        errors.append(f"shape mismatch: out={out_img.shape[:3]} ref={ref_img.shape[:3]}")

    if not np.allclose(out_img.affine, ref_img.affine, atol=1e-6):
        errors.append("affine mismatch")

    if int(out_h["qform_code"]) != int(ref_h["qform_code"]):
        errors.append(f"qform_code mismatch: out={int(out_h['qform_code'])} ref={int(ref_h['qform_code'])}")

    if int(out_h["sform_code"]) != int(ref_h["sform_code"]):
        errors.append(f"sform_code mismatch: out={int(out_h['sform_code'])} ref={int(ref_h['sform_code'])}")

    if int(out_h["xyzt_units"]) != int(ref_h["xyzt_units"]):
        errors.append(f"xyzt_units mismatch: out={int(out_h['xyzt_units'])} ref={int(ref_h['xyzt_units'])}")

    if not np.allclose(out_h["pixdim"][1:4], ref_h["pixdim"][1:4], atol=1e-6):
        errors.append("pixdim[1:4] mismatch")

    if not np.allclose(out_h["pixdim"][4:8], ref_h["pixdim"][4:8], atol=1e-6):
        errors.append("pixdim[4:8] mismatch")

    expected_dtype = expected_np_dtype(dtype_name)
    if out_h.get_data_dtype() != expected_dtype:
        errors.append(f"dtype mismatch: out={out_h.get_data_dtype()} expected={expected_dtype}")

    arr = out_img.get_fdata()
    nz = int(np.count_nonzero(arr))

    return {
        "pass": len(errors) == 0,
        "errors": errors,
        "path": str(out_path),
        "sha256": sha256_file(out_path),
        "header": header_snapshot(out_img),
        "nonzero_voxels": nz,
    }


def nii_stem(path: Path) -> str:
    name = path.name
    if name.endswith(".nii.gz"):
        return name[:-7]
    return path.stem


def inspect_header_semantics(path: Path) -> dict[str, Any]:
    img = nib.load(str(path))
    h = img.header
    snapshot = header_snapshot(img)
    issues: list[dict[str, Any]] = []

    spatial_unit = snapshot.get("spatial_unit", "unknown")
    if spatial_unit != "mm":
        issues.append(
            {
                "check": "units",
                "severity": "warning",
                "auto_fixable": True,
                "message": f"spatial unit is {spatial_unit}, expected mm",
                "reason": "registration spacing semantics should be mm",
            }
        )

    qf, qcode = img.get_qform(coded=True)
    sf, scode = img.get_sform(coded=True)

    if qf is None or int(qcode) <= 0:
        issues.append(
            {
                "check": "qform",
                "severity": "warning",
                "auto_fixable": True,
                "message": f"qform missing or qform_code<=0 (qform_code={int(qcode)})",
                "reason": "ants/nifti spatial transform should have valid qform",
            }
        )
    elif not np.allclose(qf, img.affine, atol=1e-5):
        issues.append(
            {
                "check": "qform",
                "severity": "warning",
                "auto_fixable": True,
                "message": "qform matrix differs from image affine",
                "reason": "qform and affine should be coherent for deterministic IO",
            }
        )

    if sf is None or int(scode) <= 0:
        issues.append(
            {
                "check": "sform",
                "severity": "warning",
                "auto_fixable": True,
                "message": f"sform missing or sform_code<=0 (sform_code={int(scode)})",
                "reason": "ants/nifti spatial transform should have valid sform",
            }
        )
    elif not np.allclose(sf, img.affine, atol=1e-5):
        issues.append(
            {
                "check": "sform",
                "severity": "warning",
                "auto_fixable": True,
                "message": "sform matrix differs from image affine",
                "reason": "sform and affine should be coherent for deterministic IO",
            }
        )

    axcodes = snapshot.get("axcodes", ["?", "?", "?"])
    if any(code == "?" for code in axcodes):
        issues.append(
            {
                "check": "axis_semantics",
                "severity": "error",
                "auto_fixable": False,
                "message": f"invalid axis semantics from affine: {axcodes}",
                "reason": "cannot infer reliable axis directions from affine",
            }
        )

    det = float(snapshot.get("affine_det_3x3", 0.0))
    if not np.isfinite(det) or abs(det) < 1e-8:
        issues.append(
            {
                "check": "affine_linear",
                "severity": "error",
                "auto_fixable": False,
                "message": f"affine 3x3 determinant invalid ({det})",
                "reason": "degenerate affine breaks spatial semantics",
            }
        )

    return {
        "path": str(path),
        "header": snapshot,
        "issues": issues,
        "pass": len(issues) == 0,
    }


def apply_header_autofix(input_path: Path, output_path: Path) -> dict[str, Any]:
    img = nib.load(str(input_path))
    h = img.header
    affine = img.affine.copy()
    hdr = h.copy()

    before = header_snapshot(img)
    fixes: list[dict[str, Any]] = []

    spatial_unit, time_unit = hdr.get_xyzt_units()
    if spatial_unit != "mm":
        new_time_unit = time_unit or "sec"
        old_units = {"spatial_unit": spatial_unit or "unknown", "time_unit": time_unit or "unknown"}
        hdr.set_xyzt_units("mm", new_time_unit)
        fixes.append(
            {
                "field": "xyzt_units",
                "old": old_units,
                "new": {"spatial_unit": "mm", "time_unit": new_time_unit},
                "reason": "registration contract requires mm spatial unit",
            }
        )

    qf, qcode = img.get_qform(coded=True)
    sf, scode = img.get_sform(coded=True)
    need_qform = qf is None or int(qcode) <= 0 or not np.allclose(qf, affine, atol=1e-5)
    need_sform = sf is None or int(scode) <= 0 or not np.allclose(sf, affine, atol=1e-5)

    if need_qform:
        fixes.append(
            {
                "field": "qform",
                "old": {"qform_code": int(qcode)},
                "new": {"qform_code": 1, "qform_source": "image_affine"},
                "reason": "normalize qform to affine with valid code",
            }
        )

    if need_sform:
        fixes.append(
            {
                "field": "sform",
                "old": {"sform_code": int(scode)},
                "new": {"sform_code": 1, "sform_source": "image_affine"},
                "reason": "normalize sform to affine with valid code",
            }
        )

    if not fixes:
        return {
            "applied": False,
            "input_path": str(input_path),
            "output_path": str(input_path),
            "before_header": before,
            "after_header": before,
            "fixes": [],
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = np.asarray(img.dataobj)
    out = nib.Nifti1Image(data, affine, hdr)
    if need_qform:
        out.set_qform(affine, code=1)
    else:
        out.set_qform(qf if qf is not None else affine, code=int(qcode))
    if need_sform:
        out.set_sform(affine, code=1)
    else:
        out.set_sform(sf if sf is not None else affine, code=int(scode))
    nib.save(out, str(output_path))

    after = header_snapshot(nib.load(str(output_path)))
    return {
        "applied": True,
        "input_path": str(input_path),
        "output_path": str(output_path),
        "before_header": before,
        "after_header": after,
        "fixes": fixes,
    }


def precheck_inputs(
    *,
    source_paths: dict[str, Path | None],
    precheck_dir: Path,
    auto_fix: bool,
) -> tuple[dict[str, Path | None], dict[str, Any], bool]:
    effective_paths: dict[str, Path | None] = {}
    records: dict[str, Any] = {}
    all_pass = True

    for key, path in source_paths.items():
        if path is None:
            effective_paths[key] = None
            records[key] = {
                "path": "",
                "used_path": "",
                "status": "missing_optional",
                "issues_before": [],
                "issues_after": [],
                "auto_fixes": [],
                "pass_after": True,
            }
            continue

        before = inspect_header_semantics(path)
        used_path = path
        fix_result = {
            "applied": False,
            "output_path": str(path),
            "fixes": [],
            "before_header": before.get("header", {}),
            "after_header": before.get("header", {}),
        }

        if auto_fix:
            fix_out = precheck_dir / f"{key}_{nii_stem(path)}_header_checked.nii.gz"
            fix_result = apply_header_autofix(path, fix_out)
            if fix_result.get("applied"):
                used_path = Path(fix_result["output_path"])

        after = inspect_header_semantics(used_path)
        record = {
            "path": str(path),
            "used_path": str(used_path),
            "status": "ok" if after.get("pass", False) else "issues_remaining",
            "header_before": before.get("header", {}),
            "header_after": after.get("header", {}),
            "issues_before": before.get("issues", []),
            "issues_after": after.get("issues", []),
            "auto_fixes": fix_result.get("fixes", []),
            "auto_fix_applied": bool(fix_result.get("applied", False)),
            "pass_after": bool(after.get("pass", False)),
        }
        records[key] = record
        effective_paths[key] = used_path
        all_pass = all_pass and bool(record["pass_after"])

    fixed_record = records.get("fixed", {})
    moving_record = records.get("moving", {})
    pair_issues: list[dict[str, Any]] = []
    fixed_ax = fixed_record.get("header_after", {}).get("axcodes", [])
    moving_ax = moving_record.get("header_after", {}).get("axcodes", [])
    fixed_unit = fixed_record.get("header_after", {}).get("spatial_unit", "unknown")
    moving_unit = moving_record.get("header_after", {}).get("spatial_unit", "unknown")
    if fixed_ax and moving_ax and fixed_ax != moving_ax:
        pair_issues.append(
            {
                "check": "axis_semantics_pair",
                "severity": "warning",
                "message": f"fixed axcodes={fixed_ax}, moving axcodes={moving_ax}",
                "reason": "manual orientation QC may be required before resample",
            }
        )
    if fixed_unit != moving_unit:
        pair_issues.append(
            {
                "check": "units_pair",
                "severity": "warning",
                "message": f"fixed spatial unit={fixed_unit}, moving spatial unit={moving_unit}",
                "reason": "units should match before registration",
            }
        )

    pair_pass = len(pair_issues) == 0
    records["pair_semantics"] = {
        "fixed_used_path": fixed_record.get("used_path", ""),
        "moving_used_path": moving_record.get("used_path", ""),
        "fixed_axcodes": fixed_ax,
        "moving_axcodes": moving_ax,
        "fixed_spatial_unit": fixed_unit,
        "moving_spatial_unit": moving_unit,
        "issues": pair_issues,
        "pass": pair_pass,
    }
    all_pass = all_pass and pair_pass
    return effective_paths, records, all_pass


def load_moving_orientation_manifest(path_text: str) -> dict[str, Any]:
    if not path_text.strip():
        raise RuntimeError(
            "Missing --moving-orientation-manifest. "
            "Orientation QC verified manifest is mandatory before resample_contract."
        )
    p = normalize_path(path_text)
    if not p.exists():
        raise FileNotFoundError(f"moving orientation manifest not found: {p}")
    payload = json.loads(p.read_text(encoding="utf-8"))
    if not bool(payload.get("verified", False)):
        raise RuntimeError(
            f"Orientation manifest is not verified: {p}. "
            "You must complete orientation QC and save with verified=true before continuing."
        )

    output_moving_txt = str(payload.get("output_moving", "")).strip()
    if not output_moving_txt:
        raise RuntimeError(f"Orientation manifest missing output_moving: {p}")
    output_moving = normalize_path(output_moving_txt)
    if not output_moving.exists():
        raise FileNotFoundError(f"Verified moving output not found: {output_moving} (from {p})")

    output_moving_mask_txt = str(payload.get("output_moving_mask", "")).strip()
    output_moving_mask = normalize_path(output_moving_mask_txt) if output_moving_mask_txt else None
    if output_moving_mask_txt and (output_moving_mask is None or not output_moving_mask.exists()):
        raise FileNotFoundError(
            f"Verified moving mask output not found: {output_moving_mask_txt} (from {p})"
        )

    return {
        "manifest_path": str(p),
        "verified": True,
        "selected_candidate": payload.get("selected_candidate", {}),
        "fixed_path": payload.get("fixed_path", ""),
        "input_moving": payload.get("input_moving", ""),
        "output_moving": str(output_moving),
        "input_moving_mask": payload.get("input_moving_mask", ""),
        "output_moving_mask": str(output_moving_mask) if output_moving_mask else "",
        "generated_at": payload.get("generated_at", ""),
    }


def resample_identity(
    *,
    ants_apply: str,
    input_path: Path,
    reference_path: Path,
    output_path: Path,
    interpolation: str,
    windows_mode: bool,
    env: dict[str, str],
) -> dict[str, Any]:
    cmd = [
        ants_apply,
        "-d",
        "3",
        "-i",
        to_ants_path(input_path, windows_mode),
        "-r",
        to_ants_path(reference_path, windows_mode),
        "-o",
        to_ants_path(output_path, windows_mode),
        "-n",
        interpolation,
    ]
    run(cmd, env=env)
    return {"cmd": cmd, "interpolation": interpolation}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Resample all registration inputs to one reference grid with strict header contract.")
    p.add_argument("--moving", required=True)
    p.add_argument("--fixed", required=True)
    p.add_argument("--output-dir", required=True)

    p.add_argument("--moving-mask", default="")
    p.add_argument("--fixed-mask", default="")
    p.add_argument("--labels", default="")

    p.add_argument("--reference-grid", default="", help="Existing reference grid. If omitted, build union grid from moving/fixed.")
    p.add_argument("--spacing", default="", help="Only used when building union grid; scalar or x,y,z")

    p.add_argument("--moving-name", default="moving_on_reference_grid.nii.gz")
    p.add_argument("--fixed-name", default="fixed_on_reference_grid.nii.gz")
    p.add_argument("--moving-mask-name", default="moving_mask_on_reference_grid.nii.gz")
    p.add_argument("--fixed-mask-name", default="fixed_mask_on_reference_grid.nii.gz")
    p.add_argument("--labels-name", default="labels_on_reference_grid.nii.gz")

    p.add_argument("--ants-bin", default="C:/tools/ANTs/ants-2.6.5/bin")
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--run-id", default="")
    p.add_argument(
        "--disable-header-autofix",
        action="store_true",
        help="Do not auto-fix qform/sform/units before resample; only report.",
    )
    p.add_argument(
        "--strict-precheck",
        action="store_true",
        help="Fail if pre-resample direction/unit/axis checks are not clean after optional auto-fix.",
    )
    p.add_argument(
        "--moving-orientation-manifest",
        default="",
        help="Required manifest from manual moving orientation QC (verified=true).",
    )
    p.add_argument("--strict", action="store_true", help="Fail if any contract check fails.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    requested_moving = normalize_path(args.moving)
    fixed = normalize_path(args.fixed)
    requested_moving_mask = normalize_path(args.moving_mask) if args.moving_mask.strip() else None
    fixed_mask = normalize_path(args.fixed_mask) if args.fixed_mask.strip() else None
    labels = normalize_path(args.labels) if args.labels.strip() else None

    out_dir = normalize_path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_id = args.run_id.strip() or datetime.now().strftime("RUN_RESAMPLE_%Y%m%d_%H%M%S")

    moving_orientation_provenance = load_moving_orientation_manifest(args.moving_orientation_manifest)
    manifest_fixed_txt = str(moving_orientation_provenance.get("fixed_path", "")).strip()
    if manifest_fixed_txt:
        manifest_fixed = normalize_path(manifest_fixed_txt)
        if manifest_fixed != fixed:
            raise RuntimeError(
                "Fixed image mismatch between --fixed and moving orientation manifest: "
                f"fixed={fixed}, manifest.fixed_path={manifest_fixed}"
            )

    moving = normalize_path(moving_orientation_provenance.get("output_moving", ""))
    moving_mask_manifest_txt = str(moving_orientation_provenance.get("output_moving_mask", "")).strip()
    moving_mask_manifest = normalize_path(moving_mask_manifest_txt) if moving_mask_manifest_txt else None
    if requested_moving_mask is not None:
        if moving_mask_manifest is None:
            raise RuntimeError(
                "You provided --moving-mask, but orientation manifest has no output_moving_mask. "
                "Rerun orientation QC with moving mask loaded, save verified outputs, then rerun resample_contract."
            )
        if moving_mask_manifest != requested_moving_mask:
            raise RuntimeError(
                "moving mask mismatch between --moving-mask and orientation manifest: "
                f"arg={requested_moving_mask}, manifest={moving_mask_manifest}"
            )
    moving_mask = moving_mask_manifest if moving_mask_manifest is not None else None
    requested_vs_verified_moving_match = requested_moving == moving
    requested_vs_verified_moving_mask_match = (
        (requested_moving_mask is None and moving_mask is None)
        or (requested_moving_mask is not None and moving_mask is not None and requested_moving_mask == moving_mask)
    )

    if not requested_moving.exists():
        raise FileNotFoundError(f"requested moving not found: {requested_moving}")
    if not moving.exists():
        raise FileNotFoundError(f"verified moving not found: {moving}")
    if not fixed.exists():
        raise FileNotFoundError(f"fixed not found: {fixed}")
    if requested_moving_mask and not requested_moving_mask.exists():
        raise FileNotFoundError(f"requested moving mask not found: {requested_moving_mask}")
    if moving_mask and not moving_mask.exists():
        raise FileNotFoundError(f"moving mask not found: {moving_mask}")
    if fixed_mask and not fixed_mask.exists():
        raise FileNotFoundError(f"fixed mask not found: {fixed_mask}")
    if labels and not labels.exists():
        raise FileNotFoundError(f"labels not found: {labels}")

    requested_sources: dict[str, Path | None] = {
        "moving": requested_moving,
        "fixed": fixed,
        "moving_mask": requested_moving_mask,
        "fixed_mask": fixed_mask,
        "labels": labels,
    }

    sources_for_precheck: dict[str, Path | None] = {
        "moving": moving,
        "fixed": fixed,
        "moving_mask": moving_mask,
        "fixed_mask": fixed_mask,
        "labels": labels,
    }

    precheck_dir = out_dir / "00_precheck"
    precheck_dir.mkdir(parents=True, exist_ok=True)
    effective_sources, precheck_records, precheck_all_pass = precheck_inputs(
        source_paths=sources_for_precheck,
        precheck_dir=precheck_dir,
        auto_fix=not args.disable_header_autofix,
    )
    precheck_records["moving_orientation_provenance"] = moving_orientation_provenance

    if args.strict_precheck and not precheck_all_pass:
        unresolved: list[str] = []
        for key, record in precheck_records.items():
            if key in {"pair_semantics", "moving_orientation_provenance"}:
                continue
            if not record.get("pass_after", True):
                unresolved.append(f"{key}: {record.get('issues_after', [])}")
        pair = precheck_records.get("pair_semantics", {})
        if not pair.get("pass", True):
            unresolved.append(f"pair_semantics: {pair.get('issues', [])}")
        raise RuntimeError("Precheck failed in strict-precheck mode: " + " | ".join(unresolved))

    moving = effective_sources.get("moving") or moving
    fixed = effective_sources.get("fixed") or fixed
    moving_mask = effective_sources.get("moving_mask")
    fixed_mask = effective_sources.get("fixed_mask")
    labels = effective_sources.get("labels")

    ants_bin = normalize_path(args.ants_bin)
    ants_apply = ensure_exec(ants_bin, "antsApplyTransforms")
    windows_mode = ants_apply.lower().endswith(".exe")

    env = os.environ.copy()
    env["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = str(max(args.threads, 1))
    env["OMP_NUM_THREADS"] = str(max(args.threads, 1))

    # Prepare reference grid (frozen source of truth)
    reference_grid = normalize_path(args.reference_grid) if args.reference_grid.strip() else None
    reference_out = out_dir / "reference_grid.nii.gz"

    reference_info: dict[str, Any]
    if reference_grid:
        if not reference_grid.exists():
            raise FileNotFoundError(f"reference grid not found: {reference_grid}")
        ref_img = nib.load(str(reference_grid))
        nib.save(ref_img, str(reference_out))
        canonical_ref = canonicalize_reference_grid(reference_out)
        reference_info = {
            "path": str(reference_out),
            "source": str(reference_grid),
            "shape": [int(x) for x in ref_img.shape[:3]],
            "spacing": [float(x) for x in ref_img.header.get_zooms()[:3]],
            "canonicalized": canonical_ref,
        }
    else:
        reference_info = build_union_reference_grid(
            moving_path=moving,
            fixed_path=fixed,
            out_path=reference_out,
            spacing_text=args.spacing,
        )
        canonical_ref = canonicalize_reference_grid(reference_out)
        reference_info["canonicalized"] = canonical_ref

    # If fixed mask absent but labels exist, derive native fixed mask as labels>0.
    fixed_mask_native = fixed_mask
    generated_fixed_mask_native = None
    generated_fixed_mask_precheck: dict[str, Any] = {}
    if fixed_mask_native is None and labels is not None:
        generated_fixed_mask_native = out_dir / "fixed_mask_from_labels_native.nii.gz"
        save_binary_mask_from_labels(labels, generated_fixed_mask_native)
        fixed_mask_native = generated_fixed_mask_native
        if not args.disable_header_autofix:
            fixed_mask_fix_out = precheck_dir / "fixed_mask_from_labels_header_checked.nii.gz"
            generated_fixed_mask_precheck = apply_header_autofix(fixed_mask_native, fixed_mask_fix_out)
            if generated_fixed_mask_precheck.get("applied"):
                fixed_mask_native = Path(generated_fixed_mask_precheck["output_path"])

    outputs: dict[str, Path] = {
        "moving": out_dir / args.moving_name,
        "fixed": out_dir / args.fixed_name,
    }
    if moving_mask is not None:
        outputs["moving_mask"] = out_dir / args.moving_mask_name
    if fixed_mask_native is not None:
        outputs["fixed_mask"] = out_dir / args.fixed_mask_name
    if labels is not None:
        outputs["labels"] = out_dir / args.labels_name

    resample_logs: dict[str, Any] = {}
    checks: dict[str, Any] = {}

    # Continuous outputs (linear)
    resample_logs["moving"] = resample_identity(
        ants_apply=ants_apply,
        input_path=moving,
        reference_path=reference_out,
        output_path=outputs["moving"],
        interpolation="Linear",
        windows_mode=windows_mode,
        env=env,
    )
    canonicalize_to_reference(outputs["moving"], reference_out, "float32")
    checks["moving"] = check_contract(outputs["moving"], reference_out, "float32")

    resample_logs["fixed"] = resample_identity(
        ants_apply=ants_apply,
        input_path=fixed,
        reference_path=reference_out,
        output_path=outputs["fixed"],
        interpolation="Linear",
        windows_mode=windows_mode,
        env=env,
    )
    canonicalize_to_reference(outputs["fixed"], reference_out, "float32")
    checks["fixed"] = check_contract(outputs["fixed"], reference_out, "float32")

    # Discrete outputs (NN / GenericLabel)
    if moving_mask is not None:
        resample_logs["moving_mask"] = resample_identity(
            ants_apply=ants_apply,
            input_path=moving_mask,
            reference_path=reference_out,
            output_path=outputs["moving_mask"],
            interpolation="NearestNeighbor",
            windows_mode=windows_mode,
            env=env,
        )
        canonicalize_to_reference(outputs["moving_mask"], reference_out, "uint8")
        checks["moving_mask"] = check_contract(outputs["moving_mask"], reference_out, "uint8")

    if fixed_mask_native is not None:
        resample_logs["fixed_mask"] = resample_identity(
            ants_apply=ants_apply,
            input_path=fixed_mask_native,
            reference_path=reference_out,
            output_path=outputs["fixed_mask"],
            interpolation="NearestNeighbor",
            windows_mode=windows_mode,
            env=env,
        )
        canonicalize_to_reference(outputs["fixed_mask"], reference_out, "uint8")
        checks["fixed_mask"] = check_contract(outputs["fixed_mask"], reference_out, "uint8")

    if labels is not None:
        resample_logs["labels"] = resample_identity(
            ants_apply=ants_apply,
            input_path=labels,
            reference_path=reference_out,
            output_path=outputs["labels"],
            interpolation="GenericLabel",
            windows_mode=windows_mode,
            env=env,
        )
        canonicalize_to_reference(outputs["labels"], reference_out, "uint16")
        checks["labels"] = check_contract(outputs["labels"], reference_out, "uint16")

    all_pass = all(v.get("pass", False) for v in checks.values())

    def hash_if_exists(path: Path | None) -> str:
        if path is None or not path.exists():
            return ""
        return sha256_file(path)

    manifest = {
        "run_id": run_id,
        "timestamp": now_iso(),
        "ants_bin": str(ants_bin),
        "threads": max(args.threads, 1),
        "precheck": {
            "strict_precheck": bool(args.strict_precheck),
            "auto_fix_enabled": not args.disable_header_autofix,
            "all_pass": bool(precheck_all_pass),
            "orientation_gate": {
                "required": True,
                "requested_moving": str(requested_moving),
                "verified_moving_used": str(moving),
                "requested_vs_verified_moving_match": bool(requested_vs_verified_moving_match),
                "requested_moving_mask": str(requested_moving_mask) if requested_moving_mask else "",
                "verified_moving_mask_used": str(moving_mask) if moving_mask else "",
                "requested_vs_verified_moving_mask_match": bool(requested_vs_verified_moving_mask_match),
                "verified_manifest": moving_orientation_provenance,
            },
            "records": precheck_records,
            "generated_fixed_mask_precheck": generated_fixed_mask_precheck,
        },
        "reference": {
            **reference_info,
            "sha256": sha256_file(reference_out),
            "header": header_snapshot(nib.load(str(reference_out))),
        },
        "sources": {
            "moving": str(requested_sources["moving"]) if requested_sources["moving"] else "",
            "fixed": str(requested_sources["fixed"]) if requested_sources["fixed"] else "",
            "moving_mask": str(requested_sources["moving_mask"]) if requested_sources["moving_mask"] else "",
            "fixed_mask": str(requested_sources["fixed_mask"]) if requested_sources["fixed_mask"] else "",
            "labels": str(requested_sources["labels"]) if requested_sources["labels"] else "",
            "fixed_mask_generated_from_labels": str(generated_fixed_mask_native) if generated_fixed_mask_native else "",
        },
        "sources_effective_for_resample": {
            "moving": str(moving),
            "fixed": str(fixed),
            "moving_mask": str(moving_mask) if moving_mask else "",
            "fixed_mask": str(fixed_mask_native) if fixed_mask_native else "",
            "labels": str(labels) if labels else "",
        },
        "source_hashes": {
            "moving_original": hash_if_exists(requested_sources["moving"]),
            "fixed_original": hash_if_exists(requested_sources["fixed"]),
            "moving_mask_original": hash_if_exists(requested_sources["moving_mask"]),
            "fixed_mask_original": hash_if_exists(requested_sources["fixed_mask"]),
            "labels_original": hash_if_exists(requested_sources["labels"]),
            "moving_effective": hash_if_exists(moving),
            "fixed_effective": hash_if_exists(fixed),
            "moving_mask_effective": hash_if_exists(moving_mask),
            "fixed_mask_effective": hash_if_exists(fixed_mask_native),
            "labels_effective": hash_if_exists(labels),
        },
        "outputs": {k: str(v) for k, v in outputs.items()},
        "resample_logs": resample_logs,
        "contract_checks": checks,
        "contract_all_pass": all_pass,
    }

    manifest_path = out_dir / "resample_contract_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    summary_md = out_dir / "resample_contract_summary.md"
    lines = [
        "# Resample Contract Summary",
        f"- run_id: `{run_id}`",
        f"- timestamp: `{manifest['timestamp']}`",
        f"- reference: `{reference_out}`",
        f"- precheck_all_pass: `{precheck_all_pass}` (strict_precheck={bool(args.strict_precheck)})",
        f"- contract_all_pass: `{all_pass}`",
        "",
        "## Outputs",
    ]
    for key, path in outputs.items():
        check = checks.get(key, {})
        lines.append(f"- `{key}`: `{path}` | pass={check.get('pass')} | sha256={check.get('sha256', '')}")
        if check.get("errors"):
            for err in check["errors"]:
                lines.append(f"  - error: {err}")
    summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[OK] manifest: {manifest_path}")
    print(f"[OK] summary : {summary_md}")

    if args.strict and not all_pass:
        raise RuntimeError("Resample contract check failed in strict mode.")


if __name__ == "__main__":
    main()

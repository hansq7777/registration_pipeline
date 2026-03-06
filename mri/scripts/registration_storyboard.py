#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

plt = None
import nibabel as nib
import numpy as np


@dataclass
class StoryRow:
    row_id: str
    label: str
    kind: str  # overlay | labels | scalar | missing
    fixed_path: str = ""
    moving_path: str = ""
    scalar_path: str = ""
    labels_path: str = ""
    qc_text: str = ""
    status: str = "ok"
    note: str = ""


def _ensure_matplotlib() -> Any:
    global plt
    if plt is not None:
        return plt
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as _plt
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing dependency: matplotlib. Install it in your environment, e.g. `pip install matplotlib`."
        ) from exc
    plt = _plt
    return plt


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def _fmt_qc(qc: dict[str, Any] | None) -> str:
    if not qc:
        return ""
    cc = _safe_float(qc.get("cc"))
    mi = _safe_float(qc.get("mi"))
    nmi = _safe_float(qc.get("nmi"))
    dice = _safe_float(qc.get("dice"))
    parts: list[str] = []
    if math.isfinite(cc):
        parts.append(f"CC={cc:.3f}")
    if math.isfinite(mi):
        parts.append(f"MI={mi:.3f}")
    if math.isfinite(nmi):
        parts.append(f"NMI={nmi:.3f}")
    if math.isfinite(dice):
        parts.append(f"Dice={dice:.3f}")
    return " | ".join(parts)


def _find_latest_manifest(reg_dir: Path) -> Path | None:
    candidates = sorted(
        reg_dir.glob("run_manifest_*.json"),
        key=lambda p: p.stat().st_mtime,
    )
    return candidates[-1] if candidates else None


def _resolve_manifest(reg_dir: Path, prefix: str, explicit: str) -> Path:
    if explicit.strip():
        path = Path(explicit).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Manifest not found: {path}")
        return path

    latest = _find_latest_manifest(reg_dir)
    if latest:
        return latest

    ms_summary = reg_dir / f"{prefix}multistart_summary.json"
    if ms_summary.exists():
        summary = _read_json(ms_summary)
        selected = int(summary.get("selected_seed", -1))
        if selected >= 0:
            cand = reg_dir / "multistart_runs" / f"seed_{selected}"
            latest_seed_manifest = _find_latest_manifest(cand)
            if latest_seed_manifest:
                return latest_seed_manifest

    raise FileNotFoundError(f"Cannot find run_manifest in {reg_dir}")


def _resolve_context(run_dir: Path) -> tuple[Path, Path, Path]:
    # Returns: workflow_root, reg_dir, report_dir
    if (run_dir / "02_reg").exists():
        workflow_root = run_dir
        reg_dir = run_dir / "02_reg"
    else:
        reg_dir = run_dir
        workflow_root = run_dir
        if (run_dir.parent / "01_inputs").exists() and run_dir.name == "02_reg":
            workflow_root = run_dir.parent

    report_dir = workflow_root / "05_report"
    if not report_dir.exists():
        report_dir = reg_dir
    report_dir.mkdir(parents=True, exist_ok=True)
    return workflow_root, reg_dir, report_dir


def _find_first_existing(paths: list[Path]) -> Path | None:
    for p in paths:
        if p.exists():
            return p
    return None


def _as_existing(path_str: str) -> Path | None:
    if not path_str:
        return None
    p = Path(path_str)
    return p if p.exists() else None


def _infer_stage_entries_from_filesystem(
    manifest: dict[str, Any],
    workflow_root: Path,
    *,
    prefix: str,
) -> dict[str, dict[str, Any]]:
    stages_dir = _as_existing(manifest.get("stages_dir", ""))
    if stages_dir is None:
        candidate = workflow_root / "stages"
        stages_dir = candidate if candidate.exists() else None
    if stages_dir is None:
        return {}

    out: dict[str, dict[str, Any]] = {}
    for stage in ["rigid", "affine", "syn"]:
        stage_dir = stages_dir / stage
        warped = stage_dir / f"{prefix}{stage}_Warped.nii.gz"
        if not warped.exists():
            continue
        qc = {}
        qc_path = stage_dir / f"{prefix}{stage}_qc_metrics.json"
        if qc_path.exists():
            try:
                qc = _read_json(qc_path)
            except Exception:
                qc = {}
        out[stage] = {
            "stage": stage,
            "warped": str(warped),
            "qc_metrics": qc if isinstance(qc, dict) else {},
            "inferred_from": str(stage_dir),
        }
    return out


def _load_trace_stage_order(manifest: dict[str, Any], workflow_root: Path) -> list[str]:
    trace_path = _as_existing(manifest.get("trace_path", ""))
    if trace_path is None:
        candidate = workflow_root / "transform_trace.jsonl"
        trace_path = candidate if candidate.exists() else None
    if trace_path is None:
        return []

    order: list[str] = []
    seen: set[str] = set()
    try:
        with trace_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                raw = line.strip()
                if not raw:
                    continue
                evt = json.loads(raw)
                if not isinstance(evt, dict):
                    continue
                step_name = str(evt.get("step_name", ""))
                if step_name not in {"REG_STAGE_STARTED", "REG_STAGE_DONE"}:
                    continue
                stage = str(evt.get("stage", "")).strip()
                if not stage or stage in seen:
                    continue
                seen.add(stage)
                order.append(stage)
    except Exception:
        return []
    return order


def _extract_stage_transforms(entry: dict[str, Any] | None) -> list[str]:
    if not entry:
        return []
    cmd = entry.get("command", [])
    if not isinstance(cmd, list):
        return []
    toks = [str(x) for x in cmd]
    transforms: list[str] = []
    for idx, tok in enumerate(toks):
        if tok == "-t" and idx + 1 < len(toks):
            val = toks[idx + 1].strip()
            if val:
                transforms.append(val)
    return transforms


def _format_stage_label(stage: str, entry: dict[str, Any] | None, manifest: dict[str, Any]) -> str:
    transforms = _extract_stage_transforms(entry)
    method = ""
    if transforms:
        method = " + ".join(transforms)
    elif stage == "syn":
        method = str(manifest.get("nonlinear_transform_spec", "")).strip() or str(
            manifest.get("nonlinear_transform", "")
        ).strip()
    elif stage == "rigid":
        method = "Rigid"
    elif stage == "affine":
        method = "Affine"

    reused = bool(entry and str(entry.get("reused_from_stage_dir", "")).strip())
    if reused:
        method = f"{method} (reused)".strip() if method else "reused"

    if method:
        return f"Stage: {stage}\nMethod: {method}"
    return f"Stage: {stage}"


def _build_rows(manifest: dict[str, Any], workflow_root: Path, *, prefix: str) -> list[StoryRow]:
    rows: list[StoryRow] = []

    fixed_input = _as_existing(manifest.get("fixed_image", ""))
    moving_input = _as_existing(manifest.get("moving_image", ""))
    fixed_opt = _as_existing(manifest.get("fixed_image_for_optimization", ""))
    moving_opt = _as_existing(manifest.get("moving_image_for_optimization", ""))
    warped_final = _as_existing(manifest.get("warped", ""))
    warped_full = _as_existing(manifest.get("warped_full", ""))

    qc_summary_path = workflow_root / "03_qc" / "qc_summary.json"
    moving_source = None
    fixed_source = None
    if qc_summary_path.exists():
        try:
            qcs = _read_json(qc_summary_path)
            inputs = qcs.get("inputs", {})
            moving_source = _as_existing(inputs.get("moving_source", ""))
            fixed_source = _as_existing(inputs.get("fixed_source", ""))
        except Exception:
            moving_source = None
            fixed_source = None

    rows.append(
        StoryRow(
            row_id="raw",
            label="Raw / source",
            kind="overlay",
            fixed_path=str(fixed_source or fixed_input or ""),
            moving_path=str(moving_source or moving_input or ""),
            note="source image pair",
            status="ok" if (fixed_source or fixed_input) and (moving_source or moving_input) else "missing",
        )
    )

    rows.append(
        StoryRow(
            row_id="reg_input",
            label="Registration input",
            kind="overlay",
            fixed_path=str(fixed_input or ""),
            moving_path=str(moving_input or ""),
            note="after resample/contract (if used)",
            status="ok" if fixed_input and moving_input else "missing",
        )
    )

    preproc_changed = (fixed_opt and moving_opt) and (
        str(fixed_opt) != str(fixed_input) or str(moving_opt) != str(moving_input)
    )
    if preproc_changed:
        rows.append(
            StoryRow(
                row_id="preproc",
                label="Optimization preproc",
                kind="overlay",
                fixed_path=str(fixed_opt),
                moving_path=str(moving_opt),
                note="N4 / normalization output",
            )
        )

    stage_order_raw = manifest.get("registration_stage_order", []) or []
    stage_order = [str(s).strip() for s in stage_order_raw if str(s).strip()]
    stage_entries = {
        str(x.get("stage", "")): x for x in (manifest.get("registration_stages", []) or []) if isinstance(x, dict)
    }
    inferred_stage_entries = _infer_stage_entries_from_filesystem(manifest, workflow_root, prefix=prefix)
    for stage_name, stage_entry in inferred_stage_entries.items():
        if stage_name not in stage_entries:
            stage_entries[stage_name] = stage_entry
    if not stage_order:
        inferred_order = [s for s in ["rigid", "affine", "syn"] if s in stage_entries]
        trace_order = _load_trace_stage_order(manifest, workflow_root)
        stage_order = list(inferred_order)
        for stage_name in trace_order:
            if stage_name not in stage_order:
                stage_order.append(stage_name)
        if not stage_order:
            stage_order = trace_order if trace_order else ["rigid", "affine", "syn"]

    for stage in stage_order:
        entry = stage_entries.get(stage)
        stage_label = _format_stage_label(stage, entry, manifest)
        if not entry:
            rows.append(
                StoryRow(
                    row_id=f"stage_{stage}",
                    label=stage_label,
                    kind="missing",
                    status="missing",
                    note="stage missing in this run",
                )
            )
            continue

        warped = _as_existing(entry.get("warped", ""))
        qc_text = _fmt_qc(entry.get("qc_metrics", {}))
        rows.append(
            StoryRow(
                row_id=f"stage_{stage}",
                label=stage_label,
                kind="overlay",
                fixed_path=str(fixed_opt or fixed_input or ""),
                moving_path=str(warped or ""),
                qc_text=qc_text,
                note="per-stage registration output",
                status="ok" if (fixed_opt or fixed_input) and warped else "missing",
            )
        )

    if warped_full and fixed_source:
        rows.append(
            StoryRow(
                row_id="warped_full",
                label="Warped full-res",
                kind="overlay",
                fixed_path=str(fixed_source),
                moving_path=str(warped_full),
                qc_text=_fmt_qc(manifest.get("qc_full_raw", {})),
            )
        )

    labels_path = _find_first_existing(
        [
            workflow_root / "04_labels" / "WHS12_labels_in_subject_raw.nii.gz",
            workflow_root / "04_labels" / "labels_in_native.nii.gz",
        ]
    )
    if labels_path:
        bg = moving_source or moving_input or moving_opt
        rows.append(
            StoryRow(
                row_id="labels_native",
                label="Inverse labels",
                kind="labels",
                fixed_path=str(bg or ""),
                labels_path=str(labels_path),
                note="template labels projected back to native",
                status="ok" if bg else "missing",
            )
        )

    jacobian_path = _find_first_existing(
        [
            workflow_root / "03_qc" / "jacobian_det.nii.gz",
            workflow_root / "03_qc" / "jacobian_det_param1.nii.gz",
        ]
    )
    if jacobian_path:
        rows.append(
            StoryRow(
                row_id="jacobian",
                label="Jacobian det",
                kind="scalar",
                scalar_path=str(jacobian_path),
                note="deformation sanity map",
            )
        )

    return rows


def _load_data(cache: dict[str, np.ndarray], path: str) -> np.ndarray | None:
    if not path:
        return None
    if path in cache:
        return cache[path]
    p = Path(path)
    if not p.exists():
        return None
    arr = np.asarray(nib.load(str(p)).get_fdata(), dtype=np.float32)
    cache[path] = arr
    return arr


def _robust_range(arr: np.ndarray) -> tuple[float, float]:
    vals = arr[np.isfinite(arr)]
    if vals.size == 0:
        return 0.0, 1.0
    non_zero = vals[vals != 0]
    use = non_zero if non_zero.size > 100 else vals
    lo, hi = np.percentile(use, [1, 99]).tolist()
    if hi - lo < 1e-8:
        hi = lo + 1e-6
    return float(lo), float(hi)


def _normalize(
    arr: np.ndarray,
    lo: float,
    hi: float,
    *,
    zero_as_background: bool = False,
    signed_magnitude: bool = False,
) -> np.ndarray:
    if signed_magnitude and lo < 0.0 < hi:
        # Robust-z images are centered at 0 (about half negative by design).
        # Use |z| for display intensity to avoid false "signal holes" from sign clipping.
        scale = max(abs(lo), abs(hi), 1e-8)
        out = np.abs(arr) / scale
    elif zero_as_background and lo < 0.0 < hi:
        # Anchor 0 to black so zero-valued background is not rendered mid-gray/red.
        out = arr / max(hi, 1e-8)
    else:
        out = (arr - lo) / max(hi - lo, 1e-8)
    out = np.clip(out, 0.0, 1.0)
    out[~np.isfinite(out)] = 0
    return out


def _axis_slice(arr: np.ndarray, axis: int, idx: int) -> np.ndarray:
    if axis == 0:
        sl = arr[idx, :, :]
    elif axis == 1:
        sl = arr[:, idx, :]
    else:
        sl = arr[:, :, idx]
    return np.rot90(sl)


def _overlay_pair(fixed_norm: np.ndarray, moving_norm: np.ndarray) -> np.ndarray:
    rgb = np.stack([fixed_norm, fixed_norm, fixed_norm], axis=-1)
    alpha = np.clip(moving_norm, 0.0, 1.0) * 0.65
    rgb[..., 0] = np.clip((1 - alpha) * rgb[..., 0] + alpha * 1.0, 0.0, 1.0)
    rgb[..., 1] = np.clip((1 - alpha) * rgb[..., 1] + alpha * 0.2, 0.0, 1.0)
    rgb[..., 2] = np.clip((1 - alpha) * rgb[..., 2] + alpha * 0.0, 0.0, 1.0)
    return rgb


def _overlay_labels(bg_norm: np.ndarray, labels: np.ndarray) -> np.ndarray:
    _ensure_matplotlib()
    rgb = np.stack([bg_norm, bg_norm, bg_norm], axis=-1)
    labels_i = np.rint(labels).astype(np.int32)
    nonzero = labels_i > 0
    if np.count_nonzero(nonzero) == 0:
        return rgb

    cmap = plt.get_cmap("tab20")
    mapped = (labels_i % 20) / 20.0
    colors = cmap(mapped)[..., :3]
    alpha = np.where(nonzero, 0.65, 0.0)[..., None]
    rgb = rgb * (1.0 - alpha) + colors * alpha
    return np.clip(rgb, 0.0, 1.0)


def _volume_center(arr: np.ndarray) -> tuple[int, int, int]:
    mask = np.isfinite(arr) & (arr != 0)
    if np.count_nonzero(mask) > 100:
        coords = np.argwhere(mask)
        return tuple(int(np.median(coords[:, i])) for i in range(3))
    return tuple(int((arr.shape[i] - 1) / 2) for i in range(3))


def _map_idx(idx: int, ref_dim: int, dim: int) -> int:
    if dim <= 1:
        return 0
    if ref_dim <= 1:
        return min(max(idx, 0), dim - 1)
    ratio = idx / max(ref_dim - 1, 1)
    mapped = int(round(ratio * (dim - 1)))
    return min(max(mapped, 0), dim - 1)


def _render_row(
    ax_row: np.ndarray,
    row: StoryRow,
    cache: dict[str, np.ndarray],
    norm_cache: dict[str, tuple[float, float]],
    *,
    n_cols: int,
) -> dict[str, Any]:
    axis_short = ["Sag", "Cor", "Axi"]
    row_centers: dict[str, Any] = {"row_id": row.row_id, "fixed_center": [], "moving_center": []}

    def _title(axis: int) -> str:
        return axis_short[axis]

    if row.status != "ok":
        for j in range(n_cols):
            ax = ax_row[j]
            ax.set_facecolor("#222222")
            ax.text(0.5, 0.5, f"{row.label}\nMISSING", color="white", ha="center", va="center", fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(_title(j), fontsize=8)
        row_centers["status"] = "missing"
        return row_centers

    def get_norm(path: str, arr: np.ndarray) -> tuple[float, float]:
        if path and path in norm_cache:
            return norm_cache[path]
        lo, hi = _robust_range(arr)
        if path:
            norm_cache[path] = (lo, hi)
        return lo, hi

    fixed = None
    moving = None
    if row.kind == "overlay":
        fixed = _load_data(cache, row.fixed_path)
        moving = _load_data(cache, row.moving_path)
    elif row.kind == "labels":
        fixed = _load_data(cache, row.fixed_path)
        moving = _load_data(cache, row.labels_path)
    elif row.kind == "scalar":
        scalar = _load_data(cache, row.scalar_path)
        fixed = scalar
        moving = scalar

    if fixed is None or moving is None:
        for j in range(n_cols):
            ax = ax_row[j]
            ax.set_facecolor("#222222")
            ax.text(0.5, 0.5, "MISSING", color="white", ha="center", va="center", fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(_title(j), fontsize=8)
        row_centers["status"] = "missing"
        return row_centers

    fixed_center = _volume_center(fixed)
    moving_center = _volume_center(moving)
    row_centers["fixed_center"] = list(fixed_center)
    row_centers["moving_center"] = list(moving_center)
    row_centers["status"] = "ok"

    for axis in range(3):
        ax = ax_row[axis]
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(_title(axis), fontsize=8)

        fi = fixed_center[axis]
        mi = moving_center[axis]
        fs = _axis_slice(fixed, axis, fi)
        ms = _axis_slice(moving, axis, mi)

        if row.kind == "overlay":
            flo, fhi = get_norm(row.fixed_path, fixed)
            mlo, mhi = get_norm(row.moving_path, moving)
            rgb = _overlay_pair(
                _normalize(fs, flo, fhi, zero_as_background=True, signed_magnitude=True),
                _normalize(ms, mlo, mhi, zero_as_background=True, signed_magnitude=True),
            )
            ax.imshow(rgb, interpolation="nearest")
        elif row.kind == "labels":
            blo, bhi = get_norm(row.fixed_path, fixed)
            rgb = _overlay_labels(
                _normalize(fs, blo, bhi, zero_as_background=True, signed_magnitude=True),
                ms,
            )
            ax.imshow(rgb, interpolation="nearest")
        elif row.kind == "scalar":
            slo, shi = get_norm(row.scalar_path, fixed)
            ax.imshow(
                _normalize(fs, slo, shi, zero_as_background=True, signed_magnitude=True),
                cmap="magma",
                interpolation="nearest",
            )
        else:
            ax.set_facecolor("#222222")
            ax.text(0.5, 0.5, "UNSUPPORTED", color="white", ha="center", va="center", fontsize=9)

    return row_centers


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Generate a long tri-view storyboard for one registration run. "
            "The storyboard follows the actual stage list from run_manifest."
        )
    )
    p.add_argument("--run-dir", required=True, help="Run root (contains 02_reg) or registration output directory.")
    p.add_argument("--prefix", default="template_reg_", help="Output prefix used by registration files.")
    p.add_argument("--manifest", default="", help="Optional explicit run_manifest path.")
    p.add_argument("--output", default="", help="Optional output PNG path.")
    p.add_argument("--dpi", type=int, default=180)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    _ensure_matplotlib()
    run_dir = Path(args.run_dir).expanduser().resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"run-dir not found: {run_dir}")

    workflow_root, reg_dir, report_dir = _resolve_context(run_dir)
    manifest_path = _resolve_manifest(reg_dir, args.prefix, args.manifest)
    manifest = _read_json(manifest_path)

    rows = _build_rows(manifest, workflow_root, prefix=args.prefix)
    cache: dict[str, np.ndarray] = {}
    norm_cache: dict[str, tuple[float, float]] = {}

    n_rows = max(len(rows), 1)
    n_cols = 3
    fig_h = max(2.6 * n_rows, 6)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, fig_h), squeeze=False)
    centers_per_row: list[dict[str, Any]] = []

    for i, row in enumerate(rows):
        center_info = _render_row(axes[i], row, cache, norm_cache, n_cols=n_cols)
        centers_per_row.append(center_info)
        row_label = row.label
        if row.qc_text:
            row_label = f"{row_label}\n{row.qc_text}"
        if row.note:
            row_label = f"{row_label}\n{row.note}"
        axes[i, 0].set_ylabel(row_label, fontsize=8, rotation=0, ha="right", va="center", labelpad=120)

    fig.suptitle(
        "Registration Storyboard (actual run stages)",
        fontsize=12,
        y=0.995,
    )
    plt.tight_layout(rect=[0.22, 0.01, 1, 0.985])

    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output.strip()
        else report_dir / "registration_storyboard.png"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=max(args.dpi, 72))
    plt.close(fig)

    manifest_out = output_path.with_name("registration_storyboard_manifest.json")
    derived_stage_order = [r.row_id.replace("stage_", "", 1) for r in rows if r.row_id.startswith("stage_")]
    manifest_payload = {
        "run_id": manifest.get("run_id", ""),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "run_dir": str(run_dir),
        "workflow_root": str(workflow_root),
        "reg_dir": str(reg_dir),
        "registration_manifest": str(manifest_path),
        "output_png": str(output_path),
        "center_mode": "per_image_own_center_per_row",
        "registration_stage_order": derived_stage_order,
        "storyboard_contract_note": (
            manifest.get("storyboard_contract_note", "")
            or "If pipeline steps change or are skipped in future runs, storyboard must follow that run's actual stage list."
        ),
        "centers_per_row": centers_per_row,
        "rows": [
            {
                "row_id": r.row_id,
                "label": r.label,
                "kind": r.kind,
                "status": r.status,
                "fixed_path": r.fixed_path,
                "moving_path": r.moving_path,
                "scalar_path": r.scalar_path,
                "labels_path": r.labels_path,
                "qc_text": r.qc_text,
                "note": r.note,
            }
            for r in rows
        ],
    }
    manifest_out.write_text(json.dumps(manifest_payload, indent=2), encoding="utf-8")

    print(f"[OK] storyboard: {output_path}")
    print(f"[OK] storyboard manifest: {manifest_out}")


if __name__ == "__main__":
    main()

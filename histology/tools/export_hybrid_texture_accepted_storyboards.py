from __future__ import annotations

import csv
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np

REPO_HISTOLOGY_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_HISTOLOGY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_HISTOLOGY_ROOT))

from gui_mvp.hitl_gui.application.pair_registration import (  # noqa: E402
    _ants_apply,
    _compute_stage_heatmap,
    _run_logged,
    _stage_command,
    _warp_gray_affine,
    _warp_mask_affine,
    _write_coord_images,
    build_affine_matrix,
    compute_affine_stage_heatmap,
    compute_registration_metrics,
    find_ants_bin,
    gray_preview_panel,
    overlay_preview,
    read_nifti_2d,
    render_storyboard,
    write_nifti_2d,
)
from evaluate_fake_myelin_epochs_registration import (  # noqa: E402
    RAW_NISSL_BASELINE,
    _gray_u8_to_float,
    _load_binary_mask,
    _moving_gray_for_source,
    _read_mask,
)


HYBRID_ROOT = Path(r"/mnt/d/Research/Image Analysis/Nanozoomer scans/histology_shape_fake_hybrid_20260329T191855Z")
DESKTOP_ROOT = Path(r"/mnt/c/Users/Siqi/Desktop/Priority1_mask_vs_MI_storyboards_20260329")
OUTPUT_ROOT = DESKTOP_ROOT / "06_shape_then_texture_accepted_cases"
EPOCH_NAME = "epoch30"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _blank_heatmap(shape_hw: tuple[int, int]) -> np.ndarray:
    h, w = shape_hw
    return np.full((h, w, 3), 235, dtype=np.uint8)


def _gradient_mag_display(gray: np.ndarray, mask: np.ndarray, sigma: float = 1.5) -> np.ndarray:
    gray_u8 = np.clip(np.round(gray * 255.0), 0, 255).astype(np.uint8)
    mask_bool = mask > 0
    blur = cv2.GaussianBlur(gray_u8, (0, 0), sigmaX=float(sigma), sigmaY=float(sigma), borderType=cv2.BORDER_REPLICATE)
    blur_f = blur.astype(np.float32) / 255.0
    gx = cv2.Sobel(blur_f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(blur_f, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    out = np.zeros_like(mag, dtype=np.float32)
    vals = mag[mask_bool]
    if vals.size > 0:
        vmax = float(np.percentile(vals, 99.0))
        if vmax <= 1e-8:
            vmax = float(vals.max())
        if vmax > 1e-8:
            out[mask_bool] = np.clip(mag[mask_bool] / vmax, 0.0, 1.0)
    return out


def _case_category(by_arm: dict[str, dict[str, str]]) -> tuple[str, str]:
    raw_acc = by_arm["shape_then_raw_texture_rigid"].get("texture_gate_accepted") == "True"
    fake_acc = by_arm["shape_then_fake_texture_rigid"].get("texture_gate_accepted") == "True"
    if fake_acc and raw_acc:
        return "03_both", "Both local texture refinements help after shape alignment; compare whether fake myelin gives a cleaner local correction."
    if fake_acc:
        return "01_fake_only", "Only fake-myelin local refinement passes the geometry gate here, suggesting synthetic same-modality texture adds usable local signal after shape alignment."
    if raw_acc:
        return "02_raw_only", "Only raw-Nissl local refinement passes the geometry gate here; fake myelin does not add extra value for this case."
    raise RuntimeError("Case is not texture-accepted in either arm.")


def _accepted_cases() -> dict[str, dict[str, str]]:
    rows = list(csv.DictReader((HYBRID_ROOT / "hybrid_results.csv").open(encoding="utf-8")))
    by_pair: dict[str, dict[str, str]] = {}
    for row in rows:
        pair_key = str(row["pair_key"])
        if row["arm"] not in {"shape_then_raw_texture_rigid", "shape_then_fake_texture_rigid"}:
            continue
        by_pair.setdefault(pair_key, {})[row["arm"]] = row
    out: dict[str, dict[str, str]] = {}
    for pair_key, arm_rows in by_pair.items():
        if (
            arm_rows.get("shape_then_raw_texture_rigid", {}).get("texture_gate_accepted") == "True"
            or arm_rows.get("shape_then_fake_texture_rigid", {}).get("texture_gate_accepted") == "True"
        ):
            out[pair_key] = arm_rows
    return dict(sorted(out.items()))


def _rerun_local_rigid(
    *,
    fixed_gray: np.ndarray,
    moving_gray: np.ndarray,
    fixed_mask: np.ndarray,
    moving_mask: np.ndarray,
    ants_bin: Path,
    tag: str,
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix=f"{tag}_") as tmpdir_s:
        tmpdir = Path(tmpdir_s)
        fixed_img_path = tmpdir / "fixed_gray.nii.gz"
        moving_img_path = tmpdir / "moving_gray.nii.gz"
        fixed_mask_path = tmpdir / "fixed_mask.nii.gz"
        moving_mask_path = tmpdir / "moving_mask.nii.gz"
        write_nifti_2d(fixed_img_path, fixed_gray)
        write_nifti_2d(moving_img_path, moving_gray)
        write_nifti_2d(fixed_mask_path, fixed_mask)
        write_nifti_2d(moving_mask_path, moving_mask)
        moving_coord_x, moving_coord_y = _write_coord_images(tmpdir, moving_gray.shape[:2])

        stage_dir = tmpdir / "rigid"
        stage_dir.mkdir(parents=True, exist_ok=True)
        prefix = stage_dir / "rigid_"
        cmd = _stage_command(
            ants_bin,
            "rigid",
            fixed_img_path,
            moving_img_path,
            fixed_mask_path,
            moving_mask_path,
            prefix,
            [],
            "current",
        )
        _run_logged(cmd, stage_dir / "rigid.log")
        rigid_mat = stage_dir / "rigid_0GenericAffine.mat"
        warped_mask_path = stage_dir / "rigid_warped_mask.nii.gz"
        _ants_apply(
            ants_bin,
            moving_mask_path,
            fixed_img_path,
            warped_mask_path,
            [rigid_mat],
            interpolation="NearestNeighbor",
            log_path=stage_dir / "rigid_warp_mask.log",
        )
        warped_img_path = stage_dir / "rigid_Warped.nii.gz"
        warped_gray = np.clip(read_nifti_2d(warped_img_path), 0.0, 1.0)
        warped_mask = (read_nifti_2d(warped_mask_path) > 0.5).astype(np.float32)
        metrics, metric_timing = compute_registration_metrics(fixed_gray, warped_gray, fixed_mask, warped_mask)
        heatmap_rgb, _ = _compute_stage_heatmap(
            ants_bin,
            stage_dir,
            "rigid",
            fixed_img_path,
            fixed_mask,
            moving_coord_x,
            moving_coord_y,
            rigid_mat,
            stage_dir / "affine_0GenericAffine.mat",
            warped_mask_path,
        )
        return {
            "warped_gray": warped_gray,
            "warped_mask": warped_mask,
            "metrics": metrics,
            "metric_timing_seconds": metric_timing,
            "heatmap": heatmap_rgb,
        }


def _render_case(pair_key: str, rows_by_arm: dict[str, str], ants_bin: Path) -> None:
    result_path = HYBRID_ROOT / "results" / f"{rows_by_arm['shape_then_fake_texture_rigid']['stem']}.json"
    result = _load_json(result_path)
    geom = _load_json(Path(result["geometry_manifest_path"]))

    fixed_gray = _gray_u8_to_float(_read_mask(Path(geom["cache_files"]["fixed_gray_1024"])))
    fixed_mask = _load_binary_mask(Path(geom["cache_files"]["fixed_mask_1024"]))
    moving_mask_1536 = _load_binary_mask(Path(geom["cache_files"]["moving_mask_1536"]))
    moving_mask_1024 = _load_binary_mask(Path(geom["cache_files"]["moving_mask_1024"]))

    from evaluate_fake_myelin_epochs_registration import PairGeometry  # noqa: E402

    geometry = PairGeometry(
        pair_key=str(geom["pair_key"]),
        stem=str(geom["stem"]),
        myelin_image_path=Path(str(geom["myelin_image"])),
        nissl_image_path=Path(str(geom["nissl_image"])),
        myelin_mask_path=Path(str(geom["myelin_mask"])),
        nissl_mask_path=Path(str(geom["nissl_mask"])),
        fixed_gray_1024_path=Path(str(geom["cache_files"]["fixed_gray_1024"])),
        fixed_mask_1024_path=Path(str(geom["cache_files"]["fixed_mask_1024"])),
        moving_mask_1536_path=Path(str(geom["cache_files"]["moving_mask_1536"])),
        moving_mask_1024_path=Path(str(geom["cache_files"]["moving_mask_1024"])),
        geometry_manifest_path=Path(result["geometry_manifest_path"]),
    )

    raw_gray, _ = _moving_gray_for_source(RAW_NISSL_BASELINE, geometry, moving_mask_1536, moving_mask_1024)
    fake_gray, _ = _moving_gray_for_source(EPOCH_NAME, geometry, moving_mask_1536, moving_mask_1024)
    fixed_tex = _gradient_mag_display(fixed_gray, fixed_mask)
    raw_tex = _gradient_mag_display(raw_gray, moving_mask_1024)
    fake_tex = _gradient_mag_display(fake_gray, moving_mask_1024)

    shape_params = dict(result["arms"]["mask_rigid"]["mask_rigid_transform_params"])
    shape_mat = build_affine_matrix(
        moving_mask_1024.shape[:2],
        fixed_mask.shape[:2],
        tx_px=float(shape_params["tx_px"]),
        ty_px=float(shape_params["ty_px"]),
        angle_deg=float(shape_params["angle_deg"]),
        scale=float(shape_params.get("scale", 1.0)),
    )
    shape_mask = _warp_mask_affine(moving_mask_1024, shape_mat, fixed_mask.shape[:2]).astype(np.float32)
    shape_gray_raw = np.clip(_warp_gray_affine(raw_gray, shape_mat, fixed_gray.shape[:2]), 0.0, 1.0)
    shape_gray_fake = np.clip(_warp_gray_affine(fake_gray, shape_mat, fixed_gray.shape[:2]), 0.0, 1.0)
    shape_metrics = dict(result["arms"]["mask_rigid"]["mask_rigid_metrics"])
    shape_gate = dict(result["arms"]["mask_rigid"]["mask_rigid_gate"])
    shape_heatmap, _ = compute_affine_stage_heatmap(
        np.asarray(shape_mat, dtype=np.float32),
        fixed_gray.shape[:2],
        ((fixed_mask > 0) | (shape_mask > 0)).astype(np.uint8),
        OUTPUT_ROOT / "_tmp_shape_heatmap.png",
    )
    try:
        (OUTPUT_ROOT / "_tmp_shape_heatmap.png").unlink(missing_ok=True)
    except Exception:
        pass

    raw_start_gray = shape_gray_raw if shape_gate.get("accepted") else raw_gray
    fake_start_gray = shape_gray_fake if shape_gate.get("accepted") else fake_gray
    start_mask = shape_mask if shape_gate.get("accepted") else moving_mask_1024

    raw_local = _rerun_local_rigid(
        fixed_gray=fixed_gray,
        moving_gray=raw_start_gray,
        fixed_mask=fixed_mask,
        moving_mask=start_mask,
        ants_bin=ants_bin,
        tag=f"{pair_key}_raw_local",
    )
    fake_local = _rerun_local_rigid(
        fixed_gray=fixed_gray,
        moving_gray=fake_start_gray,
        fixed_mask=fixed_mask,
        moving_mask=start_mask,
        ants_bin=ants_bin,
        tag=f"{pair_key}_fake_local",
    )
    raw_local_tex = _gradient_mag_display(raw_local["warped_gray"], raw_local["warped_mask"])
    fake_local_tex = _gradient_mag_display(fake_local["warped_gray"], fake_local["warped_mask"])

    category_slug, category_conclusion = _case_category(rows_by_arm)
    category_dir = OUTPUT_ROOT / category_slug
    category_dir.mkdir(parents=True, exist_ok=True)
    case_dir = category_dir / pair_key
    case_dir.mkdir(parents=True, exist_ok=True)

    raw_gate = dict(result["arms"]["shape_then_raw_texture_rigid"]["texture_stage_gate"])
    fake_gate = dict(result["arms"]["shape_then_fake_texture_rigid"]["texture_stage_gate"])

    rows = [
        {
            "label": "Input",
            "note": (
                f"raw Nissl input [gradient magnitude view] | Dice={float(result['input_metrics']['dice']):.4f} | "
                f"HD95={float(result['input_metrics']['hd95_px']):.2f}"
            ),
            "fixed": gray_preview_panel(fixed_tex),
            "moving": gray_preview_panel(raw_tex),
            "overlay": overlay_preview(fixed_tex, raw_tex, fixed_mask, moving_mask_1024),
            "heatmap": _blank_heatmap(fixed_gray.shape[:2]),
            "col_titles": ("Moving", "Fixed", "Overlay", "Warp Field"),
        },
        {
            "label": f"Shape Stage [{'ACCEPTED' if shape_gate.get('accepted') else 'REJECTED'}]",
            "note": (
                f"mask-rigid | Dice={float(shape_metrics['dice']):.4f} | "
                f"HD95={float(shape_metrics['hd95_px']):.2f}"
            ),
            "fixed": gray_preview_panel(fixed_gray),
            "moving": gray_preview_panel(shape_gray_raw),
            "overlay": overlay_preview(fixed_gray, shape_gray_raw, fixed_mask, shape_mask),
            "heatmap": shape_heatmap,
            "col_titles": ("Moving", "Fixed", "Overlay", "Warp Field"),
        },
        {
            "label": f"Raw Local Texture [{'ACCEPTED' if raw_gate.get('accepted') else 'REJECTED'}]",
            "note": (
                f"shape -> raw local rigid [gradient magnitude view] | Dice={float(raw_local['metrics']['dice']):.4f} | "
                f"HD95={float(raw_local['metrics']['hd95_px']):.2f}"
            ),
            "fixed": gray_preview_panel(fixed_tex),
            "moving": gray_preview_panel(raw_local_tex),
            "overlay": overlay_preview(fixed_tex, raw_local_tex, fixed_mask, raw_local["warped_mask"]),
            "heatmap": raw_local["heatmap"],
            "col_titles": ("Moving", "Fixed", "Overlay", "Warp Field"),
        },
        {
            "label": f"Fake Local Texture [{'ACCEPTED' if fake_gate.get('accepted') else 'REJECTED'}]",
            "note": (
                f"shape -> fake(epoch30) local rigid [gradient magnitude view] | Dice={float(fake_local['metrics']['dice']):.4f} | "
                f"HD95={float(fake_local['metrics']['hd95_px']):.2f}"
            ),
            "fixed": gray_preview_panel(fixed_tex),
            "moving": gray_preview_panel(fake_local_tex),
            "overlay": overlay_preview(fixed_tex, fake_local_tex, fixed_mask, fake_local["warped_mask"]),
            "heatmap": fake_local["heatmap"],
            "col_titles": ("Moving", "Fixed", "Overlay", "Warp Field"),
        },
    ]
    storyboard_path = case_dir / "shape_then_texture_storyboard.png"
    render_storyboard(rows, storyboard_path)

    case_summary = {
        "pair_key": pair_key,
        "category": category_slug,
        "category_conclusion": category_conclusion,
        "input_metrics": dict(result["input_metrics"]),
        "shape_metrics": shape_metrics,
        "shape_gate": shape_gate,
        "raw_local_metrics": dict(raw_local["metrics"]),
        "raw_local_gate": raw_gate,
        "fake_local_metrics": dict(fake_local["metrics"]),
        "fake_local_gate": fake_gate,
    }
    _write_json(case_dir / "summary.json", case_summary)

    case_readme = [
        f"# {pair_key}",
        "",
        f"Category: `{category_slug}`",
        "",
        category_conclusion,
        "",
        "## Quick read",
        "",
        f"- Shape stage: {'ACCEPTED' if shape_gate.get('accepted') else 'REJECTED'} | Dice `{float(shape_metrics['dice']):.4f}` | HD95 `{float(shape_metrics['hd95_px']):.2f}`",
        f"- Raw local texture: {'ACCEPTED' if raw_gate.get('accepted') else 'REJECTED'} | Dice `{float(raw_local['metrics']['dice']):.4f}` | HD95 `{float(raw_local['metrics']['hd95_px']):.2f}`",
        f"- Fake local texture: {'ACCEPTED' if fake_gate.get('accepted') else 'REJECTED'} | Dice `{float(fake_local['metrics']['dice']):.4f}` | HD95 `{float(fake_local['metrics']['hd95_px']):.2f}`",
        "",
        "## Files",
        "",
        "- `shape_then_texture_storyboard.png`",
        "- `summary.json`",
    ]
    (case_dir / "README.md").write_text("\n".join(case_readme), encoding="utf-8")


def main() -> None:
    ants_bin = find_ants_bin()
    if ants_bin is None:
        raise RuntimeError("ANTs not found.")
    accepted = _accepted_cases()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    top_lines = [
        "# Shape Then Texture Accepted Cases",
        "",
        "These cases are the hybrid benchmark examples where the second-stage local texture refinement was accepted by the geometry gate.",
        "",
        "Subfolders:",
        "- `01_fake_only`",
        "- `02_raw_only`",
        "- `03_both`",
        "",
        "Notes:",
        "- `Input`, `Raw local texture`, and `Fake local texture` are shown in `gradient magnitude` view.",
        "- `Shape stage` is kept in ordinary grayscale to emphasize the geometry-first step.",
        "- The gate still depends on `Dice` and `HD95`, not on appearance metrics.",
    ]
    (OUTPUT_ROOT / "README.md").write_text("\n".join(top_lines), encoding="utf-8")

    for pair_key, rows_by_arm in accepted.items():
        _render_case(pair_key, rows_by_arm, ants_bin)

    category_notes = {
        "01_fake_only": "Only fake-myelin local refinement is accepted. These are the clearest cases where synthetic same-modality texture may add real local alignment value.",
        "02_raw_only": "Only raw-Nissl local refinement is accepted. These are the cautionary cases where fake myelin does not add enough extra information.",
        "03_both": "Both raw and fake local texture refinements are accepted. These are the cases where shape-first alignment leaves a small local adjustment window that both image sources can exploit.",
    }
    for slug, note in category_notes.items():
        path = OUTPUT_ROOT / slug
        path.mkdir(parents=True, exist_ok=True)
        (path / "README.md").write_text(f"# {slug}\n\n{note}\n", encoding="utf-8")

    print(f"Exported hybrid accepted storyboards to {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()

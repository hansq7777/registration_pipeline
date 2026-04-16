from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from dataclasses import dataclass
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
    _stage_command,
    _stage_transforms,
    _write_coord_images,
    compute_registration_metrics,
    find_ants_bin,
    gray_preview_panel,
    metrics_note,
    overlay_preview,
    read_nifti_2d,
    render_storyboard,
    stage_display_name,
    strict_pareto_gate_decision,
    write_nifti_2d,
)
from run_registration_preprocessing_matrix import (  # noqa: E402
    VARIANT_SPECS,
    _gray_u8_to_float,
    preprocess_variant,
)


DESKTOP_ROOT = Path(r"/mnt/c/Users/Siqi/Desktop")
MASK_BENCH_ROOT = Path(r"/mnt/d/Research/Image Analysis/Nanozoomer scans/histology_priority1_mask_rigid_benchmark_20260329T181453Z")
MATRIX_ROOT = Path(r"/mnt/d/Research/Image Analysis/Nanozoomer scans/histology_registration_preproc_matrix_20260326T070652Z")
MI_VARIANT = "gradient_mag_blur_1.5"
OUTPUT_ROOT = DESKTOP_ROOT / "Priority1_mask_vs_MI_storyboards_20260329"


@dataclass(frozen=True)
class CaseSpec:
    pair_key: str
    group: str

    @property
    def unit_key(self) -> str:
        return f"{self.pair_key}__group_{self.group}"


CATEGORIES: list[dict[str, Any]] = [
    {
        "slug": "01_mask_accept_mi_reject",
        "title": "Category 1: Mask Accepted, MI Rejected",
        "conclusion": "Geometry-driven rigid is safer here. Shape matching improves overlap while MI-rigid is rejected by the gate.",
        "cases": [
            CaseSpec("2506_84__2506_85", "all"),
            CaseSpec("2508_162__2508_163", "all"),
            CaseSpec("2501_78__2501_79", "all"),
            CaseSpec("2506_186__2506_187", "all"),
            CaseSpec("2507_204__2507_205", "all"),
        ],
    },
    {
        "slug": "02_mi_accept_mask_reject",
        "title": "Category 2: MI Accepted, Mask Rejected",
        "conclusion": "Intensity-driven rigid still helps in a minority of cases. These are the counterexamples against claiming mask-rigid is universally best.",
        "cases": [
            CaseSpec("2504_78__2504_79", "all"),
            CaseSpec("2504_144__2504_145", "all"),
            CaseSpec("2502_120__2502_97", "all"),
            CaseSpec("2501_48__2501_49", "all"),
            CaseSpec("2506_162__2506_163", "all"),
        ],
    },
    {
        "slug": "03_both_accept_mask_better",
        "title": "Category 3: Both Accepted, But Mask Better",
        "conclusion": "Both methods help, but mask-rigid gives cleaner geometry and lower contour error in these examples.",
        "cases": [
            CaseSpec("2501_24__2501_25", "1"),
            CaseSpec("2506_42__2506_43", "all"),
            CaseSpec("2501_54__2501_55", "all"),
            CaseSpec("2507_30__2507_31", "2"),
            CaseSpec("2507_162__2507_163", "all"),
        ],
    },
    {
        "slug": "04_both_accept_mi_better",
        "title": "Category 4: Both Accepted, But MI Better",
        "conclusion": "These are the cases where appearance-based cues still provide cleaner final overlap than pure shape matching.",
        "cases": [
            CaseSpec("2501_168__2501_169", "all"),
            CaseSpec("2507_144__2507_145", "all"),
            CaseSpec("2504_138__2504_139", "all"),
            CaseSpec("2506_204__2506_205", "all"),
            CaseSpec("2504_168__2504_169", "all"),
        ],
    },
    {
        "slug": "05_both_reject_mask_tail",
        "title": "Category 5: Both Rejected, Mask Failure Tail",
        "conclusion": "These cases motivate the next method. Both rigid strategies fail, and mask-rigid can still produce heavy HD95 tails on hard morphology.",
        "cases": [
            CaseSpec("2502_156__2502_157", "all"),
            CaseSpec("2506_114__2506_115", "all"),
            CaseSpec("2501_180__2501_181", "all"),
            CaseSpec("2507_168__2507_169", "all"),
            CaseSpec("2502_132__2502_133", "all"),
        ],
    },
]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _find_mask_run_dir(case: CaseSpec) -> Path:
    pair_dir = MASK_BENCH_ROOT / "mask_rigid_runs" / case.pair_key
    if not pair_dir.exists():
        raise FileNotFoundError(pair_dir)
    suffix = f"_nissl_{case.group}_to_myelin_{case.group}"
    matches = sorted(p for p in pair_dir.iterdir() if p.is_dir() and p.name.endswith(suffix))
    if not matches:
        raise FileNotFoundError(f"No mask run found for {case.unit_key}")
    return matches[-1]


def _geometry_dir(case: CaseSpec) -> Path:
    path = MATRIX_ROOT / "geometry" / case.unit_key
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def _load_cached_geometry(case: CaseSpec) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    geom = _geometry_dir(case)
    fixed_rgb = cv2.cvtColor(cv2.imread(str(geom / "fixed_myelin_rgb_1024.png"), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    moving_rgb = cv2.cvtColor(cv2.imread(str(geom / "moving_nissl_rgb_1024.png"), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    fixed_labels = cv2.imread(str(geom / "fixed_myelin_labels_1024.png"), cv2.IMREAD_UNCHANGED).astype(np.uint8)
    moving_labels = cv2.imread(str(geom / "moving_nissl_labels_1024.png"), cv2.IMREAD_UNCHANGED).astype(np.uint8)
    return fixed_rgb, fixed_labels, moving_rgb, moving_labels


def _render_mi_storyboard(case: CaseSpec, out_dir: Path) -> tuple[Path, dict[str, Any]]:
    ants_bin = find_ants_bin()
    if ants_bin is None:
        raise RuntimeError("ANTs not found.")
    fixed_rgb, fixed_labels, moving_rgb, moving_labels = _load_cached_geometry(case)
    fixed_gray_u8, fixed_labels = preprocess_variant(fixed_rgb, fixed_labels, MI_VARIANT)
    moving_gray_u8, moving_labels = preprocess_variant(moving_rgb, moving_labels, MI_VARIANT)
    background_value = int(VARIANT_SPECS[MI_VARIANT]["background"])
    fixed_gray = _gray_u8_to_float(fixed_gray_u8, fixed_labels, background_value)
    moving_gray = _gray_u8_to_float(moving_gray_u8, moving_labels, background_value)
    fixed_mask = (fixed_labels == 1).astype(np.float32)
    moving_mask = (moving_labels == 1).astype(np.float32)

    out_dir.mkdir(parents=True, exist_ok=True)
    storyboard_path = out_dir / "mi_rigid_storyboard.png"
    manifest_path = out_dir / "mi_rigid_summary.json"

    with tempfile.TemporaryDirectory(prefix=f"{case.unit_key}_mi_rigid_") as tmpdir_s:
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

        input_metrics, input_metric_timing = compute_registration_metrics(fixed_gray, moving_gray, fixed_mask, moving_mask)
        input_note = metrics_note(input_metrics, input_metric_timing, f"{MI_VARIANT} before registration")

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
        from gui_mvp.hitl_gui.application.pair_registration import _run_logged  # noqa: E402
        _run_logged(cmd, stage_dir / "rigid.log")
        rigid_mat = stage_dir / "rigid_0GenericAffine.mat"
        tfms = _stage_transforms(stage_dir, "rigid", rigid_mat, stage_dir / "affine_0GenericAffine.mat")
        warped_mask_path = stage_dir / "rigid_warped_mask.nii.gz"
        _ants_apply(
            ants_bin,
            moving_mask_path,
            fixed_img_path,
            warped_mask_path,
            tfms,
            interpolation="NearestNeighbor",
            log_path=stage_dir / "rigid_warp_mask.log",
        )
        warped_img_path = stage_dir / "rigid_Warped.nii.gz"
        warped_gray = read_nifti_2d(warped_img_path)
        warped_mask = read_nifti_2d(warped_mask_path)
        rigid_metrics, rigid_metric_timing = compute_registration_metrics(
            fixed_gray,
            np.clip(warped_gray, 0.0, 1.0),
            fixed_mask,
            (warped_mask > 0.5).astype(np.float32),
        )
        gate = strict_pareto_gate_decision(input_metrics, rigid_metrics)
        gate["stage"] = "rigid"
        gate["best_stage_before"] = "input"
        gate["best_stage_after"] = "rigid" if bool(gate.get("accepted")) else "input"
        heatmap_rgb, heatmap_png = _compute_stage_heatmap(
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
        rows = [
            {
                "label": "Input",
                "note": input_note,
                "fixed": gray_preview_panel(fixed_gray),
                "moving": gray_preview_panel(moving_gray),
                "overlay": overlay_preview(fixed_gray, moving_gray, fixed_mask, moving_mask),
                "heatmap": np.full((*fixed_gray.shape, 3), 235, dtype=np.uint8),
                "col_titles": ("Moving", "Fixed", "Overlay", "Warp Field"),
            },
            {
                "label": f"{stage_display_name('rigid')} [{'ACCEPTED' if gate['accepted'] else 'REJECTED'}]",
                "note": (
                    f"{metrics_note(rigid_metrics, rigid_metric_timing, f'{MI_VARIANT} rigid finished')} | "
                    f"gate={'ACCEPTED' if gate['accepted'] else 'REJECTED'} vs input"
                ),
                "fixed": gray_preview_panel(fixed_gray),
                "moving": gray_preview_panel(np.clip(warped_gray, 0.0, 1.0)),
                "overlay": overlay_preview(fixed_gray, np.clip(warped_gray, 0.0, 1.0), fixed_mask, (warped_mask > 0.5).astype(np.float32)),
                "heatmap": heatmap_rgb,
                "col_titles": ("Moving", "Fixed", "Overlay", "Warp Field"),
            },
        ]
        render_storyboard(rows, storyboard_path)
        summary = {
            "pair_key": case.pair_key,
            "group": case.group,
            "variant": MI_VARIANT,
            "input_metrics": input_metrics,
            "rigid_metrics": rigid_metrics,
            "input_metric_timing_seconds": input_metric_timing,
            "rigid_metric_timing_seconds": rigid_metric_timing,
            "gate": gate,
            "rigid_warped_image": str(warped_img_path),
            "rigid_warped_mask": str(warped_mask_path),
            "rigid_heatmap_png": str(heatmap_png),
        }
        manifest_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return storyboard_path, summary


def _compose_side_by_side(mask_storyboard: Path, mi_storyboard: Path, out_path: Path, title: str, summary_lines: list[str]) -> None:
    mask_bgr = cv2.imread(str(mask_storyboard), cv2.IMREAD_COLOR)
    mi_bgr = cv2.imread(str(mi_storyboard), cv2.IMREAD_COLOR)
    if mask_bgr is None or mi_bgr is None:
        raise FileNotFoundError("Failed to read storyboard image.")
    h = max(mask_bgr.shape[0], mi_bgr.shape[0])
    pad = 20
    header_h = 120
    total_w = mask_bgr.shape[1] + mi_bgr.shape[1] + pad * 3
    canvas = np.full((header_h + h + pad * 2, total_w), 255, dtype=np.uint8)
    canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
    cv2.putText(canvas, title, (pad, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (20, 20, 20), 2, cv2.LINE_AA)
    cv2.putText(canvas, "Left: Mask-rigid [grayscale]", (pad, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 90, 160), 2, cv2.LINE_AA)
    cv2.putText(
        canvas,
        "Right: MI-rigid [gradient magnitude]",
        (mask_bgr.shape[1] + pad * 2, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (160, 60, 0),
        2,
        cv2.LINE_AA,
    )
    y = 90
    for line in summary_lines[:2]:
        cv2.putText(canvas, line, (pad, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (55, 55, 55), 1, cv2.LINE_AA)
        y += 22
    canvas[header_h : header_h + mask_bgr.shape[0], pad : pad + mask_bgr.shape[1]] = mask_bgr
    x2 = pad * 2 + mask_bgr.shape[1]
    canvas[header_h : header_h + mi_bgr.shape[0], x2 : x2 + mi_bgr.shape[1]] = mi_bgr
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), canvas)


def _format_case_conclusion(category_title: str, mask_manifest: dict[str, Any], mi_summary: dict[str, Any]) -> list[str]:
    mask_gate = dict((mask_manifest.get("stages") or {}).get("mask_rigid", {}).get("gate") or {})
    mi_gate = dict(mi_summary.get("gate") or {})
    mask_metrics = dict((mask_manifest.get("stages") or {}).get("mask_rigid", {}).get("metrics") or {})
    mi_metrics = dict(mi_summary.get("rigid_metrics") or {})
    lines = [
        f"{category_title}",
        (
            f"Mask-rigid: {'ACCEPTED' if mask_gate.get('accepted') else 'REJECTED'} | "
            f"Dice={float(mask_metrics.get('dice', float('nan'))):.4f} | "
            f"HD95={float(mask_metrics.get('hd95_px', float('nan'))):.2f}"
        ),
        (
            f"MI-rigid: {'ACCEPTED' if mi_gate.get('accepted') else 'REJECTED'} | "
            f"Dice={float(mi_metrics.get('dice', float('nan'))):.4f} | "
            f"HD95={float(mi_metrics.get('hd95_px', float('nan'))):.2f}"
        ),
    ]
    return lines


def main() -> int:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    top_lines = [
        "# Priority 1 Storyboard Comparisons",
        "",
        "- Left panel in each comparison: mask-rigid, shown in ordinary grayscale.",
        f"- Right panel in each comparison: MI-rigid (`{MI_VARIANT}` baseline), shown in gradient magnitude view.",
        f"- Mask benchmark root: `{MASK_BENCH_ROOT}`",
        f"- MI matrix root: `{MATRIX_ROOT}`",
        "",
    ]
    for category in CATEGORIES:
        cat_dir = OUTPUT_ROOT / category["slug"]
        cat_dir.mkdir(parents=True, exist_ok=True)
        (cat_dir / "README.md").write_text(
            "\n".join(
                [
                    f"# {category['title']}",
                    "",
                    category["conclusion"],
                    "",
                    "Display convention:",
                    "- mask-rigid: grayscale",
                    f"- MI-rigid: gradient magnitude (`{MI_VARIANT}`)",
                    "",
                    "Cases:",
                    *[f"- `{case.pair_key}` group `{case.group}`" for case in category["cases"]],
                    "",
                ]
            ),
            encoding="utf-8",
        )
        top_lines.append(f"- [{category['title']}](./{category['slug']}/README.md)")
        for idx, case in enumerate(category["cases"], start=1):
            case_dir = cat_dir / f"{idx:02d}_{case.pair_key}__group_{case.group}"
            case_dir.mkdir(parents=True, exist_ok=True)

            mask_run_dir = _find_mask_run_dir(case)
            mask_storyboard = mask_run_dir / "storyboard.png"
            mask_manifest_path = mask_run_dir / "run_manifest.json"
            mask_manifest = _load_json(mask_manifest_path)
            shutil.copy2(mask_storyboard, case_dir / "mask_rigid_storyboard.png")
            shutil.copy2(mask_manifest_path, case_dir / "mask_rigid_run_manifest.json")

            mi_storyboard, mi_summary = _render_mi_storyboard(case, case_dir)

            summary_lines = _format_case_conclusion(category["title"], mask_manifest, mi_summary)
            _compose_side_by_side(
                case_dir / "mask_rigid_storyboard.png",
                mi_storyboard,
                case_dir / "side_by_side_comparison.png",
                f"{case.pair_key} | group {case.group}",
                summary_lines,
            )

            case_readme = [
                f"# {case.pair_key} | group {case.group}",
                "",
                f"Category: {category['title']}",
                "",
                category["conclusion"],
                "",
                "## Files",
                "",
                "- `side_by_side_comparison.png`",
                "- `mask_rigid_storyboard.png`",
                "- `mi_rigid_storyboard.png`",
                "- `mask_rigid_run_manifest.json`",
                "- `mi_rigid_summary.json`",
                "",
                "## Display Convention",
                "",
                "- `mask_rigid_storyboard.png`: ordinary grayscale",
                f"- `mi_rigid_storyboard.png`: gradient magnitude (`{MI_VARIANT}`)",
                "",
                "## Quick Comparison",
                "",
                f"- {summary_lines[1]}",
                f"- {summary_lines[2]}",
                "",
            ]
            (case_dir / "README.md").write_text("\n".join(case_readme), encoding="utf-8")

    (OUTPUT_ROOT / "README.md").write_text("\n".join(top_lines) + "\n", encoding="utf-8")
    print(f"Exported storyboard cases to {OUTPUT_ROOT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

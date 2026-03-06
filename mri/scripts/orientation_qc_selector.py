#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def normalize_path(path_str: str) -> Path:
    raw = path_str.strip().strip('"')
    if os.name != "nt":
        m = re.match(r"^([A-Za-z]):[\\/](.*)$", raw)
        if m:
            drive = m.group(1).lower()
            tail = m.group(2).replace("\\", "/")
            return Path(f"/mnt/{drive}/{tail}").resolve()
    return Path(raw).expanduser().resolve()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def nii_stem(path: Path) -> str:
    name = path.name
    if name.endswith(".nii.gz"):
        return name[:-7]
    return path.stem


def header_snapshot(img: nib.Nifti1Image) -> dict[str, Any]:
    h = img.header
    spatial_unit, time_unit = h.get_xyzt_units()
    try:
        axcodes_raw = nib.aff2axcodes(img.affine)
    except Exception:
        axcodes_raw = (None, None, None)
    axcodes = [str(x) if x is not None else "?" for x in axcodes_raw]
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
    }


def robust_range(arr: np.ndarray) -> tuple[float, float]:
    vals = arr[np.isfinite(arr)]
    if vals.size == 0:
        return 0.0, 1.0
    non_zero = vals[vals != 0]
    use = non_zero if non_zero.size > 100 else vals
    lo, hi = np.percentile(use, [1, 99]).tolist()
    if hi - lo < 1e-8:
        hi = lo + 1e-6
    return float(lo), float(hi)


def normalize(arr: np.ndarray, lo: float, hi: float) -> np.ndarray:
    out = (arr - lo) / max(hi - lo, 1e-8)
    out = np.clip(out, 0.0, 1.0)
    out[~np.isfinite(out)] = 0.0
    return out


def volume_center(arr: np.ndarray) -> tuple[int, int, int]:
    mask = np.isfinite(arr) & (arr != 0)
    if np.count_nonzero(mask) > 100:
        coords = np.argwhere(mask)
        return tuple(int(np.median(coords[:, i])) for i in range(3))
    return tuple(int((arr.shape[i] - 1) / 2) for i in range(3))


def axis_slice(arr: np.ndarray, axis: int, idx: int) -> np.ndarray:
    if axis == 0:
        sl = arr[idx, :, :]
    elif axis == 1:
        sl = arr[:, idx, :]
    else:
        sl = arr[:, :, idx]
    return np.rot90(sl)


def overlay_pair(fixed_norm: np.ndarray, moving_norm: np.ndarray) -> np.ndarray:
    rgb = np.stack([fixed_norm, fixed_norm, fixed_norm], axis=-1)
    alpha = np.clip(moving_norm, 0.0, 1.0) * 0.68
    rgb[..., 0] = np.clip((1.0 - alpha) * rgb[..., 0] + alpha * 1.0, 0.0, 1.0)
    rgb[..., 1] = np.clip((1.0 - alpha) * rgb[..., 1] + alpha * 0.2, 0.0, 1.0)
    rgb[..., 2] = np.clip((1.0 - alpha) * rgb[..., 2] + alpha * 0.0, 0.0, 1.0)
    return rgb


def apply_flip_axes(arr: np.ndarray, flip_axes: list[int]) -> np.ndarray:
    out = arr
    for axis in flip_axes:
        out = np.flip(out, axis=axis)
    return out


def hconcat(images: list[np.ndarray], gap: int = 2) -> np.ndarray:
    if not images:
        return np.zeros((1, 1, 3), dtype=np.uint8)
    max_h = max(img.shape[0] for img in images)
    padded: list[np.ndarray] = []
    for img in images:
        if img.shape[0] < max_h:
            pad_h = max_h - img.shape[0]
            img = np.pad(img, ((0, pad_h), (0, 0), (0, 0)), mode="constant", constant_values=0)
        padded.append(img)
    if gap <= 0:
        out = np.concatenate(padded, axis=1)
    else:
        spacer = np.zeros((max_h, gap, 3), dtype=np.uint8)
        parts: list[np.ndarray] = []
        for i, img in enumerate(padded):
            if i > 0:
                parts.append(spacer)
            parts.append(img)
        out = np.concatenate(parts, axis=1)
    return out


def build_triview_overlay(
    fixed_arr: np.ndarray,
    moving_arr: np.ndarray,
    *,
    fixed_range: tuple[float, float],
    moving_range: tuple[float, float],
) -> np.ndarray:
    f_center = volume_center(fixed_arr)
    m_center = volume_center(moving_arr)
    panels: list[np.ndarray] = []
    for axis in range(3):
        fs = axis_slice(fixed_arr, axis, f_center[axis])
        ms = axis_slice(moving_arr, axis, m_center[axis])
        rgb = overlay_pair(normalize(fs, *fixed_range), normalize(ms, *moving_range))
        panels.append((rgb * 255.0).astype(np.uint8))
    return hconcat(panels, gap=3)


def to_pixmap(rgb: np.ndarray) -> QtGui.QPixmap:
    arr = np.ascontiguousarray(rgb.astype(np.uint8))
    h, w, _ = arr.shape
    qimg = QtGui.QImage(arr.data, w, h, w * 3, QtGui.QImage.Format_RGB888)
    return QtGui.QPixmap.fromImage(qimg.copy())


def candidate_specs() -> list[dict[str, Any]]:
    combos = [
        [],
        [0],
        [1],
        [2],
        [0, 1],
        [0, 2],
        [1, 2],
        [0, 1, 2],
    ]
    names = {
        "": "none",
        "0": "flip_x",
        "1": "flip_y",
        "2": "flip_z",
        "0,1": "flip_xy",
        "0,2": "flip_xz",
        "1,2": "flip_yz",
        "0,1,2": "flip_xyz",
    }
    out: list[dict[str, Any]] = []
    for idx, axes in enumerate(combos):
        key = ",".join(str(a) for a in axes)
        out.append({"id": idx, "flip_axes": axes, "name": names.get(key, f"flip_{key}")})
    return out


class OrientationQcWindow(QtWidgets.QWidget):
    def __init__(
        self,
        *,
        fixed_path: Path,
        moving_path: Path,
        moving_mask_path: Path | None,
        output_dir: Path,
        output_moving: Path,
        output_moving_mask: Path | None,
        manifest_path: Path,
    ) -> None:
        super().__init__()
        self.setWindowTitle("Orientation QC Selector (moving 8 candidates)")
        self.resize(1500, 980)

        self.fixed_path = fixed_path
        self.moving_path = moving_path
        self.moving_mask_path = moving_mask_path
        self.output_dir = output_dir
        self.output_moving = output_moving
        self.output_moving_mask = output_moving_mask
        self.manifest_path = manifest_path

        self.fixed_img = nib.load(str(fixed_path))
        self.moving_img = nib.load(str(moving_path))
        self.moving_mask_img = nib.load(str(moving_mask_path)) if moving_mask_path else None

        self.fixed_arr = np.asarray(self.fixed_img.get_fdata(), dtype=np.float32)
        self.moving_arr = np.asarray(self.moving_img.get_fdata(), dtype=np.float32)
        self.fixed_range = robust_range(self.fixed_arr)
        self.moving_range = robust_range(self.moving_arr)
        self.candidates = candidate_specs()
        self.button_group = QtWidgets.QButtonGroup(self)
        self.button_group.setExclusive(True)

        self._build_ui()

    def _build_ui(self) -> None:
        root = QtWidgets.QVBoxLayout(self)
        info = QtWidgets.QLabel(
            "Fixed 保持不变；Moving 生成 8 个翻转组合。选择最匹配 fixed 的组合，勾选 verified 后保存。"
        )
        info.setWordWrap(True)
        root.addWidget(info)

        path_text = QtWidgets.QLabel(
            f"Fixed: {self.fixed_path}\n"
            f"Moving: {self.moving_path}\n"
            f"Output moving: {self.output_moving}\n"
            f"Output manifest: {self.manifest_path}"
        )
        path_text.setWordWrap(True)
        root.addWidget(path_text)

        fixed_preview = build_triview_overlay(
            self.fixed_arr,
            self.fixed_arr,
            fixed_range=self.fixed_range,
            moving_range=self.fixed_range,
        )
        fixed_box = QtWidgets.QGroupBox("Fixed tri-view (reference)")
        fixed_layout = QtWidgets.QVBoxLayout(fixed_box)
        fixed_label = QtWidgets.QLabel()
        fixed_label.setPixmap(to_pixmap(fixed_preview))
        fixed_label.setAlignment(QtCore.Qt.AlignCenter)
        fixed_layout.addWidget(fixed_label)
        root.addWidget(fixed_box)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_widget = QtWidgets.QWidget()
        grid = QtWidgets.QGridLayout(scroll_widget)
        grid.setSpacing(10)

        for i, cand in enumerate(self.candidates):
            flipped = apply_flip_axes(self.moving_arr, cand["flip_axes"])
            preview = build_triview_overlay(
                self.fixed_arr,
                flipped,
                fixed_range=self.fixed_range,
                moving_range=self.moving_range,
            )
            box = QtWidgets.QGroupBox(f"Candidate {cand['id']}: {cand['name']}")
            vbox = QtWidgets.QVBoxLayout(box)
            radio = QtWidgets.QRadioButton(f"Use candidate {cand['id']} ({cand['name']})")
            self.button_group.addButton(radio, cand["id"])
            if i == 0:
                radio.setChecked(True)
            vbox.addWidget(radio)

            img_label = QtWidgets.QLabel()
            img_label.setPixmap(to_pixmap(preview))
            img_label.setAlignment(QtCore.Qt.AlignCenter)
            vbox.addWidget(img_label)

            row = i // 2
            col = i % 2
            grid.addWidget(box, row, col)

        scroll.setWidget(scroll_widget)
        root.addWidget(scroll, stretch=1)

        self.verified_box = QtWidgets.QCheckBox("Mark as orientation/axis verified for downstream processing")
        self.verified_box.setChecked(True)
        root.addWidget(self.verified_box)

        btn_row = QtWidgets.QHBoxLayout()
        btn_cancel = QtWidgets.QPushButton("Cancel")
        btn_cancel.clicked.connect(self.close)
        btn_save = QtWidgets.QPushButton("Save selected orientation")
        btn_save.clicked.connect(self._save_selected)
        btn_row.addWidget(btn_cancel)
        btn_row.addStretch(1)
        btn_row.addWidget(btn_save)
        root.addLayout(btn_row)

    def _save_selected(self) -> None:
        cand_id = self.button_group.checkedId()
        cand = next((c for c in self.candidates if int(c["id"]) == int(cand_id)), None)
        if cand is None:
            QtWidgets.QMessageBox.warning(self, "No candidate", "请选择一个 moving 方向候选。")
            return

        if not self.verified_box.isChecked():
            answer = QtWidgets.QMessageBox.question(
                self,
                "未勾选 verified",
                "你没有勾选 verified。仍然保存吗？",
            )
            if answer != QtWidgets.QMessageBox.Yes:
                return

        self.output_dir.mkdir(parents=True, exist_ok=True)

        moving_data_raw = np.asarray(self.moving_img.dataobj)
        moving_out_data = apply_flip_axes(moving_data_raw, cand["flip_axes"])
        moving_hdr = self.moving_img.header.copy()
        moving_out = nib.Nifti1Image(moving_out_data, self.moving_img.affine, moving_hdr)
        moving_qf, moving_qc = self.moving_img.get_qform(coded=True)
        moving_sf, moving_sc = self.moving_img.get_sform(coded=True)
        moving_out.set_qform(moving_qf if moving_qf is not None else self.moving_img.affine, code=int(moving_qc))
        moving_out.set_sform(moving_sf if moving_sf is not None else self.moving_img.affine, code=int(moving_sc))
        nib.save(moving_out, str(self.output_moving))

        output_moving_mask_path = ""
        if self.moving_mask_img is not None and self.output_moving_mask is not None:
            mask_data_raw = np.asarray(self.moving_mask_img.dataobj)
            mask_out_data = apply_flip_axes(mask_data_raw, cand["flip_axes"])
            mask_hdr = self.moving_mask_img.header.copy()
            mask_out = nib.Nifti1Image(mask_out_data, self.moving_mask_img.affine, mask_hdr)
            mask_qf, mask_qc = self.moving_mask_img.get_qform(coded=True)
            mask_sf, mask_sc = self.moving_mask_img.get_sform(coded=True)
            mask_out.set_qform(mask_qf if mask_qf is not None else self.moving_mask_img.affine, code=int(mask_qc))
            mask_out.set_sform(mask_sf if mask_sf is not None else self.moving_mask_img.affine, code=int(mask_sc))
            nib.save(mask_out, str(self.output_moving_mask))
            output_moving_mask_path = str(self.output_moving_mask)

        out_moving_img = nib.load(str(self.output_moving))
        out_mask_img = nib.load(output_moving_mask_path) if output_moving_mask_path else None
        manifest = {
            "run_id": datetime.now().strftime("RUN_ORIENTQC_%Y%m%d_%H%M%S"),
            "generated_at": now_iso(),
            "fixed_path": str(self.fixed_path),
            "input_moving": str(self.moving_path),
            "input_moving_mask": str(self.moving_mask_path) if self.moving_mask_path else "",
            "output_moving": str(self.output_moving),
            "output_moving_mask": output_moving_mask_path,
            "verified": bool(self.verified_box.isChecked()),
            "selected_candidate": cand,
            "all_candidates": self.candidates,
            "adjustment_trace": [
                {
                    "field": "moving_voxel_axis_flip",
                    "old": "identity",
                    "new": f"flip_axes={cand['flip_axes']}",
                    "reason": "manual fixed-vs-moving tri-view orientation QC selection",
                }
            ],
            "header_before": {
                "fixed": header_snapshot(self.fixed_img),
                "moving": header_snapshot(self.moving_img),
                "moving_mask": header_snapshot(self.moving_mask_img) if self.moving_mask_img else {},
            },
            "header_after": {
                "moving": header_snapshot(out_moving_img),
                "moving_mask": header_snapshot(out_mask_img) if out_mask_img else {},
            },
            "sha256": {
                "fixed": sha256_file(self.fixed_path),
                "input_moving": sha256_file(self.moving_path),
                "input_moving_mask": sha256_file(self.moving_mask_path) if self.moving_mask_path else "",
                "output_moving": sha256_file(self.output_moving),
                "output_moving_mask": sha256_file(Path(output_moving_mask_path)) if output_moving_mask_path else "",
            },
            "notes": [
                "this operation is a voxel-axis flip redefinition for moving image",
                "moving affine/qform/sform are preserved intentionally; voxel content is flipped per selected candidate",
                "pass this manifest to resample_contract --moving-orientation-manifest for provenance tracking",
            ],
        }
        self.manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

        QtWidgets.QMessageBox.information(
            self,
            "Saved",
            f"[OK] output moving:\n{self.output_moving}\n\n[OK] manifest:\n{self.manifest_path}",
        )
        self.close()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Manual orientation QC selector: fixed stays unchanged; moving shows 8 axis-flip candidates. "
            "User selects one candidate and saves orientation-verified outputs."
        )
    )
    p.add_argument("--fixed", required=True)
    p.add_argument("--moving", required=True)
    p.add_argument("--moving-mask", default="")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--output-moving", default="")
    p.add_argument("--output-moving-mask", default="")
    p.add_argument("--manifest", default="")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    fixed = normalize_path(args.fixed)
    moving = normalize_path(args.moving)
    moving_mask = normalize_path(args.moving_mask) if args.moving_mask.strip() else None
    out_dir = normalize_path(args.output_dir)

    if not fixed.exists():
        raise FileNotFoundError(f"fixed not found: {fixed}")
    if not moving.exists():
        raise FileNotFoundError(f"moving not found: {moving}")
    if moving_mask and not moving_mask.exists():
        raise FileNotFoundError(f"moving mask not found: {moving_mask}")

    out_moving = normalize_path(args.output_moving) if args.output_moving.strip() else out_dir / f"{nii_stem(moving)}_orientation_verified.nii.gz"
    out_mask = None
    if moving_mask is not None:
        out_mask = (
            normalize_path(args.output_moving_mask)
            if args.output_moving_mask.strip()
            else out_dir / f"{nii_stem(moving_mask)}_orientation_verified.nii.gz"
        )
    manifest = normalize_path(args.manifest) if args.manifest.strip() else out_dir / "orientation_qc_manifest.json"

    app = QtWidgets.QApplication(sys.argv)
    window = OrientationQcWindow(
        fixed_path=fixed,
        moving_path=moving,
        moving_mask_path=moving_mask,
        output_dir=out_dir,
        output_moving=out_moving,
        output_moving_mask=out_mask,
        manifest_path=manifest,
    )
    window.show()
    code = app.exec()
    if code == 0 and manifest.exists():
        print(f"[OK] orientation manifest: {manifest}")
        print(f"[OK] output moving: {out_moving}")
        if out_mask and out_mask.exists():
            print(f"[OK] output moving mask: {out_mask}")
    sys.exit(code)


if __name__ == "__main__":
    main()


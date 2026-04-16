from __future__ import annotations

from collections import OrderedDict
import ctypes
from dataclasses import dataclass
from datetime import datetime, timezone
import os
import threading
import traceback
from pathlib import Path
import shlex
import sys
from time import perf_counter
from typing import Callable
import json

import cv2
import numpy as np
from PySide6.QtCore import QObject, QSignalBlocker, QThread, QRectF, Qt, Signal, QTimer
from PySide6.QtGui import QColor, QPen, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGraphicsRectItem,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QStackedWidget,
    QTextEdit,
    QMessageBox,
    QVBoxLayout,
    QWidget,
)

from ..application import WorkflowService
from ..application.section_workspace import (
    WorkspaceSection,
    find_external_prediction_mask,
    list_workspace_sections,
    load_masks_from_label_path,
    load_workspace_prediction_input,
    load_workspace_section,
    prepare_workspace_work_image,
    save_workspace_review,
    write_mask_labels_file,
    write_workspace_prediction,
)
from ..application.pair_workspace import (
    WorkspacePair,
    default_pair_registration_masks_root,
    default_pair_registry_path,
    list_cross_stain_pairs,
    load_pair_registry,
    pair_registration_mask_paths,
    save_pair_registry,
)
from ..application.pair_registration import (
    PairRegistrationConfig,
    component_rank_map,
    default_pair_registration_runs_root,
    find_ants_bin,
    keep_group,
    latest_registration_run_dir,
    registration_support_mask,
    run_pair_registration,
)
from ..application.roi_mapping import (
    current_step6_state,
    default_pair_roi_root,
    load_approved_registration_context,
    map_step7_full_crop_polygon_to_step6_side,
    map_step7_scene_polygon_to_step6_side,
    map_step7_scene_mask_to_step6_side,
    save_step6_roi as save_step6_roi_outputs,
    update_step6_roi_mapping,
)
from ..application.confocal_registration import (
    ConfocalAutoScaleConfig,
    ConfocalFrontierConfig,
    ConfocalSeedScreenConfig,
    ConfocalRigidConfig,
    STEP7_REGISTRATION_INPUT_PROFILE,
    STEP7_TARGET_UM_PER_PX,
    analyze_confocal_duplicate_stacks,
    build_confocal_tile_defs,
    build_step7_scene_fov_masks,
    _invert_confocal_u8,
    _resample_mask_to_target_um_per_px,
    default_confocal_registration_root,
    export_confocal_step7_session,
    export_confocal_full_report,
    load_confocal_projection,
    load_step7_handoff_payload,
    prepare_myelin_confocal_fixed_bundle,
    _resample_projection_to_target_um_per_px,
    run_confocal_auto_scale_sweep,
    run_confocal_frontier_propagation,
    run_confocal_seed_tile_screening,
    run_confocal_rigid_registration,
)
from ..db import connect_db, transaction
from ..domain import LoadedSlide, ProposalBox
from ..pipeline_adapters import (
    MASK_COMPUTE_PROFILE_FAST,
    MASK_COMPUTE_PROFILE_FULL,
    MASK_COMPUTE_PROFILE_STANDARD,
    MASK_PRESET_HYBRID_BALANCED,
    MASK_PRESET_M3_HYST_ENTRES_GUARD,
    MASK_PRESET_LATEST_CONTEXTUAL,
    MASK_PRESET_LEGACY_SIMPLE,
    compute_auto_masks,
    compute_auto_masks_with_info,
    compute_auto_masks_resampled,
    default_mask_preset_for_stain,
    extract_crop_for_preview,
    mask_compute_profile_max_long_edge,
)
from ..pipeline_adapters.slide_io import effective_crop_rect_overview, extract_level0_bbox_rgb, load_slide_header_only, open_slide_handle
from ..repositories import RevisionRepository, SectionRepository
from ..widgets.graphics import ConfocalAlignmentView, DraggableProposalItem, ImageSceneView, qimage_from_rgb_array
from ..widgets.mask_editor import MaskEditorLabel
from ..widgets.proposal_card import ProposalCard


if sys.platform == "win32":
    try:
        import winsound
    except Exception:  # pragma: no cover - defensive on Windows env mismatch
        winsound = None

    class _FLASHWINFO(ctypes.Structure):
        _fields_ = [
            ("cbSize", ctypes.c_uint),
            ("hwnd", ctypes.c_void_p),
            ("dwFlags", ctypes.c_uint),
            ("uCount", ctypes.c_uint),
            ("dwTimeout", ctypes.c_uint),
        ]


    _FLASHW_TRAY = 0x00000002
    _FLASHW_TIMERNOFG = 0x0000000C


    def _flash_taskbar_icon(hwnd: int) -> bool:
        if hwnd <= 0:
            return False
        try:
            flash = _FLASHWINFO(
                cbSize=ctypes.sizeof(_FLASHWINFO),
                hwnd=ctypes.c_void_p(hwnd),
                dwFlags=_FLASHW_TRAY | _FLASHW_TIMERNOFG,
                uCount=8,
                dwTimeout=0,
            )
            return bool(ctypes.windll.user32.FlashWindowEx(ctypes.byref(flash)))
        except Exception:
            return False


    def _play_attention_sound() -> None:
        if winsound is None:
            return
        try:
            winsound.MessageBeep(winsound.MB_ICONASTERISK)
        except Exception:
            pass


else:
    def _flash_taskbar_icon(hwnd: int) -> bool:
        return False


    def _play_attention_sound() -> None:
        return None


def _fit_step7_procrustes(src_xy: np.ndarray, dst_xy: np.ndarray, *, allow_scale: bool) -> dict[str, object]:
    src = np.asarray(src_xy, dtype=np.float64)
    dst = np.asarray(dst_xy, dtype=np.float64)
    src_ctr = src.mean(axis=0)
    dst_ctr = dst.mean(axis=0)
    src_c = src - src_ctr
    dst_c = dst - dst_ctr
    cov = src_c.T @ dst_c
    u, s, vt = np.linalg.svd(cov)
    rot = vt.T @ u.T
    if np.linalg.det(rot) < 0:
        vt[-1, :] *= -1.0
        rot = vt.T @ u.T
    scale = 1.0
    if allow_scale:
        denom = float(np.sum(src_c**2))
        if denom > 1e-8:
            scale = float(np.sum(s) / denom)
    pred = (scale * (src @ rot.T)) + (dst_ctr - scale * (src_ctr @ rot.T))
    residual_vec = pred - dst
    residual_norm = np.linalg.norm(residual_vec, axis=1)
    translation = dst_ctr - scale * (src_ctr @ rot.T)
    return {
        "scale": float(scale),
        "rotation_matrix": rot.astype(float).tolist(),
        "rotation_deg": float(np.degrees(np.arctan2(rot[1, 0], rot[0, 0]))),
        "translation_xy": translation.astype(float).tolist(),
        "pred_xy": pred.astype(float).tolist(),
        "residual_xy": residual_vec.astype(float).tolist(),
        "residual_norm": residual_norm.astype(float).tolist(),
        "rms_px": float(np.sqrt(np.mean(residual_norm**2))) if residual_norm.size else float("nan"),
        "mean_px": float(np.mean(residual_norm)) if residual_norm.size else float("nan"),
        "max_px": float(np.max(residual_norm)) if residual_norm.size else float("nan"),
    }


def _fit_step7_affine(src_xy: np.ndarray, dst_xy: np.ndarray) -> dict[str, object]:
    src = np.asarray(src_xy, dtype=np.float64)
    dst = np.asarray(dst_xy, dtype=np.float64)
    src_aug = np.concatenate([src, np.ones((src.shape[0], 1), dtype=np.float64)], axis=1)
    coeff, _, _, _ = np.linalg.lstsq(src_aug, dst, rcond=None)
    pred = src_aug @ coeff
    residual_vec = pred - dst
    residual_norm = np.linalg.norm(residual_vec, axis=1)
    mat = coeff.T
    linear = mat[:, :2]
    col0 = linear[:, 0]
    col1 = linear[:, 1]
    scale_x = float(np.linalg.norm(col0))
    shear = float(np.dot(col0 / max(scale_x, 1e-8), col1)) if scale_x > 1e-8 else 0.0
    return {
        "matrix_2x3": mat.astype(float).tolist(),
        "pred_xy": pred.astype(float).tolist(),
        "residual_xy": residual_vec.astype(float).tolist(),
        "residual_norm": residual_norm.astype(float).tolist(),
        "rms_px": float(np.sqrt(np.mean(residual_norm**2))) if residual_norm.size else float("nan"),
        "mean_px": float(np.mean(residual_norm)) if residual_norm.size else float("nan"),
        "max_px": float(np.max(residual_norm)) if residual_norm.size else float("nan"),
        "scale_x_like": scale_x,
        "scale_y_like": float(np.linalg.norm(col1)),
        "shear_like": shear,
    }


class SaveRevisionWorker(QObject):
    finished = Signal(str)
    failed = Signal(str)

    def __init__(
        self,
        *,
        db_path: Path,
        workspace_root: Path,
        section_uid: str,
        tissue_mask: np.ndarray,
        artifact_mask: np.ndarray,
        mirror_enabled: bool,
        bbox_overview: dict,
        notes: str = "",
    ) -> None:
        super().__init__()
        self.db_path = db_path
        self.workspace_root = workspace_root
        self.section_uid = section_uid
        self.tissue_mask = tissue_mask
        self.artifact_mask = artifact_mask
        self.mirror_enabled = mirror_enabled
        self.bbox_overview = bbox_overview
        self.notes = notes

    def run(self) -> None:
        conn = None
        try:
            conn = connect_db(self.db_path)
            revision_repository = RevisionRepository(conn, self.workspace_root)
            section_repository = SectionRepository(conn)
            with transaction(conn):
                revision_id = revision_repository.create_mask_revision(
                    section_uid=self.section_uid,
                    tissue_mask=self.tissue_mask,
                    artifact_mask=self.artifact_mask,
                    mirror_enabled=self.mirror_enabled,
                    bbox_overview=self.bbox_overview,
                    notes=self.notes,
                )
                current_version = section_repository.get_manual_mask_version(self.section_uid)
                section_repository.update_review_state(
                    section_uid=self.section_uid,
                    mirror_state="mirrored_lr" if self.mirror_enabled else "original",
                    review_status="mask_reviewed",
                    manual_mask_version=current_version + 1,
                    notes=self.notes,
                )
            self.finished.emit(revision_id)
        except Exception:
            self.failed.emit(f"Save revision failed:\n{traceback.format_exc()}")
        finally:
            if conn is not None:
                conn.close()


def _clone_proposal_snapshot(proposal: ProposalBox) -> ProposalBox:
    return ProposalBox(
        label=proposal.label,
        stain=proposal.stain,
        sample_id=proposal.sample_id,
        section_id=proposal.section_id,
        proposal_rank=proposal.proposal_rank,
        x=int(proposal.x),
        y=int(proposal.y),
        w=int(proposal.w),
        h=int(proposal.h),
        mask_preset=proposal.mask_preset,
        mirror_enabled=proposal.mirror_enabled,
    )


@dataclass
class _PrecomputeTask:
    generation: int
    proposal_index: int
    proposal_label: str
    target_proposal: ProposalBox
    all_proposals: list[ProposalBox]
    mask_method: str


class BackgroundPrecomputeWorker(QObject):
    status = Signal(str)
    section_ready = Signal(int, object, int, str)
    idle = Signal(int)
    failed = Signal(str)

    def __init__(self, auto_mask_cache) -> None:
        super().__init__()
        self.auto_mask_cache = auto_mask_cache
        self._condition = threading.Condition()
        self._stop = False
        self._paused = True
        self._loaded_slide: LoadedSlide | None = None
        self._queue: list[_PrecomputeTask] = []
        self._crop_level = 4
        self._generation = 0
        self._slide_handle = None
        self._slide_key: tuple | None = None

    def configure(
        self,
        loaded_slide: LoadedSlide,
        tasks: list[_PrecomputeTask],
        *,
        crop_level: int,
        generation: int,
    ) -> None:
        with self._condition:
            self._loaded_slide = loaded_slide
            self._queue = list(tasks)
            self._crop_level = int(crop_level)
            self._generation = int(generation)
            self._condition.notify_all()

    def set_paused(self, paused: bool) -> None:
        with self._condition:
            self._paused = bool(paused)
            self._condition.notify_all()

    def stop(self) -> None:
        with self._condition:
            self._stop = True
            self._condition.notify_all()

    def run(self) -> None:
        try:
            while True:
                with self._condition:
                    while not self._stop and (self._paused or not self._queue or self._loaded_slide is None):
                        self._condition.wait(timeout=0.25)
                    if self._stop:
                        break
                    task = self._queue.pop(0)
                    loaded_slide = self._loaded_slide
                    crop_level = self._crop_level
                    generation = self._generation

                if loaded_slide is None:
                    continue

                try:
                    self._ensure_slide_handle(loaded_slide)
                    self.status.emit(f"Background precompute: {task.proposal_label} ...")
                    crop_rgb = extract_crop_for_preview(
                        loaded_slide,
                        task.target_proposal,
                        crop_level=crop_level,
                        slide_handle=self._slide_handle,
                    )
                    cache_state = "hit"
                    if not self.auto_mask_cache.contains(
                        loaded_slide,
                        task.target_proposal,
                        task.all_proposals,
                        crop_level=crop_level,
                        mask_method=task.mask_method,
                    ):
                        tissue, artifact = compute_auto_masks(
                            crop_rgb,
                            loaded_slide.stain,
                            method=task.mask_method,
                            loaded_slide=loaded_slide,
                            target_proposal=task.target_proposal,
                            all_proposals=task.all_proposals,
                            crop_level=crop_level,
                        )
                        self.auto_mask_cache.put(
                            loaded_slide,
                            task.target_proposal,
                            task.all_proposals,
                            crop_level=crop_level,
                            mask_method=task.mask_method,
                            tissue=tissue,
                            artifact=artifact,
                        )
                        cache_state = "computed"
                    self.section_ready.emit(task.proposal_index, crop_rgb, generation, cache_state)
                except Exception:
                    self.failed.emit(f"Background precompute failed for {task.proposal_label}:\n{traceback.format_exc()}")

                with self._condition:
                    is_idle = not self._queue and generation == self._generation
                if is_idle:
                    self.idle.emit(generation)
        finally:
            self._close_slide_handle()

    def _ensure_slide_handle(self, loaded_slide: LoadedSlide) -> None:
        slide_key = (
            str(loaded_slide.slide_path.resolve()),
            loaded_slide.backend,
            loaded_slide.slide_path.stat().st_size,
            int(getattr(loaded_slide.slide_path.stat(), "st_mtime_ns", int(loaded_slide.slide_path.stat().st_mtime * 1e9))),
        )
        if self._slide_handle is not None and self._slide_key == slide_key:
            return
        self._close_slide_handle()
        self._slide_handle = open_slide_handle(loaded_slide)
        self._slide_key = slide_key

    def _close_slide_handle(self) -> None:
        if self._slide_handle is not None:
            try:
                self._slide_handle.close()
            except Exception:
                pass
        self._slide_handle = None
        self._slide_key = None


class MaskPredictionWorker(QObject):
    progress = Signal(str)
    stage_progress = Signal(object)
    finished = Signal(object)
    failed = Signal(str)

    def __init__(
        self,
        items: list[WorkspaceSection],
        *,
        mask_method_override: str | None = None,
        compute_profile: str = MASK_COMPUTE_PROFILE_STANDARD,
    ) -> None:
        super().__init__()
        self.items = items
        self.mask_method_override = mask_method_override
        self.compute_profile = compute_profile

    @staticmethod
    def _algorithm_version(stain: str, method: str) -> str:
        stain_key = stain.lower()
        if method == MASK_PRESET_LEGACY_SIMPLE:
            return "gui_legacy_simple_autoseg_v1"
        if method == MASK_PRESET_M3_HYST_ENTRES_GUARD:
            return "gui_m3_hyst_entres_guard_autoseg_v1"
        if method == MASK_PRESET_HYBRID_BALANCED:
            return "gui_hybrid_balanced_autoseg_v2"
        if stain_key == "nissl":
            return "gui_nissl_tool_baseline_autoseg_v1"
        if stain_key == "gallyas":
            return "gui_contextual_autoseg_v1"
        return "gui_simple_autoseg_v1"

    def run(self) -> None:
        try:
            predicted: list[str] = []
            total = len(self.items)
            for idx, item in enumerate(self.items, start=1):
                mask_method = self.mask_method_override or default_mask_preset_for_stain(item.stain)
                section_t0 = perf_counter()
                step_count = 4

                load_t0 = perf_counter()
                self.stage_progress.emit(
                    {
                        "item_label": item.label,
                        "item_index": idx,
                        "total_items": total,
                        "step_index": 1,
                        "step_count": step_count,
                        "stage": "load_work_image",
                        "stage_elapsed_s": 0.0,
                        "section_elapsed_s": 0.0,
                        "progress_percent": int(round(((idx - 1) + 0.02) / max(1, total) * 100)),
                    }
                )
                _, work_rgb, work_info = load_workspace_prediction_input(item, compute_profile=self.compute_profile)
                load_s = perf_counter() - load_t0
                self.stage_progress.emit(
                    {
                        "item_label": item.label,
                        "item_index": idx,
                        "total_items": total,
                        "step_index": 1,
                        "step_count": step_count,
                        "stage": "load_work_image",
                        "stage_elapsed_s": round(load_s, 3),
                        "section_elapsed_s": round(perf_counter() - section_t0, 3),
                        "progress_percent": int(round(((idx - 1) + 0.20) / max(1, total) * 100)),
                    }
                )

                compute_t0 = perf_counter()
                self.stage_progress.emit(
                    {
                        "item_label": item.label,
                        "item_index": idx,
                        "total_items": total,
                        "step_index": 2,
                        "step_count": step_count,
                        "stage": "compute_mask",
                        "stage_elapsed_s": 0.0,
                        "section_elapsed_s": round(perf_counter() - section_t0, 3),
                        "progress_percent": int(round(((idx - 1) + 0.28) / max(1, total) * 100)),
                    }
                )
                tissue, artifact, algo_info = compute_auto_masks_with_info(
                    work_rgb,
                    item.stain,
                    method=mask_method,
                    input_scale=float(work_info.get("working_scale", 1.0)),
                )
                compute_s = perf_counter() - compute_t0
                self.stage_progress.emit(
                    {
                        "item_label": item.label,
                        "item_index": idx,
                        "total_items": total,
                        "step_index": 2,
                        "step_count": step_count,
                        "stage": "compute_mask",
                        "stage_elapsed_s": round(compute_s, 3),
                        "section_elapsed_s": round(perf_counter() - section_t0, 3),
                        "progress_percent": int(round(((idx - 1) + 0.62) / max(1, total) * 100)),
                    }
                )

                upscale_t0 = perf_counter()
                self.stage_progress.emit(
                    {
                        "item_label": item.label,
                        "item_index": idx,
                        "total_items": total,
                        "step_index": 3,
                        "step_count": step_count,
                        "stage": "upscale_masks",
                        "stage_elapsed_s": 0.0,
                        "section_elapsed_s": round(perf_counter() - section_t0, 3),
                        "progress_percent": int(round(((idx - 1) + 0.70) / max(1, total) * 100)),
                    }
                )
                orig_h, orig_w = [int(v) for v in work_info.get("original_shape_hw", list(work_rgb.shape[:2]))]
                if (orig_h, orig_w) != work_rgb.shape[:2]:
                    tissue = cv2.resize(tissue.astype(np.uint8), (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
                    artifact = cv2.resize(artifact.astype(np.uint8), (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
                upscale_s = perf_counter() - upscale_t0
                self.stage_progress.emit(
                    {
                        "item_label": item.label,
                        "item_index": idx,
                        "total_items": total,
                        "step_index": 3,
                        "step_count": step_count,
                        "stage": "upscale_masks",
                        "stage_elapsed_s": round(upscale_s, 3),
                        "section_elapsed_s": round(perf_counter() - section_t0, 3),
                        "progress_percent": int(round(((idx - 1) + 0.82) / max(1, total) * 100)),
                    }
                )

                write_t0 = perf_counter()
                self.stage_progress.emit(
                    {
                        "item_label": item.label,
                        "item_index": idx,
                        "total_items": total,
                        "step_index": 4,
                        "step_count": step_count,
                        "stage": "write_outputs",
                        "stage_elapsed_s": 0.0,
                        "section_elapsed_s": round(perf_counter() - section_t0, 3),
                        "progress_percent": int(round(((idx - 1) + 0.88) / max(1, total) * 100)),
                    }
                )
                compute_info = {
                    **dict(work_info),
                    **dict(algo_info or {}),
                    "compute_profile": self.compute_profile,
                    "timing_s": {
                        "load_work_image": round(load_s, 3),
                        "compute_mask": round(compute_s, 3),
                        "upscale_masks": round(upscale_s, 3),
                    },
                }
                write_workspace_prediction(
                    item,
                    tissue,
                    artifact,
                    mask_preset=mask_method,
                    mask_algorithm_version=self._algorithm_version(item.stain, mask_method),
                    mask_compute_profile=self.compute_profile,
                    compute_info=compute_info,
                )
                write_s = perf_counter() - write_t0
                section_total_s = perf_counter() - section_t0
                self.stage_progress.emit(
                    {
                        "item_label": item.label,
                        "item_index": idx,
                        "total_items": total,
                        "step_index": 4,
                        "step_count": step_count,
                        "stage": "write_outputs",
                        "stage_elapsed_s": round(write_s, 3),
                        "section_elapsed_s": round(section_total_s, 3),
                        "progress_percent": int(round(((idx - 1) + 1.0) / max(1, total) * 100)),
                    }
                )
                predicted.append(item.label)
                self.progress.emit(
                    f"Predicted: {item.label} | profile={self.compute_profile} | "
                    f"load={load_s:.2f}s compute={compute_s:.2f}s upscale={upscale_s:.2f}s write={write_s:.2f}s total={section_total_s:.2f}s"
                )
            self.finished.emit({"predicted": predicted, "compute_profile": self.compute_profile})
        except Exception:
            self.failed.emit(f"Mask prediction failed:\n{traceback.format_exc()}")


class DownsamplePrepareWorker(QObject):
    progress = Signal(str)
    stage_progress = Signal(object)
    finished = Signal(object)
    failed = Signal(str)

    def __init__(
        self,
        items: list[WorkspaceSection],
        *,
        compute_profile: str = MASK_COMPUTE_PROFILE_STANDARD,
    ) -> None:
        super().__init__()
        self.items = items
        self.compute_profile = compute_profile

    def run(self) -> None:
        try:
            prepared: list[str] = []
            total = len(self.items)
            max_long_edge = mask_compute_profile_max_long_edge(self.compute_profile)
            for idx, item in enumerate(self.items, start=1):
                section_t0 = perf_counter()
                self.stage_progress.emit(
                    {
                        "item_label": item.label,
                        "item_index": idx,
                        "total_items": total,
                        "step_index": 1,
                        "step_count": 1,
                        "stage": "prepare_work_image",
                        "stage_elapsed_s": 0.0,
                        "section_elapsed_s": 0.0,
                        "progress_percent": int(round(((idx - 1) + 0.05) / max(1, total) * 100)),
                    }
                )
                info = prepare_workspace_work_image(
                    item,
                    compute_profile=self.compute_profile,
                    max_long_edge=max_long_edge,
                )
                section_total_s = perf_counter() - section_t0
                self.stage_progress.emit(
                    {
                        "item_label": item.label,
                        "item_index": idx,
                        "total_items": total,
                        "step_index": 1,
                        "step_count": 1,
                        "stage": "prepare_work_image",
                        "stage_elapsed_s": round(section_total_s, 3),
                        "section_elapsed_s": round(section_total_s, 3),
                        "progress_percent": int(round(((idx - 1) + 1.0) / max(1, total) * 100)),
                    }
                )
                prepared.append(item.label)
                timings = dict(info.get("timing_s") or {})
                self.progress.emit(
                    f"Prepared work image: {item.label} | profile={self.compute_profile} | "
                    f"shape={info.get('working_shape_hw')} scale={info.get('working_scale'):.4f} "
                    f"| load={float(timings.get('load', 0.0)):.2f}s resize={float(timings.get('resize', 0.0)):.2f}s "
                    f"write={float(timings.get('write', 0.0)):.2f}s total={section_total_s:.2f}s"
                )
            self.finished.emit({"prepared": prepared, "compute_profile": self.compute_profile})
        except Exception:
            self.failed.emit(f"Downsample preparation failed:\n{traceback.format_exc()}")


class PairRegistrationWorker(QObject):
    progress = Signal(str)
    stage_update = Signal(object)
    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, cfg: PairRegistrationConfig) -> None:
        super().__init__()
        self.cfg = cfg

    def run(self) -> None:
        try:
            summary = run_pair_registration(
                self.cfg,
                progress_cb=self.stage_update.emit,
            )
            self.finished.emit(summary)
        except Exception:
            self.failed.emit(f"Step 5 registration failed:\n{traceback.format_exc()}")


class ConfocalRigidWorker(QObject):
    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, cfg: ConfocalRigidConfig) -> None:
        super().__init__()
        self.cfg = cfg

    def run(self) -> None:
        try:
            summary = run_confocal_rigid_registration(self.cfg)
            self.finished.emit(summary)
        except Exception:
            self.failed.emit(f"Step 7 confocal registration failed:\n{traceback.format_exc()}")


class ConfocalSeedScreenWorker(QObject):
    stage_progress = Signal(object)
    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, cfg: ConfocalSeedScreenConfig) -> None:
        super().__init__()
        self.cfg = cfg

    def run(self) -> None:
        try:
            summary = run_confocal_seed_tile_screening(
                self.cfg,
                progress_cb=self.stage_progress.emit,
            )
            self.finished.emit(summary)
        except Exception:
            self.failed.emit(f"Step 7 seed screening failed:\n{traceback.format_exc()}")


class ConfocalAutoScaleWorker(QObject):
    stage_progress = Signal(object)
    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, cfg: ConfocalAutoScaleConfig) -> None:
        super().__init__()
        self.cfg = cfg

    def run(self) -> None:
        try:
            summary = run_confocal_auto_scale_sweep(
                self.cfg,
                progress_cb=self.stage_progress.emit,
            )
            self.finished.emit(summary)
        except Exception:
            self.failed.emit(f"Step 7 auto scale sweep failed:\n{traceback.format_exc()}")


class ConfocalFrontierWorker(QObject):
    stage_progress = Signal(object)
    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, cfg: ConfocalFrontierConfig) -> None:
        super().__init__()
        self.cfg = cfg

    def run(self) -> None:
        try:
            summary = run_confocal_frontier_propagation(
                self.cfg,
                progress_cb=self.stage_progress.emit,
            )
            self.finished.emit(summary)
        except Exception:
            self.failed.emit(f"Step 7 frontier propagation failed:\n{traceback.format_exc()}")


class WorkflowWindow(QWidget):
    PAGE_HOME = 0
    PAGE_STAGE1 = 1
    PAGE_STAGE2 = 2
    PAGE_STAGE3 = 3
    PAGE_STAGE4 = 4
    PAGE_STAGE5 = 5
    PAGE_STAGE6 = 6
    PAGE_STAGE7 = 7
    PAGE_STAGE8 = 8

    def __init__(self, workflow_service: WorkflowService) -> None:
        super().__init__()
        self.setWindowTitle("Histology HITL Workflow")
        self.workflow_service = workflow_service

        self.current_folder: Path | None = None
        self.current_proposal_index: int = 0
        self.workspace_root: Path | None = None
        self.review_mask_root: Path | None = None
        self.workspace_sections: list[WorkspaceSection] = []
        self.current_workspace_index: int = 0
        self.step4_myelin_root: Path | None = None
        self.step4_nissl_root: Path | None = None
        self.step4_pairs: list[WorkspacePair] = []
        self.current_pair_index: int = 0
        self.step4_pair_registry: dict[str, dict] = {}
        self.step4_all_pairs: list[WorkspacePair] = []
        self.step4_active_editor: MaskEditorLabel | None = None
        self.step4_active_side: str = "myelin"
        self.step4_myelin_crop_rgb: np.ndarray | None = None
        self.step4_nissl_crop_rgb: np.ndarray | None = None
        self.step4_component_groups: dict[str, dict[int, int]] = {"myelin": {}, "nissl": {}}
        self.step4_group_flips: dict[str, dict[int, bool]] = {"myelin": {}, "nissl": {}}
        self.step4_preserve_component_marks_once: dict[str, bool] = {"myelin": False, "nissl": False}
        self.step4_pair_cache_lock = threading.Lock()
        self.step4_pair_cache_generation: int = 0
        self.step4_pair_cache: dict[str, dict[str, tuple[dict, np.ndarray, np.ndarray, np.ndarray, dict[str, str]]]] = {}
        self.step4_pair_cache_order: list[str] = []
        self.step4_pair_prefetch_inflight: set[str] = set()
        self.step4_pair_cache_capacity: int = 5
        self.step5_pairs: list[WorkspacePair] = []
        self.current_step5_pair_index: int = 0
        self.step5_run_thread: QThread | None = None
        self.step5_run_worker = None
        self.step6_pairs: list[WorkspacePair] = []
        self.current_step6_pair_index: int = 0
        self.step6_roi_root: Path | None = None
        self.step6_current_context = None
        self.step6_current_mapping_result: dict[str, object] | None = None
        self.step6_preview_stale: bool = False
        self.step6_source_side: str = "nissl"
        self.step6_last_updated_source_roi_highres: np.ndarray | None = None
        self.step6_last_updated_target_roi_highres: np.ndarray | None = None
        self.step6_confocal_handoff_path: Path | None = None
        self.step6_confocal_handoff: dict[str, object] | None = None
        self.step6_confocal_handoff_origin: str = "none"
        self.step6_confocal_overlay_visible: bool = True
        self.step6_confocal_support_bbox_cache: dict[str, tuple[int, int, int, int]] = {}
        self.step6_confocal_overlay_masks_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self.step6_hires_section_cache: dict[str, dict[str, object]] = {}
        self.step6_hires_loaded_slide: LoadedSlide | None = None
        self.step6_hires_slide_handle = None
        self.step6_hires_slide_key: tuple[object, ...] | None = None
        self.step6_hires_last_request_key: tuple[object, ...] | None = None
        self.step6_hires_view_timer = QTimer(self)
        self.step6_hires_view_timer.setSingleShot(True)
        self.step6_hires_view_timer.timeout.connect(self._refresh_step6_hires_source_patch)
        self.step7_myelin_root: Path | None = None
        self.step7_sections: list[WorkspaceSection] = []
        self.current_step7_section_index: int = -1
        self.step7_confocal_paths: list[Path] = []
        self.step7_confocal_source_mode: str = "none"
        self.step7_duplicate_stack_report: dict[str, object] | None = None
        self.step7_duplicate_stack_warning_shown: bool = False
        self.step7_projection_info: dict[str, object] | None = None
        self.step7_confocal_projection_raw_u8: np.ndarray | None = None
        self.step7_confocal_projection_u8: np.ndarray | None = None
        self.step7_confocal_projection_mask_raw_u8: np.ndarray | None = None
        self.step7_confocal_projection_mask_u8: np.ndarray | None = None
        self.step7_fixed_rgb: np.ndarray | None = None
        self.step7_fixed_labels: np.ndarray | None = None
        self.step7_fixed_info: dict[str, object] | None = None
        self.step7_fixed_cache: dict[str, tuple[np.ndarray, np.ndarray, dict[str, object]]] = {}
        self.step7_progress_state: dict[str, object] | None = None
        self.step7_run_thread: QThread | None = None
        self.step7_run_worker = None
        self.step7_auto_scale_thread: QThread | None = None
        self.step7_auto_scale_worker = None
        self.step7_seed_screen_thread: QThread | None = None
        self.step7_seed_screen_worker = None
        self.step7_frontier_thread: QThread | None = None
        self.step7_frontier_worker = None
        self.step7_last_manual_action: str | None = None
        self.step7_last_run_dir: Path | None = None
        self.step7_last_frontier_dir: Path | None = None
        self.step7_diagnostic_log: list[str] = []
        self.step7_last_run_summary_lines: list[str] = []
        self.step7_last_auto_scale_summary_lines: list[str] = []
        self.step7_last_seed_screen_summary_lines: list[str] = []
        self.step7_last_frontier_summary_lines: list[str] = []
        self.step7_last_auto_scale_dir: Path | None = None
        self.step7_last_seed_screen_dir: Path | None = None
        self.step7_last_seed_screen_rows: list[dict[str, object]] = []
        self.step7_last_frontier_rows: list[dict[str, object]] = []
        self.step7_tile_result_rows: dict[int, dict[str, object]] = {}
        self.step7_accepted_tile_indices: set[int] = set()
        self.step7_hold_tile_indices: set[int] = set()
        self.step7_frozen_tile_indices: set[int] = set()
        self.step7_frontier_tile_indices: set[int] = set()
        self.step7_last_export_dir: Path | None = None
        self.proposal_items: list[DraggableProposalItem] = []
        self.crop_outline_items: list[QGraphicsRectItem] = []
        self.proposal_cards: list[ProposalCard] = []
        self.default_ndpi_root = Path("D:/Research/Image Analysis/Nanozoomer scans")
        self.export_thread: QThread | None = None
        self.export_worker = None
        self.prepare_thread: QThread | None = None
        self.prepare_worker = None
        self.predict_thread: QThread | None = None
        self.predict_worker = None
        self.save_thread: QThread | None = None
        self.save_worker = None
        self.step5_runs_root: Path | None = None
        self.bg_precompute_thread: QThread | None = None
        self.bg_precompute_worker: BackgroundPrecomputeWorker | None = None
        self.bg_precompute_generation: int = 0
        self.bg_precompute_active_generation: int = 0
        self.editor_painting_active: bool = False
        self._after_save_action: Callable[[], None] | None = None

        self.pages = QStackedWidget()
        self.page_home = self._build_home_page()
        self.page_stage1 = self._build_stage1_page()
        self.page_stage2 = self._build_stage2_page()
        self.page_stage3 = self._build_stage3_page()
        self.page_stage4 = self._build_stage4_page()
        self.page_stage5 = self._build_stage5_page()
        self.page_stage6 = self._build_stage6_page()
        self.page_stage7 = self._build_stage7_page()
        self.page_stage8 = self._build_stage8_page()
        self.pages.addWidget(self.page_home)
        self.pages.addWidget(self.page_stage1)
        self.pages.addWidget(self.page_stage2)
        self.pages.addWidget(self.page_stage3)
        self.pages.addWidget(self.page_stage4)
        self.pages.addWidget(self.page_stage5)
        self.pages.addWidget(self.page_stage6)
        self.pages.addWidget(self.page_stage7)
        self.pages.addWidget(self.page_stage8)

        root = QVBoxLayout()
        root.addWidget(self.pages)
        self.setLayout(root)
        self._init_background_precompute()

    @property
    def current_slide(self):
        return self.workflow_service.current_slide

    def _build_home_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout()

        title = QLabel("Histology Workflow Steps")
        title.setStyleSheet("font-size: 24px; font-weight: 600;")
        subtitle = QLabel(
            "Choose a workflow step to enter. Step 1 reviews bbox proposals and exports section crop folders. "
            "Step 2 batch-predicts masks from exported crop folders. Step 3 edits masks from those folders. "
            "Step 4 reviews paired Nissl/Myelin sections for registration readiness. "
            "Step 5 prepares the usable reviewed pairs for downstream registration. "
            "Step 6 maps hand-drawn ROIs between Nissl and Myelin via an approved registration, defaulting to Nissl -> Myelin. "
            "Step 7 aligns confocal z-stacks locally onto Myelin. "
            "Step 8 will consume Step 7 session exports for fiber-density analysis."
        )
        subtitle.setWordWrap(True)

        self.step1_entry_button = QPushButton("Step 1: Histology Preview and Box Proposal")
        self.step1_entry_button.setMinimumHeight(52)
        self.step1_entry_button.clicked.connect(self.goto_stage1)

        self.step2_entry_button = QPushButton("Step 2: Batch Mask Prediction")
        self.step2_entry_button.setMinimumHeight(52)
        self.step2_entry_button.clicked.connect(self.goto_stage2)

        self.step3_entry_button = QPushButton("Step 3: Mask Review and Annotation")
        self.step3_entry_button.setMinimumHeight(52)
        self.step3_entry_button.clicked.connect(self.goto_stage3)

        self.future_step4_button = QPushButton("Step 4: Paired Slice QC / Registration Prep")
        self.future_step4_button.setMinimumHeight(44)
        self.future_step4_button.clicked.connect(self.goto_stage4)

        self.future_step5_button = QPushButton("Step 5: Registration Entry")
        self.future_step5_button.setMinimumHeight(44)
        self.future_step5_button.clicked.connect(self.goto_stage5)

        self.future_step6_button = QPushButton("Step 6: ROI Annotation and Mapping")
        self.future_step6_button.setMinimumHeight(44)
        self.future_step6_button.clicked.connect(self.goto_stage6)

        self.future_step7_button = QPushButton("Step 7: Confocal to Myelin Local Registration")
        self.future_step7_button.setMinimumHeight(44)
        self.future_step7_button.clicked.connect(self.goto_stage7)

        self.future_step8_button = QPushButton("Step 8: Fiber Density Analysis")
        self.future_step8_button.setMinimumHeight(44)
        self.future_step8_button.clicked.connect(self.goto_stage8)

        self.home_status = QTextEdit()
        self.home_status.setReadOnly(True)
        self.home_status.setMinimumHeight(160)
        self.home_status.setPlainText(
            "\n".join(
                [
                    "Current step entry points:",
                    "- Step 1: load NDPI, inspect slide thumbnail, adjust proposal boxes, export crop folders",
                    "- Step 2: batch-predict masks from exported crop folders",
                    "- Step 3: edit predicted masks from exported crop folders",
                    "- Step 4: review paired myelin/nissl sections for registration",
                    "- Step 5: inspect usable registration pairs and multi-group warnings",
                    "- Step 6: annotate ROI on Nissl and map it to Myelin via approved registration",
                    "- Step 7: generate confocal focus projection and tile-wise align it to Myelin",
                    "- Step 8: overlay Step 7 tile geometry with nnUNet myelin predictions for density analysis",
                    "",
                    "Current session:",
                    "- no slide loaded",
                    "- Step 2 / Step 3 require exported crop folders",
                    "- Step 4 uses paired section folders from myelin + nissl roots",
                ]
            )
        )

        layout.addWidget(title)
        layout.addWidget(subtitle)
        layout.addSpacing(12)
        layout.addWidget(self.step1_entry_button)
        layout.addWidget(self.step2_entry_button)
        layout.addWidget(self.step3_entry_button)
        layout.addWidget(self.future_step4_button)
        layout.addWidget(self.future_step5_button)
        layout.addWidget(self.future_step6_button)
        layout.addWidget(self.future_step7_button)
        layout.addWidget(self.future_step8_button)
        layout.addSpacing(12)
        layout.addWidget(self.home_status)
        layout.addStretch(1)
        page.setLayout(layout)
        return page

    def _build_stage1_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout()

        stage_header = QHBoxLayout()
        stage_header.addWidget(QLabel("Step 1: Histology Preview and Box Proposal"))
        stage_header.addStretch(1)
        self.home_from_stage1_button = QPushButton("Back To Step Menu")
        self.home_from_stage1_button.clicked.connect(self.goto_home)
        stage_header.addWidget(self.home_from_stage1_button)

        top_controls = QHBoxLayout()
        self.folder_button = QPushButton("Select NDPI Folder")
        self.folder_button.clicked.connect(self.select_ndpi_folder)
        self.open_file_button = QPushButton("Open Single NDPI")
        self.open_file_button.clicked.connect(self.open_single_ndpi)
        self.ndpi_status = QLabel("No folder loaded")
        top_controls.addWidget(self.folder_button)
        top_controls.addWidget(self.open_file_button)
        top_controls.addWidget(self.ndpi_status)

        main_row = QHBoxLayout()

        left_panel = QVBoxLayout()
        left_panel.addWidget(QLabel("NDPI Files"))
        self.ndpi_list = QListWidget()
        self.ndpi_list.itemSelectionChanged.connect(self.on_ndpi_selected)
        left_panel.addWidget(self.ndpi_list)
        left_panel.addWidget(QLabel("Label / Macro"))
        self.label_view = ImageSceneView()
        self.label_view.setMinimumHeight(220)
        left_panel.addWidget(self.label_view)

        center_panel = QVBoxLayout()
        center_panel.addWidget(QLabel("Slide Thumbnail + Proposal Boxes"))
        self.overview_view = ImageSceneView()
        center_panel.addWidget(self.overview_view)

        right_panel = QVBoxLayout()
        right_panel.addWidget(QLabel("Proposal Crop Previews"))
        self.cards_container = QWidget()
        self.cards_layout = QVBoxLayout()
        self.cards_layout.addStretch(1)
        self.cards_container.setLayout(self.cards_layout)
        self.cards_scroll = QScrollArea()
        self.cards_scroll.setWidgetResizable(True)
        self.cards_scroll.setWidget(self.cards_container)
        right_panel.addWidget(self.cards_scroll)

        main_row.addLayout(left_panel, 2)
        main_row.addLayout(center_panel, 5)
        main_row.addLayout(right_panel, 3)

        bottom_controls = QHBoxLayout()
        self.add_proposal_button = QPushButton("Add Proposal")
        self.add_proposal_button.clicked.connect(self.add_proposal_box)
        self.remove_proposal_button = QPushButton("Remove Selected Proposal")
        self.remove_proposal_button.clicked.connect(self.remove_selected_proposal)
        self.proposal_count_spin = QSpinBox()
        self.proposal_count_spin.setMinimum(0)
        self.proposal_count_spin.valueChanged.connect(self.ensure_proposal_count)
        self.export_crops_button = QPushButton("Confirm BBoxes + Export Crop Folders")
        self.export_crops_button.clicked.connect(self.export_crop_workspaces)
        self.next_step_button = QPushButton("Go To Step 2: Batch Mask Prediction")
        self.next_step_button.clicked.connect(self.goto_stage2)
        bottom_controls.addWidget(self.add_proposal_button)
        bottom_controls.addWidget(self.remove_proposal_button)
        bottom_controls.addWidget(QLabel("Proposal Count"))
        bottom_controls.addWidget(self.proposal_count_spin)
        bottom_controls.addSpacing(16)
        bottom_controls.addWidget(QLabel("Selected Proposal"))
        self.sel_x_spin = QSpinBox()
        self.sel_y_spin = QSpinBox()
        self.sel_w_spin = QSpinBox()
        self.sel_h_spin = QSpinBox()
        for spin in [self.sel_x_spin, self.sel_y_spin, self.sel_w_spin, self.sel_h_spin]:
            spin.setRange(0, 200000)
        self.sel_w_spin.setMinimum(1)
        self.sel_h_spin.setMinimum(1)
        bottom_controls.addWidget(QLabel("x"))
        bottom_controls.addWidget(self.sel_x_spin)
        bottom_controls.addWidget(QLabel("y"))
        bottom_controls.addWidget(self.sel_y_spin)
        bottom_controls.addWidget(QLabel("w"))
        bottom_controls.addWidget(self.sel_w_spin)
        bottom_controls.addWidget(QLabel("h"))
        bottom_controls.addWidget(self.sel_h_spin)
        self.apply_dims_button = QPushButton("Apply Box")
        self.apply_dims_button.clicked.connect(self.apply_selected_box_dimensions)
        bottom_controls.addWidget(self.apply_dims_button)
        bottom_controls.addStretch(1)
        bottom_controls.addWidget(self.export_crops_button)
        bottom_controls.addWidget(self.next_step_button)

        self.stage1_info = QTextEdit()
        self.stage1_info.setReadOnly(True)
        self.stage1_info.setMinimumHeight(110)

        layout.addLayout(stage_header)
        layout.addLayout(top_controls)
        layout.addLayout(main_row)
        layout.addLayout(bottom_controls)
        layout.addWidget(self.stage1_info)
        page.setLayout(layout)
        return page

    def _build_stage2_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout()

        stage_header = QHBoxLayout()
        stage_header.addWidget(QLabel("Step 2: Batch Mask Prediction"))
        stage_header.addStretch(1)
        self.home_from_stage2_button = QPushButton("Back To Step Menu")
        self.home_from_stage2_button.clicked.connect(self.goto_home)
        stage_header.addWidget(self.home_from_stage2_button)

        top_controls = QHBoxLayout()
        self.workspace_root_button = QPushButton("Select Crop Workspace Root")
        self.workspace_root_button.clicked.connect(self.select_workspace_root)
        self.workspace_root_label = QLabel("No crop workspace selected")
        top_controls.addWidget(self.workspace_root_button)
        top_controls.addWidget(self.workspace_root_label)

        body = QHBoxLayout()
        left = QVBoxLayout()
        left.addWidget(QLabel("Section Folders"))
        self.workspace_section_list = QListWidget()
        self.workspace_section_list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        left.addWidget(self.workspace_section_list)

        controls = QVBoxLayout()
        controls.addWidget(QLabel("Prediction Preset"))
        self.predict_preset_combo = QComboBox()
        self.predict_preset_combo.addItem("Auto By Stain", "__auto__")
        self.predict_preset_combo.addItem("M3 Hysteresis Guard (Best GT)", MASK_PRESET_M3_HYST_ENTRES_GUARD)
        self.predict_preset_combo.addItem("Latest Contextual", MASK_PRESET_LATEST_CONTEXTUAL)
        self.predict_preset_combo.addItem("Hybrid Balanced (Best GT)", MASK_PRESET_HYBRID_BALANCED)
        self.predict_preset_combo.addItem("Legacy Simple", MASK_PRESET_LEGACY_SIMPLE)
        controls.addWidget(self.predict_preset_combo)
        controls.addWidget(QLabel("Mask Compute Profile"))
        self.predict_profile_combo = QComboBox()
        self.predict_profile_combo.addItem("Standard 2048px (Recommended)", MASK_COMPUTE_PROFILE_STANDARD)
        self.predict_profile_combo.addItem("Fast 1600px", MASK_COMPUTE_PROFILE_FAST)
        self.predict_profile_combo.addItem("Full Export Resolution", MASK_COMPUTE_PROFILE_FULL)
        controls.addWidget(self.predict_profile_combo)
        self.prepare_work_button = QPushButton("Prepare Work Images For Selected")
        self.prepare_work_button.clicked.connect(self.prepare_workspace_work_images)

        self.refresh_workspace_button = QPushButton("Refresh Folder List")
        self.refresh_workspace_button.clicked.connect(self.refresh_workspace_sections)
        self.select_all_workspace_button = QPushButton("Select All Sections")
        self.select_all_workspace_button.clicked.connect(self.select_all_workspace_sections)
        self.predict_masks_button = QPushButton("Run Mask Prediction For Selected")
        self.predict_masks_button.clicked.connect(self.predict_masks_for_workspace)
        self.goto_stage3_button = QPushButton("Go To Step 3: Mask Review")
        self.goto_stage3_button.clicked.connect(self.goto_stage3)
        controls.addWidget(self.refresh_workspace_button)
        controls.addWidget(self.select_all_workspace_button)
        controls.addWidget(self.prepare_work_button)
        controls.addWidget(self.predict_masks_button)
        controls.addWidget(self.goto_stage3_button)
        controls.addStretch(1)

        body.addLayout(left, 5)
        body.addLayout(controls, 2)

        self.stage2_info = QTextEdit()
        self.stage2_info.setReadOnly(True)
        self.stage2_info.setMinimumHeight(160)
        self.stage2_progress_label = QLabel("Step 2 progress: idle")
        self.stage2_progress_bar = QProgressBar()
        self.stage2_progress_bar.setRange(0, 100)
        self.stage2_progress_bar.setValue(0)
        self.stage2_info.setPlainText(
            "\n".join(
                [
                    "Step 2: prepare downsampled work images, then batch-predict masks from exported crop folders.",
                    "- input is the crop workspace exported from Step 1",
                    "- select section folders explicitly before prepare/predict",
                    "- prepare writes crop_work_<profile>.png into each section folder",
                    "- prediction reads prepared work images and only writes mask png + metadata",
                    "- predicted masks become the starting point for Step 3 review",
                ]
            )
        )

        layout.addLayout(stage_header)
        layout.addLayout(top_controls)
        layout.addLayout(body)
        layout.addWidget(self.stage2_progress_label)
        layout.addWidget(self.stage2_progress_bar)
        layout.addWidget(self.stage2_info)
        page.setLayout(layout)
        return page

    def _build_stage3_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout()

        top = QHBoxLayout()
        self.prev_section_button = QPushButton("Prev")
        self.prev_section_button.clicked.connect(self.prev_section)
        self.next_section_button = QPushButton("Next")
        self.next_section_button.clicked.connect(self.next_section)
        self.back_button = QPushButton("Back To Step Menu")
        self.back_button.clicked.connect(self.goto_home)
        self.review_workspace_button = QPushButton("Select Crop Workspace Root")
        self.review_workspace_button.clicked.connect(self.select_workspace_root)
        self.review_mask_root_button = QPushButton("Select External Predicted Mask Root")
        self.review_mask_root_button.clicked.connect(self.select_review_mask_root)
        self.review_mask_root_clear_button = QPushButton("Use Workspace Masks")
        self.review_mask_root_clear_button.clicked.connect(self.clear_review_mask_root)
        self.review_mask_root_label = QLabel("Mask source: workspace section folders")
        self.review_section_combo = QComboBox()
        self.review_section_combo.currentIndexChanged.connect(self.on_review_section_changed)
        self.section_label = QLabel("No section selected")
        top.addWidget(self.prev_section_button)
        top.addWidget(self.next_section_button)
        top.addWidget(self.back_button)
        top.addWidget(self.review_workspace_button)
        top.addWidget(self.review_mask_root_button)
        top.addWidget(self.review_mask_root_clear_button)
        top.addWidget(self.review_mask_root_label)
        top.addWidget(self.review_section_combo)
        top.addWidget(self.section_label)

        main = QHBoxLayout()
        self.section_editor = MaskEditorLabel()
        self.section_editor.setMinimumSize(700, 520)
        self.section_editor.set_on_mask_changed(self.update_mask_stats)
        self.section_editor.set_on_painting_state_changed(self.on_editor_painting_state_changed)
        self.section_editor.set_on_active_layer_changed(self.on_editor_active_layer_changed)
        self.section_editor.set_on_close_fill_requested(self.close_and_fill_tissue_gaps)
        self.section_editor.set_on_save_and_next_requested(self.save_and_move_to_next)

        controls = QVBoxLayout()
        controls.addWidget(QLabel("Active Layer"))
        self.layer_combo = QComboBox()
        self.layer_combo.addItems(["tissue", "artifact"])
        self.layer_combo.currentTextChanged.connect(self.section_editor.set_active_layer)
        controls.addWidget(self.layer_combo)

        controls.addWidget(QLabel("Brush Radius"))
        self.brush_spin = QSpinBox()
        self.brush_spin.setRange(1, 100)
        self.brush_spin.setValue(8)
        self.brush_spin.valueChanged.connect(self.section_editor.set_brush_radius)
        controls.addWidget(self.brush_spin)

        controls.addWidget(QLabel("Mask Preset"))
        self.mask_preset_combo = QComboBox()
        self.mask_preset_combo.addItem("M3 Hysteresis Guard (Best GT)", MASK_PRESET_M3_HYST_ENTRES_GUARD)
        self.mask_preset_combo.addItem("Latest Contextual", MASK_PRESET_LATEST_CONTEXTUAL)
        self.mask_preset_combo.addItem("Hybrid Balanced (Best GT)", MASK_PRESET_HYBRID_BALANCED)
        self.mask_preset_combo.addItem("Legacy Simple", MASK_PRESET_LEGACY_SIMPLE)
        self.mask_preset_combo.currentIndexChanged.connect(self.on_mask_preset_changed)
        controls.addWidget(self.mask_preset_combo)
        controls.addWidget(QLabel("Mask Compute Profile"))
        self.review_profile_combo = QComboBox()
        self.review_profile_combo.addItem("Standard 2048px (Recommended)", MASK_COMPUTE_PROFILE_STANDARD)
        self.review_profile_combo.addItem("Fast 1600px", MASK_COMPUTE_PROFILE_FAST)
        self.review_profile_combo.addItem("Full Export Resolution", MASK_COMPUTE_PROFILE_FULL)
        controls.addWidget(self.review_profile_combo)

        self.mirror_check = QCheckBox("Mirror LR")
        self.mirror_check.toggled.connect(self.section_editor.set_mirror)
        controls.addWidget(self.mirror_check)

        self.auto_mask_button = QPushButton("Run/Refresh Auto Mask")
        self.auto_mask_button.clicked.connect(self.refresh_current_mask)
        self.close_fill_button = QPushButton("Close + Fill Tissue Gaps")
        self.close_fill_button.clicked.connect(self.close_and_fill_tissue_gaps)
        self.undo_button = QPushButton("Undo Last Edit")
        self.undo_button.clicked.connect(self.undo_last_edit)
        self.save_revision_button = QPushButton("Save Masks To Folder")
        self.save_revision_button.clicked.connect(self.save_current_revision_state)
        self.save_next_button = QPushButton("Save And Move To Next")
        self.save_next_button.clicked.connect(self.save_and_move_to_next)
        self.export_button = QPushButton("Return To Step 2")
        self.export_button.clicked.connect(self.goto_stage2)
        controls.addWidget(self.auto_mask_button)
        controls.addWidget(self.close_fill_button)
        controls.addWidget(self.undo_button)
        controls.addWidget(self.save_revision_button)
        controls.addWidget(self.save_next_button)
        controls.addWidget(self.export_button)
        self.bg_precompute_label = QLabel("Background precompute: idle")
        self.bg_precompute_label.setWordWrap(True)
        controls.addWidget(self.bg_precompute_label)
        controls.addWidget(QLabel("Current Mask Stats"))
        self.mask_stats_panel = QTextEdit()
        self.mask_stats_panel.setReadOnly(True)
        self.mask_stats_panel.setMinimumHeight(180)
        controls.addWidget(self.mask_stats_panel)

        self.section_info = QTextEdit()
        self.section_info.setReadOnly(True)
        controls.addWidget(self.section_info)
        controls.addStretch(1)

        main.addWidget(self.section_editor, 7)
        controls_widget = QWidget()
        controls_widget.setLayout(controls)
        main.addWidget(controls_widget, 3)

        layout.addLayout(top)
        layout.addLayout(main)
        page.setLayout(layout)
        return page

    def _build_stage4_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout()

        top = QHBoxLayout()
        self.step4_prev_button = QPushButton("Prev Pair")
        self.step4_prev_button.clicked.connect(self.prev_pair)
        self.step4_next_button = QPushButton("Next Pair")
        self.step4_next_button.clicked.connect(self.next_pair)
        self.step4_back_button = QPushButton("Back To Step Menu")
        self.step4_back_button.clicked.connect(self.goto_home)
        self.step4_myelin_root_button = QPushButton("Select Myelin Root")
        self.step4_myelin_root_button.clicked.connect(self.select_step4_myelin_root)
        self.step4_nissl_root_button = QPushButton("Select Nissl Root")
        self.step4_nissl_root_button.clicked.connect(self.select_step4_nissl_root)
        self.step4_refresh_button = QPushButton("Refresh Pairs")
        self.step4_refresh_button.clicked.connect(self.refresh_step4_pairs)
        self.step4_pair_label = QLabel("No pair selected")
        top.addWidget(self.step4_prev_button)
        top.addWidget(self.step4_next_button)
        top.addWidget(self.step4_back_button)
        top.addWidget(self.step4_myelin_root_button)
        top.addWidget(self.step4_nissl_root_button)
        top.addWidget(self.step4_refresh_button)
        top.addWidget(self.step4_pair_label)

        body = QHBoxLayout()

        left = QVBoxLayout()
        left.addWidget(QLabel("Paired Sections"))
        self.step4_pair_list = QListWidget()
        self.step4_pair_list.currentRowChanged.connect(self.on_step4_pair_changed)
        left.addWidget(self.step4_pair_list)
        self.step4_root_status = QTextEdit()
        self.step4_root_status.setReadOnly(True)
        self.step4_root_status.setMinimumHeight(120)
        left.addWidget(self.step4_root_status)

        center = QVBoxLayout()
        editor_row = QHBoxLayout()

        myelin_panel = QVBoxLayout()
        self.step4_myelin_title = QLabel("Myelin")
        myelin_panel.addWidget(self.step4_myelin_title)
        self.step4_myelin_editor = MaskEditorLabel()
        self.step4_myelin_editor.setMinimumSize(480, 360)
        self.step4_myelin_editor.set_on_mask_changed(lambda: self.on_step4_editor_mask_changed("myelin"))
        self.step4_myelin_editor.set_on_active_layer_changed(self.on_step4_editor_active_layer_changed)
        self.step4_myelin_editor.set_on_close_fill_requested(self.close_fill_step4_active_editor)
        self.step4_myelin_editor.set_on_save_and_next_requested(self.save_and_next_pair)
        self.step4_myelin_editor.set_on_focus_gained(lambda: self.set_step4_active_editor(self.step4_myelin_editor, "myelin"))
        self.step4_myelin_editor.set_on_mark_group_requested(
            lambda group_id: self.mark_step4_hovered_component_for_side("myelin", group_id)
        )
        myelin_panel.addWidget(self.step4_myelin_editor)
        self.step4_myelin_stats = QTextEdit()
        self.step4_myelin_stats.setReadOnly(True)
        self.step4_myelin_stats.setMinimumHeight(120)
        myelin_panel.addWidget(self.step4_myelin_stats)

        nissl_panel = QVBoxLayout()
        self.step4_nissl_title = QLabel("Nissl")
        nissl_panel.addWidget(self.step4_nissl_title)
        self.step4_nissl_editor = MaskEditorLabel()
        self.step4_nissl_editor.setMinimumSize(480, 360)
        self.step4_nissl_editor.set_on_mask_changed(lambda: self.on_step4_editor_mask_changed("nissl"))
        self.step4_nissl_editor.set_on_active_layer_changed(self.on_step4_editor_active_layer_changed)
        self.step4_nissl_editor.set_on_close_fill_requested(self.close_fill_step4_active_editor)
        self.step4_nissl_editor.set_on_save_and_next_requested(self.save_and_next_pair)
        self.step4_nissl_editor.set_on_focus_gained(lambda: self.set_step4_active_editor(self.step4_nissl_editor, "nissl"))
        self.step4_nissl_editor.set_on_mark_group_requested(
            lambda group_id: self.mark_step4_hovered_component_for_side("nissl", group_id)
        )
        nissl_panel.addWidget(self.step4_nissl_editor)
        self.step4_nissl_stats = QTextEdit()
        self.step4_nissl_stats.setReadOnly(True)
        self.step4_nissl_stats.setMinimumHeight(120)
        nissl_panel.addWidget(self.step4_nissl_stats)

        editor_row.addLayout(myelin_panel, 1)
        editor_row.addLayout(nissl_panel, 1)
        center.addLayout(editor_row)

        controls = QVBoxLayout()
        self.step4_active_editor_label = QLabel("Active panel (keyboard target): myelin")
        controls.addWidget(self.step4_active_editor_label)
        controls.addWidget(QLabel("Active Layer"))
        self.step4_layer_combo = QComboBox()
        self.step4_layer_combo.addItems(["tissue", "artifact"])
        self.step4_layer_combo.currentTextChanged.connect(self.set_step4_active_layer_all)
        controls.addWidget(self.step4_layer_combo)
        controls.addWidget(QLabel("Brush Radius"))
        self.step4_brush_spin = QSpinBox()
        self.step4_brush_spin.setRange(1, 100)
        self.step4_brush_spin.setValue(8)
        self.step4_brush_spin.valueChanged.connect(self.set_step4_brush_radius_all)
        controls.addWidget(self.step4_brush_spin)
        self.step4_myelin_flip_check = QCheckBox("Flip Myelin LR")
        self.step4_myelin_flip_check.toggled.connect(self.on_step4_flip_changed)
        self.step4_nissl_flip_check = QCheckBox("Flip Nissl LR")
        self.step4_nissl_flip_check.toggled.connect(self.on_step4_flip_changed)
        controls.addWidget(QLabel("Registration QC"))
        self.step4_registration_status_combo = QComboBox()
        self.step4_registration_status_combo.addItem("Unreviewed", "unreviewed")
        self.step4_registration_status_combo.addItem("Usable", "usable")
        self.step4_registration_status_combo.addItem("Unusable", "unusable")
        controls.addWidget(self.step4_myelin_flip_check)
        controls.addWidget(self.step4_nissl_flip_check)
        controls.addWidget(self.step4_registration_status_combo)
        self.step4_next_unreviewed_button = QPushButton("Move To Next Unreviewed")
        self.step4_next_unreviewed_button.clicked.connect(self.move_to_next_unreviewed_pair)
        controls.addWidget(self.step4_next_unreviewed_button)
        controls.addWidget(QLabel("Active Side Group Flip (registration only)"))
        self.step4_group_flip_side_label = QLabel("Active side: myelin")
        controls.addWidget(self.step4_group_flip_side_label)
        self.step4_group1_flip_check = QCheckBox("Flip Group 1 LR")
        self.step4_group1_flip_check.toggled.connect(lambda checked: self.on_step4_group_flip_changed(1, checked))
        self.step4_group2_flip_check = QCheckBox("Flip Group 2 LR")
        self.step4_group2_flip_check.toggled.connect(lambda checked: self.on_step4_group_flip_changed(2, checked))
        controls.addWidget(self.step4_group1_flip_check)
        controls.addWidget(self.step4_group2_flip_check)
        self.step4_close_fill_button = QPushButton("Close + Fill Active Panel")
        self.step4_close_fill_button.clicked.connect(self.close_fill_step4_active_editor)
        self.step4_undo_button = QPushButton("Undo Active Panel")
        self.step4_undo_button.clicked.connect(self.undo_step4_active_editor)
        self.step4_clear_hover_mark_button = QPushButton("Clear Hovered Component Mark")
        self.step4_clear_hover_mark_button.clicked.connect(self.clear_step4_hovered_component_mark)
        self.step4_reset_marks_button = QPushButton("Reset Active Side Component Marks")
        self.step4_reset_marks_button.clicked.connect(self.reset_step4_focused_component_marks)
        self.step4_save_button = QPushButton("Save Pair QC")
        self.step4_save_button.clicked.connect(self.save_current_pair_state)
        self.step4_save_next_button = QPushButton("Save Pair QC And Move To Next")
        self.step4_save_next_button.clicked.connect(self.save_and_next_pair)
        self.step4_to_stage5_button = QPushButton("Open Step 5 Registration Entry")
        self.step4_to_stage5_button.clicked.connect(self.goto_stage5)
        controls.addWidget(self.step4_close_fill_button)
        controls.addWidget(self.step4_undo_button)
        controls.addWidget(self.step4_clear_hover_mark_button)
        controls.addWidget(self.step4_reset_marks_button)
        controls.addWidget(self.step4_save_button)
        controls.addWidget(self.step4_save_next_button)
        controls.addWidget(self.step4_to_stage5_button)
        controls.addWidget(QLabel("Pair QC Info"))
        self.step4_info = QTextEdit()
        self.step4_info.setReadOnly(True)
        controls.addWidget(self.step4_info)
        controls.addStretch(1)

        body.addLayout(left, 2)
        body.addLayout(center, 8)
        controls_widget = QWidget()
        controls_widget.setLayout(controls)
        body.addWidget(controls_widget, 3)

        layout.addLayout(top)
        layout.addLayout(body)
        page.setLayout(layout)
        return page

    def _build_stage5_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout()

        top = QHBoxLayout()
        self.step5_refresh_button = QPushButton("Refresh Registration Pairs")
        self.step5_refresh_button.clicked.connect(self.refresh_step5_pairs)
        self.step5_back_button = QPushButton("Back To Step Menu")
        self.step5_back_button.clicked.connect(self.goto_home)
        self.step5_open_step4_button = QPushButton("Back To Step 4 QC")
        self.step5_open_step4_button.clicked.connect(self.goto_stage4)
        self.step5_open_step6_button = QPushButton("Open Step 6 ROI Mapping")
        self.step5_open_step6_button.clicked.connect(self.goto_stage6)
        self.step5_pair_label = QLabel("No registration pair selected")
        top.addWidget(self.step5_refresh_button)
        top.addWidget(self.step5_open_step4_button)
        top.addWidget(self.step5_open_step6_button)
        top.addWidget(self.step5_back_button)
        top.addWidget(self.step5_pair_label)

        body = QHBoxLayout()
        left = QVBoxLayout()
        left.addWidget(QLabel("Usable Registration Pairs"))
        self.step5_pair_list = QListWidget()
        self.step5_pair_list.currentRowChanged.connect(self.on_step5_pair_changed)
        left.addWidget(self.step5_pair_list)
        self.step5_root_status = QTextEdit()
        self.step5_root_status.setReadOnly(True)
        self.step5_root_status.setMinimumHeight(120)
        left.addWidget(self.step5_root_status)

        right = QVBoxLayout()
        run_controls = QHBoxLayout()
        self.step5_moving_side_combo = QComboBox()
        self.step5_moving_side_combo.addItem("Myelin", "myelin")
        self.step5_moving_side_combo.addItem("Nissl", "nissl")
        self.step5_fixed_side_combo = QComboBox()
        self.step5_fixed_side_combo.addItem("Nissl", "nissl")
        self.step5_fixed_side_combo.addItem("Myelin", "myelin")
        self.step5_moving_group_combo = QComboBox()
        self.step5_fixed_group_combo = QComboBox()
        for combo in (self.step5_moving_group_combo, self.step5_fixed_group_combo):
            combo.addItem("All kept", "all")
            combo.addItem("Group 1", "1")
            combo.addItem("Group 2", "2")
        self.step5_target_um_per_px_spin = QDoubleSpinBox()
        self.step5_target_um_per_px_spin.setRange(4.0, 30.0)
        self.step5_target_um_per_px_spin.setDecimals(1)
        self.step5_target_um_per_px_spin.setSingleStep(1.0)
        self.step5_target_um_per_px_spin.setValue(10.0)
        self.step5_target_um_per_px_spin.setSuffix(" um/px")
        self.step5_working_long_edge_combo = QComboBox()
        self.step5_working_long_edge_combo.addItem("1024 (Default)", 1024)
        self.step5_working_long_edge_combo.addItem("512 (Draft)", 512)
        self.step5_blur_sigma_spin = QDoubleSpinBox()
        self.step5_blur_sigma_spin.setRange(0.0, 5.0)
        self.step5_blur_sigma_spin.setDecimals(1)
        self.step5_blur_sigma_spin.setSingleStep(0.5)
        self.step5_blur_sigma_spin.setValue(0.0)
        self.step5_run_button = QPushButton("Run ANTs Registration")
        self.step5_run_button.clicked.connect(self.run_step5_registration)
        self.step5_approve_button = QPushButton("Approve Current Run")
        self.step5_approve_button.clicked.connect(self.approve_current_step5_run)
        run_controls.addWidget(QLabel("Moving"))
        run_controls.addWidget(self.step5_moving_side_combo)
        run_controls.addWidget(QLabel("Group"))
        run_controls.addWidget(self.step5_moving_group_combo)
        run_controls.addSpacing(10)
        run_controls.addWidget(QLabel("Fixed"))
        run_controls.addWidget(self.step5_fixed_side_combo)
        run_controls.addWidget(QLabel("Group"))
        run_controls.addWidget(self.step5_fixed_group_combo)
        run_controls.addSpacing(10)
        run_controls.addWidget(QLabel("Target"))
        run_controls.addWidget(self.step5_target_um_per_px_spin)
        run_controls.addWidget(QLabel("Working"))
        run_controls.addWidget(self.step5_working_long_edge_combo)
        run_controls.addWidget(QLabel("Blur"))
        run_controls.addWidget(self.step5_blur_sigma_spin)
        run_controls.addSpacing(10)
        run_controls.addWidget(self.step5_run_button)
        run_controls.addWidget(self.step5_approve_button)
        right.addLayout(run_controls)
        self.step5_progress_bar = QProgressBar()
        self.step5_progress_bar.setRange(0, 100)
        self.step5_progress_bar.setValue(0)
        self.step5_progress_label = QLabel("Step 5 progress: idle")
        self.step5_acceptance_label = QLabel(
            "Accepted path: input | Rejected: none | Best accepted state runner: downstream stages start from the current best state."
        )
        self.step5_acceptance_label.setWordWrap(True)
        self.step5_acceptance_label.setStyleSheet(
            "padding:6px 10px; border-radius:6px; background:#eef4fb; color:#24405c; border:1px solid #c9d8ea; font-weight:600;"
        )
        right.addWidget(self.step5_progress_bar)
        right.addWidget(self.step5_progress_label)
        right.addWidget(self.step5_acceptance_label)
        right.addWidget(QLabel("Registration Entry Info"))
        self.step5_info = QTextEdit()
        self.step5_info.setReadOnly(True)
        self.step5_info.setPlainText(
            "\n".join(
                [
                    "Step 5 entry point",
                    "- only pairs marked Usable in Step 4 are shown here",
                    "- multi-group pairs mean registration should consider 1<->1 and 2<->2",
                    "- choose moving/fixed side and group, normalize both sides to a common target um/px",
                    "- then downsample to a registration working long edge (1024 default, 512 optional) and optional blur",
                    "- then run rigid + affine + nonlinear ANTs",
                    "- later stages start from the current best accepted state, not a rejected linear result",
                    "- approve a run before Step 6 can use it for ROI mapping",
                ]
            )
        )
        right.addWidget(self.step5_info)
        right.addWidget(QLabel("Storyboard"))
        self.step5_storyboard_label = QLabel("No registration storyboard yet")
        self.step5_storyboard_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.step5_storyboard_label.setMinimumSize(640, 480)
        self.step5_storyboard_label.setStyleSheet("background:#f5f5f5; border:1px solid #cccccc;")
        self.step5_storyboard_scroll = QScrollArea()
        self.step5_storyboard_scroll.setWidgetResizable(True)
        self.step5_storyboard_scroll.setWidget(self.step5_storyboard_label)
        right.addWidget(self.step5_storyboard_scroll, 1)

        body.addLayout(left, 3)
        body.addLayout(right, 6)

        layout.addLayout(top)
        layout.addLayout(body)
        page.setLayout(layout)
        return page

    def _build_stage6_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout()

        top = QHBoxLayout()
        self.step6_refresh_button = QPushButton("Refresh Approved ROI Pairs")
        self.step6_refresh_button.clicked.connect(self.refresh_step6_pairs)
        self.step6_back_button = QPushButton("Back To Step Menu")
        self.step6_back_button.clicked.connect(self.goto_home)
        self.step6_open_step5_button = QPushButton("Back To Step 5 Registration")
        self.step6_open_step5_button.clicked.connect(self.goto_stage5)
        self.step6_pair_label = QLabel("No approved ROI mapping pair selected")
        self.step6_direction_combo = QComboBox()
        self.step6_direction_combo.addItem("Nissl -> Myelin", "nissl")
        self.step6_direction_combo.addItem("Myelin -> Nissl", "myelin")
        self.step6_direction_combo.currentIndexChanged.connect(self.on_step6_direction_changed)
        self.step6_load_step7_handoff_button = QPushButton("Load Step 7 Handoff")
        self.step6_load_step7_handoff_button.clicked.connect(self.load_step6_step7_handoff)
        self.step6_clear_step7_handoff_button = QPushButton("Clear Confocal Overlay")
        self.step6_clear_step7_handoff_button.clicked.connect(self.clear_step6_step7_handoff)
        self.step6_auto_step7_handoff_check = QCheckBox("Auto Load Latest Step 7 Handoff")
        self.step6_auto_step7_handoff_check.setChecked(True)
        self.step6_auto_step7_handoff_check.toggled.connect(self.on_step6_auto_handoff_toggled)
        top.addWidget(self.step6_refresh_button)
        top.addWidget(self.step6_open_step5_button)
        top.addWidget(self.step6_back_button)
        top.addWidget(QLabel("Direction"))
        top.addWidget(self.step6_direction_combo)
        top.addWidget(self.step6_auto_step7_handoff_check)
        top.addWidget(self.step6_load_step7_handoff_button)
        top.addWidget(self.step6_clear_step7_handoff_button)
        top.addWidget(self.step6_pair_label)

        body = QHBoxLayout()
        left = QVBoxLayout()
        left.addWidget(QLabel("Pairs With Approved Registration"))
        self.step6_pair_list = QListWidget()
        self.step6_pair_list.currentRowChanged.connect(self.on_step6_pair_changed)
        left.addWidget(self.step6_pair_list)
        self.step6_root_status = QTextEdit()
        self.step6_root_status.setReadOnly(True)
        self.step6_root_status.setMinimumHeight(72)
        self.step6_root_status.setMaximumHeight(110)
        left.addWidget(self.step6_root_status)
        self.step6_confocal_status_label = QLabel("Confocal FOV overlay: none")
        self.step6_confocal_status_label.setWordWrap(True)
        self.step6_confocal_status_label.setStyleSheet("padding:6px 8px; background:#f5f5f5; border:1px solid #d0d0d0;")
        left.addWidget(self.step6_confocal_status_label)
        self.step6_pair_list.setMaximumWidth(280)
        self.step6_root_status.setMaximumWidth(280)
        self.step6_confocal_status_label.setMaximumWidth(280)
        left_panel = QWidget()
        left_panel.setLayout(left)
        left_panel.setMaximumWidth(310)

        center = QVBoxLayout()
        self.step6_nissl_title = QLabel("Nissl ROI")
        center.addWidget(self.step6_nissl_title)
        step6_tool_row = QHBoxLayout()
        self.step6_grab_button = QPushButton("Grab")
        self.step6_grab_button.setCheckable(True)
        self.step6_grab_button.clicked.connect(lambda: self.set_step6_source_tool("grab"))
        self.step6_brush_button = QPushButton("Brush")
        self.step6_brush_button.setCheckable(True)
        self.step6_brush_button.clicked.connect(lambda: self.set_step6_source_tool("brush"))
        self.step6_eraser_button = QPushButton("Eraser")
        self.step6_eraser_button.setCheckable(True)
        self.step6_eraser_button.clicked.connect(lambda: self.set_step6_source_tool("eraser"))
        self.step6_polygon_button = QPushButton("Polygon")
        self.step6_polygon_button.setCheckable(True)
        self.step6_polygon_button.clicked.connect(lambda: self.set_step6_source_tool("polygon"))
        self.step6_polygon_fill_button = QPushButton("Fill Polygon")
        self.step6_polygon_fill_button.clicked.connect(self.step6_apply_polygon_fill)
        self.step6_polygon_clear_button = QPushButton("Clear Polygon")
        self.step6_polygon_clear_button.clicked.connect(self.step6_clear_polygon)
        self.step6_brush_spin = QSpinBox()
        self.step6_brush_spin.setRange(1, 100)
        self.step6_brush_spin.setValue(8)
        self.step6_hires_nissl_check = QCheckBox("Hi-Res Nissl View")
        self.step6_hires_nissl_check.toggled.connect(self.on_step6_hires_nissl_toggled)
        self.step6_force_level0_check = QCheckBox("Force Level0")
        self.step6_force_level0_check.toggled.connect(self.on_step6_force_level0_toggled)
        self.step6_toggle_confocal_overlay_button = QPushButton("Hide Confocal Grid")
        self.step6_toggle_confocal_overlay_button.clicked.connect(self.toggle_step6_confocal_overlay_visibility)
        step6_tool_row.addWidget(self.step6_grab_button)
        step6_tool_row.addWidget(self.step6_brush_button)
        step6_tool_row.addWidget(self.step6_eraser_button)
        step6_tool_row.addWidget(self.step6_polygon_button)
        step6_tool_row.addWidget(self.step6_polygon_fill_button)
        step6_tool_row.addWidget(self.step6_polygon_clear_button)
        step6_tool_row.addWidget(QLabel("Brush"))
        step6_tool_row.addWidget(self.step6_brush_spin)
        step6_tool_row.addWidget(self.step6_hires_nissl_check)
        step6_tool_row.addWidget(self.step6_force_level0_check)
        step6_tool_row.addWidget(self.step6_toggle_confocal_overlay_button)
        center.addLayout(step6_tool_row)
        self._sync_step6_confocal_overlay_toggle_button()
        self.step6_hires_status_label = QLabel("Hi-res patch: off")
        self.step6_hires_status_label.setWordWrap(True)
        self.step6_hires_status_label.setStyleSheet("padding:4px 8px; color:#444444; background:#f5f5f5; border:1px solid #d8d8d8;")
        center.addWidget(self.step6_hires_status_label)
        self.step6_nissl_editor = MaskEditorLabel()
        self.step6_nissl_editor.set_on_mask_changed(self.on_step6_roi_mask_changed)
        self.step6_nissl_editor.set_on_save_and_next_requested(self.save_step6_roi_and_next)
        self.step6_nissl_editor.set_on_tool_mode_changed(self.on_step6_source_tool_changed)
        self.step6_nissl_editor.set_on_view_changed(self.on_step6_source_view_changed)
        self.step6_brush_spin.valueChanged.connect(self.step6_nissl_editor.set_brush_radius)
        self.step6_nissl_editor.set_brush_radius(self.step6_brush_spin.value())
        center.addWidget(self.step6_nissl_editor, 1)

        right = QVBoxLayout()
        self.step6_mapping_status_label = QLabel("Mapped ROI preview: fresh")
        self.step6_mapping_status_label.setStyleSheet(
            "padding:6px 10px; border-radius:6px; background:#e8f6ec; color:#175c2b; font-weight:600;"
        )
        right.addWidget(self.step6_mapping_status_label)
        controls = QHBoxLayout()
        self.step6_update_button = QPushButton("Update ROI Mapping")
        self.step6_update_button.clicked.connect(self.update_step6_roi_mapping_preview)
        self.step6_save_button = QPushButton("Save ROI Outputs")
        self.step6_save_button.clicked.connect(self.save_step6_roi)
        self.step6_save_next_button = QPushButton("Save ROI Outputs + Move To Next")
        self.step6_save_next_button.clicked.connect(self.save_step6_roi_and_next)
        controls.addWidget(self.step6_update_button)
        controls.addWidget(self.step6_save_button)
        controls.addWidget(self.step6_save_next_button)
        right.addLayout(controls)
        self.step6_target_title = QLabel("Mapped Myelin ROI")
        right.addWidget(self.step6_target_title)
        target_view_controls = QHBoxLayout()
        self.step6_target_zoom_out_button = QPushButton("Zoom -")
        self.step6_target_zoom_out_button.clicked.connect(lambda: self.zoom_step6_target_view(1.0 / 1.12))
        self.step6_target_zoom_in_button = QPushButton("Zoom +")
        self.step6_target_zoom_in_button.clicked.connect(lambda: self.zoom_step6_target_view(1.12))
        self.step6_target_fit_button = QPushButton("Fit View")
        self.step6_target_fit_button.clicked.connect(self.reset_step6_target_view)
        target_view_controls.addWidget(self.step6_target_zoom_out_button)
        target_view_controls.addWidget(self.step6_target_zoom_in_button)
        target_view_controls.addWidget(self.step6_target_fit_button)
        right.addLayout(target_view_controls)
        self.step6_target_viewer = MaskEditorLabel()
        self.step6_target_viewer.set_editing_enabled(False)
        right.addWidget(self.step6_target_viewer, 1)
        self.step6_info = QTextEdit()
        self.step6_info.setReadOnly(True)
        self.step6_info.setPlainText(
            "\n".join(
                [
                    "Step 6 ROI Annotation and Mapping",
                    "- default direction is Nissl -> Myelin, but you can switch to Myelin -> Nissl",
                    "- left panel is the editable source; right panel is a read-only target viewer with pan/zoom",
                    "- tools: Grab, Brush, Eraser, Polygon; both sides also support Zoom +/- and Fit View",
                    "- Hi-Res Nissl View keeps ROI storage on the current canvas but replaces the visible left-side Nissl viewport with a dynamic high-resolution NDPI patch",
                    "- polygon: left click adds vertices, Fill Polygon or right click or C closes/fills, Clear Polygon or Esc clears",
                    "- Step 6 works independently; if Auto Load Latest Step 7 Handoff is enabled and a current myelin export exists, the confocal FOV/grid is loaded onto the current source side automatically",
                    "- optional: load a Step 7 handoff to project accepted/frozen confocal FOV and tile grid onto both source and target sides",
                    "- Update ROI Mapping downsamples through the approved Step 5 preprocessing chain, applies the approved transform, and refreshes the right target preview",
                    "- green/yellow highlights show ROI added in the current edit batch; magenta highlights show ROI removed in the current batch",
                    "- Save writes high-resolution ROI outputs plus low-resolution debug canvases",
                    "- S saves and moves to the next approved pair",
                ]
            )
        )
        right.addWidget(self.step6_info)

        body.addWidget(left_panel, 1)
        body.addLayout(center, 5)
        body.addLayout(right, 6)

        layout.addLayout(top)
        layout.addLayout(body)
        page.setLayout(layout)
        self._sync_step6_tool_buttons("grab")
        self._sync_step6_hires_nissl_controls()
        return page

    def _build_stage7_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout()

        top = QHBoxLayout()
        self.step7_refresh_button = QPushButton("Refresh Myelin Sections")
        self.step7_refresh_button.clicked.connect(self.refresh_step7_sections)
        self.step7_back_button = QPushButton("Back To Step Menu")
        self.step7_back_button.clicked.connect(self.goto_home)
        self.step7_pair_label = QLabel("No myelin section selected")
        top.addWidget(self.step7_refresh_button)
        top.addWidget(self.step7_back_button)
        top.addWidget(self.step7_pair_label)

        body = QHBoxLayout()
        left = QVBoxLayout()
        left.addWidget(QLabel("Section Selection"))
        self.step7_section_list = QListWidget()
        self.step7_section_list.currentRowChanged.connect(self.on_step7_section_changed)
        left.addWidget(self.step7_section_list)
        self.step7_root_status = QTextEdit()
        self.step7_root_status.setReadOnly(True)
        self.step7_root_status.setMinimumHeight(120)
        left.addWidget(self.step7_root_status)

        middle = QVBoxLayout()
        select_row = QHBoxLayout()
        self.step7_select_stack_button = QPushButton("Select Confocal Source(s)")
        self.step7_select_stack_button.clicked.connect(self.select_step7_confocal_stack)
        self.step7_clear_grid_button = QPushButton("Clear Current Grid")
        self.step7_clear_grid_button.clicked.connect(self.clear_step7_current_grid)
        self.step7_stack_label = QLabel("No confocal source selected")
        self.step7_stack_label.setWordWrap(True)
        select_row.addWidget(self.step7_select_stack_button)
        select_row.addWidget(self.step7_clear_grid_button)
        select_row.addWidget(self.step7_stack_label, 1)
        middle.addLayout(select_row)

        proj_row = QHBoxLayout()
        self.step7_projection_mode_combo = QComboBox()
        self.step7_projection_mode_combo.addItem("Focus / EDF", "focus")
        self.step7_projection_mode_combo.addItem("Max", "max")
        self.step7_projection_mode_combo.addItem("Mean", "mean")
        self.step7_channel_spin = QSpinBox()
        self.step7_channel_spin.setRange(0, 15)
        self.step7_channel_spin.setValue(0)
        self.step7_overlap_spin = QDoubleSpinBox()
        self.step7_overlap_spin.setRange(0.0, 0.45)
        self.step7_overlap_spin.setDecimals(3)
        self.step7_overlap_spin.setSingleStep(0.01)
        self.step7_overlap_spin.setValue(0.10)
        self.step7_generate_projection_button = QPushButton("Generate Projection")
        self.step7_generate_projection_button.clicked.connect(self.generate_step7_projection)
        proj_row.addWidget(QLabel("Projection"))
        proj_row.addWidget(self.step7_projection_mode_combo)
        proj_row.addWidget(self.step7_generate_projection_button)
        proj_row.addStretch(1)
        middle.addLayout(proj_row)

        manual_row = QHBoxLayout()
        self.step7_tx_spin = QDoubleSpinBox()
        self.step7_tx_spin.setRange(-5000.0, 5000.0)
        self.step7_tx_spin.setDecimals(1)
        self.step7_ty_spin = QDoubleSpinBox()
        self.step7_ty_spin.setRange(-5000.0, 5000.0)
        self.step7_ty_spin.setDecimals(1)
        self.step7_angle_spin = QDoubleSpinBox()
        self.step7_angle_spin.setRange(-180.0, 180.0)
        self.step7_angle_spin.setDecimals(1)
        self.step7_scale_spin = QDoubleSpinBox()
        self.step7_scale_spin.setRange(0.1, 5.0)
        self.step7_scale_spin.setDecimals(3)
        self.step7_scale_spin.setValue(1.0)
        self.step7_profile_combo = QComboBox()
        self.step7_profile_combo.addItem("Pct 1-99 + Blur8 (Default)", "paired_percentile_blur8")
        self.step7_profile_combo.addItem("Pct 1-99 + Blur6", "paired_percentile_blur6")
        self.step7_profile_combo.addItem("Pct 1-99 + Blur4", "paired_percentile_blur4")
        self.step7_profile_combo.addItem("Pct 1-99 + CLAHE2.5 + Blur6 (Rescue)", "paired_pct1_99_clahe2p5_blur6")
        self.step7_profile_combo.currentIndexChanged.connect(lambda _idx: self._update_step7_info_text())
        self.step7_refine_model_combo = QComboBox()
        self.step7_refine_model_combo.addItem("Similarity (Default)", "similarity")
        self.step7_refine_model_combo.addItem("Affine", "affine")
        self.step7_flip_lr_check = QCheckBox("Flip LR")
        self.step7_flip_ud_check = QCheckBox("Flip UD")
        self.step7_flip_lr_check.setChecked(False)
        self.step7_flip_ud_check.setChecked(True)
        self.step7_flip_lr_check.setEnabled(False)
        self.step7_flip_ud_check.setEnabled(False)
        self.step7_flip_lr_check.toggled.connect(self.on_step7_flip_changed)
        self.step7_flip_ud_check.toggled.connect(self.on_step7_flip_changed)
        self.step7_run_button = QPushButton("Fiber Registration")
        self.step7_run_button.clicked.connect(self.run_step7_registration)
        self.step7_auto_scale_button = QPushButton("Auto Scale Sweep (Sampled)")
        self.step7_auto_scale_button.clicked.connect(self.run_step7_auto_scale_sweep)
        self.step7_seed_screen_button = QPushButton("Screen Seed Tiles")
        self.step7_seed_screen_button.clicked.connect(self.run_step7_seed_screening)
        self.step7_frontier_button = QPushButton("Propagate Frontier")
        self.step7_frontier_button.clicked.connect(self.run_step7_frontier_propagation)
        self.step7_export_button = QPushButton("Export Step 7 Session")
        self.step7_export_button.clicked.connect(self.export_step7_session_package)
        self.step7_anchor_mode_button = QPushButton("Manual Anchor Mode")
        self.step7_anchor_mode_button.clicked.connect(self.start_step7_anchor_mode)
        manual_row.addWidget(QLabel("Reg Input"))
        manual_row.addWidget(self.step7_profile_combo)
        manual_row.addWidget(self.step7_auto_scale_button)
        manual_row.addWidget(self.step7_seed_screen_button)
        manual_row.addWidget(self.step7_anchor_mode_button)
        manual_row.addStretch(1)
        middle.addLayout(manual_row)

        self.step7_progress_label = QLabel("Step 7 progress: idle")
        self.step7_progress_bar = QProgressBar()
        self.step7_progress_bar.setRange(0, 100)
        self.step7_progress_bar.setValue(0)
        self.step7_progress_detail_label = QLabel("Active tiles: none")
        self.step7_progress_detail_label.setWordWrap(True)
        middle.addWidget(self.step7_progress_label)
        middle.addWidget(self.step7_progress_bar)
        middle.addWidget(self.step7_progress_detail_label)
        self.step7_info = QTextEdit()
        self.step7_info.setReadOnly(True)
        self.step7_info.setPlainText(
            "\n".join(
                [
                    "Step 7 Confocal to Myelin Local Registration",
                    "- select a whole-section myelin crop and one or more confocal sources",
                    "- Clear Current Grid removes the active confocal source(s), projection, tile states, and coarse transform while keeping the selected myelin section loaded",
                    "- single-source TIFF, full CZI, and multi-TIFF strip stitching are supported",
                    "- multi-TIFF projection uses the default tile-overlap setting plus phase-corrected strip stitching",
                    "- drag the overlay block with left mouse; right drag rotates it",
                    "- confocal is displayed inverted against myelin and both sides use a 1.0 um/px working grid",
                    "- orientation is currently locked to UD flip (x-axis mirror) for this confocal-to-nanozoomer setup",
                    "- drag / rotate / flip for coarse manual alignment using local fiber patterns; tx/ty/angle/scale are recorded but kept off the main toolbar",
                    "- Auto Scale Sweep (Sampled) tests a bounded set of global whole-grid scales on a small representative tile subset and picks the one that best improves downstream tile-local fiber matching",
                    f"- current registration input profile defaults to {STEP7_REGISTRATION_INPUT_PROFILE}: paired percentile normalization on both sides + Gaussian blur sigma=8",
                    "- rescue option available: paired_pct1_99_clahe2p5_blur6 for special tiles with strong landmark structure",
                    "- Screen Seed Tiles runs a first pass from the current whole-grid coarse placement and ranks tiles that are already locally stable enough to serve as seeds",
                    "- after reviewing and accepting/freezing some tiles, use Propagate Frontier from the lower-right QC area to expand one more frontier round",
                    "- Propagate Frontier uses accepted/frozen tiles as solved graph nodes; if none are marked yet, it falls back to the current best seed-screen tile",
                    "- local refine uses the default similarity model; affine remains available internally for targeted testing",
                    "- press F to lock the confocal grid; this does not start landmark collection by itself",
                    "- Manual Anchor Mode starts alternating anchor collection: A1 -> B1 -> A2 -> B2; keys 1-9 override the point index",
                    "- while locked and collecting, right-click or Backspace undoes the latest anchor and restores that slot",
                    "- A/B keys place anchors at the current cursor position; [ and ] change confocal overlay opacity",
                    "- clicking inside the grid selects a tile; Shift-click adds/removes tiles from the current selection for batch freezing",
                    "- the primary selected tile is highlighted in bright yellow; additional selected tiles use a lighter yellow and frozen tiles stay blue",
                    "- before any tile is frozen, drag the selected grid with left mouse, right-drag to rotate, and pull corner handles for ratio-locked scaling",
                    "- once any tile is frozen, the whole-grid transform is locked so frozen tiles stay fixed in place for later propagation rounds",
                    "- mouse wheel only zooms the full preview view; it no longer rescales the selected grid",
                    "- Export Step 7 Session writes a current-session documentation package plus a Step 8 handoff with tile geometry and transforms",
                ]
            )
        )
        middle.addWidget(self.step7_info)

        right = QVBoxLayout()
        right.addWidget(QLabel("Manual Preview"))
        self.step7_preview_view = ConfocalAlignmentView()
        self.step7_preview_view.setMinimumSize(640, 360)
        self.step7_preview_view.setStyleSheet("background:#f5f5f5; border:1px solid #cccccc;")
        self.step7_preview_view.transformEdited.connect(self.on_step7_preview_transform_edited)
        self.step7_preview_view.diagnosticPointPlaced.connect(self.on_step7_diagnostic_point_placed)
        self.step7_preview_view.diagnosticStateChanged.connect(self.on_step7_diagnostic_state_changed)
        self.step7_preview_view.tileSelectionChanged.connect(self.on_step7_tile_selection_changed)
        right.addWidget(self.step7_preview_view, 1)
        tile_qc_title_row = QHBoxLayout()
        tile_qc_title_row.addWidget(QLabel("Selected Tile QC"))
        tile_qc_title_row.addStretch(1)
        self.step7_frozen_count_label = QLabel("Frozen: 0/0")
        self.step7_frozen_count_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        tile_qc_title_row.addWidget(self.step7_frozen_count_label)
        right.addLayout(tile_qc_title_row)
        tile_qc_controls = QHBoxLayout()
        self.step7_tile_prev_button = QPushButton("Prev Tile")
        self.step7_tile_prev_button.clicked.connect(self.select_prev_step7_tile)
        self.step7_tile_next_button = QPushButton("Next Tile")
        self.step7_tile_next_button.clicked.connect(self.select_next_step7_tile)
        self.step7_tile_accept_button = QPushButton("Accept")
        self.step7_tile_accept_button.clicked.connect(self.accept_step7_selected_tile)
        self.step7_tile_hold_button = QPushButton("Hold")
        self.step7_tile_hold_button.clicked.connect(self.hold_step7_selected_tile)
        self.step7_tile_freeze_button = QPushButton("Freeze Tile")
        self.step7_tile_freeze_button.clicked.connect(self.toggle_step7_selected_tile_frozen)
        self.step7_tile_status_label = QLabel("No tile selected")
        self.step7_tile_status_label.setWordWrap(True)
        tile_qc_controls.addWidget(self.step7_tile_prev_button)
        tile_qc_controls.addWidget(self.step7_tile_next_button)
        tile_qc_controls.addWidget(self.step7_tile_accept_button)
        tile_qc_controls.addWidget(self.step7_tile_hold_button)
        tile_qc_controls.addWidget(self.step7_tile_freeze_button)
        tile_qc_controls.addWidget(self.step7_tile_status_label, 1)
        right.addLayout(tile_qc_controls)
        self.step7_storyboard_label = QLabel("No selected tile QC yet")
        self.step7_storyboard_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.step7_storyboard_label.setMinimumSize(640, 320)
        self.step7_storyboard_label.setStyleSheet("background:#f5f5f5; border:1px solid #cccccc;")
        self.step7_storyboard_scroll = QScrollArea()
        self.step7_storyboard_scroll.setWidgetResizable(False)
        self.step7_storyboard_scroll.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.step7_storyboard_scroll.setWidget(self.step7_storyboard_label)
        right.addWidget(self.step7_storyboard_scroll, 1)
        frontier_action_row = QHBoxLayout()
        frontier_action_row.addStretch(1)
        frontier_action_row.addWidget(self.step7_frontier_button)
        frontier_action_row.addWidget(self.step7_export_button)
        right.addLayout(frontier_action_row)

        body.addLayout(left, 3)
        body.addLayout(middle, 5)
        body.addLayout(right, 7)

        layout.addLayout(top)
        layout.addLayout(body)
        page.setLayout(layout)
        return page

    def _build_stage8_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout()

        top = QHBoxLayout()
        top.addWidget(QLabel("Step 8: Fiber Density Analysis"))
        top.addStretch(1)
        back = QPushButton("Back To Step Menu")
        back.clicked.connect(self.goto_home)
        top.addWidget(back)
        layout.addLayout(top)

        info = QTextEdit()
        info.setReadOnly(True)
        info.setMinimumHeight(420)
        info.setPlainText(
            "\n".join(
                [
                    "Step 8 Scaffold",
                    "- primary upstream input: Step 7 session export",
                    "- required handoff file: step8_handoff.json",
                    "- planned second input: nnUNet 3D myelin prediction / inference export",
                    "- planned functions:",
                    "  * load registered confocal tile positions and transforms",
                    "  * connect them to predicted myelin maps in the same Step 7 preview scene space",
                    "  * visualize prediction overlays on the confocal-myelin tile view",
                    "  * compute tile-wise and pooled fiber-density summaries",
                    "",
                    "Latest Step 7 export: none",
                ]
            )
        )
        self.step8_info = info
        layout.addWidget(info)
        page.setLayout(layout)
        return page

    def _init_background_precompute(self) -> None:
        self.bg_precompute_thread = QThread(self)
        self.bg_precompute_worker = BackgroundPrecomputeWorker(self.workflow_service.auto_mask_cache)
        self.bg_precompute_worker.moveToThread(self.bg_precompute_thread)
        self.bg_precompute_thread.started.connect(self.bg_precompute_worker.run)
        self.bg_precompute_worker.status.connect(self.on_background_precompute_status)
        self.bg_precompute_worker.section_ready.connect(self.on_background_precompute_section_ready)
        self.bg_precompute_worker.idle.connect(self.on_background_precompute_idle)
        self.bg_precompute_worker.failed.connect(self.on_background_precompute_failed)
        try:
            priority = QThread.Priority.LowestPriority
        except AttributeError:
            priority = QThread.LowestPriority
        self.bg_precompute_thread.start(priority)

    def _clone_all_proposals(self, proposals: list[ProposalBox]) -> list[ProposalBox]:
        return [_clone_proposal_snapshot(proposal) for proposal in proposals]

    def _remaining_precompute_indices(self) -> list[int]:
        slide = self.current_slide
        if slide is None or not slide.proposals:
            return []
        out: list[int] = []
        total = len(slide.proposals)
        for offset in range(1, total):
            idx = (self.current_proposal_index + offset) % total
            if idx != self.current_proposal_index and idx not in out:
                out.append(idx)
        return out

    def _background_precompute_should_pause(self) -> bool:
        return (
            self.pages.currentIndex() != self.PAGE_STAGE2
            or self.current_slide is None
            or self.save_thread is not None
            or self.export_thread is not None
            or self.editor_painting_active
        )

    def _schedule_background_precompute(self) -> None:
        if self.bg_precompute_worker is None:
            return
        slide = self.current_slide
        if slide is None or self.pages.currentIndex() != self.PAGE_STAGE2:
            self.bg_precompute_worker.set_paused(True)
            self.bg_precompute_label.setText("Background precompute: idle")
            return

        if self._background_precompute_should_pause():
            self.bg_precompute_worker.set_paused(True)
            if self.editor_painting_active:
                self.bg_precompute_label.setText("Background precompute: paused while painting")
            elif self.save_thread is not None:
                self.bg_precompute_label.setText("Background precompute: paused while saving")
            elif self.export_thread is not None:
                self.bg_precompute_label.setText("Background precompute: paused while exporting")
            else:
                self.bg_precompute_label.setText("Background precompute: paused")
            return

        crop_level = self.workflow_service.mask_work_crop_level_for_slide(slide)
        all_snapshots = self._clone_all_proposals(slide.proposals)
        tasks: list[_PrecomputeTask] = []
        for idx in self._remaining_precompute_indices():
            proposal = slide.proposals[idx]
            if (
                self.workflow_service.has_section_crop_cached(idx, crop_level)
                and self.workflow_service.has_auto_mask_cached(idx, crop_level, mask_method=proposal.mask_preset)
            ):
                continue
            tasks.append(
                _PrecomputeTask(
                    generation=self.bg_precompute_generation + 1,
                    proposal_index=idx,
                    proposal_label=proposal.label,
                    target_proposal=all_snapshots[idx],
                    all_proposals=all_snapshots,
                    mask_method=proposal.mask_preset,
                )
            )

        self.bg_precompute_generation += 1
        self.bg_precompute_active_generation = self.bg_precompute_generation
        if not tasks:
            self.bg_precompute_worker.configure(
                slide,
                [],
                crop_level=crop_level,
                generation=self.bg_precompute_generation,
            )
            self.bg_precompute_worker.set_paused(True)
            self.bg_precompute_label.setText("Background precompute: remaining sections already cached")
            return

        for task in tasks:
            task.generation = self.bg_precompute_generation
        self.bg_precompute_worker.configure(
            slide,
            tasks,
            crop_level=crop_level,
            generation=self.bg_precompute_generation,
        )
        self.bg_precompute_worker.set_paused(False)
        queued_labels = ", ".join(task.proposal_label for task in tasks)
        self.bg_precompute_label.setText(f"Background precompute queued: {queued_labels}")

    def on_background_precompute_status(self, message: str) -> None:
        self.bg_precompute_label.setText(message)

    def on_background_precompute_section_ready(self, idx: int, crop_rgb: object, generation: int, cache_state: str) -> None:
        slide = self.current_slide
        if slide is None or generation != self.bg_precompute_active_generation:
            return
        if not (0 <= idx < len(slide.proposals)):
            return
        if isinstance(crop_rgb, np.ndarray):
            crop_level = self.workflow_service.mask_work_crop_level_for_slide(slide)
            self.workflow_service.store_section_crop(idx, crop_level, crop_rgb)
            self.bg_precompute_label.setText(
                f"Background precompute ready: {slide.proposals[idx].label} ({cache_state})"
            )

    def on_background_precompute_idle(self, generation: int) -> None:
        if generation != self.bg_precompute_active_generation:
            return
        if self._background_precompute_should_pause():
            return
        self.bg_precompute_label.setText("Background precompute: idle")

    def on_background_precompute_failed(self, message: str) -> None:
        self.section_info.append(message)
        self.bg_precompute_label.setText("Background precompute: error")

    def on_editor_painting_state_changed(self, active: bool) -> None:
        self.editor_painting_active = active
        if active:
            if self.bg_precompute_worker is not None:
                self.bg_precompute_worker.set_paused(True)
            self.bg_precompute_label.setText("Background precompute: paused while painting")
        else:
            self._schedule_background_precompute()

    def on_editor_active_layer_changed(self, layer: str) -> None:
        idx = self.layer_combo.findText(layer)
        if idx >= 0 and self.layer_combo.currentIndex() != idx:
            self.layer_combo.blockSignals(True)
            self.layer_combo.setCurrentIndex(idx)
            self.layer_combo.blockSignals(False)
        self.section_info.append(f"Active layer switched by shortcut: {layer}")

    def goto_home(self) -> None:
        self.refresh_home_status()
        if self.bg_precompute_worker is not None:
            self.bg_precompute_worker.set_paused(True)
        self.bg_precompute_label.setText("Background precompute: idle")
        self.pages.setCurrentIndex(self.PAGE_HOME)

    def goto_stage1(self) -> None:
        self.pages.setCurrentIndex(self.PAGE_STAGE1)

    def refresh_home_status(self) -> None:
        slide = self.current_slide
        if slide is None:
            self.home_status.setPlainText(
                "\n".join(
                    [
                        "Current step entry points:",
                        "- Step 1: load NDPI, inspect slide thumbnail, adjust proposal boxes, export crop folders",
                        "- Step 2: batch-predict masks from exported crop folders",
                        "- Step 3: review and edit masks from exported crop folders",
                        "- Step 4: paired Nissl/Myelin registration QC",
                        "",
                        "Current session:",
                        "- no slide loaded",
                        "- Step 2 / Step 3 require exported crop folders",
                        "- Step 4 requires both myelin and nissl roots",
                    ]
                )
            )
            return
        self.home_status.setPlainText(
            "\n".join(
                [
                    "Current step entry points:",
                    "- Step 1: histology preview and box proposal",
                    "- Step 2: batch mask prediction from crop folders",
                    "- Step 3: mask review and annotation from crop folders",
                    "- Step 4: paired Nissl/Myelin registration QC",
                    "",
                    "Current session:",
                    f"- slide: {slide.slide_name}",
                    f"- stain: {slide.stain}",
                    f"- backend: {slide.backend}",
                    f"- proposals: {len(slide.proposals)}",
                    f"- crop export ready: {'yes' if slide.proposals else 'no'}",
                ]
            )
        )

    def select_ndpi_folder(self) -> None:
        default_root = str(self.default_ndpi_root if self.default_ndpi_root.exists() else Path("C:/"))
        path = QFileDialog.getExistingDirectory(self, "Select NDPI Folder", default_root)
        if not path:
            return
        self.current_folder = Path(path)
        self.populate_ndpi_list(self.current_folder)

    def open_single_ndpi(self) -> None:
        default_root = str(self.default_ndpi_root if self.default_ndpi_root.exists() else Path("C:/"))
        path_str, _ = QFileDialog.getOpenFileName(self, "Open NDPI", default_root, "NDPI Files (*.ndpi)")
        if not path_str:
            return
        slide_path = Path(path_str)
        self.current_folder = slide_path.parent
        self.populate_ndpi_list(self.current_folder)
        self.select_ndpi_in_list(slide_path.name)

    def populate_ndpi_list(self, folder: Path) -> None:
        self.ndpi_list.clear()
        for p in self.workflow_service.list_ndpi_files(folder):
            self.ndpi_list.addItem(p.name)
        self.ndpi_status.setText(str(folder))
        self.stage1_info.setPlainText(
            "\n".join(
                [
                    "Step 1: Histology Preview and Box Proposal",
                    f"- folder loaded: {folder}",
                    f"- ndpi files: {self.ndpi_list.count()}",
                    "- select one NDPI to load slide thumbnail and initial proposal boxes",
                    "- bbox proposal remains a manual review step before export",
                ]
            )
        )

    def select_ndpi_in_list(self, filename: str) -> None:
        for idx in range(self.ndpi_list.count()):
            item = self.ndpi_list.item(idx)
            if item.text() == filename:
                self.ndpi_list.setCurrentItem(item)
                break

    def on_ndpi_selected(self) -> None:
        item = self.ndpi_list.currentItem()
        if item is None or self.current_folder is None:
            return
        try:
            result = self.workflow_service.load_slide(self.current_folder / item.text())
            self._bind_loaded_slide(result.messages)
        except Exception as exc:
            self.stage1_info.setPlainText(f"Failed to load NDPI:\n{exc}")

    def _bind_loaded_slide(self, messages: list[str]) -> None:
        slide = self.current_slide
        if slide is None:
            return
        if self.bg_precompute_worker is not None:
            self.bg_precompute_worker.set_paused(True)
        self.bg_precompute_label.setText("Background precompute: idle")
        self.refresh_home_status()
        self.ndpi_status.setText(f"{slide.slide_name} | stain={slide.stain} | backend={slide.backend} | proposals={len(slide.proposals)}")
        self.proposal_count_spin.blockSignals(True)
        self.proposal_count_spin.setValue(len(slide.proposals))
        self.proposal_count_spin.blockSignals(False)

        self.label_view.set_rgb_image(slide.label_preview.width, slide.label_preview.height, slide.label_preview.tobytes("raw", "RGB"))
        self._rebuild_proposal_scene()
        self.rebuild_proposal_cards()
        if self.proposal_items:
            self.proposal_items[0].setSelected(True)
            self.on_proposal_selected(self.proposal_items[0])
        self.stage1_info.setPlainText(
            "\n".join(
                messages
                + [
                    "",
                    "Step 1: Histology Preview and Box Proposal",
                    "- drag red boxes to adjust proposal seed boxes",
                    "- yellow dashed boxes show the actual crop region that Step 2 will read",
                    "- proposal previews are loaded on demand to reduce slide-open latency",
                    "- right pane previews update after drag release",
                    "- each proposal card can run automatic mask generation",
                    "- bottom controls add/remove proposal boxes and continue to Step 2",
                ]
            )
        )
        self._notify_completion("BBox proposal ready")

    def _notify_completion(self, reason: str) -> None:
        app = QApplication.instance()
        if app is not None:
            try:
                app.beep()
            except Exception:
                pass
            try:
                app.alert(self, 5000)
            except Exception:
                pass
        _play_attention_sound()
        if not self.isActiveWindow():
            _flash_taskbar_icon(int(self.winId()))

    def _notify_bbox_ready(self) -> None:
        self._notify_completion("BBox proposal ready")

    def _rebuild_proposal_scene(self) -> None:
        slide = self.current_slide
        self.overview_view.set_rgb_image(slide.overview.width, slide.overview.height, slide.overview.tobytes("raw", "RGB"))
        self.proposal_items.clear()
        self.crop_outline_items.clear()
        for idx, proposal in enumerate(slide.proposals):
            x1, y1, x2, y2 = effective_crop_rect_overview(slide, proposal)
            crop_item = QGraphicsRectItem(QRectF(float(x1), float(y1), float(x2 - x1), float(y2 - y1)))
            crop_item.setPen(QPen(QColor(255, 215, 0), 2, Qt.PenStyle.DashLine))
            crop_item.setBrush(Qt.BrushStyle.NoBrush)
            crop_item.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
            crop_item.setZValue(0.5)
            self.overview_view.scene_obj.addItem(crop_item)
            self.crop_outline_items.append(crop_item)
            item = DraggableProposalItem(
                QRectF(float(proposal.x), float(proposal.y), float(proposal.w), float(proposal.h)),
                label=proposal.label,
                on_changed=self.on_proposal_moved,
                on_drag_finished=self.on_proposal_drag_finished,
                on_selected=self.on_proposal_selected,
            )
            item.setZValue(1.0)
            item.setData(0, idx)
            self.overview_view.scene_obj.addItem(item)
            self.proposal_items.append(item)

    def _update_crop_outline_item(self, idx: int) -> None:
        slide = self.current_slide
        if slide is None:
            return
        if idx < 0 or idx >= len(self.crop_outline_items) or idx >= len(slide.proposals):
            return
        x1, y1, x2, y2 = effective_crop_rect_overview(slide, slide.proposals[idx])
        self.crop_outline_items[idx].setRect(QRectF(float(x1), float(y1), float(x2 - x1), float(y2 - y1)))

    def rebuild_proposal_cards(self) -> None:
        while self.cards_layout.count():
            item = self.cards_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self.proposal_cards.clear()

        slide = self.current_slide
        if slide is None:
            return
        for idx, proposal in enumerate(slide.proposals):
            card = ProposalCard(proposal.label, proposal_index=idx, on_run_mask=self.run_mask_for_proposal)
            self.cards_layout.addWidget(card)
            self.proposal_cards.append(card)
        self.cards_layout.addStretch(1)

    def ensure_proposal_preview(self, idx: int) -> None:
        slide = self.current_slide
        if slide is None:
            return
        if idx < 0 or idx >= len(self.proposal_cards):
            return
        card = self.proposal_cards[idx]
        if card.preview_loaded:
            return
        preview = self.workflow_service.get_preview_crop(
            idx,
            crop_level=self.workflow_service.preview_crop_level_for_slide(slide),
        )
        card.set_preview(preview)

    def on_proposal_moved(self, item: DraggableProposalItem) -> None:
        slide = self.current_slide
        if slide is None:
            return
        idx = int(item.data(0))
        rect = item.scene_rect()
        proposal = slide.proposals[idx]
        proposal.x = int(round(rect.x()))
        proposal.y = int(round(rect.y()))
        proposal.w = int(round(rect.width()))
        proposal.h = int(round(rect.height()))
        self.workflow_service.session_cache.invalidate_proposal(proposal)
        self._update_crop_outline_item(idx)
        if item.isSelected():
            self._set_dimension_inputs_from_proposal(proposal)

    def on_proposal_drag_finished(self, item: DraggableProposalItem) -> None:
        slide = self.current_slide
        if slide is None:
            return
        idx = int(item.data(0))
        proposal = slide.proposals[idx]
        self.workflow_service.update_proposal_bbox(
            idx,
            x=proposal.x,
            y=proposal.y,
            w=proposal.w,
            h=proposal.h,
        )
        self._update_crop_outline_item(idx)
        preview = self.workflow_service.get_preview_crop(idx, crop_level=self.workflow_service.preview_crop_level_for_slide(slide))
        self.proposal_cards[idx].set_preview(preview)

    def on_proposal_selected(self, item: DraggableProposalItem) -> None:
        slide = self.current_slide
        if slide is None:
            return
        idx = int(item.data(0))
        proposal = slide.proposals[idx]
        self._set_dimension_inputs_from_proposal(proposal)
        self.ensure_proposal_preview(idx)
        self.stage1_info.append(f"Selected proposal: {proposal.label}")

    def _set_dimension_inputs_from_proposal(self, proposal) -> None:
        for spin, value in [
            (self.sel_x_spin, proposal.x),
            (self.sel_y_spin, proposal.y),
            (self.sel_w_spin, proposal.w),
            (self.sel_h_spin, proposal.h),
        ]:
            spin.blockSignals(True)
            spin.setValue(int(value))
            spin.blockSignals(False)

    def _selected_proposal_index(self) -> int | None:
        selected_items = [it for it in self.proposal_items if it.isSelected()]
        if not selected_items:
            return None
        return int(selected_items[-1].data(0))

    def apply_selected_box_dimensions(self) -> None:
        slide = self.current_slide
        if slide is None:
            return
        idx = self._selected_proposal_index()
        if idx is None:
            return
        proposal = self.workflow_service.update_proposal_bbox(
            idx,
            x=self.sel_x_spin.value(),
            y=self.sel_y_spin.value(),
            w=max(1, self.sel_w_spin.value()),
            h=max(1, self.sel_h_spin.value()),
        )
        item = self.proposal_items[idx]
        item.set_scene_rect(QRectF(float(proposal.x), float(proposal.y), float(proposal.w), float(proposal.h)))
        self._update_crop_outline_item(idx)
        preview = self.workflow_service.get_preview_crop(idx, crop_level=self.workflow_service.preview_crop_level_for_slide(slide))
        self.proposal_cards[idx].set_preview(preview)

    def add_proposal_box(self) -> None:
        if self.current_slide is None:
            return
        idx = self.workflow_service.add_proposal()
        self._rebuild_proposal_scene()
        self.rebuild_proposal_cards()
        self.proposal_count_spin.blockSignals(True)
        self.proposal_count_spin.setValue(len(self.current_slide.proposals))
        self.proposal_count_spin.blockSignals(False)
        self.proposal_items[idx].setSelected(True)
        self.on_proposal_selected(self.proposal_items[idx])
        self.refresh_home_status()

    def remove_selected_proposal(self) -> None:
        slide = self.current_slide
        if slide is None:
            return
        idx = self._selected_proposal_index()
        if idx is None:
            return
        self.workflow_service.remove_proposal(idx)
        self._rebuild_proposal_scene()
        self.rebuild_proposal_cards()
        self.proposal_count_spin.blockSignals(True)
        self.proposal_count_spin.setValue(len(slide.proposals))
        self.proposal_count_spin.blockSignals(False)
        if self.proposal_items:
            new_idx = max(0, idx - 1)
            self.proposal_items[new_idx].setSelected(True)
            self.on_proposal_selected(self.proposal_items[new_idx])
        self.refresh_home_status()

    def ensure_proposal_count(self, target_count: int) -> None:
        if self.current_slide is None:
            return
        self.workflow_service.ensure_proposal_count(target_count)
        self._rebuild_proposal_scene()
        self.rebuild_proposal_cards()
        self.refresh_home_status()

    def run_mask_for_proposal(self, idx: int) -> None:
        slide = self.current_slide
        if slide is None:
            return
        proposal = slide.proposals[idx]
        crop_rgb, tissue, artifact = self.workflow_service.generate_mask_preview(
            idx,
            crop_level=self.workflow_service.mask_work_crop_level_for_slide(slide),
            mask_method=proposal.mask_preset,
        )
        preview = crop_rgb.astype(np.float32)
        preview[tissue > 0] = 0.65 * preview[tissue > 0] + 0.35 * np.array([255, 0, 0], dtype=np.float32)
        preview[artifact > 0] = 0.65 * preview[artifact > 0] + 0.35 * np.array([0, 255, 255], dtype=np.float32)
        self.proposal_cards[idx].set_preview(np.clip(preview, 0, 255).astype(np.uint8))

    def goto_stage2(self) -> None:
        self.pages.setCurrentIndex(self.PAGE_STAGE2)

    def goto_stage3(self) -> None:
        self.pages.setCurrentIndex(self.PAGE_STAGE3)
        if self.workspace_root is not None and not self.workspace_sections:
            self.refresh_workspace_sections()
        if self.workspace_sections and self.review_section_combo.count() > 0:
            if self.review_section_combo.currentIndex() < 0:
                self.review_section_combo.setCurrentIndex(0)
            else:
                self.load_current_workspace_section()

    def goto_stage4(self) -> None:
        self.pages.setCurrentIndex(self.PAGE_STAGE4)
        if self.step4_myelin_root is None:
            default_myelin = self._default_step4_myelin_root()
            if default_myelin.exists():
                self.step4_myelin_root = default_myelin
        if self.step4_nissl_root is None:
            default_nissl = self._default_step4_nissl_root()
            if default_nissl.exists():
                self.step4_nissl_root = default_nissl
        self.refresh_step4_pairs()

    def goto_stage5(self) -> None:
        self.pages.setCurrentIndex(self.PAGE_STAGE5)
        if self.step4_myelin_root is None:
            default_myelin = self._default_step4_myelin_root()
            if default_myelin.exists():
                self.step4_myelin_root = default_myelin
        if self.step4_nissl_root is None:
            default_nissl = self._default_step4_nissl_root()
            if default_nissl.exists():
                self.step4_nissl_root = default_nissl
        self.refresh_step5_pairs()

    def goto_stage6(self) -> None:
        self.pages.setCurrentIndex(self.PAGE_STAGE6)
        if self.step4_myelin_root is None:
            default_myelin = self._default_step4_myelin_root()
            if default_myelin.exists():
                self.step4_myelin_root = default_myelin
        if self.step4_nissl_root is None:
            default_nissl = self._default_step4_nissl_root()
            if default_nissl.exists():
                self.step4_nissl_root = default_nissl
        self.refresh_step6_pairs()

    def goto_stage7(self) -> None:
        self.pages.setCurrentIndex(self.PAGE_STAGE7)
        if self.step7_myelin_root is None:
            default_myelin = self._default_step4_myelin_root()
            if default_myelin.exists():
                self.step7_myelin_root = default_myelin
        self.refresh_step7_sections()

    def goto_stage8(self) -> None:
        self.pages.setCurrentIndex(self.PAGE_STAGE8)
        self._refresh_step8_info()

    @staticmethod
    def _preferred_existing_path(raw: str) -> Path | None:
        candidate = Path(raw)
        if candidate.exists():
            return candidate
        text = str(raw).strip()
        if len(text) >= 3 and text[1] == ":" and text[2] in {"\\", "/"}:
            drive = text[0].lower()
            tail = text[3:].replace("\\", "/")
            alt = Path(f"/mnt/{drive}") / Path(tail)
            if alt.exists():
                return alt
        return None

    def _default_crop_workspace_root(self) -> Path:
        preferred = self._preferred_existing_path(r"D:\Research\Image Analysis\Nanozoomer scans")
        if preferred is not None:
            return preferred
        if self.workspace_root is not None and self.workspace_root.exists():
            return self.workspace_root
        if self.current_slide is not None:
            return self.current_slide.slide_path.parent
        return Path("C:/")

    def _default_step4_myelin_root(self) -> Path:
        preferred = self._preferred_existing_path(
            r"D:\Research\Image Analysis\Nanozoomer scans\20250327 rat myelin quantification\Tissue&Masks"
        )
        return preferred if preferred is not None else self._default_crop_workspace_root()

    def _default_step4_nissl_root(self) -> Path:
        preferred = self._preferred_existing_path(
            r"D:\Research\Image Analysis\Nanozoomer scans\20250424 Nissl cytoarchitectonic counterpart\Tissue&Masks"
        )
        return preferred if preferred is not None else self._default_crop_workspace_root()

    def _step4_registry_path(self) -> Path | None:
        return default_pair_registry_path(self.step4_myelin_root, self.step4_nissl_root)

    def _step4_registration_masks_root(self) -> Path | None:
        return default_pair_registration_masks_root(self.step4_myelin_root, self.step4_nissl_root)

    def _step4_registration_mask_paths(self, pair: WorkspacePair) -> dict[str, Path]:
        return pair_registration_mask_paths(self._step4_registration_masks_root(), pair.pair_key)

    def _step4_registration_mask_relpath(self, path: Path | None) -> str | None:
        if path is None:
            return None
        root = self._step4_registration_masks_root()
        if root is None:
            return str(path)
        try:
            return str(path.resolve().relative_to(root.parent.resolve()))
        except Exception:
            return str(path)

    def _load_step4_pair_section(
        self,
        pair: WorkspacePair,
        side: str,
    ) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray, dict[str, str]]:
        item = pair.myelin_item if side == "myelin" else pair.nissl_item
        metadata, crop_rgb, tissue, artifact, source_info = load_workspace_section(item)
        reg_path = self._step4_registration_mask_paths(pair).get(side)
        if reg_path is not None and reg_path.exists():
            tissue, artifact = load_masks_from_label_path(reg_path, shape_hw=crop_rgb.shape[:2])
            source_info = {
                "mask_source": "pair_registration_mask",
                "mask_path": str(reg_path),
                "artifact_source": str(reg_path),
            }
        return metadata, crop_rgb, tissue, artifact, source_info

    def _invalidate_step4_pair_cache(self) -> None:
        with self.step4_pair_cache_lock:
            self.step4_pair_cache_generation += 1
            self.step4_pair_cache.clear()
            self.step4_pair_cache_order.clear()
            self.step4_pair_prefetch_inflight.clear()

    def _step4_cached_pair_data(
        self,
        pair: WorkspacePair,
    ) -> dict[str, tuple[dict, np.ndarray, np.ndarray, np.ndarray, dict[str, str]]] | None:
        with self.step4_pair_cache_lock:
            cached = self.step4_pair_cache.get(pair.pair_key)
            if not isinstance(cached, dict):
                return None
            if "myelin" not in cached or "nissl" not in cached:
                return None
            return cached

    def _store_step4_pair_cache(
        self,
        pair_key: str,
        data: dict[str, tuple[dict, np.ndarray, np.ndarray, np.ndarray, dict[str, str]]],
        generation: int,
    ) -> None:
        with self.step4_pair_cache_lock:
            if generation != self.step4_pair_cache_generation:
                self.step4_pair_prefetch_inflight.discard(pair_key)
                return
            self.step4_pair_cache[pair_key] = data
            if pair_key in self.step4_pair_cache_order:
                self.step4_pair_cache_order.remove(pair_key)
            self.step4_pair_cache_order.append(pair_key)
            while len(self.step4_pair_cache_order) > self.step4_pair_cache_capacity:
                victim = self.step4_pair_cache_order.pop(0)
                self.step4_pair_cache.pop(victim, None)
                self.step4_pair_prefetch_inflight.discard(victim)
            self.step4_pair_prefetch_inflight.discard(pair_key)

    def _remove_step4_pair_cache(self, pair_key: str) -> None:
        with self.step4_pair_cache_lock:
            self.step4_pair_cache.pop(pair_key, None)
            if pair_key in self.step4_pair_cache_order:
                self.step4_pair_cache_order.remove(pair_key)
            self.step4_pair_prefetch_inflight.discard(pair_key)

    def _prefetch_step4_pair(self, pair: WorkspacePair, generation: int) -> None:
        try:
            data = {
                "myelin": self._load_step4_pair_section(pair, "myelin"),
                "nissl": self._load_step4_pair_section(pair, "nissl"),
            }
            self._store_step4_pair_cache(pair.pair_key, data, generation)
        except Exception:
            with self.step4_pair_cache_lock:
                self.step4_pair_prefetch_inflight.discard(pair.pair_key)

    def _schedule_step4_pair_prefetch(self, center_index: int) -> None:
        if not self.step4_pairs:
            return
        with self.step4_pair_cache_lock:
            generation = self.step4_pair_cache_generation
        target_indices: list[int] = []
        for delta in (1, 2, -1, -2):
            idx = center_index + delta
            if 0 <= idx < len(self.step4_pairs):
                target_indices.append(idx)
        for idx in target_indices:
            pair = self.step4_pairs[idx]
            pair_key = pair.pair_key
            with self.step4_pair_cache_lock:
                if pair_key in self.step4_pair_cache or pair_key in self.step4_pair_prefetch_inflight:
                    continue
                self.step4_pair_prefetch_inflight.add(pair_key)
            thread = threading.Thread(
                target=self._prefetch_step4_pair,
                args=(pair, generation),
                name=f"step4-prefetch-{pair_key}",
                daemon=True,
            )
            thread.start()

    def select_step4_myelin_root(self) -> None:
        start_dir = str(self.step4_myelin_root or self._default_step4_myelin_root())
        path = QFileDialog.getExistingDirectory(self, "Select Myelin Tissue&Masks Root", start_dir)
        if not path:
            return
        self.step4_myelin_root = Path(path)
        self.refresh_step4_pairs()

    def select_step4_nissl_root(self) -> None:
        start_dir = str(self.step4_nissl_root or self._default_step4_nissl_root())
        path = QFileDialog.getExistingDirectory(self, "Select Nissl Tissue&Masks Root", start_dir)
        if not path:
            return
        self.step4_nissl_root = Path(path)
        self.refresh_step4_pairs()

    def select_workspace_root(self) -> None:
        start_dir = str(self._default_crop_workspace_root())
        path = QFileDialog.getExistingDirectory(self, "Select Crop Workspace Root", start_dir)
        if not path:
            return
        self.workspace_root = Path(path)
        self.refresh_workspace_sections()

    def refresh_workspace_sections(self) -> None:
        self.workspace_sections = list_workspace_sections(self.workspace_root) if self.workspace_root is not None else []
        self.workspace_root_label.setText(str(self.workspace_root) if self.workspace_root is not None else "No crop workspace selected")
        self.workspace_section_list.clear()
        self.review_section_combo.blockSignals(True)
        self.review_section_combo.clear()
        for item in self.workspace_sections:
            if item.has_masks:
                suffix = " [mask]"
            elif item.has_prepared_work:
                suffix = " [crop + downsampled]"
            else:
                suffix = " [crop-only]"
            self.workspace_section_list.addItem(f"{item.label}{suffix}")
            self.review_section_combo.addItem(item.label, item.label)
        self.review_section_combo.blockSignals(False)
        self.stage2_info.append(f"Workspace refresh: {len(self.workspace_sections)} section folder(s) found.")
        if self.workspace_sections:
            self.current_workspace_index = min(self.current_workspace_index, len(self.workspace_sections) - 1)
            self.review_section_combo.setCurrentIndex(self.current_workspace_index)
            if self.pages.currentIndex() == self.PAGE_STAGE3:
                self.load_current_workspace_section()
        else:
            self.current_workspace_index = 0
            self.section_label.setText("No section selected")
            self.section_editor.set_section(np.zeros((1, 1, 3), dtype=np.uint8), np.zeros((1, 1), dtype=np.uint8), np.zeros((1, 1), dtype=np.uint8))
            self.section_info.setPlainText("No workspace section loaded")

    def select_review_mask_root(self) -> None:
        start_dir = str(self.review_mask_root or self.workspace_root or (self.current_slide.slide_path.parent if self.current_slide is not None else Path("C:/")))
        path = QFileDialog.getExistingDirectory(self, "Select External Predicted Mask Root", start_dir)
        if not path:
            return
        self.review_mask_root = Path(path)
        self.review_mask_root_label.setText(f"Mask source: {self.review_mask_root}")
        self.section_info.append(f"External predicted mask root selected: {self.review_mask_root}")
        if self.pages.currentIndex() == self.PAGE_STAGE3 and self.workspace_sections:
            self.load_current_workspace_section()

    def clear_review_mask_root(self) -> None:
        self.review_mask_root = None
        self.review_mask_root_label.setText("Mask source: workspace section folders")
        self.section_info.append("Step 3 mask source reverted to workspace section folders.")
        if self.pages.currentIndex() == self.PAGE_STAGE3 and self.workspace_sections:
            self.load_current_workspace_section()

    def select_all_workspace_sections(self) -> None:
        for idx in range(self.workspace_section_list.count()):
            self.workspace_section_list.item(idx).setSelected(True)

    def _selected_workspace_items(self) -> list[WorkspaceSection]:
        selected = sorted({index.row() for index in self.workspace_section_list.selectedIndexes()})
        if not selected:
            return []
        return [self.workspace_sections[idx] for idx in selected if 0 <= idx < len(self.workspace_sections)]

    def _set_stage2_busy(self, busy: bool, *, action_text: str | None = None) -> None:
        self.refresh_workspace_button.setEnabled(not busy)
        self.select_all_workspace_button.setEnabled(not busy)
        self.prepare_work_button.setEnabled(not busy)
        self.predict_masks_button.setEnabled(not busy)
        if not busy:
            self.prepare_work_button.setText("Prepare Work Images For Selected")
            self.predict_masks_button.setText("Run Mask Prediction For Selected")
            return
        if action_text == "prepare":
            self.prepare_work_button.setEnabled(False)
            self.prepare_work_button.setText("Preparing...")
        if action_text == "predict":
            self.predict_masks_button.setEnabled(False)
            self.predict_masks_button.setText("Predicting...")

    def _update_stage2_progress(self, payload: object) -> None:
        if not isinstance(payload, dict):
            return
        label = str(payload.get("item_label") or "?")
        item_index = int(payload.get("item_index") or 0)
        total_items = int(payload.get("total_items") or 0)
        step_index = int(payload.get("step_index") or 0)
        step_count = int(payload.get("step_count") or 0)
        stage = str(payload.get("stage") or "working")
        stage_elapsed_s = float(payload.get("stage_elapsed_s") or 0.0)
        section_elapsed_s = float(payload.get("section_elapsed_s") or 0.0)
        progress_percent = max(0, min(100, int(payload.get("progress_percent") or 0)))
        self.stage2_progress_bar.setValue(progress_percent)
        self.stage2_progress_label.setText(
            f"Step 2 progress: {item_index}/{total_items} | {label} | "
            f"step {step_index}/{step_count}: {stage} | stage {stage_elapsed_s:.2f}s | total {section_elapsed_s:.2f}s"
        )

    def prepare_workspace_work_images(self) -> None:
        if self.prepare_thread is not None or self.predict_thread is not None:
            self.stage2_info.append("Step 2 is already busy.")
            return
        items = self._selected_workspace_items()
        if not items:
            self.stage2_info.append("Select at least one section folder before preparing work images.")
            return
        profile_data = self.predict_profile_combo.currentData()
        compute_profile = profile_data if isinstance(profile_data, str) else MASK_COMPUTE_PROFILE_STANDARD
        self._set_stage2_busy(True, action_text="prepare")
        self.stage2_progress_bar.setValue(0)
        self.stage2_progress_label.setText("Step 2 progress: starting work-image preparation ...")
        self.prepare_thread = QThread(self)
        self.prepare_worker = DownsamplePrepareWorker(items, compute_profile=compute_profile)
        self.prepare_worker.moveToThread(self.prepare_thread)
        self.prepare_thread.started.connect(self.prepare_worker.run)
        self.prepare_worker.progress.connect(self.stage2_info.append)
        self.prepare_worker.stage_progress.connect(self._update_stage2_progress)
        self.prepare_worker.finished.connect(self.on_prepare_finished)
        self.prepare_worker.failed.connect(self.on_prepare_failed)
        self.prepare_worker.finished.connect(self.prepare_thread.quit)
        self.prepare_worker.failed.connect(self.prepare_thread.quit)
        self.prepare_thread.finished.connect(self.prepare_worker.deleteLater)
        self.prepare_thread.finished.connect(self.prepare_thread.deleteLater)
        self.prepare_thread.start()

    def predict_masks_for_workspace(self) -> None:
        if self.predict_thread is not None or self.prepare_thread is not None:
            self.stage2_info.append("Step 2 is already busy.")
            return
        items = self._selected_workspace_items()
        if not items:
            self.stage2_info.append("Select at least one section folder before running mask prediction.")
            return
        preset = self.predict_preset_combo.currentData()
        override = preset if isinstance(preset, str) and preset != "__auto__" else None
        profile_data = self.predict_profile_combo.currentData()
        compute_profile = profile_data if isinstance(profile_data, str) else MASK_COMPUTE_PROFILE_STANDARD
        self._set_stage2_busy(True, action_text="predict")
        self.stage2_progress_bar.setValue(0)
        self.stage2_progress_label.setText("Step 2 progress: starting mask prediction ...")
        self.predict_thread = QThread(self)
        self.predict_worker = MaskPredictionWorker(
            items,
            mask_method_override=override,
            compute_profile=compute_profile,
        )
        self.predict_worker.moveToThread(self.predict_thread)
        self.predict_thread.started.connect(self.predict_worker.run)
        self.predict_worker.progress.connect(self.stage2_info.append)
        self.predict_worker.stage_progress.connect(self._update_stage2_progress)
        self.predict_worker.finished.connect(self.on_predict_finished)
        self.predict_worker.failed.connect(self.on_predict_failed)
        self.predict_worker.finished.connect(self.predict_thread.quit)
        self.predict_worker.failed.connect(self.predict_thread.quit)
        self.predict_thread.finished.connect(self.predict_worker.deleteLater)
        self.predict_thread.finished.connect(self.predict_thread.deleteLater)
        self.predict_thread.start()

    def on_prepare_finished(self, summary: object) -> None:
        data = dict(summary) if isinstance(summary, dict) else {}
        prepared = data.get("prepared", [])
        compute_profile = data.get("compute_profile", "unknown")
        self.stage2_info.append(f"Work-image preparation finished: {len(prepared)} section(s). profile={compute_profile}")
        if prepared:
            self.stage2_info.append(f"Prepared folders: {', '.join(prepared)}")
        self.prepare_worker = None
        self.prepare_thread = None
        self._set_stage2_busy(False)
        self.stage2_progress_bar.setValue(100)
        self.stage2_progress_label.setText(
            f"Step 2 progress: work-image preparation finished for {len(prepared)} section(s) | profile={compute_profile}"
        )
        self.refresh_workspace_sections()
        self._notify_completion("Work-image preparation finished")

    def on_prepare_failed(self, message: str) -> None:
        self.stage2_info.append(message)
        self.prepare_worker = None
        self.prepare_thread = None
        self._set_stage2_busy(False)
        self.stage2_progress_label.setText("Step 2 progress: work-image preparation failed")

    def on_predict_finished(self, summary: object) -> None:
        data = dict(summary) if isinstance(summary, dict) else {}
        predicted = data.get("predicted", [])
        compute_profile = data.get("compute_profile", "unknown")
        self.stage2_info.append(f"Mask prediction finished: {len(predicted)} section(s). profile={compute_profile}")
        if predicted:
            self.stage2_info.append(f"Predicted folders: {', '.join(predicted)}")
        self.predict_worker = None
        self.predict_thread = None
        self._set_stage2_busy(False)
        self.stage2_progress_bar.setValue(100)
        self.stage2_progress_label.setText(
            f"Step 2 progress: mask prediction finished for {len(predicted)} section(s) | profile={compute_profile}"
        )
        self.refresh_workspace_sections()
        self._notify_completion("Mask batch prediction finished")

    def on_predict_failed(self, message: str) -> None:
        self.stage2_info.append(message)
        self.predict_worker = None
        self.predict_thread = None
        self._set_stage2_busy(False)
        self.stage2_progress_label.setText("Step 2 progress: mask prediction failed")

    def _current_workspace_item(self) -> WorkspaceSection | None:
        if not self.workspace_sections:
            return None
        if not (0 <= self.current_workspace_index < len(self.workspace_sections)):
            return None
        return self.workspace_sections[self.current_workspace_index]

    def on_review_section_changed(self, index: int) -> None:
        if index < 0 or index >= len(self.workspace_sections):
            return
        self.current_workspace_index = index
        self.load_current_workspace_section()

    def load_current_workspace_section(self) -> None:
        item = self._current_workspace_item()
        if item is None:
            return
        metadata, crop_rgb, tissue, artifact, source_info = load_workspace_section(
            item,
            external_mask_root=self.review_mask_root,
        )
        self.section_editor.set_section(crop_rgb, tissue, artifact)
        self.section_label.setText(f"{self.current_workspace_index + 1}/{len(self.workspace_sections)} | {item.label}")
        preset = (
            str((metadata.get("mask_prediction") or {}).get("mask_preset_selected"))
            if (metadata.get("mask_prediction") or {}).get("mask_preset_selected")
            else default_mask_preset_for_stain(item.stain)
        )
        compute_profile = (
            str((metadata.get("mask_prediction") or {}).get("mask_compute_profile"))
            if (metadata.get("mask_prediction") or {}).get("mask_compute_profile")
            else MASK_COMPUTE_PROFILE_STANDARD
        )
        self.mask_preset_combo.blockSignals(True)
        preset_index = self.mask_preset_combo.findData(preset)
        self.mask_preset_combo.setCurrentIndex(max(0, preset_index))
        self.mask_preset_combo.blockSignals(False)
        self.review_profile_combo.blockSignals(True)
        profile_index = self.review_profile_combo.findData(compute_profile)
        self.review_profile_combo.setCurrentIndex(max(0, profile_index))
        self.review_profile_combo.blockSignals(False)
        self.mirror_check.blockSignals(True)
        self.mirror_check.setChecked(False)
        self.mirror_check.blockSignals(False)
        self.section_editor.set_mirror(False)
        self.update_mask_stats()
        external_mask = find_external_prediction_mask(item, self.review_mask_root)
        active_mask_root = str(self.review_mask_root) if self.review_mask_root is not None else "workspace folder"
        self.section_info.setPlainText(
            "\n".join(
                [
                    f"label: {item.label}",
                    f"section_dir: {item.section_dir}",
                    f"stain: {item.stain}",
                    f"review_mask_root: {active_mask_root}",
                    f"mask_source: {source_info.get('mask_source')}",
                    f"mask_path: {source_info.get('mask_path')}",
                    f"pipeline_stage: {metadata.get('pipeline_stage', 'unknown')}",
                    f"workspace_status: {json.dumps(metadata.get('workspace_status', {}), ensure_ascii=True)}",
                    f"mask_compute_profile: {compute_profile}",
                    "",
                    "Editing:",
                    "- T: switch active annotation to tissue",
                    "- A: switch active annotation to artifact",
                    "- N: toggle raw image show/hide (mask-only view when hidden)",
                    "- default mode is hand/grab; press P to enter brush mode",
                    "- left mouse: paint active layer when brush mode is on",
                    "- right mouse: erase both tissue and artifact under the stroke",
                    "- M: toggle mask overlay show/hide",
                    "- P: toggle brush mode on/off",
                    "- H: force hand/grab mode; press again to return to the previous tool if H switched you there",
                    "- L: line-erase tool; click start and end to cut a thin gap through both masks",
                    "- when brush mode is off: wheel zooms and left-drag pans the image",
                    "- hover a component and press D to delete that whole connected component from the active layer",
                    "- C: run Close + Fill Tissue Gaps",
                    "- Z: undo up to the last 5 edit operations",
                    "- S: save and move to the next section",
                    "- tissue has priority: artifact painting never overwrites existing tissue pixels",
                    f"- current masks are loaded from {'external predictions' if external_mask is not None else 'the section folder'}",
                    "- Save writes back into the same folder and keeps a backup revision",
                ]
            )
        )

    def refresh_current_mask(self) -> None:
        item = self._current_workspace_item()
        if item is None:
            return
        _, crop_rgb, _, _, _ = load_workspace_section(item)
        preset = self.current_mask_preset()
        profile_data = self.review_profile_combo.currentData()
        compute_profile = profile_data if isinstance(profile_data, str) else MASK_COMPUTE_PROFILE_STANDARD
        tissue, artifact, compute_info = compute_auto_masks_resampled(
            crop_rgb,
            item.stain,
            method=preset,
            compute_profile=compute_profile,
        )
        self.section_editor.set_section(crop_rgb, tissue, artifact)
        self.update_mask_stats()
        self.section_info.append(
            f"Auto-mask refreshed with preset: {preset} | profile={compute_profile} | working={compute_info.get('working_shape_hw')}"
        )

    def current_mask_preset(self) -> str:
        data = self.mask_preset_combo.currentData()
        if not isinstance(data, str):
            return MASK_PRESET_LATEST_CONTEXTUAL
        return data

    def on_mask_preset_changed(self) -> None:
        preset = self.current_mask_preset()
        self.section_info.append(f"Mask preset selected: {preset}")

    def close_and_fill_tissue_gaps(self) -> None:
        self.section_editor.close_and_fill_tissue_gaps()
        self.update_mask_stats()

    def undo_last_edit(self) -> None:
        if self.section_editor.undo_last_action():
            self.update_mask_stats()
            self.section_info.append("Undo applied: restored previous mask state")
        else:
            self.section_info.append("Undo skipped: no previous mask state available")

    def save_current_revision_state(self) -> None:
        self._save_current_workspace_masks(move_to_next=False)

    def save_and_move_to_next(self) -> None:
        self._save_current_workspace_masks(move_to_next=True)

    def _save_current_workspace_masks(self, *, move_to_next: bool) -> None:
        item = self._current_workspace_item()
        if item is None:
            return
        current_idx = self.current_workspace_index
        _, crop_rgb, _, _, _ = load_workspace_section(item)
        tissue, artifact = self.section_editor.current_masks()
        backup_dir = save_workspace_review(
            item,
            crop_rgb,
            tissue,
            artifact,
            mask_preset=self.current_mask_preset(),
        )
        self.section_info.append(
            f"Saved masks to folder: {item.label} | backup={backup_dir if backup_dir is not None else 'none'}"
        )
        target_idx = current_idx
        if move_to_next and self.workspace_sections:
            target_idx = min(current_idx + 1, len(self.workspace_sections) - 1)
        self.current_workspace_index = target_idx
        self.refresh_workspace_sections()
        if self.review_section_combo.count() > self.current_workspace_index:
            self.review_section_combo.setCurrentIndex(self.current_workspace_index)

    def prev_section(self) -> None:
        if not self.workspace_sections:
            return
        self.current_workspace_index = max(0, self.current_workspace_index - 1)
        self.review_section_combo.setCurrentIndex(self.current_workspace_index)

    def next_section(self) -> None:
        if not self.workspace_sections:
            return
        self.current_workspace_index = min(len(self.workspace_sections) - 1, self.current_workspace_index + 1)
        self.review_section_combo.setCurrentIndex(self.current_workspace_index)

    def _current_pair(self) -> WorkspacePair | None:
        if not self.step4_pairs:
            return None
        if not (0 <= self.current_pair_index < len(self.step4_pairs)):
            return None
        return self.step4_pairs[self.current_pair_index]

    def _step4_pair_review(self, pair: WorkspacePair) -> dict:
        return dict(self.step4_pair_registry.get(pair.pair_key) or {})

    def _step4_registration_status(self, pair: WorkspacePair) -> str:
        review = self._step4_pair_review(pair)
        status = review.get("registration_status")
        if isinstance(status, str) and status in {"unreviewed", "usable", "unusable"}:
            return status
        if "registration_usable" in review:
            return "usable" if bool(review.get("registration_usable")) else "unusable"
        return "unreviewed"

    def _step4_pair_display_text(self, pair: WorkspacePair) -> str:
        status = self._step4_registration_status(pair)
        suffix = {
            "usable": "[usable]",
            "unusable": "[unusable]",
            "unreviewed": "[unreviewed]",
        }[status]
        return f"{pair.display_label} {suffix}"

    def _step4_editor_for_side(self, side: str) -> MaskEditorLabel:
        return self.step4_myelin_editor if side == "myelin" else self.step4_nissl_editor

    def _sync_step4_component_marks(self) -> None:
        self.step4_myelin_editor.set_component_group_marks(self.step4_component_groups.get("myelin", {}))
        self.step4_nissl_editor.set_component_group_marks(self.step4_component_groups.get("nissl", {}))

    def _step4_side_group_flip_map_from_review(self, review: dict, side: str) -> dict[int, bool]:
        raw = dict((review.get("group_flip_lr") or {}).get(side) or {})
        out: dict[int, bool] = {}
        for group_key, value in raw.items():
            try:
                group_id = int(group_key)
            except Exception:
                continue
            if group_id in {1, 2}:
                out[group_id] = bool(value)
        return out

    def _sync_step4_group_flip_controls(self) -> None:
        side = self.step4_active_side
        group_flips = self.step4_group_flips.get(side, {})
        self.step4_group_flip_side_label.setText(f"Active side: {side}")
        self.step4_group1_flip_check.blockSignals(True)
        self.step4_group2_flip_check.blockSignals(True)
        self.step4_group1_flip_check.setChecked(bool(group_flips.get(1, False)))
        self.step4_group2_flip_check.setChecked(bool(group_flips.get(2, False)))
        self.step4_group1_flip_check.blockSignals(False)
        self.step4_group2_flip_check.blockSignals(False)

    def _step4_side_group_map_from_review(self, review: dict, side: str) -> dict[int, int]:
        groups = dict((review.get("component_groups") or {}).get(side) or {})
        out: dict[int, int] = {}
        for group_key, ranks in groups.items():
            try:
                group_id = int(group_key)
            except Exception:
                continue
            if group_id not in {1, 2}:
                continue
            if isinstance(ranks, list) and ranks:
                try:
                    out[int(ranks[0])] = group_id
                except Exception:
                    continue
        return out

    def _normalized_step4_group_map(self, group_map: dict[int, int], component_count: int) -> dict[int, int]:
        normalized = {int(rank): int(group) for rank, group in group_map.items() if int(rank) > 0 and int(group) in {1, 2} and int(rank) <= component_count}
        if not normalized:
            return {1: 1} if component_count > 0 else {}
        selected_ranks = sorted(normalized)
        unique_groups = sorted(set(normalized.values()))
        if len(selected_ranks) == 1:
            return {selected_ranks[0]: 1}
        rank_for_group: dict[int, int] = {}
        for rank in selected_ranks:
            group = normalized[rank]
            if group not in rank_for_group:
                rank_for_group[group] = rank
        out: dict[int, int] = {}
        if 1 in rank_for_group:
            out[rank_for_group[1]] = 1
        if 2 in rank_for_group:
            out[rank_for_group[2]] = 2
        if not out:
            out[selected_ranks[0]] = 1
            if len(selected_ranks) > 1:
                out[selected_ranks[1]] = 2
        elif 1 not in out:
            only_rank = next(iter(out))
            out = {only_rank: 1}
        return out

    def _step4_selection_text(self, side: str) -> str:
        editor = self._step4_editor_for_side(side)
        summary = editor.combined_component_summary()
        component_count = len(summary)
        normalized = self._normalized_step4_group_map(self.step4_component_groups.get(side, {}), component_count)
        group_to_rank = {group: rank for rank, group in normalized.items()}
        area_by_rank = {entry["rank"]: entry["area"] for entry in summary}
        parts = [f"components={component_count}"]
        for group_id in (1, 2):
            rank = group_to_rank.get(group_id)
            if rank is None:
                parts.append(f"group{group_id}=none")
            else:
                parts.append(f"group{group_id}=rank{rank}({area_by_rank.get(rank, 0)} px)")
        return ", ".join(parts)

    def _update_step4_root_status_text(self) -> None:
        registry_path = self._step4_registry_path()
        registration_masks_root = self._step4_registration_masks_root()
        unreviewed_count = sum(1 for pair in self.step4_pairs if self._step4_registration_status(pair) == "unreviewed")
        self.step4_root_status.setPlainText(
            "\n".join(
                [
                    f"myelin_root: {self.step4_myelin_root}",
                    f"nissl_root: {self.step4_nissl_root}",
                    f"pair_combinations_all: {len(self.step4_all_pairs)}",
                    f"pair_combinations_shown: {len(self.step4_pairs)}",
                    f"unreviewed_pairs: {unreviewed_count}",
                    f"pair_registry: {registry_path if registry_path is not None else 'none'}",
                    f"registration_masks_root: {registration_masks_root if registration_masks_root is not None else 'none'}",
                    "pairing rule: same animal id and section number within +/- 1",
                ]
            )
        )

    def refresh_step4_pairs(self) -> None:
        current_key = self._current_pair().pair_key if self._current_pair() is not None else None
        self._invalidate_step4_pair_cache()
        self.step4_pair_list.clear()
        self.step4_pairs = []
        registry_path = self._step4_registry_path()
        self.step4_pair_registry = load_pair_registry(registry_path)
        if self.step4_myelin_root is None or self.step4_nissl_root is None:
            self.step4_root_status.setPlainText(
                "\n".join(
                    [
                        f"myelin_root: {self.step4_myelin_root or 'not set'}",
                        f"nissl_root: {self.step4_nissl_root or 'not set'}",
                        "pair_count: 0",
                    ]
                )
            )
            self.step4_pair_label.setText("No pair selected")
            return
        self.step4_all_pairs = list_cross_stain_pairs(self.step4_myelin_root, self.step4_nissl_root)
        self.step4_pairs = list(self.step4_all_pairs)
        for pair in self.step4_pairs:
            self.step4_pair_list.addItem(self._step4_pair_display_text(pair))
        self._update_step4_root_status_text()
        if self.step4_pairs:
            matched_idx = next((i for i, pair in enumerate(self.step4_pairs) if pair.pair_key == current_key), None)
            if matched_idx is not None:
                self.current_pair_index = matched_idx
            else:
                self.current_pair_index = min(self.current_pair_index, len(self.step4_pairs) - 1)
            self.step4_pair_list.setCurrentRow(self.current_pair_index)
            self.load_current_pair()
        else:
            self.current_pair_index = 0
            self.step4_pair_label.setText("No pair selected")
            self.step4_info.setPlainText("No cross-stain pairs found with the current roots.")

    def _step5_pair_display_text(self, pair: WorkspacePair) -> str:
        review = self._step4_pair_review(pair)
        multi_group = bool(review.get("multi_group_registration"))
        suffix = "[multi-group]" if multi_group else "[single-group]"
        return f"{pair.display_label} {suffix}"

    def _step5_runs_root(self) -> Path | None:
        self.step5_runs_root = default_pair_registration_runs_root(self.step4_myelin_root, self.step4_nissl_root)
        return self.step5_runs_root

    def _current_step5_pair(self) -> WorkspacePair | None:
        if 0 <= self.current_step5_pair_index < len(self.step5_pairs):
            return self.step5_pairs[self.current_step5_pair_index]
        return None

    def _set_step5_storyboard(self, storyboard_path: Path | None) -> None:
        if storyboard_path is None or not storyboard_path.exists():
            self.step5_storyboard_label.setText("No registration storyboard yet")
            self.step5_storyboard_label.setPixmap(QPixmap())
            return
        pixmap = QPixmap(str(storyboard_path))
        if pixmap.isNull():
            self.step5_storyboard_label.setText(f"Failed to load storyboard:\n{storyboard_path}")
            return
        self.step5_storyboard_label.setText("")
        self.step5_storyboard_label.setPixmap(pixmap)
        self.step5_storyboard_label.resize(pixmap.size())

    def refresh_step5_pairs(self) -> None:
        current_key = (
            self.step5_pairs[self.current_step5_pair_index].pair_key
            if self.step5_pairs and 0 <= self.current_step5_pair_index < len(self.step5_pairs)
            else None
        )
        runs_root = self._step5_runs_root()
        ants_bin = find_ants_bin()
        self.step5_pair_list.clear()
        self.step5_pairs = []
        registry_path = self._step4_registry_path()
        self.step4_pair_registry = load_pair_registry(registry_path)
        if self.step4_myelin_root is None or self.step4_nissl_root is None:
            self.step5_root_status.setPlainText(
                "\n".join(
                    [
                        f"myelin_root: {self.step4_myelin_root or 'not set'}",
                        f"nissl_root: {self.step4_nissl_root or 'not set'}",
                        f"registration_runs_root: {runs_root or 'not set'}",
                        f"ants_bin: {ants_bin or 'not found'}",
                        "usable_pairs: 0",
                    ]
                )
            )
            self.step5_pair_label.setText("No registration pair selected")
            self.step5_progress_bar.setValue(0)
            self.step5_progress_label.setText("Step 5 progress: idle")
            self._set_step5_acceptance_summary(None)
            self._set_step5_storyboard(None)
            return
        all_pairs = list_cross_stain_pairs(self.step4_myelin_root, self.step4_nissl_root)
        self.step5_pairs = [pair for pair in all_pairs if self._step4_registration_status(pair) == "usable"]
        for pair in self.step5_pairs:
            self.step5_pair_list.addItem(self._step5_pair_display_text(pair))
        self.step5_root_status.setPlainText(
            "\n".join(
                [
                    f"myelin_root: {self.step4_myelin_root}",
                    f"nissl_root: {self.step4_nissl_root}",
                    f"usable_pairs: {len(self.step5_pairs)}",
                    f"pair_registry: {registry_path if registry_path is not None else 'none'}",
                    f"registration_runs_root: {runs_root if runs_root is not None else 'none'}",
                    f"ants_bin: {ants_bin if ants_bin is not None else 'not found'}",
                    "only Step 4 usable pairs are shown here",
                ]
            )
        )
        if self.step5_pairs:
            matched_idx = next((i for i, pair in enumerate(self.step5_pairs) if pair.pair_key == current_key), None)
            self.current_step5_pair_index = matched_idx if matched_idx is not None else min(self.current_step5_pair_index, len(self.step5_pairs) - 1)
            self.step5_pair_list.setCurrentRow(self.current_step5_pair_index)
            self.on_step5_pair_changed(self.current_step5_pair_index)
        else:
            self.current_step5_pair_index = 0
            self.step5_pair_label.setText("No usable registration pair selected")
            self._set_step5_acceptance_summary(None)
            self.step5_info.setPlainText(
                "\n".join(
                    [
                        "No usable pair is currently available.",
                        "Review pairs in Step 4 and mark them Usable to include them here.",
                    ]
                )
            )
            self._set_step5_storyboard(None)

    def on_step4_pair_changed(self, index: int) -> None:
        if index < 0 or index >= len(self.step4_pairs):
            return
        self.current_pair_index = index
        self.load_current_pair()

    def on_step5_pair_changed(self, index: int) -> None:
        if index < 0 or index >= len(self.step5_pairs):
            return
        self.current_step5_pair_index = index
        pair = self.step5_pairs[index]
        review = self._step4_pair_review(pair)
        multi_group = bool(review.get("multi_group_registration"))
        reg_files = dict(review.get("registration_mask_files") or {})
        approved = dict(review.get("approved_registration") or {})
        latest_run = latest_registration_run_dir(self._step5_runs_root(), pair.pair_key)
        latest_manifest: dict | None = None
        if latest_run is not None:
            manifest_path = latest_run / "run_manifest.json"
            if manifest_path.exists():
                try:
                    latest_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                except Exception:
                    latest_manifest = None
        latest_storyboard = (latest_run / "storyboard.png") if latest_run is not None else None
        if latest_storyboard is not None and latest_storyboard.exists():
            self._set_step5_storyboard(latest_storyboard)
        else:
            self._set_step5_storyboard(None)
        self._set_step5_acceptance_summary(latest_manifest)
        self.step5_moving_side_combo.blockSignals(True)
        self.step5_fixed_side_combo.blockSignals(True)
        self.step5_moving_group_combo.blockSignals(True)
        self.step5_fixed_group_combo.blockSignals(True)
        self.step5_moving_side_combo.setCurrentIndex(max(0, self.step5_moving_side_combo.findData("myelin")))
        self.step5_fixed_side_combo.setCurrentIndex(max(0, self.step5_fixed_side_combo.findData("nissl")))
        self.step5_moving_group_combo.setCurrentIndex(0)
        self.step5_fixed_group_combo.setCurrentIndex(0)
        self.step5_moving_side_combo.blockSignals(False)
        self.step5_fixed_side_combo.blockSignals(False)
        self.step5_moving_group_combo.blockSignals(False)
        self.step5_fixed_group_combo.blockSignals(False)
        self.step5_pair_label.setText(f"{index + 1}/{len(self.step5_pairs)} | {pair.display_label}")
        self.step5_info.setPlainText(
            "\n".join(
                [
                    f"pair_key: {pair.pair_key}",
                    f"animal_id: {pair.animal_id}",
                    f"myelin_label: {pair.myelin_item.label}",
                    f"nissl_label: {pair.nissl_item.label}",
                    f"registration_status: {review.get('registration_status', 'unreviewed')}",
                    f"multi_group_registration: {multi_group}",
                    f"flip_myelin_lr: {bool((review.get('flip_lr') or {}).get('myelin', False))}",
                    f"flip_nissl_lr: {bool((review.get('flip_lr') or {}).get('nissl', False))}",
                    f"myelin_group_flip_lr: {json.dumps((review.get('group_flip_lr') or {}).get('myelin', {}), ensure_ascii=True)}",
                    f"nissl_group_flip_lr: {json.dumps((review.get('group_flip_lr') or {}).get('nissl', {}), ensure_ascii=True)}",
                    f"myelin_component_groups: {json.dumps((review.get('component_groups') or {}).get('myelin', {}), ensure_ascii=True)}",
                    f"nissl_component_groups: {json.dumps((review.get('component_groups') or {}).get('nissl', {}), ensure_ascii=True)}",
                    f"myelin_registration_mask: {reg_files.get('myelin', 'missing')}",
                    f"nissl_registration_mask: {reg_files.get('nissl', 'missing')}",
                    f"latest_run: {latest_run if latest_run is not None else 'none'}",
                    f"approved_run: {approved.get('run_dir', 'none')}",
                    f"approved_stage: {approved.get('approved_stage', 'none')}",
                    f"accepted_path: {' -> '.join((latest_manifest or {}).get('accepted_stage_path') or ['input'])}",
                    f"rejected_stages: {', '.join(self._step5_rejected_stages_from_manifest(latest_manifest or {})) or 'none'}",
                    "",
                    "Registration prep notes:",
                    "- only usable pairs are shown here",
                    "- if multi_group_registration is true, run registration for 1<->1 and 2<->2 separately",
                    "- choose moving/fixed side and group, then run ANTs rigid + affine + SyN",
                    "- if rigid or affine is rejected, the next stage starts from the current best accepted state",
                    "- storyboard updates after each stage and includes a displacement heatmap panel",
                ]
            )
        )

    def run_step5_registration(self) -> None:
        if self.step5_run_thread is not None:
            self.step5_info.append("Step 5 is already running a registration.")
            return
        pair = self._current_step5_pair()
        if pair is None:
            self.step5_info.append("No usable pair selected.")
            return
        if self.step4_myelin_root is None or self.step4_nissl_root is None:
            QMessageBox.warning(self, "Step 5 Registration", "Myelin and Nissl roots must both be set.")
            return
        ants_bin = find_ants_bin()
        if ants_bin is None:
            QMessageBox.warning(self, "Step 5 Registration", "Could not find a local ANTs installation.")
            return
        runs_root = self._step5_runs_root()
        if runs_root is None:
            QMessageBox.warning(self, "Step 5 Registration", "Registration runs root is not available.")
            return
        review = self._step4_pair_review(pair)
        moving_side = str(self.step5_moving_side_combo.currentData() or "myelin")
        fixed_side = str(self.step5_fixed_side_combo.currentData() or "nissl")
        moving_group = str(self.step5_moving_group_combo.currentData() or "all")
        fixed_group = str(self.step5_fixed_group_combo.currentData() or "all")
        if moving_side == fixed_side:
            QMessageBox.warning(self, "Step 5 Registration", "Moving side and fixed side must be different.")
            return
        cfg = PairRegistrationConfig(
            pair_key=pair.pair_key,
            moving_side=moving_side,
            fixed_side=fixed_side,
            moving_group=moving_group,
            fixed_group=fixed_group,
            review=review,
            common_root=runs_root.parent,
            myelin_root=self.step4_myelin_root,
            nissl_root=self.step4_nissl_root,
            ants_bin=ants_bin,
            runs_root=runs_root,
            target_um_per_px=float(self.step5_target_um_per_px_spin.value()),
            working_long_edge=int(self.step5_working_long_edge_combo.currentData() or 1024),
            pre_blur_sigma=float(self.step5_blur_sigma_spin.value()),
        )
        self.step5_run_button.setEnabled(False)
        self.step5_progress_bar.setValue(0)
        self.step5_progress_label.setText("Step 5 progress: preparing registration run ...")
        self.step5_run_thread = QThread(self)
        self.step5_run_worker = PairRegistrationWorker(cfg)
        self.step5_run_worker.moveToThread(self.step5_run_thread)
        self.step5_run_thread.started.connect(self.step5_run_worker.run)
        self.step5_run_worker.stage_update.connect(self.on_step5_registration_stage_update)
        self.step5_run_worker.finished.connect(self.on_step5_registration_finished)
        self.step5_run_worker.failed.connect(self.on_step5_registration_failed)
        self.step5_run_worker.finished.connect(self.step5_run_thread.quit)
        self.step5_run_worker.failed.connect(self.step5_run_thread.quit)
        self.step5_run_thread.finished.connect(self.step5_run_worker.deleteLater)
        self.step5_run_thread.finished.connect(self.step5_run_thread.deleteLater)
        self.step5_run_thread.start()

    def on_step5_registration_stage_update(self, payload: object) -> None:
        data = dict(payload) if isinstance(payload, dict) else {}
        stage = str(data.get("stage") or "unknown")
        percent = max(0, min(100, int(round(float(data.get("progress_percent") or 0)))))
        message = str(data.get("message") or stage)
        storyboard_path = data.get("storyboard_path")
        run_dir = data.get("run_dir")
        self.step5_progress_bar.setValue(percent)
        self.step5_progress_label.setText(f"Step 5 progress: {stage} | {percent}% | {message}")
        if storyboard_path:
            self._set_step5_storyboard(Path(str(storyboard_path)))
        if run_dir:
            self.step5_info.append(f"{stage}: {run_dir}")
            manifest_path = Path(str(run_dir)) / "run_manifest.json"
            if manifest_path.exists():
                try:
                    self._set_step5_acceptance_summary(json.loads(manifest_path.read_text(encoding="utf-8")))
                except Exception:
                    pass

    def on_step5_registration_finished(self, summary: object) -> None:
        data = dict(summary) if isinstance(summary, dict) else {}
        run_dir = data.get("run_dir", "")
        storyboard_path = data.get("storyboard_path", "")
        if storyboard_path:
            self._set_step5_storyboard(Path(str(storyboard_path)))
        if run_dir:
            manifest_path = Path(str(run_dir)) / "run_manifest.json"
            if manifest_path.exists():
                try:
                    self._set_step5_acceptance_summary(json.loads(manifest_path.read_text(encoding="utf-8")))
                except Exception:
                    pass
        self.step5_progress_bar.setValue(100)
        self.step5_progress_label.setText("Step 5 progress: registration finished")
        self.step5_info.append(f"Registration finished. run_dir={run_dir}")
        self.step5_run_button.setEnabled(True)
        self.step5_run_worker = None
        self.step5_run_thread = None
        self._notify_completion("Step 5 registration finished")
        self.on_step5_pair_changed(self.current_step5_pair_index)

    def on_step5_registration_failed(self, message: str) -> None:
        self.step5_info.append(message)
        self.step5_progress_label.setText("Step 5 progress: registration failed")
        self.step5_acceptance_label.setText(
            "Accepted path: unavailable | Rejected: unavailable | Best accepted state runner: see failure log."
        )
        self.step5_acceptance_label.setStyleSheet(
            "padding:6px 10px; border-radius:6px; background:#fff1e0; color:#8a3d00; border:1px solid #f1a552; font-weight:700;"
        )
        self.step5_run_button.setEnabled(True)
        self.step5_run_worker = None
        self.step5_run_thread = None

    def _pair_common_root(self) -> Path | None:
        roots = [p for p in (self.step4_myelin_root, self.step4_nissl_root) if p is not None]
        if not roots:
            return None
        return Path(os.path.commonpath([str(p.resolve()) for p in roots]))

    def _relpath_from_common_root(self, path: Path | None) -> str | None:
        if path is None:
            return None
        common_root = self._pair_common_root()
        if common_root is None:
            return str(path)
        try:
            return str(path.resolve().relative_to(common_root.resolve()))
        except Exception:
            return str(path)

    def _latest_completed_stage_from_manifest(self, manifest: dict) -> str:
        requested = [str(x).strip().lower() for x in manifest.get("run_stages") or [] if str(x).strip()]
        stages = dict(manifest.get("stages") or {})
        for stage in reversed(requested):
            if stage in stages:
                return stage
        for stage in ("syn", "affine", "rigid"):
            if stage in stages:
                return stage
        return "rigid"

    @staticmethod
    def _step5_rejected_stages_from_manifest(manifest: dict) -> list[str]:
        stages = dict(manifest.get("stages") or {})
        rejected: list[str] = []
        for stage in [str(x).strip().lower() for x in manifest.get("run_stages") or [] if str(x).strip()]:
            gate = dict(stages.get(stage, {}).get("gate") or {})
            if gate and not bool(gate.get("accepted")):
                rejected.append(stage)
        return rejected

    @staticmethod
    def _best_step5_approved_stage_from_manifest(manifest: dict) -> str:
        best_stage = str(manifest.get("best_stage") or "").strip().lower()
        if best_stage and best_stage != "input":
            return best_stage
        accepted = [str(x).strip().lower() for x in manifest.get("accepted_stage_path") or [] if str(x).strip()]
        for stage in reversed(accepted):
            if stage != "input":
                return stage
        return ""

    def _approved_step5_stage_from_review(self, review: dict) -> str:
        approved = dict(review.get("approved_registration") or {})
        stage = str(approved.get("approved_stage") or "").strip().lower()
        if stage and stage != "input":
            return stage
        manifest_ref = approved.get("manifest_path")
        common_root = self._pair_common_root()
        if not manifest_ref or common_root is None:
            return ""
        manifest_path = Path(str(manifest_ref))
        if not manifest_path.is_absolute():
            manifest_path = common_root / manifest_path
        if not manifest_path.exists():
            return ""
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            return ""
        return self._best_step5_approved_stage_from_manifest(manifest)

    def _set_step5_acceptance_summary(self, manifest: dict | None) -> None:
        if not manifest:
            self.step5_acceptance_label.setText(
                "Accepted path: input | Rejected: none | Best accepted state runner: downstream stages start from the current best state."
            )
            self.step5_acceptance_label.setStyleSheet(
                "padding:6px 10px; border-radius:6px; background:#eef4fb; color:#24405c; border:1px solid #c9d8ea; font-weight:600;"
            )
            return
        stages = dict(manifest.get("stages") or {})
        has_gate_bookkeeping = bool(manifest.get("accepted_stage_path")) or any(
            dict(stage_info.get("gate") or {}) for stage_info in stages.values()
        )
        if not has_gate_bookkeeping:
            self.step5_acceptance_label.setText(
                "Accepted path: legacy run | Rejected: legacy run | Best accepted state runner: bookkeeping not available for this older run."
            )
            self.step5_acceptance_label.setStyleSheet(
                "padding:6px 10px; border-radius:6px; background:#f4f4f4; color:#555555; border:1px solid #d0d0d0; font-weight:600;"
            )
            return
        accepted = [str(x) for x in (manifest.get("accepted_stage_path") or ["input"]) if str(x).strip()]
        if not accepted:
            accepted = ["input"]
        rejected = self._step5_rejected_stages_from_manifest(manifest)
        accepted_text = " -> ".join(accepted)
        rejected_text = ", ".join(rejected) if rejected else "none"
        best_stage = str(manifest.get("best_stage") or accepted[-1] or "input")
        syn_requested = "syn" in {str(x).strip().lower() for x in manifest.get("run_stages") or []}
        linear_rejected = any(stage in rejected for stage in ("rigid", "affine"))
        linear_attempted = any(stage in stages for stage in ("rigid", "affine"))
        note = "Best accepted state runner: downstream stages start from the current best state."
        if syn_requested and linear_attempted and all(stage in rejected for stage in stages if stage in {"rigid", "affine"}):
            note = "Best accepted state runner: linear stages were rejected, so nonlinear starts from the current best accepted state."
        text = f"Accepted path: {accepted_text} | Rejected: {rejected_text} | Best: {best_stage}. {note}"
        if rejected:
            style = (
                "padding:6px 10px; border-radius:6px; background:#fff1e0; color:#8a3d00; "
                "border:1px solid #f1a552; font-weight:700;"
            )
        else:
            style = (
                "padding:6px 10px; border-radius:6px; background:#e8f6ec; color:#175c2b; "
                "border:1px solid #b8dfc2; font-weight:600;"
            )
        self.step5_acceptance_label.setText(text)
        self.step5_acceptance_label.setStyleSheet(style)

    def _set_rgb_image_label(self, label: QLabel, rgb: np.ndarray | None, empty_text: str) -> None:
        if rgb is None:
            label.setText(empty_text)
            label.setPixmap(QPixmap())
            return
        pixmap = QPixmap.fromImage(qimage_from_rgb_array(rgb.astype(np.uint8)))
        if pixmap.isNull():
            label.setText(empty_text)
            label.setPixmap(QPixmap())
            return
        label.setText("")
        label.setPixmap(pixmap)
        label.resize(pixmap.size())
        label.adjustSize()

    @staticmethod
    def _fit_rgb_preview(rgb: np.ndarray, *, max_long_edge: int) -> np.ndarray:
        arr = np.asarray(rgb, dtype=np.uint8)
        h, w = arr.shape[:2]
        long_edge = max(h, w)
        if long_edge <= max_long_edge or max_long_edge <= 0:
            return arr
        scale = float(max_long_edge) / float(long_edge)
        out_w = max(1, int(round(w * scale)))
        out_h = max(1, int(round(h * scale)))
        return cv2.resize(arr, (out_w, out_h), interpolation=cv2.INTER_AREA)

    def _set_step6_target_preview(self, rgb: np.ndarray | None, empty_text: str) -> None:
        if rgb is None:
            self.step6_target_viewer.set_section(
                np.full((32, 32, 3), 255, dtype=np.uint8),
                np.zeros((32, 32), dtype=np.uint8),
                np.zeros((32, 32), dtype=np.uint8),
            )
            return
        arr = np.asarray(rgb, dtype=np.uint8)
        self.step6_target_viewer.set_section(
            arr,
            np.zeros(arr.shape[:2], dtype=np.uint8),
            np.zeros(arr.shape[:2], dtype=np.uint8),
        )

    def _set_step6_target_view(
        self,
        rgb: np.ndarray | None,
        roi_mask: np.ndarray | None = None,
        *,
        preserve_view: bool = False,
    ) -> None:
        if rgb is None:
            self._set_step6_target_preview(None, "No mapped ROI yet")
            return
        arr = np.asarray(rgb, dtype=np.uint8)
        roi = np.asarray(roi_mask, dtype=np.uint8) if roi_mask is not None else np.zeros(arr.shape[:2], dtype=np.uint8)
        if roi.shape[:2] != arr.shape[:2]:
            roi = cv2.resize(roi, (arr.shape[1], arr.shape[0]), interpolation=cv2.INTER_NEAREST)
        self.step6_target_viewer.set_section(
            arr,
            roi,
            np.zeros(arr.shape[:2], dtype=np.uint8),
            preserve_view=preserve_view,
        )
        self.step6_target_viewer.set_active_layer("tissue")

    def on_step6_hires_nissl_toggled(self, checked: bool) -> None:
        self._sync_step6_source_render_mode()
        self._sync_step6_hires_nissl_controls()
        self._update_step6_direction_labels(self._current_step6_pair())
        if checked and self.step6_source_side == "nissl":
            self._schedule_step6_hires_source_patch_refresh(delay_ms=0)
        elif not checked:
            self._clear_step6_hires_source_patch()

    def on_step6_force_level0_toggled(self, checked: bool) -> None:
        self._sync_step6_hires_nissl_controls()
        self._update_step6_direction_labels(self._current_step6_pair())
        if self.step6_hires_nissl_check.isChecked() and self.step6_source_side == "nissl":
            self.step6_hires_last_request_key = None
            self._schedule_step6_hires_source_patch_refresh(delay_ms=0)

    def on_step6_source_view_changed(self, _info: dict[str, object]) -> None:
        if self.step6_hires_nissl_check.isChecked() and self.step6_source_side == "nissl":
            self._schedule_step6_hires_source_patch_refresh()

    def _sync_step6_source_render_mode(self) -> None:
        use_tiled_raw = bool(self.step6_source_side == "nissl" and self.step6_hires_nissl_check.isChecked())
        self.step6_nissl_editor.set_full_resolution_render_enabled(use_tiled_raw)

    def _sync_step6_hires_nissl_controls(self) -> None:
        active = self.step6_source_side == "nissl" and self.step6_current_context is not None
        self.step6_hires_nissl_check.setEnabled(active)
        self.step6_force_level0_check.setEnabled(active and self.step6_hires_nissl_check.isChecked())
        if active:
            self.step6_hires_nissl_check.setToolTip(
                "Display a dynamic NDPI patch for the current left-side Nissl viewport while keeping ROI storage in the existing canvas coordinates. Zoom in to inspect cytoarchitecture in detail."
            )
            self.step6_force_level0_check.setToolTip(
                "Always read the current Nissl viewport directly from NDPI level 0. This is slower, but shows the finest raw detail."
            )
            if not self.step6_hires_nissl_check.isChecked():
                self._set_step6_hires_status("Hi-res patch: off")
        else:
            self.step6_hires_nissl_check.setToolTip("Hi-res Nissl view is only available when the editable left side is Nissl.")
            self.step6_force_level0_check.setToolTip("Force Level0 is only available while editing Nissl with Hi-Res View enabled.")
            self._clear_step6_hires_source_patch()

    def _schedule_step6_hires_source_patch_refresh(self, *, delay_ms: int = 120) -> None:
        if not self.step6_hires_nissl_check.isChecked() or self.step6_source_side != "nissl":
            self._clear_step6_hires_source_patch()
            return
        self.step6_hires_view_timer.start(max(0, int(delay_ms)))

    def _set_step6_hires_status(self, text: str, *, warn: bool = False) -> None:
        self.step6_hires_status_label.setText(str(text))
        if warn:
            self.step6_hires_status_label.setStyleSheet("padding:4px 8px; color:#8a3d00; background:#fff1e0; border:1px solid #f1a552;")
        else:
            self.step6_hires_status_label.setStyleSheet("padding:4px 8px; color:#444444; background:#f5f5f5; border:1px solid #d8d8d8;")

    def _clear_step6_hires_source_patch(self) -> None:
        self.step6_hires_view_timer.stop()
        self.step6_hires_last_request_key = None
        self.step6_nissl_editor.set_detail_patch(None)
        if self.step6_source_side != "nissl":
            self._set_step6_hires_status("Hi-res patch: unavailable on Myelin source")
        elif not self.step6_hires_nissl_check.isChecked():
            self._set_step6_hires_status("Hi-res patch: off")
        else:
            self._set_step6_hires_status("Hi-res patch: waiting for viewport")

    def _close_step6_hires_slide_handle(self) -> None:
        if self.step6_hires_slide_handle is not None:
            try:
                self.step6_hires_slide_handle.close()
            except Exception:
                pass
        self.step6_hires_slide_handle = None
        self.step6_hires_loaded_slide = None
        self.step6_hires_slide_key = None

    def _ensure_step6_hires_slide(self, slide_path: Path, stain: str) -> LoadedSlide:
        stat = slide_path.stat()
        slide_key = (
            str(slide_path.resolve()),
            int(stat.st_size),
            int(getattr(stat, "st_mtime_ns", int(stat.st_mtime * 1e9))),
            str(stain).strip().lower(),
        )
        if self.step6_hires_loaded_slide is not None and self.step6_hires_slide_key == slide_key:
            return self.step6_hires_loaded_slide
        self._close_step6_hires_slide_handle()
        loaded = load_slide_header_only(slide_path, stain)
        self.step6_hires_loaded_slide = loaded
        self.step6_hires_slide_handle = open_slide_handle(loaded)
        self.step6_hires_slide_key = slide_key
        return loaded

    @staticmethod
    def _step6_group_flip_bboxes(labels_after_whole: np.ndarray, preprocess: dict[str, object]) -> list[tuple[int, int, int, int]]:
        component_groups = dict(preprocess.get("component_groups") or {})
        group_flip_lr = dict(preprocess.get("group_flip_lr") or {})
        if not component_groups or not group_flip_lr:
            return []
        labels_cc, rank_to_label = component_rank_map(np.asarray(labels_after_whole, dtype=np.uint8))
        bboxes: list[tuple[int, int, int, int]] = []
        for raw_group_id, raw_enabled in group_flip_lr.items():
            if not bool(raw_enabled):
                continue
            ranks = component_groups.get(str(raw_group_id))
            if not isinstance(ranks, list):
                continue
            for raw_rank in ranks:
                try:
                    rank = int(raw_rank)
                except Exception:
                    continue
                label_idx = rank_to_label.get(rank)
                if label_idx is None:
                    continue
                ys, xs = np.where(labels_cc == label_idx)
                if ys.size == 0 or xs.size == 0:
                    continue
                bboxes.append((int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1))
        return bboxes

    def _step6_hires_nissl_source_cache(self) -> dict[str, object] | None:
        context = self.step6_current_context
        pair = self._current_step6_pair()
        if context is None or pair is None or self.step6_source_side != "nissl":
            return None
        section_dir = pair.nissl_item.section_dir
        try:
            cache_key = str(section_dir.resolve())
        except Exception:
            cache_key = str(section_dir)
        cached = self.step6_hires_section_cache.get(cache_key)
        if cached is not None:
            return cached
        metadata_path = section_dir / "metadata.json"
        if not metadata_path.exists():
            return None
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except Exception:
            return None
        slide_path = self._preferred_existing_path(str(dict(metadata.get("source_slide") or {}).get("path") or ""))
        if slide_path is None or not slide_path.exists():
            return None
        labels = self._step6_registration_labels_for_side(context, pair, "nissl")
        if labels is None:
            return None
        preprocess = dict(context.nissl_preprocess or {})
        whole_flip = bool(preprocess.get("whole_flip_lr", False))
        labels_after_whole = labels[:, ::-1].copy() if whole_flip else labels.copy()
        group_flip_boxes = self._step6_group_flip_bboxes(labels_after_whole, preprocess)
        labels_transformed = labels_after_whole.copy()
        for x1, y1, x2, y2 in group_flip_boxes:
            labels_transformed[y1:y2, x1:x2] = labels_transformed[y1:y2, x1:x2][:, ::-1]
        try:
            labels_kept = keep_group(
                labels_transformed,
                dict(preprocess.get("component_groups") or {}),
                str(preprocess.get("group_choice") or "all"),
            )
            support_mask = (registration_support_mask(labels_kept, context.registration_mask_mode).astype(np.uint8) * 255)
        except Exception:
            return None
        mapping = dict(metadata.get("canvas_to_slide_level0") or {})
        origin_xy = dict(mapping.get("origin_level0_xy") or {})
        scale_xy = dict(mapping.get("scale_level0_per_canvas_px") or {})
        scale_x = float(scale_xy.get("x") or 0.0)
        scale_y = float(scale_xy.get("y") or 0.0)
        if scale_x <= 0.0 or scale_y <= 0.0:
            return None
        cache = {
            "section_dir": section_dir,
            "slide_path": slide_path,
            "metadata": metadata,
            "canvas_shape_hw": (int(labels.shape[0]), int(labels.shape[1])),
            "whole_flip": whole_flip,
            "group_flip_boxes": group_flip_boxes,
            "support_mask_u8": support_mask,
            "mirror_x_applied": bool(mapping.get("mirror_x_applied", False)),
            "origin_level0_xy": (int(origin_xy.get("x", 0)), int(origin_xy.get("y", 0))),
            "scale_level0_per_canvas_px": (scale_x, scale_y),
        }
        self.step6_hires_section_cache[cache_key] = cache
        return cache

    def _step6_choose_hires_level(
        self,
        loaded_slide: LoadedSlide,
        visible_full_rect_xywh: tuple[int, int, int, int],
        visible_widget_rect_xywh: tuple[int, int, int, int],
        cache: dict[str, object],
    ) -> int:
        if self.step6_force_level0_check.isChecked():
            return 0
        _, _, rect_w, rect_h = [int(v) for v in visible_full_rect_xywh]
        _, _, widget_w, widget_h = [int(v) for v in visible_widget_rect_xywh]
        scale_x, scale_y = [float(v) for v in cache["scale_level0_per_canvas_px"]]
        raw_w = max(1.0, scale_x * float(rect_w))
        raw_h = max(1.0, scale_y * float(rect_h))
        raw_area = raw_w * raw_h
        if max(raw_w, raw_h) <= 4096.0 and raw_area <= 12_000_000.0:
            return 0
        if max(raw_w, raw_h) <= 8192.0 and raw_area <= 24_000_000.0:
            level1 = min(1, max(0, loaded_slide.level_count - 1))
            if float(loaded_slide.level_downsamples[level1]) <= 2.0 + 1e-6:
                return level1
        desired_downsample = max(
            1.0,
            scale_x * (float(rect_w) / max(float(widget_w), 1.0)),
            scale_y * (float(rect_h) / max(float(widget_h), 1.0)),
        )
        if desired_downsample <= max(4.0, min(float(scale_x), float(scale_y))):
            return 0
        # The Step 6 canvas itself is already a downsampled crop view. If we only
        # target on-screen sampling, whole-image fit view often chooses a pyramid
        # level that is equal to or coarser than the current canvas, making the
        # hi-res toggle appear to do nothing. Force at least one level finer than
        # the exported canvas basis whenever possible.
        canvas_downsample = max(1.0, float(scale_x), float(scale_y))
        target_downsample = min(desired_downsample, max(1.0, canvas_downsample / 2.0))
        levels = tuple(float(v) for v in loaded_slide.level_downsamples)
        max_allowed = target_downsample * 1.02
        finer_or_equal = [idx for idx, ds in enumerate(levels) if ds <= max_allowed + 1e-6]
        if finer_or_equal:
            return int(max(finer_or_equal, key=lambda idx: levels[idx]))
        return int(min(range(len(levels)), key=lambda idx: abs(np.log(max(levels[idx], 1e-6)) - np.log(target_downsample))))

    @staticmethod
    def _step6_source_view_raw_canvas_rect(
        visible_full_rect_xywh: tuple[int, int, int, int],
        cache: dict[str, object],
    ) -> tuple[int, int, int, int]:
        x, y, w, h = [int(v) for v in visible_full_rect_xywh]
        canvas_w = int(cache["canvas_shape_hw"][1])
        if bool(cache["whole_flip"]):
            raw_x = max(0, min(canvas_w - 1, canvas_w - (x + w)))
        else:
            raw_x = max(0, min(canvas_w - 1, x))
        return raw_x, int(y), int(w), int(h)

    @staticmethod
    def _step6_canvas_rect_to_level0_bbox(
        canvas_rect_xywh: tuple[int, int, int, int],
        cache: dict[str, object],
    ) -> tuple[int, int, int, int]:
        x, y, w, h = [int(v) for v in canvas_rect_xywh]
        canvas_w = int(cache["canvas_shape_hw"][1])
        origin_x, origin_y = [int(v) for v in cache["origin_level0_xy"]]
        scale_x, scale_y = [float(v) for v in cache["scale_level0_per_canvas_px"]]
        if bool(cache["mirror_x_applied"]):
            sx1 = float(origin_x) + (float(canvas_w) - float(x + w)) * scale_x
            sx2 = float(origin_x) + (float(canvas_w) - float(x)) * scale_x
        else:
            sx1 = float(origin_x) + float(x) * scale_x
            sx2 = float(origin_x) + float(x + w) * scale_x
        sy1 = float(origin_y) + float(y) * scale_y
        sy2 = float(origin_y) + float(y + h) * scale_y
        x0 = int(np.floor(min(sx1, sx2)))
        y0 = int(np.floor(min(sy1, sy2)))
        x1 = int(np.ceil(max(sx1, sx2)))
        y1 = int(np.ceil(max(sy1, sy2)))
        return x0, y0, max(1, int(x1 - x0)), max(1, int(y1 - y0))

    @staticmethod
    def _step6_apply_group_flips_to_patch(
        patch_rgb: np.ndarray,
        visible_full_rect_xywh: tuple[int, int, int, int],
        group_flip_boxes: list[tuple[int, int, int, int]],
    ) -> np.ndarray:
        out = np.asarray(patch_rgb, dtype=np.uint8).copy()
        if out.size == 0 or not group_flip_boxes:
            return out
        x0, y0, w, h = [int(v) for v in visible_full_rect_xywh]
        patch_h, patch_w = out.shape[:2]
        scale_x = float(patch_w) / max(float(w), 1.0)
        scale_y = float(patch_h) / max(float(h), 1.0)
        for bx1, by1, bx2, by2 in group_flip_boxes:
            ix1 = max(x0, int(bx1))
            iy1 = max(y0, int(by1))
            ix2 = min(x0 + w, int(bx2))
            iy2 = min(y0 + h, int(by2))
            if ix2 <= ix1 or iy2 <= iy1:
                continue
            px1 = max(0, min(patch_w - 1, int(round((ix1 - x0) * scale_x))))
            py1 = max(0, min(patch_h - 1, int(round((iy1 - y0) * scale_y))))
            px2 = max(px1 + 1, min(patch_w, int(round((ix2 - x0) * scale_x))))
            py2 = max(py1 + 1, min(patch_h, int(round((iy2 - y0) * scale_y))))
            out[py1:py2, px1:px2] = out[py1:py2, px1:px2][:, ::-1, :]
        return out

    @staticmethod
    def _step6_crop_mask_to_patch(mask_u8: np.ndarray, visible_full_rect_xywh: tuple[int, int, int, int], patch_shape_hw: tuple[int, int]) -> np.ndarray:
        x, y, w, h = [int(v) for v in visible_full_rect_xywh]
        crop = np.asarray(mask_u8, dtype=np.uint8)[y : y + h, x : x + w]
        out_h, out_w = [int(v) for v in patch_shape_hw]
        if crop.shape[:2] != (out_h, out_w):
            crop = cv2.resize(crop, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
        return crop

    def _build_step6_hires_source_patch(
        self,
        slide_patch_rgb: np.ndarray,
        visible_full_rect_xywh: tuple[int, int, int, int],
        cache: dict[str, object],
        context,
        pair: WorkspacePair,
    ) -> np.ndarray:
        patch = np.asarray(slide_patch_rgb, dtype=np.uint8).copy()
        if patch.size == 0:
            return patch
        if bool(cache["mirror_x_applied"]):
            patch = patch[:, ::-1, :]
        if bool(cache["whole_flip"]):
            patch = patch[:, ::-1, :]
        patch = self._step6_apply_group_flips_to_patch(
            patch,
            visible_full_rect_xywh,
            list(cache["group_flip_boxes"]),
        )
        overlay_masks = self._step6_confocal_overlay_masks_for_side(
            context,
            pair,
            "nissl",
            shape_hw=tuple(cache["canvas_shape_hw"]),
        )
        if overlay_masks is not None:
            support_full, edge_full = overlay_masks
            support_crop = self._step6_crop_mask_to_patch(support_full, visible_full_rect_xywh, patch.shape[:2])
            edge_crop = self._step6_crop_mask_to_patch(edge_full, visible_full_rect_xywh, patch.shape[:2])
            patch = self._step6_blend_confocal_overlay(patch, support_crop, edge_crop)
        return patch

    def _refresh_step6_hires_source_patch(self) -> None:
        if not self.step6_hires_nissl_check.isChecked() or self.step6_source_side != "nissl":
            self._clear_step6_hires_source_patch()
            return
        context = self.step6_current_context
        pair = self._current_step6_pair()
        if context is None or pair is None:
            self._clear_step6_hires_source_patch()
            return
        viewport = self.step6_nissl_editor.current_viewport_info()
        cache = self._step6_hires_nissl_source_cache()
        if viewport is None or cache is None:
            self._clear_step6_hires_source_patch()
            return
        visible_full_rect_xywh = tuple(int(v) for v in viewport.get("visible_full_rect_xywh") or ())
        visible_widget_rect_xywh = tuple(int(v) for v in viewport.get("visible_widget_rect_xywh") or ())
        if len(visible_full_rect_xywh) != 4 or len(visible_widget_rect_xywh) != 4:
            self._clear_step6_hires_source_patch()
            return
        loaded_slide = self._ensure_step6_hires_slide(Path(cache["slide_path"]), "nissl")
        level = self._step6_choose_hires_level(
            loaded_slide,
            visible_full_rect_xywh,
            visible_widget_rect_xywh,
            cache,
        )
        request_key = (
            pair.pair_key,
            context.nissl_label,
            visible_full_rect_xywh,
            visible_widget_rect_xywh,
            int(level),
            bool(self.step6_force_level0_check.isChecked()),
            str(self.step6_confocal_handoff_path or ""),
        )
        if request_key == self.step6_hires_last_request_key:
            return
        raw_canvas_rect = self._step6_source_view_raw_canvas_rect(visible_full_rect_xywh, cache)
        bbox_level0_xywh = self._step6_canvas_rect_to_level0_bbox(raw_canvas_rect, cache)
        try:
            slide_patch = extract_level0_bbox_rgb(
                loaded_slide,
                bbox_level0_xywh,
                level=level,
                slide_handle=self.step6_hires_slide_handle,
            )
        except Exception:
            self._clear_step6_hires_source_patch()
            self._set_step6_hires_status("Hi-res patch: failed to read NDPI patch", warn=True)
            return
        detail_patch = self._build_step6_hires_source_patch(
            slide_patch,
            visible_full_rect_xywh,
            cache,
            context,
            pair,
        )
        self.step6_nissl_editor.set_detail_patch(detail_patch, full_rect_xywh=visible_full_rect_xywh)
        self.step6_hires_last_request_key = request_key
        patch_h, patch_w = [int(v) for v in detail_patch.shape[:2]]
        _, _, widget_w, widget_h = [int(v) for v in visible_widget_rect_xywh]
        level_text = f"L{int(level)}"
        if self.step6_force_level0_check.isChecked():
            level_text += " forced"
        self._set_step6_hires_status(
            f"Hi-res patch: {level_text} | raw patch {patch_w}x{patch_h} px -> view {widget_w}x{widget_h} px"
        )

    @staticmethod
    def _binary_roi_from_masks(tissue: np.ndarray, artifact: np.ndarray) -> np.ndarray:
        return np.where(tissue > 0, 255, 0).astype(np.uint8)

    @staticmethod
    def _step6_diff_overlay_rgba(current_roi: np.ndarray, reference_roi: np.ndarray) -> np.ndarray:
        current = np.asarray(current_roi, dtype=np.uint8) > 0
        reference = np.asarray(reference_roi, dtype=np.uint8) > 0
        added = current & ~reference
        removed = reference & ~current
        overlay = np.zeros(current.shape + (4,), dtype=np.uint8)
        overlay[added] = np.array([160, 255, 80, 175], dtype=np.uint8)
        overlay[removed] = np.array([255, 80, 200, 170], dtype=np.uint8)
        return overlay

    @staticmethod
    def _step6_apply_roi_preview(rgb: np.ndarray, current_roi: np.ndarray, reference_roi: np.ndarray | None = None) -> np.ndarray:
        overlay = rgb.copy()
        current = np.asarray(current_roi, dtype=np.uint8) > 0
        if np.any(current):
            tint = np.array([80, 220, 255], dtype=np.float32)
            overlay[current] = np.clip(0.45 * overlay[current].astype(np.float32) + 0.55 * tint, 0, 255).astype(np.uint8)
        if reference_roi is not None:
            reference = np.asarray(reference_roi, dtype=np.uint8) > 0
            added = current & ~reference
            removed = reference & ~current
            if np.any(added):
                tint_add = np.array([160, 255, 80], dtype=np.float32)
                overlay[added] = np.clip(0.25 * overlay[added].astype(np.float32) + 0.75 * tint_add, 0, 255).astype(np.uint8)
            if np.any(removed):
                tint_remove = np.array([255, 80, 200], dtype=np.float32)
                overlay[removed] = np.clip(0.25 * overlay[removed].astype(np.float32) + 0.75 * tint_remove, 0, 255).astype(np.uint8)
        return overlay

    def _set_step6_stale_state(self, stale: bool, *, reason: str | None = None) -> None:
        self.step6_preview_stale = bool(stale)
        if stale:
            text = "Mapped ROI preview: STALE"
            if reason:
                text = f"{text} | {reason}"
            self.step6_mapping_status_label.setText(text)
            self.step6_mapping_status_label.setStyleSheet(
                "padding:6px 10px; border-radius:6px; background:#fff1e0; color:#8a3d00; font-weight:700; border:1px solid #f1a552;"
            )
            self.step6_update_button.setStyleSheet(
                "background:#f6b04d; color:#1e1e1e; font-weight:700; border:1px solid #cd8a2e; padding:4px 10px;"
            )
        else:
            self.step6_mapping_status_label.setText("Mapped ROI preview: fresh")
            self.step6_mapping_status_label.setStyleSheet(
                "padding:6px 10px; border-radius:6px; background:#e8f6ec; color:#175c2b; font-weight:600;"
            )
            self.step6_update_button.setStyleSheet("")

    def _current_step6_roi_highres(self) -> np.ndarray:
        tissue, artifact = self.step6_nissl_editor.current_masks()
        return self._binary_roi_from_masks(tissue, artifact)

    def _sync_step6_tool_buttons(self, tool: str | None = None) -> None:
        mode = str(tool or self.step6_nissl_editor.current_tool_mode()).strip().lower()
        button_map = {
            "grab": self.step6_grab_button,
            "brush": self.step6_brush_button,
            "eraser": self.step6_eraser_button,
            "polygon": self.step6_polygon_button,
        }
        for key, button in button_map.items():
            button.blockSignals(True)
            button.setChecked(mode == key)
            if mode == key:
                button.setStyleSheet("background:#d9edf7; border:1px solid #8fbad1; font-weight:600;")
            else:
                button.setStyleSheet("")
            button.blockSignals(False)

    def on_step6_source_tool_changed(self, tool: str) -> None:
        self._sync_step6_tool_buttons(tool)

    def set_step6_source_tool(self, tool: str) -> None:
        normalized = str(tool).strip().lower()
        if normalized == "brush":
            self.step6_nissl_editor.activate_brush_tool(add=True)
        elif normalized == "eraser":
            self.step6_nissl_editor.activate_brush_tool(add=False)
        elif normalized == "polygon":
            self.step6_nissl_editor.activate_polygon_tool()
        else:
            self.step6_nissl_editor.activate_hand_tool()
        self._sync_step6_tool_buttons(normalized)
        self.step6_nissl_editor.setFocus(Qt.FocusReason.OtherFocusReason)

    def step6_apply_polygon_fill(self) -> None:
        if self.step6_nissl_editor.apply_polygon_fill():
            self._sync_step6_tool_buttons()
            self.step6_nissl_editor.setFocus(Qt.FocusReason.OtherFocusReason)

    def step6_clear_polygon(self) -> None:
        self.step6_nissl_editor.clear_polygon()
        self.step6_nissl_editor.setFocus(Qt.FocusReason.OtherFocusReason)

    def zoom_step6_source_view(self, factor: float) -> None:
        self.step6_nissl_editor.zoom_by(float(factor))
        self.step6_nissl_editor.setFocus(Qt.FocusReason.OtherFocusReason)

    def reset_step6_source_view(self) -> None:
        self.step6_nissl_editor.reset_view()
        self.step6_nissl_editor.setFocus(Qt.FocusReason.OtherFocusReason)

    def zoom_step6_target_view(self, factor: float) -> None:
        self.step6_target_viewer.zoom_by(float(factor))
        self.step6_target_viewer.setFocus(Qt.FocusReason.OtherFocusReason)

    def reset_step6_target_view(self) -> None:
        self.step6_target_viewer.reset_view()
        self.step6_target_viewer.setFocus(Qt.FocusReason.OtherFocusReason)

    def _set_step6_confocal_status(self, text: str, *, warn: bool = False) -> None:
        self.step6_confocal_status_label.setText(text)
        if warn:
            self.step6_confocal_status_label.setStyleSheet(
                "padding:6px 8px; background:#fff1e0; color:#8a3d00; border:1px solid #f1a552; font-weight:600;"
            )
        else:
            self.step6_confocal_status_label.setStyleSheet(
                "padding:6px 8px; background:#f5f5f5; border:1px solid #d0d0d0;"
            )

    def _sync_step6_confocal_overlay_toggle_button(self) -> None:
        has_handoff = isinstance(self.step6_confocal_handoff, dict)
        self.step6_toggle_confocal_overlay_button.setEnabled(has_handoff)
        self.step6_toggle_confocal_overlay_button.setText(
            "Hide Confocal Grid" if self.step6_confocal_overlay_visible else "Show Confocal Grid"
        )

    def _refresh_step6_context_views(self, *, preserve_source_view: bool = True, preserve_target_view: bool = True) -> None:
        context = self.step6_current_context
        pair = self._current_step6_pair()
        if context is None or pair is None:
            return
        source_side = self.step6_source_side
        target_side = self._current_step6_target_side()
        state = current_step6_state(context, source_side=source_side)
        source_preview_rgb = self._step6_rgb_with_context_overlays(state["source_rgb"], context, pair, source_side)
        target_preview_rgb = self._step6_rgb_with_context_overlays(state["target_rgb"], context, pair, target_side)
        current_source_roi = self._current_step6_roi_highres()
        if current_source_roi.shape[:2] != source_preview_rgb.shape[:2]:
            current_source_roi = cv2.resize(
                np.asarray(current_source_roi, dtype=np.uint8),
                (source_preview_rgb.shape[1], source_preview_rgb.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
        target_roi = (
            np.asarray(self.step6_last_updated_target_roi_highres, dtype=np.uint8).copy()
            if self.step6_last_updated_target_roi_highres is not None
            else np.asarray(state["target_roi"], dtype=np.uint8).copy()
        )
        self._sync_step6_source_render_mode()
        self.step6_nissl_editor.set_section(
            source_preview_rgb,
            np.asarray(current_source_roi, dtype=np.uint8),
            np.zeros(source_preview_rgb.shape[:2], dtype=np.uint8),
            preserve_view=preserve_source_view,
        )
        self.step6_nissl_editor.set_active_layer("tissue")
        self.step6_nissl_editor.set_aux_overlay_rgba(None)
        self._refresh_step6_source_batch_overlay()
        self._set_step6_target_view(target_preview_rgb, target_roi, preserve_view=preserve_target_view)
        self.step6_hires_last_request_key = None
        if self.step6_hires_nissl_check.isChecked() and self.step6_source_side == "nissl":
            self._schedule_step6_hires_source_patch_refresh(delay_ms=0)
        else:
            self._clear_step6_hires_source_patch()

    def toggle_step6_confocal_overlay_visibility(self) -> None:
        self.step6_confocal_overlay_visible = not self.step6_confocal_overlay_visible
        self._sync_step6_confocal_overlay_toggle_button()
        self._refresh_step6_context_views(preserve_source_view=True, preserve_target_view=True)
        pair = self._current_step6_pair()
        if pair is not None and isinstance(self.step6_confocal_handoff, dict):
            origin = self.step6_confocal_handoff_origin or "manual"
            origin_text = "manually loaded" if origin == "manual" else "auto-loaded latest Step 7 handoff"
            path_text = f"\n{self.step6_confocal_handoff_path}" if self.step6_confocal_handoff_path is not None else ""
            visibility_text = (
                "Currently hidden in Step 6."
                if not self.step6_confocal_overlay_visible
                else "Both source and target sides show accepted+frozen confocal FOV/grid."
            )
            self._set_step6_confocal_status(
                f"Confocal FOV overlay active for {pair.myelin_item.label} ({origin_text}). {visibility_text}{path_text}",
                warn=False,
            )

    @staticmethod
    def _step6_blend_confocal_overlay(base_rgb: np.ndarray, support_mask_u8: np.ndarray, edge_mask_u8: np.ndarray) -> np.ndarray:
        out = np.asarray(base_rgb, dtype=np.uint8).copy()
        support = np.asarray(support_mask_u8, dtype=np.uint8) > 0
        edges = np.asarray(edge_mask_u8, dtype=np.uint8) > 0
        if np.any(support):
            tint = np.array([250, 220, 90], dtype=np.float32)
            out[support] = np.clip(0.78 * out[support].astype(np.float32) + 0.22 * tint, 0, 255).astype(np.uint8)
        if np.any(edges):
            out[edges] = np.array([220, 95, 20], dtype=np.uint8)
        return out

    def _current_step6_ants_bin(self, context) -> Path | None:
        if context is None:
            return None
        if context.registration_backend != "ants":
            return Path()
        ants_bin = find_ants_bin()
        if ants_bin is None:
            return None
        return ants_bin

    def _step6_registration_mask_path(self, context, side: str) -> Path | None:
        if context is None:
            return None
        preprocess = context.nissl_preprocess if str(side).strip().lower() == "nissl" else context.myelin_preprocess
        raw = str(dict(preprocess or {}).get("mask_path") or "").strip()
        return self._preferred_existing_path(raw)

    def _step6_registration_labels_for_side(self, context, pair: WorkspacePair, side: str) -> np.ndarray | None:
        section_dir = pair.nissl_item.section_dir if str(side).strip().lower() == "nissl" else pair.myelin_item.section_dir
        mask_path = self._step6_registration_mask_path(context, side)
        candidate = mask_path if mask_path is not None else (section_dir / "mask_labels.png")
        labels = cv2.imread(str(candidate), cv2.IMREAD_UNCHANGED)
        if labels is None:
            return None
        if labels.ndim == 3:
            labels = labels[..., 0]
        return np.asarray(labels, dtype=np.uint8)

    def _step6_myelin_support_bbox_canvas_xywh(self, pair: WorkspacePair) -> tuple[int, int, int, int] | None:
        if isinstance(self.step6_confocal_handoff, dict):
            scene_space = self.step6_confocal_handoff.get("scene_space")
            if isinstance(scene_space, dict):
                raw = scene_space.get("fixed_support_bbox_canvas_xywh")
                if isinstance(raw, (list, tuple)) and len(raw) >= 4:
                    try:
                        return tuple(int(v) for v in raw[:4])
                    except Exception:
                        pass
        label = str(pair.myelin_item.label or "").strip()
        if not label:
            return None
        cached = self.step6_confocal_support_bbox_cache.get(label)
        if cached is not None:
            return tuple(int(v) for v in cached)
        labels_path = pair.myelin_item.section_dir / "mask_labels.png"
        if not labels_path.exists():
            return None
        labels = cv2.imread(str(labels_path), cv2.IMREAD_UNCHANGED)
        if labels is None:
            return None
        if labels.ndim == 3:
            labels = labels[..., 0]
        ys, xs = np.where(np.asarray(labels, dtype=np.uint8) > 0)
        if ys.size == 0 or xs.size == 0:
            bbox = (0, 0, int(labels.shape[1]), int(labels.shape[0]))
        else:
            x0 = int(xs.min())
            y0 = int(ys.min())
            x1 = int(xs.max()) + 1
            y1 = int(ys.max()) + 1
            bbox = (x0, y0, int(x1 - x0), int(y1 - y0))
        self.step6_confocal_support_bbox_cache[label] = bbox
        return bbox

    @staticmethod
    def _render_step6_confocal_overlay_masks(
        polygons_xy: list[np.ndarray],
        *,
        shape_hw: tuple[int, int],
        edge_thickness_px: int = 4,
    ) -> tuple[np.ndarray, np.ndarray]:
        out_h = max(1, int(shape_hw[0]))
        out_w = max(1, int(shape_hw[1]))
        support_mask = np.zeros((out_h, out_w), dtype=np.uint8)
        edge_mask = np.zeros((out_h, out_w), dtype=np.uint8)
        for poly in polygons_xy:
            pts = np.asarray(poly, dtype=np.float32).reshape((-1, 2))
            if pts.shape[0] < 3:
                continue
            pts_i32 = np.round(pts).astype(np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(support_mask, [pts_i32], 255, lineType=cv2.LINE_8)
            cv2.polylines(edge_mask, [pts_i32], True, 255, max(1, int(edge_thickness_px)), cv2.LINE_AA)
        return support_mask, edge_mask

    def _step6_confocal_overlay_masks_for_side(
        self,
        context,
        pair: WorkspacePair,
        side: str,
        *,
        shape_hw: tuple[int, int],
    ) -> tuple[np.ndarray, np.ndarray] | None:
        if not self.step6_confocal_overlay_visible:
            return None
        if context is None or not isinstance(self.step6_confocal_handoff, dict):
            return None
        cache_key = "|".join(
            [
                str(pair.pair_key),
                str(side).strip().lower(),
                f"{int(shape_hw[0])}x{int(shape_hw[1])}",
                str(self.step6_confocal_handoff_path or ""),
                str(context.registration_backend),
            ]
        )
        cached = self.step6_confocal_overlay_masks_cache.get(cache_key)
        if cached is not None:
            return cached
        handoff = dict(self.step6_confocal_handoff)
        handoff_label = str(handoff.get("myelin_label") or "").strip()
        if handoff_label and handoff_label != pair.myelin_item.label:
            return None
        scene_space = handoff.get("scene_space") if isinstance(handoff.get("scene_space"), dict) else {}
        if context.registration_backend == "mask_shape":
            myelin_labels = self._step6_registration_labels_for_side(context, pair, "myelin")
            if myelin_labels is None:
                return None
            mapped_polygons: list[np.ndarray] = []
            preview_shape_raw = scene_space.get("fixed_preview_shape_hw") or handoff.get("fixed_preview_shape_hw") or []
            preview_shape_hw = None
            if isinstance(preview_shape_raw, (list, tuple)) and len(preview_shape_raw) >= 2:
                preview_shape_hw = (int(preview_shape_raw[0]), int(preview_shape_raw[1]))
            support_bbox_canvas_xywh = self._step6_myelin_support_bbox_canvas_xywh(pair)
            for row in list(handoff.get("tile_records") or []):
                if not isinstance(row, dict):
                    continue
                state = str(row.get("tile_state") or "").strip().lower()
                if state not in {"accepted", "frozen"}:
                    continue
                full_crop_poly = np.asarray(row.get("final_full_crop_polygon_xy") or [], dtype=np.float32)
                mapped = np.zeros((0, 2), dtype=np.float32)
                if full_crop_poly.shape == (4, 2):
                    mapped = map_step7_full_crop_polygon_to_step6_side(
                        context,
                        full_crop_poly,
                        output_side=side,
                        myelin_labels=np.asarray(myelin_labels, dtype=np.uint8),
                    )
                elif preview_shape_hw is not None and support_bbox_canvas_xywh is not None:
                    scene_poly = np.asarray(row.get("final_scene_polygon_xy") or [], dtype=np.float32)
                    if scene_poly.shape == (4, 2):
                        mapped = map_step7_scene_polygon_to_step6_side(
                            context,
                            scene_poly,
                            step7_preview_shape_hw=preview_shape_hw,
                            step7_support_bbox_canvas_xywh=support_bbox_canvas_xywh,
                            output_side=side,
                            myelin_labels=np.asarray(myelin_labels, dtype=np.uint8),
                        )
                if mapped.shape == (4, 2):
                    mapped_polygons.append(mapped)
            if mapped_polygons:
                rendered = self._render_step6_confocal_overlay_masks(mapped_polygons, shape_hw=shape_hw, edge_thickness_px=4)
                self.step6_confocal_overlay_masks_cache[cache_key] = rendered
                return rendered
            return None
        preview_shape_raw = scene_space.get("fixed_preview_shape_hw") or handoff.get("fixed_preview_shape_hw") or []
        if not isinstance(preview_shape_raw, (list, tuple)) or len(preview_shape_raw) < 2:
            return None
        preview_shape_hw = (int(preview_shape_raw[0]), int(preview_shape_raw[1]))
        support_bbox_canvas_xywh = self._step6_myelin_support_bbox_canvas_xywh(pair)
        if support_bbox_canvas_xywh is None:
            return None
        scene_masks = build_step7_scene_fov_masks(handoff, outline_thickness_px=8)
        ants_bin = self._current_step6_ants_bin(context)
        if ants_bin is None:
            return None
        support_mask = map_step7_scene_mask_to_step6_side(
            context,
            np.asarray(scene_masks["scene_support_mask_u8"], dtype=np.uint8),
            step7_preview_shape_hw=preview_shape_hw,
            step7_support_bbox_canvas_xywh=support_bbox_canvas_xywh,
            output_side=side,
            ants_bin=ants_bin,
        )
        edge_mask = map_step7_scene_mask_to_step6_side(
            context,
            np.asarray(scene_masks["scene_grid_edges_u8"], dtype=np.uint8),
            step7_preview_shape_hw=preview_shape_hw,
            step7_support_bbox_canvas_xywh=support_bbox_canvas_xywh,
            output_side=side,
            ants_bin=ants_bin,
        )
        rendered = (support_mask, edge_mask)
        self.step6_confocal_overlay_masks_cache[cache_key] = rendered
        return rendered

    def _step6_rgb_with_context_overlays(self, base_rgb: np.ndarray, context, pair: WorkspacePair | None, side: str) -> np.ndarray:
        rgb = np.asarray(base_rgb, dtype=np.uint8)
        if context is None or pair is None:
            return rgb
        overlay_masks = self._step6_confocal_overlay_masks_for_side(context, pair, side, shape_hw=rgb.shape[:2])
        if overlay_masks is None:
            return rgb
        support_mask, edge_mask = overlay_masks
        return self._step6_blend_confocal_overlay(rgb, support_mask, edge_mask)

    def _refresh_step6_source_batch_overlay(self) -> None:
        if self.step6_current_context is None:
            self.step6_nissl_editor.set_aux_overlay_rgba(None)
            return
        if self.step6_last_updated_source_roi_highres is None:
            self.step6_nissl_editor.set_aux_overlay_rgba(None)
            return
        current_roi = self._current_step6_roi_highres()
        diff_overlay = self._step6_diff_overlay_rgba(current_roi, self.step6_last_updated_source_roi_highres)
        if np.any(diff_overlay[..., 3] > 0):
            self.step6_nissl_editor.set_aux_overlay_rgba(diff_overlay)
        else:
            self.step6_nissl_editor.set_aux_overlay_rgba(None)

    def approve_current_step5_run(self) -> None:
        pair = self._current_step5_pair()
        runs_root = self._step5_runs_root()
        registry_path = self._step4_registry_path()
        if pair is None or runs_root is None or registry_path is None:
            QMessageBox.warning(self, "Approve Registration Run", "A usable pair and registration roots are required.")
            return
        latest_run = latest_registration_run_dir(runs_root, pair.pair_key)
        if latest_run is None:
            QMessageBox.warning(self, "Approve Registration Run", "No registration run exists for the current pair.")
            return
        manifest_path = latest_run / "run_manifest.json"
        if not manifest_path.exists():
            QMessageBox.warning(self, "Approve Registration Run", f"Missing run manifest:\n{manifest_path}")
            return
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        approved_stage = self._best_step5_approved_stage_from_manifest(manifest)
        if not approved_stage:
            QMessageBox.warning(
                self,
                "Approve Registration Run",
                "Latest run did not produce any non-input stage that passed gate. Step 6 requires an accepted registration stage.",
            )
            return
        review = self._step4_pair_review(pair)
        nissl_group = (
            str(manifest.get("fixed_group") or "all")
            if str(manifest.get("fixed_side") or "").strip().lower() == "nissl"
            else str(manifest.get("moving_group") or "all")
        )
        review["approved_registration"] = {
            "run_dir": self._relpath_from_common_root(latest_run),
            "manifest_path": self._relpath_from_common_root(manifest_path),
            "approved_stage": approved_stage,
            "moving_side": str(manifest.get("moving_side") or ""),
            "fixed_side": str(manifest.get("fixed_side") or ""),
            "moving_group": str(manifest.get("moving_group") or "all"),
            "fixed_group": str(manifest.get("fixed_group") or "all"),
            "group_tag": nissl_group,
            "approved_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        self.step4_pair_registry[pair.pair_key] = review
        save_pair_registry(registry_path, self.step4_pair_registry)
        self.step5_info.append(f"Approved run for downstream ROI mapping: {latest_run}")
        self.on_step5_pair_changed(self.current_step5_pair_index)

    def _step6_roi_root(self) -> Path | None:
        self.step6_roi_root = default_pair_roi_root(self.step4_myelin_root, self.step4_nissl_root)
        return self.step6_roi_root

    def _current_step6_pair(self) -> WorkspacePair | None:
        if 0 <= self.current_step6_pair_index < len(self.step6_pairs):
            return self.step6_pairs[self.current_step6_pair_index]
        return None

    @staticmethod
    def _step6_side_display_name(side: str) -> str:
        return "Nissl" if str(side).strip().lower() == "nissl" else "Myelin"

    def _current_step6_target_side(self) -> str:
        return "myelin" if self.step6_source_side == "nissl" else "nissl"

    def _update_step6_direction_labels(self, pair: WorkspacePair | None = None) -> None:
        source_side = self.step6_source_side
        target_side = self._current_step6_target_side()
        source_name = self._step6_side_display_name(source_side)
        target_name = self._step6_side_display_name(target_side)
        source_suffix = ""
        if source_side == "nissl" and self.step6_hires_nissl_check.isChecked():
            source_suffix = " | Hi-Res View"
            if self.step6_force_level0_check.isChecked():
                source_suffix += " | Force L0"
        if pair is None:
            self.step6_nissl_title.setText(f"{source_name} ROI{source_suffix}")
        elif source_side == "nissl":
            self.step6_nissl_title.setText(f"{source_name} ROI | {pair.nissl_item.label}{source_suffix}")
        else:
            self.step6_nissl_title.setText(f"{source_name} ROI | {pair.myelin_item.label}")
        self.step6_target_title.setText(f"Mapped {target_name} ROI")

    def on_step6_direction_changed(self, index: int) -> None:
        side = str(self.step6_direction_combo.itemData(index) or "nissl")
        if side == self.step6_source_side:
            return
        self.step6_source_side = side
        pair = self._current_step6_pair()
        self._sync_step6_source_render_mode()
        self._update_step6_direction_labels(pair)
        self._sync_step6_hires_nissl_controls()
        if pair is not None:
            self.on_step6_pair_changed(self.current_step6_pair_index)

    def _set_step6_confocal_handoff(
        self,
        path: Path | None,
        handoff: dict[str, object] | None,
        *,
        origin: str,
    ) -> None:
        self.step6_confocal_handoff_path = None if path is None else Path(path)
        self.step6_confocal_handoff = None if handoff is None else dict(handoff)
        self.step6_confocal_handoff_origin = str(origin or "none").strip().lower()
        self.step6_confocal_overlay_masks_cache.clear()
        self.step6_hires_last_request_key = None
        self._sync_step6_confocal_overlay_toggle_button()

    def _step6_confocal_handoff_start_dir(self) -> str:
        if self.step6_confocal_handoff_path is not None:
            return str(self.step6_confocal_handoff_path.parent)
        if self.step7_last_export_dir is not None:
            return str(self.step7_last_export_dir)
        return str(self._default_step7_confocal_root())

    def _step6_auto_handoff_search_roots(self) -> list[Path]:
        roots: list[Path] = []
        if self.step7_last_export_dir is not None:
            roots.extend([self.step7_last_export_dir.parent, self.step7_last_export_dir.parent.parent])
        export_root = self._step7_export_root()
        if export_root is not None:
            roots.append(export_root)
        if self.step6_confocal_handoff_path is not None:
            roots.extend([self.step6_confocal_handoff_path.parent.parent, self.step6_confocal_handoff_path.parent.parent.parent])
        roots.append(self._default_step7_confocal_root())
        out: list[Path] = []
        seen: set[str] = set()
        for root in roots:
            try:
                resolved = str(Path(root).resolve())
            except Exception:
                resolved = str(Path(root))
            if not resolved or resolved in seen:
                continue
            seen.add(resolved)
            out.append(Path(root))
        return out

    def _step6_candidate_handoff_paths(self, myelin_label: str) -> list[Path]:
        label = str(myelin_label or "").strip()
        if not label:
            return []
        candidates: list[Path] = []
        seen: set[str] = set()
        patterns = [
            "step7_session_export_*/step8_handoff.json",
            f"{label}/step7_session_export_*/step8_handoff.json",
            f"{label}_*/step7_session_export_*/step8_handoff.json",
            f"*/{label}/step7_session_export_*/step8_handoff.json",
            f"*/{label}_*/step7_session_export_*/step8_handoff.json",
        ]
        for root in self._step6_auto_handoff_search_roots():
            if not root.exists():
                continue
            root_is_label_dir = root.name == label
            root_patterns = patterns if not root_is_label_dir else patterns[:1]
            for pattern in root_patterns:
                try:
                    found = list(root.glob(pattern))
                except Exception:
                    found = []
                for path in found:
                    if not path.is_file():
                        continue
                    key = str(path)
                    if key in seen:
                        continue
                    seen.add(key)
                    candidates.append(path)
        candidates.sort(key=lambda p: (p.stat().st_mtime if p.exists() else 0.0, str(p)), reverse=True)
        return candidates

    def _latest_step6_auto_handoff_for_label(self, myelin_label: str) -> tuple[Path, dict[str, object]] | None:
        label = str(myelin_label or "").strip()
        if not label:
            return None
        if self.step7_last_export_dir is not None:
            direct_path = self.step7_last_export_dir / "step8_handoff.json"
            if direct_path.exists():
                try:
                    direct_handoff = load_step7_handoff_payload(direct_path)
                    build_step7_scene_fov_masks(direct_handoff)
                    if str(direct_handoff.get("myelin_label") or "").strip() == label:
                        return direct_path, dict(direct_handoff)
                except Exception:
                    pass
        for path in self._step6_candidate_handoff_paths(label):
            try:
                handoff = load_step7_handoff_payload(path)
                build_step7_scene_fov_masks(handoff)
            except Exception:
                continue
            if str(handoff.get("myelin_label") or "").strip() != label:
                continue
            return path, dict(handoff)
        return None

    def _maybe_auto_load_step6_confocal_handoff(self, pair: WorkspacePair | None) -> bool:
        if pair is None:
            return False
        if self.step6_confocal_handoff_origin == "manual":
            return False
        if not self.step6_auto_step7_handoff_check.isChecked():
            if self.step6_confocal_handoff_origin == "auto":
                self._set_step6_confocal_handoff(None, None, origin="none")
            return False
        resolved = self._latest_step6_auto_handoff_for_label(pair.myelin_item.label)
        if resolved is None:
            if self.step6_confocal_handoff_origin == "auto":
                self._set_step6_confocal_handoff(None, None, origin="none")
            return False
        path, handoff = resolved
        if self.step6_confocal_handoff_path == path and self.step6_confocal_handoff_origin == "auto":
            return True
        self._set_step6_confocal_handoff(path, handoff, origin="auto")
        return True

    def on_step6_auto_handoff_toggled(self, checked: bool) -> None:
        pair = self._current_step6_pair()
        if not checked and self.step6_confocal_handoff_origin == "auto":
            self._set_step6_confocal_handoff(None, None, origin="none")
        if pair is not None:
            self.on_step6_pair_changed(self.current_step6_pair_index)

    def load_step6_step7_handoff(self) -> None:
        start_dir = self._step6_confocal_handoff_start_dir()
        chosen, _ = QFileDialog.getOpenFileName(
            self,
            "Select Step 7 Handoff JSON",
            start_dir,
            "Step 7 handoff (step8_handoff.json *.json);;JSON (*.json)",
        )
        if not chosen:
            return
        path = Path(chosen)
        try:
            handoff = load_step7_handoff_payload(path)
            build_step7_scene_fov_masks(handoff)
        except Exception as exc:
            QMessageBox.warning(self, "Load Step 7 Handoff", f"Could not load Step 7 handoff:\n{path}\n\n{exc}")
            return
        self._set_step6_confocal_handoff(path, handoff, origin="manual")
        pair = self._current_step6_pair()
        if pair is not None:
            self.on_step6_pair_changed(self.current_step6_pair_index)

    def clear_step6_step7_handoff(self) -> None:
        if self.step6_auto_step7_handoff_check.isChecked():
            self.step6_auto_step7_handoff_check.blockSignals(True)
            self.step6_auto_step7_handoff_check.setChecked(False)
            self.step6_auto_step7_handoff_check.blockSignals(False)
        self._set_step6_confocal_handoff(None, None, origin="none")
        pair = self._current_step6_pair()
        if pair is not None:
            self.on_step6_pair_changed(self.current_step6_pair_index)

    def _step6_pair_display_text(self, pair: WorkspacePair) -> str:
        review = self._step4_pair_review(pair)
        approved = dict(review.get("approved_registration") or {})
        group_tag = str(approved.get("group_tag") or "all")
        stage = self._approved_step5_stage_from_review(review) or str(approved.get("approved_stage") or "unknown")
        return f"{pair.display_label} [approved {stage} | group {group_tag}]"

    def refresh_step6_pairs(self) -> None:
        current_key = self._current_step6_pair().pair_key if self._current_step6_pair() is not None else None
        self.step6_pair_list.clear()
        self.step6_pairs = []
        registry_path = self._step4_registry_path()
        self.step4_pair_registry = load_pair_registry(registry_path)
        roi_root = self._step6_roi_root()
        if self.step4_myelin_root is None or self.step4_nissl_root is None:
            self.step6_root_status.setPlainText("Step 6 requires both myelin and nissl roots.")
            self.step6_pair_label.setText("No approved ROI mapping pair selected")
            self.step6_info.setPlainText("Step 6 requires Step 4/5 roots and an approved registration run.")
            self._set_step6_confocal_status("Confocal FOV overlay: none", warn=False)
            return
        all_pairs = list_cross_stain_pairs(self.step4_myelin_root, self.step4_nissl_root)
        self.step6_pairs = [
            pair
            for pair in all_pairs
            if self._step4_registration_status(pair) == "usable"
            and bool(self._approved_step5_stage_from_review(self._step4_pair_review(pair)))
        ]
        for pair in self.step6_pairs:
            self.step6_pair_list.addItem(self._step6_pair_display_text(pair))
        self.step6_root_status.setPlainText(
            "\n".join(
                [
                    f"myelin_root: {self.step4_myelin_root}",
                    f"nissl_root: {self.step4_nissl_root}",
                    f"roi_root: {roi_root if roi_root is not None else 'none'}",
                    f"approved_pairs: {len(self.step6_pairs)}",
                    "only usable pairs with an approved Step 5 run are shown here",
                ]
            )
        )
        if self.step6_pairs:
            matched_idx = next((i for i, pair in enumerate(self.step6_pairs) if pair.pair_key == current_key), None)
            self.current_step6_pair_index = matched_idx if matched_idx is not None else min(self.current_step6_pair_index, len(self.step6_pairs) - 1)
            self.step6_pair_list.setCurrentRow(self.current_step6_pair_index)
            self.on_step6_pair_changed(self.current_step6_pair_index)
        else:
            self.current_step6_pair_index = 0
            self.step6_current_context = None
            self.step6_current_mapping_result = None
            self.step6_last_updated_source_roi_highres = None
            self.step6_last_updated_target_roi_highres = None
            self.step6_pair_label.setText("No approved ROI mapping pair selected")
            self._update_step6_direction_labels(None)
            self._sync_step6_source_render_mode()
            self.step6_nissl_editor.set_section(np.full((32, 32, 3), 255, dtype=np.uint8), np.zeros((32, 32), dtype=np.uint8), np.zeros((32, 32), dtype=np.uint8))
            self.step6_nissl_editor.set_aux_overlay_rgba(None)
            self._set_step6_target_preview(None, "No mapped ROI yet")
            self.step6_info.setPlainText("No usable pair currently has an approved Step 5 registration.")
            self._set_step6_confocal_status("Confocal FOV overlay: none", warn=False)
            self._set_step6_stale_state(False)
            self._sync_step6_confocal_overlay_toggle_button()
            self._sync_step6_hires_nissl_controls()
            self._clear_step6_hires_source_patch()

    def on_step6_pair_changed(self, index: int) -> None:
        if index < 0 or index >= len(self.step6_pairs):
            return
        self.current_step6_pair_index = index
        pair = self.step6_pairs[index]
        self._maybe_auto_load_step6_confocal_handoff(pair)
        review = self._step4_pair_review(pair)
        common_root = self._pair_common_root()
        roi_root = self._step6_roi_root()
        if common_root is None or roi_root is None:
            return
        context = load_approved_registration_context(
            pair.pair_key,
            review,
            common_root,
            roi_root,
            self.step4_myelin_root,
            self.step4_nissl_root,
        )
        self.step6_current_context = context
        self.step6_current_mapping_result = None
        self._update_step6_direction_labels(pair)
        if context is None:
            self.step6_last_updated_source_roi_highres = None
            self.step6_last_updated_target_roi_highres = None
            self.step6_pair_label.setText(f"{index + 1}/{len(self.step6_pairs)} | {pair.display_label}")
            self._sync_step6_source_render_mode()
            self.step6_nissl_editor.set_section(np.full((32, 32, 3), 255, dtype=np.uint8), np.zeros((32, 32), dtype=np.uint8), np.zeros((32, 32), dtype=np.uint8))
            self.step6_nissl_editor.set_aux_overlay_rgba(None)
            self.step6_info.setPlainText("Approved registration metadata is missing or points to files that no longer exist.")
            self._set_step6_target_preview(None, "No mapped ROI yet")
            self._set_step6_confocal_status("Confocal FOV overlay: none", warn=False)
            self._set_step6_stale_state(False)
            self._sync_step6_confocal_overlay_toggle_button()
            self._sync_step6_hires_nissl_controls()
            self._clear_step6_hires_source_patch()
            return
        source_side = self.step6_source_side
        target_side = self._current_step6_target_side()
        source_name = self._step6_side_display_name(source_side)
        target_name = self._step6_side_display_name(target_side)
        state = current_step6_state(context, source_side=source_side)
        self.step6_last_updated_source_roi_highres = np.asarray(state["source_roi"], dtype=np.uint8).copy()
        self.step6_last_updated_target_roi_highres = np.asarray(state["target_roi"], dtype=np.uint8).copy()
        source_preview_rgb = self._step6_rgb_with_context_overlays(state["source_rgb"], context, pair, source_side)
        target_preview_rgb = self._step6_rgb_with_context_overlays(state["target_rgb"], context, pair, target_side)
        self._sync_step6_source_render_mode()
        self.step6_nissl_editor.set_section(
            source_preview_rgb,
            state["source_roi"],
            np.zeros(state["source_roi"].shape, dtype=np.uint8),
        )
        self.step6_nissl_editor.set_active_layer("tissue")
        self.step6_nissl_editor.set_aux_overlay_rgba(None)
        self._set_step6_target_view(target_preview_rgb, state["target_roi"])
        self.step6_pair_label.setText(f"{index + 1}/{len(self.step6_pairs)} | {pair.display_label}")
        self._set_step6_stale_state(False)
        self.step6_hires_last_request_key = None
        self._sync_step6_confocal_overlay_toggle_button()
        self._sync_step6_hires_nissl_controls()
        approved = dict(review.get("approved_registration") or {})
        if isinstance(self.step6_confocal_handoff, dict):
            handoff_label = str(self.step6_confocal_handoff.get("myelin_label") or "").strip()
            origin = self.step6_confocal_handoff_origin or "manual"
            origin_text = "manually loaded" if origin == "manual" else "auto-loaded latest Step 7 handoff"
            if handoff_label and handoff_label != pair.myelin_item.label:
                self._set_step6_confocal_status(
                    f"Confocal FOV overlay {origin_text} for {handoff_label}, but current pair uses {pair.myelin_item.label}. Overlay hidden.",
                    warn=True,
                )
            else:
                path_text = f"\n{self.step6_confocal_handoff_path}" if self.step6_confocal_handoff_path is not None else ""
                visibility_text = (
                    "Currently hidden in Step 6."
                    if not self.step6_confocal_overlay_visible
                    else "Both source and target sides show accepted+frozen confocal FOV/grid."
                )
                self._set_step6_confocal_status(
                    f"Confocal FOV overlay active for {pair.myelin_item.label} ({origin_text}). "
                    f"{visibility_text}{path_text}",
                    warn=False,
                )
        elif self.step6_auto_step7_handoff_check.isChecked():
            self._set_step6_confocal_status(
                f"Confocal FOV overlay: none | no Step 7 handoff found yet for {pair.myelin_item.label}. "
                "Step 6 still works independently.",
                warn=False,
            )
        else:
            self._set_step6_confocal_status(
                "Confocal FOV overlay: none | auto-load disabled. Step 6 is running independently.",
                warn=False,
            )
        self.step6_info.setPlainText(
            "\n".join(
                [
                    f"pair_key: {pair.pair_key}",
                    f"approved_run_dir: {approved.get('run_dir', 'missing')}",
                    f"approved_stage: {approved.get('approved_stage', 'unknown')}",
                    f"registration_backend: {context.registration_backend}",
                    f"group_tag: {approved.get('group_tag', 'all')}",
                    f"roi_output_dir: {context.output_dir}",
                    "",
                    "Editing:",
                    f"- draw ROI on the high-resolution {source_name} side",
                    "- tissue brush is the intended ROI layer; artifact is ignored on update/save",
                    "- Update ROI Mapping applies the approved Step 5 transform without re-optimizing registration",
                    "- green/yellow = current batch added ROI, magenta = current batch removed ROI",
                f"- Save writes the current high-resolution {source_name} ROI and mapped {target_name} ROI",
                "- S saves and advances to the next approved pair",
                ]
            )
        )
        if self.step6_hires_nissl_check.isChecked() and self.step6_source_side == "nissl":
            self._schedule_step6_hires_source_patch_refresh(delay_ms=0)
        else:
            self._clear_step6_hires_source_patch()

    def update_step6_roi_mapping_preview(self) -> bool:
        context = self.step6_current_context
        if context is None:
            return False
        source_side = self.step6_source_side
        target_side = self._current_step6_target_side()
        source_name = self._step6_side_display_name(source_side)
        target_name = self._step6_side_display_name(target_side)
        ants_bin = find_ants_bin() if context.registration_backend == "ants" else Path()
        if context.registration_backend == "ants" and ants_bin is None:
            QMessageBox.warning(self, "Step 6 ROI Mapping", "Could not find a local ANTs installation.")
            return False
        roi_labels_highres = self._current_step6_roi_highres()
        previous_target_roi = (
            self.step6_last_updated_target_roi_highres.copy()
            if self.step6_last_updated_target_roi_highres is not None
            else None
        )
        result = update_step6_roi_mapping(context, roi_labels_highres, ants_bin, source_side=source_side)
        self.step6_current_mapping_result = result
        state = current_step6_state(context, source_side=source_side)
        mapped = np.asarray(result["target_roi_highres"], dtype=np.uint8)
        target_preview_rgb = self._step6_rgb_with_context_overlays(state["target_rgb"], context, self._current_step6_pair(), target_side)
        self._set_step6_target_view(target_preview_rgb, mapped, preserve_view=True)
        self.step6_last_updated_source_roi_highres = roi_labels_highres.copy()
        self.step6_last_updated_target_roi_highres = mapped.copy()
        self.step6_nissl_editor.set_aux_overlay_rgba(None)
        self._set_step6_stale_state(False)
        self.step6_info.append(
            f"Updated {source_name} -> {target_name} ROI mapping preview using the approved Step 5 transform. "
            "Green/yellow shows newly added ROI in this batch; magenta shows removed ROI."
        )
        return True

    def on_step6_roi_mask_changed(self) -> None:
        if self.step6_current_context is None:
            return
        self.step6_current_mapping_result = None
        source_name = self._step6_side_display_name(self.step6_source_side)
        target_name = self._step6_side_display_name(self._current_step6_target_side())
        self._refresh_step6_source_batch_overlay()
        already_stale = self.step6_preview_stale
        self._set_step6_stale_state(True, reason=f"{source_name} ROI changed; {target_name} preview is out of date")
        if not already_stale:
            self.step6_info.append(
                "ROI changed: preview mapping is now stale. "
                "Green/yellow marks added ROI in the current edit batch; magenta marks removed ROI. "
                f"Click Update ROI Mapping to refresh the {target_name} overlay."
            )

    def save_step6_roi(self) -> bool:
        pair = self._current_step6_pair()
        context = self.step6_current_context
        if pair is None or context is None:
            return False
        if self.step6_current_mapping_result is None:
            target_name = self._step6_side_display_name(self._current_step6_target_side())
            msg = QMessageBox(self)
            msg.setIcon(QMessageBox.Icon.Warning)
            msg.setWindowTitle("Step 6 ROI Mapping")
            msg.setText(f"Mapped {target_name} preview is stale.")
            msg.setInformativeText("Update ROI Mapping before saving so the right-side ROI matches the latest left-side edits.")
            update_button = msg.addButton("Update Mapping", QMessageBox.ButtonRole.AcceptRole)
            cancel_button = msg.addButton("Cancel", QMessageBox.ButtonRole.RejectRole)
            msg.setDefaultButton(update_button)
            msg.exec()
            if msg.clickedButton() is not update_button:
                return False
            if not self.update_step6_roi_mapping_preview():
                return False
        result = save_step6_roi_outputs(context, self.step6_current_mapping_result or {})
        state = current_step6_state(context, source_side=self.step6_source_side)
        target_preview_rgb = self._step6_rgb_with_context_overlays(
            state["target_rgb"],
            context,
            self._current_step6_pair(),
            self._current_step6_target_side(),
        )
        self._set_step6_target_view(target_preview_rgb, state["target_roi"], preserve_view=True)
        self.step6_last_updated_source_roi_highres = np.asarray(state["source_roi"], dtype=np.uint8).copy()
        self.step6_last_updated_target_roi_highres = np.asarray(state["target_roi"], dtype=np.uint8).copy()
        self.step6_nissl_editor.set_aux_overlay_rgba(None)
        self._set_step6_stale_state(False)
        registry_path = self._step4_registry_path()
        if registry_path is not None:
            review = self._step4_pair_review(pair)
            review["roi_mapping"] = {
                "output_dir": self._relpath_from_common_root(context.output_dir),
                "manifest_path": self._relpath_from_common_root(context.output_dir / "roi_manifest.json"),
                "source_side": str(result.get("source_side") or self.step6_source_side),
                "target_side": str(result.get("target_side") or self._current_step6_target_side()),
                "registration_backend": context.registration_backend,
                "saved_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
            self.step4_pair_registry[pair.pair_key] = review
            save_pair_registry(registry_path, self.step4_pair_registry)
        self.step6_current_mapping_result = None
        self.step6_info.append(f"Saved ROI outputs: {context.output_dir}")
        return True

    def save_step6_roi_and_next(self) -> None:
        old_index = self.current_step6_pair_index
        if not self.save_step6_roi():
            return
        if not self.step6_pairs:
            return
        target_index = min(old_index + 1, len(self.step6_pairs) - 1)
        self.current_step6_pair_index = target_index
        if self.step6_pair_list.currentRow() != target_index:
            self.step6_pair_list.setCurrentRow(target_index)
        self.on_step6_pair_changed(target_index)

    def _step7_runs_root(self) -> Path | None:
        return default_confocal_registration_root(self.step7_myelin_root)

    def _step7_export_root(self) -> Path | None:
        if self.step7_confocal_paths:
            source_dirs = [
                (path.parent if path.is_file() else path)
                for path in self.step7_confocal_paths
            ]
            try:
                common_dir = Path(os.path.commonpath([str(path) for path in source_dirs]))
            except Exception:
                common_dir = source_dirs[0]
            return common_dir
        return self._default_step7_confocal_root()

    def _latest_step7_run_dir(self, label: str) -> Path | None:
        runs_root = self._step7_runs_root()
        if runs_root is None:
            return None
        section_root = runs_root / label
        if not section_root.exists():
            return None
        candidates = [p for p in section_root.iterdir() if p.is_dir()]
        if not candidates:
            return None
        return sorted(candidates)[-1]

    def _current_step7_item(self) -> WorkspaceSection | None:
        if 0 <= self.current_step7_section_index < len(self.step7_sections):
            return self.step7_sections[self.current_step7_section_index]
        return None

    def _step7_any_worker_running(self) -> bool:
        return any(
            worker is not None
            for worker in (
                self.step7_run_thread,
                self.step7_auto_scale_thread,
                self.step7_seed_screen_thread,
                self.step7_frontier_thread,
            )
        )

    def _default_step7_confocal_root(self) -> Path:
        if self.step7_confocal_paths:
            first = self.step7_confocal_paths[0]
            return first.parent if first.is_dir() else first.parent
        preferred = self._preferred_existing_path(r"D:\Research\Image Analysis\Confocal Myelin data")
        if preferred is not None:
            return preferred
        return self._default_crop_workspace_root()

    def _default_step7_confocal_roi_root(self) -> Path | None:
        preferred = self._preferred_existing_path(r"D:\Research\Image Analysis\Confocal Myelin data\202512_8rats_3ROIs")
        if preferred is not None:
            return preferred
        for path in self.step7_confocal_paths:
            probe = path if path.is_dir() else path.parent
            for parent in (probe, *probe.parents):
                if parent.name == "202512_8rats_3ROIs" and parent.exists():
                    return parent
        candidate = self._default_step7_confocal_root()
        if candidate is not None and candidate.name == "202512_8rats_3ROIs" and candidate.exists():
            return candidate
        return None

    def _step7_available_confocal_section_labels(self) -> tuple[Path | None, set[str]]:
        roi_root = self._default_step7_confocal_roi_root()
        labels: set[str] = set()
        if roi_root is None or not roi_root.exists():
            return roi_root, labels
        try:
            for path in roi_root.iterdir():
                if not path.is_dir() or path.name.startswith("_"):
                    continue
                parts = path.name.split("_")
                if len(parts) < 4:
                    continue
                labels.add("_".join(parts[:2]))
        except Exception:
            return roi_root, set()
        return roi_root, labels

    def _describe_step7_confocal_sources(self) -> str:
        if not self.step7_confocal_paths:
            return "none"
        if len(self.step7_confocal_paths) == 1:
            return str(self.step7_confocal_paths[0])
        return f"{len(self.step7_confocal_paths)} sources | first={self.step7_confocal_paths[0].name}"

    @staticmethod
    def _format_step7_scale_arrow(source_um: object, target_um: object) -> str:
        if not isinstance(source_um, (list, tuple)) or len(source_um) != 2:
            return "unknown -> unknown"
        if not isinstance(target_um, (list, tuple)) or len(target_um) != 2:
            return f"{tuple(source_um)} -> unknown"
        return f"({float(source_um[0]):.3f}, {float(source_um[1]):.3f}) -> ({float(target_um[0]):.3f}, {float(target_um[1]):.3f})"

    @staticmethod
    def _format_step7_extent_summary(
        *,
        fixed_info: object,
        projection_info: object,
        raw_projection: object,
        scaled_projection: object,
    ) -> list[str]:
        if not isinstance(fixed_info, dict) or not isinstance(projection_info, dict):
            return []
        fixed_support_shape = fixed_info.get("support_shape_hw")
        fixed_source_um = fixed_info.get("source_um_per_px_xy")
        source_um = projection_info.get("source_um_per_px_xy")
        lines: list[str] = []
        if (
            isinstance(fixed_support_shape, (list, tuple))
            and len(fixed_support_shape) == 2
            and isinstance(fixed_source_um, (list, tuple))
            and len(fixed_source_um) == 2
        ):
            support_h = float(fixed_support_shape[0])
            support_w = float(fixed_support_shape[1])
            lines.append(
                "section_support_um_extent_xy: "
                f"({support_w * float(fixed_source_um[0]):.1f}, {support_h * float(fixed_source_um[1]):.1f})"
            )
        if (
            raw_projection is not None
            and hasattr(raw_projection, "shape")
            and isinstance(source_um, (list, tuple))
            and len(source_um) == 2
        ):
            raw_h, raw_w = raw_projection.shape[:2]
            conf_w_um = float(raw_w) * float(source_um[0])
            conf_h_um = float(raw_h) * float(source_um[1])
            lines.append(f"confocal_raw_projection_shape: ({int(raw_h)}, {int(raw_w)})")
            lines.append(f"confocal_um_extent_xy: ({conf_w_um:.1f}, {conf_h_um:.1f})")
            if (
                isinstance(fixed_support_shape, (list, tuple))
                and len(fixed_support_shape) == 2
                and isinstance(fixed_source_um, (list, tuple))
                and len(fixed_source_um) == 2
            ):
                support_h = float(fixed_support_shape[0])
                support_w = float(fixed_support_shape[1])
                sec_w_um = support_w * float(fixed_source_um[0])
                sec_h_um = support_h * float(fixed_source_um[1])
                if sec_w_um > 0 and sec_h_um > 0:
                    lines.append(
                        "confocal_vs_section_support_fraction_xy: "
                        f"({100.0 * conf_w_um / sec_w_um:.1f}%, {100.0 * conf_h_um / sec_h_um:.1f}%)"
                    )
        if scaled_projection is not None and hasattr(scaled_projection, "shape"):
            scaled_h, scaled_w = scaled_projection.shape[:2]
            lines.append(f"confocal_scaled_preview_shape: ({int(scaled_h)}, {int(scaled_w)})")
        return lines

    @staticmethod
    def _format_step7_grid_summary(stitch_info: object) -> list[str]:
        if not isinstance(stitch_info, dict):
            return []
        grid_shape = stitch_info.get("grid_shape_rc")
        tile_shape = stitch_info.get("tile_shape_hw")
        step_xy = stitch_info.get("grid_step_xy_px")
        overlap_xy = stitch_info.get("inferred_overlap_fraction_xy")
        lines: list[str] = []
        if isinstance(grid_shape, (list, tuple)) and len(grid_shape) == 2:
            lines.append(f"grid_shape_rc: ({int(grid_shape[0])}, {int(grid_shape[1])})")
        if isinstance(tile_shape, (list, tuple)) and len(tile_shape) == 2:
            lines.append(f"tile_shape_hw: ({int(tile_shape[0])}, {int(tile_shape[1])})")
        if isinstance(step_xy, (list, tuple)) and len(step_xy) == 2:
            lines.append(f"grid_step_xy_px: ({float(step_xy[0]):.1f}, {float(step_xy[1]):.1f})")
        if isinstance(overlap_xy, (list, tuple)) and len(overlap_xy) == 2:
            overlap_x = "unknown" if overlap_xy[0] is None else f"{100.0 * float(overlap_xy[0]):.1f}%"
            overlap_y = "unknown" if overlap_xy[1] is None else f"{100.0 * float(overlap_xy[1]):.1f}%"
            lines.append(f"inferred_overlap_xy: ({overlap_x}, {overlap_y})")
        return lines

    def _compute_step7_duplicate_stack_report(self) -> None:
        self.step7_duplicate_stack_report = None
        if len(self.step7_confocal_paths) <= 1:
            return
        suffixes = {path.suffix.lower() for path in self.step7_confocal_paths}
        if suffixes - {".tif", ".tiff"}:
            self.step7_duplicate_stack_report = {
                "checked": False,
                "reason": "non_tiff_sources",
                "source_count": len(self.step7_confocal_paths),
            }
            return
        try:
            self.step7_duplicate_stack_report = analyze_confocal_duplicate_stacks(self.step7_confocal_paths)
        except Exception as exc:
            self.step7_duplicate_stack_report = {
                "checked": False,
                "reason": "analysis_failed",
                "error": f"{type(exc).__name__}: {exc}",
                "source_count": len(self.step7_confocal_paths),
            }

    def _step7_duplicate_stack_summary_lines(self) -> list[str]:
        report = self.step7_duplicate_stack_report
        if not isinstance(report, dict):
            return ["duplicate_stack_check: not_run"]
        if report.get("error"):
            return [f"duplicate_stack_check: failed ({report.get('error')})"]
        if not bool(report.get("checked")):
            return [f"duplicate_stack_check: {report.get('reason', 'not_applicable')}"]
        duplicate_groups = list(report.get("duplicate_groups") or [])
        if not duplicate_groups:
            source_count = int(report.get("source_count") or len(self.step7_confocal_paths))
            return [f"duplicate_stack_check: none ({source_count} tile stacks checked)"]
        source_count = int(report.get("source_count") or 0)
        unique_count = int(report.get("unique_stack_count") or 0)
        lines = [
            (
                "duplicate_stack_check: WARNING "
                f"{int(report.get('duplicate_stack_count') or 0)} duplicate stacks across "
                f"{int(report.get('duplicate_group_count') or 0)} groups | "
                f"unique_volumes={unique_count}/{source_count} | "
                f"all_tiles_identical={bool(report.get('all_tiles_identical', False))}"
            )
        ]
        for group in duplicate_groups[:3]:
            names = list(group.get("names") or [])
            preview = ", ".join(str(name) for name in names[:4])
            truncated = int(group.get("truncated_name_count") or 0)
            if truncated > 0:
                preview += f", +{truncated} more"
            lines.append(f"duplicate_group_{int(group.get('group_index') or 0)}: {preview}")
        remaining = max(0, int(report.get("duplicate_group_count") or len(duplicate_groups)) - 3)
        if remaining > 0:
            lines.append(f"duplicate_group_more: +{remaining} additional groups")
        return lines

    def _warn_step7_duplicate_stack_report(self, *, context: str) -> None:
        report = self.step7_duplicate_stack_report
        if self.step7_duplicate_stack_warning_shown:
            return
        if not isinstance(report, dict) or not bool(report.get("checked")):
            return
        duplicate_groups = list(report.get("duplicate_groups") or [])
        if not duplicate_groups:
            return
        lines = [
            "Duplicate confocal z-stacks were detected before Step 7 import.",
            f"context: {context}",
            (
                f"unique volumes: {int(report.get('unique_stack_count') or 0)} / "
                f"{int(report.get('source_count') or len(self.step7_confocal_paths))}"
            ),
        ]
        if bool(report.get("all_tiles_identical", False)):
            lines.append("All selected tiles resolve to the same voxel volume.")
        lines.append("")
        lines.append("Example duplicate groups:")
        for group in duplicate_groups[:3]:
            names = list(group.get("names") or [])
            preview = ", ".join(str(name) for name in names[:4])
            truncated = int(group.get("truncated_name_count") or 0)
            if truncated > 0:
                preview += f", +{truncated} more"
            lines.append(f"- {preview}")
        lines.extend(
            [
                "",
                "This usually means the CZI -> OME-TIFF export is wrong.",
                "Re-extract the original .czi one series at a time before continuing.",
            ]
        )
        self.step7_duplicate_stack_warning_shown = True
        QMessageBox.warning(self, "Step 7 Duplicate Stack Warning", "\n".join(lines))

    def _update_step7_info_text(self) -> None:
        item = self._current_step7_item()
        fixed_shape = None if self.step7_fixed_rgb is None else self.step7_fixed_rgb.shape[:2]
        fixed_um = None if self.step7_fixed_info is None else self.step7_fixed_info.get("preview_um_per_px_xy")
        source_um = None if self.step7_projection_info is None else self.step7_projection_info.get("source_um_per_px_xy")
        scale_info = None if self.step7_projection_info is None else self.step7_projection_info.get("scale_to_section_preview")
        stitch_info = None if self.step7_projection_info is None else self.step7_projection_info.get("stitch_info")
        diag = self.step7_preview_view.diagnostic_snapshot() if hasattr(self, "step7_preview_view") else {}
        points_a = diag.get("points_a_scene", {}) if isinstance(diag, dict) else {}
        points_b = diag.get("points_b_raw", {}) if isinstance(diag, dict) else {}
        selected_labels = list(diag.get("selected_tile_labels") or []) if isinstance(diag, dict) else []
        selected_summary = "none"
        if selected_labels:
            preview = ", ".join(str(v) for v in selected_labels[:4])
            if len(selected_labels) > 4:
                preview += f", +{len(selected_labels) - 4} more"
            selected_summary = f"{preview} ({len(selected_labels)} selected)"
        complete_pairs = sorted({int(k) for k in points_a.keys()} & {int(k) for k in points_b.keys()})
        lines = [
            f"myelin_label: {item.label if item is not None else 'none'}",
            f"section_dir: {item.section_dir if item is not None else 'none'}",
            f"confocal_source_mode: {self.step7_confocal_source_mode}",
            f"confocal_sources: {self._describe_step7_confocal_sources()}",
            *self._step7_duplicate_stack_summary_lines(),
            f"fixed_preview_shape: {fixed_shape}",
            f"fixed_preview_um_per_px: {fixed_um}",
            f"step7_target_working_um_per_px: {float(STEP7_TARGET_UM_PER_PX):.1f}",
            f"step7_registration_input_profile: {self._step7_registration_profile_value()}",
            f"confocal_display_and_registration_polarity: inverted_for_myelin_matching",
            f"source_um_per_px -> section_preview_um_per_px: {self._format_step7_scale_arrow(source_um, fixed_um)}",
            f"manual_flip_state: LR={bool(self.step7_flip_lr_check.isChecked())}, UD={bool(self.step7_flip_ud_check.isChecked())}",
            f"anchor_mode_active: {bool(diag.get('diagnostic_active', False))}",
            f"anchor_target: {diag.get('next_group', 'A')}{diag.get('next_index', 1)}",
            f"anchor_transform_locked: {bool(diag.get('transform_locked', False))}",
            f"selected_tile: {selected_summary}",
            f"accepted_tiles: {self._describe_step7_accepted_tiles()}",
            f"frozen_tiles: {self._describe_step7_frozen_tiles()}",
            f"hold_tiles: {self._describe_step7_hold_tiles()}",
            f"frontier_tiles: {self._describe_step7_frontier_tiles()}",
            f"confocal_overlay_opacity: {float(diag.get('overlay_opacity', 0.85)):.2f}",
            f"anchor_points: A={len(points_a)} | B={len(points_b)} | complete_pairs={complete_pairs or 'none'}",
        ]
        if self.step7_last_manual_action:
            lines.append(f"last_manual_action: {self.step7_last_manual_action}")
        if isinstance(self.step7_fixed_info, dict) and self.step7_fixed_info.get("fixed_working_mode"):
            lines.append(f"fixed_working_mode: {self.step7_fixed_info.get('fixed_working_mode')}")
        if isinstance(scale_info, dict):
            lines.append(f"projection_scale_to_section: {scale_info}")
        lines.extend(
            self._format_step7_extent_summary(
                fixed_info=self.step7_fixed_info,
                projection_info=self.step7_projection_info,
                raw_projection=self.step7_confocal_projection_raw_u8,
                scaled_projection=self.step7_confocal_projection_u8,
            )
        )
        lines.extend(self._format_step7_grid_summary(stitch_info))
        lines.extend(self._format_step7_current_tracker_lines())
        lines.append("note: physical comparison is against the section support crop used in Step 7, not the full slide canvas")
        lines.extend(
            [
                    "",
                    "Workflow:",
                    "- select a confocal z-stack source, CZI, or multi-TIFF strip/grid",
                    "- generate a 2D projection",
                    "- adjust manual coarse alignment",
                    "- optionally run Auto Scale Sweep to optimize the whole-grid scale before tile screening",
                    "- optionally screen seed tiles, freeze a trusted tile, then propagate frontier neighbors",
                f"- run local refine using {self._step7_registration_profile_value()}",
            ]
        )
        if self.step7_last_run_summary_lines:
            lines.extend(["", *self.step7_last_run_summary_lines])
        if self.step7_last_auto_scale_summary_lines:
            lines.extend(["", *self.step7_last_auto_scale_summary_lines])
        if self.step7_last_seed_screen_summary_lines:
            lines.extend(["", *self.step7_last_seed_screen_summary_lines])
        if self.step7_last_frontier_summary_lines:
            lines.extend(["", *self.step7_last_frontier_summary_lines])
        if self.step7_diagnostic_log:
            lines.extend(["", "Anchor log:"])
            lines.extend(self.step7_diagnostic_log[-20:])
        self.step7_info.setPlainText("\n".join(lines))

    def _refresh_step8_info(self) -> None:
        if not hasattr(self, "step8_info") or self.step8_info is None:
            return
        latest_export = str(self.step7_last_export_dir) if self.step7_last_export_dir is not None else "none"
        lines = [
            "Step 8 Scaffold",
            "- primary upstream input: Step 7 session export",
            "- required handoff file: step8_handoff.json",
            "- planned second input: nnUNet 3D myelin prediction / inference export",
            "- planned functions:",
            "  * load registered confocal tile positions and transforms",
            "  * connect them to predicted myelin maps in the same Step 7 preview scene space",
            "  * visualize prediction overlays on the confocal-myelin tile view",
            "  * compute tile-wise and pooled fiber-density summaries",
            "",
            f"Latest Step 7 export: {latest_export}",
        ]
        if self.step7_last_export_dir is not None:
            lines.extend(
                [
                    f"- session_manifest: {self.step7_last_export_dir / 'session_manifest.json'}",
                    f"- step8_handoff: {self.step7_last_export_dir / 'step8_handoff.json'}",
                    f"- tile_transforms_csv: {self.step7_last_export_dir / 'tile_transforms.csv'}",
                ]
            )
        self.step8_info.setPlainText("\n".join(lines))

    def _format_step7_current_tracker_lines(self) -> list[str]:
        snap = self.step7_preview_view.diagnostic_snapshot() if hasattr(self, "step7_preview_view") else {}
        points_a = snap.get("points_a_scene", {}) if isinstance(snap, dict) else {}
        points_b = snap.get("points_b_raw", {}) if isinstance(snap, dict) else {}
        points_b_scene = snap.get("points_b_scene", {}) if isinstance(snap, dict) else {}
        pair_ids = sorted({int(k) for k in points_a.keys()} & {int(k) for k in points_b.keys()})
        lines = [
            "current_repro_tracker:",
            (
                "  manual_transform: "
                f"tx={float(self.step7_tx_spin.value()):.1f} "
                f"ty={float(self.step7_ty_spin.value()):.1f} "
                f"angle={float(self.step7_angle_spin.value()):.1f} "
                f"scale={float(self.step7_scale_spin.value()):.5f} "
                f"flip(LR,UD)=({bool(self.step7_flip_lr_check.isChecked())},{bool(self.step7_flip_ud_check.isChecked())})"
            ),
            (
                "  tracker_ready: "
                f"complete_anchor_pairs={pair_ids or 'none'} "
                f"source_mode={self.step7_confocal_source_mode or 'none'} "
                f"projection={str(self.step7_projection_mode_combo.currentData() or 'focus')}"
            ),
            f"  selected_tile: {snap.get('selected_tile_label') or 'none'}",
            f"  accepted_tiles: {self._describe_step7_accepted_tiles()}",
            f"  frozen_tiles: {self._describe_step7_frozen_tiles()}",
            f"  hold_tiles: {self._describe_step7_hold_tiles()}",
            f"  frontier_tiles: {self._describe_step7_frontier_tiles()}",
        ]
        if pair_ids:
            lines.append("  anchor_pairs_explicit:")
            for idx in pair_ids[:9]:
                a = points_a.get(str(idx), [float("nan"), float("nan")])
                b_scene = points_b_scene.get(str(idx), [float("nan"), float("nan")])
                b_raw = points_b.get(str(idx), [float("nan"), float("nan")])
                lines.append(
                    f"    A{idx}=({float(a[0]):.1f},{float(a[1]):.1f}) | "
                    f"B{idx}_scene=({float(b_scene[0]):.1f},{float(b_scene[1]):.1f}) | "
                    f"B{idx}_raw=({float(b_raw[0]):.1f},{float(b_raw[1]):.1f})"
                )
        return lines

    def _reset_step7_diagnostic_state(self) -> None:
        self.step7_diagnostic_log = []
        if hasattr(self, "step7_preview_view"):
            self.step7_preview_view.clear_diagnostic_points()

    def _clear_step7_storyboard_display(self, text: str = "No Step 7 fiber QC storyboard yet") -> None:
        self.step7_storyboard_label.setText(str(text))
        self.step7_storyboard_label.setPixmap(QPixmap())

    def _reset_step7_session_state(
        self,
        *,
        clear_confocal_paths: bool = False,
        clear_loaded_projection: bool = True,
        clear_duplicate_report: bool = True,
        reset_transform: bool = False,
    ) -> None:
        if clear_confocal_paths:
            self.step7_confocal_paths = []
            self.step7_confocal_source_mode = "none"
            self.step7_stack_label.setText("No confocal source selected")
        if clear_duplicate_report:
            self.step7_duplicate_stack_report = None
            self.step7_duplicate_stack_warning_shown = False
        if clear_loaded_projection:
            self.step7_projection_info = None
            self.step7_confocal_projection_raw_u8 = None
            self.step7_confocal_projection_u8 = None
            self.step7_confocal_projection_mask_raw_u8 = None
            self.step7_confocal_projection_mask_u8 = None
        if reset_transform:
            self._set_step7_transform_spins(0.0, 0.0, 0.0, 1.0)
        self.step7_last_manual_action = None
        self.step7_last_run_dir = None
        self.step7_last_auto_scale_dir = None
        self.step7_last_frontier_dir = None
        self.step7_last_run_summary_lines = []
        self.step7_last_auto_scale_summary_lines = []
        self.step7_last_frontier_summary_lines = []
        self.step7_last_seed_screen_dir = None
        self.step7_last_export_dir = None
        self.step7_last_seed_screen_rows = []
        self.step7_last_frontier_rows = []
        self.step7_tile_result_rows = {}
        self.step7_last_seed_screen_summary_lines = []
        self.step7_accepted_tile_indices = set()
        self.step7_hold_tile_indices = set()
        self.step7_frozen_tile_indices = set()
        self.step7_frontier_tile_indices = set()
        self.step7_progress_state = None
        self.step7_progress_label.setText("Step 7 progress: idle")
        self.step7_progress_bar.setValue(0)
        self.step7_progress_detail_label.setText("Active tiles: none")
        self._reset_step7_diagnostic_state()
        self._clear_step7_storyboard_display()
        if clear_loaded_projection or clear_confocal_paths:
            self._clear_step7_preview_overlay_state(reset_alignment=reset_transform)
        else:
            self._update_step7_frozen_count_label()

    @staticmethod
    def _fmt_step7_metric(value: object, *, digits: int = 4) -> str:
        try:
            val = float(value)
        except Exception:
            return "nan"
        if not np.isfinite(val):
            return "inf" if val > 0 else "-inf"
        return f"{val:.{digits}f}"

    def _step7_registration_profile_value(self) -> str:
        data = self.step7_profile_combo.currentData() if hasattr(self, "step7_profile_combo") else None
        return str(data or STEP7_REGISTRATION_INPUT_PROFILE)

    def _build_step7_run_summary_lines(self, data: dict[str, object]) -> list[str]:
        refine_model = str(
            data.get("local_refine_model")
            or (data.get("local_registration") if isinstance(data.get("local_registration"), dict) else {}).get("transform_model")
            or "similarity"
        )
        run_dir = str(data.get("run_dir") or "").strip()
        input_metrics = data.get("input_metrics") if isinstance(data.get("input_metrics"), dict) else {}
        refine_metrics = (
            data.get("refine_metrics")
            if isinstance(data.get("refine_metrics"), dict)
            else (data.get("rigid_metrics") if isinstance(data.get("rigid_metrics"), dict) else {})
        )
        full_input_metrics = data.get("full_input_metrics") if isinstance(data.get("full_input_metrics"), dict) else {}
        full_refine_metrics = (
            data.get("full_refine_metrics")
            if isinstance(data.get("full_refine_metrics"), dict)
            else (data.get("full_rigid_metrics") if isinstance(data.get("full_rigid_metrics"), dict) else {})
        )
        timing = data.get("timing_seconds") if isinstance(data.get("timing_seconds"), dict) else {}
        manual_init = data.get("manual_init") if isinstance(data.get("manual_init"), dict) else {}
        coarse = data.get("coarse_alignment_record") if isinstance(data.get("coarse_alignment_record"), dict) else {}
        local_info = data.get("local_registration") if isinstance(data.get("local_registration"), dict) else {}
        roi_bbox = local_info.get("roi_bbox_yxyx") if isinstance(local_info.get("roi_bbox_yxyx"), list) else coarse.get("local_roi_bbox_yxyx")
        zoom_bbox = local_info.get("fiber_qc_zoom_bbox_yxyx") if isinstance(local_info.get("fiber_qc_zoom_bbox_yxyx"), list) else coarse.get("fiber_qc_zoom_bbox_yxyx")
        files = data.get("files") if isinstance(data.get("files"), dict) else {}
        anchor_info = data.get("manual_anchor_mode") if isinstance(data.get("manual_anchor_mode"), dict) else {}
        lines = [
            "Last Step 7 run:",
            f"- refine_model: {refine_model}",
            (
                "- registration_input_profile: "
                f"{str(data.get('registration_input_profile') or local_info.get('registration_input_profile') or STEP7_REGISTRATION_INPUT_PROFILE)}"
            ),
            (
                "- local pattern metrics (registration input): "
                f"CC {self._fmt_step7_metric(input_metrics.get('cc'))} -> {self._fmt_step7_metric(refine_metrics.get('cc'))} | "
                f"MI {self._fmt_step7_metric(input_metrics.get('mi'))} -> {self._fmt_step7_metric(refine_metrics.get('mi'))}"
            ),
            (
                "- local geometry diagnostics: "
                f"Dice {self._fmt_step7_metric(input_metrics.get('dice'))} -> {self._fmt_step7_metric(refine_metrics.get('dice'))} | "
                f"HD95 {self._fmt_step7_metric(input_metrics.get('hd95_px'), digits=1)} -> {self._fmt_step7_metric(refine_metrics.get('hd95_px'), digits=1)}"
            ),
            (
                "- whole-slice diagnostics: "
                f"CC {self._fmt_step7_metric(full_input_metrics.get('cc'))} -> {self._fmt_step7_metric(full_refine_metrics.get('cc'))} | "
                f"MI {self._fmt_step7_metric(full_input_metrics.get('mi'))} -> {self._fmt_step7_metric(full_refine_metrics.get('mi'))}"
            ),
            (
                "- timing_seconds: "
                f"ants={self._fmt_step7_metric(timing.get('ants_registration'), digits=2)} "
                f"total={self._fmt_step7_metric(timing.get('total'), digits=2)}"
            ),
        ]
        if run_dir:
            lines.append(f"- run_dir: {run_dir}")
        if manual_init:
            lines.append(
                "- coarse manual state: "
                f"tx={self._fmt_step7_metric(manual_init.get('tx_px'), digits=1)} "
                f"ty={self._fmt_step7_metric(manual_init.get('ty_px'), digits=1)} "
                f"angle={self._fmt_step7_metric(manual_init.get('angle_deg'), digits=1)} "
                f"scale={self._fmt_step7_metric(manual_init.get('scale'), digits=4)} "
                f"flip(LR,UD)=({bool(manual_init.get('flip_lr'))},{bool(manual_init.get('flip_ud'))})"
            )
        if anchor_info:
            lines.append(
                "- manual_anchor_mode: "
                f"used={bool(anchor_info.get('used', False))} "
                f"model={anchor_info.get('model', 'none')} "
                f"pair_count={int(anchor_info.get('pair_count', 0) or 0)}"
            )
        if roi_bbox:
            lines.append(f"- local_roi_bbox_yxyx: {tuple(int(v) for v in roi_bbox)}")
        if zoom_bbox:
            lines.append(f"- fiber_qc_zoom_bbox_yxyx: {tuple(int(v) for v in zoom_bbox)}")
        if files:
            full_ref = str(files.get("coarse_alignment_overlay_full") or "").strip()
            local_ref = str(files.get("coarse_alignment_overlay_local") or "").strip()
            if full_ref or local_ref:
                lines.append(
                    "- coarse alignment snapshots: "
                    f"full=`{full_ref}` local=`{local_ref}`"
                )
            repro_tracker_ref = str(files.get("repro_tracker") or "").strip()
            if repro_tracker_ref:
                lines.append(f"- repro_tracker: `{repro_tracker_ref}`")
        return lines

    def _load_step7_fixed_section(self, item: WorkspaceSection) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
        cached = self.step7_fixed_cache.get(item.label)
        if cached is not None:
            return cached
        bundle = prepare_myelin_confocal_fixed_bundle(item, max_long_edge=None, target_um_per_px=STEP7_TARGET_UM_PER_PX)
        info = {
            "preview_um_per_px_xy": list(bundle.preview_um_per_px_xy) if bundle.preview_um_per_px_xy is not None else None,
            "source_um_per_px_xy": list(bundle.source_um_per_px_xy) if bundle.source_um_per_px_xy is not None else None,
            "support_shape_hw": [int(bundle.support_shape_hw[0]), int(bundle.support_shape_hw[1])],
            "preview_shape_hw": [int(bundle.preview_shape_hw[0]), int(bundle.preview_shape_hw[1])],
            "support_bbox_canvas_xywh": list(bundle.support_bbox_canvas_xywh) if bundle.support_bbox_canvas_xywh is not None else None,
            "fixed_working_mode": bundle.fixed_working_mode,
            "target_um_per_px_xy": list(bundle.target_um_per_px_xy) if bundle.target_um_per_px_xy is not None else None,
        }
        cached = (bundle.rgb, bundle.labels, info)
        self.step7_fixed_cache[item.label] = cached
        return cached

    def _refresh_step7_projection_to_current_section(self) -> None:
        if self.step7_confocal_projection_raw_u8 is None:
            self.step7_confocal_projection_u8 = None
            self.step7_confocal_projection_mask_u8 = None
            return
        info = dict(self.step7_projection_info or {})
        source_um = info.get("source_um_per_px_xy")
        target_um = None if self.step7_fixed_info is None else self.step7_fixed_info.get("preview_um_per_px_xy")
        projection, scale_info = _resample_projection_to_target_um_per_px(
            self.step7_confocal_projection_raw_u8,
            source_um_per_px_xy=tuple(source_um) if isinstance(source_um, (list, tuple)) and len(source_um) == 2 else None,
            target_um_per_px_xy=tuple(target_um) if isinstance(target_um, (list, tuple)) and len(target_um) == 2 else None,
        )
        self.step7_confocal_projection_u8 = projection
        raw_mask = (
            np.where(np.asarray(self.step7_confocal_projection_mask_raw_u8) > 0, 255, 0).astype(np.uint8)
            if self.step7_confocal_projection_mask_raw_u8 is not None
            else np.where(np.asarray(self.step7_confocal_projection_raw_u8) > 0, 255, 0).astype(np.uint8)
        )
        self.step7_confocal_projection_mask_u8 = _resample_mask_to_target_um_per_px(
            raw_mask,
            source_um_per_px_xy=tuple(source_um) if isinstance(source_um, (list, tuple)) and len(source_um) == 2 else None,
            target_um_per_px_xy=tuple(target_um) if isinstance(target_um, (list, tuple)) and len(target_um) == 2 else None,
        )
        if self.step7_confocal_projection_mask_u8 is not None:
            self.step7_confocal_projection_mask_u8 = np.where(self.step7_confocal_projection_mask_u8 > 0, 255, 0).astype(np.uint8)
        if self.step7_projection_info is not None:
            self.step7_projection_info["scale_to_section_preview"] = scale_info
            self.step7_projection_info["target_um_per_px_xy"] = list(target_um) if isinstance(target_um, (list, tuple)) and len(target_um) == 2 else None
            self.step7_projection_info["raw_projection_shape_hw"] = [
                int(self.step7_confocal_projection_raw_u8.shape[0]),
                int(self.step7_confocal_projection_raw_u8.shape[1]),
            ]
            self.step7_projection_info["scaled_projection_shape_hw"] = [int(projection.shape[0]), int(projection.shape[1])]

    def update_step7_tile_outline_preview(self) -> None:
        return

    def _set_step7_transform_spins(self, tx_px: float, ty_px: float, angle_deg: float, scale: float) -> None:
        for spin, value in (
            (self.step7_tx_spin, tx_px),
            (self.step7_ty_spin, ty_px),
            (self.step7_angle_spin, angle_deg),
            (self.step7_scale_spin, scale),
        ):
            spin.blockSignals(True)
            spin.setValue(float(value))
            spin.blockSignals(False)

    def _append_step7_diagnostic_log(self, line: str) -> None:
        self.step7_diagnostic_log.append(str(line))
        self._update_step7_info_text()

    def start_step7_anchor_mode(self) -> None:
        if self.step7_confocal_projection_u8 is None or self.step7_fixed_rgb is None:
            QMessageBox.warning(self, "Step 7 Anchors", "Generate a projection first, then start Manual Anchor Mode.")
            return
        if hasattr(self.step7_preview_view, "activate_anchor_mode"):
            self.step7_preview_view.activate_anchor_mode()
        self.step7_last_manual_action = "anchor_mode_started"
        self._append_step7_diagnostic_log("Anchor mode: started. Next target A1.")

    def on_step7_diagnostic_state_changed(self, payload: object) -> None:
        data = dict(payload) if isinstance(payload, dict) else {}
        event_name = str(data.get("event") or "")
        if event_name == "lock_toggled":
            self.step7_last_manual_action = f"grid_lock={'on' if bool(data.get('transform_locked')) else 'off'}"
        elif event_name == "opacity_changed":
            self.step7_last_manual_action = f"overlay_opacity={float(data.get('overlay_opacity', 0.85)):.2f}"
        elif event_name in {"target_manual", "target_advanced", "anchor_mode_started"}:
            self.step7_last_manual_action = f"anchor_target={data.get('next_group', 'A')}{data.get('next_index', 1)}"
        elif event_name in {"point_deleted", "point_undone"}:
            self.step7_last_manual_action = f"undo_to_{data.get('next_group', 'A')}{data.get('next_index', 1)}"
        elif event_name == "reset":
            self.step7_last_manual_action = "anchor_reset"
        self._update_step7_info_text()

    def on_step7_diagnostic_point_placed(self, payload: object) -> None:
        data = dict(payload) if isinstance(payload, dict) else {}
        group = str(data.get("group") or "?")
        idx = int(data.get("index") or 0)
        scene_xy = data.get("scene_xy") or [float("nan"), float("nan")]
        if group == "A":
            msg = f"A{idx}: section_scene_xy=({scene_xy[0]:.1f}, {scene_xy[1]:.1f})"
        else:
            raw_xy = data.get("overlay_raw_xy") or [float("nan"), float("nan")]
            disp_xy = data.get("overlay_display_xy") or [float("nan"), float("nan")]
            signal_class = str(data.get("signal_class") or "unknown").replace("_", " ")
            msg = (
                f"B{idx}: section_scene_xy=({scene_xy[0]:.1f}, {scene_xy[1]:.1f}) | "
                f"confocal_display_xy=({disp_xy[0]:.1f}, {disp_xy[1]:.1f}) | "
                f"confocal_raw_xy=({raw_xy[0]:.1f}, {raw_xy[1]:.1f}) | "
                f"B point {signal_class}"
            )
        self._append_step7_diagnostic_log(msg)

    def analyze_step7_landmarks(self) -> None:
        snap = self.step7_preview_view.diagnostic_snapshot()
        points_a = snap.get("points_a_scene", {}) if isinstance(snap, dict) else {}
        points_b = snap.get("points_b_raw", {}) if isinstance(snap, dict) else {}
        pair_ids = sorted({int(k) for k in points_a.keys()} & {int(k) for k in points_b.keys()})
        if len(pair_ids) < 2:
            self._append_step7_diagnostic_log("Analyze: need at least 2 complete A/B pairs.")
            return
        src = np.asarray([points_b[str(idx)] for idx in pair_ids], dtype=np.float64)
        dst = np.asarray([points_a[str(idx)] for idx in pair_ids], dtype=np.float64)
        rigid = _fit_step7_procrustes(src, dst, allow_scale=False)
        similarity = _fit_step7_procrustes(src, dst, allow_scale=True)
        affine = _fit_step7_affine(src, dst) if len(pair_ids) >= 3 else None
        lines = [
            f"Analyze: pairs={pair_ids}",
            (
                "  rigid: "
                f"rms={rigid['rms_px']:.2f}px mean={rigid['mean_px']:.2f}px max={rigid['max_px']:.2f}px "
                f"rot={rigid['rotation_deg']:.2f}deg trans=({rigid['translation_xy'][0]:.2f},{rigid['translation_xy'][1]:.2f})"
            ),
            (
                "  similarity: "
                f"rms={similarity['rms_px']:.2f}px mean={similarity['mean_px']:.2f}px max={similarity['max_px']:.2f}px "
                f"rot={similarity['rotation_deg']:.2f}deg scale={similarity['scale']:.5f}"
            ),
        ]
        diagnosis = "  diagnosis: similarity does not improve enough over rigid yet."
        rigid_rms = float(rigid["rms_px"])
        sim_rms = float(similarity["rms_px"])
        if affine is not None:
            lines.append(
                "  affine: "
                f"rms={affine['rms_px']:.2f}px mean={affine['mean_px']:.2f}px max={affine['max_px']:.2f}px "
                f"scale_x_like={affine['scale_x_like']:.5f} scale_y_like={affine['scale_y_like']:.5f} "
                f"shear_like={affine['shear_like']:.5f}"
            )
            aff_rms = float(affine["rms_px"])
            if sim_rms < rigid_rms * 0.80 and aff_rms >= sim_rms * 0.92:
                diagnosis = "  diagnosis: scale/rotation mismatch dominates; similarity likely enough."
            elif aff_rms < sim_rms * 0.80:
                diagnosis = "  diagnosis: affine gives a clear extra gain; anisotropic scale or shear is likely."
            elif aff_rms < sim_rms * 0.95:
                diagnosis = "  diagnosis: affine helps modestly; weak shear or axis-specific scale is plausible."
            elif aff_rms > sim_rms * 0.98 and sim_rms > rigid_rms * 0.98:
                diagnosis = "  diagnosis: higher-order mismatch remains; if residual vectors stay structured, nonlinear may be needed."
        elif sim_rms < rigid_rms * 0.80:
            diagnosis = "  diagnosis: two-point screen suggests scale/rotation mismatch."
        lines.append(diagnosis)
        for line in lines:
            self._append_step7_diagnostic_log(line)

    def refresh_step7_sections(self) -> None:
        if self.step7_myelin_root is None:
            self.step7_myelin_root = self._default_step4_myelin_root()
        confocal_roi_root, available_confocal_labels = self._step7_available_confocal_section_labels()
        loaded_label = self._current_step7_item().label if (self.step7_fixed_rgb is not None and self._current_step7_item() is not None) else None
        self.step7_section_list.blockSignals(True)
        self.step7_section_list.clear()
        if self.step7_myelin_root is None or not self.step7_myelin_root.exists():
            self.step7_sections = []
            self.step7_root_status.setPlainText("Step 7 myelin root is not set.")
            self.current_step7_section_index = -1
            self.step7_pair_label.setText("No myelin section selected")
            self.step7_fixed_rgb = None
            self.step7_fixed_labels = None
            self.step7_fixed_info = None
            self._reset_step7_session_state(clear_confocal_paths=True, reset_transform=True)
            self._refresh_step8_info()
            self.step7_section_list.blockSignals(False)
            self.update_step7_preview()
            return
        all_myelin_sections = [
            item
            for item in list_workspace_sections(self.step7_myelin_root)
            if item.stain in {"gallyas", "myelin", ""}
        ]
        if available_confocal_labels:
            self.step7_sections = [item for item in all_myelin_sections if str(item.label) in available_confocal_labels]
        else:
            self.step7_sections = all_myelin_sections
        for item in self.step7_sections:
            self.step7_section_list.addItem(item.label)
        status_lines = [
            f"myelin_root: {self.step7_myelin_root}",
            f"confocal_roi_root: {confocal_roi_root or 'not found'}",
            f"confocal_runs_root: {self._step7_runs_root()}",
            f"myelin_sections_shown: {len(self.step7_sections)}",
            f"myelin_sections_total: {len(all_myelin_sections)}",
            f"fixed_cache_entries: {len(self.step7_fixed_cache)}",
        ]
        if available_confocal_labels:
            status_lines.insert(4, f"confocal_sections_with_roi: {len(available_confocal_labels)}")
        else:
            status_lines.insert(4, "confocal_sections_with_roi: unavailable")
        self.step7_root_status.setPlainText(
            "\n".join(status_lines)
        )
        if self.step7_sections:
            matched_idx = next((i for i, item in enumerate(self.step7_sections) if item.label == loaded_label), None)
            if matched_idx is not None:
                self.current_step7_section_index = int(matched_idx)
                self.step7_section_list.setCurrentRow(self.current_step7_section_index)
            else:
                self.current_step7_section_index = -1
                self.step7_section_list.setCurrentRow(-1)
                self.step7_pair_label.setText("Select a myelin section to load Step 7 images")
                self.step7_fixed_rgb = None
                self.step7_fixed_labels = None
                self.step7_fixed_info = None
                self._reset_step7_session_state(clear_confocal_paths=True, reset_transform=True)
                self._refresh_step8_info()
                self._update_step7_info_text()
                self.update_step7_tile_outline_preview()
                self.update_step7_preview()
        else:
            self.current_step7_section_index = -1
            self.step7_pair_label.setText("No myelin section selected")
            self.step7_fixed_rgb = None
            self.step7_fixed_labels = None
            self.step7_fixed_info = None
            self._reset_step7_session_state(clear_confocal_paths=True, reset_transform=True)
            self._refresh_step8_info()
            self._update_step7_info_text()
            self.update_step7_tile_outline_preview()
            self.update_step7_preview()
        self.step7_section_list.blockSignals(False)

    def on_step7_section_changed(self, index: int) -> None:
        if index < 0 or index >= len(self.step7_sections):
            return
        previous_item = self._current_step7_item()
        previous_label = previous_item.label if previous_item is not None else ""
        self.current_step7_section_index = index
        item = self.step7_sections[index]
        fixed_rgb, fixed_labels, fixed_info = self._load_step7_fixed_section(item)
        self.step7_fixed_rgb = fixed_rgb
        self.step7_fixed_labels = fixed_labels
        self.step7_fixed_info = fixed_info
        section_changed = str(previous_label) != str(item.label)
        self._reset_step7_session_state(
            clear_confocal_paths=section_changed,
            clear_loaded_projection=section_changed,
            clear_duplicate_report=section_changed,
            reset_transform=section_changed,
        )
        self._refresh_step8_info()
        self._refresh_step7_projection_to_current_section()
        self.step7_pair_label.setText(f"{index + 1}/{len(self.step7_sections)} | {item.label}")
        self._update_step7_info_text()
        self.update_step7_tile_outline_preview()
        self.update_step7_preview()

    def select_step7_confocal_stack(self) -> None:
        default_dir = str(self._default_step7_confocal_root())
        file_paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Confocal Source(s)",
            default_dir,
            "Confocal stacks (*.tif *.tiff *.czi);;All Files (*)",
        )
        if not file_paths:
            return
        selected = [Path(path) for path in file_paths]
        suffixes = {path.suffix.lower() for path in selected}
        if len(selected) > 1 and ".czi" in suffixes:
            QMessageBox.warning(self, "Step 7 Confocal", "Select either one .czi file or multiple TIFF tiles, not both.")
            return
        self._reset_step7_session_state(clear_confocal_paths=True, reset_transform=True)
        self.step7_confocal_paths = sorted(selected)
        self.step7_confocal_source_mode = "czi_whole" if len(selected) == 1 and selected[0].suffix.lower() == ".czi" else ("multi_tiff_strip" if len(selected) > 1 else "single_tiff")
        self.step7_stack_label.setText(self._describe_step7_confocal_sources())
        if len(self.step7_confocal_paths) > 1:
            self.step7_progress_label.setText("Step 7 progress: checking duplicate z-stacks ...")
            QApplication.processEvents()
            self._compute_step7_duplicate_stack_report()
            self._warn_step7_duplicate_stack_report(context="selected confocal sources")
        self.step7_progress_label.setText("Step 7 progress: idle")
        self._refresh_step8_info()
        self._update_step7_info_text()

    def clear_step7_current_grid(self) -> None:
        if self._step7_any_worker_running():
            QMessageBox.warning(
                self,
                "Step 7 Clear Grid",
                "Wait for the current Step 7 job to finish before clearing the current grid.",
            )
            return
        if (
            not self.step7_confocal_paths
            and self.step7_confocal_projection_u8 is None
            and not self.step7_tile_result_rows
            and not self.step7_accepted_tile_indices
            and not self.step7_frozen_tile_indices
            and not self.step7_hold_tile_indices
            and not self.step7_frontier_tile_indices
        ):
            self.step7_info.append("Step 7 clear grid: nothing to clear.")
            return
        confirm = QMessageBox.question(
            self,
            "Step 7 Clear Grid",
            "Clear the current confocal grid, projection, tile states, and coarse transform for this myelin section?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        if confirm != QMessageBox.StandardButton.Yes:
            return
        self._reset_step7_session_state(
            clear_confocal_paths=True,
            clear_loaded_projection=True,
            clear_duplicate_report=True,
            reset_transform=True,
        )
        self._refresh_step8_info()
        self._update_step7_info_text()
        self.update_step7_tile_outline_preview()
        self.update_step7_preview()
        item = self._current_step7_item()
        if item is not None:
            self.step7_pair_label.setText(f"{self.current_step7_section_index + 1}/{len(self.step7_sections)} | {item.label}")
        self.step7_info.append("Step 7 clear grid: current confocal grid/session state cleared. Select new source(s) to start the next grid.")

    def generate_step7_projection(self) -> None:
        if not self.step7_confocal_paths:
            QMessageBox.warning(self, "Step 7 Confocal", "Select one or more confocal sources first.")
            return
        if self.step7_duplicate_stack_report is None and len(self.step7_confocal_paths) > 1:
            self.step7_progress_label.setText("Step 7 progress: checking duplicate z-stacks ...")
            QApplication.processEvents()
            self._compute_step7_duplicate_stack_report()
            self.step7_progress_label.setText("Step 7 progress: idle")
        self._warn_step7_duplicate_stack_report(context="generate projection")
        channel_index = int(self.step7_channel_spin.value())
        mode = str(self.step7_projection_mode_combo.currentData() or "focus")
        self.step7_progress_label.setText("Step 7 progress: generating projection ...")
        QApplication.processEvents()
        bundle = load_confocal_projection(
            self.step7_confocal_paths,
            mode=mode,
            channel_index=channel_index,
            nominal_overlap_fraction=float(self.step7_overlap_spin.value()),
        )
        self.step7_channel_spin.setMaximum(max(0, int(bundle.channel_count) - 1))
        self.step7_confocal_projection_raw_u8 = bundle.projection_u8
        self.step7_confocal_projection_mask_raw_u8 = np.where(bundle.projection_u8 > 0, 255, 0).astype(np.uint8)
        self.step7_confocal_source_mode = bundle.source_mode
        self.step7_projection_info = {
            "source_mode": bundle.source_mode,
            "source_paths": bundle.source_paths,
            "source_shapes": bundle.source_shapes,
            "stitch_info": bundle.stitch_info,
            "source_um_per_px_xy": list(bundle.physical_um_per_px_xy) if bundle.physical_um_per_px_xy is not None else None,
            "duplicate_stack_report": self.step7_duplicate_stack_report if isinstance(self.step7_duplicate_stack_report, dict) else None,
        }
        self._reset_step7_session_state(
            clear_confocal_paths=False,
            clear_loaded_projection=False,
            clear_duplicate_report=False,
            reset_transform=False,
        )
        self.step7_confocal_paths = [Path(p) for p in list(bundle.source_paths or self.step7_confocal_paths)]
        self.step7_stack_label.setText(self._describe_step7_confocal_sources())
        self._refresh_step8_info()
        self._refresh_step7_projection_to_current_section()
        self.step7_progress_label.setText("Step 7 progress: projection ready")
        self._update_step7_info_text()
        if bundle.source_mode in {"multi_tiff_strip", "multi_tiff_grid"}:
            self.step7_info.append(f"Stitch info: {json.dumps(bundle.stitch_info, ensure_ascii=True)}")
        self.update_step7_tile_outline_preview()
        self.update_step7_preview()

    def on_step7_preview_transform_edited(self, tx_px: float, ty_px: float, angle_deg: float, scale: float) -> None:
        self._set_step7_transform_spins(tx_px, ty_px, angle_deg, scale)
        self.step7_last_manual_action = (
            f"manual_transform_updated -> tx={float(tx_px):.1f} "
            f"ty={float(ty_px):.1f} angle={float(angle_deg):.1f} scale={float(scale):.5f}"
        )
        self._update_step7_info_text()

    def on_step7_tile_selection_changed(self, payload: object) -> None:
        data = dict(payload) if isinstance(payload, dict) else {}
        label = str(data.get("selected_tile_label") or "").strip()
        if label:
            self.step7_last_manual_action = f"tile_selected={label}"
        self._update_step7_tile_qc_display()
        self._update_step7_info_text()

    def _current_step7_tile_defs(self) -> list[dict[str, object]]:
        if self.step7_projection_info is None or self.step7_confocal_projection_u8 is None:
            return []
        stitch_info = self.step7_projection_info.get("stitch_info") if isinstance(self.step7_projection_info, dict) else None
        raw_shape_hw = self.step7_projection_info.get("raw_projection_shape_hw") if isinstance(self.step7_projection_info, dict) else None
        if not isinstance(raw_shape_hw, list) or len(raw_shape_hw) != 2:
            raw_shape_hw = list(self.step7_confocal_projection_u8.shape[:2])
        return build_confocal_tile_defs(
            stitch_info if isinstance(stitch_info, dict) else {},
            raw_shape_hw=(int(raw_shape_hw[0]), int(raw_shape_hw[1])),
            scaled_shape_hw=self.step7_confocal_projection_u8.shape[:2],
            flip_lr=bool(self.step7_flip_lr_check.isChecked()),
            flip_ud=bool(self.step7_flip_ud_check.isChecked()),
        )

    def _build_step7_auto_scale_summary_lines(self, data: dict[str, object]) -> list[str]:
        best = data.get("best_by_composite") if isinstance(data.get("best_by_composite"), dict) else {}
        best_final = data.get("best_by_mean_final_cc") if isinstance(data.get("best_by_mean_final_cc"), dict) else {}
        best_right = data.get("best_by_rightmost_abs_dx") if isinstance(data.get("best_by_rightmost_abs_dx"), dict) else {}
        manual_init = data.get("manual_init") if isinstance(data.get("manual_init"), dict) else {}
        run_dir = str(data.get("run_dir") or "").strip()
        summary_rows = data.get("summary_rows") if isinstance(data.get("summary_rows"), list) else []
        sweep = data.get("sweep") if isinstance(data.get("sweep"), dict) else {}
        chosen_scale = float(data.get("chosen_scale") or best.get("scale") or self.step7_scale_spin.value())
        initial_scale = float(manual_init.get("scale") or chosen_scale)
        applied_scale_changed = not np.isclose(float(initial_scale), float(chosen_scale), atol=5e-6)
        sampled_count = int(sweep.get("tile_count_sampled") or 0)
        total_count = int(sweep.get("tile_count_total") or 0)
        sampled_labels = [str(v) for v in list(sweep.get("sampled_tile_labels") or [])]
        lines = [
            "Last Auto Scale Sweep:",
            f"- initial_scale: {initial_scale:.5f}",
            f"- chosen_scale: {chosen_scale:.5f}",
            f"- applied_scale_changed: {'yes' if applied_scale_changed else 'no'}",
            f"- candidate_scales: {len(summary_rows)}",
        ]
        if sampled_count > 0:
            lines.append(f"- sampled_tiles: {sampled_count}/{max(sampled_count, total_count)}")
            if sampled_labels:
                lines.append("- sample_labels: " + ", ".join(sampled_labels))
        if best:
            lines.append(
                "- best_composite: "
                f"scale={float(best.get('scale', float('nan'))):.5f} "
                f"| mean_final_CC={self._fmt_step7_metric(best.get('mean_final_cc'))} "
                f"| mean|dx|={self._fmt_step7_metric(best.get('mean_abs_dx'))} "
                f"| rightmost|dx|={self._fmt_step7_metric(best.get('rightmost_abs_dx'))}"
            )
        if best_final:
            lines.append(
                "- best_mean_final_CC: "
                f"scale={float(best_final.get('scale', float('nan'))):.5f} "
                f"| mean_final_CC={self._fmt_step7_metric(best_final.get('mean_final_cc'))}"
            )
        if best_right:
            lines.append(
                "- best_right_flatten: "
                f"scale={float(best_right.get('scale', float('nan'))):.5f} "
                f"| rightmost|dx|={self._fmt_step7_metric(best_right.get('rightmost_abs_dx'))}"
            )
        if run_dir:
            lines.append(f"- run_dir: {run_dir}")
        return lines

    def _build_step7_seed_screen_summary_lines(self, data: dict[str, object]) -> list[str]:
        top = data.get("top_seed_candidates") if isinstance(data.get("top_seed_candidates"), list) else []
        run_dir = str(data.get("run_dir") or "").strip()
        rows = data.get("rows") if isinstance(data.get("rows"), list) else []
        accepted = sum(1 for row in rows if str(row.get("proposal_gate") or "") == "accepted")
        rejected = max(0, len(rows) - accepted)
        lines = [
            "Last Seed Screening:",
            f"- registration_input_profile: {str(data.get('registration_input_profile') or STEP7_REGISTRATION_INPUT_PROFILE)}",
            f"- candidate_count: {len(rows) if rows else len(self._current_step7_tile_defs())}",
            f"- accepted_shift_updates: {accepted} | kept_current: {rejected}",
        ]
        if top:
            best = top[0]
            lines.append(
                "- best_seed: "
                f"{str(best.get('label') or 'unknown')} "
                f"| current_CC={self._fmt_step7_metric(best.get('current_cc'))} "
                f"| shift=({int(best.get('best_shift_dx_px') or 0)},{int(best.get('best_shift_dy_px') or 0)}) "
                f"| score={self._fmt_step7_metric(best.get('seed_score'))}"
            )
            lines.append(
                "- top_candidates: "
                + "; ".join(
                    f"{str(row.get('label') or 'unknown')} score={self._fmt_step7_metric(row.get('seed_score'))}"
                    for row in top[:5]
                )
            )
        if run_dir:
            lines.append(f"- run_dir: {run_dir}")
        return lines

    def _build_step7_frontier_summary_lines(self, data: dict[str, object]) -> list[str]:
        top = data.get("top_frontier_candidates") if isinstance(data.get("top_frontier_candidates"), list) else []
        graph = data.get("graph_state") if isinstance(data.get("graph_state"), dict) else {}
        run_dir = str(data.get("run_dir") or "").strip()
        frontier_rows = data.get("rows") if isinstance(data.get("rows"), list) else []
        lines = [
            "Last Frontier Propagation:",
            f"- registration_input_profile: {str(data.get('registration_input_profile') or STEP7_REGISTRATION_INPUT_PROFILE)}",
            f"- solved_tiles: {graph.get('solved_tile_indices') or []}",
            f"- frontier_candidate_count: {len(frontier_rows)}",
            f"- residual_model: {str(graph.get('residual_model') or 'bounded_translation')}",
        ]
        if top:
            best = top[0]
            lines.append(
                "- best_frontier: "
                f"{str(best.get('label') or 'unknown')} "
                f"| confidence={self._fmt_step7_metric(best.get('frontier_confidence'))} "
                f"| CC {self._fmt_step7_metric(best.get('current_cc'))}->{self._fmt_step7_metric(best.get('shifted_cc'))} "
                f"| prior=({int(best.get('prior_shift_dx_px') or 0)},{int(best.get('prior_shift_dy_px') or 0)}) "
                f"| shift=({int(best.get('best_shift_dx_px') or 0)},{int(best.get('best_shift_dy_px') or 0)})"
            )
            lines.append(
                "- top_frontier: "
                + "; ".join(
                    f"{str(row.get('label') or 'unknown')} conf={self._fmt_step7_metric(row.get('frontier_confidence'))}"
                    for row in top[:5]
                )
            )
        if run_dir:
            lines.append(f"- run_dir: {run_dir}")
        return lines

    def _sync_step7_tile_state_sets_from_rows(self) -> None:
        accepted: set[int] = set()
        hold: set[int] = set()
        frontier: set[int] = set()
        for tile_index, row in self.step7_tile_result_rows.items():
            if not isinstance(row, dict):
                continue
            state = str(row.get("tile_state") or "").strip().lower()
            idx = int(tile_index)
            if state == "frozen":
                self.step7_frozen_tile_indices.add(idx)
                accepted.add(idx)
            elif state == "accepted":
                accepted.add(idx)
            elif state == "hold":
                hold.add(idx)
            elif state == "frontier":
                frontier.add(idx)
        accepted.difference_update(self.step7_frozen_tile_indices)
        hold.difference_update(self.step7_frozen_tile_indices | accepted)
        frontier.difference_update(self.step7_frozen_tile_indices | accepted | hold)
        self.step7_accepted_tile_indices = accepted
        self.step7_hold_tile_indices = hold
        self.step7_frontier_tile_indices = frontier

    def _set_step7_cached_tile_state(self, tile_index: int, state: str) -> None:
        row = self.step7_tile_result_rows.get(int(tile_index))
        if not isinstance(row, dict):
            return
        row["tile_state"] = str(state)
        if str(state) == "frozen":
            self.step7_frozen_tile_indices.add(int(tile_index))
            self.step7_accepted_tile_indices.add(int(tile_index))
            self.step7_hold_tile_indices.discard(int(tile_index))
        elif str(state) == "accepted":
            self.step7_frozen_tile_indices.discard(int(tile_index))
            self.step7_accepted_tile_indices.add(int(tile_index))
            self.step7_hold_tile_indices.discard(int(tile_index))
        elif str(state) == "hold":
            self.step7_frozen_tile_indices.discard(int(tile_index))
            self.step7_accepted_tile_indices.discard(int(tile_index))
            self.step7_hold_tile_indices.add(int(tile_index))
        else:
            self.step7_accepted_tile_indices.discard(int(tile_index))
            self.step7_hold_tile_indices.discard(int(tile_index))

    def _describe_step7_accepted_tiles(self) -> str:
        accepted_only = set(self.step7_accepted_tile_indices) - set(self.step7_frozen_tile_indices)
        if not accepted_only:
            return "none"
        defs = {int(row.get("tile_index", -1)): str(row.get("label") or f"T{int(row.get('tile_index', -1)):02d}") for row in self._current_step7_tile_defs()}
        labels = [defs.get(int(idx), f"T{int(idx):02d}") for idx in sorted(accepted_only)]
        return "[" + ", ".join(labels) + "]"

    def _describe_step7_hold_tiles(self) -> str:
        if not self.step7_hold_tile_indices:
            return "none"
        defs = {int(row.get("tile_index", -1)): str(row.get("label") or f"T{int(row.get('tile_index', -1)):02d}") for row in self._current_step7_tile_defs()}
        labels = [defs.get(int(idx), f"T{int(idx):02d}") for idx in sorted(self.step7_hold_tile_indices)]
        return "[" + ", ".join(labels) + "]"

    def _describe_step7_frozen_tiles(self) -> str:
        if not self.step7_frozen_tile_indices:
            return "none"
        labels: list[str] = []
        defs = {int(row.get("tile_index", -1)): str(row.get("label") or f"T{int(row.get('tile_index', -1)):02d}") for row in self._current_step7_tile_defs()}
        for idx in sorted(self.step7_frozen_tile_indices):
            labels.append(defs.get(int(idx), f"T{int(idx):02d}"))
        return "[" + ", ".join(labels) + "]"

    def _describe_step7_frontier_tiles(self) -> str:
        if not self.step7_frontier_tile_indices:
            return "none"
        defs = {int(row.get("tile_index", -1)): str(row.get("label") or f"T{int(row.get('tile_index', -1)):02d}") for row in self._current_step7_tile_defs()}
        labels = [defs.get(int(idx), f"T{int(idx):02d}") for idx in sorted(self.step7_frontier_tile_indices)]
        return "[" + ", ".join(labels) + "]"

    def _step7_result_row_for_tile(self, tile_index: int | None) -> dict[str, object] | None:
        if tile_index is None:
            return None
        cached = self.step7_tile_result_rows.get(int(tile_index))
        if isinstance(cached, dict):
            return cached
        for row in self.step7_last_frontier_rows:
            if int(row.get("tile_index", -1)) == int(tile_index):
                return row
        for row in self.step7_last_seed_screen_rows:
            if int(row.get("tile_index", -1)) == int(tile_index):
                return row
        return None

    def _step7_tile_order(self) -> list[int]:
        defs = self._current_step7_tile_defs()
        if defs:
            return [int(row["tile_index"]) for row in defs]
        return [int(row.get("tile_index", -1)) for row in sorted(self.step7_last_seed_screen_rows, key=lambda r: (int(r.get("row_display", 0)), int(r.get("col_display", 0))))]

    def _step7_unfrozen_tile_order(self) -> list[int]:
        return [int(idx) for idx in self._step7_tile_order() if int(idx) not in self.step7_frozen_tile_indices]

    def _selected_step7_tile_indices_from_snapshot(self) -> list[int]:
        snap = self.step7_preview_view.diagnostic_snapshot() if hasattr(self, "step7_preview_view") else {}
        selected = snap.get("selected_tile_indices") if isinstance(snap, dict) else None
        if isinstance(selected, list) and selected:
            return [int(v) for v in selected]
        selected_idx = snap.get("selected_tile_index") if isinstance(snap, dict) else None
        if selected_idx is None:
            return []
        return [int(selected_idx)]

    def _step7_loaded_tile_count(self) -> int:
        tile_defs = self._current_step7_tile_defs()
        if tile_defs:
            return int(len(tile_defs))
        return int(len(self.step7_tile_result_rows))

    def _update_step7_frozen_count_label(self) -> None:
        if not hasattr(self, "step7_frozen_count_label"):
            return
        frozen_count = int(len(self.step7_frozen_tile_indices))
        total_count = int(self._step7_loaded_tile_count())
        self.step7_frozen_count_label.setText(f"Frozen: {frozen_count}/{total_count}")

    def _clear_step7_preview_overlay_state(self, *, reset_alignment: bool) -> None:
        if not hasattr(self, "step7_preview_view"):
            return
        blocker = QSignalBlocker(self.step7_preview_view)
        try:
            self.step7_preview_view.set_overlay_gray(None)
            self.step7_preview_view.set_overlay_tiles([])
            self.step7_preview_view.set_selected_tile(None)
            self.step7_preview_view.set_frozen_tiles(set())
            self.step7_preview_view.set_accepted_tiles(set())
            self.step7_preview_view.set_hold_tiles(set())
            self.step7_preview_view.set_frontier_tiles(set())
            if reset_alignment:
                self.step7_preview_view.set_alignment(0.0, 0.0, 0.0, 1.0)
        finally:
            del blocker
        self._update_step7_tile_qc_display()

    def _refresh_step7_preview_tile_states_only(self) -> None:
        if not hasattr(self, "step7_preview_view"):
            return
        blocker = QSignalBlocker(self.step7_preview_view)
        try:
            self.step7_preview_view.set_frozen_tiles(self.step7_frozen_tile_indices)
            self.step7_preview_view.set_accepted_tiles(self.step7_accepted_tile_indices)
            self.step7_preview_view.set_hold_tiles(self.step7_hold_tile_indices)
            self.step7_preview_view.set_frontier_tiles(self.step7_frontier_tile_indices)
        finally:
            del blocker
        self._update_step7_tile_qc_display()

    def _set_step7_selected_tile(self, tile_index: int | None) -> None:
        blocker = QSignalBlocker(self.step7_preview_view)
        try:
            self.step7_preview_view.set_selected_tile(tile_index)
        finally:
            del blocker
        self._update_step7_tile_qc_display()

    def _step7_qc_panel_rgb(self, panel: np.ndarray) -> np.ndarray:
        arr = np.asarray(panel)
        if arr.ndim == 2:
            gray = arr.astype(np.uint8)
            return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        rgb = arr[..., :3]
        if np.issubdtype(rgb.dtype, np.floating):
            rgb = np.clip(np.round(rgb * (255.0 if float(np.nanmax(rgb)) <= 1.0 else 1.0)), 0, 255).astype(np.uint8)
        else:
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        return rgb

    def _compose_step7_single_tile_qc_rgb(self, row: dict[str, object]) -> np.ndarray:
        panels = []
        for key in ("moving", "fixed", "overlay", "heatmap"):
            panel = row.get(key)
            if isinstance(panel, np.ndarray):
                panels.append(self._step7_qc_panel_rgb(np.asarray(panel)))
        if not panels:
            return np.full((420, 900, 3), 245, dtype=np.uint8)
        panel_h = max(int(panel.shape[0]) for panel in panels)
        panel_w = max(int(panel.shape[1]) for panel in panels)
        norm_panels = [
            panel if panel.shape[:2] == (panel_h, panel_w) else cv2.resize(panel, (panel_w, panel_h), interpolation=cv2.INTER_LINEAR)
            for panel in panels
        ]
        pad = 16
        title_h = 58
        note_h = 52
        canvas = np.full((title_h + note_h + panel_h + pad * 2, pad * 5 + panel_w * 4, 3), 246, dtype=np.uint8)
        font = cv2.FONT_HERSHEY_SIMPLEX
        label = str(row.get("label") or "Tile")
        profile = str(row.get("registration_profile") or STEP7_REGISTRATION_INPUT_PROFILE)
        gate = str(row.get("proposal_gate") or "n/a")
        tile_state = str(row.get("tile_state") or "n/a")
        rank_bits: list[str] = []
        if row.get("rank") is not None:
            rank_bits.append(f"rank={int(row.get('rank') or 0)}")
        if row.get("frontier_confidence") is not None:
            rank_bits.append(f"frontier={self._fmt_step7_metric(row.get('frontier_confidence'))}")
        elif row.get("seed_score") is not None:
            rank_bits.append(f"score={self._fmt_step7_metric(row.get('seed_score'))}")
        metric_note = (
            f"CC {self._fmt_step7_metric(row.get('current_cc'))}->{self._fmt_step7_metric(row.get('shifted_cc'))} | "
            f"pred=({int(row.get('prior_shift_dx_px', row.get('pred_dx_px', 0)) or 0)},{int(row.get('prior_shift_dy_px', row.get('pred_dy_px', 0)) or 0)}) | "
            f"final=({int(row.get('best_shift_dx_px') or 0)},{int(row.get('best_shift_dy_px') or 0)}) | "
            f"candidate CC={self._fmt_step7_metric(row.get('candidate_shifted_cc', row.get('shifted_cc')))} | "
            f"state={tile_state} | gate={gate}"
        )
        if rank_bits:
            metric_note += " | " + " | ".join(rank_bits[:2])
        metric_note += f" | profile={profile}"
        cv2.putText(canvas, label, (pad, 24), font, 0.72, (22, 22, 22), 2, cv2.LINE_AA)
        cv2.putText(canvas, metric_note[:220], (pad, 48), font, 0.5, (70, 70, 70), 1, cv2.LINE_AA)
        titles_raw = row.get("col_titles")
        titles = tuple(str(v) for v in list(titles_raw)[:4]) if isinstance(titles_raw, (list, tuple)) and titles_raw else (
            "Raw overlay current",
            "Raw overlay shifted",
            "Processed overlay current",
            "Processed overlay shifted",
        )
        y0 = title_h + note_h
        for i, panel in enumerate(norm_panels):
            x0 = pad + i * (panel_w + pad)
            canvas[y0 : y0 + panel_h, x0 : x0 + panel_w] = panel
            cv2.rectangle(canvas, (x0, y0), (x0 + panel_w, y0 + panel_h), (188, 188, 188), 1)
            if i < len(titles):
                cv2.putText(canvas, titles[i], (x0 + 6, y0 - 8), font, 0.5, (50, 50, 50), 1, cv2.LINE_AA)
        return canvas

    def _update_step7_tile_controls(self) -> None:
        snap = self.step7_preview_view.diagnostic_snapshot() if hasattr(self, "step7_preview_view") else {}
        selected_idx = snap.get("selected_tile_index") if isinstance(snap, dict) else None
        selected_label = snap.get("selected_tile_label") if isinstance(snap, dict) else None
        selected_indices = [int(v) for v in list(snap.get("selected_tile_indices") or [])] if isinstance(snap, dict) else []
        selected_labels = [str(v) for v in list(snap.get("selected_tile_labels") or [])] if isinstance(snap, dict) else []
        frozen = bool(selected_idx in self.step7_frozen_tile_indices) if selected_idx is not None else False
        accepted = bool(selected_idx in self.step7_accepted_tile_indices) if selected_idx is not None else False
        hold = bool(selected_idx in self.step7_hold_tile_indices) if selected_idx is not None else False
        frontier = bool(selected_idx in self.step7_frontier_tile_indices) if selected_idx is not None else False
        all_selected_frozen = bool(selected_indices) and all(int(idx) in self.step7_frozen_tile_indices for idx in selected_indices)
        self._update_step7_frozen_count_label()
        self.step7_tile_prev_button.setEnabled(bool(self._step7_unfrozen_tile_order()))
        self.step7_tile_next_button.setEnabled(bool(self._step7_unfrozen_tile_order()))
        self.step7_tile_accept_button.setEnabled(selected_idx is not None)
        self.step7_tile_hold_button.setEnabled(selected_idx is not None)
        self.step7_tile_freeze_button.setEnabled(bool(selected_indices))
        if len(selected_indices) > 1:
            self.step7_tile_freeze_button.setText(
                f"{'Unfreeze' if all_selected_frozen else 'Freeze'} Selected ({len(selected_indices)})"
            )
        else:
            self.step7_tile_freeze_button.setText("Unfreeze Tile" if frozen else "Freeze Tile")
        if selected_idx is None:
            self.step7_tile_status_label.setText("No tile selected")
        else:
            status = "frozen" if frozen else ("accepted" if accepted else ("hold" if hold else ("frontier" if frontier else "unseen")))
            if len(selected_indices) > 1:
                preview = ", ".join(selected_labels[:3]) if selected_labels else ", ".join(f"T{int(idx):02d}" for idx in selected_indices[:3])
                if len(selected_indices) > 3:
                    preview += f", +{len(selected_indices) - 3} more"
                self.step7_tile_status_label.setText(f"{len(selected_indices)} selected | primary={selected_label or f'T{int(selected_idx):02d}'} | {status} | {preview}")
            else:
                self.step7_tile_status_label.setText(f"{selected_label or f'T{int(selected_idx):02d}'} | {status}")

    def _update_step7_tile_qc_display(self) -> None:
        snap = self.step7_preview_view.diagnostic_snapshot() if hasattr(self, "step7_preview_view") else {}
        selected_idx = snap.get("selected_tile_index") if isinstance(snap, dict) else None
        row = self._step7_result_row_for_tile(selected_idx)
        self._update_step7_tile_controls()
        if row is None:
            self.step7_storyboard_label.setText("No selected tile QC yet")
            self.step7_storyboard_label.setPixmap(QPixmap())
            return
        rgb = self._compose_step7_single_tile_qc_rgb(row)
        self._set_rgb_image_label(self.step7_storyboard_label, rgb, "No selected tile QC yet")

    def select_prev_step7_tile(self) -> None:
        order = self._step7_unfrozen_tile_order()
        if not order:
            return
        snap = self.step7_preview_view.diagnostic_snapshot()
        current = snap.get("selected_tile_index") if isinstance(snap, dict) else None
        if current not in order:
            self._set_step7_selected_tile(order[-1])
            return
        idx = order.index(int(current))
        self._set_step7_selected_tile(order[(idx - 1) % len(order)])

    def select_next_step7_tile(self) -> None:
        order = self._step7_unfrozen_tile_order()
        if not order:
            return
        snap = self.step7_preview_view.diagnostic_snapshot()
        current = snap.get("selected_tile_index") if isinstance(snap, dict) else None
        if current not in order:
            self._set_step7_selected_tile(order[0])
            return
        idx = order.index(int(current))
        self._set_step7_selected_tile(order[(idx + 1) % len(order)])

    def accept_step7_selected_tile(self) -> None:
        snap = self.step7_preview_view.diagnostic_snapshot()
        selected_idx = snap.get("selected_tile_index") if isinstance(snap, dict) else None
        if selected_idx is None:
            return
        idx = int(selected_idx)
        self._set_step7_cached_tile_state(idx, "accepted")
        self.step7_frontier_tile_indices.discard(idx)
        self.step7_last_manual_action = f"tile_accepted={snap.get('selected_tile_label') or f'T{idx:02d}'}"
        self._refresh_step7_preview_tile_states_only()
        self._update_step7_info_text()

    def hold_step7_selected_tile(self) -> None:
        snap = self.step7_preview_view.diagnostic_snapshot()
        selected_idx = snap.get("selected_tile_index") if isinstance(snap, dict) else None
        if selected_idx is None:
            return
        idx = int(selected_idx)
        self._set_step7_cached_tile_state(idx, "hold")
        self.step7_frontier_tile_indices.discard(idx)
        self.step7_last_manual_action = f"tile_hold={snap.get('selected_tile_label') or f'T{idx:02d}'}"
        self._refresh_step7_preview_tile_states_only()
        self._update_step7_info_text()

    def toggle_step7_selected_tile_frozen(self) -> None:
        snap = self.step7_preview_view.diagnostic_snapshot()
        selected_indices = self._selected_step7_tile_indices_from_snapshot()
        if not selected_indices:
            return
        selected_labels = [str(v) for v in list(snap.get("selected_tile_labels") or [])] if isinstance(snap, dict) else []
        all_selected_frozen = all(int(idx) in self.step7_frozen_tile_indices for idx in selected_indices)
        if all_selected_frozen:
            for idx in selected_indices:
                self.step7_frozen_tile_indices.discard(int(idx))
                self._set_step7_cached_tile_state(int(idx), "accepted")
            action_labels = selected_labels or [f"T{int(idx):02d}" for idx in selected_indices]
            self.step7_last_manual_action = f"tiles_unfrozen={','.join(action_labels)}"
        else:
            for idx in selected_indices:
                self.step7_frozen_tile_indices.add(int(idx))
                self._set_step7_cached_tile_state(int(idx), "frozen")
                self.step7_frontier_tile_indices.discard(int(idx))
            action_labels = selected_labels or [f"T{int(idx):02d}" for idx in selected_indices]
            self.step7_last_manual_action = f"tiles_frozen={','.join(action_labels)}"
        self._refresh_step7_preview_tile_states_only()
        self._update_step7_info_text()

    def update_step7_preview(self, *, preserve_view: bool = False) -> None:
        view_state = None
        if preserve_view and self.step7_fixed_rgb is not None and hasattr(self, "step7_preview_view"):
            view_state = self.step7_preview_view.capture_view_state()
        if self.step7_fixed_rgb is None or self.step7_fixed_labels is None:
            self.step7_preview_view.clear_all()
            self._update_step7_frozen_count_label()
            return
        blocker = QSignalBlocker(self.step7_preview_view)
        try:
            self.step7_preview_view.set_fixed_rgb(self.step7_fixed_rgb)
            display_projection = None if self.step7_confocal_projection_u8 is None else _invert_confocal_u8(self.step7_confocal_projection_u8)
            self.step7_preview_view.set_overlay_gray(
                display_projection,
                alpha_source_u8=self.step7_confocal_projection_u8,
                flip_lr=bool(self.step7_flip_lr_check.isChecked()),
                flip_ud=bool(self.step7_flip_ud_check.isChecked()),
            )
            self.step7_preview_view.set_overlay_tiles(self._current_step7_tile_defs())
            self.step7_preview_view.set_frozen_tiles(self.step7_frozen_tile_indices)
            self.step7_preview_view.set_accepted_tiles(self.step7_accepted_tile_indices)
            self.step7_preview_view.set_hold_tiles(self.step7_hold_tile_indices)
            self.step7_preview_view.set_frontier_tiles(self.step7_frontier_tile_indices)
            self.step7_preview_view.set_alignment(
                float(self.step7_tx_spin.value()),
                float(self.step7_ty_spin.value()),
                float(self.step7_angle_spin.value()),
                float(self.step7_scale_spin.value()),
            )
        finally:
            del blocker
        if view_state is not None:
            self.step7_preview_view.restore_view_state(view_state)
        self._update_step7_tile_qc_display()

    def on_step7_flip_changed(self) -> None:
        self.step7_last_manual_action = (
            f"flip_changed -> LR={bool(self.step7_flip_lr_check.isChecked())}, "
            f"UD={bool(self.step7_flip_ud_check.isChecked())}"
        )
        self.update_step7_preview()
        self._update_step7_info_text()

    def _reset_step7_progress_tracking(self, mode: str) -> None:
        self.step7_progress_state = {
            "mode": str(mode),
            "seed_total": 0,
            "seed_done": set(),
            "solved_total": 0,
            "solved_done": set(),
            "frontier_total": 0,
            "frontier_done": set(),
            "refresh_total": 0,
            "refresh_done": set(),
            "active_tiles": OrderedDict(),
        }
        self.step7_progress_bar.setValue(0)
        self.step7_progress_detail_label.setText("Active tiles: none")

    def _set_step7_active_tile_status(self, tile_label: str, status: str) -> None:
        if not self.step7_progress_state:
            return
        active = self.step7_progress_state.get("active_tiles")
        if not isinstance(active, OrderedDict):
            return
        key = str(tile_label).strip()
        if not key:
            return
        active[key] = str(status).strip()
        if len(active) > 8:
            active.popitem(last=False)

    def _remove_step7_active_tile(self, tile_label: str | None) -> None:
        if not self.step7_progress_state or not tile_label:
            return
        active = self.step7_progress_state.get("active_tiles")
        if isinstance(active, OrderedDict):
            active.pop(str(tile_label), None)

    def _update_step7_progress_detail_text(self, fallback: str = "") -> None:
        if not self.step7_progress_state:
            self.step7_progress_detail_label.setText("Active tiles: none")
            return
        active = self.step7_progress_state.get("active_tiles")
        if isinstance(active, OrderedDict) and active:
            preview = [f"{label} {status}".strip() for label, status in list(active.items())[:6]]
            extra = max(0, len(active) - len(preview))
            suffix = f" | +{extra} more" if extra > 0 else ""
            self.step7_progress_detail_label.setText("Active tiles: " + " | ".join(preview) + suffix)
            return
        self.step7_progress_detail_label.setText(fallback or "Active tiles: none")

    def _handle_step7_auto_scale_progress(self, data: dict[str, object]) -> None:
        if not self.step7_progress_state or str(self.step7_progress_state.get("mode")) != "auto_scale":
            self._reset_step7_progress_tracking("auto_scale")
        state = self.step7_progress_state if isinstance(self.step7_progress_state, dict) else {}
        stage = str(data.get("stage") or "running")
        tile_label = str(data.get("tile_label") or "").strip()
        total_units = int(data.get("total_units") or state.get("auto_scale_total_units") or 0)
        if total_units > 0:
            state["auto_scale_total_units"] = int(total_units)
        scale_count = int(data.get("scale_count") or state.get("auto_scale_scale_count") or 0)
        if scale_count > 0:
            state["auto_scale_scale_count"] = int(scale_count)
        tile_count = int(data.get("tile_count") or data.get("total_items") or state.get("auto_scale_tile_count") or 0)
        if tile_count > 0:
            state["auto_scale_tile_count"] = int(tile_count)
        done_units = state.get("auto_scale_done_units")
        if not isinstance(done_units, set):
            done_units = set()
            state["auto_scale_done_units"] = done_units

        if stage in {"coarse_eval", "refine_eval", "candidate_eval"}:
            step_idx = int(data.get("candidate_index") or 0)
            step_count = int(data.get("candidate_count") or 0)
            phase_name = "coarse" if stage == "coarse_eval" else ("refine" if stage == "refine_eval" else "eval")
            detail = f"{phase_name} {step_idx}/{max(1, step_count)}"
            self._set_step7_active_tile_status(tile_label, detail)
        elif stage == "tile_done":
            scale_index = int(data.get("scale_index") or 0)
            tile_index = int(data.get("tile_index") or -1)
            if scale_index > 0 and tile_index >= 0:
                done_units.add((int(scale_index), int(tile_index)))
            self._remove_step7_active_tile(tile_label)

        percent = self.step7_progress_bar.value()
        done_units_override = data.get("done_units_count")
        done_count = int(done_units_override) if done_units_override is not None else len(done_units)
        unit_total = int(state.get("auto_scale_total_units") or 0)
        current_scale_index = int(data.get("scale_index") or 0)
        current_scale_total = int(state.get("auto_scale_scale_count") or 0)
        current_tile_total = int(state.get("auto_scale_tile_count") or 0)

        if stage == "setup":
            percent = max(percent, 1)
            self.step7_progress_bar.setValue(percent)
            self.step7_progress_label.setText(
                f"Step 7 progress: auto_scale | {percent}% | setup | scales={current_scale_total} tiles={current_tile_total}"
            )
            self._update_step7_progress_detail_text("Active tiles: waiting for worker threads")
            return
        if stage == "scale_setup":
            base_percent = int(round(92.0 * (float(max(0, current_scale_index - 1) * max(1, current_tile_total)) / float(max(1, unit_total))))) if unit_total > 0 else 0
            percent = max(percent, base_percent)
            self.step7_progress_bar.setValue(percent)
            self.step7_progress_label.setText(
                f"Step 7 progress: auto_scale | {percent}% | scale {current_scale_index}/{max(1, current_scale_total)} | {float(data.get('scale') or 0.0):.5f}"
            )
            self._update_step7_progress_detail_text("Active tiles: waiting for scale worker threads")
            return
        if stage in {"coarse_eval", "refine_eval", "candidate_eval", "tile_done"}:
            base = int(round(92.0 * (float(done_count) / float(max(1, unit_total))))) if unit_total > 0 else 0
            percent = max(percent, base)
            self.step7_progress_bar.setValue(percent)
            self.step7_progress_label.setText(
                f"Step 7 progress: auto_scale | {percent}% | done {done_count}/{max(1, unit_total)} | scale {current_scale_index}/{max(1, current_scale_total)}"
            )
            self._update_step7_progress_detail_text()
            return
        if stage == "ranking":
            percent = max(percent, 96)
            self.step7_progress_bar.setValue(percent)
            self.step7_progress_label.setText(f"Step 7 progress: auto_scale | {percent}% | ranking scales")
            self._update_step7_progress_detail_text("Active tiles: ranking scale candidates")
            return
        if stage == "done":
            self.step7_progress_bar.setValue(100)
            self.step7_progress_label.setText(
                f"Step 7 progress: auto_scale | 100% | chosen scale {float(data.get('chosen_scale') or 0.0):.5f}"
            )
            self._update_step7_progress_detail_text("Active tiles: none")
            return

    def _handle_step7_seed_screen_progress(self, data: dict[str, object]) -> None:
        if not self.step7_progress_state or str(self.step7_progress_state.get("mode")) != "seed_screen":
            self._reset_step7_progress_tracking("seed_screen")
        state = self.step7_progress_state if isinstance(self.step7_progress_state, dict) else {}
        stage = str(data.get("stage") or "running")
        tile_label = str(data.get("tile_label") or "").strip()
        total_tiles = int(data.get("tile_count") or data.get("total_items") or state.get("seed_total") or 0)
        if total_tiles > 0:
            state["seed_total"] = int(total_tiles)
        seed_done = state.get("seed_done")
        if not isinstance(seed_done, set):
            seed_done = set()
            state["seed_done"] = seed_done
        if stage in {"coarse_eval", "refine_eval", "candidate_eval"}:
            step_idx = int(data.get("candidate_index") or 0)
            step_count = int(data.get("candidate_count") or 0)
            phase_name = "coarse" if stage == "coarse_eval" else ("refine" if stage == "refine_eval" else "eval")
            profile_name = str(data.get("refine_profile") or data.get("coarse_profile") or "").replace("paired_percentile_", "").replace("moving_", "")
            detail = f"{phase_name} {step_idx}/{max(1, step_count)}"
            if profile_name:
                detail += f" {profile_name}"
            self._set_step7_active_tile_status(tile_label, detail)
        elif stage == "tile_done":
            tile_index = int(data.get("tile_index") or -1)
            if tile_index >= 0:
                seed_done.add(int(tile_index))
            self._remove_step7_active_tile(tile_label)
        percent = self.step7_progress_bar.value()
        done_count = len(seed_done)
        total_count = int(state.get("seed_total") or 0)
        if stage in {"coarse_eval", "refine_eval", "candidate_eval", "tile_done"}:
            base = int(round(88.0 * (float(done_count) / float(max(1, total_count))))) if total_count > 0 else 0
            percent = max(percent, base)
            self.step7_progress_bar.setValue(percent)
            self.step7_progress_label.setText(f"Step 7 progress: seed_screen | {percent}% | tiles done {done_count}/{max(1, total_count)}")
            self._update_step7_progress_detail_text()
            return
        if stage == "setup":
            percent = max(percent, 1)
            self.step7_progress_bar.setValue(percent)
            self.step7_progress_label.setText(
                f"Step 7 progress: seed_screen | {percent}% | setup | tiles={int(state.get('seed_total') or 0)}"
            )
            self._update_step7_progress_detail_text("Active tiles: waiting for worker threads")
            return
        if stage == "ranking":
            percent = max(percent, 92)
            self.step7_progress_bar.setValue(percent)
            self.step7_progress_label.setText(f"Step 7 progress: seed_screen | {percent}% | ranking tiles")
            self._update_step7_progress_detail_text("Active tiles: ranking complete rows")
            return
        if stage == "storyboard":
            percent = max(percent, 96)
            self.step7_progress_bar.setValue(percent)
            self.step7_progress_label.setText(f"Step 7 progress: seed_screen | {percent}% | rendering storyboard")
            self._update_step7_progress_detail_text("Active tiles: rendering storyboard")
            return
        if stage == "done":
            self.step7_progress_bar.setValue(100)
            self.step7_progress_label.setText("Step 7 progress: seed_screen | 100% | finished")
            self._update_step7_progress_detail_text("Active tiles: none")
            return

    def _handle_step7_frontier_progress(self, data: dict[str, object]) -> None:
        if not self.step7_progress_state or str(self.step7_progress_state.get("mode")) != "frontier":
            self._reset_step7_progress_tracking("frontier")
        state = self.step7_progress_state if isinstance(self.step7_progress_state, dict) else {}
        stage = str(data.get("stage") or "running")
        event_mode = str(data.get("mode") or "frontier")
        tile_label = str(data.get("tile_label") or "").strip()
        if stage == "solved_setup":
            state["solved_total"] = int(data.get("tile_count") or 0)
        elif stage == "frontier_setup":
            state["frontier_total"] = int(data.get("tile_count") or 0)
        elif stage == "refresh_setup":
            state["refresh_total"] = int(data.get("tile_count") or 0)

        def _done_set(name: str) -> set[int]:
            val = state.get(name)
            if not isinstance(val, set):
                val = set()
                state[name] = val
            return val

        if stage in {"coarse_eval", "refine_eval", "candidate_eval"}:
            step_idx = int(data.get("candidate_index") or 0)
            step_count = int(data.get("candidate_count") or 0)
            phase_name = "coarse" if stage == "coarse_eval" else ("refine" if stage == "refine_eval" else "eval")
            prefix = "solved" if event_mode == "frontier_solved" else "frontier"
            detail = f"{prefix} {phase_name} {step_idx}/{max(1, step_count)}"
            self._set_step7_active_tile_status(tile_label, detail)
        elif stage == "solved_tile_done":
            tile_index = int(data.get("tile_index") or -1)
            if tile_index >= 0:
                _done_set("solved_done").add(int(tile_index))
            self._remove_step7_active_tile(tile_label)
        elif stage == "frontier_tile_done":
            tile_index = int(data.get("tile_index") or -1)
            if tile_index >= 0:
                _done_set("frontier_done").add(int(tile_index))
            self._remove_step7_active_tile(tile_label)
        elif stage == "refresh_tile_done":
            tile_index = int(data.get("tile_index") or -1)
            if tile_index >= 0:
                _done_set("refresh_done").add(int(tile_index))
            self._remove_step7_active_tile(tile_label)

        percent = self.step7_progress_bar.value()
        solved_total = int(state.get("solved_total") or 0)
        frontier_total = int(state.get("frontier_total") or 0)
        refresh_total = int(state.get("refresh_total") or 0)
        solved_done = len(_done_set("solved_done"))
        frontier_done = len(_done_set("frontier_done"))
        refresh_done = len(_done_set("refresh_done"))

        if stage == "setup":
            percent = max(percent, 1)
            self.step7_progress_bar.setValue(percent)
            self.step7_progress_label.setText(f"Step 7 progress: frontier | {percent}% | setup")
            self._update_step7_progress_detail_text("Active tiles: waiting for worker threads")
            return
        if stage == "solved_setup":
            percent = max(percent, 3)
            self.step7_progress_bar.setValue(percent)
            self.step7_progress_label.setText(f"Step 7 progress: frontier | {percent}% | solved seeds {solved_done}/{max(1, solved_total)}")
            self._update_step7_progress_detail_text("Active tiles: waiting for solved seed evaluation")
            return
        if stage in {"coarse_eval", "refine_eval", "candidate_eval", "solved_tile_done"} and event_mode == "frontier_solved":
            phase_percent = 4 + int(round(12.0 * (float(solved_done) / float(max(1, solved_total))))) if solved_total > 0 else 16
            percent = max(percent, phase_percent)
            self.step7_progress_bar.setValue(percent)
            self.step7_progress_label.setText(f"Step 7 progress: frontier | {percent}% | solved seeds {solved_done}/{max(1, solved_total)}")
            self._update_step7_progress_detail_text()
            return
        if stage == "frontier_setup":
            percent = max(percent, 18)
            self.step7_progress_bar.setValue(percent)
            self.step7_progress_label.setText(f"Step 7 progress: frontier | {percent}% | frontier tiles {frontier_done}/{max(1, frontier_total)}")
            self._update_step7_progress_detail_text("Active tiles: waiting for frontier evaluation")
            return
        if stage in {"coarse_eval", "refine_eval", "candidate_eval", "frontier_tile_done"} and event_mode == "frontier":
            phase_percent = 20 + int(round(52.0 * (float(frontier_done) / float(max(1, frontier_total))))) if frontier_total > 0 else 72
            percent = max(percent, phase_percent)
            self.step7_progress_bar.setValue(percent)
            self.step7_progress_label.setText(f"Step 7 progress: frontier | {percent}% | frontier tiles {frontier_done}/{max(1, frontier_total)}")
            self._update_step7_progress_detail_text()
            return
        if stage == "graph_solve":
            percent = max(percent, 74)
            self.step7_progress_bar.setValue(percent)
            self.step7_progress_label.setText(f"Step 7 progress: frontier | {percent}% | graph solve")
            self._update_step7_progress_detail_text("Active tiles: solving frontier subgraph")
            return
        if stage == "refresh_setup":
            percent = max(percent, 78)
            self.step7_progress_bar.setValue(percent)
            self.step7_progress_label.setText(f"Step 7 progress: frontier | {percent}% | refresh {refresh_done}/{max(1, refresh_total)}")
            self._update_step7_progress_detail_text("Active tiles: waiting for QC refresh")
            return
        if stage == "refresh_tile_done":
            phase_percent = 80 + int(round(12.0 * (float(refresh_done) / float(max(1, refresh_total))))) if refresh_total > 0 else 92
            percent = max(percent, phase_percent)
            self.step7_progress_bar.setValue(percent)
            self.step7_progress_label.setText(f"Step 7 progress: frontier | {percent}% | refresh {refresh_done}/{max(1, refresh_total)}")
            self._update_step7_progress_detail_text()
            return
        if stage == "storyboard":
            percent = max(percent, 96)
            self.step7_progress_bar.setValue(percent)
            self.step7_progress_label.setText(f"Step 7 progress: frontier | {percent}% | rendering storyboard")
            self._update_step7_progress_detail_text("Active tiles: rendering storyboard")
            return
        if stage == "done":
            self.step7_progress_bar.setValue(100)
            self.step7_progress_label.setText("Step 7 progress: frontier | 100% | finished")
            self._update_step7_progress_detail_text("Active tiles: none")
            return

    def on_step7_stage_update(self, payload: object) -> None:
        data = dict(payload) if isinstance(payload, dict) else {}
        mode = str(data.get("mode") or "")
        if mode == "auto_scale":
            self._handle_step7_auto_scale_progress(data)
            return
        if mode == "seed_screen":
            self._handle_step7_seed_screen_progress(data)
            return
        if mode in {"frontier", "frontier_solved"}:
            self._handle_step7_frontier_progress(data)
            return
        stage = str(data.get("stage") or "running")
        percent = max(0, min(100, int(round(float(data.get("progress_percent") or 0)))))
        message = str(data.get("message") or stage)
        self.step7_progress_bar.setValue(percent)
        self.step7_progress_label.setText(f"Step 7 progress: {stage} | {percent}% | {message}")
        self.step7_progress_detail_label.setText("Active tiles: none")

    def run_step7_auto_scale_sweep(self) -> None:
        if self.step7_auto_scale_thread is not None:
            self.step7_info.append("Step 7 auto scale sweep is already running.")
            return
        if self.step7_seed_screen_thread is not None or self.step7_frontier_thread is not None or self.step7_run_thread is not None:
            self.step7_info.append("Step 7 is already running another confocal task.")
            return
        item = self._current_step7_item()
        runs_root = self._step7_runs_root()
        if item is None or self.step7_fixed_rgb is None or self.step7_fixed_labels is None:
            QMessageBox.warning(self, "Step 7 Auto Scale", "Select a myelin section first.")
            return
        if self.step7_confocal_projection_u8 is None or not self.step7_confocal_paths:
            QMessageBox.warning(self, "Step 7 Auto Scale", "Generate a confocal projection first.")
            return
        if runs_root is None:
            QMessageBox.warning(self, "Step 7 Auto Scale", "Confocal registration output root is not available.")
            return
        tile_defs = self._current_step7_tile_defs()
        if not tile_defs:
            QMessageBox.warning(self, "Step 7 Auto Scale", "Current confocal source does not expose a tiled grid to evaluate.")
            return
        if (
            self.step7_tile_result_rows
            or self.step7_accepted_tile_indices
            or self.step7_hold_tile_indices
            or self.step7_frozen_tile_indices
            or self.step7_frontier_tile_indices
        ):
            QMessageBox.warning(
                self,
                "Step 7 Auto Scale",
                "Auto Scale Sweep should be run before seed screening and frontier propagation. Reload or regenerate the current Step 7 projection to start from a clean state.",
            )
            return
        snap = self.step7_preview_view.diagnostic_snapshot() if hasattr(self, "step7_preview_view") else {}
        points_a = snap.get("points_a_scene", {}) if isinstance(snap, dict) else {}
        points_b = snap.get("points_b_raw", {}) if isinstance(snap, dict) else {}
        points_b_scene = snap.get("points_b_scene", {}) if isinstance(snap, dict) else {}
        complete_pair_ids = sorted({int(k) for k in points_a.keys()} & {int(k) for k in points_b.keys()})
        anchor_pairs = [
            {
                "index": int(idx),
                "section_scene_xy": [float(points_a[str(idx)][0]), float(points_a[str(idx)][1])],
                "confocal_raw_xy": [float(points_b[str(idx)][0]), float(points_b[str(idx)][1])],
                "confocal_scene_xy": [
                    float((points_b_scene.get(str(idx)) or [float("nan"), float("nan")])[0]),
                    float((points_b_scene.get(str(idx)) or [float("nan"), float("nan")])[1]),
                ],
            }
            for idx in complete_pair_ids
        ]
        cfg = ConfocalAutoScaleConfig(
            myelin_label=item.label,
            myelin_section_dir=item.section_dir,
            myelin_rgb=self.step7_fixed_rgb,
            myelin_labels=self.step7_fixed_labels,
            myelin_fixed_info=dict(self.step7_fixed_info or {}),
            confocal_projection_u8=self.step7_confocal_projection_u8,
            confocal_signal_mask_u8=self.step7_confocal_projection_mask_u8,
            out_root=runs_root,
            confocal_sources=list(self.step7_confocal_paths),
            confocal_source_mode=self.step7_confocal_source_mode,
            nominal_overlap_fraction=float(self.step7_overlap_spin.value()),
            projection_info=dict(self.step7_projection_info or {}),
            projection_mode=str(self.step7_projection_mode_combo.currentData() or "focus"),
            channel_index=int(self.step7_channel_spin.value()),
            registration_input_profile=self._step7_registration_profile_value(),
            target_working_um_per_px=float(STEP7_TARGET_UM_PER_PX),
            invert_confocal_for_registration=True,
            tx_px=float(self.step7_tx_spin.value()),
            ty_px=float(self.step7_ty_spin.value()),
            angle_deg=float(self.step7_angle_spin.value()),
            scale=float(self.step7_scale_spin.value()),
            flip_lr=bool(self.step7_flip_lr_check.isChecked()),
            flip_ud=bool(self.step7_flip_ud_check.isChecked()),
            anchor_pairs=anchor_pairs,
            sweep_half_range=0.02,
            sweep_step=0.002,
            search_radius_px=24,
            local_refine_radius_px=2,
            sample_tile_limit=3,
            sample_strategy="rowwise_uniform",
        )
        self.step7_auto_scale_button.setEnabled(False)
        self.step7_seed_screen_button.setEnabled(False)
        self.step7_frontier_button.setEnabled(False)
        self._reset_step7_progress_tracking("auto_scale")
        self.step7_progress_label.setText("Step 7 progress: auto scale sweep ...")
        self.step7_auto_scale_thread = QThread(self)
        self.step7_auto_scale_worker = ConfocalAutoScaleWorker(cfg)
        self.step7_auto_scale_worker.moveToThread(self.step7_auto_scale_thread)
        self.step7_auto_scale_thread.started.connect(self.step7_auto_scale_worker.run)
        self.step7_auto_scale_worker.stage_progress.connect(self.on_step7_stage_update)
        self.step7_auto_scale_worker.finished.connect(self.on_step7_auto_scale_finished)
        self.step7_auto_scale_worker.failed.connect(self.on_step7_auto_scale_failed)
        self.step7_auto_scale_worker.finished.connect(self.step7_auto_scale_thread.quit)
        self.step7_auto_scale_worker.failed.connect(self.step7_auto_scale_thread.quit)
        self.step7_auto_scale_thread.finished.connect(self.step7_auto_scale_worker.deleteLater)
        self.step7_auto_scale_thread.finished.connect(self.step7_auto_scale_thread.deleteLater)
        self.step7_auto_scale_thread.start()

    def run_step7_seed_screening(self) -> None:
        if self.step7_seed_screen_thread is not None:
            self.step7_info.append("Step 7 is already screening seed tiles.")
            return
        if self.step7_auto_scale_thread is not None:
            self.step7_info.append("Wait for Auto Scale Sweep to finish first.")
            return
        item = self._current_step7_item()
        runs_root = self._step7_runs_root()
        if item is None or self.step7_fixed_rgb is None or self.step7_fixed_labels is None:
            QMessageBox.warning(self, "Step 7 Seed Screening", "Select a myelin section first.")
            return
        if self.step7_confocal_projection_u8 is None or not self.step7_confocal_paths:
            QMessageBox.warning(self, "Step 7 Seed Screening", "Generate a confocal projection first.")
            return
        if runs_root is None:
            QMessageBox.warning(self, "Step 7 Seed Screening", "Confocal registration output root is not available.")
            return
        tile_defs = self._current_step7_tile_defs()
        if not tile_defs:
            QMessageBox.warning(self, "Step 7 Seed Screening", "Current confocal source does not expose a tiled grid to screen.")
            return
        snap = self.step7_preview_view.diagnostic_snapshot() if hasattr(self, "step7_preview_view") else {}
        points_a = snap.get("points_a_scene", {}) if isinstance(snap, dict) else {}
        points_b = snap.get("points_b_raw", {}) if isinstance(snap, dict) else {}
        points_b_scene = snap.get("points_b_scene", {}) if isinstance(snap, dict) else {}
        complete_pair_ids = sorted({int(k) for k in points_a.keys()} & {int(k) for k in points_b.keys()})
        anchor_pairs = [
            {
                "index": int(idx),
                "section_scene_xy": [float(points_a[str(idx)][0]), float(points_a[str(idx)][1])],
                "confocal_raw_xy": [float(points_b[str(idx)][0]), float(points_b[str(idx)][1])],
                "confocal_scene_xy": [
                    float((points_b_scene.get(str(idx)) or [float("nan"), float("nan")])[0]),
                    float((points_b_scene.get(str(idx)) or [float("nan"), float("nan")])[1]),
                ],
            }
            for idx in complete_pair_ids
        ]
        cfg = ConfocalSeedScreenConfig(
            myelin_label=item.label,
            myelin_section_dir=item.section_dir,
            myelin_rgb=self.step7_fixed_rgb,
            myelin_labels=self.step7_fixed_labels,
            myelin_fixed_info=dict(self.step7_fixed_info or {}),
            confocal_projection_u8=self.step7_confocal_projection_u8,
            confocal_signal_mask_u8=self.step7_confocal_projection_mask_u8,
            out_root=runs_root,
            confocal_sources=list(self.step7_confocal_paths),
            confocal_source_mode=self.step7_confocal_source_mode,
            nominal_overlap_fraction=float(self.step7_overlap_spin.value()),
            projection_info=dict(self.step7_projection_info or {}),
            projection_mode=str(self.step7_projection_mode_combo.currentData() or "focus"),
            channel_index=int(self.step7_channel_spin.value()),
            registration_input_profile=self._step7_registration_profile_value(),
            target_working_um_per_px=float(STEP7_TARGET_UM_PER_PX),
            invert_confocal_for_registration=True,
            tx_px=float(self.step7_tx_spin.value()),
            ty_px=float(self.step7_ty_spin.value()),
            angle_deg=float(self.step7_angle_spin.value()),
            scale=float(self.step7_scale_spin.value()),
            flip_lr=bool(self.step7_flip_lr_check.isChecked()),
            flip_ud=bool(self.step7_flip_ud_check.isChecked()),
            anchor_pairs=anchor_pairs,
            search_radius_px=32,
            top_k_storyboard=max(1, min(6, len(tile_defs))),
        )
        self.step7_seed_screen_button.setEnabled(False)
        self._reset_step7_progress_tracking("seed_screen")
        self.step7_progress_label.setText("Step 7 progress: screening seed tiles ...")
        self.step7_seed_screen_thread = QThread(self)
        self.step7_seed_screen_worker = ConfocalSeedScreenWorker(cfg)
        self.step7_seed_screen_worker.moveToThread(self.step7_seed_screen_thread)
        self.step7_seed_screen_thread.started.connect(self.step7_seed_screen_worker.run)
        self.step7_seed_screen_worker.stage_progress.connect(self.on_step7_stage_update)
        self.step7_seed_screen_worker.finished.connect(self.on_step7_seed_screening_finished)
        self.step7_seed_screen_worker.failed.connect(self.on_step7_seed_screening_failed)
        self.step7_seed_screen_worker.finished.connect(self.step7_seed_screen_thread.quit)
        self.step7_seed_screen_worker.failed.connect(self.step7_seed_screen_thread.quit)
        self.step7_seed_screen_thread.finished.connect(self.step7_seed_screen_worker.deleteLater)
        self.step7_seed_screen_thread.finished.connect(self.step7_seed_screen_thread.deleteLater)
        self.step7_seed_screen_thread.start()

    def run_step7_frontier_propagation(self) -> None:
        if self.step7_frontier_thread is not None:
            self.step7_info.append("Step 7 frontier propagation is already running.")
            return
        if self.step7_auto_scale_thread is not None:
            self.step7_info.append("Wait for Auto Scale Sweep to finish first.")
            return
        item = self._current_step7_item()
        runs_root = self._step7_runs_root()
        if item is None or self.step7_fixed_rgb is None or self.step7_fixed_labels is None:
            QMessageBox.warning(self, "Step 7 Frontier", "Select a myelin section first.")
            return
        if self.step7_confocal_projection_u8 is None or not self.step7_confocal_paths:
            QMessageBox.warning(self, "Step 7 Frontier", "Generate a confocal projection first.")
            return
        if runs_root is None:
            QMessageBox.warning(self, "Step 7 Frontier", "Confocal registration output root is not available.")
            return
        tile_defs = self._current_step7_tile_defs()
        if not tile_defs:
            QMessageBox.warning(self, "Step 7 Frontier", "Current confocal source does not expose a tiled grid to propagate.")
            return
        snap = self.step7_preview_view.diagnostic_snapshot() if hasattr(self, "step7_preview_view") else {}
        points_a = snap.get("points_a_scene", {}) if isinstance(snap, dict) else {}
        points_b = snap.get("points_b_raw", {}) if isinstance(snap, dict) else {}
        points_b_scene = snap.get("points_b_scene", {}) if isinstance(snap, dict) else {}
        complete_pair_ids = sorted({int(k) for k in points_a.keys()} & {int(k) for k in points_b.keys()})
        anchor_pairs = [
            {
                "index": int(idx),
                "section_scene_xy": [float(points_a[str(idx)][0]), float(points_a[str(idx)][1])],
                "confocal_raw_xy": [float(points_b[str(idx)][0]), float(points_b[str(idx)][1])],
                "confocal_scene_xy": [
                    float((points_b_scene.get(str(idx)) or [float("nan"), float("nan")])[0]),
                    float((points_b_scene.get(str(idx)) or [float("nan"), float("nan")])[1]),
                ],
            }
            for idx in complete_pair_ids
        ]
        if not self.step7_frozen_tile_indices and not self.step7_accepted_tile_indices and not self.step7_tile_result_rows:
            QMessageBox.warning(
                self,
                "Step 7 Frontier",
                "Accept/freeze a tile first, or run Screen Seed Tiles so the prototype can fall back to the current best tile.",
            )
            return
        cfg = ConfocalFrontierConfig(
            myelin_label=item.label,
            myelin_section_dir=item.section_dir,
            myelin_rgb=self.step7_fixed_rgb,
            myelin_labels=self.step7_fixed_labels,
            myelin_fixed_info=dict(self.step7_fixed_info or {}),
            confocal_projection_u8=self.step7_confocal_projection_u8,
            confocal_signal_mask_u8=self.step7_confocal_projection_mask_u8,
            out_root=runs_root,
            confocal_sources=list(self.step7_confocal_paths),
            confocal_source_mode=self.step7_confocal_source_mode,
            nominal_overlap_fraction=float(self.step7_overlap_spin.value()),
            projection_info=dict(self.step7_projection_info or {}),
            projection_mode=str(self.step7_projection_mode_combo.currentData() or "focus"),
            channel_index=int(self.step7_channel_spin.value()),
            registration_input_profile=self._step7_registration_profile_value(),
            target_working_um_per_px=float(STEP7_TARGET_UM_PER_PX),
            invert_confocal_for_registration=True,
            tx_px=float(self.step7_tx_spin.value()),
            ty_px=float(self.step7_ty_spin.value()),
            angle_deg=float(self.step7_angle_spin.value()),
            scale=float(self.step7_scale_spin.value()),
            flip_lr=bool(self.step7_flip_lr_check.isChecked()),
            flip_ud=bool(self.step7_flip_ud_check.isChecked()),
            anchor_pairs=anchor_pairs,
            selected_tile_index=None,
            accepted_tile_indices=sorted(int(v) for v in self.step7_accepted_tile_indices),
            frozen_tile_indices=sorted(int(v) for v in self.step7_frozen_tile_indices),
            prior_rows=[dict(row) for row in self.step7_tile_result_rows.values()],
            search_radius_px=20,
            max_frontier_tiles=6,
            top_k_storyboard=4,
        )
        self.step7_frontier_button.setEnabled(False)
        self._reset_step7_progress_tracking("frontier")
        self.step7_progress_label.setText("Step 7 progress: propagating frontier ...")
        self.step7_frontier_thread = QThread(self)
        self.step7_frontier_worker = ConfocalFrontierWorker(cfg)
        self.step7_frontier_worker.moveToThread(self.step7_frontier_thread)
        self.step7_frontier_thread.started.connect(self.step7_frontier_worker.run)
        self.step7_frontier_worker.stage_progress.connect(self.on_step7_stage_update)
        self.step7_frontier_worker.finished.connect(self.on_step7_frontier_finished)
        self.step7_frontier_worker.failed.connect(self.on_step7_frontier_failed)
        self.step7_frontier_worker.finished.connect(self.step7_frontier_thread.quit)
        self.step7_frontier_worker.failed.connect(self.step7_frontier_thread.quit)
        self.step7_frontier_thread.finished.connect(self.step7_frontier_worker.deleteLater)
        self.step7_frontier_thread.finished.connect(self.step7_frontier_thread.deleteLater)
        self.step7_frontier_thread.start()

    def run_step7_registration(self) -> None:
        if self.step7_run_thread is not None:
            self.step7_info.append("Step 7 is already running a confocal registration.")
            return
        item = self._current_step7_item()
        runs_root = self._step7_runs_root()
        ants_bin = find_ants_bin()
        if item is None or self.step7_fixed_rgb is None or self.step7_fixed_labels is None:
            QMessageBox.warning(self, "Step 7 Confocal", "Select a myelin section first.")
            return
        if self.step7_confocal_projection_u8 is None or not self.step7_confocal_paths:
            QMessageBox.warning(self, "Step 7 Confocal", "Generate a confocal projection first.")
            return
        if runs_root is None:
            QMessageBox.warning(self, "Step 7 Confocal", "Confocal registration output root is not available.")
            return
        if ants_bin is None:
            QMessageBox.warning(self, "Step 7 Confocal", "Could not find a local ANTs installation.")
            return
        snap = self.step7_preview_view.diagnostic_snapshot() if hasattr(self, "step7_preview_view") else {}
        points_a = snap.get("points_a_scene", {}) if isinstance(snap, dict) else {}
        points_b = snap.get("points_b_raw", {}) if isinstance(snap, dict) else {}
        points_b_scene = snap.get("points_b_scene", {}) if isinstance(snap, dict) else {}
        complete_pair_ids = sorted({int(k) for k in points_a.keys()} & {int(k) for k in points_b.keys()})
        anchor_pairs = [
            {
                "index": int(idx),
                "section_scene_xy": [float(points_a[str(idx)][0]), float(points_a[str(idx)][1])],
                "confocal_raw_xy": [float(points_b[str(idx)][0]), float(points_b[str(idx)][1])],
                "confocal_scene_xy": [
                    float((points_b_scene.get(str(idx)) or [float("nan"), float("nan")])[0]),
                    float((points_b_scene.get(str(idx)) or [float("nan"), float("nan")])[1]),
                ],
            }
            for idx in complete_pair_ids
        ]
        cfg = ConfocalRigidConfig(
            myelin_label=item.label,
            myelin_section_dir=item.section_dir,
            myelin_stain=item.stain,
            myelin_rgb=self.step7_fixed_rgb,
            myelin_labels=self.step7_fixed_labels,
            myelin_fixed_info=dict(self.step7_fixed_info or {}),
            confocal_projection_u8=self.step7_confocal_projection_u8,
            confocal_signal_mask_u8=self.step7_confocal_projection_mask_u8,
            ants_bin=ants_bin,
            out_root=runs_root,
            confocal_sources=list(self.step7_confocal_paths),
            confocal_source_mode=self.step7_confocal_source_mode,
            nominal_overlap_fraction=float(self.step7_overlap_spin.value()),
            projection_info=dict(self.step7_projection_info or {}),
            projection_mode=str(self.step7_projection_mode_combo.currentData() or "focus"),
            channel_index=int(self.step7_channel_spin.value()),
            local_refine_model=str(self.step7_refine_model_combo.currentData() or "similarity"),
            registration_input_profile=self._step7_registration_profile_value(),
            target_working_um_per_px=float(STEP7_TARGET_UM_PER_PX),
            invert_confocal_for_registration=True,
            tx_px=float(self.step7_tx_spin.value()),
            ty_px=float(self.step7_ty_spin.value()),
            angle_deg=float(self.step7_angle_spin.value()),
            scale=float(self.step7_scale_spin.value()),
            flip_lr=bool(self.step7_flip_lr_check.isChecked()),
            flip_ud=bool(self.step7_flip_ud_check.isChecked()),
            anchor_pairs=anchor_pairs,
        )
        self.step7_run_button.setEnabled(False)
        self.step7_progress_label.setText(
            f"Step 7 progress: running fiber registration ({str(self.step7_refine_model_combo.currentData() or 'similarity')}) ..."
        )
        self.step7_run_thread = QThread(self)
        self.step7_run_worker = ConfocalRigidWorker(cfg)
        self.step7_run_worker.moveToThread(self.step7_run_thread)
        self.step7_run_thread.started.connect(self.step7_run_worker.run)
        self.step7_run_worker.finished.connect(self.on_step7_registration_finished)
        self.step7_run_worker.failed.connect(self.on_step7_registration_failed)
        self.step7_run_worker.finished.connect(self.step7_run_thread.quit)
        self.step7_run_worker.failed.connect(self.step7_run_thread.quit)
        self.step7_run_thread.finished.connect(self.step7_run_worker.deleteLater)
        self.step7_run_thread.finished.connect(self.step7_run_thread.deleteLater)
        self.step7_run_thread.start()

    def on_step7_auto_scale_finished(self, summary: object) -> None:
        data = dict(summary) if isinstance(summary, dict) else {}
        chosen_scale = float(data.get("chosen_scale") or self.step7_scale_spin.value())
        run_dir_ref = data.get("run_dir")
        self.step7_auto_scale_button.setEnabled(True)
        self.step7_seed_screen_button.setEnabled(True)
        self.step7_frontier_button.setEnabled(True)
        self.step7_auto_scale_worker = None
        self.step7_auto_scale_thread = None
        self.step7_last_auto_scale_dir = Path(run_dir_ref) if run_dir_ref else None
        self.step7_last_auto_scale_summary_lines = self._build_step7_auto_scale_summary_lines(data)
        self.step7_scale_spin.setValue(float(chosen_scale))
        self.step7_last_manual_action = f"auto_scale_applied={float(chosen_scale):.5f}"
        self.update_step7_preview(preserve_view=True)
        self.step7_progress_bar.setValue(100)
        self.step7_progress_label.setText(f"Step 7 progress: auto scale sweep finished ({float(chosen_scale):.5f})")
        self.step7_progress_detail_label.setText("Active tiles: none")
        self.step7_progress_state = None
        self._update_step7_info_text()
        self._notify_completion("Step 7 auto scale sweep finished")

    def on_step7_auto_scale_failed(self, message: str) -> None:
        self.step7_info.append(message)
        self.step7_progress_bar.setValue(0)
        self.step7_progress_label.setText("Step 7 progress: auto scale sweep failed")
        self.step7_progress_detail_label.setText("Active tiles: none")
        self.step7_progress_state = None
        self.step7_auto_scale_button.setEnabled(True)
        self.step7_seed_screen_button.setEnabled(True)
        self.step7_frontier_button.setEnabled(True)
        self.step7_auto_scale_worker = None
        self.step7_auto_scale_thread = None

    def on_step7_seed_screening_finished(self, summary: object) -> None:
        data = dict(summary) if isinstance(summary, dict) else {}
        files = data.get("files") if isinstance(data.get("files"), dict) else {}
        storyboard_ref = data.get("storyboard_path") or files.get("storyboard")
        run_dir_ref = data.get("run_dir")
        storyboard = Path(str(storyboard_ref)) if storyboard_ref else None
        self.step7_seed_screen_button.setEnabled(True)
        self.step7_seed_screen_worker = None
        self.step7_seed_screen_thread = None
        self.step7_last_seed_screen_dir = Path(run_dir_ref) if run_dir_ref else None
        rows = data.get("rows") if isinstance(data.get("rows"), list) else []
        self.step7_last_seed_screen_rows = [dict(row) for row in rows]
        self.step7_tile_result_rows = {
            int(row.get("tile_index", -1)): dict(row)
            for row in self.step7_last_seed_screen_rows
            if int(row.get("tile_index", -1)) >= 0
        }
        self.step7_accepted_tile_indices = set()
        self.step7_hold_tile_indices = set()
        self.step7_last_seed_screen_summary_lines = self._build_step7_seed_screen_summary_lines(data)
        self.step7_last_frontier_summary_lines = []
        self.step7_last_frontier_rows = []
        self.step7_frontier_tile_indices = set()
        top = data.get("top_seed_candidates") if isinstance(data.get("top_seed_candidates"), list) else []
        if top:
            best_index = int(top[0].get("tile_index", -1))
            if best_index >= 0:
                self._set_step7_selected_tile(best_index)
                self.step7_last_manual_action = f"screen_best_seed={str(top[0].get('label') or f'T{best_index:02d}')}"
        self.step7_progress_bar.setValue(100)
        self.step7_progress_label.setText("Step 7 progress: seed screening finished")
        self.step7_progress_detail_label.setText("Active tiles: none")
        self.step7_progress_state = None
        self._update_step7_info_text()
        if storyboard is not None and storyboard.exists():
            if not top:
                pixmap = QPixmap(str(storyboard))
                if not pixmap.isNull():
                    self.step7_storyboard_label.setText("")
                    self.step7_storyboard_label.setPixmap(pixmap)
                    self.step7_storyboard_label.resize(pixmap.size())
        self._notify_completion("Step 7 seed screening finished")

    def on_step7_seed_screening_failed(self, message: str) -> None:
        self.step7_info.append(message)
        self.step7_progress_bar.setValue(0)
        self.step7_progress_label.setText("Step 7 progress: seed screening failed")
        self.step7_progress_detail_label.setText("Active tiles: none")
        self.step7_progress_state = None
        self.step7_seed_screen_button.setEnabled(True)
        self.step7_seed_screen_worker = None
        self.step7_seed_screen_thread = None

    def on_step7_frontier_finished(self, summary: object) -> None:
        data = dict(summary) if isinstance(summary, dict) else {}
        self.step7_frontier_button.setEnabled(True)
        self.step7_frontier_worker = None
        self.step7_frontier_thread = None
        run_dir_ref = data.get("run_dir")
        self.step7_last_frontier_dir = Path(run_dir_ref) if run_dir_ref else None
        self.step7_last_frontier_rows = [dict(row) for row in (data.get("rows") if isinstance(data.get("rows"), list) else [])]
        solved_rows = [dict(row) for row in (data.get("solved_rows") if isinstance(data.get("solved_rows"), list) else [])]
        for row in solved_rows + self.step7_last_frontier_rows:
            tile_index = int(row.get("tile_index", -1))
            if tile_index >= 0:
                self.step7_tile_result_rows[tile_index] = dict(row)
        self.step7_frontier_tile_indices = {
            int(row.get("tile_index", -1))
            for row in self.step7_last_frontier_rows
            if int(row.get("tile_index", -1)) >= 0 and str(row.get("tile_state") or "").strip().lower() == "frontier"
        }
        self._sync_step7_tile_state_sets_from_rows()
        self.step7_last_frontier_summary_lines = self._build_step7_frontier_summary_lines(data)
        if self.step7_last_frontier_rows:
            best_index = int(self.step7_last_frontier_rows[0].get("tile_index", -1))
            if best_index >= 0:
                self._set_step7_selected_tile(best_index)
                self.step7_last_manual_action = f"frontier_best={str(self.step7_last_frontier_rows[0].get('label') or f'T{best_index:02d}')}"
        self.step7_progress_bar.setValue(100)
        self.step7_progress_label.setText("Step 7 progress: frontier propagation finished")
        self.step7_progress_detail_label.setText("Active tiles: none")
        self.step7_progress_state = None
        self._refresh_step7_preview_tile_states_only()
        self._update_step7_info_text()
        self._notify_completion("Step 7 frontier propagation finished")

    def on_step7_frontier_failed(self, message: str) -> None:
        self.step7_info.append(message)
        self.step7_progress_bar.setValue(0)
        self.step7_progress_label.setText("Step 7 progress: frontier propagation failed")
        self.step7_progress_detail_label.setText("Active tiles: none")
        self.step7_progress_state = None
        self.step7_frontier_button.setEnabled(True)
        self.step7_frontier_worker = None
        self.step7_frontier_thread = None

    def on_step7_registration_finished(self, summary: object) -> None:
        data = dict(summary) if isinstance(summary, dict) else {}
        files = data.get("files") if isinstance(data.get("files"), dict) else {}
        storyboard_ref = data.get("storyboard_path") or files.get("quick_qc_storyboard") or files.get("storyboard")
        run_dir_ref = data.get("run_dir")
        if not run_dir_ref and files.get("manifest"):
            try:
                run_dir_ref = str(Path(str(files.get("manifest"))).parent)
            except Exception:
                run_dir_ref = ""
        storyboard = Path(str(storyboard_ref)) if storyboard_ref else None
        refine_model = str(data.get("local_refine_model") or data.get("local_registration", {}).get("transform_model") or self.step7_refine_model_combo.currentData() or "similarity")
        self.step7_progress_label.setText(f"Step 7 progress: fiber registration finished ({refine_model})")
        self.step7_run_button.setEnabled(True)
        self.step7_run_worker = None
        self.step7_run_thread = None
        self.step7_last_run_dir = Path(run_dir_ref) if run_dir_ref else None
        self.step7_last_run_summary_lines = self._build_step7_run_summary_lines(data)
        self._update_step7_info_text()
        self._update_step7_tile_qc_display()
        if storyboard is not None and storyboard.exists() and not self.step7_last_seed_screen_rows:
            pixmap = QPixmap(str(storyboard))
            if not pixmap.isNull():
                self.step7_storyboard_label.setText("")
                self.step7_storyboard_label.setPixmap(pixmap)
                self.step7_storyboard_label.resize(pixmap.size())
        self._notify_completion(f"Step 7 confocal {refine_model} registration finished")

    def on_step7_registration_failed(self, message: str) -> None:
        self.step7_info.append(message)
        self.step7_progress_label.setText("Step 7 progress: registration failed")
        self.step7_run_button.setEnabled(True)
        self.step7_run_worker = None
        self.step7_run_thread = None

    def export_step7_session_package(self) -> None:
        item = self._current_step7_item()
        export_root = self._step7_export_root()
        if item is None or self.step7_fixed_rgb is None or self.step7_fixed_labels is None:
            QMessageBox.warning(self, "Step 7 Export", "Select a myelin section first.")
            return
        if self.step7_confocal_projection_u8 is None or not self.step7_confocal_paths:
            QMessageBox.warning(self, "Step 7 Export", "Generate a confocal projection first.")
            return
        if export_root is None:
            QMessageBox.warning(self, "Step 7 Export", "Confocal export root is not available.")
            return
        snap = self.step7_preview_view.diagnostic_snapshot() if hasattr(self, "step7_preview_view") else {}
        points_a = snap.get("points_a_scene", {}) if isinstance(snap, dict) else {}
        points_b = snap.get("points_b_raw", {}) if isinstance(snap, dict) else {}
        points_b_scene = snap.get("points_b_scene", {}) if isinstance(snap, dict) else {}
        complete_pair_ids = sorted({int(k) for k in points_a.keys()} & {int(k) for k in points_b.keys()})
        anchor_pairs = [
            {
                "index": int(idx),
                "section_scene_xy": [float(points_a[str(idx)][0]), float(points_a[str(idx)][1])],
                "confocal_raw_xy": [float(points_b[str(idx)][0]), float(points_b[str(idx)][1])],
                "confocal_scene_xy": [
                    float((points_b_scene.get(str(idx)) or [float("nan"), float("nan")])[0]),
                    float((points_b_scene.get(str(idx)) or [float("nan"), float("nan")])[1]),
                ],
            }
            for idx in complete_pair_ids
        ]
        tile_rows = [
            dict(self.step7_tile_result_rows[idx])
            for idx in sorted(self.step7_tile_result_rows.keys())
            if isinstance(self.step7_tile_result_rows.get(idx), dict)
        ]
        try:
            summary = export_confocal_step7_session(
                myelin_label=item.label,
                myelin_section_dir=item.section_dir,
                out_root=export_root,
                fixed_rgb=self.step7_fixed_rgb,
                fixed_info=dict(self.step7_fixed_info or {}),
                confocal_projection_u8=self.step7_confocal_projection_u8,
                confocal_sources=list(self.step7_confocal_paths),
                confocal_source_mode=self.step7_confocal_source_mode,
                nominal_overlap_fraction=float(self.step7_overlap_spin.value()),
                projection_info=dict(self.step7_projection_info or {}),
                projection_mode=str(self.step7_projection_mode_combo.currentData() or "focus"),
                channel_index=int(self.step7_channel_spin.value()),
                registration_input_profile=self._step7_registration_profile_value(),
                target_working_um_per_px=float(STEP7_TARGET_UM_PER_PX),
                tx_px=float(self.step7_tx_spin.value()),
                ty_px=float(self.step7_ty_spin.value()),
                angle_deg=float(self.step7_angle_spin.value()),
                scale=float(self.step7_scale_spin.value()),
                flip_lr=bool(self.step7_flip_lr_check.isChecked()),
                flip_ud=bool(self.step7_flip_ud_check.isChecked()),
                anchor_pairs=anchor_pairs,
                tile_defs=[dict(row) for row in self._current_step7_tile_defs()],
                tile_rows=tile_rows,
                accepted_tile_indices=sorted(int(v) for v in self.step7_accepted_tile_indices),
                frozen_tile_indices=sorted(int(v) for v in self.step7_frozen_tile_indices),
                hold_tile_indices=sorted(int(v) for v in self.step7_hold_tile_indices),
                frontier_tile_indices=sorted(int(v) for v in self.step7_frontier_tile_indices),
                selected_tile_indices=self._selected_step7_tile_indices_from_snapshot(),
                seed_screen_run_dir=self.step7_last_seed_screen_dir,
                frontier_run_dir=self.step7_last_frontier_dir,
            )
        except Exception:
            self.step7_info.append(f"Step 7 session export failed:\n{traceback.format_exc()}")
            return
        self.step7_last_export_dir = Path(str(summary.get("run_dir") or ""))
        self.step7_info.append(
            "Step 7 session exported. "
            f"manifest={summary.get('session_manifest', '')} | "
            f"step8_handoff={summary.get('step8_handoff', '')}"
        )
        self.step7_progress_label.setText("Step 7 progress: session export finished")
        self._refresh_step8_info()

    def export_step7_full_report(self) -> None:
        item = self._current_step7_item()
        if item is None:
            QMessageBox.warning(self, "Step 7 Confocal", "Select a myelin section first.")
            return
        run_dir = self.step7_last_run_dir
        if run_dir is None or not run_dir.exists():
            run_dir = self._latest_step7_run_dir(item.label)
        if run_dir is None or not run_dir.exists():
            QMessageBox.warning(self, "Step 7 Confocal", "No Step 7 run found for this section yet.")
            return
        try:
            summary = export_confocal_full_report(run_dir)
        except Exception:
            self.step7_info.append(f"Step 7 full-report export failed:\n{traceback.format_exc()}")
            return
        self.step7_info.append(
            "Full report exported. "
            f"storyboard={summary.get('full_report_storyboard', '')} | "
            f"metrics_md={summary.get('full_metrics_report_md', '')}"
        )
        self.step7_progress_label.setText("Step 7 progress: full report exported")

    def _ensure_step4_review_status_selected(self) -> bool:
        status = str(self.step4_registration_status_combo.currentData() or "unreviewed")
        if status in {"usable", "unusable"}:
            return True

        box = QMessageBox(self)
        box.setWindowTitle("Review Result Required")
        box.setIcon(QMessageBox.Icon.Question)
        box.setText("Current pair hasn't been reviewed.")
        box.setInformativeText("Choose a review result before saving this pair.")
        usable_btn = box.addButton("Usable", QMessageBox.ButtonRole.AcceptRole)
        unusable_btn = box.addButton("Unusable", QMessageBox.ButtonRole.DestructiveRole)
        cancel_btn = box.addButton("Cancel", QMessageBox.ButtonRole.RejectRole)
        box.setDefaultButton(cancel_btn)
        box.exec()

        clicked = box.clickedButton()
        if clicked is usable_btn:
            idx = self.step4_registration_status_combo.findData("usable")
            self.step4_registration_status_combo.setCurrentIndex(max(0, idx))
            return True
        if clicked is unusable_btn:
            idx = self.step4_registration_status_combo.findData("unusable")
            self.step4_registration_status_combo.setCurrentIndex(max(0, idx))
            return True
        return False

    def move_to_next_unreviewed_pair(self) -> None:
        pair = self._current_pair()
        if pair is None:
            return
        if self._step4_registration_status(pair) == "unreviewed":
            QMessageBox.information(self, "Pair Not Reviewed", "Current pair hasn't been reviewed.")
            return
        if not self.step4_pairs:
            return
        total = len(self.step4_pairs)
        start = self.current_pair_index
        for offset in range(1, total):
            idx = (start + offset) % total
            candidate = self.step4_pairs[idx]
            if self._step4_registration_status(candidate) == "unreviewed":
                self.current_pair_index = idx
                self.step4_pair_list.setCurrentRow(idx)
                self.load_current_pair()
                return
        QMessageBox.information(self, "No Unreviewed Pair", "No unreviewed pair remains.")

    def load_current_pair(self) -> None:
        pair = self._current_pair()
        if pair is None:
            return
        cached = self._step4_cached_pair_data(pair)
        if cached is None:
            cached = {
                "myelin": self._load_step4_pair_section(pair, "myelin"),
                "nissl": self._load_step4_pair_section(pair, "nissl"),
            }
            with self.step4_pair_cache_lock:
                generation = self.step4_pair_cache_generation
            self._store_step4_pair_cache(pair.pair_key, cached, generation)
        my_metadata, my_crop_rgb, my_tissue, my_artifact, my_source = cached["myelin"]
        ni_metadata, ni_crop_rgb, ni_tissue, ni_artifact, ni_source = cached["nissl"]
        self.step4_myelin_crop_rgb = my_crop_rgb
        self.step4_nissl_crop_rgb = ni_crop_rgb
        self.step4_myelin_editor.set_section(my_crop_rgb, my_tissue, my_artifact)
        self.step4_nissl_editor.set_section(ni_crop_rgb, ni_tissue, ni_artifact)
        self.step4_pair_label.setText(f"{self.current_pair_index + 1}/{len(self.step4_pairs)} | {pair.display_label}")
        self.step4_myelin_title.setText(f"Myelin | {pair.myelin_item.label}")
        self.step4_nissl_title.setText(f"Nissl | {pair.nissl_item.label}")
        review = dict(self.step4_pair_registry.get(pair.pair_key) or {})
        flip = dict(review.get("flip_lr") or {})
        self.step4_component_groups = {
            "myelin": self._step4_side_group_map_from_review(review, "myelin"),
            "nissl": self._step4_side_group_map_from_review(review, "nissl"),
        }
        self.step4_group_flips = {
            "myelin": self._step4_side_group_flip_map_from_review(review, "myelin"),
            "nissl": self._step4_side_group_flip_map_from_review(review, "nissl"),
        }
        self.step4_myelin_flip_check.blockSignals(True)
        self.step4_nissl_flip_check.blockSignals(True)
        self.step4_registration_status_combo.blockSignals(True)
        self.step4_myelin_flip_check.setChecked(bool(flip.get("myelin", False)))
        self.step4_nissl_flip_check.setChecked(bool(flip.get("nissl", False)))
        status = self._step4_registration_status(pair)
        status_idx = self.step4_registration_status_combo.findData(status)
        self.step4_registration_status_combo.setCurrentIndex(max(0, status_idx))
        self.step4_myelin_flip_check.blockSignals(False)
        self.step4_nissl_flip_check.blockSignals(False)
        self.step4_registration_status_combo.blockSignals(False)
        self.step4_myelin_editor.set_mirror(self.step4_myelin_flip_check.isChecked())
        self.step4_nissl_editor.set_mirror(self.step4_nissl_flip_check.isChecked())
        self.set_step4_active_layer_all(self.step4_layer_combo.currentText())
        self.set_step4_brush_radius_all(self.step4_brush_spin.value())
        self._sync_step4_component_marks()
        self.set_step4_active_editor(self.step4_myelin_editor, "myelin")
        self.update_step4_mask_stats()
        self.step4_info.setPlainText(
            "\n".join(
                [
                    f"pair_key: {pair.pair_key}",
                    f"animal_id: {pair.animal_id}",
                    f"myelin_label: {pair.myelin_item.label}",
                    f"nissl_label: {pair.nissl_item.label}",
                    f"myelin_mask_source: {my_source.get('mask_source')}",
                    f"nissl_mask_source: {ni_source.get('mask_source')}",
                    f"myelin_pipeline_stage: {my_metadata.get('pipeline_stage', 'unknown')}",
                    f"nissl_pipeline_stage: {ni_metadata.get('pipeline_stage', 'unknown')}",
                    f"registration_status: {status}",
                    f"flip_myelin_lr: {bool(flip.get('myelin', False))}",
                    f"flip_nissl_lr: {bool(flip.get('nissl', False))}",
                    f"myelin_group_flip_lr: {json.dumps(self.step4_group_flips.get('myelin', {}), ensure_ascii=True)}",
                    f"nissl_group_flip_lr: {json.dumps(self.step4_group_flips.get('nissl', {}), ensure_ascii=True)}",
                    f"myelin_component_selection: {self._step4_selection_text('myelin')}",
                    f"nissl_component_selection: {self._step4_selection_text('nissl')}",
                    "",
                    "Editing:",
                    "- click either panel to make it the active keyboard target",
                    "- panels open in hand/grab mode by default; press P to enter brush mode",
                    "- T / A / N / M / P / H / L / D / C / Z work on the active panel",
                    "- 1 / 2 mark the hovered combined component on the active panel as registration group 1 or 2",
                    "- S saves this pair QC and moves to the next pair",
                    "- L cuts a thin erase line through both masks on the active panel",
                    "- flip flags are stored for later registration prep; they do not rewrite crop_raw",
                    "- mark hovered combined component as registration group 1 or 2",
                    "- Save writes pair-specific registration masks only; canonical section masks are left unchanged",
                ]
            )
        )
        self._schedule_step4_pair_prefetch(self.current_pair_index)

    def _mask_stats_text_from_arrays(self, tissue: np.ndarray, artifact: np.ndarray) -> str:
        tissue_bin = tissue > 0
        artifact_bin = artifact > 0
        usable = tissue_bin & ~artifact_bin
        if usable.size == 0:
            return "No mask loaded"
        num, _, stats, _ = cv2.connectedComponentsWithStats(usable.astype(np.uint8), 8)
        areas = sorted([int(stats[i, cv2.CC_STAT_AREA]) for i in range(1, num)], reverse=True)
        return "\n".join(
            [
                f"usable_components: {max(0, num - 1)}",
                f"tissue_px: {int(tissue_bin.sum())}",
                f"artifact_px: {int(artifact_bin.sum())}",
                f"usable_px: {int(usable.sum())}",
                f"largest_usable_px: {areas[0] if areas else 0}",
            ]
        )

    def update_step4_mask_stats(self) -> None:
        my_tissue, my_artifact = self.step4_myelin_editor.current_masks()
        ni_tissue, ni_artifact = self.step4_nissl_editor.current_masks()
        self.step4_myelin_stats.setPlainText(self._mask_stats_text_from_arrays(my_tissue, my_artifact))
        self.step4_nissl_stats.setPlainText(self._mask_stats_text_from_arrays(ni_tissue, ni_artifact))
        if self._current_pair() is not None:
            pair = self._current_pair()
            review = dict(self.step4_pair_registry.get(pair.pair_key) or {})
            flip = dict(review.get("flip_lr") or {})
            status = self._step4_registration_status(pair)
            self.step4_info.setPlainText(
                "\n".join(
                    [
                        f"pair_key: {pair.pair_key}",
                        f"animal_id: {pair.animal_id}",
                        f"myelin_label: {pair.myelin_item.label}",
                        f"nissl_label: {pair.nissl_item.label}",
                        f"registration_status: {status}",
                        f"flip_myelin_lr: {bool(flip.get('myelin', False))}",
                        f"flip_nissl_lr: {bool(flip.get('nissl', False))}",
                        f"myelin_group_flip_lr: {json.dumps(self.step4_group_flips.get('myelin', {}), ensure_ascii=True)}",
                        f"nissl_group_flip_lr: {json.dumps(self.step4_group_flips.get('nissl', {}), ensure_ascii=True)}",
                        f"myelin_component_selection: {self._step4_selection_text('myelin')}",
                        f"nissl_component_selection: {self._step4_selection_text('nissl')}",
                        "",
                        "Save behavior:",
                        "- if no component is explicitly marked on one side, only the largest combined component is kept as group 1",
                        "- if groups 1 and 2 are marked, both are kept and later registration should consider two pairings: 1<->1 and 2<->2",
                    ]
                )
            )

    def set_step4_active_editor(self, editor: MaskEditorLabel, name: str) -> None:
        self.step4_active_editor = editor
        self.step4_active_side = name
        self.step4_active_editor_label.setText(f"Active panel (keyboard target): {name}")
        self.step4_layer_combo.blockSignals(True)
        self.step4_layer_combo.setCurrentText(editor.active_layer)
        self.step4_layer_combo.blockSignals(False)
        self._sync_step4_group_flip_controls()

    def on_step4_editor_active_layer_changed(self, layer: str) -> None:
        self.step4_layer_combo.blockSignals(True)
        self.step4_layer_combo.setCurrentText(layer)
        self.step4_layer_combo.blockSignals(False)
        self.set_step4_active_layer_all(layer)

    def set_step4_active_layer_all(self, layer: str) -> None:
        self.step4_myelin_editor.set_active_layer(layer)
        self.step4_nissl_editor.set_active_layer(layer)

    def set_step4_brush_radius_all(self, radius: int) -> None:
        self.step4_myelin_editor.set_brush_radius(radius)
        self.step4_nissl_editor.set_brush_radius(radius)

    def on_step4_flip_changed(self) -> None:
        self.step4_myelin_editor.set_mirror(self.step4_myelin_flip_check.isChecked())
        self.step4_nissl_editor.set_mirror(self.step4_nissl_flip_check.isChecked())

    def on_step4_group_flip_changed(self, group_id: int, checked: bool) -> None:
        side = self.step4_active_side
        mapping = dict(self.step4_group_flips.get(side, {}))
        mapping[int(group_id)] = bool(checked)
        self.step4_group_flips[side] = mapping
        self.update_step4_mask_stats()

    def _invalidate_step4_side_component_groups(self, side: str) -> None:
        self.step4_component_groups[side] = {}
        self._sync_step4_component_marks()
        self.update_step4_mask_stats()

    def on_step4_editor_mask_changed(self, side: str) -> None:
        if self.step4_preserve_component_marks_once.get(side, False):
            self.step4_preserve_component_marks_once[side] = False
            self._sync_step4_component_marks()
        else:
            self.step4_component_groups[side] = {}
            self._sync_step4_component_marks()
        self.update_step4_mask_stats()

    def close_fill_step4_active_editor(self) -> None:
        editor = self.step4_active_editor or self.step4_myelin_editor
        side = self.step4_active_side
        self.step4_preserve_component_marks_once[side] = True
        editor.close_and_fill_tissue_gaps()

    def undo_step4_active_editor(self) -> None:
        editor = self.step4_active_editor or self.step4_myelin_editor
        if editor.undo_last_action():
            self.step4_component_groups[self.step4_active_side] = {}
            self.update_step4_mask_stats()

    def mark_step4_hovered_component(self, group_id: int) -> None:
        self.mark_step4_hovered_component_for_side(self.step4_active_side, group_id)

    def mark_step4_hovered_component_for_side(self, side: str, group_id: int) -> None:
        editor = self._step4_editor_for_side(side)
        rank = editor.hovered_combined_component_rank()
        if rank is None:
            self.step4_info.append(f"No combined component under cursor on {side}.")
            return
        mapping = dict(self.step4_component_groups.get(side, {}))
        for existing_rank, existing_group in list(mapping.items()):
            if existing_group == group_id or existing_rank == rank:
                mapping.pop(existing_rank, None)
        mapping[rank] = group_id
        self.step4_component_groups[side] = mapping
        self._sync_step4_component_marks()
        self.update_step4_mask_stats()

    def clear_step4_hovered_component_mark(self) -> None:
        side = self.step4_active_side
        editor = self._step4_editor_for_side(side)
        rank = editor.hovered_combined_component_rank()
        if rank is None:
            self.step4_info.append(f"No combined component under cursor on {side}.")
            return
        mapping = dict(self.step4_component_groups.get(side, {}))
        mapping.pop(rank, None)
        self.step4_component_groups[side] = mapping
        self._sync_step4_component_marks()
        self.update_step4_mask_stats()

    def reset_step4_focused_component_marks(self) -> None:
        self.step4_component_groups[self.step4_active_side] = {}
        self._sync_step4_component_marks()
        self.update_step4_mask_stats()

    def _apply_step4_component_selection(self, side: str) -> dict[str, Any]:
        editor = self._step4_editor_for_side(side)
        summary = editor.combined_component_summary()
        component_count = len(summary)
        normalized = self._normalized_step4_group_map(self.step4_component_groups.get(side, {}), component_count)
        keep_ranks = set(normalized.keys())
        kept = editor.keep_only_combined_component_ranks(keep_ranks)
        normalized = {rank: group for rank, group in normalized.items() if rank in kept}
        self.step4_component_groups[side] = normalized
        self._sync_step4_component_marks()
        group_to_ranks: dict[str, list[int]] = {}
        for rank, group in sorted(normalized.items()):
            group_to_ranks.setdefault(str(group), []).append(int(rank))
        return {
            "kept_ranks": sorted(int(rank) for rank in kept),
            "component_groups": group_to_ranks,
            "component_count_before_trim": component_count,
            "multi_group": "2" in group_to_ranks,
        }

    def save_current_pair_state(self) -> bool:
        pair = self._current_pair()
        if pair is None or self.step4_myelin_crop_rgb is None or self.step4_nissl_crop_rgb is None:
            return False
        if not self._ensure_step4_review_status_selected():
            return False
        my_selection = self._apply_step4_component_selection("myelin")
        ni_selection = self._apply_step4_component_selection("nissl")
        my_tissue, my_artifact = self.step4_myelin_editor.current_masks()
        ni_tissue, ni_artifact = self.step4_nissl_editor.current_masks()
        reg_paths = self._step4_registration_mask_paths(pair)
        my_reg_path = reg_paths.get("myelin")
        ni_reg_path = reg_paths.get("nissl")
        if my_reg_path is None or ni_reg_path is None:
            QMessageBox.warning(self, "Registration Save Failed", "Registration mask root is not available.")
            return False
        write_mask_labels_file(my_reg_path, my_tissue, my_artifact)
        write_mask_labels_file(ni_reg_path, ni_tissue, ni_artifact)
        registry_path = self._step4_registry_path()
        if registry_path is not None:
            registration_status = str(self.step4_registration_status_combo.currentData() or "unreviewed")
            multi_group_registration = bool(my_selection["multi_group"] and ni_selection["multi_group"])
            self.step4_pair_registry[pair.pair_key] = {
                "animal_id": pair.animal_id,
                "myelin_label": pair.myelin_item.label,
                "nissl_label": pair.nissl_item.label,
                "registration_status": registration_status,
                "registration_usable": registration_status == "usable",
                "component_groups": {
                    "myelin": my_selection["component_groups"],
                    "nissl": ni_selection["component_groups"],
                },
                "kept_component_ranks": {
                    "myelin": my_selection["kept_ranks"],
                    "nissl": ni_selection["kept_ranks"],
                },
                "multi_group_registration": multi_group_registration,
                "flip_lr": {
                    "myelin": bool(self.step4_myelin_flip_check.isChecked()),
                    "nissl": bool(self.step4_nissl_flip_check.isChecked()),
                },
                "group_flip_lr": {
                    "myelin": {str(group): bool(flag) for group, flag in sorted(self.step4_group_flips.get("myelin", {}).items()) if int(group) in {1, 2}},
                    "nissl": {str(group): bool(flag) for group, flag in sorted(self.step4_group_flips.get("nissl", {}).items()) if int(group) in {1, 2}},
                },
                "registration_mask_files": {
                    "myelin": self._step4_registration_mask_relpath(my_reg_path),
                    "nissl": self._step4_registration_mask_relpath(ni_reg_path),
                },
                "saved_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
            save_pair_registry(registry_path, self.step4_pair_registry)
        self._remove_step4_pair_cache(pair.pair_key)
        self.step4_info.append(
            "Saved pair QC and pair-specific registration masks."
            + (" This pair needs 1<->1 and 2<->2 registration." if multi_group_registration else "")
        )
        item = self.step4_pair_list.item(self.current_pair_index) if 0 <= self.current_pair_index < self.step4_pair_list.count() else None
        if item is not None and pair is not None:
            item.setText(self._step4_pair_display_text(pair))
        self._update_step4_root_status_text()
        return True

    def save_and_next_pair(self) -> None:
        old_index = self.current_pair_index
        if not self.save_current_pair_state():
            return
        if not self.step4_pairs:
            return
        target_index = min(old_index + 1, len(self.step4_pairs) - 1)
        self.current_pair_index = target_index
        if self.step4_pair_list.currentRow() != target_index:
            self.step4_pair_list.setCurrentRow(target_index)
        self.load_current_pair()

    def prev_pair(self) -> None:
        if not self.step4_pairs:
            return
        self.current_pair_index = max(0, self.current_pair_index - 1)
        self.step4_pair_list.setCurrentRow(self.current_pair_index)

    def next_pair(self) -> None:
        if not self.step4_pairs:
            return
        self.current_pair_index = min(len(self.step4_pairs) - 1, self.current_pair_index + 1)
        self.step4_pair_list.setCurrentRow(self.current_pair_index)

    def export_crop_workspaces(self) -> None:
        slide = self.current_slide
        if slide is None:
            return
        if self.export_thread is not None:
            self.stage1_info.append("Crop export is already running.")
            return
        default_out = str(self._default_crop_workspace_root())
        out_dir = QFileDialog.getExistingDirectory(self, "Select Crop Workspace Root", default_out)
        if not out_dir:
            return
        export_root = Path(out_dir)
        plan_items, skipped_labels = self.workflow_service.plan_export(export_root)
        planned_labels = [item.proposal.label for item in plan_items]
        self.stage1_info.append(f"Crop workspace target: {export_root}")
        self.stage1_info.append(
            f"Planner: {len(plan_items)} new crop folder(s), {len(skipped_labels)} existing folder(s) skipped."
        )
        if planned_labels:
            self.stage1_info.append(f"Will write: {', '.join(planned_labels)}")
        if skipped_labels:
            self.stage1_info.append(f"Skipped existing folders: {', '.join(skipped_labels)}")
        if not plan_items:
            self.stage1_info.append("Nothing new to export.")
            self.workspace_root = export_root
            self.refresh_workspace_sections()
            return

        self.export_crops_button.setEnabled(False)
        self.export_crops_button.setText("Exporting Crops...")
        if self.bg_precompute_worker is not None:
            self.bg_precompute_worker.set_paused(True)
        self.bg_precompute_label.setText("Background precompute: paused while exporting")
        crop_level = self.workflow_service.export_crop_level_for_slide(slide)
        self.export_thread = QThread(self)
        self.export_worker = self.workflow_service.create_export_worker(
            export_root,
            crop_level,
            profile_name="crop_workspace",
            include_masks=False,
        )
        self.export_worker.moveToThread(self.export_thread)
        self.export_thread.started.connect(self.export_worker.run)
        self.export_worker.progress.connect(self.stage1_info.append)
        self.export_worker.finished.connect(self.on_export_finished)
        self.export_worker.failed.connect(self.on_export_failed)
        self.export_worker.finished.connect(self.export_thread.quit)
        self.export_worker.failed.connect(self.export_thread.quit)
        self.export_thread.finished.connect(self.export_worker.deleteLater)
        self.export_thread.finished.connect(self.export_thread.deleteLater)
        self.export_thread.start()

    def _reset_export_state(self) -> None:
        self.export_crops_button.setEnabled(True)
        self.export_crops_button.setText("Confirm BBoxes + Export Crop Folders")
        self.export_worker = None
        self.export_thread = None
        if self.bg_precompute_worker is not None:
            self.bg_precompute_worker.set_paused(True)

    def on_export_finished(self, summary: object) -> None:
        data = dict(summary) if isinstance(summary, dict) else {}
        exported = data.get("exported", [])
        skipped_during_write = data.get("skipped_during_write", [])
        export_root = data.get("export_root", "")
        self.stage1_info.append(
            f"Crop export finished. wrote={len(exported)} skipped_during_write={len(skipped_during_write)} root={export_root}"
        )
        if exported:
            self.stage1_info.append(f"Wrote folders: {', '.join(exported)}")
        if skipped_during_write:
            self.stage1_info.append(f"Skipped during write: {', '.join(skipped_during_write)}")
        if export_root:
            self.workspace_root = Path(export_root)
            self.refresh_workspace_sections()
        self._reset_export_state()

    def on_export_failed(self, message: str) -> None:
        self.stage1_info.append(message)
        self._reset_export_state()

    def update_mask_stats(self) -> None:
        usable = self.section_editor.current_usable_mask()
        mask = usable > 0
        if mask.size == 0:
            self.mask_stats_panel.setPlainText("No mask loaded")
            return
        num, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
        areas = sorted([int(stats[i, cv2.CC_STAT_AREA]) for i in range(1, num)], reverse=True)
        total = int(mask.sum())
        preview_areas = ", ".join(str(x) for x in areas[:8]) if areas else "none"
        self.mask_stats_panel.setPlainText(
            "\n".join(
                [
                    f"connected_components: {max(0, num - 1)}",
                    f"total_foreground_px: {total}",
                    f"largest_component_px: {areas[0] if areas else 0}",
                    f"areas_desc: {preview_areas}",
                    "",
                    "Goal usually: 1 dominant connected tissue component",
                ]
            )
        )

    def closeEvent(self, event) -> None:
        if self.bg_precompute_worker is not None:
            self.bg_precompute_worker.stop()
        if self.bg_precompute_thread is not None:
            self.bg_precompute_thread.quit()
            self.bg_precompute_thread.wait(2000)
        self._close_step6_hires_slide_handle()
        super().closeEvent(event)

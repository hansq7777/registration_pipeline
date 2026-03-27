from __future__ import annotations

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
from PySide6.QtCore import QObject, QThread, QRectF, Qt, Signal
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
    default_pair_registration_runs_root,
    find_ants_bin,
    latest_registration_run_dir,
    run_pair_registration,
)
from ..application.roi_mapping import (
    current_step6_state,
    default_pair_roi_root,
    load_approved_registration_context,
    save_step6_roi as save_step6_roi_outputs,
    update_step6_roi_mapping,
)
from ..application.confocal_registration import (
    ConfocalRigidConfig,
    apply_manual_transform,
    default_confocal_registration_root,
    infer_stack_channel_count,
    prepare_myelin_confocal_fixed,
    project_confocal_stack,
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
from ..pipeline_adapters.slide_io import effective_crop_rect_overview, open_slide_handle
from ..repositories import RevisionRepository, SectionRepository
from ..widgets.graphics import DraggableProposalItem, ImageSceneView, qimage_from_rgb_array
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


class WorkflowWindow(QWidget):
    PAGE_HOME = 0
    PAGE_STAGE1 = 1
    PAGE_STAGE2 = 2
    PAGE_STAGE3 = 3
    PAGE_STAGE4 = 4
    PAGE_STAGE5 = 5
    PAGE_STAGE6 = 6
    PAGE_STAGE7 = 7

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
        self.step6_last_updated_nissl_roi_highres: np.ndarray | None = None
        self.step6_last_updated_myelin_roi_highres: np.ndarray | None = None
        self.step7_myelin_root: Path | None = None
        self.step7_sections: list[WorkspaceSection] = []
        self.current_step7_section_index: int = 0
        self.step7_confocal_path: Path | None = None
        self.step7_confocal_projection_u8: np.ndarray | None = None
        self.step7_fixed_rgb: np.ndarray | None = None
        self.step7_fixed_labels: np.ndarray | None = None
        self.step7_run_thread: QThread | None = None
        self.step7_run_worker = None
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
        self.pages.addWidget(self.page_home)
        self.pages.addWidget(self.page_stage1)
        self.pages.addWidget(self.page_stage2)
        self.pages.addWidget(self.page_stage3)
        self.pages.addWidget(self.page_stage4)
        self.pages.addWidget(self.page_stage5)
        self.pages.addWidget(self.page_stage6)
        self.pages.addWidget(self.page_stage7)

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
            "Step 6 maps hand-drawn Nissl ROIs onto Myelin via an approved registration. "
            "Step 7 aligns confocal z-stacks locally onto Myelin."
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
                    "- Step 7: generate confocal focus projection and rigidly align it to Myelin",
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
        self.section_editor.setMinimumSize(900, 700)
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
        self.step4_myelin_editor.setMinimumSize(620, 520)
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
        self.step4_nissl_editor.setMinimumSize(620, 520)
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
        right.addWidget(self.step5_progress_bar)
        right.addWidget(self.step5_progress_label)
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
                    "- approve a run before Step 6 can use it for ROI mapping",
                ]
            )
        )
        right.addWidget(self.step5_info)
        right.addWidget(QLabel("Storyboard"))
        self.step5_storyboard_label = QLabel("No registration storyboard yet")
        self.step5_storyboard_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.step5_storyboard_label.setMinimumSize(900, 680)
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
        top.addWidget(self.step6_refresh_button)
        top.addWidget(self.step6_open_step5_button)
        top.addWidget(self.step6_back_button)
        top.addWidget(self.step6_pair_label)

        body = QHBoxLayout()
        left = QVBoxLayout()
        left.addWidget(QLabel("Pairs With Approved Registration"))
        self.step6_pair_list = QListWidget()
        self.step6_pair_list.currentRowChanged.connect(self.on_step6_pair_changed)
        left.addWidget(self.step6_pair_list)
        self.step6_root_status = QTextEdit()
        self.step6_root_status.setReadOnly(True)
        self.step6_root_status.setMinimumHeight(120)
        left.addWidget(self.step6_root_status)

        center = QVBoxLayout()
        self.step6_nissl_title = QLabel("Nissl ROI")
        center.addWidget(self.step6_nissl_title)
        self.step6_nissl_editor = MaskEditorLabel()
        self.step6_nissl_editor.set_on_mask_changed(self.on_step6_roi_mask_changed)
        self.step6_nissl_editor.set_on_save_and_next_requested(self.save_step6_roi_and_next)
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
        right.addWidget(QLabel("Mapped Myelin ROI"))
        self.step6_myelin_mapped_label = QLabel("No mapped ROI yet")
        self.step6_myelin_mapped_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.step6_myelin_mapped_label.setMinimumSize(900, 680)
        self.step6_myelin_mapped_label.setStyleSheet("background:#f5f5f5; border:1px solid #cccccc;")
        self.step6_myelin_mapped_scroll = QScrollArea()
        self.step6_myelin_mapped_scroll.setWidgetResizable(True)
        self.step6_myelin_mapped_scroll.setWidget(self.step6_myelin_mapped_label)
        right.addWidget(self.step6_myelin_mapped_scroll, 1)
        self.step6_info = QTextEdit()
        self.step6_info.setReadOnly(True)
        self.step6_info.setPlainText(
            "\n".join(
                [
                    "Step 6 ROI Annotation and Mapping",
                    "- draw ROI on the high-resolution Nissl panel",
                    "- Update ROI Mapping downsamples through the approved Step 5 preprocessing chain, applies the approved transform, and refreshes the high-resolution Myelin preview",
                    "- green/yellow highlights show ROI added in the current edit batch; magenta highlights show ROI removed in the current batch",
                    "- Save writes high-resolution ROI outputs plus low-resolution debug canvases",
                    "- S saves and moves to the next approved pair",
                ]
            )
        )
        right.addWidget(self.step6_info)

        body.addLayout(left, 3)
        body.addLayout(center, 5)
        body.addLayout(right, 5)

        layout.addLayout(top)
        layout.addLayout(body)
        page.setLayout(layout)
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
        left.addWidget(QLabel("Myelin Sections"))
        self.step7_section_list = QListWidget()
        self.step7_section_list.currentRowChanged.connect(self.on_step7_section_changed)
        left.addWidget(self.step7_section_list)
        self.step7_root_status = QTextEdit()
        self.step7_root_status.setReadOnly(True)
        self.step7_root_status.setMinimumHeight(120)
        left.addWidget(self.step7_root_status)

        middle = QVBoxLayout()
        select_row = QHBoxLayout()
        self.step7_select_stack_button = QPushButton("Select Confocal Z-Stack")
        self.step7_select_stack_button.clicked.connect(self.select_step7_confocal_stack)
        self.step7_stack_label = QLabel("No confocal stack selected")
        self.step7_stack_label.setWordWrap(True)
        select_row.addWidget(self.step7_select_stack_button)
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
        self.step7_generate_projection_button = QPushButton("Generate Projection")
        self.step7_generate_projection_button.clicked.connect(self.generate_step7_projection)
        proj_row.addWidget(QLabel("Projection"))
        proj_row.addWidget(self.step7_projection_mode_combo)
        proj_row.addWidget(QLabel("Channel"))
        proj_row.addWidget(self.step7_channel_spin)
        proj_row.addWidget(self.step7_generate_projection_button)
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
        self.step7_flip_lr_check = QCheckBox("Flip LR")
        self.step7_flip_ud_check = QCheckBox("Flip UD")
        self.step7_update_preview_button = QPushButton("Update Preview")
        self.step7_update_preview_button.clicked.connect(self.update_step7_preview)
        self.step7_run_button = QPushButton("Run Rigid Refine")
        self.step7_run_button.clicked.connect(self.run_step7_registration)
        manual_row.addWidget(QLabel("tx"))
        manual_row.addWidget(self.step7_tx_spin)
        manual_row.addWidget(QLabel("ty"))
        manual_row.addWidget(self.step7_ty_spin)
        manual_row.addWidget(QLabel("angle"))
        manual_row.addWidget(self.step7_angle_spin)
        manual_row.addWidget(QLabel("scale"))
        manual_row.addWidget(self.step7_scale_spin)
        manual_row.addWidget(self.step7_flip_lr_check)
        manual_row.addWidget(self.step7_flip_ud_check)
        manual_row.addWidget(self.step7_update_preview_button)
        manual_row.addWidget(self.step7_run_button)
        middle.addLayout(manual_row)

        self.step7_progress_label = QLabel("Step 7 progress: idle")
        middle.addWidget(self.step7_progress_label)
        self.step7_info = QTextEdit()
        self.step7_info.setReadOnly(True)
        self.step7_info.setPlainText(
            "\n".join(
                [
                    "Step 7 Confocal to Myelin Local Registration",
                    "- select a myelin section and a confocal z-stack",
                    "- generate a focus / EDF projection or alternative 2D projection",
                    "- adjust tx / ty / angle / scale / flips for coarse manual alignment",
                    "- run rigid refine to produce a local storyboard and metrics",
                ]
            )
        )
        middle.addWidget(self.step7_info)

        right = QVBoxLayout()
        right.addWidget(QLabel("Manual Preview"))
        self.step7_preview_label = QLabel("No confocal projection preview yet")
        self.step7_preview_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.step7_preview_label.setMinimumSize(900, 420)
        self.step7_preview_label.setStyleSheet("background:#f5f5f5; border:1px solid #cccccc;")
        self.step7_preview_scroll = QScrollArea()
        self.step7_preview_scroll.setWidgetResizable(True)
        self.step7_preview_scroll.setWidget(self.step7_preview_label)
        right.addWidget(self.step7_preview_scroll, 1)
        right.addWidget(QLabel("Rigid Storyboard"))
        self.step7_storyboard_label = QLabel("No confocal rigid storyboard yet")
        self.step7_storyboard_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.step7_storyboard_label.setMinimumSize(900, 420)
        self.step7_storyboard_label.setStyleSheet("background:#f5f5f5; border:1px solid #cccccc;")
        self.step7_storyboard_scroll = QScrollArea()
        self.step7_storyboard_scroll.setWidgetResizable(True)
        self.step7_storyboard_scroll.setWidget(self.step7_storyboard_label)
        right.addWidget(self.step7_storyboard_scroll, 1)

        body.addLayout(left, 3)
        body.addLayout(middle, 5)
        body.addLayout(right, 7)

        layout.addLayout(top)
        layout.addLayout(body)
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

    def _default_crop_workspace_root(self) -> Path:
        preferred = Path(r"D:\Research\Image Analysis\Nanozoomer scans")
        if preferred.exists():
            return preferred
        if self.workspace_root is not None and self.workspace_root.exists():
            return self.workspace_root
        if self.current_slide is not None:
            return self.current_slide.slide_path.parent
        return Path("C:/")

    def _default_step4_myelin_root(self) -> Path:
        preferred = Path(
            r"D:\Research\Image Analysis\Nanozoomer scans\20250327 rat myelin quantification\Tissue&Masks"
        )
        return preferred if preferred.exists() else self._default_crop_workspace_root()

    def _default_step4_nissl_root(self) -> Path:
        preferred = Path(
            r"D:\Research\Image Analysis\Nanozoomer scans\20250424 Nissl cytoarchitectonic counterpart\Tissue&Masks"
        )
        return preferred if preferred.exists() else self._default_crop_workspace_root()

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
        latest_storyboard = (latest_run / "storyboard.png") if latest_run is not None else None
        if latest_storyboard is not None and latest_storyboard.exists():
            self._set_step5_storyboard(latest_storyboard)
        else:
            self._set_step5_storyboard(None)
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
                    "",
                    "Registration prep notes:",
                    "- only usable pairs are shown here",
                    "- if multi_group_registration is true, run registration for 1<->1 and 2<->2 separately",
                    "- choose moving/fixed side and group, then run ANTs rigid + affine + SyN",
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

    def on_step5_registration_finished(self, summary: object) -> None:
        data = dict(summary) if isinstance(summary, dict) else {}
        run_dir = data.get("run_dir", "")
        storyboard_path = data.get("storyboard_path", "")
        if storyboard_path:
            self._set_step5_storyboard(Path(str(storyboard_path)))
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

    def _refresh_step6_nissl_batch_overlay(self) -> None:
        if self.step6_current_context is None:
            self.step6_nissl_editor.set_aux_overlay_rgba(None)
            return
        if self.step6_last_updated_nissl_roi_highres is None:
            self.step6_nissl_editor.set_aux_overlay_rgba(None)
            return
        current_roi = self._current_step6_roi_highres()
        diff_overlay = self._step6_diff_overlay_rgba(current_roi, self.step6_last_updated_nissl_roi_highres)
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
        approved_stage = self._latest_completed_stage_from_manifest(manifest)
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

    def _step6_pair_display_text(self, pair: WorkspacePair) -> str:
        review = self._step4_pair_review(pair)
        approved = dict(review.get("approved_registration") or {})
        group_tag = str(approved.get("group_tag") or "all")
        stage = str(approved.get("approved_stage") or "unknown")
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
            return
        all_pairs = list_cross_stain_pairs(self.step4_myelin_root, self.step4_nissl_root)
        self.step6_pairs = [
            pair
            for pair in all_pairs
            if self._step4_registration_status(pair) == "usable"
            and bool((self._step4_pair_review(pair).get("approved_registration") or {}))
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
            self.step6_last_updated_nissl_roi_highres = None
            self.step6_last_updated_myelin_roi_highres = None
            self.step6_pair_label.setText("No approved ROI mapping pair selected")
            self.step6_nissl_editor.set_section(np.full((32, 32, 3), 255, dtype=np.uint8), np.zeros((32, 32), dtype=np.uint8), np.zeros((32, 32), dtype=np.uint8))
            self.step6_nissl_editor.set_aux_overlay_rgba(None)
            self._set_rgb_image_label(self.step6_myelin_mapped_label, None, "No mapped ROI yet")
            self.step6_info.setPlainText("No usable pair currently has an approved Step 5 registration.")
            self._set_step6_stale_state(False)

    def on_step6_pair_changed(self, index: int) -> None:
        if index < 0 or index >= len(self.step6_pairs):
            return
        self.current_step6_pair_index = index
        pair = self.step6_pairs[index]
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
        if context is None:
            self.step6_last_updated_nissl_roi_highres = None
            self.step6_last_updated_myelin_roi_highres = None
            self.step6_pair_label.setText(f"{index + 1}/{len(self.step6_pairs)} | {pair.display_label}")
            self.step6_nissl_editor.set_section(np.full((32, 32, 3), 255, dtype=np.uint8), np.zeros((32, 32), dtype=np.uint8), np.zeros((32, 32), dtype=np.uint8))
            self.step6_nissl_editor.set_aux_overlay_rgba(None)
            self.step6_info.setPlainText("Approved registration metadata is missing or points to files that no longer exist.")
            self._set_rgb_image_label(self.step6_myelin_mapped_label, None, "No mapped ROI yet")
            self._set_step6_stale_state(False)
            return
        state = current_step6_state(context)
        self.step6_last_updated_nissl_roi_highres = np.asarray(state["nissl_roi"], dtype=np.uint8).copy()
        self.step6_last_updated_myelin_roi_highres = np.asarray(state["myelin_roi"], dtype=np.uint8).copy()
        self.step6_nissl_editor.set_section(
            state["nissl_rgb"],
            state["nissl_roi"],
            np.zeros(state["nissl_roi"].shape, dtype=np.uint8),
        )
        self.step6_nissl_editor.set_active_layer("tissue")
        self.step6_nissl_editor.set_aux_overlay_rgba(None)
        self._set_rgb_image_label(self.step6_myelin_mapped_label, state["myelin_overlay"], "No mapped ROI yet")
        self.step6_nissl_title.setText(f"Nissl ROI | {pair.nissl_item.label}")
        self.step6_pair_label.setText(f"{index + 1}/{len(self.step6_pairs)} | {pair.display_label}")
        self._set_step6_stale_state(False)
        approved = dict(review.get("approved_registration") or {})
        self.step6_info.setPlainText(
            "\n".join(
                [
                    f"pair_key: {pair.pair_key}",
                    f"approved_run_dir: {approved.get('run_dir', 'missing')}",
                    f"approved_stage: {approved.get('approved_stage', 'unknown')}",
                    f"group_tag: {approved.get('group_tag', 'all')}",
                    f"roi_output_dir: {context.output_dir}",
                    "",
                    "Editing:",
                    "- draw ROI on the high-resolution Nissl side",
                    "- tissue brush is the intended ROI layer; artifact is ignored on update/save",
                    "- Update ROI Mapping applies the approved run without re-optimizing ANTs",
                    "- green/yellow = current batch added ROI, magenta = current batch removed ROI",
                    "- Save writes the current high-resolution ROI and mapped Myelin ROI",
                    "- S saves and advances to the next approved pair",
                ]
            )
        )

    def update_step6_roi_mapping_preview(self) -> bool:
        context = self.step6_current_context
        ants_bin = find_ants_bin()
        if context is None:
            return False
        if ants_bin is None:
            QMessageBox.warning(self, "Step 6 ROI Mapping", "Could not find a local ANTs installation.")
            return False
        roi_labels_highres = self._current_step6_roi_highres()
        previous_myelin_roi = (
            self.step6_last_updated_myelin_roi_highres.copy()
            if self.step6_last_updated_myelin_roi_highres is not None
            else None
        )
        result = update_step6_roi_mapping(context, roi_labels_highres, ants_bin)
        self.step6_current_mapping_result = result
        state = current_step6_state(context)
        mapped = np.asarray(result["myelin_roi_highres"], dtype=np.uint8)
        myelin_overlay = self._step6_apply_roi_preview(state["myelin_rgb"], mapped, previous_myelin_roi)
        self._set_rgb_image_label(self.step6_myelin_mapped_label, myelin_overlay, "No mapped ROI yet")
        self.step6_last_updated_nissl_roi_highres = roi_labels_highres.copy()
        self.step6_last_updated_myelin_roi_highres = mapped.copy()
        self.step6_nissl_editor.set_aux_overlay_rgba(None)
        self._set_step6_stale_state(False)
        self.step6_info.append(
            "Updated ROI mapping preview using the approved Step 5 transform. "
            "Green/yellow shows newly added ROI in this batch; magenta shows removed ROI."
        )
        return True

    def on_step6_roi_mask_changed(self) -> None:
        if self.step6_current_context is None:
            return
        self.step6_current_mapping_result = None
        self._refresh_step6_nissl_batch_overlay()
        already_stale = self.step6_preview_stale
        self._set_step6_stale_state(True, reason="left ROI changed; right preview is out of date")
        if not already_stale:
            self.step6_info.append(
                "ROI changed: preview mapping is now stale. "
                "Green/yellow marks added ROI in the current edit batch; magenta marks removed ROI. "
                "Click Update ROI Mapping to refresh the Myelin overlay."
            )

    def save_step6_roi(self) -> bool:
        pair = self._current_step6_pair()
        context = self.step6_current_context
        if pair is None or context is None:
            return False
        if self.step6_current_mapping_result is None:
            msg = QMessageBox(self)
            msg.setIcon(QMessageBox.Icon.Warning)
            msg.setWindowTitle("Step 6 ROI Mapping")
            msg.setText("Mapped Myelin preview is stale.")
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
        state = current_step6_state(context)
        self._set_rgb_image_label(self.step6_myelin_mapped_label, state["myelin_overlay"], "No mapped ROI yet")
        self.step6_last_updated_nissl_roi_highres = np.asarray(state["nissl_roi"], dtype=np.uint8).copy()
        self.step6_last_updated_myelin_roi_highres = np.asarray(state["myelin_roi"], dtype=np.uint8).copy()
        self.step6_nissl_editor.set_aux_overlay_rgba(None)
        self._set_step6_stale_state(False)
        registry_path = self._step4_registry_path()
        if registry_path is not None:
            review = self._step4_pair_review(pair)
            review["roi_mapping"] = {
                "output_dir": self._relpath_from_common_root(context.output_dir),
                "manifest_path": self._relpath_from_common_root(context.output_dir / "roi_manifest.json"),
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

    def _current_step7_item(self) -> WorkspaceSection | None:
        if 0 <= self.current_step7_section_index < len(self.step7_sections):
            return self.step7_sections[self.current_step7_section_index]
        return None

    def refresh_step7_sections(self) -> None:
        if self.step7_myelin_root is None:
            self.step7_myelin_root = self._default_step4_myelin_root()
        current_label = self._current_step7_item().label if self._current_step7_item() is not None else None
        self.step7_section_list.clear()
        if self.step7_myelin_root is None or not self.step7_myelin_root.exists():
            self.step7_sections = []
            self.step7_root_status.setPlainText("Step 7 myelin root is not set.")
            return
        self.step7_sections = [
            item
            for item in list_workspace_sections(self.step7_myelin_root)
            if item.stain in {"gallyas", "myelin", ""}
        ]
        for item in self.step7_sections:
            self.step7_section_list.addItem(item.label)
        self.step7_root_status.setPlainText(
            "\n".join(
                [
                    f"myelin_root: {self.step7_myelin_root}",
                    f"confocal_runs_root: {self._step7_runs_root()}",
                    f"myelin_sections: {len(self.step7_sections)}",
                ]
            )
        )
        if self.step7_sections:
            matched_idx = next((i for i, item in enumerate(self.step7_sections) if item.label == current_label), None)
            self.current_step7_section_index = matched_idx if matched_idx is not None else min(self.current_step7_section_index, len(self.step7_sections) - 1)
            self.step7_section_list.setCurrentRow(self.current_step7_section_index)
            self.on_step7_section_changed(self.current_step7_section_index)

    def on_step7_section_changed(self, index: int) -> None:
        if index < 0 or index >= len(self.step7_sections):
            return
        self.current_step7_section_index = index
        item = self.step7_sections[index]
        fixed_rgb, fixed_labels = prepare_myelin_confocal_fixed(item)
        self.step7_fixed_rgb = fixed_rgb
        self.step7_fixed_labels = fixed_labels
        self.step7_pair_label.setText(f"{index + 1}/{len(self.step7_sections)} | {item.label}")
        self.step7_storyboard_label.setText("No confocal rigid storyboard yet")
        self.step7_storyboard_label.setPixmap(QPixmap())
        self.step7_info.setPlainText(
            "\n".join(
                [
                    f"myelin_label: {item.label}",
                    f"section_dir: {item.section_dir}",
                    f"confocal_stack: {self.step7_confocal_path if self.step7_confocal_path is not None else 'none'}",
                    "",
                    "Workflow:",
                    "- select a confocal z-stack",
                    "- generate a 2D projection",
                    "- adjust manual coarse alignment",
                    "- run rigid refine",
                ]
            )
        )
        self.update_step7_preview()

    def select_step7_confocal_stack(self) -> None:
        default_dir = str(self.step7_myelin_root if self.step7_myelin_root is not None else self._default_crop_workspace_root())
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Confocal Z-Stack",
            default_dir,
            "TIFF stacks (*.tif *.tiff);;All Files (*)",
        )
        if not file_path:
            return
        self.step7_confocal_path = Path(file_path)
        self.step7_confocal_projection_u8 = None
        self.step7_stack_label.setText(str(self.step7_confocal_path))
        self.step7_storyboard_label.setText("No confocal rigid storyboard yet")
        self.step7_storyboard_label.setPixmap(QPixmap())
        self.step7_preview_label.setText("Generate a projection to preview coarse alignment.")
        self.step7_preview_label.setPixmap(QPixmap())

    def generate_step7_projection(self) -> None:
        if self.step7_confocal_path is None:
            QMessageBox.warning(self, "Step 7 Confocal", "Select a confocal z-stack first.")
            return
        try:
            import tifffile
        except Exception as exc:
            QMessageBox.warning(self, "Step 7 Confocal", f"tifffile is required:\n{exc}")
            return
        stack = np.asarray(tifffile.imread(str(self.step7_confocal_path)))
        channel_count = infer_stack_channel_count(stack)
        self.step7_channel_spin.setMaximum(max(0, channel_count - 1))
        channel_index = int(self.step7_channel_spin.value())
        mode = str(self.step7_projection_mode_combo.currentData() or "focus")
        self.step7_confocal_projection_u8 = project_confocal_stack(stack, mode=mode, channel_index=channel_index)
        self.step7_info.append(
            f"Generated confocal projection: mode={mode} channel={channel_index} shape={self.step7_confocal_projection_u8.shape}"
        )
        self.update_step7_preview()

    def update_step7_preview(self) -> None:
        if self.step7_fixed_rgb is None or self.step7_fixed_labels is None:
            self._set_rgb_image_label(self.step7_preview_label, None, "No myelin section selected")
            return
        if self.step7_confocal_projection_u8 is None:
            self._set_rgb_image_label(self.step7_preview_label, self.step7_fixed_rgb, "No confocal projection preview yet")
            return
        warped, _warped_mask, _mat = apply_manual_transform(
            self.step7_confocal_projection_u8,
            self.step7_fixed_rgb.shape[:2],
            tx_px=float(self.step7_tx_spin.value()),
            ty_px=float(self.step7_ty_spin.value()),
            angle_deg=float(self.step7_angle_spin.value()),
            scale=float(self.step7_scale_spin.value()),
            flip_lr=bool(self.step7_flip_lr_check.isChecked()),
            flip_ud=bool(self.step7_flip_ud_check.isChecked()),
        )
        fixed_gray_u8 = cv2.cvtColor(self.step7_fixed_rgb, cv2.COLOR_RGB2GRAY)
        base = cv2.cvtColor(fixed_gray_u8, cv2.COLOR_GRAY2RGB)
        preview = base.copy()
        confocal_rgb = cv2.cvtColor(warped, cv2.COLOR_GRAY2RGB)
        mask = warped > 0
        if np.any(mask):
            preview[mask] = np.clip(
                0.55 * preview[mask].astype(np.float32) + 0.45 * confocal_rgb[mask].astype(np.float32),
                0,
                255,
            ).astype(np.uint8)
        self._set_rgb_image_label(self.step7_preview_label, preview, "No confocal projection preview yet")

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
        if self.step7_confocal_projection_u8 is None or self.step7_confocal_path is None:
            QMessageBox.warning(self, "Step 7 Confocal", "Generate a confocal projection first.")
            return
        if runs_root is None:
            QMessageBox.warning(self, "Step 7 Confocal", "Confocal registration output root is not available.")
            return
        if ants_bin is None:
            QMessageBox.warning(self, "Step 7 Confocal", "Could not find a local ANTs installation.")
            return
        cfg = ConfocalRigidConfig(
            myelin_label=item.label,
            myelin_rgb=self.step7_fixed_rgb,
            myelin_labels=self.step7_fixed_labels,
            confocal_projection_u8=self.step7_confocal_projection_u8,
            ants_bin=ants_bin,
            out_root=runs_root,
            confocal_source=self.step7_confocal_path,
            projection_mode=str(self.step7_projection_mode_combo.currentData() or "focus"),
            channel_index=int(self.step7_channel_spin.value()),
            tx_px=float(self.step7_tx_spin.value()),
            ty_px=float(self.step7_ty_spin.value()),
            angle_deg=float(self.step7_angle_spin.value()),
            scale=float(self.step7_scale_spin.value()),
            flip_lr=bool(self.step7_flip_lr_check.isChecked()),
            flip_ud=bool(self.step7_flip_ud_check.isChecked()),
        )
        self.step7_run_button.setEnabled(False)
        self.step7_progress_label.setText("Step 7 progress: running rigid refine ...")
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

    def on_step7_registration_finished(self, summary: object) -> None:
        data = dict(summary) if isinstance(summary, dict) else {}
        storyboard = Path(str(data.get("storyboard_path", ""))) if data.get("storyboard_path") else None
        self.step7_progress_label.setText("Step 7 progress: rigid refine finished")
        self.step7_run_button.setEnabled(True)
        self.step7_run_worker = None
        self.step7_run_thread = None
        self.step7_info.append(f"Rigid refine finished. run_dir={data.get('run_dir', '')}")
        if storyboard is not None and storyboard.exists():
            pixmap = QPixmap(str(storyboard))
            if not pixmap.isNull():
                self.step7_storyboard_label.setText("")
                self.step7_storyboard_label.setPixmap(pixmap)
                self.step7_storyboard_label.resize(pixmap.size())
        self._notify_completion("Step 7 confocal rigid registration finished")

    def on_step7_registration_failed(self, message: str) -> None:
        self.step7_info.append(message)
        self.step7_progress_label.setText("Step 7 progress: registration failed")
        self.step7_run_button.setEnabled(True)
        self.step7_run_worker = None
        self.step7_run_thread = None

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
        super().closeEvent(event)

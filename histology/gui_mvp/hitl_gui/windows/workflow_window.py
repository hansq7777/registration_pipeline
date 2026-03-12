from __future__ import annotations

from dataclasses import dataclass
import threading
import traceback
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
from PySide6.QtCore import QObject, QThread, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QPen
from PySide6.QtWidgets import (
    QFileDialog,
    QCheckBox,
    QComboBox,
    QGraphicsRectItem,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QStackedWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ..application import WorkflowService
from ..db import connect_db, transaction
from ..domain import LoadedSlide, ProposalBox
from ..pipeline_adapters import (
    MASK_PRESET_HYBRID_BALANCED,
    MASK_PRESET_LATEST_CONTEXTUAL,
    MASK_PRESET_LEGACY_SIMPLE,
    compute_auto_masks,
    extract_crop_for_preview,
)
from ..pipeline_adapters.slide_io import effective_crop_rect_overview, open_slide_handle
from ..repositories import RevisionRepository, SectionRepository
from ..widgets.graphics import DraggableProposalItem, ImageSceneView
from ..widgets.mask_editor import MaskEditorLabel
from ..widgets.proposal_card import ProposalCard


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


class WorkflowWindow(QWidget):
    PAGE_HOME = 0
    PAGE_STAGE1 = 1
    PAGE_STAGE2 = 2

    def __init__(self, workflow_service: WorkflowService) -> None:
        super().__init__()
        self.setWindowTitle("Histology HITL Workflow")
        self.workflow_service = workflow_service

        self.current_folder: Path | None = None
        self.current_proposal_index: int = 0
        self.proposal_items: list[DraggableProposalItem] = []
        self.crop_outline_items: list[QGraphicsRectItem] = []
        self.proposal_cards: list[ProposalCard] = []
        self.default_ndpi_root = Path("D:/Research/Image Analysis/Nanozoomer scans")
        self.export_thread: QThread | None = None
        self.export_worker = None
        self.save_thread: QThread | None = None
        self.save_worker = None
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
        self.pages.addWidget(self.page_home)
        self.pages.addWidget(self.page_stage1)
        self.pages.addWidget(self.page_stage2)

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
            "Choose a workflow step to enter. Step 1 prepares slide preview and proposal boxes. "
            "Step 2 refines tissue/artifact masks and exports reviewed outputs. "
            "Additional steps are reserved for later expansion."
        )
        subtitle.setWordWrap(True)

        self.step1_entry_button = QPushButton("Step 1: Histology Preview and Box Proposal")
        self.step1_entry_button.setMinimumHeight(52)
        self.step1_entry_button.clicked.connect(self.goto_stage1)

        self.step2_entry_button = QPushButton("Step 2: Mask Generation and Annotation")
        self.step2_entry_button.setMinimumHeight(52)
        self.step2_entry_button.clicked.connect(self.goto_stage2)

        self.future_step3_button = QPushButton("Step 3: Reserved For Pair Review")
        self.future_step3_button.setMinimumHeight(44)
        self.future_step3_button.setEnabled(False)

        self.future_step4_button = QPushButton("Step 4: Reserved For Export / Registration Prep")
        self.future_step4_button.setMinimumHeight(44)
        self.future_step4_button.setEnabled(False)

        self.home_status = QTextEdit()
        self.home_status.setReadOnly(True)
        self.home_status.setMinimumHeight(160)
        self.home_status.setPlainText(
            "\n".join(
                [
                    "Current step entry points:",
                    "- Step 1: load NDPI, inspect slide thumbnail, adjust proposal boxes",
                    "- Step 2: generate masks, annotate tissue/artifact, export reviewed crops",
                    "",
                    "Current session:",
                    "- no slide loaded",
                    "- Step 2 requires a slide with at least one proposal",
                ]
            )
        )

        layout.addWidget(title)
        layout.addWidget(subtitle)
        layout.addSpacing(12)
        layout.addWidget(self.step1_entry_button)
        layout.addWidget(self.step2_entry_button)
        layout.addWidget(self.future_step3_button)
        layout.addWidget(self.future_step4_button)
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
        self.next_step_button = QPushButton("Go To Step 2: Mask Generation and Annotation")
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

        top = QHBoxLayout()
        self.prev_section_button = QPushButton("Prev")
        self.prev_section_button.clicked.connect(self.prev_section)
        self.next_section_button = QPushButton("Next")
        self.next_section_button.clicked.connect(self.next_section)
        self.back_button = QPushButton("Back To Step Menu")
        self.back_button.clicked.connect(self.goto_home)
        self.section_label = QLabel("No section selected")
        top.addWidget(self.prev_section_button)
        top.addWidget(self.next_section_button)
        top.addWidget(self.back_button)
        top.addWidget(self.section_label)

        main = QHBoxLayout()
        self.section_editor = MaskEditorLabel()
        self.section_editor.setMinimumSize(900, 700)
        self.section_editor.set_on_mask_changed(self.update_mask_stats)
        self.section_editor.set_on_painting_state_changed(self.on_editor_painting_state_changed)

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
        self.mask_preset_combo.addItem("Latest Contextual", MASK_PRESET_LATEST_CONTEXTUAL)
        self.mask_preset_combo.addItem("Hybrid Balanced (Best GT)", MASK_PRESET_HYBRID_BALANCED)
        self.mask_preset_combo.addItem("Legacy Simple", MASK_PRESET_LEGACY_SIMPLE)
        self.mask_preset_combo.currentIndexChanged.connect(self.on_mask_preset_changed)
        controls.addWidget(self.mask_preset_combo)

        self.mirror_check = QCheckBox("Mirror LR")
        self.mirror_check.toggled.connect(self.section_editor.set_mirror)
        controls.addWidget(self.mirror_check)

        self.auto_mask_button = QPushButton("Run/Refresh Auto Mask")
        self.auto_mask_button.clicked.connect(self.refresh_current_mask)
        self.keep_largest_button = QPushButton("Keep Largest Tissue Component")
        self.keep_largest_button.clicked.connect(self.keep_largest_tissue_component)
        self.shrink_mask_button = QPushButton("Shrink Active Layer")
        self.shrink_mask_button.clicked.connect(self.shrink_active_layer)
        self.expand_mask_button = QPushButton("Expand Active Layer")
        self.expand_mask_button.clicked.connect(self.expand_active_layer)
        self.close_fill_button = QPushButton("Close + Fill Tissue Gaps")
        self.close_fill_button.clicked.connect(self.close_and_fill_tissue_gaps)
        self.save_revision_button = QPushButton("Save Current Revision")
        self.save_revision_button.clicked.connect(self.save_current_revision_state)
        self.export_button = QPushButton("Save All Crops + Masks")
        self.export_button.clicked.connect(self.export_all_sections)
        controls.addWidget(self.auto_mask_button)
        controls.addWidget(self.keep_largest_button)
        controls.addWidget(self.shrink_mask_button)
        controls.addWidget(self.expand_mask_button)
        controls.addWidget(self.close_fill_button)
        controls.addWidget(self.save_revision_button)
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
                        "- Step 1: load NDPI, inspect slide thumbnail, adjust proposal boxes",
                        "- Step 2: generate masks, annotate tissue/artifact, export reviewed crops",
                        "",
                        "Current session:",
                        "- no slide loaded",
                        "- Step 2 requires a slide with at least one proposal",
                    ]
                )
            )
            return
        self.home_status.setPlainText(
            "\n".join(
                [
                    "Current step entry points:",
                    "- Step 1: histology preview and box proposal",
                    "- Step 2: mask generation and annotation",
                    "- future steps remain reserved",
                    "",
                    "Current session:",
                    f"- slide: {slide.slide_name}",
                    f"- stain: {slide.stain}",
                    f"- backend: {slide.backend}",
                    f"- proposals: {len(slide.proposals)}",
                    f"- Step 2 ready: {'yes' if slide.proposals else 'no'}",
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
        if self.current_slide is None or not self.current_slide.proposals:
            self.refresh_home_status()
            if self.current_slide is None:
                self.home_status.append("")
                self.home_status.append("Step 2 blocked: load a slide in Step 1 first.")
            else:
                self.home_status.append("")
                self.home_status.append("Step 2 blocked: current slide has no proposals.")
            self.pages.setCurrentIndex(self.PAGE_HOME)
            return
        self.current_proposal_index = 0
        self.pages.setCurrentIndex(self.PAGE_STAGE2)
        self.load_current_section_for_edit()

    def load_current_section_for_edit(self) -> None:
        slide = self.current_slide
        if slide is None:
            return
        proposal, crop_rgb, tissue, artifact = self.workflow_service.get_section_for_edit(
            self.current_proposal_index,
            crop_level=self.workflow_service.mask_work_crop_level_for_slide(slide),
            mask_method=slide.proposals[self.current_proposal_index].mask_preset,
        )
        self.section_editor.set_section(crop_rgb, tissue, artifact)
        self.section_label.setText(f"{self.current_proposal_index+1}/{len(slide.proposals)} | {proposal.label}")
        self.mask_preset_combo.blockSignals(True)
        preset_index = self.mask_preset_combo.findData(proposal.mask_preset)
        self.mask_preset_combo.setCurrentIndex(max(0, preset_index))
        self.mask_preset_combo.blockSignals(False)
        self.mirror_check.blockSignals(True)
        self.mirror_check.setChecked(proposal.mirror_enabled)
        self.mirror_check.blockSignals(False)
        self.section_editor.set_mirror(proposal.mirror_enabled)
        self.update_mask_stats()
        self.section_info.setPlainText(
            "\n".join(
                [
                    f"label: {proposal.label}",
                    f"sample_id: {proposal.sample_id}",
                    f"section_id: {proposal.section_id}",
                    f"bbox_overview: {proposal.bbox_dict()}",
                    f"mask_preset: {proposal.mask_preset}",
                    f"mask_work_level: {proposal.mask_work_level}",
                    f"mask_work_shape: {proposal.mask_work_shape}",
                    f"latest_revision_id: {proposal.latest_revision_id or 'none'}",
                    "",
                    "Editing:",
                    "- left mouse: paint active layer",
                    "- right mouse: erase active layer",
                    "- choose tissue/artifact in the layer selector",
                    "- choose latest contextual, hybrid balanced, or legacy simple auto-mask preset",
                    "- shrink/expand buttons apply to the active layer",
                    "- current GUI editing uses a lower working crop level for speed",
                    "- export remaps masks onto full-resolution crop output",
                    "- optional mirror toggle for orientation validation",
                ]
            )
        )
        self._schedule_background_precompute()

    def refresh_current_mask(self) -> None:
        slide = self.current_slide
        if slide is None:
            return
        preset = self.current_mask_preset()
        crop_rgb, tissue, artifact = self.workflow_service.generate_mask_preview(
            self.current_proposal_index,
            crop_level=self.workflow_service.mask_work_crop_level_for_slide(slide),
            mask_method=preset,
        )
        self.section_editor.set_section(crop_rgb, tissue, artifact)
        self.update_mask_stats()
        self.section_info.append(f"Auto-mask refreshed with preset: {preset}")

    def current_mask_preset(self) -> str:
        data = self.mask_preset_combo.currentData()
        if not isinstance(data, str):
            return MASK_PRESET_LATEST_CONTEXTUAL
        return data

    def on_mask_preset_changed(self) -> None:
        slide = self.current_slide
        if slide is None or not slide.proposals:
            return
        preset = self.current_mask_preset()
        slide.proposals[self.current_proposal_index].mask_preset = preset
        self.section_info.append(f"Mask preset selected: {preset}")

    def keep_largest_tissue_component(self) -> None:
        self.section_editor.keep_largest_tissue_component()
        self.update_mask_stats()

    def close_and_fill_tissue_gaps(self) -> None:
        self.section_editor.close_and_fill_tissue_gaps()
        self.update_mask_stats()

    def shrink_active_layer(self) -> None:
        self.section_editor.morph_active_layer("shrink")
        self.update_mask_stats()

    def expand_active_layer(self) -> None:
        self.section_editor.morph_active_layer("expand")
        self.update_mask_stats()

    def save_current_revision_state(self) -> None:
        self._request_save_current_revision()

    def _request_save_current_revision(self, after_save: Callable[[], None] | None = None) -> None:
        slide = self.current_slide
        if slide is None:
            return
        if self.save_thread is not None:
            self.section_info.append("Save is already running.")
            return
        tissue, artifact = self.section_editor.current_masks()
        proposal = slide.proposals[self.current_proposal_index]
        mirror_enabled = self.mirror_check.isChecked()
        proposal.tissue_mask_final = tissue.copy()
        proposal.artifact_mask_final = artifact.copy()
        proposal.mask_work_shape = tuple(int(x) for x in tissue.shape[:2])
        proposal.mirror_enabled = mirror_enabled
        proposal.notes = ""
        self._after_save_action = after_save

        section_uid = self.workflow_service.section_uid(slide, proposal)
        db_row = self.workflow_service.conn.execute("PRAGMA database_list").fetchone()
        if db_row is None:
            self.section_info.append("Save revision failed: database path unavailable.")
            self._after_save_action = None
            return
        db_path = Path(str(db_row["file"] if "file" in db_row.keys() else db_row[2]))
        workspace_root = self.workflow_service.revision_repository.workspace_root

        self._set_save_busy(True)
        self.section_info.append(f"Saving revision for {proposal.label}...")

        self.save_thread = QThread(self)
        self.save_worker = SaveRevisionWorker(
            db_path=db_path,
            workspace_root=workspace_root,
            section_uid=section_uid,
            tissue_mask=tissue.copy(),
            artifact_mask=artifact.copy(),
            mirror_enabled=mirror_enabled,
            bbox_overview=proposal.bbox_dict(),
            notes="",
        )
        self.save_worker.moveToThread(self.save_thread)
        self.save_thread.started.connect(self.save_worker.run)
        self.save_worker.finished.connect(self.on_save_revision_finished)
        self.save_worker.failed.connect(self.on_save_revision_failed)
        self.save_worker.finished.connect(self.save_thread.quit)
        self.save_worker.failed.connect(self.save_thread.quit)
        self.save_thread.finished.connect(self.save_worker.deleteLater)
        self.save_thread.finished.connect(self.save_thread.deleteLater)
        self.save_thread.start()

    def prev_section(self) -> None:
        slide = self.current_slide
        if slide is None:
            return
        target_index = max(0, self.current_proposal_index - 1)
        self._request_save_current_revision(after_save=lambda: self._load_section_after_save(target_index))

    def next_section(self) -> None:
        slide = self.current_slide
        if slide is None:
            return
        target_index = min(len(slide.proposals) - 1, self.current_proposal_index + 1)
        self._request_save_current_revision(after_save=lambda: self._load_section_after_save(target_index))

    def export_all_sections(self) -> None:
        slide = self.current_slide
        if slide is None:
            return
        if self.export_thread is not None:
            self.section_info.append("Export is already running.")
            return
        self._request_save_current_revision(after_save=self._start_export_after_save)

    def _load_section_after_save(self, target_index: int) -> None:
        self.current_proposal_index = target_index
        self.load_current_section_for_edit()

    def _start_export_after_save(self) -> None:
        slide = self.current_slide
        if slide is None:
            return
        default_out = str(slide.slide_path.parent)
        out_dir = QFileDialog.getExistingDirectory(self, "Select Export Folder", default_out)
        if not out_dir:
            return
        export_root = Path(out_dir)
        plan_items, skipped_labels = self.workflow_service.plan_export(export_root)
        planned_labels = [item.proposal.label for item in plan_items]
        self.section_info.append(f"Export target: {export_root}")
        self.section_info.append(
            f"Export planner: {len(plan_items)} new folder(s), {len(skipped_labels)} existing folder(s) skipped."
        )
        if planned_labels:
            self.section_info.append(f"Will write: {', '.join(planned_labels)}")
        if skipped_labels:
            self.section_info.append(f"Skipped existing folders: {', '.join(skipped_labels)}")
        if not plan_items:
            self.section_info.append("Nothing new to export.")
            return

        self.export_button.setEnabled(False)
        self.export_button.setText("Export Running...")
        if self.bg_precompute_worker is not None:
            self.bg_precompute_worker.set_paused(True)
        self.bg_precompute_label.setText("Background precompute: paused while exporting")
        crop_level = self.workflow_service.export_crop_level_for_slide(slide)
        self.export_thread = QThread(self)
        self.export_worker = self.workflow_service.create_export_worker(export_root, crop_level, profile_name="review_mask")
        self.export_worker.moveToThread(self.export_thread)
        self.export_thread.started.connect(self.export_worker.run)
        self.export_worker.progress.connect(self.section_info.append)
        self.export_worker.finished.connect(self.on_export_finished)
        self.export_worker.failed.connect(self.on_export_failed)
        self.export_worker.finished.connect(self.export_thread.quit)
        self.export_worker.failed.connect(self.export_thread.quit)
        self.export_thread.finished.connect(self.export_worker.deleteLater)
        self.export_thread.finished.connect(self.export_thread.deleteLater)
        self.export_thread.start()

    def _set_save_busy(self, busy: bool) -> None:
        self.save_revision_button.setEnabled(not busy)
        self.save_revision_button.setText("Saving..." if busy else "Save Current Revision")
        self.prev_section_button.setEnabled(not busy)
        self.next_section_button.setEnabled(not busy)
        self.auto_mask_button.setEnabled(not busy)
        self.keep_largest_button.setEnabled(not busy)
        self.shrink_mask_button.setEnabled(not busy)
        self.expand_mask_button.setEnabled(not busy)
        self.close_fill_button.setEnabled(not busy)
        self.mask_preset_combo.setEnabled(not busy)
        self.layer_combo.setEnabled(not busy)
        self.brush_spin.setEnabled(not busy)
        self.mirror_check.setEnabled(not busy)
        self.section_editor.setEnabled(not busy)
        if self.export_thread is None:
            self.export_button.setEnabled(not busy)
        if busy:
            if self.bg_precompute_worker is not None:
                self.bg_precompute_worker.set_paused(True)
            self.bg_precompute_label.setText("Background precompute: paused while saving")

    def _reset_save_state(self, *, reschedule_background: bool = True) -> None:
        self._set_save_busy(False)
        if reschedule_background:
            self._schedule_background_precompute()
        elif self.bg_precompute_worker is not None:
            self.bg_precompute_worker.set_paused(True)
        self.save_worker = None
        self.save_thread = None

    def on_save_revision_finished(self, revision_id: str) -> None:
        slide = self.current_slide
        if slide is not None and 0 <= self.current_proposal_index < len(slide.proposals):
            slide.proposals[self.current_proposal_index].latest_revision_id = revision_id
        self.section_info.append(f"Saved revision: {revision_id}")
        self.update_mask_stats()
        after_save = self._after_save_action
        self._after_save_action = None
        self._reset_save_state(reschedule_background=after_save is None)
        if after_save is not None:
            after_save()

    def on_save_revision_failed(self, message: str) -> None:
        self.section_info.append(message)
        self._after_save_action = None
        self._reset_save_state()

    def _reset_export_state(self) -> None:
        self.export_button.setEnabled(True)
        self.export_button.setText("Save All Crops + Masks")
        self.export_worker = None
        self.export_thread = None
        self._schedule_background_precompute()

    def on_export_finished(self, summary: object) -> None:
        data = dict(summary) if isinstance(summary, dict) else {}
        exported = data.get("exported", [])
        skipped_during_write = data.get("skipped_during_write", [])
        export_root = data.get("export_root", "")
        self.section_info.append(
            f"Export finished. wrote={len(exported)} skipped_during_write={len(skipped_during_write)} root={export_root}"
        )
        if exported:
            self.section_info.append(f"Wrote folders: {', '.join(exported)}")
        if skipped_during_write:
            self.section_info.append(f"Skipped during write: {', '.join(skipped_during_write)}")
        self._reset_export_state()

    def on_export_failed(self, message: str) -> None:
        self.section_info.append(message)
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

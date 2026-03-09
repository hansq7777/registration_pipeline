from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QThread, QRectF, Qt
from PySide6.QtWidgets import (
    QFileDialog,
    QCheckBox,
    QComboBox,
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
from ..widgets.graphics import DraggableProposalItem, ImageSceneView
from ..widgets.mask_editor import MaskEditorLabel
from ..widgets.proposal_card import ProposalCard


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
        self.proposal_cards: list[ProposalCard] = []
        self.default_ndpi_root = Path("D:/Research/Image Analysis/Nanozoomer scans")
        self.export_thread: QThread | None = None
        self.export_worker = None

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

        self.mirror_check = QCheckBox("Mirror LR")
        self.mirror_check.toggled.connect(self.section_editor.set_mirror)
        controls.addWidget(self.mirror_check)

        self.auto_mask_button = QPushButton("Run/Refresh Auto Mask")
        self.auto_mask_button.clicked.connect(self.refresh_current_mask)
        self.keep_largest_button = QPushButton("Keep Largest Tissue Component")
        self.keep_largest_button.clicked.connect(self.keep_largest_tissue_component)
        self.close_fill_button = QPushButton("Close + Fill Tissue Gaps")
        self.close_fill_button.clicked.connect(self.close_and_fill_tissue_gaps)
        self.save_revision_button = QPushButton("Save Current Revision")
        self.save_revision_button.clicked.connect(self.save_current_revision_state)
        self.export_button = QPushButton("Save All Crops + Masks")
        self.export_button.clicked.connect(self.export_all_sections)
        controls.addWidget(self.auto_mask_button)
        controls.addWidget(self.keep_largest_button)
        controls.addWidget(self.close_fill_button)
        controls.addWidget(self.save_revision_button)
        controls.addWidget(self.export_button)
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

    def goto_home(self) -> None:
        self.refresh_home_status()
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
                    "- drag red boxes to adjust proposals",
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
        for idx, proposal in enumerate(slide.proposals):
            item = DraggableProposalItem(
                QRectF(float(proposal.x), float(proposal.y), float(proposal.w), float(proposal.h)),
                label=proposal.label,
                on_changed=self.on_proposal_moved,
                on_drag_finished=self.on_proposal_drag_finished,
                on_selected=self.on_proposal_selected,
            )
            item.setData(0, idx)
            self.overview_view.scene_obj.addItem(item)
            self.proposal_items.append(item)

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
            preview = self.workflow_service.get_preview_crop(idx, crop_level=min(4, slide.level_count - 1))
            card.set_preview(preview)
            self.cards_layout.addWidget(card)
            self.proposal_cards.append(card)
        self.cards_layout.addStretch(1)

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
        preview = self.workflow_service.get_preview_crop(idx, crop_level=min(4, slide.level_count - 1))
        self.proposal_cards[idx].set_preview(preview)

    def on_proposal_selected(self, item: DraggableProposalItem) -> None:
        slide = self.current_slide
        if slide is None:
            return
        idx = int(item.data(0))
        proposal = slide.proposals[idx]
        self._set_dimension_inputs_from_proposal(proposal)
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
        preview = self.workflow_service.get_preview_crop(idx, crop_level=min(4, slide.level_count - 1))
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
        crop_rgb, tissue, artifact = self.workflow_service.generate_mask_preview(idx, crop_level=min(3, slide.level_count - 1))
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
        self.load_current_section_for_edit()
        self.pages.setCurrentIndex(self.PAGE_STAGE2)

    def load_current_section_for_edit(self) -> None:
        slide = self.current_slide
        if slide is None:
            return
        proposal, crop_rgb, tissue, artifact = self.workflow_service.get_section_for_edit(
            self.current_proposal_index,
            crop_level=min(3, slide.level_count - 1),
        )
        self.section_editor.set_section(crop_rgb, tissue, artifact)
        self.section_label.setText(f"{self.current_proposal_index+1}/{len(slide.proposals)} | {proposal.label}")
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
                    f"latest_revision_id: {proposal.latest_revision_id or 'none'}",
                    "",
                    "Editing:",
                    "- left mouse: paint active layer",
                    "- right mouse: erase active layer",
                    "- choose tissue/artifact in the layer selector",
                    "- optional mirror toggle for orientation validation",
                ]
            )
        )

    def refresh_current_mask(self) -> None:
        slide = self.current_slide
        if slide is None:
            return
        crop_rgb, tissue, artifact = self.workflow_service.generate_mask_preview(
            self.current_proposal_index,
            crop_level=min(3, slide.level_count - 1),
        )
        self.section_editor.set_section(crop_rgb, tissue, artifact)
        self.update_mask_stats()

    def keep_largest_tissue_component(self) -> None:
        self.section_editor.keep_largest_tissue_component()
        self.update_mask_stats()

    def close_and_fill_tissue_gaps(self) -> None:
        self.section_editor.close_and_fill_tissue_gaps()
        self.update_mask_stats()

    def save_current_revision_state(self) -> None:
        slide = self.current_slide
        if slide is None:
            return
        tissue, artifact = self.section_editor.current_masks()
        revision_id = self.workflow_service.save_revision(
            self.current_proposal_index,
            tissue_mask=tissue,
            artifact_mask=artifact,
            mirror_enabled=self.mirror_check.isChecked(),
            notes="",
        )
        slide.proposals[self.current_proposal_index].latest_revision_id = revision_id
        self.section_info.append(f"Saved revision: {revision_id}")
        self.update_mask_stats()

    def prev_section(self) -> None:
        slide = self.current_slide
        if slide is None:
            return
        self.save_current_revision_state()
        self.current_proposal_index = max(0, self.current_proposal_index - 1)
        self.load_current_section_for_edit()

    def next_section(self) -> None:
        slide = self.current_slide
        if slide is None:
            return
        self.save_current_revision_state()
        self.current_proposal_index = min(len(slide.proposals) - 1, self.current_proposal_index + 1)
        self.load_current_section_for_edit()

    def export_all_sections(self) -> None:
        slide = self.current_slide
        if slide is None:
            return
        if self.export_thread is not None:
            self.section_info.append("Export is already running.")
            return
        self.save_current_revision_state()
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
        crop_level = min(3, slide.level_count - 1)
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

    def _reset_export_state(self) -> None:
        self.export_button.setEnabled(True)
        self.export_button.setText("Save All Crops + Masks")
        self.export_worker = None
        self.export_thread = None

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

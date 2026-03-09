from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QRectF
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QPushButton,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ..proposal_backend import load_ndpi_with_proposals, openslide_available
from ..widgets.graphics import DraggableProposalItem, ImageSceneView


class ProjectWindow(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Histology NDPI Loader")

        self.project_label = QLabel("No NDPI loaded")
        self.slide_list = QListWidget()
        self.import_run_button = QPushButton("Open NDPI Slide")
        self.refresh_button = QPushButton("Clear")
        self.info_panel = QTextEdit()
        self.info_panel.setReadOnly(True)
        self.label_view = ImageSceneView()
        self.overview_view = ImageSceneView()
        self.current_slide = None
        self.proposal_items: list[DraggableProposalItem] = []

        self.import_run_button.clicked.connect(self.open_ndpi_file)
        self.refresh_button.clicked.connect(self.clear_loaded_slide)
        if not openslide_available():
            self.import_run_button.setEnabled(False)
            self.project_label.setText("GUI loaded, but NDPI loading is disabled because openslide is unavailable in this Python environment.")

        controls = QHBoxLayout()
        controls.addWidget(self.import_run_button)
        controls.addWidget(self.refresh_button)

        left_panel = QVBoxLayout()
        left_panel.addWidget(QLabel("Label / Macro Preview"))
        left_panel.addWidget(self.label_view)
        left_panel.addWidget(QLabel("Proposals"))
        left_panel.addWidget(self.slide_list)
        left_panel.addWidget(QLabel("Metadata"))
        left_panel.addWidget(self.info_panel)

        right_panel = QVBoxLayout()
        right_panel.addWidget(QLabel("Overview + Proposal Boxes"))
        right_panel.addWidget(self.overview_view)

        splitter = QSplitter()
        left_widget = QWidget()
        left_widget.setLayout(left_panel)
        right_widget = QWidget()
        right_widget.setLayout(right_panel)
        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        splitter.setSizes([320, 960])

        layout = QVBoxLayout()
        layout.addLayout(controls)
        layout.addWidget(self.project_label)
        layout.addWidget(splitter)
        self.setLayout(layout)

    def open_ndpi_file(self) -> None:
        path_str, _ = QFileDialog.getOpenFileName(self, "Open NDPI Slide", str(Path("C:/")), "NDPI Files (*.ndpi)")
        if not path_str:
            return
        try:
            self.load_slide(Path(path_str))
        except Exception as exc:
            self.info_panel.setPlainText(f"Failed to load NDPI:\n{exc}")

    def clear_loaded_slide(self) -> None:
        self.current_slide = None
        self.project_label.setText("No NDPI loaded")
        self.slide_list.clear()
        self.info_panel.clear()
        self.label_view.clear_all()
        self.overview_view.clear_all()
        self.proposal_items.clear()

    def load_slide(self, slide_path: Path) -> None:
        loaded = load_ndpi_with_proposals(slide_path)
        self.current_slide = loaded
        self.project_label.setText(f"{slide_path.name} | stain={loaded.stain} | proposals={len(loaded.candidates)}")

        label_bytes = loaded.label_preview.tobytes("raw", "RGB")
        self.label_view.set_rgb_image(loaded.label_preview.width, loaded.label_preview.height, label_bytes)

        overview_bytes = loaded.overview.tobytes("raw", "RGB")
        self.overview_view.set_rgb_image(loaded.overview.width, loaded.overview.height, overview_bytes)

        self.slide_list.clear()
        self.proposal_items.clear()
        for idx, cand in enumerate(loaded.candidates, start=1):
            label = cand.section.short_label if cand.section else f"cand_{idx}"
            self.slide_list.addItem(f"{idx}. {label} | x={cand.x} y={cand.y} w={cand.w} h={cand.h}")
            item = DraggableProposalItem(
                QRectF(float(cand.x), float(cand.y), float(cand.w), float(cand.h)),
                label=label,
                on_changed=self.on_proposal_moved,
            )
            self.overview_view.scene_obj.addItem(item)
            self.proposal_items.append(item)

        self.info_panel.setPlainText(
            "\n".join(
                [
                    f"slide_path: {loaded.slide_path}",
                    f"stain: {loaded.stain}",
                    f"expected_labels: {', '.join(loaded.expected_labels)}",
                    "",
                    "Interaction:",
                    "- left drag a red proposal box to move it",
                    "- this page is the first-pass NDPI loader/reviewer",
                ]
            )
        )

    def on_proposal_moved(self, item: DraggableProposalItem) -> None:
        rect = item.sceneBoundingRect()
        for idx, proposal_item in enumerate(self.proposal_items):
            if proposal_item is item:
                self.slide_list.item(idx).setText(
                    f"{idx+1}. {item.label} | x={int(round(rect.x()))} y={int(round(rect.y()))} "
                    f"w={int(round(rect.width()))} h={int(round(rect.height()))}"
                )
                break

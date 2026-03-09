from __future__ import annotations

from PySide6.QtWidgets import (
    QComboBox,
    QLabel,
    QListWidget,
    QPushButton,
    QSlider,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


class SectionReviewWindow(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Section Review")

        self.canvas = QLabel("Section canvas placeholder")
        self.layer_list = QListWidget()
        self.layer_list.addItems(
            [
                "raw_crop",
                "tissue_mask_auto",
                "artifact_mask_auto",
                "tissue_mask_final",
                "artifact_mask_final",
                "usable_tissue_mask",
                "foreground_rgba",
                "foreground_rgb_white",
                "foreground_rgb_black",
            ]
        )
        self.tool_select = QComboBox()
        self.tool_select.addItems(
            [
                "brush_tissue",
                "brush_artifact",
                "eraser",
                "polygon_add",
                "polygon_subtract",
                "keep_component",
                "remove_component",
                "warp_affine_local",
                "warp_nonrigid_local",
                "mirror_lr",
            ]
        )
        self.opacity_slider = QSlider()
        self.metadata_panel = QTextEdit()
        self.metadata_panel.setReadOnly(True)

        self.save_button = QPushButton("Save Revision")
        self.approve_button = QPushButton("Approve Mask")
        self.export_preview_button = QPushButton("Preview Export")

        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        layout.addWidget(self.layer_list)
        layout.addWidget(self.tool_select)
        layout.addWidget(self.opacity_slider)
        layout.addWidget(self.metadata_panel)
        layout.addWidget(self.save_button)
        layout.addWidget(self.approve_button)
        layout.addWidget(self.export_preview_button)
        self.setLayout(layout)

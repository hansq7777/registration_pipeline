from __future__ import annotations

from typing import Callable, Optional

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QLabel, QPushButton, QVBoxLayout, QWidget

from .graphics import qimage_from_rgb_array


class ProposalCard(QWidget):
    def __init__(self, label: str, proposal_index: int, on_run_mask: Callable[[int], None]) -> None:
        super().__init__()
        self.label = label
        self.proposal_index = proposal_index
        self.on_run_mask = on_run_mask
        self.title = QLabel(label)
        self.title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label = QLabel("No preview")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumHeight(120)
        self.run_button = QPushButton("Run Mask")
        self.run_button.clicked.connect(lambda: self.on_run_mask(self.proposal_index))

        layout = QVBoxLayout()
        layout.addWidget(self.title)
        layout.addWidget(self.image_label)
        layout.addWidget(self.run_button)
        self.setLayout(layout)

    def set_preview(self, rgb: Optional[np.ndarray]) -> None:
        if rgb is None:
            self.image_label.setText("No preview")
            self.image_label.setPixmap(QPixmap())
            return
        qimg = qimage_from_rgb_array(rgb)
        pix = QPixmap.fromImage(qimg).scaledToWidth(240, Qt.TransformationMode.SmoothTransformation)
        self.image_label.setPixmap(pix)

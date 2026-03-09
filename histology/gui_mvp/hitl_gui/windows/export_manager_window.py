from __future__ import annotations

from PySide6.QtWidgets import QComboBox, QLabel, QPushButton, QTextEdit, QVBoxLayout, QWidget


class ExportManagerWindow(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Export Manager")

        self.profile_select = QComboBox()
        self.profile_select.addItems(["review_mask", "cyclegan_train", "registration_fullres"])
        self.preview_panel = QTextEdit()
        self.preview_panel.setReadOnly(True)
        self.export_button = QPushButton("Run Export")
        self.refresh_button = QPushButton("Refresh Preview")

        layout = QVBoxLayout()
        layout.addWidget(QLabel("Export Profile"))
        layout.addWidget(self.profile_select)
        layout.addWidget(self.preview_panel)
        layout.addWidget(self.refresh_button)
        layout.addWidget(self.export_button)
        self.setLayout(layout)

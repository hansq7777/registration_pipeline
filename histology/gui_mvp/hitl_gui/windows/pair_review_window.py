from __future__ import annotations

from PySide6.QtWidgets import QLabel, QPushButton, QTextEdit, QVBoxLayout, QWidget


class PairReviewWindow(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Pair Review")

        self.left_canvas = QLabel("Nissl view placeholder")
        self.right_canvas = QLabel("Gallyas view placeholder")
        self.score_panel = QTextEdit()
        self.score_panel.setReadOnly(True)

        self.toggle_mirror_button = QPushButton("Toggle Mirror")
        self.accept_pair_button = QPushButton("Accept Pair")
        self.reject_pair_button = QPushButton("Reject Pair")
        self.override_pair_button = QPushButton("Manual Override")

        layout = QVBoxLayout()
        layout.addWidget(self.left_canvas)
        layout.addWidget(self.right_canvas)
        layout.addWidget(self.score_panel)
        layout.addWidget(self.toggle_mirror_button)
        layout.addWidget(self.accept_pair_button)
        layout.addWidget(self.reject_pair_button)
        layout.addWidget(self.override_pair_button)
        self.setLayout(layout)

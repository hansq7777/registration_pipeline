from __future__ import annotations

from PySide6.QtWidgets import QLabel, QListWidget, QPushButton, QSplitter, QTextEdit, QVBoxLayout, QWidget


class SlideReviewWindow(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Slide Review")

        self.overview_canvas = QLabel("Overview canvas placeholder")
        self.proposal_list = QListWidget()
        self.flag_panel = QTextEdit()
        self.flag_panel.setReadOnly(True)

        self.accept_button = QPushButton("Accept Proposal")
        self.reject_button = QPushButton("Reject Proposal")
        self.split_button = QPushButton("Split Proposal")
        self.merge_button = QPushButton("Merge Proposal")

        left = QVBoxLayout()
        left.addWidget(self.overview_canvas)

        right = QVBoxLayout()
        right.addWidget(self.proposal_list)
        right.addWidget(self.flag_panel)
        right.addWidget(self.accept_button)
        right.addWidget(self.reject_button)
        right.addWidget(self.split_button)
        right.addWidget(self.merge_button)

        container = QSplitter()
        left_widget = QWidget()
        left_widget.setLayout(left)
        right_widget = QWidget()
        right_widget.setLayout(right)
        container.addWidget(left_widget)
        container.addWidget(right_widget)

        root = QVBoxLayout()
        root.addWidget(container)
        self.setLayout(root)

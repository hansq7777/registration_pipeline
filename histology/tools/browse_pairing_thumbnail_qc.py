#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image
from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QFont, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QScrollBar,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from histology.gui_mvp.hitl_gui.widgets.graphics import qimage_from_rgb_array
from histology.tools.export_pairing_thumbnail_qc import ThumbSection, _collect_sections, _default_root, _fit_with_padding


VISIBLE_COUNT = 5


def _load_thumb_array(section: ThumbSection, thumb_w: int, thumb_h: int) -> np.ndarray:
    thumb = _fit_with_padding(section.crop_path, thumb_w, thumb_h)
    return np.asarray(thumb, dtype=np.uint8)


@dataclass(frozen=True)
class LoadedThumb:
    section: ThumbSection
    array: np.ndarray


@lru_cache(maxsize=128)
def _cached_thumb_array(path_str: str, thumb_w: int, thumb_h: int) -> np.ndarray:
    return _load_thumb_array(
        ThumbSection(ordinal=0, label="", sec_num=0, crop_path=Path(path_str)),
        thumb_w,
        thumb_h,
    )


class ZoomableThumbView(QGraphicsView):
    def __init__(self, width: int, height: int) -> None:
        super().__init__()
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._pixmap_item: QGraphicsPixmapItem | None = None
        self.setFixedSize(width, height)
        self.setStyleSheet("background:#121212;border:1px solid #5a5a5a;")
        self.setRenderHints(self.renderHints())
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)

    def set_loaded_thumb(self, loaded: LoadedThumb | None) -> None:
        self.resetTransform()
        self._scene.clear()
        self._pixmap_item = None
        if loaded is None:
            return
        pixmap = QPixmap.fromImage(qimage_from_rgb_array(loaded.array))
        self._pixmap_item = self._scene.addPixmap(pixmap)
        self._scene.setSceneRect(self._pixmap_item.boundingRect())
        self.fitInView(self._scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def wheelEvent(self, event) -> None:
        if self._pixmap_item is None:
            return super().wheelEvent(event)
        factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        self.scale(factor, factor)
        event.accept()


class ThumbCard(QWidget):
    def __init__(self, thumb_w: int, thumb_h: int) -> None:
        super().__init__()
        self.image_view = ZoomableThumbView(thumb_w, thumb_h)
        self.side_label = QLabel("")
        self.side_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.side_label.setWordWrap(True)
        self.side_label.setFixedWidth(108)
        self.side_label.setMinimumHeight(thumb_h)
        side_font = QFont("Consolas")
        side_font.setPointSize(11)
        side_font.setBold(True)
        self.side_label.setFont(side_font)
        self.side_label.setStyleSheet(
            "color:#f8f8f8;"
            "background:#202020;"
            "border:1px solid #5a5a5a;"
            "padding:6px;"
        )
        self.text_label = QLabel("")
        self.text_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self.text_label.setWordWrap(True)
        self.text_label.setStyleSheet("color:#e8e8e8;")

        top_row = QHBoxLayout()
        top_row.setContentsMargins(0, 0, 0, 0)
        top_row.setSpacing(8)
        top_row.addWidget(self.side_label)
        top_row.addWidget(self.image_view, 1)

        layout = QVBoxLayout()
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(6)
        layout.addLayout(top_row)
        layout.addWidget(self.text_label)
        self.setLayout(layout)

    def set_loaded_thumb(self, loaded: LoadedThumb | None) -> None:
        if loaded is None:
            self.image_view.set_loaded_thumb(None)
            self.side_label.setText("—")
            self.text_label.setText("")
            return
        self.image_view.set_loaded_thumb(loaded)
        sec = loaded.section
        self.side_label.setText(f"#{sec.ordinal}\nsec={sec.sec_num}")
        self.text_label.setText(f"{sec.label}")


class LadderPane(QWidget):
    def __init__(self, title: str, sections: list[ThumbSection], thumb_w: int, thumb_h: int) -> None:
        super().__init__()
        self.sections = sections
        self.thumb_w = thumb_w
        self.thumb_h = thumb_h
        self.title_label = QLabel(title)
        self.title_label.setStyleSheet("font-weight:600;color:#f0f0f0;")
        self.range_label = QLabel("")
        self.range_label.setStyleSheet("color:#b8b8b8;")

        self.cards = [ThumbCard(thumb_w, thumb_h) for _ in range(VISIBLE_COUNT)]
        self.scroll = QScrollBar(Qt.Orientation.Vertical)
        self.scroll.setMinimum(0)
        self.scroll.setMaximum(max(0, len(sections) - VISIBLE_COUNT))
        self.scroll.setPageStep(1)
        self.scroll.valueChanged.connect(self.refresh_visible)

        cards_layout = QVBoxLayout()
        cards_layout.setContentsMargins(0, 0, 0, 0)
        cards_layout.setSpacing(8)
        for card in self.cards:
            cards_layout.addWidget(card)
        cards_layout.addStretch(1)

        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(10)
        body.addLayout(cards_layout, 1)
        body.addWidget(self.scroll)

        layout = QVBoxLayout()
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)
        layout.addWidget(self.title_label)
        layout.addWidget(self.range_label)
        layout.addLayout(body, 1)
        self.setLayout(layout)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.refresh_visible()

    def refresh_visible(self) -> None:
        start = int(self.scroll.value())
        end = min(len(self.sections), start + VISIBLE_COUNT)
        if self.sections:
            self.range_label.setText(f"visible: {start + 1}-{end} / {len(self.sections)}")
        else:
            self.range_label.setText("visible: 0 / 0")
        for idx, card in enumerate(self.cards):
            absolute_idx = start + idx
            if absolute_idx >= len(self.sections):
                card.set_loaded_thumb(None)
                continue
            sec = self.sections[absolute_idx]
            arr = _cached_thumb_array(str(sec.crop_path), self.thumb_w, self.thumb_h)
            card.set_loaded_thumb(LoadedThumb(section=sec, array=arr))


class PairingThumbnailBrowser(QMainWindow):
    def __init__(
        self,
        *,
        animal_id: int,
        myelin_root: Path,
        nissl_root: Path,
        thumb_w: int,
        thumb_h: int,
    ) -> None:
        super().__init__()
        self.setWindowTitle(f"Histology Pairing Browser | animal {animal_id}")
        self.resize(1500, 1300)

        status = QLabel("Loading thumbnails ...")
        status.setStyleSheet("color:#d8d8d8;padding:6px;")
        container = QWidget()
        outer = QVBoxLayout()
        outer.setContentsMargins(10, 10, 10, 10)
        outer.setSpacing(10)
        outer.addWidget(status)
        container.setLayout(outer)
        self.setCentralWidget(container)
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)

        myelin_sections = _collect_sections(myelin_root, animal_id, "myelin")
        nissl_sections = _collect_sections(nissl_root, animal_id, "nissl")

        left = LadderPane(
            f"Myelin | {animal_id} | {len(myelin_sections)} sections",
            myelin_sections,
            thumb_w,
            thumb_h,
        )
        right = LadderPane(
            f"Nissl | {animal_id} | {len(nissl_sections)} sections",
            nissl_sections,
            thumb_w,
            thumb_h,
        )

        info = QLabel(
            "Drag each vertical scrollbar to change which 5 thumbnails are visible. "
            "Use the displayed #ordinal + label to report corrected cross-stain pairing."
        )
        info.setWordWrap(True)
        info.setStyleSheet("color:#c8c8c8;padding:4px;")

        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(12)
        row.addWidget(left, 1)
        row.addWidget(right, 1)

        outer.removeWidget(status)
        status.deleteLater()
        outer.addWidget(info)
        outer.addLayout(row, 1)
        QTimer.singleShot(0, QApplication.restoreOverrideCursor)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--animal-id", type=int, default=2502)
    parser.add_argument("--myelin-root", type=Path, default=_default_root("myelin"))
    parser.add_argument("--nissl-root", type=Path, default=_default_root("nissl"))
    parser.add_argument("--thumb-width", type=int, default=320)
    parser.add_argument("--thumb-height", type=int, default=220)
    args = parser.parse_args()

    app = QApplication.instance() or QApplication(sys.argv)
    window = PairingThumbnailBrowser(
        animal_id=args.animal_id,
        myelin_root=args.myelin_root,
        nissl_root=args.nissl_root,
        thumb_w=args.thumb_width,
        thumb_h=args.thumb_height,
    )
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())

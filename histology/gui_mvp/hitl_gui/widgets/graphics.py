from __future__ import annotations

from typing import Callable, Optional

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QGraphicsPixmapItem, QGraphicsRectItem, QGraphicsScene, QGraphicsView


def qimage_from_rgb_bytes(width: int, height: int, data: bytes) -> QImage:
    return QImage(data, width, height, width * 3, QImage.Format.Format_RGB888).copy()


def qimage_from_rgb_array(arr: np.ndarray) -> QImage:
    h, w = arr.shape[:2]
    return qimage_from_rgb_bytes(w, h, arr.astype(np.uint8).tobytes())


def qimage_from_rgba_array(arr: np.ndarray) -> QImage:
    h, w = arr.shape[:2]
    return QImage(arr.astype(np.uint8).tobytes(), w, h, w * 4, QImage.Format.Format_RGBA8888).copy()


class DraggableProposalItem(QGraphicsRectItem):
    def __init__(
        self,
        rect: QRectF,
        label: str,
        on_changed: Optional[Callable[["DraggableProposalItem"], None]] = None,
        on_drag_finished: Optional[Callable[["DraggableProposalItem"], None]] = None,
        on_selected: Optional[Callable[["DraggableProposalItem"], None]] = None,
    ) -> None:
        super().__init__(0.0, 0.0, rect.width(), rect.height())
        self.label = label
        self.on_changed = on_changed
        self.on_drag_finished = on_drag_finished
        self.on_selected = on_selected
        self._handle_margin = 10.0
        self._drag_mode = "move"
        self._press_scene_pos = QPointF()
        self._press_geom = QRectF()
        self.setPos(rect.x(), rect.y())
        self.setAcceptHoverEvents(True)
        self.setFlags(
            QGraphicsRectItem.GraphicsItemFlag.ItemIsSelectable
            | QGraphicsRectItem.GraphicsItemFlag.ItemSendsGeometryChanges
        )
        self.setAcceptedMouseButtons(Qt.MouseButton.LeftButton)
        self.setPen(QPen(QColor(255, 0, 0), 3))

    def itemChange(self, change, value):
        result = super().itemChange(change, value)
        if change == QGraphicsRectItem.GraphicsItemChange.ItemSelectedHasChanged:
            self.setPen(QPen(QColor(0, 170, 255) if bool(value) else QColor(255, 0, 0), 3))
            if bool(value) and self.on_selected is not None:
                self.on_selected(self)
        return result

    def scene_rect(self) -> QRectF:
        return QRectF(self.pos().x(), self.pos().y(), self.rect().width(), self.rect().height())

    def set_scene_rect(self, rect: QRectF) -> None:
        self.setPos(rect.x(), rect.y())
        self.setRect(0.0, 0.0, max(1.0, rect.width()), max(1.0, rect.height()))
        if self.on_changed is not None:
            self.on_changed(self)

    def _detect_drag_mode(self, scene_pos: QPointF) -> str:
        rect = self.scene_rect()
        left = abs(scene_pos.x() - rect.left()) <= self._handle_margin
        right = abs(scene_pos.x() - rect.right()) <= self._handle_margin
        top = abs(scene_pos.y() - rect.top()) <= self._handle_margin
        bottom = abs(scene_pos.y() - rect.bottom()) <= self._handle_margin
        if left and top:
            return "resize_lt"
        if right and top:
            return "resize_rt"
        if left and bottom:
            return "resize_lb"
        if right and bottom:
            return "resize_rb"
        if left:
            return "resize_l"
        if right:
            return "resize_r"
        if top:
            return "resize_t"
        if bottom:
            return "resize_b"
        return "move"

    def _apply_cursor_for_mode(self, mode: str) -> None:
        cursor_map = {
            "move": Qt.CursorShape.SizeAllCursor,
            "resize_l": Qt.CursorShape.SizeHorCursor,
            "resize_r": Qt.CursorShape.SizeHorCursor,
            "resize_t": Qt.CursorShape.SizeVerCursor,
            "resize_b": Qt.CursorShape.SizeVerCursor,
            "resize_lt": Qt.CursorShape.SizeFDiagCursor,
            "resize_rb": Qt.CursorShape.SizeFDiagCursor,
            "resize_rt": Qt.CursorShape.SizeBDiagCursor,
            "resize_lb": Qt.CursorShape.SizeBDiagCursor,
        }
        self.setCursor(cursor_map.get(mode, Qt.CursorShape.ArrowCursor))

    def hoverMoveEvent(self, event) -> None:
        self._apply_cursor_for_mode(self._detect_drag_mode(event.scenePos()))
        super().hoverMoveEvent(event)

    def hoverLeaveEvent(self, event) -> None:
        self.setCursor(Qt.CursorShape.ArrowCursor)
        super().hoverLeaveEvent(event)

    def mousePressEvent(self, event) -> None:
        self.setSelected(True)
        if self.on_selected is not None:
            self.on_selected(self)
        self._press_scene_pos = event.scenePos()
        self._press_geom = self.scene_rect()
        self._drag_mode = self._detect_drag_mode(event.scenePos())
        self._apply_cursor_for_mode(self._drag_mode)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        delta = event.scenePos() - self._press_scene_pos
        rect = QRectF(self._press_geom)
        min_size = 20.0

        if self._drag_mode == "move":
            rect.translate(delta)
        else:
            if "l" in self._drag_mode:
                rect.setLeft(min(rect.right() - min_size, rect.left() + delta.x()))
            if "r" in self._drag_mode:
                rect.setRight(max(rect.left() + min_size, rect.right() + delta.x()))
            if "t" in self._drag_mode:
                rect.setTop(min(rect.bottom() - min_size, rect.top() + delta.y()))
            if "b" in self._drag_mode:
                rect.setBottom(max(rect.top() + min_size, rect.bottom() + delta.y()))

        self.set_scene_rect(rect)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        super().mouseReleaseEvent(event)
        self._apply_cursor_for_mode("move")
        if self.on_drag_finished is not None:
            self.on_drag_finished(self)


class ImageSceneView(QGraphicsView):
    def __init__(self) -> None:
        super().__init__()
        self.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._pixmap_item: Optional[QGraphicsPixmapItem] = None

    def clear_all(self) -> None:
        self._scene.clear()
        self._pixmap_item = None

    def set_rgb_image(self, width: int, height: int, data: bytes) -> None:
        self.clear_all()
        image = qimage_from_rgb_bytes(width, height, data)
        pixmap = QPixmap.fromImage(image)
        self._pixmap_item = self._scene.addPixmap(pixmap)
        self._scene.setSceneRect(QRectF(0, 0, width, height))
        self.fitInView(self._scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    @property
    def scene_obj(self) -> QGraphicsScene:
        return self._scene

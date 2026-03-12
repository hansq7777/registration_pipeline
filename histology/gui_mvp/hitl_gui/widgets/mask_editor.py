from __future__ import annotations

from typing import Callable, Optional

import cv2
import numpy as np
from PySide6.QtCore import QPoint, QRect, QRectF, Qt
from PySide6.QtGui import QColor, QMouseEvent, QPainter, QPaintEvent, QPixmap, QWheelEvent
from PySide6.QtWidgets import QWidget
from scipy.ndimage import binary_fill_holes

from .graphics import qimage_from_rgb_array, qimage_from_rgba_array


class MaskEditorLabel(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.raw_rgb_full: Optional[np.ndarray] = None
        self.tissue_mask_full: Optional[np.ndarray] = None
        self.artifact_mask_full: Optional[np.ndarray] = None

        self.raw_rgb_display: Optional[np.ndarray] = None
        self.tissue_mask_display: Optional[np.ndarray] = None
        self.artifact_mask_display: Optional[np.ndarray] = None

        self.base_pixmap: Optional[QPixmap] = None
        self.overlay_rgba_display: Optional[np.ndarray] = None
        self.overlay_pixmap: Optional[QPixmap] = None
        self.stroke_mask_display: Optional[np.ndarray] = None
        self.stroke_rgba_display: Optional[np.ndarray] = None
        self.stroke_pixmap: Optional[QPixmap] = None

        self.active_layer: str = "tissue"
        self.brush_radius: int = 8
        self.mirror_enabled: bool = False
        self.display_scale: float = 1.0
        self.view_scale: float = 1.0
        self._image_draw_rect = QRectF()

        self.hover_pos_display: Optional[tuple[int, int]] = None
        self.on_mask_changed: Optional[Callable[[], None]] = None
        self.on_painting_state_changed: Optional[Callable[[bool], None]] = None

        self._painting = False
        self._last_draw_coord_display: Optional[tuple[int, int]] = None
        self._stroke_points_display: list[tuple[int, int]] = []
        self._stroke_add_mode: bool = True
        self._stroke_dirty_display_rect: Optional[QRect] = None

        self.setMouseTracking(True)
        self.setMinimumSize(900, 700)
        self.setAutoFillBackground(False)

    def set_section(self, raw_rgb: np.ndarray, tissue_mask: np.ndarray, artifact_mask: np.ndarray) -> None:
        self.raw_rgb_full = raw_rgb.copy()
        self.tissue_mask_full = tissue_mask.copy()
        self.artifact_mask_full = artifact_mask.copy()
        self.hover_pos_display = None
        self._last_draw_coord_display = None
        self._stroke_points_display = []
        self._stroke_dirty_display_rect = None
        self._rebuild_display_buffers()
        self.refresh()

    def set_active_layer(self, layer: str) -> None:
        self.active_layer = layer

    def set_brush_radius(self, radius: int) -> None:
        old_rect = self._hover_widget_rect()
        self.brush_radius = max(1, radius)
        self._update_widget_rect(old_rect.united(self._hover_widget_rect()))

    def set_mirror(self, enabled: bool) -> None:
        if self.mirror_enabled == enabled:
            return
        self.mirror_enabled = enabled
        self.hover_pos_display = None
        self._last_draw_coord_display = None
        self._stroke_points_display = []
        self._stroke_dirty_display_rect = None
        self._rebuild_display_buffers()
        self.refresh()

    def current_masks(self) -> tuple[np.ndarray, np.ndarray]:
        if self.tissue_mask_full is None or self.artifact_mask_full is None:
            return np.zeros((1, 1), dtype=np.uint8), np.zeros((1, 1), dtype=np.uint8)
        return self.tissue_mask_full.copy(), self.artifact_mask_full.copy()

    def current_usable_mask(self) -> np.ndarray:
        if self.tissue_mask_full is None or self.artifact_mask_full is None:
            return np.zeros((1, 1), dtype=np.uint8)
        usable = (self.tissue_mask_full > 0) & ~(self.artifact_mask_full > 0)
        return (usable.astype(np.uint8) * 255)

    def set_on_mask_changed(self, callback: Callable[[], None]) -> None:
        self.on_mask_changed = callback

    def set_on_painting_state_changed(self, callback: Callable[[bool], None]) -> None:
        self.on_painting_state_changed = callback

    def _rebuild_display_buffers(self) -> None:
        if self.raw_rgb_full is None or self.tissue_mask_full is None or self.artifact_mask_full is None:
            self.raw_rgb_display = None
            self.tissue_mask_display = None
            self.artifact_mask_display = None
            self.base_pixmap = None
            self.overlay_rgba_display = None
            self.overlay_pixmap = None
            self.stroke_mask_display = None
            self.stroke_rgba_display = None
            self.stroke_pixmap = None
            return

        h, w = self.raw_rgb_full.shape[:2]
        max_dim = 1400
        self.display_scale = min(1.0, max_dim / max(h, w))
        dw = max(1, int(round(w * self.display_scale)))
        dh = max(1, int(round(h * self.display_scale)))

        if self.display_scale < 1.0:
            raw_disp = cv2.resize(self.raw_rgb_full, (dw, dh), interpolation=cv2.INTER_AREA)
            tissue_disp = cv2.resize(self.tissue_mask_full, (dw, dh), interpolation=cv2.INTER_NEAREST)
            artifact_disp = cv2.resize(self.artifact_mask_full, (dw, dh), interpolation=cv2.INTER_NEAREST)
        else:
            raw_disp = self.raw_rgb_full.copy()
            tissue_disp = self.tissue_mask_full.copy()
            artifact_disp = self.artifact_mask_full.copy()

        if self.mirror_enabled:
            raw_disp = raw_disp[:, ::-1, :].copy()
            tissue_disp = tissue_disp[:, ::-1].copy()
            artifact_disp = artifact_disp[:, ::-1].copy()

        self.raw_rgb_display = raw_disp
        self.tissue_mask_display = tissue_disp
        self.artifact_mask_display = artifact_disp
        self.base_pixmap = QPixmap.fromImage(qimage_from_rgb_array(raw_disp))
        self._rebuild_overlay_full()
        self._reset_stroke_overlay()
        self._update_draw_rect()

    def _rebuild_overlay_full(self) -> None:
        if self.tissue_mask_display is None or self.artifact_mask_display is None:
            self.overlay_rgba_display = None
            self.overlay_pixmap = None
            return
        h, w = self.tissue_mask_display.shape[:2]
        overlay = np.zeros((h, w, 4), dtype=np.uint8)
        tissue = self.tissue_mask_display > 0
        artifact = self.artifact_mask_display > 0
        overlay[tissue] = np.array([255, 0, 0, 96], dtype=np.uint8)
        overlay[artifact] = np.array([0, 255, 255, 110], dtype=np.uint8)
        self.overlay_rgba_display = overlay
        self.overlay_pixmap = QPixmap.fromImage(qimage_from_rgba_array(overlay))

    def _reset_stroke_overlay(self) -> None:
        if self.raw_rgb_display is None:
            self.stroke_mask_display = None
            self.stroke_rgba_display = None
            self.stroke_pixmap = None
            return
        h, w = self.raw_rgb_display.shape[:2]
        self.stroke_mask_display = np.zeros((h, w), dtype=np.uint8)
        self.stroke_rgba_display = np.zeros((h, w, 4), dtype=np.uint8)
        self.stroke_pixmap = QPixmap.fromImage(qimage_from_rgba_array(self.stroke_rgba_display))

    def _update_overlay_subrect(self, rect: QRect) -> None:
        if self.overlay_rgba_display is None or self.overlay_pixmap is None:
            return
        if self.tissue_mask_display is None or self.artifact_mask_display is None:
            return
        img_h, img_w = self.tissue_mask_display.shape[:2]
        x1 = max(0, rect.left())
        y1 = max(0, rect.top())
        x2 = min(img_w, rect.right() + 1)
        y2 = min(img_h, rect.bottom() + 1)
        if x1 >= x2 or y1 >= y2:
            return

        overlay = np.zeros((y2 - y1, x2 - x1, 4), dtype=np.uint8)
        tissue = self.tissue_mask_display[y1:y2, x1:x2] > 0
        artifact = self.artifact_mask_display[y1:y2, x1:x2] > 0
        overlay[tissue] = np.array([255, 0, 0, 96], dtype=np.uint8)
        overlay[artifact] = np.array([0, 255, 255, 110], dtype=np.uint8)
        self.overlay_rgba_display[y1:y2, x1:x2] = overlay

        painter = QPainter(self.overlay_pixmap)
        painter.drawImage(QPoint(x1, y1), qimage_from_rgba_array(overlay))
        painter.end()

    def _stroke_overlay_color(self) -> np.ndarray:
        if self._stroke_add_mode:
            if self.active_layer == "tissue":
                return np.array([255, 96, 96, 150], dtype=np.uint8)
            return np.array([80, 255, 255, 150], dtype=np.uint8)
        return np.array([255, 255, 80, 170], dtype=np.uint8)

    def _update_stroke_overlay_subrect(self, rect: QRect) -> None:
        if self.stroke_mask_display is None or self.stroke_rgba_display is None or self.stroke_pixmap is None:
            return
        img_h, img_w = self.stroke_mask_display.shape[:2]
        x1 = max(0, rect.left())
        y1 = max(0, rect.top())
        x2 = min(img_w, rect.right() + 1)
        y2 = min(img_h, rect.bottom() + 1)
        if x1 >= x2 or y1 >= y2:
            return
        overlay = np.zeros((y2 - y1, x2 - x1, 4), dtype=np.uint8)
        stroke = self.stroke_mask_display[y1:y2, x1:x2] > 0
        overlay[stroke] = self._stroke_overlay_color()
        self.stroke_rgba_display[y1:y2, x1:x2] = overlay

        painter = QPainter(self.stroke_pixmap)
        painter.drawImage(QPoint(x1, y1), qimage_from_rgba_array(overlay))
        painter.end()

    def _update_draw_rect(self) -> None:
        if self.raw_rgb_display is None:
            self._image_draw_rect = QRectF()
            self.view_scale = 1.0
            return
        img_h, img_w = self.raw_rgb_display.shape[:2]
        if img_w <= 0 or img_h <= 0 or self.width() <= 0 or self.height() <= 0:
            self._image_draw_rect = QRectF()
            self.view_scale = 1.0
            return
        self.view_scale = min(self.width() / img_w, self.height() / img_h)
        draw_w = img_w * self.view_scale
        draw_h = img_h * self.view_scale
        draw_x = (self.width() - draw_w) / 2.0
        draw_y = (self.height() - draw_h) / 2.0
        self._image_draw_rect = QRectF(draw_x, draw_y, draw_w, draw_h)

    def refresh(self) -> None:
        self._update_draw_rect()
        self.update()

    def _widget_to_display_xy(self, pos: QPoint) -> Optional[tuple[int, int]]:
        if self.raw_rgb_display is None or self._image_draw_rect.isNull():
            return None
        rect = self._image_draw_rect
        if not rect.contains(pos.x(), pos.y()):
            return None
        img_h, img_w = self.raw_rgb_display.shape[:2]
        rel_x = (pos.x() - rect.x()) / max(1e-6, rect.width())
        rel_y = (pos.y() - rect.y()) / max(1e-6, rect.height())
        ix = int(np.clip(round(rel_x * (img_w - 1)), 0, img_w - 1))
        iy = int(np.clip(round(rel_y * (img_h - 1)), 0, img_h - 1))
        return ix, iy

    def _display_to_full_xy(self, coord: tuple[int, int]) -> tuple[int, int]:
        if self.raw_rgb_full is None:
            return coord
        px, py = coord
        full_h, full_w = self.raw_rgb_full.shape[:2]
        fx = int(round(px / max(self.display_scale, 1e-6)))
        fy = int(round(py / max(self.display_scale, 1e-6)))
        fx = max(0, min(full_w - 1, fx))
        fy = max(0, min(full_h - 1, fy))
        if self.mirror_enabled:
            fx = full_w - 1 - fx
        return fx, fy

    def _display_rect_from_points(self, points: list[tuple[int, int]], radius: int) -> QRect:
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        return QRect(min(xs) - radius - 2, min(ys) - radius - 2, max(xs) - min(xs) + 2 * radius + 5, max(ys) - min(ys) + 2 * radius + 5)

    def _display_rect_to_widget_rect(self, rect: QRect) -> QRect:
        if self._image_draw_rect.isNull():
            return QRect()
        x = int(np.floor(self._image_draw_rect.x() + rect.x() * self.view_scale))
        y = int(np.floor(self._image_draw_rect.y() + rect.y() * self.view_scale))
        w = int(np.ceil(rect.width() * self.view_scale))
        h = int(np.ceil(rect.height() * self.view_scale))
        return QRect(x, y, max(1, w), max(1, h)).adjusted(-3, -3, 3, 3)

    def _hover_widget_rect(self, coord: Optional[tuple[int, int]] = None) -> QRect:
        c = coord if coord is not None else self.hover_pos_display
        if c is None:
            return QRect()
        radius = max(1, int(round(self.brush_radius * self.display_scale)))
        rect = QRect(c[0] - radius - 2, c[1] - radius - 2, 2 * radius + 5, 2 * radius + 5)
        return self._display_rect_to_widget_rect(rect)

    def _update_widget_rect(self, rect: QRect) -> None:
        if rect.isNull():
            self.update()
        else:
            self.update(rect)

    def _set_hover_coord(self, coord: Optional[tuple[int, int]]) -> None:
        old_rect = self._hover_widget_rect()
        self.hover_pos_display = coord
        self._update_widget_rect(old_rect.united(self._hover_widget_rect()))

    def _stroke_on_mask(self, mask: np.ndarray, start: tuple[int, int], end: tuple[int, int], radius: int, add: bool) -> None:
        value = 255 if add else 0
        thickness = max(1, 2 * radius)
        if start == end:
            cv2.circle(mask, start, radius, value, thickness=-1, lineType=cv2.LINE_8)
        else:
            cv2.line(mask, start, end, value, thickness=thickness, lineType=cv2.LINE_8)
            cv2.circle(mask, end, radius, value, thickness=-1, lineType=cv2.LINE_8)

    def _begin_stroke(self, add: bool) -> None:
        self._painting = True
        self._stroke_add_mode = add
        self._last_draw_coord_display = None
        self._stroke_points_display = []
        self._stroke_dirty_display_rect = None
        if self.on_painting_state_changed is not None:
            self.on_painting_state_changed(True)
        if self.stroke_mask_display is not None:
            self.stroke_mask_display.fill(0)
        if self.stroke_rgba_display is not None:
            self.stroke_rgba_display.fill(0)
        if self.stroke_pixmap is not None:
            self.stroke_pixmap.fill(Qt.GlobalColor.transparent)

    def _clear_stroke_preview(self) -> None:
        if self.stroke_mask_display is not None:
            self.stroke_mask_display.fill(0)
        if self.stroke_rgba_display is not None:
            self.stroke_rgba_display.fill(0)
        if self.stroke_pixmap is not None:
            self.stroke_pixmap.fill(Qt.GlobalColor.transparent)
        self._stroke_points_display = []
        self._stroke_dirty_display_rect = None

    def _paint_at(self, pos: QPoint, add: bool) -> None:
        if self.tissue_mask_full is None or self.artifact_mask_full is None:
            return
        if self.tissue_mask_display is None or self.artifact_mask_display is None:
            return
        if self.stroke_mask_display is None:
            return
        coord_display = self._widget_to_display_xy(pos)
        if coord_display is None:
            return

        prev_hover = self._hover_widget_rect()
        self.hover_pos_display = coord_display

        prev_display = self._last_draw_coord_display or coord_display

        display_radius = max(1, int(round(self.brush_radius * self.display_scale)))
        dirty_display = self._display_rect_from_points([prev_display, coord_display], display_radius)
        dirty_widget = self._display_rect_to_widget_rect(dirty_display).united(prev_hover).united(self._hover_widget_rect(coord_display))

        self._stroke_on_mask(self.stroke_mask_display, prev_display, coord_display, display_radius, add=True)
        self._update_stroke_overlay_subrect(dirty_display)
        self._last_draw_coord_display = coord_display
        self._stroke_points_display.append(coord_display)
        if self._stroke_dirty_display_rect is None:
            self._stroke_dirty_display_rect = dirty_display
        else:
            self._stroke_dirty_display_rect = self._stroke_dirty_display_rect.united(dirty_display)
        self._update_widget_rect(dirty_widget)

    def _commit_stroke(self) -> None:
        if not self._stroke_points_display:
            self._clear_stroke_preview()
            return
        if self.tissue_mask_full is None or self.artifact_mask_full is None:
            self._clear_stroke_preview()
            return
        if self.tissue_mask_display is None or self.artifact_mask_display is None:
            self._clear_stroke_preview()
            return

        target_display = self.tissue_mask_display if self.active_layer == "tissue" else self.artifact_mask_display
        target_full = self.tissue_mask_full if self.active_layer == "tissue" else self.artifact_mask_full
        points = self._stroke_points_display

        if len(points) == 1:
            points = [points[0], points[0]]

        display_radius = max(1, int(round(self.brush_radius * self.display_scale)))
        last_full: Optional[tuple[int, int]] = None
        last_display: Optional[tuple[int, int]] = None
        for coord_display in points:
            coord_full = self._display_to_full_xy(coord_display)
            if last_display is None or last_full is None:
                last_display = coord_display
                last_full = coord_full
            self._stroke_on_mask(target_display, last_display, coord_display, display_radius, add=self._stroke_add_mode)
            self._stroke_on_mask(target_full, last_full, coord_full, self.brush_radius, add=self._stroke_add_mode)
            last_display = coord_display
            last_full = coord_full

        dirty_display = self._stroke_dirty_display_rect or self._display_rect_from_points(points, display_radius)
        self._update_overlay_subrect(dirty_display)
        self._clear_stroke_preview()
        dirty_widget = self._display_rect_to_widget_rect(dirty_display).united(self._hover_widget_rect())
        self._update_widget_rect(dirty_widget)

    def paintEvent(self, event: QPaintEvent) -> None:
        painter = QPainter(self)
        painter.setClipRect(event.rect())
        painter.fillRect(self.rect(), QColor(26, 26, 26))

        if self.base_pixmap is None or self._image_draw_rect.isNull():
            painter.setPen(QColor(210, 210, 210))
            painter.drawText(self.rect(), Qt.AlignCenter, "No section loaded")
            return

        target = self._image_draw_rect
        source_rect = QRectF(self.base_pixmap.rect())
        painter.drawPixmap(target, self.base_pixmap, source_rect)
        if self.overlay_pixmap is not None:
            painter.drawPixmap(target, self.overlay_pixmap, QRectF(self.overlay_pixmap.rect()))
        if self.stroke_pixmap is not None:
            painter.drawPixmap(target, self.stroke_pixmap, QRectF(self.stroke_pixmap.rect()))

        if self.hover_pos_display is not None:
            hx, hy = self.hover_pos_display
            display_radius = max(1, int(round(self.brush_radius * self.display_scale)))
            cx = target.x() + (hx + 0.5) * self.view_scale
            cy = target.y() + (hy + 0.5) * self.view_scale
            radius = display_radius * self.view_scale
            painter.setPen(QColor(80, 255, 80))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawEllipse(QPoint(int(round(cx)), int(round(cy))), int(round(radius)), int(round(radius)))

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._update_draw_rect()
        self.update()

    def leaveEvent(self, event) -> None:
        self._set_hover_coord(None)
        super().leaveEvent(event)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._begin_stroke(add=True)
            self._paint_at(event.position().toPoint(), add=True)
            event.accept()
            return
        if event.button() == Qt.MouseButton.RightButton:
            self._begin_stroke(add=False)
            self._paint_at(event.position().toPoint(), add=False)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        coord = self._widget_to_display_xy(event.position().toPoint())
        if self._painting:
            self.hover_pos_display = coord
            add = bool(event.buttons() & Qt.MouseButton.LeftButton)
            self._paint_at(event.position().toPoint(), add=add)
        else:
            self._set_hover_coord(coord)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        was_painting = self._painting
        self._painting = False
        self._last_draw_coord_display = None
        if was_painting:
            self._commit_stroke()
        if was_painting and self.on_mask_changed is not None:
            self.on_mask_changed()
        if was_painting and self.on_painting_state_changed is not None:
            self.on_painting_state_changed(False)
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event: QWheelEvent) -> None:
        delta = event.angleDelta().y()
        step = 1 if delta > 0 else -1
        self.set_brush_radius(self.brush_radius + step)
        event.accept()

    def keep_largest_tissue_component(self) -> None:
        if self.tissue_mask_full is None:
            return
        num, labels, stats, _ = cv2.connectedComponentsWithStats((self.tissue_mask_full > 0).astype(np.uint8), 8)
        if num <= 1:
            return
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        self.tissue_mask_full = (labels == largest).astype(np.uint8) * 255
        self._rebuild_display_buffers()
        self.refresh()
        if self.on_mask_changed is not None:
            self.on_mask_changed()

    def close_and_fill_tissue_gaps(self) -> None:
        if self.tissue_mask_full is None:
            return
        mask = (self.tissue_mask_full > 0).astype(np.uint8)
        num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        if num > 1:
            largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            mask = (labels == largest).astype(np.uint8)
        kernel = np.ones((9, 9), np.uint8)
        mask = cv2.morphologyEx(mask * 255, cv2.MORPH_CLOSE, kernel) > 0
        mask = binary_fill_holes(mask > 0)
        self.tissue_mask_full = (mask.astype(np.uint8) * 255)
        self._rebuild_display_buffers()
        self.refresh()
        if self.on_mask_changed is not None:
            self.on_mask_changed()

    def morph_active_layer(self, operation: str) -> None:
        if self.tissue_mask_full is None or self.artifact_mask_full is None:
            return
        target_full = self.tissue_mask_full if self.active_layer == "tissue" else self.artifact_mask_full
        mask = (target_full > 0).astype(np.uint8)
        if not mask.any():
            return
        min_dim = min(mask.shape[:2])
        kernel_size = max(3, int(round(min_dim * 0.002)))
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        if operation == "shrink":
            updated = cv2.erode(mask * 255, kernel, iterations=1) > 0
        elif operation == "expand":
            updated = cv2.dilate(mask * 255, kernel, iterations=1) > 0
        else:
            raise ValueError(f"Unknown morph operation: {operation}")

        target_full[:, :] = updated.astype(np.uint8) * 255
        self._rebuild_display_buffers()
        self.refresh()
        if self.on_mask_changed is not None:
            self.on_mask_changed()

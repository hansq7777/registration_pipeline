from __future__ import annotations

from typing import Callable, Optional

import cv2
import numpy as np
from PySide6.QtCore import QPoint, QPointF, QRect, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QMouseEvent, QPainter, QPaintEvent, QPixmap, QWheelEvent
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
        self.raw_visible: bool = True
        self.overlay_visible: bool = True
        self.brush_enabled: bool = False
        self._hand_override_active: bool = False
        self._brush_enabled_before_hand_override: Optional[bool] = None
        self._line_erase_active: bool = False
        self._brush_enabled_before_line_erase: Optional[bool] = None
        self._line_erase_start_display: Optional[tuple[int, int]] = None
        self._line_erase_preview_end_display: Optional[tuple[int, int]] = None
        self.display_scale: float = 1.0
        self.view_scale: float = 1.0
        self.zoom_factor: float = 1.0
        self.pan_offset = QPointF(0.0, 0.0)
        self._image_draw_rect = QRectF()

        self.hover_pos_display: Optional[tuple[int, int]] = None
        self.on_mask_changed: Optional[Callable[[], None]] = None
        self.on_painting_state_changed: Optional[Callable[[bool], None]] = None
        self.on_active_layer_changed: Optional[Callable[[str], None]] = None
        self.on_close_fill_requested: Optional[Callable[[], None]] = None
        self.on_save_and_next_requested: Optional[Callable[[], None]] = None
        self.on_focus_gained: Optional[Callable[[], None]] = None
        self.on_mark_group_requested: Optional[Callable[[int], None]] = None
        self.component_group_marks: dict[int, int] = {}

        self._painting = False
        self._panning = False
        self._pan_start_widget: Optional[QPoint] = None
        self._pan_start_offset = QPointF(0.0, 0.0)
        self._last_draw_coord_display: Optional[tuple[int, int]] = None
        self._stroke_points_display: list[tuple[int, int]] = []
        self._stroke_add_mode: bool = True
        self._stroke_dirty_display_rect: Optional[QRect] = None
        self._undo_stack: list[tuple[np.ndarray, np.ndarray]] = []

        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
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
        self._undo_stack = []
        self.component_group_marks = {}
        self.zoom_factor = 1.0
        self.pan_offset = QPointF(0.0, 0.0)
        self._rebuild_display_buffers()
        self.refresh()

    def set_active_layer(self, layer: str) -> None:
        if layer not in {"tissue", "artifact"}:
            return
        self.active_layer = layer
        self.refresh()

    def set_brush_radius(self, radius: int) -> None:
        old_rect = self._hover_widget_rect()
        self.brush_radius = max(1, radius)
        self._update_widget_rect(old_rect.united(self._hover_widget_rect()))

    def toggle_overlay_visibility(self) -> bool:
        self.overlay_visible = not self.overlay_visible
        self.refresh()
        return self.overlay_visible

    def toggle_raw_visibility(self) -> bool:
        self.raw_visible = not self.raw_visible
        self.refresh()
        return self.raw_visible

    def toggle_brush_mode(self) -> bool:
        self._hand_override_active = False
        self._brush_enabled_before_hand_override = None
        self._line_erase_active = False
        self._brush_enabled_before_line_erase = None
        self._line_erase_start_display = None
        self._line_erase_preview_end_display = None
        self._set_brush_enabled(not self.brush_enabled)
        return self.brush_enabled

    def _set_brush_enabled(self, enabled: bool) -> None:
        enabled = bool(enabled)
        if self._painting:
            self._painting = False
            self._clear_stroke_preview()
            if self.on_painting_state_changed is not None:
                self.on_painting_state_changed(False)
        self.brush_enabled = enabled
        if not self.brush_enabled:
            self._set_hover_coord(None)
        self.refresh()

    def toggle_hand_override(self) -> bool:
        if self._line_erase_active:
            self._line_erase_active = False
            self._brush_enabled_before_line_erase = None
            self._line_erase_start_display = None
            self._line_erase_preview_end_display = None
        if self._hand_override_active:
            self._hand_override_active = False
            restore_brush = self._brush_enabled_before_hand_override
            self._brush_enabled_before_hand_override = None
            if restore_brush:
                self._set_brush_enabled(True)
            self.refresh()
            return not self.brush_enabled

        if not self.brush_enabled:
            self.refresh()
            return True

        self._brush_enabled_before_hand_override = self.brush_enabled
        self._hand_override_active = True
        self._set_brush_enabled(False)
        return True

    def toggle_line_erase_mode(self) -> bool:
        if self._line_erase_active:
            self._line_erase_active = False
            restore_brush = self._brush_enabled_before_line_erase
            self._brush_enabled_before_line_erase = None
            self._line_erase_start_display = None
            self._line_erase_preview_end_display = None
            if restore_brush:
                self._set_brush_enabled(True)
            self.refresh()
            return False

        self._brush_enabled_before_line_erase = self.brush_enabled
        self._hand_override_active = False
        self._brush_enabled_before_hand_override = None
        self._set_brush_enabled(False)
        self._line_erase_active = True
        self._line_erase_start_display = None
        self._line_erase_preview_end_display = None
        self.refresh()
        return True

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

    def _combined_component_context(self) -> tuple[np.ndarray, dict[int, int], list[dict[str, float]]] | None:
        if self.tissue_mask_full is None or self.artifact_mask_full is None:
            return None
        combined = ((self.tissue_mask_full > 0) | (self.artifact_mask_full > 0)).astype(np.uint8)
        if not combined.any():
            return None
        num, labels, stats, centroids = cv2.connectedComponentsWithStats(combined, 8)
        components: list[dict[str, float]] = []
        for label_idx in range(1, num):
            area = int(stats[label_idx, cv2.CC_STAT_AREA])
            if area <= 0:
                continue
            components.append(
                {
                    "label": int(label_idx),
                    "area": area,
                    "centroid_x": float(centroids[label_idx, 0]),
                    "centroid_y": float(centroids[label_idx, 1]),
                }
            )
        components.sort(key=lambda x: x["area"], reverse=True)
        label_to_rank = {entry["label"]: rank + 1 for rank, entry in enumerate(components)}
        return labels, label_to_rank, components

    def combined_component_summary(self) -> list[dict[str, int]]:
        context = self._combined_component_context()
        if context is None:
            return []
        _, _, components = context
        out: list[dict[str, int]] = []
        for rank, entry in enumerate(components, start=1):
            out.append({"rank": rank, "area": int(entry["area"])})
        return out

    def hovered_combined_component_rank(self) -> int | None:
        if self.hover_pos_display is None:
            return None
        context = self._combined_component_context()
        if context is None:
            return None
        labels, label_to_rank, _ = context
        full_x, full_y = self._display_to_full_xy(self.hover_pos_display)
        label_idx = int(labels[full_y, full_x])
        if label_idx <= 0:
            return None
        return label_to_rank.get(label_idx)

    def keep_only_combined_component_ranks(self, ranks: set[int]) -> list[int]:
        context = self._combined_component_context()
        if context is None:
            return []
        labels, label_to_rank, components = context
        keep_ranks = sorted({int(rank) for rank in ranks if int(rank) > 0})
        if not keep_ranks:
            return []
        keep_labels = {label for label, rank in label_to_rank.items() if rank in keep_ranks}
        if not keep_labels:
            return []
        self._remember_undo_state()
        keep_mask = np.isin(labels, list(keep_labels))
        tissue = (self.tissue_mask_full > 0) & keep_mask
        artifact = (self.artifact_mask_full > 0) & keep_mask & ~tissue
        self.tissue_mask_full = tissue.astype(np.uint8) * 255
        self.artifact_mask_full = artifact.astype(np.uint8) * 255
        self._rebuild_display_buffers()
        self.refresh()
        if self.on_mask_changed is not None:
            self.on_mask_changed()
        valid_ranks = [rank for rank in keep_ranks if rank <= len(components)]
        return valid_ranks

    def _remember_undo_state(self) -> None:
        if self.tissue_mask_full is None or self.artifact_mask_full is None:
            return
        self._undo_stack.append(
            (
                self.tissue_mask_full.copy(),
                self.artifact_mask_full.copy(),
            )
        )
        if len(self._undo_stack) > 5:
            self._undo_stack = self._undo_stack[-5:]

    def undo_last_action(self) -> bool:
        if not self._undo_stack:
            return False
        tissue, artifact = self._undo_stack.pop()
        self.tissue_mask_full = tissue
        self.artifact_mask_full = artifact
        self._rebuild_display_buffers()
        self.refresh()
        if self.on_mask_changed is not None:
            self.on_mask_changed()
        return True

    def set_on_mask_changed(self, callback: Callable[[], None]) -> None:
        self.on_mask_changed = callback

    def set_on_painting_state_changed(self, callback: Callable[[bool], None]) -> None:
        self.on_painting_state_changed = callback

    def set_on_active_layer_changed(self, callback: Callable[[str], None]) -> None:
        self.on_active_layer_changed = callback

    def set_on_close_fill_requested(self, callback: Callable[[], None]) -> None:
        self.on_close_fill_requested = callback

    def set_on_save_and_next_requested(self, callback: Callable[[], None]) -> None:
        self.on_save_and_next_requested = callback

    def set_on_focus_gained(self, callback: Callable[[], None]) -> None:
        self.on_focus_gained = callback

    def set_on_mark_group_requested(self, callback: Callable[[int], None]) -> None:
        self.on_mark_group_requested = callback

    def set_component_group_marks(self, marks: dict[int, int]) -> None:
        self.component_group_marks = {
            int(rank): int(group)
            for rank, group in dict(marks).items()
            if int(rank) > 0 and int(group) in {1, 2}
        }
        self.refresh()

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

    def _hover_color(self) -> QColor:
        if self.active_layer == "tissue":
            return QColor(255, 120, 120)
        return QColor(80, 255, 255)

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
        fit_scale = min(self.width() / img_w, self.height() / img_h)
        self.view_scale = fit_scale * self.zoom_factor
        draw_w = img_w * self.view_scale
        draw_h = img_h * self.view_scale
        draw_x = (self.width() - draw_w) / 2.0 + self.pan_offset.x()
        draw_y = (self.height() - draw_h) / 2.0 + self.pan_offset.y()
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

    def _full_to_display_xy(self, coord: tuple[float, float]) -> tuple[float, float]:
        if self.raw_rgb_full is None:
            return coord
        px, py = float(coord[0]), float(coord[1])
        full_h, full_w = self.raw_rgb_full.shape[:2]
        if self.mirror_enabled:
            px = float(full_w - 1) - px
        dx = float(np.clip(px * self.display_scale, 0.0, max(0, (self.raw_rgb_display.shape[1] - 1) if self.raw_rgb_display is not None else 0)))
        dy = float(np.clip(py * self.display_scale, 0.0, max(0, (self.raw_rgb_display.shape[0] - 1) if self.raw_rgb_display is not None else 0)))
        return dx, dy

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

    def _line_preview_widget_rect(
        self,
        start: Optional[tuple[int, int]] = None,
        end: Optional[tuple[int, int]] = None,
    ) -> QRect:
        p1 = start if start is not None else self._line_erase_start_display
        p2 = end if end is not None else self._line_erase_preview_end_display
        if p1 is None or p2 is None:
            return QRect()
        rect = self._display_rect_from_points([p1, p2], radius=4)
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

    def _set_line_erase_preview(
        self,
        start: Optional[tuple[int, int]],
        end: Optional[tuple[int, int]],
    ) -> None:
        old_rect = self._line_preview_widget_rect()
        self._line_erase_start_display = start
        self._line_erase_preview_end_display = end
        self._update_widget_rect(old_rect.united(self._line_preview_widget_rect()))

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
        if self.active_layer == "artifact" and self._stroke_add_mode and self.tissue_mask_display is not None:
            self.stroke_mask_display[self.tissue_mask_display > 0] = 0
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

        points = self._stroke_points_display
        self._remember_undo_state()

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
            if self._stroke_add_mode:
                target_display = self.tissue_mask_display if self.active_layer == "tissue" else self.artifact_mask_display
                target_full = self.tissue_mask_full if self.active_layer == "tissue" else self.artifact_mask_full
                self._stroke_on_mask(target_display, last_display, coord_display, display_radius, add=True)
                self._stroke_on_mask(target_full, last_full, coord_full, self.brush_radius, add=True)
            else:
                self._stroke_on_mask(self.tissue_mask_display, last_display, coord_display, display_radius, add=False)
                self._stroke_on_mask(self.artifact_mask_display, last_display, coord_display, display_radius, add=False)
                self._stroke_on_mask(self.tissue_mask_full, last_full, coord_full, self.brush_radius, add=False)
                self._stroke_on_mask(self.artifact_mask_full, last_full, coord_full, self.brush_radius, add=False)
            last_display = coord_display
            last_full = coord_full

        if self._stroke_add_mode:
            if self.active_layer == "tissue":
                self.artifact_mask_display[self.tissue_mask_display > 0] = 0
                self.artifact_mask_full[self.tissue_mask_full > 0] = 0
            elif self.active_layer == "artifact":
                self.artifact_mask_display[self.tissue_mask_display > 0] = 0
                self.artifact_mask_full[self.tissue_mask_full > 0] = 0

        dirty_display = self._stroke_dirty_display_rect or self._display_rect_from_points(points, display_radius)
        self._rebuild_overlay_full()
        self._clear_stroke_preview()
        dirty_widget = self._display_rect_to_widget_rect(dirty_display).united(self._hover_widget_rect())
        self._update_widget_rect(dirty_widget)
        self.refresh()

    def paintEvent(self, event: QPaintEvent) -> None:
        painter = QPainter(self)
        painter.setClipRect(event.rect())
        painter.fillRect(self.rect(), QColor(26, 26, 26))
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        if self.base_pixmap is None or self._image_draw_rect.isNull():
            painter.setPen(QColor(210, 210, 210))
            painter.drawText(self.rect(), Qt.AlignCenter, "No section loaded")
            return

        target = self._image_draw_rect
        source_rect = QRectF(self.base_pixmap.rect())
        if self.raw_visible:
            painter.drawPixmap(target, self.base_pixmap, source_rect)
        if self.overlay_visible and self.overlay_pixmap is not None:
            painter.drawPixmap(target, self.overlay_pixmap, QRectF(self.overlay_pixmap.rect()))
        if self.stroke_pixmap is not None:
            painter.drawPixmap(target, self.stroke_pixmap, QRectF(self.stroke_pixmap.rect()))

        if self.component_group_marks:
            context = self._combined_component_context()
            if context is not None:
                _, label_to_rank, components = context
                rank_to_component = {
                    label_to_rank[int(entry["label"])]: entry
                    for entry in components
                    if int(entry["label"]) in label_to_rank
                }
                font = QFont()
                font.setBold(True)
                font.setPointSize(max(12, int(round(16 * self.view_scale))))
                painter.setFont(font)
                for rank, group_id in sorted(self.component_group_marks.items()):
                    if group_id not in {1, 2}:
                        continue
                    entry = rank_to_component.get(int(rank))
                    if entry is None:
                        continue
                    dx, dy = self._full_to_display_xy((float(entry["centroid_x"]), float(entry["centroid_y"])))
                    cx = target.x() + (dx + 0.5) * self.view_scale
                    cy = target.y() + (dy + 0.5) * self.view_scale
                    badge_radius = max(12.0, 14.0 * self.view_scale)
                    badge_color = QColor(255, 196, 64, 160) if int(group_id) == 1 else QColor(80, 220, 255, 160)
                    painter.setPen(Qt.PenStyle.NoPen)
                    painter.setBrush(badge_color)
                    painter.drawEllipse(QPointF(cx, cy), badge_radius, badge_radius)
                    painter.setPen(QColor(20, 20, 20, 220))
                    text_rect = QRectF(cx - badge_radius, cy - badge_radius, badge_radius * 2.0, badge_radius * 2.0)
                    painter.drawText(text_rect, Qt.AlignmentFlag.AlignCenter, str(int(group_id)))

        if self._line_erase_active and self._line_erase_start_display is not None and self._line_erase_preview_end_display is not None:
            sx, sy = self._line_erase_start_display
            ex, ey = self._line_erase_preview_end_display
            p1 = QPointF(target.x() + (sx + 0.5) * self.view_scale, target.y() + (sy + 0.5) * self.view_scale)
            p2 = QPointF(target.x() + (ex + 0.5) * self.view_scale, target.y() + (ey + 0.5) * self.view_scale)
            pen = QPen(QColor(255, 255, 80, 220), max(2.0, 2.0 * self.view_scale))
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawLine(p1, p2)
            painter.setBrush(QColor(255, 255, 80, 180))
            painter.drawEllipse(p1, max(3.0, 3.0 * self.view_scale), max(3.0, 3.0 * self.view_scale))
            painter.drawEllipse(p2, max(3.0, 3.0 * self.view_scale), max(3.0, 3.0 * self.view_scale))

        if self.brush_enabled and self.hover_pos_display is not None:
            hx, hy = self.hover_pos_display
            display_radius = max(1, int(round(self.brush_radius * self.display_scale)))
            cx = target.x() + (hx + 0.5) * self.view_scale
            cy = target.y() + (hy + 0.5) * self.view_scale
            radius = display_radius * self.view_scale
            painter.setPen(self._hover_color())
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawEllipse(QPoint(int(round(cx)), int(round(cy))), int(round(radius)), int(round(radius)))

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._update_draw_rect()
        self.update()

    def leaveEvent(self, event) -> None:
        self._set_hover_coord(None)
        if self._line_erase_active and self._line_erase_start_display is not None:
            self._set_line_erase_preview(self._line_erase_start_display, self._line_erase_start_display)
        super().leaveEvent(event)

    def enterEvent(self, event) -> None:
        self.setFocus(Qt.FocusReason.MouseFocusReason)
        super().enterEvent(event)

    def focusInEvent(self, event) -> None:
        if self.on_focus_gained is not None:
            self.on_focus_gained()
        super().focusInEvent(event)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        self.setFocus(Qt.FocusReason.MouseFocusReason)
        if self._line_erase_active and event.button() == Qt.MouseButton.LeftButton:
            coord = self._widget_to_display_xy(event.position().toPoint())
            if coord is not None:
                if self._line_erase_start_display is None:
                    self._set_line_erase_preview(coord, coord)
                else:
                    self._set_line_erase_preview(self._line_erase_start_display, coord)
                    self.apply_line_erase()
                event.accept()
                return
        if self._line_erase_active and event.button() == Qt.MouseButton.RightButton:
            self._set_line_erase_preview(None, None)
            event.accept()
            return
        if not self.brush_enabled and event.button() == Qt.MouseButton.LeftButton:
            self._panning = True
            self._pan_start_widget = event.position().toPoint()
            self._pan_start_offset = QPointF(self.pan_offset)
            event.accept()
            return
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
        if self._panning and self._pan_start_widget is not None:
            current = event.position().toPoint()
            delta = current - self._pan_start_widget
            self.pan_offset = QPointF(
                self._pan_start_offset.x() + float(delta.x()),
                self._pan_start_offset.y() + float(delta.y()),
            )
            self.refresh()
        elif self._painting:
            self.hover_pos_display = coord
            add = bool(event.buttons() & Qt.MouseButton.LeftButton)
            self._paint_at(event.position().toPoint(), add=add)
        else:
            self._set_hover_coord(coord)
            if self._line_erase_active and self._line_erase_start_display is not None:
                preview = coord if coord is not None else self._line_erase_start_display
                self._set_line_erase_preview(self._line_erase_start_display, preview)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        was_painting = self._painting
        was_panning = self._panning
        self._painting = False
        self._panning = False
        self._pan_start_widget = None
        self._last_draw_coord_display = None
        if was_painting:
            self._commit_stroke()
        if was_painting and self.on_mask_changed is not None:
            self.on_mask_changed()
        if was_painting and self.on_painting_state_changed is not None:
            self.on_painting_state_changed(False)
        if was_panning:
            event.accept()
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event: QWheelEvent) -> None:
        if self.brush_enabled:
            delta = event.angleDelta().y()
            step = 1 if delta > 0 else -1
            self.set_brush_radius(self.brush_radius + step)
        else:
            delta = event.angleDelta().y()
            old_rect = QRectF(self._image_draw_rect)
            old_zoom = float(self.zoom_factor)
            zoom_step = 1.12 if delta > 0 else 1 / 1.12
            new_zoom = float(np.clip(self.zoom_factor * zoom_step, 0.25, 8.0))
            if abs(new_zoom - old_zoom) < 1e-9:
                event.accept()
                return
            anchor = event.position()
            self.zoom_factor = new_zoom
            self._update_draw_rect()
            new_rect = QRectF(self._image_draw_rect)
            if not old_rect.isNull() and not new_rect.isNull():
                rel_x = (anchor.x() - old_rect.x()) / max(1e-6, old_rect.width())
                rel_y = (anchor.y() - old_rect.y()) / max(1e-6, old_rect.height())
                rel_x = float(np.clip(rel_x, 0.0, 1.0))
                rel_y = float(np.clip(rel_y, 0.0, 1.0))
                target_x = new_rect.x() + rel_x * new_rect.width()
                target_y = new_rect.y() + rel_y * new_rect.height()
                self.pan_offset = QPointF(
                    self.pan_offset.x() + (anchor.x() - target_x),
                    self.pan_offset.y() + (anchor.y() - target_y),
                )
            self.refresh()
        event.accept()

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key.Key_A:
            self.set_active_layer("artifact")
            if self.on_active_layer_changed is not None:
                self.on_active_layer_changed("artifact")
            event.accept()
            return
        if event.key() == Qt.Key.Key_T:
            self.set_active_layer("tissue")
            if self.on_active_layer_changed is not None:
                self.on_active_layer_changed("tissue")
            event.accept()
            return
        if event.key() == Qt.Key.Key_N:
            self.toggle_raw_visibility()
            event.accept()
            return
        if event.key() == Qt.Key.Key_M:
            self.toggle_overlay_visibility()
            event.accept()
            return
        if event.key() == Qt.Key.Key_P:
            self.toggle_brush_mode()
            event.accept()
            return
        if event.key() == Qt.Key.Key_H:
            self.toggle_hand_override()
            event.accept()
            return
        if event.key() == Qt.Key.Key_L:
            self.toggle_line_erase_mode()
            event.accept()
            return
        if event.key() == Qt.Key.Key_D:
            if self.delete_component_under_cursor():
                event.accept()
                return
        if event.key() == Qt.Key.Key_1:
            if self.on_mark_group_requested is not None:
                self.on_mark_group_requested(1)
                event.accept()
                return
        if event.key() == Qt.Key.Key_2:
            if self.on_mark_group_requested is not None:
                self.on_mark_group_requested(2)
                event.accept()
                return
        if event.key() == Qt.Key.Key_C:
            if self.on_close_fill_requested is not None:
                self.on_close_fill_requested()
                event.accept()
                return
        if event.key() == Qt.Key.Key_S:
            if self.on_save_and_next_requested is not None:
                self.on_save_and_next_requested()
                event.accept()
                return
        if event.key() == Qt.Key.Key_Z:
            if self.undo_last_action():
                event.accept()
                return
        super().keyPressEvent(event)

    def apply_line_erase(self) -> bool:
        if self._line_erase_start_display is None or self._line_erase_preview_end_display is None:
            return False
        if self.tissue_mask_full is None or self.artifact_mask_full is None:
            return False
        start_full = self._display_to_full_xy(self._line_erase_start_display)
        end_full = self._display_to_full_xy(self._line_erase_preview_end_display)
        self._remember_undo_state()
        thickness = 2
        for mask in (self.tissue_mask_full, self.artifact_mask_full):
            cv2.line(mask, start_full, end_full, 0, thickness=thickness, lineType=cv2.LINE_8)
            cv2.circle(mask, start_full, max(1, thickness // 2), 0, thickness=-1, lineType=cv2.LINE_8)
            cv2.circle(mask, end_full, max(1, thickness // 2), 0, thickness=-1, lineType=cv2.LINE_8)
        start = self._line_erase_start_display
        self._rebuild_display_buffers()
        self._set_line_erase_preview(None, None)
        self.refresh()
        if self.on_mask_changed is not None:
            self.on_mask_changed()
        self._set_hover_coord(start)
        return True

    def delete_component_under_cursor(self) -> bool:
        if self.hover_pos_display is None:
            return False
        if self.tissue_mask_full is None or self.artifact_mask_full is None:
            return False
        full_x, full_y = self._display_to_full_xy(self.hover_pos_display)
        target_full = self.tissue_mask_full if self.active_layer == "tissue" else self.artifact_mask_full
        if target_full[full_y, full_x] <= 0:
            return False
        binary = (target_full > 0).astype(np.uint8)
        self._remember_undo_state()
        num_labels, labels = cv2.connectedComponents(binary, connectivity=8)
        if num_labels <= 1:
            return False
        target_label = int(labels[full_y, full_x])
        if target_label <= 0:
            return False
        target_full[labels == target_label] = 0
        self._rebuild_display_buffers()
        self.refresh()
        if self.on_mask_changed is not None:
            self.on_mask_changed()
        return True

    def keep_largest_tissue_component(self) -> None:
        if self.tissue_mask_full is None:
            return
        num, labels, stats, _ = cv2.connectedComponentsWithStats((self.tissue_mask_full > 0).astype(np.uint8), 8)
        if num <= 1:
            return
        self._remember_undo_state()
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        self.tissue_mask_full = (labels == largest).astype(np.uint8) * 255
        self._rebuild_display_buffers()
        self.refresh()
        if self.on_mask_changed is not None:
            self.on_mask_changed()

    def _closing_fill_class_for_component(
        self,
        component_mask: np.ndarray,
        tissue_mask: np.ndarray,
        artifact_mask: np.ndarray,
    ) -> str:
        component_u8 = component_mask.astype(np.uint8)
        eroded = cv2.erode(component_u8, np.ones((3, 3), np.uint8), iterations=1) > 0
        boundary = component_mask & ~eroded

        def interior_area(mask: np.ndarray) -> int:
            inside = mask & component_mask
            if not inside.any():
                return 0
            num_labels, labels = cv2.connectedComponents(inside.astype(np.uint8), connectivity=8)
            area = 0
            for label_idx in range(1, num_labels):
                comp = labels == label_idx
                if not np.any(comp & boundary):
                    area += int(comp.sum())
            return area

        tissue_interior_area = interior_area(tissue_mask)
        artifact_interior_area = interior_area(artifact_mask)

        if artifact_interior_area > 0 and tissue_interior_area <= 0:
            return "artifact"
        if tissue_interior_area > 0 and artifact_interior_area <= 0:
            return "tissue"
        if artifact_interior_area > 0 and tissue_interior_area > 0:
            return "artifact" if artifact_interior_area >= tissue_interior_area else "tissue"
        return "tissue"

    def _interior_pixels_for_component(self, component_mask: np.ndarray, mask: np.ndarray) -> np.ndarray:
        component_u8 = component_mask.astype(np.uint8)
        eroded = cv2.erode(component_u8, np.ones((3, 3), np.uint8), iterations=1) > 0
        boundary = component_mask & ~eroded
        inside = mask & component_mask
        if not inside.any():
            return np.zeros_like(component_mask, dtype=bool)
        num_labels, labels = cv2.connectedComponents(inside.astype(np.uint8), connectivity=8)
        kept = np.zeros_like(component_mask, dtype=bool)
        for label_idx in range(1, num_labels):
            comp = labels == label_idx
            if not np.any(comp & boundary):
                kept |= comp
        return kept

    def close_and_fill_tissue_gaps(self) -> None:
        if self.tissue_mask_full is None or self.artifact_mask_full is None:
            return
        tissue = self.tissue_mask_full > 0
        artifact = (self.artifact_mask_full > 0) & ~tissue
        combined = tissue | artifact
        if not combined.any():
            return

        context = self._combined_component_context()
        selected_component_masks: list[np.ndarray] = []
        if context is not None and self.component_group_marks:
            labels, label_to_rank, _ = context
            keep_ranks = {int(rank) for rank in self.component_group_marks.keys() if int(rank) > 0}
            keep_labels = [label for label, rank in label_to_rank.items() if int(rank) in keep_ranks]
            for label_idx in keep_labels:
                comp = labels == int(label_idx)
                if comp.any():
                    selected_component_masks.append(comp)

        self._remember_undo_state()

        if selected_component_masks:
            selected_union = np.zeros_like(combined, dtype=bool)
            tissue_fill = np.zeros_like(tissue, dtype=bool)
            artifact_fill = np.zeros_like(artifact, dtype=bool)
            tissue_to_artifact = np.zeros_like(tissue, dtype=bool)
            artifact_to_tissue = np.zeros_like(artifact, dtype=bool)
            kernel = np.ones((9, 9), np.uint8)

            for component_mask in selected_component_masks:
                selected_union |= component_mask
                comp_tissue = tissue & component_mask
                comp_artifact = artifact & component_mask
                comp_combined = comp_tissue | comp_artifact
                if not comp_combined.any():
                    continue
                comp_closed = cv2.morphologyEx(comp_combined.astype(np.uint8) * 255, cv2.MORPH_CLOSE, kernel) > 0
                comp_closed = binary_fill_holes(comp_closed > 0)
                closed_component_mask = comp_closed
                added_component = closed_component_mask & ~comp_combined
                fill_class = self._closing_fill_class_for_component(closed_component_mask, comp_tissue, comp_artifact)
                interior_tissue = self._interior_pixels_for_component(closed_component_mask, comp_tissue)
                interior_artifact = self._interior_pixels_for_component(closed_component_mask, comp_artifact)
                if fill_class == "artifact":
                    artifact_fill |= added_component
                    tissue_to_artifact |= interior_tissue
                else:
                    tissue_fill |= added_component
                    artifact_to_tissue |= interior_artifact

            kept_tissue = tissue & selected_union
            kept_artifact = artifact & selected_union
            new_tissue = (kept_tissue & ~tissue_to_artifact) | tissue_fill | artifact_to_tissue
            new_artifact = (kept_artifact & ~artifact_to_tissue) | artifact_fill | tissue_to_artifact
            new_tissue &= ~new_artifact
            self.tissue_mask_full = (new_tissue.astype(np.uint8) * 255)
            self.artifact_mask_full = (new_artifact.astype(np.uint8) * 255)
            self._rebuild_display_buffers()
            self.refresh()
            if self.on_mask_changed is not None:
                self.on_mask_changed()
            return

        combined_u8 = combined.astype(np.uint8)
        num, labels, stats, _ = cv2.connectedComponentsWithStats(combined_u8, 8)
        if num > 1:
            largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            keep_region = labels == largest
        else:
            keep_region = combined

        tissue = tissue & keep_region
        artifact = artifact & keep_region
        combined = tissue | artifact

        kernel = np.ones((9, 9), np.uint8)
        combined_closed = cv2.morphologyEx(combined.astype(np.uint8) * 255, cv2.MORPH_CLOSE, kernel) > 0
        combined_closed = binary_fill_holes(combined_closed > 0)
        added_region = combined_closed & ~combined

        tissue_fill = np.zeros_like(tissue, dtype=bool)
        artifact_fill = np.zeros_like(artifact, dtype=bool)
        tissue_to_artifact = np.zeros_like(tissue, dtype=bool)
        artifact_to_tissue = np.zeros_like(artifact, dtype=bool)
        num_closed, closed_labels = cv2.connectedComponents(combined_closed.astype(np.uint8), connectivity=8)
        for label_idx in range(1, num_closed):
            component_mask = closed_labels == label_idx
            added_component = component_mask & added_region
            fill_class = self._closing_fill_class_for_component(component_mask, tissue, artifact)
            interior_tissue = self._interior_pixels_for_component(component_mask, tissue)
            interior_artifact = self._interior_pixels_for_component(component_mask, artifact)
            if fill_class == "artifact":
                artifact_fill |= added_component
                tissue_to_artifact |= interior_tissue
            else:
                tissue_fill |= added_component
                artifact_to_tissue |= interior_artifact

        new_tissue = (tissue & ~tissue_to_artifact) | tissue_fill | artifact_to_tissue
        new_artifact = (artifact & ~artifact_to_tissue) | artifact_fill | tissue_to_artifact
        new_tissue &= ~new_artifact
        self.tissue_mask_full = (new_tissue.astype(np.uint8) * 255)
        self.artifact_mask_full = (new_artifact.astype(np.uint8) * 255)
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
        self._remember_undo_state()
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

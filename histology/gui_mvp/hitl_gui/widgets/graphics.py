from __future__ import annotations

from typing import Callable, Optional

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QCursor, QImage, QPainter, QPen, QPixmap, QBrush, QPainterPath, QPolygonF
from PySide6.QtWidgets import (
    QGraphicsEllipseItem,
    QGraphicsPathItem,
    QGraphicsPixmapItem,
    QGraphicsPolygonItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsSimpleTextItem,
    QGraphicsView,
)


def qimage_from_rgb_bytes(width: int, height: int, data: bytes) -> QImage:
    return QImage(data, width, height, width * 3, QImage.Format.Format_RGB888).copy()


def qimage_from_rgb_array(arr: np.ndarray) -> QImage:
    h, w = arr.shape[:2]
    return qimage_from_rgb_bytes(w, h, arr.astype(np.uint8).tobytes())


def qimage_from_rgba_array(arr: np.ndarray) -> QImage:
    h, w = arr.shape[:2]
    return QImage(arr.astype(np.uint8).tobytes(), w, h, w * 4, QImage.Format.Format_RGBA8888).copy()


def _precision_anchor_cursor() -> QCursor:
    pix = QPixmap(19, 19)
    pix.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pix)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    pen_outer = QPen(QColor(255, 255, 255, 235), 3)
    pen_inner = QPen(QColor(220, 48, 48, 245), 1)
    painter.setPen(pen_outer)
    painter.drawLine(9, 1, 9, 17)
    painter.drawLine(1, 9, 17, 9)
    painter.setPen(pen_inner)
    painter.drawLine(9, 1, 9, 17)
    painter.drawLine(1, 9, 17, 9)
    painter.setBrush(QBrush(QColor(220, 48, 48, 245)))
    painter.setPen(QPen(QColor(255, 255, 255, 235), 1))
    painter.drawEllipse(7, 7, 4, 4)
    painter.end()
    return QCursor(pix, 9, 9)


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


class RibbonAnnotationView(QGraphicsView):
    curvesChanged = Signal(object)
    probeChanged = Signal(object)

    def __init__(self) -> None:
        super().__init__()
        self.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMouseTracking(True)
        self.viewport().setMouseTracking(True)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._pixmap_item: Optional[QGraphicsPixmapItem] = None
        self._tile_polygon_items: list[QGraphicsPolygonItem] = []
        self._tile_label_items: list[QGraphicsSimpleTextItem] = []
        self._curve_path_items: dict[str, QGraphicsPathItem] = {}
        self._curve_point_items: dict[str, list[QGraphicsEllipseItem]] = {"surface": [], "interface": []}
        self._preview_path_item: Optional[QGraphicsPathItem] = None
        self._curves: dict[str, list[tuple[float, float]]] = {"surface": [], "interface": []}
        self._active_curve: Optional[str] = None
        self._curve_drawing_active: bool = False
        self._last_curve_scene_pos: Optional[QPointF] = None
        self._preview_scene_pos: Optional[QPointF] = None
        self._tile_rows: list[dict[str, object]] = []
        self._tile_include_map: dict[int, bool] = {}
        self._current_tile_index: int | None = None
        self._active_probe: bool = False
        self._probe_stage: str = "idle"
        self._probe_depth_start: Optional[QPointF] = None
        self._probe_depth_end: Optional[QPointF] = None
        self._probe_width_vector: Optional[QPointF] = None
        self._probe_preview_pos: Optional[QPointF] = None
        self._probe_polygon_item: Optional[QGraphicsPolygonItem] = None
        self._probe_axis_item: Optional[QGraphicsPathItem] = None
        self._probe_tick_items: list[QGraphicsPathItem] = []
        self._saved_probe_items: list[QGraphicsPolygonItem | QGraphicsPathItem | QGraphicsSimpleTextItem] = []
        self._saved_probe_payloads: list[dict[str, object]] = []

    def clear_all(self) -> None:
        self._scene.clear()
        self._pixmap_item = None
        self._tile_polygon_items = []
        self._tile_label_items = []
        self._curve_path_items = {}
        self._curve_point_items = {"surface": [], "interface": []}
        self._preview_path_item = None
        self._preview_scene_pos = None
        self._probe_polygon_item = None
        self._probe_axis_item = None
        self._probe_tick_items = []
        self._saved_probe_items = []

    def set_rgb_image(self, width: int, height: int, data: bytes) -> None:
        self.clear_all()
        image = qimage_from_rgb_bytes(width, height, data)
        pixmap = QPixmap.fromImage(image)
        self._pixmap_item = self._scene.addPixmap(pixmap)
        self._scene.setSceneRect(QRectF(0, 0, width, height))
        self.fitInView(self._scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
        self._rebuild_overlays()

    def capture_view_state(self) -> dict[str, object]:
        center = self.mapToScene(self.viewport().rect().center())
        return {
            "center_scene_xy": [float(center.x()), float(center.y())],
            "transform": self.transform(),
        }

    def restore_view_state(self, state: dict[str, object] | None) -> None:
        if not isinstance(state, dict):
            return
        transform = state.get("transform")
        if transform is not None:
            self.setTransform(transform)
        center_xy = state.get("center_scene_xy")
        if isinstance(center_xy, (list, tuple)) and len(center_xy) == 2:
            self.centerOn(float(center_xy[0]), float(center_xy[1]))

    @property
    def scene_obj(self) -> QGraphicsScene:
        return self._scene

    def set_active_curve(self, name: Optional[str]) -> None:
        if name not in {None, "surface", "interface"}:
            return
        if name is not None:
            self._active_probe = False
            self._probe_stage = "idle"
        self._active_curve = name
        self._curve_drawing_active = False
        self._last_curve_scene_pos = None
        self._preview_scene_pos = None
        self.setDragMode(
            QGraphicsView.DragMode.NoDrag if self._active_curve is not None else QGraphicsView.DragMode.ScrollHandDrag
        )
        self.viewport().setCursor(
            Qt.CursorShape.CrossCursor if self._active_curve is not None else Qt.CursorShape.OpenHandCursor
        )
        self._rebuild_preview_path()

    def set_probe_mode(self, active: bool) -> None:
        self._active_probe = bool(active)
        if self._active_probe:
            self.set_active_curve(None)
            self._probe_stage = "depth"
            self.setDragMode(QGraphicsView.DragMode.NoDrag)
            self.viewport().setCursor(Qt.CursorShape.CrossCursor)
        else:
            self._probe_stage = "idle"
            self._probe_preview_pos = None
            self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
            self.viewport().setCursor(Qt.CursorShape.OpenHandCursor)
        self._rebuild_overlays()

    def clear_probe(self) -> None:
        self._probe_depth_start = None
        self._probe_depth_end = None
        self._probe_width_vector = None
        self._probe_preview_pos = None
        self._probe_stage = "depth" if self._active_probe else "idle"
        self._rebuild_overlays()
        self.probeChanged.emit(self.probe_snapshot())

    def reverse_probe_depth(self) -> None:
        if self._probe_depth_start is None or self._probe_depth_end is None:
            return
        self._probe_depth_start, self._probe_depth_end = self._probe_depth_end, self._probe_depth_start
        if self._probe_width_vector is not None:
            self._probe_width_vector = QPointF(-self._probe_width_vector.x(), -self._probe_width_vector.y())
        self._rebuild_overlays()
        self.probeChanged.emit(self.probe_snapshot())

    def set_saved_probes(self, probes: list[dict[str, object]] | None) -> None:
        self._saved_probe_payloads = [dict(row) for row in list(probes or [])]
        self._rebuild_overlays()

    def probe_snapshot(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "active": bool(self._active_probe),
            "stage": str(self._probe_stage),
            "complete": False,
        }
        if self._probe_depth_start is not None:
            payload["depth_start_scene_xy"] = [float(self._probe_depth_start.x()), float(self._probe_depth_start.y())]
        if self._probe_depth_end is not None:
            payload["depth_end_scene_xy"] = [float(self._probe_depth_end.x()), float(self._probe_depth_end.y())]
        if self._probe_width_vector is not None:
            payload["width_vector_scene_xy"] = [float(self._probe_width_vector.x()), float(self._probe_width_vector.y())]
            payload["width_px"] = float(2.0 * np.hypot(float(self._probe_width_vector.x()), float(self._probe_width_vector.y())))
        payload["complete"] = bool(
            self._probe_depth_start is not None
            and self._probe_depth_end is not None
            and self._probe_width_vector is not None
        )
        return payload

    def active_curve(self) -> Optional[str]:
        return self._active_curve

    def curve_points(self, name: str) -> list[tuple[float, float]]:
        return list(self._curves.get(name, []))

    def set_curve_points(self, name: str, points_xy: list[tuple[float, float]] | np.ndarray | None) -> None:
        if name not in self._curves:
            return
        pts: list[tuple[float, float]] = []
        if points_xy is not None:
            for pt in list(points_xy):
                if len(pt) < 2:
                    continue
                pts.append((float(pt[0]), float(pt[1])))
        self._curves[name] = pts
        self._rebuild_overlays()
        self.curvesChanged.emit(self.curve_snapshot())

    def clear_curve(self, name: str) -> None:
        self.set_curve_points(name, [])

    def undo_last_point(self, name: Optional[str] = None) -> None:
        target = name or self._active_curve
        if target not in self._curves:
            return
        if not self._curves[target]:
            return
        self._curves[target] = self._curves[target][:-1]
        self._rebuild_overlays()
        self.curvesChanged.emit(self.curve_snapshot())

    def curve_snapshot(self) -> dict[str, list[list[float]]]:
        return {
            "surface": [[float(x), float(y)] for x, y in self._curves.get("surface", [])],
            "interface": [[float(x), float(y)] for x, y in self._curves.get("interface", [])],
        }

    def set_tile_overlay(
        self,
        tile_rows: list[dict[str, object]],
        *,
        include_map: dict[int, bool] | None = None,
        current_tile_index: int | None = None,
    ) -> None:
        self._tile_rows = [dict(row) for row in list(tile_rows or [])]
        self._tile_include_map = {int(k): bool(v) for k, v in dict(include_map or {}).items()}
        self._current_tile_index = None if current_tile_index is None else int(current_tile_index)
        self._rebuild_overlays()

    def _scene_polygon_for_row(self, row: dict[str, object]) -> np.ndarray | None:
        bbox = row.get("final_scene_bbox_xyxy") or row.get("pred_scene_bbox_xyxy") or row.get("nominal_scene_bbox_xyxy")
        if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            x0, y0, x1, y1 = [float(v) for v in bbox]
            return np.asarray([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32)
        candidates = (
            row.get("final_scene_polygon_xy"),
            row.get("pred_scene_polygon_xy"),
            row.get("nominal_scene_polygon_xy"),
        )
        for value in candidates:
            arr = np.asarray(value, dtype=np.float32)
            if arr.ndim == 2 and arr.shape[0] >= 3 and arr.shape[1] == 2:
                return arr
        bbox = row.get("final_scene_bbox_xyxy") or row.get("pred_scene_bbox_xyxy") or row.get("nominal_scene_bbox_xyxy")
        if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            x0, y0, x1, y1 = [float(v) for v in bbox]
            return np.asarray([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32)
        return None

    def _rebuild_overlays(self) -> None:
        for item in self._tile_polygon_items:
            self._scene.removeItem(item)
        for item in self._tile_label_items:
            self._scene.removeItem(item)
        for item in self._curve_path_items.values():
            self._scene.removeItem(item)
        for items in self._curve_point_items.values():
            for item in items:
                self._scene.removeItem(item)
        for item in self._probe_tick_items:
            self._scene.removeItem(item)
        if self._probe_polygon_item is not None:
            self._scene.removeItem(self._probe_polygon_item)
        if self._probe_axis_item is not None:
            self._scene.removeItem(self._probe_axis_item)
        for item in self._saved_probe_items:
            self._scene.removeItem(item)
        self._tile_polygon_items = []
        self._tile_label_items = []
        self._curve_path_items = {}
        self._curve_point_items = {"surface": [], "interface": []}
        self._probe_polygon_item = None
        self._probe_axis_item = None
        self._probe_tick_items = []
        self._saved_probe_items = []

        for row in self._tile_rows:
            pts = self._scene_polygon_for_row(row)
            if pts is None:
                continue
            idx = int(row.get("tile_index", -1))
            include = bool(self._tile_include_map.get(idx, False))
            color = QColor(56, 168, 106, 220) if include else QColor(198, 72, 72, 220)
            if self._current_tile_index is not None and idx == int(self._current_tile_index):
                color = QColor(255, 215, 64, 245)
            poly = QPolygonF([QPointF(float(x), float(y)) for x, y in pts])
            item = QGraphicsPolygonItem(poly)
            item.setPen(QPen(color, 2))
            fill = QColor(color)
            fill.setAlpha(40 if include else 24)
            item.setBrush(QBrush(fill))
            self._scene.addItem(item)
            self._tile_polygon_items.append(item)
            cx = float(np.mean(pts[:, 0]))
            cy = float(np.mean(pts[:, 1]))
            label = QGraphicsSimpleTextItem(str(row.get("label") or f"T{idx:02d}"))
            label.setBrush(QBrush(QColor(25, 25, 25)))
            label.setPos(cx - 22.0, cy - 10.0)
            self._scene.addItem(label)
            self._tile_label_items.append(label)

        curve_styles = {
            "surface": QColor(28, 120, 255, 245),
            "interface": QColor(255, 96, 32, 245),
        }
        for name, pts in self._curves.items():
            if pts:
                path = QPainterPath(QPointF(float(pts[0][0]), float(pts[0][1])))
                for x, y in pts[1:]:
                    path.lineTo(float(x), float(y))
                path_item = QGraphicsPathItem(path)
                pen = QPen(curve_styles[name], 3)
                path_item.setPen(pen)
                self._scene.addItem(path_item)
                self._curve_path_items[name] = path_item
                point_items: list[QGraphicsEllipseItem] = []
                for i, (x, y) in enumerate(pts):
                    ell = QGraphicsEllipseItem(float(x - 4.0), float(y - 4.0), 8.0, 8.0)
                    ell.setPen(QPen(QColor(255, 255, 255, 235), 1))
                    ell.setBrush(QBrush(curve_styles[name]))
                    self._scene.addItem(ell)
                    point_items.append(ell)
                    if i == 0:
                        txt = QGraphicsSimpleTextItem("start")
                        txt.setBrush(QBrush(curve_styles[name]))
                        txt.setPos(float(x + 5.0), float(y + 5.0))
                        self._scene.addItem(txt)
                        self._tile_label_items.append(txt)
                self._curve_point_items[name] = point_items
        self._rebuild_saved_probe_items()
        self._rebuild_probe_overlay()
        self._rebuild_preview_path()

    def _probe_geometry_points(self, preview_pos: QPointF | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        if self._probe_depth_start is None or self._probe_depth_end is None:
            return None
        start = np.asarray([float(self._probe_depth_start.x()), float(self._probe_depth_start.y())], dtype=np.float32)
        end = np.asarray([float(self._probe_depth_end.x()), float(self._probe_depth_end.y())], dtype=np.float32)
        axis = end - start
        length = float(np.linalg.norm(axis))
        if length <= 1e-3:
            return None
        unit = axis / length
        normal = np.asarray([-unit[1], unit[0]], dtype=np.float32)
        width_vec = None
        if self._probe_width_vector is not None:
            width_vec = np.asarray([float(self._probe_width_vector.x()), float(self._probe_width_vector.y())], dtype=np.float32)
        elif preview_pos is not None:
            p = np.asarray([float(preview_pos.x()), float(preview_pos.y())], dtype=np.float32)
            mid = 0.5 * (start + end)
            width_vec = normal * float(np.dot(p - mid, normal))
        if width_vec is None:
            return None
        if float(np.linalg.norm(width_vec)) <= 1.0:
            return None
        corners = np.asarray([start - width_vec, start + width_vec, end + width_vec, end - width_vec], dtype=np.float32)
        return corners, start, end

    def _rebuild_saved_probe_items(self) -> None:
        color = QColor(180, 90, 220, 150)
        for row in self._saved_probe_payloads:
            corners = np.asarray(row.get("oriented_rectangle_corners_scene_xy"), dtype=np.float32)
            if corners.ndim != 2 or corners.shape[0] < 4 or corners.shape[1] != 2:
                continue
            poly = QPolygonF([QPointF(float(x), float(y)) for x, y in corners[:4]])
            item = QGraphicsPolygonItem(poly)
            item.setPen(QPen(color, 2, Qt.PenStyle.DotLine))
            item.setBrush(QBrush(QColor(180, 90, 220, 18)))
            self._scene.addItem(item)
            self._saved_probe_items.append(item)
            label = QGraphicsSimpleTextItem(str(row.get("probe_id") or "probe"))
            label.setBrush(QBrush(color))
            label.setPos(float(corners[0, 0]) + 4.0, float(corners[0, 1]) - 16.0)
            self._scene.addItem(label)
            self._saved_probe_items.append(label)

    def _rebuild_probe_overlay(self) -> None:
        geom = self._probe_geometry_points(self._probe_preview_pos)
        if geom is None:
            return
        corners, start, end = geom
        color = QColor(255, 208, 48, 245)
        poly = QPolygonF([QPointF(float(x), float(y)) for x, y in corners])
        item = QGraphicsPolygonItem(poly)
        item.setPen(QPen(color, 3))
        item.setBrush(QBrush(QColor(255, 208, 48, 28)))
        self._scene.addItem(item)
        self._probe_polygon_item = item
        path = QPainterPath(QPointF(float(start[0]), float(start[1])))
        path.lineTo(QPointF(float(end[0]), float(end[1])))
        axis_item = QGraphicsPathItem(path)
        axis_item.setPen(QPen(QColor(28, 120, 255, 245), 3))
        self._scene.addItem(axis_item)
        self._probe_axis_item = axis_item
        for frac in np.linspace(0.0, 1.0, 6, dtype=np.float32):
            p0 = corners[0] * (1.0 - float(frac)) + corners[3] * float(frac)
            p1 = corners[1] * (1.0 - float(frac)) + corners[2] * float(frac)
            tick = QPainterPath(QPointF(float(p0[0]), float(p0[1])))
            tick.lineTo(QPointF(float(p1[0]), float(p1[1])))
            tick_item = QGraphicsPathItem(tick)
            tick_item.setPen(QPen(QColor(255, 208, 48, 175), 1, Qt.PenStyle.DashLine))
            self._scene.addItem(tick_item)
            self._probe_tick_items.append(tick_item)

    def _rebuild_preview_path(self) -> None:
        if self._preview_path_item is not None:
            self._scene.removeItem(self._preview_path_item)
            self._preview_path_item = None
        if self._active_curve is None or self._preview_scene_pos is None:
            return
        pts = self._curves.get(self._active_curve, [])
        if not pts:
            return
        path = QPainterPath(QPointF(float(pts[0][0]), float(pts[0][1])))
        for x, y in pts[1:]:
            path.lineTo(float(x), float(y))
        path.lineTo(self._preview_scene_pos)
        item = QGraphicsPathItem(path)
        color = QColor(28, 120, 255, 180) if self._active_curve == "surface" else QColor(255, 96, 32, 180)
        item.setPen(QPen(color, 2, Qt.PenStyle.DashLine))
        self._scene.addItem(item)
        self._preview_path_item = item

    def _clamp_scene_pos(self, scene_pos: QPointF) -> QPointF:
        rect = self._scene.sceneRect()
        return QPointF(
            min(max(float(scene_pos.x()), float(rect.left())), float(rect.right())),
            min(max(float(scene_pos.y()), float(rect.top())), float(rect.bottom())),
        )

    def _append_curve_point_if_far_enough(self, scene_pos: QPointF) -> bool:
        if self._active_curve is None:
            return False
        last = self._last_curve_scene_pos
        if last is not None:
            dx = float(scene_pos.x() - last.x())
            dy = float(scene_pos.y() - last.y())
            if (dx * dx + dy * dy) < 9.0:
                return False
        pts = list(self._curves.get(self._active_curve, []))
        pts.append((float(scene_pos.x()), float(scene_pos.y())))
        self._curves[self._active_curve] = pts
        self._last_curve_scene_pos = QPointF(scene_pos)
        return True

    def mousePressEvent(self, event) -> None:
        if self._active_probe and event.button() == Qt.MouseButton.LeftButton:
            scene_pos = self._clamp_scene_pos(self.mapToScene(event.pos()))
            if self._probe_stage in {"depth", "idle"}:
                self._probe_depth_start = QPointF(scene_pos)
                self._probe_depth_end = QPointF(scene_pos)
                self._probe_width_vector = None
                self._probe_stage = "depth_drag"
            elif self._probe_stage == "width":
                self._probe_preview_pos = QPointF(scene_pos)
                self._probe_stage = "width_drag"
            event.accept()
            return
        if self._active_curve is not None and event.button() == Qt.MouseButton.LeftButton:
            scene_pos = self._clamp_scene_pos(self.mapToScene(event.pos()))
            self._curve_drawing_active = True
            self._preview_scene_pos = None
            changed = self._append_curve_point_if_far_enough(scene_pos)
            if changed:
                self._rebuild_overlays()
                self.curvesChanged.emit(self.curve_snapshot())
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self._active_probe:
            scene_pos = self._clamp_scene_pos(self.mapToScene(event.pos()))
            if self._probe_stage == "depth_drag":
                self._probe_depth_end = QPointF(scene_pos)
                self._rebuild_overlays()
                self.probeChanged.emit(self.probe_snapshot())
                event.accept()
                return
            if self._probe_stage == "width_drag":
                self._probe_preview_pos = QPointF(scene_pos)
                self._rebuild_overlays()
                self.probeChanged.emit(self.probe_snapshot())
                event.accept()
                return
        if self._active_curve is not None and self._curve_drawing_active:
            scene_pos = self._clamp_scene_pos(self.mapToScene(event.pos()))
            changed = self._append_curve_point_if_far_enough(scene_pos)
            if changed:
                self._rebuild_overlays()
                self.curvesChanged.emit(self.curve_snapshot())
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if self._active_probe and event.button() == Qt.MouseButton.LeftButton:
            scene_pos = self._clamp_scene_pos(self.mapToScene(event.pos()))
            if self._probe_stage == "depth_drag":
                self._probe_depth_end = QPointF(scene_pos)
                self._probe_stage = "width"
                self._probe_preview_pos = None
                self._rebuild_overlays()
                self.probeChanged.emit(self.probe_snapshot())
                event.accept()
                return
            if self._probe_stage == "width_drag":
                if self._probe_depth_start is not None and self._probe_depth_end is not None:
                    start = np.asarray([float(self._probe_depth_start.x()), float(self._probe_depth_start.y())], dtype=np.float32)
                    end = np.asarray([float(self._probe_depth_end.x()), float(self._probe_depth_end.y())], dtype=np.float32)
                    axis = end - start
                    length = float(np.linalg.norm(axis))
                    if length > 1e-3:
                        unit = axis / length
                        normal = np.asarray([-unit[1], unit[0]], dtype=np.float32)
                        mid = 0.5 * (start + end)
                        p = np.asarray([float(scene_pos.x()), float(scene_pos.y())], dtype=np.float32)
                        width_vec = normal * float(np.dot(p - mid, normal))
                        self._probe_width_vector = QPointF(float(width_vec[0]), float(width_vec[1]))
                self._probe_stage = "complete"
                self._probe_preview_pos = None
                self._rebuild_overlays()
                self.probeChanged.emit(self.probe_snapshot())
                event.accept()
                return
        if self._active_curve is not None and event.button() == Qt.MouseButton.LeftButton and self._curve_drawing_active:
            scene_pos = self._clamp_scene_pos(self.mapToScene(event.pos()))
            changed = self._append_curve_point_if_far_enough(scene_pos)
            self._curve_drawing_active = False
            self._preview_scene_pos = None
            if changed:
                self._rebuild_overlays()
                self.curvesChanged.emit(self.curve_snapshot())
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event) -> None:
        if event.angleDelta().y() == 0:
            super().wheelEvent(event)
            return
        old_anchor = self.transformationAnchor()
        try:
            self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
            factor = 1.15 if event.angleDelta().y() > 0 else (1.0 / 1.15)
            self.scale(factor, factor)
            event.accept()
        finally:
            self.setTransformationAnchor(old_anchor)


class ConfocalAlignmentView(QGraphicsView):
    transformEdited = Signal(float, float, float, float)
    diagnosticPointPlaced = Signal(object)
    diagnosticStateChanged = Signal(object)
    tileSelectionChanged = Signal(object)

    def __init__(self) -> None:
        super().__init__()
        self.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMouseTracking(True)
        self.viewport().setMouseTracking(True)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._fixed_item: Optional[QGraphicsPixmapItem] = None
        self._overlay_item: Optional[QGraphicsPixmapItem] = None
        self._overlay_selection_item: Optional[QGraphicsRectItem] = None
        self._overlay_handle_items: dict[str, QGraphicsRectItem] = {}
        self._fixed_shape_hw: tuple[int, int] = (1, 1)
        self._overlay_rgb: Optional[np.ndarray] = None
        self._overlay_plane_u8: Optional[np.ndarray] = None
        self._overlay_alpha_source_u8: Optional[np.ndarray] = None
        self._overlay_flip_lr: bool = False
        self._overlay_flip_ud: bool = False
        self._overlay_opacity: float = 0.85
        self._tx_px: float = 0.0
        self._ty_px: float = 0.0
        self._angle_deg: float = 0.0
        self._scale: float = 1.0
        self._overlay_selected: bool = False
        self._overlay_transform_locked: bool = False
        self._overlay_frozen_lock: bool = False
        self._overlay_tile_defs: list[dict[str, object]] = []
        self._overlay_tile_items: dict[int, tuple[QGraphicsRectItem, QGraphicsSimpleTextItem]] = {}
        self._selected_tile_index: int | None = None
        self._selected_tile_indices: set[int] = set()
        self._frozen_tile_indices: set[int] = set()
        self._accepted_tile_indices: set[int] = set()
        self._hold_tile_indices: set[int] = set()
        self._frontier_tile_indices: set[int] = set()
        self._drag_mode: Optional[str] = None
        self._press_scene_pos = QPointF()
        self._press_tx_px: float = 0.0
        self._press_ty_px: float = 0.0
        self._press_angle_deg: float = 0.0
        self._press_scale: float = 1.0
        self._scale_pivot_local = QPointF()
        self._scale_pivot_scene = QPointF()
        self._scale_initial_distance: float = 1.0
        self._diagnostic_active: bool = False
        self._diagnostic_next_group: str = "A"
        self._diagnostic_next_index: int = 1
        self._points_a_scene: dict[int, tuple[float, float]] = {}
        self._points_b_raw: dict[int, tuple[float, float]] = {}
        self._diagnostic_history: list[tuple[str, int, Optional[tuple[float, float]]]] = []
        self._a_marker_items: dict[int, tuple[QGraphicsEllipseItem, QGraphicsSimpleTextItem]] = {}
        self._b_marker_items: dict[int, tuple[QGraphicsEllipseItem, QGraphicsSimpleTextItem]] = {}
        self._default_cursor = QCursor(Qt.CursorShape.ArrowCursor)
        self._anchor_cursor = _precision_anchor_cursor()
        self._last_scene_pos = QPointF()

    def clear_all(self) -> None:
        self._scene.clear()
        self._fixed_item = None
        self._overlay_item = None
        self._overlay_selection_item = None
        self._overlay_handle_items.clear()
        self._overlay_rgb = None
        self._overlay_plane_u8 = None
        self._overlay_alpha_source_u8 = None
        self._fixed_shape_hw = (1, 1)
        self._overlay_selected = False
        self._overlay_tile_defs = []
        self._overlay_tile_items.clear()
        self._selected_tile_index = None
        self._selected_tile_indices.clear()
        self._frozen_tile_indices.clear()
        self._accepted_tile_indices.clear()
        self._hold_tile_indices.clear()
        self._frontier_tile_indices.clear()
        self._overlay_frozen_lock = False
        self._a_marker_items.clear()
        self._b_marker_items.clear()

    def set_fixed_rgb(self, rgb: Optional[np.ndarray]) -> None:
        if rgb is None:
            self.clear_all()
            return
        arr = np.asarray(rgb, dtype=np.uint8)
        self._scene.clear()
        self._fixed_item = self._scene.addPixmap(QPixmap.fromImage(qimage_from_rgb_array(arr)))
        self._overlay_item = None
        self._overlay_selection_item = None
        self._overlay_handle_items.clear()
        self._fixed_shape_hw = (int(arr.shape[0]), int(arr.shape[1]))
        self._scene.setSceneRect(QRectF(0.0, 0.0, float(arr.shape[1]), float(arr.shape[0])))
        self.fitInView(self._scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
        if self._overlay_rgb is not None:
            self._install_overlay_item()
        self._redraw_landmark_items()

    def set_overlay_gray(
        self,
        gray_u8: Optional[np.ndarray],
        *,
        alpha_source_u8: Optional[np.ndarray] = None,
        flip_lr: bool = False,
        flip_ud: bool = False,
    ) -> None:
        if gray_u8 is None:
            self._overlay_rgb = None
            self._overlay_plane_u8 = None
            self._overlay_alpha_source_u8 = None
            if self._overlay_item is not None:
                self._scene.removeItem(self._overlay_item)
                self._overlay_item = None
                self._overlay_selection_item = None
                self._overlay_handle_items.clear()
            self._overlay_selected = False
            self._selected_tile_index = None
            self._selected_tile_indices.clear()
            self._overlay_tile_items.clear()
            self._emit_tile_selection()
            return
        self._overlay_plane_u8 = np.asarray(gray_u8, dtype=np.uint8)
        self._overlay_alpha_source_u8 = (
            self._overlay_plane_u8
            if alpha_source_u8 is None
            else np.asarray(alpha_source_u8, dtype=np.uint8)
        )
        self._overlay_flip_lr = bool(flip_lr)
        self._overlay_flip_ud = bool(flip_ud)
        self._render_overlay_rgba()
        if self._fixed_item is not None:
            self._install_overlay_item()

    def set_alignment(self, tx_px: float, ty_px: float, angle_deg: float, scale: float) -> None:
        self._tx_px = float(tx_px)
        self._ty_px = float(ty_px)
        self._angle_deg = float(angle_deg)
        self._scale = max(0.05, float(scale))
        self._apply_overlay_transform()

    def capture_view_state(self) -> dict[str, object]:
        center = self.mapToScene(self.viewport().rect().center())
        return {
            "center_scene_xy": [float(center.x()), float(center.y())],
            "transform": self.transform(),
        }

    def restore_view_state(self, state: dict[str, object] | None) -> None:
        if not isinstance(state, dict):
            return
        transform = state.get("transform")
        if transform is not None:
            self.setTransform(transform)
        center_xy = state.get("center_scene_xy")
        if isinstance(center_xy, (list, tuple)) and len(center_xy) == 2:
            self.centerOn(float(center_xy[0]), float(center_xy[1]))

    def render_scene_rgb(self) -> np.ndarray | None:
        rect = self._scene.sceneRect()
        width = max(1, int(np.ceil(float(rect.width()))))
        height = max(1, int(np.ceil(float(rect.height()))))
        image = QImage(width, height, QImage.Format.Format_RGB888)
        image.fill(QColor(245, 245, 245))
        painter = QPainter(image)
        try:
            self._scene.render(painter, QRectF(0.0, 0.0, float(width), float(height)), rect)
        finally:
            painter.end()
        bits = image.constBits()
        try:
            arr = np.frombuffer(bits, dtype=np.uint8, count=width * height * 3).reshape(height, width, 3).copy()
        except Exception:
            return None
        return arr

    def clear_diagnostic_points(self) -> None:
        self._points_a_scene.clear()
        self._points_b_raw.clear()
        self._diagnostic_history.clear()
        self._diagnostic_active = False
        self._diagnostic_next_group = "A"
        self._diagnostic_next_index = 1
        self._overlay_transform_locked = False
        self._remove_landmark_items()
        self._redraw_landmark_items()
        self._update_interaction_cursor()
        self._emit_diagnostic_state_change("reset")

    def activate_anchor_mode(self) -> None:
        self._diagnostic_active = True
        self._overlay_transform_locked = True
        if not self._points_a_scene and not self._points_b_raw:
            self._diagnostic_next_group = "A"
            self._diagnostic_next_index = 1
        self._set_overlay_selected(False)
        self._update_interaction_cursor()
        self._emit_diagnostic_state_change("anchor_mode_started")

    def diagnostic_snapshot(self) -> dict[str, object]:
        points_b_scene: dict[str, list[float]] = {}
        for idx, raw_xy in self._points_b_raw.items():
            display_x, display_y = self._display_local_from_raw(float(raw_xy[0]), float(raw_xy[1]))
            scene_x = float("nan")
            scene_y = float("nan")
            if self._overlay_item is not None:
                scene_pt = self._overlay_item.mapToScene(QPointF(display_x, display_y))
                scene_x = float(scene_pt.x())
                scene_y = float(scene_pt.y())
            points_b_scene[str(idx)] = [scene_x, scene_y]
        return {
            "diagnostic_active": bool(self._diagnostic_active),
            "next_group": self._diagnostic_next_group,
            "next_index": int(self._diagnostic_next_index),
            "transform_locked": bool(self._transform_interaction_locked()),
            "manual_transform_locked": bool(self._overlay_transform_locked),
            "frozen_transform_locked": bool(self._overlay_frozen_lock),
            "overlay_opacity": float(self._overlay_opacity),
            "points_a_scene": {str(k): [float(v[0]), float(v[1])] for k, v in sorted(self._points_a_scene.items())},
            "points_b_raw": {str(k): [float(v[0]), float(v[1])] for k, v in sorted(self._points_b_raw.items())},
            "points_b_scene": points_b_scene,
            "selected_tile_index": None if self._selected_tile_index is None else int(self._selected_tile_index),
            "selected_tile_label": self._tile_label_for_index(self._selected_tile_index),
            "selected_tile_indices": sorted(int(v) for v in self._selected_tile_indices),
            "selected_tile_labels": [str(self._tile_label_for_index(idx) or f"T{int(idx):02d}") for idx in sorted(self._selected_tile_indices)],
            "accepted_tile_indices": sorted(int(v) for v in self._accepted_tile_indices),
            "hold_tile_indices": sorted(int(v) for v in self._hold_tile_indices),
            "frontier_tile_indices": sorted(int(v) for v in self._frontier_tile_indices),
        }

    def set_overlay_tiles(self, tile_defs: list[dict[str, object]] | None) -> None:
        self._overlay_tile_defs = [dict(row) for row in (tile_defs or [])]
        valid_indices = {int(row.get("tile_index", -1)) for row in self._overlay_tile_defs}
        self._selected_tile_indices = {idx for idx in self._selected_tile_indices if idx in valid_indices}
        if self._selected_tile_index not in valid_indices:
            self._selected_tile_index = sorted(self._selected_tile_indices)[0] if self._selected_tile_indices else None
        elif self._selected_tile_index is not None:
            self._selected_tile_indices.add(int(self._selected_tile_index))
        if self._overlay_item is not None:
            self._rebuild_tile_items()
        self._emit_tile_selection()

    def set_selected_tile(self, tile_index: int | None) -> None:
        idx = None if tile_index is None else int(tile_index)
        valid_indices = {int(row.get("tile_index", -1)) for row in self._overlay_tile_defs}
        if idx is not None and idx not in valid_indices:
            idx = None
        self._selected_tile_index = idx
        self._selected_tile_indices = set() if idx is None else {idx}
        self._refresh_tile_item_styles()
        self._emit_tile_selection()

    def set_selected_tiles(self, tile_indices: set[int] | list[int] | tuple[int, ...] | None, *, primary_tile_index: int | None = None) -> None:
        valid_indices = {int(row.get("tile_index", -1)) for row in self._overlay_tile_defs}
        selected = {int(v) for v in (tile_indices or []) if int(v) in valid_indices}
        primary = None if primary_tile_index is None else int(primary_tile_index)
        if primary is not None and primary not in selected:
            primary = None
        if primary is None and selected:
            primary = sorted(selected)[-1]
        self._selected_tile_indices = selected
        self._selected_tile_index = primary
        self._refresh_tile_item_styles()
        self._emit_tile_selection()

    def set_frozen_tiles(self, tile_indices: set[int] | list[int] | tuple[int, ...] | None) -> None:
        self._frozen_tile_indices = {int(v) for v in (tile_indices or [])}
        self._overlay_frozen_lock = bool(self._frozen_tile_indices)
        if self._overlay_frozen_lock and self._drag_mode in {"translate", "rotate", "scale_lt", "scale_rt", "scale_lb", "scale_rb"}:
            self._drag_mode = None
        self._refresh_tile_item_styles()
        self._update_overlay_selection_visuals()
        self._update_interaction_cursor()
        self._emit_tile_selection()

    def _transform_interaction_locked(self) -> bool:
        return bool(self._overlay_transform_locked or self._overlay_frozen_lock)

    def set_accepted_tiles(self, tile_indices: set[int] | list[int] | tuple[int, ...] | None) -> None:
        self._accepted_tile_indices = {int(v) for v in (tile_indices or [])}
        self._refresh_tile_item_styles()
        self._emit_tile_selection()

    def set_hold_tiles(self, tile_indices: set[int] | list[int] | tuple[int, ...] | None) -> None:
        self._hold_tile_indices = {int(v) for v in (tile_indices or [])}
        self._refresh_tile_item_styles()
        self._emit_tile_selection()

    def set_frontier_tiles(self, tile_indices: set[int] | list[int] | tuple[int, ...] | None) -> None:
        self._frontier_tile_indices = {int(v) for v in (tile_indices or [])}
        self._refresh_tile_item_styles()
        self._emit_tile_selection()

    def _tile_label_for_index(self, tile_index: int | None) -> str | None:
        if tile_index is None:
            return None
        for row in self._overlay_tile_defs:
            if int(row.get("tile_index", -1)) == int(tile_index):
                return str(row.get("label") or f"T{int(tile_index):02d}")
        return None

    def _install_overlay_item(self) -> None:
        if self._overlay_item is not None:
            self._scene.removeItem(self._overlay_item)
            self._overlay_item = None
            self._overlay_selection_item = None
        if self._overlay_rgb is None:
            return
        pixmap = QPixmap.fromImage(qimage_from_rgba_array(self._overlay_rgb))
        self._overlay_item = self._scene.addPixmap(pixmap)
        self._overlay_item.setTransformOriginPoint(pixmap.width() / 2.0, pixmap.height() / 2.0)
        self._overlay_selection_item = QGraphicsRectItem(0.0, 0.0, float(pixmap.width()), float(pixmap.height()), self._overlay_item)
        self._overlay_selection_item.setPen(QPen(QColor(255, 208, 48), 3, Qt.PenStyle.DashLine))
        self._overlay_handle_items.clear()
        for key in ("lt", "rt", "lb", "rb"):
            handle = QGraphicsRectItem(self._overlay_item)
            handle.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
            handle.setPen(QPen(QColor(255, 208, 48, 245), 2))
            handle.setBrush(QBrush(QColor(255, 208, 48, 220)))
            handle.setVisible(False)
            self._overlay_handle_items[key] = handle
        self._update_overlay_selection_visuals()
        self._rebuild_tile_items()
        self._apply_overlay_transform()
        self._redraw_landmark_items()

    def _apply_overlay_transform(self) -> None:
        if self._overlay_item is None or self._overlay_rgb is None:
            return
        fh, fw = self._fixed_shape_hw
        oh, ow = self._overlay_rgb.shape[:2]
        x = (fw / 2.0) - (ow / 2.0) + self._tx_px
        y = (fh / 2.0) - (oh / 2.0) + self._ty_px
        self._overlay_item.setPos(float(x), float(y))
        self._overlay_item.setRotation(self._angle_deg)
        self._overlay_item.setScale(self._scale)
        self._update_overlay_selection_visuals()

    def _set_overlay_selected(self, selected: bool) -> None:
        self._overlay_selected = bool(selected)
        self._update_overlay_selection_visuals()

    def _update_overlay_selection_visuals(self) -> None:
        if self._overlay_selection_item is not None:
            self._overlay_selection_item.setVisible(self._overlay_selected)
        if self._overlay_item is None:
            self._overlay_handle_items.clear()
            return
        rect = self._overlay_item.boundingRect()
        handle_half_local = max(4.0, 8.0 / max(float(self._scale), 0.1))
        handle_size_local = handle_half_local * 2.0
        corners = {
            "lt": (rect.left(), rect.top()),
            "rt": (rect.right(), rect.top()),
            "lb": (rect.left(), rect.bottom()),
            "rb": (rect.right(), rect.bottom()),
        }
        handles_visible = bool(self._overlay_selected and not self._transform_interaction_locked())
        for key, handle in self._overlay_handle_items.items():
            cx, cy = corners[key]
            handle.setRect(
                float(cx - handle_half_local),
                float(cy - handle_half_local),
                float(handle_size_local),
                float(handle_size_local),
            )
            handle.setVisible(handles_visible)

    def _overlay_scale_mode_at_scene_pos(self, scene_pos: QPointF) -> str | None:
        if self._overlay_item is None or not self._overlay_selected or self._transform_interaction_locked():
            return None
        local = self._overlay_item.mapFromScene(scene_pos)
        rect = self._overlay_item.boundingRect()
        handle_margin_local = max(8.0, 18.0 / max(float(self._scale), 0.1))
        near_left = abs(local.x() - rect.left()) <= handle_margin_local
        near_right = abs(local.x() - rect.right()) <= handle_margin_local
        near_top = abs(local.y() - rect.top()) <= handle_margin_local
        near_bottom = abs(local.y() - rect.bottom()) <= handle_margin_local
        if near_left and near_top:
            return "scale_lt"
        if near_right and near_top:
            return "scale_rt"
        if near_left and near_bottom:
            return "scale_lb"
        if near_right and near_bottom:
            return "scale_rb"
        return None

    def _cursor_shape_for_overlay_mode(self, mode: str | None):
        cursor_map = {
            "translate": Qt.CursorShape.SizeAllCursor,
            "scale_lt": Qt.CursorShape.SizeFDiagCursor,
            "scale_rb": Qt.CursorShape.SizeFDiagCursor,
            "scale_rt": Qt.CursorShape.SizeBDiagCursor,
            "scale_lb": Qt.CursorShape.SizeBDiagCursor,
        }
        return cursor_map.get(mode, Qt.CursorShape.ArrowCursor)

    def _update_overlay_hover_cursor(self, scene_pos: QPointF | None = None) -> None:
        if self._transform_interaction_locked() and self._diagnostic_active:
            self.viewport().setCursor(self._anchor_cursor)
            return
        if scene_pos is None or self._overlay_item is None:
            self.viewport().setCursor(self._default_cursor)
            return
        if self._drag_mode in {"translate", "rotate", "scale_lt", "scale_rt", "scale_lb", "scale_rb"}:
            mode = self._drag_mode if self._drag_mode != "rotate" else "translate"
            self.viewport().setCursor(QCursor(self._cursor_shape_for_overlay_mode(mode)))
            return
        if self._scene_pos_hits_overlay(scene_pos) and not self._transform_interaction_locked():
            scale_mode = self._overlay_scale_mode_at_scene_pos(scene_pos)
            if scale_mode is not None:
                self.viewport().setCursor(QCursor(self._cursor_shape_for_overlay_mode(scale_mode)))
                return
            self.viewport().setCursor(QCursor(Qt.CursorShape.SizeAllCursor))
            return
        self.viewport().setCursor(self._default_cursor)

    def _scale_pivot_for_mode(self, mode: str) -> Optional[tuple[float, float, float, float, float, float]]:
        if self._overlay_item is None:
            return None
        rect = self._overlay_item.boundingRect()
        pivot_local_map = {
            "scale_lt": (rect.right(), rect.bottom(), rect.left(), rect.top()),
            "scale_rt": (rect.left(), rect.bottom(), rect.right(), rect.top()),
            "scale_lb": (rect.right(), rect.top(), rect.left(), rect.bottom()),
            "scale_rb": (rect.left(), rect.top(), rect.right(), rect.bottom()),
        }
        vals = pivot_local_map.get(mode)
        if vals is None:
            return None
        pivot_x, pivot_y, corner_x, corner_y = [float(v) for v in vals]
        pivot_scene = self._overlay_item.mapToScene(QPointF(pivot_x, pivot_y))
        return (
            pivot_x,
            pivot_y,
            float(pivot_scene.x()),
            float(pivot_scene.y()),
            float(corner_x),
            float(corner_y),
        )

    def _remove_tile_items(self) -> None:
        if self._overlay_item is None:
            self._overlay_tile_items.clear()
            return
        for rect, text in self._overlay_tile_items.values():
            try:
                if rect.scene() is not None:
                    self._scene.removeItem(rect)
            except Exception:
                pass
            try:
                if text.scene() is not None:
                    self._scene.removeItem(text)
            except Exception:
                pass
        self._overlay_tile_items.clear()

    def _rebuild_tile_items(self) -> None:
        self._remove_tile_items()
        if self._overlay_item is None or not self._overlay_tile_defs:
            return
        for tile in self._overlay_tile_defs:
            bbox = tile.get("display_bbox_xyxy")
            if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                continue
            x0, y0, x1, y1 = [float(v) for v in bbox]
            rect = QGraphicsRectItem(x0, y0, max(1.0, x1 - x0), max(1.0, y1 - y0), self._overlay_item)
            rect.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
            text = QGraphicsSimpleTextItem(str(tile.get("label") or ""), self._overlay_item)
            text.setBrush(QBrush(QColor(44, 128, 255)))
            text.setPos(x0 + 6.0, y0 + 4.0)
            self._overlay_tile_items[int(tile.get("tile_index", -1))] = (rect, text)
        self._refresh_tile_item_styles()

    def _refresh_tile_item_styles(self) -> None:
        for tile in self._overlay_tile_defs:
            idx = int(tile.get("tile_index", -1))
            items = self._overlay_tile_items.get(idx)
            if items is None:
                continue
            rect, text = items
            is_frozen = idx in self._frozen_tile_indices
            is_accepted = idx in self._accepted_tile_indices
            is_hold = idx in self._hold_tile_indices
            is_selected = idx in self._selected_tile_indices
            is_primary_selected = self._selected_tile_index is not None and idx == int(self._selected_tile_index)
            if is_frozen:
                rect.setPen(QPen(QColor(44, 128, 255, 245), 3 if is_primary_selected else (2 if is_selected else 2)))
                rect.setBrush(QBrush(QColor(44, 128, 255, 30 if is_selected else 14)))
                text.setVisible(True)
                text.setOpacity(0.98)
            elif is_primary_selected:
                rect.setPen(QPen(QColor(255, 208, 48, 245), 3))
                rect.setBrush(QBrush(QColor(255, 208, 48, 30)))
                text.setVisible(True)
                text.setOpacity(0.98)
            elif is_selected:
                rect.setPen(QPen(QColor(255, 208, 48, 200), 2))
                rect.setBrush(QBrush(QColor(255, 208, 48, 18)))
                text.setVisible(True)
                text.setOpacity(0.95)
            elif is_accepted:
                rect.setPen(QPen(QColor(56, 174, 214, 235), 2))
                rect.setBrush(QBrush(QColor(56, 174, 214, 18)))
                text.setVisible(True)
                text.setOpacity(0.94)
            elif is_hold:
                rect.setPen(QPen(QColor(224, 134, 52, 235), 2))
                rect.setBrush(QBrush(QColor(224, 134, 52, 18)))
                text.setVisible(True)
                text.setOpacity(0.92)
            elif idx in self._frontier_tile_indices:
                rect.setPen(QPen(QColor(72, 176, 104, 235), 2))
                rect.setBrush(QBrush(QColor(72, 176, 104, 18)))
                text.setVisible(True)
                text.setOpacity(0.90)
            else:
                rect.setPen(QPen(QColor(255, 255, 255, 105), 1))
                rect.setBrush(QBrush(Qt.BrushStyle.NoBrush))
                text.setVisible(False)

    def _emit_tile_selection(self) -> None:
        payload = {
            "selected_tile_index": None if self._selected_tile_index is None else int(self._selected_tile_index),
            "selected_tile_label": self._tile_label_for_index(self._selected_tile_index),
            "selected_tile_indices": sorted(int(v) for v in self._selected_tile_indices),
            "selected_tile_labels": [str(self._tile_label_for_index(idx) or f"T{int(idx):02d}") for idx in sorted(self._selected_tile_indices)],
            "selected_tile_frozen": bool(self._selected_tile_index in self._frozen_tile_indices) if self._selected_tile_index is not None else False,
            "selected_tile_accepted": bool(self._selected_tile_index in self._accepted_tile_indices) if self._selected_tile_index is not None else False,
            "selected_tile_hold": bool(self._selected_tile_index in self._hold_tile_indices) if self._selected_tile_index is not None else False,
            "selected_tile_frontier": bool(self._selected_tile_index in self._frontier_tile_indices) if self._selected_tile_index is not None else False,
            "frozen_tile_indices": sorted(int(v) for v in self._frozen_tile_indices),
            "accepted_tile_indices": sorted(int(v) for v in self._accepted_tile_indices),
            "hold_tile_indices": sorted(int(v) for v in self._hold_tile_indices),
            "frontier_tile_indices": sorted(int(v) for v in self._frontier_tile_indices),
        }
        self.tileSelectionChanged.emit(payload)

    def _tile_index_at_scene_pos(self, scene_pos: QPointF) -> int | None:
        if self._overlay_item is None:
            return None
        local_pos = self._overlay_item.mapFromScene(scene_pos)
        for tile in self._overlay_tile_defs:
            bbox = tile.get("display_bbox_xyxy")
            if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                continue
            x0, y0, x1, y1 = [float(v) for v in bbox]
            if x0 <= float(local_pos.x()) <= x1 and y0 <= float(local_pos.y()) <= y1:
                return int(tile.get("tile_index", -1))
        return None

    def _scene_pos_hits_overlay(self, scene_pos: QPointF) -> bool:
        if self._overlay_item is None:
            return False
        item_pos = self._overlay_item.mapFromScene(scene_pos)
        rect = self._overlay_item.boundingRect()
        return bool(rect.contains(item_pos))

    def _scene_pos_hits_overlay_signal(self, scene_pos: QPointF) -> bool:
        if self._overlay_item is None:
            return False
        item_pos = self._overlay_item.mapFromScene(scene_pos)
        rect = self._overlay_item.boundingRect()
        if not rect.contains(item_pos):
            return False
        if self._overlay_alpha_source_u8 is None:
            return True
        raw_x, raw_y = self._raw_local_from_display(float(item_pos.x()), float(item_pos.y()))
        x_i = int(np.clip(round(raw_x), 0, max(0, self._overlay_alpha_source_u8.shape[1] - 1)))
        y_i = int(np.clip(round(raw_y), 0, max(0, self._overlay_alpha_source_u8.shape[0] - 1)))
        return bool(self._overlay_alpha_source_u8[y_i, x_i] > 0)

    def _render_overlay_rgba(self) -> None:
        if self._overlay_plane_u8 is None:
            self._overlay_rgb = None
            return
        plane = np.asarray(self._overlay_plane_u8, dtype=np.uint8)
        alpha_plane = (
            plane
            if self._overlay_alpha_source_u8 is None
            else np.asarray(self._overlay_alpha_source_u8, dtype=np.uint8)
        )
        if self._overlay_flip_lr:
            plane = np.fliplr(plane)
            alpha_plane = np.fliplr(alpha_plane)
        if self._overlay_flip_ud:
            plane = np.flipud(plane)
            alpha_plane = np.flipud(alpha_plane)
        plane_f = plane.astype(np.float32)
        valid = alpha_plane > 0
        if np.any(valid):
            vals = plane_f[valid]
            lo = float(np.percentile(vals, 1.0))
            hi = float(np.percentile(vals, 99.0))
            if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
                lo = float(vals.min())
                hi = float(vals.max())
            if hi > lo:
                plane_vis = np.clip((plane_f - lo) / float(hi - lo), 0.0, 1.0)
            else:
                plane_vis = np.zeros_like(plane_f, dtype=np.float32)
        else:
            plane_vis = np.zeros_like(plane_f, dtype=np.float32)
        base_alpha = np.where(
            valid,
            np.maximum(120.0, 255.0 * np.power(np.clip(plane_vis, 0.0, 1.0), 0.55)),
            0.0,
        )
        alpha = np.clip(base_alpha * self._overlay_opacity, 0, 255).astype(np.uint8)
        plane_vis_u8 = np.clip(np.round(plane_vis * 255.0), 0, 255).astype(np.uint8)
        self._overlay_rgb = np.stack(
            [
                np.clip(110.0 + plane_vis_u8.astype(np.float32) * 0.75, 0, 255).astype(np.uint8),
                np.clip(plane_vis_u8.astype(np.float32) * 0.18, 0, 255).astype(np.uint8),
                np.clip(plane_vis_u8.astype(np.float32) * 0.14, 0, 255).astype(np.uint8),
                alpha,
            ],
            axis=-1,
        )

    def _update_interaction_cursor(self) -> None:
        if self._transform_interaction_locked() and self._diagnostic_active:
            self.viewport().setCursor(self._anchor_cursor)
        else:
            self.viewport().setCursor(self._default_cursor)

    def _display_local_from_raw(self, raw_x: float, raw_y: float) -> tuple[float, float]:
        if self._overlay_rgb is None:
            return float(raw_x), float(raw_y)
        h, w = self._overlay_rgb.shape[:2]
        x = float(w - 1 - raw_x) if self._overlay_flip_lr else float(raw_x)
        y = float(h - 1 - raw_y) if self._overlay_flip_ud else float(raw_y)
        return x, y

    def _raw_local_from_display(self, display_x: float, display_y: float) -> tuple[float, float]:
        if self._overlay_rgb is None:
            return float(display_x), float(display_y)
        h, w = self._overlay_rgb.shape[:2]
        x = float(w - 1 - display_x) if self._overlay_flip_lr else float(display_x)
        y = float(h - 1 - display_y) if self._overlay_flip_ud else float(display_y)
        x = float(np.clip(x, 0.0, max(0.0, w - 1.0)))
        y = float(np.clip(y, 0.0, max(0.0, h - 1.0)))
        return x, y

    def _remove_landmark_items(self) -> None:
        for item_map in (self._a_marker_items, self._b_marker_items):
            for ellipse, text in item_map.values():
                try:
                    if ellipse.scene() is not None:
                        self._scene.removeItem(ellipse)
                except Exception:
                    pass
                try:
                    if text.scene() is not None:
                        self._scene.removeItem(text)
                except Exception:
                    pass
            item_map.clear()

    def _add_marker(
        self,
        *,
        parent,
        center_x: float,
        center_y: float,
        fill: QColor,
        pen: QColor,
        label: str,
    ) -> tuple[QGraphicsEllipseItem, QGraphicsSimpleTextItem]:
        radius = 14.0
        ellipse = QGraphicsEllipseItem(center_x - radius, center_y - radius, radius * 2.0, radius * 2.0, parent)
        ellipse.setPen(QPen(pen, 2))
        ellipse.setBrush(QBrush(fill))
        ellipse.setOpacity(0.52)
        text = QGraphicsSimpleTextItem(label, parent)
        text.setBrush(QBrush(QColor(255, 255, 255)))
        text.setOpacity(0.88)
        br = text.boundingRect()
        text.setPos(center_x - br.width() / 2.0, center_y - br.height() / 2.0 - 1.0)
        return ellipse, text

    def _redraw_landmark_items(self) -> None:
        self._remove_landmark_items()
        for idx, scene_xy in sorted(self._points_a_scene.items()):
            ellipse, text = self._add_marker(
                parent=None,
                center_x=float(scene_xy[0]),
                center_y=float(scene_xy[1]),
                fill=QColor(42, 140, 80, 175),
                pen=QColor(255, 255, 255),
                label=f"A{idx}",
            )
            self._scene.addItem(ellipse)
            self._scene.addItem(text)
            self._a_marker_items[idx] = (ellipse, text)
        if self._overlay_item is None:
            return
        for idx, raw_xy in sorted(self._points_b_raw.items()):
            dx, dy = self._display_local_from_raw(float(raw_xy[0]), float(raw_xy[1]))
            ellipse, text = self._add_marker(
                parent=self._overlay_item,
                center_x=dx,
                center_y=dy,
                fill=QColor(188, 46, 46, 175),
                pen=QColor(255, 255, 255),
                label=f"B{idx}",
            )
            self._b_marker_items[idx] = (ellipse, text)

    def _emit_diagnostic_state_change(self, event_name: str) -> None:
        payload = dict(self.diagnostic_snapshot())
        payload["event"] = str(event_name)
        self.diagnosticStateChanged.emit(payload)

    def _advance_diagnostic_target(self) -> None:
        if self._diagnostic_next_group == "A":
            self._diagnostic_next_group = "B"
        else:
            self._diagnostic_next_group = "A"
            if self._diagnostic_next_index < 9:
                self._diagnostic_next_index += 1
        self._emit_diagnostic_state_change("target_advanced")

    def _push_history(self, group: str, idx: int, prev_value: Optional[tuple[float, float]]) -> None:
        self._diagnostic_history.append((str(group), int(idx), None if prev_value is None else (float(prev_value[0]), float(prev_value[1]))))

    def _undo_last_point(self) -> bool:
        if not self._diagnostic_history:
            return False
        group, idx, prev_value = self._diagnostic_history.pop()
        target = self._points_a_scene if group == "A" else self._points_b_raw
        if prev_value is None:
            target.pop(idx, None)
        else:
            target[idx] = prev_value
        self._diagnostic_active = True
        self._diagnostic_next_group = group
        self._diagnostic_next_index = idx
        self._redraw_landmark_items()
        self._emit_diagnostic_state_change("point_undone")
        return True

    def _record_a_point(self, scene_pos: QPointF) -> None:
        idx = int(self._diagnostic_next_index)
        prev_value = self._points_a_scene.get(idx)
        self._push_history("A", idx, prev_value)
        self._points_a_scene[idx] = (float(scene_pos.x()), float(scene_pos.y()))
        self._redraw_landmark_items()
        payload = {
            "group": "A",
            "index": idx,
            "scene_xy": [float(scene_pos.x()), float(scene_pos.y())],
        }
        self.diagnosticPointPlaced.emit(payload)
        self._advance_diagnostic_target()

    def _record_b_point(self, scene_pos: QPointF, *, inside_signal: bool) -> None:
        if self._overlay_item is None:
            return
        idx = int(self._diagnostic_next_index)
        display_pos = self._overlay_item.mapFromScene(scene_pos)
        raw_x, raw_y = self._raw_local_from_display(float(display_pos.x()), float(display_pos.y()))
        prev_value = self._points_b_raw.get(idx)
        self._push_history("B", idx, prev_value)
        self._points_b_raw[idx] = (raw_x, raw_y)
        self._redraw_landmark_items()
        payload = {
            "group": "B",
            "index": idx,
            "scene_xy": [float(scene_pos.x()), float(scene_pos.y())],
            "overlay_display_xy": [float(display_pos.x()), float(display_pos.y())],
            "overlay_raw_xy": [float(raw_x), float(raw_y)],
            "signal_class": "inside_signal" if inside_signal else "low_signal_background",
        }
        self.diagnosticPointPlaced.emit(payload)
        self._advance_diagnostic_target()

    def _current_anchor_pivot(self) -> Optional[tuple[float, float, float, float]]:
        pair_ids = sorted(set(self._points_a_scene.keys()) & set(self._points_b_raw.keys()))
        if pair_ids:
            idx = int(pair_ids[0])
            scene_xy = self._points_a_scene.get(idx)
            raw_xy = self._points_b_raw.get(idx)
            if scene_xy is not None and raw_xy is not None:
                display_x, display_y = self._display_local_from_raw(float(raw_xy[0]), float(raw_xy[1]))
                return float(display_x), float(display_y), float(scene_xy[0]), float(scene_xy[1])
        if self._overlay_item is None or self._overlay_rgb is None:
            return None
        h, w = self._overlay_rgb.shape[:2]
        center_x = float(w) / 2.0
        center_y = float(h) / 2.0
        scene_pt = self._overlay_item.mapToScene(QPointF(center_x, center_y))
        return center_x, center_y, float(scene_pt.x()), float(scene_pt.y())

    def _set_scale_preserve_anchor(
        self,
        new_scale: float,
        *,
        pivot: Optional[tuple[float, float, float, float]] = None,
    ) -> bool:
        if self._overlay_item is None or self._overlay_rgb is None:
            return False
        pivot = pivot or self._current_anchor_pivot()
        if pivot is None:
            return False
        display_x, display_y, anchor_scene_x, anchor_scene_y = pivot
        new_scale = float(np.clip(new_scale, 0.05, 5.0))
        h, w = self._overlay_rgb.shape[:2]
        center_x = float(w) / 2.0
        center_y = float(h) / 2.0
        theta = np.deg2rad(float(self._angle_deg))
        cos_t = float(np.cos(theta))
        sin_t = float(np.sin(theta))
        vec_x = float(display_x - center_x)
        vec_y = float(display_y - center_y)
        rotated_x = new_scale * (cos_t * vec_x - sin_t * vec_y)
        rotated_y = new_scale * (sin_t * vec_x + cos_t * vec_y)
        overlay_x = float(anchor_scene_x - center_x - rotated_x)
        overlay_y = float(anchor_scene_y - center_y - rotated_y)
        fh, fw = self._fixed_shape_hw
        self._scale = new_scale
        self._tx_px = overlay_x - ((float(fw) / 2.0) - (float(w) / 2.0))
        self._ty_px = overlay_y - ((float(fh) / 2.0) - (float(h) / 2.0))
        self._apply_overlay_transform()
        self.transformEdited.emit(self._tx_px, self._ty_px, self._angle_deg, self._scale)
        return True

    def _place_a_at_current_cursor(self) -> bool:
        scene_pos = self._current_cursor_scene_pos()
        if scene_pos is None:
            return False
        self._record_a_point(scene_pos)
        return True

    def _place_b_at_current_cursor(self) -> bool:
        scene_pos = self._current_cursor_scene_pos()
        if scene_pos is None or not self._scene_pos_hits_overlay(scene_pos):
            return False
        self._record_b_point(scene_pos, inside_signal=self._scene_pos_hits_overlay_signal(scene_pos))
        return True

    def _current_cursor_scene_pos(self) -> Optional[QPointF]:
        viewport_pos = self.mapFromGlobal(QCursor.pos())
        if self.viewport().rect().contains(viewport_pos):
            self._last_scene_pos = self.mapToScene(viewport_pos)
            return QPointF(self._last_scene_pos)
        if self._last_scene_pos.isNull():
            return None
        return QPointF(self._last_scene_pos)

    def _handle_diagnostic_click(self, scene_pos: QPointF, overlay_hit: bool) -> bool:
        if not self._transform_interaction_locked() or not self._diagnostic_active:
            return False
        if self._diagnostic_next_group == "A":
            self._record_a_point(scene_pos)
            return True
        if self._diagnostic_next_group == "B" and overlay_hit:
            self._record_b_point(scene_pos, inside_signal=self._scene_pos_hits_overlay_signal(scene_pos))
            return True
        return False

    def mousePressEvent(self, event) -> None:
        self.setFocus()
        if self._overlay_item is None:
            super().mousePressEvent(event)
            return
        scene_pos = self.mapToScene(event.position().toPoint())
        self._last_scene_pos = QPointF(scene_pos)
        overlay_hit = self._scene_pos_hits_overlay(scene_pos)
        tile_hit = self._tile_index_at_scene_pos(scene_pos) if overlay_hit else None
        if event.button() == Qt.MouseButton.LeftButton:
            if self._transform_interaction_locked():
                if overlay_hit:
                    if tile_hit is not None and bool(event.modifiers() & Qt.KeyboardModifier.ShiftModifier):
                        updated = set(self._selected_tile_indices)
                        if int(tile_hit) in updated:
                            updated.discard(int(tile_hit))
                            next_primary = self._selected_tile_index if self._selected_tile_index in updated else (sorted(updated)[-1] if updated else None)
                        else:
                            updated.add(int(tile_hit))
                            next_primary = int(tile_hit)
                        self.set_selected_tiles(updated, primary_tile_index=next_primary)
                    else:
                        self.set_selected_tile(tile_hit)
                    self._set_overlay_selected(True)
                    self._drag_mode = None
                    event.accept()
                    return
                self.set_selected_tile(None)
                self._set_overlay_selected(False)
                self._drag_mode = "pan"
                self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
                super().mousePressEvent(event)
                return
            if overlay_hit:
                if tile_hit is not None and bool(event.modifiers() & Qt.KeyboardModifier.ShiftModifier):
                    updated = set(self._selected_tile_indices)
                    if int(tile_hit) in updated:
                        updated.discard(int(tile_hit))
                        next_primary = self._selected_tile_index if self._selected_tile_index in updated else (sorted(updated)[-1] if updated else None)
                    else:
                        updated.add(int(tile_hit))
                        next_primary = int(tile_hit)
                    self.set_selected_tiles(updated, primary_tile_index=next_primary)
                    self._set_overlay_selected(True)
                    self._drag_mode = None
                    event.accept()
                    return
                self.set_selected_tile(tile_hit)
                self._set_overlay_selected(True)
                scale_mode = self._overlay_scale_mode_at_scene_pos(scene_pos)
                self._drag_mode = scale_mode or "translate"
            else:
                self.set_selected_tile(None)
                self._set_overlay_selected(False)
                self._drag_mode = "pan"
                self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
                super().mousePressEvent(event)
                return
        elif event.button() == Qt.MouseButton.RightButton:
            if self._transform_interaction_locked() and self._diagnostic_active:
                if self._undo_last_point():
                    event.accept()
                    return
            if overlay_hit and self._overlay_selected and not self._transform_interaction_locked():
                self._drag_mode = "rotate"
            else:
                self._drag_mode = None
                super().mousePressEvent(event)
                return
        else:
            self._drag_mode = None
            super().mousePressEvent(event)
            return
        self._press_scene_pos = scene_pos
        self._press_tx_px = self._tx_px
        self._press_ty_px = self._ty_px
        self._press_angle_deg = self._angle_deg
        self._press_scale = self._scale
        self._scale_pivot_local = QPointF()
        self._scale_pivot_scene = QPointF()
        self._scale_initial_distance = 1.0
        if self._drag_mode in {"scale_lt", "scale_rt", "scale_lb", "scale_rb"}:
            pivot_info = self._scale_pivot_for_mode(self._drag_mode)
            if pivot_info is not None:
                (
                    pivot_local_x,
                    pivot_local_y,
                    pivot_scene_x,
                    pivot_scene_y,
                    corner_local_x,
                    corner_local_y,
                ) = pivot_info
                self._scale_pivot_local = QPointF(pivot_local_x, pivot_local_y)
                self._scale_pivot_scene = QPointF(pivot_scene_x, pivot_scene_y)
                corner_scene = self._overlay_item.mapToScene(QPointF(corner_local_x, corner_local_y))
                self._scale_initial_distance = max(
                    1.0,
                    float(
                        np.hypot(
                            float(corner_scene.x()) - float(pivot_scene_x),
                            float(corner_scene.y()) - float(pivot_scene_y),
                        )
                    ),
                )
            else:
                self._drag_mode = "translate"
        self._update_overlay_hover_cursor(scene_pos)
        event.accept()

    def mouseMoveEvent(self, event) -> None:
        self._last_scene_pos = self.mapToScene(event.position().toPoint())
        if self._drag_mode == "pan":
            super().mouseMoveEvent(event)
            return
        if self._overlay_item is None or self._drag_mode is None:
            self._update_overlay_hover_cursor(self._last_scene_pos)
            super().mouseMoveEvent(event)
            return
        scene_pos = self.mapToScene(event.position().toPoint())
        delta = scene_pos - self._press_scene_pos
        if self._drag_mode == "translate":
            self._tx_px = self._press_tx_px + float(delta.x())
            self._ty_px = self._press_ty_px + float(delta.y())
        elif self._drag_mode == "rotate":
            self._angle_deg = self._press_angle_deg + float(delta.x() + delta.y()) * 0.20
        elif self._drag_mode in {"scale_lt", "scale_rt", "scale_lb", "scale_rb"}:
            current_distance = max(
                1.0,
                float(
                    np.hypot(
                        float(scene_pos.x()) - float(self._scale_pivot_scene.x()),
                        float(scene_pos.y()) - float(self._scale_pivot_scene.y()),
                    )
                ),
            )
            new_scale = self._press_scale * (current_distance / max(self._scale_initial_distance, 1.0))
            if self._set_scale_preserve_anchor(
                new_scale,
                pivot=(
                    float(self._scale_pivot_local.x()),
                    float(self._scale_pivot_local.y()),
                    float(self._scale_pivot_scene.x()),
                    float(self._scale_pivot_scene.y()),
                ),
            ):
                self._update_overlay_hover_cursor(scene_pos)
                event.accept()
                return
        self._apply_overlay_transform()
        self.transformEdited.emit(self._tx_px, self._ty_px, self._angle_deg, self._scale)
        self._update_overlay_hover_cursor(scene_pos)
        event.accept()

    def mouseReleaseEvent(self, event) -> None:
        if self._drag_mode is not None:
            if self._drag_mode == "pan":
                super().mouseReleaseEvent(event)
                self.setDragMode(QGraphicsView.DragMode.NoDrag)
            self._drag_mode = None
            self._update_overlay_hover_cursor(self._last_scene_pos)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event) -> None:
        factor = 1.15 if event.angleDelta().y() > 0 else 1.0 / 1.15
        self.scale(factor, factor)
        event.accept()

    def keyPressEvent(self, event) -> None:
        handled = True
        key = event.key()
        if key == Qt.Key.Key_A:
            self._diagnostic_active = True
            self._diagnostic_next_group = "A"
            placed = False
            if self._transform_interaction_locked():
                placed = self._place_a_at_current_cursor()
            if not placed:
                self._emit_diagnostic_state_change("target_manual")
            self._update_interaction_cursor()
        elif key == Qt.Key.Key_B:
            self._diagnostic_active = True
            self._diagnostic_next_group = "B"
            placed = False
            if self._transform_interaction_locked():
                placed = self._place_b_at_current_cursor()
            if not placed:
                self._emit_diagnostic_state_change("target_manual")
            self._update_interaction_cursor()
        elif Qt.Key.Key_1 <= key <= Qt.Key.Key_9:
            self._diagnostic_next_index = int(key - Qt.Key.Key_0)
            self._emit_diagnostic_state_change("target_manual")
        elif key == Qt.Key.Key_F:
            if not self._overlay_frozen_lock:
                self._overlay_transform_locked = not self._overlay_transform_locked
                self._set_overlay_selected(False if self._overlay_transform_locked else self._overlay_selected)
                self._update_interaction_cursor()
                self._emit_diagnostic_state_change("lock_toggled")
        elif key == Qt.Key.Key_BracketLeft:
            self._overlay_opacity = max(0.10, self._overlay_opacity - 0.10)
            self._render_overlay_rgba()
            if self._fixed_item is not None:
                self._install_overlay_item()
            self._emit_diagnostic_state_change("opacity_changed")
        elif key == Qt.Key.Key_BracketRight:
            self._overlay_opacity = min(1.35, self._overlay_opacity + 0.10)
            self._render_overlay_rgba()
            if self._fixed_item is not None:
                self._install_overlay_item()
            self._emit_diagnostic_state_change("opacity_changed")
        elif key in {Qt.Key.Key_Delete, Qt.Key.Key_Backspace}:
            self._undo_last_point()
        else:
            handled = False
        if handled:
            event.accept()
            return
        super().keyPressEvent(event)

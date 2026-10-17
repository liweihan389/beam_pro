from __future__ import annotations

import math

import numpy as np
from PySide6.QtCore import QRectF, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPen, QPixmap
from PySide6.QtWidgets import QGraphicsItem, QGraphicsScene, QGraphicsView


class BeamView(QGraphicsView):
    roi_selected = Signal(object)
    frame_painted = Signal(int)

    def __init__(self):
        super().__init__()
        self.setScene(QGraphicsScene(self))
        self.image_item = self.scene().addPixmap(QPixmap())
        self.image_item.setZValue(-10)
        self.setBackgroundBrush(QColor("#080f1c"))
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setRenderHint(__import__("PySide6.QtGui", fromlist=["QPainter"]).QPainter.RenderHint.Antialiasing)
        self.setMinimumSize(480, 360)
        self.items = []
        self.selection = None
        self.selecting = False
        self.start_point = None
        self.rectangle = None
        self.dimensions = None
        self.displayed_frame_id = -1
        self.last_painted_id = -1
        self.auto_circle = True
        self.fixed_fit = None
        self.current_fit = None
        self.plugin_shapes = []
        self.track_center = True
        self.connect_centers = False

    def set_image(self, frame, fit=None):
        fit = fit if fit is not None else self.current_fit
        if fit and fit.amplitude > 0:
            yy, xx = np.ogrid[:frame.shape[0], :frame.shape[1]]
            background = (fit.background_origin if fit.background_origin is not None else fit.background)
            background = background + fit.background_slope_x * xx + fit.background_slope_y * yy
            normalized = np.clip((frame.pixels - background) / fit.amplitude, 0, 1)
            threshold = fit.boundary_fraction
        else:
            normalized = np.clip(frame.pixels / frame.full_scale, 0, 1)
            threshold = .135
        # Blue begins at the selected relative intensity, independent of exposure.
        levels = [0., threshold * .3, threshold * .8, threshold,
                  threshold + .18 * (1 - threshold), threshold + .45 * (1 - threshold),
                  threshold + .75 * (1 - threshold), 1.]
        palette = np.array([[8, 10, 32], [55, 15, 95], [170, 30, 145], [25, 65, 215],
                            [20, 190, 220], [90, 210, 70], [255, 195, 25], [255, 255, 255]])
        rgb = np.stack([np.interp(normalized, levels, palette[:, channel]) for channel in range(3)], axis=-1)
        rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
        h, w = frame.shape
        image = QImage(rgb.data, w, h, rgb.strides[0], QImage.Format.Format_RGB888).copy()
        self.image_item.setPixmap(QPixmap.fromImage(image))
        self.displayed_frame_id = frame.frame_id
        if self.dimensions != (w, h):
            self.dimensions = (w, h)
            self.setSceneRect(QRectF(0, 0, w, h))
            self.fitInView(self.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def overlays(self, fit=None, shapes=None):
        self.current_fit = fit
        self.plugin_shapes = shapes or []
        for item in self.items:
            self.scene().removeItem(item)
        self.items.clear()
        def ellipse(x, y, rx, ry, angle, color, dashed=False):
            pen = QPen(QColor(color), 1.2)
            pen.setCosmetic(True)
            if dashed:
                pen.setStyle(Qt.PenStyle.DashLine)
            item = self.scene().addEllipse(-rx, -ry, 2 * rx, 2 * ry, pen)
            item.setPos(x, y)
            item.setRotation(math.degrees(angle))
            self.items.append(item)
        def cross(x, y, color, role="", diagonal=False):
            pen = QPen(QColor(color), 1.5)
            pen.setCosmetic(True)
            outline = QPen(QColor("#101828"), 3.5)
            outline.setCosmetic(True)
            segments = [(x-4, y-4, x+4, y+4), (x-4, y+4, x+4, y-4)] if diagonal else [
                (x-5, y, x+5, y), (x, y-5, x, y+5)]
            for stroke in (outline, pen):
                for segment in segments:
                    item = self.scene().addLine(*segment, stroke)
                    item.setData(0, role)
                    self.items.append(item)
        live_valid = bool(fit and fit.valid and fit.radius_minor > 0 and np.isfinite(fit.center).all())
        circle_fit = self.fixed_fit or fit
        if self.auto_circle and self.fixed_fit and self.connect_centers and live_valid:
            reference_x, reference_y = self.fixed_fit.boundary_center
            pen = QPen(QColor("#ffcc6e"), 1.4)
            pen.setCosmetic(True)
            item = self.scene().addLine(reference_x, reference_y, fit.center_x, fit.center_y, pen)
            item.setData(0, "center_link")
            self.items.append(item)
        if self.auto_circle and circle_fit and circle_fit.radius_minor > 0:
            x, y = circle_fit.boundary_center
            ellipse(x, y, circle_fit.radius_major, circle_fit.radius_minor, circle_fit.angle_rad,
                    "#d5b4ff" if self.fixed_fit else ("#7ce5cb" if circle_fit.valid else "#f3ba62"),
                    not circle_fit.valid)
            cross(x, y, "#d5b4ff" if self.fixed_fit else ("#7ce5cb" if circle_fit.valid else "#f3ba62"),
                  "circle_center")
            if self.fixed_fit:
                text = self.scene().addText("固定参考圈")
                text.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations)
                text.setDefaultTextColor(QColor("#d5b4ff"))
                text.setPos(x + circle_fit.radius_major + 5, y)
                self.items.append(text)
        if live_valid and (self.fixed_fit is None or self.track_center):
            # With an unfixed circle, coincident centers need only one marker.
            if self.fixed_fit is not None or not self.auto_circle or np.linalg.norm(fit.center - fit.boundary_center) > .1:
                cross(fit.center_x, fit.center_y, "#f4fafc", "live_center", diagonal=True)
        for shape in shapes or []:
            try:
                if shape["kind"] == "horizontal_line":
                    y = float(shape["y"])
                    half = float(shape.get("half_height", 0))
                    if not math.isfinite(y) or not math.isfinite(half) or half < 0 or not self.dimensions:
                        continue
                    width, height = self.dimensions
                    if not 0 <= y < height:
                        continue
                    pen = QPen(QColor("#ffcc6e"), 1.5)
                    pen.setCosmetic(True)
                    item = self.scene().addLine(0, y, width, y, pen)
                    item.setData(0, "height_target")
                    self.items.append(item)
                    brush = QColor("#ffcc6e")
                    brush.setAlpha(35)
                    band = self.scene().addRect(0, max(0,y-half), width, min(height,y+half)-max(0,y-half), QPen(Qt.PenStyle.NoPen), brush)
                    band.setData(0, "height_tolerance")
                    self.items.append(band)
                    label = self.scene().addText(str(shape.get("label", "等高目标")))
                    label.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations)
                    label.setDefaultTextColor(QColor("#ffcc6e"))
                    label.setPos(4,y+3)
                    self.items.append(label)
                    continue
                if shape["kind"] in ("target", "circle", "ellipse"):
                    x, y = float(shape["x"]), float(shape["y"])
                    radius = max(0.3, float(shape.get("radius", 4)))
                    if not all(math.isfinite(v) for v in (x, y, radius)):
                        continue
                    ellipse(x, y, radius, max(0.3, float(shape.get("radius_y", radius))),
                            float(shape.get("angle_rad", 0)), "#ffcc6e")
                    cross(x, y, "#ffcc6e")
                    text = self.scene().addText(str(shape.get("label", "目标")))
                    text.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations)
                    text.setDefaultTextColor(QColor("#ffcc6e"))
                    text.setPos(x + 6, y + 6)
                    self.items.append(text)
            except (KeyError, ValueError, TypeError):
                continue
        if self.selection:
            x0, y0, x1, y1 = self.selection
            pen = QPen(QColor("#8bb9ff"), 1, Qt.PenStyle.DashLine)
            pen.setCosmetic(True)
            self.items.append(self.scene().addRect(x0, y0, x1-x0, y1-y0, pen))

    def set_auto_circle(self, enabled):
        self.auto_circle = bool(enabled)
        if not enabled:
            self.fixed_fit = None
            self.track_center, self.connect_centers = True, False
        self.overlays(self.current_fit, self.plugin_shapes)

    def toggle_fixed_circle(self):
        if self.fixed_fit is not None:
            self.fixed_fit = None
        elif self.auto_circle and self.current_fit and self.current_fit.valid:
            self.fixed_fit = self.current_fit
        self.track_center, self.connect_centers = True, False
        self.overlays(self.current_fit, self.plugin_shapes)

    def clear_fixed_circle(self):
        self.fixed_fit = None
        self.track_center, self.connect_centers = True, False
        self.overlays(self.current_fit, self.plugin_shapes)

    def set_fixed_display(self, track_center, connect_centers):
        self.track_center = bool(track_center)
        self.connect_centers = bool(connect_centers)
        self.overlays(self.current_fit, self.plugin_shapes)

    def enable_selection(self, checked):
        self.selecting = checked
        self.setDragMode(QGraphicsView.DragMode.NoDrag if checked else QGraphicsView.DragMode.ScrollHandDrag)

    def mousePressEvent(self, event):
        if self.selecting and event.button() == Qt.MouseButton.LeftButton:
            self.start_point = self.mapToScene(event.position().toPoint())
            self.rectangle = self.scene().addRect(QRectF(self.start_point, self.start_point), QPen(QColor("#8bb9ff")))
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.start_point is not None:
            self.rectangle.setRect(QRectF(self.start_point, self.mapToScene(event.position().toPoint())).normalized())
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self.start_point is not None:
            rect = self.rectangle.rect().intersected(self.sceneRect())
            self.scene().removeItem(self.rectangle)
            self.rectangle = None
            self.start_point = None
            if rect.width() >= 9 and rect.height() >= 9:
                self.selection = (int(rect.left()), int(rect.top()), int(rect.right()), int(rect.bottom()))
                self.roi_selected.emit(self.selection)
        else:
            super().mouseReleaseEvent(event)

    def wheelEvent(self, event):
        factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        scale = self.transform().m11() * factor
        if 0.05 < scale < 100:
            self.scale(factor, factor)

    def drawForeground(self, painter, rect):
        super().drawForeground(painter, rect)
        if self.displayed_frame_id != self.last_painted_id and self.displayed_frame_id >= 0:
            self.last_painted_id = self.displayed_frame_id
            self.frame_painted.emit(self.displayed_frame_id)


"""Host-owned declarative controls: plugins do not receive Qt objects."""
from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QScrollArea,
    QTableWidget,
    QTableWidgetItem,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)


class CausticPlot(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.payload = {}
        self.setMinimumHeight(170)
        self.setMaximumHeight(220)

    def set_payload(self, payload):
        self.payload = payload or {}
        self.setVisible(bool(self.payload.get("curves")))
        self.update()

    def paintEvent(self, event):
        curves = self.payload.get("curves", [])
        if not curves:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        plot = QRectF(53, 24, max(10, self.width()-67), max(10, self.height()-65))
        xmin = min(min(c["x"]) for c in curves)
        xmax = max(max(c["x"]) for c in curves)
        ymin = 0
        ymax = max(max(c["high"]) for c in curves)*1.08
        for point in self.payload.get("points", []):
            ymax = max(ymax, max(point["diameters"])*1.08)
        def xy(x, y):
            return QPointF(plot.left()+(x-xmin)/max(1e-9, xmax-xmin)*plot.width(),
                           plot.bottom()-(y-ymin)/max(1e-9, ymax-ymin)*plot.height())
        painter.setPen(QPen(QColor('#47607b'), 1))
        painter.drawRect(plot)
        for i in range(4):
            value = ymax*i/3
            position = xy(xmin, value)
            painter.drawText(QRectF(0, position.y()-9, 47, 18), Qt.AlignmentFlag.AlignRight, f'{value:.0f}')
            z = xmin+(xmax-xmin)*i/3
            painter.drawText(QRectF(xy(z, 0).x()-27, plot.bottom()+3, 54, 18), Qt.AlignmentFlag.AlignCenter, f'{z:.0f}')
        painter.drawText(5, 15, '直径 μm')
        painter.drawText(int(plot.center().x()-40), self.height()-5, '相对位置 mm')
        colors = ('#7ce5cb', '#ffbc75')
        for axis, curve in enumerate(curves):
            color = QColor(colors[axis % 2])
            shade = QColor(color)
            shade.setAlpha(35)
            polygon = [xy(x, y) for x, y in zip(curve['x'], curve['low'], strict=False)]
            polygon += [xy(x, y) for x, y in reversed(list(zip(curve['x'], curve['high'], strict=False)))]
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(shade)
            painter.drawPolygon(QPolygonF(polygon))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(QPen(color, 1.7))
            painter.drawPolyline(QPolygonF([xy(x, y) for x, y in zip(curve['x'], curve['y'], strict=False)]))
            painter.drawText(int(plot.left()+55+axis*105), 15, curve['label'])
            for point in self.payload.get('points', []):
                center = xy(point['z'], point['diameters'][axis])
                painter.drawLine(xy(point['z']-point['tolerance'], point['diameters'][axis]),
                                 xy(point['z']+point['tolerance'], point['diameters'][axis]))
                painter.drawEllipse(center, 3, 3)


class PluginPanel(QScrollArea):
    def __init__(self, send_action, parent=None, *, identity="waist", title="束腰测量"):
        super().__init__(parent)
        self.identity = identity
        self.default_title = title
        self.send_action = send_action
        self.setWidgetResizable(True)
        body = QWidget()
        self.setWidget(body)
        layout = QVBoxLayout(body)
        self.title = QLabel(title)
        self.title.setStyleSheet('font-size:17px; font-weight:600')
        self.title.setWordWrap(True)
        layout.addWidget(self.title)
        self.text = QTextBrowser()
        self.text.setMinimumHeight(130)
        self.text.setMaximumHeight(150)
        layout.addWidget(self.text)
        self.form_widget = QWidget()
        self.form = QFormLayout(self.form_widget)
        self.form.setContentsMargins(0, 0, 0, 0)
        self.controls = {}
        self.schema = None
        layout.addWidget(self.form_widget)
        self.buttons_widget = QWidget()
        self.buttons_layout = QGridLayout(self.buttons_widget)
        self.buttons_layout.setContentsMargins(0, 0, 0, 0)
        self.buttons = {}
        layout.addWidget(self.buttons_widget)
        self.table = QTableWidget()
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.verticalHeader().hide()
        self.table.verticalHeader().setDefaultSectionSize(23)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.setMinimumHeight(110)
        self.table.setMaximumHeight(175)
        layout.addWidget(self.table)
        self.plot = CausticPlot()
        layout.addWidget(self.plot)
        self.note = QLabel('')
        self.note.setWordWrap(True)
        layout.addWidget(self.note)
        layout.addStretch()
        self.set_event({'title': title, 'text': '点击下方按钮启用插件。',
                        'actions': [{'id': 'activate', 'label': '启用'+title+'插件'}]})

    def values(self):
        return {key: control.isChecked() if isinstance(control, QCheckBox) else control.value()
                for key, control in self.controls.items()}

    def set_event(self, event):
        self.title.setText(str(event.get('title', self.default_title)))
        self.text.setPlainText(str(event.get('text', '')))
        fields = event.get('fields', [])[:20]
        if self.schema != fields:
            while self.form.rowCount():
                self.form.removeRow(0)
            self.controls = {}
            for field in fields:
                if field.get('type') == 'boolean':
                    widget = QCheckBox()
                    widget.setChecked(bool(field.get('value')))
                else:
                    widget = QDoubleSpinBox()
                    widget.setRange(float(field.get('minimum', -100000)), float(field.get('maximum', 100000)))
                    widget.setDecimals(int(field.get('decimals', 2)))
                    widget.setSuffix(str(field.get('suffix', '')))
                    widget.setValue(float(field.get('value', 0)))
                    widget.setReadOnly(bool(field.get('readonly')))
                widget.setObjectName(self.identity+'_'+field['id'])
                self.controls[field['id']] = widget
                self.form.addRow(str(field.get('label', field['id'])), widget)
            self.schema = fields
        while self.buttons_layout.count():
            widget = self.buttons_layout.takeAt(0).widget()
            if widget:
                widget.deleteLater()
        self.buttons = {}
        for index, action in enumerate(event.get('actions', [])[:12]):
            button = QPushButton(str(action.get('label', action['id'])))
            button.setObjectName(self.identity+'_action_'+action['id'])
            button.clicked.connect(lambda checked=False, key=action['id']: self.send_action(key, self.values()))
            self.buttons[action['id']] = button
            self.buttons_layout.addWidget(button, index//2, index%2)
        table = event.get('table') or {}
        rows = table.get('rows', [])[:200]
        headers = table.get('headers', [])[:10]
        self.table.setVisible(bool(rows))
        self.table.setColumnCount(len(headers))
        self.table.setHorizontalHeaderLabels(headers)
        self.table.setRowCount(len(rows))
        self.table.setFixedHeight(min(180, 29+len(rows)*23))
        for i, row in enumerate(rows):
            for j, value in enumerate(row[:len(headers)]):
                self.table.setItem(i, j, QTableWidgetItem(str(value)))
        self.plot.set_payload(event.get('plot'))
        self.note.setText((event.get('plot') or {}).get('note', ''))

from __future__ import annotations

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import hashlib
import json
import sys
import threading
import time
import zipfile
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from beamic_sdk.alignment import Geometry
from PySide6.QtCore import QObject, QStandardPaths, Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from .bridge_client import BridgeClient
from .exposure import ExposureController
from .fitting import FitSettings, fit_beam
from .measurement import Capture
from .plugin_panel import PluginPanel
from .plugin_runtime import PluginProcess
from .plugin_store import PluginStore
from .preview import PreviewFrame
from .simulator import Simulator
from .stream import AdaptiveRate, Latest, RateMeter, SharedPublisher
from .view import BeamView


def assets_root():
    if os.environ.get("BEAMIC_PRO_ASSETS"):
        return Path(os.environ["BEAMIC_PRO_ASSETS"])
    if getattr(sys, "frozen", False):
        return Path(sys._MEIPASS) / "assets"
    return Path(__file__).resolve().parents[2]


class Signals(QObject):
    fit_ready = Signal(object)
    plugin_event = Signal(str, object)
    bridge_event = Signal(object)
    operation = Signal(object)


class Window(QMainWindow):
    def __init__(self, data_dir=None, demo=False):
        super().__init__()
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        self.setWindowTitle("beamic pro · 光学实验助手")
        self.resize(1360, 860)
        self.assets = assets_root()
        self.development = not getattr(sys, "frozen", False) and not os.environ.get("BEAMIC_PRO_BUNDLED")
        root = Path(data_dir) if data_dir else Path(QStandardPaths.writableLocation(QStandardPaths.StandardLocation.AppLocalDataLocation))
        self.settings_path = root / "spot-settings.json"
        try:
            self.fit_settings = FitSettings(**json.loads(self.settings_path.read_text(encoding="utf-8")))
        except (OSError, ValueError, TypeError):
            self.fit_settings = FitSettings()
        runtime = Path(sys.executable).parent / "runtime" / "python.exe"
        if not self.development and not runtime.exists():
            runtime = Path(sys.executable).parent / "runtime" / "python.exe"
        self.plugin_python = Path(os.environ.get("BEAMIC_PRO_PYTHON", str(Path(sys.executable) if self.development else runtime)))
        if self.plugin_python.name.lower() == "pythonw.exe":
            self.plugin_python = self.plugin_python.with_name("python.exe")
        wheels = sorted((self.assets / "wheels").glob("beamic_pro_sdk-*.whl"))
        self.store = PluginStore(root / "plugins", self.plugin_python, wheels[-1] if wheels else self.assets / "sdk")
        self.source = None
        self.source_generation = 0
        self.selection_generation = 0
        self.frames = Latest()
        self.jobs = Latest()
        self.signals = Signals(self)
        self.signals.fit_ready.connect(self.fit_result)
        self.signals.plugin_event.connect(self.plugin_event)
        self.signals.bridge_event.connect(self.bridge_event)
        self.signals.operation.connect(self.operation)
        self.pool = ThreadPoolExecutor(max_workers=3)
        self.analyzer_stop = threading.Event()
        self.analyzer = threading.Thread(target=self.analyze, daemon=True)
        self.analyzer.start()
        self.exposure = ExposureController()
        self.rate = AdaptiveRate()
        self.received_rate, self.fit_rate, self.display_rate = RateMeter(), RateMeter(), RateMeter()
        self.processed_rate = RateMeter()
        self.latencies = deque(maxlen=200)
        self.last_frame = self.last_fit = None
        self.last_frame_ns = 0
        self.last_fit_ns = 0
        self.coordinate_key = None
        self.capture = None
        self.capture_owner = None
        self.pending_exposure = False
        self.publisher = SharedPublisher()
        self.plugins = {}
        self.interactive = None
        self.plugin_states = {}
        self.overlays = {}
        self.capture_ready = False
        self.capabilities = {}
        self.control_report = None
        self.installing = False
        self.closing = False
        self.build_ui()
        self.reload_plugins()
        self.tick_timer = QTimer(self)
        self.tick_timer.timeout.connect(self.tick)
        self.tick_timer.start(10)
        self.last_render = 0
        self.next_render = 0
        self.status_timer = QTimer(self)
        self.status_timer.timeout.connect(self.update_status)
        self.status_timer.start(500)
        if demo:
            QTimer.singleShot(0, self.connect_demo)

    def build_ui(self):
        container = QWidget()
        self.setCentralWidget(container)
        outer = QVBoxLayout(container)
        header = QHBoxLayout()
        title = QLabel("beamic pro")
        title.setStyleSheet("font-size:26px; font-weight:700; color:#7ce5cb")
        header.addWidget(title)
        self.source_label = QLabel("尚未连接 · 图像仅在内存中使用")
        header.addWidget(self.source_label, 1)
        demo_button = QPushButton("模拟光路")
        demo_button.clicked.connect(self.connect_demo)
        real_button = QPushButton("连接 BeamMic")
        real_button.clicked.connect(self.connect_real)
        header.addWidget(demo_button)
        header.addWidget(real_button)
        restart_button = QPushButton("开始采集／刷新预检")
        restart_button.clicked.connect(self.restart_real_capture)
        header.addWidget(restart_button)
        outer.addLayout(header)
        controls = QHBoxLayout()
        controls.addWidget(QLabel("目标新帧率"))
        self.fps_input = QDoubleSpinBox()
        self.fps_input.setRange(1, 500)
        self.fps_input.setValue(20)
        self.fps_input.setSuffix(" FPS")
        self.fps_input.valueChanged.connect(self.change_rate)
        controls.addWidget(self.fps_input)
        self.mode = QComboBox()
        self.mode.addItems(["测量 · 未饱和", "找光 · 轻微过曝"])
        self.mode.currentIndexChanged.connect(self.change_mode)
        controls.addWidget(self.mode)
        self.auto_exposure_input = QCheckBox("曝光稳定")
        self.auto_exposure_input.setChecked(True)
        self.auto_exposure_input.setToolTip("开启时自动调节曝光；关闭后保持曝光不变，可在 BeamMic 手动设置。已发出的指令会完成，旧帧和无效图像仍不进入测量。")
        self.auto_exposure_input.toggled.connect(self.change_auto_exposure)
        controls.addWidget(self.auto_exposure_input)
        select_button = QPushButton("框选主光斑")
        select_button.setCheckable(True)
        clear_button = QPushButton("清除框选")
        clear_button.clicked.connect(self.clear_selection)
        controls.addWidget(select_button)
        controls.addWidget(clear_button)
        self.exposure_label = QLabel("等待图像")
        controls.addWidget(self.exposure_label, 1)
        outer.addLayout(controls)
        splitter = QSplitter()
        self.view = BeamView()
        self.view.roi_selected.connect(self.select_roi)
        self.view.frame_painted.connect(self.painted)
        select_button.toggled.connect(self.view.enable_selection)
        image_panel = QWidget()
        image_layout = QVBoxLayout(image_panel)
        image_layout.setContentsMargins(0, 0, 0, 0)
        circle_row = QHBoxLayout()
        circle_controls = QVBoxLayout()
        self.auto_circle = QCheckBox("自动定圈")
        self.auto_circle.setChecked(True)
        self.auto_circle.toggled.connect(self.change_auto_circle)
        circle_controls.addWidget(self.auto_circle)
        self.fixed_circle_button = QPushButton("固定圈")
        self.fixed_circle_button.setEnabled(False)
        self.fixed_circle_button.clicked.connect(self.toggle_fixed_circle)
        circle_controls.addWidget(self.fixed_circle_button)
        circle_row.addLayout(circle_controls)
        self.fixed_options = QWidget()
        fixed_options_layout = QVBoxLayout(self.fixed_options)
        fixed_options_layout.setContentsMargins(4, 0, 8, 0)
        self.track_center_input = QCheckBox("追踪新光斑的几何中心")
        self.track_center_input.setChecked(True)
        self.track_center_input.setToolTip("仅显示当前主光斑的中心，不保留轨迹；关闭后定位和测量仍继续。")
        self.connect_centers_input = QCheckBox("连接新光斑中心与光圈中心")
        self.connect_centers_input.setToolTip("连接实时测量中心与固定圈中心；可独立于追踪标记开启。")
        fixed_options_layout.addWidget(self.track_center_input)
        fixed_options_layout.addWidget(self.connect_centers_input)
        self.track_center_input.toggled.connect(self.change_fixed_display)
        self.connect_centers_input.toggled.connect(self.change_fixed_display)
        circle_row.addWidget(self.fixed_options)
        self.fixed_options.hide()
        self.circle_note = QLabel("蓝色外缘对应所选相对强度；白色表示显示峰值，饱和状态以提示为准。\n圈默认标出中心；实际定位和测量持续更新。")
        self.circle_note.setWordWrap(True)
        circle_row.addWidget(self.circle_note, 1)
        image_layout.addLayout(circle_row)
        image_layout.addWidget(self.view, 1)
        self.fit_label = QLabel("强度伪彩色显示；颜色不代表实际波长。滚轮缩放，拖动平移。")
        self.fit_label.setWordWrap(True)
        image_layout.addWidget(self.fit_label)
        self.metrics_label = QLabel("新帧 0 FPS · 有效拟合 0 FPS · 显示 0 FPS")
        image_layout.addWidget(self.metrics_label)
        splitter.addWidget(image_panel)
        self.tabs = QTabWidget()
        self.tabs.setMinimumWidth(360)
        self.tabs.setMaximumWidth(460)
        experiment = QWidget()
        layout = QVBoxLayout(experiment)
        self.alignment_notice = QLabel("DM固定 · 只调蓝光两面反射镜\n光路：M1远镜 → M2近镜 → DM → CCD\n调节顺序：先M2，再M1；红光光路保持固定。")
        self.alignment_notice.setWordWrap(True)
        self.alignment_notice.setStyleSheet("color:#7ce5cb; font-weight:600")
        layout.addWidget(self.alignment_notice)
        geometry_group = QGroupBox("沿光路累计的距离（毫米）")
        form = QFormLayout(geometry_group)
        self.geometry_inputs = {}
        for key, label, value in [("z_near", "DM → 近处测量", 100), ("z_far", "DM → 远处测量", 900),
                                  ("mirror_near", "M2近反射镜 → DM", 120), ("mirror_far", "M1远反射镜 → DM", 400)]:
            spin = QDoubleSpinBox()
            spin.setRange(0, 100000)
            spin.setDecimals(2)
            spin.setValue(value)
            form.addRow(label, spin)
            self.geometry_inputs[key] = spin
        layout.addWidget(geometry_group)
        start = QPushButton("开始双镜合束（DM固定）")
        start.clicked.connect(self.start_alignment)
        layout.addWidget(start)
        self.guidance_title = QLabel("操作引导")
        self.guidance_title.setStyleSheet("font-size:17px; font-weight:600")
        self.guidance_text = QLabel("DM和红光光路保持固定。保持相机方向和倍率一致，输入距离后开始，按照提示轮流遮挡红蓝光。")
        self.guidance_text.setWordWrap(True)
        layout.addWidget(self.guidance_title)
        layout.addWidget(self.guidance_text)
        self.action_panel = QWidget()
        self.action_layout = QVBoxLayout(self.action_panel)
        self.action_layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.action_panel)
        self.capture_label = QLabel("")
        self.capture_label.setWordWrap(True)
        layout.addWidget(self.capture_label)
        self.demo_group = QGroupBox("模拟光路操作（不连接硬件）")
        demo_layout = QVBoxLayout(self.demo_group)
        row = QHBoxLayout()
        self.demo_plane, self.demo_color = QComboBox(), QComboBox()
        self.demo_plane.addItems(["相机在近处", "相机在远处"])
        self.demo_color.addItems(["显示红光（遮蓝）", "显示蓝光（遮红）"])
        self.demo_plane.currentIndexChanged.connect(self.sim_selection)
        self.demo_color.currentIndexChanged.connect(self.sim_selection)
        row.addWidget(self.demo_plane)
        row.addWidget(self.demo_color)
        demo_layout.addLayout(row)
        row = QHBoxLayout()
        self.demo_mirror = QComboBox()
        self.demo_mirror.addItems(["调M2近反射镜", "调M1远反射镜"])
        row.addWidget(self.demo_mirror)
        for label, dx, dy in [("←", -1, 0), ("→", 1, 0), ("↑", 0, -1), ("↓", 0, 1)]:
            button = QPushButton(label)
            button.setAutoRepeat(True)
            button.clicked.connect(lambda checked=False, x=dx, y=dy: self.sim_move(x, y))
            row.addWidget(button)
        demo_layout.addLayout(row)
        layout.addWidget(self.demo_group)
        self.demo_group.hide()
        layout.addStretch()
        self.tabs.addTab(experiment, "合束引导")
        self.waist_panel = PluginPanel(self.waist_action)
        self.tabs.addTab(self.waist_panel, "束腰测量")
        self.level_panel = PluginPanel(lambda action, values: self.panel_action("level", action, values),
                                       identity="level", title="单镜等高与准直")
        self.tabs.addTab(self.level_panel, "等高／准直")
        self.plugin_panels = {"waist": self.waist_panel, "level": self.level_panel}
        self.level_demo = QGroupBox("模拟单镜／coupler（不控制硬件）")
        demo_form = QFormLayout(self.level_demo)
        self.level_position = QDoubleSpinBox()
        self.level_position.setRange(0, 100000)
        self.level_position.setValue(100)
        self.level_position.setSuffix(" mm")
        self.level_position.valueChanged.connect(self.level_sim_change)
        demo_form.addRow("反射点 → 模拟CCD", self.level_position)
        self.level_focus = QDoubleSpinBox()
        self.level_focus.setRange(-1, 1.5)
        self.level_focus.setDecimals(2)
        self.level_focus.setSingleStep(.05)
        self.level_focus.setValue(.9)
        self.level_focus.valueChanged.connect(self.level_sim_change)
        demo_form.addRow("模拟调焦量（任意刻度）", self.level_focus)
        row = QHBoxLayout()
        for title, direction in [("光斑上移", -1), ("光斑下移", 1)]:
            button = QPushButton(title)
            button.clicked.connect(lambda checked=False, d=direction: self.level_sim_mirror(d))
            row.addWidget(button)
        demo_form.addRow(row)
        self.level_panel.widget().layout().insertWidget(4, self.level_demo)
        self.level_demo.hide()
        library = QWidget()
        library_layout = QVBoxLayout(library)
        library_layout.addWidget(QLabel("本地插件 · 共用独立的插件依赖环境"))
        self.plugin_list = QListWidget()
        self.plugin_list.currentItemChanged.connect(self.select_plugin)
        library_layout.addWidget(self.plugin_list)
        row = QHBoxLayout()
        for label, callback in [("启用", self.enable_selected), ("停用", self.disable_selected), ("卸载", self.uninstall_selected)]:
            button = QPushButton(label)
            button.clicked.connect(callback)
            row.addWidget(button)
        library_layout.addLayout(row)
        install_button = QPushButton("导入插件包并安装依赖…")
        install_button.clicked.connect(self.import_package)
        library_layout.addWidget(install_button)
        self.plugin_detail = QLabel("选择插件查看结果。安装失败会保留现有共享环境。")
        self.plugin_detail.setWordWrap(True)
        library_layout.addWidget(self.plugin_detail)
        self.tabs.addTab(library, "插件库")
        settings_panel = QWidget()
        settings_layout = QFormLayout(settings_panel)
        self.center_input = QComboBox()
        self.center_input.addItems(["边界拟合圆／椭圆的中心（默认）", "实际光斑区域的面积中心"])
        self.center_input.setCurrentIndex(0 if self.fit_settings.center_mode == "boundary" else 1)
        settings_layout.addRow("定位中心", self.center_input)
        self.boundary_input = QDoubleSpinBox()
        self.boundary_input.setRange(3, 80)
        self.boundary_input.setDecimals(1)
        self.boundary_input.setValue(self.fit_settings.boundary_fraction * 100)
        self.boundary_input.setSuffix(" %")
        settings_layout.addRow("相对强度边界", self.boundary_input)
        explanation = QLabel("扣除背景后，以主光斑峰值的所选百分比确定边界。\n圆孔衍射仅定位中央亮斑，外围环保留显示。\n面积中心按区域面积计算，不按亮度加权；内部暗洞不计入面积。\n更改中心或边界将清空合束测量；束腰二阶矩宽度独立于显示圈。")
        explanation.setWordWrap(True)
        settings_layout.addRow(explanation)
        self.center_input.currentIndexChanged.connect(self.change_fit_settings)
        self.boundary_input.valueChanged.connect(self.change_fit_settings)
        self.tabs.addTab(settings_panel, "定位设置")
        diagnostics = QWidget()
        dlayout = QVBoxLayout(diagnostics)
        report_button = QPushButton("导出兼容性／性能报告（不含图像）")
        report_button.clicked.connect(self.save_report)
        dlayout.addWidget(report_button)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(500)
        dlayout.addWidget(self.log)
        self.tabs.addTab(diagnostics, "状态与预检")
        self.tabs.currentChanged.connect(self.resize_plugin_tab)
        splitter.addWidget(self.tabs)
        splitter.setSizes([900, 390])
        outer.addWidget(splitter, 1)

    def resize_plugin_tab(self, index):
        waist = self.tabs.currentWidget() in self.plugin_panels.values()
        self.tabs.setMinimumWidth(560 if waist else 360)
        self.tabs.setMaximumWidth(620 if waist else 460)

    def log_message(self, message):
        self.log.appendPlainText(time.strftime("%H:%M:%S ") + str(message))

    def analyze(self):
        while not self.analyzer_stop.wait(0.005):
            job = self.jobs.take()
            if job is None:
                continue
            frame, selection, generation, selection_generation, settings = job
            start = time.monotonic()
            try:
                fit = fit_beam(frame, selection, settings)
                self.signals.fit_ready.emit((frame, fit, generation, selection_generation, time.monotonic() - start))
            except Exception as error:
                self.signals.bridge_event.emit((generation, {"type": "error", "message": "拟合失败：" + str(error)}))

    def begin_source(self):
        self.source_generation += 1
        self.frames.clear()
        self.jobs.clear()
        self.last_frame = self.last_fit = None
        self.view.current_fit = None
        self.release_fixed_circle()
        self.capture = None
        self.coordinate_key = None
        self.exposure = ExposureController()
        self.exposure.set_enabled(self.auto_exposure_input.isChecked())
        self.pending_exposure = False
        self.capture_ready = False
        self.received_rate, self.fit_rate, self.display_rate = RateMeter(), RateMeter(), RateMeter()
        self.processed_rate = RateMeter()
        self.last_frame_ns = self.last_fit_ns = 0
        self.latencies.clear()
        self.rate.configure(self.fps_input.value())
        self.next_render = 0
        self.control_report = None
        self.reset_interactive("相机来源已改变，请重新测量")
        return self.source_generation

    def connect_demo(self):
        if self.source:
            self.source.close()
        generation = self.begin_source()
        source = Simulator()
        source.rate = self.fps_input.value()
        self.source = source
        self.apply_capabilities(source.capabilities)
        source.start(lambda frame: self.frames.put((generation, frame)),
                     lambda message: self.signals.bridge_event.emit((generation, {"type": "error", "message": message})))
        self.source_label.setText("模拟光路 · 合成数据 · 不连接相机")
        self.demo_group.show()
        self.log_message("已启动内存模拟光路；此结果不属于硬件验收。")

    def connect_real(self):
        default = Path(os.environ.get("BEAMIC_PRO_APP_ROOT", str(Path(sys.executable).parent))) / "bridge" / "BeamMicBridge.exe"
        executable, _ = QFileDialog.getOpenFileName(self, "选择 BeamMicBridge.exe", str(default), "桥接程序 (*.exe)")
        if not executable:
            return
        profile, _ = QFileDialog.getOpenFileName(self, "选择设备配置（默认配置支持预览）", str(self.assets / "bridge"), "设备配置 (*.json)")
        if not profile:
            return
        old_source, self.source = self.source, None
        generation = self.begin_source()
        report_path, _ = QFileDialog.getOpenFileName(self, "选择曝光预检报告（取消将只允许预览）", str(Path(profile).parent), "预检报告 (*.json)")
        if report_path:
            try:
                report = json.loads(Path(report_path).read_text(encoding="utf-8"))
                digest = hashlib.sha256(Path(profile).read_bytes()).hexdigest()
                if report.get("status") != "ready" or not report.get("control_test_passed") or report.get("simulated") or report.get("profile_sha256") != digest:
                    raise ValueError("预检报告未通过曝光关联测试，或设备配置已改变。")
                self.control_report = report
            except Exception as error:
                self.log_message(str(error))
        self.source_label.setText("正在连接原版 BeamMic…")
        self.demo_group.hide()
        def connect():
            client = None
            def ensure_open():
                if self.closing:
                    raise RuntimeError("软件正在关闭，连接已取消")
            try:
                if old_source:
                    old_source.close()
                ensure_open()
                holder = []
                def receive(message):
                    if message.get("type") == "frame" and holder:
                        try:
                            frame = holder[0].read_frame(message["descriptor"])
                            if frame:
                                self.frames.put((generation, frame))
                        except Exception as error:
                            self.signals.bridge_event.emit((generation, {"type": "error", "message": str(error)}))
                    else:
                        self.signals.bridge_event.emit((generation, message))
                client = BridgeClient(Path(executable), Path(profile), receive)
                holder.append(client)
                deadline = time.monotonic() + 10
                while not client.capabilities and time.monotonic() < deadline and not self.closing:
                    if client.process.poll() is not None:
                        raise RuntimeError("桥接启动失败，请查看状态报告")
                    time.sleep(0.05)
                ensure_open()
                if not client.capabilities:
                    raise RuntimeError("接口能力检查超时")
                client.command("probe")
                ensure_open()
                try:
                    client.command("start")
                except RuntimeError as error:
                    self.signals.bridge_event.emit((generation, {"type": "log", "message": str(error) + "；请在原版 BeamMic 选择相机后，点击开始采集／刷新预检。"}))
                ensure_open()
                client.set_rate(self.rate.current)
                if self.control_report is not None:
                    deadline = time.monotonic() + 2
                    while not client.capabilities.get("exposure_verified") and time.monotonic() < deadline and not self.closing:
                        time.sleep(.05)
                    ensure_open()
                    try:
                        client.command("probe")
                        ensure_open()
                        client.command("claim_exposure")
                    except RuntimeError as error:
                        self.signals.bridge_event.emit((generation, {"type": "log", "message": "曝光管理尚未就绪：" + str(error)}))
                ensure_open()
                self.signals.operation.emit({"kind": "connected", "source": client, "generation": generation})
            except Exception as error:
                if client:
                    client.close()
                self.signals.operation.emit({"kind": "error", "message": str(error), "generation": generation})
        self.pool.submit(connect)

    def apply_capabilities(self, capabilities):
        self.capabilities = capabilities
        controlled = capabilities.get("simulated") or (self.control_report is not None
                     and self.control_report.get("assembly_version") == capabilities.get("assembly_version")
                     and self.control_report.get("source") == capabilities.get("source"))
        self.capture_ready = bool(controlled and capabilities.get("exposure_verified") and capabilities.get("saturation_verified")
                                  and (capabilities.get("simulated") or (capabilities.get("exposure_control") and capabilities.get("exposure_owned"))))
        for attribute, key in [("minimum_us", "minimum_us"), ("maximum_us", "maximum_us"), ("step_us", "step_us")]:
            if key in capabilities:
                setattr(self.exposure, attribute, max(0.000001, float(capabilities[key])))
        self.log_message(json.dumps(capabilities, ensure_ascii=False))

    def restart_real_capture(self):
        if not isinstance(self.source, BridgeClient):
            return
        source, generation = self.source, self.source_generation
        def restart():
            try:
                source.command("probe")
                source.command("start")
                if self.control_report is not None:
                    source.command("claim_exposure")
            except RuntimeError as error:
                self.signals.operation.emit({"kind": "error", "message": str(error), "generation": generation})
        self.pool.submit(restart)

    def bridge_event(self, payload):
        generation, event = payload
        if generation != self.source_generation or self.closing:
            return
        kind = event.get("type")
        if kind == "capabilities":
            self.apply_capabilities(event["report"])
        elif kind == "error":
            self.exposure.invalidate(event.get("message", "相机错误"))
            self.capture = None
            self.capture_label.setText("测量已暂停：" + event.get("message", "相机错误"))
            self.log_message(event.get("message", "相机错误"))
            self.reset_interactive("相机数据错误，请重新测量")
        elif kind == "log":
            self.log_message(event.get("message", ""))

    def operation(self, result):
        if self.closing:
            if result.get("kind") == "connected":
                result["source"].close()
            return
        if result.get("generation", self.source_generation) != self.source_generation:
            if result.get("kind") == "connected":
                result["source"].close()
            return
        kind = result["kind"]
        if kind == "connected":
            self.source = result["source"]
            self.apply_capabilities(self.source.capabilities)
            self.source_label.setText("BeamMic · " + str(self.source.capabilities.get("source", "相机")))
        elif kind == "exposure":
            self.pending_exposure = False
            self.exposure.acknowledge(result["value"]["exposure_us"], result["value"]["epoch"])
        elif kind == "exposure_canceled":
            self.pending_exposure = False
            self.exposure.reject_request("自动曝光已关闭，等待当前曝光的有效帧")
        elif kind == "installed":
            self.installing = False
            self.reload_plugins()
            self.log_message("插件与共享环境安装完成：" + result["manifest"]["name"])
            if result.get("activate"):
                self.activate(result["manifest"]["id"])
        elif kind == "log":
            self.log_message(result["message"])
        elif kind == "status_report":
            try:
                report = result["report"]
                if "capabilities" in result:
                    report["capabilities"] = result["capabilities"]
                if result.get("probe_error"):
                    report["probe_error"] = result["probe_error"]
                Path(result["filename"]).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
                self.log_message("状态与接口诊断报告已导出")
            except (OSError, ValueError, TypeError) as error:
                self.log_message("状态报告导出失败：" + str(error))
        elif kind == "error":
            self.pending_exposure = False
            self.installing = False
            self.exposure.reject_request(result["message"])
            self.log_message(result["message"])
            self.capture_label.setText(result["message"])

    def tick(self):
        now = time.monotonic()
        if now < self.next_render:
            return
        packet = self.frames.take()
        if packet is None:
            return
        generation, frame = packet
        if generation != self.source_generation:
            return
        if self.last_frame and frame.frame_id <= self.last_frame.frame_id:
            return
        self.last_render = now
        self.next_render = max(self.next_render + 1 / self.rate.current, now + .001)
        if self.coordinate_key is not None and frame.coordinate_key != self.coordinate_key:
            self.capture = None
            self.jobs.clear()
            self.last_fit = None
            self.view.current_fit = None
            self.view.overlays(None, [])
            self.reset_interactive("采集坐标发生变化，请重新测量")
            self.view.selection = None
            self.release_fixed_circle()
        self.coordinate_key = frame.coordinate_key
        self.last_frame = frame
        if isinstance(frame, PreviewFrame):
            self.capture_ready = False
            self.capture = None
        if self.control_report is not None and not isinstance(frame, PreviewFrame):
            self.capture_ready = bool(self.capabilities.get("exposure_control") and self.capabilities.get("exposure_owned") and frame.exposure_verified
                                      and frame.saturation_verified
                                      and self.control_report.get("coordinate_key") == frame.coordinate_key)
        self.last_frame_ns = time.monotonic_ns()
        self.received_rate.add(frame.frame_id)
        self.view.set_image(frame)
        self.jobs.put((frame, self.view.selection, generation, self.selection_generation, self.fit_settings))
        request = self.exposure.inspect(frame, self.view.selection) if self.source and self.capture_ready else None
        if not self.capture_ready:
            self.exposure.invalidate("仅预览：需通过当前配置的曝光预检")
        if request is not None and self.source and not self.pending_exposure and self.capture_ready:
            self.pending_exposure = True
            source = self.source
            def set_exposure():
                try:
                    if not self.exposure.enabled or generation != self.source_generation:
                        self.signals.operation.emit({"kind": "exposure_canceled", "generation": generation})
                        return
                    value = source.set_exposure(request)
                    self.signals.operation.emit({"kind": "exposure", "value": value, "generation": generation})
                except Exception as error:
                    self.signals.operation.emit({"kind": "error", "message": str(error), "generation": generation})
            self.pool.submit(set_exposure)
        if isinstance(frame, PreviewFrame):
            self.exposure_label.setText("曝光未知 · 仅预览，暂不自动调曝光")
            self.capture_label.setText("已读取数值图像；定位圈仅供预览，曝光与量程验证后开放测量")
        else:
            self.exposure_label.setText(f"{frame.exposure_us / 1000:.3f} ms · {self.exposure.state.reason}")

    def painted(self, identity):
        if self.last_frame and identity == self.last_frame.frame_id:
            self.display_rate.add(identity)
            self.latencies.append((time.monotonic_ns() - self.last_frame.received_ns) / 1e6)

    def fit_result(self, result):
        frame, fit, generation, selection_generation, elapsed = result
        if generation != self.source_generation or selection_generation != self.selection_generation or self.closing:
            return
        if self.last_frame and frame.coordinate_key != self.last_frame.coordinate_key:
            return
        if self.last_frame and isinstance(frame, PreviewFrame) != isinstance(self.last_frame, PreviewFrame):
            return
        if self.last_fit and fit.frame_id <= self.last_fit.frame_id:
            return
        self.last_fit, self.last_fit_ns = fit, time.monotonic_ns()
        self.processed_rate.add(frame.frame_id)
        stable = (self.capture_ready and self.exposure.state.stable and self.exposure.state.mode == "measurement"
                  and frame.exposure_epoch == self.exposure.state.epoch
                  and self.exposure.last_exposure is not None
                  and np.isclose(frame.exposure_us, self.exposure.last_exposure, rtol=.005)
                  and self.last_frame and frame.coordinate_key == self.last_frame.coordinate_key)
        if fit.valid and stable:
            self.fit_rate.add(frame.frame_id)
        self.view.overlays(fit, [shape for shapes in self.overlays.values() for shape in shapes])
        self.update_circle_button()
        if self.last_frame and frame.frame_id == self.last_frame.frame_id:
            self.view.set_image(frame, fit)
        if fit.radius_minor > 0:
            mode = "边界拟合中心" if fit.center_mode == "boundary" else "面积中心"
            self.fit_label.setText(f"{mode} ({fit.center_x:.2f}, {fit.center_y:.2f}) px · "
                                   f"{fit.boundary_fraction:.1%} 边界半轴 {fit.radius_major:.2f} / {fit.radius_minor:.2f} px · {fit.reason}\n"
                                   f"边界中心 ({fit.boundary_center[0]:.2f}, {fit.boundary_center[1]:.2f}) · "
                                   f"面积中心 ({fit.area_center_x:.2f}, {fit.area_center_y:.2f}) px")
        else:
            self.fit_label.setText(fit.reason)
        if self.capture:
            try:
                measurement = self.capture.add(frame, fit, stable)
                self.capture_label.setText(self.capture.reason)
                if self.capture_owner in self.plugin_panels:
                    self.plugin_panels[self.capture_owner].note.setText(self.capture.reason)
                if measurement:
                    owner = self.plugins.get(self.capture_owner)
                    if owner:
                        owner.send({"type": "measurement", "measurement": measurement.to_dict()})
                    self.capture_label.setText(f"已记录 {measurement.sample_count} 帧；中心抖动 {measurement.center_std_px:.3f} px")
                    if self.capture_owner in self.plugin_panels:
                        self.plugin_panels[self.capture_owner].note.setText(self.capture_label.text())
                    self.capture = None
            except Exception as error:
                self.capture = None
                self.capture_label.setText(str(error))
                self.reset_interactive(str(error))
        if self.plugins and not isinstance(frame, PreviewFrame):
            descriptor = self.publisher.publish(frame)
            descriptor["frame"]["metadata"] = {**frame.metadata, "measurement_stable": bool(stable), "quantitative_ready": self.capture_ready,
                                                "selection": self.view.selection,
                                                "auto_exposure": self.exposure.enabled}
            for plugin in list(self.plugins.values()):
                if plugin.alive:
                    plugin.publish(descriptor, fit)

    def update_status(self):
        latency = float(np.percentile(self.latencies, 95)) if self.latencies else 0
        label = "预览收帧（帧标识未验证）" if isinstance(self.last_frame, PreviewFrame) else "新帧"
        self.metrics_label.setText(f"{label} {self.received_rate.fps:.1f} FPS · 有效拟合 {self.fit_rate.fps:.1f} FPS · "
                                  f"显示 {self.display_rate.fps:.1f} FPS · 收帧→显示 P95 {latency:.1f} ms · 当前目标 {self.rate.current:g}")
        throughput = self.received_rate.fps
        if self.processed_rate.fps > 0:
            throughput = min(throughput, self.processed_rate.fps)
        lowered = self.rate.observe(throughput, latency)
        if lowered is not None:
            self.log_message(f"吞吐或延迟不稳定，目标速率自动降低到 {lowered:g} FPS")
            if self.source:
                source = self.source
                self.pool.submit(source.set_rate, lowered)
        if self.last_frame_ns and time.monotonic_ns() - self.last_frame_ns > 2_000_000_000:
            if self.exposure.state.reason != "超过2秒没有新帧":
                self.exposure.invalidate("超过2秒没有新帧")
                self.capture = None
                self.capture_label.setText("测量暂停：没有新帧，请检查相机与遮光。")
                self.reset_interactive("没有新帧，请重新测量")
        if self.last_fit_ns and time.monotonic_ns() - self.last_fit_ns > 300_000_000:
            self.view.overlays(None, [shape for shapes in self.overlays.values() for shape in shapes])
            self.update_circle_button()

    def update_circle_button(self):
        fixed = self.view.fixed_fit is not None
        self.fixed_options.setVisible(fixed)
        for checkbox, value in ((self.track_center_input, self.view.track_center),
                                (self.connect_centers_input, self.view.connect_centers)):
            checkbox.blockSignals(True)
            checkbox.setChecked(value)
            checkbox.blockSignals(False)
        if fixed:
            self.circle_note.setText("紫色＋：固定圈中心；白色×：实时中心；黄色线：两中心连线。\n"
                                     "定位不可靠时隐藏实时标记和连线，固定参考保持。")
        else:
            self.circle_note.setText("蓝色外缘对应所选相对强度；白色表示显示峰值，饱和状态以提示为准。\n"
                                     "圈默认标出中心；实际定位和测量持续更新。")
        self.fixed_circle_button.setText("解除固定" if fixed else "固定圈")
        self.fixed_circle_button.setEnabled(self.auto_circle.isChecked() and
                                            (fixed or bool(self.view.current_fit and self.view.current_fit.valid)))

    def change_auto_circle(self, enabled):
        self.view.set_auto_circle(enabled)
        self.update_circle_button()

    def toggle_fixed_circle(self):
        self.view.toggle_fixed_circle()
        self.update_circle_button()

    def change_fixed_display(self, *_):
        self.view.set_fixed_display(self.track_center_input.isChecked(), self.connect_centers_input.isChecked())

    def release_fixed_circle(self):
        self.view.clear_fixed_circle()
        self.update_circle_button()

    def change_fit_settings(self, *_):
        self.fit_settings = FitSettings(boundary_fraction=self.boundary_input.value() / 100,
                                        center_mode="boundary" if self.center_input.currentIndex() == 0 else "area")
        self.selection_generation += 1
        self.jobs.clear()
        self.last_fit = None
        self.view.current_fit = None
        self.release_fixed_circle()
        if self.interactive != "waist":
            self.reset_interactive("定位定义已改变，近远红蓝中心均需重新记录")
        self.settings_path.parent.mkdir(parents=True, exist_ok=True)
        self.settings_path.write_text(json.dumps({"center_mode": self.fit_settings.center_mode,
                                                 "boundary_fraction": self.fit_settings.boundary_fraction}), encoding="utf-8")
        self.capture_label.setText("定位定义已改变，请重新测量；设置已保存。")

    def change_rate(self, value):
        self.rate.configure(value)
        self.next_render = 0
        if self.source:
            self.pool.submit(self.source.set_rate, value)

    def change_mode(self, index):
        if index == 1 and self.interactive:
            self.mode.blockSignals(True)
            self.mode.setCurrentIndex(0)
            self.mode.blockSignals(False)
            self.log_message("交互测量期间使用未饱和测量模式。")
            return
        self.exposure.set_mode("finding" if index else "measurement")
        self.capture = None

    def change_auto_exposure(self, enabled):
        self.exposure.set_enabled(enabled)
        if self.capture:
            self.capture.fits.clear()
        self.capture_label.setText("曝光策略已切换，等待有效的新帧；当前记录将重新累计。")

    def measurement_mode(self):
        self.mode.blockSignals(True)
        self.mode.setCurrentIndex(0)
        self.mode.blockSignals(False)
        if self.exposure.state.mode != "measurement":
            self.exposure.set_mode("measurement")

    def select_roi(self, selection):
        self.release_fixed_circle()
        self.selection_generation += 1
        self.exposure.invalidate("主光斑选择已改变")
        self.capture = None
        self.jobs.clear()
        if self.interactive and self.interactive in self.plugins:
            self.plugins[self.interactive].send({"type": "action", "action": "selection_changed", "values": {}})
        self.overlays.clear()

    def clear_selection(self):
        self.view.selection = None
        self.select_roi(None)

    def sim_selection(self):
        if isinstance(self.source, Simulator):
            self.source.plane = "near" if self.demo_plane.currentIndex() == 0 else "far"
            self.source.color = "red" if self.demo_color.currentIndex() == 0 else "blue"
            self.exposure.invalidate("切换光束，等待曝光稳定")

    def sim_move(self, dx, dy):
        if isinstance(self.source, Simulator):
            self.source.move("near" if self.demo_mirror.currentIndex() == 0 else "far", dx, dy)

    def reload_plugins(self):
        self.catalog = {}
        for path in (self.assets / "plugins").glob("*/manifest.json"):
            manifest = PluginStore.validate_manifest(json.loads(path.read_text(encoding="utf-8")))
            self.catalog[manifest["id"]] = {"manifest": manifest, "directory": path.parent, "builtin": True}
        for identity, item in self.store.read()["plugins"].items():
            if item["manifest"].get("sdk_version") != 2:
                continue
            self.catalog[identity] = {"manifest": item["manifest"], "directory": self.store.root / item["path"], "builtin": False}
        selected = self.plugin_list.currentItem().data(Qt.ItemDataRole.UserRole) if self.plugin_list.currentItem() else None
        self.plugin_list.clear()
        for identity, item in self.catalog.items():
            entry = QListWidgetItem(item["manifest"]["name"] + (" · 运行中" if identity in self.plugins else ""))
            entry.setData(Qt.ItemDataRole.UserRole, identity)
            self.plugin_list.addItem(entry)
            if selected == identity:
                self.plugin_list.setCurrentItem(entry)

    def select_plugin(self, item, previous=None):
        if item:
            identity = item.data(Qt.ItemDataRole.UserRole)
            event = self.plugin_states.get(identity)
            self.plugin_detail.setText(event.get("text", "等待插件结果") if event else "启用后显示插件结果。")

    def enable_selected(self):
        item = self.plugin_list.currentItem()
        if item:
            self.activate(item.data(Qt.ItemDataRole.UserRole))

    def activate(self, identity):
        if identity in self.plugins:
            return True
        if self.installing:
            self.log_message("正在更新共享环境，请稍后启用插件。")
            return False
        item = self.catalog[identity]
        if identity in self.plugin_panels and (not self.source or not self.capture_ready):
            self.log_message("交互测量需要先连接模拟光路或通过曝光预检的 BeamMic 相机。")
            return False
        if item["manifest"].get("kind") == "interactive" and self.interactive:
            self.log_message("已有引导任务运行，请先停用该任务。")
            return False
        if identity in self.plugin_panels and isinstance(self.source, Simulator):
            self.source.waist_mode = identity == "waist"
            self.source.level_mode = identity == "level"
            self.level_demo.setVisible(identity == "level")
            self.exposure.invalidate("切换模拟测量光路，等待新帧")
        try:
            if (self.store.read()["environment"] and self.store.read().get("sdk_version") == 2
                    and (identity not in self.plugin_panels or self.store.read().get("sdk_package_version") == "2.1.0")):
                python, sdk_path = self.store.current_python(), None
            elif self.development and not self.store.read()["environment"]:
                python, sdk_path = Path(sys.executable), self.assets / "sdk" / "src"
            else:
                package = self.store.root / (identity + "-builtin.zip")
                with zipfile.ZipFile(package, "w", zipfile.ZIP_DEFLATED) as archive:
                    for path in item["directory"].rglob("*"):
                        if path.is_file() and "__pycache__" not in path.parts:
                            archive.write(path, path.relative_to(item["directory"]))
                self.install_package(package, activate=True)
                return False
            plugin = PluginProcess(python, item["directory"], item["manifest"], self.signals.plugin_event.emit, sdk_path)
            self.plugins[identity] = plugin
            if item["manifest"].get("kind") == "interactive":
                self.interactive = identity
                self.measurement_mode()
            self.reload_plugins()
            if identity in self.plugin_panels:
                self.tabs.setCurrentWidget(self.plugin_panels[identity])
            return True
        except Exception as error:
            self.log_message(str(error))
            return False

    def stop_plugin(self, identity):
        plugin = self.plugins.pop(identity, None)
        if plugin:
            plugin.close()
        self.overlays.pop(identity, None)
        if self.interactive == identity:
            self.interactive = None
            self.capture = None
            self.capture_owner = None
            self.guidance_text.setText("交互测量插件已停用。")
            self.set_actions([])
        if identity in self.plugin_panels:
            panel = self.plugin_panels[identity]
            panel.set_event({"title": panel.default_title+"已停用", "text": "测量数据已释放；重新启用后从设置开始。",
                             "actions": [{"id": "activate", "label": "启用"+panel.default_title+"插件"}]})
            if isinstance(self.source, Simulator):
                self.source.waist_mode = False
                self.source.level_mode = False
            self.level_demo.hide()
        self.reload_plugins()

    def disable_selected(self):
        item = self.plugin_list.currentItem()
        if item:
            self.stop_plugin(item.data(Qt.ItemDataRole.UserRole))

    def uninstall_selected(self):
        item = self.plugin_list.currentItem()
        if not item or self.installing:
            return
        identity = item.data(Qt.ItemDataRole.UserRole)
        self.stop_plugin(identity)
        try:
            self.store.uninstall(identity)
            self.reload_plugins()
        except Exception as error:
            self.log_message(str(error))

    def import_package(self):
        package, _ = QFileDialog.getOpenFileName(self, "导入插件包", "", "插件包 (*.zip)")
        if package:
            self.install_package(Path(package))

    def install_package(self, package, activate=False):
        if self.installing:
            return
        for identity in list(self.plugins):
            self.stop_plugin(identity)
        self.installing = True
        self.log_message("正在构建候选共享环境；通过依赖和入口验证后切换。")
        def install():
            try:
                manifest = self.store.install(package, lambda text: self.signals.operation.emit({"kind": "log", "message": text}))
                self.signals.operation.emit({"kind": "installed", "manifest": manifest, "activate": activate})
            except Exception as error:
                self.signals.operation.emit({"kind": "error", "message": str(error)})
        self.pool.submit(install)

    def start_alignment(self):
        if not self.source or not self.capture_ready or self.last_frame is None:
            self.log_message("需要先连接并核对读帧、曝光控制、曝光关联和饱和量程，才能开始测量。")
            return
        values = {key: spin.value() for key, spin in self.geometry_inputs.items()}
        try:
            geometry = Geometry(**values)
            if isinstance(self.source, Simulator):
                self.source.geometry = geometry
            if not self.activate("alignment"):
                return
            self.measurement_mode()
            self.plugins["alignment"].send({"type": "action", "action": "configure", "values": values})
            self.tabs.setCurrentIndex(0)
        except Exception as error:
            self.log_message(str(error))

    def set_actions(self, actions):
        while self.action_layout.count():
            widget = self.action_layout.takeAt(0).widget()
            if widget:
                widget.deleteLater()
        for action in actions:
            button = QPushButton(str(action.get("label", action["id"])))
            button.clicked.connect(lambda checked=False, identity=action["id"]: self.send_action(identity))
            self.action_layout.addWidget(button)

    def send_action(self, action):
        if self.interactive and self.interactive in self.plugins:
            if action == "reset":
                self.capture = None
            try:
                self.plugins[self.interactive].send({"type": "action", "action": action, "values": {}})
            except RuntimeError as error:
                self.log_message(str(error))

    def waist_action(self, action, values):
        self.panel_action("waist", action, values)

    def level_sim_change(self, *_):
        if isinstance(self.source, Simulator) and self.source.level_mode:
            changed_focus = not np.isclose(self.source.level_focus, self.level_focus.value())
            self.source.level_position_mm = self.level_position.value()
            self.source.level_focus = self.level_focus.value()
            self.exposure.invalidate("模拟位置／调焦改变，等待新帧")
            if changed_focus and "level" in self.plugins:
                self.panel_action("level", "coupler_changed", {})

    def level_sim_mirror(self, direction):
        if isinstance(self.source, Simulator) and self.source.level_mode:
            self.source.level_tilt += direction*.00002

    def panel_action(self, identity, action, values):
        panel = self.plugin_panels[identity]
        if action == "activate":
            if not self.source or not self.capture_ready:
                panel.text.setPlainText("请先连接模拟光路或通过曝光预检的 BeamMic 相机。")
                return
            if self.activate(identity):
                self.tabs.setCurrentWidget(panel)
            return
        plugin = self.plugins.get(identity)
        if plugin is None or self.interactive != identity:
            return
        if action in ("reset", "coupler_changed"):
            self.capture = None
            self.capture_owner = None
        if identity == "waist" and action == "record" and isinstance(self.source, Simulator):
            self.source.waist_position_mm = float(values.get("z_mm", 0))
            self.exposure.invalidate("模拟相机位置改变，等待新曝光稳定")
        if identity == "level" and action == "record" and isinstance(self.source, Simulator):
            state = self.plugin_states.get(identity, {}).get("data", {})
            stage = state.get("stage")
            settings = state.get("settings", {})
            if stage in ("coarse_capture", "collimation_capture"):
                z = settings.get("near_mm", 100)+float(values.get("z_mm", state.get("pending_z", 0)))
            elif stage == "level_return":
                z = settings.get("near_mm", 100)
            elif stage == "level_capture":
                z = settings.get("near_mm", 100) if not state.get("level_near_recorded") else settings.get("far_mm", 900)
            else:
                z = self.source.level_position_mm
            self.level_position.setValue(z)
            self.level_sim_change()
        values = {**values, "min_frame_id": self.last_frame.frame_id if self.last_frame else -1}
        try:
            plugin.send({"type": "action", "action": action, "values": values})
        except RuntimeError as error:
            self.log_message(str(error))

    def reset_interactive(self, reason):
        self.capture = None
        if self.interactive and self.interactive in self.plugins:
            try:
                self.plugins[self.interactive].send({"type": "action", "action": "reset", "values": {}})
            except RuntimeError:
                pass
            self.log_message(reason)
        self.overlays.clear()

    def plugin_event(self, identity, event):
        if identity not in self.plugins or self.closing:
            return
        kind = event.get("type")
        if kind == "panel":
            self.plugin_states[identity] = event
            if identity in self.plugin_panels:
                self.plugin_panels[identity].set_event(event)
            if identity == self.interactive:
                self.guidance_title.setText(event.get("title", "操作引导"))
                self.guidance_text.setText(event.get("text", ""))
                self.set_actions(event.get("actions", []) if identity not in self.plugin_panels else [])
            item = self.plugin_list.currentItem()
            if item and item.data(Qt.ItemDataRole.UserRole) == identity:
                self.plugin_detail.setText(event.get("text", ""))
        elif kind == "overlay":
            self.overlays[identity] = event.get("shapes", [])[:100]
        elif kind == "measurement_mode" and (identity == self.interactive or not self.interactive):
            self.measurement_mode()
        elif kind == "measurement_request":
            if not self.capture_ready or self.last_frame is None:
                self.log_message("定量测量接口尚未就绪。")
                return
            if self.capture or (self.interactive and identity != self.interactive):
                self.log_message("已有测量／引导任务，请等待。")
                return
            self.measurement_mode()
            self.capture_owner = identity
            self.capture = Capture(event["role"], event.get("samples", 20), self.exposure.state.epoch,
                                   self.last_frame.coordinate_key, self.last_frame.frame_id,
                                   reference_radius=event.get("reference_radius"))
            self.capture_label.setText("等待曝光稳定，记录新的有效帧…")
        elif kind == "error":
            self.log_message(f"插件 {identity}：{event.get('message', '运行错误')}")
            self.stop_plugin(identity)
        elif kind == "log":
            self.log_message(f"插件 {identity}：{event.get('message', '')}")

    def save_report(self):
        filename, _ = QFileDialog.getSaveFileName(self, "导出状态报告", "beamic-status.json", "状态报告 (*.json)")
        if not filename:
            return
        from . import __version__
        report = {"application_version": __version__, "simulated": isinstance(self.source, Simulator), "capabilities": getattr(self, "capabilities", {}),
                  "requested_fps": self.rate.requested, "current_target_fps": self.rate.current,
                  "received_fps": self.received_rate.fps, "effective_fit_fps": self.fit_rate.fps,
                  "display_fps": self.display_rate.fps, "images_saved": False,
                  "receive_to_display_p95_ms": float(np.percentile(self.latencies, 95)) if self.latencies else None,
                  "latency_scope": "host_receive_to_paint; camera/vendor latency is not included",
                  "hardware_alignment_acceptance": "pending", "active_plugins": list(self.plugins)}
        report["spot_definition"] = {"center_mode": self.fit_settings.center_mode,
                                     "boundary_fraction": self.fit_settings.boundary_fraction,
                                     "boundary_kind": "relative_intensity"}
        report["auto_exposure"] = self.exposure.enabled
        report["preview_only"] = isinstance(self.last_frame, PreviewFrame) or self.capabilities.get("preview_only", False)
        report["fps_note"] = "预览收帧速度不代表已验证的相机有效新帧速度" if report["preview_only"] else "有效新帧率仍需实机核查"
        source, generation = self.source, self.source_generation
        def export():
            event = {"kind": "status_report", "report": report, "filename": filename, "generation": generation}
            if isinstance(source, BridgeClient):
                try:
                    event["capabilities"] = source.command("probe")
                except Exception as error:
                    event["probe_error"] = str(error)
            self.signals.operation.emit(event)
        self.pool.submit(export)

    def closeEvent(self, event):
        if self.closing:
            event.accept()
            return
        self.closing = True
        self.tick_timer.stop()
        self.status_timer.stop()
        self.analyzer_stop.set()
        self.store.cancel_event.set()
        for plugin in list(self.plugins.values()):
            plugin.close()
        self.plugins.clear()
        if self.source:
            self.source.close()
        # Qt objects must outlive every worker that can emit through them.
        self.analyzer.join()
        self.pool.shutdown(wait=True, cancel_futures=True)
        self.publisher.close()
        event.accept()


def configure_application(application):
    application.setApplicationName("beamic-pro")
    application.setStyle("Fusion")
    application.setStyleSheet("""
        QWidget { background:#111c2d; color:#e1eaf3; font-size:13px; }
        QMainWindow { background:#0b1320; }
        QPushButton { background:#21384d; border:1px solid #38546a; border-radius:6px; padding:8px 10px; }
        QPushButton:hover { background:#2c4f66; }
        QPushButton:checked { background:#235951; }
        QLineEdit,QDoubleSpinBox,QComboBox,QPlainTextEdit,QListWidget { background:#0b1524; border:1px solid #2d4358; border-radius:4px; padding:5px; }
        QGroupBox { border:1px solid #2d4358; border-radius:6px; margin-top:12px; padding:12px; }
        QGroupBox::title { subcontrol-origin:margin; padding:0 6px; color:#9dc5d9; }
        QTabBar::tab { padding:9px 12px; background:#18283a; }
        QTabBar::tab:selected { color:#7ce5cb; background:#254051; }
        QSplitter::handle { background:#24364b; width:5px; }
    """)


def main(argv=None):
    parser = argparse.ArgumentParser(description="beamic pro 光学实验助手")
    parser.add_argument("--demo", action="store_true", help="使用内存模拟光路")
    parser.add_argument("--data-dir", type=Path)
    args = parser.parse_args(argv)
    application = QApplication(sys.argv[:1])
    configure_application(application)
    window = Window(args.data_dir, args.demo)
    window.show()
    return application.exec()


if __name__ == "__main__":
    raise SystemExit(main())


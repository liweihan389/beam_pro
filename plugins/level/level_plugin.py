"""Guided single-mirror levelling and independently verified collimation."""

from __future__ import annotations

import numpy as np
from beamic_sdk import Plugin
from level_math import (
    fit_interval,
    geometry,
    height_target,
    plot_payload,
    validate_pair,
    verify_height,
)
from width_math import measure_width, summarize


class LevelPlugin(Plugin):
    stage_titles = {
        "setup": "模式与设置",
        "level_capture": "近远光高测量",
        "level_guide": "远处单镜调节",
        "level_return": "返回近处校验",
        "level_done": "等高结果",
        "level_outside": "目标超出画面",
        "coarse_capture": "近远准直粗查",
        "focus": "手动调焦反馈",
        "collimation_capture": "多点准直确认",
        "collimation_initial": "冻结预测，尚未验证",
        "collimation_final": "准直验收结果",
    }

    def on_activate(self):
        self.settings = {
            "level_enabled": True,
            "collimate_enabled": False,
            "near_mm": 100.0,
            "far_mm": 900.0,
            "wavelength_nm": 780.0,
            "position_tol_mm": 10.0,
            "height_tol_mm": 0.01,
            "height_limit_mm": 0.05,
            "diameter_pct": 5.0,
            "pixel_pitch_um": 3.45,
            "magnification": 1.0,
            "scale_pct": 2.0,
            "coupler_confirmed": False,
        }
        self.latest = -1
        self.pitch = None
        self.signature = None
        self.generation = 0
        self.audit = []
        self.clear()
        self.publish(
            "默认只开启光高：同一固定高度支架／导轨，CCD竖轴对应平台法线，方向和倍率保持一致。"
        )

    def clear(self):
        self.stage = "setup"
        self.waiting = False
        self.points = []
        self.coarse = []
        self.frozen = self.model = None
        self.base_count = 3
        self.height = {}
        self.target = None
        self.height_result = None
        self.collimation_result = None
        self.joint = None
        self.arrivals = 0
        self.reached = False
        self.widths = []
        self.identities = []
        self.live_reason = ""
        self.live_width_um = None
        self.feedback_ns = -1
        self.pending_z = 0.0
        self.host.overlay([])

    @staticmethod
    def number(key, label, value, minimum=0, maximum=100000, suffix="", decimals=2):
        return {
            "id": key,
            "label": label,
            "type": "number",
            "value": value,
            "minimum": minimum,
            "maximum": maximum,
            "suffix": suffix,
            "decimals": decimals,
        }

    def fields(self):
        s = self.settings
        if self.stage == "setup":
            result = [
                {"id": k, "label": label, "type": "boolean", "value": s[k]}
                for k, label in [
                    ("level_enabled", "光高：近远等高"),
                    ("collimate_enabled", "准直：五点独立验证"),
                ]
            ]
            specs = [
                ("near_mm", "反射点 → 近处感光面", 0, " mm"),
                ("far_mm", "反射点 → 远处感光面", 0, " mm"),
                ("wavelength_nm", "波长（仅准直使用）", 1, " nm"),
                ("position_tol_mm", "位置读数公差 ±", 0, " mm"),
                ("height_tol_mm", "支架高度重复定位公差 ±", 0, " mm"),
                ("height_limit_mm", "等高允许高差 ≤", 0.001, " mm"),
                ("diameter_pct", "准直区间直径变化 ≤", 0.1, " %"),
                ("pixel_pitch_um", "有效像元（按帧核对）", 0.01, " μm"),
                ("magnification", "成像倍率，直测为1", 0.01, " ×"),
                ("scale_pct", "共同尺度公差 ±", 0, " %"),
            ]
            result += [
                self.number(
                    k,
                    label,
                    s[k],
                    minimum,
                    100000,
                    suffix,
                    3
                    if k in ("height_tol_mm", "height_limit_mm", "magnification")
                    else 2,
                )
                for k, label, minimum, suffix in specs
            ]
            result.append(
                {
                    "id": "coupler_confirmed",
                    "label": "启用准直时：coupler确有独立轴向调焦功能",
                    "type": "boolean",
                    "value": s["coupler_confirmed"],
                }
            )
            return result
        if self.stage == "collimation_capture" and not self.waiting:
            return [
                self.number(
                    "z_mm",
                    "本点距近处的实际位置",
                    self.pending_z,
                    0,
                    s["far_mm"] - s["near_mm"],
                    " mm",
                )
            ]
        return []

    def actions(self):
        if self.stage == "setup":
            return [{"id": "configure", "label": "确认设置，开始近处测量"}]
        actions = []
        if not self.waiting:
            if self.stage in (
                "level_capture",
                "level_return",
                "coarse_capture",
                "collimation_capture",
            ):
                actions.append({"id": "record", "label": "记录本点 · 20个有效帧"})
            elif self.stage == "level_guide" and self.reached:
                actions.append(
                    {"id": "confirm_far", "label": "远处已到目标，记录后回近处"}
                )
            elif self.stage == "focus":
                actions.append(
                    {"id": "start_validation", "label": "调焦完成，开始五点确认"}
                )
                actions.append({"id": "coarse_again", "label": "重新测近远两点粗查"})
            elif self.stage == "collimation_initial":
                actions.append(
                    {
                        "id": "continue",
                        "label": f"冻结{self.base_count}点预测，测两个新位置",
                    }
                )
            elif self.stage == "collimation_final":
                if (
                    self.collimation_result
                    and self.collimation_result["passed"]
                    and self.joint
                    and self.joint["passed"]
                ):
                    if self.settings["level_enabled"]:
                        actions.append(
                            {"id": "start_level", "label": "准直通过，开始单镜等高"}
                        )
                else:
                    actions.append({"id": "add", "label": "补测一个新位置"})
        if self.settings["collimate_enabled"]:
            actions.append(
                {"id": "coupler_changed", "label": "已调coupler：清空本轮并重新确认"}
            )
        actions.append({"id": "reset", "label": "重新设置／开始"})
        return actions

    def publish(self, message):
        table = {
            "headers": ["点", "相对近处 mm", "X直径 μm", "Y直径 μm", "用途"],
            "rows": [],
        }
        for i, p in enumerate(self.points):
            table["rows"].append(
                [
                    i + 1,
                    f"{p.z_mm:.1f}",
                    f"{p.diameters_um[0]:.2f}",
                    f"{p.diameters_um[1]:.2f}",
                    "建模"
                    if i < self.base_count
                    else "独立验证"
                    if i < self.base_count + 2
                    else "补测",
                ]
            )
        if not self.points:
            for i, p in enumerate(self.coarse):
                table["rows"].append(
                    [
                        i + 1,
                        f"{p.z_mm:.1f}",
                        f"{p.diameters_um[0]:.2f}",
                        f"{p.diameters_um[1]:.2f}",
                        "粗查，非验收",
                    ]
                )
        if self.live_width_um is not None and self.stage == "focus":
            message += f"\n当前D4σ直径：X {self.live_width_um[0]:.2f} μm / Y {self.live_width_um[1]:.2f} μm（辅助反馈，非正式记录）。"
        if self.model:
            result = self.model.variation(self.settings["diameter_pct"])
            upper = result["joint_upper_95_pct"]
            message += (
                f"\n区间X/Y直径变化：{result['nominal_pct'][0]:.2f}% / {result['nominal_pct'][1]:.2f}%；联合95%上限："
                + (f"{upper:.2f}%" if upper is not None else "不足以确定")
            )
            message += (
                "\n波长物理一致性："
                + {
                    "consistent": "一致",
                    "unresolved": "物理参数未能确定；不要求锁定束腰",
                    "conflict": "与自由传播物理约束明显冲突",
                }[self.model.physical]
            )
        message += (
            f"\n本轮支架高度重复定位公差±{self.settings['height_tol_mm']:.3f} mm；"
            "请按实际能力填写，未实测预算不能视为仪器准确度证明。"
        )
        data = {
            "stage": self.stage,
            "waiting": self.waiting,
            "count": len(self.points),
            "base_count": self.base_count,
            "collimation_passed": bool(
                self.collimation_result
                and self.collimation_result["passed"]
                and self.joint
                and self.joint["passed"]
            ),
            "coarse_count": len(self.coarse),
            "level_near_recorded": "near" in self.height,
            "samples": len(self.widths),
            "settings": self.settings,
            "pending_z": self.pending_z,
            "generation": self.generation,
            "target": self.target,
            "reached": self.reached,
            "height_result": self.height_result,
            "collimation_result": self.collimation_result,
            "joint_validation": self.joint,
            "audit": self.audit[-20:],
            "live_reason": self.live_reason,
            "live_width_um": self.live_width_um,
            "coarse_widths": [list(p.diameters_um) for p in self.coarse],
        }
        self.host.panel(
            "单镜等高／准直 · " + self.stage_titles[self.stage],
            message,
            self.actions(),
            fields=self.fields(),
            table=table,
            plot=plot_payload(self.model, self.points) if self.model else {},
            data=data,
        )

    def begin_level(self):
        self.stage = "level_capture"
        self.height = {}
        self.target = None
        self.waiting = False
        self.host.overlay([])
        self.publish(
            "将CCD放在近处，保持固定高度。记录近处几何中心；此后仅调整这一面反射镜的竖直指向。"
        )

    def prepare_width(self, z, coarse=False):
        self.pending_z = float(z)
        self.waiting = False
        self.stage = "coarse_capture" if coarse else "collimation_capture"
        self.publish(
            f"{'近远粗查，不能据此通过准直' if coarse else '五点确认'}：相对近处 {z:g} mm。移动CCD后核对实际位置，再记录。"
        )

    def on_action(self, action, values):
        try:
            if action == "configure" and self.stage == "setup":
                s = {
                    k: (bool(values[k]) if isinstance(v, bool) else float(values[k]))
                    if k in values
                    else v
                    for k, v in self.settings.items()
                }
                if not np.isfinite(
                    [v for v in s.values() if not isinstance(v, bool)]
                ).all():
                    raise ValueError("设置须为有限数值")
                geometry(s["near_mm"], s["far_mm"], s["position_tol_mm"])
                if not (s["level_enabled"] or s["collimate_enabled"]):
                    raise ValueError("至少选择光高或准直")
                if (
                    min(
                        s["pixel_pitch_um"],
                        s["magnification"],
                        s["height_limit_mm"],
                        s["diameter_pct"],
                    )
                    <= 0
                    or s["height_tol_mm"] < 0
                    or not 0 <= s["scale_pct"] <= 20
                ):
                    raise ValueError("像元、倍率或门槛无效")
                if s["collimate_enabled"] and (
                    not s["coupler_confirmed"] or s["wavelength_nm"] <= 0
                ):
                    raise ValueError("请确认coupler具有独立调焦功能，并填写有效波长")
                if self.pitch is not None and not np.isclose(
                    s["pixel_pitch_um"], self.pitch, rtol=0.005
                ):
                    raise ValueError("有效像元与当前帧不一致")
                self.settings = s
                self.minimum_frame = max(
                    self.latest, int(values.get("min_frame_id", -1))
                )
                self.signature = None
                self.host.measurement_mode()
                if s["collimate_enabled"]:
                    self.prepare_width(0, True)
                else:
                    self.begin_level()
            elif action == "record" and not self.waiting:
                if self.stage in ("level_capture", "level_return"):
                    role = (
                        "level-near-new"
                        if self.stage == "level_return"
                        else "level-near"
                        if "near" not in self.height
                        else "level-far"
                    )
                    self.waiting = True
                    self.host.request_measurement(role)
                    self.publish("记录几何中心：等待20个有效的新曝光帧。")
                elif self.stage in ("coarse_capture", "collimation_capture"):
                    z = float(values.get("z_mm", self.pending_z))
                    span = self.settings["far_mm"] - self.settings["near_mm"]
                    if not np.isfinite(z) or not 0 <= z <= span:
                        raise ValueError("测量位置须在近远工作区间内")
                    if self.stage == "collimation_capture":
                        if (
                            len(self.points) == 0
                            and z != 0
                            or len(self.points) == 1
                            and z != span
                        ):
                            raise ValueError("前两点必须依次为近端0和远端")
                        if any(
                            abs(p.z_mm - z) <= 2 * self.settings["position_tol_mm"]
                            for p in self.points
                        ):
                            raise ValueError("新位置与已有点过近，不能算独立空间采样")
                    self.pending_z = z
                    self.widths = []
                    self.identities = []
                    self.waiting = True
                    self.minimum_frame = max(
                        self.latest, int(values.get("min_frame_id", -1))
                    )
                    self.epoch = None
                    self.exposure_us = None
                    self.publish("记录二阶矩宽度：0/20；不使用显示圈作为宽度。")
            elif (
                action == "confirm_far" and self.stage == "level_guide" and self.reached
            ):
                self.waiting = True
                self.host.request_measurement("level-far-new")
                self.publish("停止调镜，重新记录远处中心，再返回近处校验。")
            elif action == "start_validation" and self.stage == "focus":
                self.points = []
                self.coarse = []
                self.model = self.frozen = None
                self.joint = self.collimation_result = None
                self.audit.append(
                    {"event": "new_focus_state", "generation": self.generation}
                )
                self.prepare_width(0)
            elif action == "coarse_again" and self.stage == "focus":
                self.coarse = []
                self.prepare_width(0, True)
            elif action == "continue" and self.stage == "collimation_initial":
                self.prepare_width(self.new_position(self.span * 0.25))
            elif (
                action == "start_level"
                and self.stage == "collimation_final"
                and self.collimation_result
                and self.collimation_result["passed"]
                and self.joint
                and self.joint["passed"]
            ):
                self.begin_level()
            elif action == "add" and self.stage == "collimation_final":
                candidates = np.linspace(0, self.span, 17)[1:-1]
                available = [
                    z
                    for z in candidates
                    if all(
                        abs(p.z_mm - z) > 2 * self.settings["position_tol_mm"]
                        for p in self.points
                    )
                ]
                if not available:
                    raise ValueError("区间内没有足够分离的新位置，请重新选择距离")
                self.prepare_width(
                    max(
                        available,
                        key=lambda z: min(abs(p.z_mm - z) for p in self.points),
                    )
                )
            elif action == "coupler_changed" and self.settings["collimate_enabled"]:
                self.generation += 1
                self.clear()
                self.stage = "focus"
                self.audit.append(
                    {
                        "event": "coupler_changed_all_records_invalidated",
                        "generation": self.generation,
                    }
                )
                self.publish(
                    "已调coupler：所有旧宽度、预测、等高目标和中心记录均已清空。调焦稳定后重新开始五点确认；不提供未校准的旋钮方向／圈数。"
                )
            elif action in ("reset", "selection_changed"):
                self.clear()
                self.signature = None
                self.publish(
                    "重新核对设置：来源、坐标、选区或定位定义变化会使旧测量失效。"
                )
        except (ValueError, TypeError, OverflowError, np.linalg.LinAlgError) as error:
            self.publish("输入或测量未采用：" + str(error))

    @property
    def span(self):
        return self.settings["far_mm"] - self.settings["near_mm"]

    def new_position(self, preferred):
        available = [
            z
            for z in [preferred, *np.linspace(0, self.span, 17)[1:-1]]
            if all(
                abs(p.z_mm - z) > 2 * self.settings["position_tol_mm"]
                for p in self.points
            )
        ]
        if not available:
            raise ValueError("区间内没有足够分离的新位置，请重新选择距离")
        return float(available[0])

    def on_measurement(self, measurement):
        if not self.waiting or not measurement.role.startswith("level-"):
            return
        self.waiting = False
        if measurement.role == "level-near":
            self.height["near"] = measurement
            self.publish("近处完成。移动CCD到远处，保持高度、方向和倍率，再记录中心。")
        elif measurement.role == "level-far":
            self.height["far"] = measurement
            s = self.settings
            n = self.height["near"]
            self.target = height_target(
                s["near_mm"],
                s["far_mm"],
                n.center_y,
                measurement.center_y,
                n.uncertainty_px,
                measurement.uncertainty_px,
                s["pixel_pitch_um"] / s["magnification"],
                s["height_tol_mm"],
                s["position_tol_mm"],
                s["scale_pct"],
            )
            if self.target["y_px"] < 0 or self.target["y_px"] >= self.shape[0]:
                self.stage = "level_outside"
                self.host.overlay([])
                self.publish(
                    "目标高度超出画面。先粗调单镜或重新选择测量位置，再重新测量；不能按原近处位置代替目标。"
                )
            else:
                self.stage = "level_guide"
                self.arrivals = 0
                self.reached = False
                self.publish(
                    f"相机留在远处。计算目标 y={self.target['y_px']:.2f} px。只调单镜竖直指向，使当前几何中心进入水平目标带；横向位置不作为本次约束。辅助目标带不代替最终近远复测。"
                )
        elif measurement.role == "level-far-new":
            self.height["far_new"] = measurement
            self.stage = "level_return"
            self.host.overlay([])
            self.publish(
                "远处新中心已记录。停止调镜，将CCD返回近处并记录；近处应到计算后的共同高度，而非原近处中心。"
            )
        elif measurement.role == "level-near-new":
            self.height["near_new"] = measurement
            s = self.settings
            f = self.height["far_new"]
            self.height_result = verify_height(
                s["near_mm"],
                s["far_mm"],
                measurement.center_y,
                f.center_y,
                measurement.uncertainty_px,
                f.uncertainty_px,
                s["pixel_pitch_um"] / s["magnification"],
                s["height_limit_mm"],
                s["height_tol_mm"],
                s["position_tol_mm"],
                s["scale_pct"],
            )
            self.stage = "level_done"
            result = self.height_result
            self.publish(
                (
                    "等高通过（按输入误差预算）"
                    if result["passed"]
                    else "等高未达标／精度不足，请重新测量调整"
                )
                + f"\n近远高差 {result['difference_mm']:.4f} mm；绝对高差95%上限 {result['absolute_difference_95_mm']:.4f} mm / 门槛 {result['limit_mm']:.4f} mm。"
            )

    def on_frame(self, frame, fit):
        self.latest = max(self.latest, frame.frame_id)
        self.shape = frame.shape
        if self.stage == "setup":
            if frame.pixel_pitch_um and frame.pixel_pitch_um != self.pitch:
                self.pitch = frame.pixel_pitch_um
                self.settings["pixel_pitch_um"] = float(self.pitch)
                self.publish(
                    "已读取有效像元。默认只开启光高；填写反射点到两处感光面的距离，保持CCD同高同方向。"
                )
            return
        if frame.frame_id <= getattr(self, "minimum_frame", -1):
            return
        signature = (
            frame.coordinate_key,
            frame.shape,
            frame.pixel_pitch_um,
            tuple(frame.metadata.get("selection") or ()),
        )
        if self.signature is None:
            self.signature = signature
        elif self.signature != signature:
            self.clear()
            self.signature = None
            self.pitch = frame.pixel_pitch_um
            self.publish("来源、有效像元、坐标或选区变化，旧测量已清空。")
            return
        ready = (
            frame.exposure_verified
            and frame.saturation_verified
            and frame.metadata.get("measurement_stable")
            and frame.metadata.get("quantitative_ready")
        )
        if self.stage == "focus":
            if ready and frame.received_ns - self.feedback_ns >= 500_000_000:
                self.feedback_ns = frame.received_ns
                try:
                    width, _ = measure_width(frame, frame.metadata.get("selection"))
                    self.live_width_um = (
                        width
                        * self.settings["pixel_pitch_um"]
                        / self.settings["magnification"]
                    ).tolist()
                    self.publish(
                        "手动小步调焦；每次调过请清空旧状态。不要只追求远处最小光斑；近远等大也不能直接通过准直。"
                    )
                except ValueError as error:
                    self.live_width_um = None
                    if self.live_reason != str(error):
                        self.live_reason = str(error)
                        self.publish("调焦反馈暂停：" + str(error))
            return
        if self.stage == "level_guide":
            self.minimum_frame = frame.frame_id
            s = self.settings
            half = (
                s["height_limit_mm"]
                * 1000
                * s["far_mm"]
                / self.span
                / (s["pixel_pitch_um"] / s["magnification"])
            )
            self.host.overlay(
                [
                    {
                        "kind": "horizontal_line",
                        "y": self.target["y_px"],
                        "half_height": half,
                        "label": "单镜等高目标（只约束光高）",
                    }
                ],
                frame.frame_id,
            )
            hit = bool(
                ready
                and fit
                and fit.valid
                and abs(fit.center_y - self.target["y_px"]) <= half
            )
            self.arrivals = self.arrivals + 1 if hit else 0
            reached = self.arrivals >= 3
            if reached != self.reached:
                self.reached = reached
                self.publish(
                    "远处目标已到达：请停止调镜并记录，再返回近处校验。"
                    if reached
                    else "中心尚未稳定进入目标带，继续调整单镜竖直指向。"
                )
            return
        if not self.waiting or self.stage not in (
            "coarse_capture",
            "collimation_capture",
        ):
            return
        if frame.pixel_pitch_um and not np.isclose(
            frame.pixel_pitch_um, self.settings["pixel_pitch_um"], rtol=0.005
        ):
            self.waiting = False
            self.publish("帧有效像元与设置不符，本点未录入。")
            return
        if frame.metadata.get("simulated") and not np.isclose(
            frame.metadata.get("level_position_mm", -1),
            self.settings["near_mm"] + self.pending_z,
        ):
            return
        if not ready:
            self.widths = []
            self.identities = []
            return
        if self.epoch != frame.exposure_epoch or self.exposure_us != frame.exposure_us:
            self.widths = []
            self.identities = []
            self.epoch = frame.exposure_epoch
            self.exposure_us = frame.exposure_us
        try:
            width, _ = measure_width(frame, frame.metadata.get("selection"))
        except ValueError as error:
            self.widths = []
            self.identities = []
            if self.live_reason != str(error):
                self.live_reason = str(error)
                self.publish("本点暂停：" + str(error))
            return
        self.live_reason = ""
        self.minimum_frame = frame.frame_id
        self.widths.append(width)
        self.identities.append(frame.frame_id)
        if len(self.widths) < 20:
            if len(self.widths) % 5 == 0:
                self.publish(f"记录中：{len(self.widths)}/20")
            return
        self.waiting = False
        try:
            s = self.settings
            point = summarize(
                self.pending_z,
                s["position_tol_mm"],
                self.widths,
                self.identities,
                s["pixel_pitch_um"],
                s["magnification"],
            )
            if self.stage == "coarse_capture":
                self.coarse.append(point)
                if len(self.coarse) == 1:
                    self.prepare_width(self.span, True)
                else:
                    self.stage = "focus"
                    self.publish(
                        "近远粗查完成，仅供调焦参考，不能排除中间聚焦。手动小步调coupler；每次调过请点击清空本轮。完成后开始五点确认。"
                    )
                return
            self.points.append(point)
            if len(self.points) < 3:
                self.prepare_width(
                    self.span if len(self.points) == 1 else self.span * 0.5
                )
                return
            if self.frozen is None:
                self.model = fit_interval(
                    self.points, s["wavelength_nm"], s["scale_pct"]
                )
                self.frozen = self.model
                self.base_count = len(self.points)
                self.stage = "collimation_initial"
                self.audit.append(
                    {
                        "event": "three_points_frozen",
                        "base_count": self.base_count,
                        "generation": self.generation,
                    }
                )
                self.publish(
                    f"{self.base_count}点初算完成，尚未验证。冻结预测后，在两个未测过的新位置验证；不要求精确锁定束腰。"
                )
            elif len(self.points) == self.base_count + 1:
                self.prepare_width(self.new_position(self.span * 0.75))
            else:
                self.joint = validate_pair(
                    self.frozen, self.points[self.base_count : self.base_count + 2]
                )
                self.model = fit_interval(
                    self.points, s["wavelength_nm"], s["scale_pct"]
                )
                self.collimation_result = self.model.variation(s["diameter_pct"])
                self.stage = "collimation_final"
                self.audit.append(
                    {
                        "event": "held_out_pair",
                        "passed": self.joint["passed"],
                        "generation": self.generation,
                    }
                )
                passed = self.joint["passed"] and self.collimation_result["passed"]
                self.publish(
                    "五点独立验证通过，指定区间准直达标。"
                    if passed
                    else "验证不一致，需检查／重新调焦并重新确认。"
                    if not self.joint["passed"]
                    else "准直变化未达标或不确定度过大：继续调焦或补测。"
                )
        except (ValueError, np.linalg.LinAlgError) as error:
            self.collimation_result = None
            self.stage = "collimation_final"
            self.publish("测量信息不足：" + str(error) + "；可补测或重新调焦。")

"""Interactive, process-isolated second-moment waist measurement."""
from __future__ import annotations

import numpy as np
from beamic_sdk import Plugin
from waist_math import (
    curve_payload,
    fit_caustic,
    measure_width,
    predict,
    suggested_positions,
    summarize,
    validate_point,
)


class WaistPlugin(Plugin):
    def on_activate(self):
        self.settings = {"tolerance_mm": 10., "diameter_pct": 5., "position_mm": 5.,
                         "pixel_pitch_um": 3.45, "magnification": 1., "scale_pct": 2.,
                         "absolute_enabled": False, "front_distance_mm": 0., "sensor_offset_mm": 4.5}
        self.points = []
        self.frozen = self.result = None
        self.base_count = 3
        self.stage = "setup"
        self.key = None
        self.waiting = False
        self.latest_frame_id = -1
        self.latest_pitch = None
        self.context_frame_id = -1
        self.audit = []
        self.live_reason = ""
        self.next_z = 0.
        self.publish("首次感光面设为0；正值沿光束传播方向。移动同一相机，保持方向、倍率、滤片和选区一致。")

    @staticmethod
    def field(key, label, value, minimum, maximum, suffix="", decimals=2):
        return {"id": key, "label": label, "type": "number", "value": value,
                "minimum": minimum, "maximum": maximum, "decimals": decimals, "suffix": suffix}

    def fields(self):
        s = self.settings
        if self.stage == "setup":
            return [self.field("tolerance_mm", "默认读数公差 ±", s["tolerance_mm"], 0, 1000, " mm"),
                    self.field("pixel_pitch_um", "有效像素尺寸", s["pixel_pitch_um"], .01, 1000, " μm"),
                    self.field("magnification", "成像倍率（直测为1）", s["magnification"], .01, 1000, " ×", 3),
                    self.field("scale_pct", "共同宽度系统公差 ±", s["scale_pct"], 0, 20, " %"),
                    self.field("diameter_pct", "直径区间相对半宽 ≤", s["diameter_pct"], .1, 50, " %"),
                    self.field("position_mm", "位置区间半宽 ≤", s["position_mm"], .01, 1000, " mm"),
                    {"id": "absolute_enabled", "label": "同时显示相对外部参考点的位置", "type": "boolean", "value": s["absolute_enabled"]},
                    self.field("front_distance_mm", "参考点 → 首次本体前端面", s["front_distance_mm"], 0, 100000, " mm"),
                    self.field("sensor_offset_mm", "前端面 → 感光面偏移", s["sensor_offset_mm"], 0, 100, " mm")]
        if self.waiting:
            return []
        if self.stage == "capture":
            number = self.slot+1
            label = "第1点 l1=0（零点）" if self.slot == 0 else f"第{number}点 l{number}（相对首点）"
            field = self.field("z_mm", label, self.next_z, -100000, 100000, " mm")
            field["readonly"] = self.slot == 0
            return [field, self.field("tolerance_mm", "本点读数公差 ±", self.next_tolerance, 0, 1000, " mm")]
        if self.points:
            return [self.field("redo_index", "需要重录的点编号", len(self.points), 1, len(self.points), decimals=0)]
        return []

    def actions(self):
        if self.stage == "setup":
            return [{"id": "configure", "label": "确认设置，开始第1点"}]
        if self.waiting:
            return [{"id": "cancel", "label": "取消本点记录"}, {"id": "reset", "label": "重新开始"}]
        if self.stage == "capture":
            actions = [{"id": "preview", "label": "查看此位置的预测范围"}] if self.frozen is not None else []
            return actions+[{"id": "record", "label": f"记录第{self.slot+1}点 · 20个有效帧"},
                    {"id": "reset", "label": "重新开始"}]
        if self.stage == "initial":
            return [{"id": "continue", "label": "保留初算预测，开始独立验证"},
                    {"id": "redo", "label": "重录所选点"}, {"id": "reset", "label": "重新开始"}]
        actions = [{"id": "add", "label": f"补测第{len(self.points)+1}点"}]
        if self.result is not None and self.result.goodness:
            actions.append({"id": "revalidate", "label": "以当前数据重新预测，再测两个新点"})
        return actions+[{"id": "redo", "label": "重录所选点"}, {"id": "reset", "label": "重新开始"}]

    def outcome(self):
        if self.frozen is None or len(self.points) < self.base_count+2:
            return "尚未完成两个独立验证点"
        checks = [validate_point(self.frozen, p) for p in self.points[self.base_count:self.base_count+2]]
        if not all(c["passed"] for c in checks):
            return "验证不一致"
        if self.result is None or not self.result.goodness:
            return "传播模型不一致，需要检查／补测"
        if np.min(self.result.parameters[:, 1]) < 10*self.settings["pixel_pitch_um"]/self.settings["magnification"]:
            return "预测束腰采样不足，需要放大成像后重新测量"
        if not self.result.precision(self.settings["diameter_pct"], self.settings["position_mm"]):
            return "预测一致，需要补测"
        return "验证通过，精度达标"

    def summary(self, result):
        lines = []
        for axis, label in enumerate(("X方向", "Y方向")):
            value = result.axis(axis)
            dlo, dhi = value["diameter_interval_um"]
            zlo, zhi = value["z_interval_mm"]
            lines.append(f"{label}：最小直径 {value['diameter_um']:.2f} μm [{dlo:.2f}, {dhi:.2f}]；"
                         f"位置 {value['z_mm']:.2f} mm [{zlo:.2f}, {zhi:.2f}]")
        lines.append("以上为基于输入公差的95%不确定区间；两个固定相机方向分别拟合，允许像散。")
        if self.settings["absolute_enabled"]:
            # The first reading error is shared with the relative-position draws.
            offset = self.settings["front_distance_mm"]+self.settings["sensor_offset_mm"]
            sensor = np.random.default_rng(21).uniform(-.11, .11, len(result.draws))
            absolute = result.draws[:, :, 0]+result.datum_errors[:, None]+offset+sensor[:, None]
            for axis, label in enumerate(("X", "Y")):
                lo, hi = np.quantile(absolute[:, axis], [.025, .975])
                lines.append(f"{label}绝对位置：{result.parameters[axis, 0]+offset:.2f} mm [{lo:.2f}, {hi:.2f}]"
                             "（含SP932U感光面±0.11 mm；外部参考点自身需固定）")
        return "\n".join(lines)

    def publish(self, message):
        table = {"headers": ["点", "位置 mm", "± mm", "X直径 μm", "Y直径 μm", "用途"], "rows": []}
        for i, p in enumerate(self.points):
            role = "建模"
            if self.frozen is not None and self.base_count <= i < self.base_count+2:
                role = "验证一致" if validate_point(self.frozen, p)["passed"] else "验证不一致"
            elif self.frozen is not None and i >= self.base_count+2:
                role = "补测"
            table["rows"].append([str(i+1), f"{p.z_mm:.1f}", f"{p.tolerance_mm:.1f}",
                                  f"{p.diameters_um[0]:.2f}", f"{p.diameters_um[1]:.2f}", role])
        result = self.result or self.frozen
        text = message
        if result:
            text += "\n\n"+self.summary(result)
        text += "\n\n二阶矩直径 D4σ，与显示圈独立。±读数公差按均匀分布；零点误差共享；宽度系统公差为共同误差假设。"
        if self.stage == "setup":
            text += "\n像素尺寸默认SP932U原始像元3.45 μm；合并、缩放需填有效值。共同宽度公差默认±2%为保守预算，不是像元制造公差。"
        if self.settings["absolute_enabled"]:
            text += "\n绝对位置基准是本体前端平面，不能使用滤片外缘。"
        data = {"stage": self.stage, "waiting": self.waiting, "count": len(self.points),
                "base_count": self.base_count, "outcome": self.outcome(), "samples": len(getattr(self, 'widths', [])),
                "settings": self.settings, "audit": self.audit[-12:], "live_reason": self.live_reason,
                "axes": [result.axis(i) for i in range(2)] if result else [],
                "valid_fraction": result.valid_fraction if result else None}
        self.host.panel("束腰测量 · "+("设置" if self.stage == "setup" else self.outcome() if self.stage == "final" else message.split('\n')[0]),
                        text, self.actions(), fields=self.fields(), table=table,
                        plot=curve_payload(result, self.points) if result else {}, data=data)

    def prepare(self, slot=None, z=None):
        self.slot = len(self.points) if slot is None else slot
        self.next_z = float(z if z is not None else (0 if not self.points else self.points[-1].z_mm+150))
        self.next_tolerance = self.settings["tolerance_mm"]
        self.stage = "capture"
        self.waiting = False
        kind = "独立验证" if self.frozen is not None and self.base_count <= self.slot < self.base_count+2 else "宽度测量"
        self.publish(f"第{self.slot+1}点 · {kind}\n将相机移动后填写实际相对位置。理论建议需确认位于透镜之后、可摆放且光斑完整。")

    def on_action(self, action, values):
        try:
            if action == "configure" and self.stage == "setup":
                s = dict(self.settings)
                for key in s:
                    if key in values:
                        s[key] = bool(values[key]) if key == "absolute_enabled" else float(values[key])
                if not all(np.isfinite(v) for k, v in s.items() if k != "absolute_enabled") or min(s["pixel_pitch_um"], s["magnification"], s["diameter_pct"], s["position_mm"]) <= 0 or s["tolerance_mm"] < 0 or not 0 <= s["scale_pct"] <= 20:
                    raise ValueError("测量设置无效")
                self.settings = s
                self.key = None
                self.latest_pitch = None
                self.context_frame_id = int(values.get("min_frame_id", self.latest_frame_id))
                self.host.measurement_mode()
                self.prepare(0, 0)
            elif action == "record" and self.stage == "capture" and not self.waiting:
                z = float(values.get("z_mm", self.next_z))
                tolerance = float(values.get("tolerance_mm", self.settings["tolerance_mm"]))
                if not np.isfinite([z, tolerance]).all() or tolerance < 0 or self.slot == 0 and z != 0:
                    raise ValueError("距离或公差无效，首次位置必须为0")
                if any(i != self.slot and abs(p.z_mm-z) < 1e-6 for i, p in enumerate(self.points)):
                    raise ValueError("这个位置已经记录，请选择不同位置")
                if self.latest_pitch is not None and not np.isclose(self.latest_pitch, self.settings["pixel_pitch_um"], rtol=.005):
                    raise ValueError(f"帧报告像素尺寸{self.latest_pitch:g} μm，与设置不符；请重新开始并核对有效像元")
                self.pending_z, self.pending_tolerance = z, tolerance
                self.widths, self.identities = [], []
                self.min_frame_id = max(self.latest_frame_id, int(values.get("min_frame_id", -1)))
                self.epoch = self.exposure_us = None
                self.waiting = True
                self.publish(f"第{self.slot+1}点记录中 · 0/20\n等待未饱和、稳定曝光的新帧。")
            elif action == "preview" and self.frozen is not None and not self.waiting:
                z = float(values.get("z_mm", self.next_z))
                tolerance = float(values.get("tolerance_mm", self.next_tolerance))
                if not np.isfinite([z, tolerance]).all() or tolerance < 0:
                    raise ValueError("预测位置或公差无效")
                self.next_z, self.next_tolerance = z, tolerance
                ranges = np.quantile(predict(self.frozen, z, tolerance), [.025, .975], axis=0)
                self.publish(f"第{self.slot+1}点 · 预测范围\n位置 {z:g} ±{tolerance:g} mm："
                             f"X直径 [{ranges[0, 0]:.2f}, {ranges[1, 0]:.2f}] μm；"
                             f"Y直径 [{ranges[0, 1]:.2f}, {ranges[1, 1]:.2f}] μm。\n"
                             "以上含位置与模型不确定度；正式联合检验还会加入新测量误差。")
            elif action == "cancel":
                self.waiting = False
                self.prepare(self.slot, self.pending_z)
            elif action == "continue" and self.frozen is not None:
                self.prepare(z=suggested_positions(self.frozen, self.points)[0])
            elif action == "add" and not self.waiting:
                if self.result:
                    candidates = suggested_positions(self.result, self.points)
                    z = candidates[0] if not any(abs(p.z_mm-candidates[0]) < max(1, p.tolerance_mm) for p in self.points) else candidates[1]
                else:
                    z = max(p.z_mm for p in self.points)+150
                self.prepare(z=z)
            elif action == "revalidate" and self.result is not None and self.result.goodness:
                self.frozen = self.result
                self.base_count = len(self.points)
                self.audit.append({"event": "new_prediction", "base_count": self.base_count})
                self.prepare(z=suggested_positions(self.frozen, self.points)[0])
            elif action == "redo" and self.points:
                index = int(values.get("redo_index", len(self.points)))-1
                if not 0 <= index < len(self.points):
                    raise ValueError("重录点编号无效")
                if index == 0:
                    self.on_activate()
                    self.publish("重录首点会改变零点，所有相对距离需重新记录。请确认设置。")
                else:
                    z = self.points[index].z_mm
                    if index < self.base_count:
                        self.points = self.points[:self.base_count]
                        self.frozen = self.result = None
                        self.audit.append({"event": "prediction_invalidated", "point": index+1})
                    self.prepare(index, z)
            elif action in ("reset", "selection_changed"):
                self.on_activate()
                self.publish("来源、坐标、选区或测量定义改变／重新开始：旧测量已清空，请核对设置。")
        except (ValueError, TypeError, OverflowError) as error:
            self.publish("输入未采用："+str(error))

    def on_frame(self, frame, fit):
        self.latest_frame_id = max(self.latest_frame_id, frame.frame_id)
        if self.stage == "setup" or frame.frame_id <= self.context_frame_id:
            return
        self.latest_pitch = frame.pixel_pitch_um
        signature = (frame.coordinate_key, frame.shape, frame.pixel_pitch_um,
                     tuple(frame.metadata.get("selection") or ()))
        if self.key is None:
            self.key = signature
        elif signature != self.key:
            self.on_activate()
            self.key = signature
            self.latest_frame_id = frame.frame_id
            self.latest_pitch = frame.pixel_pitch_um
            self.publish("采集坐标、有效像元或积分选区改变，测量已清空。")
            return
        if not self.waiting or frame.frame_id <= self.min_frame_id:
            return
        if frame.pixel_pitch_um is not None and not np.isclose(frame.pixel_pitch_um, self.settings["pixel_pitch_um"], rtol=.005):
            self.waiting = False
            self.publish(f"本点未录入：帧像素尺寸{frame.pixel_pitch_um:g} μm与设置不符，请重新开始并核对有效像元。")
            return
        if frame.metadata.get("simulated") and "waist_position_mm" in frame.metadata and not np.isclose(frame.metadata["waist_position_mm"], self.pending_z):
            return
        ready = frame.exposure_verified and frame.saturation_verified and frame.metadata.get("measurement_stable") and frame.metadata.get("quantitative_ready")
        if not ready:
            if self.widths:
                self.widths, self.identities = [], []
                self.publish(f"第{self.slot+1}点记录暂停：等待有效的新曝光帧，累计已重新开始。")
            return
        if self.epoch != frame.exposure_epoch or self.exposure_us != frame.exposure_us:
            self.widths, self.identities = [], []
            self.epoch, self.exposure_us = frame.exposure_epoch, frame.exposure_us
        try:
            width, _ = measure_width(frame, frame.metadata.get("selection"))
        except ValueError as error:
            self.widths, self.identities = [], []
            reason = str(error)
            if reason != self.live_reason:
                self.live_reason = reason
                self.publish(f"第{self.slot+1}点暂停：{reason}")
            return
        self.live_reason = ""
        self.min_frame_id = frame.frame_id
        self.widths.append(width)
        self.identities.append(frame.frame_id)
        if len(self.widths) < 20:
            if len(self.widths) % 5 == 0:
                self.publish(f"第{self.slot+1}点记录中 · {len(self.widths)}/20")
            return
        self.waiting = False
        try:
            point = summarize(self.pending_z, self.pending_tolerance, self.widths, self.identities,
                              self.settings["pixel_pitch_um"], self.settings["magnification"])
        except ValueError as error:
            self.publish("本点未录入："+str(error))
            return
        if self.slot < len(self.points):
            self.points[self.slot] = point
        else:
            self.points.append(point)
        self.after_point()

    def after_point(self):
        if self.frozen is None:
            if len(self.points) < 3:
                self.prepare(z=150 if len(self.points) == 1 else 350)
                return
            try:
                self.result = fit_caustic(self.points, self.settings["scale_pct"])
                self.frozen = self.result
                self.base_count = len(self.points)
                self.stage = "initial"
                self.audit.append({"event": "initial_prediction", "base_count": self.base_count})
                fourth, fifth = suggested_positions(self.result, self.points)
                self.publish(f"{len(self.points)}点初算完成 · 尚未验证\n建议下一点靠近 {fourth:g} mm，再取扩张区域约 {fifth:g} mm。距离为理论建议，填写实际读数。")
            except ValueError as error:
                self.stage = "final"
                self.result = None
                self.publish("初测不足："+str(error)+"\n可以增加建模点；得到可靠预测后仍需两个新的独立验证点。")
            return
        if len(self.points) < self.base_count+2:
            checked = validate_point(self.frozen, self.points[-1])
            self.audit.append({"event": "held_out_point", "point": len(self.points), **checked})
            self.prepare(z=suggested_positions(self.frozen, self.points)[1])
            self.publish(f"第{len(self.points)}点独立检查：{'一致' if checked['passed'] else '不一致'}\n请再记录一个新位置，完成两点联合验证。")
            return
        checked = [validate_point(self.frozen, p) for p in self.points[self.base_count:self.base_count+2]]
        self.audit.append({"event": "held_out_pair", "base_count": self.base_count,
                           "passed": all(p["passed"] for p in checked)})
        self.stage = "final"
        try:
            self.result = fit_caustic(self.points, self.settings["scale_pct"])
            self.publish(f"{len(self.points)}点重算完成 · {self.outcome()}\n原预测独立验证记录保留；同一位置重复拍帧不能消除摆放公差。")
        except ValueError as error:
            self.result = None
            self.publish("最终拟合暂不可用："+str(error)+"\n可重录异常点或增加测点。")

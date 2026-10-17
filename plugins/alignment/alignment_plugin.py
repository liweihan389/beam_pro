import numpy as np
from beamic_sdk import Plugin
from beamic_sdk.alignment import Geometry, compute_targets, verify


class AlignmentPlugin(Plugin):
    labels = {"near_red": "近处红光（遮蓝光）", "near_blue": "近处蓝光（遮红光）",
              "far_red": "远处红光（遮蓝光）", "far_blue": "远处蓝光（遮红光）"}

    def on_activate(self):
        self.geometry = None
        self.measurements = {}
        self.stage = "setup"
        self.index = 0
        self.round = 0
        self.targets = None
        self.shape = None
        self.near_target_reached = False
        self.far_target_reached = False
        self.waiting = False
        self.arrivals = 0
        self.host.panel("双镜合束 · DM固定", "DM和红光光路保持固定。只调蓝光的两面反射镜：M1远镜 → M2近镜 → DM。先输入四个沿光路累计的距离（毫米），再点击开始。", [])

    def actions(self):
        if self.stage in ("capture", "verify") and not self.waiting:
            return [{"id": "record", "label": "记录：" + self.labels[self.roles[self.index]]},
                    {"id": "reset", "label": "重新开始"}]
        if self.stage == "near_mirror":
            return ([{"id": "confirm_near", "label": "M2已到目标，进入M1调节"}] if self.near_target_reached else []) + [{"id": "reset", "label": "重新测量"}]
        if self.stage == "far_mirror":
            return ([{"id": "confirm_far", "label": "M1已到目标，开始近远复测"}] if self.far_target_reached else []) + [{"id": "reset", "label": "重新测量"}]
        return [{"id": "reset", "label": "重新开始"}]

    def show(self, text):
        self.host.panel(f"双镜合束 · 第{self.round}轮 · DM固定", text+"\nDM和红光光路保持固定。", self.actions())

    def on_action(self, action, values):
        if action == "configure":
            self.geometry = Geometry(float(values["z_near"]), float(values["z_far"]),
                                     float(values["mirror_near"]), float(values["mirror_far"]))
            self.round = 1
            self.begin_capture()
        elif action == "reset":
            if self.geometry is not None:
                self.round += 1
                self.begin_capture()
        elif action == "record" and self.stage in ("capture", "verify") and not self.waiting:
            self.waiting = True
            role = self.roles[self.index]
            red = self.measurements.get(role.replace("blue", "red")) if role.endswith("blue") else None
            self.host.request_measurement(role, reference_radius=red.radius_minor if red else None)
            self.show("等待曝光稳定并记录20个有效帧：" + self.labels[role])
        elif action == "confirm_near" and self.stage == "near_mirror" and self.near_target_reached:
            self.stage = "far_mirror"
            self.arrivals = 0
            self.show("相机保持在远处，红光保持遮挡。只调M1远反射镜（蓝光），把蓝光移到红光参考中心；M2保持不动。")
        elif action == "confirm_far" and self.stage == "far_mirror" and self.far_target_reached:
            self.stage = "verify"
            self.index = 0
            self.measurements = {}
            self.host.overlay([])
            self.show("现在将相机移回近处，保持方向和倍率，分别复测红蓝光。")
        elif action == "selection_changed":
            self.waiting = False
            self.arrivals = 0
            self.near_target_reached = self.far_target_reached = False
            if self.stage in ("capture", "verify"):
                self.show("主光斑选区已改变；已有测量保留，请重新记录当前项：" + self.labels[self.roles[self.index]])
            elif self.stage in ("near_mirror", "far_mirror"):
                mirror = "M2近反射镜" if self.stage == "near_mirror" else "M1远反射镜"
                self.show(f"主光斑选区已改变，等待新拟合。相机保持在远处，只调{mirror}（蓝光）。")

    def begin_capture(self):
        self.roles = ["near_red", "near_blue", "far_red", "far_blue"]
        self.measurements = {}
        self.index = 0
        self.waiting = False
        self.stage = "capture"
        self.targets = None
        self.near_target_reached = self.far_target_reached = False
        self.arrivals = 0
        self.host.overlay([])
        self.host.measurement_mode()
        self.show("将相机放在近处。保持相机方向、倍率和采集格式一致，先遮蓝光，记录红光。")

    def on_measurement(self, measurement):
        if not self.waiting or measurement.role != self.roles[self.index]:
            return
        self.waiting = False
        reference = next(iter(self.measurements.values())) if self.measurements else None
        if reference and (measurement.coordinate_key, measurement.center_mode, measurement.boundary_fraction, measurement.boundary_kind) != (
                reference.coordinate_key, reference.center_mode, reference.boundary_fraction, reference.boundary_kind):
            self.begin_capture()
            self.show("坐标或定位定义已变化，四次测量已清空，请重新记录。")
            return
        self.measurements[measurement.role] = measurement
        self.index += 1
        if self.index < 4:
            role = self.roles[self.index]
            move = "将相机移到远处，保持方向和倍率。" if self.index == 2 else "保持相机位置不动，切换遮光。"
            self.show(move + "下一项：" + self.labels[role])
            return
        if self.stage == "verify":
            result = verify(self.measurements)
            description = "；".join(f"{'近处' if plane == 'near' else '远处'}误差 {result[plane]['error_px']:.3f} px / 容差 {result[plane]['tolerance_px']:.3f} px" for plane in ("near", "far"))
            if result["passed"]:
                self.stage = "done"
                self.show("合束通过。" + description + "；容差为各处红光所选边界短半轴的5%。")
                return
            self.round += 1
            self.calculate(description + "；尚未通过，用本轮复测数据重新计算。")
        else:
            self.calculate("")

    def calculate(self, prefix):
        self.targets = compute_targets(self.geometry, self.measurements)
        if self.shape and not self.targets.visible(self.shape[1], self.shape[0]):
            self.stage = "outside"
            self.show(prefix + "中间目标超出画面，请先粗调光路再重新测量。")
            return
        blue = self.measurements["far_blue"]
        c, s = np.cos(blue.angle_rad), np.sin(blue.angle_rad)
        extent = 1.6 * np.sqrt(np.array([(blue.radius_major*c)**2 + (blue.radius_minor*s)**2,
                                       (blue.radius_major*s)**2 + (blue.radius_minor*c)**2]))
        if self.shape:
            for target in (self.targets.intermediate, self.targets.final):
                center = np.array(target)
                if np.any(center - extent < 0) or np.any(center + extent > np.array([self.shape[1]-1, self.shape[0]-1])):
                    self.stage = "outside"
                    self.show(prefix + "目标太靠近画面边缘，光斑会被截断。请粗调或横向调整 CCD 位置，再重新测量。")
                    return
        self.stage = "near_mirror"
        self.near_target_reached = self.far_target_reached = False
        x, y = self.targets.intermediate
        self.show(prefix + f"相机留在远处，遮红光。只调M2近反射镜（蓝光），将蓝光移到中间目标 ({x:.2f}, {y:.2f}) px；M1保持不动。")

    def on_frame(self, frame, fit):
        self.shape = frame.shape
        if self.measurements and fit:
            reference = next(iter(self.measurements.values()))
            if (fit.center_mode, fit.boundary_fraction, fit.boundary_kind) != (
                    reference.center_mode, reference.boundary_fraction, reference.boundary_kind):
                self.begin_capture()
                self.show("定位定义已改变，请重新测量。")
                return
        if self.measurements and frame.coordinate_key != next(iter(self.measurements.values())).coordinate_key:
            self.begin_capture()
            self.show("采集坐标已改变，请重新测量。")
            return
        if self.stage not in ("near_mirror", "far_mirror") or self.targets is None:
            return
        target = self.targets.intermediate if self.stage == "near_mirror" else self.targets.final
        self.host.overlay([{"kind": "target", "x": target[0], "y": target[1], "radius": self.targets.tolerance_px,
                            "label": "M2中间目标" if self.stage == "near_mirror" else "红光参考中心"}], frame.frame_id)
        if fit is None or not fit.valid or not frame.exposure_verified or not frame.saturation_verified or not frame.metadata.get("measurement_stable") or not frame.metadata.get("quantitative_ready"):
            reached = False
        else:
            reached = np.linalg.norm(fit.center - np.array(target)) <= self.targets.tolerance_px
        self.arrivals = self.arrivals + 1 if reached else 0
        reached = self.arrivals >= 3
        attr = "near_target_reached" if self.stage == "near_mirror" else "far_target_reached"
        if reached != getattr(self, attr):
            setattr(self, attr, bool(reached))
            mirror = "M2近反射镜" if self.stage == "near_mirror" else "M1远反射镜"
            self.show(f"{mirror}目标{'已到达，请确认下一步' if reached else '尚未到达，请继续调节'}。相机保持在远处，红光保持遮挡。")


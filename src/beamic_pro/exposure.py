from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from beamic_sdk import Frame


@dataclass
class ExposureState:
    mode: str = "measurement"
    stable: bool = False
    reason: str = "等待相机"
    epoch: int = 0
    stable_frames: int = 0


class ExposureController:
    def __init__(self):
        self.enabled = True
        self.state = ExposureState()
        self.last_id = -1
        self.last_exposure = None
        self.minimum_us, self.maximum_us, self.step_us = 1.0, 1_000_000.0, 1.0
        self.pending_us = None
        self.pending_epoch = None
        self.dark_attempts = 0

    def set_enabled(self, enabled):
        self.enabled = bool(enabled)
        self.dark_attempts = 0
        self.invalidate("等待自动曝光稳定" if self.enabled else "自动曝光已关闭，等待当前曝光的有效帧")

    def set_mode(self, mode):
        if mode not in ("finding", "measurement"):
            raise ValueError("曝光模式无效")
        self.state.mode = mode
        self.dark_attempts = 0
        self.invalidate("等待曝光稳定")

    def reject_request(self, reason):
        self.pending_us = self.pending_epoch = None
        self.invalidate(reason)

    def invalidate(self, reason):
        self.state.stable = False
        self.state.stable_frames = 0
        self.state.reason = reason

    def acknowledge(self, exposure_us, epoch):
        self.pending_us = float(exposure_us)
        self.pending_epoch = int(epoch)
        self.state.epoch = int(epoch)
        self.invalidate("等待新曝光帧")

    def inspect(self, frame: Frame, selection=None):
        """Returns a requested exposure, never directly writes camera settings."""
        if frame.frame_id <= self.last_id:
            return None
        self.last_id = frame.frame_id
        if not frame.exposure_verified or not frame.saturation_verified:
            self.invalidate("接口尚未验证曝光关联或饱和量程")
            return None
        if self.pending_us is not None:
            if frame.exposure_epoch < self.pending_epoch or not math.isclose(frame.exposure_us, self.pending_us, rel_tol=0.005):
                self.invalidate("等待新曝光帧")
                return None
            self.pending_us = None
        if self.last_exposure is not None and not math.isclose(frame.exposure_us, self.last_exposure, rel_tol=0.005):
            self.invalidate("曝光变化，重新稳定")
        self.last_exposure = frame.exposure_us
        self.state.epoch = frame.exposure_epoch
        pixels = frame.pixels
        if selection:
            x0, y0, x1, y1 = selection
            pixels = pixels[max(0, y0):min(pixels.shape[0], y1), max(0, x0):min(pixels.shape[1], x1)]
        if pixels.size == 0:
            self.invalidate("选择区域无效")
            return None
        peak = float(np.max(pixels)) / frame.full_scale
        saturated = float(np.mean(pixels >= frame.full_scale * 0.999))
        stride = max(1, max(pixels.shape) // 128)
        sample = pixels[::stride, ::stride]
        background = float(np.percentile(sample, 30))
        noise = max(.15, 1.4826 * float(np.median(np.abs(sample - np.median(sample)))))
        signal = float(np.max(pixels)) - background
        if signal < max(frame.full_scale * .005, noise * 8):
            self.invalidate("无信号或信号过弱，暂停测量")
            if self.enabled and self.dark_attempts < 4 and frame.exposure_us < self.maximum_us:
                self.dark_attempts += 1
                requested = min(self.maximum_us, max(self.minimum_us, round(frame.exposure_us * 2 / self.step_us) * self.step_us))
                self.pending_us, self.pending_epoch = requested, frame.exposure_epoch + 1
                self.state.reason = "信号过弱，正在限次搜索曝光"
                return requested
            return None
        self.dark_attempts = 0
        if not self.enabled:
            if saturated and (self.state.mode == "measurement" or saturated > .005):
                self.invalidate("自动曝光已关闭；光斑饱和，请在 BeamMic 手动降低曝光")
                return None
            self.state.stable_frames += 1
            self.state.stable = self.state.stable_frames >= 3
            self.state.reason = "自动曝光已关闭 · 当前曝光稳定" if self.state.stable else "自动曝光已关闭 · 等待连续有效帧"
            return None
        good = 0.72 <= peak <= 0.88 and saturated == 0
        if self.state.mode == "finding":
            good = peak >= 0.999 and 0 < saturated <= 0.005
        if good:
            self.state.stable_frames += 1
            self.state.stable = self.state.stable_frames >= 3
            self.state.reason = "曝光稳定" if self.state.stable else "等待连续稳定帧"
            return None
        self.invalidate("正在调整曝光")
        ratio = 0.80 / max(peak, 0.01)
        if self.state.mode == "finding":
            ratio = 1.015 / max(peak, 0.01) if saturated <= 0.005 else 0.90
        requested = frame.exposure_us * min(2.0, max(0.25, ratio))
        requested = round(requested / self.step_us) * self.step_us
        requested = min(self.maximum_us, max(self.minimum_us, requested))
        if math.isclose(requested, frame.exposure_us, rel_tol=0.002):
            self.state.reason = "曝光已到设备限制，请调整光学衰减"
            return None
        self.pending_us = requested
        self.pending_epoch = frame.exposure_epoch + 1
        return requested


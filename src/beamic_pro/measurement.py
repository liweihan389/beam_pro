from __future__ import annotations

import math

import numpy as np
from beamic_sdk import BeamFit, Frame, Measurement


class Capture:
    """Consumes unique, verified, post-exposure frames; keeps only fit summaries."""
    def __init__(self, role: str, samples: int = 20, exposure_epoch: int = 0,
                 coordinate_key: str | None = None, after_frame_id: int = -1, reference_radius: float | None = None):
        if not 2 <= samples <= 1000:
            raise ValueError("记录帧数须在2到1000之间")
        self.role, self.samples, self.epoch = role, samples, exposure_epoch
        if reference_radius is not None and (not math.isfinite(reference_radius) or reference_radius <= 0):
            raise ValueError("参考光斑半径无效")
        self.reference_radius = reference_radius
        self.coordinate_key = coordinate_key
        self.last_id = after_frame_id
        self.fits: list[BeamFit] = []
        self.first_id = None
        self.exposure_us = None
        self.definition = None
        self.reason = "等待曝光稳定后的新帧"

    def add(self, frame: Frame, fit: BeamFit, stable: bool) -> Measurement | None:
        if frame.frame_id <= self.last_id or fit.frame_id != frame.frame_id:
            self.reason = "忽略旧帧或重复帧"
            return None
        self.last_id = frame.frame_id
        if not frame.exposure_verified or not frame.saturation_verified or not stable or frame.exposure_epoch < self.epoch:
            self.fits.clear()
            self.reason = "等待新曝光对应的有效帧"
            return None
        if self.coordinate_key is not None and frame.coordinate_key != self.coordinate_key:
            raise ValueError("采集坐标已改变，请重新开始测量")
        if self.exposure_us is not None and not math.isclose(frame.exposure_us, self.exposure_us, rel_tol=0.005):
            self.fits.clear()
            self.reason = "曝光发生变化，重新累计"
        self.exposure_us = frame.exposure_us
        if not fit.valid:
            self.fits.clear()
            self.reason = fit.reason
            return None
        definition = (fit.center_mode, fit.boundary_fraction, fit.boundary_kind)
        if self.definition is not None and definition != self.definition:
            raise ValueError("中心模式或边界定义已改变，请重新测量")
        self.definition = definition
        self.coordinate_key = frame.coordinate_key
        if not self.fits:
            self.first_id = frame.frame_id
        self.fits.append(fit)
        self.reason = f"正在记录 {len(self.fits)}/{self.samples}"
        if len(self.fits) < self.samples:
            return None
        centers = np.array([f.center for f in self.fits])
        radius = float(np.mean([f.radius_minor for f in self.fits]))
        jitter = float(np.sqrt(np.var(centers, axis=0, ddof=1).sum()))
        # Do not average away large mechanical excursions to claim precise alignment.
        budget_radius = min(radius, self.reference_radius) if self.reference_radius is not None else radius
        if jitter > 0.02 * budget_radius or np.max(np.linalg.norm(centers - centers.mean(axis=0), axis=1)) > 0.05 * budget_radius:
            self.fits.clear()
            self.reason = "光斑抖动超过测量预算，请稳定光路后重新记录"
            return None
        if np.std([f.radius_minor for f in self.fits], ddof=1) > 0.03 * radius:
            self.fits.clear()
            self.reason = "光斑尺寸不稳定，请重新记录"
            return None
        mean = centers.mean(axis=0)
        return Measurement(self.role, float(mean[0]), float(mean[1]), radius,
                           float(np.mean([f.radius_major for f in self.fits])),
                           self.fits[-1].angle_rad, self.samples, jitter,
                           jitter / math.sqrt(self.samples), frame.coordinate_key,
                           frame.exposure_us, self.first_id, frame.frame_id,
                           center_mode=fit.center_mode, boundary_fraction=fit.boundary_fraction,
                           boundary_kind=fit.boundary_kind)


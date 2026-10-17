"""Two-plane alignment: blue M1 (far) -> M2 (near) -> fixed DM -> CCD.

Only the two blue steering mirrors supply angular corrections. The DM and
the red reference path are fixed; distance parameters do not control DM pose.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .types import Measurement


@dataclass(frozen=True)
class Geometry:
    z_near: float
    z_far: float
    mirror_near: float
    mirror_far: float

    def __post_init__(self):
        values = (self.z_near, self.z_far, self.mirror_near, self.mirror_far)
        if not all(math.isfinite(x) for x in values):
            raise ValueError("距离必须是有限数值")
        if not 0 <= self.z_near < self.z_far:
            raise ValueError("测量距离须满足 0 ≤ 近处 < 远处")
        if not 0 < self.mirror_near < self.mirror_far:
            raise ValueError("镜距须满足 0 < 近镜 < 远镜（沿传播路径累计）")


@dataclass(frozen=True)
class Targets:
    intermediate: tuple[float, float]
    final: tuple[float, float]
    error_at_dm: tuple[float, float]
    slope_px_per_mm: tuple[float, float]
    tolerance_px: float

    def visible(self, width: int, height: int):
        x, y = self.intermediate
        return 0 <= x < width and 0 <= y < height


def compute_targets(geometry: Geometry, measurements: dict[str, Measurement]) -> Targets:
    roles = ("near_red", "near_blue", "far_red", "far_blue")
    if any(role not in measurements for role in roles):
        raise ValueError("需要近处红/蓝和远处红/蓝四次测量")
    points = [measurements[r] for r in roles]
    if len({p.coordinate_key for p in points}) != 1:
        raise ValueError("采集坐标、方向或倍率已改变，请重新测量")
    if len({(p.center_mode, p.boundary_fraction, p.boundary_kind) for p in points}) != 1:
        raise ValueError("中心模式或边界定义不一致，请重新测量")
    if any(not np.isfinite(p.center).all() or p.radius_minor <= 0 for p in points):
        raise ValueError("光斑测量无效")
    nr, nb, fr, fb = points
    e1, e2 = nb.center - nr.center, fb.center - fr.center
    slope = (e2 - e1) / (geometry.z_far - geometry.z_near)
    e0 = e1 - geometry.z_near * slope
    mid = fr.center + (geometry.z_far + geometry.mirror_far) / (
        geometry.mirror_far - geometry.mirror_near
    ) * (e0 - geometry.mirror_near * slope)
    if not np.isfinite(mid).all():
        raise ValueError("目标无法计算，请检查距离")
    return Targets(tuple(mid), tuple(fr.center), tuple(e0), tuple(slope), 0.05 * fr.radius_minor)


def verify(measurements: dict[str, Measurement]) -> dict:
    if len({(p.center_mode, p.boundary_fraction, p.boundary_kind, p.coordinate_key)
            for p in measurements.values()}) != 1:
        raise ValueError("验收测量的坐标、中心或边界定义不一致")
    result = {}
    for plane in ("near", "far"):
        red, blue = measurements[f"{plane}_red"], measurements[f"{plane}_blue"]
        if red.coordinate_key != blue.coordinate_key:
            raise ValueError("红蓝测量坐标不一致")
        if red.radius_minor <= 0 or not np.isfinite(red.center).all() or not np.isfinite(blue.center).all():
            raise ValueError("验收测量无效")
        error = float(np.linalg.norm(blue.center - red.center))
        tolerance = 0.05 * red.radius_minor
        result[plane] = {"error_px": error, "tolerance_px": tolerance, "passed": error <= tolerance}
    result["passed"] = result["near"]["passed"] and result["far"]["passed"]
    return result


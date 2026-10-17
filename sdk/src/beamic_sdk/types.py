from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, field

import numpy as np


@dataclass(frozen=True)
class Frame:
    frame_id: int
    pixels: np.ndarray
    received_ns: int
    exposure_us: float
    exposure_epoch: int
    coordinate_key: str
    full_scale: float
    pixel_pitch_um: float | None = None
    timestamp_source: str = "host_monotonic"
    exposure_verified: bool = False
    saturation_verified: bool = False
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.pixels.ndim != 2 or not np.isfinite(self.pixels).all():
            raise ValueError("图像必须是有限的二维数值数组")
        if self.full_scale <= 0 or not math.isfinite(self.full_scale):
            raise ValueError("缺少有效的饱和量程")
        if not math.isfinite(self.exposure_us) or self.exposure_us <= 0:
            raise ValueError("曝光值无效")
        self.pixels.flags.writeable = False

    @property
    def shape(self):
        return self.pixels.shape

    def description(self):
        return {k: v for k, v in vars(self).items() if k != "pixels"}


@dataclass(frozen=True)
class BeamFit:
    frame_id: int
    center_x: float = 0.0
    center_y: float = 0.0
    radius_major: float = 0.0
    radius_minor: float = 0.0
    angle_rad: float = 0.0
    amplitude: float = 0.0
    background: float = 0.0
    residual_fraction: float = math.inf
    saturated_fraction: float = 0.0
    valid: bool = False
    reason: str = "尚未拟合"
    roi: tuple[int, int, int, int] | None = None
    center_mode: str = "boundary"
    boundary_fraction: float = .135
    boundary_kind: str = "relative_intensity"
    boundary_center_x: float | None = None
    boundary_center_y: float | None = None
    area_center_x: float | None = None
    area_center_y: float | None = None
    shape_kind: str = "ellipse"
    boundary_coverage: float = 0.0
    defect_level: str = "none"
    warnings: tuple[str, ...] = ()
    background_origin: float | None = None
    background_slope_x: float = 0.0
    background_slope_y: float = 0.0

    @property
    def boundary_center(self):
        return np.array([self.center_x if self.boundary_center_x is None else self.boundary_center_x,
                         self.center_y if self.boundary_center_y is None else self.boundary_center_y])

    @property
    def center(self):
        return np.array([self.center_x, self.center_y], dtype=float)

    def to_dict(self):
        value = asdict(self)
        # Strict JSON must not contain Infinity/NaN, including failed fits.
        for key, item in value.items():
            if isinstance(item, float) and not math.isfinite(item):
                value[key] = None
        return value

    @classmethod
    def from_dict(cls, value):
        data = dict(value)
        if data.get("residual_fraction") is None:
            data["residual_fraction"] = math.inf
        if data.get("roi") is not None:
            data["roi"] = tuple(data["roi"])
        if "warnings" in data:
            data["warnings"] = tuple(data["warnings"])
        return cls(**data)


@dataclass(frozen=True)
class Measurement:
    role: str
    center_x: float
    center_y: float
    radius_minor: float
    radius_major: float
    angle_rad: float
    sample_count: int
    center_std_px: float
    uncertainty_px: float
    coordinate_key: str
    exposure_us: float
    first_frame_id: int
    last_frame_id: int
    captured_ns: int = field(default_factory=time.monotonic_ns)
    center_mode: str = "boundary"
    boundary_fraction: float = .135
    boundary_kind: str = "relative_intensity"

    @property
    def center(self):
        return np.array([self.center_x, self.center_y], dtype=float)

    def to_dict(self):
        return asdict(self)


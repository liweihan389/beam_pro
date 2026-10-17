"""Display-only raster. Unknown exposure and saturation remain unknown.

This type is deliberately not a plugin SDK Frame and cannot be published as a
quantitative measurement. The display range is computed by the view only.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class PreviewFrame:
    frame_id: int
    pixels: np.ndarray
    received_ns: int
    coordinate_key: str
    exposure_us: None = None
    full_scale: None = None
    exposure_epoch: int = 0
    pixel_pitch_um: None = None
    timestamp_source: str = "host_receive_only"
    exposure_verified: bool = False
    saturation_verified: bool = False
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.pixels.ndim != 2 or self.pixels.size == 0 or not np.isfinite(self.pixels).all():
            raise ValueError("预览图像必须是有限且非空的二维数值数组")
        if self.exposure_us is not None or self.full_scale is not None or self.pixel_pitch_um is not None:
            raise ValueError("预览帧不能携带未经验证的曝光、量程或像素标定")
        if self.exposure_verified or self.saturation_verified:
            raise ValueError("预览帧不能标记为已验证")
        self.pixels.flags.writeable = False

    @property
    def shape(self):
        return self.pixels.shape

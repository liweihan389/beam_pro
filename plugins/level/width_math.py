"""Shared second-moment acquisition, frozen from waist plugin 0.1.0.

Intervals are conditional on the entered error budget, not ISO certification.
The first sensor plane is the origin. Reading errors share its datum error.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class WidthPoint:
    z_mm: float
    tolerance_mm: float
    diameters_um: tuple[float, float]
    sigma_um: tuple[float, float]
    samples: int = 20
    first_frame: int = 0
    last_frame: int = 0


def _moments(signal, bounds):
    x0, y0, x1, y1 = bounds
    image = signal[y0:y1, x0:x1]
    yy, xx = np.mgrid[y0:y1, x0:x1]
    total = image.sum()
    if total <= 0:
        raise ValueError("扣背景后有效光能不足")
    cx, cy = (image * xx).sum() / total, (image * yy).sum() / total
    vx = (image * (xx - cx) ** 2).sum() / total
    vy = (image * (yy - cy) ** 2).sum() / total
    if min(vx, vy) <= 0:
        raise ValueError("二阶矩无效，请检查背景")
    return np.array([cx, cy]), np.sqrt([vx, vy])


def measure_width(frame, selection=None):
    """Signed background-subtracted integrals; never use a geometric contour."""
    image = np.asarray(frame.pixels, dtype=float)
    xoff = yoff = 0
    if selection is not None:
        xoff, yoff, x1, y1 = map(int, selection)
        image = image[yoff:y1, xoff:x1]
    h, w = image.shape
    if min(h, w) < 24:
        raise ValueError("积分选区太小，请包含完整光斑及背景")
    if (image >= frame.full_scale * 0.999).any():
        raise ValueError("图像饱和，暂停宽度记录")
    yy, xx = np.mgrid[:h, :w]
    edge = (xx < w * 0.12) | (xx >= w * 0.88) | (yy < h * 0.12) | (yy >= h * 0.88)
    indices = np.flatnonzero(edge)[:: max(1, int(edge.sum()) // 16000)]
    design = np.column_stack(
        [np.ones(len(indices)), xx.ravel()[indices] / w, yy.ravel()[indices] / h]
    )
    values = image.ravel()[indices]
    keep = np.ones(len(values), dtype=bool)
    for _ in range(4):
        beta = np.linalg.lstsq(design[keep], values[keep], rcond=None)[0]
        residual = values - design @ beta
        noise = max(
            0.1, 1.4826 * np.median(np.abs(residual[keep] - np.median(residual[keep])))
        )
        keep = np.abs(residual - np.median(residual[keep])) < 3.5 * noise
        if keep.sum() < 20:
            raise ValueError("背景不足，不能可靠计算二阶矩")
    signal = image - (beta[0] + beta[1] * xx / w + beta[2] * yy / h)
    peak = signal.max()
    if peak < 30 * noise:
        raise ValueError("宽度测量信噪比不足")
    provisional = np.where(signal > max(6 * noise, peak * 0.015), signal, 0)
    center, sigma = _moments(provisional, (0, 0, w, h))

    def bounds(factor):
        start = np.floor(center - factor * sigma).astype(int)
        end = np.ceil(center + factor * sigma).astype(int) + 1
        if np.any(start < 0) or np.any(end > [w, h]):
            raise ValueError("光斑尾部或积分选区被截断，请扩大画面／选区")
        return int(start[0]), int(start[1]), int(end[0]), int(end[1])

    for _ in range(4):
        region = bounds(4.5)
        next_center, next_sigma = _moments(signal, region)
        change = np.max(np.abs(next_sigma / sigma - 1))
        center, sigma = next_center, next_sigma
        if change < 0.0005:
            break
    _, wide_sigma = _moments(signal, bounds(5.5))
    if np.max(np.abs(wide_sigma / sigma - 1)) > 0.02:
        raise ValueError("宽度随积分范围变化超过2%，请检查尾部、背景或杂光")
    diameter = 4 * sigma
    if diameter.min() < 10:
        raise ValueError("最小方向不足10个像素，束腰采样不足")
    return diameter, center + np.array([xoff, yoff])


def summarize(z, tolerance, widths, identities, pitch_um, magnification):
    values = np.asarray(widths) * pitch_um / magnification
    n = len(values)
    sigma = values.std(axis=0, ddof=1)
    if np.any(sigma / values.mean(axis=0) > 0.03):
        raise ValueError("本点宽度波动超过3%，请等待光束／曝光稳定后重录")
    # Positive serial correlation reduces the effective number of frames.
    centered = values - values.mean(axis=0)
    denominator = np.sum(centered[:-1] ** 2, axis=0)
    rho = np.divide(
        np.sum(centered[:-1] * centered[1:], axis=0),
        denominator,
        out=np.zeros(2),
        where=denominator > 0,
    )
    rho = np.clip(rho, 0, 0.9)
    effective = np.maximum(2, n * (1 - rho) / (1 + rho))
    mean = values.mean(axis=0)
    uncertainty = np.maximum(sigma / np.sqrt(effective), mean * 0.002)
    return WidthPoint(
        float(z),
        float(tolerance),
        tuple(mean),
        tuple(uncertainty),
        n,
        identities[0],
        identities[-1],
    )

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from beamic_sdk import BeamFit, Frame
from scipy import ndimage, optimize, spatial


@dataclass(frozen=True)
class FitSettings:
    boundary_fraction: float = .135
    center_mode: str = "boundary"
    min_snr: float = 12.0
    ambiguity_ratio: float = .25
    max_points: int = 1800

    def __post_init__(self):
        if not .03 <= self.boundary_fraction <= .80:
            raise ValueError("边界强度须在3%到80%之间")
        if self.center_mode not in ("boundary", "area"):
            raise ValueError("未知的几何中心模式")


DEFAULT_SETTINGS = FitSettings()


def gaussian(x, y, amplitude, cx, cy, major, minor, angle, background):
    c, s = np.cos(angle), np.sin(angle)
    u, v = c * (x - cx) + s * (y - cy), -s * (x - cx) + c * (y - cy)
    return background + amplitude * np.exp(-2 * ((u / major) ** 2 + (v / minor) ** 2))


def _background(data):
    """Fit a plane to background pixels, excluding light iteratively."""
    h, w = data.shape
    step = max(1, math.ceil(math.sqrt(data.size / 20000)))
    yy, xx = np.mgrid[:h:step, :w:step]
    design = np.column_stack((np.ones(xx.size), xx.ravel() / w, yy.ravel() / h))
    values = data[::step, ::step].ravel()
    selected = values <= np.percentile(values, 60)
    plane, noise = np.array([np.percentile(values, 20), 0., 0.]), .15
    for _ in range(5):
        if selected.sum() < 30:
            break
        plane = np.linalg.lstsq(design[selected], values[selected], rcond=None)[0]
        residual = values - design @ plane
        median = float(np.median(residual[selected]))
        noise = max(.15, 1.4826 * float(np.median(np.abs(residual[selected] - median))))
        selected = (residual < 3 * noise) & (residual > -3 * noise)
    y, x = np.mgrid[:h, :w]
    return plane[0] + plane[1] * x / w + plane[2] * y / h, noise


def _crossings(signal, level, filled):
    """Subpixel outer-edge crossings; internal dark holes are excluded."""
    points = []
    for axis in (0, 1):
        a = signal[:-1, :] if axis == 0 else signal[:, :-1]
        b = signal[1:, :] if axis == 0 else signal[:, 1:]
        ma = filled[:-1, :] if axis == 0 else filled[:, :-1]
        mb = filled[1:, :] if axis == 0 else filled[:, 1:]
        yy, xx = np.nonzero((ma != mb) & ((a - level) * (b - level) <= 0))
        delta = (level - a[yy, xx]) / (b[yy, xx] - a[yy, xx])
        points.append(np.column_stack((xx + (delta if axis == 1 else 0), yy + (delta if axis == 0 else 0))))
    return np.concatenate(points)


def _ellipse_distance(points, parameters):
    cx, cy, major, minor, angle = parameters
    c, s = np.cos(angle), np.sin(angle)
    dx, dy = points[:, 0] - cx, points[:, 1] - cy
    return (np.sqrt(((c * dx + s * dy) / major) ** 2 + ((-s * dx + c * dy) / minor) ** 2) - 1) * minor


def fit_beam(frame: Frame, selection=None, settings: FitSettings = DEFAULT_SETTINGS) -> BeamFit:
    """Fit a relative-intensity contour without assuming Gaussian irradiance.

    Coordinates use the camera raster. Area centers include actual missing
    regions and holes; the reference ellipse keeps its separate fitted center.
    """
    height, width = frame.shape
    semantics = {"center_mode": settings.center_mode, "boundary_fraction": settings.boundary_fraction,
                 "boundary_kind": "relative_intensity"}
    def fail(reason, roi=None):
        return BeamFit(frame.frame_id, reason=reason, roi=roi, **semantics)
    x0, y0, x1, y1 = (0, 0, width, height) if selection is None else (int(v) for v in selection)
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(width, x1), min(height, y1)
    roi = (x0, y0, x1, y1)
    if x1 - x0 < 9 or y1 - y0 < 9:
        return fail("选择区域过小", roi)
    raw = frame.pixels[y0:y1, x0:x1].astype(float)
    background, noise = _background(raw)
    corrected = raw - background
    # Isolated bad pixels cannot define the peak or the outer edge.
    median = ndimage.median_filter(corrected, size=3)
    signal = ndimage.gaussian_filter(median, .65)
    peak = float(signal.max())
    scale = frame.full_scale
    if peak < settings.min_snr * noise or (scale is not None and peak < .005 * scale):
        return fail("无信号或信噪比不足", roi)
    level = peak * settings.boundary_fraction
    if level < 5 * noise:
        return fail("所选边界低于可靠噪声水平，请提高边界百分比", roi)
    # Connectivity is always assessed at 1/e² of the observed background-
    # subtracted global peak, independently of the display contour setting.
    structure = ndimage.generate_binary_structure(2, 2)
    reference_level = peak * math.exp(-2)
    reference_labels, reference_count = ndimage.label(signal >= reference_level, structure)
    indices = range(1, reference_count + 1)
    energies = ndimage.sum(np.maximum(signal, 0), reference_labels, indices)
    if not len(energies):
        return fail("未找到主光斑", roi)
    primary = int(np.argmax(energies)) + 1
    reference = reference_labels == primary
    reference_areas = ndimage.sum(np.ones_like(signal), reference_labels, indices)
    reference_peaks = ndimage.maximum(signal, reference_labels, indices)
    ambiguous = any(i + 1 != primary and reference_areas[i] >= 9
                    and reference_peaks[i] > reference_level + 5 * noise for i in range(reference_count))
    ry, rx = np.nonzero(reference)
    if len(rx) < 30:
        return fail("光斑采样不足", roi)
    reference_minor = 2 * math.sqrt(max(float(np.linalg.eigvalsh(np.cov([rx, ry], bias=True))[0]), 1))
    window = max(5, int(reference_minor * .7) | 1)
    maxima = (signal == ndimage.maximum_filter(signal, size=window)) & reference
    py, px = np.nonzero(maxima)
    distinct = []
    for i in np.argsort(signal[py, px])[::-1]:
        point = np.array([px[i], py[i]])
        if all(np.linalg.norm(point - other) > .65 * reference_minor for other in distinct):
            distinct.append(point)
    bimodal = len(distinct) == 2
    labels, _ = ndimage.label(signal >= level, structure)
    selected_labels = np.unique(labels[reference & (signal >= level)])
    selected_labels = selected_labels[selected_labels != 0]
    component = np.isin(labels, selected_labels)
    if bimodal and any(signal[p[1], p[0]] < level for p in distinct):
        return fail("1/e²处双峰连通，但所选边界未包含两峰；请降低边界百分比", roi)
    filled = ndimage.binary_fill_holes(component)
    ys, xs = np.nonzero(filled)
    if len(xs) < 30:
        return fail("光斑采样不足", roi)
    points = _crossings(signal, level, filled)
    if len(points) < 20:
        return fail("边界不足，无法可靠拟合", roi)
    observed_points = points
    if bimodal:
        if reference_level < 5 * noise:
            return fail("1/e²处连通性低于可靠噪声水平，无法确认双峰是否连接", roi)
        # Fit the outer envelope of the connected pair. Do not force the neck
        # between its lobes to follow a single Gaussian ellipse.
        try:
            hull = points[spatial.ConvexHull(points).vertices]
        except spatial.QhullError:
            return fail("双峰外边界退化，无法可靠拟合整体椭圆", roi)
        edges = np.roll(hull, -1, axis=0) - hull
        points = np.concatenate([start + edge * np.linspace(0, 1, max(2, int(np.linalg.norm(edge) * 2)), endpoint=False)[:, None]
                                 for start, edge in zip(hull, edges, strict=True)])
    if len(points) > settings.max_points:
        points = points[np.linspace(0, len(points) - 1, settings.max_points).astype(int)]
    mean = np.array([xs.mean(), ys.mean()])
    eigenvalues, vectors = np.linalg.eigh(np.cov(np.array([xs, ys]), bias=True))
    major, minor = 2 * np.sqrt(np.maximum(eigenvalues[::-1], 1))
    angle = math.atan2(vectors[1, 1], vectors[0, 1])
    initial = [*mean, major, minor, angle]
    bounds = ([-raw.shape[1], -raw.shape[0], 1, 1, -2 * math.pi],
              [raw.shape[1] * 2, raw.shape[0] * 2, width * 2, height * 2, 2 * math.pi])
    try:
        solution = optimize.least_squares(lambda p: _ellipse_distance(points, p), initial,
                                          bounds=bounds, loss="linear" if bimodal else "soft_l1", f_scale=.6, max_nfev=60)
        parameters = solution.x
        for _ in range(0 if bimodal else 3):
            inliers = np.abs(_ellipse_distance(points, parameters)) < max(.8, .025 * min(parameters[2:4]))
            if inliers.sum() < 20:
                break
            solution = optimize.least_squares(lambda p, mask=inliers: _ellipse_distance(points[mask], p), parameters,
                                              bounds=bounds, loss="soft_l1", f_scale=.35, max_nfev=35)
            parameters = solution.x
    except (ValueError, FloatingPointError, np.linalg.LinAlgError):
        return fail("边界椭圆拟合失败", roi)
    cx, cy, major, minor, angle = (float(v) for v in parameters)
    if minor > major:
        major, minor, angle = minor, major, angle + math.pi / 2
    angle = (angle + math.pi / 2) % math.pi - math.pi / 2
    shape_kind = "ellipse"
    if major / minor < 1.015:
        circle = optimize.least_squares(lambda p: np.hypot(points[:, 0] - p[0], points[:, 1] - p[1]) - p[2],
                                         [cx, cy, (major + minor) / 2], loss="soft_l1", f_scale=.35)
        cx, cy, major = (float(v) for v in circle.x)
        minor, angle, shape_kind = major, 0., "circle"
    parameters = [cx, cy, major, minor, angle]
    if bimodal:
        points = observed_points
    distances = _ellipse_distance(points, parameters)
    inliers = np.abs(distances) < max(1., .035 * minor)
    c, s = math.cos(angle), math.sin(angle)
    dx, dy = points[:, 0] - cx, points[:, 1] - cy
    angles = np.arctan2((-s * dx + c * dy) / minor, (c * dx + s * dy) / major)
    bins = np.floor((angles + math.pi) / (2 * math.pi) * 36).astype(int) % 36
    covered = np.zeros(36, dtype=bool)
    covered[bins[inliers]] = True
    coverage = float(covered.mean())
    longest_gap, gap = 0, 0
    for occupied in np.tile(covered, 2):
        gap = 0 if occupied else gap + 1
        longest_gap = max(longest_gap, gap)
    y, x = np.mgrid[:raw.shape[0], :raw.shape[1]]
    u, v = c * (x - cx) + s * (y - cy), -s * (x - cx) + c * (y - cy)
    ideal = (u / major) ** 2 + (v / minor) ** 2 <= 1
    missing = float(np.count_nonzero(ideal & ~filled) / max(ideal.sum(), 1))
    holes = float(np.count_nonzero(filled & ~component) / max(filled.sum(), 1))
    area_y, area_x = np.nonzero(component)
    area_cx, area_cy = float(area_x.mean()), float(area_y.mean())
    saturation = ((raw >= scale * .999) & ndimage.binary_dilation(filled, iterations=3)
                  if scale is not None else np.zeros(raw.shape, dtype=bool))
    saturated_fraction = float(saturation.sum() / max(filled.sum(), 1))
    warnings = []
    if bimodal:
        warnings.append("1/e²连通双峰 · 整体椭圆包络")
    if holes > .003:
        warnings.append("内部暗洞")
    if missing > .005 or coverage < 1 or np.mean(~inliers) > .03:
        warnings.append("边界缺损或不规则")
    if np.any((corrected - median > max(peak * .25, noise * 10)) & filled):
        warnings.append("局部亮点")
    residuals = distances if bimodal else distances[inliers]
    rms = float(np.sqrt(np.mean(residuals ** 2)) / minor) if len(residuals) else math.inf
    valid, reason = True, "有效" if not warnings else "有效 · " + "、".join(warnings)
    visible_mask = reference if bimodal else filled
    touches = bool(visible_mask[0].any() or visible_mask[-1].any() or visible_mask[:, 0].any() or visible_mask[:, -1].any())
    if saturation.any():
        valid, reason = False, "光斑饱和；虚线圈仅供找光，请等待降低曝光"
    elif touches:
        valid, reason = False, "光斑被画面或选择区域截断"
    elif ambiguous or len(distinct) > 2:
        valid, reason = False, "1/e²处分离的光斑或超过两个主峰，请框选单个主光斑"
    elif not solution.success or minor < 2 or holes > .20 or (not bimodal and (rms > .04 or coverage < .80 or longest_gap > 5 or missing > .15)):
        valid, reason = False, "明显遮挡、边界不足或拟合不可靠，暂停测量"
    if scale is None:
        reason = "预览定位圈；饱和与数据处理未验证 · " + reason
    selected_cx, selected_cy = (cx, cy) if settings.center_mode == "boundary" else (area_cx, area_cy)
    return BeamFit(frame.frame_id, selected_cx + x0, selected_cy + y0, major, minor, angle,
                   peak, float(background.mean()), rms, saturated_fraction, valid, reason, roi,
                   boundary_center_x=cx + x0, boundary_center_y=cy + y0,
                   area_center_x=area_cx + x0, area_center_y=area_cy + y0, shape_kind=shape_kind,
                   boundary_coverage=coverage, defect_level="mild" if valid and warnings else ("none" if valid else "severe"),
                   warnings=tuple(warnings),
                   background_origin=float(background[0, 0] - x0 * (background[0, -1] - background[0, 0]) / max(raw.shape[1] - 1, 1)
                                           - y0 * (background[-1, 0] - background[0, 0]) / max(raw.shape[0] - 1, 1)),
                   background_slope_x=float((background[0, -1] - background[0, 0]) / max(raw.shape[1] - 1, 1)),
                   background_slope_y=float((background[-1, 0] - background[0, 0]) / max(raw.shape[0] - 1, 1)), **semantics)

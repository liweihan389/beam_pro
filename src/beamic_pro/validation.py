"""Deterministic synthetic fixtures with analytic ground truth, never CCD data."""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, replace

import numpy as np
from beamic_sdk import Frame
from scipy import optimize, special

from .fitting import DEFAULT_SETTINGS, fit_beam, gaussian


@dataclass
class SpotCase:
    name: str
    label: str
    frame: Frame
    center: tuple[float, float] | None
    radii: tuple[float, float] | None
    expected_valid: bool
    kind: str
    angle: float = 0.


def airy_profile(argument):
    argument = np.asarray(argument, dtype=float)
    ratio = np.ones_like(argument)
    np.divide(2 * special.j1(argument), argument, out=ratio, where=argument != 0)
    return ratio ** 2


def clean_case(kind="gaussian", seed=1, center=(153.7, 114.3), radii=(33., 23.), angle=.65,
               amplitude=3100., noise=3., fraction=.135, gradient=0.):
    y, x = np.mgrid[:240, :320]
    background = 30. + gradient * (x / 320 + .6 * y / 240)
    if kind == "airy":
        radius = radii[0]
        signal = amplitude * airy_profile(special.jn_zeros(1, 1)[0] * np.hypot(x - center[0], y - center[1]) / radius)
        root = optimize.brentq(lambda u: float(airy_profile(u)) - fraction, 0., special.jn_zeros(1, 1)[0])
        edge_radius = radius * root / special.jn_zeros(1, 1)[0]
        edge_radii = (edge_radius, edge_radius)
        angle = 0.
    else:
        signal = gaussian(x, y, amplitude, *center, *radii, angle, 0)
        edge_radii = tuple(r * math.sqrt(-math.log(fraction) / 2) for r in radii)
    pixels = np.clip(background + signal + np.random.default_rng(seed).normal(0, noise, x.shape), 0, 4095).astype(np.float32)
    frame = Frame(seed, pixels, time.monotonic_ns(), 1000, 1, "synthetic_fixed_axes", 4095,
                  exposure_verified=True, saturation_verified=True, metadata={"simulated": True, "kind": kind})
    return SpotCase(f"{kind}-{seed}", "圆孔衍射中央亮斑" if kind == "airy" else "椭圆高斯",
                    frame, center, edge_radii, True, kind, angle)


def showcase_cases():
    circle = clean_case(seed=1, radii=(25, 25), angle=0)
    circle.name, circle.label = "round", "圆高斯"
    ellipse = clean_case(seed=2)
    ellipse.name, ellipse.label = "ellipse", "旋转椭圆高斯"
    airy = clean_case("airy", seed=3, radii=(40, 40))
    airy.name = "airy"
    gradient = clean_case(seed=4, gradient=160)
    gradient.name, gradient.label = "gradient", "非均匀背景"
    noisy = clean_case(seed=5, noise=25)
    noisy.name, noisy.label = "noise", "含噪椭圆高斯"
    base = clean_case(seed=6, radii=(30, 25), angle=0)
    y, x = np.mgrid[:240, :320]
    dx, dy = x - base.center[0], y - base.center[1]
    def defect(name, label, pixels, valid):
        return SpotCase(name, label, replace(base.frame, pixels=np.asarray(pixels, dtype=np.float32)),
                        base.center, None, valid, name)
    holes = np.array(base.frame.pixels)
    holes[(dx - 10) ** 2 + dy ** 2 < 4.5 ** 2] = 30.
    hole = defect("hole", "局部内部暗洞", holes, True)
    notch_pixels = np.array(base.frame.pixels)
    notch_pixels[(np.abs(np.arctan2(dy / 25, dx / 30)) < .14) & ((dx / 30) ** 2 + (dy / 25) ** 2 > .65 ** 2)] = 30.
    notch = defect("notch", "局部边界缺口", notch_pixels, True)
    hotspot_pixels = np.array(base.frame.pixels)
    hotspot_pixels[107, 163] = 3900.
    hotspot = defect("hotspot", "孤立亮点", hotspot_pixels, True)
    blocked_pixels = np.array(base.frame.pixels)
    blocked_pixels[x > base.center[0] + 4] = 30.
    blocked = defect("occlusion", "明显遮挡", blocked_pixels, False)
    clipped = clean_case(seed=10, center=(8, 120))
    clipped.name, clipped.label, clipped.expected_valid, clipped.radii = "clipped", "画面截断", False, None
    saturated = clean_case(seed=11, amplitude=6500)
    saturated.name, saturated.label, saturated.expected_valid, saturated.radii = "saturated", "中心饱和", False, None
    a, b = clean_case(seed=12, center=(139, 120), radii=(20, 15), amplitude=1800), clean_case(seed=13, center=(164, 120), radii=(20, 15), amplitude=1800)
    double = defect("multipeak", "1/e²连通双峰", np.clip(a.frame.pixels + b.frame.pixels - 30, 0, 4095), True)
    double.center = (151.5,120)
    wide_a = clean_case(seed=16,center=(133,120),radii=(20,15),angle=0,amplitude=1800)
    wide_b = clean_case(seed=17,center=(178,120),radii=(20,15),angle=0,amplitude=1800)
    neck = defect("necked_pair","1/e²处细颈连接的双峰",wide_a.frame.pixels+wide_b.frame.pixels-30,True)
    neck.center=(155.5,120)
    apart_b = clean_case(seed=18,center=(190,120),radii=(20,15),amplitude=1800)
    separated = defect("separated_pair","1/e²处分离的两光斑",a.frame.pixels+apart_b.frame.pixels-30,False)
    separated.center=None
    triple_frames=[clean_case(seed=20+i,center=(x,120),radii=(15,15),amplitude=1600) for i,x in enumerate((130,151.5,173))]
    triple=defect("triple","三个连接主峰",sum(c.frame.pixels for c in triple_frames)-60,False)
    triple.center=None
    donut = defect("donut", "中央大面积暗洞", 30 + 2500 * np.exp(-((np.hypot(dx, dy) - 23) / 4) ** 2), False)
    dark = clean_case(seed=15, amplitude=0)
    dark.name, dark.label, dark.center, dark.radii, dark.expected_valid = "dark", "无信号", None, None, False
    return [circle, ellipse, airy, gradient, noisy, hole, notch, hotspot, blocked, clipped, saturated, double, neck, separated, triple, donut, dark]


def evaluate(case, settings=DEFAULT_SETTINGS):
    fit = fit_beam(case.frame, settings=settings)
    error = float(np.linalg.norm(fit.center - case.center)) if case.center is not None and fit.radius_minor > 0 else None
    axes_error = [abs(value / truth - 1) for value, truth in zip((fit.radius_major, fit.radius_minor), case.radii, strict=True)] if case.radii else None
    center_limit = max(.5, .01 * case.radii[1]) if case.radii else .5
    passed = fit.valid == case.expected_valid
    if case.expected_valid and case.radii:
        passed = passed and error is not None and error <= center_limit and max(axes_error) <= .03
    return fit, {"name": case.name, "label": case.label, "expected_valid": case.expected_valid,
                 "valid": fit.valid, "passed": bool(passed), "center_error_px": error,
                 "center_limit_px": center_limit if case.radii else None,
                 "axes_relative_errors": axes_error, "reason": fit.reason,
                 "coverage": fit.boundary_coverage, "defect_level": fit.defect_level,
                 "mode": fit.center_mode, "boundary_fraction": fit.boundary_fraction}

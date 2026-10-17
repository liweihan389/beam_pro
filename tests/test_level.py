import math
import sys
from pathlib import Path

import numpy as np
import pytest
from beamic_sdk import Host

DIRECTORY = Path(__file__).parents[1] / "plugins/level"
sys.path.insert(0, str(DIRECTORY))
from level_math import (
    fit_interval,
    geometry,
    height_target,
    validate_pair,
    verify_height,
)
from level_plugin import LevelPlugin
from width_math import WidthPoint


def point(z, diameter=4000.0, other=None, tol=10.0, noise=8.0):
    return WidthPoint(
        z, tol, (diameter, diameter if other is None else other), (noise, noise)
    )


@pytest.mark.parametrize(
    "values", [(1, 1, 0), (-1, 10, 0), (100, 105, 10), (0, 20, 10), (0, math.nan, 0)]
)
def test_invalid_distances(values):
    with pytest.raises(ValueError):
        geometry(*values)


@pytest.mark.parametrize(
    "near,far,offset,tilt",
    [(100, 1000, 50.0, 0.002), (20, 300, -40.0, -0.1), (0, 900, 130.0, 0.001)],
)
def test_single_mirror_target_levels_both_planes(near, far, offset, tilt):
    y1, y2 = offset + near * tilt, offset + far * tilt
    result = height_target(near, far, y1, y2, 0, 0, 3.45, 0, 0, 0)
    assert result["y_px"] == pytest.approx(offset)
    correction = (result["y_px"] - y2) / far
    assert y1 + near * correction == pytest.approx(result["y_px"])


def test_height_uncertainty_cannot_be_averaged_away():
    assert verify_height(100, 900, 120, 120, 0.001, 0.001, 3.45)["passed"]
    assert not verify_height(
        100, 900, 120, 120, 0.001, 0.001, 3.45, height_tolerance_mm=0.1
    )["passed"]


def test_flat_interval_accepts_without_resolving_waist():
    points = [point(z) for z in [0, 1000, 500]]
    result = fit_interval(points)
    assert result.physical == "unresolved"
    assert result.variation()["passed"]
    assert validate_pair(result, [point(250), point(750)])["passed"]


def test_equal_endpoints_with_focus_in_middle_fails():
    result = fit_interval(
        [point(0, 2000), point(1000, 2000), point(500, 250)], draws=2000
    )
    assert not result.variation()["passed"]
    assert min(result.variation()["nominal_pct"]) > 500


def test_one_bad_axis_and_bad_independent_validation():
    points = [point(z, 4000, 4000 * (1 + 0.15 * z / 1000)) for z in [0, 1000, 500]]
    result = fit_interval(points)
    assert not result.variation()["passed"]
    flat = fit_interval([point(z) for z in [0, 1000, 500]])
    assert not validate_pair(flat, [point(250, 4300), point(750)])["passed"]


def test_invalid_or_repeated_new_positions():
    for points in [[point(0), point(10), point(20)], [point(0), point(1000), point(0)]]:
        with pytest.raises(ValueError):
            fit_interval(points)


def test_coupler_change_invalidates_every_record_and_target():
    plugin = LevelPlugin(Host(lambda event: None))
    plugin.on_activate()
    plugin.settings["collimate_enabled"] = True
    plugin.points = [point(0), point(1000), point(500)]
    plugin.model = plugin.frozen = fit_interval(plugin.points, draws=1000)
    plugin.target = {"y_px": 20}
    plugin.height = {"near": object()}
    plugin.on_action("coupler_changed", {})
    assert not plugin.points and not plugin.height and plugin.target is None
    assert plugin.frozen is None and plugin.model is None and plugin.generation == 1


def test_default_modes_and_coupler_requirement():
    events = []
    plugin = LevelPlugin(Host(events.append))
    plugin.on_activate()
    assert plugin.settings["level_enabled"] and not plugin.settings["collimate_enabled"]
    plugin.on_action("configure", {"collimate_enabled": True})
    assert plugin.stage == "setup" and "调焦功能" in events[-1]["text"]


def test_width_collection_drops_duplicate_unstable_and_changed_exposure_frames():
    from dataclasses import replace

    from beamic_pro.simulator import Simulator

    source = Simulator()
    source.level_mode = True
    events = []
    plugin = LevelPlugin(Host(events.append))
    plugin.on_activate()

    def frame():
        f = source.next_frame()
        return replace(
            f,
            metadata={
                **f.metadata,
                "measurement_stable": True,
                "quantitative_ready": True,
            },
        )

    first = frame()
    plugin.on_frame(first, None)
    plugin.on_action(
        "configure",
        {"collimate_enabled": True, "level_enabled": False, "coupler_confirmed": True},
    )
    plugin.on_action("record", {})
    second = frame()
    plugin.on_frame(second, None)
    assert len(plugin.widths) == 1
    plugin.on_frame(second, None)
    assert len(plugin.widths) == 1
    source.set_exposure(900)
    third = frame()
    plugin.on_frame(third, None)
    assert len(plugin.widths) == 1 and plugin.epoch == third.exposure_epoch
    fourth = frame()
    plugin.on_frame(
        replace(fourth, metadata={**fourth.metadata, "measurement_stable": False}), None
    )
    assert not plugin.widths
    fifth = frame()
    plugin.on_frame(replace(fifth, coordinate_key="changed"), None)
    assert plugin.stage == "setup" and not plugin.points


def test_ideal_gaussian_physical_invariant_units():
    z = np.array([0, 1000, 500.0])
    radius = 1000.0
    wavelength = 0.78
    widths = 2 * np.sqrt(radius**2 + (wavelength * 1000 * z / (np.pi * radius)) ** 2)
    result = fit_interval(
        [
            point(float(zi), float(di), noise=0.1, tol=0)
            for zi, di in zip(z, widths, strict=True)
        ]
    )
    a, b, c = result.coefficients[0]
    assert (a * c - b * b / 4) / result.span**2 == pytest.approx(
        (4 * 780 / np.pi) ** 2, rel=1e-6
    )

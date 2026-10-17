import math
from dataclasses import replace

import numpy as np
import pytest
from conftest import image_frame

from beamic_pro.fitting import FitSettings, fit_beam


def pair(distance=40, ratio=1., angle=0., noise=2):
    middle = np.array([155., 120.])
    offset = distance / 2 * np.array([math.cos(angle), math.sin(angle)])
    first = image_frame(center=middle-offset, radii=(20, 15), angle=angle, amplitude=1800, noise=noise)
    second = image_frame(center=middle+offset, radii=(20, 15), angle=angle, amplitude=1800*ratio, noise=noise, frame_id=2)
    return replace(first, pixels=(first.pixels+second.pixels-12).astype(np.float32))


@pytest.mark.parametrize("distance", [25, 35, 45])
@pytest.mark.parametrize("angle", [0., .6, 1.3])
@pytest.mark.parametrize("mode", ["boundary", "area"])
def test_connected_pairs_are_valid_geometrical_envelopes(distance, angle, mode):
    fit = fit_beam(pair(distance, angle=angle), settings=FitSettings(center_mode=mode))
    assert fit.valid, fit.reason
    assert "连通双峰" in fit.reason
    np.testing.assert_allclose(fit.center, (155, 120), atol=.5)
    assert fit.radius_major > fit.radius_minor


@pytest.mark.parametrize("fraction", [.03, .135, .3, .5, .8])
def test_separation_is_assessed_at_one_over_e_squared_not_display_level(fraction):
    fit = fit_beam(pair(51), settings=FitSettings(boundary_fraction=fraction))
    assert not fit.valid
    assert "分离" in fit.reason


@pytest.mark.parametrize("fraction", [.135, .3, .5, .8])
def test_connected_pair_can_have_two_display_contours(fraction):
    fit = fit_beam(pair(40), settings=FitSettings(boundary_fraction=fraction))
    assert fit.valid, fit.reason
    np.testing.assert_allclose(fit.center, (155, 120), atol=.5)


@pytest.mark.parametrize("ratio", [.3, .5, .8])
def test_unequal_connected_peaks_can_be_fitted(ratio):
    fit = fit_beam(pair(35, ratio=ratio))
    assert fit.valid, fit.reason
    assert 137.5 < fit.center_x < 172.5


def test_three_peaks_and_saturation_remain_invalid():
    frames = [image_frame(center=(x,120),radii=(15,15),amplitude=1700) for x in (130,151.5,173)]
    pixels = sum(f.pixels for f in frames)-24
    fit = fit_beam(replace(frames[0], pixels=pixels.astype(np.float32)))
    assert not fit.valid
    assert "两个" in fit.reason
    saturated = replace(pair(), pixels=np.clip(pair().pixels*3,0,4095).astype(np.float32))
    assert not fit_beam(saturated).valid

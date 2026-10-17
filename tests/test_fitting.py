from dataclasses import replace

import numpy as np
import pytest
from conftest import image_frame

from beamic_pro.fitting import fit_beam


@pytest.mark.parametrize("radii,angle", [((25, 25), 0), ((45, 15), .8), ((25, 18), -1.2)])
def test_gaussian_center_and_radii(radii, angle):
    frame = image_frame(center=(143.7, 111.3), radii=radii, angle=angle)
    result = fit_beam(frame)
    assert result.valid, result.reason
    np.testing.assert_allclose(result.center, [143.7, 111.3], atol=.04)
    np.testing.assert_allclose([result.radius_major, result.radius_minor], radii, atol=.1)


@pytest.mark.parametrize("frame", [image_frame(amplitude=6000), image_frame(center=(5, 120)), image_frame(amplitude=0)])
def test_reject_saturated_truncated_or_dark(frame):
    result = fit_beam(frame)
    assert not result.valid


def test_multiple_spots_require_selection():
    first, second = image_frame(center=(100, 100)), image_frame(center=(230, 140), amplitude=2000)
    frame = replace(first, pixels=(first.pixels + second.pixels - 12).astype(np.float32))
    assert not fit_beam(frame).valid
    selected = fit_beam(frame, (30, 35, 165, 165))
    assert selected.valid, selected.reason
    np.testing.assert_allclose(selected.center, (100, 100), atol=.1)


def test_overlapping_pair_is_accepted_as_geometrical_envelope():
    first, second = image_frame(center=(138, 120), radii=(20, 15), amplitude=1800), image_frame(center=(163, 120), radii=(20, 15), amplitude=1800)
    frame = replace(first, pixels=(first.pixels + second.pixels - 12).astype(np.float32))
    result = fit_beam(frame)
    assert result.valid, result.reason
    assert "连通双峰" in result.reason


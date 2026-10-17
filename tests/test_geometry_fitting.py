from dataclasses import replace

import numpy as np
import pytest
from scipy import ndimage

from beamic_pro.fitting import FitSettings, _background, fit_beam
from beamic_pro.validation import clean_case, evaluate, showcase_cases


@pytest.mark.parametrize("case", showcase_cases(), ids=lambda case: case.name)
def test_defects_and_diffraction(case):
    fit, result = evaluate(case)
    assert result["passed"], result
    if case.name in ("hole", "notch", "hotspot"):
        assert fit.defect_level == "mild"
        assert np.linalg.norm(fit.boundary_center - case.center) < .5
    if case.name == "airy":
        assert fit.shape_kind == "circle"
        assert fit.radius_minor < 40  # Main lobe only, excludes side rings.


@pytest.mark.parametrize("kind", ["gaussian", "airy"])
@pytest.mark.parametrize("mode", ["boundary", "area"])
def test_randomized_centers_radii_rotation_noise(kind, mode):
    rng = np.random.default_rng(20261014)
    for seed in range(20):
        major, minor = rng.uniform(24, 48), rng.uniform(14, 23)
        radii = (major, major) if kind == "airy" else (major, minor)
        case = clean_case(kind, seed=seed + 30, center=tuple(rng.uniform([100, 80], [215, 155])),
                          radii=radii, angle=rng.uniform(-1.5, 1.5), amplitude=rng.uniform(1700, 3400),
                          noise=rng.uniform(2, 18), gradient=rng.uniform(0, 90))
        _, result = evaluate(case, FitSettings(center_mode=mode))
        assert result["passed"], result


@pytest.mark.parametrize("fraction", [.03, .05, .135, .3, .5, .8])
@pytest.mark.parametrize("kind", ["gaussian", "airy"])
def test_boundary_percent_changes_geometry_with_known_truth(fraction, kind):
    case = clean_case(kind, radii=(42, 30) if kind == "gaussian" else (42, 42), fraction=fraction, noise=2)
    _, result = evaluate(case, FitSettings(boundary_fraction=fraction))
    assert result["passed"], result


def test_area_center_responds_to_missing_area_not_brightness():
    case = next(c for c in showcase_cases() if c.name == "hole")
    fit = fit_beam(case.frame, settings=FitSettings(center_mode="area"))
    assert fit.valid
    assert fit.center_x < fit.boundary_center_x - .20
    np.testing.assert_allclose(fit.center, [fit.area_center_x, fit.area_center_y])
    assert np.linalg.norm(fit.boundary_center - case.center) < .05
    # Independently calculate the observed footprint centroid at the chosen level.
    background, _ = _background(case.frame.pixels.astype(float))
    signal = ndimage.gaussian_filter(ndimage.median_filter(case.frame.pixels - background, size=3), .65)
    y, x = np.nonzero(signal >= signal.max() * .135)
    np.testing.assert_allclose(fit.center, (x.mean(), y.mean()), atol=1e-8)


def test_boundary_ignores_exposure_scaling_and_preserves_raster_coordinates():
    case = clean_case()
    dim = replace(case.frame, pixels=(case.frame.pixels * .6).astype(np.float32), exposure_us=600)
    before, after = fit_beam(case.frame), fit_beam(dim)
    np.testing.assert_allclose(before.center, after.center, atol=.005)
    np.testing.assert_allclose([before.radius_major, before.radius_minor], [after.radius_major, after.radius_minor], atol=.02)
    selected = fit_beam(case.frame, (70, 45, 235, 195))
    assert selected.valid
    np.testing.assert_allclose(selected.center, before.center, atol=.04)


def test_invalid_settings_are_rejected():
    with pytest.raises(ValueError):
        FitSettings(center_mode="weighted")
    with pytest.raises(ValueError):
        FitSettings(boundary_fraction=0)

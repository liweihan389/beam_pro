import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from beamic_sdk import Host
from conftest import image_frame

sys.path.insert(0, str(Path(__file__).parents[1]/'plugins'/'waist'))
from waist_math import WidthPoint, fit_caustic, measure_width, summarize, validate_point
from waist_plugin import WaistPlugin


def points(z=(0, 150, 350, 170, 520), tolerance=0, diameters=(110, 140), waists=(160, 180)):
    return [WidthPoint(float(position), tolerance,
                       tuple(np.sqrt(np.array(diameters)**2+(np.array([.7, .8])*(position-np.array(waists)))**2)),
                       (.25, .25)) for position in z]


@pytest.mark.parametrize('angle', [0, .3, 1.2])
@pytest.mark.parametrize('noise', [0, 2, 5])
def test_second_moments_rotated_gaussian_with_background(angle, noise):
    frame = image_frame(radii=(30, 20), angle=angle, noise=noise)
    yy, xx = np.mgrid[:240, :320]
    frame = replace(frame, pixels=frame.pixels+.01*xx-.012*yy)
    actual, center = measure_width(frame)
    expected = 2*np.sqrt([(30*np.cos(angle))**2+(20*np.sin(angle))**2,
                         (30*np.sin(angle))**2+(20*np.cos(angle))**2])
    np.testing.assert_allclose(actual, expected, rtol=.015)
    np.testing.assert_allclose(center, (150, 120), atol=.1)


def test_bimodal_width_counts_both_peaks_not_the_envelope():
    first = image_frame(center=(130, 120), radii=(20, 20), angle=0, noise=0, amplitude=1400)
    second = image_frame(center=(170, 120), radii=(20, 20), angle=0, noise=0, amplitude=1400)
    frame = replace(first, pixels=first.pixels+second.pixels-12)
    width, center = measure_width(frame)
    np.testing.assert_allclose(width, (4*np.sqrt(10**2+20**2), 40), rtol=.015)
    np.testing.assert_allclose(center, (150, 120), atol=.05)


@pytest.mark.parametrize('inner', [0, 20])
def test_uniform_disk_and_annulus_second_moment_width(inner):
    yy, xx = np.mgrid[:240, :320]
    r2 = (xx-150)**2+(yy-120)**2
    frame = replace(image_frame(), pixels=(12+2500*((r2 <= 30**2) & (r2 >= inner**2))).astype(np.float32))
    width, center = measure_width(frame)
    np.testing.assert_allclose(width, 2*np.sqrt(30**2+inner**2), rtol=.015)
    np.testing.assert_allclose(center, (150, 120), atol=.05)


def test_gaussian_with_central_dark_defect_measures_remaining_energy():
    yy, xx = np.mgrid[:240, :320]
    frame = image_frame(radii=(20, 20), angle=0, noise=0)
    pixels = frame.pixels.copy()
    pixels[(xx-150)**2+(yy-120)**2 < 8**2] = 12
    width, _ = measure_width(replace(frame, pixels=pixels))
    np.testing.assert_allclose(width, 2*np.sqrt(20**2+2*8**2), rtol=.015)


@pytest.mark.parametrize('kind', ['saturated', 'truncated', 'dark', 'undersampled'])
def test_invalid_widths_are_rejected(kind):
    kwargs = {'saturated': {'amplitude': 6000}, 'truncated': {'center': (10, 120)},
              'dark': {'amplitude': 10}, 'undersampled': {'radii': (3, 3)}}[kind]
    with pytest.raises(ValueError):
        measure_width(image_frame(**kwargs))


def test_partial_roi_is_rejected_and_readonly_frame_is_unchanged():
    frame = image_frame()
    before = frame.pixels.copy()
    with pytest.raises(ValueError):
        measure_width(frame, (130, 100, 170, 140))
    np.testing.assert_array_equal(frame.pixels, before)
    assert not frame.pixels.flags.writeable


def test_noiseless_three_points_recover_both_waists_and_nonzero_uncertainty():
    result = fit_caustic(points()[:3], scale_tolerance_pct=2)
    np.testing.assert_allclose(result.parameters[:, :2], [(160, 110), (180, 140)], atol=.001)
    assert np.all(result.intervals[1, :, 1]-result.intervals[0, :, 1] > 2)
    assert result.valid_fraction > .99


def test_five_points_independent_prediction_rejects_wrong_spot():
    data = points(tolerance=.5)
    initial = fit_caustic(data[:3])
    snapshot = initial.parameters.copy()
    assert all(validate_point(initial, point)['passed'] for point in data[3:])
    bad = replace(data[3], diameters_um=(data[3].diameters_um[0]*1.25, data[3].diameters_um[1]))
    assert not validate_point(initial, bad)['passed']
    np.testing.assert_array_equal(initial.parameters, snapshot)
    final = fit_caustic(data)
    assert final.precision() and final.goodness


def test_coarse_placement_produces_a_real_position_interval():
    fine = fit_caustic(points(tolerance=.1))
    coarse = fit_caustic(points(tolerance=10))
    fine_width = fine.intervals[1, :, 0]-fine.intervals[0, :, 0]
    coarse_width = coarse.intervals[1, :, 0]-coarse.intervals[0, :, 0]
    assert np.all(coarse_width > 5*fine_width)
    assert not coarse.precision(position_mm=.1)


def test_common_width_scale_does_not_move_the_waist():
    data = points(tolerance=1)
    a = fit_caustic(data, 0)
    b = fit_caustic([replace(p, diameters_um=tuple(np.array(p.diameters_um)*1.03),
                             sigma_um=tuple(np.array(p.sigma_um)*1.03)) for p in data], 0)
    np.testing.assert_allclose(a.parameters[:, 0], b.parameters[:, 0], atol=1e-5)
    np.testing.assert_allclose(b.parameters[:, 1]/a.parameters[:, 1], 1.03)


@pytest.mark.parametrize('data', [points(z=(0, 0, 200)), points(z=(0, 1, 2), tolerance=10),
                                points(z=(0, 150)), points(z=(10, 150, 350))])
def test_bad_positions_refused(data):
    with pytest.raises(ValueError):
        fit_caustic(data)


def test_nonphysical_curvature_and_zero_minimum_are_not_clipped_into_a_waist():
    for diameters in ((100., 200., 300.), (300., 200., 100.)):
        with pytest.raises(ValueError, match='正曲率'):
            fit_caustic([WidthPoint(z, 0, (d, d), (.2, .2)) for z, d in zip((0, 100, 200), diameters, strict=True)])


def test_all_point_fit_rejects_an_incompatible_propagation_curve():
    data = points(tolerance=0)
    data[-1] = replace(data[-1], diameters_um=tuple(np.array(data[-1].diameters_um)*1.3))
    result = fit_caustic(data)
    assert not result.goodness


def test_pixel_calibration_mismatch_cannot_enter_even_the_first_capture():
    plugin = WaistPlugin(Host(lambda event: None))
    plugin.on_activate()
    plugin.on_action('configure', {'pixel_pitch_um': 3.45})
    plugin.on_action('record', {'z_mm': 0})
    frame = replace(image_frame(), pixel_pitch_um=5.2,
                    metadata={'measurement_stable': True, 'quantitative_ready': True})
    plugin.on_frame(frame, None)
    assert not plugin.points and not plugin.waiting


def test_frame_average_accounts_for_serial_correlation_and_width_drift():
    values = np.column_stack([50+np.sin(np.arange(20)*.1)*.15, 40+np.sin(np.arange(20)*.1)*.1])
    point = summarize(100, 10, values, list(range(20)), 3.45, 1)
    assert point.sigma_um[0] >= values[:, 0].std(ddof=1)*3.45/np.sqrt(20)
    with pytest.raises(ValueError, match='波动'):
        summarize(100, 10, np.column_stack([np.arange(20)+20, np.ones(20)*40]), list(range(20)), 3.45, 1)


def test_capture_excludes_old_unverified_duplicate_and_cross_exposure_frames():
    events = []
    plugin = WaistPlugin(Host(events.append))
    plugin.on_activate()
    plugin.on_action('configure', {'pixel_pitch_um': 3.45})
    plugin.on_action('record', {'z_mm': 0, 'min_frame_id': 10})
    good = replace(image_frame(frame_id=11, epoch=1), pixel_pitch_um=3.45,
                   metadata={'measurement_stable': True, 'quantitative_ready': True})
    plugin.on_frame(replace(good, frame_id=9), None)
    assert not plugin.widths
    plugin.on_frame(replace(good, exposure_verified=False), None)
    assert not plugin.widths
    plugin.on_frame(good, None)
    plugin.on_frame(good, None)
    assert len(plugin.widths) == 1
    plugin.on_frame(replace(good, frame_id=12, exposure_epoch=2, exposure_us=1200), None)
    assert len(plugin.widths) == 1 and plugin.identities == [12]
    plugin.on_frame(replace(good, frame_id=13, exposure_epoch=2, exposure_us=1200,
                            metadata={'measurement_stable': False, 'quantitative_ready': True}), None)
    assert not plugin.widths


def test_new_coordinate_or_pixel_calibration_invalidates_prior_data():
    events = []
    plugin = WaistPlugin(Host(events.append))
    plugin.on_activate()
    plugin.points = points()
    plugin.stage = 'final'
    frame = replace(image_frame(), pixel_pitch_um=3.45)
    plugin.on_frame(frame, None)
    plugin.on_frame(replace(frame, frame_id=2, coordinate_key='changed'), None)
    assert not plugin.points and plugin.stage == 'setup'


def test_delayed_source_frame_cannot_reset_a_newly_configured_session():
    plugin = WaistPlugin(Host(lambda event: None))
    plugin.on_activate()
    plugin.on_action('configure', {'min_frame_id': 20})
    old = replace(image_frame(frame_id=19), coordinate_key='old-format')
    plugin.on_frame(old, None)
    assert plugin.key is None and plugin.stage == 'capture'
    plugin.on_frame(replace(old, frame_id=21, coordinate_key='new-format'), None)
    assert plugin.key[0] == 'new-format' and plugin.stage == 'capture'


def test_replacing_build_point_requires_fresh_holdouts_and_first_point_resets_origin():
    plugin = WaistPlugin(Host(lambda event: None))
    plugin.on_activate()
    plugin.points = points()
    plugin.frozen = fit_caustic(plugin.points[:3])
    plugin.result = fit_caustic(plugin.points)
    plugin.stage = 'final'
    plugin.on_action('redo', {'redo_index': 2})
    assert len(plugin.points) == 3 and plugin.frozen is None
    plugin.on_action('redo', {'redo_index': 1})
    assert not plugin.points and plugin.stage == 'setup'


def test_future_host_requirement_is_checked():
    import json

    from beamic_pro.plugin_store import PluginStore
    manifest = json.loads((Path(__file__).parents[1]/'plugins/waist/manifest.json').read_text())
    assert PluginStore.validate_manifest(manifest)
    with pytest.raises(ValueError, match='更新版本'):
        PluginStore.validate_manifest({**manifest, 'minimum_host_version': '999.0'})

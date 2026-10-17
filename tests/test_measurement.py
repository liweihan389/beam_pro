from dataclasses import replace

from conftest import image_frame

from beamic_pro.exposure import ExposureController
from beamic_pro.measurement import Capture


def test_capture_only_counts_new_verified_matching_epoch_frames(good_fit):
    capture = Capture("near_red", 20, exposure_epoch=2, after_frame_id=10)
    frame = image_frame(frame_id=9, epoch=2)
    assert capture.add(frame, replace(good_fit, frame_id=9), True) is None
    assert not capture.fits
    frame = image_frame(frame_id=11, epoch=1)
    capture.add(frame, replace(good_fit, frame_id=11), True)
    assert not capture.fits
    result = None
    for identity in range(12, 32):
        frame = image_frame(frame_id=identity, epoch=2)
        fit = replace(good_fit, frame_id=identity)
        result = capture.add(frame, fit, True)
        capture.add(frame, fit, True)  # Duplicate does not count.
    assert result.sample_count == 20
    assert result.first_frame_id == 12 and result.last_frame_id == 31


def test_exposure_changes_restart_capture(good_fit):
    capture = Capture("red", 4)
    for i in range(1, 3):
        capture.add(image_frame(frame_id=i), replace(good_fit, frame_id=i), True)
    capture.add(image_frame(frame_id=3, exposure=1200), replace(good_fit, frame_id=3), True)
    assert len(capture.fits) == 1


def test_jitter_is_not_averaged_into_false_precision(good_fit):
    capture = Capture("red", 10)
    for identity in range(1, 11):
        fit = replace(good_fit, frame_id=identity, center_x=150 + (-1) ** identity * 2)
        assert capture.add(image_frame(frame_id=identity), fit, True) is None
    assert "抖动" in capture.reason
    assert not capture.fits


def test_unverified_and_invalid_fit_cannot_record(good_fit):
    capture = Capture("red", 2)
    capture.add(image_frame(frame_id=1, verified=False), good_fit, True)
    assert not capture.fits
    capture.add(image_frame(frame_id=2), replace(good_fit, frame_id=2, valid=False), True)
    assert not capture.fits


def test_exposure_controller_waits_for_acknowledged_frame():
    controller = ExposureController()
    controller.acknowledge(500, 2)
    old = image_frame(exposure=1000, epoch=1, frame_id=1)
    assert controller.inspect(old) is None
    assert not controller.state.stable
    for i in range(2, 5):
        frame = image_frame(exposure=500, epoch=2, frame_id=i, amplitude=3276)
        assert controller.inspect(frame) is None
    assert controller.state.stable
    assert controller.state.epoch == 2


def test_exposure_reduces_saturation_and_respects_limits():
    controller = ExposureController()
    request = controller.inspect(image_frame(amplitude=6000))
    assert request < 1000
    controller = ExposureController()
    controller.maximum_us = 1100
    request = controller.inspect(image_frame(amplitude=1000))
    assert request == 1100



def test_capture_rejects_definition_change_and_transmits_selected_definition(good_fit):
    import pytest
    capture = Capture("red", 2)
    capture.add(image_frame(frame_id=1), good_fit, True)
    with pytest.raises(ValueError, match="定义"):
        capture.add(image_frame(frame_id=2), replace(good_fit, frame_id=2, center_mode="area"), True)
    capture = Capture("red", 2)
    result = None
    for i in (1, 2):
        result = capture.add(image_frame(frame_id=i), replace(good_fit, frame_id=i, center_mode="area", boundary_fraction=.3), True)
    assert result.center_mode == "area"
    assert result.boundary_fraction == .3


def test_manual_exposure_keeps_value_and_allows_lower_peak():
    controller = ExposureController()
    controller.set_enabled(False)
    for i in range(1,4):
        assert controller.inspect(image_frame(frame_id=i, amplitude=1200, exposure=700)) is None
    assert controller.state.stable
    assert controller.pending_us is None
    controller.set_enabled(True)
    assert controller.inspect(image_frame(frame_id=4, amplitude=1200, exposure=700)) > 700


def test_manual_exposure_never_adjusts_saturation_or_dark_signal():
    for amplitude in (0,6000):
        controller=ExposureController()
        controller.set_enabled(False)
        assert controller.inspect(image_frame(amplitude=amplitude)) is None
        assert not controller.state.stable
        assert controller.pending_us is None


def test_manual_exposure_still_waits_for_acknowledged_new_frames():
    controller=ExposureController()
    controller.acknowledge(500,2)
    controller.set_enabled(False)
    assert controller.inspect(image_frame(frame_id=1, exposure=1000,epoch=1,amplitude=1200)) is None
    assert not controller.state.stable
    for i in range(2,5):
        assert controller.inspect(image_frame(frame_id=i,exposure=500,epoch=2,amplitude=1200)) is None
    assert controller.state.stable
    assert controller.state.epoch==2


def test_manual_exposure_does_not_bypass_unverified_frames():
    controller=ExposureController()
    controller.set_enabled(False)
    for i in range(1,5):
        assert controller.inspect(image_frame(frame_id=i,verified=False)) is None
    assert not controller.state.stable
    assert controller.pending_us is None

from dataclasses import replace

from conftest import image_frame

from beamic_pro.measurement import Capture


def test_broad_blue_beam_cannot_relax_red_jitter_budget(good_fit):
    capture = Capture("near_blue", 10, reference_radius=5)
    for identity in range(1, 11):
        fit = replace(good_fit, frame_id=identity, radius_minor=80,
                      center_x=150 + (-1)**identity * .3)
        assert capture.add(image_frame(frame_id=identity), fit, True) is None
    assert "抖动" in capture.reason


def test_unstable_radius_does_not_inflate_acceptance_tolerance(good_fit):
    capture = Capture("near_red", 10)
    for identity in range(1, 11):
        fit = replace(good_fit, frame_id=identity, radius_minor=20 + (-1)**identity * 3)
        assert capture.add(image_frame(frame_id=identity), fit, True) is None
    assert "尺寸" in capture.reason


from dataclasses import replace

from beamic_pro.view import BeamView


def tagged(view, name):
    return [item for item in view.items if item.data(0) == name]


def test_reference_center_and_connection_use_correct_geometrical_points(
    application, good_fit
):
    view = BeamView()
    reference = replace(
        good_fit,
        center_mode="area",
        center_x=147,
        center_y=121,
        boundary_center_x=150,
        boundary_center_y=120,
    )
    current = replace(
        reference,
        frame_id=2,
        center_x=174,
        center_y=133,
        boundary_center_x=170,
        boundary_center_y=130,
    )
    try:
        view.overlays(reference)
        assert tagged(view, "circle_center")
        view.toggle_fixed_circle()
        view.overlays(current)
        assert view.fixed_fit is reference
        assert tagged(view, "circle_center") and tagged(view, "live_center")
        assert not tagged(view, "center_link")
        # Hiding the live marker does not stop the independently enabled link.
        view.set_fixed_display(False, True)
        assert not tagged(view, "live_center")
        line = tagged(view, "center_link")[0].line()
        assert (line.x1(), line.y1()) == (
            150,
            120,
        )  # Circle center, not old area center.
        assert (line.x2(), line.y2()) == (174, 133)  # Selected live area center.
        for identity in range(3, 8):
            view.overlays(replace(current, frame_id=identity, center_x=174 + identity))
            assert len(tagged(view, "center_link")) == 1
        view.set_fixed_display(True, False)
        assert tagged(view, "live_center") and not tagged(view, "center_link")
        view.toggle_fixed_circle()
        assert view.fixed_fit is None
        assert not tagged(view, "center_link")
    finally:
        view.close()


def test_unreliable_or_missing_centers_hide_link_but_preserve_reference(
    application, good_fit
):
    view = BeamView()
    try:
        view.overlays(good_fit)
        view.toggle_fixed_circle()
        view.set_fixed_display(True, True)
        for failed in (replace(good_fit, valid=False, center_x=300), None):
            view.overlays(failed)
            assert not tagged(view, "live_center")
            assert not tagged(view, "center_link")
            assert tagged(view, "circle_center")
            assert view.fixed_fit is good_fit
        view.overlays(replace(good_fit, frame_id=2, center_x=175))
        assert tagged(view, "live_center") and tagged(view, "center_link")
        view.set_auto_circle(False)
        assert view.fixed_fit is None
        assert not tagged(view, "center_link")
        assert not tagged(view, "circle_center")
        assert tagged(view, "live_center")
    finally:
        view.close()


def test_horizontal_height_target_is_not_a_two_axis_target(application):
    view = BeamView()
    view.dimensions = (320, 240)
    try:
        view.overlays(
            None,
            [{"kind": "horizontal_line", "y": 120, "half_height": 3, "label": "等高"}],
        )
        target = tagged(view, "height_target")[0].line()
        assert (target.x1(), target.x2(), target.y1(), target.y2()) == (
            0,
            320,
            120,
            120,
        )
        assert tagged(view, "height_tolerance")
        for y in [-1, 241, float("nan")]:
            view.overlays(None, [{"kind": "horizontal_line", "y": y}])
            assert not tagged(view, "height_target")
    finally:
        view.close()

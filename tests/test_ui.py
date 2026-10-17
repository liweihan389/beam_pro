import time

from beamic_pro.app import Window


def pump(application, seconds):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        application.processEvents()
        time.sleep(.005)


def test_live_demo_render_fit_and_plugin_activation(application, tmp_path):
    window = Window(tmp_path, demo=True)
    window.show()
    try:
        pump(application, 2)
        assert window.last_frame is not None
        assert window.last_fit and window.last_fit.valid
        assert window.exposure.state.stable
        assert window.received_rate.fps >= 15
        assert window.display_rate.fps >= 15
        assert window.activate("basic_analysis")
        assert window.activate("alignment")
        pump(application, 1)
        assert "13.5%" in window.plugin_states.get("basic_analysis", {}).get("text", "")
        window.start_alignment()
        pump(application, .3)
        assert "近处" in window.guidance_text.text()
        window.send_action("record")
        pump(application, 2)
        assert "近处蓝光" in window.guidance_text.text()
    finally:
        window.close()
        pump(application, .1)



def test_fixed_reference_does_not_freeze_measurement_or_plugin_targets(application, tmp_path):
    from PySide6.QtWidgets import QGraphicsEllipseItem
    window = Window(tmp_path, demo=True)
    window.show()
    try:
        pump(application, 1.5)
        assert window.auto_circle.isChecked()
        assert window.fixed_circle_button.isEnabled()
        start = window.last_fit
        window.fixed_circle_button.click()
        fixed = window.view.fixed_fit
        assert fixed is start
        assert window.fixed_circle_button.text() == "解除固定"
        assert window.fixed_options.isVisible()
        assert window.track_center_input.isChecked()
        assert not window.connect_centers_input.isChecked()
        window.demo_color.setCurrentIndex(1)
        pump(application, 1.5)
        assert window.last_fit.center_x > start.center_x + 8
        assert window.view.fixed_fit is fixed
        assert window.last_fit.frame_id > fixed.frame_id
        assert window.exposure.state.stable
        window.connect_centers_input.setChecked(True)
        window.track_center_input.setChecked(False)
        # Between new preview arrival and its fit, the previous centre is reference
        # only. Inspect the next fresh result rather than an arbitrary refresh.
        deadline = time.monotonic() + 2
        links = []
        while time.monotonic() < deadline and not links:
            application.processEvents()
            links = [item for item in window.view.items if item.data(0) == "center_link"]
            if not links:
                time.sleep(.002)
        assert len(links) == 1
        assert not any(item.data(0) == "live_center" for item in window.view.items)
        assert links[0].line().x1() == fixed.boundary_center[0]
        assert links[0].line().x2() == window.last_fit.center_x
        # This target survives hiding the automatic contour.
        window.overlays["test"] = [{"kind": "target", "x": 100, "y": 80, "radius": 3}]
        pump(application, .15)
        window.auto_circle.setChecked(False)
        assert window.view.fixed_fit is None
        assert not window.fixed_options.isVisible()
        assert not any(item.data(0) == "center_link" for item in window.view.items)
        assert not window.fixed_circle_button.isEnabled()
        assert len([item for item in window.view.items if isinstance(item, QGraphicsEllipseItem)]) == 1
        from beamic_pro.measurement import Capture
        window.capture = Capture("test", 3, window.exposure.state.epoch,
                                 window.last_frame.coordinate_key, window.last_frame.frame_id)
        pump(application, .5)
        assert window.capture is None
        assert "已记录 3 帧" in window.capture_label.text()
        window.auto_circle.setChecked(True)
        assert window.fixed_circle_button.isEnabled()
        window.fixed_circle_button.click()
        window.fixed_circle_button.click()
        assert window.view.fixed_fit is None
    finally:
        window.close()
        pump(application, .1)


def test_definition_change_resets_alignment_and_settings_persist(application, tmp_path):
    window = Window(tmp_path, demo=True)
    window.show()
    try:
        pump(application, 1.5)
        window.start_alignment()
        pump(application, .3)
        window.send_action("record")
        pump(application, 1.4)
        assert "近处蓝光" in window.guidance_text.text()
        window.fixed_circle_button.click()
        window.center_input.setCurrentIndex(1)
        window.boundary_input.setValue(30)
        pump(application, .3)
        assert window.capture is None
        assert window.view.fixed_fit is None
        assert "先遮蓝光" in window.guidance_text.text()
        assert window.last_fit.center_mode == "area"
        assert window.last_fit.boundary_fraction == .3
    finally:
        window.close()
        pump(application, .1)
    reopened = Window(tmp_path)
    try:
        assert reopened.center_input.currentIndex() == 1
        assert reopened.boundary_input.value() == 30
        assert reopened.auto_circle.isChecked()
    finally:
        reopened.close()
        pump(application, .1)


def test_complete_alignment_with_geometric_measurements(application, tmp_path):
    import numpy as np
    window = Window(tmp_path, demo=True)
    window.show()
    def until(condition, seconds=5):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            application.processEvents()
            if condition():
                return
            time.sleep(.005)
        raise AssertionError(window.guidance_text.text() + " / " + window.capture_label.text())
    def record(index, expected_next):
        window.demo_plane.setCurrentIndex(index // 2)
        window.demo_color.setCurrentIndex(index % 2)
        until(lambda: window.exposure.state.stable)
        window.send_action("record")
        until(lambda: expected_next in window.guidance_text.text())
    try:
        until(lambda: window.last_fit and window.last_fit.valid and window.exposure.state.stable)
        window.start_alignment()
        until(lambda: "先遮蓝光" in window.guidance_text.text())
        record(0, "近处蓝光")
        record(1, "远处红光")
        record(2, "远处蓝光")
        record(3, "中间目标")
        source = window.source
        g = source.geometry
        source.near_kick = (source.offset - g.mirror_far * source.slope) / (g.mirror_far - g.mirror_near)
        until(lambda: "目标已到达" in window.guidance_text.text())
        window.send_action("confirm_near")
        until(lambda: "只调M1远反射镜" in window.guidance_text.text())
        source.far_kick = -(source.offset - g.mirror_near * source.slope) / (g.mirror_far - g.mirror_near)
        np.testing.assert_allclose(source.slope + source.near_kick + source.far_kick, 0, atol=1e-12)
        until(lambda: "目标已到达" in window.guidance_text.text())
        window.send_action("confirm_far")
        until(lambda: "移回近处" in window.guidance_text.text())
        record(0, "近处蓝光")
        record(1, "远处红光")
        record(2, "远处蓝光")
        record(3, "合束通过")
        assert "所选边界短半轴的5%" in window.guidance_text.text()
    finally:
        window.close()
        pump(application, .1)


def test_exposure_checkbox_keeps_manual_value_and_measurement_active(application,tmp_path):
    from beamic_pro.measurement import Capture
    window=Window(tmp_path,demo=True)
    window.show()
    try:
        pump(application,1.3)
        assert window.auto_exposure_input.isChecked()
        value=window.last_frame.exposure_us
        window.auto_exposure_input.setChecked(False)
        window.demo_color.setCurrentIndex(1)
        pump(application,.6)
        assert not window.exposure.enabled
        assert window.last_frame.exposure_us==value
        assert window.source.exposure_us==value
        assert window.exposure.state.stable
        window.capture=Capture('manual',4,window.exposure.state.epoch,
                               window.last_frame.coordinate_key,window.last_frame.frame_id)
        pump(application,.4)
        assert window.capture is None
        assert '已记录 4 帧' in window.capture_label.text()
        window.start_alignment()
        pump(application,.25)
        assert not window.auto_exposure_input.isChecked()
        assert not window.exposure.enabled
        window.auto_exposure_input.setChecked(True)
        pump(application,.6)
        assert window.last_frame.exposure_us > value
    finally:
        window.close()
        pump(application,.1)

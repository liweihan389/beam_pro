import time

from beamic_pro.app import Window


def wait(app, window, predicate, timeout=15):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        app.processEvents()
        if predicate():
            return
        time.sleep(0.004)
    raise AssertionError(
        str(window.plugin_states.get("level", {}))[:2500] + window.log.toPlainText()
    )


def start(app, window, collimate=False, both=False):
    wait(app, window, lambda: window.capture_ready and window.exposure.state.stable)
    window.level_panel.buttons["activate"].click()
    wait(
        app,
        window,
        lambda: (
            "configure" in window.level_panel.buttons
            and window.last_frame.coordinate_key.startswith("sim-level")
            and window.exposure.state.stable
        ),
    )
    wait(
        app, window, lambda: window.level_panel.controls["pixel_pitch_um"].value() == 20
    )
    if collimate:
        window.level_panel.controls["collimate_enabled"].setChecked(True)
        window.level_panel.controls["level_enabled"].setChecked(both)
        window.level_panel.controls["coupler_confirmed"].setChecked(True)
    window.level_panel.buttons["configure"].click()
    wait(app, window, lambda: window.plugin_states["level"]["data"]["stage"] != "setup")


def record(app, window, predicate, z=None):
    wait(app, window, lambda: "record" in window.level_panel.buttons)
    if z is not None:
        window.level_panel.controls["z_mm"].setValue(z)
    window.level_panel.buttons["record"].click()
    wait(app, window, predicate)


def state(window):
    return window.plugin_states["level"]["data"]


def test_entire_level_workflow_preserves_horizontal_freedom(application, tmp_path):
    window = Window(tmp_path, demo=True)
    window.show()
    try:
        start(application, window)
        assert (
            window.level_panel.controls == {}
            or not state(window)["settings"]["collimate_enabled"]
        )
        record(application, window, lambda: state(window)["level_near_recorded"])
        record(application, window, lambda: state(window)["stage"] == "level_guide")
        wait(
            application,
            window,
            lambda: any(
                x.get("kind") == "horizontal_line"
                for x in window.overlays.get("level", [])
            ),
        )
        assert all(x.get("kind") != "target" for x in window.overlays["level"])
        # Move only the simulated mirror through actual UI controls.
        buttons = window.level_demo.findChildren(
            __import__("PySide6.QtWidgets", fromlist=["QPushButton"]).QPushButton
        )
        up = next(b for b in buttons if b.text() == "光斑上移")
        for _ in range(21):
            up.click()
        wait(application, window, lambda: state(window)["reached"])
        window.level_panel.buttons["confirm_far"].click()
        wait(application, window, lambda: state(window)["stage"] == "level_return")
        record(application, window, lambda: state(window)["stage"] == "level_done")
        assert state(window)["height_result"]["passed"]
        window.stop_plugin("level")
        assert not window.source.level_mode
    finally:
        window.close()
        application.processEvents()


def test_collimation_five_points_and_state_reset(application, tmp_path):
    window = Window(tmp_path, demo=True)
    window.show()
    try:
        start(application, window, True)
        for n in [1, 2]:
            record(application, window, lambda n=n: state(window)["coarse_count"] == n)
        assert state(window)["stage"] == "focus"
        window.level_focus.setValue(0)
        wait(application, window, lambda: state(window)["generation"] == 1)
        window.level_panel.buttons["start_validation"].click()
        for n in range(1, 6):
            if n == 4:
                wait(
                    application,
                    window,
                    lambda: "continue" in window.level_panel.buttons,
                )
                window.level_panel.buttons["continue"].click()
            record(application, window, lambda n=n: state(window)["count"] == n)
        assert state(window)["stage"] == "collimation_final"
        assert state(window)["joint_validation"]["passed"]
        assert state(window)["collimation_result"]["passed"]
        assert "start_level" not in window.level_panel.buttons
        window.level_focus.setValue(0.1)
        wait(application, window, lambda: state(window)["generation"] == 2)
        assert (
            state(window)["count"] == 0 and state(window)["collimation_result"] is None
        )
    finally:
        window.close()
        application.processEvents()

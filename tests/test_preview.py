import json
import time
from dataclasses import replace

import numpy as np
import pytest

from beamic_pro.fitting import fit_beam
from beamic_pro.preview import PreviewFrame
from conftest import image_frame


def preview(identity=1, center=(150, 120)):
    original = image_frame(frame_id=identity, center=center)
    return PreviewFrame(identity, original.pixels.copy(), time.monotonic_ns(), "preview-coordinates")


def pump_until(application, condition, seconds=3):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        application.processEvents()
        if condition():
            return
        time.sleep(.005)
    raise AssertionError("Timed out waiting for preview")


def test_unknown_exposure_and_scale_cannot_be_promoted_to_measurement():
    frame = preview()
    assert frame.exposure_us is None and frame.full_scale is None
    assert not frame.pixels.flags.writeable
    for kwargs in ({"exposure_us": 1000}, {"full_scale": 4095}, {"pixel_pitch_um": 3.45},
                   {"exposure_verified": True}, {"saturation_verified": True}):
        with pytest.raises(ValueError):
            replace(frame, **kwargs)
    fit = fit_beam(frame)
    np.testing.assert_allclose(fit.center, [150, 120], atol=.15)
    assert fit.radius_minor > 10
    assert not fit.valid
    assert "预览" in fit.reason


def test_preview_raster_reaches_view_and_localization_but_not_plugins(application, tmp_path):
    from beamic_pro.app import Window
    window = Window(tmp_path)
    window.show()
    class NeverPublish:
        alive = True
        def publish(self, *_):
            raise AssertionError("Unverified preview must not reach plugins")
        def close(self):
            pass
    try:
        generation = window.begin_source()
        window.plugins["guard"] = NeverPublish()
        window.capture_ready = True  # Even stale readiness must be cleared by preview.
        frame = preview()
        window.frames.put((generation, frame))
        pump_until(application, lambda: window.last_fit is not None)
        assert window.last_frame is frame
        assert not window.view.image_item.pixmap().isNull()
        assert any(item.data(0) == "circle_center" for item in window.view.items)
        assert not window.last_fit.valid and not window.capture_ready
        assert window.fit_rate.fps == 0
        assert "曝光未知" in window.exposure_label.text()
        assert not window.fixed_circle_button.isEnabled()
        frame2 = preview(2, (170, 130))
        window.frames.put((generation, frame2))
        pump_until(application, lambda: window.last_fit.frame_id == 2)
        np.testing.assert_allclose(window.last_fit.center, [170, 130], atol=.15)
        # Switching back to verified data cannot display a late preview fit.
        late = window.last_fit
        window.last_frame = image_frame(frame_id=3, key=frame.coordinate_key)
        window.last_fit = None
        window.fit_result((frame2, late, generation, window.selection_generation, .01))
        assert window.last_fit is None
    finally:
        window.close()


def test_preview_report_export_does_not_require_real_camera(application, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QFileDialog
    from beamic_pro.app import Window
    path = tmp_path / "report.json"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: (str(path), ""))
    window = Window(tmp_path)
    try:
        window.last_frame = preview()
        window.save_report()
        pump_until(application, path.exists)
        report = json.loads(path.read_text())
        assert report["preview_only"]
        assert report["images_saved"] is False
        assert "pixels" not in report
        assert "capabilities" in report
    finally:
        window.close()


def test_shared_memory_descriptor_selects_preview_without_fake_metadata():
    import threading
    from beamic_pro.bridge_client import BridgeClient
    frame = preview()
    shape = frame.shape
    size = 8 + frame.pixels.nbytes
    memory = bytearray(size)
    np.ndarray((1,), dtype=np.uint64, buffer=memory)[0] = 2
    np.ndarray(shape, dtype=np.float32, buffer=memory, offset=8)[:] = frame.pixels
    client = object.__new__(BridgeClient)
    client.map_lock = threading.Lock()
    client.maps = {"test": memory}
    descriptor = {"map_name": "test", "map_size": size, "offset": 0, "sequence": 2,
                  "shape": shape, "preview_only": True,
                  "frame": {"frame_id": 1, "coordinate_key": "preview", "exposure_us": None,
                            "full_scale": None, "received_ns": 0}}
    received = client.read_frame(descriptor)
    assert isinstance(received, PreviewFrame)
    np.testing.assert_array_equal(received.pixels, frame.pixels)
    assert received.received_ns > 0
    # A changed sequence is discarded before constructing a frame.
    np.ndarray((1,), dtype=np.uint64, buffer=memory)[0] = 3
    assert client.read_frame(descriptor) is None


def test_report_refreshes_runtime_diagnostics_before_writing(application, tmp_path, monkeypatch):
    import beamic_pro.app as app_module
    from PySide6.QtWidgets import QFileDialog
    path = tmp_path / "refreshed.json"
    commands = []
    class FakeBridge:
        def command(self, name):
            commands.append(name)
            return {"diagnostics": {"frame_samples": [{"marker": "new"}]}, "preview_only": True}
        def close(self):
            pass
    monkeypatch.setattr(app_module, "BridgeClient", FakeBridge)
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: (str(path), ""))
    window = app_module.Window(tmp_path)
    try:
        window.source = FakeBridge()
        window.last_frame = preview()
        window.save_report()
        pump_until(application, path.exists)
        report = json.loads(path.read_text())
        assert commands == ["probe"]
        assert report["capabilities"]["diagnostics"]["frame_samples"][0]["marker"] == "new"
        assert report["application_version"] == "0.5.1"
    finally:
        window.close()

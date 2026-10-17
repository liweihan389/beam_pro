import json
import threading
import time
from dataclasses import replace

import numpy as np
import pytest

from beamic_pro.diagnostics import DiagnosticRecorder
from beamic_pro.preview import PreviewFrame
from conftest import image_frame


def raw_frame(identity):
    pixels = np.array([[1.00000000000001, 2.123456789012345], [3., 4.]], dtype=np.float64)
    return PreviewFrame(identity, pixels, time.monotonic_ns(), "raw", metadata={"preview_only": True})


def wait_for(condition):
    deadline = time.monotonic() + 4
    while not condition():
        if time.monotonic() > deadline:
            raise AssertionError("Diagnostic writer timed out")
        time.sleep(.01)


def test_disabled_then_lossless_local_recording(tmp_path):
    recorder = DiagnosticRecorder()
    frame = raw_frame(1)
    recorder.offer(frame)
    assert not list(tmp_path.iterdir())
    folder = recorder.start(tmp_path, count=2)
    recorder.offer(frame)
    recorder.offer(raw_frame(2))
    wait_for(lambda: not recorder.active)
    manifest = json.loads((folder / "manifest.json").read_text())
    assert manifest["state"] == "complete" and manifest["saved"] == 2
    with np.load(folder / "frame-0001.npz", allow_pickle=False) as data:
        assert data["pixels"].dtype == np.float64
        np.testing.assert_array_equal(data["pixels"], frame.pixels)
        metadata = json.loads(str(data["metadata_json"]))
        assert metadata["exposure_us"] is None
        assert metadata["full_scale"] is None
    assert manifest["local_only"]
    with pytest.raises(ValueError):
        recorder.start(tmp_path, 21)


def test_slow_disk_has_bounded_queue_and_does_not_block_ingress(tmp_path, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    original = np.savez_compressed
    def slow(*args, **kwargs):
        entered.set()
        assert release.wait(3)
        return original(*args, **kwargs)
    monkeypatch.setattr(np, "savez_compressed", slow)
    recorder = DiagnosticRecorder()
    recorder.start(tmp_path, count=5)
    recorder.offer(raw_frame(1))
    assert entered.wait(1)
    start = time.monotonic()
    for i in range(2, 15):
        recorder.offer(raw_frame(i))
    assert time.monotonic() - start < .3
    assert recorder.items.qsize() <= 2
    assert recorder.status["dropped"] > 0
    recorder.stop()
    release.set()
    wait_for(lambda: not recorder.active)


def test_disk_error_and_timeout_preserve_partial_manifest(tmp_path, monkeypatch):
    recorder = DiagnosticRecorder()
    folder = recorder.start(tmp_path, timeout=.1)
    wait_for(lambda: not recorder.active)
    assert json.loads((folder / "manifest.json").read_text())["state"] == "partial"
    def fail(*a, **k):
        raise OSError("disk full")
    monkeypatch.setattr(np, "savez_compressed", fail)
    folder = recorder.start(tmp_path)
    recorder.offer(raw_frame(1))
    wait_for(lambda: not recorder.active)
    manifest = json.loads((folder / "manifest.json").read_text())
    assert manifest["state"] == "failed"
    assert "disk full" in manifest["reason"]


def test_scaled_preview_keeps_original_scene_coordinates(application, good_fit):
    from beamic_pro.view import BeamView
    frame = image_frame(shape=(1536, 2048), center=(1024, 768))
    view = BeamView()
    view.resize(600, 450)
    try:
        view.set_image(frame)
        assert view.render_shape[1] < 2048
        assert view.sceneRect().width() == 2048
        assert view.sceneRect().height() == 1536
        assert view.image_item.transform().m11() >= 2
        fit = replace(good_fit, center_x=1024, center_y=768)
        view.overlays(fit)
        assert any(item.data(0) == "circle_center" for item in view.items)
        view.set_fit_freshness(False, 1.3)
        assert any(item.data(0) == "stale_fit" for item in view.items)
        assert not any(item.data(0) == "live_center" for item in view.items)
        view.set_fit_freshness(False, 2.01)
        assert view.current_fit is None
        assert not any(item.data(0) == "circle_center" for item in view.items)
        # Explicit fixed reference survives expiry of the live result.
        view.set_fit_freshness(True, 0)
        view.overlays(fit)
        view.toggle_fixed_circle()
        view.set_fit_freshness(False, 3)
        assert view.fixed_fit is fit
        assert any(item.data(0) == "circle_center" for item in view.items)
    finally:
        view.close()


def test_old_control_generation_and_float64_transport():
    from beamic_pro.bridge_client import BridgeClient
    client = object.__new__(BridgeClient)
    client.map_lock = threading.Lock()
    client.control_generation = 3
    original = raw_frame(1)
    memory = bytearray(8 + original.pixels.nbytes)
    np.ndarray((1,), np.uint64, buffer=memory)[0] = 2
    np.ndarray(original.shape, np.float64, buffer=memory, offset=8)[:] = original.pixels
    client.maps = {"raw": memory}
    descriptor = {"map_name": "raw", "shape": original.shape, "dtype": "<f8", "offset": 0, "sequence": 2,
                  "control_generation": 2, "preview_only": True,
                  "frame": {"frame_id": 1, "coordinate_key": "raw", "received_ns": 0}}
    assert client.read_frame(descriptor) is None
    descriptor["control_generation"] = 3
    received = client.read_frame(descriptor)
    assert received.pixels.dtype == np.float64
    np.testing.assert_array_equal(received.pixels, original.pixels)


def test_plugin_receives_each_fits_own_frame_even_when_preview_advanced(application, tmp_path, good_fit):
    from types import SimpleNamespace
    from beamic_sdk import BeamFit
    from beamic_pro.app import Window
    good_fit = replace(good_fit, area_center_x=150., area_center_y=120.)
    window = Window(tmp_path)
    delivered = []
    publisher = window.publisher
    window.publisher = SimpleNamespace(publish=lambda frame: {"frame": frame.description()})
    window.plugins["test"] = SimpleNamespace(alive=True, publish=lambda descriptor, fit: delivered.append((descriptor, fit)))
    try:
        first = image_frame(frame_id=1)
        window.last_frame = image_frame(frame_id=2)
        window.fit_result((first, good_fit, window.source_generation, window.selection_generation, .1))
        assert delivered[0][0]["frame"]["frame_id"] == 1
        assert delivered[0][1] is good_fit
        assert not delivered[0][0]["frame"]["metadata"]["matches_display_frame"]
        assert not window.view.fit_current
        failed = BeamFit(2, reason="无信号")
        window.fit_result((window.last_frame, failed, window.source_generation, window.selection_generation, .1))
        assert window.view.current_fit is good_fit  # Reference drawing only.
        assert delivered[1][1] is failed  # Never reuse that reference as plugin input.
        window.last_control_generation = 1
        window.fit_result((image_frame(frame_id=3), replace(good_fit, frame_id=3),
                           window.source_generation, window.selection_generation, .1))
        assert len(delivered) == 2
        window.update_status()
        assert not window.view.fit_current
        window.capture_ready = True
        window.exposure.state.stable = True
        window.last_stable_fit_ns = time.monotonic_ns() - 3_000_000_000
        assert not window.allow_guidance_action("confirm_far")
        window.last_stable_fit_ns = time.monotonic_ns()
        assert window.allow_guidance_action("confirm_far")
        window.bridge_event((window.source_generation, {"type": "control_state", "generation": 2,
                                                       "phase": "transition", "reason": "stop"}))
        assert not window.allow_guidance_action("confirm_near")
    finally:
        window.plugins.clear()
        window.publisher = publisher
        window.close()
        application.processEvents()

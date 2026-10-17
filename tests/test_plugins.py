import json
import queue
import sys
import time
import zipfile
from pathlib import Path

import pytest
from conftest import image_frame

from beamic_pro.plugin_runtime import PluginProcess
from beamic_pro.plugin_store import PluginStore
from beamic_pro.stream import Latest, SharedPublisher


def package(path, identity="example", dependencies=None):
    manifest = {"id": identity, "name": identity, "version": "1", "sdk_version": 2,
                "entry": "example:Example", "dependencies": dependencies or [], "kind": "analysis"}
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("manifest.json", json.dumps(manifest))
        archive.writestr("example.py", "class Example: pass\n")
    return path


def test_failed_install_keeps_registry_and_old_environment(tmp_path):
    fail = {"value": False}
    calls = []
    def runner(command, progress):
        calls.append(command)
        if fail["value"] and "install" in command and "conflict>=2" in command:
            raise RuntimeError("ResolutionImpossible")
        return "numpy==2.3.5\n" if "freeze" in command else ""
    store = PluginStore(tmp_path / "store", Path(sys.executable), tmp_path / "sdk", runner)
    store.install(package(tmp_path / "first.zip"))
    original = store.registry.read_bytes()
    fail["value"] = True
    with pytest.raises(RuntimeError, match="ResolutionImpossible"):
        store.install(package(tmp_path / "bad.zip", "conflicting", ["conflict>=2"]))
    assert store.registry.read_bytes() == original
    assert len(list((store.root / "environments").iterdir())) == 0  # mocked venv has no filesystem effects
    assert any("numpy==2.3.5" in command for command in calls)


def test_bad_archive_cannot_escape_destination(tmp_path):
    package_path = tmp_path / "bad.zip"
    with zipfile.ZipFile(package_path, "w") as archive:
        archive.writestr("../escape.py", "no")
    with pytest.raises(ValueError, match="越界"):
        PluginStore.extract(package_path, tmp_path / "destination")
    assert not (tmp_path / "escape.py").exists()


def test_invalid_sdk_and_entry_rejected():
    manifest = {"id": "example", "name": "x", "version": "1", "sdk_version": 99,
                "entry": "example:Example", "dependencies": []}
    with pytest.raises(ValueError, match="SDK"):
        PluginStore.validate_manifest(manifest)


@pytest.mark.parametrize("shape", [(240, 320), (241, 321)])
def test_slow_worker_does_not_queue_every_frame(tmp_path, shape):
    (tmp_path / "slow.py").write_text("from beamic_sdk import Plugin\nimport time\nclass Slow(Plugin):\n def on_frame(self,frame,fit):\n  time.sleep(.2)\n  self.host.log(str(frame.frame_id))\n")
    events = queue.Queue()
    manifest = {"id": "slow", "entry": "slow:Slow"}
    sdk = Path(__file__).parents[1] / "sdk" / "src"
    worker = PluginProcess(Path(sys.executable), tmp_path, manifest, lambda identity, event: events.put(event), sdk)
    publisher = SharedPublisher()
    try:
        assert worker.ready.wait(5)
        for identity in range(1, 31):
            frame = image_frame(frame_id=identity, shape=shape)
            worker.publish(publisher.publish(frame), None)
            time.sleep(.005)
        deadline = time.monotonic() + 3
        ids = []
        while time.monotonic() < deadline:
            try:
                event = events.get(timeout=.1)
                if event["type"] == "log":
                    ids.append(int(event["message"]))
                    if ids[-1] == 30:
                        break
                assert event["type"] != "error", event
            except queue.Empty:
                pass
        assert ids and ids[-1] == 30
        assert len(ids) < 10
        assert worker.controls.empty()
    finally:
        worker.close()
        publisher.close()


def test_crashing_worker_reports_error_and_exits(tmp_path):
    (tmp_path / "crash.py").write_text("from beamic_sdk import Plugin\nclass Crash(Plugin):\n def on_activate(self): raise RuntimeError('intentional failure')\n")
    events = queue.Queue()
    sdk = Path(__file__).parents[1] / "sdk" / "src"
    worker = PluginProcess(Path(sys.executable), tmp_path, {"id": "crash", "entry": "crash:Crash"},
                           lambda identity, event: events.put(event), sdk)
    try:
        event = events.get(timeout=5)
        assert event["type"] == "error"
        assert "intentional failure" in event["message"]
    finally:
        worker.close()


def test_latest_mailbox_is_bounded():
    mailbox = Latest()
    for identity in range(10000):
        mailbox.put(identity)
    assert mailbox.take() == 9999
    assert mailbox.take() is None



def test_legacy_sdk_migration_preserves_old_environment(tmp_path):
    def runner(command, progress):
        return "numpy==2.3.5\n" if "freeze" in command else ""
    store = PluginStore(tmp_path / "store", Path(sys.executable), tmp_path / "sdk", runner)
    old = {"environment": "environments/old", "lock": ["numpy==2.3.5"],
           "plugins": {"legacy": {"manifest": {"sdk_version": 1}, "path": "packages/old"}}}
    store._commit(old)
    store.install(package(tmp_path / "new.zip"))
    state = store.read()
    assert state["sdk_version"] == 2
    assert state["previous_environment"] == old["environment"]
    assert state["legacy_plugins"]["legacy"] == old["plugins"]["legacy"]
    assert "legacy" not in state["plugins"]


def test_closed_worker_with_buffered_input_cannot_abort_host_cleanup():
    import subprocess
    import threading
    worker=PluginProcess.__new__(PluginProcess)
    worker.process=subprocess.Popen([sys.executable,'-c','pass'],stdin=subprocess.PIPE,
                                    stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    worker.process.wait(timeout=5)
    worker.process.stdin.write(b'buffered frame after worker exit')
    worker.controls=queue.Queue()
    worker.stop_event=threading.Event()
    worker._threads=[]
    worker.close()
    assert worker.closing and worker.stop_event.is_set()
    assert all(stream.closed for stream in (worker.process.stdin,worker.process.stdout,worker.process.stderr))

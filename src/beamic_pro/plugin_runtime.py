from __future__ import annotations

import json
import os
import queue
import subprocess
import threading
import time
from pathlib import Path

from .stream import Latest


class PluginProcess:
    def __init__(self, python: Path, directory: Path, manifest: dict, on_event, sdk_path=None):
        self.manifest, self.on_event = manifest, on_event
        environment = os.environ.copy()
        environment["PYTHONUTF8"] = "1"
        # Only development workers use source paths; installed environments contain the SDK wheel.
        if sdk_path:
            environment["PYTHONPATH"] = str(sdk_path)
        else:
            environment.pop("PYTHONPATH", None)
        self.process = subprocess.Popen(
            [str(python), "-m", "beamic_sdk.worker", str(directory), manifest["entry"]],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8", bufsize=1, env=environment,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        self.latest = Latest()
        self.controls = queue.Queue(maxsize=64)
        self.ready = threading.Event()
        self.stop_event = threading.Event()
        self.last_ack = time.monotonic()
        self.last_dispatch = 0.0
        self.dispatching = False
        self.closing = False
        self.started_at = time.monotonic()
        self._threads = [threading.Thread(target=f, daemon=True) for f in (self._read, self._write, self._stderr)]
        for thread in self._threads:
            thread.start()

    @property
    def alive(self):
        return self.process.poll() is None and not self.stop_event.is_set()

    def send(self, message):
        if not self.alive:
            raise RuntimeError("插件进程未运行")
        try:
            self.controls.put_nowait(message)
        except queue.Full as error:
            raise RuntimeError("插件操作过多，请等待当前操作完成") from error

    def publish(self, descriptor, fit):
        self.latest.put({"type": "frame", "descriptor": descriptor, "fit": fit.to_dict() if fit else None})

    def _write(self):
        try:
            while not self.stop_event.is_set():
                message = None
                try:
                    message = self.controls.get(timeout=0.01)
                except queue.Empty:
                    if self.ready.is_set():
                        message = self.latest.take()
                        if message:
                            self.ready.clear()
                            self.last_dispatch = time.monotonic()
                            self.dispatching = True
                if message:
                    self.process.stdin.write(json.dumps(message, ensure_ascii=False, allow_nan=False) + "\n")
                    self.process.stdin.flush()
                if self.dispatching and not self.ready.is_set() and time.monotonic() - self.last_dispatch > 5:
                    self.on_event(self.manifest["id"], {"type": "error", "message": "插件处理超时，已停用；图像流继续运行"})
                    self.stop_event.set()
                    self.process.kill()
                elif not self.ready.is_set() and not self.dispatching and time.monotonic() - self.started_at > 15:
                    self.on_event(self.manifest["id"], {"type": "error", "message": "插件启动超时，已停用"})
                    self.stop_event.set()
                    self.process.kill()
        except (BrokenPipeError, OSError, ValueError) as error:
            if not self.stop_event.is_set():
                self.on_event(self.manifest["id"], {"type": "error", "message": str(error)})

    def _read(self):
        try:
            for line in self.process.stdout:
                message = json.loads(line)
                if message.get("type") == "ready":
                    self.last_ack = time.monotonic()
                    self.dispatching = False
                    self.ready.set()
                else:
                    self.on_event(self.manifest["id"], message)
        except (ValueError, OSError) as error:
            self.on_event(self.manifest["id"], {"type": "error", "message": str(error)})
        finally:
            if not self.stop_event.is_set() and not self.closing:
                self.on_event(self.manifest["id"], {"type": "error", "message": "插件进程已退出"})

    def _stderr(self):
        for line in self.process.stderr:
            if line.strip():
                self.on_event(self.manifest["id"], {"type": "log", "message": line.strip()})

    def close(self):
        self.closing = True
        if self.process.poll() is None:
            try:
                self.controls.put_nowait({"type": "stop"})
                self.process.wait(timeout=1)
            except (subprocess.TimeoutExpired, queue.Full):
                self.process.kill()
                self.process.wait(timeout=2)
        self.stop_event.set()
        for thread in self._threads:
            if thread is not threading.current_thread():
                thread.join(timeout=.5)
        for stream in (self.process.stdin, self.process.stdout, self.process.stderr):
            try:
                stream.close()
            except OSError:
                # A terminated worker may leave an unflushed input frame. Its
                # closed pipe must not abort the host's remaining cleanup.
                pass


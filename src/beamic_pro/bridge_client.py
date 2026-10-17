from __future__ import annotations

import json
import mmap
import os
import queue
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
from beamic_sdk import Frame

from .preview import PreviewFrame


class BridgeClient:
    def __init__(self, executable: Path, profile: Path | None, event_callback):
        if os.name != "nt":
            raise RuntimeError("BeamMic 实机桥接需要 Windows；本机可以运行模拟光路")
        self.event_callback = event_callback
        self.process = subprocess.Popen([str(executable), *(["--profile", str(profile)] if profile else [])],
                                        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                        text=True, encoding="utf-8", errors="replace", bufsize=1,
                                        creationflags=subprocess.CREATE_NO_WINDOW)
        self.pending = {}
        self.command_id = 0
        self.lock = threading.Lock()
        self.maps = {}
        self.map_lock = threading.Lock()
        self.capabilities = {}
        self.control_generation = 0
        self.receive_count = 0
        self.copy_times_ms = []
        self.closed = False
        self._reader = threading.Thread(target=self._read, daemon=True)
        self._reader.start()
        self._stderr = threading.Thread(target=self._read_errors, daemon=True)
        self._stderr.start()

    def _read_errors(self):
        for line in self.process.stderr:
            self.event_callback({"type": "log", "message": line.rstrip()})

    def _read(self):
        try:
            for line in self.process.stdout:
                try:
                    message = json.loads(line)
                    if message.get("type") == "control_state":
                        self.control_generation = max(self.control_generation, int(message["generation"]))
                    if message.get("type") == "frame":
                        self.receive_count += 1
                        message["descriptor"]["_host_received_ns"] = time.monotonic_ns()
                    if message.get("type") == "reply":
                        result = self.pending.get(message["id"])
                        if result:
                            result.put(message)
                    else:
                        if message.get("type") == "capabilities":
                            self.capabilities = message["report"]
                        self.event_callback(message)
                except (ValueError, KeyError) as error:
                    self.event_callback({"type": "error", "message": "桥接消息无效：" + str(error)})
        finally:
            if not self.closed:
                self.event_callback({"type": "error", "message": "BeamMic 桥接已退出"})

    def command(self, method, **values):
        timeout = values.pop("_timeout", 10)
        with self.lock:
            self.command_id += 1
            identity = self.command_id
            result = queue.Queue(maxsize=1)
            self.pending[identity] = result
            message = {"id": identity, "method": method, **values}
            try:
                self.process.stdin.write(json.dumps(message, ensure_ascii=False) + "\n")
                self.process.stdin.flush()
            except (BrokenPipeError, OSError) as error:
                self.pending.pop(identity, None)
                raise RuntimeError("桥接连接已断开") from error
        try:
            reply = result.get(timeout=timeout)
            if not reply.get("ok"):
                raise RuntimeError(reply.get("error", "桥接操作失败"))
            return reply.get("result", {})
        except queue.Empty as error:
            raise RuntimeError("桥接操作超时") from error
        finally:
            self.pending.pop(identity, None)

    def read_frame(self, descriptor):
        with self.map_lock:
            return self._read_frame(descriptor)

    def _read_frame(self, descriptor):
        if int(descriptor.get("control_generation", 0)) < getattr(self, "control_generation", 0):
            return None
        received_ns = int(descriptor.get("_host_received_ns", time.monotonic_ns()))
        copy_started = time.monotonic_ns()
        name = descriptor["map_name"]
        if name not in self.maps:
            for old in self.maps.values():
                old.close()
            self.maps.clear()
            self.maps[name] = mmap.mmap(-1, int(descriptor["map_size"]), tagname=name, access=mmap.ACCESS_READ)
        mapping = self.maps[name]
        offset, expected = int(descriptor["offset"]), int(descriptor["sequence"])
        header = np.ndarray((1,), dtype=np.uint64, buffer=mapping, offset=offset)
        if int(header[0]) != expected or expected % 2:
            del header
            return None
        dtype = np.dtype(descriptor.get("dtype", "<f4"))
        if dtype.kind not in "fiu" or dtype.itemsize not in (1, 2, 4, 8):
            raise ValueError("桥接像素类型无效")
        pixels = np.ndarray(tuple(descriptor["shape"]), dtype=dtype, buffer=mapping, offset=offset + 8).copy()
        good = int(header[0]) == expected
        del header
        if not good:
            return None
        metadata = dict(descriptor["frame"])
        metadata["received_ns"] = received_ns
        duration_ms = (time.monotonic_ns() - copy_started) / 1e6
        if hasattr(self, "copy_times_ms"):
            self.copy_times_ms.append(duration_ms)
            del self.copy_times_ms[:-200]
        metadata["metadata"] = {**metadata.get("metadata", {}),
                                "host_mapping_copy_ms": duration_ms,
                                "transport_received_ns": received_ns}
        if descriptor.get("preview_only"):
            return PreviewFrame(pixels=pixels, **metadata)
        return Frame(pixels=pixels, **metadata)

    def set_exposure(self, value):
        return self.command("set_exposure", exposure_us=value)

    def set_rate(self, value):
        return self.command("set_rate", fps=value)

    def close(self):
        self.closed = True
        try:
            self.command("shutdown", _timeout=1)
        except RuntimeError:
            pass
        try:
            self.process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=2)
        with self.map_lock:
            for mapping in self.maps.values():
                mapping.close()
            self.maps.clear()


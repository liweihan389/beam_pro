"""Opt-in, bounded, local-only lossless snapshot recording."""
from __future__ import annotations

import json
import copy
import queue
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


class DiagnosticRecorder:
    def __init__(self, notify=lambda status: None):
        self.notify = notify
        self.lock = threading.Lock()
        self.thread = None
        self.cancel = threading.Event()
        self.active = False
        self.status = {"state": "off", "saved": 0, "dropped": 0}

    def start(self, parent, count=5, context=None, timeout=30):
        if not 1 <= int(count) <= 20:
            raise ValueError("每次诊断保存1～20张，默认5张")
        with self.lock:
            if self.thread and self.thread.is_alive():
                raise RuntimeError("上一轮诊断仍在结束，请稍后再试")
            folder = Path(parent) / ("beamic-diagnostic-" + datetime.now().strftime("%Y%m%d-%H%M%S-%f"))
            folder.mkdir(parents=True, exist_ok=False)
            self.folder, self.target = folder, int(count)
            self.items = queue.Queue(maxsize=2)
            self.cancel = threading.Event()
            self.seen = set()
            self.accepted = 0
            self.context = dict(context or {})
            self.timeout = min(120, max(.1, float(timeout)))
            self.started = time.monotonic()
            self.active = True
            self.status = {"state": "recording", "saved": 0, "dropped": 0, "requested": self.target,
                           "folder": str(folder), "local_only": True, "frames": [],
                           "started_utc": datetime.now(timezone.utc).isoformat(), "context": self.context}
            self.thread = threading.Thread(target=self._run, name="local-diagnostics", daemon=True)
            self.thread.start()
        self._notify()
        return folder

    def _notify(self):
        with self.lock:
            value = {k: v for k, v in self.status.items() if k != "frames"}
        self.notify(value)

    def offer(self, frame):
        # The ingress frame owns a read-only array. Never block acquisition or copy pixels here.
        with self.lock:
            if not self.active or self.cancel.is_set() or self.accepted >= self.target:
                return
            identity = (frame.metadata.get("control_generation", 0), frame.frame_id)
            if identity in self.seen:
                return
            metadata = {k: v for k, v in vars(frame).items() if k != "pixels"}
            metadata["metadata"] = dict(frame.metadata)
            try:
                self.items.put_nowait((frame.pixels, metadata))
            except queue.Full:
                self.status["dropped"] += 1
                return
            self.seen.add(identity)
            self.accepted += 1

    def _manifest(self):
        with self.lock:
            data = copy.deepcopy(self.status)
            data["images_saved"] = data["saved"] > 0
        temporary = self.folder / "manifest.json.tmp"
        temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        temporary.replace(self.folder / "manifest.json")

    def _run(self):
        try:
            self._manifest()
            while True:
                with self.lock:
                    finished = self.accepted >= self.target and self.items.empty()
                if finished or self.cancel.is_set() or time.monotonic() - self.started >= self.timeout:
                    break
                try:
                    pixels, metadata = self.items.get(timeout=.1)
                except queue.Empty:
                    continue
                with self.lock:
                    number = self.status["saved"] + 1
                name = f"frame-{number:04d}.npz"
                # Numeric .npy payload and JSON only; no pickle/object arrays, no precision conversion.
                np.savez_compressed(self.folder / name, pixels=pixels,
                                    metadata_json=np.array(json.dumps(metadata, ensure_ascii=False, allow_nan=False)))
                with self.lock:
                    self.status["saved"] += 1
                    self.status["frames"].append({"file": name, "frame_id": metadata["frame_id"],
                                                  "shape": list(pixels.shape), "dtype": pixels.dtype.str,
                                                  "raw_bytes": pixels.nbytes})
                self._manifest()
                self._notify()
            with self.lock:
                self.status["state"] = "complete" if self.status["saved"] == self.target else "partial"
                if self.status["state"] == "partial":
                    self.status["reason"] = "用户停止／采集条件改变" if self.cancel.is_set() else "超时，保留已保存部分"
        except Exception as error:
            with self.lock:
                self.status["state"] = "failed"
                self.status["reason"] = str(error)
        finally:
            with self.lock:
                self.active = False
            try:
                self._manifest()
            except OSError:
                pass
            # Release queued raster references even after disk failure/cancellation.
            while not self.items.empty():
                try:
                    self.items.get_nowait()
                except queue.Empty:
                    break
            self._notify()

    def stop(self, wait=False):
        self.cancel.set()
        if wait and self.thread:
            self.thread.join(timeout=5)

    def note(self, event):
        with self.lock:
            if not self.active:
                return
            notes = self.status.setdefault("events", [])
            notes.append({"host_monotonic_ns": time.monotonic_ns(), **event})
            del notes[:-100]

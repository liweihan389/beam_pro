from __future__ import annotations

import threading
import time
from collections import deque
from multiprocessing import shared_memory

import numpy as np
from beamic_sdk import Frame


class Latest:
    """Bounded mailbox: publishers overwrite stale pending work."""
    def __init__(self):
        self._value = None
        self._lock = threading.Lock()

    def put(self, value):
        with self._lock:
            self._value = value

    def take(self):
        with self._lock:
            value, self._value = self._value, None
        return value

    def clear(self):
        self.take()


class RateMeter:
    def __init__(self):
        self.times = deque()
        self.last_id = None

    def add(self, identity, now=None):
        if identity == self.last_id:
            return
        self.last_id = identity
        now = time.monotonic() if now is None else now
        self.times.append(now)
        self._trim(now)

    def _trim(self, now):
        while self.times and self.times[0] < now - 2:
            self.times.popleft()

    @property
    def fps(self):
        now = time.monotonic()
        self._trim(now)
        if len(self.times) < 2:
            return 0.0
        return (len(self.times) - 1) / max(now - self.times[0], 1e-6)


class AdaptiveRate:
    def __init__(self, requested=20.0):
        self.requested = float(requested)
        self.current = float(requested)
        self.last_check = time.monotonic()
        self.failures = 0

    def configure(self, requested):
        if not 1 <= requested <= 500:
            raise ValueError("目标帧率须在1到500之间；实际能力以设备为准")
        self.requested = self.current = float(requested)
        self.last_check = time.monotonic()
        self.failures = 0

    def observe(self, actual_fps, latency_ms, now=None):
        now = time.monotonic() if now is None else now
        if now - self.last_check < 3:
            return None
        self.last_check = now
        if actual_fps > 0 and (actual_fps < self.current * 0.75 or latency_ms > 200):
            self.failures += 1
        else:
            self.failures = 0
        if self.failures >= 2 and self.current > 1:
            self.current = max(1.0, round(self.current * 0.75, 1))
            self.failures = 0
            return self.current
        return None


class SharedPublisher:
    """Three-slot seqlock ring; a slow plugin never delays the image stream."""
    def __init__(self):
        self.shm = None
        self.capacity = 0
        self.slot = -1
        self.sequence = 0
        self.slot_span = 0
        self.retired = []

    def publish(self, frame: Frame):
        self.collect_retired()
        size = frame.pixels.nbytes
        if self.shm is None or size > self.capacity:
            if self.shm is not None:
                self.retired.append((self.shm, time.monotonic()))
            self.capacity = size
            self.slot_span = 8 + (size + 7) // 8 * 8
            self.shm = shared_memory.SharedMemory(create=True, size=3 * self.slot_span)
            self.slot = -1
        self.slot = (self.slot + 1) % 3
        offset = self.slot * self.slot_span
        header = np.ndarray((1,), dtype=np.uint64, buffer=self.shm.buf, offset=offset)
        self.sequence += 2
        header[0] = self.sequence - 1
        view = np.ndarray(frame.shape, dtype=frame.pixels.dtype, buffer=self.shm.buf, offset=offset + 8)
        view[:] = frame.pixels
        header[0] = self.sequence
        return {"name": self.shm.name, "offset": offset, "sequence": self.sequence,
                "shape": list(frame.shape), "dtype": frame.pixels.dtype.str,
                "frame": frame.description()}

    def collect_retired(self, force=False):
        retained = []
        for shm, created in self.retired:
            if force or time.monotonic() - created > 10:
                shm.close()
                shm.unlink()
            else:
                retained.append((shm, created))
        self.retired = retained

    def close(self):
        self.collect_retired(force=True)
        if self.shm is not None:
            self.shm.close()
            self.shm.unlink()
            self.shm = None


"""JSON-line plugin process. Numerical images arrive through shared memory."""
from __future__ import annotations

import importlib
import json
import os
import sys
import threading
import traceback
from multiprocessing import shared_memory

import numpy as np

from . import BeamFit, Frame, Host, Measurement


def read_frame(descriptor):
    # POSIX resource_tracker must not unlink host-owned memory on worker exit.
    if sys.version_info >= (3, 13):
        shm = shared_memory.SharedMemory(name=descriptor["name"], track=False)
    else:
        shm = shared_memory.SharedMemory(name=descriptor["name"])
        if os.name != "nt":
            from multiprocessing import resource_tracker
            resource_tracker.unregister(shm._name, "shared_memory")
    try:
        offset = int(descriptor["offset"])
        sequence = np.ndarray((1,), dtype=np.uint64, buffer=shm.buf, offset=offset)
        expected = int(descriptor["sequence"])
        if int(sequence[0]) != expected or expected % 2:
            del sequence
            return None
        pixels = np.ndarray(tuple(descriptor["shape"]), dtype=np.dtype(descriptor["dtype"]),
                            buffer=shm.buf, offset=offset + 8).copy()
        if int(sequence[0]) != expected:
            del sequence
            return None
        del sequence
        return Frame(pixels=pixels, **descriptor["frame"])
    finally:
        shm.close()


def main():
    directory, entry = sys.argv[1:3]
    protocol = sys.stdout
    sys.stdout = sys.stderr  # Plugin prints cannot corrupt protocol messages.
    lock = threading.Lock()
    def send(event):
        with lock:
            protocol.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")
            protocol.flush()
    plugin = None
    try:
        sys.path.insert(0, directory)
        module, symbol = entry.split(":", 1)
        plugin = getattr(importlib.import_module(module), symbol)(Host(send))
        plugin.on_activate()
        send({"type": "ready"})
        for line in sys.stdin:
            message = json.loads(line)
            command = message["type"]
            if command == "stop":
                break
            if command == "frame":
                frame = read_frame(message["descriptor"])
                if frame is not None:
                    fit = BeamFit.from_dict(message["fit"]) if message.get("fit") else None
                    plugin.on_frame(frame, fit)
                send({"type": "ready"})
            elif command == "action":
                plugin.on_action(message["action"], message.get("values", {}))
            elif command == "measurement":
                plugin.on_measurement(Measurement(**message["measurement"]))
    except Exception as error:
        send({"type": "error", "message": str(error), "traceback": traceback.format_exc()})
        return 1
    finally:
        if plugin is not None:
            try:
                plugin.on_deactivate()
            except Exception:
                traceback.print_exc(file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


from __future__ import annotations

from .types import BeamFit, Frame, Measurement


class Host:
    """Plugins send declarative events; camera settings remain host-owned."""

    def __init__(self, send):
        self._send = send

    def panel(self, title: str, text: str, actions: list[dict] | None = None, *,
              fields=None, table=None, plot=None, data=None):
        event = {"type": "panel", "title": title, "text": text, "actions": actions or []}
        for key, value in (("fields", fields), ("table", table), ("plot", plot), ("data", data)):
            if value is not None:
                event[key] = value
        self._send(event)

    def overlay(self, shapes: list[dict], frame_id: int | None = None):
        self._send({"type": "overlay", "shapes": shapes, "frame_id": frame_id})

    def request_measurement(self, role: str, samples: int = 20, reference_radius: float | None = None):
        if not role or not 2 <= samples <= 1000:
            raise ValueError("测量角色或帧数无效")
        self._send({"type": "measurement_request", "role": role, "samples": samples, "reference_radius": reference_radius})

    def measurement_mode(self):
        self._send({"type": "measurement_mode"})

    def log(self, message: str):
        self._send({"type": "log", "message": str(message)})


class Plugin:
    def __init__(self, host: Host):
        self.host = host

    def on_activate(self):
        pass

    def on_frame(self, frame: Frame, fit: BeamFit | None):
        pass

    def on_action(self, action: str, values: dict):
        pass

    def on_measurement(self, measurement: Measurement):
        pass

    def on_deactivate(self):
        pass


from __future__ import annotations

import threading
import time

import numpy as np
from beamic_sdk import Frame
from beamic_sdk.alignment import Geometry


class Simulator:
    """In-memory optical bench. Never masquerades as a BeamMic connection."""
    def __init__(self):
        self.rate = 20.0
        self.exposure_us = 1000.0
        self.epoch = 0
        self.frame_id = 0
        self.geometry = Geometry(100, 900, 120, 400)
        self.plane = "near"
        self.color = "red"
        self.offset = np.array([10.0, -7.0])
        self.slope = np.array([0.006, -0.004])
        self.near_kick = np.zeros(2)
        self.far_kick = np.zeros(2)
        self._stop = threading.Event()
        self._thread = None
        self.rng = np.random.default_rng(23)
        self.waist_mode = False
        self.level_mode = False
        self.level_position_mm = 100.
        self.level_tilt = .0004
        self.level_focus = .9
        self.waist_position_mm = 0.
        self.waist_truth = {"z_mm": (160., 180.), "radius_um": (55., 70.), "slope": (.00035, .00040)}
        self.capabilities = {"simulated": True, "ready": True, "source": "内存模拟光路",
                             "minimum_us": 50, "maximum_us": 10000, "step_us": 1,
                             "exposure_verified": True, "saturation_verified": True}

    def set_exposure(self, value):
        self.exposure_us = min(10000.0, max(50.0, float(value)))
        self.epoch += 1
        return {"exposure_us": self.exposure_us, "epoch": self.epoch}

    def set_rate(self, value):
        self.rate = float(value)

    def move(self, mirror, dx, dy):
        if mirror == "near":
            self.near_kick += np.array([dx, dy]) * 0.0003
        elif mirror == "far":
            self.far_kick += np.array([dx, dy]) * 0.0003

    def next_frame(self):
        self.frame_id += 1
        if self.level_mode:
            return self.level_frame()
        if self.waist_mode:
            return self.waist_frame()
        z = self.geometry.z_near if self.plane == "near" else self.geometry.z_far
        red = np.array([159.0, 119.0]) + np.array([z * 0.003, z * 0.001])
        center = red.copy()
        if self.color == "blue":
            center += self.offset + z * self.slope
            center += (z + self.geometry.mirror_near) * self.near_kick
            center += (z + self.geometry.mirror_far) * self.far_kick
        radius = 20.0 + z * 0.004
        y, x = np.mgrid[:240, :320]
        angle = 0.25
        u = np.cos(angle) * (x - center[0]) + np.sin(angle) * (y - center[1])
        v = -np.sin(angle) * (x - center[0]) + np.cos(angle) * (y - center[1])
        amplitude = (3300 if self.color == "red" else 2100) * self.exposure_us / 1000
        image = 12 + amplitude * np.exp(-2 * ((u / (radius * 1.4)) ** 2 + (v / radius) ** 2))
        image += self.rng.normal(0, 2.0, image.shape)
        image = np.clip(image, 0, 4095).astype(np.float32)
        return Frame(self.frame_id, image, time.monotonic_ns(), self.exposure_us,
                     self.epoch, "sim-320x240-1x-fixed", 4095,
                     pixel_pitch_um=5.2, exposure_verified=True, saturation_verified=True,
                     metadata={"simulated": True, "plane": self.plane, "color": self.color})

    def level_frame(self):
        z = self.level_position_mm
        pitch = 20.
        radii = np.array([1., .95])
        widths = np.sqrt((radii*(1+self.level_focus*.0006*z))**2+(.00078*z/(np.pi*radii))**2)*1000/pitch
        center = np.array([319.5, 239.5+z*(self.level_tilt+self.level_focus*.00003)*1000/pitch])
        y,x = np.mgrid[:480,:640]
        amplitude = 3100*self.exposure_us/1000
        pixels = 12+.003*x-.002*y+amplitude*np.exp(-2*((x-center[0])**2/widths[0]**2+(y-center[1])**2/widths[1]**2))
        pixels += self.rng.normal(0,1.,pixels.shape)
        pixels = np.clip(pixels,0,4095).astype(np.float32)
        return Frame(self.frame_id,pixels,time.monotonic_ns(),self.exposure_us,self.epoch,"sim-level-640x480-20um",4095,
                     pixel_pitch_um=pitch,exposure_verified=True,saturation_verified=True,
                     metadata={"simulated":True,"level_position_mm":z,"level_focus":self.level_focus})

    def waist_frame(self):
        z = self.waist_position_mm
        truth = self.waist_truth
        radius = np.sqrt(np.array(truth["radius_um"])**2+
                         (np.array(truth["slope"])*1000*(z-np.array(truth["z_mm"])))**2)/3.45
        y, x = np.mgrid[:360, :480]
        amplitude = 3200*self.exposure_us/1000
        pixels = 12+.003*x-.002*y+amplitude*np.exp(-2*((x-239.5)**2/radius[0]**2+(y-179.5)**2/radius[1]**2))
        pixels += self.rng.normal(0, 1.5, pixels.shape)
        pixels = np.clip(pixels, 0, 4095).astype(np.float32)
        return Frame(self.frame_id, pixels, time.monotonic_ns(), self.exposure_us,
                     self.epoch, "sim-waist-480x360-3.45um", 4095, pixel_pitch_um=3.45,
                     exposure_verified=True, saturation_verified=True,
                     metadata={"simulated": True, "waist_position_mm": z})

    def start(self, callback, error_callback=None):
        def run():
            deadline = time.monotonic()
            while not self._stop.is_set():
                try:
                    callback(self.next_frame())
                except Exception as error:
                    if error_callback:
                        error_callback(str(error))
                    break
                period = 1 / self.rate
                deadline += period
                now = time.monotonic()
                if deadline < now - period:
                    deadline = now
                self._stop.wait(max(0.001, deadline - now))
        self._stop.clear()
        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()

    def close(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)


import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import gc
import time

import numpy as np
import pytest
from beamic_sdk import BeamFit, Frame, Measurement

from beamic_pro.fitting import gaussian


@pytest.fixture(scope="session")
def application():
    from PySide6.QtCore import QCoreApplication, QEvent
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    app.setQuitOnLastWindowClosed(False)
    yield app
    gc.collect()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    app.processEvents()
    gc.collect()


def image_frame(center=(150, 120), radii=(30, 20), angle=0.3, noise=2, amplitude=3200,
                shape=(240, 320), frame_id=1, epoch=1, exposure=1000, key="fixed", verified=True):
    y, x = np.mgrid[:shape[0], :shape[1]]
    pixels = gaussian(x, y, amplitude, center[0], center[1], radii[0], radii[1], angle, 12)
    pixels += np.random.default_rng(frame_id).normal(0, noise, shape)
    pixels = np.clip(pixels, 0, 4095).astype(np.float32)
    return Frame(frame_id, pixels, time.monotonic_ns(), exposure, epoch, key, 4095,
                 exposure_verified=verified, saturation_verified=verified)


def point(role, center, radius=20, key="fixed"):
    return Measurement(role, *center, radius, radius * 1.2, 0, 20, 0.01, 0.002,
                       key, 1000, 1, 20)


@pytest.fixture
def good_fit():
    return BeamFit(1, 150, 120, 30, 20, 0.3, 3200, 12, .001, 0, True, "有效")


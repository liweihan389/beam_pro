"""Check the fixed-center controls and capture synthetic UI examples only."""
import json
import tempfile
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
from PySide6.QtWidgets import QApplication

from beamic_pro.app import Window, configure_application
from beamic_pro.measurement import Capture
from beamic_pro.validation import clean_case

ROOT = Path(__file__).resolve().parents[1]


def main():
    application = QApplication([])
    configure_application(application)
    def pump(seconds):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            application.processEvents()
            time.sleep(.003)
    with tempfile.TemporaryDirectory(prefix="beamic-center-controls-") as data_dir:
        window = Window(data_dir, demo=True)
        window.show()
        try:
            pump(1.4)
            assert window.last_fit and window.last_fit.valid
            assert not window.fixed_options.isVisible()
            window.fixed_circle_button.click()
            reference = window.view.fixed_fit
            assert window.track_center_input.isChecked()
            assert not window.connect_centers_input.isChecked()
            window.source.offset = np.array([30., -18.])
            window.demo_color.setCurrentIndex(1)
            pump(1.1)
            window.connect_centers_input.setChecked(True)
            pump(.1)
            figures = ROOT / "docs" / "figures"
            figures.mkdir(exist_ok=True)
            window.grab().save(str(figures / "ui-fixed-circle.png"))
            current = window.last_fit
            links = [item for item in window.view.items if item.data(0) == "center_link"]
            assert len(links) == 1
            line = links[0].line()
            assert (line.x1(), line.y1()) == tuple(reference.boundary_center)
            assert (line.x2(), line.y2()) == tuple(current.center)
            window.track_center_input.setChecked(False)
            pump(.1)
            assert not any(item.data(0) == "live_center" for item in window.view.items)
            assert any(item.data(0) == "center_link" for item in window.view.items)
            window.grab().save(str(figures / "ui-independent-link.png"))
            window.capture = Capture("display_only", 3, window.exposure.state.epoch,
                                     window.last_frame.coordinate_key, window.last_frame.frame_id)
            pump(.35)
            assert window.capture is None and "已记录 3 帧" in window.capture_label.text()
            window.track_center_input.setChecked(True)
            window.tabs.setCurrentIndex(2)
            pump(.1)
            window.grab().save(str(figures / "ui-location-settings.png"))
            window.tabs.setCurrentIndex(0)
            window.source.close()
            frame = replace(clean_case(amplitude=0).frame, frame_id=window.last_frame.frame_id + 1,
                            coordinate_key=window.last_frame.coordinate_key,
                            exposure_us=window.last_frame.exposure_us, exposure_epoch=window.last_frame.exposure_epoch)
            window.frames.put((window.source_generation, frame))
            pump(.2)
            assert window.last_fit and not window.last_fit.valid
            assert window.view.fixed_fit is reference
            assert any(item.data(0) == "circle_center" for item in window.view.items)
            assert not any(item.data(0) in ("center_link", "live_center") for item in window.view.items)
            window.grab().save(str(figures / "ui-tracking-lost.png"))
            report = {"simulated": True, "version": "0.3.0", "passed": True,
                      "circle_center_marked": True, "track_default": True, "connect_default": False,
                      "independent_link_verified": True, "measurement_continues_when_marker_hidden": True,
                      "missing_spot_hides_live_center_and_link": True, "fixed_reference_preserved": True,
                      "reference_center_px": reference.boundary_center.tolist(),
                      "live_center_px": current.center.tolist(), "hardware_acceptance": "pending"}
            (ROOT / "docs" / "center-controls-check.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            print(json.dumps(report, ensure_ascii=False, indent=2))
        finally:
            window.close()
            application.processEvents()


if __name__ == "__main__":
    main()

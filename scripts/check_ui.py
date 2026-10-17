"""Explicit development QA; uses synthetic frames only, never a real camera."""
import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
from PySide6.QtWidgets import QApplication

from beamic_pro.app import Window, configure_application


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--screenshot", type=Path)
    parser.add_argument("--seconds", type=float, default=8)
    args = parser.parse_args()
    application = QApplication([])
    configure_application(application)
    window = Window(args.data_dir, demo=True)
    window.show()
    started = time.monotonic()
    enabled = False
    while time.monotonic() - started < args.seconds:
        application.processEvents()
        if not enabled and window.last_fit and window.last_fit.valid and window.exposure.state.stable:
            window.activate("basic_analysis")
            window.start_alignment()
            enabled = True
        time.sleep(.002)
    application.processEvents()
    report = {"simulated": True, "hardware_acceptance": "pending", "duration_seconds": args.seconds,
              "received_fps": window.received_rate.fps, "valid_fit_fps": window.fit_rate.fps,
              "display_fps": window.display_rate.fps,
              "receive_to_display_p95_ms": float(np.percentile(window.latencies,95)) if window.latencies else None,
              "latency_scope": "host_receive_to_paint_only", "exposure_stable": window.exposure.state.stable,
              "guidance": window.guidance_text.text()}
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.screenshot:
        window.grab().save(str(args.screenshot))
    window.close()
    application.processEvents()
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


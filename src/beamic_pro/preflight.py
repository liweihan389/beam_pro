from __future__ import annotations

import argparse
import hashlib
import json
import platform
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np

from .bridge_client import BridgeClient


def validate_control_frames(baseline: list[dict], changed: list[dict], requested_us: float) -> tuple[bool, str]:
    """Reject setting-only metadata masquerading as frame-associated exposure."""
    if len(baseline) < 5 or len(changed) < 5:
        return False, "有效帧不足，无法验证曝光关联"
    old_us = float(np.median([f["exposure_us"] for f in baseline]))
    old_energy = float(np.median([f["energy"] for f in baseline]))
    if old_energy <= 0 or abs(requested_us / old_us - 1) < .15:
        return False, "信号不足或曝光变化太小"
    consistent_new = 0
    for frame in changed:
        measured_us, energy = frame["exposure_us"], frame["energy"]
        ratio = energy / old_energy
        if abs(measured_us / old_us - 1) < .02:
            expected = 1.0
        elif abs(measured_us / requested_us - 1) < .02:
            expected = requested_us / old_us
            consistent_new += 1
        else:
            return False, "帧曝光值与旧值、新设置都不匹配"
        if abs(ratio - expected) > max(.08, .25 * expected):
            return False, "帧元数据与实际图像亮度变化不一致，禁止用当前设置值代替逐帧曝光"
    return (True, "曝光元数据与图像变化一致") if consistent_new >= 5 else (False, "新曝光对应的帧不足")


def run_preflight(executable, profile, seconds=10, control_test=False):
    events, summaries = [], deque(maxlen=400)
    event_lock = threading.Lock()
    holder = []
    def receive(value):
        stamp = time.monotonic()
        if value.get("type") == "frame" and holder:
            frame = holder[0].read_frame(value["descriptor"])
            if frame is not None:
                stride = max(1, max(frame.shape) // 200)
                sample = frame.pixels[::stride, ::stride]
                bg = float(np.percentile(sample, 20))
                summary = {"stamp": stamp, "frame_id": frame.frame_id,
                           "vendor_frame_id": frame.metadata.get("vendor_frame_id"),
                           "coordinate_key": frame.coordinate_key,
                           "energy": float(np.maximum(sample - bg, 0).sum()),
                           "exposure_us": frame.exposure_us, "shape": list(frame.shape),
                           "exposure_verified": frame.exposure_verified,
                           "saturation_verified": frame.saturation_verified,
                           "peak_fraction": float(sample.max() / frame.full_scale)}
                with event_lock:
                    summaries.append(summary)
        with event_lock:
            if value.get("type") != "frame":
                events.append((stamp, value))
    report = {"status": "not_run", "simulated": False, "platform": platform.platform(),
              "hardware_acceptance": "pending", "images_saved": False,
              "control_test_passed": False}
    if profile and Path(profile).exists():
        report["profile_sha256"] = hashlib.sha256(Path(profile).read_bytes()).hexdigest()
    if platform.system() != "Windows":
        report.update(status="requires_windows", reason="此预检需要实验室 Windows 电脑和已安装的原版 BeamMic")
        return report
    client, original_exposure = None, None
    try:
        client = BridgeClient(Path(executable), Path(profile) if profile else None, receive)
        holder.append(client)
        deadline = time.monotonic() + 10
        while not client.capabilities and time.monotonic() < deadline:
            if client.process.poll() is not None:
                raise RuntimeError("桥接未成功启动")
            time.sleep(0.05)
        client.command("probe")
        client.command("start")
        deadline = time.monotonic() + min(seconds, 5)
        while not client.capabilities.get("exposure_verified") and time.monotonic() < deadline:
            time.sleep(.05)
        client.command("probe")
        if client.capabilities.get("exposure_verified") and client.capabilities.get("exposure_control"):
            client.command("claim_exposure")
        time.sleep(seconds)
        client.command("probe")
        report["capabilities"] = dict(client.capabilities)
        report["assembly_version"] = client.capabilities.get("assembly_version")
        report["source"] = client.capabilities.get("source")
        with event_lock:
            baseline = list(summaries)
        report["frames_received"] = len(baseline)
        report["actual_fps"] = (len(baseline)-1) / (baseline[-1]["stamp"]-baseline[0]["stamp"]) if len(baseline)>1 else 0
        report["unique_frame_ids"] = len({f["frame_id"] for f in baseline})
        report["coordinate_key"] = baseline[-1]["coordinate_key"] if baseline else None
        ready = len(baseline) >= 5 and all(f["exposure_verified"] and f["saturation_verified"] for f in baseline)
        ready = ready and len({f["coordinate_key"] for f in baseline}) == 1
        ready = ready and client.capabilities.get("exposure_control", False)
        report["status"] = "ready_for_control_test" if ready else "incompatible_or_unverified"
        if control_test and ready:
            baseline = baseline[-10:]
            if any(f["peak_fraction"] > .98 or f["peak_fraction"] < .05 for f in baseline):
                raise RuntimeError("曝光测试需稳定、未饱和且足够明亮的单束连续光；请先在原版软件调整光强")
            original_exposure = float(np.median([f["exposure_us"] for f in baseline]))
            requested = max(float(client.capabilities["minimum_us"]), original_exposure * .5)
            reply = client.set_exposure(requested)
            changed_after = time.monotonic()
            time.sleep(max(2, min(seconds, 5)))
            with event_lock:
                changed = [f for f in summaries if f["stamp"] >= changed_after]
            passed, reason = validate_control_frames(baseline, changed, reply["exposure_us"])
            report["control_test_passed"] = passed
            report["control_test_reason"] = reason
            report["status"] = "ready" if passed else "exposure_association_failed"
        with event_lock:
            report["errors"] = [item["message"] for _, item in events if item.get("type") == "error"]
        report["note"] = "此报告不证明真实合束达标；还需测量接口/显示延迟与近远处合束误差。"
    except Exception as error:
        report.update(status="failed", reason=str(error))
    finally:
        if client:
            if original_exposure is not None:
                try:
                    client.set_exposure(original_exposure)
                except RuntimeError as error:
                    report["restore_error"] = str(error)
            client.close()
    return report


def main():
    parser = argparse.ArgumentParser(description="BeamMic 兼容性预检（不保存图像）")
    parser.add_argument("--bridge", default="BeamMicBridge.exe")
    parser.add_argument("--profile")
    parser.add_argument("--seconds", type=float, default=10)
    parser.add_argument("--control-test", action="store_true", help="在稳定单束连续光下减半曝光，检查逐帧元数据关联，并恢复曝光")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run_preflight(args.bridge, args.profile, min(60, max(1, args.seconds)), args.control_test)
    content = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(content, encoding="utf-8")
    print(content)
    return 0 if report["status"] in ("ready", "ready_for_control_test") else 2


if __name__ == "__main__":
    raise SystemExit(main())


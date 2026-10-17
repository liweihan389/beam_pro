from beamic_pro.preflight import run_preflight, validate_control_frames


def test_setting_only_exposure_metadata_is_rejected():
    baseline = [{"exposure_us": 1000, "energy": 10000} for _ in range(5)]
    falsely_labeled = [{"exposure_us": 500, "energy": 10000} for _ in range(5)]
    assert not validate_control_frames(baseline, falsely_labeled, 500)[0]


def test_delayed_old_frames_are_accepted_only_with_old_metadata():
    baseline = [{"exposure_us": 1000, "energy": 10000} for _ in range(5)]
    changed = [{"exposure_us": 1000, "energy": 10000} for _ in range(2)]
    changed += [{"exposure_us": 500, "energy": 5000} for _ in range(5)]
    assert validate_control_frames(baseline, changed, 500)[0]


def test_non_windows_preflight_cannot_claim_hardware_ready(monkeypatch):
    monkeypatch.setattr("beamic_pro.preflight.platform.system", lambda: "Darwin")
    report = run_preflight("missing.exe", None)
    assert report["status"] == "requires_windows"
    assert not report["control_test_passed"]
    assert report["hardware_acceptance"] == "pending"


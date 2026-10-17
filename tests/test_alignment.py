import numpy as np
import pytest
from beamic_sdk.alignment import Geometry, compute_targets, verify
from conftest import point


@pytest.mark.parametrize("values", [(1, 1, 2, 3), (-1, 3, 2, 4), (1, 3, 0, 4), (1, 3, 4, 4), (1, 3, 5, 4), (float("nan"), 3, 2, 4)])
def test_invalid_geometry(values):
    with pytest.raises(ValueError):
        Geometry(*values)


def test_two_step_targets_eliminate_both_position_and_slope():
    rng = np.random.default_rng(13)
    for _ in range(1000):
        dn = rng.uniform(10, 300)
        df = dn + rng.uniform(20, 500)
        z1 = rng.uniform(0, 200)
        z2 = z1 + rng.uniform(100, 2000)
        geometry = Geometry(z1, z2, dn, df)
        e0, slope = rng.uniform(-60, 60, 2), rng.uniform(-.04, .04, 2)
        # Independent camera translations at the two planes cancel in red-blue differences.
        r1, r2 = rng.uniform(100, 300, 2), rng.uniform(50, 200, 2)
        e1, e2 = e0 + z1 * slope, e0 + z2 * slope
        data = {"near_red": point("near_red", r1), "near_blue": point("near_blue", r1 + e1),
                "far_red": point("far_red", r2), "far_blue": point("far_blue", r2 + e2)}
        targets = compute_targets(geometry, data)
        intermediate_error = np.array(targets.intermediate) - r2
        near_kick = (intermediate_error - e2) / (z2 + dn)
        far_kick = -intermediate_error / (z2 + df)
        for z in (0, z1, z2, 2 * z2):
            residual = e0 + z * slope + near_kick * (z + dn) + far_kick * (z + df)
            np.testing.assert_allclose(residual, 0, atol=1e-9)


def test_intermediate_is_not_generally_red_center():
    data = {r: point(r, c) for r, c in [("near_red", (100, 100)), ("near_blue", (110, 105)),
                                       ("far_red", (100, 100)), ("far_blue", (120, 110))]}
    targets = compute_targets(Geometry(100, 1000, 100, 300), data)
    assert targets.intermediate != targets.final
    data["far_blue"] = point("far_blue", (120, 110), key="rotated")
    with pytest.raises(ValueError, match="坐标"):
        compute_targets(Geometry(100, 1000, 100, 300), data)


def test_verify_uses_each_local_red_short_radius():
    data = {"near_red": point("near_red", (100, 100), 10), "near_blue": point("near_blue", (100.4, 100)),
            "far_red": point("far_red", (100, 100), 40), "far_blue": point("far_blue", (101.5, 100))}
    assert verify(data)["passed"]
    data["near_blue"] = point("near_blue", (100.6, 100))
    assert not verify(data)["passed"]
    assert verify(data)["far"]["passed"]



def test_measurement_definitions_cannot_be_mixed():
    from dataclasses import replace
    data = {role: point(role, (100, 100)) for role in ("near_red", "near_blue", "far_red", "far_blue")}
    for changed in ({"center_mode": "area"}, {"boundary_fraction": .30}, {"boundary_kind": "gaussian_1e2"}):
        invalid = dict(data)
        invalid["far_blue"] = replace(data["far_blue"], **changed)
        with pytest.raises(ValueError, match="定义"):
            compute_targets(Geometry(100, 900, 120, 400), invalid)
        with pytest.raises(ValueError, match="定义"):
            verify(invalid)

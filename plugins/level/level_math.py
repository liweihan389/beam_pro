"""Single-mirror levelling and finite-interval D4sigma propagation.

All intervals are conditional on user-supplied error budgets. Near-flat curves
remain usable without dividing by curvature or claiming a waist location.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def geometry(near, far, tolerance=0):
    if (
        not np.isfinite([near, far, tolerance]).all()
        or tolerance < 0
        or near < 0
        or far <= near
    ):
        raise ValueError("须满足0≤近处距离<远处距离，公差不能为负")
    if far - near <= max(1, 2 * tolerance):
        raise ValueError("近远间距须大于两倍位置公差，且大于1 mm")


def height_target(
    near,
    far,
    y1,
    y2,
    sigma1,
    sigma2,
    pitch_um,
    height_tolerance_mm=0.01,
    position_tolerance_mm=10,
    scale_pct=2,
    draws=10000,
):
    geometry(near, far, position_tolerance_mm)
    if (
        not np.isfinite(
            [y1, y2, sigma1, sigma2, pitch_um, height_tolerance_mm, scale_pct]
        ).all()
        or min(sigma1, sigma2, height_tolerance_mm) < 0
        or pitch_um <= 0
        or not 0 <= scale_pct <= 20
    ):
        raise ValueError("中心、像元或误差预算无效")
    rng = np.random.default_rng(531)
    lengths = rng.uniform(
        [near - position_tolerance_mm, far - position_tolerance_mm],
        [near + position_tolerance_mm, far + position_tolerance_mm],
        (draws, 2),
    )
    scale = (
        pitch_um / 1000 * (1 + rng.uniform(-scale_pct / 100, scale_pct / 100, draws))
    )
    h = np.array([y1, y2])[None] + rng.normal(0, [sigma1, sigma2], (draws, 2))
    h += (
        rng.uniform(-height_tolerance_mm, height_tolerance_mm, (draws, 2))
        / scale[:, None]
    )
    targets = h[:, 1] - lengths[:, 1] / (lengths[:, 1] - lengths[:, 0]) * (
        h[:, 1] - h[:, 0]
    )
    nominal = y2 - far / (far - near) * (y2 - y1)
    return {
        "y_px": float(nominal),
        "interval_px": np.quantile(targets, [0.025, 0.975]).tolist(),
        "tilt_mrad": float((y2 - y1) * pitch_um / (far - near)),
        "near_expected_px": float(nominal),
    }


def verify_height(
    near,
    far,
    y1,
    y2,
    sigma1,
    sigma2,
    pitch_um,
    limit_mm=0.05,
    height_tolerance_mm=0.01,
    position_tolerance_mm=10,
    scale_pct=2,
    draws=10000,
):
    geometry(near, far, position_tolerance_mm)
    if min(pitch_um, limit_mm) <= 0 or min(sigma1, sigma2, height_tolerance_mm) < 0:
        raise ValueError("等高验收参数无效")
    rng = np.random.default_rng(543)
    scales = (
        pitch_um / 1000 * (1 + rng.uniform(-scale_pct / 100, scale_pct / 100, draws))
    )
    difference = (y2 - y1 + rng.normal(0, np.hypot(sigma1, sigma2), draws)) * scales
    difference += np.diff(
        rng.uniform(-height_tolerance_mm, height_tolerance_mm, (draws, 2)), axis=1
    )[:, 0]
    lengths = (
        far
        - near
        + np.diff(
            rng.uniform(-position_tolerance_mm, position_tolerance_mm, (draws, 2)),
            axis=1,
        )[:, 0]
    )
    upper = float(np.quantile(np.abs(difference), 0.95))
    return {
        "passed": upper <= limit_mm,
        "difference_mm": float((y2 - y1) * pitch_um / 1000),
        "difference_interval_mm": np.quantile(difference, [0.025, 0.975]).tolist(),
        "absolute_difference_95_mm": upper,
        "limit_mm": limit_mm,
        "tilt_interval_mrad": np.quantile(
            difference / lengths * 1000, [0.025, 0.975]
        ).tolist(),
    }


def _fit(z, widths, covariance, shift, span):
    x = (z - shift) / span
    design = np.stack([x * x, x, np.ones_like(x)], axis=-1)
    inverse = np.linalg.inv(covariance)
    lhs = np.einsum("dni,nm,dmj->dij", design, inverse, design)
    rhs = np.einsum("dni,nm,dm->di", design, inverse, widths**2)
    coefficients = np.linalg.solve(lhs, rhs[..., None])[..., 0]
    residual = widths**2 - np.einsum("dni,di->dn", design, coefficients)
    score = np.einsum("dn,nm,dm->d", residual, inverse, residual)
    return coefficients, score


def extrema(coefficients, left, right):
    """Include the interior vertex; no clipping/dropping bad simulated curves."""
    a, b, c = np.moveaxis(coefficients, -1, 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        vertex = -b / (2 * a)
    vertex = np.where((vertex >= left) & (vertex <= right), vertex, left)
    values = np.stack(
        [
            a * left**2 + b * left + c,
            a * right**2 + b * right + c,
            a * vertex**2 + b * vertex + c,
        ],
        axis=-1,
    )
    return values.min(axis=-1), values.max(axis=-1)


@dataclass
class IntervalModel:
    coefficients: np.ndarray
    draws: np.ndarray
    datum: np.ndarray
    scales: np.ndarray
    right_draws: np.ndarray
    shift: float
    span: float
    goodness: bool
    physical: str

    def evaluate(self, z):
        x = (np.asarray(z) - self.shift) / self.span
        return np.sqrt(
            np.maximum(
                0,
                self.coefficients[:, 0, None] * x * x
                + self.coefficients[:, 1, None] * x
                + self.coefficients[:, 2, None],
            )
        )

    def variation(self, limit_pct=5):
        lo, hi = extrema(self.coefficients, -0.5, 0.5)
        nominal = np.where(
            lo > 0, np.sqrt(np.maximum(hi, 0) / np.maximum(lo, 1e-20)) - 1, np.inf
        )
        lower, upper = extrema(
            self.draws, -0.5, (self.right_draws[:, None] - self.shift) / self.span
        )
        with np.errstate(invalid="ignore", divide="ignore"):
            samples = np.sqrt(upper / lower) - 1
        samples = np.where((lower > 0) & np.isfinite(samples), samples, np.inf)
        joint = np.max(samples, axis=1)
        # Order statistic keeps infinite/invalid upper tails conservative.
        bound = float(
            np.sort(joint)[min(len(joint) - 1, int(np.ceil(0.95 * len(joint))) - 1)]
        )
        return {
            "nominal_pct": (nominal * 100).tolist(),
            "joint_upper_95_pct": bound * 100 if np.isfinite(bound) else None,
            "passed": bool(
                self.goodness
                and self.physical != "conflict"
                and bound <= limit_pct / 100
            ),
            "limit_pct": limit_pct,
            "invalid_fraction": float(np.mean(~np.isfinite(joint))),
            "physical": self.physical,
        }


def fit_interval(points, wavelength_nm=780, scale_pct=2, draws=10000, seed=20261015):
    if len(points) < 3:
        raise ValueError("至少需要三个不同位置，包含近端、远端及中间")
    z = np.array([p.z_mm for p in points])
    tol = np.array([p.tolerance_mm for p in points])
    widths = np.array([p.diameters_um for p in points])
    noise = np.array([p.sigma_um for p in points])
    if (
        not np.isfinite(
            np.r_[z, tol, widths.ravel(), noise.ravel(), wavelength_nm, scale_pct]
        ).all()
        or np.any(tol < 0)
        or np.any(widths <= 0)
        or np.any(noise <= 0)
        or wavelength_nm <= 0
        or not 0 <= scale_pct <= 20
    ):
        raise ValueError("位置、波长、尺寸或误差预算无效")
    span = z[1]
    geometry(0, span, float(tol.max()))
    if z[0] != 0 or np.any(z < 0) or np.any(z > span) or len(np.unique(z)) != len(z):
        raise ValueError("首点为近处0，第二点为远处，各点须在区间内且位置不重复")
    if min(abs(a - b) for i, a in enumerate(z) for b in z[i + 1 :]) <= 2 * tol.max():
        raise ValueError("任意两点的间距须大于两倍位置公差")
    rng = np.random.default_rng(seed)
    errors = rng.uniform(-tol, tol, (draws, len(points)))
    positions = z + errors - errors[:, :1]
    scales = 1 + rng.uniform(-scale_pct / 100, scale_pct / 100, draws)
    values = (widths + rng.normal(0, noise, (draws, *widths.shape))) * scales[
        :, None, None
    ]
    covz = np.diag(tol**2 / 3) + tol[0] ** 2 / 3 * np.ones((len(points), len(points)))
    covz[0, :] = covz[:, 0] = 0
    coefficients = []
    samples = []
    scores = []
    shift = span / 2
    for axis in range(2):
        initial = np.polyfit((z - shift) / span, widths[:, axis] ** 2, 2)
        gradient = (2 * initial[0] * (z - shift) / span + initial[1]) / span
        covariance = (
            np.diag((2 * widths[:, axis] * noise[:, axis]) ** 2)
            + gradient[:, None] * covz * gradient[None, :]
        )
        nominal, score = _fit(z[None], widths[None, :, axis], covariance, shift, span)
        sampled, _ = _fit(positions, values[:, :, axis], covariance, shift, span)
        coefficients.append(nominal[0])
        samples.append(sampled)
        scores.append(float(score[0]))
    coefficients = np.array(coefficients)
    samples = np.stack(samples, axis=1)
    minimum, _ = extrema(coefficients, -0.5, 0.5)
    if np.any(minimum <= 0):
        raise ValueError("区间内预测尺寸无效，请重录或补测")
    invariant = (
        samples[:, :, 0] * samples[:, :, 2] - samples[:, :, 1] ** 2 / 4
    ) / span**2
    physical = (samples[:, :, 0] > 0) & (invariant >= (4 * wavelength_nm / np.pi) ** 2)
    physical_fraction = float(np.mean(np.all(physical, axis=1)))
    classification = (
        "conflict"
        if physical_fraction < 0.005
        else "consistent"
        if physical_fraction >= 0.95
        else "unresolved"
    )
    threshold = (
        float(np.quantile(rng.chisquare(len(points) - 3, 50000), 0.975))
        if len(points) > 3
        else np.inf
    )
    return IntervalModel(
        coefficients,
        samples,
        errors[:, 0],
        scales,
        positions[:, 1],
        shift,
        span,
        all(s <= threshold for s in scores),
        classification,
    )


def predict(model, z, tolerance=0, seed=154):
    rng = np.random.default_rng(seed)
    positions = z + rng.uniform(-tolerance, tolerance, len(model.draws)) - model.datum
    x = (positions - model.shift) / model.span
    q = (
        model.draws[:, :, 0] * x[:, None] ** 2
        + model.draws[:, :, 1] * x[:, None]
        + model.draws[:, :, 2]
    )
    if np.any(q <= 0):
        raise ValueError("冻结预测包含无效尺寸，需补测建立新预测")
    return np.sqrt(q)


def validate_pair(model, points):
    if len(points) != 2 or points[0].z_mm == points[1].z_mm:
        raise ValueError("需要两个不同的新位置进行独立验证")
    checks = []
    for i, p in enumerate(points):
        predicted = predict(model, p.z_mm, p.tolerance_mm, seed=154 + i)
        rng = np.random.default_rng(640 + i)
        observed = (
            np.array(p.diameters_um) + rng.normal(0, p.sigma_um, predicted.shape)
        ) * model.scales[:, None]
        limits = np.quantile(predicted - observed, [0.00625, 0.99375], axis=0)
        checks.append(
            {
                "passed": bool(np.all(limits[0] <= 0) & np.all(limits[1] >= 0)),
                "innovation_joint_um": limits.T.tolist(),
            }
        )
    return {"passed": all(c["passed"] for c in checks), "points": checks}


def plot_payload(model, points):
    z = np.linspace(0, model.span, 100)
    x = (z - model.shift) / model.span
    curves = []
    for axis, label in enumerate(("X方向", "Y方向")):
        q = (
            model.draws[:, axis, 0, None] * x * x
            + model.draws[:, axis, 1, None] * x
            + model.draws[:, axis, 2, None]
        )
        values = np.sqrt(np.maximum(q, 0))
        low, high = np.quantile(values, [0.025, 0.975], axis=0)
        curves.append(
            {
                "label": label,
                "x": z.tolist(),
                "y": model.evaluate(z)[axis].tolist(),
                "low": low.tolist(),
                "high": high.tolist(),
            }
        )
    return {
        "curves": curves,
        "points": [
            {
                "z": p.z_mm,
                "diameters": list(p.diameters_um),
                "tolerance": p.tolerance_mm,
            }
            for p in points
        ],
        "note": "仅评估所选近远区间；阴影为95%参数区间，不承诺全程零发散。",
    }

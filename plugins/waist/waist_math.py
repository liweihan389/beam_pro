"""Second-moment caustics and propagation of externally assigned uncertainties.

Intervals are conditional on the entered error budget, not ISO certification.
The first sensor plane is the origin. Reading errors share its datum error.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class WidthPoint:
    z_mm: float
    tolerance_mm: float
    diameters_um: tuple[float, float]
    sigma_um: tuple[float, float]
    samples: int = 20
    first_frame: int = 0
    last_frame: int = 0


def _moments(signal, bounds):
    x0, y0, x1, y1 = bounds
    image = signal[y0:y1, x0:x1]
    yy, xx = np.mgrid[y0:y1, x0:x1]
    total = image.sum()
    if total <= 0:
        raise ValueError("扣背景后有效光能不足")
    cx, cy = (image * xx).sum() / total, (image * yy).sum() / total
    vx = (image * (xx-cx)**2).sum()/total
    vy = (image * (yy-cy)**2).sum()/total
    if min(vx, vy) <= 0:
        raise ValueError("二阶矩无效，请检查背景")
    return np.array([cx, cy]), np.sqrt([vx, vy])


def measure_width(frame, selection=None):
    """Signed background-subtracted integrals; never use a geometric contour."""
    image = np.asarray(frame.pixels, dtype=float)
    xoff = yoff = 0
    if selection is not None:
        xoff, yoff, x1, y1 = map(int, selection)
        image = image[yoff:y1, xoff:x1]
    h, w = image.shape
    if min(h, w) < 24:
        raise ValueError("积分选区太小，请包含完整光斑及背景")
    if (image >= frame.full_scale*.999).any():
        raise ValueError("图像饱和，暂停宽度记录")
    yy, xx = np.mgrid[:h, :w]
    edge = (xx < w*.12) | (xx >= w*.88) | (yy < h*.12) | (yy >= h*.88)
    indices = np.flatnonzero(edge)[::max(1, int(edge.sum())//16000)]
    design = np.column_stack([np.ones(len(indices)), xx.ravel()[indices]/w, yy.ravel()[indices]/h])
    values = image.ravel()[indices]
    keep = np.ones(len(values), dtype=bool)
    for _ in range(4):
        beta = np.linalg.lstsq(design[keep], values[keep], rcond=None)[0]
        residual = values-design@beta
        noise = max(.1, 1.4826*np.median(np.abs(residual[keep]-np.median(residual[keep]))))
        keep = np.abs(residual-np.median(residual[keep])) < 3.5*noise
        if keep.sum() < 20:
            raise ValueError("背景不足，不能可靠计算二阶矩")
    signal = image-(beta[0]+beta[1]*xx/w+beta[2]*yy/h)
    peak = signal.max()
    if peak < 30*noise:
        raise ValueError("宽度测量信噪比不足")
    provisional = np.where(signal > max(6*noise, peak*.015), signal, 0)
    center, sigma = _moments(provisional, (0, 0, w, h))

    def bounds(factor):
        start = np.floor(center-factor*sigma).astype(int)
        end = np.ceil(center+factor*sigma).astype(int)+1
        if np.any(start < 0) or np.any(end > [w, h]):
            raise ValueError("光斑尾部或积分选区被截断，请扩大画面／选区")
        return int(start[0]), int(start[1]), int(end[0]), int(end[1])

    for _ in range(4):
        region = bounds(4.5)
        next_center, next_sigma = _moments(signal, region)
        change = np.max(np.abs(next_sigma/sigma-1))
        center, sigma = next_center, next_sigma
        if change < .0005:
            break
    _, wide_sigma = _moments(signal, bounds(5.5))
    if np.max(np.abs(wide_sigma/sigma-1)) > .02:
        raise ValueError("宽度随积分范围变化超过2%，请检查尾部、背景或杂光")
    diameter = 4*sigma
    if diameter.min() < 10:
        raise ValueError("最小方向不足10个像素，束腰采样不足")
    return diameter, center+np.array([xoff, yoff])


def summarize(z, tolerance, widths, identities, pitch_um, magnification):
    values = np.asarray(widths)*pitch_um/magnification
    n = len(values)
    sigma = values.std(axis=0, ddof=1)
    if np.any(sigma/values.mean(axis=0) > .03):
        raise ValueError("本点宽度波动超过3%，请等待光束／曝光稳定后重录")
    # Positive serial correlation reduces the effective number of frames.
    centered = values-values.mean(axis=0)
    denominator = np.sum(centered[:-1]**2, axis=0)
    rho = np.divide(np.sum(centered[:-1]*centered[1:], axis=0), denominator,
                    out=np.zeros(2), where=denominator > 0)
    rho = np.clip(rho, 0, .9)
    effective = np.maximum(2, n*(1-rho)/(1+rho))
    mean = values.mean(axis=0)
    uncertainty = np.maximum(sigma/np.sqrt(effective), mean*.002)
    return WidthPoint(float(z), float(tolerance), tuple(mean), tuple(uncertainty),
                      n, identities[0], identities[-1])


def _quadratics(z, widths, covariance, shift, span):
    # A fixed whitening transform retains the shared zero-reading covariance.
    x = (z-shift)/span
    a = np.stack([x*x, x, np.ones_like(x)], axis=-1)
    inverse = np.linalg.inv(covariance)
    lhs = np.einsum('dni,nm,dmj->dij', a, inverse, a)
    rhs = np.einsum('dni,nm,dm->di', a, inverse, widths**2)
    regularization = np.eye(3)[None]*np.max(np.diagonal(lhs, axis1=1, axis2=2), axis=1)[:, None, None]*1e-13
    coeff = np.linalg.solve(lhs+regularization, rhs[..., None])[..., 0]
    aa, bb, cc = coeff.T
    with np.errstate(invalid='ignore', divide='ignore'):
        z0 = shift-span*bb/(2*aa)
        minimum = cc-bb*bb/(4*aa)
        d0 = np.sqrt(np.maximum(minimum, 0))
        slope = np.sqrt(np.maximum(aa, 0))/span
    valid = (aa > 0) & (minimum > np.max(widths**2, axis=1)*1e-10) & np.isfinite(z0)
    residual = widths**2-np.einsum('dni,di->dn', a, coeff)
    score = np.einsum('dn,nm,dm->d', residual, inverse, residual)
    return np.stack([z0, d0, slope], axis=-1), valid, score


@dataclass
class Caustic:
    parameters: np.ndarray
    draws: np.ndarray
    datum_errors: np.ndarray
    scale_factors: np.ndarray
    intervals: np.ndarray
    valid_fraction: float
    goodness: bool
    scores: list[float]

    def axis(self, index):
        z, diameter, slope = self.parameters[index]
        ci = self.intervals[:, index, :]
        return {"z_mm": float(z), "diameter_um": float(diameter),
                "z_interval_mm": ci[:, 0].tolist(), "diameter_interval_um": ci[:, 1].tolist(),
                "rayleigh_mm": float(diameter/slope)}

    def precision(self, diameter_pct=5, position_mm=5):
        lo, hi = self.intervals
        half_width = np.maximum(np.abs(lo-self.parameters), np.abs(hi-self.parameters))
        return bool(np.all(half_width[:, 0] <= position_mm) and
                    np.all(half_width[:, 1]/self.parameters[:, 1] <= diameter_pct/100))


def fit_caustic(points, scale_tolerance_pct=2, draws=2500, seed=20261014):
    if len(points) < 3:
        raise ValueError("至少需要三个不同位置")
    z = np.array([p.z_mm for p in points])
    tol = np.array([p.tolerance_mm for p in points])
    widths = np.array([p.diameters_um for p in points])
    noise = np.array([p.sigma_um for p in points])
    if z[0] != 0 or len(np.unique(z)) != len(z):
        raise ValueError("首次位置必须为0，各次位置不能重复")
    if not np.isfinite(np.r_[z, tol, widths.ravel(), noise.ravel()]).all() or (tol < 0).any() or (widths <= 0).any() or (noise <= 0).any():
        raise ValueError("位置、公差或尺寸数据无效")
    if not 0 <= scale_tolerance_pct <= 20:
        raise ValueError("共同宽度系统公差必须在0–20%之间")
    span = np.ptp(z)
    if span <= max(1, 2*tol.max()):
        raise ValueError("取样距离范围太小，需大于两倍摆放公差")
    shift = np.mean(z)
    # l_i = reading_i - reading_1; l_1 is exactly zero by definition.
    cov_z = np.diag(tol**2/3)+tol[0]**2/3*np.ones((len(z), len(z)))
    cov_z[0, :] = cov_z[:, 0] = 0
    rng = np.random.default_rng(seed)
    errors = rng.uniform(-tol, tol, size=(draws, len(z)))
    sampled_z = z+errors-errors[:, :1]
    scales = 1+rng.uniform(-scale_tolerance_pct/100, scale_tolerance_pct/100, draws)
    sampled_widths = (widths+rng.normal(0, noise, size=(draws, *widths.shape)))*scales[:, None, None]
    models, masks, nominal, scores = [], [], [], []
    for axis in range(2):
        # Estimate width-square sensitivity for weighting; no residual-derived
        # noise estimate is used, including when there are exactly three points.
        preliminary = np.polyfit((z-shift)/span, widths[:, axis]**2, 2)
        gradient = (2*preliminary[0]*(z-shift)/span+preliminary[1])/span
        covariance = np.diag((2*widths[:, axis]*noise[:, axis])**2)+gradient[:, None]*cov_z*gradient[None, :]
        base, physical, score = _quadratics(z[None], widths[None, :, axis], covariance, shift, span)
        if not physical[0]:
            raise ValueError("初测未形成正曲率且最小尺寸为正的传播曲线，请补测／重录")
        model, mask, _ = _quadratics(sampled_z, sampled_widths[:, :, axis], covariance, shift, span)
        models.append(model)
        masks.append(mask)
        nominal.append(base[0])
        scores.append(float(score[0]))
    good = masks[0] & masks[1]
    fraction = float(good.mean())
    if fraction < .90:
        raise ValueError(f"仅{fraction:.0%}误差模拟得到物理解，三点不足以可靠锁定束腰，请扩大范围／补测")
    model_draws = np.stack(models, axis=1)[good]
    threshold = float(np.quantile(rng.chisquare(len(z)-3, 20000), .975)) if len(z) > 3 else np.inf
    return Caustic(np.array(nominal), model_draws, errors[good, 0], scales[good],
                   np.quantile(model_draws, [.025, .975], axis=0), fraction,
                   bool(all(s <= threshold for s in scores)), scores)


def predict(result, z, tolerance=0, seed=17):
    rng = np.random.default_rng(seed)
    positions = z+rng.uniform(-tolerance, tolerance, len(result.draws))-result.datum_errors
    model = result.draws
    prediction = np.sqrt(model[:, :, 1]**2+(model[:, :, 2]*(positions[:, None]-model[:, :, 0]))**2)
    return prediction


def validate_point(result, point):
    predicted = predict(result, point.z_mm, point.tolerance_mm)
    rng = np.random.default_rng(43)
    observed = (np.array(point.diameters_um)+rng.normal(0, point.sigma_um, predicted.shape))*result.scale_factors[:, None]
    innovation = predicted-observed
    # Bonferroni: two held-out positions x two fixed image axes, nominal 95%
    # simultaneous coverage, conditional on the supplied uncertainty model.
    limits = np.quantile(innovation, [.00625, .99375], axis=0)
    passed = bool(np.all(limits[0] <= 0) and np.all(limits[1] >= 0))
    return {"passed": passed, "prediction_95_um": np.quantile(predicted, [.025, .975], axis=0).T.tolist(),
            "innovation_joint_um": limits.T.tolist()}


def suggested_positions(result, points):
    z0 = float(result.parameters[:, 0].mean())
    rayleigh = float(np.max(result.parameters[:, 1]/result.parameters[:, 2]))
    far = z0+2*rayleigh
    if any(abs(far-p.z_mm) < max(1, p.tolerance_mm*2) for p in points):
        far = z0-2*rayleigh
    return round(z0, 1), round(far, 1)


def curve_payload(result, points):
    near, far = suggested_positions(result, points)
    left = min([p.z_mm for p in points]+[near, far])
    right = max([p.z_mm for p in points]+[near, far])
    z = np.linspace(left-(right-left)*.05, right+(right-left)*.05, 100)
    curves = []
    for axis, name in enumerate(('X方向', 'Y方向')):
        model = result.draws[:, axis]
        values = np.sqrt(model[:, 1, None]**2+(model[:, 2, None]*(z[None]-model[:, 0, None]))**2)
        low, high = np.quantile(values, [.025, .975], axis=0)
        center = result.parameters[axis]
        mean = np.sqrt(center[1]**2+(center[2]*(z-center[0]))**2)
        curves.append({"label": name, "x": z.tolist(), "y": mean.tolist(), "low": low.tolist(), "high": high.tolist()})
    return {"curves": curves, "points": [{"z": p.z_mm, "tolerance": p.tolerance_mm,
            "diameters": list(p.diameters_um), "sigma": list(p.sigma_um)} for p in points],
            "note": "线：拟合；阴影：参数95%区间；点：实测；横线：摆放公差"}

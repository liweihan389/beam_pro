"""Reproducible sensitivity checks, separate from camera or coverage validation."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "plugins/level"))
from level_math import fit_interval, validate_pair
from width_math import WidthPoint


def main():
    rng = np.random.default_rng(15494)
    z = np.array([0, 800, 400, 200, 600.0])
    records = []
    for i in range(120):
        radii = rng.uniform(900, 1200, 2)
        waists = rng.uniform(-100, 500, 2)
        truth = 2 * np.sqrt(
            radii**2 + (0.78 * 1000 * (z[:, None] - waists) / (np.pi * radii)) ** 2
        )
        widths = truth + rng.normal(0, truth * 0.002)
        points = [
            WidthPoint(float(zi), 10, tuple(w), tuple(t * 0.002))
            for zi, w, t in zip(z, widths, truth, strict=True)
        ]
        frozen = fit_interval(points[:3], draws=10000, seed=9200 + i)
        validation = validate_pair(frozen, points[3:])
        final = fit_interval(points, draws=10000, seed=9400 + i).variation()
        records.append(
            {"index": i, "independent_consistent": validation["passed"], "final": final}
        )
    report = {
        "date": "2026-10-15",
        "version": "0.5.0",
        "simulated": True,
        "hardware_acceptance": "pending",
        "conditional_trials": len(records),
        "independent_pair_consistent": sum(
            r["independent_consistent"] for r in records
        ),
        "notes": (
            "Sensitivity check on generated Gaussian beams under stated noise; "
            "not a hardware accuracy or guaranteed coverage proof. "
            "Noise is generated in width only; position and scale budgets are "
            "assumed in inference, not sampled in these trial inputs. "
            "The final variation flag alone does not imply acceptance: "
            "the plugin additionally requires the held-out pair to pass."
        ),
        "seed": 15494,
        "draws_per_fit": 10000,
        "positions_mm": z.tolist(),
        "noise_relative_sigma": 0.002,
        "records": records,
    }
    (ROOT / "docs/level-numerical-check.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in report.items() if k != "records"}, indent=2))


if __name__ == "__main__":
    main()

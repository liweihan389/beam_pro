"""Evaluate the unchanged production fitter on public experimental images.

No reference annotations are supplied to the fitter. All outputs describe single
archived frames, not camera throughput, physical accuracy or exposure control.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import platform
import re
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import PIL
import scipy
from beamic_sdk import Frame
from PIL import Image
from scipy import ndimage

from beamic_pro import __version__
from beamic_pro.fitting import (
    FitSettings,
    _background,
    _crossings,
    _ellipse_distance,
    fit_beam,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "build/real-spots"
OUTPUT = ROOT / "docs/real-spots"


def read_image(path, source):
    """Keep native geometry; use encoding limits, never observed peak as full scale."""
    with Image.open(path) as image:
        rgb = image.mode in ("RGB", "RGBA")
        pixels = np.asarray(image.convert("L") if rgb else image)
        bits = image.tag_v2.get(258, (8,))[0] if hasattr(image, "tag_v2") else None
    transform = "RGB转灰度亮度；未作线性相机响应校正" if rgb else "原始灰度，未缩放或裁剪"
    if path.suffix.lower() == ".pgm":
        header = re.match(rb"P5\s+(\d+)\s+(\d+)\s+(\d+)\s", path.read_bytes()[:128])
        if not header:
            raise ValueError("unsupported PGM header")
        full_scale = int(header.group(3))
        if full_scale == 65535:
            # The author's documentation specifies 12-bit samples in high bits.
            pixels = pixels >> 4
            full_scale = 4095
            transform = "按作者说明将16位容器右移4位，保留12位数值；未清除原图异常像素"
    else:
        full_scale = (1 << (bits or (8 if pixels.dtype == np.uint8 else 16))) - 1
    if pixels.ndim != 2:
        raise ValueError("not a two dimensional intensity image")
    return pixels.astype(np.float32), float(full_scale), transform


def oxford_reference(values):
    amplitude, x, y, a, b, angle, background = values
    # Dataset pixels are centered at .5, 1.5, ...; the app uses 0, 1, ... .
    if b > a:
        a, b, angle = b, a, angle + math.pi / 2
    scale = math.sqrt(-math.log(.135) / 2)
    return {"x": x - .5, "y": y - .5, "major": a * scale, "minor": b * scale,
            "angle": (angle + math.pi / 2) % math.pi - math.pi / 2,
            "amplitude": amplitude, "background": background}


def load_cases():
    cases = []
    for annotation in sorted((DATA / "oxford").rglob("data_*.json")):
        table = json.loads(annotation.read_text())
        base = annotation.parent.parent
        for key in table["run"]:
            row = {column: values[key] for column, values in table.items()}
            path = base / row["gray_paths"].replace("\\", "/").lstrip("/")
            references = [oxford_reference(v) for v in row["fits"]]
            cases.append({"id": f"oxford-{row['run']}", "source": "oxford", "path": str(path.relative_to(DATA)),
                          "split": "train" if "train" in annotation.name else "validation",
                          "beam_count": row["num_beams"], "references": references})
    annotated_paths = {case["path"] for case in cases}
    for path in sorted((DATA / "oxford").rglob("*.tiff")):
        if str(path.relative_to(DATA)) not in annotated_paths:
            cases.append({"id": f"oxford-{path.stem}", "source": "oxford", "path": str(path.relative_to(DATA)),
                          "split": "unannotated", "beam_count": None, "references": []})
    for path in sorted((DATA / "laserbeamsize").glob("*")):
        if path.suffix in (".pgm", ".png"):
            cases.append({"id": "lbs-" + path.stem, "source": "laserbeamsize",
                          "path": str(path.relative_to(DATA)), "references": []})
    for path in sorted((DATA / "nist").rglob("LaserSpotMeasure70x70.tif")):
        cases.append({"id": "nist-spot", "source": "nist", "path": str(path.relative_to(DATA)), "references": []})
    for annotation in sorted((DATA / "rgb").rglob("instances_default.json")):
        table = json.loads(annotation.read_text())
        by_image = defaultdict(list)
        for entry in table["annotations"]:
            if entry["category_id"] == 1:
                by_image[entry["image_id"]].append(entry["bbox"])
        for image in table["images"]:
            path = annotation.parent.parent / "images" / image["file_name"]
            group = "phone" if "test_comparison" in str(annotation) else "camera"
            cases.append({"id": f"rgb-{group}-{image['id']}", "source": "rgb", "split": group,
                          "path": str(path.relative_to(DATA)), "references": [], "boxes": by_image[image["id"]]})
    return cases


def contour_metrics(pixels, fit):
    """Geometric agreement with the observed contour, not independent accuracy."""
    if fit.radius_minor <= 0:
        return {}
    y, x = np.mgrid[:pixels.shape[0], :pixels.shape[1]]
    if fit.background_origin is None:
        background, _ = _background(pixels)
    else:
        background = fit.background_origin + fit.background_slope_x * x + fit.background_slope_y * y
    signal = ndimage.gaussian_filter(ndimage.median_filter(pixels - background, size=3), .65)
    structure = ndimage.generate_binary_structure(2, 2)
    labels, count = ndimage.label(signal >= signal.max() * math.exp(-2), structure)
    energies = ndimage.sum(np.maximum(signal, 0), labels, range(1, count + 1))
    reference = labels == (int(np.argmax(energies)) + 1)
    level = signal.max() * .135
    labels, _ = ndimage.label(signal >= level, structure)
    selected = np.unique(labels[reference & (signal >= level)])
    observed = ndimage.binary_fill_holes(np.isin(labels, selected[selected > 0]))
    points = _crossings(signal, level, observed)
    residual = _ellipse_distance(points, [*fit.boundary_center, fit.radius_major, fit.radius_minor, fit.angle_rad])
    c, s = math.cos(fit.angle_rad), math.sin(fit.angle_rad)
    dx, dy = x - fit.boundary_center[0], y - fit.boundary_center[1]
    ideal = ((c * dx + s * dy) / fit.radius_major) ** 2 + ((-s * dx + c * dy) / fit.radius_minor) ** 2 <= 1
    return {"contour_iou": float(np.count_nonzero(ideal & observed) / max(1, np.count_nonzero(ideal | observed))),
            "all_contour_rms_fraction": float(np.sqrt(np.mean(residual ** 2)) / fit.radius_minor),
            "contour_point_count": len(points)}


def evaluate(case):
    path = DATA / case["path"]
    pixels, full_scale, transform = read_image(path, case["source"])
    frame = Frame(1, pixels, 0, 1, 0, "archived-offline", full_scale,
                  metadata={"real_image": True, "exposure_unknown": True})
    started = time.perf_counter()
    fit = fit_beam(frame, settings=FitSettings())
    elapsed = time.perf_counter() - started
    row = {**case, "shape": list(pixels.shape), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
           "full_scale": full_scale, "transform": transform, "fit": fit.to_dict(), "fit_seconds": elapsed,
           **contour_metrics(pixels, fit)}
    if case["source"] == "oxford" and case["beam_count"] == 1 and fit.radius_minor > 0:
        ref = case["references"][0]
        error = float(np.linalg.norm(fit.center - [ref["x"], ref["y"]]))
        area_error = float(np.linalg.norm(np.array([fit.area_center_x, fit.area_center_y]) - [ref["x"], ref["y"]]))
        row.update(center_error_px=error, center_error_fraction=error / ref["minor"], area_center_error_px=area_error,
                   major_relative_error=abs(fit.radius_major / ref["major"] - 1),
                   minor_relative_error=abs(fit.radius_minor / ref["minor"] - 1))
        row["within_5pct_reference_minor"] = bool(fit.valid and error <= .05 * ref["minor"])
        if ref["major"] / ref["minor"] >= 1.15:
            row["angle_error_deg"] = abs(math.degrees((fit.angle_rad - ref["angle"] + math.pi / 2) % math.pi - math.pi / 2))
    if case["source"] == "rgb":
        row["inside_annotated_box"] = bool(fit.valid and any(
            x <= fit.center_x + .5 <= x + w and y <= fit.center_y + .5 <= y + h for x, y, w, h in case["boxes"]))
    return row


def describe(rows):
    valid = [r for r in rows if r["fit"]["valid"]]
    result = {"total": len(rows), "accepted": len(valid), "rejected": len(rows) - len(valid),
              "reasons": dict(Counter(r["fit"]["reason"] for r in rows if not r["fit"]["valid"]))}
    for field in ["center_error_px", "area_center_error_px", "center_error_fraction", "major_relative_error",
                  "minor_relative_error", "angle_error_deg", "contour_iou", "all_contour_rms_fraction"]:
        values = np.array([r[field] for r in valid if r.get(field) is not None])
        if len(values):
            result[field] = {"n": len(values), "median": float(np.median(values)), "p95": float(np.percentile(values, 95)),
                             "min": float(values.min()), "max": float(values.max()), "rmse": float(np.sqrt(np.mean(values ** 2)))}
    if rows and rows[0]["source"] == "oxford" and all(r["beam_count"] == 1 for r in rows):
        result["within_5pct_reference_minor"] = sum(r.get("within_5pct_reference_minor", False) for r in rows)
    if rows and rows[0]["source"] == "rgb":
        result["annotated_images"] = sum(bool(r["boxes"]) for r in rows)
        result["accepted_inside_box"] = sum(r["inside_annotated_box"] for r in rows)
        result["unannotated_images"] = sum(not r["boxes"] for r in rows)
    return result


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    cases = load_cases()
    assert sum(c["source"] == "oxford" for c in cases) == 1500
    assert sum(c["source"] == "oxford" and c["split"] != "unannotated" for c in cases) == 1050
    assert len({c["path"] for c in cases}) == len(cases)
    fitter = ROOT / "src/beamic_pro/fitting.py"
    frozen_hash = hashlib.sha256(fitter.read_bytes()).hexdigest()
    rows = []
    with ProcessPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(evaluate, case): case for case in cases}
        for future in as_completed(futures):
            rows.append(future.result())
            if len(rows) % 100 == 0:
                print(f"Completed {len(rows)}/{len(cases)}", flush=True)
    assert hashlib.sha256(fitter.read_bytes()).hexdigest() == frozen_hash
    rows.sort(key=lambda r: r["id"])
    summary = {source: describe([r for r in rows if r["source"] == source])
               for source in sorted({r["source"] for r in rows})}
    summary["oxford_annotated"] = describe([r for r in rows if r["source"] == "oxford" and r["split"] != "unannotated"])
    summary["oxford_unannotated"] = describe([r for r in rows if r["source"] == "oxford" and r["split"] == "unannotated"])
    for count in range(1, 6):
        summary[f"oxford_{count}_beam"] = describe([r for r in rows if r["source"] == "oxford" and r["beam_count"] == count])
    for split in ["train", "validation"]:
        summary[f"oxford_single_{split}"] = describe([r for r in rows if r["source"] == "oxford" and r["beam_count"] == 1 and r["split"] == split])
    manifest = json.loads((DATA / "download-manifest.json").read_text())
    result = {"date": "2026-10-15", "version": __version__, "settings": vars(FitSettings()),
              "fitter_sha256": frozen_hash, "real_archived_data": True, "hardware_acceptance": "pending",
              "environment": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__, "pillow": PIL.__version__},
              "summary": summary, "download_manifest": manifest, "rows": rows}
    (OUTPUT / "results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    fields = ["id", "source", "split", "path", "valid", "reason", "center_x", "center_y", "radius_major", "radius_minor",
              "center_error_px", "area_center_error_px", "center_error_fraction", "major_relative_error", "minor_relative_error",
              "angle_error_deg", "contour_iou", "all_contour_rms_fraction", "inside_annotated_box", "full_scale", "transform", "sha256"]
    with (OUTPUT / "results.csv").open("w", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(output, fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, **row["fit"]})
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

"""Check dataset coordinate and encoding adapters independently of real data."""
import importlib.util
import math
from pathlib import Path

import numpy as np

SPEC = importlib.util.spec_from_file_location("real_spots", Path(__file__).parents[1] / "scripts/check_real_spots.py")
real_spots = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(real_spots)


def test_oxford_pixel_centers_and_intensity_boundary_conversion():
    ref = real_spots.oxford_reference([20, 10.5, 8.5, 8, 4, .3, 3])
    assert (ref["x"], ref["y"]) == (10, 8)
    assert math.isclose(math.exp(-2 * (ref["major"] / 8) ** 2), .135)
    assert math.isclose(math.exp(-2 * (ref["minor"] / 4) ** 2), .135)


def test_oxford_axis_swap_preserves_ellipse_geometry():
    ref = real_spots.oxford_reference([20, 10.5, 8.5, 4, 8, .3, 3])
    assert ref["major"] > ref["minor"]
    assert math.isclose((ref["angle"] - .3) % math.pi, math.pi / 2)


def test_padded_12bit_pgm_uses_encoding_limit_instead_of_observed_peak(tmp_path):
    path = tmp_path / "input.pgm"
    data = np.array([[0, 500, 1000], [2000, 3000, 4095]], dtype=">u2")
    path.write_bytes(b"P5\n3 2\n65535\n" + (data.astype(np.uint16) << 4).astype(">u2").tobytes())
    pixels, scale, _ = real_spots.read_image(path, "laserbeamsize")
    np.testing.assert_array_equal(pixels, data)
    assert scale == 4095


def test_rgb_conversion_keeps_all_channels_and_dimensions(tmp_path):
    from PIL import Image
    path = tmp_path / "input.png"
    Image.new("RGB", (12, 9), (255, 0, 0)).save(path)
    pixels, scale, transform = real_spots.read_image(path, "rgb")
    assert pixels.shape == (9, 12)
    assert 70 < pixels[0, 0] < 90
    assert scale == 255 and "RGB" in transform

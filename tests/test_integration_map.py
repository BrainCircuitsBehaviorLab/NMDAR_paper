import numpy as np
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from process.common import integration_map_2d


def _same_length_convolve(row, kernel):
    convolved = np.convolve(row, kernel, mode="same")
    if convolved.shape[0] == row.shape[0]:
        return convolved
    start = (convolved.shape[0] - row.shape[0]) // 2
    return convolved[start : start + row.shape[0]]


def _reference_matlab_integration_map(x, y, values, *, bnd, dx, sigma):
    n_conv = sigma * 4.0
    sigma_bins = sigma / dx
    n_conv_bins = sigma_bins * 4.0

    boundaries = np.arange(-(bnd + n_conv), bnd + n_conv + dx * 0.5, dx)
    centers = np.concatenate(([boundaries[0] - dx / 2.0], boundaries + dx / 2.0))
    hist_edges = np.concatenate(([-np.inf], boundaries, [np.inf]))

    weighted_sum, _, _ = np.histogram2d(x, y, bins=(hist_edges, hist_edges), weights=values)
    counts, _, _ = np.histogram2d(x, y, bins=(hist_edges, hist_edges))

    kernel_axis = np.arange(-n_conv_bins, n_conv_bins + 1.0, 1.0)
    kernel = np.exp(-(kernel_axis**2) / (2.0 * sigma_bins**2))

    weighted_sum = np.apply_along_axis(_same_length_convolve, 0, weighted_sum, kernel)
    weighted_sum = np.apply_along_axis(_same_length_convolve, 1, weighted_sum, kernel)
    counts = np.apply_along_axis(_same_length_convolve, 0, counts, kernel)
    counts = np.apply_along_axis(_same_length_convolve, 1, counts, kernel)

    keep = (centers > -bnd) & (centers < bnd)
    weighted_sum = weighted_sum[np.ix_(keep, keep)]
    counts = counts[np.ix_(keep, keep)]
    mean_map = np.divide(
        weighted_sum,
        counts,
        out=np.full_like(weighted_sum, np.nan, dtype=float),
        where=counts > 1e-9,
    )
    return mean_map, counts, centers[keep]


def test_integration_map_matches_matlab_algorithm_for_symmetric_axes():
    x_grid = np.tile(np.linspace(-2.0, 2.0, 11), 11)
    y_grid = np.repeat(np.linspace(-2.0, 2.0, 11), 11)
    values = (x_grid + 0.5 * y_grid > 0.0).astype(float)

    result = integration_map_2d(
        x_grid,
        y_grid,
        values,
        bnd=2.0,
        dx=0.5,
        sigma=2.5,
        fill_empty=False,
    )
    expected_map, expected_counts, expected_centers = _reference_matlab_integration_map(
        x_grid,
        y_grid,
        values,
        bnd=2.0,
        dx=0.5,
        sigma=2.5,
    )

    np.testing.assert_allclose(result["map"], expected_map)
    np.testing.assert_allclose(result["n_datapoints"], expected_counts)
    np.testing.assert_allclose(result["x_centers"], expected_centers)
    np.testing.assert_allclose(result["y_centers"], expected_centers)
    assert result["sigma_x_bins"] == 5.0
    assert result["sigma_y_bins"] == 5.0

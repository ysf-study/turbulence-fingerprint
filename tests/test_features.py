"""Invariants of the feature extractor.

These are structural properties, not regression values: a change that breaks
them would change what the four statistics mean, not just their third decimal.
OpenCV is required and is listed in requirements-dev.txt; without it the whole
module is skipped rather than failed, because the released tables already exist.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("cv2")

from feature_utils import compute_tf_features, helmholtz_solenoidal  # noqa: E402


def _smooth_flow(n_frames: int = 6, size: int = 32, seed: int = 0) -> np.ndarray:
    """A spatially coherent flow sequence: random noise averaged over 5x5 windows."""
    rng = np.random.default_rng(seed)
    flow = rng.normal(size=(n_frames, size, size, 2))
    kernel = np.ones((5, 5)) / 25.0
    for t in range(n_frames):
        for c in range(2):
            a = flow[t, :, :, c]
            padded = np.pad(a, 2, mode="reflect")
            flow[t, :, :, c] = sum(
                kernel[i, j] * padded[i : i + size, j : j + size]
                for i in range(5)
                for j in range(5)
            )
    return flow


def test_helmholtz_projection_removes_divergence() -> None:
    """A curl-free field (a gradient) has no solenoidal content and is annihilated."""
    size = 64
    y, x = np.mgrid[0:size, 0:size]
    k = 2 * np.pi / size
    # u = d(phi)/dx, v = d(phi)/dy with phi = sin(3kx) cos(2ky): irrotational
    u = 3 * k * np.cos(3 * k * x) * np.cos(2 * k * y)
    v = -2 * k * np.sin(3 * k * x) * np.sin(2 * k * y)
    us, vs = helmholtz_solenoidal(u, v)
    scale = np.abs(u).mean() + np.abs(v).mean()
    assert (np.abs(us).mean() + np.abs(vs).mean()) < 1e-8 * scale


def test_solenoidal_field_is_untouched_by_the_projection() -> None:
    """A field built from a stream function is already divergence-free.

    u = -d(psi)/dy, v = d(psi)/dx is exactly the representation the projection
    reconstructs, so the projection must return the input unchanged.
    """
    size = 64
    y, x = np.mgrid[0:size, 0:size]
    k = 2 * np.pi / size
    u = -(k * np.sin(k * x) * np.cos(k * y))
    v = k * np.cos(k * x) * np.sin(k * y)
    us, vs = helmholtz_solenoidal(u, v)
    scale = np.abs(u).mean() + np.abs(v).mean()
    assert np.abs(us - u).mean() + np.abs(vs - v).mean() < 1e-6 * scale


def test_features_are_finite_and_ordered() -> None:
    feats = compute_tf_features(_smooth_flow())
    assert feats.shape == (4,)
    assert np.isfinite(feats).all()
    f_low, omega_max, max_temporal, p999_temporal = feats
    # flatness is a fourth moment over a squared second moment, so it is >= 1
    assert f_low >= 1.0
    assert omega_max >= 0.0
    assert max_temporal >= p999_temporal >= 0.0


def test_static_flow_gives_near_zero_amplitude() -> None:
    """A constant flow carries no turbulence: after global-motion removal, no signal."""
    flow = np.ones((6, 32, 32, 2), dtype=float) * 2.0
    feats = compute_tf_features(flow)
    assert feats[1] < 1e-6
    assert feats[2] < 1e-6


def test_amplitude_statistics_track_the_flow_magnitude() -> None:
    """Tripling the flow triples the temporal increments; flatness is scale-free."""
    base = compute_tf_features(_smooth_flow(seed=2), remove_global=False)
    louder = compute_tf_features(3.0 * _smooth_flow(seed=2), remove_global=False)
    assert louder[2] == pytest.approx(3.0 * base[2], rel=0.05)
    assert np.isfinite(louder[0]) and louder[0] > 0

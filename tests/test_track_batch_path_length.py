from __future__ import annotations

import importlib.util
import sys
import warnings
from pathlib import Path

import numpy as np


def _load_track_batch_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "track_batch.py"
    spec = importlib.util.spec_from_file_location(
        "track_batch_path_length_under_test",
        path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(path.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


def test_path_length_avoids_intermediate_square_overflow() -> None:
    track_batch = _load_track_batch_module()
    coordinates = np.array([0.0, 1.0e200], dtype=float)

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        length = track_batch._path_length(coordinates, coordinates)

    assert np.isfinite(length)
    np.testing.assert_allclose(length, np.hypot(1.0e200, 1.0e200))

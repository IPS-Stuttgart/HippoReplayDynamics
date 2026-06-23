from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import struct
import sys

import numpy as np
import pandas as pd


def _load_linearizer_module():
    repo_root = Path(__file__).resolve().parents[1]
    src_path = repo_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    module_path = repo_root / "scripts" / "linearize_olafsdottir_ztrack.py"
    spec = importlib.util.spec_from_file_location("linearize_olafsdottir_ztrack", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_minimal_pos(path: Path, samples: list[tuple[int, int, int]]) -> None:
    header = (
        "timebase 50 hz\n"
        "sample_rate 50.0 hz\n"
        "bytes_per_timestamp 4\n"
        "bytes_per_coord 2\n"
        "pixels_per_metre 100\n"
        f"num_pos_samples {len(samples)}\n"
        "pos_format t,x1,y1,x2,y2,numpix1,numpix2\n"
        "data_start"
    ).encode("ascii")
    payload = b"".join(
        struct.pack(">I8h", frame, x, y, 1023, 1023, 1, 0, 1, 0)
        for frame, x, y in samples
    )
    path.write_bytes(header + payload)


def test_project_points_to_centerline_returns_linear_coordinate() -> None:
    module = _load_linearizer_module()
    centerline = np.array([[0.0, 0.0], [100.0, 0.0], [100.0, 100.0], [200.0, 100.0]])
    xy = np.array([[10.0, 2.0], [98.0, 20.0], [180.0, 102.0]])

    linear, error = module.project_points_to_centerline(xy, np.ones(3, dtype=bool), centerline)

    np.testing.assert_allclose(linear, np.array([10.0, 120.0, 280.0]), atol=2.0)
    assert np.nanmax(error) <= 2.0


def test_linearize_pos_file_writes_required_outputs_with_configured_centerline(tmp_path: Path) -> None:
    module = _load_linearizer_module()
    pos_path = tmp_path / "track.pos"
    _write_minimal_pos(
        pos_path,
        [
            (0, 0, 0),
            (1, 50, 0),
            (2, 100, 0),
            (3, 100, 50),
            (4, 100, 100),
        ],
    )
    centerline_path = tmp_path / "centerline.json"
    centerline_path.write_text(
        json.dumps({"points_cm": [[0, 0], [100, 0], [100, 100]]}),
        encoding="utf-8",
    )
    output = tmp_path / "linearized"

    summary = module.linearize_pos_file(
        pos_path,
        output,
        centerline_path=centerline_path,
        smoothing_window_samples=1,
        occupancy_bin_size_cm=50.0,
    )

    assert Path(summary["linearized_position"]).is_file()
    assert Path(summary["track_geometry"]).is_file()
    assert Path(summary["linearization_diagnostics"]).is_file()
    linearized = pd.read_csv(output / "linearized_position.csv")
    assert list(linearized.columns) == [
        "time_s",
        "x_cm",
        "y_cm",
        "linear_position_cm",
        "speed_cm_s",
        "valid_position",
    ]
    np.testing.assert_allclose(linearized["linear_position_cm"], [0, 50, 100, 150, 200], atol=1e-6)
    assert linearized["valid_position"].all()

    geometry = json.loads((output / "track_geometry.json").read_text(encoding="utf-8"))
    assert geometry["source"] == "configured_centerline"
    assert geometry["track_length_cm"] == 200.0

    diagnostics = pd.read_csv(output / "linearization_diagnostics.csv")
    assert {
        "fraction_valid_position",
        "median_projection_error_cm",
        "max_projection_error_cm",
        "track_length_cm",
        "position_start_time_s",
        "position_end_time_s",
        "session_duration_s",
    }.issubset(set(diagnostics["metric"]))
    diag = diagnostics.set_index("metric")["value"]
    assert diag["position_start_time_s"] == 0.0
    assert diag["position_end_time_s"] == 0.08
    assert diag["session_duration_s"] == 0.08
    occupancy = diagnostics[diagnostics["metric"] == "occupancy_by_linear_bin"]
    assert not occupancy.empty
    assert occupancy["value"].sum() > 0.0


def test_infer_centerline_from_occupied_samples() -> None:
    module = _load_linearizer_module()
    xy = np.array([[float(x), 0.0] for x in range(0, 101, 5)])
    valid = np.ones(xy.shape[0], dtype=bool)

    centerline = module.infer_centerline_from_positions(
        xy,
        valid,
        bin_size_cm=5.0,
        simplify_step_cm=10.0,
    )

    assert centerline.shape[1] == 2
    assert centerline.shape[0] >= 2
    assert np.linalg.norm(centerline[-1] - centerline[0]) >= 90.0

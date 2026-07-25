from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np


def _load_module():
    repo_root = Path(__file__).resolve().parents[1]
    src_path = repo_root / "src"
    scripts_path = repo_root / "scripts"
    for path in (src_path, scripts_path):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    module_path = scripts_path / "summarize_olafsdottir_sleeppost_event_detection_qc.py"
    spec = importlib.util.spec_from_file_location("summarize_olafsdottir_sleeppost_speed_test", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_sleeppost_speed_uses_nonuniform_timestamp_coordinates(tmp_path: Path, monkeypatch) -> None:
    module = _load_module()
    sleep_stem = tmp_path / "sleep"
    sleep_stem.with_suffix(".pos").touch()
    times = np.array([0.0, 1.0, 3.0])

    class Position:
        pass

    position = Position()
    position.x_cm = times**2
    position.y_cm = np.zeros_like(times)
    position.valid = np.ones(times.shape, dtype=bool)
    position.times_s = times
    monkeypatch.setattr(module, "read_axona_pos", lambda _path: position)

    speed = module.load_sleep_speed(sleep_stem)

    assert speed is not None
    np.testing.assert_allclose(speed.speed_cm_s, np.array([1.0, 2.0, 4.0]))

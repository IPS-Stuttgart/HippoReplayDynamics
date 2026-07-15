from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _load_script_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "run_replay_control_benchmark.py"
    spec = importlib.util.spec_from_file_location("run_replay_control_benchmark", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_parse_int_ranges_preserves_ascending_ranges_and_deduplicates() -> None:
    module = _load_script_module()

    assert module._parse_int_ranges("0-2, 2 4") == [0, 1, 2, 4]


def test_parse_int_ranges_rejects_descending_range() -> None:
    module = _load_script_module()

    with pytest.raises(ValueError, match="event range must be ascending"):
        module._parse_int_ranges("5-3")

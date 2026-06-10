import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "scripts"))

from off_swr_trajectory_discovery import _as_bool  # noqa: E402


def test_off_swr_trajectory_discovery_as_bool_accepts_csv_numeric_flags():
    values = ["1.0", 1.0, 2.0, "0.0", 0.0, "False", "True"]
    expected = [True, True, True, False, False, False, True]

    assert [_as_bool(value) for value in values] == expected

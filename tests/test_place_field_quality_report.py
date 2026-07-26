import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "scripts"))

from place_field_quality_report import _format_stable_cell_ids  # noqa: E402


def test_format_stable_cell_ids_does_not_emit_phantom_empty_record():
    assert _format_stable_cell_ids([]) == ""


def test_format_stable_cell_ids_writes_one_integer_per_line():
    assert _format_stable_cell_ids([3, 7]) == "3\n7\n"

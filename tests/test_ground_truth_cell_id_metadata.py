import numpy as np
import pytest

import hipporeplayimm.ground_truth as ground_truth_module
from hipporeplayimm.ground_truth import _parse_cell_ids
from hipporeplayimm.ground_truth_cell_id_metadata import apply_ground_truth_cell_id_metadata_patch


def test_parse_cell_ids_accepts_integer_valued_metadata():
    np.testing.assert_array_equal(_parse_cell_ids("1 2.0 3"), np.array([1, 2, 3]))
    np.testing.assert_array_equal(_parse_cell_ids([1, 2.0, np.int64(3)]), np.array([1, 2, 3]))


@pytest.mark.parametrize(
    "value",
    [
        b"[1, 2]",
        np.bytes_("[1, 2]"),
        bytearray(b"[1, 2]"),
        memoryview(b"[1, 2]"),
    ],
)
def test_parse_cell_ids_accepts_byte_backed_text_metadata(value: object):
    np.testing.assert_array_equal(_parse_cell_ids(value), np.array([1, 2]))


def test_parse_cell_ids_accepts_buffer_elements_without_ascii_expansion():
    np.testing.assert_array_equal(
        _parse_cell_ids([bytearray(b"1"), memoryview(b"2")]),
        np.array([1, 2]),
    )


@pytest.mark.parametrize("value", [b"\xff", bytearray(b"\xff"), memoryview(b"\xff")])
def test_parse_cell_ids_rejects_invalid_utf8_buffer_metadata(value: object):
    with pytest.raises(ValueError, match="cell ID metadata"):
        _parse_cell_ids(value)


def test_parse_cell_ids_rejects_fractional_metadata():
    fractional_cell = 2 + 0.5
    with pytest.raises(ValueError, match="cell ID metadata"):
        _parse_cell_ids([1, fractional_cell])


def test_parse_cell_ids_rejects_boolean_metadata():
    with pytest.raises(ValueError, match="boolean identifiers"):
        _parse_cell_ids([1, True])
    with pytest.raises(ValueError, match="boolean identifiers"):
        _parse_cell_ids(np.array([False], dtype=bool))


@pytest.mark.parametrize("value", [np.iinfo(int).max + 1, np.iinfo(int).min - 1])
def test_parse_cell_ids_rejects_values_outside_platform_integer_range(value: int):
    with pytest.raises(ValueError, match="platform integer range"):
        _parse_cell_ids([value])


def test_ground_truth_cell_id_patch_refreshes_stale_flag(monkeypatch: pytest.MonkeyPatch):
    def stale_parse_cell_ids(_value: object) -> np.ndarray:
        return np.array([0], dtype=int)

    monkeypatch.setattr(ground_truth_module, "_parse_cell_ids", stale_parse_cell_ids)
    monkeypatch.setattr(
        ground_truth_module,
        "_ground_truth_strict_cell_id_metadata_patch_applied",
        True,
        raising=False,
    )

    apply_ground_truth_cell_id_metadata_patch()

    np.testing.assert_array_equal(ground_truth_module._parse_cell_ids("1 2"), np.array([1, 2]))
    with pytest.raises(ValueError, match="boolean identifiers"):
        ground_truth_module._parse_cell_ids([True])

from __future__ import annotations

import hipporeplayimm.olafsdottir2016 as olafsdottir2016
from hipporeplayimm.axona_data_end_footer import apply_axona_data_end_footer_patch
from hipporeplayimm.olafsdottir2016 import _header_float, _header_int, _strip_axona_data_end


def test_axona_data_end_without_line_delimiter_is_payload() -> None:
    payload = bytes([1, 2]) + b"data_end"

    assert _strip_axona_data_end(payload) == payload


def test_axona_data_end_with_line_delimiter_is_footer() -> None:
    payload = bytes([1, 2]) + b"\r\ndata_end\r\n"

    assert _strip_axona_data_end(payload) == bytes([1, 2])


def test_axona_header_float_parses_scientific_notation() -> None:
    assert _header_float({"timebase": "1e6 hz"}, "timebase", 50.0) == 1_000_000.0
    assert _header_float({"sample_rate": "4.8E+3 hz"}, "sample_rate", 0.0) == 4800.0
    assert _header_float({"pixels_per_metre": ".5e2"}, "pixels_per_metre", 0.0) == 50.0


def test_axona_header_int_accepts_integer_scientific_notation() -> None:
    assert _header_int({"num_pos_samples": "3e0"}, "num_pos_samples", 0) == 3


def test_axona_header_int_does_not_round_fractional_metadata() -> None:
    assert _header_int({"num_pos_samples": "3.5"}, "num_pos_samples", 0) == 0


def test_axona_patch_refreshes_stale_flagged_functions(monkeypatch) -> None:
    def stale_strip(payload: bytes) -> bytes:
        return b"stale"

    def stale_header_float(header: dict[str, str], key: str, default: float) -> float:
        return -1.0

    def stale_header_int(header: dict[str, str], key: str, default: int) -> int:
        return -1

    monkeypatch.setattr(olafsdottir2016, "_strip_axona_data_end", stale_strip)
    monkeypatch.setattr(olafsdottir2016, "_header_float", stale_header_float)
    monkeypatch.setattr(olafsdottir2016, "_header_int", stale_header_int)
    monkeypatch.setattr(olafsdottir2016, "_axona_data_end_footer_patch_applied", True, raising=False)

    apply_axona_data_end_footer_patch()

    assert olafsdottir2016._strip_axona_data_end(bytes([1, 2]) + b"\r\ndata_end\r\n") == bytes([1, 2])
    assert olafsdottir2016._header_float({"timebase": "1e6 hz"}, "timebase", 50.0) == 1_000_000.0
    assert olafsdottir2016._header_int({"num_pos_samples": "3e0"}, "num_pos_samples", 0) == 3

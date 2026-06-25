from __future__ import annotations

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


def test_axona_header_int_rounds_scientific_notation() -> None:
    assert _header_int({"num_pos_samples": "3e0"}, "num_pos_samples", 0) == 3

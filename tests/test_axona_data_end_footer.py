from __future__ import annotations

from hipporeplayimm.olafsdottir2016 import _strip_axona_data_end


def test_axona_data_end_without_line_delimiter_is_payload() -> None:
    payload = bytes([1, 2]) + b"data_end"

    assert _strip_axona_data_end(payload) == payload


def test_axona_data_end_with_line_delimiter_is_footer() -> None:
    payload = bytes([1, 2]) + b"\r\ndata_end\r\n"

    assert _strip_axona_data_end(payload) == bytes([1, 2])

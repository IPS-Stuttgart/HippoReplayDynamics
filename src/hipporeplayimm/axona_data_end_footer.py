"""Axona reader runtime fixes for data_end footers and numeric headers."""

from __future__ import annotations

import math
import re

_PATCHED_FLAG = "_axona_data_end_footer_patch_applied"
_NUMERIC_TOKEN = re.compile(r"[-+]?(?:(?:\d+(?:\.\d*)?)|(?:\.\d+))(?:[eE][-+]?\d+)?")


def _strip_axona_data_end(payload: bytes) -> bytes:
    """Strip newline-delimited Axona ``data_end`` footers only.

    Binary Axona payloads can contain arbitrary bytes.  A payload that happens to
    end in the ASCII bytes ``data_end`` is not necessarily carrying the textual
    footer; the standard footer is written as a separate line after the binary
    payload.  Preserve undecorated trailing bytes and strip only when the marker
    is preceded by CR/LF line separation.
    """

    marker = b"data_end"
    end = len(payload)
    while end > 0 and payload[end - 1 : end] in {b"\r", b"\n", b"\t", b" "}:
        end -= 1
    marker_start = end - len(marker)
    if marker_start < 0 or payload[marker_start:end] != marker:
        return payload
    if marker_start >= 2 and payload[marker_start - 2 : marker_start] == b"\r\n":
        return payload[: marker_start - 2]
    if marker_start >= 1 and payload[marker_start - 1 : marker_start] in {b"\r", b"\n"}:
        return payload[: marker_start - 1]
    return payload


def _header_float(header: dict[str, str], key: str, default: float) -> float:
    """Parse Axona numeric header values, including exponent notation."""

    raw = header.get(key)
    if raw is None:
        return float(default)
    match = _NUMERIC_TOKEN.search(str(raw))
    return float(match.group(0)) if match else float(default)


def _header_int(header: dict[str, str], key: str, default: int) -> int:
    """Parse Axona integer-valued header fields without rounding fractions."""

    value = _header_float(header, key, float(default))
    if not math.isfinite(value):
        return int(default)
    rounded = round(value)
    if not math.isclose(value, rounded, rel_tol=0.0, abs_tol=0.0):
        return int(default)
    return int(rounded)


def apply_axona_data_end_footer_patch() -> None:
    """Install strict Axona footer stripping and exponent-aware header parsing."""

    from . import olafsdottir2016

    if (
        getattr(olafsdottir2016, _PATCHED_FLAG, False)
        and getattr(olafsdottir2016, "_strip_axona_data_end", None) is _strip_axona_data_end
        and getattr(olafsdottir2016, "_header_float", None) is _header_float
        and getattr(olafsdottir2016, "_header_int", None) is _header_int
    ):
        return
    olafsdottir2016._strip_axona_data_end = _strip_axona_data_end
    olafsdottir2016._header_float = _header_float
    olafsdottir2016._header_int = _header_int
    setattr(olafsdottir2016, _PATCHED_FLAG, True)


__all__ = ["apply_axona_data_end_footer_patch"]

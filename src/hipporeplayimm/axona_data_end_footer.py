"""Axona reader runtime fixes for binary framing, footers, and numeric headers."""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
from functools import wraps
from pathlib import Path
import re
import sys
from typing import Any

import numpy as np

_PATCHED_FLAG = "_axona_data_end_footer_patch_applied"
_NUMERIC_TOKEN = re.compile(r"[-+]?(?:(?:\d+(?:\.\d*)?)|(?:\.\d+))(?:[eE][-+]?\d+)?")
_CUT_READER_WRAPPER_ATTR = "_axona_cut_label_count_validation_wrapper"
_EGF_READER_WRAPPER_ATTR = "_axona_egf_payload_count_validation_wrapper"
_DATA_START_READER_WRAPPER_ATTR = "_axona_data_start_validation_wrapper"
_DATA_START_MARKER = b"data_start"
_DATA_START_LINE = re.compile(rb"(?:^|[\r\n])data_start")
_AXONA_HEADER_SCAN_BYTES = 64 * 1024
_BINARY_READER_NAMES = (
    "read_axona_pos",
    "read_axona_egf",
    "read_axona_tetrode_spike_times",
)
_ORIGINAL_ATTR = "__hipporeplayimm_original__"


def _strip_axona_data_end(payload: bytes) -> bytes:
    """Strip newline-delimited Axona ``data_end`` footers only.

    Binary Axona payloads can contain arbitrary bytes. A payload that happens to
    end in the ASCII bytes ``data_end`` is not necessarily carrying the textual
    footer; the standard footer is written as a separate line after the binary
    payload. Preserve undecorated trailing bytes and strip only when the marker
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
    """Parse Axona integer-valued header fields without float precision loss."""

    raw = header.get(key)
    if raw is None:
        return int(default)
    match = _NUMERIC_TOKEN.search(str(raw))
    if match is None:
        return int(default)
    try:
        value = Decimal(match.group(0))
    except InvalidOperation:
        return int(default)
    if not value.is_finite() or value < -sys.maxsize - 1 or value > sys.maxsize:
        return int(default)
    integral = value.to_integral_value()
    if value != integral:
        return int(default)
    return int(integral)


def _payload_record_count(payload: bytes, record_size: int, header_count: int) -> int:
    """Return the usable Axona record count without hiding truncated payloads."""

    record_size = int(record_size)
    header_count = int(header_count)
    if record_size <= 0:
        raise ValueError("Axona record size must be positive")

    available, remainder = divmod(len(payload), record_size)
    if remainder:
        raise ValueError(
            f"Axona payload has {remainder} trailing byte(s); expected complete "
            f"{record_size}-byte records"
        )
    if header_count > available:
        raise ValueError(
            f"Axona header declares {header_count} records but payload contains only "
            f"{available} complete records"
        )
    return available if header_count <= 0 else header_count


def _wrapper_chain(function: Any):
    """Yield one wrapper lineage while tolerating mixed legacy wrapper styles."""

    seen: set[int] = set()
    current = function
    while callable(current) and id(current) not in seen:
        seen.add(id(current))
        yield current
        original = getattr(current, _ORIGINAL_ATTR, None)
        wrapped = getattr(current, "__wrapped__", None)
        if callable(original) and original is not current:
            current = original
        elif callable(wrapped) and wrapped is not current:
            current = wrapped
        else:
            break


def _has_owned_wrapper(function: Any, marker: str) -> bool:
    """Return whether a wrapper in the lineage owns ``marker``.

    ``functools.wraps`` copies function attributes. Storing the wrapper itself
    as the marker value distinguishes the real owner from copied attributes on
    later outer wrappers.
    """

    return any(getattr(wrapper, marker, None) is wrapper for wrapper in _wrapper_chain(function))


def _declared_cut_spike_count(path: str | Path) -> int | None:
    """Return the spike count declared by an Axona ``.cut`` header."""

    text = Path(path).read_text(encoding="latin-1", errors="ignore")
    for line in text.splitlines():
        if "Exact_cut_for" not in line or "spikes:" not in line:
            continue
        match = re.search(r"spikes:\s*(\d+)", line)
        return int(match.group(1)) if match else None
    return None


def _require_axona_data_start(path: str | Path) -> None:
    """Reject binary Axona files whose payload delimiter is absent.

    Real Axona files and the repository fixtures permit the first binary byte to
    follow ``data_start`` immediately. Require a line-start marker, but do not
    require a newline after it.
    """

    with Path(path).open("rb") as handle:
        prefix = handle.read(_AXONA_HEADER_SCAN_BYTES + len(_DATA_START_MARKER))
    if _DATA_START_LINE.search(prefix) is None:
        raise ValueError(
            f"Axona binary file {path} is missing a data_start marker in its header"
        )


def _wrap_binary_reader_data_start(current):
    """Require a payload delimiter before invoking an Axona binary reader."""

    if _has_owned_wrapper(current, _DATA_START_READER_WRAPPER_ATTR):
        return current

    @wraps(current)
    def read_axona_binary(path, *args, **kwargs):
        _require_axona_data_start(path)
        return current(path, *args, **kwargs)

    setattr(
        read_axona_binary,
        _DATA_START_READER_WRAPPER_ATTR,
        read_axona_binary,
    )
    setattr(read_axona_binary, _ORIGINAL_ATTR, current)
    return read_axona_binary


def _wrap_read_axona_cut(current):
    """Reject cut files whose declared labels are truncated."""

    if _has_owned_wrapper(current, _CUT_READER_WRAPPER_ATTR):
        return current

    @wraps(current)
    def read_axona_cut(path, tetrode_path=None):
        declared_spikes = _declared_cut_spike_count(path)
        result = current(path, tetrode_path)
        observed_labels = len(result.labels)
        if declared_spikes is not None and observed_labels != declared_spikes:
            raise ValueError(
                f"Axona cut file declares {declared_spikes} spikes but contains "
                f"{observed_labels} labels"
            )
        return result

    setattr(read_axona_cut, _CUT_READER_WRAPPER_ATTR, read_axona_cut)
    setattr(read_axona_cut, _ORIGINAL_ATTR, current)
    return read_axona_cut


def _wrap_read_axona_egf(current):
    """Parse only complete EGF samples and honor the declared sample count."""

    if _has_owned_wrapper(current, _EGF_READER_WRAPPER_ATTR):
        return current

    @wraps(current)
    def read_axona_egf(path):
        from . import olafsdottir2016

        header, payload = olafsdottir2016._read_axona_header_and_payload(path)
        n_samples = olafsdottir2016._payload_record_count(
            payload,
            2,
            olafsdottir2016._header_int(header, "num_EGF_samples", 0),
        )
        signal = np.frombuffer(
            payload[: n_samples * 2],
            dtype=">i2",
        ).astype(np.int16)
        sample_rate_hz = olafsdottir2016._header_float(
            header,
            "sample_rate",
            4800.0,
        )
        times_s = np.arange(signal.shape[0], dtype=float) / float(sample_rate_hz)
        return olafsdottir2016.AxonaEgf(
            header=header,
            signal=signal,
            sample_rate_hz=float(sample_rate_hz),
            times_s=times_s,
        )

    setattr(read_axona_egf, _EGF_READER_WRAPPER_ATTR, read_axona_egf)
    setattr(read_axona_egf, _ORIGINAL_ATTR, current)
    return read_axona_egf


def apply_axona_data_end_footer_patch() -> None:
    """Install strict Axona framing, footer, header, and record validation."""

    from . import olafsdottir2016

    current_cut_reader = getattr(olafsdottir2016, "read_axona_cut", None)
    current_binary_readers = {
        name: getattr(olafsdottir2016, name, None)
        for name in _BINARY_READER_NAMES
    }
    if (
        getattr(olafsdottir2016, _PATCHED_FLAG, False)
        and getattr(olafsdottir2016, "_strip_axona_data_end", None) is _strip_axona_data_end
        and getattr(olafsdottir2016, "_header_float", None) is _header_float
        and getattr(olafsdottir2016, "_header_int", None) is _header_int
        and getattr(olafsdottir2016, "_payload_record_count", None) is _payload_record_count
        and _has_owned_wrapper(current_cut_reader, _CUT_READER_WRAPPER_ATTR)
        and _has_owned_wrapper(
            current_binary_readers["read_axona_egf"],
            _EGF_READER_WRAPPER_ATTR,
        )
        and all(
            _has_owned_wrapper(reader, _DATA_START_READER_WRAPPER_ATTR)
            for reader in current_binary_readers.values()
        )
    ):
        return

    olafsdottir2016._strip_axona_data_end = _strip_axona_data_end
    olafsdottir2016._header_float = _header_float
    olafsdottir2016._header_int = _header_int
    olafsdottir2016._payload_record_count = _payload_record_count
    olafsdottir2016.read_axona_cut = _wrap_read_axona_cut(current_cut_reader)

    current_binary_readers["read_axona_egf"] = _wrap_read_axona_egf(
        current_binary_readers["read_axona_egf"]
    )
    for name, current_reader in current_binary_readers.items():
        setattr(
            olafsdottir2016,
            name,
            _wrap_binary_reader_data_start(current_reader),
        )
    setattr(olafsdottir2016, _PATCHED_FLAG, True)


__all__ = ["apply_axona_data_end_footer_patch"]

from __future__ import annotations

from pathlib import Path
import struct

import pytest

import hipporeplayimm.olafsdottir2016 as olafsdottir2016
from hipporeplayimm.olafsdottir2016 import read_axona_egf, read_axona_tetrode_spike_times
from hipporeplayimm.olafsdottir_header_float_patch import apply_olafsdottir_header_float_patch


def test_axona_tetrode_timebase_accepts_scientific_notation(tmp_path: Path) -> None:
    path = tmp_path / "session.1"
    header = (
        "num_spikes 1\n"
        "timebase 9.6e4 hz\n"
        "samples_per_spike 2\n"
        "data_start"
    ).encode("ascii")
    payload = struct.pack(">I", 96000) + b"\x00" * 8
    path.write_bytes(header + payload)

    times = read_axona_tetrode_spike_times(path)

    assert times.tolist() == pytest.approx([1.0])


def test_axona_tetrode_timebase_falls_back_from_nonfinite_overflow(tmp_path: Path) -> None:
    path = tmp_path / "session.1"
    header = (
        "num_spikes 1\n"
        "timebase 1e309 hz\n"
        "samples_per_spike 2\n"
        "data_start"
    ).encode("ascii")
    payload = struct.pack(">I", 96000) + b"\x00" * 8
    path.write_bytes(header + payload)

    times = read_axona_tetrode_spike_times(path)

    assert times.tolist() == pytest.approx([1.0])


def test_axona_tetrode_timebase_falls_back_from_malformed_exponent(tmp_path: Path) -> None:
    path = tmp_path / "session.1"
    header = (
        "num_spikes 1\n"
        "timebase 1e hz\n"
        "samples_per_spike 2\n"
        "data_start"
    ).encode("ascii")
    payload = struct.pack(">I", 96000) + b"\x00" * 8
    path.write_bytes(header + payload)

    times = read_axona_tetrode_spike_times(path)

    assert times.tolist() == pytest.approx([1.0])


def test_header_float_does_not_extract_numbers_from_words() -> None:
    assert olafsdottir2016._header_float({"timebase": "channel1"}, "timebase", 96000.0) == pytest.approx(96000.0)


@pytest.mark.parametrize(
    ("key", "default"),
    [
        ("timebase", 96000.0),
        ("sample_rate", 4800.0),
        ("pixels_per_metre", 100.0),
    ],
)
@pytest.mark.parametrize("raw_value", ["0", "-1"])
def test_header_float_rejects_nonpositive_physical_scales(
    key: str,
    default: float,
    raw_value: str,
) -> None:
    with pytest.raises(ValueError, match=rf"Axona header {key} must be positive"):
        olafsdottir2016._header_float({key: raw_value}, key, default)


def test_axona_egf_rejects_zero_sample_rate(tmp_path: Path) -> None:
    path = tmp_path / "session.egf"
    header = b"sample_rate 0 hz\ndata_start\n"
    payload = struct.pack(">hh", -1, 1)
    path.write_bytes(header + payload)

    with pytest.raises(ValueError, match="Axona header sample_rate must be positive"):
        read_axona_egf(path)


def test_header_float_patch_refreshes_stale_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    def stale_header_float(header: dict[str, str], key: str, default: float) -> float:
        return float(default)

    monkeypatch.setattr(olafsdottir2016, "_header_float", stale_header_float)
    monkeypatch.setattr(olafsdottir2016, "_olafsdottir_header_float_patch_applied", True, raising=False)

    apply_olafsdottir_header_float_patch()

    assert olafsdottir2016._header_float({"timebase": "9.6e4 hz"}, "timebase", 0.0) == pytest.approx(96000.0)

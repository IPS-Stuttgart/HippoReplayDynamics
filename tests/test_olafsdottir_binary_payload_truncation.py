from __future__ import annotations

from pathlib import Path

import pytest

from hipporeplayimm.olafsdottir2016 import read_axona_pos, read_axona_tetrode_spike_times


def _position_record(timestamp: int = 0) -> bytes:
    coordinates = (100, 200, 101, 201, 1, 1, 1, 1)
    return int(timestamp).to_bytes(4, "big", signed=False) + b"".join(
        int(value).to_bytes(2, "big", signed=True) for value in coordinates
    )


def test_read_axona_pos_rejects_truncated_declared_record_count(tmp_path: Path) -> None:
    path = tmp_path / "session.pos"
    header = (
        "num_pos_samples 2\n"
        "timebase 50 hz\n"
        "pixels_per_metre 100\n"
        "bytes_per_timestamp 4\n"
        "bytes_per_coord 2\n"
        "data_start\n"
    ).encode("ascii")
    path.write_bytes(header + _position_record() + b"\ndata_end\n")

    with pytest.raises(ValueError, match=r"declares 2 records.*only 1 complete records"):
        read_axona_pos(path)


def test_read_axona_tetrode_rejects_truncated_declared_record_count(tmp_path: Path) -> None:
    path = tmp_path / "session.1"
    samples_per_spike = 2
    n_channels = 4
    header = (
        "num_spikes 2\n"
        "timebase 96000 hz\n"
        f"samples_per_spike {samples_per_spike}\n"
        f"num_chans {n_channels}\n"
        "bytes_per_timestamp 4\n"
        "bytes_per_sample 1\n"
        "data_start\n"
    ).encode("ascii")
    payload = bytearray()
    for channel in range(n_channels):
        payload.extend((0).to_bytes(4, "big", signed=False))
        payload.extend(bytes((10, 20 + channel)))
    path.write_bytes(header + payload + b"\ndata_end\n")

    with pytest.raises(ValueError, match=r"declares 2 records.*only 1 complete records"):
        read_axona_tetrode_spike_times(path)


def test_read_axona_pos_rejects_partial_binary_record(tmp_path: Path) -> None:
    path = tmp_path / "session.pos"
    header = (
        "timebase 50 hz\n"
        "pixels_per_metre 100\n"
        "bytes_per_timestamp 4\n"
        "bytes_per_coord 2\n"
        "data_start\n"
    ).encode("ascii")
    path.write_bytes(header + _position_record() + b"\x00" + b"\ndata_end\n")

    with pytest.raises(ValueError, match=r"1 trailing byte"):
        read_axona_pos(path)

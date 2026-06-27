from __future__ import annotations

from pathlib import Path
import struct

import pytest

from hipporeplayimm.olafsdottir2016 import read_axona_tetrode_spike_times


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

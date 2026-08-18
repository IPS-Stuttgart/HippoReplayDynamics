from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from hipporeplayimm.olafsdottir2016 import read_axona_tetrode_spike_times


def _write_tetrode_file(
    path: Path,
    timestamps: list[int],
    *,
    mismatched_contact: bool = False,
) -> None:
    samples_per_spike = 2
    n_channels = 4
    header = (
        f"num_spikes {len(timestamps)}\n"
        "timebase 96000 hz\n"
        f"samples_per_spike {samples_per_spike}\n"
        f"num_chans {n_channels}\n"
        "bytes_per_timestamp 4\n"
        "bytes_per_sample 1\n"
        "data_start\n"
    ).encode("ascii")
    payload = bytearray()
    for spike_index, timestamp in enumerate(timestamps):
        for channel in range(n_channels):
            contact_timestamp = timestamp
            if mismatched_contact and spike_index == 1 and channel == 2:
                contact_timestamp += 1
            payload.extend(int(contact_timestamp).to_bytes(4, "big", signed=False))
            payload.extend(bytes((10 + spike_index, 20 + channel)))
    path.write_bytes(header + payload + b"\ndata_end\n")


def test_read_axona_tetrode_spike_times_uses_per_contact_record_layout(tmp_path: Path) -> None:
    path = tmp_path / "session.1"
    _write_tetrode_file(path, [0, 96000, 192000])

    spike_times = read_axona_tetrode_spike_times(path)

    np.testing.assert_allclose(spike_times, np.array([0.0, 1.0, 2.0]))


def test_read_axona_tetrode_spike_times_rejects_disagreeing_contact_timestamps(
    tmp_path: Path,
) -> None:
    path = tmp_path / "session.1"
    _write_tetrode_file(path, [0, 96000, 192000], mismatched_contact=True)

    with pytest.raises(ValueError, match="contact timestamps disagree"):
        read_axona_tetrode_spike_times(path)

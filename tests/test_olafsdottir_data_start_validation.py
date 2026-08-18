from __future__ import annotations

from pathlib import Path

import pytest

from hipporeplayimm.olafsdottir2016 import (
    read_axona_egf,
    read_axona_pos,
    read_axona_set,
    read_axona_tetrode_spike_times,
)


@pytest.mark.parametrize(
    ("filename", "reader", "header"),
    [
        (
            "broken.pos",
            read_axona_pos,
            b"bytes_per_timestamp 4\nbytes_per_coord 2\n",
        ),
        (
            "broken.egf",
            read_axona_egf,
            b"sample_rate 4800 hz\n",
        ),
        (
            "broken.1",
            read_axona_tetrode_spike_times,
            b"samples_per_spike 50\nbytes_per_timestamp 4\nbytes_per_sample 1\n",
        ),
    ],
)
def test_binary_axona_readers_reject_missing_data_start(
    tmp_path: Path,
    filename: str,
    reader,
    header: bytes,
) -> None:
    path = tmp_path / filename
    path.write_bytes(header)

    with pytest.raises(ValueError, match="missing a data_start marker"):
        reader(path)


def test_axona_set_remains_valid_without_data_start(tmp_path: Path) -> None:
    path = tmp_path / "session.set"
    path.write_bytes(b"duration 10\nsample_rate 4800 hz\n")

    header = read_axona_set(path)

    assert header["duration"] == "10"
    assert header["sample_rate"] == "4800 hz"

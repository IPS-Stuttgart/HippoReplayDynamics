from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import pytest

import hipporeplayimm
from hipporeplayimm import olafsdottir2016


@pytest.mark.parametrize(
    ("filename", "reader_name", "header"),
    [
        (
            "broken.pos",
            "read_axona_pos",
            b"bytes_per_timestamp 4\nbytes_per_coord 2\n",
        ),
        (
            "broken.egf",
            "read_axona_egf",
            b"sample_rate 4800 hz\n",
        ),
        (
            "broken.1",
            "read_axona_tetrode_spike_times",
            b"samples_per_spike 50\nbytes_per_timestamp 4\nbytes_per_sample 1\n",
        ),
    ],
)
def test_binary_axona_readers_reject_missing_data_start(
    tmp_path: Path,
    filename: str,
    reader_name: str,
    header: bytes,
) -> None:
    path = tmp_path / filename
    path.write_bytes(header)

    reader = getattr(olafsdottir2016, reader_name)
    with pytest.raises(ValueError, match="missing a data_start marker"):
        reader(path)


def test_axona_set_remains_valid_without_data_start(tmp_path: Path) -> None:
    path = tmp_path / "session.set"
    path.write_bytes(b"duration 10\nsample_rate 4800 hz\n")

    header = olafsdottir2016.read_axona_set(path)

    assert header["duration"] == "10"
    assert header["sample_rate"] == "4800 hz"


def test_binary_marker_may_be_followed_immediately_by_payload(tmp_path: Path) -> None:
    path = tmp_path / "session.egf"
    path.write_bytes(
        b"sample_rate 4800 hz\n"
        b"num_EGF_samples 2\n"
        b"data_start"
        + np.array([-2, 3], dtype=">i2").tobytes()
    )

    result = olafsdottir2016.read_axona_egf(path)

    assert result.signal.tolist() == [-2, 3]


def test_data_start_text_inside_header_value_is_not_a_delimiter(tmp_path: Path) -> None:
    path = tmp_path / "broken.egf"
    path.write_bytes(
        b"comment this mentions data_start but has no delimiter\n"
        b"sample_rate 4800 hz\n"
    )

    with pytest.raises(ValueError, match="missing a data_start marker"):
        olafsdottir2016.read_axona_egf(path)


def test_runtime_refresh_restores_axona_binary_guards_after_reload(
    tmp_path: Path,
) -> None:
    reloaded = importlib.reload(olafsdottir2016)
    hipporeplayimm.apply_runtime_patches()

    missing = tmp_path / "missing.egf"
    missing.write_bytes(b"sample_rate 4800 hz\n")
    with pytest.raises(ValueError, match="missing a data_start marker"):
        reloaded.read_axona_egf(missing)

    truncated = tmp_path / "truncated.egf"
    truncated.write_bytes(
        b"sample_rate 4800 hz\n"
        b"num_EGF_samples 2\n"
        b"data_start"
        + (1).to_bytes(2, "big", signed=True)
    )
    with pytest.raises(
        ValueError,
        match=r"declares 2 records.*only 1 complete records",
    ):
        reloaded.read_axona_egf(truncated)

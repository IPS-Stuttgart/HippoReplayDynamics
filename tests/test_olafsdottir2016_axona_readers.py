from __future__ import annotations

from pathlib import Path

import numpy as np

from hipporeplayimm.olafsdottir2016 import (
    read_axona_cut,
    read_axona_egf,
    read_axona_pos,
    read_axona_set,
)


FIXTURE = Path(__file__).resolve().parent / "fixtures" / "axona_minimal"


def test_read_axona_set_parses_header_metadata() -> None:
    header = read_axona_set(FIXTURE / "mini.set")

    assert header["duration"] == "10"
    assert header["sample_rate"] == "4800 hz"


def test_read_axona_pos_parses_position_time_series() -> None:
    pos = read_axona_pos(FIXTURE / "mini.pos")

    np.testing.assert_allclose(pos.times_s, np.array([0.0, 0.02, 0.04]))
    np.testing.assert_allclose(pos.x_cm, np.array([10.0, 21.0, 40.0]))
    np.testing.assert_allclose(pos.y_cm, np.array([20.0, 31.0, 50.0]))
    assert pos.valid.tolist() == [True, True, True]
    assert pos.pixels_per_metre == 100.0


def test_read_axona_egf_parses_int16_lfp() -> None:
    egf = read_axona_egf(FIXTURE / "mini.egf")

    assert egf.sample_rate_hz == 4800.0
    np.testing.assert_array_equal(egf.signal, np.array([-2, -1, 0, 1], dtype=np.int16))
    np.testing.assert_allclose(egf.times_s, np.arange(4) / 4800.0)


def test_read_axona_cut_reads_labels_and_optional_tetrode_times() -> None:
    cut = read_axona_cut(FIXTURE / "mini_1.cut", tetrode_path=FIXTURE / "mini.1")

    np.testing.assert_array_equal(cut.labels, np.array([1, 0, 2]))
    assert cut.spike_times_s is not None
    np.testing.assert_allclose(cut.spike_times_s, np.array([0.0, 1.0, 2.0]))

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from hipporeplayimm.olafsdottir2016 import read_axona_cut


def test_read_axona_cut_rejects_truncated_declared_labels(tmp_path: Path) -> None:
    cut_path = tmp_path / "truncated.cut"
    cut_path.write_text("Exact_cut_for sample spikes: 3\n1 2\n", encoding="latin-1")

    with pytest.raises(ValueError, match=r"declares 3 spikes but contains 2 labels"):
        read_axona_cut(cut_path)


def test_read_axona_cut_accepts_complete_declared_labels(tmp_path: Path) -> None:
    cut_path = tmp_path / "complete.cut"
    cut_path.write_text("Exact_cut_for sample spikes: 3\n1 2 3\n", encoding="latin-1")

    result = read_axona_cut(cut_path)

    np.testing.assert_array_equal(result.labels, np.array([1, 2, 3]))

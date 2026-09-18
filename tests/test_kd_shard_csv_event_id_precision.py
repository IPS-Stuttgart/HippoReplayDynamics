from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from aggregate_kd_model_evidence_shards import _read_csv_with_exact_event_index  # noqa: E402


def test_base_csv_reader_preserves_adjacent_decimal_event_ids(
    tmp_path: Path,
) -> None:
    first = 2**53
    second = first + 1
    path = tmp_path / "event_model_evidence.csv"
    pd.DataFrame(
        {
            "event_index": [f"{first}.0", f"{second}.0"],
            "model": ["diffusion", "diffusion"],
        }
    ).to_csv(path, index=False)

    loaded = _read_csv_with_exact_event_index(path)

    assert loaded["event_index"].tolist() == [first, second]


def test_base_csv_reader_rejects_fractional_event_ids(tmp_path: Path) -> None:
    path = tmp_path / "event_model_evidence.csv"
    pd.DataFrame({"event_index": ["10.5"]}).to_csv(path, index=False)

    with pytest.raises(ValueError, match="finite integer values"):
        _read_csv_with_exact_event_index(path)

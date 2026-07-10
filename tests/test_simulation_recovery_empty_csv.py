from __future__ import annotations

import pandas as pd

from hipporeplayimm.simulation_recovery import SimulationRecoveryResult
from hipporeplayimm.simulation_recovery_empty_csv import EMPTY_RECOVERY_CSV_SCHEMAS


def test_empty_simulation_recovery_csvs_keep_parseable_headers(tmp_path) -> None:
    result = SimulationRecoveryResult(
        event_scores=pd.DataFrame(),
        confusion_matrix=pd.DataFrame(),
        summary=pd.DataFrame(),
        settings={},
        certified_vs_exact_summary=pd.DataFrame(),
    )

    result.write(tmp_path)

    for filename, expected_columns in EMPTY_RECOVERY_CSV_SCHEMAS.items():
        table = pd.read_csv(tmp_path / filename)
        assert table.empty
        assert tuple(table.columns) == expected_columns

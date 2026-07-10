"""Keep empty simulation-recovery CSV artifacts readable.

Pandas serializes a completely columnless data frame as a zero-byte file.  Empty
or early-stopped recovery runs can legitimately produce columnless summary and
diagnostic tables, but downstream sweep aggregation still calls ``read_csv`` on
the advertised artifacts.  Repair only zero-byte outputs with stable header-only
schemas so an empty run remains distinguishable from a corrupt artifact.
"""

from __future__ import annotations

from functools import wraps
from pathlib import Path

import pandas as pd

_PATCHED_FLAG = "_simulation_recovery_empty_csv_patch_applied"
_WRAPPER_MARKER = "_simulation_recovery_empty_csv_wrapper"

EMPTY_RECOVERY_CSV_SCHEMAS: dict[str, tuple[str, ...]] = {
    "simulation_recovery_event_scores.csv": ("status",),
    "simulation_recovery_confusion_matrix.csv": ("true_model",),
    "simulation_recovery_summary.csv": (
        "true_model",
        "expected_model",
        "simulated_events",
        "recovered_events",
        "recovery_accuracy",
    ),
    "simulation_recovery_certified_vs_exact_summary.csv": (
        "true_model",
        "expected_model",
        "simulated_events",
        "certified_vs_exact_recovered_events",
        "certified_vs_exact_recovery_accuracy",
    ),
    "simulation_recovery_diagnostic_event_table.csv": (
        "session",
        "event_index",
        "true_model",
        "expected_model",
        "failure_mode",
    ),
    "simulation_recovery_diagnostic_summary.csv": (
        "true_model",
        "events",
        "strict_recovery_accuracy",
        "certified_vs_exact_recovery_accuracy",
    ),
    "simulation_recovery_certified_vs_exact_events.csv": (
        "session",
        "event_index",
        "true_model",
        "expected_model",
        "certified_vs_exact_recovered_expected_model",
        "certified_vs_exact_reason",
    ),
}


def apply_simulation_recovery_empty_csv_patch() -> None:
    """Wrap result writing so empty CSV artifacts retain parseable headers."""

    from . import simulation_recovery

    current_write = simulation_recovery.SimulationRecoveryResult.write
    if getattr(current_write, _WRAPPER_MARKER, False):
        setattr(simulation_recovery, _PATCHED_FLAG, True)
        return

    @wraps(current_write)
    def write_with_empty_csv_schemas(self, output) -> None:
        current_write(self, output)
        _repair_zero_byte_recovery_csvs(Path(output))

    setattr(write_with_empty_csv_schemas, _WRAPPER_MARKER, True)
    simulation_recovery.SimulationRecoveryResult.write = write_with_empty_csv_schemas
    setattr(simulation_recovery, _PATCHED_FLAG, True)


def _repair_zero_byte_recovery_csvs(output: Path) -> None:
    for filename, columns in EMPTY_RECOVERY_CSV_SCHEMAS.items():
        path = output / filename
        if path.exists() and path.stat().st_size == 0:
            pd.DataFrame(columns=columns).to_csv(path, index=False)


__all__ = [
    "EMPTY_RECOVERY_CSV_SCHEMAS",
    "apply_simulation_recovery_empty_csv_patch",
]

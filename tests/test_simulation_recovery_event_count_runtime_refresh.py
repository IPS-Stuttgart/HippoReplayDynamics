import importlib

import pandas as pd

import hipporeplayimm
import hipporeplayimm.simulation_recovery as simulation_recovery


def test_runtime_patches_restore_event_count_wrappers_after_recovery_reload() -> None:
    reloaded = importlib.reload(simulation_recovery)

    # importlib.reload() keeps dynamically added module attributes, so the old
    # module-level sentinels remain even though the source functions were
    # restored and no longer carry the event-count behavior.
    assert getattr(
        reloaded,
        "_simulation_recovery_session_event_count_patch_applied",
        False,
    )
    assert getattr(
        reloaded,
        "_simulation_recovery_certified_event_duplicate_model_patch_applied",
        False,
    )

    hipporeplayimm.apply_runtime_patches()

    assert getattr(
        reloaded.recovery_summary,
        "_simulation_recovery_session_event_count_patch_applied",
        False,
    )
    assert getattr(
        reloaded.certified_vs_exact_recovery_summary,
        "_simulation_recovery_session_event_count_patch_applied",
        False,
    )
    assert getattr(
        reloaded.certified_vs_exact_event_recovery,
        "_simulation_recovery_certified_event_duplicate_model_patch_applied",
        False,
    )

    rows = pd.DataFrame(
        [
            _row("Rat1/Open1", -1.0),
            _row("Rat1/Open1", -3.0, model="sorted-spike-state-space-diffusion"),
            _row("Rat2/Open1", -2.0),
            _row("Rat2/Open1", -4.0, model="sorted-spike-state-space-diffusion"),
        ]
    )
    scored = reloaded.add_evidence_columns(rows)

    summary = reloaded.recovery_summary(scored)
    certified = reloaded.certified_vs_exact_recovery_summary(scored)

    overall = summary[summary["true_model"] == "overall"].iloc[0]
    certified_overall = certified[certified["true_model"] == "overall"].iloc[0]
    assert overall["simulated_events"] == 2
    assert certified_overall["simulated_events"] == 2


def _row(
    session: str,
    log_evidence: float,
    *,
    model: str = "sorted-spike-state-space-stationary",
) -> dict[str, object]:
    return {
        "status": "success",
        "session": session,
        "event_index": 0,
        "true_model": "stationary",
        "expected_model": "sorted-spike-state-space-stationary",
        "model": model,
        "log_evidence": log_evidence,
        "n_time": 3,
        "n_spikes": 5,
    }

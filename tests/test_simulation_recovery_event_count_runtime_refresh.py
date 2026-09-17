from __future__ import annotations

import subprocess
import sys
import textwrap


def test_runtime_patches_restore_event_count_wrappers_after_recovery_reload() -> None:
    # Reloading simulation_recovery mutates a shared module object and intentionally
    # leaves arbitrary dynamic sentinels behind.  Exercise that process boundary in
    # isolation so the regression test cannot perturb unrelated patch-order tests.
    script = textwrap.dedent(
        r'''
        import importlib

        import pandas as pd

        import hipporeplayimm
        import hipporeplayimm.simulation_recovery as recovery

        reloaded = importlib.reload(recovery)

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
        assert not getattr(
            reloaded.recovery_summary,
            "_simulation_recovery_session_event_count_patch_applied",
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
                {
                    "status": "success",
                    "session": "Rat1/Open1",
                    "event_index": 0,
                    "true_model": "stationary",
                    "expected_model": "sorted-spike-state-space-stationary",
                    "model": "sorted-spike-state-space-stationary",
                    "log_evidence": -1.0,
                    "n_time": 3,
                    "n_spikes": 5,
                },
                {
                    "status": "success",
                    "session": "Rat1/Open1",
                    "event_index": 0,
                    "true_model": "stationary",
                    "expected_model": "sorted-spike-state-space-stationary",
                    "model": "sorted-spike-state-space-diffusion",
                    "log_evidence": -3.0,
                    "n_time": 3,
                    "n_spikes": 5,
                },
                {
                    "status": "success",
                    "session": "Rat2/Open1",
                    "event_index": 0,
                    "true_model": "stationary",
                    "expected_model": "sorted-spike-state-space-stationary",
                    "model": "sorted-spike-state-space-stationary",
                    "log_evidence": -2.0,
                    "n_time": 3,
                    "n_spikes": 5,
                },
                {
                    "status": "success",
                    "session": "Rat2/Open1",
                    "event_index": 0,
                    "true_model": "stationary",
                    "expected_model": "sorted-spike-state-space-stationary",
                    "model": "sorted-spike-state-space-diffusion",
                    "log_evidence": -4.0,
                    "n_time": 3,
                    "n_spikes": 5,
                },
            ]
        )
        scored = reloaded.add_evidence_columns(rows)

        summary = reloaded.recovery_summary(scored)
        certified = reloaded.certified_vs_exact_recovery_summary(scored)

        overall = summary[summary["true_model"] == "overall"].iloc[0]
        certified_overall = certified[certified["true_model"] == "overall"].iloc[0]
        assert overall["simulated_events"] == 2
        assert certified_overall["simulated_events"] == 2
        '''
    )

    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr

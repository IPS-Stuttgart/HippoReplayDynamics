import pandas as pd

import hipporeplayimm  # noqa: F401 - ensure runtime patches are installed
from hipporeplayimm.evidence_reporting import EXACT_EVIDENCE_SUPPORT, simulation_event_best_rows


def test_simulation_best_rows_ignore_unique_stale_flag_below_max_evidence():
    rows = pd.DataFrame(
        [
            {
                "session": "RatX/OpenY",
                "simulation_random_seed": 1,
                "event_index": 0,
                "true_model": "diffusion",
                "expected_model": "sorted-spike-state-space-diffusion",
                "model": "sorted-spike-state-space-diffusion",
                "status": "success",
                "log_evidence": 0.0,
                "evidence_support": EXACT_EVIDENCE_SUPPORT,
                "evidence_comparable": True,
                "is_best_model": False,
                "best_model": "stale",
                "n_time": 3,
                "n_spikes": 5,
            },
            {
                "session": "RatX/OpenY",
                "simulation_random_seed": 1,
                "event_index": 0,
                "true_model": "diffusion",
                "expected_model": "sorted-spike-state-space-diffusion",
                "model": "sorted-spike-state-space-momentum-exact-sparse",
                "status": "success",
                "log_evidence": -10.0,
                "evidence_support": EXACT_EVIDENCE_SUPPORT,
                "evidence_comparable": True,
                "is_best_model": True,
                "best_model": "stale",
                "n_time": 3,
                "n_spikes": 5,
            },
        ]
    )

    best = simulation_event_best_rows(rows)

    assert len(best) == 1
    assert best.loc[0, "model"] == "sorted-spike-state-space-diffusion"
    assert float(best.loc[0, "log_evidence"]) == 0.0

import pandas as pd

from hipporeplayimm.evidence_reporting import simulation_event_best_rows


def _row(event_index: int, model: str, log_evidence: float, **extra: object) -> dict[str, object]:
    row = {
        "status": "success",
        "session": "RatX/OpenY",
        "event_index": event_index,
        "true_model": "diffusion",
        "expected_model": "sorted-spike-state-space-diffusion",
        "model": model,
        "log_evidence": log_evidence,
        "n_time": 3,
        "n_spikes": 5,
    }
    row.update(extra)
    return row


def test_best_rows_refreshes_partial_and_stale_winner_markers():
    scores = pd.DataFrame(
        [
            _row(
                0,
                "sorted-spike-state-space-stationary",
                -2.0,
                is_best_model=True,
                best_model="sorted-spike-state-space-stationary",
            ),
            _row(
                0,
                "sorted-spike-state-space-diffusion",
                -1.0,
                is_best_model=False,
                best_model="sorted-spike-state-space-stationary",
            ),
            _row(
                1,
                "sorted-spike-state-space-stationary",
                -3.0,
                is_best_model=False,
                best_model="",
            ),
            _row(
                1,
                "sorted-spike-state-space-diffusion",
                -0.5,
                is_best_model=False,
                best_model="",
            ),
        ]
    )

    best = simulation_event_best_rows(scores).sort_values("event_index").reset_index(drop=True)

    assert best["event_index"].tolist() == [0, 1]
    assert best["model"].tolist() == [
        "sorted-spike-state-space-diffusion",
        "sorted-spike-state-space-diffusion",
    ]
    assert best["best_model"].tolist() == best["model"].tolist()
    assert best["is_best_model"].tolist() == [True, True]
    assert best["recovered_expected_model"].tolist() == [True, True]

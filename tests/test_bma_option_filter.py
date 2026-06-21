from __future__ import annotations

import hipporeplayimm.clusterless_ground_truth as clusterless_ground_truth


def test_option_filter_keeps_bma_settings() -> None:
    options = {
        "include_bayesian_model_average": False,
        "bayesian_model_average_name": "custom-bma",
        "bayesian_model_average_evidence_column": "heldout_log_likelihood",
        "clusterless_mark_likelihood": "local-kde",
        "state_space_max_step_sigma": 4.0,
    }

    filtered = clusterless_ground_truth._drop_clusterless_kwargs(options)

    assert filtered["include_bayesian_model_average"] is False
    assert filtered["bayesian_model_average_name"] == "custom-bma"
    assert filtered["bayesian_model_average_evidence_column"] == "heldout_log_likelihood"
    assert filtered["state_space_max_step_sigma"] == 4.0
    assert "clusterless_mark_likelihood" not in filtered

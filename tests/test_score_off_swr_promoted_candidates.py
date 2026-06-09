import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1] / "scripts"))

from score_off_swr_promoted_candidates import (  # noqa: E402
    FULL_CORE_REQUIRED_MODELS,
    PROMOTED_WINDOW_ROLE,
    _encoding_config_from_args,
    build_parser,
    select_candidate_windows,
    write_validation_outputs,
)


def test_parser_defaults_build_current_encoding_config():
    args = build_parser().parse_args(
        [
            "--dataset-root",
            "data/DataSetFromPfeifferFoster",
            "--candidate-table",
            "off_swr_high_specificity_candidate_table.csv",
        ]
    )

    config = _encoding_config_from_args(args)

    assert config.bin_size_cm == args.bin_size_cm
    assert config.smoothing_sigma_bins == args.smoothing_sigma_bins
    assert config.min_speed_cm_s == args.min_speed_cm_s
    assert args.null_random_seed == 1


def test_select_candidate_windows_filters_promotion_ready_and_strong_immobile():
    table = pd.DataFrame(
        [
            _candidate("Rat1/Open1", 0, 0, rank=2, promoted=False, margin=120.0, state="running", distance=2.0),
            _candidate("Rat2/Open2", 1, 0, rank=1, promoted=True, margin=60.0, state="immobile", distance=1.2),
            _candidate("Rat2/Open2", 2, 0, rank=3, promoted=True, margin=40.0, state="immobile", distance=1.2),
            _candidate("Rat3/Open1", 3, 0, rank=4, promoted=True, margin=80.0, state="immobile", distance=0.5),
        ]
    )

    promoted = select_candidate_windows(table, candidate_filter="promotion-ready")

    assert promoted["event_index"].tolist() == [1, 2, 3]
    assert promoted["validation_candidate_index"].tolist() == [0, 1, 2]
    assert promoted["validation_window_role"].eq(PROMOTED_WINDOW_ROLE).all()

    rat2_strong = select_candidate_windows(
        table,
        candidate_filter="strong-immobile",
        session_filter=("Rat2/Open2",),
    )

    assert rat2_strong["event_index"].tolist() == [1]


def test_write_validation_outputs_reports_exact_core_promoted_candidate_support(tmp_path):
    candidates = pd.DataFrame(
        [
            _candidate("Rat2/Open2", 1, 0, rank=1, promoted=True, margin=60.0, state="immobile", distance=1.2),
            _candidate("Rat2/Open2", 2, 1, rank=2, promoted=True, margin=70.0, state="immobile", distance=1.5),
        ]
    )
    candidates["validation_candidate_index"] = [0, 1]
    candidates["validation_window_role"] = PROMOTED_WINDOW_ROLE
    scores = pd.DataFrame(
        [
            *_score_rows("Rat2/Open2", 1, 0, stationary=0.0, trajectory=60.0),
            *_score_rows("Rat2/Open2", 2, 1, stationary=5.0, trajectory=80.0),
        ]
    )

    outputs = write_validation_outputs(
        candidates=candidates,
        scores=scores,
        output=tmp_path,
        comparison_scope="full-core",
        candidate_filter="promotion-ready",
        required_models=FULL_CORE_REQUIRED_MODELS,
        margin_threshold=5.5,
    )

    expected_files = {
        "promoted_off_swr_candidate_exact_core_event_model_evidence.csv",
        "promoted_off_swr_candidate_exact_core_decisions.csv",
        "promoted_off_swr_candidate_exact_core_summary.csv",
        "promoted_off_swr_candidate_exact_core_session_summary.csv",
        "promoted_off_swr_candidate_exact_core_rat_summary.csv",
        "promoted_off_swr_candidate_exact_core_gate_summary.csv",
    }
    assert expected_files.issubset(outputs)
    for filename in expected_files | {"promoted_off_swr_candidate_exact_core_manifest.json"}:
        assert (tmp_path / filename).exists()

    summary = outputs["promoted_off_swr_candidate_exact_core_summary.csv"].iloc[0]
    assert int(summary["selected_candidates"]) == 2
    assert int(summary["required_complete_candidates"]) == 2
    assert int(summary["trajectory_confident_claims"]) == 2
    assert int(summary["strong_exact_candidates"]) == 2
    assert summary["validation_status"] == "exact_core_supports_promoted_off_swr_candidates"

    rat_summary = outputs["promoted_off_swr_candidate_exact_core_rat_summary.csv"].iloc[0]
    assert rat_summary["group"] == "Rat2"
    assert int(rat_summary["trajectory_confident_claims"]) == 2

    gates = outputs["promoted_off_swr_candidate_exact_core_gate_summary.csv"].set_index("gate")
    assert bool(gates.loc["overall", "passed"])


def test_write_validation_outputs_keeps_nonpromoted_exact_core_result_exploratory(tmp_path):
    candidates = pd.DataFrame([_candidate("Rat2/Open2", 1, 0, rank=1, promoted=True, margin=60.0)])
    scores = pd.DataFrame([*_score_rows("Rat2/Open2", 1, 0, stationary=30.0, trajectory=10.0)])

    outputs = write_validation_outputs(
        candidates=candidates,
        scores=scores,
        output=tmp_path,
        comparison_scope="full-core",
        candidate_filter="promotion-ready",
        required_models=FULL_CORE_REQUIRED_MODELS,
        margin_threshold=5.5,
    )

    summary = outputs["promoted_off_swr_candidate_exact_core_summary.csv"].iloc[0]
    assert int(summary["trajectory_confident_claims"]) == 0
    assert int(summary["nontrajectory_confident_claims"]) == 1
    assert summary["validation_status"] == "exact_core_does_not_promote_candidates"


def _candidate(
    session: str,
    event_index: int,
    null_index: int,
    *,
    rank: int,
    promoted: bool,
    margin: float,
    state: str = "immobile",
    distance: float = 1.2,
) -> dict[str, object]:
    return {
        "session": session,
        "event_index": event_index,
        "null_index": null_index,
        "candidate_rank": rank,
        "window_start_s": 10.0 + event_index,
        "window_end_s": 10.12 + event_index,
        "candidate_specificity_label": "interesting_off_swr_trajectory_candidate",
        "candidate_tier": "strong" if margin < 100.0 else "extreme",
        "high_specificity_label": "promotion_ready_high_specificity_candidate",
        "passes_high_specificity_promotion_filter": promoted,
        "trajectory_family_margin": margin,
        "trajectory_confidence": 0.99,
        "run_or_immobility_state": state,
        "animal_speed_mean": 1.0 if state == "immobile" else 20.0,
        "distance_to_nearest_swr_s": distance,
        "candidate_cluster_id": 1,
    }


def _score_rows(session: str, event_index: int, null_index: int, *, stationary: float, trajectory: float) -> list[dict[str, object]]:
    models = [
        ("sorted-spike-state-space-stationary", stationary),
        ("sorted-spike-state-space-diffusion", trajectory - 3.0),
        ("sorted-spike-state-space-fragmented", trajectory - 2.0),
        ("sorted-spike-state-space-first-order-imm", trajectory),
        ("sorted-spike-state-space-momentum-exact-sparse", trajectory - 1.0),
    ]
    return [
        {
            "status": "success",
            "session": session,
            "event_index": event_index,
            "window_role": PROMOTED_WINDOW_ROLE,
            "event_window_variant": "off_swr_promoted_candidate",
            "null_index": null_index,
            "matched_null_rank": null_index + 1,
            "template_event_index": event_index,
            "window_start_s": 10.0 + event_index,
            "window_end_s": 10.12 + event_index,
            "window_duration_s": 0.12,
            "model": model,
            "requested_model": model,
            "log_evidence": log_evidence,
            "n_time": 30,
            "n_spikes": 12,
            "null_active_cell_count": 6,
            "real_n_spikes": 12,
            "n_spikes_delta": 0,
            "n_spikes_relative_delta": 0.0,
            "evidence_comparable": True,
        }
        for model, log_evidence in models
    ]

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from scripts.report_hc11_native_ripple_holdout_replication import build_report
from scripts.score_hc11_webshare_native_ripple_evidence import (
    DECODER_OUTPUT,
    DIRECTION_OUTPUT,
    EVIDENCE_OUTPUT,
    EXCLUSION_OUTPUT,
    GATE_OUTPUT,
    MANIFEST_OUTPUT,
    MODELS,
    SELECTION_OUTPUT,
    UNIT_OUTPUT,
)


def _write_shard(root: Path, *, animal: str, session: str, geometry: str, strict: bool) -> None:
    shard = root / session
    shard.mkdir(parents=True)
    selection = pd.DataFrame(
        {"animal": [animal], "session": [session], "event_id": [1], "geometry": [geometry]}
    )
    rows = []
    values = {
        "stationary": 0.0,
        "diffusion": 1.0,
        "fragmented": 2.0,
        "first_order_imm": 8.0 if strict else 3.0,
    }
    for variant in ("pooled", "direction_mixture"):
        for model in MODELS:
            rows.append(
                {
                    "animal": animal,
                    "session": session,
                    "geometry": geometry,
                    "maze_type": "test maze",
                    "event_id": 1,
                    "selection_rank_within_session": 1,
                    "encoding_variant": variant,
                    "model": model,
                    "log_evidence": values[model],
                    "status": "success",
                    "duration_ms": 50.0,
                    "raw_ripple_duration_ms": 50.0,
                    "n_spikes": 20,
                    "raw_ripple_n_spikes": 20,
                    "n_active_units": 8,
                    "n_time_bins": 5,
                    "mean_stationary_mode_probability": 0.2,
                    "mean_nonstationary_mode_probability": 0.8,
                    "fraction_time_map_nonstationary": 0.8,
                    "posterior_expected_path_length_cm": 30.0,
                    "posterior_net_displacement_cm": 20.0,
                    "posterior_path_speed_cm_s": 600.0,
                }
            )
    pd.DataFrame(rows).to_csv(shard / EVIDENCE_OUTPUT, index=False)
    selection.to_csv(shard / SELECTION_OUTPUT, index=False)
    pd.DataFrame({"animal": [animal], "session": [session], "encoding_units": [10]}).to_csv(
        shard / DECODER_OUTPUT, index=False
    )
    pd.DataFrame({"animal": [animal], "session": [session], "unit_id": [1]}).to_csv(
        shard / UNIT_OUTPUT, index=False
    )
    pd.DataFrame(
        {"animal": [animal], "session": [session], "event_id": [99], "source_count": [1]}
    ).to_csv(shard / EXCLUSION_OUTPUT, index=False)
    pd.DataFrame(
        {
            "gate": ["selected_events_exclude_prior_pilots"],
            "passed": [True],
            "detail": ["overlap=0"],
        }
    ).to_csv(shard / GATE_OUTPUT, index=False)
    pd.DataFrame({"animal": [animal], "session": [session], "event_id": [1]}).to_csv(
        shard / DIRECTION_OUTPUT, index=False
    )
    manifest = {
        "event_definition": "native",
        "cohort_label": "holdout",
        "models": list(MODELS),
        "parameters": {"time_bin_s": 0.01, "event_padding_s": 0.0, "event_ranking": "spike_support"},
    }
    (shard / MANIFEST_OUTPUT).write_text(json.dumps(manifest), encoding="utf-8")


def test_report_stops_when_strict_subset_is_not_distributed(tmp_path: Path) -> None:
    shards = tmp_path / "shards"
    _write_shard(shards, animal="RatA", session="RatA_day1", geometry="linear", strict=True)
    _write_shard(shards, animal="RatB", session="RatB_day2", geometry="circular", strict=False)

    result = build_report(
        shard_root=shards,
        output_dir=tmp_path / "report",
        expected_events_per_session=1,
        margin_threshold=5.5,
    )

    assert result["verdict"] == "external_clean_imm_replication_not_established"
    gates = result["gates"].set_index("gate")
    assert bool(gates.loc["overall_technical", "passed"])
    assert not bool(gates.loc["strict_clean_imm_spans_both_animals", "passed"])
    assert result["model_summary"].iloc[0]["strict_clean_imm_count"] == 1
    report = (tmp_path / "report" / "hc11_native_ripple_holdout_replication_report.md").read_text()
    assert "stop_full_gate_ladder_no_distributed_strict_subset" in report

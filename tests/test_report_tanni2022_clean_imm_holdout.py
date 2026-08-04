from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from scripts.report_tanni2022_clean_imm_holdout import build_report
from scripts.score_tanni2022_clean_imm_holdout import (
    EVIDENCE_OUTPUT,
    GATE_OUTPUT,
    MANIFEST_OUTPUT,
    MODELS,
    SELECTION_OUTPUT,
    UNIT_OUTPUT,
)


def _write_shard(root: Path, *, animal: str, strict: bool) -> None:
    shard = root / animal
    shard.mkdir(parents=True)
    selection = pd.DataFrame(
        {
            "animal": [animal],
            "session": [f"{animal}_session"],
            "event_index": [1],
            "selection_rank_within_animal": [1],
            "n_spikes": [30],
            "n_active_cells": [12],
            "peak_ripple_z": [15.0],
            "excluded_prior_model_event": [False],
        }
    )
    values = {
        "stationary": 0.0,
        "diffusion": 2.0,
        "fragmented": 1.0,
        "first_order_imm": 8.0 if strict else 3.0,
        "exact_sparse_momentum": 4.0,
    }
    evidence = pd.DataFrame(
        {
            "animal": animal,
            "session": f"{animal}_session",
            "event_index": 1,
            "model": list(MODELS),
            "log_evidence": [values[model] for model in MODELS],
            "status": "success",
            "evidence_comparable": True,
        }
    )
    selection.to_csv(shard / SELECTION_OUTPUT, index=False)
    evidence.to_csv(shard / EVIDENCE_OUTPUT, index=False)
    pd.DataFrame({"animal": [animal], "unit_id": [1]}).to_csv(shard / UNIT_OUTPUT, index=False)
    pd.DataFrame({"gate": ["overall_technical"], "passed": [True], "detail": ["pass"]}).to_csv(
        shard / GATE_OUTPUT, index=False
    )
    manifest = {
        "parameters": {"animal": [animal], "decode_bin_s": 0.02},
        "input_file_sha256": {"selection_csv": "frozen-selection-hash"},
    }
    (shard / MANIFEST_OUTPUT).write_text(json.dumps(manifest), encoding="utf-8")


def test_report_rejects_clean_imm_when_subset_is_not_distributed(tmp_path: Path) -> None:
    shards = tmp_path / "shards"
    _write_shard(shards, animal="RatA", strict=True)
    _write_shard(shards, animal="RatB", strict=False)

    result = build_report(
        shard_root=shards,
        output_dir=tmp_path / "report",
        expected_animals=2,
        events_per_animal=1,
        margin_threshold=5.5,
        minimum_positive_events=2,
        minimum_positive_animals=2,
    )

    assert result["verdict"] == "large_2d_clean_imm_replication_not_established"
    assert result["geometry_verdict"] == "unconstrained_2d_not_sufficient_for_clean_imm"
    gates = result["gates"].set_index("gate")
    assert bool(gates.loc["overall_technical", "passed"])
    assert not bool(gates.loc["strict_clean_imm_subset_distributed", "passed"])
    report = (tmp_path / "report" / "tanni2022_clean_imm_holdout_report.md").read_text()
    assert "not sufficient" in report

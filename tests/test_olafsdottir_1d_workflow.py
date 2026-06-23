from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys

import pandas as pd


def _load_workflow_module():
    repo_root = Path(__file__).resolve().parents[1]
    src_path = repo_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    module_path = repo_root / "scripts" / "benchmark_olafsdottir_1d_replay_evidence.py"
    spec = importlib.util.spec_from_file_location("benchmark_olafsdottir_1d_replay_evidence", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_olafsdottir_1d_summary_writer_emits_required_tables(tmp_path: Path) -> None:
    module = _load_workflow_module()
    scores = _synthetic_scores(module)
    conversion = tmp_path / "derived" / "olafsdottir_ztrack_conversion_summary.csv"
    conversion.parent.mkdir()
    pd.DataFrame([{"session": "R2142/ZTrack20140806", "ripple_events": 21, "included_cells": 34}]).to_csv(conversion, index=False)

    tables = module.write_olafsdottir_outputs(
        scores,
        tmp_path / "out",
        session="R2142/ZTrack20140806",
        max_events=2,
        margin_threshold=5.5,
        models=module.DEFAULT_MODELS,
        derived_root=conversion.parent,
        benchmark_output=tmp_path / "benchmark",
        conversion_summary=conversion,
    )

    for name in module.REQUIRED_OUTPUTS:
        assert (tmp_path / "out" / name).is_file()
    evidence = pd.read_csv(tmp_path / "out" / "olafsdottir_1d_event_model_evidence.csv")
    assert len(evidence) == 10
    decisions = tables["family_margin_decisions"]
    assert len(decisions) == 2
    assert decisions["complete_exact_core"].map(bool).all()

    family_summary = pd.read_csv(tmp_path / "out" / "olafsdottir_1d_family_margin_summary.csv")
    assert int(family_summary.loc[0, "events"]) == 2
    assert int(family_summary.loc[0, "trajectory_confident_claims"]) == 1
    assert int(family_summary.loc[0, "ambiguous_events"]) == 1

    exact_claims = pd.read_csv(tmp_path / "out" / "olafsdottir_1d_exact_core_model_claim_summary.csv")
    claim_by_model = exact_claims.set_index("model")
    assert int(claim_by_model.loc[module.FIRST_ORDER_IMM_MODEL, "raw_best_events"]) == 1
    assert int(claim_by_model.loc[module.MOMENTUM_MODEL, "raw_best_events"]) == 1

    paired = pd.read_csv(tmp_path / "out" / "olafsdottir_1d_paired_momentum_diffusion_summary.csv")
    assert int(paired.loc[0, "paired_events"]) == 2
    assert int(paired.loc[0, "momentum_raw_wins"]) == 1
    assert int(paired.loc[0, "diffusion_raw_wins"]) == 1

    gates = pd.read_csv(tmp_path / "out" / "olafsdottir_1d_control_gate_summary.csv")
    assert set(gates["gate"]) >= {
        "derived_session_conversion_available",
        "event_model_evidence_nonempty",
        "exact_core_models_present",
        "all_scored_events_have_exact_core",
        "no_failed_model_rows",
        "summary_tables_written",
        "biological_claim_not_assessed",
    }
    assert gates["passed"].map(bool).all()


def test_command_builders_wire_bridge_adapter_and_exact_core_models(tmp_path: Path) -> None:
    module = _load_workflow_module()
    args = argparse.Namespace(
        extracted_root=Path("/home/github-runner/.cache/datasets/olafsdottir2016/extracted"),
        derived_root=tmp_path / "derived",
        session="R2142/ZTrack20140806",
        tetrode_mode="hippocampus",
        lfp_detector_mode="mean-envelope",
        min_event_spikes=5,
        min_event_active_cells=3,
        lfp_channels="1-4",
        ripple_high_threshold_z=2.25,
        ripple_low_threshold_z=0.75,
        events="all",
        models=" ".join(module.DEFAULT_MODELS),
        bin_size_cm=5.0,
        min_speed_cm_s=4.0,
        time_bin_s=0.02,
        max_events=5,
    )

    prepare = module.build_prepare_command(args, tmp_path / "derived")
    benchmark = module.build_benchmark_command(args, tmp_path / "derived", tmp_path / "benchmark")

    assert "prepare_olafsdottir_ztrack_sessions.py" in " ".join(prepare)
    assert "--sessions" in prepare
    assert "R2142/ZTrack20140806" in prepare
    assert "--min-event-spikes" in prepare
    assert "benchmark_model_evidence.py" in " ".join(benchmark)
    assert "--max-events" in benchmark
    assert "5" in benchmark
    benchmark_text = " ".join(benchmark)
    for model in module.DEFAULT_MODELS:
        assert model in benchmark_text


def test_workflow_dispatch_wires_1d_evidence_smoke_outputs() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    workflow = (repo_root / ".github" / "workflows" / "olafsdottir-1d-evidence.yml").read_text(encoding="utf-8")

    assert "workflow_dispatch:" in workflow
    assert "runs-on: self-hosted" in workflow
    assert "scripts/benchmark_olafsdottir_1d_replay_evidence.py" in workflow
    assert "max_events:" in workflow
    assert "R2142/ZTrack20140806" in workflow
    assert "actions/upload-artifact@v7" in workflow
    for model in _load_workflow_module().DEFAULT_MODELS:
        assert model in workflow
    for name in _load_workflow_module().REQUIRED_OUTPUTS:
        assert name in workflow


def _synthetic_scores(module) -> pd.DataFrame:
    rows = []
    log_evidence_by_event = {
        0: {
            module.STATIONARY_MODEL: 1.0,
            module.DIFFUSION_MODEL: 6.0,
            module.FRAGMENTED_MODEL: 4.0,
            module.FIRST_ORDER_IMM_MODEL: 10.0,
            module.MOMENTUM_MODEL: 5.0,
        },
        1: {
            module.STATIONARY_MODEL: 10.0,
            module.DIFFUSION_MODEL: 9.0,
            module.FRAGMENTED_MODEL: 8.0,
            module.FIRST_ORDER_IMM_MODEL: 8.5,
            module.MOMENTUM_MODEL: 11.0,
        },
    }
    for event_index, model_values in log_evidence_by_event.items():
        for model, value in model_values.items():
            rows.append(
                {
                    "session": "R2142/ZTrack20140806",
                    "event_index": event_index,
                    "model": model,
                    "requested_model": model,
                    "model_family": module.model_family(model),
                    "status": "success",
                    "log_evidence": value,
                    "n_spikes": 8 + event_index,
                    "duration_s": 0.08,
                    "runtime_s": 0.0,
                    "diagnostic_evidence_support": "exact_full_grid",
                    "diagnostic_evidence_comparable": True,
                    "diagnostic_evidence_comparison": "exact_full_grid",
                }
            )
    return pd.DataFrame(rows)

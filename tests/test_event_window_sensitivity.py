from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from aggregate_event_window_sensitivity import aggregate_event_window_sensitivity  # noqa: E402
from benchmark_model_evidence_improved import (  # noqa: E402
    _event_windows,
    _parse_window_variant_specs,
)


def test_named_window_variant_specs_build_expected_windows():
    assert _parse_window_variant_specs("core:0:0 shifted_pre:0.05:-0.05") == (
        ("core", 0.0, 0.0),
        ("shifted_pre", 0.05, -0.05),
    )

    args = SimpleNamespace(
        window_variant_specs="core:0:0 contracted:-0.004:-0.004 shifted_pre:0.05:-0.05",
        window_pre_pads_s="0.0",
        window_post_pads_s="0.0",
        window_min_duration_s=0.004,
    )
    event = SimpleNamespace(start=10.0, end=10.04)

    windows = _event_windows(args, event)

    assert windows["event_window_variant"].tolist() == [
        "core",
        "contracted",
        "shifted_pre",
    ]
    assert windows.loc[0, "window_start_s"] == 10.0
    assert windows.loc[1, "window_start_s"] == 10.004
    assert windows.loc[1, "window_end_s"] == 10.036
    assert windows.loc[2, "window_start_s"] == 9.95
    assert windows.loc[2, "window_end_s"] == pytest.approx(9.99)


def test_event_window_aggregator_compares_variants_to_core(tmp_path):
    shard_root = tmp_path / "shards" / "shard0"
    shard_root.mkdir(parents=True)
    rows = []
    for variant, trajectory_logz, stationary_logz in (
        ("core", 20.0, 0.0),
        ("contracted", 16.0, 0.0),
        ("expanded", 14.0, 0.0),
        ("shifted_pre", 2.0, 0.0),
    ):
        rows.extend(
            [
                _row(variant, "sorted-spike-state-space-stationary", "nontrajectory", stationary_logz),
                _row(variant, "sorted-spike-state-space-diffusion", "trajectory", trajectory_logz - 3.0),
                _row(variant, "sorted-spike-state-space-fragmented", "trajectory", trajectory_logz - 2.0),
                _row(variant, "sorted-spike-state-space-first-order-imm", "trajectory", trajectory_logz),
                _row(
                    variant,
                    "sorted-spike-state-space-momentum-exact-sparse",
                    "trajectory",
                    trajectory_logz - 1.0,
                ),
            ]
        )
    pd.DataFrame(rows).to_csv(shard_root / "event_model_evidence.csv", index=False)

    out = tmp_path / "out"
    aggregate_event_window_sensitivity(str(tmp_path / "shards" / "**" / "event_model_evidence.csv"), out)

    summary = pd.read_csv(out / "event_window_family_margin_summary.csv")
    comparison = pd.read_csv(out / "event_window_comparison_to_core.csv")
    gates = pd.read_csv(out / "event_window_control_gate_summary.csv")

    core = summary.set_index("event_window_variant").loc["core"]
    shifted = comparison.set_index("event_window_variant").loc["shifted_pre"]
    overall = gates.set_index("gate").loc["overall"]
    assert core["trajectory_confident_claims"] == 1
    assert shifted["mean_best_trajectory_log_evidence_minus_core"] < 0.0
    assert shifted["mean_family_margin_minus_core"] < 0.0
    assert bool(overall["passed"])


def test_event_window_sensitivity_workflow_runs_named_variants():
    workflow = Path(".github/workflows/event-window-sensitivity.yml").read_text(encoding="utf-8")

    assert "name: Event-window sensitivity controls" in workflow
    assert "window_variant_specs:" in workflow
    assert "core:0.000:0.000" in workflow
    assert "contracted:-0.004:-0.004" in workflow
    assert "shifted_pre:0.050:-0.050" in workflow
    assert "scripts/benchmark_model_evidence_improved.py" in workflow
    assert "--window-variant-specs" in workflow
    assert "timeout-minutes: 360" in workflow
    assert (
        "timeout 350m python scripts/benchmark_model_evidence_improved.py"
        in workflow
    )
    assert "shard_status.csv" in workflow
    assert "event_window_shard_status.csv" in workflow
    assert "needs.plan-session-event-shards.result == 'success'" in workflow
    assert "scripts/aggregate_event_window_sensitivity.py" in workflow
    assert "event-window-sensitivity-${{ github.run_id }}" in workflow


def _row(
    variant: str,
    model: str,
    family: str,
    log_evidence: float,
) -> dict[str, object]:
    return {
        "status": "success",
        "session": "Rat1/Open1",
        "event_index": 0,
        "window_index": 0,
        "event_window_variant": variant,
        "window_pre_pad_s": 0.0,
        "window_post_pad_s": 0.0,
        "window_start_s": 1.0,
        "window_end_s": 1.04,
        "window_duration_s": 0.04,
        "model": model,
        "requested_model": model,
        "model_family": family,
        "log_evidence": log_evidence,
        "n_time": 10,
        "n_spikes": 20,
        "runtime_s": 0.1,
        "error": "",
        "bin_size_cm": 6.0,
        "smoothing_sigma_bins": 2.0,
        "min_speed_cm_s": 5.0,
        "time_bin_s": 0.004,
    }

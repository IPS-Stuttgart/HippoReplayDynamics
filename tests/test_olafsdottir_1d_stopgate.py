from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pandas as pd


def _load_module():
    repo_root = Path(__file__).resolve().parents[1]
    scripts_path = repo_root / "scripts"
    if str(scripts_path) not in sys.path:
        sys.path.insert(0, str(scripts_path))
    module_path = scripts_path / "summarize_olafsdottir_1d_stopgate.py"
    spec = importlib.util.spec_from_file_location("summarize_olafsdottir_1d_stopgate", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_stopgate_records_do_not_scale_decision(tmp_path: Path) -> None:
    module = _load_module()
    reports = _write_report_manifests(tmp_path)
    comparison_dir = _write_comparison_pack(
        tmp_path / "comparison",
        rows=[
            _tier_row("balanced_debug", events=20, trajectory_confident=1, trajectory_median=-0.80, imm_confident=2, imm_median=2.53),
            _tier_row("high_information_debug", events=20, trajectory_confident=4, trajectory_median=-0.88, imm_confident=5, imm_median=4.15),
            _tier_row("high_information_holdout19_debug", events=19, trajectory_confident=4, trajectory_median=-0.91, imm_confident=5, imm_median=4.48),
        ],
        normalized=[
            _normalized_row("balanced_debug", "trajectory_minus_stationary_per_second", -23.5),
            _normalized_row("balanced_debug", "trajectory_minus_stationary_per_spike", -0.005),
            _normalized_row("high_information_debug", "trajectory_minus_stationary_per_second", -21.8),
            _normalized_row("high_information_debug", "trajectory_minus_stationary_per_spike", -0.002),
            _normalized_row("high_information_holdout19_debug", "trajectory_minus_stationary_per_second", -22.6),
            _normalized_row("high_information_holdout19_debug", "trajectory_minus_stationary_per_spike", -0.0025),
        ],
        recommendation="continue_imm_fragmented_taxonomy_audit_only",
    )
    out = tmp_path / "stopgate"

    tables = module.run_stopgate_summary(
        balanced_report_dir=reports["balanced_debug"],
        high_information_report_dir=reports["high_information_debug"],
        holdout_report_dir=reports["high_information_holdout19_debug"],
        comparison_dir=comparison_dir,
        output_dir=out,
    )

    assert (out / module.SUMMARY_OUTPUT).is_file()
    assert (out / module.GATE_OUTPUT).is_file()
    assert (out / module.DECISION_OUTPUT).is_file()
    summary = tables["summary"].iloc[0]
    assert bool(summary["technical_scoreable"])
    assert not bool(summary["trajectory_family_over_static_supported"])
    assert summary["imm_fragmented_axis_supported"] == "weak_or_partial"
    assert summary["recommended_next_action"] == "continue_imm_fragmented_taxonomy_audit_only"
    assert bool(summary["forbid_pilot50_biology"])
    assert bool(summary["raw_trajectory_medians_negative"])
    assert bool(summary["normalized_trajectory_medians_negative"])
    gates = tables["gates"].set_index("gate")
    assert bool(gates.loc["technical_scoreable", "passed"])
    assert not bool(gates.loc["trajectory_family_over_static_supported", "passed"])
    assert bool(gates.loc["forbid_pilot50_biology", "passed"])
    decision = (out / module.DECISION_OUTPUT).read_text(encoding="utf-8")
    assert "technical portability and specificity result" in decision
    assert "forbid_pilot50_biology: true" in decision


def test_stopgate_allows_positive_holdout_case(tmp_path: Path) -> None:
    module = _load_module()
    reports = _write_report_manifests(tmp_path)
    comparison_dir = _write_comparison_pack(
        tmp_path / "comparison",
        rows=[
            _tier_row("balanced_debug", events=20, trajectory_confident=2, trajectory_median=-0.5, imm_confident=2, imm_median=2.0),
            _tier_row("high_information_debug", events=20, trajectory_confident=12, trajectory_median=4.0, imm_confident=10, imm_median=6.0),
            _tier_row("high_information_holdout19_debug", events=20, trajectory_confident=12, trajectory_median=4.0, imm_confident=10, imm_median=6.0),
        ],
        normalized=[
            _normalized_row("balanced_debug", "trajectory_minus_stationary_per_second", -10.0),
            _normalized_row("balanced_debug", "trajectory_minus_stationary_per_spike", -0.01),
            _normalized_row("high_information_debug", "trajectory_minus_stationary_per_second", 80.0),
            _normalized_row("high_information_debug", "trajectory_minus_stationary_per_spike", 0.05),
            _normalized_row("high_information_holdout19_debug", "trajectory_minus_stationary_per_second", 80.0),
            _normalized_row("high_information_holdout19_debug", "trajectory_minus_stationary_per_spike", 0.05),
        ],
        recommendation="define_frozen_high_information_confirmation_tier",
    )

    tables = module.run_stopgate_summary(
        balanced_report_dir=reports["balanced_debug"],
        high_information_report_dir=reports["high_information_debug"],
        holdout_report_dir=reports["high_information_holdout19_debug"],
        comparison_dir=comparison_dir,
        output_dir=tmp_path / "stopgate",
    )

    summary = tables["summary"].iloc[0]
    assert bool(summary["trajectory_family_over_static_supported"])
    assert summary["imm_fragmented_axis_supported"] == "supported"
    assert summary["recommended_next_action"] == "scale_biological_1d_pilot"
    assert not bool(summary["forbid_pilot50_biology"])


def _write_report_manifests(root: Path) -> dict[str, Path]:
    paths = {}
    for label in ["balanced_debug", "high_information_debug", "high_information_holdout19_debug"]:
        path = root / label
        path.mkdir()
        (path / "olafsdottir_1d_sleep_debug_report_manifest.json").write_text(
            '{"technical_classification": "technical-pass", "biological_classification": "biological-ambiguous"}\n',
            encoding="utf-8",
        )
        paths[label] = path
    return paths


def _write_comparison_pack(
    root: Path,
    *,
    rows: list[dict[str, object]],
    normalized: list[dict[str, object]],
    recommendation: str,
) -> Path:
    root.mkdir()
    pd.DataFrame(rows).to_csv(root / "olafsdottir_1d_pilot_tier_comparison.csv", index=False)
    pd.DataFrame(normalized).to_csv(root / "olafsdottir_1d_pilot_tier_normalized_margin_comparison.csv", index=False)
    pd.DataFrame(
        [
            {
                "recommendation": recommendation,
                "localized_signal": False,
            }
        ]
    ).to_csv(root / "olafsdottir_1d_pilot_tier_decision_summary.csv", index=False)
    return root


def _tier_row(
    tier: str,
    *,
    events: int,
    trajectory_confident: int,
    trajectory_median: float,
    imm_confident: int,
    imm_median: float,
) -> dict[str, object]:
    return {
        "tier_label": tier,
        "pilot_tier": tier,
        "events": events,
        "animals": 6,
        "pairs": 10,
        "trajectory_confident_events": trajectory_confident,
        "nontrajectory_confident_events": 0,
        "imm_confident_events": imm_confident,
        "fragmented_confident_events": 0,
        "positive_trajectory_margin_events": trajectory_confident,
        "positive_imm_margin_events": events - 2,
        "mean_delta_best_trajectory_minus_stationary": trajectory_median,
        "median_delta_best_trajectory_minus_stationary": trajectory_median,
        "mean_delta_imm_minus_fragmented": imm_median,
        "median_delta_imm_minus_fragmented": imm_median,
        "biological_claim_assessed": False,
    }


def _normalized_row(tier: str, margin: str, median: float) -> dict[str, object]:
    return {
        "tier_label": tier,
        "margin": margin,
        "events_with_finite_margin": 20,
        "mean_margin": median,
        "median_margin": median,
        "p25_margin": median,
        "p75_margin": median,
        "positive_margin_events": 12 if median > 0 else 2,
        "positive_margin_fraction": 0.6 if median > 0 else 0.1,
    }

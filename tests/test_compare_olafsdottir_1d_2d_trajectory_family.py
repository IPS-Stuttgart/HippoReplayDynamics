from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pandas as pd


def _load_compare_module():
    repo_root = Path(__file__).resolve().parents[1]
    src_path = repo_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    module_path = repo_root / "scripts" / "compare_olafsdottir_1d_2d_trajectory_family.py"
    spec = importlib.util.spec_from_file_location("compare_olafsdottir_1d_2d_trajectory_family", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_compare_1d_2d_writes_primary_and_normalized_columns(tmp_path: Path) -> None:
    module = _load_compare_module()
    one_d = _scores(module, "R2142/ZTrack20140806", [0.2, -0.1, 0.0, 0.4], [module.STATIONARY_MODEL] * 4)
    two_d = _scores(
        module,
        "Rat1/Open1",
        [12.0, 20.0, 18.0, 15.0, 22.0, 17.0],
        [module.FIRST_ORDER_IMM_MODEL] * 5 + [module.MOMENTUM_MODEL],
    )

    tables = module.build_comparison(
        one_d_scores=one_d,
        two_d_scores=two_d,
        output=tmp_path,
        min_robust_1d_events=50,
    )

    summary_path = tmp_path / module.SUMMARY_OUTPUT
    interpretation_path = tmp_path / module.INTERPRETATION_OUTPUT
    assert summary_path.is_file()
    assert interpretation_path.is_file()
    summary = pd.read_csv(summary_path)
    interpretation = pd.read_csv(interpretation_path)
    assert list(summary.columns[: len(module.PRIMARY_COLUMNS)]) == list(module.PRIMARY_COLUMNS)
    assert set(summary["environment_type"]) == {"1D_Z_track", "2D_open_field"}
    for column in (
        "mean_family_margin_per_spike",
        "median_family_margin_per_spike",
        "mean_family_margin_per_time_bin",
        "median_family_margin_per_time_bin",
        "mean_spikes_per_event",
        "median_spikes_per_event",
        "mean_time_bins_per_event",
        "median_time_bins_per_event",
    ):
        assert column in summary
        assert summary[column].notna().all()
    one_row = summary[summary["environment_type"] == "1D_Z_track"].iloc[0]
    two_row = summary[summary["environment_type"] == "2D_open_field"].iloc[0]
    assert one_row["trajectory_confident_claim_fraction"] == 0.0
    assert two_row["trajectory_confident_claim_fraction"] == 1.0
    assert two_row["first_order_imm_raw_best_fraction"] > one_row["first_order_imm_raw_best_fraction"]

    result = interpretation.iloc[0]
    assert result["interpretation_class"] == "sparse_or_data_limited_feasibility_result"
    assert result["directional_pattern"] == "weaker_1d_signal"
    assert result["claim_strength"] == "hypothesis_generating_smoke"
    assert "Do not claim IMM is only apparent in 2D" in result["hard_caveat"]
    assert tables["one_d_event_metrics"].shape[0] == 4
    assert tables["two_d_event_metrics"].shape[0] == 6


def test_interpretation_supports_strong_trajectory_but_weaker_imm_dominance() -> None:
    module = _load_compare_module()
    summary = pd.DataFrame(
        [
            {
                "dataset": "Olafsdottir2016",
                "environment_type": "1D_Z_track",
                "events": 80,
                "trajectory_confident_claim_fraction": 0.75,
                "nontrajectory_confident_claim_fraction": 0.0,
                "median_family_margin": 12.0,
                "median_family_margin_per_spike": 1.0,
                "median_family_margin_per_time_bin": 2.0,
                "first_order_imm_raw_best_fraction": 0.30,
                "momentum_raw_best_fraction": 0.35,
            },
            {
                "dataset": "PfeifferFoster",
                "environment_type": "2D_open_field",
                "events": 160,
                "trajectory_confident_claim_fraction": 0.80,
                "nontrajectory_confident_claim_fraction": 0.0,
                "median_family_margin": 76.0,
                "median_family_margin_per_spike": 1.0,
                "median_family_margin_per_time_bin": 2.0,
                "first_order_imm_raw_best_fraction": 0.79,
                "momentum_raw_best_fraction": 0.14,
            },
        ]
    )

    interpretation = module.interpretation_summary(
        summary,
        margin_threshold=5.5,
        min_robust_1d_events=50,
        weaker_fraction_delta=0.20,
        similar_fraction_delta=0.10,
    )

    row = interpretation.iloc[0]
    assert row["interpretation_class"] == "strong_trajectory_family_but_weaker_imm_dominance"
    assert row["claim_strength"] == "robust_comparison_candidate"
    assert "weaker first-order IMM dominance" in row["paper_safe_statement"]


def test_resolve_event_table_accepts_artifact_directories(tmp_path: Path) -> None:
    module = _load_compare_module()
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    expected = artifact / "olafsdottir_1d_event_model_evidence.csv"
    expected.write_text("session,event_index,model,log_evidence\n", encoding="utf-8")

    assert module.resolve_event_table(artifact) == expected
    assert module.resolve_event_table(expected) == expected


def _scores(module, session: str, margins: list[float], best_models: list[str]) -> pd.DataFrame:
    rows = []
    for event_index, (margin, best_model) in enumerate(zip(margins, best_models, strict=True)):
        stationary = 0.0
        values = {
            module.STATIONARY_MODEL: stationary,
            module.DIFFUSION_MODEL: stationary + margin - 2.0,
            module.FRAGMENTED_MODEL: stationary + margin - 4.0,
            module.FIRST_ORDER_IMM_MODEL: stationary + margin - 1.0,
            module.MOMENTUM_MODEL: stationary + margin - 3.0,
        }
        if best_model in values:
            values[best_model] = stationary + margin
        if best_model == module.STATIONARY_MODEL:
            values[module.STATIONARY_MODEL] = stationary + max(0.0, margin) + 1.0
        for model, log_evidence in values.items():
            rows.append(
                {
                    "session": session,
                    "event_index": event_index,
                    "model": model,
                    "status": "success",
                    "log_evidence": log_evidence,
                    "n_spikes": 10 + event_index,
                    "n_time": 5 + event_index,
                    "diagnostic_evidence_support": "exact_full_grid",
                    "diagnostic_evidence_comparable": True,
                    "diagnostic_evidence_comparison": "exact_model_evidence",
                }
            )
    return pd.DataFrame(rows)

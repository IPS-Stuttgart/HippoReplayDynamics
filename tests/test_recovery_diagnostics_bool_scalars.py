from hipporeplayimm.recovery_diagnostics import _classify_failure_mode


def test_recovery_failure_mode_parses_csv_float_bool_scalars():
    row = {
        "successful_scores": 1,
        "expected_model_scored": "1.0",
        "strict_recovered_expected_model": "0.0",
        "certified_vs_exact_recovered_expected_model": "1.0",
        "true_model": "momentum",
        "exact_displacement_momentum_beats_diffusion_exact": "0.0",
        "expected_model_evidence_support": "truncated_full_grid",
        "comparable_scores": 1,
        "expected_model_evidence_comparable": "0.0",
    }

    assert _classify_failure_mode(row) == "strict_gate_excluded_certified_lower_bound"

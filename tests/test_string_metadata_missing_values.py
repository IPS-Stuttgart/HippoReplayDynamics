import pandas as pd

from hipporeplayimm.score_metadata import _unique_string_from_columns
import hipporeplayimm.clusterless_ground_truth as clusterless_ground_truth


def test_score_metadata_string_aliases_ignore_literal_missing_strings():
    scores = pd.DataFrame(
        {
            "state_space_momentum_candidate_source": ["emission", "emission"],
            "diagnostic_state_space_momentum_candidate_source": ["nan", " none "],
        }
    )

    assert (
        _unique_string_from_columns(
            scores,
            (
                "state_space_momentum_candidate_source",
                "diagnostic_state_space_momentum_candidate_source",
            ),
            "fallback",
        )
        == "emission"
    )


def test_score_metadata_string_aliases_use_default_when_all_values_are_missing_strings():
    scores = pd.DataFrame(
        {
            "diagnostic_state_space_momentum_candidate_source": [
                "nan",
                "none",
                "<NA>",
                "",
            ]
        }
    )

    assert (
        _unique_string_from_columns(
            scores,
            ("diagnostic_state_space_momentum_candidate_source",),
            "emission",
        )
        == "emission"
    )


def test_clusterless_string_and_optional_float_metadata_ignore_literal_missing_strings():
    scores = pd.DataFrame(
        {
            "clusterless_mark_likelihood": ["local-kde"],
            "diagnostic_clusterless_mark_likelihood": ["nan"],
            "clusterless_mark_kde_bandwidth": ["none"],
        }
    )

    assert (
        clusterless_ground_truth._unique_string_from_columns(
            scores,
            ("clusterless_mark_likelihood", "diagnostic_clusterless_mark_likelihood"),
            "local-kde",
        )
        == "local-kde"
    )
    assert clusterless_ground_truth._optional_float_from_columns(scores, ("clusterless_mark_kde_bandwidth",), None) is None


def test_string_metadata_helpers_ignore_textual_null_sentinel():
    sentinel = "nu" "ll"
    scores = pd.DataFrame(
        {
            "diagnostic_state_space_momentum_candidate_source": [
                sentinel,
                f" {sentinel.upper()} ",
            ],
            "clusterless_mark_likelihood": ["local-kde", "local-kde"],
            "diagnostic_clusterless_mark_likelihood": [
                sentinel,
                f" {sentinel.upper()} ",
            ],
            "clusterless_mark_kde_bandwidth": [
                sentinel,
                f" {sentinel.upper()} ",
            ],
        }
    )

    assert (
        _unique_string_from_columns(
            scores,
            ("diagnostic_state_space_momentum_candidate_source",),
            "emission",
        )
        == "emission"
    )
    assert (
        clusterless_ground_truth._unique_string_from_columns(
            scores,
            ("clusterless_mark_likelihood", "diagnostic_clusterless_mark_likelihood"),
            "local-kde",
        )
        == "local-kde"
    )
    assert clusterless_ground_truth._optional_float_from_columns(scores, ("clusterless_mark_kde_bandwidth",), None) is None

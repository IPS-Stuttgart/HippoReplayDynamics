from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hipporeplayimm.sign_flip_report import (
    main,
    paired_sign_flip_test,
    score_table_sign_flip_summary,
)


def test_paired_sign_flip_uses_exact_small_sample_distribution() -> None:
    result = paired_sign_flip_test([1.0, 2.0, 3.0])

    assert result.method == "exact"
    assert result.permutations_evaluated == 8
    assert result.random_seed is None
    assert result.p_value == pytest.approx(0.25)


def test_zero_deltas_do_not_expand_exact_enumeration() -> None:
    result = paired_sign_flip_test([1.0, 2.0, 3.0, 0.0, 0.0])

    assert result.n_observations == 5
    assert result.n_nonzero == 3
    assert result.permutations_evaluated == 8
    assert result.p_value == pytest.approx(0.25)


def test_all_zero_deltas_return_unit_exact_p_value() -> None:
    result = paired_sign_flip_test(np.zeros(5))

    assert result.method == "exact"
    assert result.permutations_evaluated == 1
    assert result.p_value == 1.0


def test_exact_enumeration_rejects_more_sign_bits_than_uint64_can_encode() -> None:
    with pytest.raises(ValueError, match="at most 64 nonzero observations"):
        paired_sign_flip_test(np.ones(65), max_exact_n=65)


def test_oversized_sample_can_fall_back_to_monte_carlo() -> None:
    result = paired_sign_flip_test(
        np.ones(65),
        max_exact_n=64,
        n_permutations=20,
        random_seed=7,
        chunk_size=5,
    )

    assert result.method == "monte_carlo"
    assert result.permutations_evaluated == 20
    assert result.random_seed == 7


def test_monte_carlo_path_is_reproducible() -> None:
    values = np.linspace(-2.0, 3.0, 12)

    first = paired_sign_flip_test(
        values,
        max_exact_n=4,
        n_permutations=500,
        random_seed=17,
        chunk_size=37,
    )
    second = paired_sign_flip_test(
        values,
        max_exact_n=4,
        n_permutations=500,
        random_seed=17,
        chunk_size=37,
    )

    assert first == second
    assert first.method == "monte_carlo"
    assert first.permutations_evaluated == 500
    assert first.random_seed == 17
    assert 0.0 < first.p_value <= 1.0


def test_score_table_summary_reports_method_per_model() -> None:
    frame = pd.DataFrame(
        {
            "model": ["imm"] * 3 + ["momentum"] * 6,
            "delta_vs_best_static": [1.0, 2.0, 3.0, 1.0, -1.0, 2.0, -2.0, 3.0, -3.0],
        }
    )

    summary = score_table_sign_flip_summary(
        frame,
        max_exact_n=3,
        n_permutations=100,
        random_seed=4,
    )

    assert summary["model"].tolist() == ["imm", "momentum"]
    assert summary["method"].tolist() == ["exact", "monte_carlo"]
    assert summary["permutations_evaluated"].tolist() == [8, 100]
    assert pd.isna(summary.loc[0, "random_seed"])
    assert summary.loc[1, "random_seed"] == 5


def test_score_table_summary_rejects_populated_nonnumeric_values() -> None:
    frame = pd.DataFrame(
        {
            "model": ["imm", "imm"],
            "delta_vs_best_static": [1.0, "not-a-delta"],
        }
    )

    with pytest.raises(ValueError, match="populated nonnumeric"):
        score_table_sign_flip_summary(frame)


def test_cli_writes_summary_csv(tmp_path) -> None:
    scores = tmp_path / "scores.csv"
    output = tmp_path / "nested" / "sign_flip_summary.csv"
    pd.DataFrame(
        {
            "model": ["imm", "imm", "imm"],
            "delta_vs_best_static": [1.0, 2.0, 3.0],
        }
    ).to_csv(scores, index=False)

    assert main(["--scores", str(scores), "--output", str(output)]) == 0

    written = pd.read_csv(output)
    assert written.loc[0, "model"] == "imm"
    assert written.loc[0, "method"] == "exact"
    assert written.loc[0, "p_value"] == pytest.approx(0.25)

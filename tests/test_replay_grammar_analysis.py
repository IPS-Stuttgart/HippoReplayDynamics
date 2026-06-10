import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1] / "scripts"))

from replay_grammar_analysis import (  # noqa: E402
    GRAMMAR_MODES,
    MODE_DURATION_OUTPUT,
    MODE_SEQUENCE_OUTPUT,
    MOTIF_OUTPUT,
    RAT_SUMMARY_OUTPUT,
    GrammarInferenceConfig,
    parse_mode_mean_duration_bins,
    rat_replay_grammar_summary,
    write_replay_grammar_outputs,
)


def test_replay_grammar_outputs_compositional_motif(tmp_path: Path):
    scores = _segment_scores(
        [
            (0, 2, "stationary", 20.0),
            (2, 4, "momentum", 18.0),
            (4, 6, "fragmented", 16.0),
            (0, 6, "diffusion", 5.0),
        ],
        n_time=6,
    )

    outputs = write_replay_grammar_outputs(
        scores,
        tmp_path,
        config=GrammarInferenceConfig(
            max_segments=3,
            min_segment_bins=1,
            duration_prior_log_sd=10.0,
            mode_mean_duration_bins={"stationary": 2.0, "momentum": 2.0, "fragmented": 2.0},
        ),
    )

    assert set(outputs) == {MODE_SEQUENCE_OUTPUT, MOTIF_OUTPUT, MODE_DURATION_OUTPUT, RAT_SUMMARY_OUTPUT}
    sequence = outputs[MODE_SEQUENCE_OUTPUT]
    assert sequence["mode"].tolist() == ["stationary", "momentum", "fragmented"]
    assert sequence["start_bin"].tolist() == [0, 2, 4]
    assert sequence["end_bin_exclusive"].tolist() == [2, 4, 6]

    motifs = outputs[MOTIF_OUTPUT]
    motif = motifs.iloc[0]
    assert motif["motif"] == "stationary->momentum->fragmented"
    assert motif["motif_family"] == "prelude_trajectory_endpoint"
    assert bool(motif["compositional_replay"])
    assert bool(motif["has_momentum_segment"])
    assert bool(motif["has_stationary_prelude"])
    assert bool(motif["has_fragmented_endpoint"])
    assert motif["trajectory_duration_fraction"] == pytest.approx(1 / 3)

    durations = outputs[MODE_DURATION_OUTPUT]
    assert set(durations["mode"]) == {"fragmented", "momentum", "stationary"}
    assert durations.set_index("mode").loc["momentum", "median_duration_bins"] == pytest.approx(2.0)

    rat = outputs[RAT_SUMMARY_OUTPUT].iloc[0]
    assert rat["rat"] == "Rat1"
    assert int(rat["events"]) == 1
    assert rat["most_common_motif"] == "stationary->momentum->fragmented"

    for filename in outputs:
        path = tmp_path / filename
        assert path.exists()
        assert path.stat().st_size > 0


def test_duration_prior_can_prefer_single_coherent_segment(tmp_path: Path):
    scores = _segment_scores(
        [
            (0, 1, "stationary", 10.0),
            (1, 2, "diffusion", 10.0),
            (0, 2, "stationary", 19.9),
        ],
        n_time=2,
    )

    outputs = write_replay_grammar_outputs(
        scores,
        tmp_path,
        config=GrammarInferenceConfig(
            max_segments=2,
            min_segment_bins=1,
            duration_prior_log_sd=0.1,
            mode_mean_duration_bins={"stationary": 2.0, "diffusion": 2.0},
        ),
    )

    sequence = outputs[MODE_SEQUENCE_OUTPUT]
    assert sequence["mode"].tolist() == ["stationary"]
    assert sequence.iloc[0]["segment_duration_bins"] == 2


def test_parse_mode_duration_overrides():
    parsed = parse_mode_mean_duration_bins("stationary:2, diffusion:5 momentum:7")
    assert parsed == {"stationary": 2.0, "diffusion": 5.0, "momentum": 7.0}
    with pytest.raises(ValueError, match="unknown grammar mode"):
        parse_mode_mean_duration_bins("teleport:3")


def test_rat_replay_grammar_summary_parses_csv_bool_flags():
    motifs = pd.DataFrame(
        [
            {
                "session": "Rat1/Open1",
                "motif": "stationary",
                "compositional_replay": "False",
                "has_trajectory_segment": "0.0",
                "has_momentum_segment": "False",
                "has_stationary_prelude": "False",
                "has_fragmented_endpoint": "False",
                "trajectory_duration_fraction": 0.0,
            },
            {
                "session": "Rat1/Open1",
                "motif": "diffusion->momentum",
                "compositional_replay": "True",
                "has_trajectory_segment": "1.0",
                "has_momentum_segment": 1.0,
                "has_stationary_prelude": "False",
                "has_fragmented_endpoint": "False",
                "trajectory_duration_fraction": 1.0,
            },
        ]
    )

    summary = rat_replay_grammar_summary(motifs).iloc[0]

    assert summary["compositional_events"] == 1
    assert summary["compositional_fraction"] == pytest.approx(0.5)
    assert summary["events_with_trajectory_segment"] == 1
    assert summary["events_with_momentum_segment"] == 1


def _segment_scores(overrides: list[tuple[int, int, str, float]], *, n_time: int) -> pd.DataFrame:
    values = {(start, end, mode): value for start, end, mode, value in overrides}
    rows = []
    for start in range(n_time):
        for end in range(start + 1, n_time + 1):
            for mode in GRAMMAR_MODES:
                rows.append(
                    {
                        "status": "success",
                        "session": "Rat1/Open1",
                        "event_index": 0,
                        "mode": mode,
                        "start_bin": start,
                        "end_bin_exclusive": end,
                        "segment_start_time_s": start * 0.003,
                        "segment_end_time_s": end * 0.003,
                        "segment_duration_s": (end - start) * 0.003,
                        "segment_log_evidence": values.get((start, end, mode), -100.0),
                        "n_time": end - start,
                        "n_spikes": 5,
                        "event_n_time": n_time,
                        "event_n_spikes": 15,
                        "scored_model": f"replay-grammar-{mode}",
                    }
                )
    return pd.DataFrame(rows)

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from hipporeplayimm.encoding import EmissionConfig


def test_duration_patched_emission_builders_are_synchronized() -> None:
    import hipporeplayimm
    import hipporeplayimm.benchmarks as benchmarks
    import hipporeplayimm.encoding as encoding
    import hipporeplayimm.ground_truth as ground_truth

    hipporeplayimm.apply_runtime_patches()
    hipporeplayimm.apply_runtime_patches()

    # Several modules import build_emissions by value before runtime patches are
    # installed. Keep this guard so duration metadata is not silently dropped by
    # stale aliases after import-order refactors.
    assert benchmarks.build_emissions is encoding.build_emissions
    assert ground_truth.build_emissions is encoding.build_emissions


def test_runtime_patches_replay_import_metadata_hooks(monkeypatch) -> None:
    import hipporeplayimm

    calls: list[str] = []

    def recorder(name: str):
        def apply() -> None:
            calls.append(name)

        return apply

    monkeypatch.setattr(
        hipporeplayimm._pyrecest_score_metadata,
        "apply_pyrecest_score_metadata_patch",
        recorder("pyrecest"),
    )
    monkeypatch.setattr(
        hipporeplayimm._goal_state_space_integration,
        "apply_goal_state_space_patch",
        recorder("goal_state_space"),
    )
    monkeypatch.setattr(
        hipporeplayimm._spike_rate_metadata,
        "apply_spike_rate_metadata_patch",
        recorder("spike_rate"),
    )

    hipporeplayimm.apply_runtime_patches()

    assert calls == ["pyrecest", "goal_state_space", "spike_rate"]


def test_spike_rate_metadata_patch_refreshes_stale_true_flag(monkeypatch) -> None:
    import hipporeplayimm
    import hipporeplayimm.benchmarks as benchmarks
    import hipporeplayimm.ground_truth as ground_truth
    import hipporeplayimm.score_metadata as score_metadata

    def stale_emission_config_for_scores(scores_frame: pd.DataFrame, fallback: EmissionConfig) -> EmissionConfig:
        return EmissionConfig(
            time_bin_s=fallback.time_bin_s,
            spike_rate_scale=fallback.spike_rate_scale,
            likelihood_temperature=fallback.likelihood_temperature,
            negative_binomial_overdispersion=fallback.negative_binomial_overdispersion,
        )

    def stale_benchmark_config_metadata(config) -> dict[str, object]:
        return {"emission_time_bin_s": float(config.emissions.time_bin_s)}

    monkeypatch.setattr(score_metadata, "emission_config_for_scores", stale_emission_config_for_scores)
    monkeypatch.setattr(ground_truth, "_emission_config_for_scores", stale_emission_config_for_scores)
    monkeypatch.setattr(benchmarks, "_benchmark_config_metadata", stale_benchmark_config_metadata)
    monkeypatch.setattr(score_metadata, "_spike_rate_metadata_patch_applied", True, raising=False)

    hipporeplayimm.apply_runtime_patches()

    assert getattr(score_metadata.emission_config_for_scores, "_spike_rate_metadata_emission_config_wrapper", False)
    assert ground_truth._emission_config_for_scores is score_metadata.emission_config_for_scores

    scores = pd.DataFrame(
        {
            "emission_time_bin_s": [0.004],
            "emission_spike_rate_scale": [2.5],
            "emission_likelihood_temperature": [0.75],
            "emission_negative_binomial_overdispersion": [0.2],
        }
    )
    config = score_metadata.emission_config_for_scores(scores, EmissionConfig())
    assert config.time_bin_s == 0.004
    assert config.spike_rate_scale == 2.5
    assert config.likelihood_temperature == 0.75
    assert config.negative_binomial_overdispersion == 0.2

    metadata = benchmarks._benchmark_config_metadata(
        SimpleNamespace(
            emissions=EmissionConfig(
                spike_rate_scale=3.0,
                likelihood_temperature=0.5,
                negative_binomial_overdispersion=0.1,
            )
        )
    )
    assert getattr(benchmarks._benchmark_config_metadata, "_spike_rate_metadata_wrapped", False)
    assert metadata["emission_spike_rate_scale"] == 3.0
    assert metadata["emission_likelihood_temperature"] == 0.5
    assert metadata["emission_negative_binomial_overdispersion"] == 0.1

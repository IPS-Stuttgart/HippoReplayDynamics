from __future__ import annotations


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

    monkeypatch.setattr(
        hipporeplayimm._pyrecest_score_metadata,
        "apply_pyrecest_score_metadata_patch",
        lambda: calls.append("pyrecest"),
    )
    monkeypatch.setattr(
        hipporeplayimm._goal_state_space_integration,
        "apply_goal_state_space_patch",
        lambda: calls.append("goal"),
    )
    monkeypatch.setattr(
        hipporeplayimm._spike_rate_metadata,
        "apply_spike_rate_metadata_patch",
        lambda: calls.append("spike_rate"),
    )

    hipporeplayimm.apply_runtime_patches()

    assert calls == ["pyrecest", "goal", "spike_rate"]

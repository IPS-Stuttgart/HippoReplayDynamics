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


def test_apply_runtime_patches_replays_import_time_metadata_hooks(monkeypatch) -> None:
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

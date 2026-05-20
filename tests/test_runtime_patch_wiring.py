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

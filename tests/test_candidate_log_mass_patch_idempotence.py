from __future__ import annotations

import numpy as np


def _wrapper_chain(function) -> list[object]:
    chain: list[object] = []
    seen: set[int] = set()
    current = function
    while callable(current) and id(current) not in seen:
        seen.add(id(current))
        chain.append(current)
        current = getattr(current, "__hipporeplayimm_original__", None)
    return chain


def test_candidate_log_mass_runtime_refresh_does_not_stack_wrappers() -> None:
    import hipporeplayimm
    from hipporeplayimm import candidate_log_mass_validation as array_patch
    from hipporeplayimm import candidate_support_quality_patch as bool_patch
    from hipporeplayimm import result_improvements

    hipporeplayimm.apply_runtime_patches()
    initial_chain = _wrapper_chain(result_improvements._first_finite_numeric_value)

    for _ in range(5):
        hipporeplayimm.apply_runtime_patches()

    current = result_improvements._first_finite_numeric_value
    refreshed_chain = _wrapper_chain(current)
    assert len(refreshed_chain) == len(initial_chain)
    assert any(
        getattr(wrapper, bool_patch._MIN_LOG_MASS_BOOL_WRAPPER_ATTR, False)
        for wrapper in refreshed_chain
    )
    assert any(
        getattr(wrapper, array_patch._REPORTED_MINIMUM_WRAPPER_ATTR, False)
        for wrapper in refreshed_chain
    )

    assert current(np.asarray([-0.005, -1.0])) == -1.0
    assert current(np.asarray([False, -1.0], dtype=object)) is None

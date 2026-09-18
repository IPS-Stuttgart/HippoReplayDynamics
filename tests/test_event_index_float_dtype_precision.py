from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

PARSER_MODULES = (
    "build_replay_dynamics_axis",
    "build_sota_comparator_pack",
    "candidate_support_convergence",
    "clean_imm_time_order_shuffle_control",
    "compare_wrong_map_evidence_controls",
    "paper_state_space_effects",
    "plot_imm_superiority_figures",
    "replay_behavior_alignment",
    "trajectory_imm_mode_superiority",
)


@pytest.mark.parametrize("module_name", PARSER_MODULES)
def test_event_index_parser_rejects_float32_alias_at_exact_integer_boundary(
    module_name: str,
) -> None:
    parser = getattr(importlib.import_module(module_name), "_exact_event_index")
    value = np.float32(2**24 + 1)

    assert value == np.float32(2**24)
    with pytest.raises(ValueError, match=r"at or above 2\*\*24 is unsafe"):
        parser(value)


@pytest.mark.parametrize("module_name", PARSER_MODULES)
def test_event_index_parser_preserves_exact_extended_precision_identifier(
    module_name: str,
) -> None:
    precision_bits = int(np.finfo(np.longdouble).nmant) + 1
    if precision_bits <= 53:
        pytest.skip("platform longdouble has no extra integer precision")

    parser = getattr(importlib.import_module(module_name), "_exact_event_index")
    expected = 2**53 + 1
    value = np.longdouble(str(expected))

    assert parser(value) == expected

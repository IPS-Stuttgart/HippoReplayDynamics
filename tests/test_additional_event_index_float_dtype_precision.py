from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

MODULES = (
    "audit_first_order_imm_mode_usage",
    "deduplicate_off_swr_candidates",
    "model_evidence_support_audit",
    "off_swr_ripple_negativity_validation",
    "triage_momentum_recovery",
)


def _parse_event_index(module_name: str, value: object) -> int:
    module = importlib.import_module(module_name)
    if module_name == "audit_first_order_imm_mode_usage":
        return int(module._exact_integer_identifier(value, "event_index"))
    if module_name == "deduplicate_off_swr_candidates":
        return int(module._integer_identifier(value, "event_index"))
    if module_name == "off_swr_ripple_negativity_validation":
        return int(
            module._exact_integer_identifier(
                value,
                "event_index",
                allow_missing=False,
            )
        )
    if module_name == "triage_momentum_recovery":
        return int(module._exact_integer_id(value, "event_index"))
    if module_name == "model_evidence_support_audit":
        event_values: object = [value]
        if isinstance(value, np.floating):
            # Build from an ndarray so pandas does not coerce longdouble through
            # a Python-float path before the function under test sees it.
            event_values = np.asarray([value], dtype=value.dtype)
        frame = module._exact_grouping_event_keys(
            pd.DataFrame({"event_index": event_values}),
            ["event_index"],
        )
        return int(frame.loc[0, "event_index"])
    raise AssertionError(f"unexpected module: {module_name}")


def _event_frame(value: object) -> pd.DataFrame:
    event_values: object = [value]
    if isinstance(value, np.floating):
        event_values = np.asarray([value], dtype=value.dtype)
    return pd.DataFrame({"event_index": event_values})


def _parse_event_index_from_frame(module_name: str, value: object) -> int:
    module = importlib.import_module(module_name)
    frame = _event_frame(value)
    if module_name == "audit_first_order_imm_mode_usage":
        module._coerce_exact_integer_column(frame, "event_index")
        return int(frame.loc[0, "event_index"])
    if module_name == "deduplicate_off_swr_candidates":
        values = module._integer_identifier_series(frame["event_index"], "event_index")
        return int(values.iloc[0])
    if module_name == "model_evidence_support_audit":
        normalized = module._exact_grouping_event_keys(frame, ["event_index"])
        return int(normalized.loc[0, "event_index"])
    if module_name == "off_swr_ripple_negativity_validation":
        normalized = module._normalize_key_columns(frame)
        return int(normalized.loc[0, "event_index"])
    if module_name == "triage_momentum_recovery":
        normalized = module._normalize_integer_id_columns(frame)
        return int(normalized.loc[0, "event_index"])
    raise AssertionError(f"unexpected module: {module_name}")


@pytest.mark.parametrize("module_name", MODULES)
def test_additional_event_index_parsers_reject_float32_alias(
    module_name: str,
) -> None:
    value = np.float32(2**24 + 1)

    assert value == np.float32(2**24)
    with pytest.raises(ValueError):
        _parse_event_index(module_name, value)


@pytest.mark.parametrize("module_name", MODULES)
def test_additional_event_index_parsers_preserve_extended_precision(
    module_name: str,
) -> None:
    precision_bits = int(np.finfo(np.longdouble).nmant) + 1
    if precision_bits <= 53:
        pytest.skip("platform longdouble has no extra integer precision")

    expected = 2**53 + 1
    value = np.longdouble(str(expected))

    assert _parse_event_index(module_name, value) == expected


@pytest.mark.parametrize("module_name", MODULES)
def test_dataframe_event_index_paths_reject_float32_alias(
    module_name: str,
) -> None:
    value = np.float32(2**24 + 1)

    assert value == np.float32(2**24)
    with pytest.raises(ValueError):
        _parse_event_index_from_frame(module_name, value)


@pytest.mark.parametrize("module_name", MODULES)
def test_dataframe_event_index_paths_preserve_extended_precision(
    module_name: str,
) -> None:
    precision_bits = int(np.finfo(np.longdouble).nmant) + 1
    if precision_bits <= 53:
        pytest.skip("platform longdouble has no extra integer precision")

    expected = 2**53 + 1
    value = np.longdouble(str(expected))

    assert _parse_event_index_from_frame(module_name, value) == expected

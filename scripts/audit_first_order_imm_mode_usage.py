#!/usr/bin/env python3
"""Audit whether first-order IMM wins are driven by nonstationary modes.

The implementation lives in ``_audit_first_order_imm_mode_usage_impl``.  This
front-end keeps the existing public script/import path while hardening integer
scope identifiers before they are used for grouping or joins.
"""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
from pathlib import Path

import numpy as np
import pandas as pd

import _audit_first_order_imm_mode_usage_impl as _impl


_FLOAT_EXACT_INTEGER_LIMIT = 2**53


def _exact_integer_identifier(value: object, column: str) -> int:
    """Return an exact integer identifier without lossy binary64 coercion."""

    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{column} must contain integer identifiers, not booleans")

    if isinstance(value, (int, np.integer)):
        return int(value)

    if isinstance(value, np.floating):
        if not bool(np.isfinite(value)):
            raise ValueError(f"{column} must contain finite integer identifiers")
        if not bool(value.is_integer()):
            raise ValueError(f"{column} must contain integer-valued identifiers")
        precision_bits = int(np.finfo(value.dtype).nmant) + 1
        if abs(value) >= 1 << precision_bits:
            raise ValueError(
                f"{column} contains a floating-point identifier outside the exact integer range; "
                "use integer or string IDs"
            )
        return int(value)

    if isinstance(value, float):
        numeric = float(value)
        if not np.isfinite(numeric):
            raise ValueError(f"{column} must contain finite integer identifiers")
        if not numeric.is_integer():
            raise ValueError(f"{column} must contain integer-valued identifiers")
        if abs(numeric) >= _FLOAT_EXACT_INTEGER_LIMIT:
            raise ValueError(
                f"{column} contains a floating-point identifier outside the exact integer range; "
                "use integer or string IDs"
            )
        return int(numeric)

    text = str(value).strip()
    try:
        numeric = Decimal(text)
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError(f"{column} must contain finite integer identifiers") from exc
    if not numeric.is_finite():
        raise ValueError(f"{column} must contain finite integer identifiers")
    integral = numeric.to_integral_value()
    if numeric != integral:
        raise ValueError(f"{column} must contain integer-valued identifiers")
    return int(integral)


def _coerce_exact_integer_column(frame: pd.DataFrame, column: str) -> None:
    frame[column] = pd.Series(
        [_exact_integer_identifier(value, column) for value in frame[column]],
        index=frame.index,
        dtype=object,
    )


def _read_event_model_evidence(path: Path) -> pd.DataFrame:
    """Read event evidence while preserving exact decimal-form event IDs."""

    frame = pd.read_csv(path, dtype={"event_index": "string"})
    return _read_event_model_evidence_from_frame(frame)


def _read_event_model_evidence_from_frame(frame: pd.DataFrame) -> pd.DataFrame:
    tmp = frame.copy()
    required = {"session", "event_index", "model", "log_evidence"}
    missing = sorted(required.difference(tmp.columns))
    if missing:
        raise ValueError(f"event evidence is missing required columns: {missing}")
    if "status" in tmp:
        tmp = tmp[tmp["status"].astype(str).eq("success")].copy()
    tmp["session"] = tmp["session"].astype(str)
    tmp["rat"] = tmp["session"].map(_impl._rat_from_session)
    _coerce_exact_integer_column(tmp, "event_index")
    tmp["model"] = tmp["model"].astype(str)
    tmp["log_evidence"] = pd.to_numeric(tmp["log_evidence"], errors="coerce")
    tmp = tmp.dropna(subset=["log_evidence"]).copy()
    if "evidence_comparable" not in tmp:
        tmp["evidence_comparable"] = True
    tmp["evidence_comparable"] = tmp["evidence_comparable"].map(_impl._as_bool)
    return tmp


def _selected_one_per_source_evidence(
    promoted_event_model_evidence: pd.DataFrame,
    one_per_source_decisions: pd.DataFrame,
    *,
    selection_rule: str,
) -> pd.DataFrame:
    """Join selected off-SWR rows without rounding event/null identifiers."""

    required = {"session", "event_index", "null_index", "selection_rule"}
    missing = sorted(required.difference(one_per_source_decisions.columns))
    if missing:
        raise ValueError(f"one-per-source decisions are missing required columns: {missing}")

    selected = one_per_source_decisions[
        one_per_source_decisions["selection_rule"].astype(str).eq(selection_rule)
    ].copy()
    if selected.empty:
        raise ValueError(f"no one-per-source decisions found for selection rule {selection_rule!r}")

    evidence = promoted_event_model_evidence.copy()
    for current in (selected, evidence):
        current["session"] = current["session"].astype(str)
        _coerce_exact_integer_column(current, "event_index")
        _coerce_exact_integer_column(current, "null_index")

    decision_columns = [
        column
        for column in (
            "session",
            "event_index",
            "null_index",
            "source_event_group_id",
            "selection_rule",
        )
        if column in selected
    ]
    return evidence.merge(
        selected[decision_columns],
        on=["session", "event_index", "null_index"],
        how="inner",
        suffixes=("", "_selected"),
    )


# The existing implementation resolves these helpers through its module globals
# at call time, so replacing them here fixes both imported and CLI workflows
# without duplicating the scientific/audit logic.
_impl._read_event_model_evidence = _read_event_model_evidence
_impl._read_event_model_evidence_from_frame = _read_event_model_evidence_from_frame
_impl._selected_one_per_source_evidence = _selected_one_per_source_evidence


def main(argv: list[str] | None = None) -> int:
    args = _impl.build_parser().parse_args(argv)
    event_model_evidence = _read_event_model_evidence(Path(args.event_model_evidence))
    promoted = (
        _read_event_model_evidence(Path(args.promoted_off_swr_event_model_evidence))
        if args.promoted_off_swr_event_model_evidence
        else None
    )
    one_per = (
        pd.read_csv(
            args.one_per_source_decisions,
            dtype={"event_index": "string", "null_index": "string"},
        )
        if args.one_per_source_decisions
        else None
    )
    if promoted is None:
        outputs = _impl.write_first_order_imm_mode_usage_audit(
            event_model_evidence,
            Path(args.output),
            margin_threshold=args.margin_threshold,
        )
    else:
        outputs = _impl.write_first_order_imm_mode_usage_comparison_audit(
            event_model_evidence,
            Path(args.output),
            promoted_off_swr_event_model_evidence=promoted,
            one_per_source_decisions=one_per,
            one_per_source_selection_rule=args.one_per_source_selection_rule,
            margin_threshold=args.margin_threshold,
        )
    print("First-order IMM mode usage summary:")
    print(outputs["first_order_imm_mode_usage_summary.csv"].to_string(index=False))
    print("\nFirst-order IMM mode usage gates:")
    print(outputs["first_order_imm_mode_usage_gate_summary.csv"].to_string(index=False))
    if "swr_off_swr_first_order_imm_mode_usage_comparison.csv" in outputs:
        print("\nSWR/off-SWR first-order IMM mode usage comparison:")
        print(outputs["swr_off_swr_first_order_imm_mode_usage_comparison.csv"].to_string(index=False))
    return 0


def __getattr__(name: str):
    """Delegate the unchanged public API to the implementation module."""

    return getattr(_impl, name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(dir(_impl)))


__all__ = [name for name in dir(_impl) if not name.startswith("_")]


if __name__ == "__main__":
    raise SystemExit(main())

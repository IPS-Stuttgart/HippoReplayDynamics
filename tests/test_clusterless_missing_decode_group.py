from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

import hipporeplayimm.clusterless_ground_truth as clusterless_gt


class _FakeClusterlessEncoding:
    bin_centers = np.asarray([[0.0, 0.0]], dtype=float)
    occupancy_s = np.asarray([1.0], dtype=float)


class _FakeGT(SimpleNamespace):
    pass


def test_clusterless_ground_truth_keeps_missing_decode_group_keys() -> None:
    scores = pd.DataFrame(
        {
            "session": ["s1"],
            "event_index": [0],
            "model": ["clusterless-state-space-stationary"],
            "benchmark_cell_split_index": [np.nan],
            "heldout_log_likelihood": [0.0],
            "train_log_likelihood": [0.0],
            "joint_log_likelihood": [0.0],
        }
    )
    ground_truth = pd.DataFrame(
        {
            "session": ["s1"],
            "event_index": [0],
            "valid_label": [False],
        }
    )

    def model_names(frame: pd.DataFrame) -> tuple[str, ...]:
        return tuple(dict.fromkeys(frame["model"].astype(str)))

    def group_key_values(columns: list[str], group_key: object) -> dict[str, object]:
        values = group_key if isinstance(group_key, tuple) else (group_key,)
        return dict(zip(columns, values))

    def attach_decode_group_values(row: dict[str, object], values: dict[str, object]) -> dict[str, object]:
        out = dict(row)
        for column, value in values.items():
            if column not in {"session", "event_index", "model"}:
                out[column] = value
        return out

    def decoded_merge_columns(
        scores_frame: pd.DataFrame,
        decoded: pd.DataFrame,
        benchmark_decode: bool,
    ) -> list[str]:
        columns = ["session", "event_index", "model"]
        if (
            benchmark_decode
            and "benchmark_cell_split_index" in scores_frame.columns
            and "benchmark_cell_split_index" in decoded.columns
        ):
            columns.append("benchmark_cell_split_index")
        return columns

    fake_gt = _FakeGT(
        _model_names_for_scores=model_names,
        _load_or_generate_ground_truth=lambda *_args, **_kwargs: ground_truth,
        load_open_field_sessions=lambda _root: [SimpleNamespace(session_id="s1")],
        _score_table_is_heldout_benchmark=lambda frame: {
            "heldout_log_likelihood",
            "train_log_likelihood",
            "joint_log_likelihood",
        }.issubset(frame.columns),
        _decode_group_columns=lambda _frame, benchmark_decode: [
            "session",
            "benchmark_cell_split_index",
        ]
        if benchmark_decode
        else ["session"],
        _group_key_values=group_key_values,
        _build_models=lambda config, session=None: {
            "clusterless-state-space-stationary": object()
        },
        infer_well_locations=lambda *_args, **_kwargs: pd.DataFrame(
            columns=["well_id", "well_x", "well_y"]
        ),
        fit_place_field_encoding=lambda *_args, **_kwargs: SimpleNamespace(),
        _cell_split_for_score_rows=lambda *_args, **_kwargs: (
            np.asarray([1], dtype=int),
            np.asarray([2], dtype=int),
        ),
        _session_with_mark_cell_subset=lambda session, *_args, **_kwargs: session,
        fit_clusterless_mark_encoding=lambda *_args, **_kwargs: _FakeClusterlessEncoding(),
        build_clusterless_mark_emissions=lambda *_args, **_kwargs: SimpleNamespace(
            n_time=1
        ),
        _requested_model_name=lambda _row, fallback: fallback,
        _score_joint_for_ground_truth=lambda *_args, **_kwargs: SimpleNamespace(
            terminal_log_posterior=np.log(np.asarray([1.0])),
            trajectory_log_posterior=None,
        ),
        _attach_decode_group_values=attach_decode_group_values,
        _decoded_row=lambda session, event_index, model, *_args, **_kwargs: {
            "session": session,
            "event_index": event_index,
            "model": model,
            "decoded": True,
        },
        _decoded_merge_columns=decoded_merge_columns,
        _add_ground_truth_metrics=lambda comparison, *_args, **_kwargs: comparison,
    )

    comparison = clusterless_gt._compare_clusterless_scores_to_ground_truth(
        fake_gt,
        "unused-root",
        scores,
    )

    assert comparison["decoded"].tolist() == [True]
    assert comparison["benchmark_cell_split_index"].isna().all()

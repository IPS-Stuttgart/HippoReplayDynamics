from __future__ import annotations

import numpy as np
import pandas as pd

from hipporeplayimm.result_improvement_extensions import add_model_averaged_endpoint_columns


def _row(
    *,
    variant: str,
    window_index: int,
    model: str,
    probability: float,
    endpoint_x: float,
    endpoint_y: float = 0.0,
    log_evidence: float = 0.0,
) -> dict[str, object]:
    return {
        "session": "Rat1/Open1",
        "event_index": 7,
        "window_index": int(window_index),
        "event_window_variant": str(variant),
        "window_start_s": 10.0 + float(window_index),
        "window_end_s": 10.1 + float(window_index),
        "window_duration_s": 0.1,
        "model": str(model),
        "log_evidence": float(log_evidence),
        "model_probability": float(probability),
        "evidence_comparable": True,
        "diagnostic_decoded_endpoint_x": float(endpoint_x),
        "diagnostic_decoded_endpoint_y": float(endpoint_y),
    }


def test_model_averaged_endpoint_scopes_replay_window_variants() -> None:
    frame = pd.DataFrame(
        [
            _row(variant="core", window_index=0, model="a", probability=0.25, endpoint_x=10.0, log_evidence=0.0),
            _row(variant="core", window_index=0, model="b", probability=0.75, endpoint_x=20.0, log_evidence=1.0),
            _row(variant="expanded", window_index=1, model="a", probability=0.50, endpoint_x=100.0, log_evidence=0.0),
            _row(variant="expanded", window_index=1, model="b", probability=0.50, endpoint_x=200.0, log_evidence=0.0),
        ]
    )

    out = add_model_averaged_endpoint_columns(frame)

    core = out[out["event_window_variant"].eq("core")]
    expanded = out[out["event_window_variant"].eq("expanded")]
    np.testing.assert_allclose(core["model_averaged_endpoint_x"], 17.5)
    np.testing.assert_allclose(expanded["model_averaged_endpoint_x"], 150.0)
    assert core["model_averaged_endpoint_models"].tolist() == [2, 2]
    assert expanded["model_averaged_endpoint_models"].tolist() == [2, 2]


def test_model_averaged_endpoint_scopes_stochastic_random_seeds() -> None:
    rows = [
        _row(variant="core", window_index=0, model="a", probability=0.75, endpoint_x=10.0, log_evidence=2.0),
        _row(variant="core", window_index=0, model="b", probability=0.25, endpoint_x=20.0, log_evidence=0.0),
        _row(variant="core", window_index=0, model="a", probability=0.25, endpoint_x=100.0, log_evidence=1.0),
        _row(variant="core", window_index=0, model="b", probability=0.75, endpoint_x=200.0, log_evidence=3.0),
    ]
    for row, seed in zip(rows, (1, 1, 2, 2), strict=True):
        row["random_seed"] = seed
    frame = pd.DataFrame(rows)

    out = add_model_averaged_endpoint_columns(frame)

    seed_one = out[out["random_seed"].eq(1)]
    seed_two = out[out["random_seed"].eq(2)]
    np.testing.assert_allclose(seed_one["model_averaged_endpoint_x"], 12.5)
    np.testing.assert_allclose(seed_two["model_averaged_endpoint_x"], 175.0)
    assert seed_one["model_averaged_endpoint_models"].tolist() == [2, 2]
    assert seed_two["model_averaged_endpoint_models"].tolist() == [2, 2]


def test_model_averaged_endpoint_ignores_nonfinite_endpoint_rows() -> None:
    frame = pd.DataFrame(
        [
            _row(variant="core", window_index=0, model="finite", probability=0.50, endpoint_x=10.0, endpoint_y=5.0),
            _row(variant="core", window_index=0, model="bad-x", probability=0.25, endpoint_x=np.inf, endpoint_y=100.0),
            _row(variant="core", window_index=0, model="bad-y", probability=0.25, endpoint_x=100.0, endpoint_y=-np.inf),
        ]
    )

    out = add_model_averaged_endpoint_columns(frame)

    np.testing.assert_allclose(out["model_averaged_endpoint_x"], 10.0)
    np.testing.assert_allclose(out["model_averaged_endpoint_y"], 5.0)
    assert out["model_averaged_endpoint_models"].tolist() == [1, 1, 1]


def _cell_split_row(
    *,
    train_cell_ids: object,
    test_cell_ids: object,
    model: str,
    probability: float,
    endpoint_x: float,
) -> dict[str, object]:
    row = _row(
        variant="core",
        window_index=0,
        model=model,
        probability=probability,
        endpoint_x=endpoint_x,
    )
    row["train_cell_ids"] = train_cell_ids
    row["test_cell_ids"] = test_cell_ids
    return row


def _cell_tuple(value: object) -> tuple[int, ...]:
    return tuple(int(cell_id) for cell_id in np.asarray(value, dtype=int).reshape(-1))


def test_model_averaged_endpoint_scopes_explicit_cell_split_metadata() -> None:
    frame = pd.DataFrame(
        [
            _cell_split_row(
                train_cell_ids=np.array([1, 2], dtype=int),
                test_cell_ids=np.array([3], dtype=int),
                model="a",
                probability=0.50,
                endpoint_x=10.0,
            ),
            _cell_split_row(
                train_cell_ids=np.array([1, 2], dtype=int),
                test_cell_ids=np.array([3], dtype=int),
                model="b",
                probability=0.50,
                endpoint_x=20.0,
            ),
            _cell_split_row(
                train_cell_ids=[1, 3],
                test_cell_ids=[2],
                model="a",
                probability=0.50,
                endpoint_x=100.0,
            ),
            _cell_split_row(
                train_cell_ids=[1, 3],
                test_cell_ids=[2],
                model="b",
                probability=0.50,
                endpoint_x=200.0,
            ),
        ]
    )

    out = add_model_averaged_endpoint_columns(frame)

    split_a = out[out["train_cell_ids"].map(lambda value: _cell_tuple(value) == (1, 2))]
    split_b = out[out["train_cell_ids"].map(lambda value: _cell_tuple(value) == (1, 3))]
    np.testing.assert_allclose(split_a["model_averaged_endpoint_x"], 15.0)
    np.testing.assert_allclose(split_b["model_averaged_endpoint_x"], 150.0)
    assert split_a["model_averaged_endpoint_models"].tolist() == [2, 2]
    assert split_b["model_averaged_endpoint_models"].tolist() == [2, 2]

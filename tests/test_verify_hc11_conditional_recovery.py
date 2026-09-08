import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from scripts import simulate_hc11_conditional_prediction_recovery as producer
from scripts import verify_hc11_conditional_recovery as verifier


@pytest.mark.parametrize("generator", producer.GENERATORS)
@pytest.mark.parametrize("topology", ["linear", "circular"])
def test_independent_generation(generator, topology):
    rates = np.random.default_rng(41).uniform(0.1, 20, (5, 6))
    data = {
        "centers": np.arange(6) * 4.0 + 2.0,
        "topology": np.array(topology),
        "track_length": np.array(24.0),
        "counts_POST_14": np.array([[0, 0, 0, 0, 0], [1, 2, 1, 0, 1], [1, 0, 3, 1, 1], [0, 1, 0, 1, 0]]),
        "edges_POST_14": np.array([12.0, 12.02, 12.04, 12.06, 12.067]),
        "rates_direction_mixture_0": rates,
        "rates_direction_mixture_1": rates[:, ::-1].copy(),
    }
    kernels = producer.frozen.transitions(data["centers"], data["edges_POST_14"], topology, 24.0)
    for replicate in (0, 24, 49):
        a = producer.generated_event(data, "POST_14", generator, replicate, "session", kernels)
        b = verifier.independent_draw(data, "session", "POST", 14, generator, replicate)
        for left, right in zip(a, b[:5], strict=True):
            assert_allclose(left, right, atol=0)
        verifier.check_path(a[1], a[2], data, "POST_14", generator)


def test_independent_bootstrap_weights():
    cohort = pd.DataFrame({"rat": ["a", "a", "b", "b", "b", "b"], "session": ["a1", "a1", "b1", "b1", "b2", "b2"]})
    assert_allclose(producer.bootstrap_weights(cohort, draws=100, seed=22), verifier.independent_weights(cohort, draws=100, seed=22), atol=1e-14)


def test_corrupted_static_path_fails():
    data = {"counts_POST_1": np.ones((3, 4)), "centers": np.arange(4)}
    with pytest.raises(ValueError):
        verifier.check_path(np.array([0, 1, 2]), np.full(3, -1), data, "POST_1", "static_location")


def test_reconstructed_paired_medians():
    rows = []
    for split, gain in enumerate([0, 1, 2, 3, 100]):
        rows.append(
            {
                "session": "s",
                "rat": "r",
                "phase": "POST",
                "event_index": 1,
                "generator": "diffusion",
                "replicate": 0,
                "encoding_variant": "pooled",
                "split": split,
                "score_iid_position": -20.0,
                "score_static_location": -21.0,
                "score_diffusion": -20.0 + gain,
                "score_first_order_imm": -20.0 + 2 * gain,
                "score_oracle": -1.0,
            }
        )
    scores = pd.DataFrame(rows)
    a = producer.event_contrasts(scores.copy())
    b = verifier.reconstruct_events(scores)
    for col in producer.CONTRASTS:
        assert_allclose(a[col], b[col], atol=0)
    assert a.raw_predictive_winner.tolist() == b.raw_predictive_winner.tolist()


def scope_fixture():
    frame = pd.MultiIndex.from_product(
        [producer.MODELS, range(50), ("direction_mixture", "pooled"), range(5)],
        names=["generator", "replicate", "encoding_variant", "split"],
    ).to_frame(index=False)
    frame = frame.assign(session="s", phase="POST", event_index=1, status="success")
    for model in (*producer.MODELS, "oracle"):
        frame[f"score_{model}"] = -1.0
    return frame


def test_complete_score_scope():
    verifier.verify_score_scope(scope_fixture())


@pytest.mark.parametrize("corruption", ["empty", "duplicate", "missing", "factor", "nan", "status", "positive_score"])
def test_invalid_score_scope(corruption):
    scores = scope_fixture()
    if corruption == "empty":
        scores = scores.iloc[:0]
    elif corruption == "duplicate":
        scores = pd.concat([scores, scores.iloc[:1]])
    elif corruption == "missing":
        scores = scores.iloc[1:]
    elif corruption == "factor":
        scores.loc[0, "encoding_variant"] = "unknown"
    elif corruption == "nan":
        scores.loc[0, "score_first_order_imm"] = np.nan
    elif corruption == "positive_score":
        scores.loc[0, "score_first_order_imm"] = 1.0
    else:
        scores.loc[0, "status"] = "failure"
    with pytest.raises(ValueError):
        verifier.verify_score_scope(scores)

import numpy as np
import pandas as pd
import pytest
from scripts.plot_replay_coverage_matched_events import choose_examples, pair_decisions, posterior_diagnostics


def fixture():
    rows = []
    for event, (a, b) in enumerate([(True, False), (True, True), (False, True), (False, False)]):
        for fraction, decision in [(1, a), (.5, b)]:
            rows.append(dict(dataset="test", animal="r1", session="s1", window_uid=f"e{event}",
                observation="original_order", detector="source_high_mua", bin_filter="edge_only", min_frames=10,
                population_replicate=0, cell_fraction=fraction, geometric_pass=decision, cell_identity_p=.01,
                xy_roll_p=.01, retained_cells=4 * fraction, **{"accepted_alpha_0.02": decision}))
    return pd.DataFrame(rows)


def test_matched_categories_and_deterministic_examples():
    data = fixture()
    paired = pair_decisions(data)
    assert paired.category.tolist() == ["lost", "retained", "gained", "rejected"]
    a, available = choose_examples(paired)
    b, _ = choose_examples(pair_decisions(data.sample(frac=1, random_state=5)))
    pd.testing.assert_frame_equal(a, b)
    assert available.eligible_examples.sum() == 4
    _, missing = choose_examples(paired[paired.category.eq("lost")])
    assert missing.status.eq("unavailable").sum() == 3
    with pytest.raises(ValueError, match="paired"):
        pair_decisions(data.iloc[1:])


def test_flat_prior_posterior_mean_radius_and_marginals():
    grid = np.array([[0., 0.], [8., 0.], [0., 8.], [8., 8.]])
    p, mean, radius, marginal = posterior_diagnostics(np.array([[0, 0], [2, 3]]), np.ones((2, 4)), grid)
    np.testing.assert_allclose(p, .25)
    np.testing.assert_allclose(mean, 4)
    np.testing.assert_allclose(radius, np.sqrt(32))
    for _, m in marginal:
        np.testing.assert_allclose(m, .5)
    p, _, _, _ = posterior_diagnostics(np.array([[10]]), np.array([[1., 100., 1., 1.]]), grid)
    assert p.argmax() == 1

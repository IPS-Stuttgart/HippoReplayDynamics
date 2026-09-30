import numpy as np
import pandas as pd

from hipporeplayimm.regional_content_bounds import (
    calibration_intervals,
    identification_set,
    observed_intervals,
)
from scripts.audit_regional_content_bounds import calibration, independent_lp
from scripts.report_regional_content_bounds import aggregate


def test_independent_calibration_and_fractional_program():
    rng = np.random.default_rng(198)
    z = np.repeat([0, 1], 1000)
    y = np.where(z[:, None] == 1, 2, 0)*np.ones((1, 4), int)
    y[rng.random(y.shape) < .03] = 1
    target = y[rng.choice(len(y), 400)]
    ci, _, _ = calibration_intervals(y, z)
    np.testing.assert_allclose(ci, calibration(y, z))
    for slack in (0., .1, 1.):
        model = identification_set(observed_intervals(target), ci, slack)
        result = model.prevalence()
        np.testing.assert_allclose([result["lower"], result["upper"]], independent_lp(target, ci, slack), atol=1e-7)
        result = model.conditional(80)
        np.testing.assert_allclose([result["lower"], result["upper"]], independent_lp(target, ci, slack, 80), atol=1e-7)


def test_aggregation_does_not_weight_animals_by_session_count():
    rows, score_rows, conditional = [], [], []
    for animal, sessions, width in (("a", 3, .2), ("b", 1, .8)):
        for session in range(sessions):
            meta = {"animal": animal, "session": str(session), "source": "real", "calibration": "known", "transfer_slack": 0.}
            for rep in range(2):
                rows.append(dict(**meta, width=width, useful=width <= .2, status="feasible", covers_truth=np.nan))
                score_rows.append({"animal": animal, "session": str(session), "source": "real", "latent_prevalence": width})
            conditional.append(dict(**meta, lower=0., upper=width, width=width, events=20))
    summary, _, score, _, event = aggregate(pd.DataFrame(rows), pd.DataFrame(score_rows), pd.DataFrame(conditional))
    assert summary.width.iloc[0] == .5
    assert score.latent_prevalence.iloc[0] == .5
    assert event.weighted_width.iloc[0] == .5


def test_target_truth_never_enters_bound_estimation():
    cal = np.array([[0]*4]*200 + [[2]*4]*200)
    z = np.r_[np.zeros(200), np.ones(200)]
    ci, _, _ = calibration_intervals(cal, z)
    target = cal[:300]
    # The API consumes target readouts, not target labels; changing evaluation
    # truth cannot change its interval or make shared false agreement correct.
    first = identification_set(observed_intervals(target), ci).prevalence()
    second = identification_set(observed_intervals(target.copy()), ci.copy()).prevalence()
    assert first == second

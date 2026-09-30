import numpy as np
import pandas as pd

from hipporeplayimm.regional_content_frontier import regional_log_bf
from scripts.audit_regional_content_frontier import independent_bf
from scripts.report_regional_content_frontier import summarize_animals


def test_summary_weights_animals_not_number_of_sessions():
    frame = pd.DataFrame({"setting": ["x"]*4, "animal": ["a", "a", "a", "b"],
                          "auc": [.9, .9, .9, .5]})
    _, summary = summarize_animals(frame, ["setting"], ["auc"])
    assert summary.iloc[0].auc == .7


def test_independent_likelihood_reconstruction():
    rng = np.random.default_rng(3)
    rates = rng.uniform(.01, 10, (9, 13))
    region = np.arange(13) < 3
    counts = rng.poisson(.5, (29, 9))
    for conditional in (False, True):
        expected = regional_log_bf(counts, rates, region, .04, conditional)
        np.testing.assert_allclose(independent_bf(counts, rates, region, .04, conditional), expected)

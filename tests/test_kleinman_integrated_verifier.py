import numpy as np
import pandas as pd

from scripts.calibrate_kleinman_integrated_extent import fit_events, template_library
from scripts.verify_kleinman_integrated_extent import reference_fit, reference_library


def test_scalar_reference_library_fit_and_predictive_score():
    x = np.arange(1, 100, 2)
    fields = 0.03 + 20 * np.exp(-(((x[:, None] - np.linspace(1, 99, 10)) / 8) ** 2))
    model = {"rates": np.block([[fields, fields * 0.1], [fields * 0.1, fields]]), "edges": np.arange(0, 102, 2), "ends": np.array([10.0, 90.0]), "support": np.ones(100, bool)}
    model["support"][[2, 15, 97]] = False
    q, meta = template_library(model, 0.2)
    ref_q, ref_meta = reference_library(model, 0.2)
    pd.testing.assert_frame_equal(meta, ref_meta)
    np.testing.assert_allclose(q, ref_q)
    counts = np.random.default_rng(77).poisson(0.1, size=(4, 20, 20))
    fits = fit_events(counts, q, meta)
    for i, observed in enumerate(counts):
        reference = reference_fit(observed, ref_q, ref_meta)
        for name, value in reference.items():
            np.testing.assert_allclose(fits.iloc[i][name], value, atol=1e-10)

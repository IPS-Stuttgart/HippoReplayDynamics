from types import SimpleNamespace

import numpy as np

from scripts.calibrate_kleinman_replay_content import ARMS, early_late, measure, observations, posterior
from scripts.verify_kleinman_replay_content import reference


def test_independent_reference_reproduces_three_arms_both_sides_and_timing_profiles():
    locations = np.arange(1, 100, 2)
    fields = 0.02 + 40 * np.exp(-(((locations[:, None] - np.linspace(5, 95, 12)) / 9) ** 2))
    model = {"rates": np.block([[fields, 0.1 * fields], [0.1 * fields, fields]]), "edges": np.arange(0, 102, 2), "ends": np.array([10.0, 90.0]), "support": np.ones(100, bool)}
    model["support"][[2, 4, 90]] = False
    for side in [0, 1]:
        for profile in ["linear", "cosine", "pause_step"]:
            args = {"duration": 0.2, "extent": 0.5, "side": side, "profile": profile, "expected_spikes": 48, "event_index": 17}
            data = observations(model, **args)
            for arm in ARMS:
                row = SimpleNamespace(**{**args, "duration_s": args["duration"], "arm": arm})
                ref = reference(row, model)
                p = posterior(data["counts"], model["rates"], model["support"], data["gain"], arm)
                metrics = measure(p, model, data["centers"], side)
                for name in ["displacement_fraction", "incoming_direction_mass", "weighted_correlation", "reverse_content_call"]:
                    np.testing.assert_allclose(metrics[name], ref[name], atol=1e-10)
                truth = (1 - 2 * side) * early_late(data["truth_windows"], data["centers"]) / 80
                np.testing.assert_allclose(truth, ref["true_displacement_fraction"])

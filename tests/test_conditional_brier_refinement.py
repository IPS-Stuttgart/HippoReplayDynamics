from fractions import Fraction as F
import json

import pytest

from scripts import conditional_brier_refinement as m


def test_full_support_informative_channel_can_fail_conditional_brier_guard():
    prior = F(3, 100)
    coarse = m.risks(prior, [prior], [F(1)])
    values, weights = m.channel(prior, F(1, 1000), F(9, 10))
    refined = m.risks(prior, values, weights)
    joints = m.independent_joint_check(prior, values, weights, refined)
    assert all(v["joint_mass"] > 0 for v in joints)
    assert refined["overall_brier"] < coarse["overall_brier"]
    assert refined["positive_class_brier"] < coarse["positive_class_brier"]
    assert refined["negative_class_brier"] > coarse["negative_class_brier"]
    for key in ("positive_class_absolute_error", "negative_class_absolute_error"):
        assert refined[key] < coarse[key]
    assert coarse["overall_brier"] - refined["overall_brier"] == refined["posterior_variance"]


def test_symbolic_two_point_boundary_and_signs():
    for prior in (F(1, 100), F(3, 100), F(1, 5)):
        before = m.risks(prior, [prior], [F(1)])
        for high in (F(1, 4), F(1, 2), 1 - prior, F(199, 200), F(1)):
            values, weights = m.channel(prior, F(0), high)
            after = m.risks(prior, values, weights)
            change = after["negative_class_brier"] - before["negative_class_brier"]
            assert change == prior / (1 - prior) * (high - prior) * (1 - prior - high)
            assert (change > 0) == (prior < high < 1 - prior)


def test_every_case_independently_reconstructed():
    for _, prior, values, weights in m.cases():
        m.independent_joint_check(prior, values, weights, m.risks(prior, values, weights))


@pytest.mark.parametrize("values,weights", [([F(1, 2)], [F(1)]), ([F(3, 100)], [F(1, 2)]), ([F(3, 100)], [F(-1)]), ([F(-1)], [F(1)])])
def test_invalid_refinements_are_rejected(values, weights):
    with pytest.raises(ValueError):
        m.risks(F(3, 100), values, weights)


def test_corrupted_risk_rejected_by_joint_enumeration():
    prior = F(3, 100)
    values, weights = m.channel(prior, F(0), F(9, 10))
    measured = m.risks(prior, values, weights)
    measured["negative_class_brier"] += F(1, 100000)
    with pytest.raises(AssertionError):
        m.independent_joint_check(prior, values, weights, measured)


def test_cli_artifacts_do_not_claim_remedy(tmp_path):
    output = tmp_path / "control"
    m.run(output)
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["cases"] == 5 and manifest["independent_joint_enumeration"]
    assert not manifest["validated_remedy"] and not manifest["external_validation"] and not manifest["neural_data_used"]
    for name, expected in manifest["output_sha256"].items():
        assert m.file_sha256(output / name) == expected

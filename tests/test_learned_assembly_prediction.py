"""Known-assembly recovery, exact kernels and target-cell separation."""

import itertools
import json
from dataclasses import asdict, replace

import numpy as np
import pandas as pd
import pytest
from scipy.special import logsumexp

pytest.importorskip("hmmlearn")
from hipporeplayimm.assembly_predictive_control import multinomial_ll
from hipporeplayimm.learned_assembly_prediction import (
    fit_learned_assembly,
    infer_states,
    predict_learned_assembly,
    validate_sequences,
)
from scripts.audit_2d_learned_assembly import contrasts, file_sha256, load_fit, reuse_completed
from scripts.report_2d_learned_assembly import ENDPOINTS, decisions
from scripts.verify_2d_learned_assembly import forward_backward, reference_prediction


def simulated(seed, events=120):
    rng = np.random.default_rng(seed)
    p = np.full((12, 3), 0.015)
    for state in range(3):
        p[state * 4:(state + 1) * 4, state] = 0.24
    p /= p.sum(axis=0)
    a = np.array([[0.65, 0.33, 0.02], [0.02, 0.65, 0.33], [0.33, 0.02, 0.65]])
    sequences = []
    for _ in range(events):
        state = rng.integers(3)
        rows = []
        for _ in range(18):
            rows.append(rng.multinomial(rng.integers(4, 13), p[:, state]))
            state = rng.choice(3, p=a[state])
        sequences.append(np.array(rows))
    return sequences


def test_forward_backward_matches_exhaustive_paths():
    ll = np.array([[-900, -903], [-500, -499], [-800, -796]], float)
    pi, a = np.array([0.7, 0.3]), np.array([[0.8, 0.2], [0.4, 0.6]])
    paths = list(itertools.product(range(2), repeat=3))
    weights = np.array([np.log(pi[z[0]]) + sum(ll[t, z[t]] for t in range(3)) + sum(np.log(a[z[t-1], z[t]]) for t in range(1, 3)) for z in paths])
    z, posterior = infer_states(ll, pi, a)
    assert z == pytest.approx(logsumexp(weights))
    for t in range(3):
        for state in range(2):
            value = logsumexp(weights[[path[t] == state for path in paths]]) - z
            assert posterior[t, state] == pytest.approx(value)


def test_known_assembly_recovers_prediction_and_learned_transitions():
    fit = fit_learned_assembly(simulated(9), 3, 54)
    assert fit.converged
    assert np.diff(fit.objective_trace).min() >= -1e-7
    assert np.allclose(fit.transition.sum(1), 1)
    assert not np.allclose(fit.transition, 0.5 * np.eye(3) + 0.5 / 3)
    test = simulated(83, 60)
    tr, he = np.arange(0, 12, 2), np.arange(1, 12, 2)
    scores = [predict_learned_assembly(x, tr, he, fit)[0] for x in test]
    assert np.mean([s['learned_hmm'] - s['nonspatial_global'] for s in scores]) > 10
    assert np.mean([s['learned_hmm'] - s['same_emissions_iid'] for s in scores]) > 0


def test_target_heldout_spikes_cannot_change_inferred_posterior():
    fit = fit_learned_assembly(simulated(3, 40), 3, 19)
    x = simulated(12, 1)[0]
    tr, he = np.arange(0, 12, 2), np.arange(1, 12, 2)
    first, hashes = predict_learned_assembly(x, tr, he, fit)
    changed = x.copy()
    changed[:, he] = changed[::-1][:, he]
    second, changed_hashes = predict_learned_assembly(changed, tr, he, fit)
    assert hashes == changed_hashes
    assert first['learned_hmm'] != second['learned_hmm']
    with pytest.raises(ValueError, match='partition'):
        predict_learned_assembly(x, tr, tr, fit)


def test_state_labels_are_invariant():
    fit = fit_learned_assembly(simulated(41, 30), 3, 92)
    order = np.array([2, 0, 1])
    shuffled = replace(fit, probabilities=fit.probabilities[:, order], initial=fit.initial[order], transition=fit.transition[order][:, order], occupancy=fit.occupancy[order])
    x = simulated(93, 1)[0]
    first = predict_learned_assembly(x, np.arange(6), np.arange(6, 12), fit)[0]
    second = predict_learned_assembly(x, np.arange(6), np.arange(6, 12), shuffled)[0]
    assert first == pytest.approx(second)


def test_zero_counts_and_independent_transition_limit():
    counts = np.zeros((3, 4), dtype=int)
    ll = multinomial_ll(counts, np.ones((4, 2)))
    pi = np.array([0.4, 0.6])
    z, p = infer_states(ll, pi, np.tile(pi, (2, 1)))
    assert z == pytest.approx(0)
    assert np.allclose(np.exp(p), np.tile(pi, (3, 1)))


def test_invalid_calibration_rejected():
    for seq in ([], [np.zeros((0, 3))], [np.array([[1.1, 2]])], [np.ones((2, 3)), np.ones((2, 4))]):
        with pytest.raises(ValueError):
            validate_sequences(seq)
    with pytest.raises(ValueError):
        fit_learned_assembly([np.zeros((20, 4), int)], 3, 1)
    with pytest.raises(ValueError):
        infer_states(np.ones((2, 2)), [0.5, 0.5], [[0.5, 0.2], [0.5, 0.5]])


def paired_tables():
    identity = {"dataset": 'synthetic', "animal": 'rat', "session": 'day', "event_id": 1, "split": 0}
    rows = [identity | {"n_states": 50, "shuffle": k, "n_heldout_spikes": 10,
                           "score_learned_hmm": -10 if k == -1 else -15,
                           "score_same_emissions_iid": -20, "score_nonspatial_global": -25} for k in range(-1, 20)]
    spatial = pd.DataFrame([identity | {"alpha": 100, "map": 'real', "score_first_order_imm": -7,
                                         "score_iid_position": -12, "score_event_global": -22}])
    order = pd.DataFrame([identity | {"contrast": 'alpha100__first_order_imm__real_order_advantage', "delta": 8}])
    return pd.DataFrame(rows), spatial, order


def test_paired_comparison_uses_signed_differences():
    split, events = contrasts(*paired_tables())
    result = events.set_index('contrast').delta
    assert result['k50__spatial_imm_minus_learned_hmm'] == 3
    assert result['k50__learned_hmm_order_advantage'] == 5
    assert result['k50__spatial_minus_assembly_order_advantage'] == 3
    assert len(split) == len(events) == 7


def test_missing_shuffles_or_spatial_pair_fail():
    scores, spatial, order = paired_tables()
    with pytest.raises(ValueError, match='incomplete'):
        contrasts(scores.iloc[:-1], spatial, order)
    with pytest.raises(ValueError, match='missing'):
        contrasts(scores, spatial.assign(event_id=2), order)
    with pytest.raises(ValueError, match='duplicate'):
        contrasts(pd.concat([scores, scores.iloc[:1]]), spatial, order)


def test_independent_predictor_and_fit_archive(tmp_path):
    fit = fit_learned_assembly(simulated(3, 30), 3, 43)
    path = tmp_path / 'fit.npz'
    np.savez_compressed(path, **asdict(fit))
    restored = load_fit(path)
    x = simulated(61, 1)[0]
    tr, he = np.arange(0, 12, 2), np.arange(1, 12, 2)
    actual = predict_learned_assembly(x, tr, he, restored)[0]
    expected = reference_prediction(x, tr, he, restored)
    assert actual == pytest.approx(expected)
    ll = multinomial_ll(x[:, tr], restored.probabilities[tr])
    a, q = infer_states(ll, restored.initial, restored.transition)
    b, r = forward_backward(ll, restored.initial, restored.transition)
    assert a == pytest.approx(b)
    assert np.allclose(q, r)
    assert restored.restart_objectives[restored.restart] == max(restored.restart_objectives)


def test_iteration_cap_is_not_convergence():
    fit = fit_learned_assembly(simulated(9, 30), 3, 54, max_iter=2)
    assert not fit.converged


def test_declared_capacity_can_initialize_with_replacement():
    x = np.zeros((4, 4), dtype=int)
    x[0, 0] = 3
    x[2, 2] = 4
    fit = fit_learned_assembly([x], 5, 3)
    assert fit.initialization_with_replacement
    assert fit.probabilities.shape == (4, 5)
    assert np.allclose(fit.probabilities.sum(axis=0), 1)
    assert np.isfinite(fit.objective_trace).all()


def decision_tables():
    summary = pd.DataFrame([
        {'dataset': d, 'contrast': f'k{k}__{axis}', 'mean': 2.0, 'ci_low': 1.0, 'ci_high': 3.0,
         'positive_animals': n, 'animals': n}
        for d, n in (('pfeiffer_foster', 4), ('tanni2022', 5))
        for k in (20, 50, 100) for axis in ENDPOINTS
    ])
    fits = pd.DataFrame([{'dataset': d, 'n_states': 50, 'fit_converged': True} for d in ('pfeiffer_foster', 'tanni2022')])
    return summary, fits


def test_weak_comparator_cannot_validate_spatial_claim():
    summary, fits = decision_tables()
    assert decisions(summary, fits).spatial_advantage_over_tested_assembly_supported.all()
    mask = summary.contrast.eq('k50__learned_hmm_minus_nonspatial_global')
    summary.loc[mask, 'ci_low'] = -1
    result = decisions(summary, fits)
    assert not result.spatial_advantage_over_tested_assembly_supported.any()
    assert result.verdict.eq('comparator_adequacy_not_established').all()


def test_uncertain_separation_is_not_equivalence():
    summary, fits = decision_tables()
    summary.loc[summary.contrast.eq('k50__spatial_imm_minus_learned_hmm'), 'ci_low'] = -1
    result = decisions(summary, fits)
    assert result.verdict.eq('no_rat_uniform_separation').all()
    with pytest.raises(ValueError, match='missing'):
        decisions(summary.iloc[:-1], fits)
    assert not decisions(summary, fits.assign(fit_converged=False)).learned_comparator_beats_global.any()


def test_reuse_checks_terminal_state_and_artifact_hashes(tmp_path):
    previous, out = tmp_path / 'old', tmp_path / 'new'
    previous.mkdir()
    out.mkdir()
    keys = ('parent', 'rate', 'protocol', 'hmmlearn_hmm.py', 'hmmlearn_base.py', 'hmmlearn__emissions.py', 'hmmlearn_compiled_kernel')
    inputs = {key: tmp_path / key for key in keys}
    for key, path in inputs.items():
        path.write_text(key)
    items = [{'tag': 'session'}]
    output_name = 'session__fold0__k20_fit.npz'
    (previous / output_name).write_bytes(b'fixture')
    result = {'tag': 'session', 'fold': 0, 'n_states': 20, 'output_sha256': {output_name: file_sha256(previous / output_name)}}
    name = 'session__fold0__k20_manifest.json'
    (previous / name).write_text(json.dumps(result))
    manifest = {'status': 'failed', 'selected_sessions': items, 'code_commit': 'fixture',
                'input_file_sha256': {key: file_sha256(path) for key, path in inputs.items()},
                'output_sha256': result['output_sha256'] | {name: file_sha256(previous / name)}}
    (previous / 'learned_assembly_manifest.json').write_text(json.dumps(manifest))
    reused = reuse_completed(previous, out, items, inputs)
    assert len(reused) == 1
    assert reused[0]['reused_producer_commit'] == 'fixture'
    assert (out / output_name).read_bytes() == b'fixture'
    (previous / output_name).write_bytes(b'tampered')
    with pytest.raises(ValueError, match='differs'):
        reuse_completed(previous, out, items, inputs)
    manifest['status'] = 'running'
    (previous / 'learned_assembly_manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='terminal'):
        reuse_completed(previous, out, items, inputs)

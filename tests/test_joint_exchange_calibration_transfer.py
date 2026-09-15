import json
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from scripts import audit_joint_exchange_calibration_transfer as m


def fixture():
    rng = np.random.default_rng(44)
    rates = rng.lognormal(2.5, 1.5, (12, 2))
    position = np.tile(np.arange(2), 20)
    grid = np.array([[0.0, 0.0], [20.0, 0.0]])
    enc = dict(early_run=rates, full_run=rates, grid_cm=grid, near=np.array([True, False]), high_indices=np.arange(6), low_indices=np.arange(4, 10), cell_ids=np.arange(12))
    first_counts = rng.poisson(0.02 * rates[:, position].T)
    counts = np.zeros((80, 12), int)
    counts[::2] = first_counts
    bank = dict(
        counts=counts,
        truth_cm=np.repeat(grid[position], 2, axis=0),
        labels=np.repeat(position == 0, 2),
        starts_s=np.arange(80) * 0.02,
        ends_s=(np.arange(80) + 1) * 0.02,
        parent_ids=np.repeat(np.arange(40), 2),
    )
    original = dict(high=list(range(6)), low=list(range(4, 10)))
    final = dict(high=[0, 3, 4, 5, 7, 9], low=[1, 2, 4, 5, 6, 8])
    first_bank = dict(counts=first_counts, truth_cm=grid[position], labels=position == 0)
    _, _, before_risks, _, before_j = m.truth.proof.state_for(original, enc, first_bank)
    _, _, after_risks, _, after_j = m.truth.proof.state_for(final, enc, first_bank)
    choice = dict(
        original=original,
        final_pair=final,
        net_exchanged=2,
        chosen="frozen",
        baseline_j=before_j,
        final_j=after_j,
        baseline_risks=before_risks.tolist(),
        candidates=[dict(name="frozen", risks=after_risks.tolist())],
    )
    return enc, bank, choice


def test_phase_partition_and_native_time_validation():
    _, bank, _ = fixture()
    phases = m.phase_indices(bank)
    assert np.array_equal(phases["first"], np.arange(0, 80, 2))
    assert np.array_equal(phases["remaining"], np.arange(1, 80, 2))
    assert set(phases["first"]) | set(phases["remaining"]) == set(phases["all"])
    for field in ("starts_s", "ends_s", "parent_ids"):
        bad = deepcopy(bank)
        if field == "parent_ids":
            bad[field] = np.arange(80)
        else:
            bad[field][3] += 1
        with pytest.raises(ValueError):
            m.phase_indices(bad)


def test_calibration_pass_can_coexist_with_unselected_phase_failure():
    enc, bank, choice = fixture()
    rows, summary = m.inspect_bank(enc, bank, choice, "Rat1/Open1", "run_q3")
    table = pd.DataFrame(summary).set_index("phase")
    assert len(rows) == 12
    assert table.loc["first", "all_eight_guards_pass"]
    assert not table.loc["remaining", "all_eight_guards_pass"]
    assert table.loc["first", "observations"] == table.loc["remaining", "observations"] == 40
    bad = deepcopy(choice)
    bad["final_j"] += 1
    with pytest.raises(AssertionError):
        m.inspect_bank(enc, bank, bad, "Rat1/Open1", "run_q3")


def test_full_fixed_choice_native_diagnostic_and_hashes(tmp_path):
    source, selected = tmp_path / "source", tmp_path / "selected"
    selected.mkdir()
    truth_dir = tmp_path / "truth"
    truth_dir.mkdir()
    inputs, choices = {}, {}
    for session in m.truth.exact.base.SESSIONS:
        enc, bank, choice = fixture()
        folder = source / session.replace("/", "_")
        folder.mkdir(parents=True)
        np.savez(folder / "encoding.npz", **enc)
        for name in ("run_q3", "run_q4"):
            np.savez(folder / f"{name}.npz", **bank)
        inputs.update({str(p): m.file_sha256(p) for p in folder.iterdir()})
        choices[session] = choice
    (selected / "selection.json").write_text(json.dumps(choices))
    selection_inputs = {p: h for p, h in inputs.items() if not p.endswith("/run_q4.npz")}
    (selected / "manifest.json").write_text(json.dumps(dict(input_file_sha256=selection_inputs, output_sha256={"selection.json": m.file_sha256(selected / "selection.json")})))
    (selected / "independent_audit.json").write_text(json.dumps(dict(status="pass", manifest_sha256=m.file_sha256(selected / "manifest.json"))))
    inputs[str(selected / "manifest.json")] = m.file_sha256(selected / "manifest.json")
    (truth_dir / "manifest.json").write_text(json.dumps(dict(input_file_sha256=inputs, output_sha256={})))
    (truth_dir / "independent_audit.json").write_text(json.dumps(dict(status="pass", manifest_sha256=m.file_sha256(truth_dir / "manifest.json"))))
    output = tmp_path / "output"
    m.run(selected, source, output, truth_dir)
    assert len(pd.read_csv(output / "classwise_native_transfer.csv")) == 96
    assert len(pd.read_csv(output / "native_transfer_summary.csv")) == 24
    record = json.loads((output / "manifest.json").read_text())
    assert record["posthoc_diagnostic"] and not record["replay_scored"] and not record["validated_remedy"] and not record["external_validation"]
    for name, sha in record["output_sha256"].items():
        assert m.file_sha256(output / name) == sha
    # A valid audit of a different selection is not a valid handoff.
    inputs.pop(str(selected / "manifest.json"))
    (truth_dir / "manifest.json").write_text(json.dumps(dict(input_file_sha256=inputs, output_sha256={})))
    (truth_dir / "independent_audit.json").write_text(json.dumps(dict(status="pass", manifest_sha256=m.file_sha256(truth_dir / "manifest.json"))))
    with pytest.raises(ValueError, match="another selection"):
        m.run(selected, source, tmp_path / "bad", truth_dir)

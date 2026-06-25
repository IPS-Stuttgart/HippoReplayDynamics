from pathlib import Path

import pandas as pd

from scripts.summarize_frank_eden_dataset_qc import (
    build_file_inventory,
    build_gate_summary,
    build_session_day_qc,
    build_source_data_inventory,
    write_qc_outputs,
)


def test_frank_eden_qc_detects_sorted_and_clusterless_candidate_days(tmp_path: Path):
    root = tmp_path / "frank_eden"
    root.mkdir()
    for name in [
        "bon_task01.mat",
        "bon_pos01.mat",
        "bon_linpos01.mat",
        "bon_spikes01.mat",
        "bon_marks01.mat",
        "bon_ripples01.mat",
        "bon_tetinfo.mat",
        "bon_cellinfo.mat",
        "elife-64505-fig3-data1-v2.csv",
    ]:
        (root / name).write_text("placeholder", encoding="utf-8")

    inventory = build_file_inventory(root)
    day_qc = build_session_day_qc(inventory)
    source_data = build_source_data_inventory(inventory)
    gates = build_gate_summary(inventory, day_qc, source_data)

    assert len(inventory) == 9
    assert len(source_data) == 1
    assert len(day_qc) == 1
    row = day_qc.iloc[0]
    assert row["animal"] == "bon"
    assert row["day"] == "01"
    assert row["candidate_replay_day"]
    assert row["candidate_clusterless_day"]
    assert row["has_linearized_position"]
    assert row["has_ripples"]
    assert gates[gates["gate"].eq("overall")]["passed"].iloc[0]


def test_frank_eden_qc_reports_missing_ripples(tmp_path: Path):
    root = tmp_path / "frank_eden"
    root.mkdir()
    for name in ["davtask02.mat", "davpos02.mat", "davspikes02.mat"]:
        (root / name).write_text("placeholder", encoding="utf-8")

    inventory = build_file_inventory(root)
    day_qc = build_session_day_qc(inventory)
    gates = build_gate_summary(inventory, day_qc, pd.DataFrame())

    assert len(day_qc) == 1
    row = day_qc.iloc[0]
    assert not row["candidate_replay_day"]
    assert "ripples" in row["missing_for_sorted_replay"]
    assert not gates[gates["gate"].eq("overall")]["passed"].iloc[0]


def test_frank_eden_qc_writes_expected_outputs(tmp_path: Path):
    root = tmp_path / "frank_eden"
    root.mkdir()
    for name in [
        "con_task1.mat",
        "con_pos1.mat",
        "con_spikes1.mat",
        "con_ripples1.mat",
    ]:
        (root / name).write_text("placeholder", encoding="utf-8")

    out = tmp_path / "results"
    paths = write_qc_outputs(root, out)

    assert set(paths) == {"inventory", "day_qc", "source_data", "gates", "summary"}
    for path in paths.values():
        assert path.exists()
    summary = paths["summary"].read_text(encoding="utf-8")
    assert "Frank/Eden Denovellis2021 Dataset QC Summary" in summary
    assert "10.7272/Q61N7ZC3" in summary

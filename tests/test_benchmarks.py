import numpy as np
import pandas as pd

from hipporeplayimm.benchmarks import (
    BenchmarkConfig,
    BenchmarkResult,
    _add_relative_metrics,
    _build_models,
    _is_best_static_baseline_model,
    _score_session,
    _session_mark_diagnostics,
    bootstrap_delta_ci,
)
from hipporeplayimm.clusterless import (
    ClusterlessMarkConfig,
    ClusterlessStateSpaceReplayModel,
    fit_clusterless_mark_encoding,
)
from hipporeplayimm.data import ReplaySession, SpikeMarkData
from hipporeplayimm.encoding import EmissionConfig, EncodingConfig
from hipporeplayimm.evidence_reporting import TRUNCATED_EVIDENCE_SUPPORT
from hipporeplayimm.models import EventScore


def test_benchmark_summary_and_bootstrap_ci():
    rows = pd.DataFrame(
        {
            "model": ["diffusion", "imm", "diffusion", "imm"],
            "heldout_log_likelihood": [-10.0, -9.0, -12.0, -10.0],
            "delta_vs_best_static": [0.0, 1.0, 0.0, 2.0],
            "bits_per_spike_vs_best_static": [0.0, 0.1, 0.0, 0.2],
        }
    )
    result = BenchmarkResult(rows)
    summary = result.summary()
    ci = bootstrap_delta_ci(rows, model="imm", n_bootstrap=100, random_seed=0)

    assert set(summary["model"]) == {"diffusion", "imm"}
    assert np.isfinite(ci[0])
    assert np.isfinite(ci[1])


def test_build_models_includes_opt_in_pyrecest_model():
    models = _build_models(
        BenchmarkConfig(
            models=("pyrecest-goal-particle",),
            pyrecest_particles=64,
            pyrecest_position_proposal_probability=0.5,
        )
    )

    assert set(models) == {"pyrecest-goal-particle"}
    assert models["pyrecest-goal-particle"].position_proposal_probability == 0.5


def test_build_models_includes_opt_in_pyrecest_imm_model():
    models = _build_models(
        BenchmarkConfig(models=("pyrecest-goal-particle-imm",), pyrecest_particles=64)
    )

    assert set(models) == {"pyrecest-goal-particle-imm"}


def test_state_space_aliases_canonicalize_sorted_spike_model_name():
    models = _build_models(BenchmarkConfig(models=("state-space-diffusion",)))

    assert models["state-space-diffusion"].name == "sorted-spike-state-space-diffusion"


def test_build_models_includes_exact_first_order_imm_models():
    models = _build_models(
        BenchmarkConfig(
            models=(
                "sorted-spike-state-space-first-order-imm",
                "state-space-first-order-imm",
                "clusterless-state-space-first-order-imm",
            )
        )
    )

    assert models["sorted-spike-state-space-first-order-imm"].name == "sorted-spike-state-space-first-order-imm"
    assert models["state-space-first-order-imm"].name == "sorted-spike-state-space-first-order-imm"
    assert isinstance(models["clusterless-state-space-first-order-imm"], ClusterlessStateSpaceReplayModel)


def test_build_models_includes_clusterless_state_space_model():
    models = _build_models(BenchmarkConfig(models=("clusterless-state-space-diffusion",)))

    assert isinstance(models["clusterless-state-space-diffusion"], ClusterlessStateSpaceReplayModel)


def test_session_mark_diagnostics_reports_available_clusterless_likelihood():
    diagnostics = _session_mark_diagnostics(_clusterless_benchmark_session())

    assert diagnostics["clusterless_mark_likelihood_available"]
    assert diagnostics["clusterless_mark_likelihood"] == ClusterlessMarkConfig().mark_likelihood

    diagnostics = _session_mark_diagnostics(
        _clusterless_benchmark_session(),
        clusterless_mark_likelihood="diag",
    )
    assert diagnostics["clusterless_mark_likelihood"] == "diagonal-gaussian"


def test_score_session_uses_clusterless_emissions_for_clusterless_models(monkeypatch):
    seen_cell_ids: list[tuple[int, ...]] = []

    def score_spy(self, emissions, bin_centers, candidate_indices=None):
        del candidate_indices
        seen_cell_ids.append(tuple(int(cell_id) for cell_id in emissions.cell_ids))
        return EventScore(
            str(self.name),
            float(emissions.n_spikes),
            emissions.n_time,
            emissions.n_spikes,
            diagnostics={"captured_cell_ids": ",".join(str(int(x)) for x in emissions.cell_ids)},
            terminal_log_posterior=np.zeros(emissions.n_bins),
        )

    monkeypatch.setattr(ClusterlessStateSpaceReplayModel, "score", score_spy)

    rows = _score_session(
        _clusterless_benchmark_session(),
        BenchmarkConfig(
            encoding=EncodingConfig(
                bin_size_cm=10.0,
                smoothing_sigma_bins=0.0,
                min_speed_cm_s=0.0,
                arena_padding_cm=5.0,
            ),
            emissions=EmissionConfig(time_bin_s=0.5),
            test_cell_fraction=0.5,
            random_seed=0,
            max_events_per_session=1,
            models=("clusterless-state-space-diffusion",),
        ),
    )

    assert len(rows) == 1
    assert seen_cell_ids == [(0,), (0,)]
    assert rows[0]["model"] == "clusterless-state-space-diffusion"
    assert rows[0]["test_spikes"] == 1
    assert rows[0]["diagnostic_captured_cell_ids"] == "0"
    assert rows[0]["clusterless_mark_likelihood"] == ClusterlessMarkConfig().mark_likelihood


def test_score_session_fits_clusterless_encoding_on_train_cells_only(monkeypatch):
    fitted_cell_ids: list[tuple[int, ...]] = []

    def fit_spy(session, config):
        marks = session.spike_marks
        assert marks is not None
        fitted_cell_ids.append(tuple(sorted({int(cell_id) for cell_id in marks.cell_ids})))
        return fit_clusterless_mark_encoding(session, config)

    def score_stub(self, emissions, bin_centers, candidate_indices=None):
        del bin_centers, candidate_indices
        return EventScore(
            str(self.name),
            float(emissions.n_spikes),
            emissions.n_time,
            emissions.n_spikes,
            diagnostics={},
            terminal_log_posterior=np.zeros(emissions.n_bins),
        )

    monkeypatch.setattr(
        "hipporeplayimm.benchmarks.fit_clusterless_mark_encoding",
        fit_spy,
    )
    monkeypatch.setattr(ClusterlessStateSpaceReplayModel, "score", score_stub)

    rows = _score_session(
        _clusterless_benchmark_session(),
        BenchmarkConfig(
            encoding=EncodingConfig(
                bin_size_cm=10.0,
                smoothing_sigma_bins=0.0,
                min_speed_cm_s=0.0,
                arena_padding_cm=5.0,
            ),
            emissions=EmissionConfig(time_bin_s=0.5),
            test_cell_fraction=0.5,
            random_seed=0,
            max_events_per_session=1,
            models=("clusterless-state-space-diffusion",),
        ),
    )

    assert len(rows) == 1
    assert rows[0]["train_cell_ids"] == "2"
    assert rows[0]["test_cell_ids"] == "1"
    assert rows[0]["test_spikes"] == 1
    assert fitted_cell_ids == [(2,)]


def test_best_static_baseline_includes_state_space_single_mode_models():
    assert _is_best_static_baseline_model("random")
    assert _is_best_static_baseline_model("diffusion")
    assert _is_best_static_baseline_model("momentum")
    assert _is_best_static_baseline_model("sorted-spike-state-space-stationary")
    assert _is_best_static_baseline_model("sorted-spike-state-space-diffusion")
    assert _is_best_static_baseline_model("sorted-spike-state-space-fragmented")
    assert _is_best_static_baseline_model("sorted-spike-state-space-jump")
    assert _is_best_static_baseline_model("sorted-spike-state-space-momentum")
    assert _is_best_static_baseline_model("state-space-diffusion")
    assert _is_best_static_baseline_model("clusterless-state-space-stationary")
    assert _is_best_static_baseline_model("clusterless-state-space-diffusion")
    assert _is_best_static_baseline_model("clusterless-state-space-fragmented")
    assert _is_best_static_baseline_model("clusterless-state-space-jump")
    assert _is_best_static_baseline_model("clusterless-state-space-momentum")
    assert not _is_best_static_baseline_model("imm")
    assert not _is_best_static_baseline_model("state-space-first-order-imm")
    assert not _is_best_static_baseline_model("sorted-spike-state-space-first-order-imm")
    assert not _is_best_static_baseline_model("sorted-spike-state-space-imm")
    assert not _is_best_static_baseline_model("clusterless-state-space-first-order-imm")
    assert not _is_best_static_baseline_model("clusterless-state-space-imm")
    assert not _is_best_static_baseline_model("pyrecest-goal-particle")
    assert not _is_best_static_baseline_model("pyrecest-goal-particle-imm")


def test_add_relative_metrics_uses_state_space_single_mode_baselines():
    rows = pd.DataFrame(
        {
            "session": ["s1", "s1", "s1", "s1"],
            "event_index": [7, 7, 7, 7],
            "model": [
                "sorted-spike-state-space-diffusion",
                "sorted-spike-state-space-momentum",
                "sorted-spike-state-space-jump",
                "sorted-spike-state-space-imm",
            ],
            "heldout_log_likelihood": [-8.0, -7.0, -9.0, -6.0],
            "test_spikes": [2, 2, 2, 2],
        }
    )

    result = _add_relative_metrics(rows)
    deltas = dict(zip(result["model"], result["delta_vs_best_static"]))

    assert deltas["sorted-spike-state-space-diffusion"] == -1.0
    assert deltas["sorted-spike-state-space-momentum"] == 0.0
    assert deltas["sorted-spike-state-space-jump"] == -2.0
    assert deltas["sorted-spike-state-space-imm"] == 1.0
    assert result["best_static_heldout_log_likelihood"].notna().all()


def test_add_relative_metrics_uses_clusterless_single_mode_baselines():
    rows = pd.DataFrame(
        {
            "session": ["s1", "s1", "s1", "s1"],
            "event_index": [7, 7, 7, 7],
            "model": [
                "clusterless-state-space-diffusion",
                "clusterless-state-space-momentum",
                "clusterless-state-space-jump",
                "clusterless-state-space-imm",
            ],
            "heldout_log_likelihood": [-8.0, -7.0, -9.0, -6.0],
            "test_spikes": [2, 2, 2, 2],
        }
    )

    result = _add_relative_metrics(rows)
    deltas = dict(zip(result["model"], result["delta_vs_best_static"]))

    assert deltas["clusterless-state-space-diffusion"] == -1.0
    assert deltas["clusterless-state-space-momentum"] == 0.0
    assert deltas["clusterless-state-space-jump"] == -2.0
    assert deltas["clusterless-state-space-imm"] == 1.0
    assert result["best_static_heldout_log_likelihood"].notna().all()


def test_add_relative_metrics_keeps_nan_when_no_static_baseline_is_present():
    rows = pd.DataFrame(
        {
            "session": ["s1"],
            "event_index": [7],
            "model": ["sorted-spike-state-space-imm"],
            "heldout_log_likelihood": [-6.0],
            "test_spikes": [2],
        }
    )

    result = _add_relative_metrics(rows)

    assert result["best_static_heldout_log_likelihood"].isna().all()
    assert result["delta_vs_best_static"].isna().all()


def test_add_relative_metrics_does_not_mix_exact_and_truncated_static_baselines():
    rows = pd.DataFrame(
        {
            "session": ["s1", "s1", "s1", "s1", "s1"],
            "event_index": [7, 7, 7, 7, 7],
            "model": ["random", "stationary", "diffusion", "momentum", "imm"],
            "heldout_log_likelihood": [-10.0, -9.0, -2.0, -4.0, -8.0],
            "test_spikes": [2, 2, 2, 2, 2],
            "diagnostic_candidate_evidence_support": [
                np.nan,
                np.nan,
                TRUNCATED_EVIDENCE_SUPPORT,
                TRUNCATED_EVIDENCE_SUPPORT,
                TRUNCATED_EVIDENCE_SUPPORT,
            ],
        }
    )

    result = _add_relative_metrics(rows)
    by_model = result.set_index("model")

    assert by_model["best_static_heldout_log_likelihood"].eq(-9.0).all()
    assert by_model.loc["random", "delta_vs_best_static"] == -1.0
    assert by_model.loc["stationary", "delta_vs_best_static"] == 0.0
    assert np.isnan(by_model.loc["diffusion", "delta_vs_best_static"])
    assert np.isnan(by_model.loc["momentum", "delta_vs_best_static"])
    assert np.isnan(by_model.loc["imm", "delta_vs_best_static"])
    assert by_model.loc["diffusion", "lower_bound_delta_vs_best_static"] == 7.0
    assert by_model.loc["momentum", "lower_bound_delta_vs_best_static"] == 5.0
    assert by_model.loc["imm", "lower_bound_delta_vs_best_static"] == 1.0
    assert by_model["best_static_truncated_lower_bound_heldout_log_likelihood"].eq(-2.0).all()
    assert by_model.loc["diffusion", "delta_vs_best_static_truncated_lower_bound"] == 0.0
    assert by_model.loc["momentum", "delta_vs_best_static_truncated_lower_bound"] == -2.0
    assert by_model.loc["imm", "delta_vs_best_static_truncated_lower_bound"] == -6.0


def test_benchmark_summary_separates_exact_and_truncated_support():
    rows = pd.DataFrame(
        {
            "session": ["s1", "s1", "s1"],
            "event_index": [7, 7, 7],
            "model": ["random", "stationary", "diffusion"],
            "heldout_log_likelihood": [-10.0, -9.0, -2.0],
            "test_spikes": [2, 2, 2],
            "diagnostic_candidate_evidence_support": [
                np.nan,
                np.nan,
                TRUNCATED_EVIDENCE_SUPPORT,
            ],
        }
    )

    summary = BenchmarkResult(_add_relative_metrics(rows)).summary().set_index("model")

    assert summary.loc["random", "evidence_comparable"]
    assert summary.loc["stationary", "evidence_comparable"]
    assert not summary.loc["diffusion", "evidence_comparable"]
    assert summary.loc["diffusion", "evidence_support"] == TRUNCATED_EVIDENCE_SUPPORT
    assert np.isnan(summary.loc["diffusion", "mean_delta_vs_best_static"])
    assert summary.loc["diffusion", "mean_lower_bound_delta_vs_best_static"] == 7.0


def _clusterless_benchmark_session() -> ReplaySession:
    position_times = np.linspace(0.0, 6.0, 61)
    x = np.where(position_times < 3.0, 0.0, 10.0)
    y = np.zeros_like(x)
    position = np.column_stack([position_times, x, y, np.zeros_like(x)])
    mark_times = np.array([0.5, 1.0, 3.5, 4.2, 4.4])
    cell_ids = np.array([1, 1, 2, 2, 1])
    marks = np.array([[0.0], [0.1], [10.0], [10.1], [0.05]])
    spikes = np.column_stack([mark_times, cell_ids])
    return ReplaySession(
        rat="RatX",
        name="OpenX",
        path=None,
        position=position,
        spikes=spikes,
        tetrode_cell_ids=np.array([[1, 1], [1, 2]]),
        excitatory_neurons=np.array([1, 2]),
        inhibitory_neurons=np.array([]),
        ripple_events=np.array([[4.0, 5.0, 4.5, 0.0, 0.0, 0.0]]),
        run_times=np.array([[0.0, 6.0]]),
        sleep_box_immobile_times=np.empty((0, 2)),
        sleep_times=np.empty((0, 2)),
        rem_times=np.empty((0, 2)),
        well_sequence=None,
        metadata={},
        spike_marks=SpikeMarkData(
            times=mark_times,
            marks=marks,
            source_file="Spike_Data.mat",
            source_variable="Spike_Amplitude_Marks",
            feature_names=("amp",),
            cell_ids=cell_ids,
        ),
    )

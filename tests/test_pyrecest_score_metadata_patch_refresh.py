from __future__ import annotations

import pandas as pd

from hipporeplayimm.pyrecest_score_metadata import (
    _BENCHMARK_METADATA_WRAPPER_MARKER,
    _COMPARE_WRAPPER_MARKER,
    _MODEL_SCORE_WRAPPER_MARKER,
    apply_pyrecest_score_metadata_patch,
)


def _unpatched_metadata(config):
    del config
    return {}


def _unpatched_compare(root, scores, **kwargs):
    del root, scores, kwargs
    return pd.DataFrame()


def _unpatched_score(self, emissions, bin_centers):
    del self, emissions, bin_centers
    return None


def test_pyrecest_metadata_patch_refreshes_stale_module_sentinel(monkeypatch) -> None:
    import hipporeplayimm.benchmarks as benchmarks
    import hipporeplayimm.ground_truth as ground_truth
    from hipporeplayimm.pyrecest_models import (
        PyRecEstGoalParticleIMMModel,
        PyRecEstGoalParticleModel,
    )

    monkeypatch.setattr(
        ground_truth,
        "_pyrecest_score_metadata_patch_applied",
        True,
        raising=False,
    )
    monkeypatch.setattr(benchmarks, "_benchmark_config_metadata", _unpatched_metadata)
    monkeypatch.setattr(ground_truth, "compare_scores_to_ground_truth", _unpatched_compare)
    monkeypatch.setattr(PyRecEstGoalParticleModel, "score", _unpatched_score)
    monkeypatch.setattr(PyRecEstGoalParticleIMMModel, "score", _unpatched_score)

    apply_pyrecest_score_metadata_patch()

    assert getattr(
        benchmarks._benchmark_config_metadata,
        _BENCHMARK_METADATA_WRAPPER_MARKER,
        False,
    )
    assert getattr(
        ground_truth.compare_scores_to_ground_truth,
        _COMPARE_WRAPPER_MARKER,
        False,
    )
    assert getattr(PyRecEstGoalParticleModel.score, _MODEL_SCORE_WRAPPER_MARKER, False)
    assert getattr(PyRecEstGoalParticleIMMModel.score, _MODEL_SCORE_WRAPPER_MARKER, False)

    metadata_wrapper = benchmarks._benchmark_config_metadata
    compare_wrapper = ground_truth.compare_scores_to_ground_truth
    model_wrapper = PyRecEstGoalParticleModel.score
    imm_wrapper = PyRecEstGoalParticleIMMModel.score

    apply_pyrecest_score_metadata_patch()

    assert benchmarks._benchmark_config_metadata is metadata_wrapper
    assert ground_truth.compare_scores_to_ground_truth is compare_wrapper
    assert PyRecEstGoalParticleModel.score is model_wrapper
    assert PyRecEstGoalParticleIMMModel.score is imm_wrapper

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Event, Lock
from types import SimpleNamespace


def test_pyrecest_score_calls_serialize_legacy_numpy_rng_use(monkeypatch):
    from hipporeplayimm import benchmarks as bench
    from hipporeplayimm import ground_truth as ground_truth
    from hipporeplayimm import pyrecest_score_metadata as metadata
    from hipporeplayimm.pyrecest_models import PyRecEstGoalParticleModel

    state_lock = Lock()
    start = Barrier(3)
    release = Event()
    two_calls_inside = Event()
    active_calls = 0
    max_active_calls = 0

    def fake_score(self, emissions, bin_centers):
        nonlocal active_calls, max_active_calls
        with state_lock:
            active_calls += 1
            max_active_calls = max(max_active_calls, active_calls)
            if active_calls == 2:
                two_calls_inside.set()
        try:
            if not release.wait(timeout=5.0):
                raise TimeoutError("test did not release the serialized score call")
            return SimpleNamespace(diagnostics={})
        finally:
            with state_lock:
                active_calls -= 1

    # Record every mutation performed by reapplying the idempotent runtime patch
    # so pytest restores the already-imported package after this focused test.
    monkeypatch.setattr(bench, "_benchmark_config_metadata", bench._benchmark_config_metadata)
    monkeypatch.setattr(
        ground_truth,
        "compare_scores_to_ground_truth",
        ground_truth.compare_scores_to_ground_truth,
    )
    monkeypatch.setattr(ground_truth, "_pyrecest_score_metadata_patch_applied", False)
    monkeypatch.setattr(PyRecEstGoalParticleModel, "score", fake_score)
    metadata.apply_pyrecest_score_metadata_patch()

    models = [PyRecEstGoalParticleModel(random_seed=1), PyRecEstGoalParticleModel(random_seed=2)]

    def call_score(model):
        start.wait(timeout=5.0)
        return model.score(None, None)

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(call_score, model) for model in models]
        start.wait(timeout=5.0)
        try:
            assert not two_calls_inside.wait(timeout=0.2)
        finally:
            release.set()
        results = [future.result(timeout=5.0) for future in futures]

    assert max_active_calls == 1
    assert [result.diagnostics["pyrecest_particles"] for result in results] == [512, 512]

import json

import numpy as np
import pandas as pd
import pytest
from scipy.spatial import Delaunay

from hipporeplayimm.fresh_path_clocks import draw_path, family_indices, generate_event, rng
from hipporeplayimm.literal_replay_clock import SpatialRates
from hipporeplayimm.unknown_path_clocks import MODELS
from scripts.run_fresh_path_clock_recovery import generate_repeat
from scripts.verify_fresh_path_clock_recovery import audit_generation, check_summary, reference_event


def example():
    centers = np.array([(x, y) for x in range(0, 65, 8) for y in range(0, 65, 8)], float)
    peaks = np.array([[8, 8], [56, 8], [8, 56], [56, 56], [32, 32]])
    rates = 0.2 + 10 * np.exp(-np.sum((centers[None] - peaks[:, None]) ** 2, axis=2) / 500)
    descriptors = np.array([[10, 70, 0], [10, 70, 1], [10, 70, -1]])
    return centers, rates, descriptors


def test_families_have_equal_mass_not_uniform_union():
    _, _, d = example()
    families = family_indices(d)
    g = rng("family")
    counts = np.bincount([draw_path(families, g) for _ in range(12000)], minlength=3) / 12000
    np.testing.assert_allclose(counts, [0.5, 0.25, 0.25], atol=0.015)
    with pytest.raises(ValueError):
        family_indices(d[:1])


@pytest.mark.parametrize("model", MODELS)
def test_counts_and_latents_match_independent_generation(model):
    centers, rates, d = example()
    families = family_indices(d)
    totals = np.array([0, 3, 10, 4])
    a, ids = generate_event(totals, model, SpatialRates(centers, rates), rates, d, families, rng(model))
    b, other = reference_event(totals, model, centers, rates, Delaunay(centers), d, families, rng(model))
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(ids, other)
    np.testing.assert_array_equal(a.sum(axis=1), totals)
    if not model.endswith("reset"):
        assert len(set(ids)) == 1


def test_fresh_seeds_do_not_reuse_one_path_or_spike_draw():
    centers, rates, d = example()
    families = family_indices(d)
    spatial = SpatialRates(centers, rates)
    observations = [generate_event([20] * 4, "physical_reset", spatial, rates, d, families, rng("event", i)) for i in range(10)]
    assert len({tuple(ids) for _, ids in observations}) > 3
    assert len({tuple(x.ravel()) for x, _ in observations}) == 10
    with pytest.raises(ValueError):
        generate_event([1.2], "physical", spatial, rates, d, families, rng("bad"))


def test_missing_summary_cannot_pass_vacuously():
    empty = pd.DataFrame(columns=["dataset", "teacher", "scenario", "repeat", "support"])
    with pytest.raises(ValueError):
        check_summary(empty, empty, empty)


def test_generation_archive_roundtrip_and_corruption(tmp_path):
    centers, rates, descriptors = example()
    previous, source, output = [tmp_path / name for name in ("previous", "source", "output")]
    for folder in (previous, source, output):
        folder.mkdir()
    tag = "mini"
    metadata = pd.DataFrame(
        {
            "repeat": 0,
            "row_index": range(5),
            "observation_index": range(5),
            "generator": MODELS,
            "source_teacher": "fresh_stream_0",
            "event_in_population": range(5),
            "scenario": 0.5,
        }
    )
    metadata.to_csv(output / f"{tag}_observations.csv.gz", index=False)
    np.save(output / f"{tag}_prior_descriptors.npy", descriptors)
    np.savez(source / f"{tag}_cache.npz", centers=centers, rates=rates)
    original = np.ones((20, 5), dtype=np.int32)
    np.savez(previous / f"{tag}_r000_observations.npz", counts=original, offsets=np.arange(0, 21, 4))
    (previous / "exact_path_clock_manifest.json").write_text(json.dumps({"parent_dir": str(previous)}))
    record = generate_repeat(tag, 0, previous, source, output)
    manifest = {"source_dir": str(source), "parent_dir": str(previous)}
    result = audit_generation(record, manifest, output)
    assert result["observations_regenerated"] == 5 and result["count_vectors_regenerated"] == 20
    path = output / record["file"]
    with np.load(path) as z:
        corrupt = {name: z[name] for name in z.files}
    corrupt["latent"][0] += 1
    np.savez(path, **corrupt)
    with pytest.raises(AssertionError):
        audit_generation(record, manifest, output)

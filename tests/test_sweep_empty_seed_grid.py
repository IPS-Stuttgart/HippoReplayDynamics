import pytest

from hipporeplayimm.sweeps import PyRecEstSweepConfig, pyrecest_parameter_grid


def test_empty_random_seed_grid_is_rejected():
    config = PyRecEstSweepConfig(random_seeds=())

    with pytest.raises(ValueError):
        pyrecest_parameter_grid(config)

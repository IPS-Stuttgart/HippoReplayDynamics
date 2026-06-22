from hipporeplayimm.simulation_recovery import model_family


def test_state_space_velocity_momentum_alias_is_trajectory_family():
    assert model_family("state-space-velocity-momentum") == "trajectory"

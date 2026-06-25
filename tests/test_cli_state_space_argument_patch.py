from __future__ import annotations

from argparse import ArgumentParser

from hipporeplayimm import cli
from hipporeplayimm.benchmarks import BenchmarkConfig


def test_shared_state_space_predicted_candidate_top_k_cli_option_is_registered() -> None:
    parser = ArgumentParser()
    cli._add_state_space_arguments(parser)

    args = parser.parse_args(["--state-space-momentum-predicted-candidate-top-k", "13"])
    kwargs = cli._state_space_scalar_kwargs(args)

    assert kwargs["state_space_momentum_predicted_candidate_top_k"] == 13
    assert BenchmarkConfig(**kwargs).state_space_momentum_predicted_candidate_top_k == 13

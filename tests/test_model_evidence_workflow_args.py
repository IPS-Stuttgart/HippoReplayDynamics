import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = (
    ROOT / ".github" / "workflows" / "model-evidence.yml",
    ROOT / ".github" / "workflows" / "model-evidence-event-sharded.yml",
    ROOT / ".github" / "workflows" / "model-evidence-all-sessions.yml",
)
SCRIPT = ROOT / "scripts" / "benchmark_model_evidence.py"

_REQUIRED_WORKFLOW_FLAGS = {
    "--clusterless-mark-smoothing-sigma-bins",
    "--clusterless-mark-prior-count",
    "--clusterless-mark-variance-floor",
    "--clusterless-rate-floor-hz",
    "--clusterless-mark-likelihood",
    "--clusterless-mark-kde-bandwidth",
    "--clusterless-mark-kde-spatial-sigma-bins",
    "--clusterless-mark-kde-max-neighbors",
    "--spike-rate-scale",
    "--emission-likelihood-temperature",
    "--emission-negative-binomial-overdispersion",
    "--time-bin-s",
    "--bin-size-cm",
    "--smoothing-sigma-bins",
    "--min-speed-cm-s",
    "--state-space-valid-occupancy-threshold-s",
    "--state-space-momentum-initial-sigma-cm-sqrt-s",
}


@pytest.mark.parametrize("workflow", WORKFLOWS)
def test_model_evidence_workflows_pass_only_supported_model_evidence_args(workflow: Path):
    workflow_flags = _workflow_benchmark_flags(workflow)
    script_flags = _benchmark_script_flags()

    unsupported = workflow_flags - script_flags

    assert not unsupported, f"{workflow.name} passes unsupported benchmark flags: {sorted(unsupported)}"


@pytest.mark.parametrize("workflow", WORKFLOWS)
def test_model_evidence_workflows_wire_clusterless_and_encoder_knobs(workflow: Path):
    workflow_flags = _workflow_benchmark_flags(workflow)

    missing = _REQUIRED_WORKFLOW_FLAGS - workflow_flags

    assert not missing, f"{workflow.name} does not pass expected benchmark flags: {sorted(missing)}"


def test_event_sharded_workflow_includes_exact_first_order_imm_default():
    text = (ROOT / ".github" / "workflows" / "model-evidence-event-sharded.yml").read_text(encoding="utf-8")

    assert (
        "sorted-spike-state-space-first-order-imm"
        in _workflow_dispatch_default_models(text)
    )


@pytest.mark.parametrize(
    "workflow_name",
    (
        "model-evidence-event-sharded.yml",
        "model-evidence-all-sessions.yml",
    ),
)
def test_reproducible_evidence_workflows_expose_momentum_initial_sigma_input(
    workflow_name: str,
):
    text = (ROOT / ".github" / "workflows" / workflow_name).read_text(encoding="utf-8")

    assert "state_space_momentum_initial_sigma_cm_sqrt_s:" in text
    assert (
        "STATE_SPACE_MOMENTUM_INITIAL_SIGMA_CM_SQRT_S: "
        "${{ inputs.state_space_momentum_initial_sigma_cm_sqrt_s }}"
    ) in text
    assert "inputs.state_space_momentum_initial_sigma_cm_sqrt_s" in _workflow_concurrency_group(
        text
    )


def test_clusterless_first_order_imm_is_not_silently_skipped_by_model_evidence_script():
    text = SCRIPT.read_text(encoding="utf-8")

    assert 'if name == "clusterless-state-space-first-order-imm"' not in text
    assert "clusterless-state-space-first-order-imm" in text


def _workflow_benchmark_flags(workflow: Path) -> set[str]:
    text = workflow.read_text(encoding="utf-8")
    start = text.index("python scripts/benchmark_model_evidence.py")
    end = text.index("--continue-on-error", start) + len("--continue-on-error")
    block = text[start:end]
    return set(re.findall(r"--[a-z0-9-]+", block))


def _benchmark_script_flags() -> set[str]:
    text = SCRIPT.read_text(encoding="utf-8")
    return set(re.findall(r"add_argument\(\s*[\"'](--[a-z0-9-]+)[\"']", text, flags=re.MULTILINE))


def _workflow_concurrency_group(text: str) -> str:
    match = re.search(r"concurrency:\n\s+group:\s+([^\n]+)", text)
    assert match is not None
    return match.group(1)


def _workflow_dispatch_default_models(text: str) -> str:
    match = re.search(
        r"models:\n(?:\s+[^\n]*\n)*?\s+default:\s+\"([^\"]+)\"",
        text,
    )
    assert match is not None
    return match.group(1)

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "model-evidence-event-sharded.yml"
SCRIPT = ROOT / "scripts" / "benchmark_model_evidence.py"

_REQUIRED_WORKFLOW_FLAGS = {
    "--clusterless-mark-smoothing-sigma-bins",
    "--clusterless-mark-prior-count",
    "--clusterless-mark-variance-floor",
    "--clusterless-rate-floor-hz",
    "--spike-rate-scale",
    "--time-bin-s",
    "--bin-size-cm",
    "--smoothing-sigma-bins",
    "--min-speed-cm-s",
}


def test_event_sharded_workflow_passes_only_supported_model_evidence_args():
    workflow_flags = _workflow_benchmark_flags()
    script_flags = _benchmark_script_flags()

    unsupported = workflow_flags - script_flags

    assert not unsupported, f"Workflow passes unsupported benchmark flags: {sorted(unsupported)}"


def test_event_sharded_workflow_wires_clusterless_and_encoder_knobs():
    workflow_flags = _workflow_benchmark_flags()

    missing = _REQUIRED_WORKFLOW_FLAGS - workflow_flags

    assert not missing, f"Workflow does not pass expected benchmark flags: {sorted(missing)}"


def _workflow_benchmark_flags() -> set[str]:
    text = WORKFLOW.read_text(encoding="utf-8")
    start = text.index("python scripts/benchmark_model_evidence.py")
    end = text.index("--continue-on-error", start) + len("--continue-on-error")
    block = text[start:end]
    return set(re.findall(r"--[a-z0-9-]+", block))


def _benchmark_script_flags() -> set[str]:
    text = SCRIPT.read_text(encoding="utf-8")
    return set(re.findall(r"add_argument\(\s*[\"'](--[a-z0-9-]+)[\"']", text, flags=re.MULTILINE))

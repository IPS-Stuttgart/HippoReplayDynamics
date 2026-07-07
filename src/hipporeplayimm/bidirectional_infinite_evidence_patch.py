"""Keep bidirectional replay mixtures finite and endpoint-complete."""

from __future__ import annotations

from .bidirectional_infinite_evidence_patch_impl import apply_bidirectional_infinite_evidence_patch

__all__ = ["apply_bidirectional_infinite_evidence_patch"]

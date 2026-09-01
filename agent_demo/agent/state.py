from __future__ import annotations

from typing import Any, TypedDict


class FaultAgentState(TypedDict, total=False):
    """Shared state passed through the diagnosis workflow."""

    signal: list[float]
    sampling_rate: float
    diagnosis: dict[str, Any]
    signal_features: dict[str, Any]
    retrieved_knowledge: list[dict[str, Any]]
    final_report: str

# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Reward-tagged trajectory reservoir and proposal history for co-evolution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from openjiuwen.agent_evolving.skill_bank.models import (
    ProposalStatus,
    param_error as _param,
)


@dataclass(frozen=True)
class ReservoirEntry:
    """One reward-tagged rollout observation under a concrete bank version.

    ``payload`` optionally carries a compact trajectory digest (for example a
    tool-call chain and final response) so the ReSkill creator can mine retained
    successes and failures without re-reading rollout persistence.
    """

    version_id: str
    origin_task_id: str
    task_id: str
    rollout_id: str | None
    reward: float
    success: bool
    task_key: str | None = None
    payload: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if not str(self.version_id).strip():
            raise _param("reservoir entry requires version_id")
        if not str(self.origin_task_id).strip():
            raise _param("reservoir entry requires origin_task_id")
        if self.payload is not None and not isinstance(self.payload, dict):
            raise _param("reservoir entry payload must be an object")


@dataclass(frozen=True)
class ProposalCycleRecord:
    """Audit trail of one adoption-cycle decision made by the controller."""

    cycle: int
    proposal_id: str
    status: ProposalStatus
    baseline_version_id: str
    candidate_version_id: str
    baseline_posterior_mean: float
    candidate_posterior_mean: float
    observation_count: int

    def as_metrics(self) -> dict[str, float]:
        """Return this cycle's decision as ``skill_rl/*`` training metrics."""
        return {
            "skill_rl/candidate_accepted": float(self.status is ProposalStatus.ACCEPTED),
            "skill_rl/baseline_posterior_mean": self.baseline_posterior_mean,
            "skill_rl/candidate_posterior_mean": self.candidate_posterior_mean,
            "skill_rl/observations": float(self.observation_count),
        }


class TrajectoryReservoir:
    """Bounded recent reservoir of reward-tagged rollouts.

    ReSkill retains the latest success and failure for each task/version pair.
    Per-version cycle counts remain exact regardless of retained-entry
    replacement or eviction.
    """

    def __init__(self, capacity: int = 200) -> None:
        if capacity <= 0:
            raise _param("reservoir capacity must be positive")
        self._capacity = capacity
        self._entries: dict[tuple[str, str, bool], ReservoirEntry] = {}
        # Exact per-version (successes, observations) for the current cycle;
        # reset at each cycle boundary, unaffected by reservoir eviction.
        self._cycle_counts: dict[str, tuple[int, int]] = {}

    def add(self, entry: ReservoirEntry) -> None:
        """Record one observation and retain the latest matching outcome."""
        successes, observations = self._cycle_counts.get(entry.version_id, (0, 0))
        self._cycle_counts[entry.version_id] = (successes + int(entry.success), observations + 1)

        task_key = entry.task_key or entry.origin_task_id
        key = (task_key, entry.version_id, entry.success)
        self._entries.pop(key, None)
        self._entries[key] = entry
        while len(self._entries) > self._capacity:
            self._entries.pop(next(iter(self._entries)))

    def entries(self, version_id: str | None = None) -> tuple[ReservoirEntry, ...]:
        """Return retained entries, optionally filtered to one version."""
        retained = reversed(self._entries.values())
        if version_id is None:
            return tuple(retained)
        return tuple(entry for entry in retained if entry.version_id == version_id)

    def cycle_counts(self, version_id: str) -> tuple[int, int]:
        """Return exact (successes, observations) for the current cycle."""
        return self._cycle_counts.get(version_id, (0, 0))

    def reset_cycle(self) -> None:
        """Start a new adoption cycle; retained trajectories are kept."""
        self._cycle_counts.clear()


__all__ = [
    "ProposalCycleRecord",
    "ReservoirEntry",
    "TrajectoryReservoir",
]

# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Deterministic trajectory assertions used by skill-bank diagnosis."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Sequence

from openjiuwen.agent_evolving.skill_bank.models import param_error as _param
from openjiuwen.agent_evolving.skill_bank.reservoir import ReservoirEntry


class AssertionKind(str, Enum):
    """Supported rule-based checks over compact rollout trajectories."""

    ACTION_MATCHES = "action_matches"
    ACTION_ABSENT = "action_absent"
    ACTION_ORDER = "action_order"
    MAX_ACTION_REPEATS = "max_action_repeats"
    MAX_STEPS = "max_steps"
    OBSERVATION_MATCHES = "observation_matches"


@dataclass(frozen=True)
class TrajectoryAssertion:
    """One serializable predicate maintained by the creator pipeline."""

    name: str
    kind: AssertionKind
    pattern: str = ""
    second_pattern: str = ""
    limit: int = 0

    def __post_init__(self) -> None:
        name = str(self.name).strip()
        if not name:
            raise _param("trajectory assertion requires a name")
        try:
            kind = self.kind if isinstance(self.kind, AssertionKind) else AssertionKind(self.kind)
        except (TypeError, ValueError) as exc:
            raise _param(f"unsupported trajectory assertion kind: {self.kind}") from exc
        if kind in {
            AssertionKind.ACTION_MATCHES,
            AssertionKind.ACTION_ABSENT,
            AssertionKind.ACTION_ORDER,
            AssertionKind.OBSERVATION_MATCHES,
        }:
            _validate_pattern(self.pattern, field="pattern")
        if kind is AssertionKind.ACTION_ORDER:
            _validate_pattern(self.second_pattern, field="second_pattern")
        if kind in {AssertionKind.MAX_ACTION_REPEATS, AssertionKind.MAX_STEPS} and self.limit <= 0:
            raise _param(f"assertion '{name}' requires a positive limit")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "kind", kind)

    def evaluate(self, payload: dict[str, Any]) -> bool:
        """Evaluate this assertion without invoking an LLM."""
        actions = trajectory_actions(payload)
        observations = trajectory_observations(payload)
        if self.kind is AssertionKind.ACTION_MATCHES:
            return any(re.search(self.pattern, action) for action in actions)
        if self.kind is AssertionKind.ACTION_ABSENT:
            return not any(re.search(self.pattern, action) for action in actions)
        if self.kind is AssertionKind.ACTION_ORDER:
            first = next((index for index, action in enumerate(actions) if re.search(self.pattern, action)), None)
            return first is not None and any(re.search(self.second_pattern, action) for action in actions[first + 1 :])
        if self.kind is AssertionKind.MAX_ACTION_REPEATS:
            return _max_consecutive_repeats(actions) <= self.limit
        if self.kind is AssertionKind.MAX_STEPS:
            return len(actions) <= self.limit
        return any(re.search(self.pattern, observation) for observation in observations)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"name": self.name, "kind": self.kind.value}
        if self.pattern:
            payload["pattern"] = self.pattern
        if self.second_pattern:
            payload["second_pattern"] = self.second_pattern
        if self.limit:
            payload["limit"] = self.limit
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TrajectoryAssertion":
        try:
            return cls(
                name=data.get("name", ""),
                kind=data.get("kind", ""),
                pattern=str(data.get("pattern", "")),
                second_pattern=str(data.get("second_pattern", "")),
                limit=int(data.get("limit", 0)),
            )
        except (TypeError, ValueError) as exc:
            raise _param("invalid trajectory assertion payload", cause=exc) from exc


@dataclass(frozen=True)
class AssertionGrade:
    """Aggregate result of applying one assertion to the reservoir."""

    assertion: TrajectoryAssertion
    passed: int
    total: int

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total if self.total else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.assertion.to_dict(),
            "passed": self.passed,
            "total": self.total,
            "pass_rate": self.pass_rate,
        }


def grade_assertions(
    assertions: Sequence[TrajectoryAssertion],
    entries: Sequence[ReservoirEntry],
) -> tuple[AssertionGrade, ...]:
    """Apply all assertions to every retained trajectory with a payload."""
    payloads = [entry.payload for entry in entries if entry.payload is not None]
    return tuple(
        AssertionGrade(
            assertion=assertion,
            passed=sum(assertion.evaluate(payload) for payload in payloads),
            total=len(payloads),
        )
        for assertion in assertions
    )


def trajectory_actions(payload: dict[str, Any]) -> list[str]:
    """Return normalized action strings from a compact trajectory payload."""
    turns = payload.get("turns", [])
    if not isinstance(turns, list):
        return []
    actions = []
    for turn in turns:
        if isinstance(turn, dict) and (text := serialize_trajectory_value(turn.get("action"))):
            actions.append(text)
    return actions


def trajectory_observations(payload: dict[str, Any]) -> list[str]:
    """Return normalized observation strings from a compact trajectory payload."""
    turns = payload.get("turns", [])
    if not isinstance(turns, list):
        return []
    observations = []
    for turn in turns:
        if isinstance(turn, dict) and (text := serialize_trajectory_value(turn.get("observation"))):
            observations.append(text)
    return observations


def serialize_trajectory_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        value = dump()
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    except (TypeError, ValueError):
        return str(value)


def _validate_pattern(pattern: str, *, field: str) -> None:
    if not pattern:
        raise _param(f"trajectory assertion requires {field}")
    try:
        re.compile(pattern)
    except re.error as exc:
        raise _param(f"invalid trajectory assertion {field}: {exc}", cause=exc) from exc


def _max_consecutive_repeats(actions: Sequence[str]) -> int:
    longest = current = 0
    previous = None
    for action in actions:
        current = current + 1 if action == previous else 1
        longest = max(longest, current)
        previous = action
    return longest


__all__ = [
    "AssertionGrade",
    "AssertionKind",
    "TrajectoryAssertion",
    "grade_assertions",
    "serialize_trajectory_value",
    "trajectory_actions",
    "trajectory_observations",
]

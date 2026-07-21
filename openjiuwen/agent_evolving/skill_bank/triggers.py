# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Deterministic ReSkill trigger parsing, selection, and validation."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Sequence

from openjiuwen.agent_evolving.skill_bank.assertions import serialize_trajectory_value, trajectory_actions
from openjiuwen.agent_evolving.skill_bank.models import param_error as _param
from openjiuwen.agent_evolving.skill_bank.reservoir import ReservoirEntry
from openjiuwen.agent_evolving.utils import parse_top_level_frontmatter


class SkillTriggerType(str, Enum):
    """Trigger types supported by ReSkill's deterministic runtime gate."""

    GENERAL = "general"
    BEGINNING = "beginning"
    ACTION_PATTERN = "action_pattern"


@dataclass(frozen=True)
class SkillTrigger:
    """Trigger metadata read from one existing SKILL.md package."""

    trigger_type: SkillTriggerType = SkillTriggerType.GENERAL
    pattern: str = ""
    when_to_use: str = ""

    def __post_init__(self) -> None:
        try:
            trigger_type = (
                self.trigger_type
                if isinstance(self.trigger_type, SkillTriggerType)
                else SkillTriggerType(self.trigger_type)
            )
        except (TypeError, ValueError) as exc:
            raise _param(f"unsupported skill trigger type: {self.trigger_type}") from exc
        if trigger_type is SkillTriggerType.ACTION_PATTERN:
            if not self.pattern:
                raise _param("action_pattern skill trigger requires trigger_pattern")
            try:
                re.compile(self.pattern)
            except re.error as exc:
                raise _param(f"invalid skill trigger pattern: {exc}", cause=exc) from exc
        object.__setattr__(self, "trigger_type", trigger_type)

    def fires(self, *, step: int, previous_action: str | None) -> bool:
        if self.trigger_type is SkillTriggerType.GENERAL:
            return True
        if self.trigger_type is SkillTriggerType.BEGINNING:
            return step == 0
        return previous_action is not None and re.search(self.pattern, previous_action) is not None

    @classmethod
    def from_skill_md(cls, content: str) -> "SkillTrigger":
        metadata = parse_top_level_frontmatter(content)
        raw_type = metadata.get("trigger_type", SkillTriggerType.GENERAL.value).strip().lower()
        return cls(
            trigger_type=raw_type,
            pattern=metadata.get("trigger_pattern", "").strip().strip("'\""),
            when_to_use=metadata.get("when_to_use", "").strip().strip("'\""),
        )


class SkillTriggerSelector:
    """Select triggered skills from the existing SkillUseRail skill list."""

    def __init__(self) -> None:
        self._cache: dict[Path, SkillTrigger] = {}

    def __call__(self, skills: Sequence[Any], ctx: Any) -> list[Any]:
        actions = _message_actions(getattr(ctx.inputs, "messages", []))
        previous_action = actions[-1] if actions else None
        selected = []
        for skill in skills:
            skill_md = Path(skill.directory) / "SKILL.md"
            trigger = self._cache.get(skill_md)
            if trigger is None:
                trigger = SkillTrigger.from_skill_md(skill_md.read_text(encoding="utf-8", errors="replace"))
                self._cache[skill_md] = trigger
            if trigger.fires(step=len(actions), previous_action=previous_action):
                selected.append(skill)
        return selected


def validate_skill_trigger(
    skill_name: str,
    skill_md: str,
    entries: Sequence[ReservoirEntry],
    *,
    minimum_fire_rate: float = 0.5,
) -> str | None:
    """Return validation feedback for one authored trigger, or ``None``."""
    if not 0.0 <= minimum_fire_rate <= 1.0:
        raise _param("minimum trigger fire rate must be between zero and one")
    metadata = parse_top_level_frontmatter(skill_md)
    if metadata.get("scope", "").strip().lower() not in {"general", "task"}:
        return f"skill '{skill_name}' must declare scope as general or task"
    trigger = SkillTrigger.from_skill_md(skill_md)
    if not trigger.when_to_use:
        return f"skill '{skill_name}' must declare when_to_use"
    if trigger.trigger_type is not SkillTriggerType.ACTION_PATTERN:
        return None
    rate = skill_trigger_fire_rate(trigger, entries)
    if rate < minimum_fire_rate:
        return (
            f"skill '{skill_name}' trigger fires on {rate:.1%} of retained trajectories; "
            f"minimum is {minimum_fire_rate:.1%}"
        )
    return None


def skill_trigger_fire_rate(trigger: SkillTrigger, entries: Sequence[ReservoirEntry]) -> float:
    """Estimate episode-level trigger coverage over retained trajectories."""
    if trigger.trigger_type is not SkillTriggerType.ACTION_PATTERN:
        return 1.0
    payloads = [entry.payload for entry in entries if entry.payload is not None]
    fired = sum(
        any(
            trigger.fires(step=index + 1, previous_action=action)
            for index, action in enumerate(trajectory_actions(payload))
        )
        for payload in payloads
    )
    return fired / len(payloads) if payloads else 0.0


def _message_actions(messages: Sequence[Any]) -> list[str]:
    actions = []
    for message in messages:
        if not isinstance(message, dict):
            dump = getattr(message, "model_dump", None)
            message = dump() if callable(dump) else {}
        if message.get("role") != "assistant" or not message.get("tool_calls"):
            continue
        actions.append(serialize_trajectory_value(message["tool_calls"]))
    return actions


__all__ = [
    "SkillTrigger",
    "SkillTriggerSelector",
    "SkillTriggerType",
    "skill_trigger_fire_rate",
    "validate_skill_trigger",
]

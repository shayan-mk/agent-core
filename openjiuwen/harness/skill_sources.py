# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Shared source ordering and effective-state helpers for harness skills."""

from __future__ import annotations

import json
from os import PathLike
from pathlib import Path
from typing import Optional, Sequence

from openjiuwen.core.common.logging import logger
from openjiuwen.harness.workspace.workspace import Workspace


def collect_disabled_skills_from_state(skills_dirs: Sequence[str | PathLike]) -> list[str]:
    """Read each root's state and return the disabled-skill union."""
    disabled: set[str] = set()
    for skills_dir in skills_dirs:
        state_path = Path(skills_dir) / "skills_state.json"
        if not state_path.is_file():
            continue
        try:
            data = json.loads(state_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            logger.warning("Failed to read skills_state.json at %s", state_path)
            continue
        skill_configs = data.get("skill_configs", {})
        for name, config in skill_configs.items():
            if isinstance(config, dict) and config.get("enabled") is False:
                disabled.add(name)
    return sorted(disabled)


def resolve_skill_roots(
    workspace: Workspace,
    explicit_roots: Optional[Sequence[str | PathLike]] = None,
) -> list[str]:
    """Resolve ordered skill roots with the precedence used by DeepAgent."""
    if explicit_roots is not None:
        return [str(Path(root).expanduser().resolve()) for root in explicit_roots]

    roots: list[str] = []
    skills_base = workspace.get_node_path("skills")
    if skills_base:
        roots.append(str(skills_base))
    for _team_id, target_path in workspace.list_team_links():
        roots.append(str(Path(target_path) / "skills"))
    return roots


__all__ = ["collect_disabled_skills_from_state", "resolve_skill_roots"]

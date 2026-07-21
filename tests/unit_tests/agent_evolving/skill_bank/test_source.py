# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Effective-source materialization coverage for skill-bank initialization."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from openjiuwen.agent_evolving.skill_bank import (
    SkillBankStore,
    SkillSource,
    initialize_skill_bank,
)
from openjiuwen.harness.rails import SkillUseRail
from openjiuwen.harness.skill_sources import collect_disabled_skills_from_state


def _write_skill(root: Path, name: str, description: str, body: str) -> None:
    skill_dir = root / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {description}\n---\n\n{body}\n",
        encoding="utf-8",
    )
    (skill_dir / "reference.txt").write_text(body, encoding="utf-8")


class _LocalFs:
    async def read_file(self, path: str, **_kwargs):
        content = Path(path).read_text(encoding="utf-8")
        return SimpleNamespace(code=0, data=SimpleNamespace(content=content))


class _LocalOperation:
    @staticmethod
    def fs() -> _LocalFs:
        return _LocalFs()


@pytest.mark.asyncio
async def test_materialized_bank_matches_ordered_rail_view(tmp_path: Path) -> None:
    primary = tmp_path / "primary"
    secondary = tmp_path / "secondary"
    _write_skill(primary, "shared", "Primary shared", "primary body")
    _write_skill(primary, "alpha", "Alpha", "alpha body")
    (primary / "alpha" / ".git").mkdir()
    (primary / "alpha" / ".git" / "config").write_text("local metadata", encoding="utf-8")
    _write_skill(secondary, "shared", "Secondary shared", "secondary body")
    _write_skill(secondary, "beta", "Beta", "beta body")
    (secondary / "skills_state.json").write_text(
        json.dumps({"skill_configs": {"beta": {"enabled": False}}}),
        encoding="utf-8",
    )

    sources = [
        SkillSource(primary, source_kind="workspace"),
        SkillSource(secondary, source_kind="team", revision="team-rev"),
    ]
    store = SkillBankStore(tmp_path / "bank")
    active = initialize_skill_bank(store, sources)
    disabled = collect_disabled_skills_from_state([primary, secondary])
    operation = _LocalOperation()
    live_rail = SkillUseRail(
        skills_dir=[str(primary), str(secondary)],
        disabled_skills=disabled,
        include_tools=False,
    )
    bank_rail = SkillUseRail(
        skills_dir=str(active.skills_dir),
        disabled_skills=list(active.manifest.disabled_skills),
        include_tools=False,
    )
    live_rail.set_sys_operation(operation)  # type: ignore[arg-type]
    bank_rail.set_sys_operation(operation)  # type: ignore[arg-type]

    await live_rail.reload_skills()
    await bank_rail.reload_skills()

    assert [(skill.name, skill.description) for skill in live_rail.skills_meta] == [
        (skill.name, skill.description) for skill in bank_rail.skills_meta
    ]
    assert [skill.name for skill in bank_rail.skills_meta] == ["alpha", "shared"]
    assert store.resolve().version_id == active.version_id
    assert "primary body" in (active.skills_dir / "shared" / "SKILL.md").read_text(encoding="utf-8")
    assert "secondary body" not in (active.skills_dir / "shared" / "SKILL.md").read_text(encoding="utf-8")
    assert not (active.skills_dir / "alpha" / ".git").exists()
    assert active.manifest.disabled_skills == ("beta",)
    assert [(source.skill_name, source.source_kind) for source in active.manifest.sources] == [
        ("alpha", "workspace"),
        ("beta", "team"),
        ("shared", "workspace"),
    ]
    assert all(source.content_sha256 for source in active.manifest.sources)
    assert next(source for source in active.manifest.sources if source.skill_name == "beta").revision == "team-rev"

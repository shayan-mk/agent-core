# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Candidate construction coverage."""

from __future__ import annotations

from pathlib import Path

import pytest

from openjiuwen.agent_evolving.skill_bank import SkillBankStore
from openjiuwen.agent_evolving.skill_bank.candidate import SkillBankCandidateBuilder
from openjiuwen.agent_evolving.skill_bank.models import (
    SkillBankProposal,
    SkillOperation,
    SkillOperationType,
)
from openjiuwen.core.common.exception.errors import BaseError


def _write_skill(root: Path, name: str, body: str) -> Path:
    package = root / name
    package.mkdir(parents=True, exist_ok=True)
    (package / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: Test skill\n---\n\n{body}\n",
        encoding="utf-8",
    )
    return package


def test_candidate_builder_applies_repo_operations_without_mutating_sources(tmp_path: Path) -> None:
    parent_root = tmp_path / "parent"
    _write_skill(parent_root, "modify-me", "old")
    _write_skill(parent_root, "delete-me", "delete")
    replacement = _write_skill(tmp_path / "inputs", "replacement", "new")
    addition = _write_skill(tmp_path / "inputs", "addition", "added")
    store = SkillBankStore(tmp_path / "bank")
    parent = store.create_snapshot(parent_root)
    proposal = SkillBankProposal(
        proposal_id="package-ops",
        parent_version_id=parent.version_id,
        operations=(
            SkillOperation(SkillOperationType.MODIFY, "modify-me", source_path=str(replacement)),
            SkillOperation(SkillOperationType.DELETE, "delete-me"),
            SkillOperation(SkillOperationType.ADD, "added-skill", source_path=str(addition)),
        ),
    )
    result = SkillBankCandidateBuilder(store).build(proposal)

    assert result.proposal.candidate_version_id == result.candidate.version_id
    assert result.candidate.manifest.parent_version_id == parent.version_id
    assert result.candidate.manifest.proposal_id == proposal.proposal_id
    assert result.candidate.manifest.skill_names == ("added-skill", "modify-me")
    assert "new" in (result.candidate.skills_dir / "modify-me" / "SKILL.md").read_text(encoding="utf-8")
    assert not (result.candidate.skills_dir / "delete-me").exists()
    assert "old" in (parent_root / "modify-me" / "SKILL.md").read_text(encoding="utf-8")
    assert (parent_root / "delete-me").is_dir()


def test_candidate_builder_enforces_bank_capacity(tmp_path: Path) -> None:
    parent_root = tmp_path / "parent"
    _write_skill(parent_root, "first", "one")
    _write_skill(parent_root, "second", "two")
    addition = _write_skill(tmp_path / "inputs", "third", "three")
    store = SkillBankStore(tmp_path / "bank")
    parent = store.create_snapshot(parent_root)
    proposal = SkillBankProposal(
        proposal_id="over-capacity",
        parent_version_id=parent.version_id,
        operations=(
            SkillOperation(
                SkillOperationType.ADD,
                "third",
                source_path=str(addition),
            ),
        ),
    )

    with pytest.raises(BaseError, match="maximum is 2"):
        SkillBankCandidateBuilder(store, max_bank_skills=2).build(proposal)

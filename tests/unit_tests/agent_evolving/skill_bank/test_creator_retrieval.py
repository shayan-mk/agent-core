# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Tests for package creation, scheduling, and retrieval."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from openjiuwen.agent_evolving.skill_bank import (
    ProposalStatus,
    SkillBankAdoptionCycle,
    SkillBankStore,
)
from openjiuwen.agent_evolving.skill_bank.candidate import SkillBankCandidateBuilder, SkillBankPackageCreator
from openjiuwen.agent_evolving.skill_bank.models import SkillOperationType
from openjiuwen.agent_evolving.skill_bank.reservoir import ReservoirEntry
from openjiuwen.agent_evolving.skill_bank.retrieval import Bm25SkillRetriever


def _write_skill(
    root: Path,
    name: str,
    description: str,
    body: str = "",
    *,
    scope: str | None = None,
) -> Path:
    skill_dir = root / name
    skill_dir.mkdir(parents=True)
    scope_line = f"scope: {scope}\n" if scope else ""
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {description}\n{scope_line}---\n\n{body}",
        encoding="utf-8",
    )
    return skill_dir


def _entry(success: bool, *, task: str = "task-1") -> ReservoirEntry:
    return ReservoirEntry(
        version_id="bank_000001",
        origin_task_id=f"origin-{task}",
        task_id=task,
        rollout_id=None,
        reward=1.0 if success else 0.0,
        success=success,
        payload={"tool_call_chain": "read_file -> bash", "final_response": "done"},
    )


def _store_with_baseline(tmp_path: Path) -> tuple[SkillBankStore, "object"]:
    store = SkillBankStore(tmp_path / "bank")
    bank_dir = tmp_path / "baseline"
    calculator = _write_skill(bank_dir, "calculator", "Calculate arithmetic expressions")
    (calculator / "references").mkdir()
    (calculator / "references" / "examples.md").write_text(
        "Existing calculator examples", encoding="utf-8"
    )
    baseline = store.create_snapshot(bank_dir)
    store.promote(baseline.version_id)
    return store, baseline


@pytest.mark.asyncio
async def test_creator_builds_candidate_from_llm_operations(tmp_path: Path, monkeypatch) -> None:
    store, baseline = _store_with_baseline(tmp_path)
    creator = SkillBankPackageCreator(SkillBankCandidateBuilder(store), MagicMock(), "mock-model", language="en")

    async def fake_invoke(llm, model, prompt, *, policy, **kwargs):
        assert "Calculate arithmetic expressions" in prompt
        assert '"version_id": "bank_000001"' in prompt
        return json.dumps({
            "assertion": "A csv-parsing skill prevents the observed format failures",
            "operations": [
                {
                    "action": "add",
                    "skill_name": "csv_parser",
                    "skill_md": (
                        "---\nname: csv_parser\ndescription: Parse CSV files\nscope: task\n"
                        "when_to_use: When a task contains CSV data\ntrigger_type: general\n---\n\nUse pandas."
                    ),
                    "files": {"scripts/parse.py": "print('parse')\n"},
                },
                {
                    "action": "modify",
                    "skill_name": "calculator",
                    "skill_md": (
                        "---\nname: calculator\ndescription: Calculate reliably\nscope: task\n"
                        "when_to_use: When arithmetic is required\ntrigger_type: general\n---\n"
                    ),
                },
            ],
        })

    monkeypatch.setattr(
        "openjiuwen.agent_evolving.skill_bank.candidate.invoke_text_with_retry", fake_invoke
    )
    result = await creator.create(
        baseline, [_entry(True, task="t1"), _entry(False, task="t2")], proposal_id="prop-1"
    )

    assert result is not None
    assert result.proposal.status is ProposalStatus.PENDING
    assert result.proposal.source_trajectory_ids == ("t1", "t2")
    add_operation, modify_operation = result.proposal.operations
    assert add_operation.action is SkillOperationType.ADD
    assert modify_operation.action is SkillOperationType.MODIFY
    assert "csv-parsing" in add_operation.summary
    assert all(
        operation.provenance.source_kind == "skill-bank-creator"
        for operation in result.proposal.operations
    )
    assert sorted(result.candidate.manifest.skill_names) == ["calculator", "csv_parser"]
    assert (result.candidate.skills_dir / "csv_parser" / "scripts" / "parse.py").is_file()
    assert (
        result.candidate.skills_dir / "calculator" / "references" / "examples.md"
    ).read_text(encoding="utf-8") == "Existing calculator examples"


def test_idle_cycle_requests_a_candidate_on_schedule(tmp_path: Path) -> None:
    store, baseline = _store_with_baseline(tmp_path)
    supplied = []
    cycle = SkillBankAdoptionCycle(
        store, steps_per_cycle=2,
        candidate_creator=lambda ref, entries: supplied.append(ref.version_id),
    )
    assert cycle.on_train_step(1) is None and supplied == []
    assert cycle.on_train_step(2) is None and supplied == [baseline.version_id]


def test_bm25_retriever_ranks_and_limits(tmp_path: Path) -> None:
    bank_dir = tmp_path / "skills"
    _write_skill(bank_dir, "general_help", "General task guidance", scope="general")
    _write_skill(
        bank_dir, "csv_parser", "Parse CSV spreadsheet files", "Handles csv columns."
    )
    _write_skill(
        bank_dir, "web_search", "Search the public web", "Queries search engines.", scope="task"
    )
    retriever = Bm25SkillRetriever(bank_dir)

    assert retriever.top_skills("parse csv", k=1) == ["general_help", "csv_parser"]
    assert retriever.top_skills("") == ["general_help"]
    assert retriever.top_skills("quantum chromodynamics") == ["general_help"]

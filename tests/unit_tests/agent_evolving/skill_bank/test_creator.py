# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Focused coverage for the ReSkill creator and deterministic runtime gates."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from openjiuwen.agent_evolving.skill_bank import (
    ProposalStatus,
    SkillBankCreatorPipeline,
    SkillBankProposal,
    SkillBankStore,
)
from openjiuwen.agent_evolving.skill_bank.assertions import (
    AssertionKind,
    TrajectoryAssertion,
    grade_assertions,
)
from openjiuwen.agent_evolving.skill_bank.models import (
    SkillOperation,
    SkillOperationType,
)
from openjiuwen.agent_evolving.skill_bank.reservoir import ReservoirEntry
from openjiuwen.agent_evolving.skill_bank.triggers import (
    SkillTriggerSelector,
    validate_skill_trigger,
)


def _skill(root: Path, name: str, *, trigger_type: str = "general", pattern: str = "") -> Path:
    package = root / name
    package.mkdir(parents=True)
    pattern_line = f"trigger_pattern: '{pattern}'\n" if pattern else ""
    (package / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {name}\nscope: task\n"
        f"when_to_use: Use {name} when applicable\ntrigger_type: {trigger_type}\n{pattern_line}---\n",
        encoding="utf-8",
    )
    return package


def _entry(success: bool, *, version: str = "bank_000001") -> ReservoirEntry:
    return ReservoirEntry(
        version_id=version,
        origin_task_id="origin-1",
        task_id="task-1",
        rollout_id=None,
        reward=float(success),
        success=success,
        task_key="shared-task",
        payload={
            "turns": [
                {"action": [{"name": "search", "arguments": "x"}], "observation": "result"},
                {"action": [{"name": "answer", "arguments": "y"}], "observation": "done"},
            ]
        },
    )


def test_assertions_and_skill_triggers_are_deterministic(tmp_path: Path) -> None:
    entries = [_entry(True), _entry(False)]
    assertion = TrajectoryAssertion(
        name="search-before-answer",
        kind=AssertionKind.ACTION_ORDER,
        pattern="search",
        second_pattern="answer",
    )
    grade = grade_assertions([assertion], entries)[0]
    assert (grade.passed, grade.total, grade.pass_rate) == (2, 2, 1.0)

    action_skill = _skill(tmp_path, "after_search", trigger_type="action_pattern", pattern="search")
    beginning_skill = _skill(tmp_path, "plan_first", trigger_type="beginning")
    general_skill = _skill(tmp_path, "always")
    assert (
        validate_skill_trigger("after_search", (action_skill / "SKILL.md").read_text(encoding="utf-8"), entries) is None
    )
    missed_trigger = (action_skill / "SKILL.md").read_text(encoding="utf-8").replace(
        "trigger_pattern: 'search'", "trigger_pattern: 'never-fired'"
    )
    issue = validate_skill_trigger("after_search", missed_trigger, entries)
    assert issue is not None and "0.0%" in issue

    selector = SkillTriggerSelector()
    skills = [
        SimpleNamespace(name=path.name, directory=path) for path in (action_skill, beginning_skill, general_skill)
    ]
    first_ctx = SimpleNamespace(inputs=SimpleNamespace(messages=[]))
    later_ctx = SimpleNamespace(
        inputs=SimpleNamespace(messages=[{"role": "assistant", "tool_calls": [{"name": "search"}]}])
    )
    assert [skill.name for skill in selector(skills, first_ctx)] == ["plan_first", "always"]
    assert [skill.name for skill in selector(skills, later_ctx)] == ["after_search", "always"]


@pytest.mark.asyncio
async def test_creator_pipeline_uses_grouped_evidence_assertions_and_history(tmp_path: Path, monkeypatch) -> None:
    store = SkillBankStore(tmp_path / "bank")
    source = tmp_path / "source"
    _skill(source, "search")
    baseline = store.create_snapshot(source)
    store.promote(baseline.version_id)
    previous = SkillBankProposal(
        proposal_id="previous-proposal",
        parent_version_id=baseline.version_id,
        operations=(
            SkillOperation(
                action=SkillOperationType.MODIFY,
                skill_name="search",
                summary="previous attempt",
            ),
        ),
    )
    store.record_proposal(previous)
    store.decide_proposal(previous.proposal_id, ProposalStatus.REJECTED, reason="did not improve reward")

    async def fake_invoke(llm, model, prompt, *, policy, **kwargs):
        if prompt.startswith("Compare"):
            assert prompt.count('"outcome"') == 2
            return json.dumps({"insight": "search before answering", "failure_mode": "premature answer"})
        if prompt.startswith("Maintain"):
            return json.dumps(
                {
                    "assertion_operations": [
                        {
                            "op": "add",
                            "assertion": {
                                "name": "uses-search",
                                "kind": "action_matches",
                                "pattern": "search",
                            },
                        }
                    ],
                    "diagnosis": "Some failures answer before checking evidence.",
                    "insight_groups": [{"label": "premature", "insight_indices": [0]}],
                }
            )
        assert "previous-proposal" in prompt
        assert '"name": "uses-search"' in prompt
        assert "premature" in prompt
        return json.dumps(
            {
                "assertion": "A verification skill prevents premature answers",
                "operations": [
                    {
                        "action": "add",
                        "skill_name": "verify_answer",
                        "skill_md": (
                            "---\nname: verify_answer\ndescription: Verify answers\nscope: general\n"
                            "when_to_use: Before returning a final answer\ntrigger_type: general\n---\n\n"
                            "Check the evidence before answering."
                        ),
                    }
                ],
            }
        )

    monkeypatch.setattr("openjiuwen.agent_evolving.skill_bank.creator.invoke_text_with_retry", fake_invoke)
    monkeypatch.setattr("openjiuwen.agent_evolving.skill_bank.candidate.invoke_text_with_retry", fake_invoke)
    pipeline = SkillBankCreatorPipeline(tmp_path / "bank", MagicMock(), "creator-model", language="en")
    proposal_id = await pipeline(baseline, [_entry(True), _entry(False)])

    proposal = store.get_proposal(proposal_id)
    assert proposal.status is ProposalStatus.PENDING
    assert proposal.candidate_version_id is not None
    assert [assertion.name for assertion in pipeline.assertions] == ["uses-search"]
    candidate = store.resolve(proposal.candidate_version_id)
    assert sorted(candidate.manifest.skill_names) == ["search", "verify_answer"]

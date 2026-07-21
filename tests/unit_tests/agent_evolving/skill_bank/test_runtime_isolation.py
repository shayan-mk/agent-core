# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Concurrent offline-runtime isolation coverage for immutable skill banks."""

# pylint: disable=protected-access

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from openjiuwen.agent_evolving.agent_rl.config.offline_config import SkillRLConfig
from openjiuwen.agent_evolving.agent_rl.offline.runtime.agent_factory import AgentFactory
from openjiuwen.agent_evolving.agent_rl.offline.runtime.collector import TrajectoryCollector
from openjiuwen.agent_evolving.agent_rl.offline.runtime.runtime_executor import RuntimeExecutor
from openjiuwen.agent_evolving.agent_rl.schemas import RLTask
from openjiuwen.agent_evolving.skill_bank import SkillBankStore
from openjiuwen.agent_evolving.trajectory import Trajectory
from openjiuwen.core.runner import Runner
from openjiuwen.harness.rails import SkillUseRail


def _write_bank(root: Path, body: str) -> Path:
    skill_dir = root / "shared-skill"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: shared-skill\ndescription: Shared test skill\n---\n\n" + body,
        encoding="utf-8",
    )
    return root


class _LocalFs:
    async def read_file(self, path: str, **_kwargs):
        content = Path(path).read_text(encoding="utf-8")
        return SimpleNamespace(code=0, data=SimpleNamespace(content=content))


class _LocalOperation:
    @staticmethod
    def fs() -> _LocalFs:
        return _LocalFs()


@pytest.mark.asyncio
async def test_offline_runtime_isolates_concurrent_bank_versions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline_source = _write_bank(tmp_path / "baseline", "baseline-only-content")
    candidate_source = _write_bank(tmp_path / "candidate", "candidate-only-content")
    store = SkillBankStore(tmp_path / "bank")
    baseline = store.create_snapshot(baseline_source)
    candidate = store.create_snapshot(candidate_source, parent_version_id=baseline.version_id)
    rails: dict[str, SkillUseRail] = {}
    tool_ids: dict[str, str] = {}
    operation_ids: dict[str, str] = {}
    workspaces: dict[str, Path] = {}
    observed: dict[str, tuple[str, str]] = {}
    factory = AgentFactory(
        system_prompt="test",
        tools=[],
        tool_names=[],
        temperature=0.0,
        max_new_tokens=128,
        top_p=1.0,
        presence_penalty=0.0,
        frequency_penalty=0.0,
        skill_bank_config=SkillRLConfig(
            bank_root=str(tmp_path / "bank"),
            workspace=str(tmp_path / "workspaces"),
        ),
    )
    factory.proxy_url = "http://localhost:8000"

    async def collect_without_model(
        _collector: TrajectoryCollector,
        agent: Any,
        _inputs: dict,
        *,
        session_id: str,
        **_kwargs,
    ) -> Trajectory:
        rail = next(item for item in agent.configured_rails() if isinstance(item, SkillUseRail))
        agent.strip_rails_by_type((SkillUseRail,))
        await agent.register_rail(rail)
        rail.set_sys_operation(_LocalOperation())  # type: ignore[arg-type]
        await rail.reload_skills()
        rails[session_id] = rail
        workspaces[session_id] = Path(agent._runtime_owned_workspace)
        operation_ids[session_id] = agent._deep_config.sys_operation.id
        tool_card = agent.ability_manager.get("skill_tool")
        assert tool_card is not None
        tool_ids[session_id] = tool_card.id
        assert Runner.resource_mgr.get_tool(tool_card.id) is not None
        skill = rail.skills_meta[0]
        content = (skill.directory / "SKILL.md").read_text(encoding="utf-8")
        observed[session_id] = (str(skill.directory), content)
        return Trajectory(otlp_trace={"resourceSpans": []})

    monkeypatch.setattr(TrajectoryCollector, "collect", collect_without_model)
    executor = RuntimeExecutor(agent_factory=factory)
    tasks = [
        RLTask(
            task_id="baseline-task",
            origin_task_id="task",
            task_sample={},
            skill_bank_version=baseline.version_id,
        ),
        RLTask(
            task_id="candidate-task",
            origin_task_id="task",
            task_sample={},
            skill_bank_version=candidate.version_id,
        ),
    ]
    messages = await asyncio.gather(*(executor.execute_async(task) for task in tasks))

    assert all(message.global_reward is None for message in messages)
    assert observed["baseline-task"][0] != observed["candidate-task"][0]
    assert "baseline-only-content" in observed["baseline-task"][1]
    assert "candidate-only-content" in observed["candidate-task"][1]
    assert rails["baseline-task"] is not rails["candidate-task"]
    assert not any(workspace.exists() for workspace in workspaces.values())
    assert all(Runner.resource_mgr.get_tool(tool_id) is None for tool_id in tool_ids.values())
    assert all(Runner.resource_mgr.get_sys_operation(operation_id) is None for operation_id in operation_ids.values())

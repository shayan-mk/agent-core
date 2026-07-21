# -*- coding: utf-8 -*-
"""Unit tests for RuntimeExecutor (offline runtime executor behavior)."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from openjiuwen.agent_evolving.agent_rl.offline.runtime.runtime_executor import RuntimeExecutor
from openjiuwen.agent_evolving.agent_rl.schemas import RLTask
from openjiuwen.core.runner import Runner
from openjiuwen.core.sys_operation import LocalWorkConfig, OperationMode, SysOperationCard


@pytest.fixture
def sample_task():
    return RLTask(task_id="t1", origin_task_id="o1", task_sample={}, round_num=0)


@pytest.mark.asyncio
async def test_execute_async_neither_agent_factory_returns_empty_rollout(sample_task):
    executor = RuntimeExecutor()
    result = await executor.execute_async(sample_task)
    assert result.rollout_info == []
    assert result.reward_list == []
    assert result.turn_count == 0
    assert result.task_id == sample_task.task_id
    assert result.origin_task_id == sample_task.origin_task_id


@pytest.mark.asyncio
async def test_execute_async_agent_factory_exception_returns_empty_rollout(sample_task):
    def agent_factory(_: RLTask):
        raise ValueError("fail")

    executor = RuntimeExecutor(agent_factory=agent_factory)
    result = await executor.execute_async(sample_task)
    assert result is not None
    assert result.rollout_info == []
    assert result.reward_list == []


@pytest.mark.asyncio
async def test_execute_async_keeps_skill_bank_version_on_failure():
    task = RLTask(
        task_id="t1", origin_task_id="o1", task_sample={}, round_num=0,
        skill_bank_version="bank_000001",
    )

    def agent_factory(_: RLTask):
        raise ValueError("fail")

    executor = RuntimeExecutor(agent_factory=agent_factory)
    result = await executor.execute_async(task)
    assert result.skill_bank_version == "bank_000001"
    assert result.rollout_info == []


@pytest.mark.asyncio
async def test_execute_async_tears_down_agent_after_collection_failure(sample_task, monkeypatch, tmp_path: Path):
    from openjiuwen.agent_evolving.agent_rl.offline.runtime.collector import TrajectoryCollector

    owner_id = "rollout-owner"
    operation_id = "rollout-operation"
    card = SysOperationCard(
        id=operation_id,
        mode=OperationMode.LOCAL,
        work_config=LocalWorkConfig(shell_allowlist=[]),
    )
    Runner.resource_mgr.add_sys_operation(card, tag=owner_id)
    operation = Runner.resource_mgr.get_sys_operation(operation_id)
    rail = object()
    agent = MagicMock()
    agent.configured_rails.return_value = [rail]
    agent.unregister_rail = AsyncMock()
    agent.ability_manager.teardown_tools = MagicMock()
    agent._deep_config = SimpleNamespace(sys_operation=operation)
    agent.card = SimpleNamespace(id=owner_id)
    agent._runtime_owned_agent = True
    owned_workspace = tmp_path / "rollout-workspace"
    owned_workspace.mkdir()
    setattr(agent, "_runtime_owned_workspace", owned_workspace)
    monkeypatch.setattr(
        TrajectoryCollector,
        "collect",
        AsyncMock(side_effect=RuntimeError("collection failed")),
    )
    executor = RuntimeExecutor(agent_factory=lambda _task: agent)

    try:
        result = await executor.execute_async(sample_task)

        assert result.rollout_info == []
        agent.unregister_rail.assert_awaited_once_with(rail)
        agent.ability_manager.teardown_tools.assert_called_once_with()
        assert Runner.resource_mgr.get_sys_operation(operation_id) is None
        assert not owned_workspace.exists()
    finally:
        if Runner.resource_mgr.get_sys_operation(operation_id) is not None:
            Runner.resource_mgr.remove_sys_operation(operation_id, tag=owner_id)

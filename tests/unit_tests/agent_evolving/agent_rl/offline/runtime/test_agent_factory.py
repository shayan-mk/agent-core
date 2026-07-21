# -*- coding: utf-8 -*-
"""Unit tests for AgentFactory and build_agent_factory."""

import shutil
from pathlib import Path

import pytest

from openjiuwen.agent_evolving.agent_rl.config.offline_config import AgentRuntimeConfig, SkillRLConfig
from openjiuwen.agent_evolving.agent_rl.schemas import RLTask
from openjiuwen.agent_evolving.agent_rl.offline.runtime.agent_factory import (
    AgentFactory,
    build_agent_factory,
)
from openjiuwen.agent_evolving.skill_bank import SkillBankStore
from openjiuwen.core.runner import Runner
from openjiuwen.harness.rails import LLMRetryRail, SecurityRail, SkillUseRail


def _remove_rollout_resources(agent) -> None:
    operation = agent._deep_config.sys_operation
    if operation is not None and Runner.resource_mgr.get_sys_operation(operation.id) is operation:
        Runner.resource_mgr.remove_sys_operation(operation.id, tag=agent.card.id)
    workspace = getattr(agent, "_runtime_owned_workspace", None)
    if workspace is not None:
        shutil.rmtree(workspace, ignore_errors=True)


class TestBuildAgentFactory:
    @staticmethod
    def test_build_agent_factory_returns_factory():
        cfg = AgentRuntimeConfig(
            system_prompt="You are a helpful assistant.",
            temperature=0.7,
            top_p=0.9,
            max_new_tokens=512,
        )
        factory = build_agent_factory(cfg, tools=[], tool_names=[])
        assert isinstance(factory, AgentFactory)
        assert factory.proxy_url is None


class TestAgentFactoryCallWithoutProxyUrl:
    @staticmethod
    def test_call_without_proxy_url_raises():
        factory = AgentFactory(
            system_prompt="test",
            tools=[],
            tool_names=[],
            temperature=0.7,
            max_new_tokens=128,
            top_p=0.9,
            presence_penalty=0.0,
            frequency_penalty=0.0,
        )
        task = RLTask(task_id="t1", origin_task_id="o1", task_sample={})
        with pytest.raises(Exception) as exc_info:
            factory(task)
        assert "proxy_url" in str(exc_info.value).lower() or "proxy" in str(exc_info.value).lower()


def test_factory_mounts_assigned_skill_bank_version(tmp_path: Path) -> None:
    skill_dir = tmp_path / "source" / "calculator"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: calculator\ndescription: Calculate\n---\n",
        encoding="utf-8",
    )
    store = SkillBankStore(tmp_path / "bank")
    version = store.create_snapshot(tmp_path / "source")
    factory = AgentFactory(
        system_prompt="test",
        tools=[],
        tool_names=[],
        temperature=0.7,
        max_new_tokens=128,
        top_p=0.9,
        presence_penalty=0.0,
        frequency_penalty=0.0,
        skill_bank_config=SkillRLConfig(
            bank_root=str(tmp_path / "bank"),
            workspace=str(tmp_path / "workspace"),
        ),
    )
    factory.proxy_url = "http://localhost:8000"
    bank_task = RLTask(
        task_id="with-bank",
        origin_task_id="o1",
        task_sample={},
        skill_bank_version=version.version_id,
    )

    bank_agent = factory(bank_task)
    try:
        bank_rails = bank_agent.configured_rails()
        skill_rails = [rail for rail in bank_rails if isinstance(rail, SkillUseRail)]
        assert len(skill_rails) == 1
        assert bank_agent._runtime_owned_agent is True
        assert skill_rails[0].skills_dir == [str(version.skills_dir)]
        assert skill_rails[0].evolution_store is None
        assert skill_rails[0].enabled_skills == set()
        assert any(isinstance(rail, SecurityRail) for rail in bank_rails)
        assert any(isinstance(rail, LLMRetryRail) for rail in bank_rails)
    finally:
        _remove_rollout_resources(bank_agent)

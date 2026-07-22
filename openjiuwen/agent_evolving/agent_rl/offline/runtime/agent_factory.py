# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

"""
Built-in agent factory builder.

Constructs a default agent factory callable from RLConfig.runtime
parameters and registered tools, so users don't need to write one
for standard use cases.
"""

import shutil
import uuid
from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING, Any, List

from openjiuwen.core.common.exception.codes import StatusCode
from openjiuwen.core.common.exception.errors import build_error
from openjiuwen.agent_evolving.agent_rl.config.offline_config import AgentRuntimeConfig, SkillRLConfig
from openjiuwen.agent_evolving.agent_rl.schemas import RLTask

if TYPE_CHECKING:
    from openjiuwen.agent_evolving.skill_bank.models import BankVersionRef


_MAX_CACHED_SKILL_BANK_VERSIONS = 2


class _StoreSkillBankVersionResolver:
    """Resolve immutable versions lazily in the process running the agent."""

    def __init__(self, bank_root: str | Path) -> None:
        self._bank_root = str(bank_root)
        self._store = None
        self._cache: dict[str, "BankVersionRef"] = {}

    def __call__(self, rl_task: RLTask) -> "BankVersionRef":
        version_id = rl_task.skill_bank_version
        if version_id is None:
            raise build_error(
                StatusCode.TOOLCHAIN_EVOLVING_SKILL_BANK_PARAM_ERROR,
                error_msg="skill-bank RL task is missing its assigned version",
            )
        cached = self._cache.get(version_id)
        if cached is not None:
            return cached
        if self._store is None:
            from openjiuwen.agent_evolving.skill_bank.store import SkillBankStore

            self._store = SkillBankStore(self._bank_root)
        resolved = self._store.resolve(version_id, verify=True)
        if len(self._cache) >= _MAX_CACHED_SKILL_BANK_VERSIONS:
            self._cache.clear()
        self._cache[version_id] = resolved
        return resolved

    def __getstate__(self):
        return {"_bank_root": self._bank_root, "_store": None, "_cache": {}}


class AgentFactory:
    """Callable factory that creates DeepAgent instances for each RL task.

    ``proxy_url`` must be set (by MainTrainer) before the first call.
    """

    def __init__(
        self,
        system_prompt: str,
        tools: List[Any],
        tool_names: List[str],
        temperature: float,
        max_new_tokens: int,
        top_p: float,
        presence_penalty: float,
        frequency_penalty: float,
        skill_bank_config: SkillRLConfig | None = None,
    ) -> None:
        self._system_prompt = system_prompt
        self._tools = tools
        self._tool_names = tool_names
        self._temperature = temperature
        self._max_new_tokens = max_new_tokens
        self._top_p = top_p
        self._presence_penalty = presence_penalty
        self._frequency_penalty = frequency_penalty
        self._skill_bank_config = skill_bank_config
        self._skill_bank_version_resolver = (
            _StoreSkillBankVersionResolver(skill_bank_config.bank_root)
            if skill_bank_config is not None
            else None
        )
        self._skill_retrievers = {}
        self.proxy_url: str | None = None

    def __call__(self, rl_task: RLTask):
        """Create and configure a DeepAgent instance for the given RL task."""
        skill_bank_version = (
            self._skill_bank_version_resolver(rl_task)
            if self._skill_bank_version_resolver is not None
            else None
        )
        from openjiuwen.core.foundation.llm.model import Model
        from openjiuwen.core.foundation.llm.schema.config import (
            ModelClientConfig,
            ModelRequestConfig,
        )
        from openjiuwen.core.single_agent.schema.agent_card import AgentCard
        from openjiuwen.harness.deep_agent import DeepAgent
        from openjiuwen.harness.factory import apply_deep_agent_parts, resolve_deep_agent_parts
        from openjiuwen.harness.schema.config import DeepAgentConfig

        if not self.proxy_url:
            raise build_error(
                StatusCode.AGENT_RL_PROXY_NOT_INITIALIZED,
                error_msg="proxy_url has not been set on AgentFactory, "
                          "BackendProxy must be started before creating agents",
            )

        agent_card = AgentCard(
            id=(
                f"rl_agent_{rl_task.task_id}_{skill_bank_version.version_id}"
                if skill_bank_version is not None
                else f"rl_agent_{rl_task.task_id}"
            ),
            name="RLTrainingAgent",
            description="RL training agent based on DeepAgent",
        )

        client_config = ModelClientConfig(
            client_provider="OpenAI",
            api_key="EMPTY",
            api_base=f"{self.proxy_url}/v1",
            timeout=300,
            verify_ssl=False,  # Disable SSL verification for local vLLM proxy
        )

        request_config_kwargs: dict[str, Any] = {
            "model": "agentrl",
            "temperature": self._temperature,
            "top_p": self._top_p,
            "max_tokens": self._max_new_tokens,
        }
        if self._presence_penalty != 0.0:
            request_config_kwargs["presencePenalty"] = self._presence_penalty
        if self._frequency_penalty != 0.0:
            request_config_kwargs["frequencyPenalty"] = self._frequency_penalty
        request_config = ModelRequestConfig(**request_config_kwargs)

        model = Model(
            model_client_config=client_config,
            model_config=request_config,
        )

        agent = DeepAgent(card=agent_card)
        if self._skill_bank_version_resolver is None:
            config = DeepAgentConfig(
                model=model,
                card=agent_card,
                system_prompt=self._system_prompt,
                max_iterations=10,
                enable_task_loop=False,  # RL training requires single-round mode
            )
            agent.configure(config)
        else:
            rollout_workspace = self._resolve_rollout_workspace(skill_bank_version)
            rollout_workspace.mkdir(parents=True, exist_ok=False)
            operation = None
            try:
                skill_names = self._select_skill_names(skill_bank_version, rl_task)
                skill_roots = [skill_bank_version.skills_dir]
                operation = self._register_rollout_sys_operation(
                    agent_card.id,
                    [rollout_workspace, *skill_roots],
                )
                parts = resolve_deep_agent_parts(
                    model,
                    card=agent_card,
                    system_prompt=self._system_prompt,
                    max_iterations=10,
                    enable_task_loop=False,
                    workspace=rollout_workspace,
                    skills=skill_names,
                    skill_roots=skill_roots,
                    skill_selector=self._build_skill_selector(),
                    inline_selected_skills=self._skill_bank_config.triggered_loading,
                    sys_operation=operation,
                    restrict_to_work_dir=True,
                )
                apply_deep_agent_parts(agent, parts)
                setattr(agent, "_runtime_owned_workspace", rollout_workspace)
                setattr(agent, "_runtime_owned_agent", True)
            except Exception:
                if operation is not None:
                    from openjiuwen.core.runner import Runner

                    with suppress(Exception):
                        Runner.resource_mgr.remove_sys_operation(operation.id, tag=agent_card.id)
                with suppress(Exception):
                    agent.ability_manager.teardown_tools()
                shutil.rmtree(rollout_workspace, ignore_errors=True)
                raise

        # Enable token ID + logprobs for RL trajectory training
        agent.react_agent.config.llm_return_token_ids = True

        # Register tools on DeepAgent (will be shared with inner ReActAgent)
        if self._tools:
            self._register_tools(agent)

        return agent

    def _build_skill_selector(self):
        if self._skill_bank_config is None or not self._skill_bank_config.triggered_loading:
            return None
        from openjiuwen.agent_evolving.skill_bank.triggers import SkillTriggerSelector

        return SkillTriggerSelector()

    def _select_skill_names(
        self,
        version: "BankVersionRef",
        rl_task: RLTask,
    ) -> list[str] | None:
        if self._skill_bank_config is None or self._skill_bank_config.retrieval_top_k is None:
            return None

        from openjiuwen.agent_evolving.skill_bank.retrieval import Bm25SkillRetriever

        retriever = self._skill_retrievers.get(version.version_id)
        if retriever is None:
            retriever = Bm25SkillRetriever(version)
            if len(self._skill_retrievers) >= _MAX_CACHED_SKILL_BANK_VERSIONS:
                self._skill_retrievers.clear()
            self._skill_retrievers[version.version_id] = retriever
        return retriever.top_skills(
            str(rl_task.task_sample.get("query", "")),
            k=self._skill_bank_config.retrieval_top_k,
        )

    def _resolve_rollout_workspace(
        self,
        skill_bank_version: "BankVersionRef",
    ) -> Path:
        """Return a fresh workspace below the configured root."""
        assert self._skill_bank_config is not None
        return (
            Path(self._skill_bank_config.workspace).expanduser().resolve()
            / f"{skill_bank_version.version_id}_{uuid.uuid4().hex}"
        )

    @staticmethod
    def _register_rollout_sys_operation(agent_id: str, sandbox_roots: list[Path]):
        """Register the rollout-owned filesystem operation."""
        from openjiuwen.core.runner import Runner
        from openjiuwen.core.sys_operation import LocalWorkConfig, OperationMode, SysOperationCard

        operation_id = f"RLTrainingAgent_{agent_id}_{uuid.uuid4().hex}"
        card = SysOperationCard(
            id=operation_id,
            mode=OperationMode.LOCAL,
            work_config=LocalWorkConfig(
                shell_allowlist=None,
                restrict_to_sandbox=True,
                sandbox_root=[str(root.resolve()) for root in sandbox_roots],
            ),
        )
        result = Runner.resource_mgr.add_sys_operation(card, tag=agent_id)
        if result.is_err():
            cause = result.msg()
            raise build_error(
                StatusCode.DEEPAGENT_RUNTIME_ERROR,
                error_msg=f"failed to register rollout SysOperation '{operation_id}': {cause}",
                cause=cause,
            )
        operation = Runner.resource_mgr.get_sys_operation(operation_id)
        if operation is None:
            raise build_error(
                StatusCode.DEEPAGENT_RUNTIME_ERROR,
                error_msg=f"failed to resolve rollout SysOperation '{operation_id}'",
            )
        return operation

    def _register_tools(self, agent) -> None:
        """Register tools on the agent's ability_manager and Runner.resource_mgr.

        ``ability_manager.add()`` only accepts ToolCard objects, but ``@tool``
        decorated functions are ``LocalFunction`` (a Tool subclass) instances
        that carry a ``.card`` attribute.  We must therefore:
        1. Add the ToolCard to ability_manager (so the LLM schema is available).
        2. Register the Tool instance with Runner.resource_mgr (so execution can
           retrieve it by id when the model calls the tool).

        NOTE: We always reconstruct a fresh LocalFunction instance in this process
        rather than using the cloudpickled one.  When AgentFactory is serialized
        by Ray (cloudpickle), LocalFunction.invoke is a closure whose globals
        include _TRANSFORM_NOOP (a plain object()).  cloudpickle bakes in the
        *value* of that object, creating a new object() on deserialization.
        The worker's callback framework also returns its own new object() from
        trigger_transform, making the `result is _TRANSFORM_NOOP` identity check
        always fail — so invoke() returns the sentinel instead of the real tool
        result.  Re-constructing LocalFunction here lets _ToolMeta.__call__ run
        fresh in this worker, capturing the worker's own _TRANSFORM_NOOP.
        """
        from openjiuwen.core.foundation.tool.base import Tool as FoundationTool
        from openjiuwen.core.foundation.tool import ToolCard
        from openjiuwen.core.foundation.tool.function.function import LocalFunction
        from openjiuwen.core.runner import Runner

        for t in self._tools:
            if isinstance(t, FoundationTool) and hasattr(t, 'card') and t.card:
                agent.ability_manager.add(t.card)
                if not Runner.resource_mgr.get_tool(tool_id=t.card.id):
                    func = getattr(t, '_func', None)
                    if func is not None:
                        t = LocalFunction(card=t.card, func=func)
                    Runner.resource_mgr.add_tool(t)
            elif isinstance(t, ToolCard):
                agent.ability_manager.add(t)
            else:
                from openjiuwen.core.common.logging import logger
                logger.warning(
                    "AgentFactory: unrecognized tool type %s, skipping.", type(t)
                )


def build_agent_factory(
    runtime_cfg: AgentRuntimeConfig,
    tools: List[Any],
    tool_names: List[str],
    *,
    skill_bank_config: SkillRLConfig | None = None,
) -> AgentFactory:
    """Build a default AgentFactory from runtime config + tools."""
    from openjiuwen.core.foundation.prompt import PromptTemplate

    system_prompt = runtime_cfg.system_prompt
    if isinstance(system_prompt, PromptTemplate):
        system_prompt = system_prompt.content

    return AgentFactory(
        system_prompt=system_prompt,
        tools=list(tools),
        tool_names=list(tool_names),
        temperature=runtime_cfg.temperature,
        max_new_tokens=runtime_cfg.max_new_tokens,
        top_p=runtime_cfg.top_p,
        presence_penalty=runtime_cfg.presence_penalty,
        frequency_penalty=runtime_cfg.frequency_penalty,
        skill_bank_config=skill_bank_config,
    )

# -*- coding: utf-8 -*-
"""Unit tests for TrainingCoordinator skill-bank version assignment and observation."""

import asyncio
import importlib.util
import sys
from unittest.mock import MagicMock

import pytest

pytest.importorskip("torch")
_stubbed_tensordict = (
    "tensordict" not in sys.modules
    and importlib.util.find_spec("tensordict") is None
)
if _stubbed_tensordict:
    sys.modules["tensordict"] = MagicMock()
    sys.modules["tensordict"].TensorDict = MagicMock()

from openjiuwen.agent_evolving.agent_rl.offline.coordinator.training_coordinator import (  # noqa: E402
    TrainingCoordinator,
)
from openjiuwen.agent_evolving.agent_rl.schemas import RolloutMessage  # noqa: E402
from openjiuwen.agent_evolving.skill_bank import SkillBankAdoptionCycle  # noqa: E402

if _stubbed_tensordict:
    sys.modules.pop("tensordict", None)


def _coordinator_config():
    return {
        "data": {
            "max_prompt_length": 32,
            "max_response_length": 16,
        },
        "trainer": {
            "runtime_parallel_num": 2,
        },
        "actor_rollout_ref": {
            "rollout": {"n": 4},
        },
        "JiuwenRL": {
            "whole_trajectory": False,
            "custom_fn": {
                "classifier": "default_classify_rollouts",
                "validator": "default_validate_stop",
                "sampler": "default_sampling",
            },
        },
    }


@pytest.fixture
def coordinator(mock_tokenizer):
    return TrainingCoordinator(config=_coordinator_config(), tokenizer=mock_tokenizer, persistence=None)


def test_build_initial_tasks_assigns_bank_version_to_each_rollout(coordinator):
    cycle = MagicMock()
    cycle.assign.side_effect = ["bank_000001", "bank_000002"] * 4
    cycle.task_key.side_effect = SkillBankAdoptionCycle.task_key
    coordinator.configure_skill_bank(cycle)
    tasks = coordinator._build_initial_tasks({"query": ["q1", "q2"]})

    assert len(tasks) == 8
    assert cycle.assign.call_count == 8
    versions = [task.skill_bank_version for task in tasks.values()]
    assert versions.count("bank_000001") == 4
    assert versions.count("bank_000002") == 4

def test_collect_round_observer_sees_every_message_before_filters(coordinator, monkeypatch):
    cycle = MagicMock()
    coordinator.configure_skill_bank(cycle)

    completed = RolloutMessage(
        task_id="t1", origin_task_id="o1", rollout_info=[], global_reward=1.0,
        skill_bank_version="bank_000001",
    )
    failed = RolloutMessage(
        task_id="t2", origin_task_id="o2", rollout_info=[], global_reward=None,
    )

    async def fake_wait(round_id, poll_interval=1):
        return {"t1": completed, "t2": failed}

    monkeypatch.setattr(coordinator, "_wait_for_tasks_completion", fake_wait)
    collected = asyncio.run(coordinator._collect_round_mdp(round_id=0))

    # Both messages reach the observer even though training filters drop both.
    assert [call.args[0] for call in cycle.observe.call_args_list] == [completed, failed]
    assert collected == {}

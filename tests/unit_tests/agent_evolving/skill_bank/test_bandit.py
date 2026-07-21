# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Tests for the reservoir, discounted bandit, and adoption cycle."""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from openjiuwen.agent_evolving.skill_bank.bandit import (
    DiscountedBetaArm,
    ThompsonAllocation,
    _AdaptiveMemoryEstimator,
)
from openjiuwen.agent_evolving.agent_rl.config.offline_config import (
    RLConfig,
    SkillRLConfig,
    TrainingConfig,
)
from openjiuwen.agent_evolving.agent_rl.optimizer.rl_optimizer import OfflineRLOptimizer
from openjiuwen.agent_evolving.agent_rl.offline.runtime.agent_factory import AgentFactory
from openjiuwen.agent_evolving.agent_rl.schemas import Rollout, RolloutMessage
from openjiuwen.agent_evolving.skill_bank import (
    ProposalStatus,
    SkillBankAdoptionCycle,
    SkillBankProposal,
    SkillBankStore,
)
from openjiuwen.agent_evolving.skill_bank.models import (
    SkillOperation,
    SkillOperationType,
)
from openjiuwen.agent_evolving.skill_bank.reservoir import ReservoirEntry, TrajectoryReservoir


def _write_bank(root: Path, body: str) -> Path:
    skill_dir = root / "calculator"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: calculator\ndescription: Calculate values\n---\n\n" + body,
        encoding="utf-8",
    )
    (root / "skills_state.json").write_text(
        json.dumps({"skill_configs": {"calculator": {"enabled": True}}}),
        encoding="utf-8",
    )
    return root


def _message(version_id: str | None, reward: float | None, *, turns: int = 1) -> RolloutMessage:
    return RolloutMessage(
        task_id="task-1",
        origin_task_id="origin-1",
        rollout_id="rollout-1",
        rollout_info=[
            Rollout(turn_id=i, output_response={"content": "final answer"})
            for i in range(turns)
        ],
        global_reward=reward,
        skill_bank_version=version_id,
    )


def _store_with_trial(tmp_path: Path) -> tuple[SkillBankStore, str, str, str]:
    """Create a store with a promoted baseline and a pending candidate proposal."""
    store = SkillBankStore(tmp_path / "bank")
    baseline = store.create_snapshot(_write_bank(tmp_path / "baseline", "v1"))
    store.promote(baseline.version_id)
    proposal = SkillBankProposal(
        proposal_id="prop-1",
        parent_version_id=baseline.version_id,
        operations=(SkillOperation(action=SkillOperationType.MODIFY, skill_name="calculator"),),
    )
    store.record_proposal(proposal)
    candidate = store.create_snapshot(
        _write_bank(tmp_path / "candidate", "v2"),
        parent_version_id=baseline.version_id,
        proposal_id="prop-1",
    )
    store.attach_candidate("prop-1", candidate.version_id)
    return store, baseline.version_id, candidate.version_id, "prop-1"


def test_discounted_beta_arm_applies_reskill_update() -> None:
    arm = DiscountedBetaArm(memory=200.0)
    arm.update(successes=3, observations=4)
    weight = 1.0 / (1.0 + 4 / 200.0)
    assert arm.alpha == pytest.approx(weight * 1.0 + 3)
    assert arm.beta == pytest.approx(weight * 1.0 + 1)
    assert arm.mean() == pytest.approx(arm.alpha / (arm.alpha + arm.beta))
    arm.reset()
    assert (arm.alpha, arm.beta) == (1.0, 1.0)


def test_adaptive_memory_starts_unweighted_then_fits_completed_tests() -> None:
    estimator = _AdaptiveMemoryEstimator()
    assert estimator.add_test([(3, 4), (2, 4)], [(1, 4), (3, 4)]) == float("inf")
    assert estimator.add_test([(2, 4), (1, 4)], [(3, 4), (4, 4)]) == 2000.0


def test_thompson_allocation_prefers_better_arm_with_exploration_floor() -> None:
    baseline_arm = DiscountedBetaArm()
    candidate_arm = DiscountedBetaArm()
    baseline_arm.update(successes=2, observations=100)
    candidate_arm.update(successes=98, observations=100)
    allocation = ThompsonAllocation(baseline_arm, candidate_arm, floor=0.15)
    rng = random.Random(0)
    probability = allocation.candidate_probability(rng)
    assert probability == pytest.approx(0.85)


def test_reservoir_bounds_entries_and_keeps_exact_cycle_counts() -> None:
    reservoir = TrajectoryReservoir(capacity=3)
    for index in range(10):
        reservoir.add(
            ReservoirEntry(
                version_id="bank_000001",
                origin_task_id=f"origin-{index}",
                task_id=f"task-{index}",
                rollout_id=None,
                reward=1.0 if index % 2 == 0 else 0.0,
                success=index % 2 == 0,
            )
        )
    assert len(reservoir.entries()) == 3
    assert reservoir.cycle_counts("bank_000001") == (5, 10)
    reservoir.reset_cycle()
    assert reservoir.cycle_counts("bank_000001") == (0, 0)
    assert len(reservoir.entries()) == 3

    recent = TrajectoryReservoir(capacity=10)
    first = ReservoirEntry("bank_000001", "origin", "old", None, 0.0, False, task_key="stable")
    latest = ReservoirEntry("bank_000001", "origin", "new", None, 0.2, False, task_key="stable")
    recent.add(first)
    recent.add(latest)
    assert recent.entries() == (latest,)


def test_offline_optimizer_builds_version_aware_factory_for_skill_rl(tmp_path: Path) -> None:
    config = RLConfig(
        training=TrainingConfig(whole_trajectory=True),
        skill_rl=SkillRLConfig(
            bank_root=str(tmp_path / "bank"),
            workspace=str(tmp_path / "workspaces"),
        ),
    )

    factory = OfflineRLOptimizer(config)._resolve_agent_factory()

    assert isinstance(factory, AgentFactory)


def test_adoption_cycle_accepts_better_candidate_and_promotes(tmp_path: Path) -> None:
    store, baseline_id, candidate_id, proposal_id = _store_with_trial(tmp_path)
    cycle = SkillBankAdoptionCycle(
        store,
        steps_per_cycle=2,
        min_observations_per_arm=2,
        candidate_creator=lambda _parent, _entries: None,
        seed=1,
    )
    cycle.begin_trial(proposal_id)

    cycle.observe(_message(baseline_id, 0.0))
    cycle.observe(_message(candidate_id, 1.0))
    assert cycle.on_train_step(step=1) is None
    assert cycle.reservoir.entries()[0].payload["turns"][0]["response"] == "final answer"
    assert store.get_proposal(proposal_id).status is ProposalStatus.PENDING

    cycle.observe(_message(baseline_id, 0.0))
    cycle.observe(_message(candidate_id, 1.0))
    checkpointed = []
    record = cycle.on_train_step(
        step=2,
        before_decision=lambda: checkpointed.append(store.resolve().version_id),
    )

    assert record is not None and record.status is ProposalStatus.ACCEPTED
    assert checkpointed == [baseline_id]
    assert store.resolve().version_id == candidate_id
    decided = store.get_proposal(proposal_id)
    assert decided.status is ProposalStatus.ACCEPTED
    assert len(decided.evidence) == 4
    assert cycle.baseline.version_id == candidate_id and cycle.candidate is None


def test_adoption_cycle_skips_incomparable_and_rejects_worse_candidate(tmp_path: Path) -> None:
    store, baseline_id, candidate_id, proposal_id = _store_with_trial(tmp_path)
    cycle = SkillBankAdoptionCycle(
        store,
        min_observations_per_arm=2,
        candidate_creator=lambda _parent, _entries: None,
        seed=1,
    )
    cycle.begin_trial(proposal_id)

    # Executor failure (no rollouts) and reward failure (None) are not evidence.
    cycle.observe(_message(baseline_id, 0.0, turns=0))
    cycle.observe(_message(candidate_id, None))
    cycle.observe(_message("bank_999999", 1.0))
    # Underexposed boundary leaves the proposal pending.
    cycle.observe(_message(baseline_id, 1.0))
    cycle.observe(_message(candidate_id, 0.0))
    assert cycle.on_train_step(step=1) is None
    assert store.get_proposal(proposal_id).status is ProposalStatus.PENDING

    cycle.observe(_message(baseline_id, 1.0))
    cycle.observe(_message(candidate_id, 0.0))
    record = cycle.on_train_step(step=2)

    assert record is not None and record.status is ProposalStatus.REJECTED
    assert store.resolve().version_id == baseline_id
    assert store.get_proposal(proposal_id).status is ProposalStatus.REJECTED
    assert cycle.baseline.version_id == baseline_id and cycle.candidate is None

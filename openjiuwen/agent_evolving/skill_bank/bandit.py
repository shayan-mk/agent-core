# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Discounted old/new bank allocation and adoption cycle control."""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import math
import random
from typing import Any, Awaitable, Callable, Sequence

from openjiuwen.agent_evolving.agent_rl.schemas import RolloutMessage
from openjiuwen.agent_evolving.skill_bank.models import (
    BankVersionRef,
    EvaluationEvidence,
    ProposalStatus,
    param_error as _param,
)
from openjiuwen.agent_evolving.skill_bank.reservoir import (
    ProposalCycleRecord,
    ReservoirEntry,
    TrajectoryReservoir,
)
from openjiuwen.agent_evolving.skill_bank.store import SkillBankStore
from openjiuwen.core.common.logging import logger


CandidateCreator = Callable[
    [BankVersionRef, Sequence[ReservoirEntry]],
    str | None | Awaitable[str | None],
]
_THOMPSON_SAMPLES = 10_000


def _compact_trajectory(message: RolloutMessage) -> dict:
    """Retain the action/observation sequence needed by the ReSkill creator."""
    turns = []
    for rollout in message.rollout_info:
        prompt = rollout.input_prompt or {}
        messages = prompt.get("message", [])
        response = rollout.output_response or {}
        turns.append(
            {
                "observation": messages[-1] if messages else None,
                "action": (
                    response.get("tool_calls")
                    if isinstance(response, dict)
                    else None
                ),
                "response": (
                    response.get("content")
                    if isinstance(response, dict)
                    else response
                ),
                "active_skills": (
                    list(rollout.active_skills)
                    if rollout.active_skills is not None
                    else None
                ),
            }
        )
    return {"turns": turns}


class DiscountedBetaArm:
    """ReSkill's discounted Beta posterior for one bank arm.

    Per policy step the update is ``alpha <- w*alpha + m`` and
    ``beta <- w*beta + (n - m)`` with ``w = (1 + n/M)^-1``, where ``m`` is the
    step's success count, ``n`` its observation count, and ``M`` the memory.
    """

    def __init__(self, *, memory: float = math.inf) -> None:
        if memory <= 0:
            raise _param("bandit memory must be positive")
        self._memory = memory
        self.alpha = 1.0
        self.beta = 1.0

    def reset(self) -> None:
        """Restore the per-cycle Beta(1, 1) prior."""
        self.alpha = 1.0
        self.beta = 1.0

    def set_memory(self, memory: float) -> None:
        if memory <= 0:
            raise _param("bandit memory must be positive")
        self._memory = memory

    def update(self, successes: int, observations: int) -> None:
        """Apply one discounted update from exact step counts."""
        if observations < 0 or successes < 0 or successes > observations:
            raise _param("bandit update requires 0 <= successes <= observations")
        if observations == 0:
            return
        weight = 1.0 if math.isinf(self._memory) else 1.0 / (1.0 + observations / self._memory)
        self.alpha = weight * self.alpha + successes
        self.beta = weight * self.beta + (observations - successes)

    def mean(self) -> float:
        """Posterior mean E[p]."""
        return self.alpha / (self.alpha + self.beta)

    def sample(self, rng: random.Random) -> float:
        return rng.betavariate(self.alpha, self.beta)


class _AdaptiveMemoryEstimator:
    """Estimate ReSkill's scalar memory from completed A/B test windows."""

    def __init__(self) -> None:
        self._tests: list[tuple[tuple[tuple[int, int], ...], tuple[tuple[int, int], ...]]] = []
        self.memory = math.inf

    def add_test(
        self,
        baseline_steps: Sequence[tuple[int, int]],
        candidate_steps: Sequence[tuple[int, int]],
    ) -> float:
        self._tests.append((tuple(baseline_steps), tuple(candidate_steps)))
        if len(self._tests) < 2:
            return self.memory
        self.memory = float(max(range(1, 2001), key=self._log_likelihood))
        return self.memory

    def _log_likelihood(self, memory: int) -> float:
        return sum(
            self._arm_log_likelihood(steps, memory)
            for test in self._tests
            for steps in test
        )

    @staticmethod
    def _arm_log_likelihood(steps: Sequence[tuple[int, int]], memory: int) -> float:
        alpha = 1.0
        beta = 1.0
        total = 0.0
        for successes, observations in steps:
            weight = memory / (memory + observations)
            discounted_alpha = weight * alpha
            discounted_beta = weight * beta
            failures = observations - successes
            total += (
                math.lgamma(observations + 1)
                - math.lgamma(successes + 1)
                - math.lgamma(failures + 1)
                + math.lgamma(successes + discounted_alpha)
                + math.lgamma(failures + discounted_beta)
                - math.lgamma(observations + discounted_alpha + discounted_beta)
                - math.lgamma(discounted_alpha)
                - math.lgamma(discounted_beta)
                + math.lgamma(discounted_alpha + discounted_beta)
            )
            alpha = discounted_alpha + successes
            beta = discounted_beta + failures
        return total


class ThompsonAllocation:
    """Per-rollout Thompson choice with ReSkill's 0.15 exploration floor per arm."""

    def __init__(
        self,
        baseline_arm: DiscountedBetaArm,
        candidate_arm: DiscountedBetaArm,
        *,
        floor: float = 0.15,
    ):
        if not 0.0 < floor < 0.5:
            raise _param("exploration floor must be in (0, 0.5)")
        self._baseline_arm = baseline_arm
        self._candidate_arm = candidate_arm
        self._floor = floor

    def candidate_probability(self, rng: random.Random) -> float:
        """Estimate and clip ``P(candidate > baseline)`` as in ReSkill."""
        wins = sum(
            self._candidate_arm.sample(rng) > self._baseline_arm.sample(rng)
            for _ in range(_THOMPSON_SAMPLES)
        )
        return min(1.0 - self._floor, max(self._floor, wins / _THOMPSON_SAMPLES))


class SkillBankAdoptionCycle:
    """Centralized reservoir, allocation, and adoption hook for offline RL.

    Wire ``assign`` and ``observe`` into ``TrainingCoordinator`` (via
    ``configure_skill_bank``) and call ``on_train_step`` at ``MainTrainer``'s
    rollout/train boundary. Rollout workers resolve each assigned version once
    and reuse the cached reference. Executor or reward failures are never
    counted as evidence; an undecidable cycle leaves the proposal pending.
    """

    def __init__(
        self,
        store: SkillBankStore,
        *,
        success_threshold: float = 1.0,
        steps_per_cycle: int = 1,
        min_version_episodes: int = 50,
        min_reservoir_size: int = 100,
        memory: float | None = None,
        reservoir_capacity: int = 200,
        candidate_creator: CandidateCreator,
        exploration_floor: float = 0.15,
        seed: int | None = None,
    ) -> None:
        if steps_per_cycle <= 0:
            raise _param("steps_per_cycle must be positive")
        if min_version_episodes <= 0:
            raise _param("min_version_episodes must be positive")
        if min_reservoir_size <= 0:
            raise _param("min_reservoir_size must be positive")
        self._store = store
        self._success_threshold = float(success_threshold)
        self._steps_per_cycle = steps_per_cycle
        self._min_version_episodes = min_version_episodes
        self._min_reservoir_size = min_reservoir_size
        self._candidate_creator = candidate_creator
        self._rng = random.Random(seed)

        self.reservoir = TrajectoryReservoir(reservoir_capacity)
        initial_memory = memory if memory is not None else math.inf
        self._baseline_arm = DiscountedBetaArm(memory=initial_memory)
        self._candidate_arm = DiscountedBetaArm(memory=initial_memory)
        self._memory_estimator = _AdaptiveMemoryEstimator() if memory is None else None
        self._thompson = ThompsonAllocation(
            self._baseline_arm,
            self._candidate_arm,
            floor=exploration_floor,
        )

        self._baseline: BankVersionRef = store.resolve(verify=True)
        self._candidate: BankVersionRef | None = None
        self._proposal_id: str | None = None
        self._cycle = 0
        self._steps_in_cycle = 0
        self._cycle_evidence: list[EvaluationEvidence] = []
        self._pending_entries: list[ReservoirEntry] = []
        self._pending_evidence: list[EvaluationEvidence] = []
        self._allocation_probability: float | None = None
        self._step_history: dict[str, list[tuple[int, int]]] = {}

    # -- trial lifecycle -----------------------------------------------------

    def begin_trial(self, proposal_id: str) -> BankVersionRef:
        """Start trialling a pending proposal's already-built candidate."""
        if self._proposal_id is not None:
            raise _param(f"a trial for proposal '{self._proposal_id}' is already running")
        # The store's ACTIVE pointer is the single authority on the baseline.
        self._baseline = self._store.resolve(verify=True)
        proposal = self._store.get_proposal(proposal_id)
        if proposal.status is not ProposalStatus.PENDING or not proposal.candidate_version_id:
            raise _param(f"proposal '{proposal_id}' is not pending with a candidate")
        if proposal.parent_version_id != self._baseline.version_id:
            raise _param(f"proposal '{proposal_id}' parent is not the current baseline")
        self._candidate = self._store.resolve(proposal.candidate_version_id, verify=True)
        self._proposal_id = proposal_id
        self._start_cycle()
        return self._candidate

    @property
    def baseline(self) -> BankVersionRef:
        return self._baseline

    @property
    def candidate(self) -> BankVersionRef | None:
        return self._candidate

    # -- coordinator callbacks -------------------------------------------------

    @staticmethod
    def task_key(task_sample: dict[str, Any]) -> str:
        """Return a stable reservoir key identifying a task sample."""
        for field in ("index", "data_id", "id"):
            value = task_sample.get(field)
            if value is not None:
                return str(value)
        identity = {
            key: task_sample[key]
            for key in ("query", "ground_truth")
            if key in task_sample
        }
        encoded = json.dumps(identity or task_sample, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def assign(self) -> str:
        """Assign the next rollout to the baseline or candidate bank."""
        if self._candidate is None:
            return self._baseline.version_id
        if self._allocation_probability is None:
            self._allocation_probability = self._thompson.candidate_probability(self._rng)
        assigned_candidate = self._rng.random() < self._allocation_probability
        if assigned_candidate:
            return self._candidate.version_id
        return self._baseline.version_id

    def observe(self, message: RolloutMessage) -> None:
        """Count one comparable observation per completed rollout."""
        version_id = message.skill_bank_version
        arm_ids = {self._baseline.version_id}
        if self._candidate is not None:
            arm_ids.add(self._candidate.version_id)
        if version_id not in arm_ids:
            return
        if not message.rollout_info or message.global_reward is None or not message.origin_task_id:
            # Executor or reward-pipeline failure: incomparable, never evidence.
            return
        reward = float(message.global_reward)
        success = reward >= self._success_threshold
        self._pending_entries.append(
            ReservoirEntry(
                version_id=version_id,
                origin_task_id=message.origin_task_id,
                task_id=message.task_id or "",
                rollout_id=message.rollout_id,
                reward=reward,
                success=success,
                task_key=message.skill_bank_task_key,
                payload=_compact_trajectory(message),
            )
        )
        if self._proposal_id is not None:
            self._pending_evidence.append(
                EvaluationEvidence(
                    task_id=message.task_id or message.origin_task_id or "",
                    version_id=version_id,
                    reward=reward,
                    success=success,
                    rollout_id=message.rollout_id,
                    metadata={"proposal_id": self._proposal_id, "cycle": self._cycle},
                )
            )

    # -- trainer hook ----------------------------------------------------------

    def on_train_step(
        self,
        step: int,
        *,
        before_decision: Callable[[], None] | None = None,
    ) -> ProposalCycleRecord | None:
        """Commit one successfully trained rollout step and advance the cycle."""
        step_counts: dict[str, tuple[int, int]] = {}
        for entry in self._pending_entries:
            self.reservoir.add(entry)
            successes, observations = step_counts.get(entry.version_id, (0, 0))
            step_counts[entry.version_id] = (successes + int(entry.success), observations + 1)
        self._pending_entries.clear()
        self._cycle_evidence.extend(self._pending_evidence)
        self._pending_evidence.clear()

        if self._candidate is not None:
            baseline_counts = step_counts.get(self._baseline.version_id, (0, 0))
            candidate_counts = step_counts.get(self._candidate.version_id, (0, 0))
            self._baseline_arm.update(*baseline_counts)
            self._candidate_arm.update(*candidate_counts)
            if self._memory_estimator is not None:
                self._step_history.setdefault(self._baseline.version_id, []).append(baseline_counts)
                self._step_history.setdefault(self._candidate.version_id, []).append(candidate_counts)
        self._allocation_probability = None

        if self._proposal_id is None or self._candidate is None:
            # No running trial: give the creator a chance to start one from
            # the baseline evidence gathered so far (deterministic triggers).
            self._steps_in_cycle += 1
            if self._steps_in_cycle >= self._steps_per_cycle:
                self._steps_in_cycle = 0
                self._next_trial()
            return None
        self._steps_in_cycle += 1
        if self._steps_in_cycle < self._steps_per_cycle:
            return None

        baseline_m, baseline_n = self.reservoir.cycle_counts(self._baseline.version_id)
        candidate_m, candidate_n = self.reservoir.cycle_counts(self._candidate.version_id)
        total_episodes = baseline_n + candidate_n
        if total_episodes < self._min_version_episodes:
            logger.info(
                "Adoption cycle %d at step %d has %d/%d episodes; extending",
                self._cycle,
                step,
                total_episodes,
                self._min_version_episodes,
            )
            return None

        if before_decision is not None:
            before_decision()
        accepted = self._candidate_arm.mean() > self._baseline_arm.mean()
        if self._memory_estimator is not None:
            self._memory_estimator.add_test(
                self._step_history.get(self._baseline.version_id, ()),
                self._step_history.get(self._candidate.version_id, ()),
            )
        reason = (
            f"adoption cycle {self._cycle}: baseline m/n={baseline_m}/{baseline_n} "
            f"E[p]={self._baseline_arm.mean():.6f}, candidate m/n={candidate_m}/{candidate_n} "
            f"E[p]={self._candidate_arm.mean():.6f}"
        )
        evidence = tuple(self._cycle_evidence)
        if accepted:
            self._store.accept_proposal(self._proposal_id, evidence=evidence, reason=reason)
        else:
            self._store.decide_proposal(
                self._proposal_id, ProposalStatus.REJECTED, evidence=evidence, reason=reason
            )
        record = ProposalCycleRecord(
            cycle=self._cycle,
            proposal_id=self._proposal_id,
            status=ProposalStatus.ACCEPTED if accepted else ProposalStatus.REJECTED,
            baseline_version_id=self._baseline.version_id,
            candidate_version_id=self._candidate.version_id,
            baseline_posterior_mean=self._baseline_arm.mean(),
            candidate_posterior_mean=self._candidate_arm.mean(),
            observation_count=baseline_n + candidate_n,
        )
        logger.info("Skill-bank proposal %s %s (%s)", self._proposal_id, record.status.value, reason)

        # Re-derive the baseline from the store's ACTIVE pointer rather than
        # mirroring the promotion locally.
        self._baseline = self._store.resolve(verify=False)
        self._candidate = None
        self._proposal_id = None
        self._steps_in_cycle = 0
        self._store.garbage_collect()
        self._next_trial()
        return record

    def discard_pending_step(self) -> None:
        """Discard rollout evidence when its corresponding policy step fails."""
        self._pending_entries.clear()
        self._pending_evidence.clear()
        self._allocation_probability = None

    # -- internals ---------------------------------------------------------

    def _start_cycle(self) -> None:
        self._cycle += 1
        self._steps_in_cycle = 0
        self._cycle_evidence = []
        self._step_history = {}
        self.reservoir.reset_cycle()
        # Per-cycle Beta(1, 1) priors, per the ReSkill defaults.
        self._baseline_arm.reset()
        self._candidate_arm.reset()
        if self._memory_estimator is not None:
            self._baseline_arm.set_memory(self._memory_estimator.memory)
            self._candidate_arm.set_memory(self._memory_estimator.memory)

    def _next_trial(self) -> None:
        entries = self.reservoir.entries()
        if len(entries) < self._min_reservoir_size:
            return
        try:
            proposal_id = self._candidate_creator(self._baseline, entries)
            if inspect.isawaitable(proposal_id):
                proposal_id = asyncio.run(proposal_id)
        except Exception as exc:
            logger.error("Skill-bank creator failed: %s", exc)
            return
        if proposal_id:
            self.begin_trial(proposal_id)


__all__ = [
    "DiscountedBetaArm",
    "SkillBankAdoptionCycle",
    "ThompsonAllocation",
]

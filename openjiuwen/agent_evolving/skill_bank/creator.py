# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""History-informed ReSkill creator pipeline over retained RL trajectories."""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any, Sequence

from openjiuwen.agent_evolving.optimizer.llm_resilience import LLMInvokePolicy, invoke_text_with_retry
from openjiuwen.agent_evolving.skill_bank.assertions import (
    AssertionGrade,
    TrajectoryAssertion,
    grade_assertions,
)
from openjiuwen.agent_evolving.skill_bank.candidate import (
    SkillBankCandidateBuilder,
    SkillBankPackageCreator,
)
from openjiuwen.agent_evolving.skill_bank.models import BankVersionRef, param_error as _param
from openjiuwen.agent_evolving.skill_bank.reservoir import ReservoirEntry
from openjiuwen.agent_evolving.skill_bank.store import SkillBankStore
from openjiuwen.agent_evolving.skill_bank.templates import (
    SKILL_BANK_CONTRASTIVE_PROMPT,
    SKILL_BANK_DIAGNOSER_PROMPT,
)
from openjiuwen.agent_evolving.skill_bank.triggers import SkillTrigger, skill_trigger_fire_rate
from openjiuwen.agent_evolving.utils import TuneUtils
from openjiuwen.core.common.exception.errors import BaseError
from openjiuwen.core.common.logging import logger
from openjiuwen.core.foundation.llm.model import Model

_PIPELINE_LLM_POLICY = LLMInvokePolicy(attempt_timeout_secs=120.0, total_budget_secs=300.0)


class SkillBankCreatorPipeline:
    """Create candidates from grouped contrasts, assertions, and proposal history.

    The same creator model performs diagnosis and authoring sequentially. This
    preserves ReSkill's data flow without requiring another resident model.
    """

    def __init__(
        self,
        bank_root: str | Path,
        llm: Model,
        model: str,
        *,
        language: str = "cn",
        max_task_groups: int = 6,
        minimum_trigger_fire_rate: float = 0.5,
    ) -> None:
        if language not in SKILL_BANK_CONTRASTIVE_PROMPT:
            raise _param(f"unsupported creator language: {language}")
        if max_task_groups <= 0:
            raise _param("max task groups must be positive")
        self._bank_root = str(Path(bank_root).expanduser())
        self._llm = llm
        self._model = model
        self._language = language
        self._assertions: dict[str, TrajectoryAssertion] = {}
        self._max_task_groups = max_task_groups
        self._minimum_trigger_fire_rate = minimum_trigger_fire_rate
        self._store: SkillBankStore | None = None
        self._package_creator: SkillBankPackageCreator | None = None

    @property
    def assertions(self) -> tuple[TrajectoryAssertion, ...]:
        return tuple(self._assertions.values())

    async def __call__(
        self,
        parent: BankVersionRef,
        entries: Sequence[ReservoirEntry],
    ) -> str | None:
        evidence = [entry for entry in entries if entry.payload is not None]
        if not evidence or all(entry.success for entry in evidence):
            return None
        store, package_creator = self._components()
        insights = await self._contrast_task_groups(evidence)
        grades_before = grade_assertions(self.assertions, evidence)
        diagnosis = await self._diagnose(parent, insights, grades_before)
        self._apply_assertion_operations(diagnosis.get("assertion_operations", []))
        analysis = {
            "diagnosis": str(diagnosis.get("diagnosis", "")),
            "insights": insights,
            "insight_groups": diagnosis.get("insight_groups", []),
            "assertion_grades": [grade.to_dict() for grade in grade_assertions(self.assertions, evidence)],
            "skill_trigger_rates": self._skill_trigger_rates(parent, evidence),
        }
        result = await package_creator.create(
            parent,
            evidence,
            proposal_id=f"skill-rl-{uuid.uuid4().hex}",
            creator="reskill-creator",
            analysis=analysis,
            proposal_history=store.list_proposals(),
        )
        return result.proposal.proposal_id if result is not None else None

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_store"] = None
        state["_package_creator"] = None
        return state

    def _components(self) -> tuple[SkillBankStore, SkillBankPackageCreator]:
        if self._store is None:
            self._store = SkillBankStore(self._bank_root)
        if self._package_creator is None:
            self._package_creator = SkillBankPackageCreator(
                SkillBankCandidateBuilder(self._store),
                self._llm,
                self._model,
                language=self._language,
                minimum_trigger_fire_rate=self._minimum_trigger_fire_rate,
            )
        return self._store, self._package_creator

    async def _contrast_task_groups(self, entries: Sequence[ReservoirEntry]) -> list[dict[str, Any]]:
        insights = []
        for _, group in _task_groups(entries)[: self._max_task_groups]:
            episodes = [
                {
                    "version_id": entry.version_id,
                    "outcome": "success" if entry.success else "failure",
                    "reward": entry.reward,
                    "trajectory": entry.payload,
                }
                for entry in group
            ]
            prompt = SKILL_BANK_CONTRASTIVE_PROMPT[self._language].format(
                episodes=json.dumps(episodes, ensure_ascii=False, indent=2)
            )
            raw = await invoke_text_with_retry(self._llm, self._model, prompt, policy=_PIPELINE_LLM_POLICY)
            parsed = TuneUtils.parse_json_from_llm_response(raw)
            if isinstance(parsed, dict):
                insights.append(parsed)
            else:
                logger.warning("Skill-bank contrastive analyzer returned unparseable output")
        return insights

    async def _diagnose(
        self,
        parent: BankVersionRef,
        insights: Sequence[dict],
        grades: Sequence[AssertionGrade],
    ) -> dict[str, Any]:
        current_skills = SkillBankPackageCreator.format_current_skills(parent)
        prompt = SKILL_BANK_DIAGNOSER_PROMPT[self._language].format(
            assertion_grades=json.dumps([grade.to_dict() for grade in grades], ensure_ascii=False, indent=2),
            insights=json.dumps(list(insights), ensure_ascii=False, indent=2),
            current_skills=current_skills,
        )
        raw = await invoke_text_with_retry(self._llm, self._model, prompt, policy=_PIPELINE_LLM_POLICY)
        parsed = TuneUtils.parse_json_from_llm_response(raw)
        if not isinstance(parsed, dict):
            logger.warning("Skill-bank assertion diagnoser returned unparseable output")
            return {}
        return parsed

    @staticmethod
    def _skill_trigger_rates(
        parent: BankVersionRef,
        entries: Sequence[ReservoirEntry],
    ) -> dict[str, float]:
        rates = {}
        for skill_name in parent.manifest.skill_names:
            content = (parent.skills_dir / skill_name / "SKILL.md").read_text(encoding="utf-8", errors="replace")
            rates[skill_name] = skill_trigger_fire_rate(SkillTrigger.from_skill_md(content), entries)
        return rates

    def _apply_assertion_operations(self, operations: Any) -> None:
        if not isinstance(operations, list):
            return
        for operation in operations:
            if not isinstance(operation, dict):
                continue
            action = str(operation.get("op", "")).strip().lower()
            assertion_data = operation.get("assertion", {})
            if not isinstance(assertion_data, dict):
                continue
            name = str(assertion_data.get("name", "")).strip()
            try:
                if action == "delete":
                    self._assertions.pop(name, None)
                elif action in {"add", "modify"}:
                    assertion = TrajectoryAssertion.from_dict(assertion_data)
                    self._assertions[assertion.name] = assertion
            except BaseError as exc:
                logger.warning("Ignoring invalid trajectory assertion operation: %s", exc)


def _task_groups(entries: Sequence[ReservoirEntry]) -> list[tuple[str, list[ReservoirEntry]]]:
    grouped: dict[str, list[ReservoirEntry]] = {}
    for entry in entries:
        grouped.setdefault(entry.task_key or entry.origin_task_id, []).append(entry)
    return sorted(
        grouped.items(),
        key=lambda item: (
            -len({entry.success for entry in item[1]}),
            -len({entry.version_id for entry in item[1]}),
            item[0],
        ),
    )


__all__ = ["SkillBankCreatorPipeline"]

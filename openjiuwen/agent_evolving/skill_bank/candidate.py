# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Compile package proposals into immutable bank candidates."""

from __future__ import annotations

import json
import shutil
import stat
import tempfile
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from openjiuwen.agent_evolving.optimizer.llm_resilience import LLMInvokePolicy, invoke_text_with_retry
from openjiuwen.agent_evolving.skill_bank.models import (
    BankVersionRef,
    SkillBankProposal,
    SkillOperation,
    SkillOperationType,
    SkillSourceProvenance,
    TRANSIENT_SKILL_DIR_NAMES,
    dump_json_text,
    is_skill_package,
    param_error as _param,
)
from openjiuwen.agent_evolving.skill_bank.reservoir import ReservoirEntry
from openjiuwen.agent_evolving.skill_bank.source import (
    build_skill_provenance,
    validate_skill_package,
)
from openjiuwen.agent_evolving.skill_bank.store import SkillBankStore
from openjiuwen.agent_evolving.skill_bank.templates import SKILL_BANK_CREATOR_PROMPT
from openjiuwen.agent_evolving.skill_bank.triggers import validate_skill_trigger
from openjiuwen.agent_evolving.utils import TuneUtils, frontmatter_body
from openjiuwen.core.common.exception.errors import BaseError
from openjiuwen.core.common.logging import logger
from openjiuwen.core.foundation.llm.model import Model
from openjiuwen.harness.prompts.sections.skills import MAX_INLINE_SKILL_GUIDANCE_CHARS


@dataclass(frozen=True)
class CandidateBuildResult:
    """Recorded proposal and its resolved immutable candidate version."""

    proposal: SkillBankProposal
    candidate: BankVersionRef


class SkillBankCandidateBuilder:
    """Apply bank operations only inside a writable parent-version copy."""

    def __init__(self, store: SkillBankStore, *, max_bank_skills: int = 8) -> None:
        if max_bank_skills <= 0:
            raise _param("max bank skills must be positive")
        self._store = store
        self.max_bank_skills = max_bank_skills

    def build(
        self,
        proposal: SkillBankProposal,
    ) -> CandidateBuildResult:
        """Create, snapshot, and audit one isolated candidate."""
        if proposal.candidate_version_id is not None:
            raise _param("proposal already has a candidate version")
        self._validate_operations(proposal.operations)
        parent = self._store.resolve(proposal.parent_version_id, verify=True)
        expected = self._expected_skills(parent, proposal.operations)
        if len(expected) > self.max_bank_skills:
            raise _param(
                f"candidate has {len(expected)} skills; maximum is {self.max_bank_skills}"
            )
        self._store.record_proposal(proposal)

        with tempfile.TemporaryDirectory(prefix="openjiuwen-skill-candidate-") as temporary:
            candidate_root = Path(temporary) / "skills"
            shutil.copytree(parent.skills_dir, candidate_root, copy_function=shutil.copy2)
            _make_writable(candidate_root)
            self._apply_package_operations(candidate_root, proposal.operations)
            self._validate_final_skills(expected, candidate_root)
            sources = self._build_candidate_provenance(parent, candidate_root, proposal)
            candidate = self._store.create_snapshot(
                candidate_root,
                parent_version_id=parent.version_id,
                proposal_id=proposal.proposal_id,
                sources=sources,
            )

        recorded = self._store.attach_candidate(proposal.proposal_id, candidate.version_id)
        return CandidateBuildResult(proposal=recorded, candidate=candidate)

    @staticmethod
    def _validate_operations(operations: Sequence[SkillOperation]) -> None:
        names = [operation.skill_name for operation in operations]
        if len(names) != len(set(names)):
            raise _param("a proposal can contain at most one operation per skill")

    @staticmethod
    def _apply_package_operations(root: Path, operations: Sequence[SkillOperation]) -> None:
        deleted: set[str] = set()
        for operation in operations:
            target = root / operation.skill_name
            if operation.action is SkillOperationType.DELETE:
                if not target.is_dir():
                    raise _param(f"cannot delete missing skill '{operation.skill_name}'")
                shutil.rmtree(target)
                deleted.add(operation.skill_name)
                continue

            if operation.action is SkillOperationType.ADD and target.exists():
                raise _param(f"cannot add existing skill '{operation.skill_name}'")
            if operation.action is SkillOperationType.MODIFY and not target.is_dir():
                raise _param(f"cannot modify missing skill '{operation.skill_name}'")
            if operation.source_path is None:
                if operation.action is SkillOperationType.ADD:
                    raise _param(f"ADD for '{operation.skill_name}' requires source_path")
                continue

            source = validate_skill_package(operation.source_path)
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(
                source,
                target,
                copy_function=shutil.copy2,
                ignore=shutil.ignore_patterns(*TRANSIENT_SKILL_DIR_NAMES),
            )

        if deleted:
            _remove_deleted_state(root, deleted)

    @staticmethod
    def _expected_skills(
        parent: BankVersionRef,
        operations: Sequence[SkillOperation],
    ) -> set[str]:
        """Return the skill names the candidate must end up with."""
        expected = set(parent.manifest.skill_names)
        for operation in operations:
            if operation.action is SkillOperationType.ADD:
                expected.add(operation.skill_name)
            elif operation.action is SkillOperationType.DELETE:
                expected.discard(operation.skill_name)
        return expected

    @staticmethod
    def _validate_final_skills(expected: set[str], candidate_root: Path) -> None:
        actual = {package.name for package in candidate_root.iterdir() if is_skill_package(package)}
        if actual != expected:
            raise _param("candidate skill set does not match the proposal operations")

    @staticmethod
    def _build_candidate_provenance(
        parent: BankVersionRef,
        candidate_root: Path,
        proposal: SkillBankProposal,
    ) -> tuple[SkillSourceProvenance, ...]:
        inherited = {source.skill_name: source for source in parent.manifest.sources}
        operations = {operation.skill_name: operation for operation in proposal.operations}
        result: list[SkillSourceProvenance] = []
        for package in sorted(candidate_root.iterdir(), key=lambda item: item.name):
            if not is_skill_package(package):
                continue
            operation = operations.get(package.name)
            if operation is not None and operation.provenance is not None:
                metadata = operation.provenance
                result.append(
                    build_skill_provenance(
                        package,
                        skill_name=package.name,
                        source_kind=metadata.source_kind,
                        location=metadata.location,
                        revision=metadata.revision,
                        license=metadata.license,
                    )
                )
                continue
            previous = inherited.get(package.name)
            if operation is None and previous is not None:
                result.append(previous)
                continue
            result.append(
                build_skill_provenance(
                    package,
                    skill_name=package.name,
                    source_kind="evolution" if operation and operation.source_path is None else "local",
                    location=(
                        operation.source_path
                        if operation and operation.source_path
                        else f"skill-bank:{parent.version_id}/{package.name}"
                    ),
                    revision=proposal.proposal_id,
                    license=previous.license if previous else None,
                )
            )
        return tuple(result)


_CREATOR_LLM_POLICY = LLMInvokePolicy(attempt_timeout_secs=120.0, total_budget_secs=300.0)
_MAX_CREATOR_OPERATIONS = 3
_MAX_CREATOR_FILES_PER_SKILL = 16
_MAX_AUTHORING_ATTEMPTS = 3
_MAX_EXAMPLES_PER_OUTCOME = 6


class SkillBankPackageCreator:
    """Author a package candidate from retained success/failure examples.

    This adapter turns one structured LLM response into operations for
    ``SkillBankCandidateBuilder`` and validates authored trigger coverage.
    """

    def __init__(
        self,
        builder: SkillBankCandidateBuilder,
        llm: Model,
        model: str,
        *,
        language: str = "cn",
        minimum_trigger_fire_rate: float = 0.5,
        action_vocabulary: Sequence[str] = (),
    ) -> None:
        if language not in SKILL_BANK_CREATOR_PROMPT:
            raise _param(f"unsupported creator language: {language}")
        if not 0.0 <= minimum_trigger_fire_rate <= 1.0:
            raise _param("minimum trigger fire rate must be between zero and one")
        self._builder = builder
        self._llm = llm
        self._model = model
        self._language = language
        self._minimum_trigger_fire_rate = minimum_trigger_fire_rate
        self._action_vocabulary = tuple(dict.fromkeys(str(name) for name in action_vocabulary if name))

    async def create(
        self,
        parent: BankVersionRef,
        entries: Sequence[ReservoirEntry],
        *,
        proposal_id: str,
        creator: str = "skill-bank-creator",
        analysis: Mapping[str, object] | None = None,
        proposal_history: Sequence[SkillBankProposal] = (),
    ) -> CandidateBuildResult | None:
        """Propose and build one candidate from reservoir evidence, or skip."""
        evidence = [entry for entry in entries if entry.payload is not None]
        if not evidence:
            return None

        # Only trigger_feedback changes between authoring attempts.
        prompt_args = {
            "bank_version": parent.version_id,
            "bank_size": len(parent.manifest.skill_names),
            "max_bank_skills": self._builder.max_bank_skills,
            "action_vocabulary": self._format_action_vocabulary(evidence),
            "current_skills": self.format_current_skills(parent),
            "analysis": json.dumps(analysis or {}, ensure_ascii=False, indent=2),
            "proposal_history": self._format_proposal_history(proposal_history),
            "successes": self._format_examples(evidence, success=True),
            "failures": self._format_examples(evidence, success=False),
            "max_operations": _MAX_CREATOR_OPERATIONS,
            "max_inline_chars": MAX_INLINE_SKILL_GUIDANCE_CHARS,
        }
        trigger_feedback = "(none)"
        parsed = None
        for attempt in range(_MAX_AUTHORING_ATTEMPTS):
            prompt = SKILL_BANK_CREATOR_PROMPT[self._language].format(
                **prompt_args,
                trigger_feedback=trigger_feedback,
            )
            raw = await invoke_text_with_retry(self._llm, self._model, prompt, policy=_CREATOR_LLM_POLICY)
            parsed = _parse_creator_response(raw)
            if parsed is None:
                logger.warning("Skill-bank creator returned unparseable output (preview: %s)", raw[:200])
                return None
            _, raw_operations = parsed
            if not raw_operations:
                return None
            trigger_feedback = self._validate_triggers(raw_operations, evidence) or ""
            if not trigger_feedback:
                break
            logger.info(
                "Skill-bank authoring attempt %d failed trigger validation: %s",
                attempt + 1,
                trigger_feedback,
            )
        if trigger_feedback:
            return None
        assertion, raw_operations = parsed

        with ExitStack() as stack:
            operations = tuple(
                self._to_operation(item, assertion, proposal_id, parent, stack)
                for item in raw_operations
            )
            proposal = SkillBankProposal(
                proposal_id=proposal_id,
                parent_version_id=parent.version_id,
                operations=operations,
                source_trajectory_ids=tuple(
                    dict.fromkeys(entry.task_key or entry.task_id for entry in evidence)
                ),
                creator=creator,
                model=self._model,
            )
            return self._builder.build(proposal)

    def _validate_triggers(
        self,
        operations: Sequence[dict],
        evidence: Sequence[ReservoirEntry],
    ) -> str | None:
        feedback = []
        for operation in operations:
            if str(operation.get("action", "")).strip().lower() == SkillOperationType.DELETE.value:
                continue
            skill_md = operation.get("skill_md")
            if not isinstance(skill_md, str):
                continue
            body = frontmatter_body(skill_md)
            if not body:
                issue = "inline SKILL.md guidance must not be empty"
            elif len(body) > MAX_INLINE_SKILL_GUIDANCE_CHARS:
                issue = (
                    f"inline SKILL.md guidance has {len(body)} characters; "
                    f"maximum is {MAX_INLINE_SKILL_GUIDANCE_CHARS}; move detail to support files"
                )
            else:
                try:
                    issue = validate_skill_trigger(
                        str(operation.get("skill_name", "")),
                        skill_md,
                        evidence,
                        minimum_fire_rate=self._minimum_trigger_fire_rate,
                    )
                except BaseError as exc:
                    issue = str(exc)
            if issue:
                feedback.append(issue)
        return "; ".join(feedback) or None

    @staticmethod
    def _format_proposal_history(proposals: Sequence[SkillBankProposal]) -> str:
        if not proposals:
            return "(none)"
        return json.dumps(
            [
                {
                    "proposal_id": proposal.proposal_id,
                    "status": proposal.status.value,
                    "operations": [
                        {
                            "action": operation.action.value,
                            "skill_name": operation.skill_name,
                            "summary": operation.summary,
                        }
                        for operation in proposal.operations
                    ],
                    "decision_reason": proposal.decision_reason,
                }
                for proposal in sorted(proposals, key=lambda item: item.created_at)[-12:]
            ],
            ensure_ascii=False,
            indent=2,
        )

    @staticmethod
    def format_current_skills(parent: BankVersionRef) -> str:
        skills = []
        for skill_name in parent.manifest.skill_names:
            skill_md = parent.skills_dir / skill_name / "SKILL.md"
            skills.append(
                {
                    "skill_name": skill_name,
                    "skill_md": skill_md.read_text(encoding="utf-8", errors="replace"),
                }
            )
        return json.dumps(skills, ensure_ascii=False, indent=2)

    def _format_examples(self, evidence: Sequence[ReservoirEntry], *, success: bool) -> str:
        selected = [entry for entry in evidence if entry.success is success][:_MAX_EXAMPLES_PER_OUTCOME]
        if not selected:
            return "(none)"
        return json.dumps(
            [
                {
                    "task": entry.origin_task_id,
                    "version_id": entry.version_id,
                    "reward": entry.reward,
                    "trajectory": entry.payload,
                }
                for entry in selected
            ],
            ensure_ascii=False,
            indent=2,
        )

    def _format_action_vocabulary(self, evidence: Sequence[ReservoirEntry]) -> str:
        names = set(self._action_vocabulary)
        for entry in evidence:
            for turn in (entry.payload or {}).get("turns", []):
                action = turn.get("action") if isinstance(turn, dict) else None
                if not isinstance(action, list):
                    continue
                for call in action:
                    if not isinstance(call, dict):
                        continue
                    function = call.get("function")
                    source = function if isinstance(function, dict) else call
                    name = source.get("name")
                    if name:
                        names.add(str(name))
        return json.dumps(sorted(names), ensure_ascii=False)

    def _to_operation(
        self,
        item: dict,
        assertion: str,
        proposal_id: str,
        parent: BankVersionRef,
        stack: ExitStack,
    ) -> SkillOperation:
        try:
            action = SkillOperationType(str(item.get("action", "")).strip().lower())
        except ValueError as exc:
            raise _param(f"creator emitted unsupported operation action: {item.get('action')!r}") from exc
        skill_name = str(item.get("skill_name", "")).strip()
        source_path: str | None = None
        if action is not SkillOperationType.DELETE:
            skill_md = item.get("skill_md")
            if not isinstance(skill_md, str) or not skill_md.strip():
                raise _param(f"creator {action.value} for '{skill_name}' requires skill_md content")
            package = Path(stack.enter_context(tempfile.TemporaryDirectory(prefix="openjiuwen-skill-creator-")))
            package = package / skill_name
            if action is SkillOperationType.MODIFY:
                source = parent.skills_dir / skill_name
                if not is_skill_package(source):
                    raise _param(f"creator cannot modify missing skill '{skill_name}'")
                shutil.copytree(source, package, copy_function=shutil.copy2)
                _make_writable(package)
            else:
                package.mkdir()
            (package / "SKILL.md").write_text(skill_md, encoding="utf-8")
            self._write_creator_files(package, item.get("files") or {})
            source_path = str(package)
        return SkillOperation(
            action=action,
            skill_name=skill_name,
            source_path=source_path,
            summary=assertion,
            provenance=(
                SkillSourceProvenance(
                    skill_name=skill_name,
                    source_kind="skill-bank-creator",
                    location=f"proposal:{proposal_id}/{skill_name}",
                )
                if source_path is not None
                else None
            ),
        )

    @staticmethod
    def _write_creator_files(package: Path, files: dict) -> None:
        if not isinstance(files, dict):
            raise _param("creator files must be an object of relative paths to text")
        if len(files) > _MAX_CREATOR_FILES_PER_SKILL:
            raise _param(f"creator emitted more than {_MAX_CREATOR_FILES_PER_SKILL} files for one skill")
        for relative, content in files.items():
            relative_path = Path(str(relative))
            if relative_path.is_absolute() or ".." in relative_path.parts or not str(relative).strip():
                raise _param(f"creator file path must stay inside the skill package: {relative!r}")
            if relative_path == Path("SKILL.md"):
                raise _param("creator files cannot replace the validated SKILL.md")
            target = package / relative_path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(str(content), encoding="utf-8")


def _parse_creator_response(raw: str) -> tuple[str, list[dict]] | None:
    data = TuneUtils.parse_json_from_llm_response(raw)
    if not isinstance(data, dict) or not isinstance(data.get("operations"), list):
        return None
    operations = [item for item in data["operations"] if isinstance(item, dict)]
    if len(operations) > _MAX_CREATOR_OPERATIONS:
        return None
    assertion = str(data.get("assertion", "")).strip()
    if operations and not assertion:
        return None
    return assertion, operations


def _remove_deleted_state(root: Path, deleted: set[str]) -> None:
    state_path = root / "skills_state.json"
    if not state_path.is_file():
        return
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
        skill_configs = state.get("skill_configs", {})
        if isinstance(skill_configs, dict):
            for name in deleted:
                skill_configs.pop(name, None)
        state_path.write_text(dump_json_text(state), encoding="utf-8")
    except (OSError, json.JSONDecodeError, TypeError) as exc:
        raise _param(f"invalid skills_state.json in parent version: {exc}") from exc


def _make_writable(root: Path) -> None:
    for path in (root, *root.rglob("*")):
        if path.is_symlink():
            raise _param(f"candidate copy contains symbolic link: {path}")
        path.chmod(stat.S_IMODE(path.stat().st_mode) | stat.S_IWUSR)


__all__ = [
    "CandidateBuildResult",
    "SkillBankPackageCreator",
    "SkillBankCandidateBuilder",
]

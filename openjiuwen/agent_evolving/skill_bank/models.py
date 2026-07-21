# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Data models for immutable skill-bank versions and proposal audit."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from openjiuwen.core.common.exception.codes import StatusCode
from openjiuwen.core.common.exception.errors import BaseError, build_error

_SAFE_SKILL_NAME = re.compile(r"^[A-Za-z0-9_-][A-Za-z0-9._-]{0,127}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
TRANSIENT_SKILL_DIR_NAMES = frozenset({".git", "__pycache__"})


def utc_now_iso() -> str:
    """Return a timezone-aware timestamp in the repository's standard format."""
    return datetime.now(tz=timezone.utc).isoformat()


def dump_json_text(payload: dict[str, Any]) -> str:
    """Serialize skill-bank JSON files in one canonical text form."""
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def param_error(message: str, *, cause: BaseException | None = None) -> BaseError:
    """Build the shared skill-bank parameter error."""
    return build_error(
        StatusCode.TOOLCHAIN_EVOLVING_SKILL_BANK_PARAM_ERROR,
        error_msg=message,
        cause=cause,
    )


def not_found_error(message: str) -> BaseError:
    """Build the shared skill-bank not-found error."""
    return build_error(
        StatusCode.TOOLCHAIN_EVOLVING_SKILL_BANK_NOT_FOUND,
        error_msg=message,
    )


def execution_error(message: str, *, cause: BaseException | None = None) -> BaseError:
    """Build the shared skill-bank execution error."""
    return build_error(
        StatusCode.TOOLCHAIN_EVOLVING_SKILL_BANK_EXECUTION_ERROR,
        error_msg=message,
        cause=cause,
    )


def integrity_error(message: str, *, cause: BaseException | None = None) -> BaseError:
    """Build the shared skill-bank integrity error."""
    return build_error(
        StatusCode.TOOLCHAIN_EVOLVING_SKILL_BANK_INTEGRITY_INVALID,
        error_msg=message,
        cause=cause,
    )


def validate_safe_name(value: object, *, field: str) -> None:
    """Reject names outside the shared safe-name rule used for path segments."""
    if not isinstance(value, str) or not _SAFE_SKILL_NAME.fullmatch(value):
        raise param_error(f"invalid {field}: {value!r}")


def is_transient_skill_path(relative_path: Path) -> bool:
    """Return whether a relative skill path contains transient local state."""
    return any(part in TRANSIENT_SKILL_DIR_NAMES for part in relative_path.parts)


def is_skill_package(path: Path) -> bool:
    """Return whether a directory is an ordinary skill package root."""
    return path.is_dir() and (path / "SKILL.md").is_file()


class SkillOperationType(str, Enum):
    """Repository-level operation proposed for one skill."""

    ADD = "add"
    MODIFY = "modify"
    DELETE = "delete"


class ProposalStatus(str, Enum):
    """Lifecycle state of a skill-bank proposal."""

    PENDING = "pending"
    ACCEPTED = "accepted"
    REJECTED = "rejected"


@dataclass(frozen=True)
class SkillSourceProvenance:
    """Origin metadata for one package in an effective skill bank."""

    skill_name: str
    source_kind: str
    location: str
    revision: str | None = None
    content_sha256: str | None = None
    license: str | None = None

    def __post_init__(self) -> None:
        skill_name = str(self.skill_name).strip()
        source_kind = str(self.source_kind).strip()
        location = str(self.location).strip()
        revision = str(self.revision).strip() if self.revision is not None else None
        content_sha256 = str(self.content_sha256).strip().lower() if self.content_sha256 is not None else None
        license_name = str(self.license).strip() if self.license is not None else None
        validate_safe_name(skill_name, field="provenance skill name")
        if not source_kind:
            raise param_error("skill source provenance requires source_kind")
        if not location:
            raise param_error("skill source provenance requires location")
        if content_sha256 and not _SHA256.fullmatch(content_sha256):
            raise param_error("skill source provenance content_sha256 must be a SHA-256 digest")
        object.__setattr__(self, "skill_name", skill_name)
        object.__setattr__(self, "source_kind", source_kind)
        object.__setattr__(self, "location", location)
        object.__setattr__(self, "revision", revision or None)
        object.__setattr__(self, "content_sha256", content_sha256 or None)
        object.__setattr__(self, "license", license_name or None)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "skill_name": self.skill_name,
            "source_kind": self.source_kind,
            "location": self.location,
        }
        if self.revision:
            payload["revision"] = self.revision
        if self.content_sha256:
            payload["content_sha256"] = self.content_sha256
        if self.license:
            payload["license"] = self.license
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SkillSourceProvenance":
        try:
            return cls(
                skill_name=data.get("skill_name", ""),
                source_kind=data.get("source_kind", ""),
                location=data.get("location", ""),
                revision=data.get("revision"),
                content_sha256=data.get("content_sha256"),
                license=data.get("license"),
            )
        except BaseError:
            raise
        except Exception as exc:
            raise param_error("invalid skill source provenance payload") from exc


@dataclass(frozen=True)
class SkillOperation:
    """One auditable repository-level skill operation."""

    action: SkillOperationType
    skill_name: str
    source_path: str | None = None
    summary: str = ""
    provenance: SkillSourceProvenance | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.skill_name, str):
            raise param_error("skill operation name must be a string")
        try:
            action = self.action if isinstance(self.action, SkillOperationType) else SkillOperationType(self.action)
        except (TypeError, ValueError) as exc:
            raise param_error(f"unsupported skill operation: {self.action}") from exc
        skill_name = self.skill_name.strip()
        if not skill_name:
            raise param_error("skill operation requires skill_name")
        validate_safe_name(skill_name, field="skill operation name")
        if self.provenance is not None:
            if not isinstance(self.provenance, SkillSourceProvenance):
                raise param_error("skill operation provenance must be SkillSourceProvenance")
            if self.provenance.skill_name != skill_name:
                raise param_error("skill operation provenance name must match skill_name")
        object.__setattr__(self, "action", action)
        object.__setattr__(self, "skill_name", skill_name)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "action": self.action.value,
            "skill_name": self.skill_name,
        }
        if self.source_path:
            payload["source_path"] = self.source_path
        if self.summary:
            payload["summary"] = self.summary
        if self.provenance:
            payload["provenance"] = self.provenance.to_dict()
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SkillOperation":
        try:
            return cls(
                action=data.get("action", ""),
                skill_name=data.get("skill_name", ""),
                source_path=data.get("source_path"),
                summary=str(data.get("summary", "")),
                provenance=(SkillSourceProvenance.from_dict(data["provenance"]) if data.get("provenance") else None),
            )
        except BaseError:
            raise
        except Exception as exc:
            raise param_error("invalid skill operation payload") from exc


@dataclass(frozen=True)
class EvaluationEvidence:
    """One reward observation attached to a proposal decision."""

    task_id: str
    version_id: str
    reward: float
    success: bool | None = None
    rollout_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.task_id, str) or not isinstance(self.version_id, str):
            raise param_error("evaluation evidence task_id and version_id must be strings")
        task_id = self.task_id.strip()
        version_id = self.version_id.strip()
        if not task_id:
            raise param_error("evaluation evidence requires task_id")
        if not version_id:
            raise param_error("evaluation evidence requires version_id")
        try:
            reward = float(self.reward)
        except (TypeError, ValueError) as exc:
            raise param_error("evaluation evidence reward must be numeric") from exc
        if not math.isfinite(reward):
            raise param_error("evaluation evidence reward must be finite")
        if self.success is not None and not isinstance(self.success, bool):
            raise param_error("evaluation evidence success must be a boolean")
        if not isinstance(self.metadata, dict):
            raise param_error("evaluation evidence metadata must be an object")
        object.__setattr__(self, "task_id", task_id)
        object.__setattr__(self, "version_id", version_id)
        object.__setattr__(self, "reward", reward)
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "task_id": self.task_id,
            "version_id": self.version_id,
            "reward": float(self.reward),
        }
        if self.success is not None:
            payload["success"] = self.success
        if self.rollout_id:
            payload["rollout_id"] = self.rollout_id
        if self.metadata:
            payload["metadata"] = self.metadata
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvaluationEvidence":
        try:
            return cls(
                task_id=data.get("task_id", ""),
                version_id=data.get("version_id", ""),
                reward=data.get("reward", 0.0),
                success=data.get("success"),
                rollout_id=data.get("rollout_id"),
                metadata=dict(data.get("metadata", {})),
            )
        except BaseError:
            raise
        except Exception as exc:
            raise param_error("invalid evaluation evidence payload") from exc


@dataclass(frozen=True)
class SkillBankProposal:
    """Persistent proposal metadata and its eventual adoption evidence."""

    proposal_id: str
    parent_version_id: str
    operations: tuple[SkillOperation, ...]
    source_trajectory_ids: tuple[str, ...] = ()
    creator: str = ""
    model: str = ""
    candidate_version_id: str | None = None
    status: ProposalStatus = ProposalStatus.PENDING
    evidence: tuple[EvaluationEvidence, ...] = ()
    decision_reason: str = ""
    created_at: str = field(default_factory=utc_now_iso)
    decided_at: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.proposal_id, str) or not isinstance(self.parent_version_id, str):
            raise param_error("proposal_id and parent_version_id must be strings")
        try:
            status = self.status if isinstance(self.status, ProposalStatus) else ProposalStatus(self.status)
        except (TypeError, ValueError) as exc:
            raise param_error(f"unsupported proposal status: {self.status}") from exc
        proposal_id = self.proposal_id.strip()
        parent_version_id = self.parent_version_id.strip()
        if not proposal_id:
            raise param_error("proposal_id is required")
        if not parent_version_id:
            raise param_error("parent_version_id is required")
        if not self.operations or not all(isinstance(operation, SkillOperation) for operation in self.operations):
            raise param_error("proposal operations must contain SkillOperation values")
        if self.evidence is None or not all(isinstance(item, EvaluationEvidence) for item in self.evidence):
            raise param_error("proposal evidence must contain EvaluationEvidence values")
        if self.source_trajectory_ids is None:
            raise param_error("proposal source_trajectory_ids must be a tuple of strings")
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "proposal_id", proposal_id)
        object.__setattr__(self, "parent_version_id", parent_version_id)
        object.__setattr__(self, "operations", tuple(self.operations))
        object.__setattr__(self, "source_trajectory_ids", tuple(str(item) for item in self.source_trajectory_ids))
        object.__setattr__(self, "evidence", tuple(self.evidence))

    def with_decision(
        self,
        status: ProposalStatus,
        *,
        evidence: tuple[EvaluationEvidence, ...] = (),
        reason: str = "",
        decided_at: str | None = None,
    ) -> "SkillBankProposal":
        """Return a finalized copy while preventing ambiguous re-decisions."""
        try:
            normalized = status if isinstance(status, ProposalStatus) else ProposalStatus(status)
        except (TypeError, ValueError) as exc:
            raise param_error(f"unsupported proposal status: {status}") from exc
        if normalized is ProposalStatus.PENDING:
            raise param_error("proposal decisions must be accepted or rejected")
        if self.status is not ProposalStatus.PENDING:
            raise param_error(f"proposal '{self.proposal_id}' is already {self.status.value}")
        return replace(
            self,
            status=normalized,
            evidence=tuple(evidence),
            decision_reason=reason,
            decided_at=decided_at or utc_now_iso(),
        )

    def with_candidate_version(self, version_id: str) -> "SkillBankProposal":
        """Attach the immutable candidate produced for a pending proposal."""
        if self.status is not ProposalStatus.PENDING:
            raise param_error("cannot attach a candidate to a decided proposal")
        if self.candidate_version_id is not None:
            raise param_error(f"proposal '{self.proposal_id}' already has a candidate version")
        version_id = str(version_id).strip()
        if not version_id:
            raise param_error("candidate version_id is required")
        return replace(self, candidate_version_id=version_id)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "proposal_id": self.proposal_id,
            "parent_version_id": self.parent_version_id,
            "operations": [operation.to_dict() for operation in self.operations],
            "source_trajectory_ids": list(self.source_trajectory_ids),
            "creator": self.creator,
            "model": self.model,
            "status": self.status.value,
            "evidence": [item.to_dict() for item in self.evidence],
            "decision_reason": self.decision_reason,
            "created_at": self.created_at,
        }
        if self.candidate_version_id:
            payload["candidate_version_id"] = self.candidate_version_id
        if self.decided_at:
            payload["decided_at"] = self.decided_at
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SkillBankProposal":
        try:
            return cls(
                proposal_id=data.get("proposal_id", ""),
                parent_version_id=data.get("parent_version_id", ""),
                operations=tuple(SkillOperation.from_dict(item) for item in data.get("operations", [])),
                source_trajectory_ids=tuple(str(item) for item in data.get("source_trajectory_ids", [])),
                creator=str(data.get("creator", "")),
                model=str(data.get("model", "")),
                candidate_version_id=data.get("candidate_version_id"),
                status=data.get("status", ProposalStatus.PENDING.value),
                evidence=tuple(EvaluationEvidence.from_dict(item) for item in data.get("evidence", [])),
                decision_reason=str(data.get("decision_reason", "")),
                created_at=str(data.get("created_at", "")) or utc_now_iso(),
                decided_at=data.get("decided_at"),
            )
        except BaseError:
            raise
        except Exception as exc:
            raise param_error("invalid skill-bank proposal payload") from exc


@dataclass(frozen=True)
class SkillBankManifest:
    """Deterministic metadata for one immutable whole-bank snapshot."""

    version_id: str
    created_at: str
    content_sha256: str
    file_count: int
    total_bytes: int
    skill_names: tuple[str, ...]
    disabled_skills: tuple[str, ...] = ()
    sources: tuple[SkillSourceProvenance, ...] = ()
    parent_version_id: str | None = None
    proposal_id: str | None = None
    schema_version: int = 1

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": self.schema_version,
            "version_id": self.version_id,
            "created_at": self.created_at,
            "content_sha256": self.content_sha256,
            "file_count": self.file_count,
            "total_bytes": self.total_bytes,
            "skill_names": list(self.skill_names),
            "disabled_skills": list(self.disabled_skills),
            "sources": [source.to_dict() for source in self.sources],
        }
        if self.parent_version_id:
            payload["parent_version_id"] = self.parent_version_id
        if self.proposal_id:
            payload["proposal_id"] = self.proposal_id
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SkillBankManifest":
        try:
            return cls(
                schema_version=int(data.get("schema_version", 1)),
                version_id=str(data.get("version_id", "")),
                created_at=str(data.get("created_at", "")),
                content_sha256=str(data.get("content_sha256", "")),
                file_count=int(data.get("file_count", 0)),
                total_bytes=int(data.get("total_bytes", 0)),
                skill_names=tuple(str(item) for item in data.get("skill_names", [])),
                disabled_skills=tuple(str(item) for item in data.get("disabled_skills", [])),
                sources=tuple(SkillSourceProvenance.from_dict(item) for item in data.get("sources", [])),
                parent_version_id=data.get("parent_version_id"),
                proposal_id=data.get("proposal_id"),
            )
        except BaseError:
            raise
        except Exception as exc:
            raise param_error("invalid skill-bank manifest payload") from exc


@dataclass(frozen=True)
class BankVersionRef:
    """Resolved immutable bank version and its concrete skill root."""

    version_id: str
    version_dir: Path
    manifest: SkillBankManifest

    @property
    def skills_dir(self) -> Path:
        """Return the root that must be mounted by one rollout's SkillUseRail."""
        return self.version_dir / "skills"

__all__ = [
    "BankVersionRef",
    "EvaluationEvidence",
    "ProposalStatus",
    "SkillBankManifest",
    "SkillBankProposal",
    "SkillOperation",
    "SkillOperationType",
    "SkillSourceProvenance",
]

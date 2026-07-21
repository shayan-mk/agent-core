# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Filesystem-backed immutable whole-bank snapshot store."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import uuid
from contextlib import suppress
from pathlib import Path
from typing import Any, Iterable

from filelock import FileLock

from openjiuwen.agent_evolving.skill_bank.models import (
    BankVersionRef,
    EvaluationEvidence,
    ProposalStatus,
    SkillBankManifest,
    SkillBankProposal,
    SkillSourceProvenance,
    TRANSIENT_SKILL_DIR_NAMES,
    dump_json_text,
    execution_error,
    integrity_error,
    is_transient_skill_path,
    not_found_error,
    param_error,
    utc_now_iso,
    validate_safe_name,
)
from openjiuwen.core.common.exception.errors import BaseError

_VERSION_PATTERN = re.compile(r"^bank_[0-9]{6,}$")
_MANIFEST_FILE = "manifest.json"
_ACTIVE_FILE = "ACTIVE"
_COUNTER_FILE = "VERSION_COUNTER"


class SkillBankStore:
    """Create, resolve, activate, audit, and collect immutable skill banks."""

    def __init__(self, bank_root: str | Path) -> None:
        try:
            self.root = Path(bank_root).expanduser().resolve()
            self.versions_dir = self.root / "versions"
            self.proposals_dir = self.root / "proposals"
            self.active_path = self.root / _ACTIVE_FILE
            self.counter_path = self.root / _COUNTER_FILE
            self.versions_dir.mkdir(parents=True, exist_ok=True)
            self.proposals_dir.mkdir(parents=True, exist_ok=True)
            self._lock = FileLock(str(self.root / ".skill_bank.lock"))
        except BaseError:
            raise
        except Exception as exc:
            raise execution_error(f"failed to initialize skill bank at {bank_root}: {exc}", cause=exc)

    def create_snapshot(
        self,
        skills_dir: str | Path,
        *,
        parent_version_id: str | None = None,
        proposal_id: str | None = None,
        sources: Iterable[SkillSourceProvenance] = (),
    ) -> BankVersionRef:
        """Copy a complete skills root into a new immutable version."""
        try:
            source = Path(skills_dir).expanduser().resolve()
            self._validate_snapshot_source(source)
            if parent_version_id is not None:
                self._validate_version_id(parent_version_id)
            if proposal_id is not None:
                validate_safe_name(proposal_id, field="proposal_id")
            source_records = tuple(sources)
            if not all(isinstance(item, SkillSourceProvenance) for item in source_records):
                raise param_error("snapshot sources must contain SkillSourceProvenance values")
            source_records = tuple(sorted(source_records, key=lambda item: item.skill_name))
            with self._lock:
                if parent_version_id is not None:
                    self._resolve_unlocked(parent_version_id, verify=True)
                version_id = self._allocate_version_id_unlocked()
                final_dir = self.versions_dir / version_id
                staging_dir = self.versions_dir / f".{version_id}.{uuid.uuid4().hex}.tmp"
                try:
                    payload_dir = staging_dir / "skills"
                    shutil.copytree(
                        source,
                        payload_dir,
                        copy_function=shutil.copy2,
                        ignore=shutil.ignore_patterns(*TRANSIENT_SKILL_DIR_NAMES),
                    )
                    copied_digest = self._digest_directory(payload_dir)
                    skill_names, disabled_skills = self._validate_snapshot_source(payload_dir)
                    if source_records:
                        source_names = [item.skill_name for item in source_records]
                        if len(source_names) != len(set(source_names)):
                            raise param_error("snapshot sources contain duplicate skill names")
                        if sorted(source_names) != skill_names:
                            raise param_error("snapshot sources must describe every effective skill exactly once")
                    content_sha256, file_count, total_bytes = copied_digest
                    manifest = SkillBankManifest(
                        version_id=version_id,
                        created_at=utc_now_iso(),
                        content_sha256=content_sha256,
                        file_count=file_count,
                        total_bytes=total_bytes,
                        skill_names=tuple(skill_names),
                        disabled_skills=tuple(disabled_skills),
                        sources=source_records,
                        parent_version_id=parent_version_id,
                        proposal_id=proposal_id,
                    )
                    self._write_json(staging_dir / _MANIFEST_FILE, manifest.to_dict())
                    self._make_read_only(staging_dir)
                    os.replace(staging_dir, final_dir)
                except Exception:
                    with suppress(OSError):
                        self._remove_tree(staging_dir)
                    raise
                return BankVersionRef(version_id=version_id, version_dir=final_dir, manifest=manifest)
        except BaseError:
            raise
        except Exception as exc:
            raise execution_error(f"failed to snapshot skills from {skills_dir}: {exc}", cause=exc)

    def resolve(self, version_id: str | None = None, *, verify: bool = True) -> BankVersionRef:
        """Resolve an explicit version, or the atomically selected active version."""
        try:
            with self._lock:
                if version_id is None:
                    pointer = self._read_active_pointer()
                    version_id = pointer["active_version_id"]
                return self._resolve_unlocked(version_id, verify=verify)
        except BaseError:
            raise
        except Exception as exc:
            raise execution_error(f"failed to resolve skill-bank version: {exc}", cause=exc)

    def list_versions(self, *, verify: bool = False) -> list[BankVersionRef]:
        """List snapshots in allocation order."""
        try:
            with self._lock:
                return [self._resolve_unlocked(version_id, verify=verify) for version_id in self._list_version_ids()]
        except BaseError:
            raise
        except Exception as exc:
            raise execution_error(f"failed to list skill-bank versions: {exc}", cause=exc)

    def promote(self, version_id: str) -> BankVersionRef:
        """Atomically make a verified version the inference default."""
        self._validate_version_id(version_id)
        try:
            with self._lock:
                return self._promote_unlocked(self._resolve_unlocked(version_id, verify=True))
        except BaseError:
            raise
        except Exception as exc:
            raise execution_error(f"failed to promote skill-bank version '{version_id}': {exc}", cause=exc)

    def rollback(self, version_id: str | None = None) -> BankVersionRef:
        """Atomically activate an explicit version or the previous active version."""
        if version_id is not None:
            self._validate_version_id(version_id)
        try:
            with self._lock:
                current = self._read_active_pointer()
                target = version_id or current.get("previous_version_id")
                if not target:
                    raise not_found_error("no previous active skill-bank version is available")
                ref = self._resolve_unlocked(target, verify=True)
                if target == current["active_version_id"]:
                    return ref
                pointer = {
                    "active_version_id": target,
                    "previous_version_id": current["active_version_id"],
                    "updated_at": utc_now_iso(),
                }
                self._write_json_atomic(self.active_path, pointer)
                return ref
        except BaseError:
            raise
        except Exception as exc:
            raise execution_error(f"failed to roll back skill bank: {exc}", cause=exc)

    def record_proposal(self, proposal: SkillBankProposal) -> SkillBankProposal:
        """Persist a new proposal without allowing an existing audit record to be overwritten."""
        validate_safe_name(proposal.proposal_id, field="proposal_id")
        if proposal.status is not ProposalStatus.PENDING:
            raise param_error("new proposal audit records must be pending")
        try:
            with self._lock:
                self._resolve_unlocked(proposal.parent_version_id, verify=True)
                if proposal.candidate_version_id:
                    candidate = self._resolve_unlocked(proposal.candidate_version_id, verify=True)
                    self._validate_candidate_linkage(proposal, candidate)
                path = self._proposal_path(proposal.proposal_id)
                if path.exists():
                    existing = self.get_proposal(proposal.proposal_id)
                    if existing == proposal:
                        return existing
                    raise param_error(f"proposal '{proposal.proposal_id}' already exists")
                self._write_json_atomic(path, proposal.to_dict())
                return proposal
        except BaseError:
            raise
        except Exception as exc:
            raise execution_error(f"failed to record proposal '{proposal.proposal_id}': {exc}", cause=exc)

    def get_proposal(self, proposal_id: str) -> SkillBankProposal:
        """Load one proposal audit record."""
        validate_safe_name(proposal_id, field="proposal_id")
        path = self._proposal_path(proposal_id)
        try:
            if not path.is_file():
                raise not_found_error(f"proposal '{proposal_id}' does not exist")
            data = self._read_json(path)
        except BaseError:
            raise
        except Exception as exc:
            raise integrity_error(f"invalid proposal audit record '{proposal_id}': {exc}", cause=exc)
        try:
            return SkillBankProposal.from_dict(data)
        except BaseError as exc:
            raise integrity_error(f"invalid proposal audit record '{proposal_id}'", cause=exc)

    def list_proposals(self) -> list[SkillBankProposal]:
        """List proposal records in stable identifier order."""
        try:
            return [self.get_proposal(path.stem) for path in sorted(self.proposals_dir.glob("*.json"))]
        except BaseError:
            raise
        except Exception as exc:
            raise execution_error(f"failed to list skill-bank proposals: {exc}", cause=exc)

    def decide_proposal(
        self,
        proposal_id: str,
        status: ProposalStatus,
        *,
        evidence: Iterable[EvaluationEvidence] = (),
        reason: str = "",
    ) -> SkillBankProposal:
        """Atomically finalize a pending proposal with its evaluation evidence."""
        validate_safe_name(proposal_id, field="proposal_id")
        try:
            normalized_status = status if isinstance(status, ProposalStatus) else ProposalStatus(status)
        except (TypeError, ValueError) as exc:
            raise param_error(f"unsupported proposal status: {status}", cause=exc)
        if normalized_status is ProposalStatus.ACCEPTED:
            return self.accept_proposal(proposal_id, evidence=evidence, reason=reason)
        try:
            with self._lock:
                proposal = self.get_proposal(proposal_id)
                decided = proposal.with_decision(normalized_status, evidence=tuple(evidence), reason=reason)
                self._write_json_atomic(self._proposal_path(proposal_id), decided.to_dict())
                return decided
        except BaseError:
            raise
        except Exception as exc:
            raise execution_error(f"failed to decide proposal '{proposal_id}': {exc}", cause=exc)

    def attach_candidate(self, proposal_id: str, candidate_version_id: str) -> SkillBankProposal:
        """Atomically attach one immutable candidate to a pending proposal."""
        validate_safe_name(proposal_id, field="proposal_id")
        self._validate_version_id(candidate_version_id)
        try:
            with self._lock:
                proposal = self.get_proposal(proposal_id)
                if proposal.candidate_version_id == candidate_version_id:
                    return proposal
                candidate = self._resolve_unlocked(candidate_version_id, verify=True)
                self._validate_candidate_linkage(proposal, candidate)
                attached = proposal.with_candidate_version(candidate_version_id)
                self._write_json_atomic(self._proposal_path(proposal_id), attached.to_dict())
                return attached
        except BaseError:
            raise
        except Exception as exc:
            raise execution_error(f"failed to attach candidate to proposal '{proposal_id}': {exc}", cause=exc)

    def accept_proposal(
        self,
        proposal_id: str,
        *,
        evidence: Iterable[EvaluationEvidence] = (),
        reason: str = "",
    ) -> SkillBankProposal:
        """Promote a candidate and finalize its proposal under one store lock."""
        validate_safe_name(proposal_id, field="proposal_id")
        try:
            with self._lock:
                proposal = self.get_proposal(proposal_id)
                if not proposal.candidate_version_id:
                    raise param_error(f"proposal '{proposal_id}' has no candidate version")
                candidate = self._resolve_unlocked(proposal.candidate_version_id, verify=True)
                try:
                    self._validate_candidate_linkage(proposal, candidate)
                except BaseError as exc:
                    raise integrity_error("candidate version does not match its proposal", cause=exc)

                current = self._read_active_pointer(required=False)
                if current and current["active_version_id"] not in {
                    proposal.parent_version_id,
                    candidate.version_id,
                }:
                    raise param_error(
                        f"proposal '{proposal_id}' parent is no longer the active skill-bank version"
                    )

                decided = proposal.with_decision(
                    ProposalStatus.ACCEPTED,
                    evidence=tuple(evidence),
                    reason=reason,
                )
                self._promote_unlocked(candidate)
                self._write_json_atomic(self._proposal_path(proposal_id), decided.to_dict())
                return decided
        except BaseError:
            raise
        except Exception as exc:
            raise execution_error(f"failed to accept proposal '{proposal_id}': {exc}", cause=exc)

    def garbage_collect(
        self,
        *,
        keep_latest: int = 2,
        keep_versions: Iterable[str] = (),
    ) -> list[str]:
        """Delete unreferenced snapshots while preserving rollback and pending work.

        The active and previous versions, explicitly retained versions, the newest
        ``keep_latest`` versions, and versions referenced by pending proposals are
        always retained. Proposal JSON remains as audit history after rejected or
        old accepted payloads become eligible for collection.
        """
        if keep_latest < 0:
            raise param_error("keep_latest must be non-negative")
        explicit = set(keep_versions)
        for version_id in explicit:
            self._validate_version_id(version_id)

        try:
            with self._lock:
                for version_id in explicit:
                    self._resolve_unlocked(version_id, verify=True)
                version_ids = self._list_version_ids()
                retained = set(explicit)
                pointer = self._read_active_pointer(required=False)
                if pointer:
                    retained.add(pointer["active_version_id"])
                    if pointer.get("previous_version_id"):
                        retained.add(pointer["previous_version_id"])
                if keep_latest:
                    retained.update(version_ids[-keep_latest:])
                for proposal in self.list_proposals():
                    if proposal.status is not ProposalStatus.PENDING:
                        continue
                    retained.add(proposal.parent_version_id)
                    if proposal.candidate_version_id:
                        retained.add(proposal.candidate_version_id)

                removed: list[str] = []
                for version_id in version_ids:
                    if version_id in retained:
                        continue
                    self._remove_tree(self.versions_dir / version_id)
                    removed.append(version_id)
                return removed
        except BaseError:
            raise
        except Exception as exc:
            raise execution_error(f"failed to garbage-collect skill bank: {exc}", cause=exc)

    def _resolve_unlocked(self, version_id: str, *, verify: bool) -> BankVersionRef:
        self._validate_version_id(version_id)
        version_dir = self.versions_dir / version_id
        manifest_path = version_dir / _MANIFEST_FILE
        skills_dir = version_dir / "skills"
        if not manifest_path.is_file() or not skills_dir.is_dir():
            raise not_found_error(f"skill-bank version '{version_id}' does not exist")
        try:
            manifest = SkillBankManifest.from_dict(self._read_json(manifest_path))
        except BaseError as exc:
            raise integrity_error(f"invalid manifest for skill-bank version '{version_id}'", cause=exc)
        except Exception as exc:
            raise integrity_error(f"invalid manifest for skill-bank version '{version_id}': {exc}", cause=exc)
        if manifest.schema_version != 1:
            raise integrity_error(
                f"unsupported manifest schema {manifest.schema_version} for skill-bank version '{version_id}'"
            )
        if manifest.version_id != version_id:
            raise integrity_error(f"manifest version mismatch for '{version_id}'")
        if verify:
            digest, file_count, total_bytes = self._digest_directory(skills_dir)
            if (
                digest != manifest.content_sha256
                or file_count != manifest.file_count
                or total_bytes != manifest.total_bytes
            ):
                raise integrity_error(f"snapshot payload for '{version_id}' does not match its manifest")
            try:
                skill_names, disabled_skills = self._validate_snapshot_source(skills_dir)
            except BaseError as exc:
                raise integrity_error(f"snapshot payload for '{version_id}' is invalid", cause=exc)
            if tuple(skill_names) != manifest.skill_names or tuple(disabled_skills) != manifest.disabled_skills:
                raise integrity_error(f"effective skill state for '{version_id}' does not match its manifest")
            if manifest.sources:
                source_names = [item.skill_name for item in manifest.sources]
                if len(source_names) != len(set(source_names)) or sorted(source_names) != skill_names:
                    raise integrity_error(f"source provenance for '{version_id}' does not match its skills")
            self._assert_read_only(version_dir, version_id)
        return BankVersionRef(version_id=version_id, version_dir=version_dir, manifest=manifest)

    def _promote_unlocked(self, ref: BankVersionRef) -> BankVersionRef:
        current = self._read_active_pointer(required=False)
        if current is not None and current["active_version_id"] == ref.version_id:
            return ref
        pointer = {
            "active_version_id": ref.version_id,
            "previous_version_id": current["active_version_id"] if current else None,
            "updated_at": utc_now_iso(),
        }
        self._write_json_atomic(self.active_path, pointer)
        return ref

    @staticmethod
    def _validate_candidate_linkage(proposal: SkillBankProposal, candidate: BankVersionRef) -> None:
        if candidate.manifest.parent_version_id != proposal.parent_version_id:
            raise param_error("candidate version parent does not match the proposal parent")
        if candidate.manifest.proposal_id != proposal.proposal_id:
            raise param_error("candidate version proposal_id does not match the proposal audit record")

    def _allocate_version_id_unlocked(self) -> str:
        highest = max((self._version_number(item) for item in self._list_version_ids()), default=0)
        counter = 0
        if self.counter_path.is_file():
            try:
                counter = int(self.counter_path.read_text(encoding="utf-8").strip() or "0")
            except (OSError, ValueError) as exc:
                raise integrity_error(f"invalid version counter at {self.counter_path}: {exc}", cause=exc)
        next_number = max(highest, counter) + 1
        self._write_text_atomic(self.counter_path, f"{next_number}\n")
        return f"bank_{next_number:06d}"

    def _read_active_pointer(self, *, required: bool = True) -> dict[str, Any] | None:
        if not self.active_path.is_file():
            if required:
                raise not_found_error("no active skill-bank version is configured")
            return None
        try:
            pointer = self._read_json(self.active_path)
        except Exception as exc:
            raise integrity_error(f"invalid active skill-bank pointer: {exc}", cause=exc)
        active = str(pointer.get("active_version_id", ""))
        previous = pointer.get("previous_version_id")
        try:
            self._validate_version_id(active)
            if previous is not None:
                self._validate_version_id(str(previous))
        except BaseError as exc:
            raise integrity_error("active skill-bank pointer contains an invalid version id", cause=exc)
        return pointer

    def _validate_snapshot_source(self, source: Path) -> tuple[list[str], list[str]]:
        if not source.is_dir():
            raise param_error(f"skills directory does not exist or is not a directory: {source}")
        if self.root == source or self.root.is_relative_to(source):
            raise param_error("bank_root cannot be inside the skills directory being snapshotted")

        paths = sorted(source.rglob("*"), key=lambda item: item.relative_to(source).as_posix())
        for path in paths:
            if is_transient_skill_path(path.relative_to(source)):
                continue
            if path.is_symlink():
                raise param_error(f"skill banks cannot contain symbolic links: {path.relative_to(source)}")
            if not path.is_dir() and not path.is_file():
                raise param_error(f"skill banks can contain only regular files and directories: {path}")

        skill_names: list[str] = []
        for path in sorted(source.iterdir(), key=lambda item: item.name):
            if is_transient_skill_path(path.relative_to(source)):
                continue
            if not path.is_dir():
                continue
            validate_safe_name(path.name, field="skill name")
            if not (path / "SKILL.md").is_file():
                raise param_error(f"skill '{path.name}' is missing SKILL.md")
            skill_names.append(path.name)

        disabled_skills: list[str] = []
        state_path = source / "skills_state.json"
        if state_path.exists():
            try:
                state = self._read_json(state_path)
            except Exception as exc:
                raise param_error(f"invalid skills_state.json: {exc}", cause=exc)
            skill_configs = state.get("skill_configs", {})
            if not isinstance(skill_configs, dict):
                raise param_error("skills_state.json skill_configs must be an object")
            for name, config in skill_configs.items():
                validate_safe_name(str(name), field="skills_state skill name")
                if not isinstance(config, dict):
                    raise param_error(f"skills_state config for '{name}' must be an object")
                if "enabled" in config and not isinstance(config["enabled"], bool):
                    raise param_error(f"skills_state enabled flag for '{name}' must be a boolean")
                if config.get("enabled") is False:
                    disabled_skills.append(str(name))
        return skill_names, sorted(disabled_skills)

    @staticmethod
    def _digest_directory(root: Path) -> tuple[str, int, int]:
        """Hash the complete bank tree, including normalized file and directory modes."""
        digest = hashlib.sha256()
        file_count = 0
        total_bytes = 0
        entries = sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix())
        for path in entries:
            relative_path = path.relative_to(root)
            if is_transient_skill_path(relative_path):
                continue
            relative = relative_path.as_posix().encode("utf-8")
            if path.is_symlink():
                raise integrity_error(f"snapshot contains symbolic link: {path.relative_to(root)}")
            normalized_mode = stat.S_IMODE(path.stat().st_mode) & ~0o222
            if path.is_dir():
                digest.update(b"D\0" + relative + b"\0" + str(normalized_mode).encode("ascii") + b"\0")
                continue
            if not path.is_file():
                raise integrity_error(f"snapshot contains non-regular file: {path.relative_to(root)}")
            size = path.stat().st_size
            digest.update(
                b"F\0"
                + relative
                + b"\0"
                + str(normalized_mode).encode("ascii")
                + b"\0"
                + str(size).encode("ascii")
                + b"\0"
            )
            with path.open("rb") as handle:
                while chunk := handle.read(1024 * 1024):
                    digest.update(chunk)
            digest.update(b"\0")
            file_count += 1
            total_bytes += size
        return digest.hexdigest(), file_count, total_bytes

    @staticmethod
    def _make_read_only(root: Path) -> None:
        entries = sorted(root.rglob("*"), key=lambda item: len(item.parts), reverse=True)
        for path in entries:
            if path.is_symlink():
                continue
            path.chmod(stat.S_IMODE(path.stat().st_mode) & ~0o222)
        root.chmod(stat.S_IMODE(root.stat().st_mode) & ~0o222)

    @staticmethod
    def _assert_read_only(root: Path, version_id: str) -> None:
        for path in (root, *root.rglob("*")):
            if path.is_symlink() or stat.S_IMODE(path.stat().st_mode) & 0o222:
                raise integrity_error(f"snapshot '{version_id}' is not immutable at {path}")

    @staticmethod
    def _remove_tree(path: Path) -> None:
        if not path.exists() and not path.is_symlink():
            return
        if path.is_symlink() or path.is_file():
            with suppress(OSError):
                path.chmod(stat.S_IMODE(path.lstat().st_mode) | stat.S_IWUSR)
            path.unlink()
            return
        for item in path.rglob("*"):
            if item.is_symlink():
                continue
            with suppress(OSError):
                item.chmod(stat.S_IMODE(item.stat().st_mode) | stat.S_IWUSR)
        with suppress(OSError):
            path.chmod(stat.S_IMODE(path.stat().st_mode) | stat.S_IWUSR)
        shutil.rmtree(path)

    def _list_version_ids(self) -> list[str]:
        return sorted(
            (
                path.name
                for path in self.versions_dir.iterdir()
                if not path.is_symlink() and path.is_dir() and _VERSION_PATTERN.fullmatch(path.name)
            ),
            key=self._version_number,
        )

    @staticmethod
    def _version_number(version_id: str) -> int:
        return int(version_id.removeprefix("bank_"))

    def _proposal_path(self, proposal_id: str) -> Path:
        return self.proposals_dir / f"{proposal_id}.json"

    @staticmethod
    def _write_json(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(dump_json_text(payload), encoding="utf-8")

    def _write_json_atomic(self, path: Path, payload: dict[str, Any]) -> None:
        self._write_text_atomic(path, dump_json_text(payload))

    def _write_text_atomic(self, path: Path, content: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        try:
            with temporary.open("x", encoding="utf-8") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            with suppress(OSError):
                temporary.unlink()

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise integrity_error(f"expected a JSON object in {path}")
        return data

    @staticmethod
    def _validate_version_id(version_id: str) -> None:
        if not isinstance(version_id, str) or not _VERSION_PATTERN.fullmatch(version_id):
            raise param_error(f"invalid skill-bank version id: {version_id!r}")


__all__ = ["SkillBankStore"]

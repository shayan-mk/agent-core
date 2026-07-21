# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Tests for immutable whole-bank snapshots and proposal audit."""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path

import pytest

from openjiuwen.agent_evolving.skill_bank import ProposalStatus, SkillBankProposal, SkillBankStore
from openjiuwen.agent_evolving.skill_bank.models import (
    EvaluationEvidence,
    SkillOperation,
    SkillOperationType,
)
from openjiuwen.core.common.exception.codes import StatusCode
from openjiuwen.core.common.exception.errors import BaseError


def _write_bank(root: Path, body: str, *, disabled: bool = False) -> Path:
    skill_dir = root / "calculator"
    scripts_dir = skill_dir / "scripts"
    evolution_dir = skill_dir / "evolution"
    archive_dir = skill_dir / "archive"
    scripts_dir.mkdir(parents=True)
    evolution_dir.mkdir()
    archive_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        "---\nname: calculator\ndescription: Calculate values\n---\n\n" + body,
        encoding="utf-8",
    )
    script_path = scripts_dir / "calculate.sh"
    script_path.write_text("#!/bin/sh\necho 42\n", encoding="utf-8")
    script_path.chmod(0o755)
    (skill_dir / "evolutions.json").write_text(
        json.dumps({"skill_id": "calculator", "entries": []}),
        encoding="utf-8",
    )
    (evolution_dir / "troubleshooting.md").write_text("# Troubleshooting\n", encoding="utf-8")
    (archive_dir / "SKILL.v20260713T000000.md").write_text(body, encoding="utf-8")
    (archive_dir / "evolutions.v20260713T000000.json").write_text(
        json.dumps({"skill_id": "calculator", "entries": []}),
        encoding="utf-8",
    )
    (root / "skills_state.json").write_text(
        json.dumps({"skill_configs": {"calculator": {"enabled": not disabled}}}),
        encoding="utf-8",
    )
    return root


def _payload_bytes(root: Path) -> dict[str, bytes]:
    return {path.relative_to(root).as_posix(): path.read_bytes() for path in sorted(root.rglob("*")) if path.is_file()}


def test_snapshot_is_complete_deterministic_and_read_only(tmp_path: Path) -> None:
    source = _write_bank(tmp_path / "source", "Use the baseline algorithm.", disabled=True)
    cache_dir = source / "calculator" / "scripts" / "__pycache__"
    cache_dir.mkdir()
    (cache_dir / "calculate.cpython-311.pyc").write_bytes(b"first cache")
    git_dir = source / "calculator" / ".git"
    git_dir.mkdir()
    (git_dir / "config").write_text("local repository state", encoding="utf-8")
    store = SkillBankStore(tmp_path / "bank")

    first = store.create_snapshot(source)
    before = _payload_bytes(first.skills_dir)
    resolved = store.resolve(first.version_id)
    (cache_dir / "calculate.cpython-311.pyc").write_bytes(b"different cache")
    second = store.create_snapshot(source, parent_version_id=first.version_id)

    assert resolved.manifest == first.manifest
    assert _payload_bytes(resolved.skills_dir) == before
    assert second.manifest.content_sha256 == first.manifest.content_sha256
    assert first.manifest.skill_names == ("calculator",)
    assert first.manifest.disabled_skills == ("calculator",)
    assert (first.skills_dir / "calculator" / "scripts" / "calculate.sh").is_file()
    assert (first.skills_dir / "calculator" / "evolutions.json").is_file()
    assert (first.skills_dir / "calculator" / "evolution" / "troubleshooting.md").is_file()
    assert (first.skills_dir / "calculator" / "archive" / "SKILL.v20260713T000000.md").is_file()
    assert not (first.skills_dir / "calculator" / "scripts" / "__pycache__").exists()
    assert not (first.skills_dir / "calculator" / ".git").exists()

    script_mode = stat.S_IMODE((first.skills_dir / "calculator" / "scripts" / "calculate.sh").stat().st_mode)
    assert script_mode & stat.S_IXUSR
    for path in (first.version_dir, *first.version_dir.rglob("*")):
        assert stat.S_IMODE(path.stat().st_mode) & 0o222 == 0


def test_snapshot_rejects_package_symlinks(tmp_path: Path) -> None:
    store = SkillBankStore(tmp_path / "bank")
    source = _write_bank(tmp_path / "source", "body")
    outside = tmp_path / "outside.txt"
    outside.write_text("outside", encoding="utf-8")
    os.symlink(outside, source / "calculator" / "outside-link")

    with pytest.raises(BaseError) as exc_info:
        store.create_snapshot(source)
    assert exc_info.value.status == StatusCode.TOOLCHAIN_EVOLVING_SKILL_BANK_PARAM_ERROR


def test_resolution_detects_snapshot_mutation(tmp_path: Path) -> None:
    source = _write_bank(tmp_path / "source", "original")
    store = SkillBankStore(tmp_path / "bank")
    version = store.create_snapshot(source)
    skill_file = version.skills_dir / "calculator" / "SKILL.md"
    skill_file.chmod(0o644)
    skill_file.write_text("modified", encoding="utf-8")
    skill_file.chmod(0o444)

    with pytest.raises(BaseError) as exc_info:
        store.resolve(version.version_id)
    assert exc_info.value.status == StatusCode.TOOLCHAIN_EVOLVING_SKILL_BANK_INTEGRITY_INVALID


def test_promotion_and_rollback_are_atomic(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = _write_bank(tmp_path / "source", "baseline")
    store = SkillBankStore(tmp_path / "bank")
    baseline = store.create_snapshot(source)
    (source / "calculator" / "SKILL.md").write_text("candidate", encoding="utf-8")
    candidate = store.create_snapshot(source, parent_version_id=baseline.version_id)

    store.promote(baseline.version_id)
    real_replace = os.replace

    def fail_active_replace(source_path, destination_path) -> None:
        if Path(destination_path) == store.active_path:
            raise OSError("injected activation failure")
        real_replace(source_path, destination_path)

    with monkeypatch.context() as patcher:
        patcher.setattr(os, "replace", fail_active_replace)
        with pytest.raises(BaseError) as exc_info:
            store.promote(candidate.version_id)
        assert exc_info.value.status == StatusCode.TOOLCHAIN_EVOLVING_SKILL_BANK_EXECUTION_ERROR
        assert store.resolve().version_id == baseline.version_id

    store.promote(candidate.version_id)
    assert store.rollback().version_id == baseline.version_id


def test_proposal_audit_and_conservative_garbage_collection(tmp_path: Path) -> None:
    source = _write_bank(tmp_path / "source", "version 1")
    store = SkillBankStore(tmp_path / "bank")
    versions = []
    parent = None
    proposal_id = "proposal_001"
    for index in range(1, 5):
        (source / "calculator" / "SKILL.md").write_text(f"version {index}", encoding="utf-8")
        if index == 4:
            version = store.create_snapshot(
                source,
                parent_version_id=versions[0].version_id,
                proposal_id=proposal_id,
            )
        else:
            version = store.create_snapshot(source, parent_version_id=parent)
        versions.append(version)
        parent = version.version_id

    store.promote(versions[1].version_id)
    store.promote(versions[2].version_id)
    proposal = SkillBankProposal(
        proposal_id=proposal_id,
        parent_version_id=versions[0].version_id,
        candidate_version_id=versions[3].version_id,
        operations=(
            SkillOperation(
                action=SkillOperationType.MODIFY,
                skill_name="calculator",
                summary="Use a safer calculation flow",
            ),
        ),
        source_trajectory_ids=("trajectory-1",),
        creator="test-creator",
        model="test-model",
    )
    store.record_proposal(proposal)

    assert store.record_proposal(proposal) == proposal

    assert store.garbage_collect(keep_latest=0) == []
    evidence = EvaluationEvidence(
        task_id="task-1",
        version_id=versions[3].version_id,
        reward=0.0,
        success=False,
        rollout_id="rollout-1",
    )
    decided = store.decide_proposal(
        proposal.proposal_id,
        ProposalStatus.REJECTED,
        evidence=(evidence,),
        reason="candidate regressed",
    )
    assert decided.status is ProposalStatus.REJECTED
    assert store.get_proposal(proposal.proposal_id) == decided
    with pytest.raises(BaseError) as exc_info:
        store.decide_proposal(proposal.proposal_id, ProposalStatus.ACCEPTED)
    assert exc_info.value.status == StatusCode.TOOLCHAIN_EVOLVING_SKILL_BANK_PARAM_ERROR

    removed = store.garbage_collect(keep_latest=0)
    assert removed == [versions[0].version_id, versions[3].version_id]
    assert store.get_proposal(proposal.proposal_id).decision_reason == "candidate regressed"

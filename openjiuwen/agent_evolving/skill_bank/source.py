# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Materialize existing ordered skill roots into one effective local bank."""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
import uuid
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from openjiuwen.agent_evolving.skill_bank.models import (
    BankVersionRef,
    SkillSourceProvenance,
    TRANSIENT_SKILL_DIR_NAMES,
    dump_json_text,
    execution_error,
    is_skill_package,
    is_transient_skill_path,
    param_error as _param,
)
from openjiuwen.agent_evolving.skill_bank.store import SkillBankStore
from openjiuwen.core.common.exception.errors import BaseError
from openjiuwen.harness.skill_sources import collect_disabled_skills_from_state


@dataclass(frozen=True)
class SkillSource:
    """One ordered, already-local source of ordinary skill packages."""

    root: Path
    source_kind: str = "local"
    location: str | None = None
    revision: str | None = None
    license: str | None = None

    def __post_init__(self) -> None:
        root = Path(self.root).expanduser().resolve()
        source_kind = str(self.source_kind).strip()
        if not source_kind:
            raise _param("skill source requires source_kind")
        object.__setattr__(self, "root", root)
        object.__setattr__(self, "source_kind", source_kind)
        object.__setattr__(self, "location", str(self.location or root))


@dataclass(frozen=True)
class MaterializedSkillBank:
    """Canonical first-root-wins view ready for immutable snapshotting."""

    skills_dir: Path
    skill_names: tuple[str, ...]
    disabled_skills: tuple[str, ...]
    sources: tuple[SkillSourceProvenance, ...]


def materialize_effective_skill_bank(
    sources: Iterable[SkillSource | str | Path],
    destination: str | Path,
) -> MaterializedSkillBank:
    """Copy the effective view of ordered roots into one canonical directory.

    This mirrors ``SkillUseRail`` precedence: roots are visited in order and
    duplicate skill directory names keep the first package. Missing roots are
    ignored. Disabled state is the union consumed by the harness factory.
    """
    normalized = tuple(_normalize_source(source) for source in sources)
    destination_path = Path(destination).expanduser().resolve()
    if destination_path.exists() or destination_path.is_symlink():
        raise _param(f"materialization destination already exists: {destination_path}")

    destination_path.parent.mkdir(parents=True, exist_ok=True)
    staging = destination_path.with_name(f".{destination_path.name}.{uuid.uuid4().hex}.tmp")
    selected_names: set[str] = set()
    provenance: list[SkillSourceProvenance] = []
    try:
        staging.mkdir()
        for source in normalized:
            if not source.root.exists():
                continue
            if not source.root.is_dir():
                raise _param(f"skill source is not a directory: {source.root}")
            _reject_nested_destination(source.root, destination_path)
            for package in sorted(source.root.iterdir(), key=lambda item: item.name):
                if not is_skill_package(package):
                    continue
                if package.name in selected_names:
                    continue
                _validate_package(package)
                shutil.copytree(
                    package,
                    staging / package.name,
                    copy_function=shutil.copy2,
                    ignore=shutil.ignore_patterns(*TRANSIENT_SKILL_DIR_NAMES),
                )
                selected_names.add(package.name)
                provenance.append(
                    SkillSourceProvenance(
                        skill_name=package.name,
                        source_kind=source.source_kind,
                        location=source.location,
                        revision=source.revision,
                        content_sha256=_digest_package(staging / package.name),
                        license=source.license,
                    )
                )

        disabled = tuple(collect_disabled_skills_from_state([str(source.root) for source in normalized]))
        if disabled:
            state = {"skill_configs": {name: {"enabled": False} for name in disabled}}
            (staging / "skills_state.json").write_text(dump_json_text(state), encoding="utf-8")
        os.replace(staging, destination_path)
    except Exception as exc:
        with suppress(OSError):
            shutil.rmtree(staging)
        if isinstance(exc, BaseError):
            raise
        raise execution_error(f"failed to materialize effective skill bank: {exc}", cause=exc) from exc

    return MaterializedSkillBank(
        skills_dir=destination_path,
        skill_names=tuple(sorted(selected_names)),
        disabled_skills=disabled,
        sources=tuple(sorted(provenance, key=lambda item: item.skill_name)),
    )


def initialize_skill_bank(
    store: SkillBankStore,
    sources: Iterable[SkillSource | str | Path],
) -> BankVersionRef:
    """Materialize existing sources and activate the initial immutable version."""
    with tempfile.TemporaryDirectory(prefix="openjiuwen-skill-bank-") as temporary:
        materialized = materialize_effective_skill_bank(sources, Path(temporary) / "skills")
        version = store.create_snapshot(materialized.skills_dir, sources=materialized.sources)
    return store.promote(version.version_id)


def build_skill_provenance(
    package: str | Path,
    *,
    skill_name: str | None = None,
    source_kind: str = "local",
    location: str | None = None,
    revision: str | None = None,
    license: str | None = None,
) -> SkillSourceProvenance:
    """Build immutable source metadata for one local skill package."""
    package_path = validate_skill_package(package)
    return SkillSourceProvenance(
        skill_name=skill_name or package_path.name,
        source_kind=source_kind,
        location=location or str(package_path),
        revision=revision,
        content_sha256=_digest_package(package_path),
        license=license,
    )


def validate_skill_package(package: str | Path) -> Path:
    """Validate a local package without computing its provenance digest."""
    package_path = Path(package).expanduser().resolve()
    if not is_skill_package(package_path):
        raise _param(f"skill package is missing a directory or SKILL.md: {package_path}")
    _validate_package(package_path)
    return package_path


def _normalize_source(source: SkillSource | str | Path) -> SkillSource:
    return source if isinstance(source, SkillSource) else SkillSource(root=Path(source))


def _validate_package(package: Path) -> None:
    for path in (package, *package.rglob("*")):
        if path != package and is_transient_skill_path(path.relative_to(package)):
            continue
        if path.is_symlink():
            raise _param(f"skill packages cannot contain symbolic links: {path}")
        if not path.is_dir() and not path.is_file():
            raise _param(f"skill packages can contain only regular files and directories: {path}")


def _reject_nested_destination(source_root: Path, destination: Path) -> None:
    if destination == source_root or destination.is_relative_to(source_root):
        raise _param("materialization destination cannot be inside a source root")


def _digest_package(package: Path) -> str:
    """Hash package paths and bytes for provenance, intentionally ignoring modes."""
    digest = hashlib.sha256()
    for path in sorted(package.rglob("*"), key=lambda item: item.relative_to(package).as_posix()):
        if is_transient_skill_path(path.relative_to(package)):
            continue
        if not path.is_file():
            continue
        relative = path.relative_to(package).as_posix().encode("utf-8")
        size = path.stat().st_size
        digest.update(relative + b"\0" + str(size).encode("ascii") + b"\0")
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


__all__ = [
    "MaterializedSkillBank",
    "SkillSource",
    "build_skill_provenance",
    "initialize_skill_bank",
    "materialize_effective_skill_bank",
    "validate_skill_package",
]

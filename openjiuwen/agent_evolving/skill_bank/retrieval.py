# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Cheap task-level skill retrieval for comparison against ``all``/``auto_list``.

Produces an ``enabled_skills`` allow-list for the existing harness assembly
(``skills=`` on ``resolve_deep_agent_parts``); it does not introduce a second
skill loading path. It provides the BM25 baseline described in the proposal;
the retrieval budget remains caller-configurable.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from pathlib import Path

from openjiuwen.agent_evolving.skill_bank.models import (
    BankVersionRef,
    is_skill_package,
    param_error as _param,
)
from openjiuwen.agent_evolving.utils import parse_top_level_frontmatter

_TOKEN_RE = re.compile(r"[a-z0-9]+|[\u3400-\u4dbf\u4e00-\u9fff]")
_BODY_CHARS = 2000
_K1 = 1.5
_B = 0.75


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class Bm25SkillRetriever:
    """Rank a bank's skills against one task query with BM25 (Okapi).

    Indexes each package's SKILL.md name, frontmatter description, and leading
    body text once at construction; ``top_skills`` then scores a task query
    and returns at most ``k`` skill names for use as a harness allow-list.
    """

    def __init__(self, skills_dir: str | Path | BankVersionRef) -> None:
        root = Path(skills_dir.skills_dir if isinstance(skills_dir, BankVersionRef) else skills_dir)
        if not root.is_dir():
            raise _param(f"skills directory does not exist: {root}")
        self._documents: dict[str, tuple[Counter[str], int]] = {}
        self._general_skills: set[str] = set()
        for package in sorted(root.iterdir(), key=lambda item: item.name):
            if not is_skill_package(package):
                continue
            content = (package / "SKILL.md").read_text(encoding="utf-8", errors="replace")
            frontmatter = parse_top_level_frontmatter(content)
            if frontmatter.get("scope", "").strip().lower() == "general":
                self._general_skills.add(package.name)
            # The frontmatter name/description also appear inside the body
            # slice; repeating them is a deliberate boost for the two
            # highest-signal fields.
            tokens = _tokenize(
                " ".join((package.name, frontmatter.get("name", ""), frontmatter.get("description", ""),
                          content[:_BODY_CHARS]))
            )
            self._documents[package.name] = (Counter(tokens), len(tokens))
        self._average_length = (
            sum(length for _, length in self._documents.values()) / len(self._documents)
            if self._documents
            else 0.0
        )
        self._document_frequency: Counter[str] = Counter()
        for counter, _ in self._documents.values():
            self._document_frequency.update(counter.keys())

    def top_skills(self, query: str, *, k: int = 6) -> list[str]:
        """Return general skills plus up to ``k`` ranked task-specific skills."""
        if k <= 0:
            raise _param("retrieval k must be positive")
        general = sorted(self._general_skills)
        query_terms = _tokenize(query)
        if not query_terms or not self._documents:
            return general
        total = len(self._documents)
        idf_by_term = {
            term: math.log(1 + (total - self._document_frequency[term] + 0.5) / (self._document_frequency[term] + 0.5))
            for term in set(query_terms)
        }
        scored: list[tuple[float, str]] = []
        for name, (counter, length) in self._documents.items():
            if name in self._general_skills:
                continue
            score = 0.0
            length_norm = _K1 * (1 - _B + _B * length / (self._average_length or 1.0))
            for term in query_terms:
                frequency = counter.get(term, 0)
                if not frequency:
                    continue
                score += idf_by_term[term] * frequency * (_K1 + 1) / (frequency + length_norm)
            if score > 0.0:
                scored.append((score, name))
        scored.sort(key=lambda item: (-item[0], item[1]))
        return general + [name for _, name in scored[:k]]


__all__ = ["Bm25SkillRetriever"]

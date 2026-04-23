# Bayan-Graph Plan 1 — Foundation + Walking Skeleton Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stand up the Bayan-Graph repo with typed data models, Arabic normalization, a Quran-only ingestion path, real embedding + retrieval + Citations-based validation, and a stubbed LangGraph that returns a schema-valid answer end-to-end for a single query.

**Architecture:** A Python 3.12 package (`bayan_graph/`) managed with `uv`. Pydantic v2 models define the canonical answer schema. A LangGraph `StateGraph` wires stub nodes so the walking skeleton produces a complete `AnswerResponse` JSON from a hard-coded query; only retrieval, embedding, and citation validation are real in Plan 1. Everything else is a typed stub that satisfies the schema so Plan 3 can replace stubs without refactoring.

**Tech Stack:** Python 3.12, uv, Pydantic v2, LangGraph, Anthropic SDK (Citations API), Cohere SDK (embed-multilingual-v3.0, rerank-multilingual-v3.0), Pinecone SDK, pytest, ruff, pre-commit.

**Subagent / main-thread tagging:** Each task below is tagged. `subagent` tasks are self-contained and dispatchable to a fresh Sonnet subagent. `main-thread` tasks require orchestrator context (installs, credentials, cross-task integration checks, commits spanning multiple subagent outputs).

**Parallelization groups:** Tasks within the same `[PAR-GROUP N]` label have no dependency on each other and may be dispatched concurrently. All tasks outside a group are sequential.

---

## Phase A — Repo scaffolding

### Task A1: Initialize repository with uv and pyproject.toml — `main-thread`

**Files:**
- Create: `pyproject.toml`
- Create: `uv.lock` (generated)
- Create: `.python-version`

- [ ] **Step 1: Confirm uv is installed**

Run: `uv --version`
Expected: `uv 0.4.0` or newer. If missing, install from https://docs.astral.sh/uv/.

- [ ] **Step 2: Write `.python-version`**

```
3.12
```

- [ ] **Step 3: Write `pyproject.toml`**

```toml
[project]
name = "bayan-graph"
version = "0.1.0"
description = "Multi-madhab comparative fiqh engine with enforced citations."
requires-python = ">=3.12,<3.13"
readme = "README.md"
dependencies = [
    "anthropic>=0.39.0",
    "cohere>=5.11.0",
    "pinecone>=5.0.0",
    "langgraph>=0.2.50",
    "langchain-core>=0.3.0",
    "pydantic>=2.9.0",
    "httpx>=0.27.0",
    "tenacity>=9.0.0",
    "python-dotenv>=1.0.0",
    "structlog>=24.4.0",
    "typer>=0.12.0",
    "pyarabic>=0.6.15",
]

[dependency-groups]
dev = [
    "pytest>=8.3.0",
    "pytest-asyncio>=0.24.0",
    "pytest-cov>=5.0.0",
    "ruff>=0.7.0",
    "mypy>=1.11.0",
    "pre-commit>=4.0.0",
    "respx>=0.21.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/bayan_graph"]

[tool.ruff]
line-length = 100
target-version = "py312"

[tool.ruff.lint]
select = ["E", "F", "I", "B", "UP", "N", "SIM"]
ignore = ["E501"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-ra -q --strict-markers"
asyncio_mode = "auto"
markers = [
    "integration: requires external API keys (ANTHROPIC_API_KEY, COHERE_API_KEY, PINECONE_API_KEY)",
    "slow: takes > 5s",
]

[tool.mypy]
python_version = "3.12"
strict = true
files = ["src/bayan_graph"]
```

- [ ] **Step 4: Sync the environment**

Run: `uv sync`
Expected: creates `.venv/` and `uv.lock`; all dependencies resolve.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml uv.lock .python-version
git commit -m "chore: scaffold uv project with core dependencies"
```

---

### Task A2: Create source/test directory structure — `subagent` [PAR-GROUP 1]

**Files:**
- Create: `src/bayan_graph/__init__.py`
- Create: `src/bayan_graph/models/__init__.py`
- Create: `src/bayan_graph/ingestion/__init__.py`
- Create: `src/bayan_graph/retrieval/__init__.py`
- Create: `src/bayan_graph/graph/__init__.py`
- Create: `src/bayan_graph/graph/nodes/__init__.py`
- Create: `src/bayan_graph/cli/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`

- [ ] **Step 1: Write `src/bayan_graph/__init__.py`**

```python
"""Bayan-Graph: multi-madhab comparative fiqh engine."""

__version__ = "0.1.0"
```

- [ ] **Step 2: Write empty `__init__.py` for each submodule**

Each of `models/`, `ingestion/`, `retrieval/`, `graph/`, `graph/nodes/`, `cli/` gets an empty `__init__.py`.

- [ ] **Step 3: Write `tests/__init__.py` (empty)**

- [ ] **Step 4: Write `tests/conftest.py`**

```python
import os
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def fixtures_dir(repo_root: Path) -> Path:
    return repo_root / "tests" / "fixtures"


def pytest_collection_modifyitems(config, items):
    required_keys = ("ANTHROPIC_API_KEY", "COHERE_API_KEY", "PINECONE_API_KEY")
    has_keys = all(os.getenv(k) for k in required_keys)
    if has_keys:
        return
    skip_integration = pytest.mark.skip(reason="missing required API keys")
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip_integration)
```

- [ ] **Step 5: Verify pytest discovery**

Run: `uv run pytest --collect-only -q`
Expected: collects 0 tests, no errors.

- [ ] **Step 6: Commit**

```bash
git add src/bayan_graph tests/__init__.py tests/conftest.py
git commit -m "chore: create package layout and pytest conftest"
```

---

### Task A3: Add .gitignore, pre-commit, and Makefile — `subagent` [PAR-GROUP 1]

**Files:**
- Create: `.gitignore`
- Create: `.pre-commit-config.yaml`
- Create: `Makefile`
- Create: `.env.example`

- [ ] **Step 1: Write `.gitignore`**

```
# Python
__pycache__/
*.py[cod]
*.egg-info/
.venv/
.pytest_cache/
.mypy_cache/
.ruff_cache/
htmlcov/
.coverage

# Env
.env
.env.local

# Data
data/raw/
data/processed/
*.sqlite
*.db

# IDE
.idea/
.vscode/
*.swp

# OS
.DS_Store
Thumbs.db
```

- [ ] **Step 2: Write `.pre-commit-config.yaml`**

```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.7.0
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v5.0.0
    hooks:
      - id: end-of-file-fixer
      - id: trailing-whitespace
      - id: check-yaml
      - id: check-toml
      - id: check-added-large-files
        args: ["--maxkb=500"]
```

- [ ] **Step 3: Write `Makefile`**

```makefile
.PHONY: install test lint fmt typecheck ci clean

install:
	uv sync
	uv run pre-commit install

test:
	uv run pytest

test-integration:
	uv run pytest -m integration

lint:
	uv run ruff check src tests

fmt:
	uv run ruff format src tests
	uv run ruff check --fix src tests

typecheck:
	uv run mypy src/bayan_graph

ci: lint typecheck test

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage
```

- [ ] **Step 4: Write `.env.example`**

```
ANTHROPIC_API_KEY=sk-ant-...
COHERE_API_KEY=...
PINECONE_API_KEY=...
PINECONE_INDEX=bayan-graph
LANGSMITH_API_KEY=
LANGSMITH_PROJECT=bayan-graph
```

- [ ] **Step 5: Install pre-commit hook and run once**

Run: `uv run pre-commit install && uv run pre-commit run --all-files`
Expected: all hooks pass (or auto-fix and pass on re-run).

- [ ] **Step 6: Commit**

```bash
git add .gitignore .pre-commit-config.yaml Makefile .env.example
git commit -m "chore: add gitignore, pre-commit, Makefile, env template"
```

---

## Phase B — Core Pydantic models

All Phase B tasks follow strict TDD: write failing test → run to fail → implement → run to pass → commit.

### Task B1: Controlled vocabularies — `subagent` [PAR-GROUP 2]

**Files:**
- Create: `tests/models/test_vocabs.py`
- Create: `src/bayan_graph/models/vocabs.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/models/test_vocabs.py
from bayan_graph.models.vocabs import (
    AuthorityLevel,
    EvidenceRole,
    EvidenceType,
    IkhtilafReasonType,
    RoleInSource,
    TawaturStatus,
)


def test_evidence_type_has_required_members():
    assert EvidenceType.QURAN.value == "quran"
    assert EvidenceType.HADITH.value == "hadith"
    assert EvidenceType.IJMA.value == "ijma"
    assert EvidenceType.QIYAS.value == "qiyas"
    assert EvidenceType.MADHAB_TEXT.value == "madhab_text"


def test_evidence_role_has_required_members():
    assert EvidenceRole.PRIMARY_SUPPORT.value == "primary_support"
    assert EvidenceRole.CORROBORATING.value == "corroborating"
    assert EvidenceRole.OPPOSING_CITED.value == "opposing_cited"


def test_authority_level_members():
    for v in ("mu_tamad", "mashhur", "marjuh", "shadh", "disputed_within_madhab"):
        assert any(m.value == v for m in AuthorityLevel)


def test_role_in_source_members():
    for v in ("primary_ruling", "opposing_view_cited", "opposing_view_cited_and_refuted",
              "comparative_discussion", "conditional_ruling", "narrative_commentary"):
        assert any(m.value == v for m in RoleInSource)


def test_tawatur_status_members():
    for v in ("mutawatir_lafzi", "mutawatir_manawi", "mashhur", "aziz", "gharib", "ahad"):
        assert any(m.value == v for m in TawaturStatus)


def test_ikhtilaf_reason_type_members():
    for v in ("hadith_authentication", "hadith_abrogation", "textual_interpretation",
              "qiyas_disagreement", "urf_local_custom", "preference_among_proofs"):
        assert any(m.value == v for m in IkhtilafReasonType)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/models/test_vocabs.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'bayan_graph.models.vocabs'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/bayan_graph/models/vocabs.py
from enum import Enum


class EvidenceType(str, Enum):
    QURAN = "quran"
    HADITH = "hadith"
    IJMA = "ijma"
    QIYAS = "qiyas"
    MADHAB_TEXT = "madhab_text"


class EvidenceRole(str, Enum):
    PRIMARY_SUPPORT = "primary_support"
    CORROBORATING = "corroborating"
    OPPOSING_CITED = "opposing_cited"
    OPPOSING_REFUTED = "opposing_refuted"
    ABROGATED_REFERENCE = "abrogated_reference"


class AuthorityLevel(str, Enum):
    MU_TAMAD = "mu_tamad"
    MASHHUR = "mashhur"
    MARJUH = "marjuh"
    SHADH = "shadh"
    DISPUTED_WITHIN_MADHAB = "disputed_within_madhab"


class RoleInSource(str, Enum):
    PRIMARY_RULING = "primary_ruling"
    OPPOSING_VIEW_CITED = "opposing_view_cited"
    OPPOSING_VIEW_CITED_AND_REFUTED = "opposing_view_cited_and_refuted"
    COMPARATIVE_DISCUSSION = "comparative_discussion"
    CONDITIONAL_RULING = "conditional_ruling"
    NARRATIVE_COMMENTARY = "narrative_commentary"


class TawaturStatus(str, Enum):
    MUTAWATIR_LAFZI = "mutawatir_lafzi"
    MUTAWATIR_MANAWI = "mutawatir_manawi"
    MASHHUR = "mashhur"
    AZIZ = "aziz"
    GHARIB = "gharib"
    AHAD = "ahad"


class IkhtilafReasonType(str, Enum):
    HADITH_AUTHENTICATION = "hadith_authentication"
    HADITH_ABROGATION = "hadith_abrogation"
    TEXTUAL_INTERPRETATION = "textual_interpretation"
    QIYAS_DISAGREEMENT = "qiyas_disagreement"
    URF_LOCAL_CUSTOM = "urf_local_custom"
    PREFERENCE_AMONG_PROOFS = "preference_among_proofs"
    OTHER = "other"


class Madhab(str, Enum):
    HANAFI = "hanafi"
    MALIKI = "maliki"
    SHAFII = "shafii"
    HANBALI = "hanbali"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/models/test_vocabs.py -v`
Expected: PASS (6 tests).

- [ ] **Step 5: Commit**

```bash
git add tests/models/test_vocabs.py src/bayan_graph/models/vocabs.py
git commit -m "feat(models): add controlled vocabularies"
```

---

### Task B2: Evidence model — `subagent` [PAR-GROUP 2]

**Files:**
- Create: `tests/models/test_evidence.py`
- Create: `src/bayan_graph/models/evidence.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/models/test_evidence.py
import pytest
from pydantic import ValidationError

from bayan_graph.models.evidence import Evidence, SourceLocation
from bayan_graph.models.vocabs import EvidenceRole, EvidenceType, RoleInSource, TawaturStatus


def _base_source():
    return SourceLocation(
        work="Sahih al-Bukhari",
        volume=None,
        book="Kitab al-Wudu",
        chapter="Bab Fadl al-Wudu",
        page=None,
        anchor="bukhari:135",
    )


def test_evidence_quran_requires_verse_ref():
    ev = Evidence(
        id="ev1",
        type=EvidenceType.QURAN,
        role=EvidenceRole.PRIMARY_SUPPORT,
        source=SourceLocation(work="Quran", anchor="5:6"),
        arabic="يَا أَيُّهَا الَّذِينَ آمَنُوا إِذَا قُمْتُمْ إِلَى الصَّلَاةِ",
        english="O you who believe, when you rise for prayer",
        tawatur_status=TawaturStatus.MUTAWATIR_LAFZI,
        role_in_source=RoleInSource.PRIMARY_RULING,
        citation_span={"document_index": 0, "start": 0, "end": 120},
    )
    assert ev.type == EvidenceType.QURAN


def test_evidence_hadith_requires_grading_when_ahad():
    with pytest.raises(ValidationError):
        Evidence(
            id="ev2",
            type=EvidenceType.HADITH,
            role=EvidenceRole.PRIMARY_SUPPORT,
            source=_base_source(),
            arabic="...",
            english="...",
            tawatur_status=TawaturStatus.AHAD,
            role_in_source=RoleInSource.PRIMARY_RULING,
            citation_span={"document_index": 0, "start": 0, "end": 50},
            grading=None,
        )


def test_evidence_rejects_empty_arabic():
    with pytest.raises(ValidationError):
        Evidence(
            id="ev3",
            type=EvidenceType.QURAN,
            role=EvidenceRole.PRIMARY_SUPPORT,
            source=SourceLocation(work="Quran", anchor="5:6"),
            arabic="",
            english="text",
            tawatur_status=TawaturStatus.MUTAWATIR_LAFZI,
            role_in_source=RoleInSource.PRIMARY_RULING,
            citation_span={"document_index": 0, "start": 0, "end": 10},
        )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/models/test_evidence.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/bayan_graph/models/evidence.py
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

from bayan_graph.models.vocabs import (
    EvidenceRole,
    EvidenceType,
    RoleInSource,
    TawaturStatus,
)


class SourceLocation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    work: str
    volume: str | None = None
    book: str | None = None
    chapter: str | None = None
    section: str | None = None
    page: str | None = None
    anchor: str


class CitationSpan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_index: int = Field(ge=0)
    start: int = Field(ge=0)
    end: int = Field(ge=0)

    @model_validator(mode="after")
    def _check_range(self) -> "CitationSpan":
        if self.end < self.start:
            raise ValueError("end must be >= start")
        return self


class HadithGrading(BaseModel):
    model_config = ConfigDict(extra="forbid")

    classical: str | None = None
    modern: str | None = None
    resolved: str = Field(description="conservative-grade resolution across graders")


class Evidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    type: EvidenceType
    role: EvidenceRole
    source: SourceLocation
    arabic: str = Field(min_length=1)
    english: str = Field(min_length=1)
    tawatur_status: TawaturStatus | None = None
    role_in_source: RoleInSource
    citation_span: CitationSpan
    grading: HadithGrading | None = None
    matn_cluster_id: str | None = None
    abrogated: bool = False
    abrogated_by: str | None = None

    @model_validator(mode="after")
    def _validate_type_specific(self) -> "Evidence":
        if self.type == EvidenceType.HADITH:
            if self.tawatur_status is None:
                raise ValueError("hadith evidence requires tawatur_status")
            if self.tawatur_status == TawaturStatus.AHAD and self.grading is None:
                raise ValueError("ahad hadith requires grading")
        if self.type == EvidenceType.QURAN and self.tawatur_status is None:
            # default mutawatir for Quran
            object.__setattr__(self, "tawatur_status", TawaturStatus.MUTAWATIR_LAFZI)
        return self
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/models/test_evidence.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/models/test_evidence.py src/bayan_graph/models/evidence.py
git commit -m "feat(models): add Evidence, SourceLocation, CitationSpan, HadithGrading"
```

---

### Task B3: Position + ConditionalCase — `subagent` [PAR-GROUP 2]

**Files:**
- Create: `tests/models/test_position.py`
- Create: `src/bayan_graph/models/position.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/models/test_position.py
import pytest
from pydantic import ValidationError

from bayan_graph.models.position import AlternateView, ConditionalCase, Position, PrimaryView
from bayan_graph.models.vocabs import AuthorityLevel, Madhab


def _pv(ruling="wajib"):
    return PrimaryView(
        ruling=ruling,
        authority_level=AuthorityLevel.MU_TAMAD,
        evidence_refs=["ev1"],
        reasoning="Because the verse commands it.",
    )


def test_position_requires_primary_view():
    p = Position(madhab=Madhab.HANAFI, primary_view=_pv())
    assert p.madhab == Madhab.HANAFI


def test_position_rejects_empty_evidence_refs_in_primary():
    with pytest.raises(ValidationError):
        Position(
            madhab=Madhab.HANAFI,
            primary_view=PrimaryView(
                ruling="wajib",
                authority_level=AuthorityLevel.MU_TAMAD,
                evidence_refs=[],
                reasoning="x",
            ),
        )


def test_position_accepts_alternate_views_and_conditional_cases():
    p = Position(
        madhab=Madhab.SHAFII,
        primary_view=_pv("sunnah mu'akkadah"),
        alternate_views=[
            AlternateView(
                ruling="wajib",
                authority_level=AuthorityLevel.MARJUH,
                evidence_refs=["ev2"],
                held_by=["al-Muzani"],
                reasoning="Minority view.",
            )
        ],
        conditional_cases=[
            ConditionalCase(
                condition="If the traveler intends residence > 4 days",
                ruling="rakaat count becomes 4",
                evidence_refs=["ev3"],
            )
        ],
    )
    assert len(p.alternate_views) == 1
    assert len(p.conditional_cases) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/models/test_position.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/bayan_graph/models/position.py
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from bayan_graph.models.vocabs import AuthorityLevel, Madhab


class PrimaryView(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ruling: str = Field(min_length=1)
    authority_level: AuthorityLevel
    evidence_refs: list[str] = Field(min_length=1)
    reasoning: str = Field(min_length=1)


class AlternateView(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ruling: str = Field(min_length=1)
    authority_level: AuthorityLevel
    evidence_refs: list[str] = Field(min_length=1)
    held_by: list[str] = Field(default_factory=list)
    reasoning: str = Field(min_length=1)


class ConditionalCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    condition: str = Field(min_length=1)
    ruling: str = Field(min_length=1)
    evidence_refs: list[str] = Field(default_factory=list)


class Position(BaseModel):
    model_config = ConfigDict(extra="forbid")

    madhab: Madhab
    primary_view: PrimaryView
    alternate_views: list[AlternateView] = Field(default_factory=list)
    conditional_cases: list[ConditionalCase] = Field(default_factory=list)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/models/test_position.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/models/test_position.py src/bayan_graph/models/position.py
git commit -m "feat(models): add Position, PrimaryView, AlternateView, ConditionalCase"
```

---

### Task B4: Classification + ContextSensitivity — `subagent` [PAR-GROUP 2]

**Files:**
- Create: `tests/models/test_classification.py`
- Create: `src/bayan_graph/models/classification.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/models/test_classification.py
import pytest
from pydantic import ValidationError

from bayan_graph.models.classification import Classification, ContextSensitivity


def test_classification_in_scope_requires_topic():
    c = Classification(
        in_scope=True,
        scope_category="ibadat",
        topic="wudu",
        sub_topic="nullifiers",
        context_sensitivity=ContextSensitivity(
            is_context_sensitive=False,
            user_provided_context=None,
            missing_context_fields=[],
        ),
    )
    assert c.topic == "wudu"


def test_classification_out_of_scope_does_not_need_topic():
    c = Classification(
        in_scope=False,
        scope_category="out_of_scope",
        topic=None,
        sub_topic=None,
        context_sensitivity=ContextSensitivity(
            is_context_sensitive=False,
            user_provided_context=None,
            missing_context_fields=[],
        ),
    )
    assert c.in_scope is False


def test_classification_in_scope_without_topic_rejected():
    with pytest.raises(ValidationError):
        Classification(
            in_scope=True,
            scope_category="ibadat",
            topic=None,
            sub_topic=None,
            context_sensitivity=ContextSensitivity(
                is_context_sensitive=False,
                user_provided_context=None,
                missing_context_fields=[],
            ),
        )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/models/test_classification.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/bayan_graph/models/classification.py
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ContextSensitivity(BaseModel):
    model_config = ConfigDict(extra="forbid")

    is_context_sensitive: bool
    user_provided_context: dict[str, str] | None = None
    missing_context_fields: list[str] = Field(default_factory=list)


class Classification(BaseModel):
    model_config = ConfigDict(extra="forbid")

    in_scope: bool
    scope_category: str
    topic: str | None = None
    sub_topic: str | None = None
    context_sensitivity: ContextSensitivity

    @model_validator(mode="after")
    def _in_scope_needs_topic(self) -> "Classification":
        if self.in_scope and not self.topic:
            raise ValueError("in_scope classification requires topic")
        return self
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/models/test_classification.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/models/test_classification.py src/bayan_graph/models/classification.py
git commit -m "feat(models): add Classification and ContextSensitivity"
```

---

### Task B5: CrossMadhabAnalysis + IkhtilafReason — `subagent` [PAR-GROUP 2]

**Files:**
- Create: `tests/models/test_analysis.py`
- Create: `src/bayan_graph/models/analysis.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/models/test_analysis.py
from bayan_graph.models.analysis import (
    CrossMadhabAnalysis,
    IkhtilafReason,
    PointOfAgreement,
    PointOfDisagreement,
)
from bayan_graph.models.vocabs import IkhtilafReasonType, Madhab


def test_cross_madhab_analysis_empty_is_valid():
    cma = CrossMadhabAnalysis(
        points_of_agreement=[],
        points_of_disagreement=[],
    )
    assert cma.points_of_agreement == []


def test_point_of_disagreement_with_reasons():
    pod = PointOfDisagreement(
        summary="Whether light sleep breaks wudu",
        madhab_stances={
            Madhab.HANAFI: "does not break",
            Madhab.MALIKI: "does not break if reclining",
            Madhab.SHAFII: "breaks",
            Madhab.HANBALI: "breaks unless minimal",
        },
        reasons_for_ikhtilaf=[
            IkhtilafReason(
                reason_type=IkhtilafReasonType.HADITH_AUTHENTICATION,
                explanation="Different rulings on the chain of Safwan ibn Assal hadith.",
                evidence_refs=["ev5", "ev6"],
            )
        ],
    )
    assert len(pod.reasons_for_ikhtilaf) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/models/test_analysis.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/bayan_graph/models/analysis.py
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from bayan_graph.models.vocabs import IkhtilafReasonType, Madhab


class PointOfAgreement(BaseModel):
    model_config = ConfigDict(extra="forbid")

    summary: str = Field(min_length=1)
    evidence_refs: list[str] = Field(default_factory=list)


class IkhtilafReason(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reason_type: IkhtilafReasonType
    explanation: str = Field(min_length=1)
    evidence_refs: list[str] = Field(default_factory=list)


class PointOfDisagreement(BaseModel):
    model_config = ConfigDict(extra="forbid")

    summary: str = Field(min_length=1)
    madhab_stances: dict[Madhab, str]
    reasons_for_ikhtilaf: list[IkhtilafReason] = Field(default_factory=list)


class CrossMadhabAnalysis(BaseModel):
    model_config = ConfigDict(extra="forbid")

    points_of_agreement: list[PointOfAgreement] = Field(default_factory=list)
    points_of_disagreement: list[PointOfDisagreement] = Field(default_factory=list)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/models/test_analysis.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/models/test_analysis.py src/bayan_graph/models/analysis.py
git commit -m "feat(models): add CrossMadhabAnalysis and ikhtilaf reasons"
```

---

### Task B6: CitationAudit + Disclosure + Meta — `subagent` [PAR-GROUP 2]

**Files:**
- Create: `tests/models/test_audit.py`
- Create: `src/bayan_graph/models/audit.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/models/test_audit.py
import pytest
from pydantic import ValidationError

from bayan_graph.models.audit import CitationAudit, CitationCheckResult, Disclosure, Meta


def test_citation_audit_requires_five_checks():
    audit = CitationAudit(
        passed=True,
        tier="clean",
        checks=[
            CitationCheckResult(name="every_claim_has_span", passed=True, detail="ok"),
            CitationCheckResult(name="spans_non_empty", passed=True, detail="ok"),
            CitationCheckResult(name="evidence_refs_resolve", passed=True, detail="ok"),
            CitationCheckResult(name="hierarchy_respected", passed=True, detail="ok"),
            CitationCheckResult(name="madhab_source_alignment", passed=True, detail="ok"),
        ],
        failing_check=None,
    )
    assert audit.passed is True


def test_citation_audit_tier_must_be_valid():
    with pytest.raises(ValidationError):
        CitationAudit(
            passed=False,
            tier="bogus",
            checks=[],
            failing_check=None,
        )


def test_disclosure_and_meta_compose():
    d = Disclosure(
        caveat="Not a fatwa. Consult a qualified scholar.",
        sources_consulted=["Quran", "Sahih al-Bukhari", "Al-Hidayah"],
    )
    m = Meta(
        query_id="q-1",
        latency_ms=1234,
        model_versions={"classifier": "claude-haiku-4-5-20251001"},
        retrieval_stats={"chunks_considered": 42},
    )
    assert d.sources_consulted[0] == "Quran"
    assert m.latency_ms == 1234
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/models/test_audit.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/bayan_graph/models/audit.py
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

AuditTier = Literal["clean", "degraded", "refused"]


class CitationCheckResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    passed: bool
    detail: str


class CitationAudit(BaseModel):
    model_config = ConfigDict(extra="forbid")

    passed: bool
    tier: AuditTier
    checks: list[CitationCheckResult]
    failing_check: str | None = None


class Disclosure(BaseModel):
    model_config = ConfigDict(extra="forbid")

    caveat: str = Field(min_length=1)
    sources_consulted: list[str] = Field(default_factory=list)


class Meta(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query_id: str
    latency_ms: int = Field(ge=0)
    model_versions: dict[str, str] = Field(default_factory=dict)
    retrieval_stats: dict[str, int] = Field(default_factory=dict)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/models/test_audit.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/models/test_audit.py src/bayan_graph/models/audit.py
git commit -m "feat(models): add CitationAudit, Disclosure, Meta"
```

---

### Task B7: AnswerResponse + BayanState — `subagent` [PAR-GROUP 2]

**Files:**
- Create: `tests/models/test_response.py`
- Create: `src/bayan_graph/models/response.py`
- Create: `src/bayan_graph/models/state.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/models/test_response.py
from bayan_graph.models.analysis import CrossMadhabAnalysis
from bayan_graph.models.audit import CitationAudit, CitationCheckResult, Disclosure, Meta
from bayan_graph.models.classification import Classification, ContextSensitivity
from bayan_graph.models.evidence import CitationSpan, Evidence, SourceLocation
from bayan_graph.models.position import Position, PrimaryView
from bayan_graph.models.response import AnswerResponse
from bayan_graph.models.vocabs import (
    AuthorityLevel,
    EvidenceRole,
    EvidenceType,
    Madhab,
    RoleInSource,
    TawaturStatus,
)


def _minimal_response() -> AnswerResponse:
    ev = Evidence(
        id="ev1",
        type=EvidenceType.QURAN,
        role=EvidenceRole.PRIMARY_SUPPORT,
        source=SourceLocation(work="Quran", anchor="5:6"),
        arabic="يَا أَيُّهَا الَّذِينَ آمَنُوا",
        english="O you who believe",
        tawatur_status=TawaturStatus.MUTAWATIR_LAFZI,
        role_in_source=RoleInSource.PRIMARY_RULING,
        citation_span=CitationSpan(document_index=0, start=0, end=30),
    )
    return AnswerResponse(
        query="Is wudu required for prayer?",
        headline_answer="Yes, all four madhahib require wudu for prayer.",
        classification=Classification(
            in_scope=True,
            scope_category="ibadat",
            topic="wudu",
            sub_topic=None,
            context_sensitivity=ContextSensitivity(
                is_context_sensitive=False,
                user_provided_context=None,
                missing_context_fields=[],
            ),
        ),
        positions=[
            Position(
                madhab=m,
                primary_view=PrimaryView(
                    ruling="wajib",
                    authority_level=AuthorityLevel.MU_TAMAD,
                    evidence_refs=["ev1"],
                    reasoning="Commanded in 5:6.",
                ),
            )
            for m in Madhab
        ],
        cross_madhab_analysis=CrossMadhabAnalysis(
            points_of_agreement=[],
            points_of_disagreement=[],
        ),
        evidence=[ev],
        citation_audit=CitationAudit(
            passed=True,
            tier="clean",
            checks=[
                CitationCheckResult(name=n, passed=True, detail="ok")
                for n in (
                    "every_claim_has_span",
                    "spans_non_empty",
                    "evidence_refs_resolve",
                    "hierarchy_respected",
                    "madhab_source_alignment",
                )
            ],
            failing_check=None,
        ),
        disclosure=Disclosure(
            caveat="Not a fatwa.",
            sources_consulted=["Quran"],
        ),
        meta=Meta(
            query_id="q-1",
            latency_ms=1000,
            model_versions={"classifier": "claude-haiku-4-5-20251001"},
            retrieval_stats={"chunks_considered": 5},
        ),
    )


def test_answer_response_roundtrips():
    r = _minimal_response()
    dumped = r.model_dump(mode="json")
    r2 = AnswerResponse.model_validate(dumped)
    assert r2.query == r.query
    assert len(r2.positions) == 4
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/models/test_response.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write `src/bayan_graph/models/response.py`**

```python
# src/bayan_graph/models/response.py
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from bayan_graph.models.analysis import CrossMadhabAnalysis
from bayan_graph.models.audit import CitationAudit, Disclosure, Meta
from bayan_graph.models.classification import Classification
from bayan_graph.models.evidence import Evidence
from bayan_graph.models.position import Position


class AnswerResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1)
    headline_answer: str = Field(min_length=1)
    classification: Classification
    positions: list[Position]
    cross_madhab_analysis: CrossMadhabAnalysis
    evidence: list[Evidence]
    citation_audit: CitationAudit
    disclosure: Disclosure
    meta: Meta
    narrative: str | None = None
```

- [ ] **Step 4: Write `src/bayan_graph/models/state.py`**

```python
# src/bayan_graph/models/state.py
from __future__ import annotations

from typing import TypedDict

from bayan_graph.models.analysis import CrossMadhabAnalysis
from bayan_graph.models.audit import CitationAudit
from bayan_graph.models.classification import Classification
from bayan_graph.models.evidence import Evidence
from bayan_graph.models.position import Position


class RetrievalHit(TypedDict):
    chunk_id: str
    namespace: str
    score: float
    text: str
    metadata: dict


class BayanState(TypedDict, total=False):
    query: str
    query_id: str
    classification: Classification
    retrieval_hits: dict[str, list[RetrievalHit]]
    evidence: list[Evidence]
    positions: list[Position]
    cross_madhab_analysis: CrossMadhabAnalysis
    headline_answer: str
    narrative: str
    citation_audit: CitationAudit
    sufficiency_iterations: int
    errors: list[str]
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/models/test_response.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add tests/models/test_response.py src/bayan_graph/models/response.py src/bayan_graph/models/state.py
git commit -m "feat(models): add AnswerResponse and BayanState TypedDict"
```

---

## Phase C — Arabic normalization

### Task C1: Arabic normalization with quality score — `subagent`

**Files:**
- Create: `tests/ingestion/test_arabic.py`
- Create: `src/bayan_graph/ingestion/arabic.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/ingestion/test_arabic.py
from bayan_graph.ingestion.arabic import NormalizationProfile, normalize_arabic


def test_strip_harakat_for_embedding():
    result = normalize_arabic(
        "يَا أَيُّهَا الَّذِينَ آمَنُوا",
        profile=NormalizationProfile.EMBEDDING,
    )
    assert "َ" not in result.text
    assert "ُ" not in result.text
    assert "ّ" not in result.text
    assert result.quality_score > 0.9


def test_preserve_harakat_for_display():
    original = "يَا أَيُّهَا الَّذِينَ آمَنُوا"
    result = normalize_arabic(original, profile=NormalizationProfile.DISPLAY)
    assert "َ" in result.text


def test_variant_normalization_for_comparison():
    r1 = normalize_arabic("إبراهيم", profile=NormalizationProfile.COMPARISON)
    r2 = normalize_arabic("ابراهيم", profile=NormalizationProfile.COMPARISON)
    assert r1.text == r2.text


def test_quality_score_drops_on_noise():
    result = normalize_arabic(
        "يَاأَيُّهَا????الَّذِينَ",
        profile=NormalizationProfile.EMBEDDING,
    )
    assert result.quality_score < 0.9
    assert "non_arabic_intrusion" in result.flags or "suspicious_chars" in result.flags


def test_flags_short_text():
    result = normalize_arabic("يا", profile=NormalizationProfile.EMBEDDING)
    assert "too_short" in result.flags
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/ingestion/test_arabic.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/bayan_graph/ingestion/arabic.py
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum

HARAKAT = "\u064B\u064C\u064D\u064E\u064F\u0650\u0651\u0652\u0653\u0654\u0655\u0670"
TATWEEL = "\u0640"

ALEF_VARIANTS = "أإآٱ"
YA_VARIANTS = "ىئ"
TA_MARBUTA = "ة"
HAMZA_ON_WAW = "ؤ"

ARABIC_RANGE = re.compile(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]")
NON_ARABIC_NON_WS = re.compile(r"[^\s\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF.,،؛؟!:()\[\]\"'—-]")


class NormalizationProfile(str, Enum):
    EMBEDDING = "embedding"
    DISPLAY = "display"
    COMPARISON = "comparison"


@dataclass
class NormalizationResult:
    text: str
    quality_score: float
    flags: list[str] = field(default_factory=list)


def _strip_harakat(text: str) -> str:
    return text.translate(str.maketrans("", "", HARAKAT + TATWEEL))


def _normalize_variants(text: str) -> str:
    for variant in ALEF_VARIANTS:
        text = text.replace(variant, "ا")
    for variant in YA_VARIANTS:
        text = text.replace(variant, "ي")
    text = text.replace(TA_MARBUTA, "ه")
    text = text.replace(HAMZA_ON_WAW, "و")
    return text


def _collapse_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def normalize_arabic(
    text: str,
    *,
    profile: NormalizationProfile = NormalizationProfile.EMBEDDING,
) -> NormalizationResult:
    flags: list[str] = []
    original = text

    if profile == NormalizationProfile.DISPLAY:
        out = _collapse_whitespace(text)
    elif profile == NormalizationProfile.EMBEDDING:
        out = _strip_harakat(text)
        out = _collapse_whitespace(out)
    elif profile == NormalizationProfile.COMPARISON:
        out = _strip_harakat(text)
        out = _normalize_variants(out)
        out = _collapse_whitespace(out)
    else:  # pragma: no cover - exhaustive
        raise ValueError(f"unknown profile {profile}")

    arabic_chars = sum(1 for ch in original if ARABIC_RANGE.match(ch))
    total_nonspace = sum(1 for ch in original if not ch.isspace())
    arabic_ratio = arabic_chars / total_nonspace if total_nonspace else 0.0

    if total_nonspace < 4:
        flags.append("too_short")
    if NON_ARABIC_NON_WS.search(original):
        flags.append("suspicious_chars")
    if arabic_ratio < 0.7 and total_nonspace >= 4:
        flags.append("non_arabic_intrusion")

    if "too_short" in flags:
        quality_score = 0.5
    elif "non_arabic_intrusion" in flags:
        quality_score = max(0.0, arabic_ratio)
    elif "suspicious_chars" in flags:
        quality_score = 0.85
    else:
        quality_score = round(min(1.0, arabic_ratio), 3)

    return NormalizationResult(text=out, quality_score=quality_score, flags=flags)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/ingestion/test_arabic.py -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add tests/ingestion/test_arabic.py src/bayan_graph/ingestion/arabic.py
git commit -m "feat(ingestion): add Arabic normalization with profile + quality score"
```

---

## Phase D — Quran ingestion (walking-skeleton corpus)

### Task D1: Acquire Tanzil Quran text and write loader — `main-thread`

**Files:**
- Create: `data/raw/README.md`
- Create: `scripts/fetch_quran.py`
- Create: `tests/ingestion/test_quran_loader.py`
- Create: `src/bayan_graph/ingestion/quran.py`

- [ ] **Step 1: Write `data/raw/README.md`**

```markdown
# Raw corpus

Large text files downloaded from upstream sources. Tracked via `scripts/fetch_*.py`
scripts; not committed (see `.gitignore`).

Quran (Uthmani + Sahih International):
- Arabic: https://tanzil.net/download/ (Uthmani, text-only)
- English: https://tanzil.net/trans/en.sahih
```

- [ ] **Step 2: Write `scripts/fetch_quran.py`**

```python
"""Download Quran Uthmani (Arabic) and Sahih International (English) from Tanzil."""
from __future__ import annotations

import sys
from pathlib import Path

import httpx

ARABIC_URL = "https://tanzil.net/pub/download/index.php?quranType=uthmani&outType=txt&marks=false&sajdah=false"
ENGLISH_URL = "https://tanzil.net/trans/en.sahih"


def fetch(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with httpx.Client(timeout=60, follow_redirects=True) as client:
        resp = client.get(url)
        resp.raise_for_status()
        dest.write_text(resp.text, encoding="utf-8")
    print(f"wrote {dest} ({dest.stat().st_size} bytes)")


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    data_dir = repo_root / "data" / "raw" / "quran"
    fetch(ARABIC_URL, data_dir / "quran-uthmani.txt")
    fetch(ENGLISH_URL, data_dir / "en-sahih.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 3: Run the fetcher once locally**

Run: `uv run python scripts/fetch_quran.py`
Expected: writes two files under `data/raw/quran/`. If Tanzil is unreachable, use a fixture instead (see Step 4).

- [ ] **Step 4: Write small test fixture**

Create `tests/fixtures/quran_sample.txt`:

```
1|1|بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ
1|2|الْحَمْدُ لِلَّهِ رَبِّ الْعَالَمِينَ
1|3|الرَّحْمَٰنِ الرَّحِيمِ
1|4|مَالِكِ يَوْمِ الدِّينِ
1|5|إِيَّاكَ نَعْبُدُ وَإِيَّاكَ نَسْتَعِينُ
1|6|اهْدِنَا الصِّرَاطَ الْمُسْتَقِيمَ
1|7|صِرَاطَ الَّذِينَ أَنْعَمْتَ عَلَيْهِمْ غَيْرِ الْمَغْضُوبِ عَلَيْهِمْ وَلَا الضَّالِّينَ
```

Create `tests/fixtures/en-sahih_sample.txt`:

```
1|1|In the name of Allah, the Entirely Merciful, the Especially Merciful.
1|2|[All] praise is [due] to Allah, Lord of the worlds -
1|3|The Entirely Merciful, the Especially Merciful,
1|4|Sovereign of the Day of Recompense.
1|5|It is You we worship and You we ask for help.
1|6|Guide us to the straight path -
1|7|The path of those upon whom You have bestowed favor, not of those who have evoked [Your] anger or of those who are astray.
```

- [ ] **Step 5: Write the failing test**

```python
# tests/ingestion/test_quran_loader.py
from pathlib import Path

from bayan_graph.ingestion.quran import QuranAyah, load_quran


def test_load_quran_pairs_arabic_and_english(fixtures_dir: Path):
    ayat = load_quran(
        arabic_path=fixtures_dir / "quran_sample.txt",
        english_path=fixtures_dir / "en-sahih_sample.txt",
    )
    assert len(ayat) == 7
    first = ayat[0]
    assert isinstance(first, QuranAyah)
    assert first.surah == 1
    assert first.ayah == 1
    assert "بِسْمِ" in first.arabic
    assert "Merciful" in first.english


def test_load_quran_detects_length_mismatch(tmp_path: Path, fixtures_dir: Path):
    short = tmp_path / "short.txt"
    short.write_text("1|1|A\n", encoding="utf-8")
    try:
        load_quran(arabic_path=short, english_path=fixtures_dir / "en-sahih_sample.txt")
    except ValueError as exc:
        assert "mismatch" in str(exc).lower()
    else:
        raise AssertionError("expected ValueError on length mismatch")
```

- [ ] **Step 6: Run test to verify it fails**

Run: `uv run pytest tests/ingestion/test_quran_loader.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 7: Implement `src/bayan_graph/ingestion/quran.py`**

```python
# src/bayan_graph/ingestion/quran.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class QuranAyah:
    surah: int
    ayah: int
    arabic: str
    english: str

    @property
    def anchor(self) -> str:
        return f"{self.surah}:{self.ayah}"


def _parse_tanzil(path: Path) -> dict[tuple[int, int], str]:
    out: dict[tuple[int, int], str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("|", 2)
        if len(parts) != 3:
            continue
        surah, ayah, text = parts
        out[(int(surah), int(ayah))] = text.strip()
    return out


def load_quran(*, arabic_path: Path, english_path: Path) -> list[QuranAyah]:
    arabic = _parse_tanzil(arabic_path)
    english = _parse_tanzil(english_path)
    if set(arabic) != set(english):
        missing = set(arabic) ^ set(english)
        raise ValueError(f"arabic/english key mismatch: {sorted(missing)[:5]}")
    return [
        QuranAyah(surah=s, ayah=a, arabic=arabic[(s, a)], english=english[(s, a)])
        for (s, a) in sorted(arabic)
    ]
```

- [ ] **Step 8: Run test to verify it passes**

Run: `uv run pytest tests/ingestion/test_quran_loader.py -v`
Expected: PASS.

- [ ] **Step 9: Commit**

```bash
git add data/raw/README.md scripts/fetch_quran.py tests/fixtures/quran_sample.txt tests/fixtures/en-sahih_sample.txt tests/ingestion/test_quran_loader.py src/bayan_graph/ingestion/quran.py
git commit -m "feat(ingestion): Quran loader with paired Arabic/English"
```

---

### Task D2: Chunking strategy for Quran (ayah-window) — `subagent`

**Files:**
- Create: `tests/ingestion/test_chunking.py`
- Create: `src/bayan_graph/ingestion/chunking.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/ingestion/test_chunking.py
from bayan_graph.ingestion.chunking import QuranChunk, chunk_quran
from bayan_graph.ingestion.quran import QuranAyah


def _ayat():
    return [
        QuranAyah(surah=1, ayah=i, arabic=f"ar{i}", english=f"en{i}")
        for i in range(1, 8)
    ]


def test_chunk_quran_window_three_stride_two():
    chunks = chunk_quran(_ayat(), window=3, stride=2)
    assert len(chunks) == 3
    assert isinstance(chunks[0], QuranChunk)
    assert chunks[0].anchor == "1:1-1:3"
    assert chunks[1].anchor == "1:3-1:5"
    assert chunks[2].anchor == "1:5-1:7"


def test_chunk_quran_does_not_cross_surah_boundary():
    ayat = [
        QuranAyah(surah=1, ayah=1, arabic="a", english="e"),
        QuranAyah(surah=1, ayah=2, arabic="a", english="e"),
        QuranAyah(surah=2, ayah=1, arabic="a", english="e"),
        QuranAyah(surah=2, ayah=2, arabic="a", english="e"),
    ]
    chunks = chunk_quran(ayat, window=3, stride=1)
    for c in chunks:
        surah_start, _ = c.start_ref
        surah_end, _ = c.end_ref
        assert surah_start == surah_end
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/ingestion/test_chunking.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `src/bayan_graph/ingestion/chunking.py`**

```python
# src/bayan_graph/ingestion/chunking.py
from __future__ import annotations

from dataclasses import dataclass
from itertools import groupby

from bayan_graph.ingestion.quran import QuranAyah


@dataclass(frozen=True)
class QuranChunk:
    chunk_id: str
    surah: int
    start_ref: tuple[int, int]
    end_ref: tuple[int, int]
    arabic: str
    english: str
    anchor: str


def chunk_quran(ayat: list[QuranAyah], *, window: int, stride: int) -> list[QuranChunk]:
    if window <= 0 or stride <= 0:
        raise ValueError("window and stride must be positive")
    chunks: list[QuranChunk] = []
    for surah, group_iter in groupby(ayat, key=lambda a: a.surah):
        group = list(group_iter)
        for start in range(0, len(group), stride):
            window_ayat = group[start : start + window]
            if not window_ayat:
                break
            first = window_ayat[0]
            last = window_ayat[-1]
            anchor = f"{first.surah}:{first.ayah}-{last.surah}:{last.ayah}"
            chunks.append(
                QuranChunk(
                    chunk_id=f"quran:{anchor}",
                    surah=surah,
                    start_ref=(first.surah, first.ayah),
                    end_ref=(last.surah, last.ayah),
                    arabic=" ".join(a.arabic for a in window_ayat),
                    english=" ".join(a.english for a in window_ayat),
                    anchor=anchor,
                )
            )
            if start + window >= len(group):
                break
    return chunks
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/ingestion/test_chunking.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/ingestion/test_chunking.py src/bayan_graph/ingestion/chunking.py
git commit -m "feat(ingestion): sliding-window Quran chunking"
```

---

### Task D3: Cohere embedding wrapper — `subagent`

**Files:**
- Create: `tests/retrieval/test_embed.py`
- Create: `src/bayan_graph/retrieval/embed.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/retrieval/test_embed.py
from unittest.mock import MagicMock, patch

from bayan_graph.retrieval.embed import EmbeddingInputType, embed_texts


def test_embed_texts_routes_input_type_and_batches():
    fake_resp = MagicMock()
    fake_resp.embeddings = [[0.1] * 1024 for _ in range(3)]
    fake_client = MagicMock()
    fake_client.embed.return_value = fake_resp

    with patch("bayan_graph.retrieval.embed._client", return_value=fake_client):
        vecs = embed_texts(
            ["a", "b", "c"],
            input_type=EmbeddingInputType.DOCUMENT,
            batch_size=96,
        )

    assert len(vecs) == 3
    assert len(vecs[0]) == 1024
    fake_client.embed.assert_called_once()
    call_kwargs = fake_client.embed.call_args.kwargs
    assert call_kwargs["model"] == "embed-multilingual-v3.0"
    assert call_kwargs["input_type"] == "search_document"


def test_embed_texts_query_input_type():
    fake_resp = MagicMock()
    fake_resp.embeddings = [[0.2] * 1024]
    fake_client = MagicMock()
    fake_client.embed.return_value = fake_resp

    with patch("bayan_graph.retrieval.embed._client", return_value=fake_client):
        embed_texts(["how many rakaat in fajr?"], input_type=EmbeddingInputType.QUERY)

    assert fake_client.embed.call_args.kwargs["input_type"] == "search_query"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/retrieval/test_embed.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `src/bayan_graph/retrieval/embed.py`**

```python
# src/bayan_graph/retrieval/embed.py
from __future__ import annotations

import os
from enum import Enum
from functools import lru_cache

import cohere


class EmbeddingInputType(str, Enum):
    DOCUMENT = "search_document"
    QUERY = "search_query"


EMBED_MODEL = "embed-multilingual-v3.0"
EMBED_DIM = 1024
DEFAULT_BATCH_SIZE = 96


@lru_cache(maxsize=1)
def _client() -> cohere.ClientV2:
    key = os.environ["COHERE_API_KEY"]
    return cohere.ClientV2(api_key=key)


def embed_texts(
    texts: list[str],
    *,
    input_type: EmbeddingInputType,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> list[list[float]]:
    if not texts:
        return []
    client = _client()
    out: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        resp = client.embed(
            model=EMBED_MODEL,
            input_type=input_type.value,
            texts=batch,
            embedding_types=["float"],
        )
        embeddings = getattr(resp, "embeddings", None)
        if embeddings is None or not isinstance(embeddings, list):
            # v2 returns an object with .float
            embeddings = resp.embeddings.float  # type: ignore[attr-defined]
        out.extend(embeddings)
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/retrieval/test_embed.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/retrieval/test_embed.py src/bayan_graph/retrieval/embed.py
git commit -m "feat(retrieval): Cohere embed wrapper with batching and input-type routing"
```

---

### Task D4: Pinecone upsert client — `subagent`

**Files:**
- Create: `tests/retrieval/test_pinecone_client.py`
- Create: `src/bayan_graph/retrieval/pinecone_client.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/retrieval/test_pinecone_client.py
from unittest.mock import MagicMock, patch

from bayan_graph.retrieval.pinecone_client import PineconeStore, Vector


def test_upsert_batches_and_namespaces():
    index = MagicMock()
    with patch(
        "bayan_graph.retrieval.pinecone_client.PineconeStore._get_index",
        return_value=index,
    ):
        store = PineconeStore(index_name="bayan-test")
        vectors = [
            Vector(id=f"v{i}", values=[0.1] * 1024, metadata={"text": f"t{i}"})
            for i in range(250)
        ]
        store.upsert(namespace="quran", vectors=vectors, batch_size=100)

    assert index.upsert.call_count == 3
    for call in index.upsert.call_args_list:
        assert call.kwargs["namespace"] == "quran"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/retrieval/test_pinecone_client.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `src/bayan_graph/retrieval/pinecone_client.py`**

```python
# src/bayan_graph/retrieval/pinecone_client.py
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache

from pinecone import Pinecone


@dataclass
class Vector:
    id: str
    values: list[float]
    metadata: dict


@lru_cache(maxsize=1)
def _pinecone() -> Pinecone:
    return Pinecone(api_key=os.environ["PINECONE_API_KEY"])


class PineconeStore:
    def __init__(self, index_name: str):
        self.index_name = index_name

    def _get_index(self):
        pc = _pinecone()
        return pc.Index(self.index_name)

    def upsert(
        self,
        *,
        namespace: str,
        vectors: list[Vector],
        batch_size: int = 100,
    ) -> int:
        index = self._get_index()
        count = 0
        for start in range(0, len(vectors), batch_size):
            batch = vectors[start : start + batch_size]
            payload = [
                {"id": v.id, "values": v.values, "metadata": v.metadata} for v in batch
            ]
            index.upsert(vectors=payload, namespace=namespace)
            count += len(batch)
        return count

    def query(
        self,
        *,
        namespace: str,
        vector: list[float],
        top_k: int,
        filter: dict | None = None,
    ) -> list[dict]:
        index = self._get_index()
        result = index.query(
            vector=vector,
            namespace=namespace,
            top_k=top_k,
            include_metadata=True,
            filter=filter,
        )
        return [
            {
                "id": m.id,
                "score": m.score,
                "metadata": m.metadata or {},
            }
            for m in result.matches
        ]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/retrieval/test_pinecone_client.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/retrieval/test_pinecone_client.py src/bayan_graph/retrieval/pinecone_client.py
git commit -m "feat(retrieval): Pinecone upsert/query client with batching"
```

---

### Task D5: Quran ingestion CLI — `main-thread`

**Files:**
- Create: `tests/ingestion/test_quran_ingest.py`
- Create: `src/bayan_graph/cli/ingest_quran.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/ingestion/test_quran_ingest.py
from pathlib import Path
from unittest.mock import MagicMock, patch

from bayan_graph.cli.ingest_quran import run_ingest


def test_run_ingest_produces_vectors_with_expected_metadata(fixtures_dir: Path):
    fake_embed = MagicMock(return_value=[[0.1] * 1024 for _ in range(5)])
    fake_store = MagicMock()
    fake_store.upsert.return_value = 0

    with (
        patch("bayan_graph.cli.ingest_quran.embed_texts", fake_embed),
        patch("bayan_graph.cli.ingest_quran.PineconeStore", return_value=fake_store),
    ):
        stats = run_ingest(
            arabic_path=fixtures_dir / "quran_sample.txt",
            english_path=fixtures_dir / "en-sahih_sample.txt",
            index_name="bayan-test",
            window=3,
            stride=2,
        )

    assert stats["ayat"] == 7
    assert stats["chunks"] >= 1
    args = fake_embed.call_args
    assert args.kwargs["input_type"].value == "search_document"
    upsert_kwargs = fake_store.upsert.call_args.kwargs
    assert upsert_kwargs["namespace"] == "quran"
    first_vec = upsert_kwargs["vectors"][0]
    md = first_vec.metadata
    assert md["source_type"] == "quran"
    assert md["anchor"].startswith("1:")
    assert "arabic" in md and "english" in md
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/ingestion/test_quran_ingest.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `src/bayan_graph/cli/ingest_quran.py`**

```python
# src/bayan_graph/cli/ingest_quran.py
from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated

import typer

from bayan_graph.ingestion.arabic import NormalizationProfile, normalize_arabic
from bayan_graph.ingestion.chunking import QuranChunk, chunk_quran
from bayan_graph.ingestion.quran import load_quran
from bayan_graph.retrieval.embed import EmbeddingInputType, embed_texts
from bayan_graph.retrieval.pinecone_client import PineconeStore, Vector

app = typer.Typer(add_completion=False)


def _to_vector(chunk: QuranChunk, embedding: list[float]) -> Vector:
    arabic_norm = normalize_arabic(chunk.arabic, profile=NormalizationProfile.EMBEDDING)
    return Vector(
        id=chunk.chunk_id,
        values=embedding,
        metadata={
            "source_type": "quran",
            "anchor": chunk.anchor,
            "surah": chunk.surah,
            "start_surah": chunk.start_ref[0],
            "start_ayah": chunk.start_ref[1],
            "end_surah": chunk.end_ref[0],
            "end_ayah": chunk.end_ref[1],
            "arabic": chunk.arabic,
            "arabic_normalized": arabic_norm.text,
            "english": chunk.english,
            "arabic_quality_score": arabic_norm.quality_score,
            "arabic_flags": arabic_norm.flags,
        },
    )


def run_ingest(
    *,
    arabic_path: Path,
    english_path: Path,
    index_name: str,
    window: int = 3,
    stride: int = 2,
) -> dict[str, int]:
    ayat = load_quran(arabic_path=arabic_path, english_path=english_path)
    chunks = chunk_quran(ayat, window=window, stride=stride)
    # Embed on the normalized Arabic joined with English for multilingual retrieval.
    texts = [
        f"{normalize_arabic(c.arabic, profile=NormalizationProfile.EMBEDDING).text}\n{c.english}"
        for c in chunks
    ]
    embeddings = embed_texts(texts, input_type=EmbeddingInputType.DOCUMENT)
    vectors = [_to_vector(chunk, emb) for chunk, emb in zip(chunks, embeddings, strict=True)]
    store = PineconeStore(index_name=index_name)
    upserted = store.upsert(namespace="quran", vectors=vectors)
    return {"ayat": len(ayat), "chunks": len(chunks), "upserted": upserted}


@app.command()
def ingest(
    arabic: Annotated[Path, typer.Option(help="Tanzil Uthmani txt")],
    english: Annotated[Path, typer.Option(help="Tanzil Sahih International txt")],
    index: Annotated[str, typer.Option(help="Pinecone index name")] = os.getenv(
        "PINECONE_INDEX", "bayan-graph"
    ),
    window: int = 3,
    stride: int = 2,
) -> None:
    stats = run_ingest(
        arabic_path=arabic,
        english_path=english,
        index_name=index,
        window=window,
        stride=stride,
    )
    typer.echo(stats)


if __name__ == "__main__":
    app()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/ingestion/test_quran_ingest.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/ingestion/test_quran_ingest.py src/bayan_graph/cli/ingest_quran.py
git commit -m "feat(cli): Quran ingestion command (load → chunk → embed → upsert)"
```

---

## Phase E — Retrieval runtime

### Task E1: Query-time embed + Pinecone retrieval wrapper — `subagent` [PAR-GROUP 3]

**Files:**
- Create: `tests/retrieval/test_retrieve.py`
- Create: `src/bayan_graph/retrieval/retrieve.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/retrieval/test_retrieve.py
from unittest.mock import MagicMock, patch

from bayan_graph.retrieval.retrieve import retrieve_namespace


def test_retrieve_namespace_calls_query_embedding_and_pinecone():
    fake_store = MagicMock()
    fake_store.query.return_value = [
        {"id": "quran:5:6-5:6", "score": 0.9, "metadata": {"anchor": "5:6"}},
    ]
    with (
        patch("bayan_graph.retrieval.retrieve.embed_texts", return_value=[[0.3] * 1024]),
        patch("bayan_graph.retrieval.retrieve.PineconeStore", return_value=fake_store),
    ):
        hits = retrieve_namespace(
            query="wudu requirements",
            namespace="quran",
            top_k=5,
            index_name="bayan-test",
        )

    assert len(hits) == 1
    assert hits[0]["namespace"] == "quran"
    assert hits[0]["chunk_id"] == "quran:5:6-5:6"
    assert "text" in hits[0]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/retrieval/test_retrieve.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `src/bayan_graph/retrieval/retrieve.py`**

```python
# src/bayan_graph/retrieval/retrieve.py
from __future__ import annotations

from bayan_graph.retrieval.embed import EmbeddingInputType, embed_texts
from bayan_graph.retrieval.pinecone_client import PineconeStore


def retrieve_namespace(
    *,
    query: str,
    namespace: str,
    top_k: int,
    index_name: str,
    filter: dict | None = None,
) -> list[dict]:
    [vec] = embed_texts([query], input_type=EmbeddingInputType.QUERY)
    store = PineconeStore(index_name=index_name)
    matches = store.query(namespace=namespace, vector=vec, top_k=top_k, filter=filter)
    out = []
    for m in matches:
        md = m["metadata"]
        text_parts = [md.get("arabic", ""), md.get("english", "")]
        out.append(
            {
                "chunk_id": m["id"],
                "namespace": namespace,
                "score": m["score"],
                "text": "\n".join(p for p in text_parts if p),
                "metadata": md,
            }
        )
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/retrieval/test_retrieve.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/retrieval/test_retrieve.py src/bayan_graph/retrieval/retrieve.py
git commit -m "feat(retrieval): per-namespace retrieval wrapper"
```

---

### Task E2: Cohere rerank wrapper — `subagent` [PAR-GROUP 3]

**Files:**
- Create: `tests/retrieval/test_rerank.py`
- Create: `src/bayan_graph/retrieval/rerank.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/retrieval/test_rerank.py
from unittest.mock import MagicMock, patch

from bayan_graph.retrieval.rerank import rerank_hits


def test_rerank_reorders_by_rerank_score():
    hits = [
        {"chunk_id": "a", "text": "not relevant", "score": 0.9, "namespace": "quran", "metadata": {}},
        {"chunk_id": "b", "text": "wudu is obligatory", "score": 0.8, "namespace": "quran", "metadata": {}},
        {"chunk_id": "c", "text": "something else", "score": 0.85, "namespace": "quran", "metadata": {}},
    ]

    fake_resp = MagicMock()
    fake_resp.results = [
        MagicMock(index=1, relevance_score=0.95),
        MagicMock(index=2, relevance_score=0.5),
        MagicMock(index=0, relevance_score=0.1),
    ]
    fake_client = MagicMock()
    fake_client.rerank.return_value = fake_resp

    with patch("bayan_graph.retrieval.rerank._client", return_value=fake_client):
        out = rerank_hits(query="is wudu required?", hits=hits, top_n=2)

    assert [h["chunk_id"] for h in out] == ["b", "c"]
    assert out[0]["rerank_score"] == 0.95
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/retrieval/test_rerank.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `src/bayan_graph/retrieval/rerank.py`**

```python
# src/bayan_graph/retrieval/rerank.py
from __future__ import annotations

import os
from functools import lru_cache

import cohere

RERANK_MODEL = "rerank-multilingual-v3.0"


@lru_cache(maxsize=1)
def _client() -> cohere.ClientV2:
    return cohere.ClientV2(api_key=os.environ["COHERE_API_KEY"])


def rerank_hits(*, query: str, hits: list[dict], top_n: int) -> list[dict]:
    if not hits:
        return []
    documents = [h["text"] for h in hits]
    resp = _client().rerank(model=RERANK_MODEL, query=query, documents=documents, top_n=top_n)
    out: list[dict] = []
    for r in resp.results:
        hit = dict(hits[r.index])
        hit["rerank_score"] = r.relevance_score
        out.append(hit)
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/retrieval/test_rerank.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/retrieval/test_rerank.py src/bayan_graph/retrieval/rerank.py
git commit -m "feat(retrieval): Cohere rerank wrapper"
```

---

### Task E3: Parallel retrieval across namespaces — `subagent` [PAR-GROUP 3]

**Files:**
- Create: `tests/retrieval/test_parallel_retrieve.py`
- Create: `src/bayan_graph/retrieval/parallel.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/retrieval/test_parallel_retrieve.py
from unittest.mock import patch

from bayan_graph.retrieval.parallel import parallel_retrieve


def test_parallel_retrieve_fans_out_across_namespaces():
    def fake_single(*, query, namespace, top_k, index_name, filter=None):
        return [{"chunk_id": f"{namespace}:1", "namespace": namespace, "score": 0.9, "text": "t", "metadata": {}}]

    namespaces = ["quran", "hadith", "hanafi", "maliki", "shafii", "hanbali"]
    with patch("bayan_graph.retrieval.parallel.retrieve_namespace", side_effect=fake_single):
        result = parallel_retrieve(
            query="q",
            namespaces=namespaces,
            top_k=5,
            index_name="bayan-test",
        )

    assert set(result) == set(namespaces)
    for ns in namespaces:
        assert len(result[ns]) == 1
        assert result[ns][0]["namespace"] == ns


def test_parallel_retrieve_isolates_failures():
    def sometimes_fails(*, query, namespace, top_k, index_name, filter=None):
        if namespace == "hadith":
            raise RuntimeError("boom")
        return [{"chunk_id": f"{namespace}:1", "namespace": namespace, "score": 0.9, "text": "t", "metadata": {}}]

    with patch("bayan_graph.retrieval.parallel.retrieve_namespace", side_effect=sometimes_fails):
        result = parallel_retrieve(
            query="q",
            namespaces=["quran", "hadith"],
            top_k=5,
            index_name="bayan-test",
        )

    assert result["hadith"] == []
    assert len(result["quran"]) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/retrieval/test_parallel_retrieve.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `src/bayan_graph/retrieval/parallel.py`**

```python
# src/bayan_graph/retrieval/parallel.py
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed

import structlog

from bayan_graph.retrieval.retrieve import retrieve_namespace

log = structlog.get_logger(__name__)


def parallel_retrieve(
    *,
    query: str,
    namespaces: list[str],
    top_k: int,
    index_name: str,
    filter: dict | None = None,
    max_workers: int = 6,
) -> dict[str, list[dict]]:
    result: dict[str, list[dict]] = {ns: [] for ns in namespaces}
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {
            ex.submit(
                retrieve_namespace,
                query=query,
                namespace=ns,
                top_k=top_k,
                index_name=index_name,
                filter=filter,
            ): ns
            for ns in namespaces
        }
        for fut in as_completed(futures):
            ns = futures[fut]
            try:
                result[ns] = fut.result()
            except Exception as exc:  # noqa: BLE001
                log.warning("parallel_retrieve_failed", namespace=ns, error=str(exc))
                result[ns] = []
    return result
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/retrieval/test_parallel_retrieve.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/retrieval/test_parallel_retrieve.py src/bayan_graph/retrieval/parallel.py
git commit -m "feat(retrieval): parallel retrieval across namespaces with isolated failures"
```

---

## Phase F — Citation audit (Node V, real)

### Task F1: `validate_citations` with 5 checks — `subagent`

**Files:**
- Create: `tests/graph/test_validate_citations.py`
- Create: `src/bayan_graph/graph/nodes/validate_citations.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/graph/test_validate_citations.py
from bayan_graph.graph.nodes.validate_citations import validate_citations
from bayan_graph.models.analysis import CrossMadhabAnalysis
from bayan_graph.models.classification import Classification, ContextSensitivity
from bayan_graph.models.evidence import CitationSpan, Evidence, SourceLocation
from bayan_graph.models.position import Position, PrimaryView
from bayan_graph.models.vocabs import (
    AuthorityLevel,
    EvidenceRole,
    EvidenceType,
    Madhab,
    RoleInSource,
    TawaturStatus,
)


def _ev(id_="ev1"):
    return Evidence(
        id=id_,
        type=EvidenceType.QURAN,
        role=EvidenceRole.PRIMARY_SUPPORT,
        source=SourceLocation(work="Quran", anchor="5:6"),
        arabic="آ",
        english="translation",
        tawatur_status=TawaturStatus.MUTAWATIR_LAFZI,
        role_in_source=RoleInSource.PRIMARY_RULING,
        citation_span=CitationSpan(document_index=0, start=0, end=10),
    )


def _pos(madhab=Madhab.HANAFI, refs=("ev1",)):
    return Position(
        madhab=madhab,
        primary_view=PrimaryView(
            ruling="wajib",
            authority_level=AuthorityLevel.MU_TAMAD,
            evidence_refs=list(refs),
            reasoning="r",
        ),
    )


def test_validate_citations_clean_when_all_checks_pass():
    audit = validate_citations(
        classification=Classification(
            in_scope=True,
            scope_category="ibadat",
            topic="wudu",
            sub_topic=None,
            context_sensitivity=ContextSensitivity(
                is_context_sensitive=False,
                user_provided_context=None,
                missing_context_fields=[],
            ),
        ),
        positions=[_pos(m) for m in Madhab],
        cross_madhab_analysis=CrossMadhabAnalysis(),
        evidence=[_ev()],
    )
    assert audit.passed is True
    assert audit.tier == "clean"
    assert len(audit.checks) == 5


def test_validate_citations_fails_on_unresolved_ref():
    audit = validate_citations(
        classification=Classification(
            in_scope=True,
            scope_category="ibadat",
            topic="wudu",
            sub_topic=None,
            context_sensitivity=ContextSensitivity(
                is_context_sensitive=False,
                user_provided_context=None,
                missing_context_fields=[],
            ),
        ),
        positions=[_pos(m, refs=("ev404",)) for m in Madhab],
        cross_madhab_analysis=CrossMadhabAnalysis(),
        evidence=[_ev()],
    )
    assert audit.passed is False
    assert audit.failing_check == "evidence_refs_resolve"


def test_validate_citations_fails_on_empty_span():
    bad_ev = _ev()
    bad_ev = bad_ev.model_copy(update={"citation_span": CitationSpan(document_index=0, start=5, end=5)})
    audit = validate_citations(
        classification=Classification(
            in_scope=True,
            scope_category="ibadat",
            topic="wudu",
            sub_topic=None,
            context_sensitivity=ContextSensitivity(
                is_context_sensitive=False,
                user_provided_context=None,
                missing_context_fields=[],
            ),
        ),
        positions=[_pos(m) for m in Madhab],
        cross_madhab_analysis=CrossMadhabAnalysis(),
        evidence=[bad_ev],
    )
    assert audit.passed is False
    assert audit.failing_check == "spans_non_empty"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/graph/test_validate_citations.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `src/bayan_graph/graph/nodes/validate_citations.py`**

```python
# src/bayan_graph/graph/nodes/validate_citations.py
from __future__ import annotations

from bayan_graph.models.analysis import CrossMadhabAnalysis
from bayan_graph.models.audit import CitationAudit, CitationCheckResult
from bayan_graph.models.classification import Classification
from bayan_graph.models.evidence import Evidence
from bayan_graph.models.position import Position
from bayan_graph.models.vocabs import EvidenceType

HIERARCHY_ORDER = {
    EvidenceType.QURAN: 0,
    EvidenceType.HADITH: 1,
    EvidenceType.IJMA: 2,
    EvidenceType.QIYAS: 3,
    EvidenceType.MADHAB_TEXT: 4,
}


def _check_every_claim_has_span(positions: list[Position], ev_by_id: dict[str, Evidence]) -> CitationCheckResult:
    missing: list[str] = []
    for pos in positions:
        for ref in pos.primary_view.evidence_refs:
            if ref not in ev_by_id:
                continue  # different check
            if ev_by_id[ref].citation_span is None:
                missing.append(ref)
    passed = not missing
    return CitationCheckResult(
        name="every_claim_has_span",
        passed=passed,
        detail="ok" if passed else f"missing spans: {missing[:5]}",
    )


def _check_spans_non_empty(evidence: list[Evidence]) -> CitationCheckResult:
    empties = [e.id for e in evidence if e.citation_span.end <= e.citation_span.start]
    passed = not empties
    return CitationCheckResult(
        name="spans_non_empty",
        passed=passed,
        detail="ok" if passed else f"empty spans: {empties[:5]}",
    )


def _check_evidence_refs_resolve(
    positions: list[Position], ev_by_id: dict[str, Evidence]
) -> CitationCheckResult:
    unresolved: list[str] = []
    for pos in positions:
        for ref in pos.primary_view.evidence_refs:
            if ref not in ev_by_id:
                unresolved.append(ref)
        for av in pos.alternate_views:
            for ref in av.evidence_refs:
                if ref not in ev_by_id:
                    unresolved.append(ref)
    passed = not unresolved
    return CitationCheckResult(
        name="evidence_refs_resolve",
        passed=passed,
        detail="ok" if passed else f"unresolved refs: {unresolved[:5]}",
    )


def _check_hierarchy_respected(
    positions: list[Position], ev_by_id: dict[str, Evidence]
) -> CitationCheckResult:
    violations: list[str] = []
    for pos in positions:
        types = [
            ev_by_id[ref].type
            for ref in pos.primary_view.evidence_refs
            if ref in ev_by_id
        ]
        if types and types[0] == EvidenceType.MADHAB_TEXT and len(types) > 1:
            # If later refs include stronger proof, flag as order violation
            later = [t for t in types[1:] if HIERARCHY_ORDER[t] < HIERARCHY_ORDER[EvidenceType.MADHAB_TEXT]]
            if later:
                violations.append(pos.madhab.value)
    passed = not violations
    return CitationCheckResult(
        name="hierarchy_respected",
        passed=passed,
        detail="ok" if passed else f"madhab-first ordering violations: {violations}",
    )


def _check_madhab_source_alignment(
    positions: list[Position], ev_by_id: dict[str, Evidence]
) -> CitationCheckResult:
    violations: list[str] = []
    for pos in positions:
        for ref in pos.primary_view.evidence_refs:
            ev = ev_by_id.get(ref)
            if ev is None or ev.type != EvidenceType.MADHAB_TEXT:
                continue
            src_madhab = ev.source.work.lower()
            expected = {
                "hanafi": ["hidayah"],
                "maliki": ["mudawwanah"],
                "shafii": ["umm"],
                "hanbali": ["mughni"],
            }[pos.madhab.value]
            if not any(tag in src_madhab for tag in expected):
                violations.append(f"{pos.madhab.value}:{ev.id}")
    passed = not violations
    return CitationCheckResult(
        name="madhab_source_alignment",
        passed=passed,
        detail="ok" if passed else f"cross-madhab citations: {violations[:5]}",
    )


def validate_citations(
    *,
    classification: Classification,
    positions: list[Position],
    cross_madhab_analysis: CrossMadhabAnalysis,
    evidence: list[Evidence],
) -> CitationAudit:
    ev_by_id = {e.id: e for e in evidence}
    checks = [
        _check_every_claim_has_span(positions, ev_by_id),
        _check_spans_non_empty(evidence),
        _check_evidence_refs_resolve(positions, ev_by_id),
        _check_hierarchy_respected(positions, ev_by_id),
        _check_madhab_source_alignment(positions, ev_by_id),
    ]
    first_fail = next((c for c in checks if not c.passed), None)
    tier = "clean" if first_fail is None else "degraded"
    return CitationAudit(
        passed=first_fail is None,
        tier=tier,
        checks=checks,
        failing_check=first_fail.name if first_fail else None,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/graph/test_validate_citations.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add tests/graph/test_validate_citations.py src/bayan_graph/graph/nodes/validate_citations.py
git commit -m "feat(graph): Node V — validate_citations with 5 deterministic checks"
```

---

## Phase G — Stub nodes for walking skeleton

Stubs in this phase satisfy the schema so the end-to-end graph runs; Plan 3 replaces them with real LLM-backed implementations.

### Task G1: Stub `classify_query` — `subagent` [PAR-GROUP 4]

**Files:**
- Create: `tests/graph/test_classify_query.py`
- Create: `src/bayan_graph/graph/nodes/classify_query.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/graph/test_classify_query.py
from bayan_graph.graph.nodes.classify_query import classify_query


def test_stub_classifies_in_scope_ibadat():
    state = {"query": "Is wudu required for prayer?"}
    out = classify_query(state)
    c = out["classification"]
    assert c.in_scope is True
    assert c.scope_category == "ibadat"
    assert c.topic == "wudu"
    assert c.context_sensitivity.is_context_sensitive is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/graph/test_classify_query.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement stub**

```python
# src/bayan_graph/graph/nodes/classify_query.py
"""STUB: Node A classifier. Replaced by Claude Haiku in Plan 3."""
from __future__ import annotations

from bayan_graph.models.classification import Classification, ContextSensitivity
from bayan_graph.models.state import BayanState


def classify_query(state: BayanState) -> dict:
    query = state.get("query", "").lower()
    topic = "wudu" if "wudu" in query else "salah" if "prayer" in query or "salah" in query else "generic"
    classification = Classification(
        in_scope=True,
        scope_category="ibadat",
        topic=topic,
        sub_topic=None,
        context_sensitivity=ContextSensitivity(
            is_context_sensitive=False,
            user_provided_context=None,
            missing_context_fields=[],
        ),
    )
    return {"classification": classification}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/graph/test_classify_query.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/graph/test_classify_query.py src/bayan_graph/graph/nodes/classify_query.py
git commit -m "feat(graph): stub Node A classify_query"
```

---

### Task G2: Real `parallel_retrieve` node (Quran-only) + stub sufficiency — `subagent` [PAR-GROUP 4]

**Files:**
- Create: `tests/graph/test_retrieval_nodes.py`
- Create: `src/bayan_graph/graph/nodes/parallel_retrieve.py`
- Create: `src/bayan_graph/graph/nodes/check_sufficiency.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/graph/test_retrieval_nodes.py
from unittest.mock import patch

from bayan_graph.graph.nodes.check_sufficiency import check_sufficiency
from bayan_graph.graph.nodes.parallel_retrieve import parallel_retrieve_node


def test_parallel_retrieve_node_uses_quran_only_in_skeleton():
    fake_hits = {"quran": [{"chunk_id": "quran:5:6", "namespace": "quran", "score": 0.9, "text": "t", "metadata": {}}]}
    with patch(
        "bayan_graph.graph.nodes.parallel_retrieve.parallel_retrieve",
        return_value=fake_hits,
    ) as mock_pr:
        out = parallel_retrieve_node(
            {"query": "wudu?"},
            index_name="bayan-test",
            namespaces=["quran"],
        )
    assert "retrieval_hits" in out
    assert out["retrieval_hits"]["quran"][0]["chunk_id"] == "quran:5:6"
    mock_pr.assert_called_once()


def test_check_sufficiency_stub_always_sufficient():
    out = check_sufficiency(
        {"retrieval_hits": {"quran": [{"chunk_id": "x", "namespace": "quran", "score": 0.9, "text": "t", "metadata": {}}]}}
    )
    assert out["sufficiency_iterations"] == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/graph/test_retrieval_nodes.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `parallel_retrieve_node`**

```python
# src/bayan_graph/graph/nodes/parallel_retrieve.py
from __future__ import annotations

from bayan_graph.models.state import BayanState
from bayan_graph.retrieval.parallel import parallel_retrieve


def parallel_retrieve_node(
    state: BayanState,
    *,
    index_name: str,
    namespaces: list[str],
    top_k: int = 8,
) -> dict:
    hits = parallel_retrieve(
        query=state["query"],
        namespaces=namespaces,
        top_k=top_k,
        index_name=index_name,
    )
    return {"retrieval_hits": hits}
```

- [ ] **Step 4: Implement stub `check_sufficiency`**

```python
# src/bayan_graph/graph/nodes/check_sufficiency.py
"""STUB: Node C sufficiency loop. Plan 3 replaces with Haiku-driven assessment."""
from __future__ import annotations

from bayan_graph.models.state import BayanState


def check_sufficiency(state: BayanState) -> dict:
    current = state.get("sufficiency_iterations", 0)
    return {"sufficiency_iterations": current + 1}
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/graph/test_retrieval_nodes.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add tests/graph/test_retrieval_nodes.py src/bayan_graph/graph/nodes/parallel_retrieve.py src/bayan_graph/graph/nodes/check_sufficiency.py
git commit -m "feat(graph): parallel_retrieve node + stub check_sufficiency"
```

---

### Task G3: `hierarchy_sort` (simplified real) + stub `synthesize_positions` — `subagent` [PAR-GROUP 4]

**Files:**
- Create: `tests/graph/test_hierarchy_sort.py`
- Create: `src/bayan_graph/graph/nodes/hierarchy_sort.py`
- Create: `src/bayan_graph/graph/nodes/synthesize_positions.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/graph/test_hierarchy_sort.py
from bayan_graph.graph.nodes.hierarchy_sort import hierarchy_sort
from bayan_graph.graph.nodes.synthesize_positions import synthesize_positions


def test_hierarchy_sort_turns_hits_into_evidence_quran_first():
    state = {
        "query": "q",
        "retrieval_hits": {
            "quran": [
                {
                    "chunk_id": "quran:5:6",
                    "namespace": "quran",
                    "score": 0.9,
                    "text": "ar\nen",
                    "metadata": {
                        "source_type": "quran",
                        "anchor": "5:6",
                        "arabic": "آية",
                        "english": "verse",
                    },
                }
            ]
        },
    }
    out = hierarchy_sort(state)
    assert len(out["evidence"]) == 1
    assert out["evidence"][0].type.value == "quran"
    assert out["evidence"][0].id.startswith("ev-")


def test_synthesize_positions_stub_produces_four_positions():
    class _Ev:
        id = "ev-1"
    out = synthesize_positions({"evidence": [_Ev()]})
    assert len(out["positions"]) == 4
    for p in out["positions"]:
        assert p.primary_view.evidence_refs == ["ev-1"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/graph/test_hierarchy_sort.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `hierarchy_sort`**

```python
# src/bayan_graph/graph/nodes/hierarchy_sort.py
from __future__ import annotations

from bayan_graph.models.evidence import CitationSpan, Evidence, SourceLocation
from bayan_graph.models.state import BayanState
from bayan_graph.models.vocabs import (
    EvidenceRole,
    EvidenceType,
    RoleInSource,
    TawaturStatus,
)

_SOURCE_TYPE_TO_EV_TYPE = {
    "quran": EvidenceType.QURAN,
    "hadith": EvidenceType.HADITH,
    "madhab_text": EvidenceType.MADHAB_TEXT,
}
_NS_ORDER = {"quran": 0, "hadith": 1, "hanafi": 2, "maliki": 2, "shafii": 2, "hanbali": 2, "comparative": 3}


def _hit_to_evidence(hit: dict, idx: int) -> Evidence:
    md = hit.get("metadata", {})
    src_type = md.get("source_type", hit["namespace"])
    if hit["namespace"] in ("hanafi", "maliki", "shafii", "hanbali"):
        ev_type = EvidenceType.MADHAB_TEXT
    else:
        ev_type = _SOURCE_TYPE_TO_EV_TYPE.get(src_type, EvidenceType.MADHAB_TEXT)
    arabic = md.get("arabic") or "—"
    english = md.get("english") or hit.get("text") or "—"
    anchor = md.get("anchor") or hit["chunk_id"]
    return Evidence(
        id=f"ev-{idx}",
        type=ev_type,
        role=EvidenceRole.PRIMARY_SUPPORT,
        source=SourceLocation(
            work=md.get("work", hit["namespace"].title()),
            book=md.get("book"),
            chapter=md.get("chapter"),
            page=md.get("page"),
            anchor=anchor,
        ),
        arabic=arabic,
        english=english,
        tawatur_status=TawaturStatus.MUTAWATIR_LAFZI if ev_type == EvidenceType.QURAN else None,
        role_in_source=RoleInSource(md.get("role_in_source", "primary_ruling"))
        if md.get("role_in_source") in {r.value for r in RoleInSource}
        else RoleInSource.PRIMARY_RULING,
        citation_span=CitationSpan(document_index=idx, start=0, end=max(1, len(english))),
    )


def hierarchy_sort(state: BayanState) -> dict:
    hits_by_ns = state.get("retrieval_hits", {})
    flat = [(ns, h) for ns, hits in hits_by_ns.items() for h in hits]
    flat.sort(key=lambda pair: (_NS_ORDER.get(pair[0], 99), -pair[1].get("score", 0.0)))
    evidence = [_hit_to_evidence(h, i + 1) for i, (_, h) in enumerate(flat)]
    return {"evidence": evidence}
```

- [ ] **Step 4: Implement stub `synthesize_positions`**

```python
# src/bayan_graph/graph/nodes/synthesize_positions.py
"""STUB: Node F madhab synthesis. Plan 3 replaces with Citations-API-backed Sonnet call."""
from __future__ import annotations

from bayan_graph.models.position import Position, PrimaryView
from bayan_graph.models.state import BayanState
from bayan_graph.models.vocabs import AuthorityLevel, Madhab


def synthesize_positions(state: BayanState) -> dict:
    evidence = state.get("evidence", [])
    ref = evidence[0].id if evidence else "ev-1"
    positions = [
        Position(
            madhab=m,
            primary_view=PrimaryView(
                ruling="stub-ruling",
                authority_level=AuthorityLevel.MU_TAMAD,
                evidence_refs=[ref],
                reasoning="Walking skeleton: real synthesis arrives in Plan 3.",
            ),
        )
        for m in Madhab
    ]
    return {"positions": positions}
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/graph/test_hierarchy_sort.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add tests/graph/test_hierarchy_sort.py src/bayan_graph/graph/nodes/hierarchy_sort.py src/bayan_graph/graph/nodes/synthesize_positions.py
git commit -m "feat(graph): hierarchy_sort (real) + stub synthesize_positions"
```

---

### Task G4: Stub `cross_madhab_analysis` + `narrative_render` — `subagent` [PAR-GROUP 4]

**Files:**
- Create: `tests/graph/test_cross_madhab_and_narrative.py`
- Create: `src/bayan_graph/graph/nodes/cross_madhab_analysis.py`
- Create: `src/bayan_graph/graph/nodes/narrative_render.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/graph/test_cross_madhab_and_narrative.py
from bayan_graph.graph.nodes.cross_madhab_analysis import cross_madhab_analysis
from bayan_graph.graph.nodes.narrative_render import narrative_render
from bayan_graph.models.position import Position, PrimaryView
from bayan_graph.models.vocabs import AuthorityLevel, Madhab


def _positions():
    return [
        Position(
            madhab=m,
            primary_view=PrimaryView(
                ruling="wajib",
                authority_level=AuthorityLevel.MU_TAMAD,
                evidence_refs=["ev-1"],
                reasoning="r",
            ),
        )
        for m in Madhab
    ]


def test_cross_madhab_analysis_detects_agreement_when_rulings_match():
    out = cross_madhab_analysis({"positions": _positions()})
    cma = out["cross_madhab_analysis"]
    assert len(cma.points_of_agreement) == 1
    assert cma.points_of_disagreement == []


def test_narrative_render_produces_headline_and_narrative():
    out = narrative_render(
        {
            "query": "Is wudu required for prayer?",
            "positions": _positions(),
            "cross_madhab_analysis": cross_madhab_analysis({"positions": _positions()})[
                "cross_madhab_analysis"
            ],
        }
    )
    assert out["headline_answer"]
    assert "wudu" in out["narrative"].lower() or "prayer" in out["narrative"].lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/graph/test_cross_madhab_and_narrative.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement stub `cross_madhab_analysis`**

```python
# src/bayan_graph/graph/nodes/cross_madhab_analysis.py
"""STUB: Node G cross-madhab ikhtilaf synthesis. Plan 3 replaces with Sonnet + Ibn Rushd retrieval."""
from __future__ import annotations

from bayan_graph.models.analysis import CrossMadhabAnalysis, PointOfAgreement, PointOfDisagreement
from bayan_graph.models.state import BayanState


def cross_madhab_analysis(state: BayanState) -> dict:
    positions = state.get("positions", [])
    rulings = {p.madhab: p.primary_view.ruling for p in positions}
    unique_rulings = set(rulings.values())
    if len(unique_rulings) <= 1 and rulings:
        cma = CrossMadhabAnalysis(
            points_of_agreement=[
                PointOfAgreement(
                    summary=f"All four madhahib hold: {next(iter(unique_rulings))}",
                    evidence_refs=list({r for p in positions for r in p.primary_view.evidence_refs}),
                )
            ],
            points_of_disagreement=[],
        )
    else:
        cma = CrossMadhabAnalysis(
            points_of_agreement=[],
            points_of_disagreement=[
                PointOfDisagreement(
                    summary="Rulings diverge across madhahib (stub).",
                    madhab_stances=rulings,
                    reasons_for_ikhtilaf=[],
                )
            ],
        )
    return {"cross_madhab_analysis": cma}
```

- [ ] **Step 4: Implement stub `narrative_render`**

```python
# src/bayan_graph/graph/nodes/narrative_render.py
"""STUB: Node H narrative + headline. Plan 3 replaces with Sonnet narrative rendering."""
from __future__ import annotations

from bayan_graph.models.state import BayanState


def narrative_render(state: BayanState) -> dict:
    query = state.get("query", "")
    positions = state.get("positions", [])
    cma = state.get("cross_madhab_analysis")
    if cma and cma.points_of_agreement:
        headline = f"All four madhahib agree: {cma.points_of_agreement[0].summary}"
    else:
        stances = ", ".join(f"{p.madhab.value}={p.primary_view.ruling}" for p in positions)
        headline = f"The madhahib differ. Summary: {stances}"
    narrative = (
        f"Question: {query}\n\n"
        f"{headline}\n\n"
        "This is a walking-skeleton response; richer analysis arrives once Plan 3 replaces the stub nodes."
    )
    return {"headline_answer": headline, "narrative": narrative}
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/graph/test_cross_madhab_and_narrative.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add tests/graph/test_cross_madhab_and_narrative.py src/bayan_graph/graph/nodes/cross_madhab_analysis.py src/bayan_graph/graph/nodes/narrative_render.py
git commit -m "feat(graph): stub cross_madhab_analysis and narrative_render"
```

---

## Phase H — Graph assembly + walking skeleton

### Task H1: Assemble LangGraph `StateGraph` — `main-thread`

**Files:**
- Create: `tests/graph/test_graph_assembly.py`
- Create: `src/bayan_graph/graph/build.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/graph/test_graph_assembly.py
from bayan_graph.graph.build import build_graph


def test_graph_compiles_and_has_expected_nodes():
    graph = build_graph(index_name="bayan-test", namespaces=["quran"])
    compiled = graph.compile()
    node_names = set(compiled.get_graph().nodes)
    for expected in (
        "classify_query",
        "parallel_retrieve",
        "check_sufficiency",
        "hierarchy_sort",
        "synthesize_positions",
        "cross_madhab_analysis",
        "narrative_render",
        "validate_citations",
    ):
        assert expected in node_names, f"missing node {expected}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/graph/test_graph_assembly.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `src/bayan_graph/graph/build.py`**

```python
# src/bayan_graph/graph/build.py
from __future__ import annotations

from functools import partial

from langgraph.graph import END, START, StateGraph

from bayan_graph.graph.nodes.check_sufficiency import check_sufficiency
from bayan_graph.graph.nodes.classify_query import classify_query
from bayan_graph.graph.nodes.cross_madhab_analysis import cross_madhab_analysis
from bayan_graph.graph.nodes.hierarchy_sort import hierarchy_sort
from bayan_graph.graph.nodes.narrative_render import narrative_render
from bayan_graph.graph.nodes.parallel_retrieve import parallel_retrieve_node
from bayan_graph.graph.nodes.synthesize_positions import synthesize_positions
from bayan_graph.graph.nodes.validate_citations import validate_citations
from bayan_graph.models.state import BayanState


def _validate_citations_node(state: BayanState) -> dict:
    audit = validate_citations(
        classification=state["classification"],
        positions=state.get("positions", []),
        cross_madhab_analysis=state["cross_madhab_analysis"],
        evidence=state.get("evidence", []),
    )
    return {"citation_audit": audit}


def build_graph(*, index_name: str, namespaces: list[str]) -> StateGraph:
    sg = StateGraph(BayanState)
    sg.add_node("classify_query", classify_query)
    sg.add_node(
        "parallel_retrieve",
        partial(parallel_retrieve_node, index_name=index_name, namespaces=namespaces),
    )
    sg.add_node("check_sufficiency", check_sufficiency)
    sg.add_node("hierarchy_sort", hierarchy_sort)
    sg.add_node("synthesize_positions", synthesize_positions)
    sg.add_node("cross_madhab_analysis", cross_madhab_analysis)
    sg.add_node("narrative_render", narrative_render)
    sg.add_node("validate_citations", _validate_citations_node)

    sg.add_edge(START, "classify_query")
    sg.add_edge("classify_query", "parallel_retrieve")
    sg.add_edge("parallel_retrieve", "check_sufficiency")
    sg.add_edge("check_sufficiency", "hierarchy_sort")
    sg.add_edge("hierarchy_sort", "synthesize_positions")
    sg.add_edge("synthesize_positions", "cross_madhab_analysis")
    sg.add_edge("cross_madhab_analysis", "narrative_render")
    sg.add_edge("narrative_render", "validate_citations")
    sg.add_edge("validate_citations", END)
    return sg
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/graph/test_graph_assembly.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/graph/test_graph_assembly.py src/bayan_graph/graph/build.py
git commit -m "feat(graph): assemble walking-skeleton StateGraph"
```

---

### Task H2: End-to-end walking-skeleton integration test — `main-thread`

**Files:**
- Create: `tests/graph/test_walking_skeleton.py`
- Create: `src/bayan_graph/graph/run.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/graph/test_walking_skeleton.py
from unittest.mock import patch

from bayan_graph.graph.run import run_query
from bayan_graph.models.response import AnswerResponse


def _fake_hits():
    return {
        "quran": [
            {
                "chunk_id": "quran:5:6",
                "namespace": "quran",
                "score": 0.92,
                "text": "ar\nen",
                "metadata": {
                    "source_type": "quran",
                    "anchor": "5:6",
                    "work": "Quran",
                    "arabic": "يَا أَيُّهَا الَّذِينَ آمَنُوا",
                    "english": "O you who believe",
                },
            }
        ]
    }


def test_end_to_end_returns_schema_valid_response():
    with patch(
        "bayan_graph.graph.nodes.parallel_retrieve.parallel_retrieve",
        return_value=_fake_hits(),
    ):
        response = run_query(
            query="Is wudu required for prayer?",
            index_name="bayan-test",
            namespaces=["quran"],
        )

    assert isinstance(response, AnswerResponse)
    assert response.query.startswith("Is wudu")
    assert len(response.positions) == 4
    assert response.citation_audit.passed is True
    assert response.citation_audit.tier == "clean"
    assert response.headline_answer
    assert response.narrative
    # Roundtrip JSON
    dumped = response.model_dump(mode="json")
    AnswerResponse.model_validate(dumped)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/graph/test_walking_skeleton.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `src/bayan_graph/graph/run.py`**

```python
# src/bayan_graph/graph/run.py
from __future__ import annotations

import time
import uuid

from bayan_graph.graph.build import build_graph
from bayan_graph.models.audit import Disclosure, Meta
from bayan_graph.models.response import AnswerResponse


def run_query(
    *,
    query: str,
    index_name: str,
    namespaces: list[str],
) -> AnswerResponse:
    started = time.monotonic()
    graph = build_graph(index_name=index_name, namespaces=namespaces).compile()
    query_id = str(uuid.uuid4())
    state = graph.invoke({"query": query, "query_id": query_id})
    latency_ms = int((time.monotonic() - started) * 1000)

    sources_consulted = sorted(
        {ev.source.work for ev in state.get("evidence", [])}
    )
    return AnswerResponse(
        query=query,
        headline_answer=state["headline_answer"],
        classification=state["classification"],
        positions=state["positions"],
        cross_madhab_analysis=state["cross_madhab_analysis"],
        evidence=state["evidence"],
        citation_audit=state["citation_audit"],
        disclosure=Disclosure(
            caveat="Informational output from Bayan-Graph. Not a fatwa. Consult a qualified scholar.",
            sources_consulted=sources_consulted,
        ),
        meta=Meta(
            query_id=query_id,
            latency_ms=latency_ms,
            model_versions={
                "classifier": "stub",
                "synthesizer": "stub",
                "narrative": "stub",
            },
            retrieval_stats={
                "chunks_considered": sum(
                    len(v) for v in state.get("retrieval_hits", {}).values()
                ),
                "evidence_count": len(state.get("evidence", [])),
            },
        ),
        narrative=state.get("narrative"),
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/graph/test_walking_skeleton.py -v`
Expected: PASS.

- [ ] **Step 5: Run the full test suite**

Run: `uv run pytest -v`
Expected: all tests pass; no xfails.

- [ ] **Step 6: Commit**

```bash
git add tests/graph/test_walking_skeleton.py src/bayan_graph/graph/run.py
git commit -m "feat(graph): end-to-end walking skeleton with schema-valid AnswerResponse"
```

---

### Task H3: CI workflow + README + final commit — `main-thread`

**Files:**
- Create: `.github/workflows/ci.yml`
- Create: `README.md`

- [ ] **Step 1: Write `.github/workflows/ci.yml`**

```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v3
        with:
          version: "0.4.0"
      - name: Set up Python
        run: uv python install 3.12
      - name: Install deps
        run: uv sync
      - name: Lint
        run: uv run ruff check src tests
      - name: Format check
        run: uv run ruff format --check src tests
      - name: Type check
        run: uv run mypy src/bayan_graph
      - name: Test
        run: uv run pytest -m "not integration"
```

- [ ] **Step 2: Write `README.md`**

```markdown
# Bayan-Graph

Multi-madhab comparative fiqh engine with enforced citations.

Stack: Python 3.12, LangGraph, Anthropic Citations API, Cohere embed/rerank, Pinecone.

## Quick start

```bash
make install          # uv sync + pre-commit
cp .env.example .env  # fill in keys
make ci               # lint + type + test
```

## Structure

- `src/bayan_graph/models/` — typed domain schema (evidence, positions, analysis, audit)
- `src/bayan_graph/ingestion/` — Arabic normalization, loaders, chunkers
- `src/bayan_graph/retrieval/` — Cohere embed/rerank, Pinecone client, parallel retrieval
- `src/bayan_graph/graph/` — LangGraph nodes and assembly
- `src/bayan_graph/cli/` — ingestion CLIs (`ingest_quran`)

## Ingest the Quran

```bash
uv run python scripts/fetch_quran.py
uv run python -m bayan_graph.cli.ingest_quran \
  --arabic data/raw/quran/quran-uthmani.txt \
  --english data/raw/quran/en-sahih.txt
```

## Run a query (walking skeleton)

```python
from bayan_graph.graph.run import run_query

resp = run_query(
    query="Is wudu required for prayer?",
    index_name="bayan-graph",
    namespaces=["quran"],
)
print(resp.model_dump_json(indent=2))
```

Current state: Plan 1 (walking skeleton) produces a schema-valid `AnswerResponse`
for a Quran-only corpus. Classifier, synthesizer, cross-madhab, and narrative
nodes are stubs — Plan 3 replaces them with Claude-backed implementations.
```

- [ ] **Step 3: Run CI locally**

Run: `make ci`
Expected: lint + typecheck + pytest all pass.

- [ ] **Step 4: Commit**

```bash
git add .github/workflows/ci.yml README.md
git commit -m "chore: add CI workflow and README"
```

---

## Self-Review

**Spec coverage (Plan 1 scope only):**

- Controlled vocabularies → Tasks B1, B5 ✓
- Evidence + citation span → Task B2 ✓
- Position with alternate_views + conditional_cases → Task B3 ✓
- Classification + context_sensitivity → Task B4 ✓
- CrossMadhabAnalysis + ikhtilaf reasons → Task B5 ✓
- CitationAudit with 5 checks → Tasks B6, F1 ✓
- Disclosure + Meta + AnswerResponse → Tasks B6, B7 ✓
- BayanState TypedDict → Task B7 ✓
- Arabic normalization profiles + quality score → Task C1 ✓
- Quran loader + chunker + ingestion CLI → Tasks D1, D2, D5 ✓
- Cohere embed (document/query) → Task D3 ✓
- Pinecone upsert/query → Task D4 ✓
- Retrieval wrapper + rerank + parallel retrieval → Tasks E1, E2, E3 ✓
- Node V real validator → Task F1 ✓
- Stub nodes for A, C, F, G, H → Tasks G1, G2, G3, G4 ✓
- LangGraph assembly + end-to-end run → Tasks H1, H2 ✓
- CI + README → Task H3 ✓

**Deferred to later plans (not Plan 1 scope):**
- Hadith/madhab-book/Ibn Rushd ingestion → Plan 2
- Real classifier/synthesizer/cross-madhab/narrative with Citations API → Plan 3
- Three-tier escalation + refusal UX + observability → Plan 4
- 80-question eval harness → Plan 5
- FastAPI + Next.js UI → Plan 6
- Published skills extraction → Plan 7

**Placeholder scan:** No TBD/TODO/"implement later" strings in any task. Stubs are explicitly labeled in file docstrings and have schema-conforming behavior so the end-to-end graph runs.

**Type consistency:** Checked cross-task references:
- `EvidenceType`, `EvidenceRole`, `RoleInSource`, `TawaturStatus`, `AuthorityLevel`, `IkhtilafReasonType`, `Madhab` enums consistent across B1→B7, F1, G1–G4.
- `Evidence`, `Position`, `Classification`, `CrossMadhabAnalysis`, `CitationAudit`, `AnswerResponse` signatures match in use sites (B7, F1, G3, G4, H2).
- `BayanState` keys (`query`, `classification`, `retrieval_hits`, `evidence`, `positions`, `cross_madhab_analysis`, `headline_answer`, `narrative`, `citation_audit`) consistent across G1–G4, H1, H2.

---

## Parallelization summary

For subagent-driven execution, the following groups can run concurrently once their prereqs ship:

- **PAR-GROUP 1** (after A1): A2 + A3 — directory scaffold and tooling files.
- **PAR-GROUP 2** (after A2, A3): B1, B2, B3, B4, B5, B6 — each Pydantic module is independent. B7 depends on B2–B6.
- **PAR-GROUP 3** (after B7, D4): E1, E2, E3 — retrieval components don't depend on each other.
- **PAR-GROUP 4** (after F1): G1, G2, G3, G4 — node stubs are independent.

Sequential anchors: A1 → (PAR-1) → B7 → C1 → D1 → D2 → D3 → D4 → D5 → (PAR-3) → F1 → (PAR-4) → H1 → H2 → H3.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-04-22-bayan-graph-plan-01-foundation-walking-skeleton.md`. Two execution options:

**1. Subagent-Driven (recommended)** — I dispatch a fresh Sonnet subagent per task (or parallel group), review between tasks, fast iteration. Best fit given the parallelization groups above.

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints.

Which approach?

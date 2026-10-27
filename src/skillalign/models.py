"""Data contracts shared by every stage (PLAN.md §4–§5.5, IMPLEMENTATION.md §3).

Gold annotations and system predictions use the same `Annotation` model, so the
evaluation compares like with like. Validators here check internal consistency
(IDs resolve, each finding type has the fields it needs). Checking evidence
against the files on disk lives in `spans.py`.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", use_enum_values=False)


# ---------------------------------------------------------------- enums


class Capability(StrEnum):
    """Capability taxonomy v1 (PLAN.md §4.3)."""

    FS_READ = "FS_READ"
    FS_WRITE = "FS_WRITE"
    FS_DELETE = "FS_DELETE"
    FS_ENUMERATE = "FS_ENUMERATE"
    ENV_READ = "ENV_READ"
    CREDENTIAL_ACCESS = "CREDENTIAL_ACCESS"
    NET_OUTBOUND = "NET_OUTBOUND"
    NET_INBOUND = "NET_INBOUND"
    DNS = "DNS"
    DOWNLOAD_EXEC = "DOWNLOAD_EXEC"
    PROC_EXEC = "PROC_EXEC"
    CODE_EVAL = "CODE_EVAL"
    PRIV_ESCALATION = "PRIV_ESCALATION"
    PERSISTENCE = "PERSISTENCE"
    SYS_CONFIG_MODIFY = "SYS_CONFIG_MODIFY"
    PKG_INSTALL = "PKG_INSTALL"
    ENCODE_OBFUSCATE = "ENCODE_OBFUSCATE"
    DATA_TRANSFORM = "DATA_TRANSFORM"
    AGENT_INSTRUCTION = "AGENT_INSTRUCTION"
    TOOL_PERMISSION = "TOOL_PERMISSION"
    EXTERNAL_API = "EXTERNAL_API"
    USER_INTERACTION = "USER_INTERACTION"
    COMPOSITE = "COMPOSITE"  # a source→transform→sink path; parts listed in composite_of


class FileRole(StrEnum):
    SKILL_MD = "skill_md"
    DOC = "doc"
    CODE = "code"
    CONFIG = "config"
    DATA = "data"
    VIRTUAL = "virtual"  # fenced code block extracted from a Markdown file


class Polarity(StrEnum):
    POSITIVE = "positive"
    NEGATIVE = "negative"  # "never uploads data", "works offline"


class Modality(StrEnum):
    FUNCTIONAL = "functional"
    SECURITY = "security"
    SCOPE = "scope"
    DATA_HANDLING = "data-handling"


class Origin(StrEnum):
    CODE = "code"
    AGENT_INSTRUCTION = "agent-instruction"
    CONFIG = "config"
    DEPENDENCY = "dependency"


class Reachability(StrEnum):
    ENTRY = "entry"
    REACHABLE = "reachable"
    DORMANT = "dormant"


class AlignmentStatus(StrEnum):
    FULFILLED = "fulfilled"
    PARTIAL = "partial"
    UNFULFILLED = "unfulfilled"


class FindingType(StrEnum):
    UI = "UI"
    UI_PARTIAL = "UI-partial"
    UB = "UB"
    SD = "SD"


class Axis(StrEnum):
    OBJECT = "object"
    SCOPE = "scope"
    DESTINATION = "destination"
    CONDITION = "condition"
    MAGNITUDE = "magnitude"
    NEGATIVE_CLAIM = "negative-claim"


class Risk(StrEnum):
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Suspected(StrEnum):
    DRIFT = "drift"
    DELIBERATE = "deliberate"
    UNCLEAR = "unclear"


class Split(StrEnum):
    FIXTURE = "fixture"
    NATURAL_RANDOM = "natural-random"
    NATURAL_ENRICHED = "natural-enriched"
    DRIFT = "drift"
    INJECTED = "injected"


class SkillLabel(StrEnum):
    ALIGNED = "aligned"
    MISALIGNED = "misaligned"


# ---------------------------------------------------------------- evidence


class Span(_Model):
    """A line range in one file, relative to the skill root. 1-based, inclusive."""

    file: str = Field(min_length=1)
    line_start: int = Field(ge=1)
    line_end: int = Field(ge=1)

    @model_validator(mode="after")
    def _ordered(self) -> Self:
        if self.line_end < self.line_start:
            raise ValueError(f"line_end {self.line_end} < line_start {self.line_start}")
        if self.file.startswith("/") or ".." in self.file.split("/"):
            raise ValueError(f"span file must be relative to the skill root: {self.file!r}")
        return self


class Evidence(Span):
    """A span plus the text it points at. `quote` must appear within those lines."""

    quote: str = Field(min_length=1)


# ---------------------------------------------------------------- skill


class FileArtifact(_Model):
    path: str
    role: FileRole
    language: str | None = None  # "python", "shell", "javascript", "markdown", ...
    sha256: str | None = None
    parent: str | None = None  # for virtual files: the Markdown file they came from
    line_offset: int = 0  # virtual line 1 == parent line (line_offset + 1)


class Skill(_Model):
    skill_id: str
    platform: str | None = None  # "github", "clawhub", "official", "fixture", ...
    source_url: str | None = None
    commit: str | None = None
    license: str | None = None
    files: list[FileArtifact] = Field(default_factory=list)


# ---------------------------------------------------------------- intents and behaviors


class Intent(_Model):
    """One atomic claim. For a negative claim ("never sends data"), `action`/`object`
    describe what is ruled out and `expected_capabilities` lists the capabilities it forbids."""

    id: str = Field(pattern=r"^I\d+$")
    category: str | None = None
    action: str
    object: str
    constraints: list[str] = Field(default_factory=list)
    polarity: Polarity = Polarity.POSITIVE
    modality: Modality = Modality.FUNCTIONAL
    expected_capabilities: list[Capability] = Field(default_factory=list)
    sources: list[Evidence] = Field(min_length=1)


class Behavior(_Model):
    id: str = Field(pattern=r"^B\d+$")
    capability: Capability
    action: str
    target: str | None = None  # what is read/written/run: a path, env var, command
    destination: str | None = None  # where data goes: a host, a path
    conditions: list[str] = Field(default_factory=list)
    origin: Origin = Origin.CODE
    reachability: Reachability = Reachability.REACHABLE
    evidence: list[Evidence] = Field(min_length=1)
    call_chain: list[str] = Field(default_factory=list)
    composite_of: list[str] = Field(default_factory=list)
    observed_dynamically: bool = False

    @model_validator(mode="after")
    def _composite(self) -> Self:
        if self.capability is Capability.COMPOSITE and len(self.composite_of) < 2:
            raise ValueError(f"{self.id}: a COMPOSITE behavior needs at least 2 parts")
        if self.capability is not Capability.COMPOSITE and self.composite_of:
            raise ValueError(f"{self.id}: only COMPOSITE behaviors may list composite_of")
        return self


# ---------------------------------------------------------------- judge I/O


class Card(_Model):
    """Templated evidence summary of one behavior: the only view of code the judge gets."""

    behavior_id: str
    text: str
    snippet: str
    span: Span
    chain: list[str] = Field(default_factory=list)


class ConstraintAnswer(_Model):
    constraint: str
    respected: Literal["yes", "no", "unknown"]


class ChecklistAnswer(_Model):
    """D2 judge output for one (intent, candidate cards) pair. Also the SFT target."""

    achieves_action: Literal["yes", "partly", "no"]
    constraints: list[ConstraintAnswer] = Field(default_factory=list)
    axis_mismatch: Axis | None = None
    contradicts_negative: bool = False
    cited_cards: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------- findings and annotations


class Alignment(_Model):
    intent: str
    behaviors: list[str] = Field(default_factory=list)
    status: AlignmentStatus


class Finding(_Model):
    id: str = Field(pattern=r"^F\d+$")
    type: FindingType
    intent: str | None = None
    behaviors: list[str] = Field(default_factory=list)
    divergence_axis: Axis | None = None
    expected_result: str | None = None
    actual_result: str | None = None
    evidence: list[Span] = Field(min_length=1)
    call_chain: list[str] = Field(default_factory=list)
    risk: Risk = Risk.LOW
    intent_suspected: Suspected = Suspected.UNCLEAR
    rationale: str = ""

    @model_validator(mode="after")
    def _type_fields(self) -> Self:
        t = self.type
        if t in (FindingType.UI, FindingType.UI_PARTIAL) and self.intent is None:
            raise ValueError(f"{self.id}: {t} needs an intent")
        if t is FindingType.UB:
            if self.intent is not None:
                raise ValueError(f"{self.id}: UB must not link an intent")
            if not self.behaviors:
                raise ValueError(f"{self.id}: UB needs at least one behavior")
        if t is FindingType.SD:
            if self.intent is None or not self.behaviors:
                raise ValueError(f"{self.id}: SD needs an intent and at least one behavior")
            if self.divergence_axis is None:
                raise ValueError(f"{self.id}: SD needs a divergence_axis")
        if t is not FindingType.SD and self.divergence_axis is not None:
            raise ValueError(f"{self.id}: only SD findings have a divergence_axis")
        return self


class Provenance(_Model):
    """Who produced an annotation: a person, the pre-annotator, SkillAlign, a baseline."""

    system: str  # "human", "preannotate", "skillalign", "baseline:L1", ...
    model: str | None = None
    prompt_hash: str | None = None
    seed: int | None = None
    notes: str | None = None


class Annotation(_Model):
    """One skill's labels (gold) or one system's output (prediction). PLAN.md §5.5."""

    skill_id: str
    split: Split | None = None
    category: str | None = None
    intents: list[Intent] = Field(default_factory=list)
    behaviors: list[Behavior] = Field(default_factory=list)
    alignment: list[Alignment] = Field(default_factory=list)
    findings: list[Finding] = Field(default_factory=list)
    skill_label: SkillLabel
    annotators: list[str] = Field(default_factory=list)
    prefilled: bool = False
    adjudicated: bool = False
    minutes_spent: float | None = None
    provenance: Provenance | None = None

    @model_validator(mode="after")
    def _references(self) -> Self:
        errors: list[str] = []
        intent_ids = _unique_ids("intent", [i.id for i in self.intents], errors)
        behavior_ids = _unique_ids("behavior", [b.id for b in self.behaviors], errors)
        _unique_ids("finding", [f.id for f in self.findings], errors)

        for b in self.behaviors:
            for part in b.composite_of:
                if part not in behavior_ids:
                    errors.append(f"{b.id}.composite_of: unknown behavior {part}")
                elif part == b.id:
                    errors.append(f"{b.id}.composite_of: refers to itself")
        for a in self.alignment:
            if a.intent not in intent_ids:
                errors.append(f"alignment: unknown intent {a.intent}")
            errors += [f"alignment[{a.intent}]: unknown behavior {x}"
                       for x in a.behaviors if x not in behavior_ids]
        for f in self.findings:
            if f.intent is not None and f.intent not in intent_ids:
                errors.append(f"{f.id}: unknown intent {f.intent}")
            errors += [f"{f.id}: unknown behavior {x}" for x in f.behaviors
                       if x not in behavior_ids]

        has_findings = bool(self.findings)
        if has_findings != (self.skill_label is SkillLabel.MISALIGNED):
            errors.append(
                f"skill_label={self.skill_label} but there are {len(self.findings)} findings"
            )
        if errors:
            raise ValueError("; ".join(errors))
        return self

    def intent(self, intent_id: str) -> Intent:
        return next(i for i in self.intents if i.id == intent_id)

    def behavior(self, behavior_id: str) -> Behavior:
        return next(b for b in self.behaviors if b.id == behavior_id)


def _unique_ids(kind: str, ids: list[str], errors: list[str]) -> set[str]:
    seen: set[str] = set()
    for x in ids:
        if x in seen:
            errors.append(f"duplicate {kind} id {x}")
        seen.add(x)
    return seen


#: Models exported to docs/schema/ by `skillalign schema`.
SCHEMA_MODELS: dict[str, type[BaseModel]] = {
    "annotation": Annotation,
    "intent": Intent,
    "behavior": Behavior,
    "finding": Finding,
    "card": Card,
    "checklist_answer": ChecklistAnswer,
    "skill": Skill,
}

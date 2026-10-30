"""L1: zero-shot, the whole skill (cut to the context) in one prompt, findings in our schema.

It reuses the pre-annotation prompt and checks, so L1-32B is the pre-annotator scored as a
system. That makes it an upper-bound-ish reference rather than a fair competitor (PLAN.md §7).
Run it with the 8B model for L1 and the 32B model for L1-32B.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from skillalign.bench.preannotate import preannotate_skill
from skillalign.llm import LLMClient
from skillalign.models import Annotation, Provenance, SkillLabel


def run_l1(client: LLMClient, skill_dir: Path, skill_id: str) -> Annotation:
    """The model's `Annotation` for one skill. A failed reply is an empty, aligned prediction."""
    with tempfile.TemporaryDirectory() as tmp:
        out = preannotate_skill(client, skill_dir, skill_id, Path(tmp))
        system = f"baseline:L1[{client.model}]"
        if out.status != "done" or out.path is None:
            return Annotation(skill_id=skill_id, skill_label=SkillLabel.ALIGNED,
                              provenance=Provenance(system=system, model=client.model,
                                                    notes=f"no valid reply: {out.error}"))
        ann = Annotation.model_validate_json(out.path.read_text())
    prov = (ann.provenance or Provenance(system=system)).model_copy(update={"system": system})
    return ann.model_copy(update={"provenance": prov, "prefilled": False})

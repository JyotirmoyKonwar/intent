import json
from pathlib import Path
from skillalign.config import get_settings
from skillalign.intents.extract import file_stem
from skillalign.models import Annotation
from skillalign.spans import check_annotation
s = get_settings(); b = s.data_dir / "bench"
bad = ok = 0
for e in json.loads((b / "sample.json").read_text()):
    f = b / "preannotated" / f"{file_stem(e['skill_id'])}.json"
    if not f.exists(): continue
    errs = check_annotation(Annotation.model_validate_json(f.read_text()), s.data_dir / "corpus" / e["path"])
    ok += not errs; bad += bool(errs)
    if errs: print(e["skill_id"][-50:], errs[:2])
print("clean", ok, "with span errors", bad)

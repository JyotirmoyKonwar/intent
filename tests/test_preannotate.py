import json

from skillalign.bench.preannotate import preannotate_skill, run_batch, skill_text
from skillalign.models import Annotation
from skillalign.normalize.loader import load_skill

from .conftest import FIXTURES, GOLD, SKILLS
from .fake_llm import Router, client_for

GOLD_11 = (GOLD / "11-sd-destination.json").read_text()
ID_OK = "fixture/11-sd-destination"
ID_BAD = "fixture/12-sd-negative-claim"


def route(prompt: str):
    if "fixture/12-sd-negative-claim" in prompt:
        return "not json at all", "stop"
    return GOLD_11, "stop"


def test_one_skill_passes_one_fails_and_rerun_skips(tmp_path):
    out_dir = tmp_path / "preannotated"
    router = Router(route)
    client = client_for(router)

    ok = preannotate_skill(client, SKILLS / "11-sd-destination", ID_OK, out_dir)
    bad = preannotate_skill(client, SKILLS / "12-sd-negative-claim", ID_BAD, out_dir)

    assert ok.status == "done" and bad.status == "failed"
    ann = Annotation.model_validate_json((out_dir / "fixture__11-sd-destination.json").read_text())
    assert ann.skill_id == ID_OK and ann.prefilled
    assert ann.provenance.system == "preannotate"
    assert ann.provenance.model == "test-model"
    assert ann.provenance.prompt_hash and ann.provenance.seed == client.settings.seed
    assert "date" in ann.provenance.notes
    failed = json.loads((out_dir / "fixture__12-sd-negative-claim.failed.json").read_text())
    assert failed["skill_id"] == ID_BAD and failed["error"]

    calls_before = router.count("fixture/11-sd-destination")
    again = preannotate_skill(client, SKILLS / "11-sd-destination", ID_OK, out_dir)
    assert again.status == "skipped"
    assert router.count("fixture/11-sd-destination") == calls_before


def test_span_errors_are_recorded_not_dropped(tmp_path):
    # The gold reply cites lines of 11-sd-destination; run it on another skill, so the
    # quotes do not match and the spans are wrong.
    out_dir = tmp_path / "preannotated"
    outcome = preannotate_skill(client_for(Router(route)), SKILLS / "13-sd-scope",
                                "fixture/13-sd-scope", out_dir)
    assert outcome.status == "done"
    ann = Annotation.model_validate_json((out_dir / "fixture__13-sd-scope.json").read_text())
    assert "span errors" in ann.provenance.notes
    assert (out_dir / "fixture__13-sd-scope.failed.json").exists()


def test_run_batch_limit_on_a_sample(tmp_path):
    sample = [
        {"key": "a", "skill_id": ID_OK, "path": "skills/11-sd-destination", "stratum": "random"},
        {"key": "b", "skill_id": ID_BAD, "path": "skills/12-sd-negative-claim",
         "stratum": "random"},
        {"key": "c", "skill_id": "fixture/13-sd-scope", "path": "skills/13-sd-scope",
         "stratum": "enriched"},
        {"key": "d", "skill_id": "fixture/14-sd-condition", "path": "skills/14-sd-condition",
         "stratum": "enriched"},
    ]
    out_dir = tmp_path / "preannotated"
    outcomes = run_batch(client_for(Router(route)), sample, FIXTURES, out_dir, limit=3)
    assert [o.status for o in outcomes] == ["done", "failed", "done"]
    assert not (out_dir / "fixture__14-sd-condition.json").exists()


def test_skill_text_numbers_lines_and_puts_skill_md_first():
    loaded = load_skill(SKILLS / "11-sd-destination", skill_id=ID_OK)
    text, left_out = skill_text(loaded)
    assert left_out == []
    assert text.startswith("=== FILE SKILL.md ===\n1| ")
    assert "\n1| " in text


def test_quotes_are_copied_from_the_files_not_taken_from_the_model(tmp_path):
    reply = json.loads(GOLD_11)
    for intent in reply["intents"]:
        for src in intent["sources"]:
            src["quote"] = "words the model made up"
    out_dir = tmp_path / "preannotated"
    outcome = preannotate_skill(client_for(Router(lambda prompt: (json.dumps(reply), "stop"))),
                                SKILLS / "11-sd-destination", ID_OK, out_dir)
    assert outcome.status == "done" and outcome.error is None
    ann = Annotation.model_validate_json((out_dir / "fixture__11-sd-destination.json").read_text())
    lines = (SKILLS / "11-sd-destination" / "SKILL.md").read_text().splitlines()
    for intent in ann.intents:
        for src in intent.sources:
            assert src.quote in "\n".join(lines[src.line_start - 1 : src.line_end])
    assert "made up" not in out_dir.joinpath("fixture__11-sd-destination.json").read_text()


def test_a_timeout_is_logged_and_the_batch_goes_on(tmp_path):
    import httpx
    import openai

    def route(prompt: str):
        if "fixture/12-sd-negative-claim" in prompt:
            raise openai.APITimeoutError(request=httpx.Request("POST", "http://test"))
        return GOLD_11, "stop"

    sample = [{"key": "b", "skill_id": ID_BAD, "path": "skills/12-sd-negative-claim",
               "stratum": "random"},
              {"key": "a", "skill_id": ID_OK, "path": "skills/11-sd-destination",
               "stratum": "random"}]
    out_dir = tmp_path / "preannotated"
    outcomes = run_batch(client_for(Router(route)), sample, FIXTURES, out_dir)
    assert [o.status for o in outcomes] == ["failed", "done"]
    assert (out_dir / "fixture__12-sd-negative-claim.failed.json").exists()


def test_cited_file_names_are_matched_case_insensitively(tmp_path):
    skill = tmp_path / "skill"
    (skill / "scripts").mkdir(parents=True)
    for src in (SKILLS / "11-sd-destination").rglob("*"):
        if src.is_file():
            rel = src.relative_to(SKILLS / "11-sd-destination")
            (skill / ("SKILL.MD" if rel.name == "SKILL.md" else rel)).write_text(
                src.read_text())
    out_dir = tmp_path / "out"
    outcome = preannotate_skill(client_for(Router(route)), skill, ID_OK, out_dir)
    assert outcome.status == "done" and outcome.error is None
    ann = Annotation.model_validate_json((out_dir / "fixture__11-sd-destination.json").read_text())
    assert {s.file for i in ann.intents for s in i.sources} == {"SKILL.MD"}


def test_skill_text_respects_the_token_budget(tmp_path):
    (tmp_path / "SKILL.md").write_text("---\nname: big\ndescription: d\n---\n\n"
                                       + "\n".join("字" * 40 for _ in range(2000)) + "\n")
    (tmp_path / "extra.md").write_text("more\n")
    from skillalign.bench.preannotate import MAX_SKILL_TOKENS, est_tokens
    text, left_out = skill_text(load_skill(tmp_path))
    assert est_tokens(text) <= MAX_SKILL_TOKENS + 200
    assert left_out == ["SKILL.md (cut)", "extra.md"]

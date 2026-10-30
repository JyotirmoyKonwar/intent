import json

from skillalign.intents.extract import extract_claims, file_stem, save_claims
from skillalign.llm.templates import load_prompt, with_line_numbers
from skillalign.models import Intent

from .fake_llm import TRUNC, Router, claims_json, client_for

SKILL_MD = """---
name: demo
description: Reads CSV files and writes JSON.
---

# Demo

Never uploads data to any server.
Reads ./input.csv and writes ./output.json.
"""

GOOD_NEG = {"action": "uploads", "object": "data", "constraints": ["to any server"],
            "polarity": "negative", "quote": "Never uploads data to any server.",
            "line_start": 8, "line_end": 8}
BAD_QUOTE = {"action": "sends", "object": "files", "constraints": [],
             "polarity": "positive", "quote": "Sends files to a server.",
             "line_start": 8, "line_end": 8}
DESCRIPTION = {"action": "reads", "object": "CSV files", "constraints": [],
               "polarity": "positive", "quote": "Reads CSV files and writes JSON.",
               "line_start": 3, "line_end": 3}


def make_skill(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "SKILL.md").write_text(SKILL_MD)
    return tmp_path


# The section chunk is the one whose numbered line 8 is the negative claim. Matching the
# numbered line (not the bare sentence) keeps the prompt template's own examples out of it.
SECTION_MARK = "8| Never uploads"


def route_default(prompt: str):
    if SECTION_MARK in prompt:
        return claims_json(GOOD_NEG, BAD_QUOTE), "stop"
    return claims_json(DESCRIPTION), "stop"


def test_good_bad_and_fenced_replies(tmp_path):
    skill = make_skill(tmp_path)
    fenced_description = "```json\n" + claims_json(DESCRIPTION) + "\n```"
    router = Router(lambda p: (route_default(p)[0], "stop") if SECTION_MARK in p
                        else (fenced_description, "stop"))
    result = extract_claims(client_for(router), skill, skill_id="demo")

    assert result.stats.chunks >= 2
    assert result.stats.claims_raw == 3
    assert result.stats.dropped_bad_span == 1
    assert result.stats.claims_kept == 2
    assert [i.id for i in result.intents] == ["I1", "I2"]
    neg = result.intents[1]
    assert neg.polarity == "negative"
    assert neg.sources[0].file == "SKILL.md"
    assert neg.sources[0].quote == GOOD_NEG["quote"]


def test_description_chunk_is_not_told_about_itself(tmp_path):
    router = Router(route_default)
    extract_claims(client_for(router), make_skill(tmp_path), skill_id="demo")
    description_prompts = [r["messages"][-1]["content"] for r in router.requests
                           if SECTION_MARK not in r["messages"][-1]["content"]]
    section_prompts = [r["messages"][-1]["content"] for r in router.requests
                       if SECTION_MARK in r["messages"][-1]["content"]]
    assert "Already known from the skill's description (do not list these again): (none)" in \
        description_prompts[0]
    assert "Reads CSV files and writes JSON." in section_prompts[0]


def test_truncated_reply_is_retried_and_not_cached(tmp_path):
    script = [("{\"claims\": [", TRUNC)]

    def route(prompt: str):
        if SECTION_MARK in prompt and script:
            return script.pop()
        return route_default(prompt)

    router = Router(route)
    result = extract_claims(client_for(router), make_skill(tmp_path), skill_id="demo")
    assert result.stats.failed_chunks == 0
    assert result.stats.claims_kept == 2
    assert router.count(SECTION_MARK) == 2  # the cut-off reply, then one retry that is valid


def test_chunk_with_no_valid_reply_is_counted(tmp_path):
    router = Router(lambda p: ("no json here", "stop") if SECTION_MARK in p
                    else route_default(p))
    result = extract_claims(client_for(router), make_skill(tmp_path), skill_id="demo")
    assert result.stats.failed_chunks == 1
    assert result.stats.claims_kept == 1


def test_save_claims_writes_list_and_stats(tmp_path):
    skill = make_skill(tmp_path / "skill")
    result = extract_claims(client_for(Router(route_default)), skill,
                            skill_id="github/owner/repo/@abc")
    out = save_claims(result, tmp_path / "claims")
    assert out.name == "github__owner__repo__@abc.json"
    assert file_stem("github/owner/repo/@abc") == "github__owner__repo__@abc"
    data = json.loads(out.read_text())
    assert [Intent.model_validate(d).id for d in data] == ["I1", "I2"]
    stats = json.loads((tmp_path / "claims" / "github__owner__repo__@abc.stats.json").read_text())
    assert stats["claims_raw"] == 3 and stats["dropped_bad_span"] == 1
    assert stats["claims_kept"] == 2 and stats["failed_chunks"] == 0


def test_prompt_templates_load_and_have_hashes():
    b1 = load_prompt("claims_b1")
    pre = load_prompt("preannotate")
    assert len(b1.sha256) == 64 and b1.sha256 != pre.sha256
    rendered = b1.render(known="k", file="f", section="s")
    assert "$known" not in rendered and "$section" not in rendered
    assert "Section from f" in rendered


def test_with_line_numbers():
    assert with_line_numbers("a\nb", start=9) == "9| a\n10| b"
    assert with_line_numbers("x") == "1| x"
    assert with_line_numbers("") == ""


def test_quote_at_the_wrong_line_is_relocated_but_a_false_quote_is_still_dropped(tmp_path):
    wrong_line = {**GOOD_NEG, "line_start": 3, "line_end": 3}  # the quote is on line 8
    router = Router(lambda prompt: (claims_json(wrong_line, BAD_QUOTE), "stop")
                    if SECTION_MARK in prompt else (claims_json(), "stop"))
    result = extract_claims(client_for(router), make_skill(tmp_path), skill_id="demo")
    assert result.stats.relocated == 1 and result.stats.dropped_bad_span == 1
    [intent] = result.intents
    assert (intent.sources[0].line_start, intent.sources[0].line_end) == (8, 8)
    assert intent.sources[0].quote == "Never uploads data to any server."

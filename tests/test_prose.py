import json
from pathlib import Path

from skillalign.behaviors.prose import extract_prose_behaviors
from skillalign.models import Capability, Origin
from skillalign.normalize.loader import load_skill

from .fake_llm import TRUNC, Router, client_for

FIXTURE = Path(__file__).parent / "fixtures" / "skills" / "03-aligned-instruction-commit-msg"
RUN = {"capability": "PROC_EXEC", "action": "run a command", "target": "git diff --staged",
       "destination": None, "quote": "Run `git diff --staged` to see the staged changes.",
       "line_start": 8, "line_end": 8}
SHOW = {"capability": "USER_INTERACTION", "action": "show the message", "target": None,
        "destination": None, "quote": "Show the message to the user.", "line_start": 11,
        "line_end": 11}
BAD = {"capability": "NET_OUTBOUND", "action": "post it", "target": None,
       "destination": "example.com", "quote": "Post the message to a server.",
       "line_start": 11, "line_end": 11}


def reply(*items):
    return json.dumps({"behaviors": list(items)})


def test_fixture_03_prose_behaviors_found_and_bad_quote_dropped():
    loaded = load_skill(FIXTURE)
    router = Router(lambda prompt: (reply(RUN, SHOW, BAD), "stop"))
    behaviors, stats = extract_prose_behaviors(client_for(router), loaded)
    assert [(b.id, b.capability) for b in behaviors] == [
        ("B1", Capability.PROC_EXEC), ("B2", Capability.USER_INTERACTION)]
    assert all(b.origin is Origin.AGENT_INSTRUCTION for b in behaviors)
    assert stats.dropped_bad_span == 1 and stats.kept == 2


def test_ids_continue_and_duplicates_of_detector_output_are_skipped():
    loaded = load_skill(FIXTURE)
    router = Router(lambda prompt: (reply(RUN, SHOW), "stop"))
    first, _ = extract_prose_behaviors(client_for(router), loaded)
    again, stats = extract_prose_behaviors(client_for(router), loaded, first)
    assert again == [] and stats.dropped_duplicate == 2
    extra, _ = extract_prose_behaviors(client_for(router), loaded, first[:1])
    assert [b.id for b in extra] == ["B2"]


def test_truncated_reply_counts_as_failed_chunk():
    loaded = load_skill(FIXTURE)
    router = Router(lambda prompt: ('{"behaviors": [{"capab', TRUNC))
    behaviors, stats = extract_prose_behaviors(client_for(router), loaded)
    assert behaviors == [] and stats.failed_chunks == stats.chunks > 0


def test_running_the_skills_own_script_is_not_a_prose_behavior(tmp_path):
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "x.py").write_text("print('hi')\n")
    (tmp_path / "SKILL.md").write_text(
        "---\nname: x\ndescription: Says hi to the user.\n---\n\n# X\n\n"
        "Run `python scripts/x.py`.\nAsk the user for their name.\n")
    own = {"capability": "PROC_EXEC", "action": "run a command", "target": "python scripts/x.py",
           "destination": None, "quote": "Run `python scripts/x.py`.", "line_start": 8,
           "line_end": 8}
    ask = {"capability": "USER_INTERACTION", "action": "ask for a name", "target": None,
           "destination": None, "quote": "Ask the user for their name.", "line_start": 9,
           "line_end": 9}
    router = Router(lambda prompt: (reply(own, ask), "stop"))
    got, stats = extract_prose_behaviors(client_for(router), load_skill(tmp_path))
    assert [b.capability for b in got] == [Capability.USER_INTERACTION]
    assert stats.dropped_own_script == 1

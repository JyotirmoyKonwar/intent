from skillalign.intents.vocab import (
    ACTION_VOCAB,
    canonical_verb,
    classify_category,
    enrich_intents,
    expected_capabilities,
)
from skillalign.models import Capability as C
from skillalign.models import Evidence, Intent, Polarity


def intent(action, obj="x", polarity=Polarity.POSITIVE):
    return Intent(id="I1", action=action, object=obj, polarity=polarity,
                  sources=[Evidence(file="SKILL.md", line_start=1, line_end=1, quote="q")])


def test_vocabulary_size_and_inflections():
    assert 40 <= len(ACTION_VOCAB) <= 80
    assert canonical_verb("downloads") == "download"
    assert canonical_verb("Converting") == "convert"
    assert canonical_verb("saved") == "save"
    assert canonical_verb("frobnicate") is None


def test_expected_capabilities():
    assert expected_capabilities("downloads", "a file") == [C.NET_OUTBOUND]
    assert expected_capabilities("saves", "the report to a remote server") == [C.NET_OUTBOUND]
    assert expected_capabilities("saves", "the report") == [C.FS_WRITE]
    assert expected_capabilities("frobnicate") == []


def test_enrich_keeps_existing_and_fills_negative_claims():
    keep = intent("reads").model_copy(update={"expected_capabilities": [C.ENV_READ]})
    [a, b, c] = enrich_intents([keep, intent("uploads", "data"),
                                intent("leaves", "the machine", Polarity.NEGATIVE)])
    assert a.expected_capabilities == [C.ENV_READ]
    assert b.expected_capabilities == [C.NET_OUTBOUND]
    assert c.expected_capabilities == [C.NET_OUTBOUND]


def test_category_classifier():
    assert classify_category("pdf-tools", "Convert PDF files to text") == "file-conversion"
    assert classify_category("x", "Scrape web pages with playwright") == "browser-automation"
    assert classify_category("commit-msg", "Write a git commit message") == "git-devops"
    assert classify_category("n", "Be kind", has_code=False) == "instruction-only"
    assert classify_category("n", "Be kind", has_code=True) == "other"


def test_file_verbs_need_a_file_like_object():
    assert expected_capabilities("removes", "trailing whitespace") == [C.DATA_TRANSFORM]
    assert expected_capabilities("removes", "build artifacts") == [C.FS_DELETE]
    assert expected_capabilities("creates", "SkeletonHelper") == [C.DATA_TRANSFORM]
    assert expected_capabilities("creates", "the report file") == [C.FS_WRITE]
    assert expected_capabilities("lists", "files under src/") == [C.FS_ENUMERATE]

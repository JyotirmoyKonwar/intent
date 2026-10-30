import json

from typer.testing import CliRunner

from skillalign.cli import app

from .conftest import GOLD, GOLD_FILES, SKILLS

runner = CliRunner()


def test_schema_export(tmp_path):
    result = runner.invoke(app, ["schema", "--out", str(tmp_path)])
    assert result.exit_code == 0
    schema = json.loads((tmp_path / "annotation.schema.json").read_text())
    assert schema["title"] == "Annotation"


def test_help_lists_bench_and_extract_commands():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "extract" in result.output and "bench" in result.output
    bench = runner.invoke(app, ["bench", "--help"])
    for command in ("sample", "preannotate", "render"):
        assert command in bench.output


def test_validate_gold():
    result = runner.invoke(app, ["validate", *map(str, GOLD_FILES), "--root", str(SKILLS)])
    assert result.exit_code == 0, result.output


def test_validate_reports_bad_span(tmp_path):
    data = json.loads((GOLD / "13-sd-scope.json").read_text())
    data["behaviors"][0]["evidence"][0]["quote"] = "not in the file"
    bad = tmp_path / "13-sd-scope.json"
    bad.write_text(json.dumps(data))
    result = runner.invoke(app, ["validate", str(bad), "--root", str(SKILLS)])
    assert result.exit_code == 1
    assert "quote not found" in result.output


def test_help_lists_new_commands():
    out = runner.invoke(app, ["--help"]).output
    for command in ("detect", "baseline", "eval"):
        assert command in out
    assert "inject" in runner.invoke(app, ["bench", "--help"]).output


def test_baseline_and_eval_round_trip(tmp_path):
    skill = SKILLS / "07-ub-exfil-crossfile"
    pred = tmp_path / "pred"
    pred.mkdir()
    r = runner.invoke(app, ["baseline", "r2", str(skill), "--out", str(pred / "p.json")])
    assert r.exit_code == 0 and "misaligned" in r.output
    # score the baseline under the gold skill id
    data = json.loads((pred / "p.json").read_text())
    data["skill_id"] = "fixture/07-ub-exfil-crossfile"
    (pred / "p.json").write_text(json.dumps(data))
    r = runner.invoke(app, ["eval", str(GOLD), str(pred), "--boot", "20"])
    assert r.exit_code == 0, r.output
    assert "skill level" in r.output and "macro F1" in r.output and "UB" in r.output
    j = runner.invoke(app, ["eval", str(GOLD), str(pred), "--json"])
    assert json.loads(j.output)["skills"] == 17


def test_bench_inject_cli(tmp_path):
    r = runner.invoke(app, ["bench", "inject", str(SKILLS), str(GOLD), str(tmp_path),
                            "--templates", "test", "--operator", "VAGUE_DOC"])
    assert r.exit_code == 0, r.output
    assert "injected" in r.output and (tmp_path / "manifest.json").exists()
    bad = runner.invoke(app, ["bench", "inject", str(SKILLS), str(GOLD), str(tmp_path / "x"),
                              "--operator", "NOPE"])
    assert bad.exit_code != 0

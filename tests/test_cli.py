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

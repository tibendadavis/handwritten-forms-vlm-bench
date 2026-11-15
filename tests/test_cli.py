import json
import shutil
from pathlib import Path

import langfuse as langfuse_package
import pytest
from rich.console import Console

from formbench import cli, model
from tests.conftest import FIXTURES, FakeLangfuse

ROOT = Path(__file__).parent.parent


@pytest.fixture(autouse=True)
def wide_console(monkeypatch):
    """Long messages on one line, as in a wide terminal."""
    monkeypatch.setattr(cli, "console", Console(width=300))


@pytest.fixture
def samples(tmp_path):
    shutil.copytree(FIXTURES / "cv-form", tmp_path / "cv-form")
    return tmp_path / "cv-form" / "samples"


def test_check_on_a_good_folder(capsys):
    assert cli.main(["check", "--folder", str(FIXTURES / "cv-form" / "samples")]) == 0
    out = capsys.readouterr().out
    assert "Weekly tally sheet" in out and "3 photo(s) to read" in out
    assert "skipped: sheet-02_A.jpg (no sheet-02.json)" in out


def test_check_on_a_register_folder(capsys):
    folder = FIXTURES / "register" / "pages"
    assert (
        cli.main(
            ["check", "--folder", str(folder), "--form", str(FIXTURES / "register" / "form.json")]
        )
        == 0
    )
    assert "2 photo(s) to read" in capsys.readouterr().out


def test_check_on_a_broken_folder(samples, capsys):
    (samples / "sheet-01.json").write_text('{"Region": 0001}')
    (samples / "sheet-03.json").write_text('{"Colour": "red"}')
    assert cli.main(["check", "--folder", str(samples)]) == 1
    out = capsys.readouterr().out
    assert "sheet-01.json: not valid JSON at line 1" in out
    assert "sheet-03.json: unknown field 'Colour'" in out
    assert "2 truth file(s) cannot be used." in out


def test_check_with_a_broken_form(samples, capsys):
    form = json.loads((samples.parent / "form.json").read_text())
    form["sections"]["A"]["fields"][1].pop("options")
    (samples.parent / "form.json").write_text(json.dumps(form))
    assert cli.main(["check", "--folder", str(samples)]) == 1
    assert 'sections.A.fields[1] (Region): a choice needs its "options"' in capsys.readouterr().out


def test_check_pages_with_a_register_fails(capsys):
    folder = FIXTURES / "register" / "pages"
    code = cli.main(
        [
            "check",
            "--folder",
            str(folder),
            "--form",
            str(FIXTURES / "register" / "form.json"),
            "--pages",
        ]
    )
    assert code == 1 and "--pages is for cv-forms" in capsys.readouterr().out


def test_run_prints_everything(samples, monkeypatch, capsys):
    fake = FakeLangfuse()
    monkeypatch.setattr(langfuse_package, "get_client", lambda: fake)
    replies = {
        "sheet-01_A.jpg": '{"region": "North"}',
        "sheet-01_B.jpg": '{"screened_6_11_m_male": null, "screened_6_11_m_female": null}',
        "sheet-01_C.jpg": '{"supervisor_signature": "signed"}',
    }
    monkeypatch.setattr(model, "complete_json", lambda image, *a, **kw: replies[image.name])
    out_dir = samples.parent / "results"
    args = ["run", "--folder", str(samples), "--model", "small:1b", "--out", str(out_dir)]
    assert cli.main(args) == 0
    out = capsys.readouterr().out
    assert "Prompt: tally-weekly-tally v1" in out
    assert "empty reading: sheet-01_B.jpg (B)" in out
    assert "2 photo(s) skipped" in out
    assert "Langfuse: https://langfuse.test/runs/run-1" in out
    assert "Saved to" in out and len(list(out_dir.glob("*.json"))) == 1
    assert "All runs on this folder side by side: formbench compare --folder samples" in out


def test_run_without_langfuse_stops(samples, monkeypatch, capsys):
    monkeypatch.setattr(langfuse_package, "get_client", lambda: FakeLangfuse(down=True))
    assert cli.main(["run", "--folder", str(samples), "--model", "m"]) == 1
    assert "Langfuse is required for runs" in capsys.readouterr().out


def saved_run(path, model_tag="m", version=1, counts=None, latency=10.0, photos=3):
    run = {
        "settings": {
            "folder": "samples",
            "form": "weekly-tally",
            "kind": "cv-form",
            "mode": "sections",
            "model": model_tag,
            "think": False,
            "max_side": 4032,
            "prompt_name": "tally-weekly-tally",
            "prompt_version": version,
        },
        "counts": counts
        or {
            "pages": 3,
            "json_valid": 3.0,
            "field_cells": 10,
            "field_correct": 5,
            "field_tp": 5,
            "field_fp": 5,
            "field_fn": 5,
        },
        "latency_s": latency,
        "photo_count": photos,
        "metrics": {},
        "outcomes": [],
    }
    path.write_text(json.dumps(run))
    return path


def test_compare_pools_runs_with_the_same_label(tmp_path, capsys):
    a = saved_run(tmp_path / "a.json")
    b = saved_run(
        tmp_path / "b.json",
        counts={
            "pages": 1,
            "json_valid": 1.0,
            "field_cells": 90,
            "field_correct": 90,
            "field_tp": 90,
            "field_fp": 0,
            "field_fn": 0,
        },
        latency=2.0,
    )
    c = saved_run(tmp_path / "c.json", version=2)
    (tmp_path / "other.json").write_text("[]")
    assert cli.main(["compare", str(a), str(b), str(c), str(tmp_path / "other.json")]) == 0
    out = capsys.readouterr().out
    assert "samples: 3 photo(s), 3 run(s)" in out
    assert "m prompt v1" in out and "m prompt v2" in out
    assert "95.0%" in out  # (5 + 90) / (10 + 90) pooled, not the 75% average of 50% and 100%
    assert "3.0" in out  # (10 + 2) seconds / 4 photos
    assert "not a formbench run, left out" in out


def test_compare_filters_by_folder(tmp_path, capsys):
    a = saved_run(tmp_path / "a.json")
    assert cli.main(["compare", str(a), "--folder", "elsewhere"]) == 1
    assert "No saved runs" in capsys.readouterr().out


def test_breakdown_and_rescore(samples, monkeypatch, capsys):
    fake = FakeLangfuse()
    monkeypatch.setattr(langfuse_package, "get_client", lambda: fake)
    replies = {
        # facility code: digits keep their leading zeros, so 42 is a misread of 0042
        "sheet-01_A.jpg": '{"week_starting": "2/3/26", "region": "South", "facility_code": "42"}',
        "sheet-01_B.jpg": '{"screened_6_11_m_male": "12", "screened_6_11_m_female": "3"}',
        "sheet-01_C.jpg": '{"supervisor_signature": "signed"}',
    }
    monkeypatch.setattr(model, "complete_json", lambda image, *a, **kw: replies[image.name])
    out_dir = samples.parent / "results"
    assert cli.main(["run", "--folder", str(samples), "--model", "m", "--out", str(out_dir)]) == 0
    (path,) = out_dir.glob("*.json")
    capsys.readouterr()

    assert cli.main(["breakdown", str(path)]) == 0
    out = capsys.readouterr().out
    lines = {line.split("│")[1].strip(): line for line in out.splitlines() if "│" in line}
    assert [c.strip() for c in lines["A"].split("│")[2:8]] == ["4", "1", "2", "1", "0", "25.0%"]
    assert [c.strip() for c in lines["All"].split("│")[2:8]] == ["7", "3", "2", "1", "1", "42.9%"]

    assert cli.main(["breakdown", str(path), "--by", "type"]) == 0
    out = capsys.readouterr().out
    assert "digits" in out and "choice" in out and "mark" in out

    saved = json.loads(path.read_text())
    original = saved["metrics"]
    saved["metrics"], saved["counts"], saved["outcomes"] = {}, {}, []
    path.write_text(json.dumps(saved))
    assert cli.main(["rescore", str(path), "--folder", str(samples)]) == 0
    again = json.loads(path.read_text())
    assert again["metrics"] == original and len(again["outcomes"]) == 7
    assert "Re-scored: samples" in capsys.readouterr().out


def test_scores_lists_every_score(capsys):
    assert cli.main(["scores"]) == 0
    out = capsys.readouterr().out
    for name in (
        "overall_accuracy",
        "completeness",
        "reliability",
        "rows_detected",
        "processing_success_rate",
        "processing_time",
    ):
        assert name in out

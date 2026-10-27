import datetime as dt
import json
from pathlib import Path

import pytest
from test_register_aggregate import page, row

from formbench.register_payload import ApprovalRefused, approve, data_value_set

GO_LIVE = dt.date(2025, 6, 2)
COVER = {"facility_name": "Amani HC", "mfl_code": "MFL-1", "facility_type": "Fixed/Facility"}


def make_run(folder: Path, week: str, rows, org_unit="ouAmani"):
    folder.mkdir(parents=True)
    readings = {
        "settings": {"week": week},
        "cover": {"cover": COVER},
        "pages": [{**page(*rows), "page_id": "2025-04-s1"}],
        "org_unit": {"uid": org_unit, "warning": None},
    }
    (folder / "readings.json").write_text(json.dumps(readings))
    return folder


def do_approve(run, ds, state, today=dt.date(2025, 6, 20), confirm=True):
    return approve(run, "Alice", ds=ds, go_live=GO_LIVE, today=today,
                   confirm=lambda draft: confirm, state_root=state)  # fmt: skip


def test_data_value_set_shape():
    values = [{"day": "2025-06-12", "data_element": "deOPV", "coc": "c1", "value": 3},
              {"day": "2025-06-12", "data_element": "deOPV", "coc": "c2", "value": 0}]  # fmt: skip
    payload = data_value_set(values, data_set="ds", org_unit=None, attribute_option="aoc0")
    assert payload["dataValues"] == [
        {"dataElement": "deOPV", "period": "20250612", "orgUnit": None,
         "categoryOptionCombo": "c1", "attributeOptionCombo": "aoc0", "value": "3"}
    ]  # fmt: skip


def test_refuses_while_flags_are_open_and_writes_nothing(ri_metadata, tmp_path):
    run = make_run(tmp_path / "run", "2025W24", [row("1", birth=None, opv1="12-06-25")])
    with pytest.raises(ApprovalRefused, match="birth_date_missing"):
        do_approve(run, ri_metadata, tmp_path / "state")
    assert not (run / "datavalueset.json").exists()


def test_refuses_without_confirmation_even_with_no_flags(ri_metadata, tmp_path):
    run = make_run(tmp_path / "run", "2025W24", [row("1", opv1="12-06-25")])
    with pytest.raises(ApprovalRefused, match="Not confirmed"):
        do_approve(run, ri_metadata, tmp_path / "state", confirm=False)
    assert not (run / "datavalueset.json").exists()


def test_approval_writes_payload_and_remembers_the_week(ri_metadata, tmp_path):
    run = make_run(tmp_path / "run", "2025W24", [row("1", opv1="12-06-25")])
    record = do_approve(run, ri_metadata, tmp_path / "state")
    assert record["pushed_to_dhis2"] is False and record["values"] == 2  # OPV dose + child
    payload = json.loads((run / "datavalueset.json").read_text())
    assert payload["orgUnit"] == "ouAmani" and payload["attributeOptionCombo"] == "aoc0"
    assert {v["period"] for v in payload["dataValues"]} == {"20250612"}
    with pytest.raises(ApprovalRefused, match="already approved"):
        do_approve(run, ri_metadata, tmp_path / "state")
    weeks = json.loads((tmp_path / "state" / "mfl_1_fixed_facility" / "weeks.json").read_text())
    assert weeks["2025W24"]["reviewer"] == "Alice"


def test_late_entry_next_week_produces_a_correction_payload(ri_metadata, tmp_path):
    state = tmp_path / "state"
    do_approve(
        make_run(tmp_path / "w24", "2025W24", [row("1", opv1="12-06-25")]), ri_metadata, state
    )
    rows = [row("1", opv1="12-06-25"), row("2", opv1="12-06-25"), row("3", opv1="17-06-25")]
    run = make_run(tmp_path / "w25", "2025W25", rows)
    do_approve(run, ri_metadata, state, today=dt.date(2025, 6, 27))
    fixes = json.loads((run / "corrections.json").read_text())["dataValues"]
    opv = [v for v in fixes if v["dataElement"] == "deOPV"]
    assert [(v["period"], v["value"]) for v in opv] == [("20250612", "2")]  # full new total
    week = json.loads((run / "datavalueset.json").read_text())["dataValues"]
    assert {v["period"] for v in week} == {"20250617"}


def test_reviewer_fix_is_kept_for_later_rereads(ri_metadata, tmp_path):
    state = tmp_path / "state"
    run = make_run(tmp_path / "w24", "2025W24", [row("1", birth=None, opv1="12-06-25")])
    decisions = 'pages: {"2025-04-s1": {rows: {"1": {birth_date: "10-04-25"}}}}\n'
    (run / "decisions.yaml").write_text(decisions)
    do_approve(run, ri_metadata, state)
    # Next week the birth date still reads blank on the photo: no new flag, the fix is kept.
    run = make_run(tmp_path / "w25", "2025W25", [row("1", birth=None, opv1="12-06-25", opv2="17-06-25")])  # fmt: skip
    record = do_approve(run, ri_metadata, state, today=dt.date(2025, 6, 27))
    assert record["values"] == 2


def test_no_http_writes_anywhere_in_the_code():
    src = Path(__file__).parents[1] / "src" / "formbench"
    code = "\n".join(p.read_text() for p in src.glob("*.py"))
    for verb in (".post(", ".put(", ".patch(", ".delete("):
        assert verb not in code

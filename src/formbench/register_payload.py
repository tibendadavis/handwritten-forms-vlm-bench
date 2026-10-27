"""Approval of a reviewed register run and the RI v2 dataValueSet it produces.

The payload is only written here, after a named person approves, and it is never sent:
importing it into DHIS2 is a separate, deliberate step outside this project.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any, Callable

from dotenv import find_dotenv, load_dotenv
from rich.console import Console
from rich.table import Table

from formbench.register_run import (
    METADATA,
    compute_draft,
    go_live_from_env,
    load_decisions,
    names,
)
from formbench.register_spec import load_spec
from formbench.register_state import STATE_DIR, RegisterState


class ApprovalRefused(ValueError):
    """The run cannot be approved yet; the message says why."""


def data_value_set(
    values: list[dict[str, Any]], *, data_set: str, org_unit: str | None, attribute_option: str
) -> dict[str, Any]:
    """DHIS2 dataValueSet JSON: daily periods (yyyyMMdd), values as strings, no blanks."""
    return {
        "dataSet": data_set,
        "orgUnit": org_unit,
        "attributeOptionCombo": attribute_option,
        "dataValues": [
            {
                "dataElement": v["data_element"],
                "period": v["day"].replace("-", ""),
                "orgUnit": org_unit,
                "categoryOptionCombo": v["coc"],
                "attributeOptionCombo": attribute_option,
                "value": str(v["value"]),
            }
            for v in values
            if v["value"]
        ],
    }


def approve(
    run_dir: Path,
    reviewer: str,
    *,
    ds: dict[str, Any],
    go_live: dt.date,
    today: dt.date,
    confirm: Callable[[dict[str, Any]], bool],
    state_root: Path = STATE_DIR,
) -> dict[str, Any]:
    """Recompute the draft with the reviewer's decisions; refuse while any blocking flag is
    open; ask for confirmation; then write the payload(s) and record the approval."""
    if not reviewer.strip():
        raise ApprovalRefused("A reviewer name is required.")
    if (run_dir / "approval.json").exists():
        raise ApprovalRefused(f"{run_dir} is already approved.")
    readings = json.loads((run_dir / "readings.json").read_text())
    spec = load_spec(ds)
    draft = compute_draft(
        readings["pages"], readings["cover"]["cover"], spec, names(ds),
        week=readings["settings"]["week"], go_live=go_live, today=today,
        state_root=state_root, decisions=load_decisions(run_dir),
    )  # fmt: skip
    if draft["unresolved_blocking"]:
        listed = "\n  ".join(draft["unresolved_blocking"])
        raise ApprovalRefused(f"Resolve these flags in decisions.yaml first:\n  {listed}")
    if not confirm(draft):
        raise ApprovalRefused("Not confirmed; nothing was written.")

    org_unit = (readings.get("org_unit") or {}).get("uid")
    aoc = draft["attribute_option"]["uid"]
    payload = data_value_set(draft["values"], data_set=spec.dataset_id, org_unit=org_unit, attribute_option=aoc)  # fmt: skip
    (run_dir / "datavalueset.json").write_text(json.dumps(payload, indent=2) + "\n")
    if draft["corrections"]:
        fixes = data_value_set(draft["corrections"], data_set=spec.dataset_id, org_unit=org_unit, attribute_option=aoc)  # fmt: skip
        (run_dir / "corrections.json").write_text(json.dumps(fixes, indent=2) + "\n")
    when = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    RegisterState(draft["book"], root=state_root).approve(
        draft["week"], draft["pages"], reviewer, str(run_dir), when
    )
    record = {
        "reviewer": reviewer,
        "approved_at": when,
        "week": draft["week"],
        "values": len(payload["dataValues"]),
        "corrections": len(draft["corrections"]),
        "decisions": load_decisions(run_dir),
        "pushed_to_dhis2": False,
    }
    (run_dir / "approval.json").write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
    (run_dir / "draft.json").write_text(json.dumps(draft, indent=2, ensure_ascii=False) + "\n")
    return record


def _confirm_in_terminal(draft: dict[str, Any]) -> bool:
    console = Console()
    for title, rows in (("Daily values", draft["values"]), ("Corrections", draft["corrections"])):
        if not rows:
            continue
        table = Table(title=title)
        for column in ("Day", "Data element", "Disaggregation", "Value"):
            table.add_column(column, justify="right" if column == "Value" else "left")
        for v in rows:
            table.add_row(v["day"], v["data_element_name"], v["coc_name"], str(v["value"]))
        console.print(table)
    notes = [f for f in draft["flags"] if not f["blocking"] or f["resolved"]]
    console.print(f"{len(notes)} note(s)/resolved flag(s) are listed in review.html.")
    return input("Type APPROVE to write the DHIS2 payload (it will not be sent): ") == "APPROVE"


def main() -> None:
    parser = argparse.ArgumentParser(description="Approve a reviewed register run.")
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--reviewer", required=True)
    parser.add_argument("--metadata", type=Path, default=METADATA)
    args = parser.parse_args()

    load_dotenv(find_dotenv(usecwd=True))
    try:
        record = approve(
            args.run, args.reviewer, ds=json.loads(args.metadata.read_text()),
            go_live=go_live_from_env(), today=dt.date.today(), confirm=_confirm_in_terminal,
        )  # fmt: skip
    except ApprovalRefused as error:
        raise SystemExit(f"register-approve: {error}") from None
    print(
        f"Approved by {record['reviewer']}: {record['values']} values written to "
        f"{args.run / 'datavalueset.json'}"
        + (f" and {record['corrections']} corrections" if record["corrections"] else "")
        + ". Nothing was sent to DHIS2."
    )


if __name__ == "__main__":
    main()

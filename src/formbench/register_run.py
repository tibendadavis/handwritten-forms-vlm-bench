"""One weekly run: read photos, build a draft with flags, write files for human review.

Nothing here is final: the draft only becomes a DHIS2 payload through register-approve.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import time
from pathlib import Path
from typing import Any

import yaml
from dotenv import find_dotenv, load_dotenv

from formbench.register_aggregate import (
    Flag,
    aggregate,
    apply_decisions,
    classify_days,
    iso_week_days,
    resolve_attribute_option,
    select_days,
)
from formbench.register_spec import RegisterSpec, load_spec
from formbench.register_state import (
    RegisterState,
    book_id,
    corrections,
    discrepancies,
    page_identity,
)

METADATA = Path("schemas/ri_v2/metadata.json")
RUNS_DIR = Path("results/register")


def go_live_from_env() -> dt.date:
    value = os.environ.get("REGISTER_GO_LIVE", "").strip()
    if not value:
        raise SystemExit("Set REGISTER_GO_LIVE=YYYY-MM-DD in .env (pilot go-live date).")
    return dt.date.fromisoformat(value)


def names(ds: dict[str, Any]) -> dict[str, str]:
    """UID -> readable name for data elements and category option combos (for review)."""
    out = {}
    for dse in ds["dataSetElements"]:
        de = dse["dataElement"]
        out[de["id"]] = de.get("formName") or de["name"]
        combo = dse.get("categoryCombo") or de["categoryCombo"]
        out.update({c["id"]: c["name"] for c in combo["categoryOptionCombos"]})
    return out


def _value_rows(totals, provenance, labels, previous=None):
    rows = []
    for (day, de, coc), value in totals.items():
        row = {
            "day": day.isoformat(),
            "data_element": de,
            "data_element_name": labels.get(de, de),
            "coc": coc,
            "coc_name": labels.get(coc, coc),
            "value": value,
            "sources": [list(s) for s in provenance.get((day, de, coc), [])],
        }
        if previous is not None:
            row["previous"] = previous.get((day, de, coc))
        rows.append(row)
    return rows


def compute_draft(
    pages: list[dict[str, Any]],
    cover: dict[str, Any],
    spec: RegisterSpec,
    labels: dict[str, str],
    *,
    week: str,
    go_live: dt.date,
    today: dt.date,
    state_root: Path,
    decisions: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Everything the reviewer needs, recomputed from readings + decisions every time."""
    decisions = decisions or {}
    resolved = frozenset(decisions.get("resolved") or {})
    flags: list[Flag] = []

    aoc, aoc_flag = resolve_attribute_option(
        cover.get("facility_type"), spec, decisions.get("attribute_option")
    )
    if aoc_flag:
        flags.append(aoc_flag)
    aoc_name = next((n for n, uid in spec.attribute_options.items() if uid == aoc), None)
    book = book_id(cover.get("mfl_code"), aoc_name or "unresolved")
    state = RegisterState(book, root=state_root)

    fixed = apply_decisions(pages, decisions)
    agg = aggregate(fixed, spec, go_live=go_live, today=today, resolved=resolved)
    flags += agg.flags
    approved = state.approved_pages() if aoc else {}
    for page in fixed:
        if page["page_id"] in approved:
            flags += discrepancies(page, approved[page["page_id"]])

    approved_weeks = set(state.approved_weeks()) if aoc else set()
    days = classify_days(agg, week, approved_weeks)
    for day in sorted(days["unreported"]):
        flags.append(Flag("earlier_week_unreported", f"Doses dated {day} belong to a week that was never approved; they are not counted here", False))  # fmt: skip
    values = select_days(agg, iso_week_days(week))
    fixes, after = ({}, None)
    if aoc:
        fixes, after = corrections(fixed, state, spec, go_live=go_live, today=today, resolved=resolved)  # fmt: skip
    previous = {}
    if fixes:
        stored = [{"page_id": k, **v} for k, v in state.approved_pages().items()]
        previous = aggregate(stored, spec, go_live=go_live, today=today).totals

    unique = {f.id: f for f in flags}
    flag_rows = [
        {**f.__dict__, "id": f.id, "resolved": f.id in resolved}
        for f in sorted(unique.values(), key=lambda f: (not f.blocking, f.id))
    ]
    return {
        "week": week,
        "days": [d.isoformat() for d in iso_week_days(week)],
        "go_live": go_live.isoformat(),
        "book": book,
        "data_set": spec.dataset_id,
        "attribute_option": {"name": aoc_name, "uid": aoc},
        "cover": cover,
        "values": _value_rows(values, agg.provenance, labels),
        "corrections": _value_rows(fixes, after.provenance if after else {}, labels, previous),
        "flags": flag_rows,
        "unresolved_blocking": [f["id"] for f in flag_rows if f["blocking"] and not f["resolved"]],
    }


def resolve_org_unit(mfl_code: str | None) -> tuple[str | None, str | None]:
    """Org unit UID whose code equals the MFL code (read-only lookup); else a warning."""
    if not mfl_code:
        return None, "MFL Code is blank on the cover; orgUnit will be null"
    from formbench.dhis2 import Dhis2Client

    try:
        client = Dhis2Client.from_env()
        units = client.find_org_units_by_code(mfl_code.strip())
    except (SystemExit, Exception) as error:  # a failed lookup must never stop a run
        return None, f"Org unit lookup failed ({error}); orgUnit will be null"
    if len(units) != 1:
        return None, f"{len(units)} org units have code {mfl_code!r}; orgUnit will be null"
    return units[0]["id"], None


def main() -> None:
    parser = argparse.ArgumentParser(description="Read register photos into a draft for review.")
    parser.add_argument("--week", required=True, help="ISO week, e.g. 2026W44")
    parser.add_argument("--cover", type=Path, required=True, help="photo of the register cover")
    parser.add_argument("--pages", type=Path, nargs="+", required=True, help="cohort page photos")
    parser.add_argument("--engine", required=True, help="ollama:<model-tag>")
    parser.add_argument("--mode", choices=["tiles", "full"], default="tiles")
    parser.add_argument("--think", action="store_true")
    parser.add_argument("--metadata", type=Path, default=METADATA)
    parser.add_argument("--out", type=Path, default=RUNS_DIR)
    args = parser.parse_args()

    kind, _, model = args.engine.partition(":")
    if kind != "ollama" or not model:
        raise SystemExit("register-run: --engine must be 'ollama:<model-tag>'")
    load_dotenv(find_dotenv(usecwd=True))
    go_live = go_live_from_env()
    iso_week_days(args.week)  # validate early
    ds = json.loads(args.metadata.read_text())
    spec = load_spec(ds)

    from langfuse import get_client

    from formbench.register_extract import read_cover, read_page, tiles
    from formbench.register_review import write_review
    from formbench.register_template import load_template

    template, layout = load_template()
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = args.out / f"{args.week}_{stamp}"
    (run_dir / "strips").mkdir(parents=True, exist_ok=True)
    langfuse = get_client()
    settings = {"week": args.week, "engine": args.engine, "mode": args.mode, "think": args.think}
    try:
        with langfuse.start_as_current_span(name="register-run", metadata=settings):
            langfuse.update_current_trace(name=f"register {args.week}", metadata=settings, tags=["register"])  # fmt: skip
            start = time.perf_counter()
            cover = read_cover(args.cover, model, think=args.think)
            pages, page_info = [], []
            for photo in args.pages:
                result = read_page(photo, model, mode=args.mode, template=template, layout=layout, think=args.think)  # fmt: skip
                pid = page_identity(result["reading"], photo.stem)
                pages.append({"page_id": pid, **result["reading"]})
                page_info.append({"page_id": pid, "photo": str(photo), "mode": result["mode"],
                                  "json_valid": result["json_valid"], "errors": result["errors"],
                                  "notes": result["notes"], "raw": result["raw"]})  # fmt: skip
                if result["aligned"] is not None:  # only the counting columns are saved
                    for i, (strip, _) in enumerate(tiles(result["aligned"], layout)):
                        strip.save(run_dir / "strips" / f"{pid}-{i}.jpg", quality=85)
            latency = round(time.perf_counter() - start, 2)
    finally:
        langfuse.flush()

    org_unit, warning = resolve_org_unit(cover["cover"].get("mfl_code"))
    readings = {"settings": settings, "latency_s": latency, "cover": cover, "pages": pages,
                "page_info": page_info, "org_unit": {"uid": org_unit, "warning": warning}}  # fmt: skip
    (run_dir / "readings.json").write_text(
        json.dumps(readings, indent=2, ensure_ascii=False) + "\n"
    )
    draft = compute_draft(
        pages, cover["cover"], spec, names(ds), week=args.week, go_live=go_live,
        today=dt.date.today(), state_root=Path("data/private/register/state"),
    )  # fmt: skip
    draft["org_unit"] = {"uid": org_unit, "warning": warning}
    (run_dir / "draft.json").write_text(json.dumps(draft, indent=2, ensure_ascii=False) + "\n")
    write_review(run_dir, draft, readings, list(spec.attribute_options))
    if warning:
        print(f"Warning: {warning}")
    print(
        f"{len(draft['values'])} values, {len(draft['corrections'])} corrections, "
        f"{len(draft['unresolved_blocking'])} flags to resolve.\n"
        f"Review {run_dir / 'review.html'}, fill {run_dir / 'decisions.yaml'}, then run:\n"
        f'  uv run register-approve --run {run_dir} --reviewer "Your Name"'
    )


def load_decisions(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "decisions.yaml"
    return (yaml.safe_load(path.read_text()) or {}) if path.exists() else {}


def strips_for(run_dir: Path, page_id: str) -> list[Path]:
    return sorted((run_dir / "strips").glob(f"{page_id}-*.jpg"))


if __name__ == "__main__":
    main()

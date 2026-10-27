"""Synthetic register scenarios over several weeks, with ground truth.

Children are born into cohort pages, follow the EPI schedule with realistic delays and
missed visits, and each dose is written on the day it is given (or copied from the card at
the first visit). Each week, the pages that got new entries are "photographed" showing
everything written so far. Known edge cases are injected so we can check they get flagged.
Everything here is made up; no real child, facility or person is represented.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import random
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from formbench.register_aggregate import iso_week, iso_week_days
from formbench.register_spec import DOSE_COLUMNS, load_spec
from formbench.register_state import RegisterState, page_identity

# Visits of the EPI schedule (a typical national schedule): days after birth and doses given together.
VISITS = [
    (0, ("bcg", "opv0")),
    (42, ("opv1", "penta1", "pcv1", "rota1")),
    (70, ("opv2", "penta2", "pcv2")),
    (98, ("opv3", "penta3", "pcv3", "ipv1")),
    (274, ("mcv1", "ipv2")),
    (548, ("mcv2",)),
]
CLINIC_DAYS = (0, 1, 2, 3)  # Monday to Thursday
MISSED_VISIT = 0.08

FIRST = ["Anna", "Grace", "Maria", "Sara", "Lina", "Ruth", "Nora", "Joy", "Ella",
         "Mary", "David", "John", "Peter", "James", "Daniel", "Samuel", "Paul", "Mark",
         "Simon", "Isaac", "Joseph", "Michael"]  # fmt: skip
MOTHERS = ["Esther", "Rose", "Alice", "Clara", "Helen", "Lucy", "Agnes", "Irene",
           "Martha", "Rita", "Diana", "Eva"]  # fmt: skip
VILLAGES = ["Riverside", "New Town", "Block 4", "Camp Hope", "Hilltop", "Lakeview",
            "Camp Unity", "East Gate"]  # fmt: skip
FACILITY_TYPES = ["Fixed/Facility", "Fixed/Facility", "fixed facility", "Outreach/Mobile",
                  "Outreach", "Outrech"]  # fmt: skip


@dataclass
class Child:
    serial: int
    sex: str
    birth: dt.date
    card: str
    names: str
    guardian: str
    phone: str
    village: str
    first_visit: dt.date = None  # type: ignore[assignment]
    doses: dict[str, dt.date] = field(default_factory=dict)
    written: dict[str, dt.date] = field(default_factory=dict)  # when each dose was written
    polio: dt.date | None = None
    # Edge cases: how the row is written differs from the truth.
    hide_birth: bool = False
    typo: str | None = None  # dose column written with the wrong year (looks like the future)
    crossed: str | None = None  # dose column with a crossed-out wrong date next to it


@dataclass
class Page:
    year: int
    month: int
    children: list[Child]
    month_text: str = ""
    year_text: str = ""


def clinic_day(day: dt.date) -> dt.date:
    while day.weekday() not in CLINIC_DAYS:
        day += dt.timedelta(days=1)
    return day


def fmt_date(day: dt.date, style: str) -> str:
    if style == "slash":
        return f"{day.day:02d}/{day.month:02d}/{day.year % 100:02d}"
    if style == "short":
        return f"{day.day}/{day.month}/{day.year % 100:02d}"
    return f"{day.day:02d}-{day.month:02d}-{day.year % 100:02d}"


def age_notation(birth: dt.date, on: dt.date) -> str:
    days = (on - birth).days
    if days < 7:
        return f"{days}/7"
    if days < 30:
        return f"{days // 7}/52"
    if days < 365:
        return f"{days * 12 // 365}/12"
    return str(days // 365)


def _child(rng: random.Random, serial: int, year: int, month: int) -> Child:
    days_in_month = (
        (dt.date(year + month // 12, month % 12 + 1, 1)) - dt.date(year, month, 1)
    ).days
    sex = rng.choice("FM")
    first = rng.choice(FIRST[:10] if sex == "F" else FIRST[10:])
    father, grandfather = rng.sample(FIRST[10:], 2)
    return Child(
        serial=serial,
        sex=sex,
        birth=dt.date(year, month, rng.randint(1, days_in_month)),
        card=f"{rng.randint(100, 9999)}",
        names=f"{first} {father} {grandfather}",
        guardian=f"{rng.choice(MOTHERS)} {rng.choice(FIRST[10:])}",
        phone=f"07{rng.choice('1235')}{rng.randint(1000000, 9999999)}",
        village=rng.choice(VILLAGES),
    )


def _vaccinate(rng: random.Random, child: Child, until: dt.date) -> None:
    """Visits follow the schedule with delays; a missed visit is never made up here."""
    previous = None
    for due, columns in VISITS:
        if due == 0:
            day = child.birth + dt.timedelta(days=rng.randint(0, 10))
        else:
            if rng.random() < MISSED_VISIT:
                break  # defaulter: stops coming
            day = child.birth + dt.timedelta(days=due + int(rng.expovariate(1 / 6)))
            if previous:  # at least four weeks between doses of a series
                day = max(day, previous + dt.timedelta(days=28))
            day = clinic_day(day)
        previous = day
        if day > until:
            break
        for column in columns:
            child.doses[column] = day
    if child.doses and rng.random() < 0.15:
        child.polio = min(child.doses.values())  # e.g. a campaign polio dose from the card


def _register(child: Child, rng: random.Random) -> None:
    """First visit here: everything before it is copied from the card on that day."""
    visits = sorted(set(child.doses.values()))
    child.first_visit = visits[0] if rng.random() < 0.7 else visits[min(1, len(visits) - 1)]
    for column, day in child.doses.items():
        child.written[column] = max(day, child.first_visit)


def simulate(seed: int, weeks: int, pages: int, go_live: dt.date) -> dict[str, Any]:
    """Children, doses and edge cases for `weeks` ISO weeks starting the week of go_live."""
    rng = random.Random(f"register-{seed}")
    first_week = iso_week(go_live)
    week_ids = [
        iso_week(iso_week_days(first_week)[0] + dt.timedelta(weeks=i)) for i in range(weeks)
    ]
    end = iso_week_days(week_ids[-1])[-1]

    # Cohorts whose children are due for something during the scenario.
    offsets = rng.sample([1, 2, 3, 4, 9, 10, 17, 18], k=min(pages, 8))
    cohorts = sorted({(go_live.replace(day=1) - dt.timedelta(days=30 * o)).replace(day=1) for o in offsets})  # fmt: skip
    book: list[Page] = []
    for start in cohorts:
        children = []
        for serial in range(1, rng.randint(8, 15) + 1):
            child = _child(rng, serial, start.year, start.month)
            _vaccinate(rng, child, end)
            if child.doses:
                _register(child, rng)
                children.append(child)
        for i, child in enumerate(children, start=1):
            child.serial = i
        style = rng.choice(["name", "abbr", "number"])
        month_text = {"name": start.strftime("%B"), "abbr": start.strftime("%b"),
                      "number": f"{start.month:02d}"}[style]  # fmt: skip
        year_text = str(start.year) if rng.random() < 0.6 else f"{start.year % 100:02d}"
        book.append(Page(start.year, start.month, children, month_text, year_text))

    # A catch-up page: a child over five gets MCV 2, which has no DHIS2 age band.
    if pages >= 3:
        old = go_live.replace(day=1) - dt.timedelta(days=365 * 5 + 40)
        child = _child(rng, 1, old.year, old.month)
        child.doses = {"mcv1": child.birth + dt.timedelta(days=300)}
        child.first_visit = clinic_day(go_live + dt.timedelta(days=rng.randint(0, 7 * weeks - 5)))
        child.doses["mcv2"] = child.first_visit
        child.written = {k: child.first_visit for k in child.doses}
        book.append(Page(old.year, old.month, [child], old.strftime("%B"), str(old.year)))

    edges = _inject_edges(rng, book, go_live, end)
    return {"book": book, "weeks": week_ids, "go_live": go_live, "end": end, "edges": edges}


def _in_scenario(child: Child, go_live: dt.date, end: dt.date) -> list[str]:
    return [c for c, d in child.doses.items() if go_live <= d <= end]


def _inject_edges(rng: random.Random, book: list[Page], go_live: dt.date, end: dt.date) -> list[dict[str, Any]]:  # fmt: skip
    """At most one of each edge case, on children with doses inside the scenario."""
    active = [(p, c) for p in book for c in p.children if _in_scenario(c, go_live, end)]
    rng.shuffle(active)
    edges = []

    def take():
        return active.pop() if active else (None, None)

    page, child = take()
    if child:
        child.hide_birth = True
        edges.append({"code": "birth_date_missing", "page": page, "child": child})
    page, child = take()
    if child:  # born the month before, but written on this cohort's page
        previous = child.birth.replace(day=1) - dt.timedelta(days=1)
        child.birth = previous.replace(day=min(child.birth.day, previous.day))
        edges.append({"code": "wrong_cohort", "page": page, "child": child})
    page, child = take()
    if child:
        child.typo = rng.choice(_in_scenario(child, go_live, end))
        edges.append({"code": "future_date", "page": page, "child": child, "column": child.typo})
    page, child = take()
    if child:
        child.crossed = rng.choice(_in_scenario(child, go_live, end))
        edges.append({"code": "crossed_out", "page": page, "child": child, "column": child.crossed})
    page, child = take()
    if child:  # written a week late: becomes a correction for the earlier week
        column = rng.choice(_in_scenario(child, go_live, end))
        late = child.written[column] + dt.timedelta(days=7)
        if late <= end:
            child.written[column] = late
            edges.append({"code": "late_entry", "page": page, "child": child, "column": column})
    return edges


def reading(page: Page, until: dt.date, date_style: str) -> dict[str, Any]:
    """What is written on the page by `until`, as the model should transcribe it."""
    rows = []
    for c in page.children:
        if c.first_visit > until:
            continue
        row: dict[str, Any] = {
            "serial": str(c.serial),
            "card_number": c.card,
            "sex": c.sex,
            "birth_date": None if c.hide_birth else fmt_date(c.birth, date_style),
            "age": age_notation(c.birth, c.first_visit),
        }
        for column in DOSE_COLUMNS:
            if column.key == "polio":
                day = c.polio if c.polio and c.first_visit <= until else None
            else:
                written = c.written.get(column.key)
                day = c.doses[column.key] if written and written <= until else None
            text = fmt_date(day, date_style) if day else None
            if text and column.key == c.typo:
                text = fmt_date(day.replace(year=day.year + 1), date_style)
            row[column.key] = text
        rows.append(row)
    return {"cohort_month": page.month_text, "cohort_year": page.year_text, "rows": rows}


def private_fields(page: Page) -> dict[str, dict[str, str]]:
    """Fields written on paper for realism but never extracted: keyed by serial."""
    return {str(c.serial): {"first_visit": c.first_visit, "names": c.names,
                            "guardian": c.guardian, "phone": c.phone, "village": c.village,
                            "crossed": c.crossed} for c in page.children}  # fmt: skip


def decisions_for(page_id: str, page: Page, date_style: str, written: dict[str, Any]) -> dict[str, Any]:  # fmt: skip
    """What a careful reviewer decides for this page's injected problems."""
    rows: dict[str, Any] = {}
    for c in page.children:
        row = next((r for r in written["rows"] if r["serial"] == str(c.serial)), None)
        if row is None:
            continue
        fix: dict[str, Any] = {}
        if c.hide_birth:
            fix["birth_date"] = fmt_date(c.birth, date_style)  # from the child's card
        if (c.birth.year, c.birth.month) != (page.year, page.month):
            fix["accept_cohort"] = True  # the doses are real; the row is just on the wrong page
        if c.typo and row.get(c.typo):
            fix["cells"] = {c.typo: fmt_date(c.doses[c.typo], date_style)}
        if fix:
            rows[str(c.serial)] = fix
    return {"rows": rows} if rows else {}


def build_truth(sim: dict[str, Any], ds: dict[str, Any], cover: dict[str, Any], styles: list[str]) -> list[dict[str, Any]]:  # fmt: skip
    """Per week: photos with their true readings, reviewer decisions and the expected draft
    (values and corrections) from running the real pipeline on the true readings."""
    from formbench.register_run import compute_draft, names

    spec, labels = load_spec(ds), names(ds)
    option = _intended_option(cover["facility_type"])
    state_root = Path(tempfile.mkdtemp(prefix="register-truth-"))
    weeks = []
    try:
        for week in sim["weeks"]:
            days = iso_week_days(week)
            photos, decisions = [], {"pages": {}}
            if option != cover["facility_type"]:
                decisions["attribute_option"] = option
            for index, page in enumerate(sim["book"]):
                new = any(days[0] <= d <= days[-1] for c in page.children for d in c.written.values())  # fmt: skip
                if not new:
                    continue
                written = reading(page, days[-1], styles[index])
                pid = page_identity(written, f"p{index}")
                photos.append({"page_index": index, "page_id": pid, "reading": written})
                page_decisions = decisions_for(pid, page, styles[index], written)
                if page_decisions:
                    decisions["pages"][pid] = page_decisions
            pages = [{"page_id": p["page_id"], **p["reading"]} for p in photos]
            draft = compute_draft(pages, cover, spec, labels, week=week, go_live=sim["go_live"],
                                  today=days[-1], state_root=state_root, decisions=decisions)  # fmt: skip
            if draft["unresolved_blocking"]:
                raise AssertionError(f"Truth for {week} leaves flags open: {draft['unresolved_blocking']}")  # fmt: skip
            RegisterState(draft["book"], root=state_root).approve(week, draft["pages"], "truth", "synthetic", str(days[-1]))  # fmt: skip
            weeks.append({
                "week": week, "today": days[-1].isoformat(), "photos": photos, "decisions": decisions,
                "expected": {"values": draft["values"], "corrections": draft["corrections"],
                             "notes": sorted({f["code"] for f in draft["flags"] if not f["blocking"]})},
            })  # fmt: skip
    finally:
        shutil.rmtree(state_root, ignore_errors=True)
    return weeks


def _intended_option(written: str) -> str:
    text = written.casefold()
    return "Outreach/Mobile" if text.startswith("out") else "Fixed/Facility"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a synthetic handwritten register scenario.")  # fmt: skip
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--weeks", type=int, default=3)
    parser.add_argument("--pages", type=int, default=3, help="cohort pages in the register")
    parser.add_argument("--go-live", type=dt.date.fromisoformat, default=dt.date(2026, 9, 7))
    parser.add_argument("--quality", choices=["mixed", "scan", "good", "poor"], default="mixed")
    parser.add_argument("--metadata", type=Path, default=Path("schemas/dataset/metadata.json"))
    parser.add_argument("--out", type=Path, default=Path("data/private/synthetic"))
    args = parser.parse_args()

    from formbench.register_render import render_scenario

    rng = random.Random(f"scenario-{args.seed}")
    sim = simulate(args.seed, args.weeks, args.pages, args.go_live)
    styles = [rng.choice(["dash", "slash", "short"]) for _ in sim["book"]]
    cover = {"facility_name": "Riverside Health Centre", "mfl_code": f"MFL-{rng.randint(1000, 9999)}",  # fmt: skip
             "facility_type": rng.choice(FACILITY_TYPES), "district": "Hillside", "region": "Northern"}  # fmt: skip
    truth = build_truth(sim, json.loads(args.metadata.read_text()), cover, styles)
    folder = args.out / f"seed-{args.seed}"
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir(parents=True)
    render_scenario(folder, sim, truth, cover, styles, quality=args.quality, seed=args.seed)
    edges = [{"code": e["code"], "serial": str(e["child"].serial), "cohort": f"{e['page'].year}-{e['page'].month:02d}",
              "column": e.get("column")} for e in sim["edges"]]  # fmt: skip
    meta = {"seed": args.seed, "go_live": args.go_live.isoformat(), "cover": cover,
            "intended_option": _intended_option(cover["facility_type"]), "edges": edges, "weeks": truth}  # fmt: skip
    (folder / "truth.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False, default=str) + "\n"
    )
    photos = sum(len(w["photos"]) for w in truth)
    print(f"Scenario seed-{args.seed}: {len(truth)} weeks, {photos} page photos -> {folder}/")


if __name__ == "__main__":
    main()

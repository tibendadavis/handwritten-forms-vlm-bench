"""Benchmark the register pipeline on a synthetic scenario, end to end.

Each week of the scenario is read by the model, turned into a draft, reviewed by an oracle
reviewer who fixes only what the system flagged (using the truth, like a careful health
worker with the register and the child's card in hand), and approved. Scores, in layers:

- transcription: rows found or invented, cell accuracy, blanks filled or missed;
- review: blocking flags per week (reviewer workload), injected problems caught, false alarms;
- values: the approved daily values against the expected ones. Mismatches here are errors
  that reached the payload unflagged, which is what data integrity is about.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

from dotenv import find_dotenv, load_dotenv
from rich.console import Console
from rich.table import Table

from formbench.handwriting import Unreadable, parse_date
from formbench.register_aggregate import parse_cohort, resolve_attribute_option
from formbench.register_run import compute_draft, names
from formbench.register_spec import DOSE_COLUMNS, load_spec
from formbench.register_state import RegisterState, page_identity
from formbench.schema_gen import slugify

DOSES = [c.key for c in DOSE_COLUMNS]
CHILDREN = "Children vaccinated"


def _blank(value: Any) -> bool:
    return value is None or not str(value).strip()


def same_value(truth: Any, predicted: Any, key: str) -> bool:
    if _blank(truth) or _blank(predicted):
        return _blank(truth) and _blank(predicted)
    if key in ("sex",):
        return str(truth).strip()[:1].upper() == str(predicted).strip()[:1].upper()
    if key in ("birth_date", *DOSES):
        try:
            return parse_date(str(truth)) == parse_date(str(predicted))
        except Unreadable:
            return False
    return " ".join(str(truth).split()).casefold() == " ".join(str(predicted).split()).casefold()


def score_transcription(predicted: dict[str, Any], truth: dict[str, Any]) -> dict[str, Any]:
    """Counts for one page photo; summed over pages before rates are taken."""
    truth_rows = {r["serial"]: r for r in truth["rows"]}
    pred_rows: dict[str, dict[str, Any]] = {}
    invented = 0
    for row in predicted["rows"]:
        serial = str(row.get("serial") or "").strip()
        if serial in truth_rows and serial not in pred_rows:
            pred_rows[serial] = row
        else:
            invented += 1
    counts = {"rows": len(truth_rows), "rows_found": len(pred_rows), "rows_invented": invented,
              "date_cells": 0, "date_correct": 0, "blank_cells": 0, "blank_filled": 0,
              "written_cells": 0, "written_missed": 0, "birth": 0, "birth_correct": 0,
              "sex_correct": 0, "card_correct": 0}  # fmt: skip
    for serial, t in truth_rows.items():
        p = pred_rows.get(serial)
        if p is None:
            continue
        counts["birth"] += not _blank(t.get("birth_date"))
        counts["birth_correct"] += not _blank(t.get("birth_date")) and same_value(t["birth_date"], p.get("birth_date"), "birth_date")  # fmt: skip
        counts["sex_correct"] += same_value(t.get("sex"), p.get("sex"), "sex")
        counts["card_correct"] += same_value(t.get("card_number"), p.get("card_number"), "card_number")  # fmt: skip
        for key in DOSES:
            tv, pv = t.get(key), p.get(key)
            if _blank(tv):
                counts["blank_cells"] += 1
                counts["blank_filled"] += not _blank(pv)
            else:
                counts["written_cells"] += 1
                counts["written_missed"] += _blank(pv)
                counts["date_cells"] += 1
                counts["date_correct"] += same_value(tv, pv, key)
    try:
        counts["cohort_correct"] = int(parse_cohort(predicted.get("cohort_month"), predicted.get("cohort_year"))
                                       == parse_cohort(truth["cohort_month"], truth["cohort_year"]))  # fmt: skip
    except Unreadable:
        counts["cohort_correct"] = 0
    return counts


def _key(v: dict[str, Any]) -> tuple[str, str, str]:
    return v["day"], v["data_element"], v["coc"]


def score_values(produced: list[dict[str, Any]], expected: list[dict[str, Any]]) -> dict[str, int]:
    got = {_key(v): v["value"] for v in produced}
    want = {_key(v): v["value"] for v in expected}
    children = {_key(v) for v in expected + produced if v["data_element_name"] == CHILDREN}
    keys = set(got) | set(want)
    return {
        "values": len(keys),
        "values_exact": sum(got.get(k) == want.get(k) for k in keys),
        "values_missing": sum(k not in got for k in want),
        "values_extra": sum(k not in want for k in got),
        "abs_error": sum(abs(got.get(k, 0) - want.get(k, 0)) for k in keys),
        "children": len(children),
        "children_exact": sum(got.get(k) == want.get(k) for k in children),
    }


def oracle_decisions(draft: dict[str, Any], pages: list[dict[str, Any]], photos: list[dict[str, Any]],
                     week_truth: dict[str, Any], intended_option: str) -> dict[str, Any]:  # fmt: skip
    """Resolve every open blocking flag the way a reviewer with the truth would."""
    decisions: dict[str, Any] = {"pages": {}, "resolved": {}}
    by_page = {p["page_id"]: (p, photo) for p, photo in zip(pages, photos)}
    for flag in draft["flags"]:
        if not flag["blocking"] or flag["resolved"]:
            continue
        code, pid, serial, column = flag["code"], flag["page"], flag["serial"], flag["column"]
        if code in ("facility_type_unclear", "attribute_option_invalid"):
            decisions["attribute_option"] = intended_option
            continue
        if pid not in by_page:
            decisions["resolved"][flag["id"]] = "oracle"
            continue
        _, photo = by_page[pid]
        truth_page = photo["reading"]
        true_fix = (
            week_truth["decisions"].get("pages", {}).get(photo["page_id"], {}).get("rows", {})
        )
        page_dec = decisions["pages"].setdefault(pid, {"rows": {}})
        if code == "cohort_unreadable":
            year, month = parse_cohort(truth_page["cohort_month"], truth_page["cohort_year"])
            page_dec["cohort"] = f"{year}-{month:02d}"
            continue
        truth_row = next((r for r in truth_page["rows"] if r["serial"] == serial), None)
        if truth_row is None:  # an invented row: tell the system it is not real
            decisions["resolved"][flag["id"]] = "oracle: not a real row"
            for key in DOSES:
                page_dec["rows"].setdefault(serial, {}).setdefault("cells", {})[key] = None
            continue
        row_dec = page_dec["rows"].setdefault(serial, {})
        fix = true_fix.get(serial, {})
        if code in ("birth_date_missing", "birth_date_unreadable"):
            row_dec["birth_date"] = fix.get("birth_date") or truth_row["birth_date"]
        elif code == "sex_unreadable":
            row_dec["sex"] = truth_row["sex"]
        elif code == "wrong_cohort":
            if fix.get("accept_cohort"):
                row_dec["accept_cohort"] = True
            else:
                row_dec["birth_date"] = truth_row["birth_date"]
                row_dec["accept_cohort"] = True
        elif code in ("date_unreadable", "future_date", "before_birth", "changed_since_approved") and column:  # fmt: skip
            value = (fix.get("cells") or {}).get(column, truth_row.get(column))
            if column in ("birth_date", "sex"):
                row_dec[column] = value
            else:
                row_dec.setdefault("cells", {})[column] = value
            if code == "changed_since_approved":
                decisions["resolved"][flag["id"]] = "oracle"
        else:
            decisions["resolved"][flag["id"]] = "oracle"
    return decisions


def _merge(base: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    out = json.loads(json.dumps(base))
    if extra.get("attribute_option"):
        out["attribute_option"] = extra["attribute_option"]
    out.setdefault("resolved", {}).update(extra.get("resolved", {}))
    for pid, page in extra.get("pages", {}).items():
        target = out.setdefault("pages", {}).setdefault(pid, {"rows": {}})
        if "cohort" in page:
            target["cohort"] = page["cohort"]
        for serial, row in page.get("rows", {}).items():
            merged = target.setdefault("rows", {}).setdefault(serial, {})
            cells = {**merged.get("cells", {}), **row.get("cells", {})}
            merged.update(row)
            if cells:
                merged["cells"] = cells
    return out


def review_and_approve(pages, photos, cover, spec, labels, week_truth, *, go_live, intended_option, state_root):  # fmt: skip
    """Draft -> oracle review until no blocking flag is open -> approve. Returns the
    first draft (what the reviewer saw) and the approved one."""
    week = week_truth["week"]
    today = dt.date.fromisoformat(week_truth["today"])

    def draft_with(decisions):
        return compute_draft(pages, cover, spec, labels, week=week, go_live=go_live, today=today,
                             state_root=state_root, decisions=decisions)  # fmt: skip

    decisions: dict[str, Any] = {}
    first = current = draft_with(decisions)
    for _ in range(4):
        if not current["unresolved_blocking"]:
            break
        decisions = _merge(decisions, oracle_decisions(current, pages, photos, week_truth, intended_option))  # fmt: skip
        current = draft_with(decisions)
    if current["unresolved_blocking"]:  # should not happen; resolve the rest blindly
        decisions["resolved"].update(
            {fid: "oracle: forced" for fid in current["unresolved_blocking"]}
        )
        current = draft_with(decisions)
    RegisterState(current["book"], root=state_root).approve(week, current["pages"], "oracle", "bench", week_truth["today"])  # fmt: skip
    return first, current


def flag_catch(drafts: list[dict[str, Any]], edges: list[dict[str, Any]], cover_flag_expected: bool) -> dict[str, int]:  # fmt: skip
    """Injected problems that should raise a blocking flag, and how many of them did."""
    expected = {(e["code"], e["serial"]) for e in edges if e["code"] in ("birth_date_missing", "wrong_cohort", "future_date")}  # fmt: skip
    raised = {(f["code"], f["serial"]) for d in drafts for f in d["flags"] if f["blocking"]}
    caught = {e for e in expected if e in raised}
    if cover_flag_expected:
        expected.add(("facility_type_unclear", None))
        if any(f["code"] == "facility_type_unclear" for d in drafts for f in d["flags"]):
            caught.add(("facility_type_unclear", None))
    return {"edges": len(expected), "edges_caught": len(caught),
            "blocking_flags": sum(len(d["unresolved_blocking"]) for d in drafts)}  # fmt: skip


ROW_FLAGS = {"birth_date_missing", "birth_date_unreadable", "sex_unreadable", "wrong_cohort",
             "duplicate_serial", "row_missing"}  # fmt: skip
CHECKED = ("sex", "birth_date", *DOSES)


def review_counts(flags: list[dict[str, Any]], pages: list[dict[str, Any]], photos: list[dict[str, Any]],
                  edges: list[dict[str, Any]]) -> dict[str, int]:  # fmt: skip
    """Did flags point the reviewer at the model's misreads, and how many flags were raised on
    readings that were actually right (false alarms)?

    A row-level flag (missing birth date, wrong cohort, ...) puts the whole row in front of the
    reviewer, so it covers every cell of that row; a cell flag covers its cell.
    """
    edge_rows = {(e["code"], e["serial"]) for e in edges}
    misread: set[tuple[str, str, str]] = set()
    truth_rows: dict[str, dict[str, Any]] = {}
    for page, photo in zip(pages, photos):
        truth = {r["serial"]: r for r in photo["reading"]["rows"]}
        truth_rows[page["page_id"]] = truth
        for row in page["rows"]:
            serial = str(row.get("serial") or "")
            t = truth.get(serial)
            for key in CHECKED:
                if t is None:  # invented row: anything it says is a misread
                    if not _blank(row.get(key)):
                        misread.add((page["page_id"], serial, key))
                elif not same_value(t.get(key), row.get(key), key):
                    misread.add((page["page_id"], serial, key))
    covered: set[tuple[str, str, str]] = set()
    false_alarms = 0
    for f in flags:
        if not f["blocking"] or not f["page"]:
            continue
        pid, serial = f["page"], f["serial"]
        if f["code"] == "cohort_unreadable":
            cells = {m for m in misread if m[0] == pid}
            useful = True
        elif f["code"] in ROW_FLAGS or not f["column"]:
            cells = {m for m in misread if m[:2] == (pid, serial)}
            useful = (
                bool(cells)
                or (f["code"], serial) in edge_rows
                or serial not in truth_rows.get(pid, {})
            )
        elif f["code"] == "changed_since_approved":
            # Two readings of one cell disagree, so one of them is wrong: if today's is right,
            # the flag exposes a wrong value approved earlier. Never a false alarm.
            cells = {(pid, serial, f["column"])} & misread
            useful = True
        else:
            cells = {(pid, serial, f["column"])} & misread
            useful = bool(cells) or (f["code"], serial) in edge_rows
        covered |= cells
        false_alarms += not useful
    return {
        "misreads": len(misread),
        "misreads_flagged": len(covered),
        "false_alarms": false_alarms,
    }


def _sum(dicts: list[dict[str, int]]) -> dict[str, int]:
    out: dict[str, int] = {}
    for d in dicts:
        for k, v in d.items():
            out[k] = out.get(k, 0) + v
    return out


def _rate(a: int, b: int) -> float | None:
    return round(a / b, 4) if b else None


def summarize(counts: dict[str, int], weeks: int, pages: int, latency: float, valid: int) -> dict[str, Any]:  # fmt: skip
    return {
        "json_valid": _rate(valid, pages),
        "row_recall": _rate(counts["rows_found"], counts["rows"]),
        "rows_invented": counts["rows_invented"],
        "date_cell_accuracy": _rate(counts["date_correct"], counts["date_cells"]),
        "blank_hallucination_rate": _rate(counts["blank_filled"], counts["blank_cells"]),
        "written_omission_rate": _rate(counts["written_missed"], counts["written_cells"]),
        "birth_date_accuracy": _rate(counts["birth_correct"], counts["birth"]),
        "sex_accuracy": _rate(counts["sex_correct"], counts["rows_found"]),
        "card_accuracy": _rate(counts["card_correct"], counts["rows_found"]),
        "cohort_accuracy": _rate(counts["cohort_correct"], pages),
        "flags_per_week": round(counts["blocking_flags"] / weeks, 2),
        "edge_recall": _rate(counts["edges_caught"], counts["edges"]),
        "misread_flag_recall": _rate(counts["misreads_flagged"], counts["misreads"]),
        "false_alarms": counts["false_alarms"],
        "value_accuracy": _rate(counts["values_exact"], counts["values"]),
        "values_missing": counts["values_missing"],
        "values_extra": counts["values_extra"],
        "value_abs_error": counts["abs_error"],
        "children_accuracy": _rate(counts["children_exact"], counts["children"]),
        "latency_s_per_page": round(latency / max(pages, 1), 1),
    }


def score_run(weeks_out: list[dict[str, Any]], truth: dict[str, Any], ds: dict[str, Any],
              latency: float) -> dict[str, Any]:  # fmt: skip
    """All scores from what a run recorded, so saved runs can be re-scored without a model."""
    spec = load_spec(ds)
    page_counts, value_counts, review, drafts = [], [], [], []
    pages_total = valid = 0
    for week, out in zip(truth["weeks"], weeks_out):
        pages = [{"page_id": r["page_id"], **r["reading"]} for r in out["pages"]]
        for reading, photo in zip(out["pages"], week["photos"]):
            pages_total += 1
            valid += int(reading["json_valid"])
            page_counts.append(score_transcription(reading["reading"], photo["reading"]))
        first = {"flags": out["first_flags"],
                 "unresolved_blocking": [f["id"] for f in out["first_flags"] if f["blocking"] and not f["resolved"]]}  # fmt: skip
        drafts.append(first)
        review.append(review_counts(out["first_flags"], pages, week["photos"], truth["edges"]))
        value_counts.append(score_values(out["approved_values"] + out["approved_corrections"],
                                         week["expected"]["values"] + week["expected"]["corrections"]))  # fmt: skip
    cover_expected = resolve_attribute_option(truth["cover"]["facility_type"], spec)[1] is not None
    counts = {**_sum(page_counts), **_sum(value_counts), **_sum(review),
              **flag_catch(drafts, truth["edges"], cover_expected)}  # fmt: skip
    metrics = summarize(counts, len(truth["weeks"]), pages_total, latency, valid)
    return {"metrics": metrics, "counts": counts}


def run_bench(scenario: Path, model: str, mode: str, ds: dict[str, Any], think: bool = False) -> dict[str, Any]:  # fmt: skip
    from formbench.register_extract import read_cover, read_page
    from formbench.register_template import load_template

    truth = json.loads((scenario / "truth.json").read_text())
    spec, labels = load_spec(ds), names(ds)
    template, layout = load_template()
    go_live = dt.date.fromisoformat(truth["go_live"])
    state_root = Path(tempfile.mkdtemp(prefix="register-bench-"))
    start = time.perf_counter()
    weeks_out = []
    try:
        cover = read_cover(scenario / "cover.jpg", model, think=think)
        for week in truth["weeks"]:
            pages, readings = [], []
            for photo in week["photos"]:
                result = read_page(scenario / photo["file"], model, mode=mode, template=template, layout=layout, think=think)  # fmt: skip
                pid = page_identity(result["reading"], Path(photo["file"]).stem)
                pages.append({"page_id": pid, **result["reading"]})
                readings.append({"file": photo["file"], "page_id": pid, "notes": result["notes"],
                                 "json_valid": result["json_valid"], "reading": result["reading"]})  # fmt: skip
            first, approved = review_and_approve(
                pages, week["photos"], cover["cover"], spec, labels, week,
                go_live=go_live, intended_option=truth["intended_option"], state_root=state_root,
            )  # fmt: skip
            weeks_out.append({"week": week["week"], "pages": readings, "first_flags": first["flags"],
                              "approved_values": approved["values"], "approved_corrections": approved["corrections"]})  # fmt: skip
        latency = time.perf_counter() - start
    finally:
        shutil.rmtree(state_root, ignore_errors=True)
    scored = score_run(weeks_out, truth, ds, latency)
    return {**scored, "latency_s": round(latency, 1), "cover": cover["cover"], "weeks": weeks_out}


def _table(title: str, rows: list[tuple[str, str, dict[str, Any]]]) -> Table:
    table = Table(title=title)
    table.add_column("Metric")
    for engine, mode, _ in rows:
        table.add_column(f"{engine.split(':', 1)[-1][-28:]}\n{mode}", justify="right")
    for name in rows[0][2]:
        cells = []
        for _, _, metrics in rows:
            value = metrics[name]
            percent = isinstance(value, float) and name.endswith(
                ("accuracy", "rate", "recall", "valid")
            )
            cells.append("n/a" if value is None else (f"{value:.1%}" if percent else str(value)))
        table.add_row(name, *cells)
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark VLMs on a synthetic register scenario.")
    parser.add_argument(
        "--scenario", type=Path, required=True, help="data/private/synthetic/seed-N"
    )
    parser.add_argument("--engine", action="append", help="ollama:<tag>; repeatable")
    parser.add_argument("--mode", action="append", choices=["tiles", "full"], help="repeatable")
    parser.add_argument("--think", action="store_true")
    parser.add_argument(
        "--rescore", type=Path, nargs="+", help="re-score saved run files, no model"
    )
    parser.add_argument("--metadata", type=Path, default=Path("schemas/ri_v2/metadata.json"))
    parser.add_argument("--out", type=Path, default=Path("results/register/bench"))
    args = parser.parse_args()

    load_dotenv(find_dotenv(usecwd=True))
    ds = json.loads(args.metadata.read_text())
    console = Console()
    rows = []
    if args.rescore:
        truth = json.loads((args.scenario / "truth.json").read_text())
        for path in args.rescore:
            run = json.loads(path.read_text())
            latency = run.get("latency_s") or run["metrics"]["latency_s_per_page"] * sum(len(w["pages"]) for w in run["weeks"])  # fmt: skip
            run.update(score_run(run["weeks"], truth, ds, latency), latency_s=latency)
            path.write_text(json.dumps(run, indent=2, ensure_ascii=False, default=str) + "\n")
            rows.append((run["settings"]["engine"], run["settings"]["mode"], run["metrics"]))
        console.print(_table(f"Register benchmark (re-scored): {args.scenario.name}", rows))
        return
    if not args.engine:
        raise SystemExit("register-bench: give --engine (or --rescore run files)")

    from langfuse import get_client

    langfuse = get_client()
    args.out.mkdir(parents=True, exist_ok=True)
    for engine in args.engine:
        kind, _, model = engine.partition(":")
        if kind != "ollama" or not model:
            raise SystemExit("register-bench: --engine must be 'ollama:<model-tag>'")
        for mode in args.mode or ["tiles"]:
            settings = {"scenario": args.scenario.name, "engine": engine, "mode": mode, "think": args.think}  # fmt: skip
            console.print(f"Running {engine} ({mode}) on {args.scenario.name}...")
            try:
                with langfuse.start_as_current_span(name="register-bench", metadata=settings):
                    langfuse.update_current_trace(name=f"register-bench {args.scenario.name}", metadata=settings, tags=["register-bench", mode])  # fmt: skip
                    result = run_bench(args.scenario, model, mode, ds, think=args.think)
                    for name, value in result["metrics"].items():
                        if value is not None:
                            langfuse.score_current_trace(name=name, value=float(value))
                    trace_id = langfuse.get_current_trace_id()
            finally:
                langfuse.flush()
            stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            path = args.out / f"{stamp}_{args.scenario.name}_{slugify(engine)}_{mode}.json"
            path.write_text(json.dumps({"settings": settings, "trace_id": trace_id, **result}, indent=2, ensure_ascii=False, default=str) + "\n")  # fmt: skip
            rows.append((engine, mode, result["metrics"]))
    console.print(_table(f"Register benchmark: {args.scenario.name}", rows))
    console.print(f"Runs saved under {args.out}/")


if __name__ == "__main__":
    main()

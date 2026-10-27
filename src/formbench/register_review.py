"""Local review page and decisions file for a weekly register run."""

from __future__ import annotations

import base64
import html
from collections import defaultdict
from pathlib import Path
from typing import Any

from formbench.register_spec import DOSE_COLUMNS

HOW_TO = {
    "facility_type_unclear": "set attribute_option to one of the options",
    "attribute_option_invalid": "set attribute_option to one of the options",
    "cohort_unreadable": 'pages.<page>.cohort: "YYYY-MM"',
    "birth_date_missing": "pages.<page>.rows.<serial>.birth_date (from the child's card)",
    "birth_date_unreadable": "pages.<page>.rows.<serial>.birth_date",
    "sex_unreadable": "pages.<page>.rows.<serial>.sex: F or M",
    "wrong_cohort": "pages.<page>.rows.<serial>.accept_cohort: true, or fix birth_date",
    "date_unreadable": "pages.<page>.rows.<serial>.cells.<column>: the date, or null",
    "future_date": "pages.<page>.rows.<serial>.cells.<column>: the real date, or null",
    "before_birth": "fix birth_date or cells.<column>",
    "duplicate_serial": "resolved.<id>: why both rows are real (or fix a serial)",
    "changed_since_approved": "cells.<column> with the true value, then resolved.<id>",
    "row_missing": "resolved.<id>: why the row is gone",
}

CSS = """
:root { --bg:#fbfbf8; --fg:#1d1d1b; --muted:#6b6b66; --line:#dddcd5; --bad:#b3261e;
        --warn:#8a5a00; --ok:#1e6b35; --card:#ffffff; }
@media (prefers-color-scheme: dark) { :root { --bg:#161614; --fg:#ecebe6; --muted:#a4a39c;
        --line:#3a3a36; --bad:#ff8a80; --warn:#ffcc66; --ok:#7ed69a; --card:#1f1f1c; } }
body { background:var(--bg); color:var(--fg); font:15px/1.5 system-ui, sans-serif;
       margin:0 auto; max-width:1200px; padding:16px; }
h1 { font-size:22px; margin:8px 0 4px; } h2 { font-size:18px; margin:28px 0 8px; }
.muted { color:var(--muted); } .bad { color:var(--bad); } .warn { color:var(--warn); }
.ok { color:var(--ok); }
table { border-collapse:collapse; width:100%; background:var(--card); margin:8px 0; }
th, td { border-bottom:1px solid var(--line); padding:6px 8px; text-align:left;
         vertical-align:top; } th { font-weight:600; }
td.num { text-align:right; font-variant-numeric:tabular-nums; }
.wrap { overflow-x:auto; } img { max-width:100%; border:1px solid var(--line); }
code { font-size:13px; } a { color:inherit; }
"""


def _e(value: Any) -> str:
    return html.escape("" if value is None else str(value))


def _src_link(source: list[str]) -> str:
    page, serial, column = source
    label = "row" if column == "*" else column
    return f'<a href="#row-{_e(page)}-{_e(serial)}">{_e(page)} #{_e(serial)} {_e(label)}</a>'


def _img(path: Path) -> str:
    data = base64.b64encode(path.read_bytes()).decode()
    return f'<img alt="{_e(path.stem)}" src="data:image/jpeg;base64,{data}">'


def render_html(run_dir: Path, draft: dict[str, Any], readings: dict[str, Any]) -> str:
    parts = [f"<!doctype html><html lang=en><meta charset=utf-8>"
             f'<meta name=viewport content="width=device-width,initial-scale=1">'
             f"<title>Register review {_e(draft['week'])}</title><style>{CSS}</style><body>"]  # fmt: skip
    cover = draft["cover"]
    aoc = draft["attribute_option"]
    org = draft.get("org_unit") or {}
    blocking = draft["unresolved_blocking"]
    parts.append(
        f"<h1>Week {_e(draft['week'])}: {_e(draft['days'][0])} to {_e(draft['days'][-1])}</h1>"
        f"<p>{_e(cover.get('facility_name'))} · MFL {_e(cover.get('mfl_code'))} · "
        f"session type <b>{_e(aoc['name'] or 'unresolved')}</b> · orgUnit "
        f"<code>{_e(org.get('uid') or 'null')}</code>"
        + (f' <span class="warn">({_e(org.get("warning"))})</span>' if org.get("warning") else "")
        + "</p>"
        + (
            f'<p class="bad"><b>{len(blocking)} flag(s) must be resolved before approval.</b></p>'
            if blocking
            else '<p class="ok">No open blocking flags. A person must still approve.</p>'
        )  # fmt: skip
    )

    flags = draft["flags"]
    if flags:
        parts.append("<h2>Flags</h2><div class=wrap><table><tr><th></th><th>What</th><th>Where</th><th>How to resolve</th></tr>")  # fmt: skip
        for f in flags:
            state = "resolved" if f["resolved"] else ("must resolve" if f["blocking"] else "note")
            css = "ok" if f["resolved"] else ("bad" if f["blocking"] else "muted")
            where = _src_link([f["page"], f["serial"], f["column"] or "*"]) if f["page"] and f["serial"] else _e(f["page"] or "")  # fmt: skip
            how = HOW_TO.get(f["code"], "") if f["blocking"] else ""
            parts.append(f'<tr><td class={css}>{state}</td><td>{_e(f["message"])}<br><code class=muted>{_e(f["id"])}</code></td><td>{where}</td><td>{_e(how)}</td></tr>')  # fmt: skip
        parts.append("</table></div>")

    for title, rows, has_prev in (("Daily values", draft["values"], False), ("Corrections to approved weeks", draft["corrections"], True)):  # fmt: skip
        if not rows and has_prev:
            continue
        parts.append(f"<h2>{title}</h2>")
        if not rows:
            parts.append('<p class="muted">No doses counted for this week.</p>')
            continue
        head = "<th>Before</th>" if has_prev else ""
        parts.append(f"<div class=wrap><table><tr><th>Day</th><th>Data element</th><th>Disaggregation</th>{head}<th>Value</th><th>From</th></tr>")  # fmt: skip
        for v in rows:
            prev = f"<td class=num>{_e(v.get('previous'))}</td>" if has_prev else ""
            sources = ", ".join(_src_link(s) for s in v["sources"])
            parts.append(f"<tr><td>{_e(v['day'])}</td><td>{_e(v['data_element_name'])}</td><td>{_e(v['coc_name'])}</td>{prev}<td class=num><b>{v['value']}</b></td><td>{sources}</td></tr>")  # fmt: skip
        parts.append("</table></div>")

    parts.append("<h2>Rows as read</h2><p class=muted>Only the counting columns are shown; names and contacts are never read.</p>")  # fmt: skip
    keys = [c.key for c in DOSE_COLUMNS]
    for page in readings["pages"]:
        pid = page["page_id"]
        parts.append(f"<h3>{_e(pid)} · cohort {_e(page.get('cohort_month'))} {_e(page.get('cohort_year'))}</h3>")  # fmt: skip
        by_strip: dict[Any, list[dict[str, Any]]] = defaultdict(list)
        for row in page["rows"]:
            by_strip[row.get("_strip")].append(row)
        strips = sorted((run_dir / "strips").glob(f"{pid}-*.jpg"))
        for index, rows in sorted(by_strip.items(), key=lambda kv: (kv[0] is None, kv[0] or 0)):
            shown = [strips[index]] if index is not None and index < len(strips) else strips
            parts.extend(_img(p) for p in shown)
            parts.append("<div class=wrap><table><tr><th>Serial</th><th>Sex</th><th>Birth</th><th>Age</th>" + "".join(f"<th>{_e(k)}</th>" for k in keys) + "</tr>")  # fmt: skip
            for row in rows:
                anchor = f"row-{_e(pid)}-{_e(row.get('serial'))}"
                cells = "".join(f"<td>{_e(row.get(k))}</td>" for k in keys)
                parts.append(f"<tr id='{anchor}'><td>{_e(row.get('serial'))}</td><td>{_e(row.get('sex'))}</td><td>{_e(row.get('birth_date'))}</td><td>{_e(row.get('age'))}</td>{cells}</tr>")  # fmt: skip
            parts.append("</table></div>")
    parts.append("</body></html>")
    return "\n".join(parts)


def decisions_template(draft: dict[str, Any], options: list[str]) -> str:
    lines = [
        "# Reviewer decisions for this run. Edit, then run register-approve.",
        "# Every flag marked 'must resolve' in review.html needs a fix below or an entry",
        "# under 'resolved'. Approval is always required, even with no flags.",
        "#",
        f"# Session type options: {', '.join(options)}",
        "attribute_option: null",
        "pages: {}",
        "#  <page_id>:",
        '#    cohort: "2025-04"',
        "#    rows:",
        '#      "<serial>": {birth_date: "10-04-25", sex: F, accept_cohort: true,',
        '#                   cells: {opv1: "12-06-25"}}',
        "resolved: {}",
        "#  <flag id>: reason it is acceptable as read",
        "#",
        "# Open flags:",
    ]
    lines += [f"#  {fid}" for fid in draft["unresolved_blocking"]] or ["#  (none)"]
    return "\n".join(lines) + "\n"


def write_review(run_dir: Path, draft: dict[str, Any], readings: dict[str, Any], options: list[str] | None = None) -> None:  # fmt: skip
    (run_dir / "review.html").write_text(render_html(run_dir, draft, readings))
    decisions = run_dir / "decisions.yaml"
    if not decisions.exists():  # never overwrite a reviewer's work
        decisions.write_text(decisions_template(draft, options or []))

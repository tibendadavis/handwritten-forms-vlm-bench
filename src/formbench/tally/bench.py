"""Benchmark models on photos of tally sheet sections, or of whole pages, one model per run.

A folder holds one photo per section, named <sheet>_<section>.jpg (sheet-01_A.jpg, sheet-01_B.jpg,
...), and one truth file per filled sheet, <sheet>.json, with every field of the sheet. The
form is described in form.json in the folder or the folder above it (tally/form.py).

The same folder can hold whole-page photos of the same sheets: <sheet>.jpg is page 1 and
<sheet>_p2.jpg page 2. With pages=True they are benchmarked instead of the section photos, each
page read in one call and scored on the sections printed on it, so the two ways of taking
photos are compared on the same truth.

As for registers, each folder is a Langfuse dataset and each run one of its runs; the scores
are the register's reading scores, so `formbench compare` shows tally runs the same way.
"""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path
from typing import Any

from rich.console import Console

from formbench.evaluation.bench import PHOTO_TYPES, run_label, score_table
from formbench.evaluation.scoring import comments, metrics, sum_counts
from formbench.evaluation.tracking import evaluations, item_input, score_configs, sync_dataset
from formbench.files import slugify, stamp, write_json
from formbench.prompt_store import fetch_prompt, prompt_version
from formbench.register.reader import FULL_MAX_SIDE
from formbench.tally.form import (
    DEFAULT_PAGE_PROMPT,
    DEFAULT_PROMPT,
    PAGE_MIN_SIDE,
    Form,
    find_form,
    load_form,
    load_sheet_truth,
    read_page,
    read_section,
    score_section,
)

RESULTS = Path("results/tally/bench")
PHOTO = re.compile(r"^(?P<sheet>.+)[_-](?P<section>[A-Za-z])$")
PAGE_PHOTO = re.compile(r"^(?P<sheet>.+?)(?:[_-]p(?P<page>\d+))?$")


def truth_file(folder: Path, sheet: str) -> Path | None:
    return next((folder / n for n in (f"{sheet}.truth.json", f"{sheet}.json") if (folder / n).exists()), None)  # fmt: skip


def photo_sections(photo: Path, form: Form) -> tuple[str, str, int | None] | None:
    """(sheet, section letters, page number) of a photo by its name: a section photo has no page
    number. None when the name is neither."""
    match = PHOTO.match(photo.stem)
    if match and match["section"].upper() in form.sections:
        return match["sheet"], match["section"].upper(), None
    match = PAGE_PHOTO.match(photo.stem)
    page = int(match["page"] or 1)
    return (match["sheet"], "".join(form.pages[page - 1]), page) if 1 <= page <= len(form.pages) else None  # fmt: skip


def load_items(folder: Path, form: Form, pages: bool = False) -> tuple[list[dict[str, Any]], list[str]]:  # fmt: skip
    """The section photos (or with pages, the whole-page photos) of a folder with what is really
    written in them, as dataset items, and the photos left out (with why). Photos of the other
    kind are passed over without a word."""
    items, skipped, truths = [], [], {}
    for photo in sorted(p for p in folder.iterdir() if p.suffix.lower() in PHOTO_TYPES):
        found = photo_sections(photo, form)
        truth = found and truth_file(folder, found[0])
        if found and truth and (found[2] is not None) != pages:
            continue  # a photo of the other kind
        if not found or (found[2] is None) == pages:
            want = "<sheet>.jpg or <sheet>_p<page>.jpg" if pages else f"<sheet>_<section>.jpg, sections {', '.join(form.sections)}"  # fmt: skip
            skipped.append(f"{photo.name} (name it {want})")
            continue
        sheet, letters, page = found
        if truth is None:
            skipped.append(f"{photo.name} (no {sheet}.json)")
            continue
        if sheet not in truths:
            truths[sheet] = load_sheet_truth(truth, form)
        expected = {f.id: truths[sheet].get(f.id) for letter in letters for f in form.sections[letter].fields}  # fmt: skip
        where = {"page": page, "sections": letters} if pages else {"section": letters}
        items.append({"id": f"{folder.name}/{photo.name}", "input": {"photo": photo.name, **where},
                      "expected_output": expected, "metadata": {"sheet": sheet, **where}})  # fmt: skip
    return items, skipped


def score_pages(pages: list[dict[str, Any]], items: list[dict[str, Any]], form: Form, latency: float) -> dict[str, Any]:  # fmt: skip
    """All scores of a run from what it recorded, so saved runs can be re-scored without a model.
    `pages` has one record per section read; a whole-page photo counts once as a photo."""
    expected = {i["input"]["photo"]: i["expected_output"] for i in items}
    scored = [p for p in pages if p["file"] in expected]
    total = sum_counts([score_section(p["reading"], expected[p["file"]], form.sections[p["section"]], p["json_valid"])
                        for p in scored])  # fmt: skip
    valid = {p["file"]: p["json_valid"] for p in scored}
    total.update(pages=len(valid), json_valid=float(sum(valid.values())))
    return {"metrics": metrics(total, latency=latency), "counts": total}


def run_tally(folder: Path, form: Form, model: str, *, langfuse: Any, configs: dict[str, str] | None,
              think: bool = False, max_side: int | None = None, prompt: Any = None,
              pages: bool = False) -> dict[str, Any]:  # fmt: skip
    """Benchmark one model on a folder as a Langfuse experiment: one trace per section photo (or
    with pages, per whole-page photo), scored against its sections of the truth, then the
    whole-run scores on the dataset run. With configs None (Langfuse not reachable) the same
    run happens on local items."""
    items, skipped = load_items(folder, form, pages)
    if not items:
        raise SystemExit(f"formbench tally: no {'page' if pages else 'section'} photo in {folder} has a truth file")  # fmt: skip
    side, version = max_side or FULL_MAX_SIDE, prompt_version(prompt)
    settings = {"folder": folder.name, "form": form.id, "model": model, "think": think,
                "max_side": side, "prompt_version": version,
                **({"photos": "pages", "min_side": PAGE_MIN_SIDE} if pages else {})}  # fmt: skip
    data = items
    if configs is not None:
        data = sync_dataset(langfuse, folder.name, items, f"{form.name}: section photos and their truth")  # fmt: skip

    def task(*, item: Any, **_: Any) -> dict[str, Any]:
        photo = item_input(item)
        langfuse.update_current_trace(name=f"{model} {photo['photo']}", tags=["formbench", "tally"], metadata=settings)  # fmt: skip
        start = time.perf_counter()
        if pages:
            result = read_page(folder / photo["photo"], form, photo["page"], model, think=think,
                               max_side=side, prompt=prompt)  # fmt: skip
        else:
            result = read_section(folder / photo["photo"], form, photo["section"], model,
                                  think=think, max_side=side, prompt=prompt)  # fmt: skip
            result["readings"] = {photo["section"]: result.pop("reading")}
        return {**result, "latency_s": round(time.perf_counter() - start, 1)}

    def evaluator(*, output: dict[str, Any], expected_output: Any, **_: Any) -> list[Any]:
        counts = sum_counts([score_section(reading, expected_output, form.sections[letter], output["json_valid"])
                             for letter, reading in output["readings"].items()])  # fmt: skip
        # one photo, however many sections it holds
        counts.update(pages=1, json_valid=float(output["json_valid"]))
        latency = output["latency_s"]
        return evaluations(metrics(counts, latency=latency), configs or {}, comments(counts, latency=latency))  # fmt: skip

    run_name = " ".join([model, *(["full pages"] if pages else []), *([f"prompt v{version}"] if version else []),
                         *(["think"] if think else []), *([f"max-side {side}"] if side != FULL_MAX_SIDE else []), stamp()])  # fmt: skip
    result = langfuse.run_experiment(
        name=folder.name, run_name=run_name, data=data, task=task, evaluators=[evaluator],
        max_concurrency=1, metadata=settings,
    )  # fmt: skip
    outputs = {item_input(r.item)["photo"]: r.output for r in result.item_results if r.output}
    failed = [i["input"]["photo"] for i in items if i["input"]["photo"] not in outputs]
    if len(failed) == len(items):
        raise SystemExit("formbench tally: every photo failed. Is Ollama running (`ollama list`)?")
    read = []  # one record per section, whatever photo it was read from
    for item in items:
        photo = item["input"]["photo"]
        output = outputs.get(photo) or {"readings": {}, "json_valid": False}
        for letter in item["input"].get("sections") or item["input"]["section"]:
            read.append({"file": photo, "section": letter, "json_valid": output["json_valid"],
                         "reading": output["readings"].get(letter, {})})  # fmt: skip
    latency = sum(o["latency_s"] for o in outputs.values())
    record = score_pages(read, items, form, latency)
    if result.dataset_run_id:
        notes = comments(record["counts"], latency=latency)
        for name, value in record["metrics"].items():
            langfuse.create_score(dataset_run_id=result.dataset_run_id, name=name, value=float(value),
                                  config_id=(configs or {}).get(name), comment=notes.get(name))  # fmt: skip
    langfuse.flush()
    return {**record, "pages": read, "latency_s": round(latency, 1), "page_count": len(items),
            "run_name": run_name, "dataset_run_url": result.dataset_run_url, "failed": failed,
            "skipped": skipped}  # fmt: skip


def command(args: argparse.Namespace) -> None:
    import json

    from langfuse import get_client

    from formbench.cli import model_tag

    folder = args.folder
    form = load_form(find_form(folder, args.form))
    console = Console()
    rows = []
    if args.rescore:
        for path in args.rescore:
            run = json.loads(path.read_text())
            items, _ = load_items(folder, form, run["settings"].get("photos") == "pages")
            run.update(score_pages(run["pages"], items, form, run.get("latency_s") or 0.0))
            write_json(path, run)
            rows.append((run_label(run["settings"]), run["metrics"]))
        console.print(score_table(f"Re-scored: {folder.name}", rows))
        return
    if not args.engine:
        raise SystemExit("formbench tally: give --engine (or --rescore saved run files)")

    langfuse = get_client()
    try:
        configs: dict[str, str] | None = score_configs(langfuse)
    except Exception as error:  # no keys, or the server is down: still run, save locally
        console.print(f"[yellow]Langfuse not reachable ({error}); results are saved locally only.[/yellow]")  # fmt: skip
        configs = None
    name, default = (form.page_prompt_name, DEFAULT_PAGE_PROMPT) if args.pages else (form.prompt_name, DEFAULT_PROMPT)  # fmt: skip
    prompt = fetch_prompt(langfuse, name, default, args.prompt)
    version = prompt_version(prompt)
    console.print(f"{form.name}. Prompt: {f'{name} v{version}' if version else 'the prompt in code (Langfuse not reachable)'}")  # fmt: skip
    for engine in args.engine:
        model = model_tag(engine)
        console.print(f"Running {engine} on {folder.name}...")
        result = run_tally(folder, form, model, langfuse=langfuse, configs=configs, think=args.think,
                           max_side=args.max_side, prompt=prompt, pages=args.pages)  # fmt: skip
        settings = {"scenario": folder.name, "form": form.id, "engine": engine, "think": args.think,
                    "max_side": args.max_side or FULL_MAX_SIDE, "prompt_version": version,
                    **({"photos": "pages", "min_side": PAGE_MIN_SIDE} if args.pages else {})}  # fmt: skip
        path = write_json(args.out / f"{stamp()}_{folder.name}_{slugify(engine)}.json", {"settings": settings, **result})  # fmt: skip
        for note, photos in (
            ("not read (failed)", result["failed"]),
            ("skipped", result["skipped"]),
        ):
            if photos:
                console.print(
                    f"[yellow]{len(photos)} photo(s) {note}: {', '.join(photos)}[/yellow]"
                )
        if result["dataset_run_url"]:
            console.print(f"Langfuse: {result['dataset_run_url']}")
        console.print(f"Saved to {path}")
        rows.append((run_label(settings), result["metrics"]))
    console.print(score_table(f"{folder.name}: this run", rows))
    console.print(f"All runs on this folder side by side: formbench compare --folder {folder.name}")

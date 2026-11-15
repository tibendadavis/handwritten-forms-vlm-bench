"""One benchmark run: a folder's photos read by one model, scored, tracked and saved.

Each run is a Langfuse experiment on the folder's dataset: one trace per photo, scored against
its truth, then the whole-run scores on the dataset run. The same run is saved locally as
JSON with everything needed to re-score it without a model.
"""

from __future__ import annotations

import datetime as dt
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from formbench.forms.base import FormKind, ReadOptions
from formbench.model import DEFAULT_MAX_SIDE
from formbench.prompts import fetch_prompt, prompt_version
from formbench.scoring import comments, metrics, sum_counts
from formbench.tracking import (
    RunError,
    evaluations,
    item_input,
    require_langfuse,
    score_run,
    sync_dataset,
)

NOT_READ_ERROR = "not read: the model call failed"


def utc_stamp() -> str:
    """UTC time for run names and file names, e.g. 20261030T101500Z."""
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def model_slug(model: str) -> str:
    """A model tag for file names: 'qwen3.5:9b' -> 'qwen3_5_9b'."""
    return re.sub(r"[^a-z0-9]+", "_", model.lower()).strip("_")


def mode_of(form: FormKind, pages: bool) -> str:
    if form.kind == "register":
        return "register"
    return "pages" if pages else "sections"


def run_label(settings: dict[str, Any]) -> str:
    """The model, whole pages if so, the prompt version, thinking and a non-default photo
    size: what makes two runs comparable or not. Runs with the same label are pooled."""
    side = settings.get("max_side") or DEFAULT_MAX_SIDE
    return " ".join(
        [
            settings["model"],
            *(["full pages"] if settings.get("mode") == "pages" else []),
            f"prompt v{settings.get('prompt_version')}",
            *(["think"] if settings.get("think") else []),
            *([f"{side}px"] if side != DEFAULT_MAX_SIDE else []),
        ]
    )


def run_name(settings: dict[str, Any], stamp: str) -> str:
    """The Langfuse run name: like the label, plus a UTC stamp so every run is unique."""
    side = settings["max_side"]
    return " ".join(
        [
            settings["model"],
            *(["full pages"] if settings["mode"] == "pages" else []),
            f"prompt v{settings['prompt_version']}",
            *(["think"] if settings["think"] else []),
            *([f"max-side {side}"] if side != DEFAULT_MAX_SIDE else []),
            stamp,
        ]
    )


@dataclass
class Plan:
    """Everything a run needs before the first model call."""

    folder: Path
    form: FormKind
    pages: bool
    options: ReadOptions
    items: list[dict[str, Any]]
    skipped: list[str]
    configs: dict[str, str]
    prompt: Any
    prompt_name: str

    @property
    def version(self) -> int | None:
        return prompt_version(self.prompt)


def prepare(
    langfuse: Any,
    folder: Path,
    form: FormKind,
    *,
    pages: bool,
    options: ReadOptions,
    prompt_choice: str | None,
) -> Plan:
    """Check Langfuse (score configs), the photos and the prompt before any model call."""
    configs = require_langfuse(langfuse)
    if pages and form.kind == "register":
        raise RunError("--pages is for cv-forms; a register is always one photo per page")
    items, skipped = form.items(folder, pages)
    if not items:
        raise RunError(f"no photo in {folder} has a truth file (formbench check lists why)")
    name, default = form.prompt_spec(pages)
    prompt = fetch_prompt(langfuse, name, default, prompt_choice)
    return Plan(folder, form, pages, options, items, skipped, configs, prompt, name)


def photo_part(item: dict[str, Any]) -> Any:
    """What part of the form a photo shows: a section letter, a page number, or None."""
    photo = item_input(item)
    return photo.get("section") or photo.get("page")


def score_photos(
    form: FormKind, items: list[dict[str, Any]], photos: list[dict[str, Any]]
) -> dict[str, Any]:
    """All scores of a run from its saved photo records and the truth in `items`, so a saved
    run can be re-scored without a model. Photos whose truth is gone are left out."""
    by_photo = {item_input(i)["photo"]: i for i in items}
    counts, outcomes = [], []
    for photo in photos:
        item = by_photo.get(photo["file"])
        if item is None:
            continue
        c, cells = form.score(
            item_input(item), photo["reading"], item["expected_output"], photo["json_valid"]
        )
        counts.append(c)
        outcomes += [{"file": photo["file"], **cell} for cell in cells]
    total = sum_counts(counts)
    latency = round(sum(p["latency_s"] or 0 for p in photos if p["file"] in by_photo), 1)
    return {
        "metrics": metrics(total, latency),
        "counts": total,
        "latency_s": latency,
        "outcomes": outcomes,
    }


def run_model(plan: Plan, model: str, langfuse: Any, out_dir: Path) -> dict[str, Any]:
    """Benchmark one model on the plan's photos; returns the saved record, with its path."""
    form, folder, options = plan.form, plan.folder, plan.options
    settings = {
        "folder": folder.name,
        "form": form.id,
        "kind": form.kind,
        "mode": mode_of(form, plan.pages),
        "model": model,
        "think": options.think,
        "max_side": options.max_side,
        "prompt_name": plan.prompt_name,
        "prompt_version": plan.version,
    }
    stamp = utc_stamp()
    name = run_name(settings, stamp)
    data = sync_dataset(langfuse, folder.name, plan.items, f"{form.name}: photos and their truth")

    def task(*, item: Any, **_: Any) -> dict[str, Any]:
        photo = item_input(item)
        langfuse.update_current_trace(
            name=f"{model} {photo['photo']}", tags=["formbench", form.kind], metadata=settings
        )
        start = time.perf_counter()
        result = form.read(folder, photo, model, plan.prompt, options)
        return {**result, "latency_s": round(time.perf_counter() - start, 1)}

    def evaluator(*, input: Any, output: Any, expected_output: Any, **_: Any) -> list[Any]:
        counts, _ = form.score(input, output["reading"], expected_output, output["json_valid"])
        latency = output["latency_s"]
        return evaluations(metrics(counts, latency), plan.configs, comments(counts, latency))

    result = langfuse.run_experiment(
        name=folder.name,
        run_name=name,
        data=data,
        task=task,
        evaluators=[evaluator],
        max_concurrency=1,  # one call at a time, so timings are fair on one machine
        metadata=settings,
    )
    outputs = {item_input(r.item)["photo"]: r.output for r in result.item_results if r.output}
    failed = [item_input(i)["photo"] for i in plan.items if item_input(i)["photo"] not in outputs]
    if len(failed) == len(plan.items):
        raise RunError(
            "every photo failed. Is Ollama running (ollama list)? Is the model name right?"
        )

    photos, empty = [], []
    for item in plan.items:
        photo = item_input(item)["photo"]
        output = outputs.get(photo) or {
            "reading": {},
            "json_valid": False,
            "errors": [NOT_READ_ERROR],
            "latency_s": 0.0,
        }
        photos.append(
            {
                "file": photo,
                "part": photo_part(item),
                "json_valid": output["json_valid"],
                "errors": output["errors"],
                "latency_s": output["latency_s"],
                "reading": output["reading"],
            }
        )
        if photo in outputs and output["json_valid"]:
            parts = form.is_empty_reading(
                item_input(item), output["reading"], item["expected_output"]
            )
            if parts:
                empty.append({"file": photo, "parts": parts})

    scored = score_photos(form, plan.items, photos)
    score_run(
        langfuse,
        result.dataset_run_id,
        scored["metrics"],
        plan.configs,
        comments(scored["counts"], scored["latency_s"]),
    )
    langfuse.flush()
    record = {
        "settings": settings,
        "run_name": name,
        "dataset_run_url": result.dataset_run_url,
        "metrics": scored["metrics"],
        "counts": scored["counts"],
        "latency_s": scored["latency_s"],
        "photo_count": len(plan.items),
        "photos": photos,
        "outcomes": scored["outcomes"],
        "failed": failed,
        "skipped": plan.skipped,
        "empty_readings": empty,
    }
    path = out_dir / f"{stamp}_{folder.name}_{model_slug(model)}.json"
    write_json(path, record)
    return {**record, "path": str(path)}


def write_json(path: Path, data: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def rescore(path: Path, folder: Path, form: FormKind) -> dict[str, Any]:
    """Re-score a saved run from its photos and the current truth files; writes it back."""
    run = json.loads(path.read_text(encoding="utf-8"))
    items, _ = form.items(folder, run["settings"].get("mode") == "pages")
    run.update(score_photos(form, items, run["photos"]))
    write_json(path, run)
    return run

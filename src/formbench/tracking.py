"""Langfuse for benchmark runs: score configs, datasets and scores.

- Score configs: each score is registered once with its description and range, so the
  Langfuse UI explains every number and rejects a value outside its range.
- Datasets: a photo folder is a dataset; each photo is an item whose expected output is what
  is really written on it.
- Runs: benchmarking one model on a folder is one dataset run. Photo scores land on each
  photo's trace; the same scores over the whole folder land on the run.
"""

from __future__ import annotations

from typing import Any

from langfuse import Evaluation
from langfuse.api import CreateScoreConfigRequest, ScoreDataType

from formbench.scoring import BY_NAME, SCORES


class RunError(SystemExit):
    """A run cannot go on; the message says why and what to check."""


def score_configs(langfuse: Any) -> dict[str, str]:
    """Score name -> Langfuse score config id. Existing configs (not archived) are reused by
    name; the ones missing are created."""
    existing: dict[str, str] = {}
    page = 1
    while True:
        found = langfuse.api.score_configs.get(page=page, limit=100)
        existing.update({c.name: c.id for c in found.data if not c.is_archived})
        if page >= found.meta.total_pages:
            break
        page += 1
    for score in SCORES:
        if score.name in existing:
            continue
        request = CreateScoreConfigRequest(
            name=score.name,
            data_type=ScoreDataType.NUMERIC,
            min_value=0,
            max_value=score.max_value,
            description=score.description,
        )
        existing[score.name] = langfuse.api.score_configs.create(request=request).id
    return existing


def require_langfuse(langfuse: Any) -> dict[str, str]:
    """The score configs, loaded before anything else in a run: a run that cannot be tracked
    is not started."""
    try:
        return score_configs(langfuse)
    except Exception as error:
        raise RunError(
            f"Langfuse is required for runs and is not reachable: {error}. Check "
            "LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY and LANGFUSE_HOST in .env."
        ) from None


def evaluations(
    scores: dict[str, float], configs: dict[str, str], notes: dict[str, str]
) -> list[Evaluation]:
    """Scores as Langfuse evaluations, each tied to its score config and with a comment
    saying how it was worked out."""
    return [
        Evaluation(
            name=name, value=float(value), config_id=configs.get(name), comment=notes.get(name)
        )
        for name, value in scores.items()
        if name in BY_NAME
    ]


def sync_dataset(
    langfuse: Any, name: str, items: list[dict[str, Any]], description: str
) -> list[Any]:
    """Create the dataset if needed and upsert its items (stable ids, so re-syncing is safe).
    Returns the dataset items for exactly these photos."""
    try:
        langfuse.get_dataset(name)
    except Exception:  # not found: the SDK raises the API's error type
        langfuse.create_dataset(name=name, description=description)
    for item in items:
        langfuse.create_dataset_item(
            dataset_name=name,
            id=item["id"],
            input=item["input"],
            expected_output=item["expected_output"],
            metadata=item["metadata"],
        )
    wanted = {item["id"] for item in items}
    return [item for item in langfuse.get_dataset(name).items if item.id in wanted]


def item_input(item: Any) -> dict[str, Any]:
    """The input of a Langfuse dataset item or of a local item (a plain dict)."""
    return item["input"] if isinstance(item, dict) else item.input


def score_run(
    langfuse: Any,
    dataset_run_id: str | None,
    scores: dict[str, float],
    configs: dict[str, str],
    notes: dict[str, str],
) -> None:
    """Put the whole-run scores on the dataset run, so runs sit side by side in Langfuse."""
    if not dataset_run_id:
        return
    for name, value in scores.items():
        langfuse.create_score(
            dataset_run_id=dataset_run_id,
            name=name,
            value=float(value),
            config_id=configs.get(name),
            comment=notes.get(name),
        )

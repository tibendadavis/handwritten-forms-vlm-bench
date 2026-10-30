"""Langfuse for runs that have a truth: score definitions, datasets and experiment runs.

- Score definitions: every score in SCORES is registered once (as a Langfuse score config)
  with its description and range, so the Langfuse UI explains each number and rejects a value
  outside its range.
- Datasets: a scenario folder, or a folder of real photos with truth files, becomes a dataset.
  Each photo is an item whose expected output is what is really written on it.
- Experiments: benchmarking one model on a dataset is one dataset run. Page scores land on
  each photo's trace; whole-run scores (values after review, misread flag recall, ...) land on
  the run, so runs of different models sit side by side under Datasets in Langfuse.

Nothing here is needed for scoring itself: without Langfuse the benchmark still runs and saves
its results locally.
"""

from __future__ import annotations

from typing import Any

from formbench.evaluation.scoring import BY_NAME, SCORES


def score_configs(langfuse: Any) -> dict[str, str]:
    """Score name -> Langfuse score config id, creating the configs that do not exist yet."""
    from langfuse.api import CreateScoreConfigRequest, ScoreDataType

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


def evaluations(scores: dict[str, float], configs: dict[str, str]) -> list[Any]:
    """Scores as Langfuse evaluations, each tied to its score definition when there is one."""
    from langfuse import Evaluation

    return [
        Evaluation(name=name, value=float(value), config_id=configs.get(name))
        for name, value in scores.items()
        if name in BY_NAME
    ]


def sync_dataset(langfuse: Any, name: str, items: list[dict[str, Any]], description: str) -> list[Any]:  # fmt: skip
    """Create or update the dataset and its items (stable ids, so re-syncing is safe) and
    return the dataset items for exactly these photos."""
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
            metadata=item.get("metadata"),
        )
    wanted = {item["id"] for item in items}
    dataset = langfuse.get_dataset(name)
    return [item for item in dataset.items if item.id in wanted]


def item_input(item: Any) -> dict[str, Any]:
    """The input of a dataset item or of a local item (a plain dict)."""
    return item["input"] if isinstance(item, dict) else item.input

"""Shared test fakes. Unit tests never need the network, Ollama or Langfuse."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from langfuse.api import NotFoundError
from langfuse.api.resources.prompts.types.prompt import Prompt_Text
from langfuse.model import TextPromptClient
from PIL import Image

FIXTURES = Path(__file__).parent / "fixtures"


def make_prompt(name: str, version: int, text: str, labels: list[str]) -> TextPromptClient:
    """A real Langfuse prompt object, so compile() behaves exactly as in a run."""
    data = Prompt_Text(
        name=name, version=version, prompt=text, config={}, labels=labels, tags=[], type="text"
    )
    return TextPromptClient(data)


class FakeLangfuse:
    """Records every call and keeps prompts, datasets, score configs and scores in memory.

    `down=True` makes every call fail as an unreachable server does.
    """

    def __init__(self, *, down: bool = False, configs: list[str] | None = None) -> None:
        self.down = down
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.prompts: dict[str, list[TextPromptClient]] = {}
        self.datasets: dict[str, dict[str, SimpleNamespace]] = {}
        self.configs = [
            SimpleNamespace(name=n, id=f"cfg-{n}", is_archived=False) for n in configs or []
        ]
        self.scores: list[dict[str, Any]] = []
        self.traces: list[dict[str, Any]] = []
        self.experiments: list[dict[str, Any]] = []
        self.flushed = 0
        self.api = SimpleNamespace(
            score_configs=SimpleNamespace(get=self._get_configs, create=self._create_config)
        )

    def _record(self, method: str, /, **kwargs: Any) -> None:
        self.calls.append((method, kwargs))
        if self.down:
            raise ConnectionError("connection refused")

    # prompts
    def add_prompt(self, name: str, text: str, labels: list[str] | None = None) -> None:
        versions = self.prompts.setdefault(name, [])
        versions.append(make_prompt(name, len(versions) + 1, text, labels or []))

    def get_prompt(self, name: str, *, version=None, label=None, **options: Any):
        self._record("get_prompt", name=name, version=version, label=label, **options)
        assert version is None or label is None, "Langfuse rejects version and label together"
        versions = self.prompts.get(name, [])
        if version is not None:
            found = [p for p in versions if p.version == version]
        elif label == "latest":
            found = versions[-1:]
        else:
            found = [p for p in versions if label in p.labels]
        if not found:
            raise NotFoundError({"message": f"prompt {name} not found"})
        return found[-1]

    def create_prompt(self, *, name: str, prompt: str, labels, tags, commit_message):
        self._record(
            "create_prompt",
            name=name,
            prompt=prompt,
            labels=labels,
            tags=tags,
            commit_message=commit_message,
        )
        self.add_prompt(name, prompt, labels)
        return self.prompts[name][-1]

    # score configs (two pages, to exercise paging)
    def _get_configs(self, *, page: int, limit: int):
        self._record("score_configs.get", page=page, limit=limit)
        half = (len(self.configs) + 1) // 2
        data = self.configs[:half] if page == 1 else self.configs[half:]
        return SimpleNamespace(data=data, meta=SimpleNamespace(total_pages=2))

    def _create_config(self, *, request):
        self._record("score_configs.create", request=request)
        config = SimpleNamespace(name=request.name, id=f"cfg-{request.name}", is_archived=False)
        self.configs.append(config)
        return config

    # datasets
    def get_dataset(self, name: str):
        self._record("get_dataset", name=name)
        if name not in self.datasets:
            raise NotFoundError({"message": f"dataset {name} not found"})
        return SimpleNamespace(name=name, items=list(self.datasets[name].values()))

    def create_dataset(self, *, name: str, description: str):
        self._record("create_dataset", name=name, description=description)
        self.datasets[name] = {}

    def create_dataset_item(self, *, dataset_name, id, input, expected_output, metadata):
        self._record(
            "create_dataset_item",
            dataset_name=dataset_name,
            id=id,
            input=input,
            expected_output=expected_output,
            metadata=metadata,
        )
        self.datasets[dataset_name][id] = SimpleNamespace(
            id=id, input=input, expected_output=expected_output, metadata=metadata
        )

    # experiments, traces and scores
    def update_current_trace(self, **kwargs: Any) -> None:
        self._record("update_current_trace", **kwargs)
        self.traces.append(kwargs)

    def create_score(self, **kwargs: Any) -> None:
        self._record("create_score", **kwargs)
        self.scores.append(kwargs)

    def flush(self) -> None:
        self.flushed += 1

    def run_experiment(self, *, name, run_name, data, task, evaluators, max_concurrency, metadata):
        """Like Langfuse: items one by one; an item whose task raises is left out."""
        self._record("run_experiment", name=name, run_name=run_name, metadata=metadata)
        self.experiments.append(
            {"name": name, "run_name": run_name, "max_concurrency": max_concurrency}
        )
        results = []
        for item in data:
            try:
                output = task(item=item)
            except Exception:
                continue
            evaluations = []
            for evaluator in evaluators:
                evaluations += evaluator(
                    input=item.input,
                    output=output,
                    expected_output=item.expected_output,
                    metadata=item.metadata,
                )
            results.append(SimpleNamespace(item=item, output=output, evaluations=evaluations))
        return SimpleNamespace(
            item_results=results,
            dataset_run_id="run-1",
            dataset_run_url="https://langfuse.test/runs/run-1",
        )


@pytest.fixture
def langfuse() -> FakeLangfuse:
    return FakeLangfuse()


def tiny_image(path: Path, size: tuple[int, int] = (40, 30)) -> Path:
    """A small white photo, enough for the code that opens images."""
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, "white").save(path, "PNG" if path.suffix == ".png" else "JPEG")
    return path


# Old DHIS2-era fixtures, used only by the old pipeline's tests until it is removed.


AGE_BANDS = ["0-5 months", "6-11 months", "12-23 months", "24-59 months"]
CHILD_BANDS = ["0-11 months default or 6-11 months (zero-dose)", "12-23 months", "24-59 months"]
VACCINES = ["BCG", "OPV", "Pentavalent", "IPV", "PCV", "Rota", "MRI"]
SESSIONS = ["Fixed/Facility", "Outreach/Mobile", "Health camp", "Unclear", "Not filled"]


def coc_uid(name):
    return "coc_" + "".join(ch for ch in name if ch.isalnum())


def _combo(uid, names):
    return {
        "id": uid,
        "name": uid,
        "categoryOptionCombos": [{"id": coc_uid(n), "name": n} for n in names],
    }


def _element(uid, name, combo):
    return {"dataElement": {"id": uid, "name": name, "valueType": "NUMBER", "categoryCombo": combo}}


@pytest.fixture
def ri_metadata():
    """A small stand-in for the immunisation data set with the same naming conventions."""
    doses = _combo(
        "ccDose",
        [
            f"Dose {d}, {band}, {sex}"
            for d in range(4)
            for band in AGE_BANDS
            for sex in ("Female", "Male")
        ],
    )
    children = _combo(
        "ccChild", [f"{sex}, {band}" for sex in ("Female", "Male") for band in CHILD_BANDS]
    )
    stock = _combo("ccStock", ["Issued", "Returned/Remaining"])
    elements = [_element(f"de{v}", v, doses) for v in VACCINES]
    elements += [
        _element("deChildren", "Children vaccinated", children),
        _element("deStock", "OPV Inventory", stock),
    ]
    return {
        "id": "dsRI",
        "name": "RI test",
        "periodType": "Daily",
        "categoryCombo": {
            "id": "ccSession",
            "name": "Session type",
            "categoryOptionCombos": [{"id": f"aoc{i}", "name": n} for i, n in enumerate(SESSIONS)],
        },
        "dataSetElements": elements,
        "sections": [],
    }
